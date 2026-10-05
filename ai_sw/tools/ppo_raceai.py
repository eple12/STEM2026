"""PPO for the race driver's decision layer (game/raceai.py), stage 3.

    python tools/ppo_raceai.py --bc policy/raceai/bc.npz --out policy/raceai/ppo \
        --iters 400 --episodes 160 --jobs 4

Starts from the behaviour-cloned policy (tools/train_raceai.py bc) and improves
it on the scenarios (game/raceenv.py) against the rule-based drivers, rewarded by
progress and places and punished by what the stewards punish.

Why PPO and not the project's IQN: the task is a handful of discrete decisions in
short episodes with a policy that has to stay close to a good prior, and what it
learns is a *distribution* (when to pull out, when to cover) rather than a
greedy argmax over values. A categorical PPO policy has none of the Gaussian
trouble the solo-lap runs hit -- there is no continuous noise to be optimised
around -- and the prior is a few lines of KL.

* **Exploration.** The BC policy is confident, so a plain softmax would never try
  a pass. Sampling uses softmax(logits / T) with T > 1 (``--T``), the same
  distribution the update sees; the deployed policy is the argmax, which T does
  not change.
* **Prior.** Where nothing is near the car the rule layer's plan following is
  already right, so there the policy is held to the BC policy with a KL penalty
  (``--kl``); in traffic it is free to change.
* **Held-out circuits.** ``HOLDOUT`` are never trained on; the evaluation reports
  them apart, to say whether it learned racing or learned circuits.
* **Evaluation** is greedy, ego-only, rules everywhere else, on fixed seeds that
  the training seeds never reach -- the same numbers tools/race_metrics.py prints.

Workers are processes (the simulation is Python and is what costs); the update is
a few small matrices. Checkpoints (``ppo_state.pt``) are written every iteration,
so a killed run resumes (``--resume``); ``--max-hours`` ends one cleanly.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

#: What the trainer draws scenes from. ``start`` (the grid launching into the first
#: braking zone, twenty cars) is slow, so it has a smaller share than its worth.
TRAIN_KINDS = {"tow": 0.28, "sbs": 0.14, "defend": 0.22, "merge": 0.06, "pack": 0.15,
               "start": 0.15}
EVAL_CIRCUITS = ("Monza", "Spa", "Austin", "Silverstone", "Catalunya", "Sepang",
                 "Hockenheim", "Montreal")
HOLDOUT = ("Sepang", "Hockenheim", "Montreal")
KINDS = ("tow", "sbs", "defend", "merge", "pack")
GAMMA, LAM = 0.99, 0.95


# --------------------------------------------------------------------------
# workers
# --------------------------------------------------------------------------
def forward(w, x):
    h = (x - w["mean"]) / w["std"]
    n = int(w["layers"])
    for i in range(n):
        h = h @ w[f"W{i}"] + w[f"b{i}"]
        if i < n - 1:
            h = np.maximum(h, 0.0)
    return h


def _scene_egos(kind, rng):
    """Which cars of a scene the policy drives in training."""
    if kind == "merge":
        return [int(i) for i in rng.permutation([1, 2, 3, 4])[:2]]
    if kind == "pack":
        return [int(i) for i in rng.permutation(8)[:3]]
    if kind == "start":
        return [int(i) for i in rng.permutation(19)[:5]]
    return [0]


def rollout(args):
    w, T, circuit, kind, seed = args
    from game import raceenv
    rng = np.random.default_rng(seed)
    egos = _scene_egos(kind, rng)
    sc = raceenv.make_scenario(circuit, kind, seed, ego=egos[0])
    sc.egos = egos
    env = raceenv.RaceEnv(sc)
    obs = env.reset()
    traj = {i: ([], [], [], []) for i in egos}
    parts_sum = {}
    done = False
    while not done:
        acts = {}
        for i, o in obs.items():
            z = forward(w, o) / T
            z = z - z.max()
            p = np.exp(z)
            p /= p.sum()
            a = int(rng.choice(len(p), p=p))
            t = traj[i]
            t[0].append(o)
            t[1].append(a)
            t[2].append(float(np.log(p[a] + 1e-12)))
            acts[i] = a
        obs, rew, done, info = env.step(acts)
        for i in egos:
            traj[i][3].append(rew[i])
        for k, v in info["parts"][egos[0]].items():
            parts_sum[k] = parts_sum.get(k, 0.0) + v
    rep = env.ref.report(egos)
    rep.update(ego_rank_end=raceenv._rank(env.fld, egos[0]))
    if kind == "start":
        rep["goal_reached"] = True
    rep.update(kind=kind, circuit=circuit, goal_reached=bool(
        env.fld.cars[egos[0]].limits.progress is not None
        and env.fld.cars[egos[0]].limits.progress - env.ref.start_progress[egos[0]] >= sc.goal))
    ok = raceenv.success(kind, rep)
    out = {i: (np.asarray(t[0], np.float32), np.asarray(t[1], np.int64),
               np.asarray(t[2], np.float32), np.asarray(t[3], np.float32))
           for i, t in traj.items()}
    return out, {"kind": kind, "circuit": circuit, "ok": bool(ok), "parts": parts_sum,
                 "ret": float(sum(parts_sum.values())), "hits": rep["ego_hits"],
                 "pen": rep["ego_penalty_s"], "t": rep["t"]}


def eval_episode(args):
    w, circuit, kind, seed = args
    from game import raceai, raceenv
    sc = raceenv.make_scenario(circuit, kind, seed)
    rep = raceenv.run(sc, raceai.Policy(w))
    return {"kind": kind, "circuit": circuit, "ok": bool(raceenv.success(kind, rep)),
            "hits": rep["ego_hits"], "pen": rep["ego_penalty_s"],
            "passes": rep["passes_by_ego"], "rank": rep["ego_rank_end"]}


# --------------------------------------------------------------------------
# learner
# --------------------------------------------------------------------------
def gae(r, v):
    n = len(r)
    adv = np.zeros(n, np.float32)
    last = 0.0
    for t in reversed(range(n)):
        nxt = v[t + 1] if t + 1 < n else 0.0           # the episode ends: nothing after
        delta = r[t] + GAMMA * nxt - v[t]
        last = delta + GAMMA * LAM * last
        adv[t] = last
    return adv


def summarise_eval(res):
    out = {"all": {}, "holdout": {}}
    for kind in KINDS:
        rs = [r for r in res if r["kind"] == kind]
        hs = [r for r in rs if r["circuit"] in HOLDOUT]
        out["all"][kind] = float(np.mean([r["ok"] for r in rs])) if rs else float("nan")
        out["holdout"][kind] = float(np.mean([r["ok"] for r in hs])) if hs else float("nan")
    out["score"] = float(np.mean(list(out["all"].values())))
    out["score_holdout"] = float(np.mean(list(out["holdout"].values())))
    out["passes_tow"] = float(np.mean([r["passes"] for r in res if r["kind"] == "tow"] or [0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc", default="policy/raceai/bc.npz")
    ap.add_argument("--out", default="policy/raceai/ppo")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--episodes", type=int, default=160, help="episodes per iteration")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--T", type=float, default=1.5, help="sampling temperature")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-v", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--kl", type=float, default=0.3, help="KL to BC where nothing is near")
    ap.add_argument("--value-warmup", type=int, default=2, help="iterations that train only V")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-seeds", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=100000)
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from game import raceai, raceenv
    from train_raceai import build, export, load_into

    torch.set_num_threads(max(args.jobs, 2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    circuits = [c for c in raceenv.circuits() if c not in HOLDOUT]
    print(f"{len(circuits)} training circuits, held out: {', '.join(HOLDOUT)}", flush=True)

    pi = build([128, 128], raceai.OBS_DIM, raceai.N_ACTIONS)
    mean, std = load_into(pi, args.bc)
    bc = build([128, 128], raceai.OBS_DIM, raceai.N_ACTIONS)
    load_into(bc, args.bc)
    for p in bc.parameters():
        p.requires_grad_(False)
    v_net = build([128, 128], raceai.OBS_DIM, 1)
    mean_t, std_t = torch.tensor(mean), torch.tensor(std)
    opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr)
    opt_v = torch.optim.Adam(v_net.parameters(), lr=args.lr_v)
    alone_col = raceai.OBS_NAMES.index("n0_valid")
    it0, best, counter = 0, -1.0, args.seed0
    state_path = out / "ppo_state.pt"
    if args.resume and state_path.exists():
        st = torch.load(state_path, weights_only=False)
        pi.load_state_dict(st["pi"])
        v_net.load_state_dict(st["v"])
        opt_pi.load_state_dict(st["opt_pi"])
        opt_v.load_state_dict(st["opt_v"])
        it0, best, counter = st["it"], st["best"], st["counter"]
        print(f"resumed at iteration {it0}, best score {best:.3f}", flush=True)

    def norm(x):
        return (x - mean_t) / std_t

    def weights():
        path = out / "_cur.npz"
        export(pi, mean, std, path)
        z = np.load(path)
        return {k: z[k] for k in z.files}

    rng = np.random.default_rng(args.seed0 + it0)
    kinds, kp = list(TRAIN_KINDS), np.array(list(TRAIN_KINDS.values()))
    kp = kp / kp.sum()
    log = open(out / "ppo_log.jsonl", "a", encoding="utf-8")
    t_start = time.time()
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as ex:

        def evaluate(w):
            tasks = [(w, c, k, s) for c in EVAL_CIRCUITS for k in KINDS
                     for s in range(args.eval_seeds)]
            return summarise_eval(list(ex.map(eval_episode, tasks, chunksize=2)))

        for it in range(it0, args.iters):
            t0 = time.time()
            w = weights()
            tasks = []
            for _ in range(args.episodes):
                counter += 1
                tasks.append((w, args.T, circuits[int(rng.integers(len(circuits)))],
                              kinds[int(rng.choice(len(kinds), p=kp))], counter))
            results = list(ex.map(rollout, tasks, chunksize=2))
            t_roll = time.time() - t0

            obs_l, act_l, logp_l, adv_l, ret_l = [], [], [], [], []
            with torch.no_grad():
                for traj, _info in results:
                    for o, a, lp, r in traj.values():
                        if len(a) == 0:
                            continue
                        v = v_net(norm(torch.tensor(o))).squeeze(1).numpy()
                        adv = gae(r, v)
                        obs_l.append(o)
                        act_l.append(a)
                        logp_l.append(lp)
                        adv_l.append(adv)
                        ret_l.append(adv + v)
            obs = torch.tensor(np.concatenate(obs_l))
            act = torch.tensor(np.concatenate(act_l))
            logp_old = torch.tensor(np.concatenate(logp_l))
            adv = torch.tensor(np.concatenate(adv_l))
            ret = torch.tensor(np.concatenate(ret_l))
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            alone = (obs[:, alone_col] == 0.0).float()
            n = len(act)
            x = norm(obs)
            with torch.no_grad():
                p_bc = F.softmax(bc(x) / args.T, dim=1)
                logp_bc = torch.log(p_bc + 1e-9)
            stats = {"pl": 0.0, "vl": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0}
            k_upd = 0
            for _ep in range(args.epochs):
                perm = torch.randperm(n)
                for i in range(0, n, args.batch):
                    idx = perm[i:i + args.batch]
                    xb = x[idx]
                    v_pred = v_net(xb).squeeze(1)
                    vl = F.mse_loss(v_pred, ret[idx])
                    opt_v.zero_grad()
                    vl.backward()
                    torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
                    opt_v.step()
                    stats["vl"] += float(vl)
                    if it < args.value_warmup:
                        k_upd += 1
                        continue
                    logits = pi(xb) / args.T
                    logp_all = F.log_softmax(logits, dim=1)
                    lp = logp_all.gather(1, act[idx][:, None]).squeeze(1)
                    ratio = torch.exp(lp - logp_old[idx])
                    a = adv[idx]
                    pl = -torch.min(ratio * a, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
                    p = logp_all.exp()
                    ent = -(p * logp_all).sum(1).mean()
                    kl_full = (p_bc[idx] * (logp_bc[idx] - logp_all)).sum(1)
                    kl = (kl_full * alone[idx]).sum() / alone[idx].sum().clamp(min=1.0)
                    loss = pl - args.ent * ent + args.kl * kl
                    opt_pi.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(pi.parameters(), 0.5)
                    opt_pi.step()
                    stats["pl"] += float(pl)
                    stats["ent"] += float(ent)
                    stats["kl"] += float(kl)
                    stats["clipfrac"] += float(((ratio - 1).abs() > args.clip).float().mean())
                    k_upd += 1
            for k in stats:
                stats[k] /= max(k_upd, 1)

            infos = [r[1] for r in results]
            per_kind = {k: float(np.mean([i["ok"] for i in infos if i["kind"] == k] or [float("nan")]))
                        for k in kinds}
            rec = {"it": it + 1, "samples": int(n), "ret": float(np.mean([i["ret"] for i in infos])),
                   "train_ok": per_kind, "hits": float(np.mean([i["hits"] for i in infos])),
                   "roll_s": round(t_roll, 1), "iter_s": round(time.time() - t0, 1), **{k: round(v, 4) for k, v in stats.items()}}
            if (it + 1) % args.eval_every == 0 or it + 1 == args.iters:
                ev = evaluate(weights())
                rec["eval"] = ev
                if ev["score"] > best:
                    best = ev["score"]
                    export(pi, mean, std, out / "ppo_best.npz")
                    rec["best"] = True
            export(pi, mean, std, out / "ppo_last.npz")
            torch.save({"pi": pi.state_dict(), "v": v_net.state_dict(), "opt_pi": opt_pi.state_dict(),
                        "opt_v": opt_v.state_dict(), "it": it + 1, "best": best, "counter": counter},
                       state_path)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            line = (f"it {it + 1:4d}  n {n:6d}  ret {rec['ret']:6.2f}  hits {rec['hits']:.2f}  "
                    f"ok " + " ".join(f"{k}:{v:.2f}" for k, v in per_kind.items())
                    + f"  ent {stats['ent']:.3f} kl {stats['kl']:.3f} clip {stats['clipfrac']:.2f}"
                    f"  {rec['iter_s']:.0f}s (roll {t_roll:.0f}s)")
            print(line, flush=True)
            if "eval" in rec:
                e = rec["eval"]
                print("   EVAL  " + " ".join(f"{k}:{e['all'][k]:.2f}/{e['holdout'][k]:.2f}" for k in KINDS)
                      + f"   score {e['score']:.3f} (holdout {e['score_holdout']:.3f})  tow passes {e['passes_tow']:.2f}"
                      + ("  *best*" if rec.get("best") else ""), flush=True)
            if args.max_hours and (time.time() - t_start) > args.max_hours * 3600:
                print("max hours reached", flush=True)
                break
    (out / "_cur.npz").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
