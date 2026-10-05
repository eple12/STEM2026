"""PPO for the steering/pedal network (game/drivenet.py), after DAgger.

    python tools/ppo_drive.py --init policy/drivenet/run1/dagger_last.npz --out policy/drivenet/ppo1

DAgger (tools/dagger_drive.py) makes the network drive like the hand-built
follower -- which includes everything the follower does that a person would not:
it rides the kerb (the minimum-time plan itself puts two wheels on it through
a fifth of every corner), it makes hard, jerky inputs, it goes off the road
under a cap with nobody near it. What "natural" means is not in the teacher, so
here the network is trained against what a spectator would mind:

* leaving the road, a recovery, contact -- as before;
* **kerb**: two wheels or more past the white line (``raceenv.KERB_GRIP``),
  which is the corner cutting that looks wrong; one wheel is let off lightly;
* **line error** past a metre, and speed away from the one asked for (too slow
  and too fast, the second dearer) -- so it does not simply crawl to be safe;
* **smoothness**: change of wheel and pedal from one decision to the next.

Every car of a scene shares the network and is rewarded alone. The update is PPO
with a fixed-width Gaussian on the two outputs, anchored to the DAgger network
by an L2 term on its mean where it matters least to change it. Workers are
processes (the simulation is what costs). ``ppo_state.pt`` is written every
iteration (``--resume``); ``--max-hours`` ends a run cleanly.
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

import dagger_drive as dd

TRAIN_KINDS = {"start": 0.35, "race": 0.15, "pack": 0.15, "sbs": 0.10, "defend": 0.10,
               "tow": 0.10, "merge": 0.05}
GAMMA, LAM = 0.995, 0.95
DT = 1.0 / 60.0

#: Reward (per second unless said otherwise).
R_RECOVERY = 6.0          # each
R_HIT = 2.0               # each
R_OFF = 0.6
R_KERB = 0.8              # two wheels or more beyond the white line
R_KERB1 = 0.08            # any wheel
R_ERR = 0.12              # per m^2 past a metre of line error
R_SLOW = 0.025            # per m/s under the speed asked for
R_FAST = 0.05             # per m/s over it
R_JERK = 0.15             # per unit change of wheel / pedal between decisions


def mlp(w, x):
    h = x
    n = int(w["layers"])
    for i in range(n):
        h = h @ w[f"W{i}"] + w[f"b{i}"]
        if i < n - 1:
            h = np.maximum(h, 0.0)
    return h


class Hook:
    """The network with exploration noise; keeps what PPO needs."""

    needs_teacher = False

    def __init__(self, w, sigma, rng, sample):
        self.w, self.sigma, self.rng, self.sample = w, sigma, rng, sample
        self.obs, self.act, self.logp, self.rew = [], [], [], []
        self.last_x = None
        self.last_a = np.zeros(2, np.float32)
        self.jerk = 0.0

    def act_(self, x):
        mu = mlp(self.w, x)
        self.last_x = x
        if not self.sample:
            return np.clip(mu, -1.0, 1.0)
        a = mu + self.sigma * self.rng.standard_normal(2).astype(np.float32)
        z = (a - mu) / self.sigma
        self.obs.append(x)
        self.act.append(a)
        self.logp.append(float(-0.5 * (z * z).sum() - 2.0 * np.log(self.sigma)))
        self.rew.append(0.0)
        c = np.clip(a, -1.0, 1.0)
        self.rew[-1] -= R_JERK * float(np.abs(c - self.last_a).sum())
        self.last_a = c
        return c


class _Adapter:
    needs_teacher = False

    def __init__(self, hook):
        self.hook = hook

    def act(self, x, teacher=None):
        return self.hook.act_(x)


def rollout(args):
    w, sigma, circuit, kind, seed = args
    from game import config
    config.DRIVE_AI = "rules"
    rng = np.random.default_rng(seed)
    fld = dd._scene(circuit, kind, seed)
    horizon = dd.HORIZON.get(kind, 35.0)
    hooks = {}
    for e in fld.cars:
        if e.driver is None:
            continue
        h = Hook(w, sigma, rng, True)
        e.driver.follow.drive = _Adapter(h)
        hooks[e.idx] = h
    prev = {i: (fld.cars[i].driver.recoveries, fld.cars[i].hits) for i in hooks}
    stats = dict(rec=0, hits=0, n=0, kerb=0, kerb2=0, off=0, err2=0.0)
    while fld.t < horizon:
        fld.step(DT)
        for i, h in hooks.items():
            if not h.rew or h.last_x is None:
                continue
            e = fld.cars[i]
            d = e.driver
            r0, h0 = prev[i]
            drec, dhit = d.recoveries - r0, e.hits - h0
            prev[i] = (d.recoveries, e.hits)
            stats["rec"] += drec
            stats["hits"] += dhit
            stats["n"] += 1
            gs = e.vehicle.grip_scale
            on = e.vehicle.on_track
            x = h.last_x
            err = abs(float(x[0])) * 3.0
            stats["err2"] += err * err
            stats["kerb"] += gs < 0.98
            stats["kerb2"] += gs < 0.9
            stats["off"] += not on
            dv = min(float(x[13]) * 15.0, float(x[14]) * 20.0)       # + : slower than asked
            r = -R_RECOVERY * drec - R_HIT * dhit
            r -= DT * (R_OFF * (not on) + R_KERB * (gs < 0.9) + R_KERB1 * (gs < 0.98)
                       + R_ERR * max(err - 1.0, 0.0) ** 2
                       + (R_SLOW * dv if dv > 0.0 else R_FAST * -dv))
            h.rew[-1] += r
    traj = [(np.asarray(h.obs, np.float32), np.asarray(h.act, np.float32),
             np.asarray(h.logp, np.float32), np.asarray(h.rew, np.float32))
            for h in hooks.values() if h.act]
    n = max(stats["n"], 1)
    ret = float(sum(t[3].sum() for t in traj) / max(len(traj), 1))
    return traj, {"kind": kind, "circuit": circuit, "ret": ret, "rec": stats["rec"],
                  "hits": stats["hits"] / 2.0, "cars": len(hooks),
                  "kerb_pct": 100.0 * stats["kerb"] / n, "kerb2_pct": 100.0 * stats["kerb2"] / n,
                  "off_pct": 100.0 * stats["off"] / n,
                  "rms_err": float(np.sqrt(stats["err2"] / n))}


def gae(r, v):
    n = len(r)
    adv = np.zeros(n, np.float32)
    last = 0.0
    for t in reversed(range(n)):
        nxt = v[t + 1] if t + 1 < n else 0.0
        delta = r[t] + GAMMA * nxt - v[t]
        last = delta + GAMMA * LAM * last
        adv[t] = last
    return adv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="the DAgger network (.npz)")
    ap.add_argument("--out", default="policy/drivenet/ppo")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.06)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-v", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--anchor", type=float, default=0.5, help="L2 pull of the mean towards the start")
    ap.add_argument("--value-warmup", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-seeds", type=int, default=1)
    ap.add_argument("--eval-circuits", nargs="*", default=list(dd.EVAL_CIRCUITS))
    ap.add_argument("--seed0", type=int, default=300000)
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from game import drivenet, raceenv

    torch.set_num_threads(max(args.jobs, 2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    circuits = [c for c in raceenv.circuits() if c not in dd.HOLDOUT]
    print(f"{len(circuits)} training circuits, held out: {', '.join(dd.HOLDOUT)}", flush=True)

    pi = dd.build_net()
    z = np.load(args.init)
    for i, m in enumerate([m for m in pi if hasattr(m, "weight")]):
        m.weight.data = torch.tensor(z[f"W{i}"].T.copy())
        m.bias.data = torch.tensor(z[f"b{i}"].copy())
    pi0 = dd.build_net()
    pi0.load_state_dict(pi.state_dict())
    for p in pi0.parameters():
        p.requires_grad_(False)
    v_net = nn.Sequential(nn.Linear(drivenet.OBS_DIM, 128), nn.ReLU(), nn.Linear(128, 128),
                          nn.ReLU(), nn.Linear(128, 1))
    opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr)
    opt_v = torch.optim.Adam(v_net.parameters(), lr=args.lr_v)
    it0, best, counter = 0, 1e9, args.seed0
    state_path = out / "ppo_state.pt"
    if args.resume and state_path.exists():
        st = torch.load(state_path, weights_only=False)
        pi.load_state_dict(st["pi"])
        v_net.load_state_dict(st["v"])
        opt_pi.load_state_dict(st["opt_pi"])
        opt_v.load_state_dict(st["opt_v"])
        it0, best, counter = st["it"], st["best"], st["counter"]
        print(f"resumed at iteration {it0}", flush=True)

    def weights():
        return dd.weights_of(pi, out / "_cur.npz")

    rng = np.random.default_rng(args.seed0 + it0)
    kinds, kp = list(TRAIN_KINDS), np.array(list(TRAIN_KINDS.values()))
    kp = kp / kp.sum()
    log = open(out / "ppo_log.jsonl", "a", encoding="utf-8")
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as ex:

        def evaluate(w):
            tasks = [(w, 0.0, c, k, s, False) for c in args.eval_circuits for k in dd.EVAL_KINDS
                     for s in range(args.eval_seeds)]
            return dd.summarise([r[2] for r in ex.map(dd.episode, tasks, chunksize=1)])

        def score(ev):
            # What we want small: recoveries, kerb, off-road, line error.
            return sum(v["rec"] + 0.2 * v["kerb2_pct"] + 2.0 * v["off_pct"] + 2.0 * v["rms_err"]
                       for v in ev.values())

        if it0 == 0:
            base = evaluate(weights())
            print("START    " + dd.fmt(base), flush=True)
            log.write(json.dumps({"it": 0, "eval": base}) + "\n")
            log.flush()
            best = score(base)
        for it in range(it0, args.iters):
            t0 = time.time()
            w = weights()
            tasks = []
            for _ in range(args.episodes):
                counter += 1
                tasks.append((w, args.sigma, circuits[int(rng.integers(len(circuits)))],
                              kinds[int(rng.choice(len(kinds), p=kp))], counter))
            results = list(ex.map(rollout, tasks, chunksize=1))
            t_roll = time.time() - t0
            obs_l, act_l, logp_l, adv_l, ret_l = [], [], [], [], []
            with torch.no_grad():
                for traj, _info in results:
                    for o, a, lp, r in traj:
                        v = v_net(torch.tensor(o)).squeeze(1).numpy()
                        adv = gae(r, v)
                        obs_l.append(o)
                        act_l.append(a)
                        logp_l.append(lp)
                        adv_l.append(adv)
                        ret_l.append(adv + v)
            x = torch.tensor(np.concatenate(obs_l))
            act = torch.tensor(np.concatenate(act_l))
            logp_old = torch.tensor(np.concatenate(logp_l))
            adv = torch.tensor(np.concatenate(adv_l))
            ret = torch.tensor(np.concatenate(ret_l))
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            n = len(act)
            stats = {"pl": 0.0, "vl": 0.0, "anchor": 0.0, "clipfrac": 0.0}
            k_upd = 0
            for _ep in range(args.epochs):
                perm = torch.randperm(n)
                for i in range(0, n, args.batch):
                    idx = perm[i:i + args.batch]
                    xb = x[idx]
                    vl = F.mse_loss(v_net(xb).squeeze(1), ret[idx])
                    opt_v.zero_grad()
                    vl.backward()
                    torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
                    opt_v.step()
                    stats["vl"] += float(vl.detach())
                    if it < args.value_warmup:
                        k_upd += 1
                        continue
                    mu = pi(xb)
                    zz = (act[idx] - mu) / args.sigma
                    lp = -0.5 * (zz * zz).sum(1) - 2.0 * np.log(args.sigma)
                    ratio = torch.exp(lp - logp_old[idx])
                    a = adv[idx]
                    pl = -torch.min(ratio * a, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
                    anc = ((mu - pi0(xb)) ** 2).sum(1).mean()
                    loss = pl + args.anchor * anc
                    opt_pi.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(pi.parameters(), 0.5)
                    opt_pi.step()
                    stats["pl"] += float(pl.detach())
                    stats["anchor"] += float(anc.detach())
                    stats["clipfrac"] += float(((ratio - 1).abs() > args.clip).float().mean())
                    k_upd += 1
            for k in stats:
                stats[k] /= max(k_upd, 1)
            infos = [r[1] for r in results]
            rec = {"it": it + 1, "samples": int(n), "ret": float(np.mean([i["ret"] for i in infos])),
                   "rec_per_car": float(np.mean([i["rec"] / max(i["cars"], 1) for i in infos])),
                   "kerb2_pct": float(np.mean([i["kerb2_pct"] for i in infos])),
                   "off_pct": float(np.mean([i["off_pct"] for i in infos])),
                   "rms_err": float(np.mean([i["rms_err"] for i in infos])),
                   "roll_s": round(t_roll, 1), "iter_s": round(time.time() - t0, 1),
                   **{k: round(v, 4) for k, v in stats.items()}}
            if (it + 1) % args.eval_every == 0 or it + 1 == args.iters:
                ev = evaluate(weights())
                rec["eval"] = ev
                sc = score(ev)
                if sc < best:
                    best = sc
                    dd.export(pi, out / "ppo_best.npz")
                    rec["best"] = True
            dd.export(pi, out / "ppo_last.npz")
            torch.save({"pi": pi.state_dict(), "v": v_net.state_dict(), "opt_pi": opt_pi.state_dict(),
                        "opt_v": opt_v.state_dict(), "it": it + 1, "best": best, "counter": counter},
                       state_path)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"it {it + 1:4d}  n {n:7d}  ret {rec['ret']:7.2f}  rec/car {rec['rec_per_car']:.3f}"
                  f"  kerb2 {rec['kerb2_pct']:.1f}%  off {rec['off_pct']:.2f}%  rms {rec['rms_err']:.2f}"
                  f"  anchor {stats['anchor']:.4f}  clip {stats['clipfrac']:.2f}  {rec['iter_s']:.0f}s", flush=True)
            if "eval" in rec:
                print("    EVAL  " + dd.fmt(rec["eval"]) + ("  *best*" if rec.get("best") else ""), flush=True)
            if args.max_hours and (time.time() - t_start) > args.max_hours * 3600:
                print("max hours reached", flush=True)
                break
    (out / "_cur.npz").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
