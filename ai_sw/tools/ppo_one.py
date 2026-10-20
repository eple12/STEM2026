"""PPO for the one network that decides and drives (game/onenet.py), after DAgger.

    python tools/ppo_one.py --init policy/onenet/one_a/one_last.npz --out policy/onenet/ppo1

Both heads learn at once, each from its own stream of reward, on the same trunk:

* the **fast head** (wheel and pedal, 30 Hz, Gaussian) from the driver's reward of
  tools/ppo_drive.py -- off the road, on the kerb, line error, hits, recoveries,
  the jerk of the controls;
* the **slow head** (the 24 decision actions, 7.5 Hz, categorical) from the
  racecraft reward of game/raceenv.py (``REWARD``) -- places, held-up time with
  open road beside, contact, stewards -- exactly what trained the decision policy.

Every AI car of a scene is driven by the net; each keeps its own trajectory. Two
critics (one per stream). The start is held in place by a pull of each head
towards the DAgger network it began as (``--anchor``, ``--kl``).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import dagger_drive as dd
import dagger_one as d1
import ppo_drive as pdv

TRAIN_KINDS = {"start": 0.25, "race": 0.15, "pack": 0.15, "sbs": 0.10, "defend": 0.12,
               "tow": 0.18, "merge": 0.05}
HORIZON = {"race": 100.0}
GAMMA_F, LAM = 0.995, 0.95
#: Reward per metre driven (the speed is the car's own business: there is no speed
#: to keep to), and whether slowing for a car close ahead is exempt from the
#: slowness term (0: not).
R_PROG = float(os.environ.get("PO_R_PROG", 0.0))
EXEMPT = int(os.environ.get("PO_EXEMPT", 1))
#: Share of episodes in which some cars are thrown off the road (and the metres
#: beyond the white line they land).
KICK_P = float(os.environ.get("PO_KICK_P", 0.0))
GAMMA_S = 0.99
DT = 1.0 / 60.0


# --------------------------------------------------------------------------
# workers
# --------------------------------------------------------------------------
class Hook:
    """The net as a car's driver and decision-maker, with exploration; keeps what
    PPO needs of both heads."""

    needs_teacher = False
    free = True
    one = True

    def __init__(self, net, sigma, T, rng, sample):
        self.net, self.sigma, self.T, self.rng, self.sample = net, sigma, T, rng, sample
        self.fx, self.fa, self.flp, self.fr = [], [], [], []
        self.dx, self.da, self.dlp, self.dr = [], [], [], []
        self.last_x = None
        self.last_a = np.zeros(2, np.float32)
        self.prev_rec = 0
        self.prev_hit = 0

    def act(self, x, teacher=None):
        mu = self.net.forward(x)[:2]
        self.last_x = x
        if not self.sample:
            return np.clip(mu, -1.0, 1.0)
        a = mu + self.sigma * self.rng.standard_normal(2).astype(np.float32)
        z = (a - mu) / self.sigma
        self.fx.append(x)
        self.fa.append(a)
        self.flp.append(float(-0.5 * (z * z).sum() - 2.0 * np.log(self.sigma)))
        self.fr.append(0.0)
        c = np.clip(a, -1.0, 1.0)
        self.fr[-1] -= pdv.R_JERK * float(np.abs(c - self.last_a).sum())
        self.last_a = c
        return c

    def decide(self, x):
        logits = self.net.forward(x)[2:]
        if not self.sample:
            return int(np.argmax(logits))
        z = logits / self.T
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        a = int(self.rng.choice(len(p), p=p))
        self.dx.append(x)
        self.da.append(a)
        self.dlp.append(float(np.log(p[a] + 1e-12)))
        self.dr.append(0.0)
        return a


def _scenario(circuit, kind, seed):
    """A scene with every AI car an ego (the net drives them all)."""
    from game import config, grandprix, raceenv
    config.DRIVE_AI = "rules"
    if kind == "race":
        fld = grandprix.build(raceenv.track_of(circuit), 6, 3, seed, player=False)
        fld.lights_out()
        fld.views = fld._views()
        sc = raceenv.Scenario(circuit, "race", seed, fld, [], 0, 1e12, HORIZON["race"])
    else:
        sc = raceenv.make_scenario(circuit, kind, seed)
    sc.egos = [i for i, e in enumerate(sc.fld.cars) if e.driver is not None]
    return sc


def _kick(fld, i, rng):
    """Throw car *i* off the road: beyond a white line, at an angle, slower."""
    from game import raceenv
    e = fld.cars[i]
    me = fld.views[i]
    k = fld.frame.node(me.s)
    side = 1.0 if rng.random() < 0.5 else -1.0
    edge = float(fld.track.w_right[k] if side > 0 else fld.track.w_left[k])
    raceenv._place(fld, e, me.s, 0.0, lateral_abs=side * (edge + float(rng.uniform(1.5, 5.0))),
                   yaw_off=float(rng.normal(0.0, 0.5)), speed=float(rng.uniform(8.0, 28.0)))


def rollout(args):
    w, sigma, T, circuit, kind, seed, sample = args
    from game import drivenet, onenet, raceenv
    rng = np.random.default_rng(seed)
    net = onenet.OneNet(w)
    sc = _scenario(circuit, kind, seed)
    env = raceenv.RaceEnv(sc)
    env.reset()
    fld = env.fld
    hooks = {}
    for i in sc.egos:
        d = fld.cars[i].driver
        h = Hook(net, sigma, T, rng, sample)
        h.prev_rec, h.prev_hit = d.recoveries, fld.cars[i].hits
        d.follow.drive = h
        d.policy = onenet.OnePolicy(h)
        hooks[i] = h
    v1 = drivenet.OBS_DIM_V1
    stats = dict(rec=0, hits=0, n=0, kerb2=0, off=0)
    parts_sum = {}
    plan_every = fld.frame.plan_every
    tick = 0
    kicks = []
    if sample and KICK_P > 0.0 and rng.random() < KICK_P and kind in ("start", "race", "pack"):
        for i in rng.permutation(sc.egos)[:max(1, len(sc.egos) // 6)]:
            kicks.append((float(rng.uniform(4.0, 14.0)), int(i)))
    while fld.t < sc.max_t:
        fld.step(DT)
        env.ref.update()
        tick += 1
        while kicks and fld.t >= kicks[0][0]:
            _, ki = kicks.pop(0)
            _kick(fld, ki, rng)
        for i, h in hooks.items():
            e = fld.cars[i]
            d = e.driver
            drec, dhit = d.recoveries - h.prev_rec, e.hits - h.prev_hit
            h.prev_rec, h.prev_hit = d.recoveries, e.hits
            stats["rec"] += drec
            stats["hits"] += dhit
            if h.last_x is None:
                continue
            stats["n"] += 1
            gs, on = e.vehicle.grip_scale, e.vehicle.on_track
            stats["kerb2"] += gs < 0.9
            stats["off"] += not on
            if not (sample and h.fr):
                continue
            x = h.last_x
            err = abs(float(x[0])) * 3.0
            dv = min(float(x[13]) * 15.0, float(x[14]) * 20.0)
            if EXEMPT and x[v1 + 4] > 0.5 and x[v1] < 0.8:
                dv = 0.0                  # a car close ahead in the way: slowing for it is not slowness
            r = -pdv.R_RECOVERY * drec - pdv.R_HIT * dhit
            r -= DT * (pdv.R_OFF * (not on) + pdv.R_KERB * (gs < 0.9) + pdv.R_KERB1 * (gs < 0.98)
                       + pdv.R_ERR * max(err - 1.0, 0.0) ** 2
                       + (pdv.R_SLOW * dv if dv > 0.0 else pdv.R_FAST * -dv))
            h.fr[-1] += r + R_PROG * float(e.vehicle.speed) * DT
        if tick % plan_every == 0:
            rewards, parts = env.reward_step()
            for i, h in hooks.items():
                if sample and h.dr:
                    h.dr[-1] += rewards[i]
            for k, v in parts[sc.egos[0]].items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v
    ref = env.ref
    n_cars = len(hooks)
    queue_pct = 100.0 * float(sum(ref.queue_s[i] for i in hooks)) / max(float(sum(ref.racing_s[i] for i in hooks)), 1e-6)
    pens = float(sum(fld.rc.penalty(i) for i in hooks))
    traj = []
    if sample:
        for h in hooks.values():
            traj.append(((np.asarray(h.fx, np.float32), np.asarray(h.fa, np.float32),
                          np.asarray(h.flp, np.float32), np.asarray(h.fr, np.float32)),
                         (np.asarray(h.dx, np.float32), np.asarray(h.da, np.int64),
                          np.asarray(h.dlp, np.float32), np.asarray(h.dr, np.float32))))
    n = max(stats["n"], 1)
    info = {"kind": kind, "circuit": circuit, "cars": n_cars,
            "rec_per_car": stats["rec"] / n_cars, "hits": stats["hits"] / 2.0,
            "off_pct": 100.0 * stats["off"] / n, "kerb2_pct": 100.0 * stats["kerb2"] / n,
            "queue_pct": queue_pct, "passes": len(ref.passes), "pen_s": pens,
            "t": fld.t, "ret_f": float(sum(t[0][3].sum() for t in traj) / max(len(traj), 1)),
            "ret_s": float(sum(t[1][3].sum() for t in traj) / max(len(traj), 1)),
            "parts": parts_sum}
    return traj, info


# --------------------------------------------------------------------------
# learner
# --------------------------------------------------------------------------
def gae(r, v, gamma):
    n = len(r)
    adv = np.zeros(n, np.float32)
    last = 0.0
    for t in reversed(range(n)):
        nxt = v[t + 1] if t + 1 < n else 0.0
        delta = r[t] + gamma * nxt - v[t]
        last = delta + gamma * LAM * last
        adv[t] = last
    return adv


def summarise(infos):
    out = {}
    for kind in sorted({i["kind"] for i in infos}):
        rs = [i for i in infos if i["kind"] == kind]
        out[kind] = {k: float(np.mean([r[k] for r in rs]))
                     for k in ("rec_per_car", "hits", "off_pct", "kerb2_pct", "queue_pct", "passes", "pen_s")}
    return out


def fmt(s):
    return "  ".join(f"{k}: rec/car {v['rec_per_car']:.2f} hits {v['hits']:.1f} pass {v['passes']:.1f} "
                     f"queue {v['queue_pct']:.1f}% pen {v['pen_s']:.0f}s off {v['off_pct']:.2f}% kerb2 {v['kerb2_pct']:.1f}%"
                     for k, v in s.items())


def score(ev):
    """Higher is better: passing for it, contact, recoveries, penalties and being
    held up against it (per scene, averaged over kinds)."""
    return float(np.mean([v["passes"] / 3.0 - 0.5 * v["hits"] - 5.0 * v["rec_per_car"]
                          - 0.05 * v["pen_s"] - 0.1 * v["queue_pct"] for v in ev.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="the DAgger one-network (.npz)")
    ap.add_argument("--out", default="policy/onenet/ppo")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--episodes", type=int, default=14)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.06)
    ap.add_argument("--T", type=float, default=1.3, help="sampling temperature of the decisions")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-v", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--anchor", type=float, default=0.5, help="L2 pull of the wheel / pedal mean towards the start")
    ap.add_argument("--kl", type=float, default=0.1, help="pull of the decisions towards the start, where no car is near")
    ap.add_argument("--ent", type=float, default=0.005)
    ap.add_argument("--value-warmup", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-seeds", type=int, default=1)
    ap.add_argument("--eval-circuits", nargs="*", default=list(dd.EVAL_CIRCUITS))
    ap.add_argument("--seed0", type=int, default=500000)
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reward", default="", help='JSON of raceenv.REWARD terms to change, e.g. {"place": 6}')
    ap.add_argument("--r-hit", type=float, default=None)
    ap.add_argument("--r-rec", type=float, default=None)
    ap.add_argument("--r-off", type=float, default=None)
    ap.add_argument("--r-kerb", type=float, default=None)
    ap.add_argument("--r-prog", type=float, default=None, help="reward per metre driven")
    ap.add_argument("--r-fast", type=float, default=None, help="penalty per m/s over the plan's speed")
    ap.add_argument("--r-slow", type=float, default=None, help="penalty per m/s under it")
    ap.add_argument("--exempt", type=int, default=None, help="1: slowing for a car close ahead costs nothing")
    ap.add_argument("--kick-p", type=float, default=None, help="share of episodes with cars thrown off the road")
    ap.add_argument("--free-off-t", type=float, default=None,
                    help="seconds off the road before the recovery routine takes over")
    args = ap.parse_args()
    for flag, env in ((args.r_hit, "PD_R_HIT"), (args.r_rec, "PD_R_REC"), (args.r_off, "PD_R_OFF"),
                      (args.r_kerb, "PD_R_KERB"), (args.r_prog, "PO_R_PROG"), (args.r_fast, "PD_R_FAST"),
                      (args.r_slow, "PD_R_SLOW"), (args.exempt, "PO_EXEMPT"), (args.kick_p, "PO_KICK_P"),
                      (args.free_off_t, "FORMULA_FREE_OFF_T")):
        if flag is not None:
            os.environ[env] = str(flag)
    if args.reward:
        os.environ["RACEAI_REWARD"] = args.reward

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from game import onenet, raceai, raceenv

    torch.set_num_threads(max(args.jobs, 2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    circuits = [c for c in raceenv.circuits() if c not in dd.HOLDOUT]
    print(f"{len(circuits)} training circuits, held out: {', '.join(dd.HOLDOUT)}", flush=True)

    pi = d1.build_net()
    z = np.load(args.init)
    _load(pi, z)
    pi0 = d1.build_net()
    pi0.load_state_dict(pi.state_dict())
    for p in pi0.parameters():
        p.requires_grad_(False)

    def critic():
        return nn.Sequential(nn.Linear(onenet.IN_DIM, 128), nn.ReLU(), nn.Linear(128, 128),
                             nn.ReLU(), nn.Linear(128, 1))

    vf, vs = critic(), critic()
    opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr)
    opt_v = torch.optim.Adam(list(vf.parameters()) + list(vs.parameters()), lr=args.lr_v)
    alone_col = onenet.drivenet.OBS_DIM + raceai.OBS_NAMES.index("n0_valid")
    it0, best, counter = 0, -1e9, args.seed0
    state_path = out / "ppo_state.pt"
    if args.resume and state_path.exists():
        st = torch.load(state_path, weights_only=False)
        pi.load_state_dict(st["pi"])
        vf.load_state_dict(st["vf"])
        vs.load_state_dict(st["vs"])
        opt_pi.load_state_dict(st["opt_pi"])
        opt_v.load_state_dict(st["opt_v"])
        it0, best, counter = st["it"], st["best"], st["counter"]
        print(f"resumed at iteration {it0}", flush=True)

    def weights():
        return d1.weights_of(pi, out / "_cur.npz")

    rng = np.random.default_rng(args.seed0 + it0)
    kinds, kp = list(TRAIN_KINDS), np.array(list(TRAIN_KINDS.values()))
    kp = kp / kp.sum()
    log = open(out / "ppo_log.jsonl", "a", encoding="utf-8")
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as ex:

        def evaluate(w):
            tasks = [(w, 0.0, 1.0, c, k, s, False) for c in args.eval_circuits for k in ("start", "race")
                     for s in range(args.eval_seeds)]
            return summarise([r[1] for r in ex.map(rollout, tasks, chunksize=1)])

        if it0 == 0:
            base = evaluate(weights())
            print("START    " + fmt(base), flush=True)
            log.write(json.dumps({"it": 0, "eval": base, "score": score(base)}) + "\n")
            log.flush()
            best = score(base)
        for it in range(it0, args.iters):
            t0 = time.time()
            w = weights()
            tasks = []
            for _ in range(args.episodes):
                counter += 1
                tasks.append((w, args.sigma, args.T, circuits[int(rng.integers(len(circuits)))],
                              kinds[int(rng.choice(len(kinds), p=kp))], counter, True))
            results = list(ex.map(rollout, tasks, chunksize=1))
            t_roll = time.time() - t0
            fx, fa, flp, fadv, fret = [], [], [], [], []
            dx, da, dlp, dadv, dret = [], [], [], [], []
            with torch.no_grad():
                for traj, _info in results:
                    for (x, a, lp, r), (x2, a2, lp2, r2) in traj:
                        if len(a):
                            v = vf(torch.tensor(x)).squeeze(1).numpy()
                            adv = gae(r, v, GAMMA_F)
                            fx.append(x), fa.append(a), flp.append(lp), fadv.append(adv), fret.append(adv + v)
                        if len(a2):
                            v = vs(torch.tensor(x2)).squeeze(1).numpy()
                            adv = gae(r2, v, GAMMA_S)
                            dx.append(x2), da.append(a2), dlp.append(lp2), dadv.append(adv), dret.append(adv + v)
            X = torch.tensor(np.concatenate(fx))
            A = torch.tensor(np.concatenate(fa))
            LP = torch.tensor(np.concatenate(flp))
            ADV = torch.tensor(np.concatenate(fadv))
            RET = torch.tensor(np.concatenate(fret))
            ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-6)
            X2 = torch.tensor(np.concatenate(dx))
            A2 = torch.tensor(np.concatenate(da))
            LP2 = torch.tensor(np.concatenate(dlp))
            ADV2 = torch.tensor(np.concatenate(dadv))
            RET2 = torch.tensor(np.concatenate(dret))
            ADV2 = (ADV2 - ADV2.mean()) / (ADV2.std() + 1e-6)
            alone = (X2[:, alone_col] == 0.0).float()
            with torch.no_grad():
                logp0 = F.log_softmax(pi0(X2)[1] / args.T, dim=1)
            nf, ns = len(A), len(A2)
            stats = {"plf": 0.0, "pls": 0.0, "vlf": 0.0, "vls": 0.0, "anchor": 0.0, "kl": 0.0,
                     "ent": 0.0, "clipf": 0.0, "clips": 0.0}
            k_upd = 0
            n_batches = max((nf + args.batch - 1) // args.batch, (ns + args.batch - 1) // args.batch)
            for _ep in range(args.epochs):
                pf = torch.randperm(nf)
                ps = torch.randperm(ns)
                for b in range(n_batches):
                    idf = pf[(b * args.batch) % nf:][:args.batch]
                    ids = ps[(b * args.batch) % ns:][:args.batch]
                    if len(idf) == 0 or len(ids) == 0:
                        continue
                    xb, x2b = X[idf], X2[ids]
                    vlf = F.mse_loss(vf(xb).squeeze(1), RET[idf])
                    vls = F.mse_loss(vs(x2b).squeeze(1), RET2[ids])
                    opt_v.zero_grad()
                    (vlf + vls).backward()
                    torch.nn.utils.clip_grad_norm_(list(vf.parameters()) + list(vs.parameters()), 1.0)
                    opt_v.step()
                    stats["vlf"] += float(vlf.detach())
                    stats["vls"] += float(vls.detach())
                    if it < args.value_warmup:
                        k_upd += 1
                        continue
                    mu, _ = pi(xb)
                    zz = (A[idf] - mu) / args.sigma
                    lp = -0.5 * (zz * zz).sum(1) - 2.0 * np.log(args.sigma)
                    ratio = torch.exp(lp - LP[idf])
                    a = ADV[idf]
                    plf = -torch.min(ratio * a, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
                    with torch.no_grad():
                        mu0 = pi0(xb)[0]
                    anc = ((mu - mu0) ** 2).sum(1).mean()
                    _, logits = pi(x2b)
                    logp_all = F.log_softmax(logits / args.T, dim=1)
                    lp2 = logp_all.gather(1, A2[ids][:, None]).squeeze(1)
                    ratio2 = torch.exp(lp2 - LP2[ids])
                    a2 = ADV2[ids]
                    pls = -torch.min(ratio2 * a2, torch.clamp(ratio2, 1 - args.clip, 1 + args.clip) * a2).mean()
                    p = logp_all.exp()
                    ent = -(p * logp_all).sum(1).mean()
                    kl_full = (logp0[ids].exp() * (logp0[ids] - logp_all)).sum(1)
                    kl = (kl_full * alone[ids]).sum() / alone[ids].sum().clamp(min=1.0)
                    loss = plf + args.anchor * anc + pls - args.ent * ent + args.kl * kl
                    opt_pi.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(pi.parameters(), 0.5)
                    opt_pi.step()
                    stats["plf"] += float(plf.detach())
                    stats["pls"] += float(pls.detach())
                    stats["anchor"] += float(anc.detach())
                    stats["kl"] += float(kl.detach())
                    stats["ent"] += float(ent.detach())
                    stats["clipf"] += float(((ratio - 1).abs() > args.clip).float().mean())
                    stats["clips"] += float(((ratio2 - 1).abs() > args.clip).float().mean())
                    k_upd += 1
            for k in stats:
                stats[k] /= max(k_upd, 1)
            infos = [r[1] for r in results]
            rec = {"it": it + 1, "samples": [int(nf), int(ns)],
                   "ret_f": float(np.mean([i["ret_f"] for i in infos])),
                   "ret_s": float(np.mean([i["ret_s"] for i in infos])),
                   "rec_per_car": float(np.mean([i["rec_per_car"] for i in infos])),
                   "hits": float(np.mean([i["hits"] for i in infos])),
                   "queue_pct": float(np.mean([i["queue_pct"] for i in infos])),
                   "passes": float(np.mean([i["passes"] for i in infos])),
                   "off_pct": float(np.mean([i["off_pct"] for i in infos])),
                   "roll_s": round(t_roll, 1), "iter_s": round(time.time() - t0, 1),
                   **{k: round(v, 4) for k, v in stats.items()}}
            if (it + 1) % args.eval_every == 0 or it + 1 == args.iters:
                ev = evaluate(weights())
                rec["eval"] = ev
                rec["score"] = score(ev)
                if rec["score"] > best:
                    best = rec["score"]
                    d1.export(pi, out / "ppo_best.npz")
                    rec["best"] = True
            d1.export(pi, out / "ppo_last.npz")
            torch.save({"pi": pi.state_dict(), "vf": vf.state_dict(), "vs": vs.state_dict(),
                        "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "it": it + 1,
                        "best": best, "counter": counter}, state_path)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"it {it + 1:4d}  n {nf:7d}/{ns:6d}  ret {rec['ret_f']:7.2f}/{rec['ret_s']:7.2f}  "
                  f"rec/car {rec['rec_per_car']:.2f}  hits {rec['hits']:.1f}  pass {rec['passes']:.1f}  "
                  f"queue {rec['queue_pct']:.1f}%  off {rec['off_pct']:.2f}%  anchor {stats['anchor']:.4f}  "
                  f"kl {stats['kl']:.3f}  {rec['iter_s']:.0f}s", flush=True)
            if "eval" in rec:
                print("    EVAL  " + fmt(rec["eval"]) + f"  score {rec['score']:.2f}"
                      + ("  *best*" if rec.get("best") else ""), flush=True)
            if args.max_hours and (time.time() - t_start) > args.max_hours * 3600:
                print("max hours reached", flush=True)
                break
    (out / "_cur.npz").unlink(missing_ok=True)


def _load(net, z):
    """Weights of a one-network .npz into the torch module."""
    import torch
    lin = [m for m in net.trunk if hasattr(m, "weight")]
    slow = [m for m in net.slow if hasattr(m, "weight")]
    pairs = [(lin[0], "W0", "b0"), (lin[1], "W1", "b1"), (net.fast, "Wf", "bf"),
             (slow[0], "Ws0", "bs0"), (slow[1], "Ws1", "bs1")]
    for m, wk, bk in pairs:
        m.weight.data = torch.tensor(z[wk].T.copy())
        m.bias.data = torch.tensor(z[bk].copy())


if __name__ == "__main__":
    main()
