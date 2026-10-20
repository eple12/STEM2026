"""Teach a network to steer and work the pedals (game/drivenet.py): DAgger.

    python tools/dagger_drive.py train --out policy/drivenet/run1 --jobs 4
    python tools/dagger_drive.py eval --net policy/drivenet/run1/dagger_last.npz --jobs 4

The teacher is ``PlanFollower`` -- the hand-built tracker of the minimum-time
plan. Behaviour cloning alone does not work for a car on the limit: the
network's first small error puts it in a state the teacher never drove, where
it errs more. DAgger fixes the data, not the loss: the *student* drives, the
teacher says what it would have done in every state the student reaches, and
those labels are added to everything gathered so far.

The cars are driven in the situations the follower is for: the grid launching
into the first corner, the few-car scenes of raceenv (tow, sbs, defend, merge,
pack), and a stretch of a whole race. The orders it gets (line offsets, caps,
paces, flags) come from the decision layer exactly as in the game, so the
network meets lane changes, queueing, cars alongside and an incident ahead
as the teacher does.

Evaluation is student-only on fixed seeds (never in training) against the
teacher on the same episodes: recoveries, contact, time on the kerbs, line error
and speed. ``Sepang``, ``Hockenheim`` and ``Montreal`` are never trained on.
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

TRAIN_KINDS = {"start": 0.30, "race": 0.20, "pack": 0.15, "sbs": 0.10, "defend": 0.10,
               "tow": 0.10, "merge": 0.05}
HORIZON = {"start": 40.0, "race": 100.0}
EVAL_CIRCUITS = ("Monza", "Spa", "Austin", "Silverstone", "Catalunya", "Sepang",
                 "Hockenheim", "Montreal")
HOLDOUT = ("Sepang", "Hockenheim", "Montreal")
EVAL_KINDS = ("start", "race")
DT = 1.0 / 60.0
HIDDEN = 256


# --------------------------------------------------------------------------
# workers
# --------------------------------------------------------------------------
class Hook:
    """What a follower asks every REPEAT ticks. Remembers the question and the
    teacher's answer; drives with the student's, or (``beta``) the teacher's."""

    needs_teacher = True

    def __init__(self, net, beta, rng, record):
        self.net, self.beta, self.rng, self.record = net, beta, rng, record
        #: The student sees the cars around (and is not held to the traffic
        #: rules); the teacher's answer is still the rules'. A teacher alone
        #: (no net) is recorded in the free form, to be learned from.
        self.free = net.free if net is not None else True
        self.obs, self.lab = [], []

    def act(self, x, teacher):
        if self.record:
            self.obs.append(x)
            self.lab.append(teacher)
        if self.net is None or (self.beta > 0.0 and self.rng.random() < self.beta):
            return teacher
        return self.net.act(x)


def _scene(circuit, kind, seed):
    from game import config, grandprix, raceenv
    config.DRIVE_AI = "rules"
    if kind == "race":
        fld = grandprix.build(raceenv.track_of(circuit), 6, 3, seed, player=False)
        fld.lights_out()
        fld.views = fld._views()
        return fld
    return raceenv.make_scenario(circuit, kind, seed).fld


def episode(args):
    """One scene driven by ``net`` (None: by the teacher). Returns the labelled
    states and what happened."""
    w, beta, circuit, kind, seed, record = args
    from game import drivenet
    net = drivenet.DriveNet(w) if w is not None else None
    rng = np.random.default_rng(seed)
    fld = _scene(circuit, kind, seed)
    horizon = HORIZON.get(kind, 35.0)
    hooks = {}
    plain = w is None and not record        # the follower alone, at its own rate
    for e in fld.cars:
        if e.driver is None:
            continue
        h = Hook(net, beta, rng, record)
        if not plain:
            e.driver.follow.drive = h
        hooks[e.idx] = h
    prev = {i: (fld.cars[i].driver.recoveries, fld.cars[i].hits) for i in hooks}
    st = dict(rec=0, hits=0, n=0, kerb=0, kerb2=0, off=0, err2=0.0, speed=0.0, corner=0, ckerb=0)
    plan = next(e.driver.plan for e in fld.cars if e.driver is not None)
    fr = fld.frame
    while fld.t < horizon:
        fld.step(DT)
        for i, h in hooks.items():
            e = fld.cars[i]
            d = e.driver
            r0, h0 = prev[i]
            st["rec"] += d.recoveries - r0
            st["hits"] += e.hits - h0
            prev[i] = (d.recoveries, e.hits)
            if fld.views[i].racing is False and kind != "start":
                continue
            st["n"] += 1
            gs = e.vehicle.grip_scale
            st["kerb"] += gs < 0.98
            st["kerb2"] += gs < 0.9
            st["off"] += not e.vehicle.on_track
            st["err2"] += d.follow.lat_err ** 2
            st["speed"] += e.vehicle.speed
            if abs(float(d.plan.kappa[fr.node(fld.views[i].s)])) > 3e-3:
                st["corner"] += 1
                st["ckerb"] += gs < 0.98
    obs = np.concatenate([np.asarray(h.obs, np.float32) for h in hooks.values() if h.obs]) \
        if record else np.zeros((0, drivenet.OBS_DIM), np.float32)
    lab = np.concatenate([np.asarray(h.lab, np.float32) for h in hooks.values() if h.lab]) \
        if record else np.zeros((0, 2), np.float32)
    n = max(st["n"], 1)
    info = {"kind": kind, "circuit": circuit, "rec": st["rec"], "hits": st["hits"] / 2.0,
            "off_pct": 100.0 * st["off"] / n, "kerb_pct": 100.0 * st["kerb"] / n,
            "kerb2_pct": 100.0 * st["kerb2"] / n,
            "corner_kerb_pct": 100.0 * st["ckerb"] / max(st["corner"], 1),
            "rms_err": float(np.sqrt(st["err2"] / n)), "speed": st["speed"] / n,
            "cars": len(hooks)}
    return obs, lab, info


def summarise(infos):
    out = {}
    for kind in sorted({i["kind"] for i in infos}):
        rs = [i for i in infos if i["kind"] == kind]
        out[kind] = {k: float(np.mean([r[k] for r in rs]))
                     for k in ("rec", "hits", "off_pct", "kerb_pct", "kerb2_pct",
                               "corner_kerb_pct", "rms_err", "speed")}
        hs = [r for r in rs if r["circuit"] in HOLDOUT]
        out[kind]["rec_holdout"] = float(np.mean([r["rec"] for r in hs])) if hs else float("nan")
    return out


def fmt(s):
    return "  ".join(f"{k}: rec {v['rec']:.2f} hits {v['hits']:.2f} off {v['off_pct']:.2f}% kerb {v['kerb_pct']:.1f}%"
                     f" rms {v['rms_err']:.2f} v {v['speed']:.1f}" for k, v in s.items())


# --------------------------------------------------------------------------
# learner
# --------------------------------------------------------------------------
def build_net():
    import torch.nn as nn
    from game import drivenet
    return nn.Sequential(nn.Linear(drivenet.OBS_DIM, HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, drivenet.ACT_DIM))


def export(net, path):
    lin = [m for m in net if hasattr(m, "weight")]
    d = {"layers": np.int64(len(lin))}
    for i, m in enumerate(lin):
        d[f"W{i}"] = m.weight.detach().numpy().T.astype(np.float32)
        d[f"b{i}"] = m.bias.detach().numpy().astype(np.float32)
    np.savez(path, **d)


def weights_of(net, tmp):
    export(net, tmp)
    z = np.load(tmp)
    return {k: z[k] for k in z.files}


def fit(net, opt, X, Y, steps, batch):
    import torch
    import torch.nn.functional as F
    n = len(X)
    tot = 0.0
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,))
        loss = F.smooth_l1_loss(net(X[idx]), Y[idx], beta=0.05)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        tot += float(loss)
    return tot / steps


def cmd_train(args):
    import torch
    from game import raceenv

    torch.set_num_threads(max(args.jobs, 2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    circuits = [c for c in raceenv.circuits() if c not in HOLDOUT]
    print(f"{len(circuits)} training circuits, held out: {', '.join(HOLDOUT)}", flush=True)
    net = build_net()
    if args.init:
        z = np.load(args.init)
        from game import drivenet
        for i, m in enumerate([m for m in net if hasattr(m, "weight")]):
            W = z[f"W{i}"]
            if i == 0 and W.shape[0] < drivenet.OBS_DIM:
                # A network from before it saw the cars: the new inputs start at zero.
                W = np.vstack([W, np.zeros((drivenet.OBS_DIM - W.shape[0], W.shape[1]), W.dtype)])
            m.weight.data = torch.tensor(W.T.copy())
            m.bias.data = torch.tensor(z[f"b{i}"].copy())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    it0, counter = 0, args.seed0
    state_path = out / "dagger_state.pt"
    if args.resume and state_path.exists():
        st = torch.load(state_path, weights_only=False)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        it0, counter = st["it"], st["counter"]
        print(f"resumed at iteration {it0}", flush=True)
    rng = np.random.default_rng(args.seed0 + it0)
    kinds, kp = list(TRAIN_KINDS), np.array(list(TRAIN_KINDS.values()))
    kp = kp / kp.sum()
    log = open(out / "dagger_log.jsonl", "a", encoding="utf-8")
    Xs, Ys = [], []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as ex:

        def evaluate(w):
            tasks = [(w, 0.0, c, k, s, False) for c in args.eval_circuits for k in EVAL_KINDS
                     for s in range(args.eval_seeds)]
            return summarise([r[2] for r in ex.map(episode, tasks, chunksize=1)])

        if it0 == 0:
            base = evaluate(None)
            print("TEACHER  " + fmt(base), flush=True)
            log.write(json.dumps({"it": 0, "teacher": base}) + "\n")
            log.flush()
        for it in range(it0, args.iters):
            t0 = time.time()
            w = None if it == 0 and not args.init else weights_of(net, out / "_cur.npz")
            beta = max(0.0, args.beta0 * (1.0 - it / max(args.beta_iters, 1)))
            tasks = []
            for _ in range(args.episodes):
                counter += 1
                tasks.append((w, beta, circuits[int(rng.integers(len(circuits)))],
                              kinds[int(rng.choice(len(kinds), p=kp))], counter, True))
            res = list(ex.map(episode, tasks, chunksize=1))
            t_roll = time.time() - t0
            for o, l, _ in res:
                if len(o):
                    Xs.append(o)
                    Ys.append(l)
            total = sum(len(x) for x in Xs)
            while total > args.max_samples and len(Xs) > 1:
                total -= len(Xs.pop(0))
                Ys.pop(0)
            X = torch.tensor(np.concatenate(Xs))
            Y = torch.tensor(np.concatenate(Ys))
            loss = fit(net, opt, X, Y, args.steps, args.batch)
            info = summarise([r[2] for r in res])
            rec = {"it": it + 1, "beta": round(beta, 3), "samples": int(len(X)), "loss": round(loss, 5),
                   "train": info, "roll_s": round(t_roll, 1), "iter_s": round(time.time() - t0, 1)}
            if (it + 1) % args.eval_every == 0 or it + 1 == args.iters:
                rec["eval"] = evaluate(weights_of(net, out / "_cur.npz"))
            export(net, out / "dagger_last.npz")
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "it": it + 1,
                        "counter": counter}, state_path)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"it {it + 1:3d}  beta {beta:.2f}  n {len(X):8d}  loss {loss:.4f}  {rec['iter_s']:.0f}s "
                  f"(roll {t_roll:.0f}s)\n    train " + fmt(info), flush=True)
            if "eval" in rec:
                print("    EVAL  " + fmt(rec["eval"]), flush=True)
            if args.max_hours and (time.time() - t_start) > args.max_hours * 3600:
                print("max hours reached", flush=True)
                break
    (out / "_cur.npz").unlink(missing_ok=True)


def cmd_eval(args):
    w = None
    if args.net:
        z = np.load(args.net)
        w = {k: z[k] for k in z.files}
    circuits = args.circuits or list(EVAL_CIRCUITS)
    kinds = args.kinds or list(EVAL_KINDS)
    tasks = [(w, 0.0, c, k, s, False) for c in circuits for k in kinds for s in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as ex:
        infos = [r[2] for r in ex.map(episode, tasks, chunksize=1)]
    s = summarise(infos)
    print(("student " + args.net) if args.net else "teacher", fmt(s), flush=True)
    if args.name:
        p = Path(args.name)
        p.write_text(json.dumps({"summary": s, "episodes": infos}, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", default="policy/drivenet/run")
    t.add_argument("--iters", type=int, default=60)
    t.add_argument("--episodes", type=int, default=24)
    t.add_argument("--jobs", type=int, default=4)
    t.add_argument("--steps", type=int, default=1500, help="gradient steps per iteration")
    t.add_argument("--batch", type=int, default=4096)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--beta0", type=float, default=1.0, help="share of decisions driven by the teacher at first")
    t.add_argument("--beta-iters", type=int, default=3)
    t.add_argument("--max-samples", type=int, default=8_000_000)
    t.add_argument("--eval-every", type=int, default=3)
    t.add_argument("--eval-seeds", type=int, default=1)
    t.add_argument("--eval-circuits", nargs="*", default=list(EVAL_CIRCUITS))
    t.add_argument("--seed0", type=int, default=200000)
    t.add_argument("--max-hours", type=float, default=0.0)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--init", default=None)
    e = sub.add_parser("eval")
    e.add_argument("--net", default=None, help="weights (.npz); omitted: the teacher")
    e.add_argument("--circuits", nargs="*", default=None)
    e.add_argument("--kinds", nargs="*", default=None)
    e.add_argument("--seeds", type=int, default=2)
    e.add_argument("--jobs", type=int, default=4)
    e.add_argument("--name", default=None)
    args = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
