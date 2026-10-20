"""Teach ONE network to decide and to drive (game/onenet.py): DAgger.

    python tools/dagger_one.py train --out policy/onenet/run1 --jobs 4
    python tools/dagger_one.py eval --net policy/onenet/run1/one_last.npz --jobs 4

The two networks it replaces are the teachers -- the decision policy
(``assets/policies/raceai.npz``) for the slow head and the driver
(``assets/policies/drivenet.npz``) for the fast head. Each is a function of its
own observation, so a teacher's answer to any input the student meets is just that
function evaluated there: no teacher has to run in the simulation. The student
drives the cars (every car of a scene); the teachers label what it saw; the labels
pile up and the student is fitted to all of them (tools/dagger_drive.py says why
the student, not the teacher, has to be the one driving).

The student starts as the driver: its trunk and fast head are the driver's weights
(the decision half of the input at zero), so training begins with a net that
drives and has still to learn to decide.
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

ROOT = Path(__file__).resolve().parents[1]
TEACH_DRIVE = ROOT / "assets" / "policies" / "drivenet.npz"
TEACH_DEC = ROOT / "assets" / "policies" / "raceai.npz"
HIDDEN = 256
SLOW_HIDDEN = 128
W_SLOW = 0.2
KEYS = ("W0", "b0", "W1", "b1", "Wf", "bf", "Ws0", "bs0", "Ws1", "bs1")


# --------------------------------------------------------------------------
# workers
# --------------------------------------------------------------------------
def _mlp(w, x):
    h = x
    n = int(w["layers"])
    for i in range(n):
        h = h @ w[f"W{i}"] + w[f"b{i}"]
        if i < n - 1:
            h = np.maximum(h, 0.0)
    return h


class Hook:
    """What the net is asked: remembers the question and what the teachers say,
    answers with the student (or, with probability ``beta``, the teachers)."""

    needs_teacher = False
    free = True
    one = True

    def __init__(self, student, teach_drive, teach_dec, beta, rng, record):
        self.student, self.td, self.tp = student, teach_drive, teach_dec
        self.beta, self.rng, self.record = beta, rng, record
        self.X, self.F, self.D = [], [], []

    def forward(self, x):
        from game import drivenet
        n = drivenet.OBS_DIM
        fast = _mlp(self.td, x[:n]).astype(np.float32)
        dec = self.tp.values(x[n:]).astype(np.float32)
        if self.record:
            self.X.append(x)
            self.F.append(fast)
            self.D.append(int(np.argmax(dec)))
        if self.student is None or (self.beta > 0.0 and self.rng.random() < self.beta):
            return np.concatenate([fast, dec])
        return self.student.forward(x)

    def act(self, x, teacher=None):
        return self.forward(x)[:2]

    def decide(self, x):
        return int(np.argmax(self.forward(x)[2:]))


def episode(args):
    """One scene driven by ``w`` (None: by the teachers). Returns the labelled
    inputs and what happened."""
    w, beta, circuit, kind, seed, record = args
    from game import drivenet, onenet, raceai
    student = onenet.OneNet(w) if w is not None else None
    td = dict(np.load(TEACH_DRIVE))
    tp = raceai.Policy.load(TEACH_DEC)
    rng = np.random.default_rng(seed)
    fld = dd._scene(circuit, kind, seed)
    horizon = dd.HORIZON.get(kind, 35.0)
    hooks = {}
    for e in fld.cars:
        if e.driver is None:
            continue
        h = Hook(student, td, tp, beta, rng, record)
        e.driver.follow.drive = h
        e.driver.policy = onenet.OnePolicy(h)
        hooks[e.idx] = h
    prev = {i: (fld.cars[i].driver.recoveries, fld.cars[i].hits) for i in hooks}
    st = dict(rec=0, hits=0, n=0, kerb=0, kerb2=0, off=0, err2=0.0, speed=0.0, corner=0, ckerb=0)
    fr = fld.frame
    while fld.t < horizon:
        fld.step(dd.DT)
        for i in hooks:
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
    if record:
        got = [h for h in hooks.values() if h.X]
        X = np.concatenate([np.asarray(h.X, np.float32) for h in got]) if got else np.zeros((0, onenet.IN_DIM), np.float32)
        F = np.concatenate([np.asarray(h.F, np.float32) for h in got]) if got else np.zeros((0, 2), np.float32)
        D = np.concatenate([np.asarray(h.D, np.int64) for h in got]) if got else np.zeros(0, np.int64)
    else:
        X, F, D = (np.zeros((0, onenet.IN_DIM), np.float32), np.zeros((0, 2), np.float32),
                   np.zeros(0, np.int64))
    n = max(st["n"], 1)
    info = {"kind": kind, "circuit": circuit, "rec": st["rec"], "hits": st["hits"] / 2.0,
            "off_pct": 100.0 * st["off"] / n, "kerb_pct": 100.0 * st["kerb"] / n,
            "kerb2_pct": 100.0 * st["kerb2"] / n,
            "corner_kerb_pct": 100.0 * st["ckerb"] / max(st["corner"], 1),
            "rms_err": float(np.sqrt(st["err2"] / n)), "speed": st["speed"] / n,
            "cars": len(hooks)}
    return X, F, D, info


# --------------------------------------------------------------------------
# learner
# --------------------------------------------------------------------------
def build_net():
    import torch.nn as nn
    from game import drivenet, onenet

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(onenet.IN_DIM, HIDDEN), nn.ReLU(),
                                       nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
            self.fast = nn.Linear(HIDDEN, drivenet.ACT_DIM)
            self.slow = nn.Sequential(nn.Linear(HIDDEN, SLOW_HIDDEN), nn.ReLU(),
                                      nn.Linear(SLOW_HIDDEN, onenet.N_DEC))

        def forward(self, x):
            h = self.trunk(x)
            return self.fast(h), self.slow(h)

    return Net()


def init_from_driver(net, path):
    """Trunk and fast head from the driver network; its input gets the decision
    half at zero."""
    import torch
    from game import drivenet
    z = np.load(path)
    lin = [m for m in net.trunk if hasattr(m, "weight")]
    W0 = z["W0"]
    W0 = np.vstack([W0, np.zeros((net.trunk[0].in_features - W0.shape[0], W0.shape[1]), W0.dtype)])
    for m, W, b in ((lin[0], W0, z["b0"]), (lin[1], z["W1"], z["b1"]), (net.fast, z["W2"], z["b2"])):
        m.weight.data = torch.tensor(W.T.copy())
        m.bias.data = torch.tensor(b.copy())


def export(net, path):
    d = {}
    lin = [m for m in net.trunk if hasattr(m, "weight")]
    slow = [m for m in net.slow if hasattr(m, "weight")]
    for name, m in (("0", lin[0]), ("1", lin[1])):
        d["W" + name] = m.weight.detach().numpy().T.astype(np.float32)
        d["b" + name] = m.bias.detach().numpy().astype(np.float32)
    d["Wf"] = net.fast.weight.detach().numpy().T.astype(np.float32)
    d["bf"] = net.fast.bias.detach().numpy().astype(np.float32)
    for name, m in (("s0", slow[0]), ("s1", slow[1])):
        d["W" + name] = m.weight.detach().numpy().T.astype(np.float32)
        d["b" + name] = m.bias.detach().numpy().astype(np.float32)
    np.savez(path, **d)


def weights_of(net, tmp):
    export(net, tmp)
    z = np.load(tmp)
    return {k: z[k] for k in z.files}


def fit(net, opt, X, F, D, steps, batch):
    import torch
    import torch.nn.functional as Fn
    n = len(X)
    tot = [0.0, 0.0, 0.0]
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,))
        fast, slow = net(X[idx])
        lf = Fn.smooth_l1_loss(fast, F[idx], beta=0.05)
        ls = Fn.cross_entropy(slow, D[idx])
        loss = lf + W_SLOW * ls
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        tot[0] += float(loss)
        tot[1] += float(lf)
        tot[2] += float(ls)
    return [t / steps for t in tot]


def accuracy(net, X, D, n=200_000):
    import torch
    with torch.no_grad():
        idx = torch.randint(0, len(X), (min(n, len(X)),))
        _, slow = net(X[idx])
        return float((slow.argmax(1) == D[idx]).float().mean())


def cmd_train(args):
    import torch
    from game import raceenv

    torch.set_num_threads(max(args.jobs, 2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    circuits = [c for c in raceenv.circuits() if c not in dd.HOLDOUT]
    print(f"{len(circuits)} training circuits, held out: {', '.join(dd.HOLDOUT)}", flush=True)
    net = build_net()
    init_from_driver(net, TEACH_DRIVE)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    it0, counter = 0, args.seed0
    state_path = out / "one_state.pt"
    if args.resume and state_path.exists():
        st = torch.load(state_path, weights_only=False)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        it0, counter = st["it"], st["counter"]
        print(f"resumed at iteration {it0}", flush=True)
    rng = np.random.default_rng(args.seed0 + it0)
    kinds, kp = list(dd.TRAIN_KINDS), np.array(list(dd.TRAIN_KINDS.values()))
    kp = kp / kp.sum()
    log = open(out / "one_log.jsonl", "a", encoding="utf-8")
    Xs, Fs, Ds = [], [], []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as ex:

        def evaluate(w):
            tasks = [(w, 0.0, c, k, s, False) for c in args.eval_circuits for k in dd.EVAL_KINDS
                     for s in range(args.eval_seeds)]
            return dd.summarise([r[3] for r in ex.map(episode, tasks, chunksize=1)])

        if it0 == 0:
            base = evaluate(None)
            print("TEACHERS  " + dd.fmt(base), flush=True)
            log.write(json.dumps({"it": 0, "teachers": base}) + "\n")
            log.flush()
        for it in range(it0, args.iters):
            t0 = time.time()
            w = weights_of(net, out / "_cur.npz")
            beta = max(0.0, args.beta0 * (1.0 - it / max(args.beta_iters, 1)))
            tasks = []
            for _ in range(args.episodes):
                counter += 1
                tasks.append((w, beta, circuits[int(rng.integers(len(circuits)))],
                              kinds[int(rng.choice(len(kinds), p=kp))], counter, True))
            res = list(ex.map(episode, tasks, chunksize=1))
            t_roll = time.time() - t0
            for X, F, D, _ in res:
                if len(X):
                    Xs.append(X)
                    Fs.append(F)
                    Ds.append(D)
            total = sum(len(x) for x in Xs)
            while total > args.max_samples and len(Xs) > 1:
                total -= len(Xs.pop(0))
                Fs.pop(0)
                Ds.pop(0)
            X = torch.tensor(np.concatenate(Xs))
            F = torch.tensor(np.concatenate(Fs))
            D = torch.tensor(np.concatenate(Ds))
            loss = fit(net, opt, X, F, D, args.steps, args.batch)
            acc = accuracy(net, X, D)
            info = dd.summarise([r[3] for r in res])
            rec = {"it": it + 1, "beta": round(beta, 3), "samples": int(len(X)),
                   "loss": [round(v, 5) for v in loss], "decision_agreement": round(acc, 4),
                   "train": info, "roll_s": round(t_roll, 1), "iter_s": round(time.time() - t0, 1)}
            if (it + 1) % args.eval_every == 0 or it + 1 == args.iters:
                rec["eval"] = evaluate(weights_of(net, out / "_cur.npz"))
            export(net, out / "one_last.npz")
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "it": it + 1,
                        "counter": counter}, state_path)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"it {it + 1:3d}  beta {beta:.2f}  n {len(X):8d}  loss {loss[1]:.4f}+{loss[2]:.3f}  "
                  f"decision agreement {acc:.3f}  {rec['iter_s']:.0f}s (roll {t_roll:.0f}s)\n    train "
                  + dd.fmt(info), flush=True)
            if "eval" in rec:
                print("    EVAL  " + dd.fmt(rec["eval"]), flush=True)
            if args.max_hours and (time.time() - t_start) > args.max_hours * 3600:
                print("max hours reached", flush=True)
                break
    (out / "_cur.npz").unlink(missing_ok=True)


def cmd_eval(args):
    w = None
    if args.net:
        z = np.load(args.net)
        w = {k: z[k] for k in z.files}
    circuits = args.circuits or list(dd.EVAL_CIRCUITS)
    kinds = args.kinds or list(dd.EVAL_KINDS)
    tasks = [(w, 0.0, c, k, s, False) for c in circuits for k in kinds for s in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as ex:
        infos = [r[3] for r in ex.map(episode, tasks, chunksize=1)]
    s = dd.summarise(infos)
    print(("student " + args.net) if args.net else "teachers", dd.fmt(s), flush=True)
    if args.name:
        Path(args.name).write_text(json.dumps({"summary": s, "episodes": infos}, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", default="policy/onenet/run")
    t.add_argument("--iters", type=int, default=60)
    t.add_argument("--episodes", type=int, default=24)
    t.add_argument("--jobs", type=int, default=4)
    t.add_argument("--steps", type=int, default=1500, help="gradient steps per iteration")
    t.add_argument("--batch", type=int, default=4096)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--beta0", type=float, default=1.0, help="share of decisions driven by the teachers at first")
    t.add_argument("--beta-iters", type=int, default=3)
    t.add_argument("--max-samples", type=int, default=6_000_000)
    t.add_argument("--eval-every", type=int, default=3)
    t.add_argument("--eval-seeds", type=int, default=1)
    t.add_argument("--eval-circuits", nargs="*", default=list(dd.EVAL_CIRCUITS))
    t.add_argument("--seed0", type=int, default=400000)
    t.add_argument("--max-hours", type=float, default=0.0)
    t.add_argument("--resume", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--net", default=None, help="weights (.npz); omitted: the teachers")
    e.add_argument("--circuits", nargs="*", default=None)
    e.add_argument("--kinds", nargs="*", default=None)
    e.add_argument("--seeds", type=int, default=2)
    e.add_argument("--jobs", type=int, default=4)
    e.add_argument("--name", default=None)
    args = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
