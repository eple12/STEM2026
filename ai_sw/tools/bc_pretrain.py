"""Behaviour-clone the scripted planner into an IQN policy, as a warm start.

Every from-scratch RL run (v10-v17, and the SAC attempts) failed the same way:
the policy never experiences a fast *clean* lap early, so the value function
has nothing to climb towards. The planner (``game.autopilot``) drives every
circuit clean at ~100-125 s. This clones its behaviour -- state -> discrete
action -- into the same IQN network ``train_iqn`` builds, so ``train_iqn.py
--init-from <this>`` starts from a policy that already gets round the lap, and
RL only has to make it quicker.

    python tools/bc_pretrain.py --circuit Monza --pairs 400000 --epochs 8

Writes ``assets/policies/<name>_bc.npz`` in the exact format
``train_iqn.export()`` produces.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game import config, raceline
from game.autopilot import Autopilot
from game.rlenv import RaceEnv
from game.rlpolicy import (ACTIONS, IQN_EMBED, N_ACTIONS, OBS_DIM, PEDAL_LEVELS,
                           STEER_LEVELS)

_STEER = np.asarray(STEER_LEVELS)
_PEDAL = np.asarray(PEDAL_LEVELS)


def _discretise(ctl) -> int:
    """Planner Controls -> nearest discrete action index. ACTIONS is
    [(s, p) for p in PEDAL_LEVELS for s in STEER_LEVELS], so the index is
    p_idx * n_steer + s_idx."""
    si = int(np.argmin(np.abs(_STEER - float(ctl.steer))))
    pedal_cmd = float(ctl.throttle) - float(ctl.brake)
    pi = int(np.argmin(np.abs(_PEDAL - pedal_cmd)))
    return pi * len(_STEER) + si


def collect(circuit: str, n_pairs: int, seed: int):
    env = RaceEnv(circuit, seed=seed, start_at_line=0.15)
    pace, tuning = raceline.load_tuning(env.track)
    line = raceline.load(env.track)
    speed_scale = raceline.load_speed(env.track)
    pilot = Autopilot(env.track, env.surface,
                      pace=pace if pace else config.GHOST_PACE,
                      line=line, tuning=tuning, speed_scale=speed_scale)
    print(f"  planner: pace {pilot.pace:.2f}, "
          f"{'raceline' if line is not None else 'centreline'}, "
          f"{'tuned' if tuning else 'default'} constants", flush=True)

    obs = np.zeros((n_pairs, OBS_DIM), np.float32)
    act = np.zeros(n_pairs, np.int64)
    o = env.reset()
    t0 = time.perf_counter()
    for k in range(n_pairs):
        a = _discretise(pilot.controls(env.vehicle))
        obs[k] = o
        act[k] = a
        o, _, done, _ = env.step(a)
        if done:
            o = env.reset()
        if (k + 1) % 50_000 == 0:
            print(f"  collected {k + 1:,} / {n_pairs:,}  "
                  f"[{time.perf_counter() - t0:.0f} s]", flush=True)
    # action histogram, so a degenerate planner (all-throttle-straight) is
    # obvious before eight epochs of training on it.
    hist = np.bincount(act, minlength=N_ACTIONS)
    print("  action use:", " ".join(f"{h}" for h in hist), flush=True)
    return obs, act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="Monza")
    ap.add_argument("--pairs", type=int, default=400_000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    torch.manual_seed(args.seed)

    print(f"{args.circuit}: collecting {args.pairs:,} planner state-action pairs",
          flush=True)
    obs, act = collect(args.circuit, args.pairs, args.seed)

    mean = obs.mean(0).astype(np.float64)
    std = obs.std(0).astype(np.float64)
    std[std < 1e-4] = 1e-4
    obs_n = (obs - mean) / std

    A, E = IQN_EMBED, 256

    class IQN(nn.Module):
        def __init__(self):
            super().__init__()
            a = nn.LeakyReLU
            self.ff = nn.Sequential(nn.Linear(OBS_DIM, E), a(),
                                    nn.Linear(E, E), a())
            self.phi = nn.Sequential(nn.Linear(A, E), a())
            self.A = nn.Sequential(nn.Linear(E, E), a(), nn.Linear(E, N_ACTIONS))
            self.V = nn.Sequential(nn.Linear(E, E), a(), nn.Linear(E, 1))

        def q_mean(self, x, nq=16):
            b = x.shape[0]
            h = self.ff(x)
            tau = torch.rand(b, nq, 1)
            ar = torch.arange(1, A + 1, dtype=torch.float32)
            emb = self.phi(torch.cos(ar * math.pi * tau))
            mixed = h.unsqueeze(1) * emb
            adv = self.A(mixed)
            v = self.V(mixed)
            q = v + adv - adv.mean(-1, keepdim=True)
            return q.mean(1)                       # (b, n_act)

    net = IQN()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * (len(obs) // args.batch))
    loss_fn = nn.CrossEntropyLoss()

    xo = torch.as_tensor(obs_n, dtype=torch.float32)
    xa = torch.as_tensor(act)
    n = len(obs)
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot = correct = seen = 0
        for i in range(0, n - args.batch, args.batch):
            idx = perm[i:i + args.batch]
            logits = net.q_mean(xo[idx])
            loss = loss_fn(logits, xa[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
            correct += int((logits.argmax(1) == xa[idx]).sum())
            seen += len(idx)
        print(f"  epoch {ep + 1}/{args.epochs}  loss {tot / seen:.3f}  "
              f"train acc {correct / seen:.3f}", flush=True)

    s = net.state_dict()
    w = {
        "ff0.w": s["ff.0.weight"].numpy().T, "ff0.b": s["ff.0.bias"].numpy(),
        "ff1.w": s["ff.2.weight"].numpy().T, "ff1.b": s["ff.2.bias"].numpy(),
        "iqn.w": s["phi.0.weight"].numpy().T, "iqn.b": s["phi.0.bias"].numpy(),
        "A0.w": s["A.0.weight"].numpy().T, "A0.b": s["A.0.bias"].numpy(),
        "A1.w": s["A.2.weight"].numpy().T, "A1.b": s["A.2.bias"].numpy(),
        "V0.w": s["V.0.weight"].numpy().T, "V0.b": s["V.0.bias"].numpy(),
        "V1.w": s["V.2.weight"].numpy().T, "V1.b": s["V.2.bias"].numpy(),
    }
    w = {k: v.astype(np.float32) for k, v in w.items()}
    name = args.out_name or f"{args.circuit}_bc"
    config.RL_POLICY.mkdir(parents=True, exist_ok=True)
    path = config.RL_POLICY / f"{name}.npz"
    np.savez(path, **w, obs_mean=mean.astype(np.float32),
             obs_std=std.astype(np.float32), circuit=args.circuit)
    print(f"\nsaved {path}", flush=True)


if __name__ == "__main__":
    main()
