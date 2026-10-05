"""Train the race driver's learned decision layer (game/raceai.py).

Stage 2 -- behaviour cloning: teach a network to make the rule layer's decisions,
so everything after starts from a driver that races as the current one does.

    python tools/raceai_collect.py                              # the pairs
    python tools/train_raceai.py bc --data policy/raceai/rules_pairs.npz
    python tools/race_metrics.py --policy policy/raceai/bc.npz --all-cars --name bc

The network is an MLP over ``raceai.OBS_DIM`` observations with ``N_ACTIONS``
outputs, written as the ``.npz`` ``raceai.Policy`` reads: ``mean``/``std`` (the
observation's normalisation), ``layers``, ``W0..``, ``b0..``. The same file is
what the game loads and what the RL stage starts from (``--init-from``).

Class weights are the inverse square root of an action's frequency by default: a plain
cross-entropy teaches "stay on the line" and little else, and the rare
decisions -- pull out, cover, lift -- are the whole point.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "policy" / "raceai"


def build(hidden, obs_dim, n_actions):
    import torch.nn as nn
    layers, d = [], obs_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, n_actions))
    return nn.Sequential(*layers)


def export(net, mean, std, path):
    import torch
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    w = {"mean": mean.astype(np.float32), "std": std.astype(np.float32),
         "layers": np.array(len(lin))}
    for i, m in enumerate(lin):
        w[f"W{i}"] = m.weight.detach().cpu().numpy().T.astype(np.float32)   # x @ W
        w[f"b{i}"] = m.bias.detach().cpu().numpy().astype(np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **w)


def load_into(net, path):
    import torch
    z = np.load(path)
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    for i, m in enumerate(lin):
        m.weight.data = torch.tensor(z[f"W{i}"].T.copy())
        m.bias.data = torch.tensor(z[f"b{i}"].copy())
    return z["mean"], z["std"]


def bc(args):
    import torch
    import torch.nn.functional as F
    from game import raceai

    torch.manual_seed(args.seed)
    z = np.load(args.data)
    obs, act = z["obs"].astype(np.float32), z["act"].astype(np.int64)
    n = len(act)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    n_val = max(int(n * 0.1), 1)
    val, tr = order[:n_val], order[n_val:]
    mean = obs[tr].mean(0)
    std = obs[tr].std(0) + 1e-3
    x = torch.tensor((obs - mean) / std)
    y = torch.tensor(act)
    freq = np.bincount(act[tr], minlength=raceai.N_ACTIONS).astype(np.float64) + 1.0
    weight = torch.tensor((freq.sum() / freq) ** args.weight_pow, dtype=torch.float32)
    weight = weight / weight.mean()
    net = build(args.hidden, raceai.OBS_DIM, raceai.N_ACTIONS)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    print(f"{n} pairs ({len(tr)} train / {len(val)} val), {raceai.OBS_DIM} -> "
          f"{args.hidden} -> {raceai.N_ACTIONS}", flush=True)
    t0 = time.time()
    for ep in range(args.epochs):
        net.train()
        perm = torch.tensor(rng.permutation(tr))
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            loss = F.cross_entropy(net(x[idx]), y[idx], weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        net.eval()
        with torch.no_grad():
            pv = net(x[val]).argmax(1)
            acc = float((pv == y[val]).float().mean())
            # accuracy on the decisions that are not "stay": the ones that matter
            m = y[val] != raceai.DEFAULT_ACTION
            acc_move = float((pv[m] == y[val][m]).float().mean()) if m.any() else float("nan")
            # lane only, ignoring pace
            lane_acc = float(((pv // raceai.N_PACES) == (y[val] // raceai.N_PACES)).float().mean())
        print(f"  epoch {ep + 1:2d}/{args.epochs}  loss {tot / len(perm):.4f}  "
              f"val acc {acc:.3f}  lane acc {lane_acc:.3f}  non-default acc {acc_move:.3f}  "
              f"{time.time() - t0:.0f}s", flush=True)
    path = Path(args.out)
    export(net, mean, std, path)
    print("->", path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bc", help="behaviour-clone the rule layer")
    b.add_argument("--data", default=str(OUT / "rules_pairs.npz"))
    b.add_argument("--out", default=str(OUT / "bc.npz"))
    b.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    b.add_argument("--epochs", type=int, default=20)
    b.add_argument("--batch", type=int, default=512)
    b.add_argument("--lr", type=float, default=2e-3)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--weight-pow", type=float, default=0.5,
                   help="class weight = (1/frequency)^this; 1.0 favours the rare decisions")
    args = ap.parse_args()
    if args.cmd == "bc":
        bc(args)


if __name__ == "__main__":
    main()
