"""How many decisions a second does the batched env manage?

    python tests/bench.py [--sizes 16 256 1024] [--device cuda]

Reports env-stepping throughput only (no learner), which is the quantity the
CPU trainer is bottlenecked on: it forks seven worker processes that each
step two environments in Python and reaches roughly 250 decisions a second
in total.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "ai_sw"))

from gpuenv.trackgpu import TrackGPU                       # noqa: E402
from gpuenv.vecenv import VecRaceEnv                       # noqa: E402

CPU_BASELINE = 250.0        # decisions/s, measured on the 14-env CPU trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 128, 1024])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    track = TrackGPU("Monza", device)
    print(f"device {device}, Monza, {args.steps} steps per size")

    for b in args.sizes:
        env = VecRaceEnv("Monza", b, device=device, track=track)
        env.reset_done(torch.ones(b, dtype=torch.bool, device=device))
        act = torch.randint(env.act_dim, (b,), device=device)

        for _ in range(5):                       # warm up kernels / caches
            env.step(act)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(args.steps):
            env.step(act)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        dps = args.steps * b / dt
        print(f"  envs {b:5d}  {dps:10,.0f} decisions/s  "
              f"({dps / CPU_BASELINE:6.1f}x the CPU trainer)  "
              f"{dt / args.steps * 1e3:6.1f} ms/step")


if __name__ == "__main__":
    main()
