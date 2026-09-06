"""Drive a trained policy for real and report the lap time.

Deterministic: the mean action, no exploration noise. Training reward is
progress per step, which is the right thing to optimise but not a number
anyone can compare -- this turns it into the one that matters, next to the
same circuit's hand-written planner and its CMA-ES trajectory.

    python tools/eval_rl.py --circuit Monza
    python tools/eval_rl.py --circuit Monza --laps 3 --verbose
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from game import config
from game.rlenv import RaceEnv
from game.rlpolicy import mlp_forward, normalise


def load_policy(circuit: str):
    path = config.RL_POLICY / f"{circuit}.npz"
    if not path.exists():
        raise SystemExit(f"no trained policy at {path}")
    data = np.load(path, allow_pickle=False)
    weights = {k: data[k] for k in data.files if k[0] in "wb" and k[1:].isdigit()}
    return weights, data["obs_mean"], data["obs_std"]


def drive(circuit: str, laps: int = 1, verbose: bool = False):
    weights, mean, std = load_policy(circuit)
    env = RaceEnv(circuit, randomise_start=False)
    obs = env.reset()

    n = env.track.count
    times, armed, start = [], False, None
    t = 0.0
    off = wall = steps = 0
    widest = 0.0
    limit = 400.0 * (laps + 1)
    while t < limit and len(times) < laps:
        act = mlp_forward(weights, normalise(obs[None], mean, std))[0]
        obs, _, done, _ = env.step(act)
        v = env.vehicle
        t += env.dt
        steps += 1
        if done:
            return None, {"reason": "episode ended", "at": t}
        if not v.on_track:
            off += 1
        if v.hit_wall:
            wall += 1
        i, lateral = env.surface.progress(v.pos)
        widest = max(widest, abs(lateral))
        if 0.4 * n <= i <= 0.6 * n:
            armed = True
        elif armed and i < 0.1 * n:
            armed = False
            if start is not None:
                times.append(t - start)
                if verbose:
                    print(f"    lap {len(times)}: {times[-1]:.3f} s")
            start = t
    if not times:
        return None, {"reason": "never completed a lap"}
    return min(times), {"off": 100.0 * off / steps, "walls": wall,
                        "widest": widest, "laps": len(times)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="Monza")
    ap.add_argument("--laps", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    best, info = drive(args.circuit, args.laps, args.verbose)
    if best is None:
        print(f"{args.circuit}: no lap -- {info['reason']}")
        return
    print(f"{args.circuit}: best {best:.3f} s over {info['laps']} laps   "
          f"off track {info['off']:.1f}%   walls {info['walls']}   "
          f"widest {info['widest']:.1f} m")


if __name__ == "__main__":
    main()
