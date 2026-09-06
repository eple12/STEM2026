"""Find a racing line for a circuit and save it for the AI to drive.

Two stages. ``min_curvature`` solves for the smoothest line inside the track --
deterministic, no starting guess, and already worth most of the gain. Then
CMA-ES searches from there on the modelled lap time, which is where "smoothest"
and "fastest" part company.

Both work on the analytic speed profile, not on simulated laps: a modelled lap
costs about 20 ms against 13.8 s simulated, and that ratio is what makes the
search affordable at all. The result is then *checked* by simulating, because
a line the planner cannot actually follow is not a faster line.

    python tools/train_raceline.py                 # every circuit, both stages
    python tools/train_raceline.py --circuit Monza --generations 400
    python tools/train_raceline.py --no-refine     # stage 1 only, seconds
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from game import config, raceline
from game.autopilot import Autopilot
from game.surface import Surface
from game.trackdata import load_track
from game.vehicle import Vehicle

DT = 1.0 / 120.0


def simulate(track, line, pace: float, limit: float = 400.0):
    """(best lap in seconds, widest excursion) driving *line* for real."""
    surf = Surface(track)
    v = Vehicle()
    v.frozen = False
    pos, yaw = track.start_pose()
    v.place(pos, yaw)
    pilot = Autopilot(track, surf, pace=pace, line=line)

    n = track.count
    t, armed, start, best, widest = 0.0, False, None, None, 0.0
    while t < limit and best is None:
        v.step(pilot.controls(v), DT, surf)
        t += DT
        if not np.isfinite(v.pos).all():
            return None, float("inf")
        i, off = surf.progress(v.pos)
        widest = max(widest, abs(off))
        if 0.4 * n <= i <= 0.6 * n:
            armed = True
        elif armed and i < 0.1 * n:
            armed = False
            if start is not None:
                best = t - start
            start = t
    return best, widest


def circuits(names):
    if names:
        return names
    from game.menu import available_circuits
    return available_circuits()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", action="append", default=None)
    ap.add_argument("--pace", type=float, default=config.GHOST_PACE)
    ap.add_argument("--generations", type=int, default=config.RACELINE_GENERATIONS)
    ap.add_argument("--controls", type=int, default=config.RACELINE_CONTROLS)
    ap.add_argument("--no-refine", action="store_true",
                    help="stop after the least-squares line")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the simulated lap (it costs ~14 s per circuit)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for name in circuits(args.circuit):
        track = load_track(name)
        zero = np.zeros(track.count)
        base_model = raceline.lap_time(track, zero, args.pace)

        t0 = time.perf_counter()
        offset = raceline.min_curvature(track)
        mc_model = raceline.lap_time(track, offset, args.pace)
        print(f"\n{name}  ({track.count} samples, {track.length:.0f} m)")
        print(f"  centreline      model {base_model:7.2f} s")
        print(f"  min-curvature   model {mc_model:7.2f} s  "
              f"({mc_model - base_model:+6.2f})   [{time.perf_counter() - t0:.1f} s]")

        if not args.no_refine:
            t0 = time.perf_counter()

            def report(gen, best, sigma):
                print(f"    gen {gen:4d}   model {best:7.2f} s   sigma {sigma:5.3f}",
                      flush=True)

            offset, model = raceline.refine(
                track, offset, args.pace, controls=args.controls,
                generations=args.generations, seed=args.seed, report=report)
            print(f"  refined         model {model:7.2f} s  "
                  f"({model - mc_model:+6.2f} on min-curvature)   "
                  f"[{time.perf_counter() - t0:.1f} s]")

        raceline.save(track, offset)
        lo, hi = raceline.bounds(track)
        print(f"  saved  |offset| mean {np.abs(offset).mean():.2f} m  "
              f"max {np.abs(offset).max():.2f} m  (limit {hi.max():.2f})")

        if not args.no_verify:
            t0 = time.perf_counter()
            was, _ = simulate(track, None, args.pace)
            now, widest = simulate(track, raceline.Line(track, offset), args.pace)
            if now is None:
                print("  SIMULATED: the car could not complete a lap on this line")
            else:
                print(f"  simulated       centreline {was:7.2f} s  ->  "
                      f"line {now:7.2f} s  ({now - was:+6.2f})   "
                      f"widest {widest:4.1f} m   [{time.perf_counter() - t0:.0f} s]")


if __name__ == "__main__":
    main()
