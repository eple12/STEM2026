"""Record a circuit's qualifying ghost lap.

The AI drives several flying laps headlessly; the fastest PERFECT one (no
physics step with all four wheels past the white line, no wall contact) is
saved to assets/ghosts/<circuit>.npz, where qualifying replays it.

    python tools/record_ghost.py --circuit Monza
    python tools/record_ghost.py --all            # every circuit with a policy
    python tools/record_ghost.py --circuit Spa --laps 8

Circuits without a trained policy fall back to the planner, as the live ghost
does -- pass them by name if a planner lap is wanted.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game import config, replay, rlpolicy
from game.menu import available_circuits
from game.trackdata import load_track
from game.ui import lap_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", action="append", default=[],
                    help="circuit folder name; repeat for several")
    ap.add_argument("--all", action="store_true",
                    help="every circuit that has a trained policy")
    ap.add_argument("--laps", type=int, default=5,
                    help="flying laps to drive (the first lap from the grid "
                         "is not one of them)")
    ap.add_argument("--attempts", type=int, default=1,
                    help="best-of-N: attempt 0 is the standing-grid start "
                         "above; every further attempt is a rolling start "
                         "150-500 m before the line with --attempt-laps "
                         "laps. Only laps whose LINE-CROSSING state is one "
                         "the AI reaches on an ordinary flying lap are "
                         "eligible (see --start-speed-tol / --start-lat-tol); "
                         "the first crossing of a rolling start almost never "
                         "is, so it is discarded. The fastest eligible CLEAN "
                         "lap of all attempts is kept.")
    ap.add_argument("--attempt-laps", type=int, default=2,
                    help="laps per rolling-start attempt")
    ap.add_argument("--start-speed-tol", type=float, default=0.005,
                    help="a rolling-start lap only counts if it crossed the "
                         "line at most this fraction faster than the fastest "
                         "natural (grid-run) lap did")
    ap.add_argument("--start-lat-tol", type=float, default=0.25,
                    help="...and within this many metres of the natural "
                         "crossing offsets")
    ap.add_argument("--dry-run", action="store_true",
                    help="drive and report, but do not write the ghost lap")
    ap.add_argument("--keep-faster", action="store_true",
                    help="only overwrite an existing ghost lap if faster")
    args = ap.parse_args()

    names = list(args.circuit)
    if args.all:
        names += [n for n in available_circuits()
                  if rlpolicy.available(n) and n not in names]
    if not names:
        ap.error("pass --circuit NAME or --all")

    failed = []
    for name in names:
        print(f"{name}: driving {args.laps} flying laps "
              f"({'policy' if rlpolicy.available(name) else 'planner'})")
        t0 = time.time()
        track = load_track(name)
        try:
            nat = []
            best, laps = replay.record(track, laps=args.laps, starts=nat)
            # What the AI actually crosses the line with after an ordinary
            # flying lap. A rolling-start attempt may only win with a lap that
            # began no faster and no better placed than that -- otherwise its
            # lap time is one the AI could never drive from a normal lap.
            if nat:
                v_max = max(x["speed"] for x in nat)
                lat_lo = min(x["lateral"] for x in nat) - args.start_lat_tol
                lat_hi = max(x["lateral"] for x in nat) + args.start_lat_tol
                hd_hi = max(abs(x["heading"]) for x in nat) + 0.02
                print(f"  natural line crossing: {v_max * 3.6:.1f} km/h max, "
                      f"offset {min(x['lateral'] for x in nat):+.2f}..."
                      f"{max(x['lateral'] for x in nat):+.2f} m")

                def accept(st, v_max=v_max, lo=lat_lo, hi=lat_hi, hd=hd_hi):
                    return (st["speed"] <= v_max * (1.0 + args.start_speed_tol)
                            and lo <= st["lateral"] <= hi
                            and abs(st["heading"]) <= hd)
            else:
                v_max, accept = 60.0, None
            rng = np.random.default_rng(0)
            for a in range(1, args.attempts):
                st = {"back_m": rng.uniform(150.0, 500.0),
                      "speed": v_max * rng.uniform(0.92, 1.0),
                      "lateral": rng.uniform(-1.2, 1.2)}
                print(f"  attempt {a}/{args.attempts - 1}: {st['back_m']:.0f} m "
                      f"before the line at {st['speed'] * 3.6:.0f} km/h, "
                      f"{st['lateral']:+.1f} m off centre")
                rs = []
                b2, l2 = replay.record(track, laps=args.attempt_laps, start=st,
                                       accept=accept, starts=rs)
                for x in rs:
                    print(f"    crossed at {x['speed'] * 3.6:.1f} km/h, "
                          f"{x['lateral']:+.2f} m, heading {x['heading']:+.3f} "
                          f"rad -> {'used' if x['accepted'] else 'not eligible'}")
                laps = laps + l2
                if b2 is not None and (best is None or b2.lap_time < best.lap_time):
                    best = b2
        except Exception as exc:
            # One circuit's broken policy (an observation size from an older
            # build, say) must not cost every circuit after it.
            print(f"{name}: driver failed -- {type(exc).__name__}: {exc}\n")
            failed.append(name)
            continue
        took = time.time() - t0
        if best is None:
            print(f"{name}: no PERFECT lap in {len(laps)} "
                  f"({took:.0f} s) -- nothing saved\n")
            failed.append(name)
            continue
        old = replay.lap_time(name)
        if args.keep_faster and old is not None and old <= best.lap_time:
            print(f"{name}: best {lap_time(best.lap_time)} is not faster "
                  f"than the saved {lap_time(old)} -- kept\n")
            continue
        if args.dry_run:
            print(f"{name}: best clean lap {lap_time(best.lap_time)} "
                  f"(dry run -- nothing saved)\n")
            continue
        replay.save(name, best)
        size = replay.lap_path(name).stat().st_size / 1024
        print(f"{name}: saved {lap_time(best.lap_time)} "
              f"({len(best.frames)} frames, {size:.0f} KB, {took:.0f} s)\n")
    if failed:
        sys.exit(f"no ghost lap for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
