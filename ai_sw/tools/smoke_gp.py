"""Run the grand prix in the real game loop, offscreen, and photograph it.

Builds the race the menu would (twenty cars, the field in its worker
process), skips the start film, lets a plan follower drive the player's car,
steps Ursina's loop frame by frame, and saves screenshots at the times asked
for -- the grid, the start, the first braking zone, mid-race -- plus the
frame times and how far the field's snapshots trailed the game's clock.

    python tools/smoke_gp.py                          # Monza, level 6
    python tools/smoke_gp.py --level 2 --shots 3 12 40
    python tools/smoke_gp.py --quali                  # qualifying instead
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from panda3d.core import loadPrcFileData

OUT = Path(__file__).resolve().parents[1] / "sim_preview"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="Monza")
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--secs", type=float, default=40.0,
                    help="game seconds to run; 0 = until the player has "
                         "finished (and four seconds more)")
    ap.add_argument("--shots", type=float, nargs="*", default=[-1.0, 2.0, 15.0, 30.0])
    ap.add_argument("--size", type=int, nargs=2, default=(1600, 900))
    ap.add_argument("--quali", action="store_true")
    ap.add_argument("--window", action="store_true",
                    help="a real window (frame times then include the swap)")
    ap.add_argument("--cam", default=None, help="camera mode to use")
    args = ap.parse_args()
    loadPrcFileData("", "sync-video 0")

    from ursina import Text, Ursina
    from game import frameloop
    frameloop.prepare()
    Ursina(window_type="onscreen" if args.window else "offscreen",
                 size=tuple(args.size), development_mode=False, vsync=False)
    import builtins as _b
    frameloop.install(_b.base)
    from game import app as ga
    from game import config
    from game.ui import pick_font
    font = pick_font()
    if font:
        Text.default_font = font
    ga.SESSION.update(laps=args.laps, mute=True, track=args.circuit,
                      level=args.level, mode="quali" if args.quali else "gp")
    t0 = time.perf_counter()
    ga._build_race(args.circuit, args.laps, True, intro=False,
                   mode="quali" if args.quali else "gp")
    print(f"built in {time.perf_counter() - t0:.1f} s")
    g = ga.GAME
    import __main__
    __main__.update = ga.update
    __main__.input = ga.input
    if args.cam:
        g.cam_idx = config.CAM_MODES.index(args.cam)
    from game.mintime_driver import MinTimeDriver, available, path_file
    if available(args.circuit, "g90"):
        drv = MinTimeDriver(g.track, g.surface, path_file(args.circuit, "g90"))
    else:
        from game.autopilot import Autopilot
        drv = Autopilot(g.track, g.surface)
    orig = g.read_controls

    def controls():
        c = orig()
        if g.state in (ga.RACING,):
            return drv.controls(g.vehicle)
        return c
    g.read_controls = controls
    OUT.mkdir(exist_ok=True)
    tag = f"{args.circuit}_{'q' if args.quali else 'gp'}_L{args.level}"
    shots = sorted(args.shots)
    frames, lags = [], []
    last = time.perf_counter()
    import builtins
    from panda3d.core import Filename
    while True:
        builtins.base.taskMgr.step()
        b = time.perf_counter()
        frames.append(b - last)
        last = b
        st = g.session_time if g.state != ga.COUNTDOWN else -1.0
        if g.field is not None and g._snap is not None and g.state == ga.RACING:
            lags.append(g.session_time - g._snap["t"])
        if shots and (st >= shots[0] or (shots[0] < 0 and g.state == ga.COUNTDOWN
                                         and g.start_t > g._T_build1)):
            name = OUT / f"{tag}_{shots[0]:+05.1f}.png"
            builtins.base.win.saveScreenshot(Filename.fromOsSpecific(str(name)))
            print(f"  shot {name.name}  (state {g.state}, t {st:.1f})")
            shots.pop(0)
        if (args.secs > 0 and st >= args.secs) or (args.secs <= 0 and g.state == ga.FINISHED
                                and g.session_time - g.finish_t > 4.0):
            builtins.base.win.saveScreenshot(Filename.fromOsSpecific(
                str(OUT / f"{tag}_end.png")))
            print(f"  shot {tag}_end.png  (state {g.state})")
            break
    f = np.array(frames[30:]) * 1000
    print(f"frames {len(f)}  mean {f.mean():.2f} ms  p95 {np.percentile(f, 95):.2f}  "
          f"max {f.max():.2f}")
    if lags:
        l = np.array(lags) * 1000
        print(f"field snapshot lag: mean {l.mean():.1f} ms  p95 {np.percentile(l, 95):.1f}  "
              f"max {l.max():.1f}")
    if g.field is not None and g._snap is not None:
        print("worker wall/sim", round(g._snap.get("wall_ratio", 0), 2),
              " overload", g._snap.get("overload"))
        tower = g._field_tower()
        print("tower:", " ".join(f"{e['pos']}:{e['tla']}{'*' if e['player'] else ''}"
                                 f"{e['gap']}" for e in tower[:8]))
        print("race control:", ([g._rc_cur[1]] if g._rc_cur else [])
              + [m[1] for m in g._rc_queue])
    g.destroy()


if __name__ == "__main__":
    main()
