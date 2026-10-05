"""Price the HUD's per-frame update, in isolation.

The overlay is redrawn every frame with a live speed, gear, lap clock and --
in a grand prix -- twenty running gaps on the tower. This builds the HUD in
an offscreen window and calls ``HUD.update`` with values that change every
frame the way a race's do, timing the Python side (the label and shape
rewrites and the buffer uploads it queues) and, separately, the frames it
renders.

    python tools/bench_hud.py                 # grand prix, 20 cars
    python tools/bench_hud.py --quali
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from panda3d.core import loadPrcFileData

loadPrcFileData("", "sync-video 0")


def standings(n, frame, quali):
    from game.ui import AMBER, RED
    rows = []
    for p in range(n):
        gap = "LEADER" if p == 0 else f"+{0.731 * p + 0.013 * ((frame + p) % 70):.3f}"
        rows.append(dict(pos=p + 1, tla=f"C{p:02d}", name=f"DRIVER {p}",
                         col=AMBER if p == 5 else RED, best=90.0 + p * 0.1,
                         player=p == 5, pen=5.0 if p == 7 else 0.0, gap=gap,
                         purple=p == 3, idx=p, status="", change=0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="Monza")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--cars", type=int, default=20)
    ap.add_argument("--quali", action="store_true")
    args = ap.parse_args()

    from ursina import Text, Ursina
    app = Ursina(window_type="offscreen", size=(1600, 900), development_mode=False)
    from game.ui import pick_font
    font = pick_font()
    if font:
        Text.default_font = font
    from game.hud import HUD
    from game.trackdata import load_track

    track = load_track(args.circuit)
    hud = HUD(track, 3, "quali" if args.quali else "gp")
    lo, hi = track.bounds()
    upd, rend = [], []
    for f in range(args.frames):
        speed = 150.0 + 120.0 * np.sin(f / 37.0)
        kw = dict(speed_kmh=speed, speed_frac=speed / 340.0, lap=2,
                  cur_t=31.0 + f / 60.0, last_t=90.123, best_t=89.876,
                  session_t=200.0 + f / 60.0,
                  sectors=[(30.1, "purple"), (31.2, "yellow"), (None, "live")],
                  standings=standings(args.cars, f, args.quali),
                  car_xz=track.center[(f * 3) % track.count],
                  throttle=0.5 + 0.5 * np.sin(f / 9.0), brake=0.0, steer=0.1,
                  slip=0.02, tc_cut=0.0, esc_cut=0.0, dt=1 / 60,
                  delta=(-0.123 + f * 0.001) if args.quali else None)
        others = getattr(hud, "set_field", None)
        t0 = time.perf_counter()
        if others is not None:
            hud.set_field([(track.center[(f * 3 + 17 * k) % track.count], (1, 0, 0, 1))
                           for k in range(args.cars - 1)])
        hud.update(**kw)
        t1 = time.perf_counter()
        base.graphicsEngine.renderFrame()  # noqa: F821
        t2 = time.perf_counter()
        if f >= 60:
            upd.append(t1 - t0)
            rend.append(t2 - t1)
    upd = np.array(upd) * 1000
    rend = np.array(rend) * 1000
    print(f"HUD.update   mean {upd.mean():5.3f} ms   p95 {np.percentile(upd, 95):5.3f} ms")
    print(f"renderFrame  mean {rend.mean():5.3f} ms   (whole offscreen frame, HUD only scene)")
    app.destroy() if hasattr(app, "destroy") else None


if __name__ == "__main__":
    main()
