"""Where a grand prix frame goes: Python (the game's update) against Panda's
cull and draw, in the real game loop with the twenty-car field.

    python tools/perf_gp.py                       # Monza, 25 s of racing
    python tools/perf_gp.py --threading Cull/Draw # Panda's pipelined renderer
    python tools/perf_gp.py --cprofile 15         # top Python costs
    python tools/perf_gp.py --set CAM_FOV_GAIN=0  # any config override

Frame time on this laptop drifts 10-30% run to run (thermals), so compare
variants by alternating runs, not by one pair.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="Monza")
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--secs", type=float, default=25.0)
    ap.add_argument("--from-t", type=float, default=3.0)
    ap.add_argument("--size", type=int, nargs=2, default=(1600, 900))
    ap.add_argument("--threading", default=None,
                    help="Panda threading-model (default: config's); "
                         "'single' for none")
    ap.add_argument("--cprofile", type=int, default=0,
                    help="print this many top functions of the update")
    ap.add_argument("--set", action="append", default=[],
                    help="config override KEY=VALUE")
    ap.add_argument("--quali", action="store_true")
    ap.add_argument("--ab", choices=("hud", "trees", "cars", "post", "aniso"), default=None,
                    help="alternate with this switched off every 120 frames")
    ap.add_argument("--ab-pixels", type=int, default=0,
                    help="alternate RENDER_MAX_PIXELS with this every 120 "
                         "frames, in the same run, and report both")
    ap.add_argument("--series", action="store_true",
                    help="print the mean frame time of every second")
    ap.add_argument("--parts", action="store_true",
                    help="time the main pieces of the game's update")
    ap.add_argument("--gpu-timing", action="store_true",
                    help="with --pstats: GPU timer queries too")
    ap.add_argument("--pstats", action="store_true",
                    help="connect to a PStats server (text-stats) on localhost")
    ap.add_argument("--census", action="store_true",
                    help="count nodes and geoms under each top-level group")
    ap.add_argument("--hide", action="append", default=[],
                    help="hide GeomNodes whose name starts with this "
                         "(repeatable), to price them; re-applied each second "
                         "so a LOD swap cannot bring one back")
    args = ap.parse_args()

    from panda3d.core import loadPrcFileData
    loadPrcFileData("", "sync-video 0")
    if args.pstats:
        loadPrcFileData("", "want-pstats 1")
        loadPrcFileData("", "pstats-host 127.0.0.1")
        if args.gpu_timing:
            loadPrcFileData("", "pstats-gpu-timing 1")
    from game import config
    for kv in args.set:
        k, v = kv.split("=", 1)
        setattr(config, k, type(getattr(config, k))(eval(v)))

    from ursina import Text, Ursina
    if args.threading is not None:
        config.RENDER_THREADING = "" if args.threading == "single" else args.threading
    from game import frameloop
    frameloop.prepare()
    Ursina(window_type="onscreen", size=tuple(args.size),
           development_mode=False, vsync=False)
    import builtins as _b
    frameloop.install(_b.base)
    from game import app as ga
    from game.ui import pick_font
    import builtins
    import __main__
    font = pick_font()
    if font:
        Text.default_font = font
    mode = "quali" if args.quali else "gp"
    ga.SESSION.update(laps=5, mute=True, track=args.circuit, level=args.level,
                      mode=mode)
    ga._build_race(args.circuit, 5, True, intro=False, mode=mode)
    g = ga.GAME
    from game.mintime_driver import MinTimeDriver, available, path_file
    tag = "g90" if available(args.circuit, "g90") else None
    if tag:
        drv = MinTimeDriver(g.track, g.surface, path_file(args.circuit, tag))
    else:
        from game.autopilot import Autopilot
        drv = Autopilot(g.track, g.surface)
    orig = g.read_controls
    g.read_controls = lambda: (drv.controls(g.vehicle)
                               if g.state == ga.RACING else orig())

    render = builtins.render

    def tops():
        # Ursina hangs everything off one "scene" node: look one level in.
        for ch in render.get_children():
            if ch.get_name() == "scene":
                yield from ch.get_children()
            else:
                yield ch
    if args.census:
        # By GeomNode name (digits dropped), with the vertex count: what is
        # actually handed to cull and draw.
        import re
        from collections import defaultdict
        groups = defaultdict(lambda: [0, 0, 0])
        for gn in render.find_all_matches("**/+GeomNode"):
            if gn.is_hidden():
                continue
            node = gn.node()
            key = re.sub(r"[0-9_.]+$", "", node.get_name())[:30] or "(unnamed)"
            g = groups[key]
            g[0] += 1
            g[1] += node.get_num_geoms()
            g[2] += sum(node.get_geom(i).get_vertex_data().get_num_rows()
                        for i in range(node.get_num_geoms()))
        tot = [0, 0, 0]
        for k, (n, geom, verts) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
            tot[0] += n
            tot[1] += geom
            tot[2] += verts
            print(f"  {k:30s} nodes {n:5d}  geoms {geom:5d}  verts {verts:8d}")
        print(f"  total nodes {tot[0]}  geoms {tot[1]}  verts {tot[2]}")
    def apply_hide():
        for prefix in args.hide:
            for gn in render.find_all_matches("**/+GeomNode"):
                if gn.node().get_name().startswith(prefix):
                    gn.hide()
    apply_hide()

    parts = {}
    if args.parts:
        import functools

        def wrap(obj, name, label):
            fn = getattr(obj, name)

            @functools.wraps(fn)
            def timed(*a, **k):
                t0 = time.perf_counter()
                try:
                    return fn(*a, **k)
                finally:
                    parts[label] = parts.get(label, 0.0) + time.perf_counter() - t0
            setattr(obj, name, timed)
        wrap(g, "_field_frame", "field_frame")
        wrap(g, "_draw_hud", "draw_hud")
        wrap(g, "_update_camera", "camera")
        wrap(g, "_tick", "tick")
        wrap(g.car, "sync", "car_sync")
        wrap(g.fx, "update", "fx")
        wrap(g.sound, "update", "sound")
        if g.gp is not None:
            wrap(g.gp, "update", "gp_cars")
        if g.field is not None:
            wrap(g.field, "poll", "poll")
            wrap(g.field, "tick", "send")
        wrap(g.hud, "update", "hud")
        wrap(g, "_field_tower", "tower_rows")
    part_log = []

    upd = {"t": 0.0}
    real_update = ga.update

    def timed_update():
        a = time.perf_counter()
        real_update()
        upd["t"] = time.perf_counter() - a
    __main__.update = timed_update
    __main__.input = ga.input

    from game import post
    ab_state = {"b": False, "n": 0}
    ab_a = config.RENDER_MAX_PIXELS
    ab_t = {False: [], True: []}

    trees = render.find_all_matches("**/trees*")
    texs = [(t, t.get_anisotropic_degree()) for t in render.find_all_textures()]
    print("textures with anisotropy > 2:",
          [(t.get_name(), d) for t, d in texs if d > 2])

    def ab_flip():
        b = ab_state["b"] = not ab_state["b"]
        if args.ab_pixels:
            config.RENDER_MAX_PIXELS = args.ab_pixels if b else ab_a
            if post._FX is not None:
                post._FX.manager.resizeBuffers()
        elif args.ab == "hud":
            g.hud.root.enabled = not b
        elif args.ab == "trees":
            for t in trees:
                (t.hide if b else t.show)()
        elif args.ab == "aniso":
            for t, d in texs:
                t.set_anisotropic_degree(min(d, 2) if b else d)
        elif args.ab == "cars":
            for c in g.gp.cars.values():
                (c.hide if b else c.show)()
            for st in g.light.field_shadow.stand_ins.values():
                (st.hide if b else st.show)()

    prof = None
    frames, ups = [], []
    last = time.perf_counter()
    while True:
        builtins.base.taskMgr.step()
        now = time.perf_counter()
        st = g.session_time if g.state != ga.COUNTDOWN else -1.0
        if args.hide and len(frames) % 60 == 0:
            apply_hide()
        if st >= args.from_t:
            frames.append(now - last)
            ups.append(upd["t"])
            if args.ab_pixels or args.ab:
                ab_state["n"] += 1
                if ab_state["n"] > 10:            # skip the frames after a switch
                    ab_t[ab_state["b"]].append(now - last)
                if ab_state["n"] >= 120:
                    ab_state["n"] = 0
                    ab_flip()
            if args.parts:
                part_log.append(dict(parts))
            if args.cprofile and prof is None:
                import cProfile
                prof = cProfile.Profile()
                prof.enable()
        last = now
        parts.clear()
        if st >= args.secs:
            break
    if prof is not None:
        prof.disable()
    f = np.array(frames) * 1000
    u = np.array(ups) * 1000
    print(f"threading {config.RENDER_THREADING or 'single'}  {args.size[0]}x{args.size[1]}  "
          f"frames {len(f)}")
    print(f"  frame  mean {f.mean():6.2f} ms ({1000 / f.mean():5.1f} fps)  "
          f"p50 {np.percentile(f, 50):6.2f}  p95 {np.percentile(f, 95):6.2f}  "
          f"p99 {np.percentile(f, 99):6.2f}")
    print(f"  python update mean {u.mean():6.2f} ms   rest (cull+draw+tasks) "
          f"{(f - u).mean():6.2f} ms")
    slow = f > np.percentile(f, 95)
    print(f"  slowest 5% of frames: python {u[slow].mean():6.2f} ms, rest "
          f"{(f - u)[slow].mean():6.2f} ms   (others: python "
          f"{u[~slow].mean():6.2f}, rest {(f - u)[~slow].mean():6.2f})")
    print(f"  python update p95 {np.percentile(u, 95):6.2f}  p99 "
          f"{np.percentile(u, 99):6.2f}  max {u.max():6.2f} ms")
    if args.series:
        t = np.cumsum(f) / 1000.0
        sec = t.astype(int)
        print("  per second:", " ".join(f"{f[sec == k].mean():.1f}"
                                         for k in range(sec.max() + 1)))
    if args.ab_pixels or args.ab:
        b_lab = f"B {args.ab_pixels}" if args.ab_pixels else f"B no {args.ab}"
        for k, lab in ((False, f"A {ab_a}" if args.ab_pixels else "A all"),
                       (True, b_lab)):
            v = np.array(ab_t[k]) * 1000
            print(f"  {lab:12s} mean {v.mean():6.2f}  p50 {np.percentile(v, 50):6.2f}  "
                  f"p95 {np.percentile(v, 95):6.2f}  ({len(v)} frames)")
    if part_log:
        keys = sorted({k for d in part_log for k in d})
        slow = f > np.percentile(f, 95)
        print("  piece           mean    p95    p99   (slow 5% mean)")
        for k in keys:
            v = np.array([d.get(k, 0.0) for d in part_log]) * 1000
            print(f"  {k:12s} {v.mean():6.2f} {np.percentile(v, 95):6.2f} "
                  f"{np.percentile(v, 99):6.2f}   ({v[slow].mean():6.2f})")
    if prof is not None:
        import pstats
        pstats.Stats(prof).sort_stats("tottime").print_stats(args.cprofile)
    g.destroy()


if __name__ == "__main__":
    main()
