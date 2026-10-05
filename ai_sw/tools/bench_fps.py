"""Measure frame times while the autopilot drives a real lap.

vsync is off on purpose: with it on the number is just the monitor's refresh
rate and says nothing about headroom. What matters for an exhibition machine is
whether the *worst* frames still clear the refresh interval, so this reports the
1% low and the single worst frame alongside the mean.

    python tools/bench_fps.py                       # default circuit, 900 frames
    python tools/bench_fps.py --circuit Spa --unlit # compare without the shader
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from panda3d.core import loadPrcFileData

# Ursina's vsync=False does not reach Panda: sync-video stays true and every
# reading is then the present rate, not the frame cost. This has to be set
# before Ursina imports/creates the window.
loadPrcFileData('', 'sync-video 0')

from ursina import Text, Ursina, application, window

from game import app as ga
from game import config
from game.ui import pick_font


def _run_baseline(app, args):
    """Frame time with nothing but a cube -- the machine's floor right now."""
    state = {"n": 0, "last": None, "dt": []}

    def update():
        now = time.perf_counter()
        if state["last"] is not None:
            state["n"] += 1
            if state["n"] > args.warmup:
                state["dt"].append(now - state["last"])
        state["last"] = now
        if state["n"] >= args.warmup + args.frames:
            dt = np.array(state["dt"]) * 1000.0
            print(f"\nbaseline (empty scene) @ {args.size[0]}x{args.size[1]}")
            print(f"  mean          {1000.0 / dt.mean():7.1f} fps   "
                  f"({dt.mean():5.2f} ms)")
            application.quit()

    import __main__
    __main__.update = update
    app.run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default=config.DEFAULT_TRACK)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--warmup", type=int, default=120)
    ap.add_argument("--size", type=int, nargs=2, default=list(config.WINDOW_SIZE))
    ap.add_argument("--unlit", action="store_true",
                    help="drop the sunset shader, to price it")
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--hud-frozen", action="store_true",
                    help="draw the HUD but never change its text")
    ap.add_argument("--no-minimap", action="store_true")
    ap.add_argument("--baseline", action="store_true",
                    help="measure an empty scene instead of the race. Absolute "
                         "fps depends on machine state -- this session drifted "
                         "34%% over an afternoon -- so any before/after "
                         "comparison needs a baseline taken alongside it.")
    ap.add_argument("--bias", type=float, default=None)
    ap.add_argument("--depth", type=float, default=None)
    ap.add_argument("--no-gc", action="store_true",
                    help="freeze the setup objects and stop cyclic collection")
    ap.add_argument("--hud-offscreen", action="store_true",
                    help="keep every HUD node alive and updated, but off screen "
                         "-- separates rasterising cost from per-entity CPU cost")
    ap.add_argument("--shadow-res", type=int, default=None)
    ap.add_argument("--shadow-samples", type=int, default=None)
    ap.add_argument("--no-shadow", action="store_true")
    ap.add_argument("--ab-static-casters", action="store_true",
                    help="alternate the static roadside in and out of the "
                         "shadow pass every --ab-block frames and report both "
                         "bins. Absolute runs cannot price this: two identical "
                         "runs of it drifted 2.2 ms apart, which is larger "
                         "than the effect. Interleaving cancels the drift.")
    ap.add_argument("--ab-block", type=int, default=60)
    ap.add_argument("--ab-post", action="store_true",
                    help="alternate the HDR camera chain (post.py) on and off")
    ap.add_argument("--ab-post-b", action="append", default=[],
                    help="with --ab-post: KEY=VALUE config overrides for the "
                         "'off' bin, which then runs the chain with them "
                         "instead of without it")
    ap.add_argument("--ab-fx", action="store_true",
                    help="alternate the sparks/smoke update on and off")
    ap.add_argument("--ab-hide", default=None,
                    choices=("foliage", "scenery", "track", "hud", "cars"),
                    help="alternate one group of the scene shown and hidden")
    ap.add_argument("--set", action="append", default=[],
                    help="KEY=VALUE config override for the whole run")
    ap.add_argument("--ghost", action="store_true",
                    help="start the AI as well -- what a real race costs")
    ap.add_argument("--ab-bake", action="store_true",
                    help="alternate the baked far-field shadow lookup on and "
                         "off, to price the extra sample")
    ap.add_argument("--shadow-car-only", action="store_true",
                    help="keep shadows, but take the static roadside out of "
                         "the shadow pass -- the ceiling on what baking the "
                         "static shadows could ever save")
    ap.add_argument("--stand-coverage", type=float, default=None,
                    help="fraction of the wall carrying grandstands, to price "
                         "them against a near-empty circuit")
    args = ap.parse_args()
    import ast
    for kv in args.set:
        k, v = kv.split("=", 1)
        setattr(config, k, ast.literal_eval(v))
    b_over = {}
    for kv in args.ab_post_b:
        k, v = kv.split("=", 1)
        b_over[k] = ast.literal_eval(v)

    if args.shadow_res:
        config.SHADOW_RESOLUTION = (args.shadow_res, args.shadow_res)
    if args.shadow_samples:
        config.SHADOW_SAMPLES = args.shadow_samples
    if args.bias is not None:
        config.SHADOW_BIAS = args.bias
    if args.depth is not None:
        config.SHADOW_DEPTH = args.depth
    if args.stand_coverage is not None:
        config.STAND_COVERAGE = args.stand_coverage

    app = Ursina(title="FORMULA-AI bench", size=tuple(args.size), vsync=False,
                 development_mode=False)
    font = pick_font()
    if font:
        Text.default_font = font

    if args.baseline:
        from ursina import Entity, color as ucolor
        Entity(model="cube", color=ucolor.azure, z=5)

    ga.SESSION.update(laps=99, mute=True, track=args.circuit)
    game = None
    if not args.baseline:
        ga._build_race(args.circuit, 99, True)
        game = ga.GAME

    if game is None:
        _run_baseline(app, args)
        return

    if args.unlit:
        from ursina.shaders import unlit_shader
        for e in game.light._lit:
            e.shader = unlit_shader
    if args.no_hud:
        game.hud.root.enabled = False
        game._reveal_hud = False
    if args.hud_frozen:
        game.hud.update = lambda **kw: None
    if args.hud_offscreen:
        game.hud.root.y = 40
    if args.no_minimap:
        game.hud.minimap.enabled = False
    if args.shadow_car_only:
        for e in game.scenery:
            if e is not None:
                e.hide(game.light.SHADOW_MASK)
    if args.no_shadow:
        game.light.sun.shadows = False
        from ursina import scene as _scene
        _scene.set_shader_input("shadow_strength", 0.0)

    # Drive properly: an idle car sits in one place and never loads the parts
    # of the circuit that cost the most.
    from game.autopilot import Autopilot
    pilot = Autopilot(game.track, game.surface)
    game.read_controls = lambda: pilot.controls(game.vehicle)
    game.state = 1                      # RACING
    game.vehicle.frozen = False
    if game.field is not None:
        # A grand prix with the twenty-car field: start it too, or the other
        # nineteen sit on the grid for the whole run.
        game.field.event("go")
    if args.ghost and game.ghost is not None:
        game.ghost.start()

    if args.no_gc:
        import gc
        gc.collect()
        gc.freeze()
        gc.disable()

    state = {"n": 0, "last": None, "dt": [], "ab_on": True, "skip": 0,
             "ab": {True: [], False: []}}

    def _bake_lookup(on: bool):
        from ursina import scene as _s
        _s.set_shader_input("bake_ready", 1.0 if on else 0.0)

    def _group():
        if args.ab_hide == "foliage":
            return [e for e in game.scenery if getattr(e, "is_foliage", False)]
        if args.ab_hide == "scenery":
            return [e for e in game.scenery
                    if not getattr(e, "is_foliage", False)]
        if args.ab_hide == "track":
            return list(game.world.scene.entities)
        if args.ab_hide == "hud":
            return [game.hud.root]
        if args.ab_hide == "cars":
            return [game.car] + ([game.ghost.car] if game.ghost else [])
        return []

    def _show(on: bool):
        for e in _group():
            if on:
                e.show()
            else:
                e.hide()

    _fx_update = game.fx.update

    def _fx(on: bool):
        game.fx.update = _fx_update if on else (lambda *a, **k: None)

    def _post(on: bool):
        from game import post
        post.disable()
        if on:
            post.enable()
        elif b_over:
            keep = {k: getattr(config, k) for k in b_over}
            for k, v in b_over.items():
                setattr(config, k, v)
            post.enable()
            for k, v in keep.items():
                setattr(config, k, v)
        else:
            from ursina import scene as _s
            _s.set_shader_input("direct_out", 1.0)

    def _static_casters(on: bool):
        for e in game.scenery:
            if e is None:
                continue
            if on:
                e.show(game.light.SHADOW_MASK)
            else:
                e.hide(game.light.SHADOW_MASK)

    def update():
        ab = (args.ab_static_casters or args.ab_bake or args.ab_post
              or args.ab_hide or args.ab_fx)
        if ab:
            want = (state["n"] // max(args.ab_block, 1)) % 2 == 0
            if want != state["ab_on"]:
                (_fx if args.ab_fx else _show if args.ab_hide else
                 _post if args.ab_post else
                 _bake_lookup if args.ab_bake else _static_casters)(want)
                state["ab_on"] = want
                # The frame that flips the state pays for the flip.
                state["skip"] = 2
        t_up = time.perf_counter()
        game.update()
        state.setdefault("up", []).append(time.perf_counter() - t_up)
        now = time.perf_counter()
        if state["last"] is not None:
            state["n"] += 1
            if state["n"] > args.warmup:
                state["dt"].append(now - state["last"])
                if ab:
                    if state["skip"] > 0:
                        state["skip"] -= 1
                    else:
                        state["ab"][state["ab_on"]].append(now - state["last"])
        state["last"] = now

        if state["n"] >= args.warmup + args.frames:
            dt = np.array(state["dt"]) * 1000.0     # ms
            fps = 1000.0 / dt
            print(f"\n{args.circuit} @ {args.size[0]}x{args.size[1]}"
                  f"{'  (unlit)' if args.unlit else '  (sunset shader)'}"
                  f"{'  (no HUD)' if args.no_hud else ''}"
                  f"{'  (HUD frozen)' if args.hud_frozen else ''}"
                  f"{'  (HUD offscreen)' if args.hud_offscreen else ''}"
                  f"{'  (no GC)' if args.no_gc else ''}"
                  f"{'  (no minimap)' if args.no_minimap else ''}"
                  f"  shadow={'off' if args.no_shadow else f'{config.SHADOW_RESOLUTION[0]}/{config.SHADOW_SAMPLES}'}"
                  f" bias={config.SHADOW_BIAS} depth={config.SHADOW_DEPTH}"
                  f", vsync off")
            print(f"  frames        {len(dt)}")
            print(f"  mean          {fps.mean():7.1f} fps   "
                  f"({dt.mean():5.2f} ms)")
            print(f"  median        {np.median(fps):7.1f} fps   "
                  f"({np.median(dt):5.2f} ms)")
            print(f"  1% low        {np.percentile(fps, 1):7.1f} fps   "
                  f"({np.percentile(dt, 99):5.2f} ms)")
            print(f"  worst frame   {fps.min():7.1f} fps   "
                  f"({dt.max():5.2f} ms)")
            print(f"  over 16.7 ms  {100.0 * (dt > 16.7).mean():5.1f} % of frames")
            up = np.array(state["up"][-len(dt):]) * 1000.0
            print(f"  game.update   {up.mean():5.2f} ms mean  (the rest -- Panda's "
                  f"cull/draw/flip and Ursina -- {dt.mean() - up.mean():5.2f} ms)")
            if ab:
                on = np.array(state["ab"][True]) * 1000.0
                off = np.array(state["ab"][False]) * 1000.0
                print()
                print(f"  interleaved, {args.ab_block}-frame blocks:")
                label_on = ("effects on        " if args.ab_fx else
                            f"{args.ab_hide} shown " if args.ab_hide else
                            "camera chain on " if args.ab_post else
                            "baked lookup on " if args.ab_bake
                            else "roadside casts shadows")
                label_off = ("effects off       " if args.ab_fx else
                             f"{args.ab_hide} hidden" if args.ab_hide else
                             "camera chain off" if args.ab_post else
                             "baked lookup off" if args.ab_bake
                             else "roadside does not   ")
                print(f"    {label_on}     {on.mean():5.2f} ms  "
                      f"(median {np.median(on):5.2f}, n={len(on)})")
                print(f"    {label_off}     {off.mean():5.2f} ms  "
                      f"(median {np.median(off):5.2f}, n={len(off)})")
                what = ("the effects cost" if args.ab_fx else
                        f"{args.ab_hide} costs" if args.ab_hide else
                        "the camera chain costs   " if args.ab_post else
                        "the far-field lookup costs" if args.ab_bake
                        else "static casters cost      ")
                print(f"    {what} {on.mean() - off.mean():5.2f} ms")
            application.quit()

    import __main__
    __main__.update = update
    app.run()


if __name__ == "__main__":
    main()
