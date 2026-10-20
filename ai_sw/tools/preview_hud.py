"""Screenshot the in-race HUD, including the cards, without driving a lap.

    python tools/preview_hud.py                  # lights, racing, spectating, paused, finish
    python tools/preview_hud.py --circuit Spa

Getting to the finish card by actually completing three laps is not a workable
way to iterate on it. This drops straight into a race, fakes a plausible set of
lap and sector times for both cars, and shoots every state into
``hud_preview/``.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ursina import (AmbientLight, DirectionalLight, Text, Ursina, Vec3,
                    application, window)

from game import app as ga
from game import config
from game import palette as pal
from game.ui import pick_font

OUT = Path(__file__).resolve().parents[1] / "hud_preview"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default=config.DEFAULT_TRACK)
    ap.add_argument("--size", type=int, nargs=2, default=(1600, 900))
    ap.add_argument("--assist", action="store_true", help="auto steering and pedals on (the assist box)")
    args = ap.parse_args()
    if args.assist:
        from game import settings
        settings.current.auto_steer = settings.current.auto_pedals = True

    app = Ursina(title="FORMULA-AI", size=tuple(args.size), vsync=False,
                 development_mode=False)
    font = pick_font()
    if font:
        Text.default_font = font
    AmbientLight(color=pal.rgb(160, 164, 172))
    DirectionalLight().look_at(Vec3(0.6, -1.0, 0.3))

    ga.SESSION.update(laps=3, mute=True, track=args.circuit)
    ga._build_race(args.circuit, 3, True, intro=False)
    OUT.mkdir(parents=True, exist_ok=True)

    def shot(name):
        from panda3d.core import Filename
        p = OUT / name
        base.win.saveScreenshot(Filename.fromOsSpecific(str(p)))  # noqa: F821
        print(f"   {p.name}")

    state = {"f": 0}

    def fake_race():
        import numpy as np
        g = ga.GAME
        import math
        g.state = 1                      # RACING
        g.vehicle.frozen = False
        g.start_t = g._T_rise1 + 1.0     # start sequence fully done
        # Into sector 3, so the lap-time block shows two sectors done (one
        # purple, one yellow) and the third live. Along the local tangent,
        # not world +z: a car flung sideways off the track reads as a spin.
        j = g.track.sector_bounds()[1] + 12
        tan = g.track.tangent[j]
        g.vehicle.place(g.track.center[j].copy(), math.atan2(tan[0], tan[1]))
        g.vehicle.vel[:] = tan * 55.0    # ~200 km/h
        g.surface.hint = j
        g._snap_camera()
        g.lap_num = 2
        g.last_t = 93.284
        g.best_t = 92.418
        g.session_time = 195.7
        g.lap_start = g.session_time - 63.4
        g.sector = 2
        g.sector_start = g.session_time - 1.15
        g.sectors = [29.950, 32.300, None]
        g.best_sectors = [30.112, 31.905, 30.401]
        if g.ghost is not None:
            gh = g.ghost
            gh.lap_num = 2
            gh.last_t = 91.902
            gh.best_t = 91.902
            gh.best_sectors = [29.980, 31.520, 30.402]
            gh.vehicle.frozen = False
            gh.vehicle.vel[:] = g.track.tangent[0] * 50.0
            # Ghost splits: it got everywhere 0.4 s sooner this lap.
            gh.splits[:] = np.linspace(0.0, 91.9, g.track.count) - 0.4
            gh.splits[0] = 0.0

    def update():
        state["f"] += 1
        f = state["f"]
        ga.update()
        if f == 4:
            from game.loading import Loading
            state["lc"] = Loading(args.circuit, lambda: None)
        elif f == 6:
            shot(f"loading_{args.circuit}.png")
            state["lc"].destroy()
        elif f == 13:
            ga.GAME.start_t = 3.2       # settle done, three lamps lit
        elif f == 15:                   # a frame later: colours reach the GPU
            shot(f"lights_{args.circuit}.png")
        elif f == 20:
            fake_race()
        elif f == 34:
            shot(f"racing_{args.circuit}.png")
        elif f == 35:
            # Two wheels on the grass: the marshal's flag.
            g = ga.GAME
            v = g.vehicle
            i = g.surface.hint
            v.pos[:] = v.pos + g.track.normal[i] * (g.track.w_right[i] + 2.5)
        elif f == 38:
            shot(f"flag_{args.circuit}.png")
            g = ga.GAME
            v = g.vehicle
            i = g.surface.hint
            v.pos[:] = g.track.center[i]
            ga.GAME.on_key("g")              # spectate the AI
        elif f == 42:
            shot(f"spectate_{args.circuit}.png")
            ga.GAME.on_key("g")
        elif f == 44:
            ga.GAME.on_key("escape")         # -> PAUSED
        elif f == 50:
            shot(f"paused_{args.circuit}.png")
            ga.GAME.on_key("escape")         # resume
        elif f == 54:
            ga.GAME.state = 2                # FINISHED
            ga.GAME.vehicle.frozen = True
        elif f == 62:
            shot(f"finish_{args.circuit}.png")
            application.quit()

    import __main__
    __main__.update = update
    app.run()


if __name__ == "__main__":
    main()
