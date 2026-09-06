"""Screenshot the start menu without having to launch and quit the game.

    python tools/preview_menu.py                 # first + a few rows down
    python tools/preview_menu.py --circuit Spa   # start on a given circuit

Iterating on UI by launching the game, alt-tabbing and squinting is slow, and
a menu bug that only shows on one circuit (a track outline that does not fit,
a caption that overflows its panel) is easy to miss that way. This renders to
``menu_preview/`` so the frames can be compared side by side.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ursina import (AmbientLight, DirectionalLight, Ursina, Vec3, application,
                    window)

from game import config
from game import palette as pal
from game.menu import StartMenu, available_circuits

OUT = Path(__file__).resolve().parents[1] / "menu_preview"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default=config.DEFAULT_TRACK,
                    help="circuit to open on")
    ap.add_argument("--steps", type=int, default=3,
                    help="how many rows to walk down, shooting each")
    ap.add_argument("--size", type=int, nargs=2, default=(1600, 900))
    args = ap.parse_args()

    app = Ursina(title="FORMULA-AI", size=tuple(args.size), vsync=False,
                 development_mode=False)
    window.color = pal.rgb(21, 21, 30)
    AmbientLight(color=pal.rgb(160, 164, 172))
    DirectionalLight().look_at(Vec3(0.6, -1.0, 0.3))

    names = available_circuits()
    if not names:
        raise SystemExit(f"no circuit data under {config.TRACK_DB}")
    menu = StartMenu(names, on_start=lambda n: print("START", n),
                     on_quit=application.quit, initial=args.circuit)
    OUT.mkdir(parents=True, exist_ok=True)

    # Two frames per shot: the change has to be on screen before the
    # framebuffer holds it.
    state = {"f": 0}

    def update():
        state["f"] += 1
        f = state["f"] - 4
        if f < 0:
            return
        idx, phase = divmod(f, 3)
        if idx > args.steps:
            application.quit()
            return
        if phase == 0 and idx > 0:
            menu.on_key("s")
        elif phase == 2:
            from panda3d.core import Filename
            name = menu.names[menu.sel]
            p = OUT / f"{idx:02d}_{name}.png"
            base.win.saveScreenshot(Filename.fromOsSpecific(str(p)))  # noqa: F821
            print(f"   {p.name}")

    print(f"{len(names)} circuits; shooting {args.steps + 1} frames from "
          f"{args.circuit}")
    import __main__
    __main__.update = update
    app.run()


if __name__ == "__main__":
    main()
