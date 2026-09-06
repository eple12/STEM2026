"""Shoot the menu <-> race transition: the loading card and its mid-fade frame,
in both directions, plus the start-lights sequence off a real menu launch.

    python tools/preview_transition.py --circuit Monza
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ursina import Text, Ursina, application

from game import app as ga
from game import config
from game.ui import pick_font

OUT = Path(__file__).resolve().parents[1] / "hud_preview"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default=config.DEFAULT_TRACK)
    ap.add_argument("--size", type=int, nargs=2, default=(1600, 900))
    args = ap.parse_args()

    Ursina(title="FORMULA-AI", size=tuple(args.size), vsync=False,
           development_mode=False)
    font = pick_font()
    if font:
        Text.default_font = font
    OUT.mkdir(parents=True, exist_ok=True)

    ga.SESSION.update(laps=3, mute=True, track=args.circuit)
    ga._build_menu()

    def shot(name):
        from panda3d.core import Filename
        base.win.saveScreenshot(Filename.fromOsSpecific(str(OUT / name)))  # noqa
        print(f"   {name}")

    st = {"f": 0}

    def update():
        st["f"] += 1
        f = st["f"]
        ga.update()
        if f == 3:
            shot("t_menu.png")
        elif f == 5:
            ga._start_race(args.circuit, 3, True)     # Enter on the menu
        elif f == 7:
            shot("t_loading_in.png")                  # card up, build pending
        elif f == 40:
            # by now built + held; nudge into the fade and catch it half done
            tr = ga.TRANSITION
            if tr is not None:
                tr._phase = "wipe"
                tr._t = tr.WIPE * 0.55
        elif f == 41:
            shot("t_wipe_in.png")                     # scene showing through
        elif f == 70:
            ga.GAME.start_t = ga.GAME._T_build1 + 0.3   # all five lit, holding
        elif f == 72:
            shot("t_lights_all.png")
        elif f == 120:
            shot("t_racing.png")                      # lights out, GO gone
        elif f == 122:
            ga.GAME.on_key("escape")                  # pause
            ga.GAME.on_key("enter")                   # exit to menu -> transition
        elif f == 124:
            shot("t_loading_out.png")                 # leaving the race
        elif f == 150:
            tr = ga.TRANSITION
            if tr is not None:
                tr._phase = "wipe"
                tr._t = tr.WIPE * 0.55
        elif f == 151:
            shot("t_wipe_out.png")                    # menu showing through
        elif f == 180:
            shot("t_menu_back.png")
            application.quit()

    import __main__
    __main__.update = update
    application.base.run()


if __name__ == "__main__":
    main()
