"""Screenshot the main menu and the settings screen.

    python tools/preview_settings.py

Writes ``menu_preview/main.png`` and ``menu_preview/settings.png``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ursina import AmbientLight, DirectionalLight, Ursina, Vec3, application, window

from game import palette as pal
from game import settings
from game.menu import ModeMenu, SettingsMenu

OUT = Path(__file__).resolve().parents[1] / "menu_preview"


def main():
    app = Ursina(title="FORMULA-AI", size=(1600, 900), vsync=False, development_mode=False)
    window.color = pal.rgb(21, 21, 30)
    AmbientLight(color=pal.rgb(160, 164, 172))
    DirectionalLight().look_at(Vec3(0.6, -1.0, 0.3))
    OUT.mkdir(parents=True, exist_ok=True)
    settings.current.auto_steer = True
    settings.current.drs = False
    menu = {"m": ModeMenu(on_pick=lambda m: None, on_quit=application.quit,
                          on_settings=lambda: None)}
    state = {"f": 0}

    def shoot(name):
        from panda3d.core import Filename
        base.win.saveScreenshot(Filename.fromOsSpecific(str(OUT / name)))  # noqa: F821

    def update():
        state["f"] += 1
        f = state["f"]
        if f == 6:
            shoot("main.png")
        elif f == 7:
            menu["m"].destroy()
            menu["m"] = SettingsMenu(on_back=lambda: None)
            menu["m"].on_key("s")
            menu["m"].on_key("s")
        elif f == 12:
            shoot("settings.png")
        elif f == 14:
            application.quit()

    import __main__
    __main__.update = update
    app.run()


if __name__ == "__main__":
    main()
