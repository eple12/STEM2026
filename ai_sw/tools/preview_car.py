"""Render the car model from several angles to PNGs in ``car_preview/``.

Iterating on the bodywork by driving a lap and squinting at the mirror is slow;
this puts the model on screen in a couple of seconds.

``--holes`` renders against a saturated magenta backdrop with no ground plane,
so any gap in the hull shows up as a magenta pixel inside the silhouette, and
reports the count per view.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ursina import (AmbientLight, DirectionalLight, Entity, Ursina, Vec3,
                    application, camera, color, scene, window)

from game import palette as pal
from game import config
from game.car import Car
from game.vehicle import Vehicle

OUT = Path(__file__).resolve().parents[1] / "car_preview"
SIZE = 760

VIEWS = [
    ("front3q", Vec3(4.6, 2.0, 6.4)),
    ("rear3q", Vec3(4.2, 2.3, -6.2)),
    ("side", Vec3(8.2, 1.2, 0.2)),
    ("top", Vec3(0.1, 8.0, -0.6)),
    ("front", Vec3(0.0, 1.2, 7.6)),
    ("low", Vec3(3.4, 0.45, 5.2)),
    ("wheel", Vec3(3.0, 0.9, 2.6)),
    ("under", Vec3(3.0, -1.6, 4.6)),
]
HOLE_BG = color.magenta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holes", action="store_true",
                    help="magenta backdrop, to spot gaps in the model")
    ap.add_argument("--model", default=None,
                    help="model name, resolved the same way config.PLAYER_MODEL "
                         "is -- see game/car.py (default: config.PLAYER_MODEL)")
    args = ap.parse_args()

    app = Ursina(title="car preview", size=(SIZE, SIZE), vsync=False,
                 development_mode=False)
    window.color = HOLE_BG if args.holes else pal.rgb(120, 126, 134)
    AmbientLight(color=pal.rgb(165, 168, 175))
    sun = DirectionalLight(color=pal.rgb(255, 250, 240))
    sun.look_at(Vec3(0.5, -1.0, 0.35))

    if not args.holes:
        Entity(parent=scene, model="plane", scale=(30, 1, 30),
               color=pal.ASPHALT, y=config.Y_ASPHALT)

    v = Vehicle()
    v.frozen = True
    v.place((0.0, 0.0), 0.0)
    car = Car(v, **({'model': args.model} if args.model else {}))
    car.sync()
    camera.fov = 40

    OUT.mkdir(parents=True, exist_ok=True)
    state = {"frame": 0}

    def _save(name):
        from panda3d.core import Filename
        base.win.saveScreenshot(  # noqa: F821
            Filename.fromOsSpecific(str(OUT / f"{name}.png")))
        print(f"SHOT {name}")

    # Driven from the frame loop rather than invoke(): the camera has to move
    # on one frame and be captured on a later one, and invoke() never fires
    # unless __main__.update exists anyway.
    def update():
        f = state["frame"]
        state["frame"] += 1
        if f < 5:
            return
        idx, phase = divmod(f - 5, 3)
        if idx >= len(VIEWS):
            application.quit()
            return
        name, pos = VIEWS[idx]
        if phase == 0:
            camera.position = pos
            camera.look_at(Vec3(0, 0.62, 0))
            camera.rotation_z = 0        # look_at rolls when off-axis
            camera.fov = 20 if name == "wheel" else 40
        elif phase == 2:
            _save(name)

    import __main__
    __main__.update = update
    app.run()


if __name__ == "__main__":
    main()
