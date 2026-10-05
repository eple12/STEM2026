"""Screenshot a race from a fixed set of camera positions, for judging the look.

    python tools/preview_scene.py                       # Monza -> scene_preview/
    python tools/preview_scene.py --circuit Spa --size 1920 1080
    python tools/preview_scene.py --set POST_ENABLED=False --tag nopost

Every shot is taken from the same places on the lap each run, so two runs --
before and after a change to the lighting, the post-processing or the road --
can be put side by side and compared like for like. ``--set`` overrides any
``config`` constant for the run (the value is parsed as a Python literal).
"""
import argparse
import ast
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panda3d.core import loadPrcFileData

loadPrcFileData('', 'sync-video 0')

from ursina import Text, Ursina, Vec3, application, camera  # noqa: E402

from game import app as ga  # noqa: E402
from game import config  # noqa: E402
from game.ui import pick_font  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "scene_preview"


def _corner_index(track, rank: int = 0) -> int:
    """The apex of the *rank*-th slowest corner, by curvature radius."""
    import numpy as np
    r = track.curv_radius.copy()
    order = np.argsort(r)
    picked: list[int] = []
    for i in order:
        if all(min(abs(i - j), track.count - abs(i - j)) > track.count // 12
               for j in picked):
            picked.append(int(i))
        if len(picked) > rank:
            break
    return picked[rank]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default=config.DEFAULT_TRACK)
    ap.add_argument("--size", type=int, nargs=2, default=(1600, 900))
    ap.add_argument("--tag", default="")
    ap.add_argument("--only", default="",
                    help="comma-separated shot names to keep")
    ap.add_argument("--set", action="append", default=[],
                    help="KEY=VALUE config override, repeatable")
    args = ap.parse_args()

    for kv in args.set:
        k, v = kv.split("=", 1)
        setattr(config, k, ast.literal_eval(v))
        if k == "LIGHTING_PRESET":
            for kk, vv in config._LIGHTING[config.LIGHTING_PRESET].items():
                setattr(config, kk, vv)

    app = Ursina(title="FORMULA-AI", size=tuple(args.size), vsync=False,
                 development_mode=False)
    font = pick_font()
    if font:
        Text.default_font = font

    ga.SESSION.update(laps=3, mute=True, track=args.circuit)
    ga._build_race(args.circuit, 3, True, intro=False)
    g = ga.GAME
    t = g.track
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    def shot(name):
        from panda3d.core import Filename
        p = OUT / f"{name}_{args.circuit}{suffix}.png"
        base.win.saveScreenshot(Filename.fromOsSpecific(str(p)))  # noqa: F821
        print(f"   {p.name}")

    def place(i, speed, lateral=0.0):
        v = g.vehicle
        tan = t.tangent[i]
        pos = t.center[i] + t.normal[i] * lateral
        v.place(pos.copy(), math.atan2(tan[0], tan[1]))
        v.vel[:] = tan * speed
        g.surface.hint = i
        g._snap_camera()

    def racing():
        g.state = 1
        g.start_t = g._T_rise1 + 1.0
        g.vehicle.frozen = False
        g.lap_num = 2
        g.last_t, g.best_t = 93.284, 92.418
        g.session_time = 195.7
        g.lap_start = g.session_time - 41.2
        g.sector = 1
        g.sector_start = g.session_time - 9.0
        g.sectors = [29.950, None, None]
        g.best_sectors = [30.112, 31.905, 30.401]

    corner = _corner_index(t, 0)
    corner2 = _corner_index(t, 1)
    # A grandstand straight: the start/finish, a little way down it.
    straight = int(0.03 * t.count)

    def trackside(i, side=+1, back=28.0, out=14.0, h=1.6, fov=26):
        """A TV camera: low, beside the track, long lens, on the car."""
        c = t.center[i]
        p = c - t.tangent[i] * back + t.normal[i] * side * (t.w_right[i] + out)
        camera.position = Vec3(float(p[0]), h, float(p[1]))
        cp = g.car.world_position
        camera.look_at(Vec3(cp.x, cp.y + 0.6, cp.z))
        camera.rotation_z = 0
        camera.fov = fov

    # (frame, action). Each shot is set up, given a few frames for the damped
    # camera and the shadow follow to settle, then saved.
    plan = []
    f = 6

    def add(name, setup, settle=10, after=None):
        nonlocal f
        plan.append((f, "setup", name, setup))
        f += settle
        plan.append((f, "shot", name, after))
        f += 2

    add("grid", lambda: setattr(g, "start_t", 3.6), settle=8)
    add("chase_straight", lambda: (racing(), place(straight, 72.0)))
    add("chase_corner", lambda: (racing(), place(corner - 9, 38.0)))
    add("chase_corner2", lambda: (racing(), place(corner2 - 7, 45.0)))

    def slide():
        # Sideways at speed, for the tyre smoke: velocity 30 degrees off
        # the heading.
        racing()
        place(straight + 60, 0.0)
        tan = t.tangent[straight + 60]
        a = math.radians(30.0)
        c, s_ = math.cos(a), math.sin(a)
        g.vehicle.vel[:] = [(tan[0] * c - tan[1] * s_) * 30.0,
                            (tan[0] * s_ + tan[1] * c) * 30.0]

    add("fx_slide", slide, settle=14)

    def onboard():
        racing()
        g.cam_idx = config.CAM_MODES.index("onboard")
        place(straight + 30, 75.0)

    add("onboard", onboard)

    def tv():
        g.cam_idx = config.CAM_MODES.index("chase")
        racing()
        place(corner - 3, 30.0)
        g.hud.root.enabled = False

    add("tv_corner", lambda: (tv(), trackside(corner - 3)), settle=4)

    def tv_straight():
        racing()
        place(straight + 12, 60.0)
        g.hud.root.enabled = False

    add("tv_straight", lambda: (tv_straight(),
                                trackside(straight + 12, side=-1, back=40.0,
                                          out=10.0, h=2.2, fov=18)),
        settle=4)

    def closeup():
        racing()
        place(straight + 40, 0.0)
        g.vehicle.vel[:] = 0.0
        g.vehicle.frozen = True
        g.hud.root.enabled = False

    def close_cam(f=4.6, r=3.4, up=1.25, fov=42):
        cp = g.car.world_position
        yaw = math.radians(g.car.rotation_y)
        fwd = Vec3(math.sin(yaw), 0, math.cos(yaw))
        right = Vec3(math.cos(yaw), 0, -math.sin(yaw))
        camera.position = cp + fwd * f + right * r + Vec3(0, up, 0)
        camera.look_at(cp + Vec3(0, 0.45, 0))
        camera.rotation_z = 0
        camera.fov = fov

    add("car_close", lambda: (closeup(), close_cam()), settle=6)
    add("car_rear", lambda: (closeup(), close_cam(-4.4, -2.6, 2.2, 40)),
        settle=6)
    add("car_side", lambda: (closeup(), close_cam(0.2, 6.5, 0.9, 38)),
        settle=6)

    def aerial():
        racing()
        g.vehicle.frozen = True
        i = corner
        c = t.center[i]
        p = c + t.normal[i] * 160 - t.tangent[i] * 120
        camera.position = Vec3(float(p[0]), 95.0, float(p[1]))
        camera.look_at(Vec3(float(c[0]), 0.0, float(c[1])))
        camera.rotation_z = 0
        camera.fov = 55
        g.hud.root.enabled = False

    add("aerial", aerial, settle=6)

    def stands():
        # From the track edge, across at the main grandstand.
        racing()
        g.vehicle.frozen = True
        i = straight + 25
        c = t.center[i]
        n = t.normal[i]
        p = c - n * (t.w_left[i] - 1.0)
        camera.position = Vec3(float(p[0]), 1.7, float(p[1]))
        q = c - n * 45.0 + t.tangent[i] * 18.0
        camera.look_at(Vec3(float(q[0]), 7.0, float(q[1])))
        camera.rotation_z = 0
        camera.fov = 48
        g.hud.root.enabled = False

    add("stands", stands, settle=6)

    def treeline():
        # Close to the trees behind the fence, on a corner.
        racing()
        g.vehicle.frozen = True
        i = corner2 + 10
        c = t.center[i]
        n = t.normal[i]
        p = c + n * (t.w_right[i] + 4.0)
        camera.position = Vec3(float(p[0]), 2.0, float(p[1]))
        q = c + n * 60.0
        camera.look_at(Vec3(float(q[0]), 6.0, float(q[1])))
        camera.rotation_z = 0
        camera.fov = 55
        g.hud.root.enabled = False

    add("treeline", treeline, settle=6)

    keep = set(filter(None, args.only.split(",")))
    pending = {"cam": None}
    state = {"f": 0}

    def update():
        state["f"] += 1
        fr = state["f"]
        # Fixed cameras fight the chase rig every frame; re-apply after it.
        ga.update()
        if pending["cam"] is not None:
            pending["cam"]()
        for when, kind, name, fn in plan:
            if when != fr:
                continue
            if kind == "setup":
                pending["cam"] = None
                g.hud.root.enabled = True
                g.cam_idx = config.CAM_MODES.index("chase")
                g.car.hull.enabled = True
                fn()
                if name.startswith(("tv_", "car_", "aerial", "stands",
                                    "treeline")):
                    pending["cam"] = fn
            elif not keep or name in keep:
                shot(name)
        if fr > plan[-1][0] + 1:
            application.quit()

    import __main__
    __main__.update = update
    app.run()


if __name__ == "__main__":
    main()
