"""Grand prix, game side: the nineteen other cars on screen.

The field runs in a worker process (``fieldproc.py``); this draws it. Each
AI car is an ordinary ``Car`` model in its team's livery, driven from a
``ProxyVehicle`` -- just enough of a ``Vehicle`` for the model, the camera
and the HUD to read -- that is set from the newest snapshot carried forward
by the car's own velocity to the instant being drawn.

Drawing twenty cars is not free, so the per-car work is graded by distance
from the camera. A car near it is the full model -- fifteen parts, wheels
turned and spun, the body rolled, pitched and squatting -- posed every frame
through Panda directly (Ursina's rotation properties read the whole rotation
back and rebuild it for every axis set, and that was most of the cost). A
car further away is a second, flattened copy of the same model in its rest
pose: the same look at that size, five draw calls instead of fifteen, and
nothing to pose but where it is.
"""
from __future__ import annotations

import math

import numpy as np
from panda3d.core import TransparencyAttrib

from . import config, teams
from .fieldproc import C
from .vehicle import Telemetry

#: Beyond this many metres from the camera a car is drawn as its flattened
#: copy (no wheel spin, steering or body attitude, no contact shadow).
DETAIL_RANGE = 30.0
#: Visual smoothing of snapshot corrections: time constant (s), and the
#: correction past which it is a cut instead (m).
ERR_T = 0.07
ERR_SNAP = 3.0
#: The car the camera is on: a camera bolted to it shows every correction, so
#: its are spread over longer.
ERR_T_WATCH = 0.30
#: How solid a car looks while it passes through the others (recovering,
#: or put back on the track).
GHOST_ALPHA = 0.45


class ProxyVehicle:
    """What Car.sync, the camera and the HUD read off a car, set from the
    field's snapshots rather than integrated here."""

    def __init__(self, track):
        self.track = track
        self.pos = np.zeros(2)
        self.prev_pos = self.pos
        self.yaw = 0.0
        self.prev_yaw = 0.0
        self.vel = np.zeros(2)
        self.yaw_rate = 0.0
        self.steer_angle = 0.0
        self.steer_input = 0.0
        self.on_track = True
        self.frozen = False
        self.tc_cut = 0.0
        self.esc_cut = 0.0
        self.traction_control = True
        self.tele = Telemetry()

    @property
    def speed(self) -> float:
        return math.hypot(float(self.vel[0]), float(self.vel[1]))

    @property
    def forward_speed(self) -> float:
        return (float(self.vel[0]) * math.sin(self.yaw)
                + float(self.vel[1]) * math.cos(self.yaw))

    @property
    def slip_angle(self) -> float:
        v = self.speed
        if v < 1.0:
            return 0.0
        lat = (float(self.vel[0]) * math.cos(self.yaw)
               - float(self.vel[1]) * math.sin(self.yaw))
        return math.atan2(lat, abs(self.forward_speed) + 1e-6)

    @property
    def lat_accel(self) -> float:
        return self.tele.lat_accel

    @property
    def long_accel(self) -> float:
        return self.tele.long_accel


def team_of(name: str):
    return next((t for t in teams.TEAMS if t.name == name), None)


class GPCars:
    """The AI cars: their models, their proxies, and keeping both current."""

    def __init__(self, track, info, player: int, light, progress=None):
        from .car import Car
        from .shaders import car_shader

        step = progress or (lambda: None)
        self.track = track
        self.info = info
        self.player = player
        #: The other cars' shadows (lighting.FieldShadow).
        self.shadows = light.field_shadow
        self.cars: dict[int, object] = {}
        self.proxies: dict[int, ProxyVehicle] = {}
        self._ghost: dict[int, bool] = {}
        self._detail: dict[int, bool] = {}
        self._spin: dict[int, float] = {}
        self._att: dict[int, list] = {}
        for c in info:
            if c["external"]:
                continue
            idx = c["idx"]
            v = ProxyVehicle(track)
            v.pos = np.asarray(c["pos"], float)
            v.prev_pos = v.pos
            v.yaw = v.prev_yaw = float(c["yaw"])
            car = Car(v, model=config.PLAYER_MODEL)
            light.apply(car, spec_strength=0.55, spec_power=64.0,
                        shader=car_shader)
            tm = team_of(c["team"])
            livery = None
            if tm is not None:
                livery = (tm.color, tm.secondary, tm.accent)
            car.apply_materials(config.PLAYER_MODEL, livery=livery)
            car.lod = self._flat_copy(car, config.PLAYER_MODEL)
            # The sun's shadow: a merged stand-in in the field's own map,
            # posed from this car every frame (the hero car's map is the
            # player's alone).
            self.shadows.add(idx, car.lod)
            # On the grid from the start: until the first snapshot (the intro
            # film runs before there is one) the stand-in would sit at the
            # origin, which is the start line.
            self.shadows.pose(idx, car)
            self.cars[idx] = car
            self.proxies[idx] = v
            self._ghost[idx] = False
            self._detail[idx] = True
            self._spin[idx] = 0.0
            self._att[idx] = [0.0, 0.0]
            step()
        self._t_snap = None
        #: Visual smoothing: the offset between where a car is drawn and
        #: where its newest snapshot says it is, decaying (see update).
        self._err = {idx: [0.0, 0.0, 0.0] for idx in self.cars}
        self._prev_snap = None

    #: The car's materials, as its parts are named (Car.apply_materials).
    PARTS = ("Livery", "Carbon", "Rubber", "Metal", "Interior")

    @classmethod
    def _flat_copy(cls, car, model: str):
        """The car's far stand-in: its parts merged by material -- all the
        rubber in one mesh, all the metal in another -- under one node per
        material that carries the material's inputs, so the parts beneath share
        a state and flatten into a single draw. Hidden until the car is far
        enough away.

        The geometry is the model's own ``<model>_lod.bam`` when there is one
        (a car seen from 30 m or more has no use for 91k triangles; the baked
        stand-in has 17k), else a copy of the car itself in its rest pose.
        """
        import builtins
        from pathlib import Path

        from panda3d.core import Filename, NodePath

        from .car import F1_DIR
        src, src_root = car, car
        lod_path = Path(F1_DIR) / f"{model}_lod.bam"
        if lod_path.is_file():
            src_root = builtins.base.loader.load_model(
                Filename.from_os_specific(str(lod_path.resolve())))
            src = src_root
        # Each material's shader inputs, as Car.apply_materials set them on the
        # car's own parts.
        inputs = {}
        for gn in car.find_all_matches("**/+GeomNode"):
            key = next((k for k in cls.PARTS if k in gn.get_name()), None)
            if key is not None and key not in inputs:
                inputs[key] = tuple(gn.get_shader_input(i)
                                    for i in ("part", "material"))
        lod = NodePath("lod")
        groups = {}
        for gn in src.find_all_matches("**/+GeomNode"):
            key = next((k for k in cls.PARTS if k in gn.get_name()), None)
            if key is None or key not in inputs:
                continue                      # the contact shadow
            grp = groups.get(key)
            if grp is None:
                grp = groups[key] = lod.attach_new_node(key)
                for inp in inputs[key]:
                    grp.set_shader_input(inp)
            c = NodePath(gn.node().make_copy())
            c.reparent_to(grp)
            c.set_transform(gn.get_transform(src_root))
            c.clear_shader_input("part")
            c.clear_shader_input("material")
            for key_ in list(c.node().get_python_tag_keys()):
                c.node().clear_python_tag(key_)
        lod.flatten_strong()
        lod.reparent_to(car)
        lod.hide()
        if src_root is not car:
            src_root.remove_node()
        return lod

    # -- per frame -------------------------------------------------------
    def update(self, snap, t_render: float, dt: float, cam_pos, watch=None):
        """Pose every AI car for this frame from the newest snapshot."""
        if snap is None:
            return
        rows = snap["rows"]
        ahead = max(0.0, min(t_render - snap["t"], 0.25)) if snap["started"] else 0.0
        cx, cz = float(cam_pos[0]), float(cam_pos[2])
        r2 = DETAIL_RANGE * DETAIL_RANGE
        # A new snapshot disagrees a little with where the old one was being
        # extrapolated to -- most of all just after a contact, when the
        # worker has pushed two cars apart and changed their speeds. Jumped
        # to, that is a car teleporting a few centimetres to half a metre.
        # Instead the difference is kept as an offset on the drawn pose and
        # bled away over ERR_T; anything bigger than ERR_SNAP (a recovery,
        # a reset) is still a cut.
        old = self._prev_snap
        fresh = old is not None and old is not snap
        if fresh:
            old_rows = old["rows"]
            old_ahead = (max(0.0, min(t_render - old["t"], 0.25))
                         if old["started"] else 0.0)
        self._prev_snap = snap
        decay = math.exp(-dt / ERR_T)
        decay_watch = math.exp(-dt / ERR_T_WATCH)
        for idx, car in self.cars.items():
            r = rows[idx]
            v = self.proxies[idx]
            vx, vz = r[C["vx"]], r[C["vz"]]
            x = r[C["x"]] + vx * ahead
            z = r[C["z"]] + vz * ahead
            yaw = r[C["yaw"]] + r[C["yaw_rate"]] * ahead
            err = self._err[idx]
            if fresh:
                o = old_rows[idx]
                ox = o[C["x"]] + o[C["vx"]] * old_ahead
                oz = o[C["z"]] + o[C["vz"]] * old_ahead
                oyaw = o[C["yaw"]] + o[C["yaw_rate"]] * old_ahead
                ex, ez = err[0] + ox - x, err[1] + oz - z
                ey = (err[2] + oyaw - yaw + math.pi) % (2 * math.pi) - math.pi
                if ex * ex + ez * ez > ERR_SNAP * ERR_SNAP:
                    ex = ez = ey = 0.0
                err[0], err[1], err[2] = ex, ez, ey
            k = decay_watch if idx == watch else decay
            err[0] *= k
            err[1] *= k
            err[2] *= k
            x += err[0]
            z += err[1]
            yaw += err[2]
            v.pos = np.array((x, z))
            v.prev_pos = v.pos
            v.yaw = v.prev_yaw = float(yaw)
            v.vel = np.array((vx, vz))
            v.yaw_rate = float(r[C["yaw_rate"]])
            v.steer_angle = float(r[C["steer"]])
            v.on_track = bool(r[C["on_track"]])
            t = v.tele
            t.lat_accel = float(r[C["lat_accel"]])
            t.long_accel = float(r[C["long_accel"]])
            t.downforce = float(r[C["downforce"]])

            ghost = bool(r[C["ghost"]])
            if ghost != self._ghost[idx]:
                self._ghost[idx] = ghost
                if ghost:
                    car.set_transparency(TransparencyAttrib.M_alpha)
                    car.set_color_scale(1.0, 1.0, 1.0, GHOST_ALPHA)
                else:
                    car.clear_color_scale()
                    car.set_transparency(TransparencyAttrib.M_none)

            near = (x - cx) ** 2 + (z - cz) ** 2 < r2
            if near != self._detail[idx]:
                self._detail[idx] = near
                if near:
                    car.lod.hide()
                    car.hull.show()
                    for w in car.wheels:
                        w.show()
                    car.contact.show()
                else:
                    car.lod.show()
                    car.hull.hide()
                    for w in car.wheels:
                        w.hide()
                    car.contact.hide()
            # Straight to Panda: Ursina's rotation_y = a is setH(-a), its
            # position (x, y, z) is setPos(x, y, z).
            car.setPosHpr(x, 0.0, z, -math.degrees(yaw), 0.0, 0.0)
            if not near:
                continue
            # Up close: the wheels and the body, as Car.sync does them.
            steer_vis = math.degrees(v.steer_angle)
            fwd_speed = vx * math.sin(yaw) + vz * math.cos(yaw)
            spin = (self._spin[idx] + math.degrees(
                fwd_speed / car._radius) * dt) % 360.0
            self._spin[idx] = spin
            for w in car.wheels:
                if w.steers:
                    w.setH(-steer_vis)
                w.spinner.setP(-spin)
            k = min(1.0, 8.0 * dt)
            att = self._att[idx]
            cap = config.BODY_ROLL_MAX
            att[0] += (max(-cap, min(cap, -t.lat_accel * config.BODY_ROLL_GAIN))
                       - att[0]) * k
            pcap = config.BODY_PITCH_MAX
            att[1] += (max(-pcap, min(pcap, -t.long_accel * config.BODY_PITCH_GAIN))
                       - att[1]) * k
            load_ratio = t.downforce / (config.CAR_MASS * config.GRAVITY)
            # hull: rotation_z = roll -> R, rotation_x = pitch -> P = -pitch.
            car.hull.setPosHpr(0.0, -config.BODY_SQUAT_MAX * min(1.0, load_ratio),
                               0.0, 0.0, -att[1], att[0])
        if not self.track.is_flat:
            self._lay_on_road()
        sh = self.shadows
        for idx, car in self.cars.items():
            sh.pose(idx, car, shown=not self._ghost[idx])

    def _lay_on_road(self):
        """Heights and attitude off the road surface, for every car in one
        query (banked circuits only; a flat one is all zero)."""
        idxs = list(self.cars)
        probes = []
        for idx in idxs:
            v = self.proxies[idx]
            px, pz, yaw = float(v.pos[0]), float(v.pos[1]), v.yaw
            c, sn = math.cos(yaw), math.sin(yaw)
            probes += [(px, pz), (px + sn * 1.6, pz + c * 1.6),
                       (px - sn * 1.6, pz - c * 1.6), (px + c, pz - sn),
                       (px - c, pz + sn)]
        h, _b, _n = self.track.surface_pose(probes)
        for j, idx in enumerate(idxs):
            car = self.cars[idx]
            hh = h[5 * j:5 * j + 5]
            car.setY(float(hh[0]))
            # rotation_x = a -> P = -a; rotation_z = a -> R = a.
            car.setP(math.degrees(math.atan2(float(hh[1] - hh[2]), 3.2)))
            car.setR(-math.degrees(math.atan2(float(hh[3] - hh[4]), 2.0)))

    def destroy(self):
        from .ui import destroy_tree
        self.shadows.clear()
        for car in self.cars.values():
            destroy_tree(car)
        self.cars.clear()
        self.proxies.clear()
