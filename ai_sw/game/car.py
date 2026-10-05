"""Visual car: the model on screen, driven from the physics car's telemetry.

Four sources are supported, picked by the model *name* alone (config.PLAYER_MODEL
/ GHOST_MODEL), so switching between them is a one-line edit in config.py:

* ``bl_red`` / ``bl_white`` -- the default. The Blender car in
  ``blender/f1_car.py``, baked by ``tools/build_blender_f1.py`` into a .bam
  under ``assets/models/f1``.
* ``rb_red`` / ``rb_white`` -- the supplied fp04rb asset, baked the same way by
  ``tools/build_f1_asset.py``.
* ``rc_*`` / ``rl_*`` -- ``assets/for+race.blend`` (full / decimated), baked by
  ``blender/export_race_car.py`` and ``tools/build_blender_race.py``.
* ``f1red`` / ``f1white`` -- built from primitives at runtime by ``f1car.py``.
* anything else -- a .glb from the Kenney Racing Kit, CC0
  (``assets/models/kenney``, see its LICENSE.txt), through ``panda3d-gltf``.

A .bam arrives already in game coordinates with its wheel nodes attached and
its livery in the vertex colours, so it is used as-is; a .glb is authored in
its creator's own space and has to be scaled onto the physics wheelbase and
turned round first.

What is done here for every source is the part specific to this game: driving
the model from telemetry so the body rolls with real lateral acceleration,
pitches with real longitudinal acceleration, squats under real downforce, and
the wheels steer and spin at the real rates.
"""
from __future__ import annotations

import math
from pathlib import Path

from ursina import Entity, Mesh, Vec3, scene

from . import config
from . import f1car
from . import palette as pal
from .vehicle import Vehicle

MODEL_DIR = config.ASSET_DIR / "models" / "kenney"
F1_DIR = config.ASSET_DIR / "models" / "f1"
DEFAULT_MODEL = config.PLAYER_MODEL

#: Ride-height drop at full downforce -- config.BODY_SQUAT_MAX, kept as a
#: module constant for the previews that import it directly.
SQUAT_MAX = config.BODY_SQUAT_MAX
# Named nodes inside the asset; the value says whether that wheel steers.
WHEEL_NODES = {
    "wheelFrontLeft": True, "wheelFrontRight": True,
    "wheelBackLeft": False, "wheelBackRight": False,
}


def lerp_pose(vehicle: Vehicle, alpha: float) -> tuple[float, float, float]:
    """Interpolated (x, z, yaw) between the last two physics states."""
    a = 0.0 if alpha < 0.0 else 1.0 if alpha > 1.0 else alpha
    x = vehicle.prev_pos[0] + (vehicle.pos[0] - vehicle.prev_pos[0]) * a
    z = vehicle.prev_pos[1] + (vehicle.pos[1] - vehicle.prev_pos[1]) * a
    # shortest-arc so a wrap past +-pi doesn't spin the car backwards
    d = (vehicle.yaw - vehicle.prev_yaw + math.pi) % (2 * math.pi) - math.pi
    return float(x), float(z), vehicle.prev_yaw + d * a


def load_gltf(path) -> Entity:
    """Load a .glb/.gltf through Panda3D's loader and wrap it in an Entity.

    Ursina's own ``model=`` lookup only resolves names inside its asset
    folders, so it cannot take an absolute path; going through the Panda loader
    is both the documented route and the one that gives the full glTF pipeline
    (materials, normals, node names).
    """
    import builtins

    from panda3d.core import Filename

    holder = Entity()
    node = builtins.base.loader.loadModel(
        Filename.fromOsSpecific(str(Path(path).resolve())))
    node.reparent_to(holder)
    holder.set_color_off()
    _bake_material_colors(node)
    return holder


def load_bam(path) -> Entity:
    """Load a .bam built by tools/build_blender_f1.py or build_f1_asset.py.

    No material baking and no refit: the builder already wrote flat vertex
    colours, game-space coordinates and the wheel nodes, so all this has to do
    is get it into the scene without Ursina's own white tint on top.
    """
    import builtins

    from panda3d.core import Filename

    holder = Entity()
    holder.faces_forward = True
    node = builtins.base.loader.loadModel(
        Filename.from_os_specific(str(Path(path).resolve())))
    node.reparent_to(holder)
    holder.set_color_off()
    return holder


def _bake_material_colors(node, shade_top: float = 1.0, shade_bottom: float = 0.72):
    """Turn glTF materials into flat vertex colours.

    panda3d-gltf loads baseColorFactor into Panda ``Material`` objects, but
    Ursina renders unlit and ignores materials, so the asset arrives pure
    white. Baking each geom's base colour into a ColorAttrib makes it render
    correctly and keeps the flat-shaded look the rest of the game uses.

    A gentle top-down gradient is folded in as well, so a curved body reads as
    a shape rather than a silhouette.
    """
    from panda3d.core import ColorAttrib, MaterialAttrib

    lo, hi = node.get_tight_bounds()
    span = max(hi.z - lo.z, hi.y - lo.y, 1e-6)
    for gn in node.find_all_matches("**/+GeomNode"):
        geom_node = gn.node()
        for i in range(geom_node.get_num_geoms()):
            state = geom_node.get_geom_state(i)
            attrib = state.get_attrib(MaterialAttrib)
            if attrib is None or attrib.get_material() is None:
                continue
            base = attrib.get_material().get_base_color()
            # height of this geom within the model, for the gradient
            gb = geom_node.get_geom(i).get_bounds()
            t = 0.5
            if not gb.is_empty():
                c = gb.get_approx_center()
                t = min(max((c.y - lo.y) / span, 0.0), 1.0)
            f = shade_bottom + (shade_top - shade_bottom) * t
            geom_node.set_geom_state(i, state.add_attrib(
                ColorAttrib.make_flat((base[0] * f, base[1] * f,
                                       base[2] * f, base[3]))))




class Car(Entity):
    def __init__(self, vehicle: Vehicle, model: str = DEFAULT_MODEL, **kw):
        super().__init__(parent=scene, **kw)
        self.vehicle = vehicle
        self._roll = 0.0
        self._pitch = 0.0
        self._spin = 0.0

        # hull carries roll/pitch; the Car entity itself only holds yaw
        self.hull = Entity(parent=self)
        bam = F1_DIR / f"{model}.bam"
        if bam.is_file():
            # A car baked to Panda's own format offline (build_blender_f1.py
            # for bl_*, build_f1_asset.py for rb_*): already in game
            # coordinates, wheels attached, livery in the vertex colours.
            self.body = load_bam(bam)
        elif model in f1car.LIVERIES:
            self.body = f1car.build(model)      # the procedural car
        else:
            self.body = load_gltf(MODEL_DIR / f"{model}.glb")
        self.body.parent = self.hull
        self._braking = None

        self._fit_to_wheelbase()
        self.wheels = self._rig_wheels()
        self._radius = self._wheel_radius()

        # Not a shadow -- the sun casts the real one (lighting.py) -- but the
        # occlusion under the floor that no shadow map draws.
        self.contact = self._add_contact_shadow()
        self.sync()

    # -- setup ---------------------------------------------------------
    def _fit_to_wheelbase(self):
        """Scale and orient the asset onto the physics car.

        The scale is derived from the asset's own axle spacing rather than
        hard-coded, so the wheels land on the simulated wheelbase whichever
        car model is loaded.
        """
        if getattr(self.body, "faces_forward", False):
            # Built for this game: already metres, already +z forward, already
            # sitting on the ground at the centre of gravity. Rescaling or
            # recentring it here would only push it off the physics box.
            self.body.rotation_y = 0.0
            return
        axles = [self.body.find(f"**/{n}") for n in WHEEL_NODES]
        zs = sorted(a.get_pos().z for a in axles if not a.is_empty())
        span = abs(zs[-1] - zs[0]) if len(zs) >= 2 else 1.0
        self.body.scale = config.WHEELBASE / max(span, 1e-6)
        # The Kenney asset faces -z and the game +z; the built car faces +z.
        self.body.rotation_y = (0.0 if getattr(self.body, "faces_forward", False)
                                else 180.0)

        lo, hi = self.body.get_tight_bounds()
        self.body.x -= (lo.x + hi.x) / 2      # centre laterally
        self.body.y -= lo.y                   # sit it on the ground
        self.body.z -= (lo.z + hi.z) / 2

    def _rig_wheels(self) -> list[Entity]:
        """Put each wheel on a hub (steering) and a spinner (rolling).

        The hubs hang off the *Car*, not off ``hull``, so that body attitude --
        roll, pitch and aero squat -- moves the bodywork around the wheels
        rather than carrying the wheels with it. That is what a suspension
        does, and the difference is not cosmetic: ``hull`` rotates about the
        centreline at ground level, so at BODY_ROLL_MAX a wheel a metre out
        travels some 75 mm vertically, and pitch adds as much again over the
        wheelbase. Left on the hull, the outer tyre sinks through the asphalt
        in every corner and the inner one hangs off the ground -- obvious on an
        open-wheel car, where the contact patch is in plain sight.
        """
        hubs = []
        for name, steers in WHEEL_NODES.items():
            np = self.body.find(f"**/{name}")
            if np.is_empty():
                continue
            # The hub has to sit at the wheel's *centre*, and the bounds are
            # taken in the Car's own space because that is where the hub now
            # lives. `np.get_pos()` is relative to the wheel's own parent
            # inside the asset (several nodes down) and points at the authoring
            # pivot, not the centre -- using it put the hub in the wrong place,
            # and the spin rotation then swung the wheel out on an arc instead
            # of rolling it.
            lo, hi = np.get_tight_bounds(self)
            hub = Entity(parent=self)
            hub.position = Vec3((lo.x + hi.x) / 2, (lo.y + hi.y) / 2,
                                (lo.z + hi.z) / 2)
            hub.steers = steers
            hub.spinner = Entity(parent=hub)
            # wrt_ keeps the wheel exactly where it is on screen while moving
            # it in the hierarchy; a plain reparent would drop it at the origin.
            np.wrt_reparent_to(hub.spinner)
            hubs.append(hub)
        return hubs

    #: Per-model colours the car shader paints with: the secondary panels,
    #: and the accent line (and the helmet).
    LIVERY = {
        "bl_red": dict(b=(16, 16, 18), c=(242, 242, 244)),
        "bl_white": dict(b=(14, 22, 58), c=(214, 24, 34)),
    }

    def apply_materials(self, model: str = DEFAULT_MODEL, livery=None):
        """Give each part of the car the surface it is made of.

        Under one material the whole car read as moulded plastic. The baked
        cars name their parts (``body__Livery``, ``FL__Rubber``...), and each
        gets its own: lacquered paint with a clear coat over it, carbon in a
        satin weave, matt rubber, dark machined rims. The car shader then
        paints the livery on the paint per pixel (see shaders.py, CAR).

        A model without named parts falls back to telling them apart by
        colour. Call after the lighting has been applied: these are per-part
        inputs, and they override the car-wide one from there.
        """
        from panda3d.core import GeomVertexReader
        from panda3d.core import Vec3 as PVec3

        from .shaders import material

        # the baked cars share their colourway by suffix: rc_white, rl_white...
        liv = self.LIVERY.get(model) or self.LIVERY.get(
            "bl_" + model.rpartition("_")[2], self.LIVERY["bl_red"])
        if livery is not None:
            # A team's colours, (r, g, b) 0..1: the paint itself as well.
            a, b, c = livery
            self.set_shader_input("livery_a", PVec3(*a))
            self.set_shader_input("livery_b", PVec3(*b))
            self.set_shader_input("livery_c", PVec3(*c))
        else:
            self.set_shader_input("livery_a", PVec3(-1.0, -1.0, -1.0))
            self.set_shader_input("livery_b", PVec3(*(v / 255.0 for v in liv["b"])))
            self.set_shader_input("livery_c", PVec3(*(v / 255.0 for v in liv["c"])))
        named = {"Livery": (1.0, material(0.30, 1.0, 1.0)),
                 "Carbon": (2.0, material(0.36, 1.3, 0.45)),
                 "Rubber": (3.0, material(0.90, 0.6, 0.0)),
                 "Metal": (4.0, material(0.24, 14.0, 0.0)),
                 "Interior": (0.0, material(0.70, 0.8, 0.0))}

        for w in self.wheels:
            for gn in w.find_all_matches("**/+GeomNode"):
                gn.set_python_tag("on_wheel", True)
        for gn in self.find_all_matches("**/+GeomNode"):
            node = gn.node()
            key = next((k for k in named if k in gn.get_name()), None)
            if key is not None:
                part, mat = named[key]
                gn.set_shader_input("part", part)
                gn.set_shader_input("material", mat)
                continue
            acc = [0.0, 0.0, 0.0]
            n = 0
            for i in range(node.get_num_geoms()):
                vdata = node.get_geom(i).get_vertex_data()
                if not vdata.has_column("color"):
                    continue
                rd = GeomVertexReader(vdata, "color")
                rows = vdata.get_num_rows()
                for r in range(0, rows, max(1, rows // 64)):
                    rd.set_row(r)
                    c = rd.get_data4()
                    acc[0] += c[0]
                    acc[1] += c[1]
                    acc[2] += c[2]
                    n += 1
            if n == 0:
                continue
            r, g, b = (v / n for v in acc)
            hi_, lo_ = max(r, g, b), min(r, g, b)
            sat = (hi_ - lo_) / max(hi_, 1e-4)
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            on_wheel = gn.has_python_tag("on_wheel")
            if (sat > 0.35 and hi_ > 0.25) or lum > 0.72:
                mat = material(0.34, 1.0, 1.0)          # lacquered paint
            elif lum < 0.16 and on_wheel:
                mat = material(0.88, 0.6, 0.0)          # tyre rubber
            elif lum < 0.16:
                mat = material(0.30, 1.1, 0.55)         # carbon, satin coat
            else:
                mat = material(0.28, 3.0, 0.0)          # machined metal
            gn.set_shader_input("material", mat)

    def _add_contact_shadow(self):
        """Ambient occlusion under the car: the soft darkening where the
        floor is a few centimetres off the road and no sky reaches. The sun's
        shadow does not do this -- in the car's own shadow, or with the sun
        overhead, the car otherwise floats."""
        import numpy as np
        from panda3d.core import TransparencyAttrib

        from .textures import panda_texture

        h, w = 128, 64
        y = (np.arange(h) + 0.5) / h * 2 - 1
        x = (np.arange(w) + 0.5) / w * 2 - 1
        X, Y = np.meshgrid(x, y)
        d = (np.abs(X) ** 4 + np.abs(Y) ** 4) ** 0.25
        a = np.clip(1.0 - d, 0.0, 1.0) ** 1.6
        img = np.zeros((h, w, 4))
        img[..., 3] = a * 255
        e = Entity(parent=self, model="quad", name="contact_shadow",
                   rotation_x=90, scale=(2.35, 4.6), y=config.Y_SHADOW,
                   color=(0, 0, 0, 0.62), unlit=True)
        e.set_texture(panda_texture(img, "contact_shadow", repeat=False), 1)
        e.set_transparency(TransparencyAttrib.M_alpha)
        e.set_depth_write(False)
        e.set_bin("transparent", 0)
        e.no_light = True
        return e

    def _wheel_radius(self) -> float:
        if not self.wheels:
            return 0.35
        lo, hi = self.wheels[0].get_tight_bounds()
        return max((hi.y - lo.y) / 2, 0.05)

    # -- push physics state into the transform ------------------------
    def sync(self, braking: bool = False, dt: float = 0.016, alpha: float = 1.0):
        """*alpha* is how far the renderer sits between the previous physics
        state and the current one -- see Vehicle.prev_pos."""
        v = self.vehicle
        t = v.tele
        px, pz, yaw = lerp_pose(v, alpha)
        self.rotation_y = math.degrees(yaw)
        # Sat on the road and lain along it, from five samples of the surface
        # taken at the *interpolated* position.
        #
        # Interpolated, because that is what fixes the stutter: the physics
        # runs at a fixed step and the frame lands somewhere between two of
        # them, so reading back the height the last step computed makes the
        # car climb a banked corner in visible stairs while its x and z glide.
        #
        # And five samples rather than a formula, because the attitude of a
        # body on a surface is the surface's own slope under it. Taking the
        # camber angle and working out which way to roll means getting a sign
        # right in a convention nobody can check; measuring the road a metre
        # to each side and asking which is higher cannot be got backwards.
        if v.track is not None:
            c, sn = math.cos(yaw), math.sin(yaw)
            fx, fz = sn, c                      # the car's forward, in world
            rx, rz = c, -sn                     # ...and its right
            probe = [(px, pz), (px + fx * 1.6, pz + fz * 1.6),
                     (px - fx * 1.6, pz - fz * 1.6),
                     (px + rx * 1.0, pz + rz * 1.0),
                     (px - rx * 1.0, pz - rz * 1.0)]
            h, _bank, _n = v.track.surface_pose(probe)
            self.position = Vec3(px, float(h[0]), pz)
            # Rolled towards the low side of the road and pitched nose-up
            # where it climbs away in front. The two signs are opposite, which
            # is not a mistake: Ursina's rotation_x and rotation_z turn the
            # model about axes of opposite handedness, so laying the same body
            # on the same surface needs one of each.
            self.rotation_x = -math.degrees(math.atan2(float(h[1] - h[2]), 3.2))
            self.rotation_z = -math.degrees(math.atan2(float(h[3] - h[4]), 2.0))
        else:
            self.position = Vec3(px, 0.0, pz)

        # Rain light on the built car: two meshes, one lit and one dark,
        # swapped on the brake. Vertex-coloured, so the ghost's alpha scale
        # still applies to it.
        if braking != self._braking and hasattr(self.body, "rain_on"):
            self._braking = braking
            self.body.rain_on.enabled = braking
            self.body.rain_off.enabled = not braking

        steer_vis = math.degrees(v.steer_angle)
        self._spin = (self._spin + math.degrees(
            v.forward_speed / self._radius) * dt) % 360.0
        for w in self.wheels:
            if w.steers:
                w.rotation_y = steer_vis
            w.spinner.rotation_x = self._spin

        # body attitude from real accelerations, smoothed
        k = min(1.0, 8.0 * dt)
        cap = config.BODY_ROLL_MAX
        self._roll += (max(-cap, min(cap, -t.lat_accel * config.BODY_ROLL_GAIN))
                       - self._roll) * k
        pcap = config.BODY_PITCH_MAX
        self._pitch += (max(-pcap, min(pcap, -t.long_accel
                                       * config.BODY_PITCH_GAIN))
                        - self._pitch) * k
        self.hull.rotation_z = self._roll
        self.hull.rotation_x = self._pitch

        # aero squat, capped so the body can never sink under the track
        load_ratio = t.downforce / (config.CAR_MASS * config.GRAVITY)
        self.hull.y = -SQUAT_MAX * min(1.0, load_ratio)
