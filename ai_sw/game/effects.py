"""Sparks, tyre smoke and dust: the motion a racing car leaves in the air.

Purely visual, read off the car's state after the physics has run -- nothing
here feeds back into it.

* **Sparks.** A modern Formula 1 car runs so low that its titanium skid blocks
  touch the road at speed, over every bump and every kerb, and throws a shower
  of sparks behind it. Here they come off the rear of the floor above about
  180 km/h, in bursts, far more over a kerb. Each is a short streak drawn
  along its motion relative to the camera, as a shutter would smear it, in a
  colour well over 1.0 so the camera chain's bloom makes it glow.
* **Smoke.** Rear tyres sliding, a locked brake, the handbrake: soft grey
  puffs that grow and fade.
* **Dust.** Wheels on the grass or the gravel at speed kick up a brown haze.

All particles of one kind are one vertex buffer, rebuilt with numpy each
frame and uploaded once: two draw calls for the lot, a few hundred
microseconds of Python at most, and nothing at all when nothing is alive.
"""
from __future__ import annotations

import math

import numpy as np

from . import config

_VERT = """#version 150
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec4 p3d_Color;
in vec2 p3d_MultiTexCoord0;
out vec4 col;
out vec2 uv;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    col = p3d_Color;
    uv = p3d_MultiTexCoord0;
}
"""
_FRAG = """#version 150
uniform float direct_out;
uniform float soft;            // 1: round puff, 0: streak
in vec4 col;
in vec2 uv;
out vec4 o;
void main() {
    vec2 q = uv * 2.0 - 1.0;
    float a;
    if (soft > 0.5) {
        float d = length(q);
        a = clamp(1.0 - d, 0.0, 1.0);
        a = a * a * (3.0 - 2.0 * a);
    } else {
        // Hot head, fading tail.
        a = (1.0 - q.y * q.y) * mix(0.25, 1.0, uv.x);
    }
    vec3 c = col.rgb;
    if (direct_out > 0.5) {
        c = c / (c + 0.7) * 1.2;
        c = pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
    }
    o = vec4(c, col.a * a);
}
"""

_SHADER = None


def _shader():
    global _SHADER
    if _SHADER is None:
        from panda3d.core import Shader
        _SHADER = Shader.make(Shader.SL_GLSL, _VERT, _FRAG)
    return _SHADER


class _Pool:
    """Particles of one kind and the buffer that draws them."""

    def __init__(self, n: int, additive: bool, soft: bool, name: str):
        from panda3d.core import (ColorBlendAttrib, Geom, GeomNode,
                                  GeomTriangles, GeomVertexData,
                                  TransparencyAttrib)
        from ursina import scene

        from .hudkit import _format

        self.n = n
        self.p = np.zeros((n, 3))
        self.v = np.zeros((n, 3))
        self.age = np.zeros(n)
        self.life = np.ones(n)
        self.size = np.zeros(n)
        self.grow = np.zeros(n)
        self.col = np.zeros((n, 4))
        self.alive = np.zeros(n, bool)
        self.verts = np.zeros((4 * n, 9), np.float32)
        q = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        self.verts[:, 7:9] = np.tile(q, (n, 1))
        self._shown = False

        vdata = GeomVertexData(name, _format(), Geom.UH_stream)
        vdata.unclean_set_num_rows(4 * n)
        memoryview(vdata.modify_array(0)).cast("B")[:] = self.verts.tobytes()
        prim = GeomTriangles(Geom.UH_static)
        prim.set_index_type(Geom.NT_uint32)
        idx = (np.arange(n, dtype=np.uint32)[:, None] * 4
               + np.array([0, 1, 2, 0, 2, 3], np.uint32)).ravel()
        h = prim.modify_vertices()
        h.unclean_set_num_rows(len(idx))
        memoryview(h).cast("B")[:] = idx.tobytes()
        geom = Geom(vdata)
        geom.add_primitive(prim)
        node = GeomNode(name)
        node.add_geom(geom)
        # Particles move; their bounds are not worth keeping up to date.
        node.set_final(True)
        from panda3d.core import OmniBoundingVolume
        node.set_bounds(OmniBoundingVolume())
        self.np = scene.attach_new_node(node)
        self.np.set_shader(_shader(), 10)
        self.np.set_shader_input("soft", 1.0 if soft else 0.0)
        self.np.set_depth_write(False)
        self.np.set_two_sided(True)
        self.np.set_light_off()
        self.np.set_bin("transparent", 10)
        if additive:
            self.np.set_attrib(ColorBlendAttrib.make(
                ColorBlendAttrib.M_add, ColorBlendAttrib.O_incoming_alpha,
                ColorBlendAttrib.O_one))
        else:
            self.np.set_transparency(TransparencyAttrib.M_alpha)
        # Not in either shadow pass.
        from .lighting import Sunset
        self.np.hide(Sunset.SHADOW_MASK | Sunset.BAKE_MASK)
        self.vdata = vdata

    def step(self, dt: float, gravity: float, drag: float):
        a = self.alive
        if not a.any():
            return
        self.age[a] += dt
        dead = a & (self.age >= self.life)
        self.alive[dead] = False
        a = self.alive
        self.v[a, 1] -= gravity * dt
        self.v[a] *= math.exp(-drag * dt)
        self.p[a] += self.v[a] * dt
        # A spark that hits the road skips along it.
        low = a & (self.p[:, 1] < 0.02)
        self.p[low, 1] = 0.02
        self.v[low, 1] = np.abs(self.v[low, 1]) * 0.35

    def upload(self):
        if not self.alive.any() and not self._shown:
            return
        self._shown = bool(self.alive.any())
        memoryview(self.vdata.modify_array(0)).cast("B")[:] = self.verts.tobytes()

    def destroy(self):
        self.np.remove_node()


class Effects:
    def __init__(self):
        # Room for the floor sparks of the cars round the camera, not just
        # the player's.
        self.sparks = _Pool(700, additive=True, soft=False, name="sparks")
        self.smoke = _Pool(140, additive=False, soft=True, name="smoke")
        self.rng = np.random.default_rng(3)
        self._t = 0.0
        self._acc = {"spark": 0.0, "smoke": 0.0, "dust": 0.0}
        self._cam_prev = None
        sun = np.array(config.LIGHT_SUN)
        sky = np.array(config.LIGHT_SKY)
        lit = sun * 0.45 + sky * 1.1
        self._smoke_col = np.r_[0.80 * lit, 0.30]
        self._dust_col = np.r_[np.array([0.42, 0.34, 0.24]) * lit, 0.26]

    # -- emission -------------------------------------------------------
    @staticmethod
    def _to_world(car, local):
        """Car-local (x right, y up, z forward) points to world."""
        yaw = math.radians(car.rotation_y)
        c, s = math.cos(yaw), math.sin(yaw)
        x, y, z = local[:, 0], local[:, 1], local[:, 2]
        wx = car.x + x * c + z * s
        wz = car.z - x * s + z * c
        return np.stack([wx, car.y + y, wz], axis=1)

    def _emit_sparks(self, dt, vehicle, car, key="you", phase=0.0):
        kmh = vehicle.speed * 3.6
        if kmh < 175.0 or not vehicle.on_track:
            return
        # Bottoming comes in bursts, not as a steady drizzle -- each car on
        # its own rhythm (phase), or the whole field would spark in step.
        t = self._t + phase
        burst = max(0.0, math.sin(t * 7.3) + math.sin(t * 12.9 + 1.3)
                    + 0.6 * math.sin(t * 31.0) - 0.7)
        kerb = getattr(vehicle, "grip_scale", 1.0) < 0.995
        rate = ((kmh - 175.0) / 110.0) ** 1.3 * 220.0 * burst
        if kerb:
            rate = rate * 3.0 + 120.0
        acc = self._acc.get(("spark", key), 0.0) + rate * dt
        k = int(acc)
        self._acc[("spark", key)] = acc - max(k, 0)
        if k <= 0:
            return
        r = self.rng
        local = np.stack([r.uniform(-0.45, 0.45, k), np.full(k, 0.04),
                          r.uniform(-1.85, -1.35, k)], axis=1)
        p = self._to_world(car, local)
        vx, vz = float(vehicle.vel[0]), float(vehicle.vel[1])
        base = np.array([vx, 0.0, vz])
        # Thrown off the floor: most of the car's speed, a little sideways
        # spray, and a hop.
        v = base[None, :] * r.uniform(0.55, 0.85, k)[:, None]
        v[:, 0] += r.normal(0.0, 1.6, k)
        v[:, 2] += r.normal(0.0, 1.6, k)
        v[:, 1] = r.uniform(0.4, 3.2, k)
        heat = r.uniform(0.7, 1.3, k)[:, None]
        col = np.c_[np.array([[14.0, 5.6, 1.4]]) * heat, np.ones(k)]
        free = np.flatnonzero(~self.sparks.alive)[:k]
        m = len(free)
        if m:
            sp = self.sparks
            sp.p[free] = p[:m]
            sp.v[free] = v[:m]
            sp.life[free] = r.uniform(0.12, 0.34, m)
            sp.col[free] = col[:m]
            sp.age[free] = 0.0
            sp.alive[free] = True

    def _emit_puffs(self, dt, vehicle, car, ctl):
        v = vehicle.speed
        if v < 4.0:
            return
        t = vehicle.tele
        slide = abs(math.degrees(vehicle.slip_angle))
        lock = (ctl is not None and ctl.brake > 0.9 and not vehicle.abs_enabled
                and v > 12.0)
        hand = ctl is not None and ctl.handbrake and v > 8.0
        smoking = (t.saturated_rear and v > 8.0) or slide > 9.0 or lock or hand
        off = not vehicle.on_track and vehicle.grip_scale < 0.8
        r = self.rng
        for kind, on, rate, col in (
                ("smoke", smoking, 26.0 + 2.0 * slide, self._smoke_col),
                ("dust", off and v > 6.0, 10.0 + v * 0.9, self._dust_col)):
            if not on:
                self._acc[kind] = 0.0
                continue
            self._acc[kind] += rate * dt
            k = int(self._acc[kind])
            if k <= 0:
                continue
            self._acc[kind] -= k
            side = r.choice([-1.0, 1.0], k)
            zf = r.choice([-1.42, 0.96], k) if (lock or kind == "dust") \
                else np.full(k, -1.42)
            local = np.stack([side * 0.9, np.full(k, 0.25), zf], axis=1)
            p = self._to_world(car, local)
            vel = np.zeros((k, 3))
            vel[:, 0] = float(vehicle.vel[0]) * 0.18 + r.normal(0, 0.6, k)
            vel[:, 2] = float(vehicle.vel[1]) * 0.18 + r.normal(0, 0.6, k)
            vel[:, 1] = r.uniform(0.3, 1.1, k)
            sm = self.smoke
            free = np.flatnonzero(~sm.alive)[:k]
            m = len(free)
            if not m:
                continue
            sm.p[free] = p[:m]
            sm.v[free] = vel[:m]
            sm.life[free] = r.uniform(1.1, 1.9, m)
            sm.size[free] = r.uniform(0.35, 0.6, m)
            sm.grow[free] = r.uniform(1.4, 2.4, m)
            sm.col[free] = col
            sm.age[free] = 0.0
            sm.alive[free] = True

    # -- drawing ------------------------------------------------------------
    def _build_sparks(self, cam_pos, cam_fwd, cam_vel):
        sp = self.sparks
        out = sp.verts
        a = sp.alive
        out[:, 0:3] = 0.0
        if not a.any():
            return
        idx = np.flatnonzero(a)
        p = sp.p[idx]
        # Smeared along the motion *relative to the camera*: the camera rides
        # with the car, so what the eye sees is the spark falling away.
        rel = sp.v[idx] - cam_vel[None, :]
        tail = p - rel * 0.030
        seg = p - tail
        side = np.cross(seg, cam_fwd[None, :])
        n = np.linalg.norm(side, axis=1, keepdims=True)
        side = np.where(n > 1e-6, side / np.maximum(n, 1e-6), [[0.0, 1.0, 0.0]])
        w = 0.030
        fade = (1.0 - sp.age[idx] / sp.life[idx]) ** 1.5
        rows = (idx * 4)[:, None] + np.arange(4)[None, :]
        out[rows[:, 0], 0:3] = tail - side * w
        out[rows[:, 1], 0:3] = p - side * w
        out[rows[:, 2], 0:3] = p + side * w
        out[rows[:, 3], 0:3] = tail + side * w
        c = sp.col[idx].copy()
        c[:, 3] = fade
        for j in range(4):
            out[rows[:, j], 3:7] = c

    def _build_smoke(self, right, up):
        sm = self.smoke
        out = sm.verts
        a = sm.alive
        out[:, 0:3] = 0.0
        if not a.any():
            return
        idx = np.flatnonzero(a)
        t = sm.age[idx] / sm.life[idx]
        s = (sm.size[idx] + sm.grow[idx] * np.sqrt(t))[:, None]
        p = sm.p[idx]
        rows = (idx * 4)[:, None] + np.arange(4)[None, :]
        for j, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            out[rows[:, j], 0:3] = p + (right * sx + up * sy)[None, :] * s
        c = sm.col[idx].copy()
        c[:, 3] *= (1.0 - t) ** 1.3 * np.clip(t * 6.0, 0.0, 1.0)
        for j in range(4):
            out[rows[:, j], 3:7] = c

    def update(self, dt: float, vehicle, car, ctl=None, others=()):
        """*others*: (key, vehicle, car) for the other cars near the camera,
        which throw their floor sparks too."""
        from ursina import camera
        if dt <= 0.0:
            return
        self._t += dt
        if vehicle is not None and not vehicle.frozen:
            self._emit_sparks(dt, vehicle, car)
            self._emit_puffs(dt, vehicle, car, ctl)
        for key, v, c in others:
            self._emit_sparks(dt, v, c, key=key, phase=1.7 * (key + 1))
        self.sparks.step(dt, gravity=9.8, drag=2.2)
        self.smoke.step(dt, gravity=-0.25, drag=1.6)
        if not (self.sparks.alive.any() or self.smoke.alive.any()
                or self.sparks._shown or self.smoke._shown):
            return
        cp = camera.world_position
        cam = np.array([cp.x, cp.y, cp.z])
        if self._cam_prev is None:
            self._cam_prev = cam
        cam_vel = (cam - self._cam_prev) / dt
        self._cam_prev = cam
        # A camera cut is not a camera moving at a kilometre a second.
        sp = float(np.linalg.norm(cam_vel))
        if sp > 120.0:
            cam_vel *= 120.0 / sp
        f, r_, u = camera.forward, camera.right, camera.up
        self._build_sparks(cam, np.array([f.x, f.y, f.z]), cam_vel)
        self._build_smoke(np.array([r_.x, r_.y, r_.z]), np.array([u.x, u.y, u.z]))
        self.sparks.upload()
        self.smoke.upload()

    def destroy(self):
        self.sparks.destroy()
        self.smoke.destroy()
