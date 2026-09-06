"""Validate the vehicle model against real-world numbers (no Ursina window)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from game import config
from game.surface import Surface
from game.trackdata import load_track
from game.vehicle import Controls, Vehicle

REAL_LENGTH = {"Monza": 5793, "Spa": 7004, "Silverstone": 5891,
               "Zandvoort": 4259, "Catalunya": 4675}

print("=== track geometry vs. the real circuits ===")
for name, real in REAL_LENGTH.items():
    tr = load_track(name)
    w = tr.w_left + tr.w_right
    err = 100.0 * (tr.length - real) / real
    print(f"{name:12s} {tr.length:7.0f} m (real {real}, {err:+5.1f}%)   "
          f"width {w.min():4.1f}-{w.max():4.1f} m   minR {tr.curv_radius.min():5.0f} m")


class FlatSurface:
    """Infinite grippy plane, for isolating the vehicle model."""
    def grip(self, p):
        return True, 1.0

    def resolve_body(self, p, yaw):
        return None, None, None


def new_car():
    v = Vehicle()
    v.frozen = False
    v.place((0.0, 0.0), 0.0)
    return v


DT = 1.0 / config.PHYSICS_HZ
flat = FlatSurface()

print("\n=== straight-line acceleration ===")
v = new_car()
marks = {}
t = 0.0
while t < 45.0:
    v.step(Controls(throttle=1.0), DT, flat)
    t += DT
    kmh = v.speed * 3.6
    for target in (100, 200, 300):
        if target not in marks and kmh >= target:
            marks[target] = t
print("   " + "   ".join(f"0-{k} km/h: {s:5.2f} s" for k, s in sorted(marks.items())))
print(f"   top speed: {v.speed * 3.6:.0f} km/h ({v.speed:.1f} m/s)")
assert 250 < v.speed * 3.6 < 360, "top speed out of a sane range"
assert 2.0 < marks.get(100, 99) < 5.0, "0-100 out of a sane range"

print("\n=== braking from 300 km/h ===")
# Start from a known speed rather than inheriting whatever the acceleration
# run above ended at, so the distance means something on its own.
v = new_car()
v.vel = np.array([0.0, 300 / 3.6])
d0, t0, peak_g = v.pos.copy(), 0.0, 0.0
while v.speed * 3.6 > 100 and t0 < 20:
    v.step(Controls(brake=1.0), DT, flat)
    # The *peak*, not the last sample. long_accel at the end of the run is the
    # decel at 100 km/h, where there is barely any downforce -- reading that as
    # the peak understates braking by ~0.7 g and reads like a regression.
    peak_g = max(peak_g, abs(v.tele.long_accel) / 9.81)
    t0 += DT
print(f"   300->100 km/h in {t0:4.2f} s over "
      f"{np.linalg.norm(v.pos - d0):5.1f} m  (peak {peak_g:.1f} g)")

print("\n=== trail braking: full brake while holding steering ===")
for speed_kmh in (120, 200):
    v = new_car()
    v.vel = np.array([0.0, speed_kmh / 3.6])
    for _ in range(int(1.5 / DT)):
        v.step(Controls(brake=1.0, steer=1.0), DT, flat)
    lat = abs(v.speed * v.yaw_rate) / 9.81
    print(f"   {speed_kmh:3d} km/h -> {abs(v.tele.long_accel) / 9.81:4.2f} g decel"
          f" + {lat:4.2f} g lateral, {v.speed * 3.6:5.1f} km/h after 1.5 s")
    assert lat > 0.3, "car will not turn under braking"
    assert abs(v.tele.long_accel) / 9.81 > 0.5, "car will not slow while turning"

print("\n=== steady-state cornering: full lock, held ===")
for speed_kmh in (60, 120, 200, 280):
    v = new_car()
    v.vel = np.array([0.0, speed_kmh / 3.6])
    # let the steering wind on fully, then measure
    for _ in range(int(2.5 / DT)):
        v.step(Controls(throttle=0.30, steer=1.0), DT, flat)
    r = abs(v.speed / v.yaw_rate) if abs(v.yaw_rate) > 1e-6 else float("inf")
    g = abs(v.speed * v.yaw_rate) / 9.81
    print(f"   {speed_kmh:3d} km/h -> radius {r:6.1f} m, {g:4.2f} g lateral, "
          f"slip F/R {math.degrees(v.tele.slip_front):5.1f}/"
          f"{math.degrees(v.tele.slip_rear):5.1f}°, "
          f"steer {math.degrees(v.steer_angle):4.1f}°")
    assert g < 4.0, "cornering g is not physical"

print("\n=== steering response: how fast does a tap take effect? ===")
v = new_car()
v.vel = np.array([0.0, 200 / 3.6])
for step in range(int(1.0 / DT) + 1):
    if step * DT in (0.0,) or abs(step * DT - round(step * DT, 2)) < 1e-9:
        pass
    v.step(Controls(throttle=0.3, steer=1.0), DT, flat)
    tt = (step + 1) * DT
    if abs(tt - 0.1) < DT / 2 or abs(tt - 0.25) < DT / 2 or abs(tt - 0.5) < DT / 2:
        print(f"   t={tt:4.2f}s  wheel {math.degrees(v.steer_angle):5.2f}°  "
              f"yaw rate {math.degrees(v.yaw_rate):6.2f}°/s")

print("\n=== full laps: autopilot vs. real GT3 lap times ===")
# The car is specced as a GT (0-100 in ~3.6 s, 279 km/h, ~2 g braking), so GT3
# times are the right yardstick -- not F1 pole.
from game.autopilot import Autopilot

REFERENCE = {"Monza": 107.0, "Spa": 137.0, "Silverstone": 118.0,
             "Zandvoort": 96.0, "Catalunya": 104.0}

for name, pole in REFERENCE.items():
    tr = load_track(name)
    surf = Surface(tr)
    v = new_car()
    pos, yaw = tr.start_pose()
    v.place(pos, yaw)
    ap = Autopilot(tr, surf)

    n = tr.count
    t, off_max, vmax = 0.0, 0.0, 0.0
    lap_times, armed, lap_start = [], False, None
    while t < 420.0 and len(lap_times) < 1:
        v.step(ap.controls(v), DT, surf)
        t += DT
        i, off = surf.progress(v.pos)
        off_max = max(off_max, abs(off))
        vmax = max(vmax, v.speed)
        assert np.isfinite(v.pos).all(), "NaN in position"
        if 0.4 * n <= i <= 0.6 * n:
            armed = True
        elif armed and i < 0.1 * n:
            armed = False
            if lap_start is not None:
                lap_times.append(t - lap_start)
            lap_start = t

    best = min(lap_times) if lap_times else float("nan")
    half = float(tr.w_left.max())
    where = "on track" if off_max <= half else \
            "runoff" if off_max <= half + config.RUNOFF_WIDTH * 0.9 else "hit wall"
    print(f"   {name:12s} lap {best:6.1f} s (GT3 ~{pole:.0f} s, "
          f"{100 * (best - pole) / pole:+4.0f}%)   "
          f"vmax {vmax * 3.6:3.0f} km/h   widest {off_max:4.1f} m ({where})")
    assert lap_times, f"autopilot never completed a lap at {name}"
    # A centreline follower is allowed to use the runoff; it must not crash.
    assert off_max < half + config.RUNOFF_WIDTH, f"{name}: left the circuit"
    assert best < pole * 1.8, f"{name}: implausibly slow"

# ---------------------------------------------------------------------------
# Wall contact: the body has extent, and an impact off the centreline spins it
# ---------------------------------------------------------------------------
print()
print("=== wall contact: body box, and rotation from an off-centre hit ===")

_tr = load_track("Monza")
_surf = Surface(_tr)
_segs = np.concatenate(_tr.barrier_lines())


def _seg_dist(p):
    """Distance from a point to the barrier polyline."""
    a, b = _segs[:, 0], _segs[:, 1]
    d = b - a
    L2 = np.maximum((d * d).sum(axis=1), 1e-12)
    t = np.clip(((p - a) * d).sum(axis=1) / L2, 0.0, 1.0)
    q = a + d * t[:, None]
    return float(np.sqrt(((p - q) ** 2).sum(axis=1)).min())


def _drive_into_wall(angle_deg: float, speed: float):
    """Aim the car at the right-hand barrier at *angle_deg* off parallel.

    Returns (metres the deepest body corner ends up past the wall, worst yaw
    rate during contact, slowest speed reached).
    """
    i = 40                                   # on the main straight
    off_r, _ = _tr.wall_offsets()
    limit = float(off_r[i])
    heading = math.atan2(_tr.tangent[i][0], _tr.tangent[i][1])
    yaw = heading + math.radians(angle_deg)
    # Close enough that even a 2 degree approach reaches the barrier inside
    # the run: starting 12 m off, the shallow case never touched it and the
    # comparison against a square hit was vacuous.
    start = _tr.center[i] + _tr.normal[i] * (limit - 2.0)
    v = new_car()
    v.place((float(start[0]), float(start[1])), yaw)
    v.vel = np.array([math.sin(yaw), math.cos(yaw)]) * speed
    v.yaw_rate = 0.0

    worst_through, spin, kept, touched = -1e9, 0.0, speed, False
    for _ in range(int(3.0 / DT)):
        v.step(Controls(), DT, _surf)
        assert np.isfinite(v.pos).all(), "NaN after wall contact"
        fwd = np.array([math.sin(v.yaw), math.cos(v.yaw)])
        right = np.array([math.cos(v.yaw), -math.sin(v.yaw)])
        hw = right * config.BODY_HALF_WIDTH
        for c in (v.pos + fwd * config.BODY_TO_FRONT + hw,
                  v.pos + fwd * config.BODY_TO_FRONT - hw,
                  v.pos - fwd * config.BODY_TO_REAR + hw,
                  v.pos - fwd * config.BODY_TO_REAR - hw):
            # Measured against the barrier the eye sees, not against a wall
            # offset taken from the nearest centreline sample: the two are
            # different curves through a corner, and the second one is not
            # what the car is being stopped by any more.
            worst_through = max(worst_through,
                                config.BARRIER_HALF_DEPTH - _seg_dist(c))
        if v.hit_wall:
            touched = True
            spin = max(spin, abs(math.degrees(v.yaw_rate)))
        kept = min(kept, v.speed)
    assert touched, f"the car never reached the wall at {angle_deg:.0f} deg"
    return worst_through, spin, kept


_wall = {}
for ang, label in ((2.0, "graze"), (20.0, "clip"), (85.0, "square")):
    through, spin, kept = _drive_into_wall(ang, 45.0)
    _wall[label] = (through, spin, kept)
    print(f"   {label:7s} at {ang:4.0f}deg:  deepest corner past the barrier "
          f"{through:+5.2f} m   spin {spin:5.1f}deg/s   "
          f"slowest {kept * 3.6:5.1f} km/h")
    # A step's worth of travel may compress into the wall before the impulse
    # answers, but the body must never end up buried in it.
    assert through < 1.0, f"{label}: body {through:.2f} m through the barrier"

assert _wall["clip"][1] > _wall["square"][1] + 5.0, (
    "an off-centre clip must rotate the car more than a square hit "
    f"({_wall['clip'][1]:.1f} vs {_wall['square'][1]:.1f} deg/s)")
assert _wall["graze"][2] > _wall["square"][2], (
    "a graze must cost less speed than a square hit "
    f"({_wall['graze'][2]:.1f} vs {_wall['square'][2]:.1f} m/s)")

print("\nOK")
