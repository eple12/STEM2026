"""Car-to-car contact: two body boxes, one impulse.

The same physics as the barrier contact in ``Vehicle.step``, with the second
body free to move: the impulse acts at the contact point, so a nudge on the
rear corner spins the car that is hit and a square rear-ending mostly trades
speed. Both cars' yaw inertia enters the effective mass, exactly as the
barrier's does.

Overlap is found with the separating-axis test on the two body rectangles
(``BODY_TO_FRONT``/``BODY_TO_REAR``/``BODY_HALF_WIDTH`` -- the box the barrier
uses), which for two rectangles needs only their four edge directions.
"""
from __future__ import annotations

import math

import numpy as np

from . import config

#: Bounce between two cars. Lower than a barrier's: bodywork crumples into
#: bodywork, and a high value makes every touch a pinball launch.
RESTITUTION = 0.15
#: Friction across the contact, as a share of the normal impulse.
FRICTION = 0.35
#: Above this many N*s the touch counts as a real hit for the race's records
#: (a car's mass times ~1 m/s of velocity change).
HIT_IMPULSE = 1200.0


def _frame(v):
    fwd = np.array([math.sin(v.yaw), math.cos(v.yaw)])
    right = np.array([math.cos(v.yaw), -math.sin(v.yaw)])
    return fwd, right


def _corners(v, fwd, right):
    hw = right * config.BODY_HALF_WIDTH
    nose = fwd * config.BODY_TO_FRONT
    tail = -fwd * config.BODY_TO_REAR
    c = v.pos
    return np.array([c + nose + hw, c + nose - hw, c + tail - hw, c + tail + hw])


def _inside(p, v, fwd, right) -> bool:
    rel = p - v.pos
    a = float(np.dot(rel, fwd))
    b = float(np.dot(rel, right))
    return (-config.BODY_TO_REAR <= a <= config.BODY_TO_FRONT
            and abs(b) <= config.BODY_HALF_WIDTH)


def overlap(a, b):
    """(depth, unit normal from a to b, contact point) or None."""
    fa, ra = _frame(a)
    fb, rb = _frame(b)
    ca = _corners(a, fa, ra)
    cb = _corners(b, fb, rb)
    depth, axis = math.inf, None
    for ax in (fa, ra, fb, rb):
        pa = ca @ ax
        pb = cb @ ax
        o = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
        if o <= 0.0:
            return None                         # a separating axis
        if o < depth:
            depth, axis = o, ax
    n = axis if float(np.dot(b.pos - a.pos, axis)) >= 0.0 else -axis
    # Contact at the corners that are actually inside the other body; a
    # glancing touch where none is (edge across edge) falls back to between
    # the two boxes along the normal.
    pts = [p for p in ca if _inside(p, b, fb, rb)]
    pts += [p for p in cb if _inside(p, a, fa, ra)]
    if pts:
        contact = np.mean(np.asarray(pts), axis=0)
    else:
        contact = 0.5 * (a.pos + b.pos)
    return depth, n, contact


def collide(a, b) -> float:
    """Resolve contact between two vehicles in place. Returns the normal
    impulse applied (0.0 when they are apart or already separating)."""
    hit = overlap(a, b)
    if hit is None:
        return 0.0
    depth, n, contact = hit
    m = config.CAR_MASS
    inertia = config.YAW_INERTIA
    # Push apart, half each, so neither is left inside the other next step.
    a.pos = a.pos - n * (0.5 * depth)
    b.pos = b.pos + n * (0.5 * depth)

    arm_a = contact - a.pos
    arm_b = contact - b.pos
    # How each contact point moves per unit yaw rate (Vehicle.step's spin).
    spin_a = np.array([arm_a[1], -arm_a[0]])
    spin_b = np.array([arm_b[1], -arm_b[0]])
    va = a.vel + a.yaw_rate * spin_a
    vb = b.vel + b.yaw_rate * spin_b
    vn = float(np.dot(vb - va, n))
    if vn >= 0.0:
        return 0.0                              # already moving apart
    sa = float(np.dot(spin_a, n))
    sb = float(np.dot(spin_b, n))
    k = 2.0 / m + sa * sa / inertia + sb * sb / inertia
    jn = -(1.0 + RESTITUTION) * vn / k
    a.vel = a.vel - (jn / m) * n
    b.vel = b.vel + (jn / m) * n
    a.yaw_rate -= jn * sa / inertia
    b.yaw_rate += jn * sb / inertia

    # Friction along the contact, capped by the normal impulse.
    tang = np.array([-n[1], n[0]])
    va = a.vel + a.yaw_rate * spin_a
    vb = b.vel + b.yaw_rate * spin_b
    vt = float(np.dot(vb - va, tang))
    ta = float(np.dot(spin_a, tang))
    tb = float(np.dot(spin_b, tang))
    kt = 2.0 / m + ta * ta / inertia + tb * tb / inertia
    jt = max(-FRICTION * jn, min(FRICTION * jn, -vt / kt))
    a.vel = a.vel - (jt / m) * tang
    b.vel = b.vel + (jt / m) * tang
    a.yaw_rate -= jt * ta / inertia
    b.yaw_rate += jt * tb / inertia
    return jn
