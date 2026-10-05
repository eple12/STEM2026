"""Where the structures that stand across the circuit go -- the start gantry
and the spectator bridges -- and the solid feet they stand on.

Kept free of Ursina so both sides can use it: the scenery places the models
from it, and the physics (``surface.Surface.resolve_body``, in the game and
in the field's worker process alike) collides the cars with their legs. One
description, so a car can never drive through a leg it can see, nor hit one
that is not there.

The model dimensions are the assets' own (``assets/models/circuit``,
authored in ``blender/circuit_kit.py``): X across the road, Z along it.
"""
from __future__ import annotations

import math

import numpy as np

from . import config

#: gantry.bam: 19.0 m wide; two lattice legs 0.99 m square at x = +-8.4.
GANTRY_W = 19.0
GANTRY_LEG_X = 8.4
GANTRY_LEG_HALF = 0.495
#: bridge.bam: 25.6 m wide; concrete towers from x = 9.6 to 12.4, 4.8 m deep.
BRIDGE_W = 25.6
BRIDGE_TOWER_X = (9.6, 12.4)
BRIDGE_TOWER_HALF_Z = 2.4


def across_span(track, i: int, clear: float,
                min_extra: float = config.CAR_BODY_WIDTH,
                leg_frac: float = 1.0) -> tuple[np.ndarray, float]:
    """(centre, full span) for a structure standing across the circuit.

    Its legs go *clear* metres outside the asphalt on both sides, never past
    the barrier, and it is centred on the road rather than midway between the
    two walls. Where the barrier is close enough that the clamp bites,
    *min_extra* is the floor and it wins: a full car's width of clear ground
    between the white line and the nearest leg. *leg_frac* is where the
    structure's legs actually stand, as a fraction of the half-span, so that
    it is the *leg*, not the span edge, that keeps clear.
    """
    off_r, off_l = track.wall_ray_offsets()
    w = float(max(track.w_right[i], track.w_left[i]))
    # Less than the leg's own half-width from the fence is a leg through
    # the fence, so the room stops short of it by more than a token metre.
    room = float(min(off_r[i], off_l[i])) - 1.5
    floor_half = (w + min_extra) / leg_frac
    want_half = (w + clear) / leg_frac
    half = max(min(want_half, max(room, floor_half)), floor_half)
    return track.center[i], 2.0 * half


def gantry_place(track):
    """(sample index, centre, span) of the start gantry."""
    p, span = across_span(track, 0, config.GANTRY_LEG_CLEAR, leg_frac=0.80)
    return 0, p, span


def bridge_places(track) -> list:
    """[(sample index, centre, span)] of the spectator bridges."""
    out = []
    n = track.count
    win = max(4, n // 50)
    for f in np.linspace(0.0, 1.0, config.BRIDGE_COUNT, endpoint=False)[1:]:
        i = int(np.searchsorted(track.arclen, f * track.length)) % n
        # Slide to the straightest sample nearby: a bridge is a rigid beam set
        # square to the tangent, so on a bend one pillar swings in towards the
        # apex and the along-normal clearance across_span guarantees is not
        # the clearance the car actually sees.
        js = (np.arange(i - win, i + win) % n)
        i = int(js[np.argmax(track.curv_radius[js])])
        p, span = across_span(track, i, config.BRIDGE_LEG_CLEAR, leg_frac=0.75)
        out.append((i, p, span))
    return out


def obstacles(track) -> list:
    """The legs as oriented boxes: [(cx, cz, ux, uz, half_across, half_along)]
    with (ux, uz) the across-the-road axis. Cached on the track."""
    got = getattr(track, "_structure_obstacles", None)
    if got is not None:
        return got
    boxes = []

    def legs(i, p, span, authored, xs, half_x, half_z):
        sx = span / authored
        t = track.tangent[i]
        # The models are placed square to the tangent (scenery: yaw_at), so
        # their X runs along the tangent's perpendicular, not along the
        # sample's stored normal.
        u = np.array((t[1], -t[0]))
        for x in xs:
            c = p + u * (x * sx)
            boxes.append((float(c[0]), float(c[1]), float(u[0]), float(u[1]),
                          half_x * sx, half_z))

    if getattr(config, "GANTRY_ENABLED", True):
        i, p, span = gantry_place(track)
        legs(i, p, span, GANTRY_W, (-GANTRY_LEG_X, GANTRY_LEG_X),
             GANTRY_LEG_HALF, GANTRY_LEG_HALF)
    if config.BRIDGE_COUNT > 1:
        mid = 0.5 * (BRIDGE_TOWER_X[0] + BRIDGE_TOWER_X[1])
        half = 0.5 * (BRIDGE_TOWER_X[1] - BRIDGE_TOWER_X[0])
        for i, p, span in bridge_places(track):
            legs(i, p, span, BRIDGE_W, (-mid, mid), half, BRIDGE_TOWER_HALF_Z)
    track._structure_obstacles = boxes
    return boxes


def box_hit(cx: float, cz: float, yaw: float, front: float, rear: float,
            half_w: float, box) -> tuple[float, float, float, float, float] | None:
    """Separating-axis test of the car's body box against one leg.

    The car: centre of mass (cx, cz), heading *yaw*, body *front* metres
    ahead and *rear* behind it, *half_w* either side. Returns (depth,
    normal x, normal z, contact x, contact z) -- the normal pointing from
    the leg towards the car -- or None when they do not touch. A leg is
    narrower than a car, so testing the car's corners against it (as the
    barriers do) would let a nose drive straight through one; this tests
    both boxes' faces."""
    ox, oz, ux, uz, hu, hv = box
    vx, vz = -uz, ux                       # along the road
    fx, fz = math.sin(yaw), math.cos(yaw)
    rx, rz = fz, -fx
    off = 0.5 * (front - rear)
    hf = 0.5 * (front + rear)
    bx, bz = cx + fx * off, cz + fz * off  # the body box's own centre
    dx, dz = bx - ox, bz - oz
    reach = hf + half_w + hu + hv
    if dx * dx + dz * dz > reach * reach:
        return None
    best = None
    for ax, az in ((fx, fz), (rx, rz), (ux, uz), (vx, vz)):
        d = dx * ax + dz * az
        ra = hf * abs(fx * ax + fz * az) + half_w * abs(rx * ax + rz * az)
        rb = hu * abs(ux * ax + uz * az) + hv * abs(vx * ax + vz * az)
        over = ra + rb - abs(d)
        if over <= 0.0:
            return None
        if best is None or over < best[0]:
            s = 1.0 if d >= 0.0 else -1.0
            best = (over, ax * s, az * s)
    depth, nx, nz = best
    # Where they touch, near enough for the impulse's arm: the point of the
    # leg nearest the body's centre.
    a = max(-hu, min(hu, dx * ux + dz * uz))
    b = max(-hv, min(hv, dx * vx + dz * vz))
    return depth, nx, nz, ox + ux * a + vx * b, oz + uz * a + vz * b
