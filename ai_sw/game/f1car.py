"""A Formula 1 car, built here from primitives -- no asset file.

Modelled on a current-generation car: a raised nose over a two-element front
wing with endplates, a narrow monocoque with the halo over the cockpit, an
airbox above the driver's head running back into a sharkfin engine cover,
undercut sidepods, a flat floor with a diffuser, a rear wing with a DRS flap
behind and above the mainplane, endplates and a beam wing on swan-neck pylons,
push-rod suspension, and 18-inch wheels with a rim recessed into a wide tyre.

Everything is dimensioned around the physics car (``config.WHEELBASE``,
``WHEEL_HALF_TRACK``, ``BODY_TO_FRONT`` / ``BODY_TO_REAR``), so the model that
is drawn is the box the collision code believes in, and ``Car`` finds it at
scale 1 with the wheel nodes it expects (``wheelFrontLeft`` etc.).

Every face is emitted with an explicit outward normal and its winding fixed to
match, because Panda culls back faces and a face wound the wrong way is a hole
you can see straight through. Caps are ear-clipped rather than fanned from the
centroid, so a concave profile (the sharkfin) closes without inverted slivers.
Anything that bridges two parts -- wishbones, wing pylons, mirror stalks --
takes its anchor from ``_tub`` / the wing profiles rather than a guessed
number, so nothing floats in mid-air and nothing pokes out the other side.

Two liveries: ``f1red`` for the player, ``f1white`` for the AI. Same shape,
different colours.

Coordinates: +z forward, +y up, +x right, origin at the centre of gravity
projected to the ground. Metres throughout.
"""
from __future__ import annotations

import math

import numpy as np
from ursina import Entity, Mesh

from . import config

X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


# --- palette --------------------------------------------------------------
def _c(r, g, b):
    return (r / 255.0, g / 255.0, b / 255.0, 1.0)


CARBON = _c(26, 26, 31)
CARBON_LT = _c(52, 53, 60)
TITANIUM = _c(150, 154, 162)
TYRE = _c(30, 30, 34)
TYRE_WALL = _c(44, 44, 50)
RIM = _c(126, 130, 140)
RIM_DARK = _c(58, 60, 68)
HUB = _c(92, 95, 103)
VISOR = _c(20, 20, 25)
GLASS = _c(150, 168, 188)
RAIN_OFF = _c(92, 12, 12)
RAIN_ON = _c(255, 46, 34)

LIVERIES = {
    "f1red": dict(body=_c(206, 26, 34), accent=_c(242, 242, 246),
                  helmet=_c(242, 242, 246)),
    "f1white": dict(body=_c(236, 236, 240), accent=_c(206, 26, 34),
                    helmet=_c(206, 26, 34)),
}

# --- key dimensions, taken from the physics box ------------------------
ZF = config.CG_TO_FRONT                 # front axle
ZR = -config.CG_TO_REAR                 # rear axle
TRACK = config.WHEEL_HALF_TRACK         # wheel centre from the centreline
NOSE = config.BODY_TO_FRONT             # front wing leading edge
TAIL = -config.BODY_TO_REAR             # rear wing trailing edge
R_WHEEL = 0.33
W_FRONT = 0.30
W_REAR = 0.40

#: Monocoque sections, front to back: (z, half width, y bottom, y top).
#: Anything that bolts to the tub reads its anchor out of this table.
TUB = [
    (2.08, 0.050, 0.275, 0.335),
    (1.86, 0.075, 0.250, 0.375),
    (1.60, 0.110, 0.215, 0.430),
    (1.28, 0.165, 0.150, 0.500),
    (0.88, 0.235, 0.100, 0.590),
    (0.42, 0.295, 0.080, 0.665),
    (-0.10, 0.315, 0.080, 0.715),
    (-0.70, 0.295, 0.080, 0.650),
    (-1.30, 0.220, 0.100, 0.555),
    (-1.85, 0.130, 0.195, 0.440),
    (-2.06, 0.080, 0.255, 0.375),
]


def _tub(z: float) -> tuple[float, float, float]:
    """(half width, y bottom, y top) of the monocoque at depth *z*."""
    if z >= TUB[0][0]:
        return TUB[0][1], TUB[0][2], TUB[0][3]
    if z <= TUB[-1][0]:
        return TUB[-1][1], TUB[-1][2], TUB[-1][3]
    for a, b in zip(TUB, TUB[1:]):
        if b[0] <= z <= a[0]:
            t = (a[0] - z) / (a[0] - b[0])
            return tuple(a[k] + (b[k] - a[k]) * t for k in (1, 2, 3))
    return TUB[-1][1], TUB[-1][2], TUB[-1][3]


# --- wing profiles, as (z, y) loops ------------------------------------
FW_MAIN = [(NOSE, 0.108), (NOSE - 0.10, 0.152), (NOSE - 0.27, 0.158),
           (NOSE - 0.40, 0.138), (NOSE - 0.38, 0.108), (NOSE - 0.21, 0.092)]
FW_FLAP = [(NOSE - 0.30, 0.186), (NOSE - 0.42, 0.244), (NOSE - 0.54, 0.278),
           (NOSE - 0.58, 0.252), (NOSE - 0.46, 0.208)]
FW_PLATE = [(NOSE + 0.035, 0.055), (NOSE + 0.035, 0.235), (NOSE - 0.24, 0.300),
            (NOSE - 0.60, 0.286), (NOSE - 0.62, 0.055)]
FW_SPAN = 1.10

RW_MAIN = [(-1.78, 0.756), (-1.90, 0.796), (-2.05, 0.790), (-2.17, 0.760),
           (-2.05, 0.733), (-1.90, 0.726)]
RW_FLAP = [(-2.05, 0.842), (-2.14, 0.890), (-2.24, 0.924), (-2.28, 0.932),
           (-2.22, 0.900), (-2.10, 0.858)]
RW_PLATE = [(-1.75, 0.700), (-1.78, 0.962), (-2.26, 0.986), (-2.30, 0.700),
            (-2.22, 0.442), (-1.90, 0.452)]
RW_BEAM = [(-2.00, 0.456), (-2.10, 0.482), (-2.20, 0.470), (-2.16, 0.440),
           (-2.03, 0.436)]
RW_SPAN = 0.49


def _profile_y(poly, z, upper=True):
    """Upper (or lower) surface height of a (z, y) loop at depth *z*."""
    ys = []
    m = len(poly)
    for i in range(m):
        z0, y0 = poly[i]
        z1, y1 = poly[(i + 1) % m]
        if z0 == z1:
            continue
        if min(z0, z1) - 1e-9 <= z <= max(z0, z1) + 1e-9:
            ys.append(y0 + (y1 - y0) * (z - z0) / (z1 - z0))
    if not ys:
        return None
    return max(ys) if upper else min(ys)


# --- mesh core ---------------------------------------------------------
class _Part:
    """Flat-shaded triangles for one mesh, each with an explicit normal."""

    def __init__(self):
        self.v, self.n, self.c = [], [], []

    def tri(self, a, b, c, col, out):
        a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
        g = np.cross(b - a, c - a)
        L = float(np.linalg.norm(g))
        if L < 1e-12:
            return                       # degenerate, nothing to draw
        g /= L
        out = np.asarray(out, dtype=float)
        ol = float(np.linalg.norm(out))
        if ol > 1e-12 and float(np.dot(g, out / ol)) < 0.0:
            b, c = c, b                  # Panda culls back faces: rewind it
            g = -g
        for p in (a, b, c):
            self.v.append((float(p[0]), float(p[1]), float(p[2])))
            self.n.append((float(g[0]), float(g[1]), float(g[2])))
            self.c.append(col)

    def quad(self, a, b, c, d, col, out):
        self.tri(a, b, c, col, out)
        self.tri(a, c, d, col, out)

    def mesh(self) -> Mesh:
        return Mesh(vertices=self.v, triangles=list(range(len(self.v))),
                    normals=self.n, colors=self.c, mode="triangle", static=True)


def _area2(p) -> float:
    s = 0.0
    m = len(p)
    for i in range(m):
        u0, v0 = p[i]
        u1, v1 = p[(i + 1) % m]
        s += u0 * v1 - u1 * v0
    return 0.5 * s


def _in_tri(q, a, b, c) -> bool:
    def side(p0, p1, p2):
        return ((p1[0] - p0[0]) * (p2[1] - p0[1])
                - (p1[1] - p0[1]) * (p2[0] - p0[0]))
    d1, d2, d3 = side(q, a, b), side(q, b, c), side(q, c, a)
    neg = (d1 < -1e-12) or (d2 < -1e-12) or (d3 < -1e-12)
    pos = (d1 > 1e-12) or (d2 > 1e-12) or (d3 > 1e-12)
    return not (neg and pos)


def _ear_clip(p):
    """Triangulate a simple 2D polygon, CCW. Falls back to a centroid fan."""
    m = len(p)
    if m < 3:
        return []
    idx = list(range(m))
    if _area2(p) < 0.0:
        idx.reverse()
    tris, guard = [], 0
    while len(idx) > 3 and guard < 4 * m + 16:
        guard += 1
        for k in range(len(idx)):
            i0 = idx[k - 1]
            i1 = idx[k]
            i2 = idx[(k + 1) % len(idx)]
            a, b, c = p[i0], p[i1], p[i2]
            cross = ((b[0] - a[0]) * (c[1] - a[1])
                     - (b[1] - a[1]) * (c[0] - a[0]))
            if cross <= 1e-12:                 # reflex or collinear
                continue
            if any(_in_tri(p[m2], a, b, c)
                   for m2 in idx if m2 not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            break
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    if not tris:                                # degenerate: fan it
        tris = [(0, i, i + 1) for i in range(1, m - 1)]
    return tris


def _cap(part, ring, col, out):
    """Close a ring that lies in a constant-z plane, facing *out*."""
    ring = np.asarray(ring, dtype=float)
    flat = [(float(q[0]), float(q[1])) for q in ring]
    for i0, i1, i2 in _ear_clip(flat):
        part.tri(ring[i0], ring[i1], ring[i2], col, out)


def _loft(part, rings, col, cap0=True, cap1=True, closed=False):
    """Skin a run of convex rings (same point count) into a tube."""
    rings = [np.asarray(r, dtype=float) for r in rings]
    pairs = list(zip(rings, rings[1:]))
    if closed:
        pairs.append((rings[-1], rings[0]))
    for r0, r1 in pairs:
        axis = (r0.mean(axis=0) + r1.mean(axis=0)) / 2.0
        k = len(r0)
        for i in range(k):
            j = (i + 1) % k
            a, b, c, d = r0[i], r0[j], r1[j], r1[i]
            part.quad(a, b, c, d, col, (a + b + c + d) / 4.0 - axis)
    if not closed:
        if cap0:
            _cap(part, rings[0], col, rings[0].mean(0) - rings[1].mean(0))
        if cap1:
            _cap(part, rings[-1], col, rings[-1].mean(0) - rings[-2].mean(0))


def _rrect(z, hw, y0, y1, r=0.38, n=3, x=0.0):
    """Rounded-rectangle ring in the x-y plane at depth z, CCW."""
    rad = r * min(hw, (y1 - y0) / 2.0)
    cx = (hw - rad, -(hw - rad), -(hw - rad), hw - rad)
    cy = (y1 - rad, y1 - rad, y0 + rad, y0 + rad)
    pts = []
    for k in range(4):
        for t in range(n + 1):
            ang = math.radians(90.0 * k + 90.0 * t / n)
            pts.append((x + cx[k] + rad * math.cos(ang),
                        cy[k] + rad * math.sin(ang), z))
    return np.asarray(pts)


def _extrude_x(part, poly_zy, x0, x1, col):
    """A closed (z, y) loop extruded across x, sealed at both ends."""
    p = [(float(z), float(y)) for z, y in poly_zy]
    if _area2(p) < 0.0:
        p = p[::-1]                       # CCW in (z, y)
    lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
    A = [np.array([lo, y, z]) for z, y in p]
    B = [np.array([hi, y, z]) for z, y in p]
    m = len(p)
    for i in range(m):
        j = (i + 1) % m
        dz = p[j][0] - p[i][0]
        dy = p[j][1] - p[i][1]
        # CCW in (z, y): the outward edge normal is (dy, -dz).
        nrm = np.array([0.0, -dz, dy])
        if float(np.linalg.norm(nrm)) < 1e-12:
            continue
        part.quad(A[i], A[j], B[j], B[i], col, nrm)
    for i0, i1, i2 in _ear_clip(p):
        part.tri(B[i0], B[i1], B[i2], col, X)
        part.tri(A[i0], A[i1], A[i2], col, -X)


def _box(part, lo, hi, col):
    x0, y0, z0 = (min(lo[k], hi[k]) for k in range(3))
    x1, y1, z1 = (max(lo[k], hi[k]) for k in range(3))
    P = lambda x, y, z: np.array([x, y, z], dtype=float)  # noqa: E731
    part.quad(P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1), col, Z)
    part.quad(P(x0, y0, z0), P(x1, y0, z0), P(x1, y1, z0), P(x0, y1, z0), col, -Z)
    part.quad(P(x1, y0, z0), P(x1, y0, z1), P(x1, y1, z1), P(x1, y1, z0), col, X)
    part.quad(P(x0, y0, z0), P(x0, y0, z1), P(x0, y1, z1), P(x0, y1, z0), col, -X)
    part.quad(P(x0, y1, z0), P(x1, y1, z0), P(x1, y1, z1), P(x0, y1, z1), col, Y)
    part.quad(P(x0, y0, z0), P(x1, y0, z0), P(x1, y0, z1), P(x0, y0, z1), col, -Y)


def _frame(d):
    d = np.asarray(d, dtype=float)
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    up = Y if abs(float(np.dot(d, Y))) < 0.95 else X
    u = np.cross(d, up)
    u /= max(float(np.linalg.norm(u)), 1e-9)
    return u, np.cross(u, d)


def _bar(part, p0, p1, hw, hh, col):
    """A rectangular-section strut; the end caps sit exactly on p0 and p1."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    u, w = _frame(p1 - p0)
    ring = lambda p: np.asarray([p + u * hw + w * hh, p - u * hw + w * hh,  # noqa: E731
                                 p - u * hw - w * hh, p + u * hw - w * hh])
    r0, r1 = ring(p0), ring(p1)
    axis = (p0 + p1) / 2.0
    for i in range(4):
        j = (i + 1) % 4
        a, b, c, d = r0[i], r0[j], r1[j], r1[i]
        part.quad(a, b, c, d, col, (a + b + c + d) / 4.0 - axis)
    for r, out in ((r0, p0 - p1), (r1, p1 - p0)):
        part.tri(r[0], r[1], r[2], col, out)
        part.tri(r[0], r[2], r[3], col, out)


def _tube(part, path, r, col, n=8, closed=False):
    """A round tube along a polyline."""
    path = [np.asarray(p, dtype=float) for p in path]
    m = len(path)
    rings = []
    for i, p in enumerate(path):
        if closed:
            d = path[(i + 1) % m] - path[i - 1]
        else:
            d = path[min(i + 1, m - 1)] - path[max(i - 1, 0)]
        u, w = _frame(d)
        rings.append(np.asarray(
            [p + (u * math.cos(2 * math.pi * k / n)
                  + w * math.sin(2 * math.pi * k / n)) * r for k in range(n)]))
    pairs = list(zip(rings, rings[1:]))
    centres = list(zip(path, path[1:]))
    if closed:
        pairs.append((rings[-1], rings[0]))
        centres.append((path[-1], path[0]))
    for (r0, r1), (c0, c1) in zip(pairs, centres):
        axis = (c0 + c1) / 2.0
        for i in range(n):
            j = (i + 1) % n
            a, b, c, d = r0[i], r0[j], r1[j], r1[i]
            part.quad(a, b, c, d, col, (a + b + c + d) / 4.0 - axis)
    if not closed:
        _cap_free(part, rings[0], col, path[0] - path[1])
        _cap_free(part, rings[-1], col, path[-1] - path[-2])


def _cap_free(part, ring, col, out):
    """Close a ring in any plane, by fanning from its centroid (convex only)."""
    ring = np.asarray(ring, dtype=float)
    cen = ring.mean(axis=0)
    k = len(ring)
    for i in range(k):
        part.tri(cen, ring[i], ring[(i + 1) % k], col, out)


def _sphere(part, centre, r, col, n_lat=8, n_lon=16):
    """A y-polar sphere; every normal is exact."""
    c = np.asarray(centre, dtype=float)

    def pt(i, j):
        phi = math.pi * i / n_lat
        th = 2.0 * math.pi * j / n_lon
        return c + r * np.array([math.sin(phi) * math.cos(th), math.cos(phi),
                                 math.sin(phi) * math.sin(th)])

    top, bot = c + Y * r, c - Y * r
    for j in range(n_lon):
        k = (j + 1) % n_lon
        part.tri(top, pt(1, j), pt(1, k), col, (pt(1, j) + pt(1, k)) / 2.0 - c)
        part.tri(bot, pt(n_lat - 1, k), pt(n_lat - 1, j), col,
                 (pt(n_lat - 1, j) + pt(n_lat - 1, k)) / 2.0 - c)
    for i in range(1, n_lat - 1):
        for j in range(n_lon):
            k = (j + 1) % n_lon
            a, b, d, e = pt(i, j), pt(i, k), pt(i + 1, k), pt(i + 1, j)
            part.quad(a, b, d, e, col, (a + b + d + e) / 4.0 - c)


# --- wheel -------------------------------------------------------------
def _wheel_part(width: float) -> "_Part":
    """A tyre with a recessed rim on both faces, axle along x, centred.

    Built face by face with exact normals so the outer face is sealed: the
    recess used to read as a tunnel straight through the wheel because a
    fanned cap can invert, and an inverted cap is invisible.
    """
    part = _Part()
    n = 28
    hw = width / 2.0
    R = R_WHEEL
    R_SH = R - 0.030        # shoulder, where the tread rolls into the sidewall
    R_RIM = 0.212
    R_HUB = 0.078
    tread_hw = hw - 0.028
    depth = 0.032           # how far the rim face sits inside the tyre face

    def rad(i):
        a = 2.0 * math.pi * i / n
        return np.array([0.0, math.cos(a), math.sin(a)])

    def ring(x, r):
        return [np.array([x, 0.0, 0.0]) + rad(i) * r for i in range(n)]

    def band(r0, r1, col, hint):
        for i in range(n):
            j = (i + 1) % n
            out = hint(i, j)
            part.quad(r0[i], r0[j], r1[j], r1[i], col, out)

    def disc(r0, r1, col, out):
        for i in range(n):
            j = (i + 1) % n
            part.quad(r0[i], r0[j], r1[j], r1[i], col, out)

    # tread
    band(ring(-tread_hw, R), ring(tread_hw, R), TYRE,
         lambda i, j: rad(i) + rad(j))
    for s in (-1.0, 1.0):
        ax = X * s
        xo, xi = s * hw, s * (hw - depth)
        # shoulder: tread edge rolled out to the sidewall plane
        band(ring(s * tread_hw, R), ring(xo, R_SH), TYRE,
             lambda i, j: rad(i) + rad(j) + ax * 0.9)
        # sidewall: flat annulus down to the rim
        disc(ring(xo, R_SH), ring(xo, R_RIM), TYRE_WALL, ax)
        # recess wall: a short cylinder seen from outside, so it faces the axle
        band(ring(xo, R_RIM), ring(xi, R_RIM), RIM_DARK,
             lambda i, j: -(rad(i) + rad(j)))
        # rim face, hub boss, and five spokes standing proud of it
        disc(ring(xi, R_RIM), ring(xi, R_HUB), RIM, ax)
        band(ring(xi, R_HUB), ring(xi + s * 0.016, R_HUB), RIM_DARK,
             lambda i, j: rad(i) + rad(j))
        _cap_free(part, ring(xi + s * 0.016, R_HUB), HUB, ax)
        for sp in range(5):
            a0 = 2.0 * math.pi * sp / 5.0
            xs = xi + s * 0.008
            pts = [np.array([xs, 0.0, 0.0]) + rad(0) * 0 for _ in range(4)]
            for k, (dt, rr) in enumerate(((-0.115, R_HUB + 0.004),
                                          (0.115, R_HUB + 0.004),
                                          (0.070, R_RIM - 0.012),
                                          (-0.070, R_RIM - 0.012))):
                ang = a0 + dt
                pts[k] = np.array([xs, rr * math.cos(ang), rr * math.sin(ang)])
            part.quad(pts[0], pts[1], pts[2], pts[3], RIM_DARK, ax)
    return part


def _wheel(width: float) -> Mesh:
    return _wheel_part(width).mesh()


# --- the car -----------------------------------------------------------
def build(name: str) -> Entity:
    liv = LIVERIES[name]
    BODY, ACCENT, HELMET = liv["body"], liv["accent"], liv["helmet"]

    body = Entity(name="body")
    body.faces_forward = True
    parts: dict[str, _Part] = {}

    def P(key) -> _Part:
        if key not in parts:
            parts[key] = _Part()
        return parts[key]

    # -- monocoque ------------------------------------------------------
    _loft(P("tub"), [_rrect(z, hw, y0, y1) for z, hw, y0, y1 in TUB], BODY)

    # cockpit opening, headrest, driver
    cp = P("cockpit")
    _box(cp, (-0.215, 0.585, -0.34), (0.215, 0.735, 0.44), CARBON)
    _box(cp, (-0.185, 0.585, -0.52), (0.185, 0.790, -0.34), CARBON_LT)
    helm = P("helmet")
    _sphere(helm, (0.0, 0.700, -0.09), 0.142, HELMET)
    _box(helm, (-0.098, 0.658, 0.020), (0.098, 0.722, 0.062), VISOR)

    # -- airbox, engine cover, sharkfin --------------------------------
    cover = P("cover")
    _loft(cover, [
        _rrect(-0.30, 0.100, 0.700, 0.940),
        _rrect(-0.60, 0.140, 0.640, 0.968),
        _rrect(-1.00, 0.140, 0.580, 0.860),
        _rrect(-1.45, 0.100, 0.500, 0.680),
        _rrect(-1.95, 0.052, 0.360, 0.462),
    ], BODY, cap0=False, cap1=True)
    _cap(cover, _rrect(-0.30, 0.086, 0.718, 0.922), CARBON, Z)
    _extrude_x(P("fin"), [(-0.62, 0.952), (-1.05, 0.880), (-1.55, 0.735),
                          (-1.96, 0.470), (-1.94, 0.428), (-1.48, 0.628),
                          (-1.02, 0.812), (-0.62, 0.906)],
               -0.013, 0.013, ACCENT)
    _box(P("tcam"), (-0.036, 0.940, -0.30), (0.036, 1.008, -0.18), CARBON)

    # -- sidepods -------------------------------------------------------
    for s in (-1.0, 1.0):
        pod, xc = P("pods"), s * 0.470
        _loft(pod, [
            _rrect(0.32, 0.150, 0.120, 0.440, x=xc),
            _rrect(-0.10, 0.170, 0.080, 0.462, x=xc),
            _rrect(-0.70, 0.160, 0.080, 0.400, x=xc),
            _rrect(-1.30, 0.100, 0.120, 0.262, x=xc),
            _rrect(-1.62, 0.050, 0.160, 0.202, x=xc),
        ], BODY, cap0=False, cap1=True)
        _cap(pod, _rrect(0.32, 0.132, 0.140, 0.418, x=xc), CARBON, Z)

    # -- floor, bargeboards, diffuser ------------------------------------
    fl = P("floor")
    rect = lambda z, hw, y0, y1: np.asarray(  # noqa: E731
        [(hw, y1, z), (-hw, y1, z), (-hw, y0, z), (hw, y0, z)])
    _loft(fl, [rect(0.82, 0.40, 0.035, 0.070), rect(0.20, 0.76, 0.035, 0.070),
               rect(-1.62, 0.76, 0.035, 0.070), rect(-2.08, 0.62, 0.150, 0.300)],
          CARBON)
    for s in (-1.0, 1.0):
        _extrude_x(P("edge"), [(0.20, 0.036), (-1.62, 0.036), (-1.62, 0.068),
                               (0.20, 0.068)], s * 0.745, s * 0.765, ACCENT)
    for s in (-1.0, 1.0):
        _extrude_x(P("boards"), [(0.80, 0.072), (0.80, 0.330), (0.58, 0.362),
                                 (0.34, 0.222), (0.34, 0.072)],
                   s * 0.360, s * 0.380, CARBON)

    # -- front wing ------------------------------------------------------
    _extrude_x(P("fwing"), FW_MAIN, -FW_SPAN, FW_SPAN, BODY)
    _extrude_x(P("fflap"), FW_FLAP, -FW_SPAN, -0.24, ACCENT)
    _extrude_x(P("fflap"), FW_FLAP, 0.24, FW_SPAN, ACCENT)
    for s in (-1.0, 1.0):
        _extrude_x(P("fplates"), FW_PLATE, s * FW_SPAN, s * (FW_SPAN + 0.024),
                   CARBON)
    # Nose pylons: from the wing's upper surface into the nose underside, so
    # the two are visibly one piece rather than a line stopping in mid-air.
    z_py = NOSE - 0.09
    y_wing = _profile_y(FW_MAIN, z_py, upper=True)
    for s in (-1.0, 1.0):
        # Both ends land *inside* solid geometry -- buried in the wing below
        # and in the nose above -- so the join reads as one piece.
        _bar(P("fpylon"), (s * 0.034, y_wing - 0.010, z_py),
             (s * 0.034, _tub(2.045)[1] + 0.030, 2.045), 0.022, 0.017, CARBON)

    # -- rear wing -------------------------------------------------------
    _extrude_x(P("rwing"), RW_MAIN, -RW_SPAN, RW_SPAN, BODY)
    _extrude_x(P("rflap"), RW_FLAP, -RW_SPAN, RW_SPAN, ACCENT)
    _extrude_x(P("rbeam"), RW_BEAM, -RW_SPAN + 0.04, RW_SPAN - 0.04, CARBON)
    for s in (-1.0, 1.0):
        _extrude_x(P("rplates"), RW_PLATE, s * RW_SPAN, s * (RW_SPAN + 0.024),
                   CARBON)
    # Swan-neck pylons. The top end stops *inside* the mainplane, never
    # through it: the old ones finished above its upper surface and showed as
    # spikes over the DRS flap.
    z_sw = -1.96
    y_in = 0.5 * (_profile_y(RW_MAIN, z_sw, True) + _profile_y(RW_MAIN, z_sw, False))
    _box(P("crash"), (-0.095, 0.250, -2.20), (0.095, 0.385, -1.98), CARBON)
    for s in (-1.0, 1.0):
        _bar(P("rpylon"), (s * 0.062, 0.352, -2.09), (s * 0.062, y_in, z_sw),
             0.018, 0.026, CARBON)
    rain_off, rain_on = _Part(), _Part()
    for pt, col in ((rain_off, RAIN_OFF), (rain_on, RAIN_ON)):
        _box(pt, (-0.044, 0.286, -2.216), (0.044, 0.336, -2.196), col)

    # -- halo ------------------------------------------------------------
    halo = P("halo")
    hoop = [(0.300 * math.sin(t), 0.812 + 0.020 * math.cos(t) ** 2,
             0.020 + 0.395 * math.cos(t))
            for t in (2 * math.pi * i / 22 for i in range(22))]
    _tube(halo, hoop, 0.026, TITANIUM, closed=True)
    _tube(halo, [(0.0, 0.600, 0.500), (0.0, 0.720, 0.470), (0.0, 0.812, 0.415)],
          0.026, TITANIUM)
    for s in (-1.0, 1.0):
        _tube(halo, [(s * 0.286, 0.830, -0.250), (s * 0.250, 0.700, -0.330),
                     (s * 0.215, 0.610, -0.360)], 0.024, TITANIUM)

    # -- mirrors, stalks anchored on the tub shoulder ---------------------
    for s in (-1.0, 1.0):
        hw, _y0, y1 = _tub(0.34)
        _bar(P("mstalk"), (s * (hw - 0.012), y1 - 0.075, 0.340),
             (s * 0.455, 0.628, 0.318), 0.014, 0.020, CARBON)
        _box(P("mirrors"), (s * 0.430, 0.596, 0.276), (s * 0.545, 0.668, 0.344),
             CARBON)
        _box(P("mglass"), (s * 0.436, 0.606, 0.270), (s * 0.539, 0.658, 0.278),
             GLASS)

    # -- suspension: wishbones and push-rods, anchored to the tub ----------
    sus = P("suspension")
    for zax, w_wheel in ((ZF, W_FRONT), (ZR, W_REAR)):
        face = TRACK - w_wheel / 2.0            # the wheel's inner face
        x_up = face - 0.010                     # upright, just inboard of it
        x_arm = face - 0.055
        for s in (-1.0, 1.0):
            for dz in (0.26, -0.26):
                zi = zax + dz
                hw, y0, y1 = _tub(zi)
                _bar(sus, (s * (hw - 0.012), y1 - 0.085, zi),
                     (s * x_arm, 0.452, zax), 0.015, 0.026, CARBON)
                _bar(sus, (s * (hw - 0.012), y0 + 0.048, zi),
                     (s * x_arm, 0.198, zax), 0.015, 0.026, CARBON)
            hw, _y0, y1 = _tub(zax + 0.12)
            _bar(sus, (s * x_arm, 0.210, zax + 0.02),
                 (s * (hw - 0.012), y1 - 0.030, zax + 0.12), 0.013, 0.022, CARBON)
            _box(sus, (s * (x_up - 0.085), 0.175, zax - 0.075),
                 (s * x_up, 0.492, zax + 0.075), CARBON)

    # -- assemble ---------------------------------------------------------
    for key, part in parts.items():
        Entity(parent=body, name=key, model=part.mesh())
    body.rain_off = Entity(parent=body, name="rainOff", model=rain_off.mesh())
    body.rain_on = Entity(parent=body, name="rainOn", model=rain_on.mesh(),
                          enabled=False)
    # One mesh per wheel: a Mesh is a NodePath and can hang in only one place.
    for nm, x, z, w in (("wheelFrontLeft", -TRACK, ZF, W_FRONT),
                        ("wheelFrontRight", TRACK, ZF, W_FRONT),
                        ("wheelBackLeft", -TRACK, ZR, W_REAR),
                        ("wheelBackRight", TRACK, ZR, W_REAR)):
        Entity(parent=body, name=nm, model=_wheel(w), position=(x, R_WHEEL, z))
    return body
