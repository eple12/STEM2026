"""Load an F1TENTH racetrack and derive the geometry the game needs.

CSV columns:  ``x_m, y_m, w_tr_right_m, w_tr_left_m``  (comma separated, one
comment header line starting with '#'). The centreline is a closed loop.

World mapping:  f1tenth (x, y)  ->  world (x, z),  world y is up.
"""
from __future__ import annotations

import math

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


def _nearest_dist(pts: np.ndarray, ref: np.ndarray, block: int = 256) -> np.ndarray:
    """For each point, the distance to the nearest point of *ref*.

    Blocked because the full pairwise matrix is n^2 and these circuits run to a
    few thousand samples.
    """
    out = np.empty(len(pts))
    for a in range(0, len(pts), block):
        b = min(a + block, len(pts))
        d = np.linalg.norm(pts[a:b, None, :] - ref[None, :, :], axis=2)
        out[a:b] = d.min(axis=1)
    return out


def _medial_distance(c: np.ndarray, u: np.ndarray, block: int = 256) -> np.ndarray:
    """How far each point can travel along *u* before something else is nearer.

    Walking out in discrete steps and testing works but is only as smooth as
    the step, which showed up as a 1.5 m staircase in the wall. There is a
    closed form: the ray ``p = c_i + u_i t`` is equidistant from ``c_i`` and
    ``c_j`` at

        t = |c_j - c_i|^2 / (2 u_i . (c_j - c_i))

    for every ``c_j`` the ray heads towards, and the smallest such t is where
    the ray meets the medial axis. One pass, exact, and it subsumes both
    failure modes at once: a neighbour just inside a corner yields t = R, the
    radius of curvature, and a point on the far side of a narrow section yields
    half the gap.
    """
    n = len(c)
    out = np.full(n, np.inf)
    src = np.zeros(n, dtype=int)
    for a in range(0, n, block):
        b = min(a + block, n)
        diff = c[None, :, :] - c[a:b, None, :]              # (blk, n, 2)
        dot = np.einsum("bnk,bk->bn", diff, u[a:b])
        sq = np.einsum("bnk,bnk->bn", diff, diff)
        # Only points the ray actually approaches can constrain it; a tangent
        # neighbour has dot ~ 0 and would otherwise blow up.
        t = np.where(dot > 1e-6, sq / (2.0 * np.maximum(dot, 1e-6)), np.inf)
        out[a:b] = t.min(axis=1)
        src[a:b] = t.argmin(axis=1)
    return out, src


def _slope_limit(a: np.ndarray, step: np.ndarray, passes: int = 3) -> np.ndarray:
    """Lower values until the profile never changes faster than *step* per sample.

    An erode-then-average smoothing spreads a drop over its window, so the
    kink per sample is just the drop divided by the window -- an 11 m drop over
    9 samples is still a 1.3 m step, and it pulls the wall in over the whole
    window whether or not that is needed. Limiting the slope instead only ever
    lowers a value, only near the constraint, and gives an exact bound on how
    sharply the wall can turn: the barrier tapers in like a real one.
    """
    out = a.copy()
    n = len(out)
    for _ in range(passes):
        for i in range(n):
            j = i - 1 if i else n - 1
            cap = out[j] + step[j]
            if out[i] > cap:
                out[i] = cap
        for i in range(n - 1, -1, -1):
            j = i + 1 if i < n - 1 else 0
            cap = out[j] + step[j]
            if out[i] > cap:
                out[i] = cap
    return out


def _spread(mask: np.ndarray, k: int) -> np.ndarray:
    """Circular dilation of a boolean ring by *k* samples either side."""
    out = mask.copy()
    for d in range(1, k + 1):
        out |= np.roll(mask, d) | np.roll(mask, -d)
    return out


def _smooth_ring(a: np.ndarray, k: int) -> np.ndarray:
    """Light circular moving average, to take the facets off a limited slope."""
    if k < 3 or len(a) < k:
        return a
    half = k // 2
    pad = np.concatenate([a[-half:], a, a[:half]])
    return np.convolve(pad, np.ones(2 * half + 1) / (2 * half + 1), mode="valid")


def _ray_first_hit(origins: np.ndarray, dirs: np.ndarray,
                   segs: np.ndarray, block: int = 128) -> np.ndarray:
    """Distance along each ray to the first segment it crosses, or inf.

    The barrier is a set of chords cut from a smoothed contour, and every
    other piece of roadside geometry used to be positioned by walking out
    along the centreline normal by ``wall_offsets()`` -- a *different*
    construction of the same line. The two agree on a straight and part
    company wherever the contour bends hard: the smoothing cuts a convex
    corner and the contour bulges past a concave one, so the run-off apron
    ended up either lying across the barrier or stopping short of it with a
    wedge of the grass plane showing through. Asking the barrier itself where
    it is removes the second construction, and with it the disagreement.
    """
    n = len(origins)
    out = np.full(n, np.inf)
    if len(segs) == 0:
        return out
    a = segs[:, 0]
    e = segs[:, 1] - segs[:, 0]
    for k0 in range(0, n, block):
        k1 = min(k0 + block, n)
        o = origins[k0:k1][:, None, :]
        d = dirs[k0:k1][:, None, :]
        den = d[..., 0] * e[None, :, 1] - d[..., 1] * e[None, :, 0]
        ok = np.abs(den) > 1e-12
        safe = np.where(ok, den, 1.0)
        ao = a[None, :, :] - o
        t = (ao[..., 0] * e[None, :, 1] - ao[..., 1] * e[None, :, 0]) / safe
        u = (ao[..., 0] * d[..., 1] - ao[..., 1] * d[..., 0]) / safe
        good = ok & (u >= 0.0) & (u <= 1.0) & (t > 0.0)
        out[k0:k1] = np.where(good, t, np.inf).min(axis=1)
    return out


def _seg_nearest(pts: np.ndarray, segs: np.ndarray,
                 block: int = 128) -> np.ndarray:
    """The closest point on *segs* to each of *pts*."""
    out = np.array(pts, dtype=float, copy=True)
    if len(pts) == 0 or len(segs) == 0:
        return out
    a = segs[:, 0]
    e = segs[:, 1] - segs[:, 0]
    L2 = np.maximum((e * e).sum(axis=1), 1e-9)
    for k0 in range(0, len(pts), block):
        k1 = min(k0 + block, len(pts))
        p = pts[k0:k1][:, None, :]
        t = np.clip(((p - a[None, :, :]) * e[None, :, :]).sum(axis=2) / L2[None, :],
                    0.0, 1.0)
        q = a[None, :, :] + e[None, :, :] * t[..., None]
        k = ((p - q) ** 2).sum(axis=2).argmin(axis=1)
        out[k0:k1] = q[np.arange(k1 - k0), k]
    return out


def _seg_distance(pts: np.ndarray, segs: np.ndarray,
                  block: int = 128) -> np.ndarray:
    """Distance from each point to the nearest of *segs*, or inf if there are
    none."""
    n = len(pts)
    out = np.full(n, np.inf)
    if n == 0 or len(segs) == 0:
        return out
    a = segs[:, 0]
    e = segs[:, 1] - segs[:, 0]
    L2 = np.maximum((e * e).sum(axis=1), 1e-9)
    for k0 in range(0, n, block):
        k1 = min(k0 + block, n)
        p = pts[k0:k1][:, None, :]
        ap = p - a[None, :, :]
        t = np.clip((ap * e[None, :, :]).sum(axis=2) / L2[None, :], 0.0, 1.0)
        q = a[None, :, :] + e[None, :, :] * t[..., None]
        out[k0:k1] = np.sqrt(((p - q) ** 2).sum(axis=2)).min(axis=1)
    return out


def _corridor_field(track, cell: float, pad: float):
    """Signed field over a grid: negative inside the drivable corridor.

    The corridor is "within the run-off distance of the centreline", which is
    a union of half-discs -- one radius to the right of each sample, another
    to the left, because a corner is given its room towards the exit. Sampling
    that union onto a grid is what lets the *outline* of the whole thing be
    taken, rather than one offset point per sample. Two legs of a chicane that
    overlap simply merge into one blob, and the outline goes round the pair.

    Returns (field, x0, z0), with field[r, c] at (x0 + c*cell, z0 + r*cell).
    """
    off_r, off_l = track.wall_offsets()
    reach = np.maximum(off_r, off_l)
    lo = track.center.min(axis=0) - (reach.max() + pad)
    hi = track.center.max(axis=0) + (reach.max() + pad)
    nx = int(np.ceil((hi[0] - lo[0]) / cell)) + 1
    nz = int(np.ceil((hi[1] - lo[1]) / cell)) + 1
    f = np.full((nz, nx), 1e9)

    for i in range(track.count):
        cx, cz = track.center[i]
        r = float(reach[i])
        c0 = max(int((cx - r - lo[0]) / cell), 0)
        c1 = min(int((cx + r - lo[0]) / cell) + 2, nx)
        r0 = max(int((cz - r - lo[1]) / cell), 0)
        r1 = min(int((cz + r - lo[1]) / cell) + 2, nz)
        if c1 <= c0 or r1 <= r0:
            continue
        dx = lo[0] + np.arange(c0, c1) * cell - cx
        dz = lo[1] + np.arange(r0, r1) * cell - cz
        d = np.sqrt(dz[:, None] ** 2 + dx[None, :] ** 2)
        # Which side of this sample a cell lies on picks its radius.
        side = (dz[:, None] * track.normal[i, 1]
                + dx[None, :] * track.normal[i, 0])
        rad = np.where(side > 0.0, off_r[i], off_l[i])
        np.minimum(f[r0:r1, c0:c1], d - rad, out=f[r0:r1, c0:c1])
    return f, float(lo[0]), float(lo[1])


def runoff_fill(track, cell: float, bias: float, skirt: float = 0.0):
    """Triangulate the drivable region: (vertices (V,2), triangles (T,3),
    outer mask (V,)).

    The run-off used to be a quad strip walked out along each sample's normal,
    and a strip cannot cover this region. Where the circuit turns, the region's
    outline turns with it and opens a wedge that no normal ever points into --
    a triangle of ground inside the fence that the apron simply did not reach,
    with the grass plane showing through it. That is the green wedge on the
    inside of Monza's first chicane, and every circuit here has several. No
    amount of clamping fixes it, because the missing ground is not on any ray.

    So the region is filled as a region. The same field the barrier contour is
    taken from is sampled onto a coarser grid; cells wholly inside become a
    quad, cells the boundary crosses are clipped against it. What comes out
    is the interior of the fence, exactly, with its edge interpolated onto the
    contour rather than stepped along the grid -- so "inside the barrier" and
    "painted" become the same statement, which is the whole point.

    *bias* is the level set to fill to. The barrier is nudged BARRIER_OUTSET
    outside the contour, so filling to the same offset tucks the apron's edge
    under the rail instead of stopping a hand's breadth short of it.
    """
    f, x0, z0 = track.corridor_field()
    fine = config.BARRIER_GRID
    nz, nx = f.shape
    # Node grid, sampled off the fine field rather than rebuilt at the coarse
    # spacing: a second sampling of the region is a second region, and the
    # barrier is drawn on the first one.
    gx = np.arange(0.0, (nx - 1) * fine + cell, cell)
    gz = np.arange(0.0, (nz - 1) * fine + cell, cell)
    ax = np.clip(gx / fine, 0.0, nx - 1.001)
    az = np.clip(gz / fine, 0.0, nz - 1.001)
    ix, iz = ax.astype(int), az.astype(int)
    tx, tz = (ax - ix)[None, :], (az - iz)[:, None]
    F = (f[np.ix_(iz, ix)] * (1 - tx) * (1 - tz)
         + f[np.ix_(iz, ix + 1)] * tx * (1 - tz)
         + f[np.ix_(iz + 1, ix)] * (1 - tx) * tz
         + f[np.ix_(iz + 1, ix + 1)] * tx * tz)
    X = x0 + gx
    Z = z0 + gz
    inside = F < bias
    mz, mx = F.shape

    c00 = inside[:-1, :-1]
    c10 = inside[:-1, 1:]
    c11 = inside[1:, 1:]
    c01 = inside[1:, :-1]
    cnt = (c00.astype(np.int8) + c10 + c11 + c01)
    full = cnt == 4
    part = (cnt > 0) & ~full

    # Compact vertex numbering: only nodes an emitted cell actually uses.
    used = np.zeros_like(inside)
    for dr, dc in ((0, 0), (0, 1), (1, 1), (1, 0)):
        sel = np.zeros_like(inside)
        sel[dr:mz - 1 + dr, dc:mx - 1 + dc] = full | part
        used |= sel & inside
    node = np.full(inside.shape, -1, dtype=np.int64)
    node[used] = np.arange(int(used.sum()))
    zz, xx = np.nonzero(used)
    verts = [(float(X[c]), float(Z[r])) for r, c in zip(zz.tolist(), xx.tolist())]

    tris = []
    fr, fc = np.nonzero(full)
    a = node[fr, fc]
    b = node[fr, fc + 1]
    c = node[fr + 1, fc + 1]
    d = node[fr + 1, fc]
    tris.append(np.stack([a, b, c], axis=1))
    tris.append(np.stack([a, c, d], axis=1))

    # Partial cells: walk the four corners and clip the square against the
    # contour, inserting the crossing on every edge that changes sign. Shared
    # crossings are keyed by the edge they sit on, so neighbouring cells reuse
    # the same vertex and the fill has no cracks in it.
    cross: dict = {}
    edges: list = []          # the contour, as pairs of crossing vertices

    def crossing(r0, c0, r1, c1):
        key = (r0, c0, r1, c1) if (r0, c0) < (r1, c1) else (r1, c1, r0, c0)
        got = cross.get(key)
        if got is not None:
            return got
        va, vb = float(F[r0, c0]) - bias, float(F[r1, c1]) - bias
        t = 0.5 if abs(vb - va) < 1e-12 else min(max(va / (va - vb), 0.0), 1.0)
        px = float(X[c0]) + (float(X[c1]) - float(X[c0])) * t
        pz = float(Z[r0]) + (float(Z[r1]) - float(Z[r0])) * t
        idx = len(verts)
        verts.append((px, pz))
        cross[key] = idx
        return idx

    for r, c in zip(*np.nonzero(part)):
        r, c = int(r), int(c)
        corners = ((r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c))
        poly = []
        for k in range(4):
            p, q = corners[k], corners[(k + 1) % 4]
            pin, qin = bool(inside[p]), bool(inside[q])
            if pin:
                poly.append(int(node[p]))
            if pin != qin:
                poly.append(crossing(p[0], p[1], q[0], q[1]))
        for k in range(1, len(poly) - 1):
            tris.append(np.array([[poly[0], poly[k], poly[k + 1]]], dtype=np.int64))
        # The two crossings of a partial cell are a piece of the contour. Kept
        # so the fill can be given a skirt: without one the ground stops dead
        # at the fence and drops to the plane behind it, which is a step you
        # can see under every barrier on the lap.
        hits = [q for q in poly if q >= len(zz)]
        if len(hits) == 2:
            gx_ = 0.5 * (F[r, c + 1] + F[r + 1, c + 1]
                         - F[r, c] - F[r + 1, c])
            gz_ = 0.5 * (F[r + 1, c] + F[r + 1, c + 1]
                         - F[r, c] - F[r, c + 1])
            n_ = math.hypot(gx_, gz_) or 1.0
            edges.append((hits[0], hits[1], gx_ / n_, gz_ / n_))

    V = np.asarray(verts, dtype=float)
    # Snap the boundary onto the fence. Everything above works off the level
    # set of the field; the barrier is that level set smoothed again, nudged
    # out and cut into chords, so the two run a metre apart at the sharpest
    # bends -- a metre of grass plane inside the rail here, a metre of gravel
    # outside it there. These are exactly the vertices that lie on the
    # boundary, so moving them onto the chords makes "the edge of the apron"
    # and "the barrier" the same polyline rather than two close ones. Capped
    # at a fraction of the cell so a snap can never turn a cell inside out.
    if cross:
        idx = np.fromiter(cross.values(), dtype=np.int64, count=len(cross))
        segs = [x for x in track.barrier_lines() if len(x)]
        if segs:
            q = _seg_nearest(V[idx], np.concatenate(segs, axis=0))
            d = q - V[idx]
            n = np.hypot(d[:, 0], d[:, 1])
            lim = np.minimum(1.0, cell * 0.45 / np.maximum(n, 1e-9))
            V[idx] = V[idx] + d * lim[:, None]
    # The skirt. Each contour edge is extruded outward along the field's own
    # gradient -- which is the direction "away from the circuit" by definition
    # -- and the ring of vertices that lands out there is reported so the
    # caller can drop it to the height of the plane beyond. Ten metres of it
    # turns a step under the barrier into a slope of four per cent.
    outer = np.zeros(len(V), dtype=bool)
    if edges and skirt > 0.0:
        base = len(V)
        add, quads, seen = [], [], {}

        def lift(idx, gx_, gz_):
            k = seen.get(idx)
            if k is None:
                k = base + len(add)
                add.append(V[idx] + np.array([gx_, gz_]) * skirt)
                seen[idx] = k
            return k

        for a_, b_, gx_, gz_ in edges:
            a2, b2 = lift(a_, gx_, gz_), lift(b_, gx_, gz_)
            quads.append([a_, a2, b2])
            quads.append([a_, b2, b_])
        V = np.concatenate([V, np.asarray(add, dtype=float)], axis=0)
        outer = np.zeros(len(V), dtype=bool)
        outer[base:] = True
        tris.append(np.asarray(quads, dtype=np.int64))
    T = np.concatenate(tris, axis=0) if tris else np.zeros((0, 3), np.int64)
    return V, T, outer


# Marching squares: for each corner pattern, which pairs of cell edges the
# zero crossing runs between. Corners are numbered from the grid-minimum one
# and edge k joins corner k to corner k+1.
_MS_CASES = {
    1: ((3, 0),), 2: ((0, 1),), 3: ((3, 1),), 4: ((1, 2),),
    6: ((0, 2),), 7: ((3, 2),), 8: ((2, 3),), 9: ((2, 0),),
    11: ((2, 1),), 12: ((1, 3),), 13: ((1, 0),), 14: ((0, 3),),
}


def _iso_contours(f: np.ndarray, x0: float, z0: float, cell: float) -> list:
    """Closed polylines along ``f == 0``, in world coordinates.

    Written out rather than pulled from a library so the game keeps its three
    dependencies. Crossing points are interpolated from a canonical corner
    order, so two cells either side of an edge produce bit-identical points
    and the segments link into loops by exact lookup.
    """
    v0, v1 = f[:-1, :-1], f[:-1, 1:]
    v2, v3 = f[1:, 1:], f[1:, :-1]
    inside = ((v0 < 0).astype(np.uint8) | ((v1 < 0) << 1)
              | ((v2 < 0) << 2) | ((v3 < 0) << 3))
    rs, cs = np.nonzero((inside != 0) & (inside != 15))

    def point(kind, r, c):
        if kind == "h":                      # between (r, c) and (r, c+1)
            a, b = f[r, c], f[r, c + 1]
            return (x0 + (c + a / (a - b)) * cell, z0 + r * cell)
        a, b = f[r, c], f[r + 1, c]          # between (r, c) and (r+1, c)
        return (x0 + c * cell, z0 + (r + a / (a - b)) * cell)

    links: dict = {}
    for r, c in zip(rs.tolist(), cs.tolist()):
        case = int(inside[r, c])
        if case in (5, 10):                  # saddle: the cell centre decides
            mid = 0.25 * (f[r, c] + f[r, c + 1]
                          + f[r + 1, c + 1] + f[r + 1, c])
            joined = (mid < 0) == (case == 5)
            pairs = ((3, 0), (1, 2)) if joined else ((0, 1), (2, 3))
        else:
            pairs = _MS_CASES[case]
        edge = {0: ("h", r, c), 1: ("v", r, c + 1),
                2: ("h", r + 1, c), 3: ("v", r, c)}
        for e_a, e_b in pairs:
            pa, pb = point(*edge[e_a]), point(*edge[e_b])
            links.setdefault(pa, []).append(pb)
            links.setdefault(pb, []).append(pa)

    loops, seen = [], set()
    for start in links:
        if start in seen:
            continue
        loop, prev, node = [], None, start
        while node is not None and node not in seen:
            seen.add(node)
            loop.append(node)
            nxt = None
            for cand in links[node]:
                if cand != prev and cand not in seen:
                    nxt = cand
                    break
            prev, node = node, nxt
        if len(loop) >= 8:
            loops.append(np.asarray(loop, dtype=float))
    return loops


def _smooth_loop(p: np.ndarray, k: int) -> np.ndarray:
    """Circular moving average over a closed polyline.

    The contour steps from grid edge to grid edge, so without this the wall
    carries a one-metre sawtooth all the way round.
    """
    if k < 3 or len(p) < 2 * k:
        return p
    half = k // 2
    pad = np.concatenate([p[-half:], p, p[:half]])
    w = np.ones(2 * half + 1) / (2 * half + 1)
    return np.stack([np.convolve(pad[:, 0], w, mode="valid"),
                     np.convolve(pad[:, 1], w, mode="valid")], axis=1)


def _resample_loop(p: np.ndarray, step: float) -> np.ndarray:
    """Even spacing along a closed polyline, so the chord walk sees a ruler."""
    d = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    if total < step * 4:
        return p
    t = np.arange(0.0, total, step)
    ring = np.vstack([p, p[:1]])
    return np.stack([np.interp(t, s, ring[:, 0]),
                     np.interp(t, s, ring[:, 1])], axis=1)


def _chords(pts: np.ndarray, max_run: float, max_bend: float) -> list:
    """Cut a closed polyline into straight runs, one per stretched module."""
    n = len(pts)
    # Walked over an explicit ring whose last point *is* the first, so the
    # final chord ends on the start by construction. Wrapping with a modulo
    # and then stitching the leftover metres afterwards closed the wall, but
    # the walk stops on the first index at or *past* the start, so the stitch
    # ran backwards -- a chord doubling back on the one before it, which is a
    # spike standing in the middle of an otherwise straight fence.
    ring = np.vstack([pts, pts[:1]])
    segs, k = [], 0
    while k < n:
        a = ring[k]
        j = k + 1
        while j < n:
            b = ring[j]
            d = b - a
            if np.hypot(d[0], d[1]) > max_run:
                # Back off to the last point that was still in range, so a run
                # never overshoots its length limit. Overshooting matters more
                # than the odd metre it costs: the chord is a straight line
                # across a curve, and its sag from the real outline grows with
                # the square of how long it is allowed to get.
                if j > k + 1:
                    j -= 1
                break
            if _turn_degrees(d, ring[j + 1] - b) > max_bend:
                j += 1
                break
            j += 1
        b = ring[j]
        k = j
        if np.hypot(*(b - a)) >= 0.05:
            segs.append((a, b))
    return segs


def _turn_degrees(a, b) -> float:
    """Angle in degrees between two 2-D directions, 0 if either is degenerate."""
    na, nb = float(np.hypot(a[0], a[1])), float(np.hypot(b[0], b[1]))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cos = float(np.clip((a[0] * b[0] + a[1] * b[1]) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


@dataclass
class Track:
    name: str
    center: np.ndarray        # (N, 2) world x,z  (scaled)
    tangent: np.ndarray       # (N, 2) unit forward
    normal: np.ndarray        # (N, 2) unit right-hand normal
    w_right: np.ndarray       # (N,) half width to the right edge (scaled, +gain)
    w_left: np.ndarray        # (N,)
    curv_radius: np.ndarray   # (N,) local radius of curvature, metres
    curvature: np.ndarray     # (N,) signed 1/R; positive = turning right
    seg_len: np.ndarray       # (N,) distance to next point
    arclen: np.ndarray        # (N,) cumulative distance from start
    length: float             # total lap length

    _wall_off: tuple | None = None   # cached wall_offsets() result
    _medial: tuple | None = None     # cached medial_offsets()
    _wall_ray: tuple | None = None   # cached wall_ray_offsets()
    _field: tuple | None = None      # cached corridor_field()
    _reach: tuple | None = None      # cached corridor_reach()
    _bank: np.ndarray | None = None  # cached bank()

    # -- helpers ----------------------------------------------------------
    @property
    def count(self) -> int:
        return len(self.center)

    def start_pose(self) -> tuple[np.ndarray, float]:
        """Return (position xz, yaw radians) on the grid, just behind the line."""
        pos = self.center[0] - self.tangent[0] * 6.0
        yaw = np.arctan2(self.tangent[0, 0], self.tangent[0, 1])
        return pos, float(yaw)

    def nearest_index(self, pos_xz, hint: int, window: int = 10) -> int:
        """Nearest centreline index, searched in a wrapping window around *hint*.

        The window is small on purpose: a physics step advances well under one
        sample spacing, so scanning a hundred candidates twice per step was
        costing more than the whole vehicle model. If the best match lands on
        the edge of the window the hint was stale (a reset, a teleport), so
        fall back to a full scan.
        """
        px, pz = float(pos_xz[0]), float(pos_xz[1])
        # Plain Python over a list: a handful of points is far below the size
        # where numpy's per-call overhead pays for itself, and this runs a
        # dozen times per car per physics step -- twenty cars' worth in a
        # grand prix.
        cl = getattr(self, "_center_list", None)
        if cl is None:
            cl = self._center_list = self.center.tolist()
        n = len(cl)
        # Walk downhill from the hint rather than scanning the whole window:
        # from the last answer the nearest sample is almost always the same
        # one or its neighbour, so this is two or three distances instead of
        # twenty-one. Walking the full window means the hint was stale (a
        # reset, a teleport): fall back to the full scan.
        k = hint % n
        cx, cz = cl[k]
        best = (cx - px) * (cx - px) + (cz - pz) * (cz - pz)
        for step in (1, -1):
            moved = 0
            while moved < window:
                j = (k + step) % n
                cx, cz = cl[j]
                d = (cx - px) * (cx - px) + (cz - pz) * (cz - pz)
                if d >= best:
                    break
                k, best = j, d
                moved += 1
            if moved >= window:
                break
            if moved:
                break
        else:
            moved = 0
        if moved < window:
            # A walk downhill finds *a* nearest point, and on the road it is
            # the nearest. Off it -- across the infield of a chicane, out
            # over a run-off -- the leg the car left can hold a local
            # minimum the walk stops in while the car is already beside the
            # next leg: the lap stopped advancing, "wrong way" came up
            # driving forwards, the walls went to the wrong segment and the
            # track limits measured nothing. Off the road, look properly,
            # over the stretch of lap a car can actually have crossed.
            wm = getattr(self, "_wmax_list", None)
            if wm is None:
                wm = self._wmax_list = (np.maximum(self.w_left, self.w_right)
                                        + 1.0).tolist()
            if best <= wm[k] * wm[k]:
                return k
            return self._nearest_near(px, pz, k)

        dx = self.center[:, 0] - px
        dz = self.center[:, 1] - pz
        return int(np.argmin(dx * dx + dz * dz))

    #: How far along the lap either side of the last answer an off-road
    #: search looks: more than any cut, far less than half a lap, so a
    #: parallel straight a kilometre on is never taken for the nearest.
    OFFROAD_SEARCH_M = 300.0

    def _nearest_near(self, px: float, pz: float, k: int) -> int:
        n = len(self.center)
        span = getattr(self, "_offroad_span", None)
        if span is None:
            step = max(float(np.median(self.seg_len)), 0.5)
            span = self._offroad_span = min(n // 2 - 1,
                                            int(self.OFFROAD_SEARCH_M / step))
        idx = np.arange(k - span, k + span + 1) % n
        c = self.center[idx]
        d = (c[:, 0] - px) ** 2 + (c[:, 1] - pz) ** 2
        return int(idx[int(np.argmin(d))])

    def barrier_lines(self) -> tuple[np.ndarray, np.ndarray]:
        """(right, left) barrier polylines, as arrays of (start, end) points.

        These are the exact chords the barrier props are placed along and the
        exact segments the car collides with, so the wall you see and the wall
        you hit cannot drift apart.

        The line is not an offset of the centreline. Offsetting sample by
        sample describes a barrier only where the circuit is a single ribbon;
        at a chicane the two legs' offsets drive through each other, and no
        amount of clamping or deleting repairs that, because the thing being
        asked for does not exist -- there is no wall between the legs of a
        chicane. What exists is the *drivable area*: every point within its
        run-off distance of the centreline. That is one region, it merges
        wherever two legs come close, and its **outline** is the barrier. A
        straight and an ordinary corner give back the offset line they always
        gave; Monza's first chicane gives one boundary wrapped round the whole
        complex with open asphalt inside it, which is what the circuit looks
        like.

        So: sample the region onto a grid, take its zero contour, smooth it
        and cut it into chords. The outline cannot cross the road, because the
        road is inside the region by construction.
        """
        if getattr(self, "_barrier_lines", None) is not None:
            return self._barrier_lines

        cell = config.BARRIER_GRID
        f, x0, z0 = self.corridor_field()
        loops = [_smooth_loop(lp, config.BARRIER_SMOOTH)
                 for lp in _iso_contours(f, x0, z0, cell)]

        # A lap is a closed ribbon, so its corridor is an annulus and its
        # boundary is exactly two closed curves: the outside of the circuit and
        # the infield. Anything else is a pocket -- ground pinched off when two
        # legs merged, which is the inside of a chicane complex. A pocket is
        # not a boundary of the drivable area seen from the road; walling it
        # gives a second, unreachable infield curve and a wall that appears to
        # have a hole in it. Keeping the two largest curves and dropping the
        # rest makes the pockets part of the complex they belong to, which is
        # what the merge was for, and it holds however close two legs come:
        # there is no threshold to tune and the count is always two.
        loops.sort(key=lambda lp: abs(0.5 * (np.dot(lp[:, 0], np.roll(lp[:, 1], -1))
                                             - np.dot(lp[:, 1], np.roll(lp[:, 0], -1)))),
                   reverse=True)
        right, left = [], []
        for loop in loops[:2]:
            loop = _resample_loop(loop, config.BARRIER_STEP)
            if len(loop) < 8:
                continue
            # A second, shorter pass, now that the loop is evenly spaced. The
            # first one runs on raw marching-squares output and its window is
            # measured in grid cells, so it takes the sawtooth off but leaves
            # the cusps -- the places where the corridor radius changes fastest
            # and the outline turns through most of a right angle inside four
            # metres. Those are precisely the corners everything on the
            # roadside used to come apart at. Cutting them costs a metre of
            # run-off at a hairpin and nothing anywhere else, and the apron,
            # the bank and the collision test all read this same line, so what
            # it costs, it costs consistently.
            loop = _smooth_loop(loop, config.BARRIER_RESAMPLE_SMOOTH)
            # Nudge outward, away from the track: the chords are straight
            # lines across a curved outline and would otherwise dip inside it.
            # Which way is out is decided per loop, because the infield of the
            # lap is a hole in the region and its outline faces the other way.
            i = np.array([int(np.argmin(((self.center - q) ** 2).sum(axis=1)))
                          for q in loop])
            sign = ((loop - self.center[i]) * self.normal[i]).sum(axis=1)
            side = 1 if np.sign(sign).sum() >= 0 else -1
            loop = loop + self.normal[i] * (side * config.BARRIER_OUTSET)
            segs = _chords(loop, config.BARRIER_MAX_RUN,
                           config.BARRIER_MAX_BEND)
            (right if side > 0 else left).extend(segs)

        self._barrier_lines = (
            np.asarray(right, dtype=float).reshape(-1, 2, 2),
            np.asarray(left, dtype=float).reshape(-1, 2, 2))
        return self._barrier_lines

    def nearest_indices(self, pts) -> np.ndarray:
        """Nearest centreline sample to each of many points.

        Coarse pass then a local refine. The whole cross product of a ground
        mesh against the centreline is twenty thousand by fourteen hundred,
        and building it in blocks moves half a gigabyte through memory for an
        answer that a two-hundred-candidate first guess plus a window of
        fifteen settles exactly.
        """
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        n = self.count
        step = max(1, n // 200)
        coarse = self.center[::step]
        d = ((p[:, None, :] - coarse[None, :, :]) ** 2).sum(axis=2)
        k = d.argmin(axis=1) * step
        win = np.arange(-step - 2, step + 3)
        cand = (k[:, None] + win[None, :]) % n
        q = self.center[cand] - p[:, None, :]
        return cand[np.arange(len(p)),
                    (q[:, :, 0] ** 2 + q[:, :, 1] ** 2).argmin(axis=1)]

    def barrier_distance(self, pts) -> np.ndarray:
        """Distance from each point to the nearest barrier chord, either side.

        The one honest answer to "how far outside the wall is this?", and the
        test everything placed on the roadside is held to. Measuring against
        the side's own chords alone is not enough: where the circuit doubles
        back, the nearest wall to a point sitting two metres behind one
        barrier can be the *other* one, three metres in front of it.
        """
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        r, l = self.barrier_lines()
        segs = [x for x in (r, l) if len(x)]
        if not segs:
            return np.full(len(pts), np.inf)
        return _seg_distance(pts, np.concatenate(segs, axis=0))

    def corridor_distance(self, pts) -> np.ndarray:
        """Signed distance to the drivable corridor: negative inside it.

        The same field the barrier contour is taken from, evaluated at
        arbitrary points. Positive means "out in the country, where a building
        may stand"; negative means "inside the fence", which for a grandstand
        or a lamp post is the whole of the failure.
        """
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        off_r, off_l = self.wall_offsets()
        c, nrm = self.center, self.normal
        out = np.full(len(pts), np.inf)
        for k0 in range(0, len(pts), 128):
            k1 = min(k0 + 128, len(pts))
            d = pts[k0:k1, None, :] - c[None, :, :]
            dist = np.sqrt((d * d).sum(axis=2))
            side = (d * nrm[None, :, :]).sum(axis=2) > 0.0
            rad = np.where(side, off_r[None, :], off_l[None, :])
            out[k0:k1] = (dist - rad).min(axis=1)
        return out

    def wall_ray_offsets(self) -> tuple[np.ndarray, np.ndarray]:
        """(right, left) distance from the centreline to the barrier *as drawn*.

        ``wall_offsets()`` is the radius the corridor was asked for; this is
        where the fence actually ended up after the contour was smoothed, nudged
        out and cut into chords. On a straight the two agree to within a
        centimetre. Where the wall bends hard they do not, and every artifact
        the run-off had at a sharp corner -- apron over the barrier on one side
        of the bend, a wedge of the grass plane inside it on the other -- was
        the gap between the two.

        Cast along each sample's own normal and take the first crossing, so the
        answer is the wall that sample can see. Where the ray finds nothing
        (a merged complex, where the fence wraps the pair a long way off) fall
        back to the requested radius.
        """
        if getattr(self, "_wall_ray", None) is not None:
            return self._wall_ray

        from . import config

        r, l = self.barrier_lines()
        segs = [x for x in (r, l) if len(x)]
        segs = np.concatenate(segs, axis=0) if segs else np.zeros((0, 2, 2))
        off = self.wall_offsets()
        medial = self.medial_offsets()
        out = []
        for k, (side, w, req) in enumerate(((+1.0, self.w_right, off[0]),
                                            (-1.0, self.w_left, off[1]))):
            dirs = self.normal * side
            t = _ray_first_hit(self.center, dirs, segs)
            # A correction, not a second construction. Where the fence is
            # locally the outline of this sample's own run-off the ray finds it
            # within a few metres of the requested radius and the answer is the
            # truth the apron and the roadside need. Where two legs have merged
            # there is no wall on this ray at all and it flies across the
            # infield to the far side of the circuit -- Zandvoort does this for
            # three quarters of a lap -- and an 800 m "run-off" is worse than
            # the disagreement being fixed. Past the halfway line to whatever
            # else is over there, everything is clipped to the medial axis
            # anyway, so falling back to the requested radius there changes
            # nothing that is drawn.
            med = medial[k]
            limit = np.maximum(med, req) * 1.25 + 6.0
            ok = (np.isfinite(t)
                  & (t > w + config.WALL_MIN_MARGIN * 0.5)
                  & (t <= limit))
            out.append(np.maximum(np.where(ok, t, req), w + 0.5))
        self._wall_ray = (out[0], out[1])
        return self._wall_ray

    def bank(self) -> np.ndarray:
        """Camber angle at each sample, radians, signed.

        Positive tilts the +normal (right-hand) side *down*, which is what a
        right-hand corner wants: the outside of the bend is the left, and the
        outside is the high side. So the sign is simply the sign of the
        curvature, and the magnitude comes from how tight the corner is --
        there is no camber in the source data, which is a centreline and two
        widths and nothing else.

        Smoothed over tens of metres of lap, because the thing that gives a
        banked corner away is not the angle at the apex, it is the road
        winding into it and out again. A profile that switched at the corner's
        first sample would be a step in the road surface.
        """
        if getattr(self, "_bank", None) is not None:
            return self._bank
        if not getattr(config, "BANKING_ENABLED", False):
            self._bank = np.zeros(self.count)
            return self._bank
        cap = math.radians(config.BANK_CIRCUIT_MAX.get(self.name,
                                                       config.BANK_MAX_DEG))
        mag = cap * np.clip(config.BANK_RADIUS
                            / np.maximum(self.curv_radius, 1.0),
                            0.0, 1.0) ** config.BANK_FALLOFF
        k = max(3, int(self.count * config.BANK_SMOOTH / max(self.length, 1.0)))
        # Smooth the signed value, not the magnitude: through an S the sign
        # flips, and averaging the magnitude across the flip would hold the
        # road banked the wrong way for half of the second corner.
        self._bank = _smooth_ring(np.sign(self.curvature) * mag, k)
        return self._bank

    def surface_y(self, i, lat) -> np.ndarray:
        """Height of the road surface *lat* metres right of sample *i*.

        Banked about the **low edge of the asphalt**, not about the
        centreline, so the surface is never below the plane the rest of the
        world sits on. Tilted about the centre, a corner digs a trench: one
        edge of the road ends up two metres down, the flat grass plane cuts
        straight through it, and the barrier on that side is buried to its top
        rail. Built up on the outside instead -- which is how a banked corner
        is actually made, and what Zandvoort's look like -- the low side stays
        at ground level and only the high side rises.

        Full camber across the asphalt, then faded out over the run-off: forty
        metres of apron tilted with the road would put its far edge in the air.
        """
        i = np.asarray(i)
        lat = np.asarray(lat, dtype=float)
        b = self.bank()[i]
        t = np.tan(b)
        # The low edge is the right one where the surface falls to the right.
        low = np.where(b > 0.0, self.w_right[i], -self.w_left[i])
        w = np.where(lat > 0.0, self.w_right[i], self.w_left[i])
        edge = np.clip(lat, -self.w_left[i], self.w_right[i])
        over = np.maximum(np.abs(lat) - w, 0.0)
        fade = np.clip(1.0 - over / max(config.BANK_RUNOFF_FADE, 1e-6), 0.0, 1.0)
        # Across the asphalt the height is exact; past it the same height
        # fades back to zero, so the ground leaves the corner and rejoins the
        # plane rather than stopping at a cliff.
        return (low - edge) * t * fade

    @property
    def is_flat(self) -> bool:
        """No camber anywhere: every surface height is zero (the f1tenth
        circuits as they come). Lets per-frame callers skip the lookups."""
        flat = getattr(self, "_is_flat", None)
        if flat is None:
            flat = self._is_flat = (not config.BANKING_ENABLED
                                    or not np.any(np.abs(self.bank()) > 1e-9))
        return flat

    def surface_pose(self, pts):
        """(height, camber, track normal) at arbitrary world points.

        For the renderer rather than the physics. The car and the camera are
        drawn between physics steps, and reading back the height the last step
        happened to compute makes the body climb a banked corner in
        eight-millisecond stairs. Sampled at the interpolated position it is
        as smooth as the road is.
        """
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        if self.is_flat:
            # Height and camber zero everywhere (and the callers do not use
            # the normal): skip the nearest-sample search -- this runs for
            # the car and the camera every frame.
            z = np.zeros(len(p))
            return z, z.copy(), None
        n = self.count
        i = self.nearest_indices(p)
        if not config.BANKING_ENABLED:
            # Flat circuit: height and camber are exactly zero everywhere, and
            # this is called for every car and the camera every frame.
            z = np.zeros(len(p))
            return z, z.copy(), self.normal[i]
        d = p - self.center[i]
        # Blended along the lap between the two samples the point lies
        # between. Snapping to the nearest one makes the surface a staircase
        # of five-metre treads -- every quantity it is built from (the camber,
        # the two half-widths) is stored per sample and steps at the boundary
        # -- and a car crossing those treads at sixty frames a second is the
        # judder this is here to remove.
        along = (d * self.tangent[i]).sum(axis=1)
        f = np.clip(along / np.maximum(self.seg_len[i], 1e-6), -1.0, 1.0)
        j = np.where(f >= 0.0, (i + 1) % n, (i - 1) % n)
        wt = np.abs(f)
        lat_i = (d * self.normal[i]).sum(axis=1)
        lat_j = ((p - self.center[j]) * self.normal[j]).sum(axis=1)
        h = (self.surface_y(i, lat_i) * (1.0 - wt)
             + self.surface_y(j, lat_j) * wt)
        b = self.bank()
        return h, b[i] * (1.0 - wt) + b[j] * wt, self.normal[i]

    def ground_y(self, pts) -> np.ndarray:
        """Height of the ground you can stand on, at arbitrary world points.

        The run-off apron's own surface, and the single answer to "how high is
        the ground here" for everything: the mesh that draws it, the cones on
        the kerb and the barrier at the far edge. Two descriptions of the same
        ground is how the fence came to float a hand's breadth over it.

        The road itself sits a little above this -- the apron is deliberately
        held under the asphalt -- which is right: a cone stands on the run-off,
        not on the road.
        """
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        surf, bank, _n = self.surface_pose(p)
        i = self.nearest_indices(p)
        lat = ((p - self.center[i]) * self.normal[i]).sum(axis=1)
        dist = np.abs(lat)
        w = np.where(lat > 0.0, self.w_right[i], self.w_left[i])
        under = np.clip((w + config.KERB_WIDTH + 14.0 - dist) / 14.0, 0.0, 1.0)
        sink = config.RUNOFF_BANK_SINK * np.clip(
            np.abs(np.degrees(bank)) / config.RUNOFF_BANK_SINK_REF, 0.0, 1.0)
        rise = np.minimum(config.RUNOFF_RISE * np.maximum(dist - w, 0.0),
                          config.RUNOFF_RISE_MAX)
        return (surf + config.Y_RUNOFF + rise
                - config.RUNOFF_UNDER_ROAD * under - sink)

    def corridor_field(self):
        """(field, x0, z0): the drivable region sampled onto a grid, cached.

        Built once and shared, because it is the definition everything else
        is derived from -- the barrier is its outline and the run-off apron
        is its interior. Two constructions of the same region is how the two
        came to disagree in the first place.
        """
        if getattr(self, "_field", None) is None:
            self._field = _corridor_field(self, config.BARRIER_GRID,
                                          config.BARRIER_OUTSET + 2.0)
        return self._field

    def corridor_reach(self) -> tuple[np.ndarray, np.ndarray]:
        """(right, left) distance along each sample's normal to the edge of the
        drivable region.

        This is the apron's outer edge, and the reason it is measured off the
        field rather than off ``wall_offsets()`` is that the two are not the
        same number. ``wall_offsets()`` is the radius *this* sample asked for;
        the region is the union of every sample's disc, so its boundary along
        this ray is wherever the *last* of those discs ends -- always at or
        past the requested radius, and metres past it wherever the run-off is
        opening out. Filling to the requested radius left a band of the grass
        plane showing inside the barrier for a quarter of the circuit.

        Marched along the same grid the contour is taken from, so the apron's
        edge and the fence drawn on that contour are the same line to within
        the grid cell.
        """
        if getattr(self, "_reach", None) is not None:
            return self._reach

        f, x0, z0 = self.corridor_field()
        nz, nx = f.shape
        cell = config.BARRIER_GRID
        off_r, off_l = self.wall_offsets()
        med_r, med_l = self.medial_offsets()
        steps = 512
        s = np.linspace(0.0, 1.0, steps)

        out = []
        for side, w, off, med in ((+1.0, self.w_right, off_r, med_r),
                                  (-1.0, self.w_left, off_l, med_l)):
            # Far enough that the crossing is always inside the march, close
            # enough that half a metre of step resolves it. Nothing past the
            # medial axis is ever drawn, so there is no reason to look there.
            # _medial_distance leaves inf where no other part of the lap
            # is ever nearer along this ray -- an isolated straight -- and an
            # infinite march is neither useful nor representable.
            tmax = np.where(np.isfinite(med), np.maximum(med, off), off) + 30.0
            t = tmax[:, None] * s[None, :]
            u = self.normal * side
            px = self.center[:, None, 0] + u[:, None, 0] * t
            pz = self.center[:, None, 1] + u[:, None, 1] * t
            gx = np.clip((px - x0) / cell, 0.0, nx - 1.001)
            gz = np.clip((pz - z0) / cell, 0.0, nz - 1.001)
            ix, iz = gx.astype(int), gz.astype(int)
            fx, fz = gx - ix, gz - iz
            v = (f[iz, ix] * (1 - fx) * (1 - fz) + f[iz, ix + 1] * fx * (1 - fz)
                 + f[iz + 1, ix] * (1 - fx) * fz + f[iz + 1, ix + 1] * fx * fz)
            outside = v >= 0.0
            outside[:, 0] = False                 # the centreline is inside
            k = np.argmax(outside, axis=1)
            found = outside.any(axis=1)
            a = v[np.arange(len(k)), np.maximum(k - 1, 0)]
            b = v[np.arange(len(k)), k]
            frac = np.where(b > a, -a / np.maximum(b - a, 1e-9), 0.0)
            step = tmax / (steps - 1.0)
            r = np.where(found, (k - 1 + np.clip(frac, 0.0, 1.0)) * step, tmax)
            out.append(np.maximum(r, w + 0.5))
        self._reach = (out[0], out[1])
        return self._reach

    def medial_offsets(self) -> tuple[np.ndarray, np.ndarray]:
        """(right, left) distance to the medial axis: how far each sample can
        reach out before some other part of the circuit is nearer."""
        if getattr(self, "_medial", None) is None:
            r, ri = _medial_distance(self.center, self.normal)
            l, li = _medial_distance(self.center, -self.normal)
            self._medial = (r, l)
            self._medial_src = (ri, li)
        return self._medial

    def wall_offsets(self) -> tuple[np.ndarray, np.ndarray]:
        """(right, left) distance from the centreline to the barrier, per sample.

        Offsetting each sample along its own normal by a fixed run-off width
        looks obvious and is wrong: where the circuit turns tighter than the
        offset, the offset curve crosses the centre of the corner and folds
        through itself, and where two parts of the lap pass close together the
        two walls drive through each other. Monza's first chicane does both.

        The corridor the *physics* uses has neither problem, because
        ``Surface.resolve_wall`` measures from the nearest centreline point --
        that is a distance field, and its level set is a proper offset curve
        that wraps a tight corner instead of folding into it. So this asks for
        the full run-off along each normal and lets the level set wrap: where
        two legs pass close enough that their corridors overlap, there is no
        boundary between them to find, and ``wall_valid`` drops the barrier
        there rather than trying to squeeze one in. One fence goes round the
        pair, which is what a chicane looks like from the air.

        The result is cached and shared by the wall mesh, the barrier props and
        the collision test, so the thing you see and the thing you hit cannot
        drift apart.
        """
        if getattr(self, "_wall_off", None) is not None:
            return self._wall_off

        from . import config

        c, n = self.center, self.normal
        # Extra room through a corner, weighted towards the exit. A constant
        # run-off is right for a straight and wrong everywhere else: a car
        # loses it at the exit of a corner, travelling outwards, and that is
        # where a real circuit puts its acres of asphalt. Asking for more than
        # fits between the two legs of a chicane is fine: the corridors merge
        # and the barrier between them disappears.
        k = max(2, self.count // 200)
        tan = self.tangent
        turn = np.sign(tan[:, 0] * np.roll(tan, -k, axis=0)[:, 1]
                       - tan[:, 1] * np.roll(tan, -k, axis=0)[:, 0])
        # Not "is this a corner" but "how much trouble is this corner". A
        # tight one taken fast is where a car ends up furthest from the road,
        # so severity rises as the radius falls, and is scaled again by how
        # straight the approach was -- Monza's last corner is sharp *and*
        # arrives at the end of a kilometre of full throttle, which is why it
        # has acres of sand and the Lesmos do not.
        sev = np.clip((config.RUNOFF_CORNER_RADIUS - self.curv_radius)
                      / max(config.RUNOFF_CORNER_RADIUS
                            - config.RUNOFF_SHARP_RADIUS, 1.0), 0.0, 1.0)
        back = max(1, int(self.count * config.RUNOFF_ENTRY_LOOK
                          / max(self.length, 1.0)))
        approach = np.maximum.reduce([np.roll(self.curv_radius, d)
                                      for d in range(1, back + 1)])
        entry = np.clip(approach / config.RUNOFF_ENTRY_RADIUS, 0.35, 1.0)
        corner = sev * sev * entry
        # Asymmetric smear: a few samples before the corner, several times as
        # many after it, so the widening opens out down the exit road.
        span = max(3, int(self.count * 60.0 / max(self.length, 1.0)))
        weight = corner.copy()
        for d in range(1, span + 1):
            weight = np.maximum(weight, np.roll(corner, d) * (1.0 - d / (span + 1.0)))
        for d in range(1, max(2, span // 3) + 1):
            weight = np.maximum(weight, np.roll(corner, -d) * 0.8)
        weight = _smooth_ring(weight, config.WALL_SMOOTH)

        medial = self.medial_offsets()
        out = []
        for side, w in ((+1.0, self.w_right), (-1.0, self.w_left)):
            # Full extra on the outside of the bend, a fraction on the inside.
            outside = np.where(turn == side, 1.0, 0.35)
            target = (w + config.RUNOFF_WIDTH
                      + config.RUNOFF_CORNER_EXTRA * weight * outside)
            # No medial clamp. Pulling every sample in to the medial axis is
            # what made a chicane's barrier dive into the gap between its two
            # legs and turn back on itself, and with the corner widening on top
            # the smoothing then dragged samples past their own medial distance
            # so the line crossed the next leg -- a wall standing on the road,
            # because the collision test reads this same line.
            #
            # The corridor a car may use is "within this distance of the
            # centreline", which is a distance field: where two legs pass
            # closer than their radii the corridors *merge*, and there is
            # genuinely no boundary between them. So the offset is simply the
            # target, and which of these points are really on the outside of
            # the merged region is a separate question, answered by
            # wall_valid(). Between the legs of a chicane the answer is none of
            # them, and the barrier wraps the pair instead of threading them.
            hard = w + config.WALL_MIN_MARGIN
            # Reach out only as far as the halfway line to whatever else is
            # over there, less the width of the strip a wall needs to stand
            # in. Without this the run-off of two legs simply swallowed the
            # ground between them, and with it the wall that separates them --
            # which is right for a chicane and wrong everywhere else, because
            # two legs sixty metres apart plainly do have a barrier between
            # them. Where the halfway line leaves no room for a strip at all
            # the legs really are one piece of road, so the clamp is dropped
            # and the corridors merge as before.
            med = medial[0] if side > 0 else medial[1]
            room = med - config.WALL_MERGE_GAP
            # ...and only between two legs the driver reaches one from the
            # other in a few seconds. That is the whole difference between a
            # chicane and two straights that happen to run side by side: a
            # chicane's legs are metres apart *and* seconds apart, a pair of
            # straights is metres apart and half a lap apart. Without the
            # second test Zandvoort threw away the walls that separate its
            # infield from itself, and the circuit stopped being a circuit.
            src = self._medial_src[0 if side > 0 else 1]
            lap = np.abs(self.arclen - self.arclen[src])
            lap = np.minimum(lap, self.length - lap)
            # Two legs also merge when there is physically no room for two
            # walls between them, however far apart along the lap they are.
            # The alternative is a wall standing past the halfway line, which
            # means two walls crossing, two aprons laid over each other, and
            # the lamp posts behind one of them standing in the other's
            # run-off.
            merged = _spread(((room < w + config.WALL_MERGE_FLOOR)
                              & (lap < config.WALL_MERGE_LAP))
                             | (med < w + config.WALL_MIN_MARGIN),
                             config.WALL_MERGE_SPREAD)
            clamped = np.minimum(np.maximum(np.minimum(target, room), hard),
                                 med)
            # Cross-fade rather than switch. A boolean here is a step of tens
            # of metres in the wall's radius over a single sample -- the fence
            # leaves the merged complex by turning through most of a right
            # angle -- and a corner that sharp is where the contour smoothing
            # and the chord walk disagree with each other most. The slope
            # limiter below could only ever pull the *wide* side in; blending
            # first means there is no step left for it to chase.
            blend = _smooth_ring(merged.astype(float),
                                 config.WALL_MERGE_BLEND)
            blend = np.clip(blend, 0.0, 1.0)
            best = clamped + (target - clamped) * blend
            best = _slope_limit(best, self.seg_len * config.WALL_MAX_SLOPE)
            best = _smooth_ring(best, config.WALL_SMOOTH)
            out.append(np.maximum(best, hard))
        self._wall_off = (out[0], out[1])
        return self._wall_off

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        edge_r = self.center + self.normal * self.w_right[:, None]
        edge_l = self.center - self.normal * self.w_left[:, None]
        allpts = np.vstack([edge_r, edge_l])
        return allpts.min(axis=0), allpts.max(axis=0)

    def sector_bounds(self) -> tuple[int, int]:
        """Centreline indices where sectors 2 and 3 begin.

        Thirds of the lap by distance, the way a circuit without official
        sector boards would be split. Sector 1 starts at the line.
        """
        return (int(np.searchsorted(self.arclen, self.length / 3.0)),
                int(np.searchsorted(self.arclen, 2.0 * self.length / 3.0)))

    def sector_of(self, i: int) -> int:
        b1, b2 = self.sector_bounds()
        return 0 if i < b1 else (1 if i < b2 else 2)


def _resolve_csv(name: str) -> Path:
    d = config.TRACK_DB / name
    if not d.is_dir():
        raise FileNotFoundError(f"track folder not found: {d}")
    for cand in (d / f"{name}_centerline.csv", *d.glob("*_centerline.csv")):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"no *_centerline.csv in {d}")


def load_track(name: str) -> Track:
    csv = _resolve_csv(name)
    raw = np.loadtxt(csv, delimiter=",", comments="#")  # (N, 4)
    xy = raw[:, :2]
    w_r = raw[:, 2]
    w_l = raw[:, 3]

    # Drop a duplicate closing point if present, then work as a cyclic loop.
    if np.linalg.norm(xy[0] - xy[-1]) < 1e-3:
        xy, w_r, w_l = xy[:-1], w_r[:-1], w_l[:-1]

    s = config.TRACK_SCALE_BY_NAME.get(name, config.TRACK_SCALE)
    center = xy * s

    # Width is renormalised rather than scaled: the source margins are sized
    # for 1:10 RC cars and would give a ~28 m wide circuit. Keep the *shape* of
    # the stored width profile but map its mean onto a realistic F1 width.
    half = config.TRACK_WIDTH_MEAN * 0.5
    stored = (w_r + w_l) * 0.5
    mean = float(np.mean(stored)) or 1.0
    shaped = 1.0 + config.TRACK_WIDTH_VARIATION * (stored / mean - 1.0)
    bias_r = w_r / np.maximum(w_r + w_l, 1e-9) * 2.0   # keep left/right asymmetry
    w_right = half * shaped * bias_r
    w_left = half * shaped * (2.0 - bias_r)

    # Central-difference tangent on the cyclic loop.
    nxt = np.roll(center, -1, axis=0)
    prv = np.roll(center, 1, axis=0)
    tan = nxt - prv
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    # Right-hand normal in the xz-plane.
    normal = np.stack([tan[:, 1], -tan[:, 0]], axis=1)

    seg = np.linalg.norm(nxt - center, axis=1)
    arclen = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    length = float(np.sum(seg))

    # Menger curvature over a wide stencil so per-point noise doesn't create
    # phantom hairpins; then smooth the radius a little.
    k = max(3, len(center) // 300)
    p0 = np.roll(center, k, axis=0)
    p2 = np.roll(center, -k, axis=0)
    a = np.linalg.norm(center - p0, axis=1)
    b = np.linalg.norm(p2 - center, axis=1)
    cc = np.linalg.norm(p2 - p0, axis=1)
    area = np.abs((center[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                  - (center[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])) * 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        radius = np.where(area > 1e-6, (a * b * cc) / (4.0 * area), 1e9)
    win = np.ones(5) / 5.0
    radius = np.convolve(np.concatenate([radius[-4:], radius, radius[:4]]),
                         win, mode="same")[4:-4]
    radius = np.clip(radius, 1.0, 1e9)

    # Signed curvature over the same stencil: which way the corner goes, which
    # a bare radius cannot say. Positive = turning right (towards +normal).
    turn = ((center[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
            - (center[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]))
    curvature = -np.sign(turn) / radius

    # Stop the ribbon folding over itself in tight chicanes: the edge can't sit
    # further from the centre than most of the local corner radius.
    w_cap = 0.82 * radius
    w_right = np.minimum(w_right, w_cap)
    w_left = np.minimum(w_left, w_cap)

    return Track(
        name=name, center=center, tangent=tan, normal=normal,
        w_right=w_right, w_left=w_left, curv_radius=radius, curvature=curvature,
        seg_len=seg, arclen=arclen, length=length,
    )
