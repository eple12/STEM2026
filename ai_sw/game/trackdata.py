"""Load an F1TENTH racetrack and derive the geometry the game needs.

CSV columns:  ``x_m, y_m, w_tr_right_m, w_tr_left_m``  (comma separated, one
comment header line starting with '#'). The centreline is a closed loop.

World mapping:  f1tenth (x, y)  ->  world (x, z),  world y is up.
"""
from __future__ import annotations

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
    for a in range(0, n, block):
        b = min(a + block, n)
        diff = c[None, :, :] - c[a:b, None, :]              # (blk, n, 2)
        dot = np.einsum("bnk,bk->bn", diff, u[a:b])
        sq = np.einsum("bnk,bnk->bn", diff, diff)
        # Only points the ray actually approaches can constrain it; a tangent
        # neighbour has dot ~ 0 and would otherwise blow up.
        t = np.where(dot > 1e-6, sq / (2.0 * np.maximum(dot, 1e-6)), np.inf)
        out[a:b] = t.min(axis=1)
    return out


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


def _smooth_ring(a: np.ndarray, k: int) -> np.ndarray:
    """Light circular moving average, to take the facets off a limited slope."""
    if k < 3 or len(a) < k:
        return a
    half = k // 2
    pad = np.concatenate([a[-half:], a, a[:half]])
    return np.convolve(pad, np.ones(2 * half + 1) / (2 * half + 1), mode="valid")


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
        n = self.count
        px, pz = float(pos_xz[0]), float(pos_xz[1])
        idx = (hint + np.arange(-window, window + 1)) % n
        pts = self.center[idx]
        dx = pts[:, 0] - px
        dz = pts[:, 1] - pz
        k = int(np.argmin(dx * dx + dz * dz))
        if 0 < k < 2 * window:
            return int(idx[k])

        dx = self.center[:, 0] - px
        dz = self.center[:, 1] - pz
        return int(np.argmin(dx * dx + dz * dz))

    def barrier_lines(self) -> tuple[np.ndarray, np.ndarray]:
        """(right, left) barrier polylines, as arrays of (start, end) points.

        These are the exact chords the barrier props are placed along, and the
        exact segments the car collides with. Before this existed the two were
        derived separately: the props were laid along chords of the wall line,
        while the collision test compared the car's distance from the nearest
        *centreline* sample against the wall offset there. On a straight the
        two agree; through a corner they do not, because a chord across a bend
        leaves the arc it was cut from. Measured over four circuits, the
        barrier you could see sat as much as 4.97 m outside the wall the car
        actually stopped at -- the car bounced off nothing, in open runoff,
        with the barrier still metres away.

        Run length and bend are capped so the chords stay close to the wall
        line, and one run becomes one stretched barrier module.
        """
        if getattr(self, "_barrier_lines", None) is not None:
            return self._barrier_lines

        off_r, off_l = self.wall_offsets()
        step = config.BARRIER_STEP
        targets = np.arange(0.0, max(self.length, step), step)
        idx = np.searchsorted(self.arclen, targets).clip(0, self.count - 1)

        out = []
        for side, off in ((+1, off_r), (-1, off_l)):
            # A hair outside the wall line: the modules are straight chords
            # across a curved wall, so on the inside of a bend they would
            # otherwise dip behind it and let the ground show through.
            pts = (self.center[idx]
                   + self.normal[idx]
                   * (side * (off[idx] + config.BARRIER_OUTSET))[:, None])
            n = len(pts)
            segs = []
            k = 0
            while k < n:
                a = pts[k]
                j = k + 1
                while j < n + 1:
                    b = pts[j % n]
                    d = b - a
                    if np.hypot(d[0], d[1]) > config.BARRIER_MAX_RUN:
                        break
                    nxt = pts[(j + 1) % n] - b
                    if _turn_degrees(d, nxt) > config.BARRIER_MAX_BEND:
                        j += 1
                        break
                    j += 1
                b = pts[j % n]
                k = j
                if np.hypot(*(b - a)) < 0.05:
                    continue
                segs.append((a, b))
            out.append(np.asarray(segs, dtype=float).reshape(-1, 2, 2))

        self._barrier_lines = (out[0], out[1])
        return self._barrier_lines

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
        that wraps a tight corner instead of folding into it. So this walks out
        along each normal and stops at the point where some *other* part of the
        centreline becomes the nearest one, which is exactly that level set.

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
        # where a real circuit puts its acres of asphalt. The medial-distance
        # clamp below still applies, so asking for more where there is none --
        # between the two legs of a chicane -- simply gets what fits.
        k = max(2, self.count // 200)
        tan = self.tangent
        turn = np.sign(tan[:, 0] * np.roll(tan, -k, axis=0)[:, 1]
                       - tan[:, 1] * np.roll(tan, -k, axis=0)[:, 0])
        corner = (self.curv_radius < config.RUNOFF_CORNER_RADIUS).astype(float)
        # Asymmetric smear: a few samples before the corner, several times as
        # many after it, so the widening opens out down the exit road.
        span = max(3, int(self.count * 60.0 / max(self.length, 1.0)))
        weight = corner.copy()
        for d in range(1, span + 1):
            weight = np.maximum(weight, np.roll(corner, d) * (1.0 - d / (span + 1.0)))
        for d in range(1, max(2, span // 3) + 1):
            weight = np.maximum(weight, np.roll(corner, -d) * 0.8)
        weight = _smooth_ring(weight, config.WALL_SMOOTH)

        out = []
        for side, w in ((+1.0, self.w_right), (-1.0, self.w_left)):
            # Full extra on the outside of the bend, a fraction on the inside.
            outside = np.where(turn == side, 1.0, 0.35)
            target = w + config.RUNOFF_WIDTH                 + config.RUNOFF_CORNER_EXTRA * weight * outside
            # Never onto the asphalt; but where the circuit doubles back on
            # itself this has to win over any prettier minimum, because there
            # genuinely is no room for a barrier between the two.
            hard = w + 0.3
            limit = _medial_distance(c, n * side)
            best = np.clip(limit - config.WALL_GAP, hard, target)
            best = _slope_limit(best, self.seg_len * config.WALL_MAX_SLOPE)
            best = _smooth_ring(best, config.WALL_SMOOTH)
            # Clamp against the medial distance *again*, not just against the
            # target. The slope limit and the smoothing both move a sample
            # towards its neighbours, and with the corner widening the target
            # now swings from 13 m to 60 m over a few samples -- so a sample
            # next to a wide corner gets dragged out past its own medial
            # distance, and the barrier crosses the next leg of the circuit.
            # Since the collision test reads this same line, that is a wall
            # standing on the road: exactly the invisible wall at Shanghai's
            # sharp turns.
            ceiling = np.minimum(target, np.maximum(limit - config.WALL_GAP,
                                                    hard))
            out.append(np.clip(best, hard, ceiling))
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
