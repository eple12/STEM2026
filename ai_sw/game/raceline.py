"""A racing line, and the lap-time model used to find it.

The planner in ``autopilot.py`` drives the centreline: it feeds forward the
centreline's curvature and pulls its cross-track error back to the centreline.
On a 13.5 m circuit it uses about 2 m of that width. Lap time in racing comes
overwhelmingly from the line -- a corner taken across the full width has a much
larger radius than the same corner taken down the middle, and corner speed goes
as the square root of radius -- so that is where the AI's deficit is, and it is
a geometry problem rather than a control one.

Two stages, in increasing cost:

* ``min_curvature`` solves a box-constrained least-squares problem for the
  smoothest line inside the track. It is deterministic, takes milliseconds, and
  lands close to optimal because in a corner "smoothest" and "fastest" mostly
  agree.
* ``refine`` runs CMA-ES on the actual modelled lap time from there. That is
  where the two disagree -- the fastest line sacrifices curvature in slow
  corners to straighten the fast ones, and gives up entry speed for exit onto a
  straight.

Everything here works on the analytic quasi-steady speed profile rather than by
simulating: a simulated lap costs 13.8 s of wall clock, a modelled one costs
about 20 ms. That ratio is the whole reason this is affordable.
"""
from __future__ import annotations

import math

import numpy as np

from . import config
from .trackdata import Track


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def line_points(track: Track, offset: np.ndarray) -> np.ndarray:
    """The line itself: the centreline pushed sideways by *offset* metres."""
    return track.center + track.normal * offset[:, None]


def line_geometry(points: np.ndarray):
    """(segment lengths, signed curvature, radius) of a closed polyline.

    Curvature from the turn between consecutive segments over the mean of their
    lengths, which is the discrete definition and needs no even spacing -- the
    samples are not evenly spaced once the line moves off the centreline.
    """
    d = np.roll(points, -1, axis=0) - points
    seg = np.hypot(d[:, 0], d[:, 1])
    heading = np.arctan2(d[:, 0], d[:, 1])
    turn = (np.roll(heading, -1) - heading + np.pi) % (2.0 * np.pi) - np.pi
    span = np.maximum(0.5 * (seg + np.roll(seg, -1)), 1e-6)
    # Curvature belongs to the joint *after* segment i, so roll it back to sit
    # on the same index as the point it describes.
    curvature = np.roll(turn / span, 1)
    radius = 1.0 / np.maximum(np.abs(curvature), 1e-6)
    return seg, curvature, radius


def bounds(track: Track, margin: float | None = None):
    """How far the line may move each way, in metres, per sample.

    Measured to the asphalt edge less half the car's width, plus whatever slice
    of the kerb ``RACELINE_KERB`` allows: a real line uses the kerb, and
    refusing to costs most of the width gain at exactly the corners where the
    width is worth the most.
    """
    if margin is None:
        margin = config.BODY_HALF_WIDTH + config.RACELINE_EDGE_MARGIN
    kerb = config.KERB_WIDTH * config.RACELINE_KERB
    left = np.maximum(track.w_left + kerb - margin, 0.0)
    right = np.maximum(track.w_right + kerb - margin, 0.0)
    # Offsets are measured along +normal, which is the right-hand side.
    return -left, right


# --------------------------------------------------------------------------
# the lap-time model
# --------------------------------------------------------------------------
def corner_speed(radius: np.ndarray, pace: float) -> np.ndarray:
    """Vectorised twin of ``Autopilot._corner_speed``.

    Same physics, and it has to stay the same: a line optimised against a
    different grip model than the one that drives it is optimised for a car
    that is not on the track.
    """
    cfg = config
    mu = cfg.TYRE_GRIP * pace
    L, m = cfg.WHEELBASE, cfg.CAR_MASS
    R = np.maximum(radius, 1.0)

    w_f = (cfg.CG_TO_REAR / L) * m * cfg.GRAVITY
    w_r = (cfg.CG_TO_FRONT / L) * m * cfg.GRAVITY
    cl_f = cfg.DOWNFORCE_COEFF * cfg.DOWNFORCE_FRONT_BIAS
    cl_r = cfg.DOWNFORCE_COEFF * (1.0 - cfg.DOWNFORCE_FRONT_BIAS)

    best = np.full(R.shape, cfg.MAX_SPEED, dtype=float)
    for w0, cl, arm in ((w_f, cl_f, cfg.CG_TO_REAR),
                        (w_r, cl_r, cfg.CG_TO_FRONT)):
        k = mu * L / (m * arm)
        denom = 1.0 / R - k * cl
        ok = denom > 0.0                      # downforce alone carries these
        cand = np.full(R.shape, cfg.MAX_SPEED, dtype=float)
        cand[ok] = np.sqrt(np.maximum(k * w0 / denom[ok], 1.0))
        best = np.minimum(best, cand)
    return np.minimum(best, cfg.MAX_SPEED)


def _sweep(v: np.ndarray, seg: np.ndarray, accel: float, forward: bool):
    """One braking or acceleration pass round the loop, in place."""
    n = len(v)
    for _ in range(2):                        # twice, to wrap the loop
        if forward:
            for i in range(n):
                ds = seg[i - 1]
                lim = math.sqrt(v[i - 1] ** 2 + 2.0 * accel * ds)
                if lim < v[i]:
                    v[i] = lim
        else:
            for i in range(n - 1, -1, -1):
                ds = seg[i]
                lim = math.sqrt(v[(i + 1) % n] ** 2 + 2.0 * accel * ds)
                if lim < v[i]:
                    v[i] = lim


def speed_profile(seg: np.ndarray, curvature: np.ndarray, radius: np.ndarray,
                  pace: float) -> np.ndarray:
    """Highest speed at every point, the same way the planner builds it."""
    cfg = config
    n = len(seg)
    win = max(4, int(28.0 / max(float(np.median(seg)), 1e-3)))

    r = radius.copy()
    for k in range(1, win + 1):
        r = np.minimum(r, np.roll(radius, k))
        r = np.minimum(r, np.roll(radius, -k))
    v = corner_speed(r, pace)

    swing = np.zeros(n)
    for k in range(1, win + 1):
        swing = np.maximum(swing, np.abs(np.roll(curvature, -k) - curvature))
    v *= 1.0 / (1.0 + 14.0 * swing)

    _sweep(v, seg, 0.62 * cfg.BRAKE_FORCE / cfg.CAR_MASS, forward=False)
    _sweep(v, seg, cfg.ENGINE_FORCE_MAX / cfg.CAR_MASS, forward=True)
    return np.clip(v, 6.0, cfg.MAX_SPEED)


def lap_time(track: Track, offset: np.ndarray, pace: float) -> float:
    """Modelled lap time for a line. Not a simulated time -- a comparable one.

    The planner cannot realise this number (it has to steer to the line, and
    the profile is quasi-steady), so it is an optimistic bound. What it has to
    be is *ordered the same way* as the real thing, which the check in
    ``tools/train_raceline.py`` confirms by simulating the result.
    """
    seg, curvature, radius = line_geometry(line_points(track, offset))
    v = speed_profile(seg, curvature, radius, pace)
    # Time over each segment at the mean of its end speeds.
    v_seg = np.maximum(0.5 * (v + np.roll(v, -1)), 1e-3)
    return float(np.sum(seg / v_seg))


# --------------------------------------------------------------------------
# stage 1: the smoothest line inside the track
# --------------------------------------------------------------------------
def min_curvature(track: Track) -> np.ndarray:
    """Box-constrained least squares for the smoothest line.

    Curvature is close to linear in the offset for offsets small against the
    radius, so ``kappa ~= A w + k0`` with A the second difference along arc
    length. Minimising ``||A w + k0||`` inside the track bounds is then a plain
    bounded least-squares problem -- milliseconds, no starting guess, no local
    minima -- and it lands near the fast line because through a corner smooth
    and fast mostly want the same thing.
    """
    from scipy.optimize import lsq_linear

    n = track.count
    ds = np.maximum(track.seg_len, 1e-3)
    h = float(np.mean(ds))

    # Second difference round the loop, scaled to arc length.
    rows = np.arange(n)
    A = np.zeros((n, n))
    A[rows, (rows - 1) % n] = 1.0 / (h * h)
    A[rows, rows] = -2.0 / (h * h)
    A[rows, (rows + 1) % n] = 1.0 / (h * h)

    lo, hi = bounds(track)
    res = lsq_linear(A, -track.curvature, bounds=(lo, hi),
                     max_iter=config.RACELINE_LSQ_ITERS)
    return np.asarray(res.x, dtype=float)


# --------------------------------------------------------------------------
# stage 2: search on the modelled lap time
# --------------------------------------------------------------------------
def _control_matrix(n: int, k: int) -> np.ndarray:
    """Maps k control values to n per-sample offsets, periodically and smoothly.

    Optimising all n offsets directly would search a space where most points
    are lines no car could follow. k control points spread round the lap, blended
    with a raised cosine, keeps every candidate driveable and cuts the search
    from a thousand dimensions to a few dozen.
    """
    s = np.arange(n) * (k / n)
    B = np.zeros((n, k))
    for j in range(k):
        d = np.abs(s - j)
        d = np.minimum(d, k - d)              # wrap
        w = np.where(d < 1.0, 0.5 * (1.0 + np.cos(np.pi * d)), 0.0)
        B[:, j] = w
    # Partition of unity, so a constant control vector is a constant offset.
    return B / np.maximum(B.sum(axis=1, keepdims=True), 1e-9)


def refine(track: Track, start: np.ndarray, pace: float,
           controls: int | None = None, generations: int | None = None,
           seed: int = 0, report=None) -> tuple[np.ndarray, float]:
    """CMA-ES on the modelled lap time, from *start*.

    An evolution strategy, like NEAT, but over a few dozen numbers that
    describe *where the car drives* rather than over the weights and topology
    of a network that would have to rediscover the track from scratch. That is
    the whole difference in cost: this converges in thousands of 20 ms model
    evaluations instead of tens of thousands of 14 s simulated laps.
    """
    n = track.count
    k = controls or config.RACELINE_CONTROLS
    gens = generations or config.RACELINE_GENERATIONS
    B = _control_matrix(n, k)
    lo, hi = bounds(track)

    # Least-squares fit of the starting line onto the control basis.
    x = np.linalg.lstsq(B, start, rcond=None)[0]

    def evaluate(vec: np.ndarray) -> float:
        return lap_time(track, np.clip(B @ vec, lo, hi), pace)

    rng = np.random.default_rng(seed)
    sigma = config.RACELINE_SIGMA
    lam = 4 + int(3 * math.log(k))            # CMA-ES default population
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mu_eff = 1.0 / np.sum(weights ** 2)

    c_sigma = (mu_eff + 2.0) / (k + mu_eff + 5.0)
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1) / (k + 1)) - 1) + c_sigma
    c_c = (4.0 + mu_eff / k) / (k + 4.0 + 2.0 * mu_eff / k)
    c_1 = 2.0 / ((k + 1.3) ** 2 + mu_eff)
    c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((k + 2) ** 2 + mu_eff))
    chi = math.sqrt(k) * (1.0 - 1.0 / (4.0 * k) + 1.0 / (21.0 * k * k))

    p_sigma = np.zeros(k)
    p_c = np.zeros(k)
    C = np.eye(k)
    best_x, best_f = x.copy(), evaluate(x)

    for g in range(gens):
        # Eigen-decomposition every few generations; it is the only O(k^3) part
        # and the covariance does not move fast enough to need it every time.
        if g % max(1, k // 4) == 0:
            C = np.triu(C) + np.triu(C, 1).T
            eig_v, eig_B = np.linalg.eigh(C)
            eig_v = np.maximum(eig_v, 1e-20)
            BD = eig_B @ np.diag(np.sqrt(eig_v))

        z = rng.standard_normal((lam, k))
        y = z @ BD.T
        pop = x + sigma * y
        fit = np.array([evaluate(p) for p in pop])
        order = np.argsort(fit)
        if fit[order[0]] < best_f:
            best_f, best_x = float(fit[order[0]]), pop[order[0]].copy()

        y_w = weights @ y[order[:mu]]
        x = x + sigma * y_w

        inv_sqrt = eig_B @ np.diag(1.0 / np.sqrt(eig_v)) @ eig_B.T
        p_sigma = ((1 - c_sigma) * p_sigma
                   + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt @ y_w))
        h_sig = (np.linalg.norm(p_sigma)
                 / math.sqrt(1 - (1 - c_sigma) ** (2 * (g + 1))) / chi
                 < 1.4 + 2.0 / (k + 1))
        p_c = ((1 - c_c) * p_c
               + h_sig * math.sqrt(c_c * (2 - c_c) * mu_eff) * y_w)
        rank_mu = sum(w * np.outer(yi, yi)
                      for w, yi in zip(weights, y[order[:mu]]))
        C = ((1 - c_1 - c_mu) * C
             + c_1 * (np.outer(p_c, p_c)
                      + (not h_sig) * c_c * (2 - c_c) * C)
             + c_mu * rank_mu)
        sigma *= math.exp((c_sigma / d_sigma)
                          * (np.linalg.norm(p_sigma) / chi - 1.0))
        sigma = float(np.clip(sigma, 1e-4, 5.0))

        if report is not None and (g + 1) % 20 == 0:
            report(g + 1, best_f, sigma)

    return np.clip(B @ best_x, lo, hi), best_f


# --------------------------------------------------------------------------
# a line the planner can drive
# --------------------------------------------------------------------------
class Line:
    """The guidance geometry of a racing line.

    Exposes the same attribute names as :class:`Track` -- ``center``,
    ``tangent``, ``normal``, ``curvature`` and the rest -- so the planner can
    be pointed at a line instead of the centreline by swapping one reference
    rather than by threading a second geometry through every calculation.

    Indexed by centreline sample, so ``Surface.progress`` still says which
    index the car is at and the two agree on what "here" means.
    """

    def __init__(self, track: Track, offset: np.ndarray, smooth: float = 0.0):
        self.offset = np.asarray(offset, dtype=float)
        self.center = line_points(track, self.offset)
        self.count = len(self.center)

        seg, curvature, radius = line_geometry(self.center)
        if smooth > 0.0:
            # Curvature measured from a polyline 5 m at a time has no physical
            # content at that scale -- the car's wheelbase is 2.65 m -- and one
            # kinked sample reads as a radius of a few metres. The speed
            # profile takes the *minimum* radius over a window, so a single
            # such sample poisons twenty-five metres of track: on the imported
            # line, a real 393 m corner came out as a 12 m one and the
            # reference speed there collapsed from 317 km/h to 26.
            win = max(3, int(smooth / max(float(np.median(seg)), 1e-3)) | 1)
            kern = np.hanning(win + 2)[1:-1]
            kern /= kern.sum()
            pad = np.r_[curvature[-win:], curvature, curvature[:win]]
            curvature = np.convolve(pad, kern, mode="same")[win:-win]
            radius = 1.0 / np.maximum(np.abs(curvature), 1e-6)
        self.seg_len = seg
        self.curvature = curvature
        self.curv_radius = radius
        self.length = float(seg.sum())
        self.arclen = np.concatenate(([0.0], np.cumsum(seg)[:-1]))

        d = np.roll(self.center, -1, axis=0) - self.center
        # The tangent at a point is the mean of the segments meeting there, or
        # the line kinks at every sample and the feedforward chatters.
        tan = d + np.roll(d, 1, axis=0)
        norm = np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-9)
        self.tangent = tan / norm[:, None]
        self.normal = np.stack([self.tangent[:, 1], -self.tangent[:, 0]], axis=1)


def load(track: Track):
    """The saved line for a circuit, or None if it has not been built."""
    path = config.RACELINE_DIR / f"{track.name}.npy"
    if not path.exists():
        return None
    offset = np.load(path)
    if len(offset) != track.count:
        print(f"raceline: {path.name} has {len(offset)} samples, "
              f"{track.name} has {track.count} -- ignoring it")
        return None
    return Line(track, offset)


def load_tuning(track: Track):
    """(pace, constants) learned alongside the line, or (None, None).

    The two are one answer, not two. Measured on Monza: the trained line driven
    with the hand-picked constants laps 141.2 s, *slower* than the centreline's
    126.0 -- because the line only pays off if the driver brakes late enough
    and holds enough entry speed to use it. Loading one without the other is
    worse than loading neither.
    """
    path = config.RACELINE_DIR / f"{track.name}_tuning.json"
    if not path.exists():
        return None, None
    import json

    data = json.loads(path.read_text())
    return float(data.get("pace", config.GHOST_PACE)), dict(data.get("tuning", {}))


def load_speed(track: Track):
    """The learned per-sample speed multiplier, or None.

    Separate from the line because it answers a different question: the line is
    where to drive, this is how much of the analytic profile's caution to keep
    there. The planner's profile takes the tightest radius over a window and
    penalises curvature reversals, which is right on average and wrong corner
    by corner -- this is what stops it braking a chicane's exit as if it were
    still the entry.
    """
    path = config.RACELINE_DIR / f"{track.name}_speed.npy"
    if not path.exists():
        return None
    scale = np.load(path)
    if len(scale) != track.count:
        return None
    return scale


def save(track: Track, offset: np.ndarray):
    config.RACELINE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.RACELINE_DIR / f"{track.name}.npy",
            np.asarray(offset, dtype=float))
