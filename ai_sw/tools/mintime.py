"""The minimum-lap-time line, solved rather than learned.

The RL driver has to *discover* how to lap fast from a reward, and every
reward is a weighted stand-in for the real question. Here the real question is
asked directly: over one closed lap of this circuit, with this game's own car,
which steering and pedal inputs give the least lap time while every wheel stays
inside the white line? That is an optimal-control problem, and with a
deterministic simulator it can be solved outright.

Formulation (the standard one for this, e.g. Perantoni & Limebeer 2014, and the
TUM minimum-time planner):

* The independent variable is distance along a reference curve, not time. Lap
  time is then simply the integral of ``1 / s_dot`` and the lap closes by
  construction (periodic boundary conditions).
* States: lateral offset ``n``, heading relative to the track ``xi``, body
  velocities ``vx``/``vy``, yaw rate ``r`` and front wheel angle ``delta``.
  Controls: steering rate, drive force, brake force.
* The car is ``game.vehicle.Vehicle`` written out as smooth equations: the same
  Pacejka curve, friction circle, longitudinal load transfer, downforce, drag,
  power limit, steering-rack rate and speed-dependent lock. The driver aids are
  not bypassed but turned into the limits they enforce: ABS and TC become caps
  on brake and drive force, the steering assist a cap on front slip. A line
  that satisfies them is one the game's own car can drive without any aid
  stepping in.
* Direct collocation (Legendre, degree 3) on every centreline sample, solved by
  IPOPT. No reward weights anywhere -- only a whisper of input-smoothness
  regularisation so the solver does not chatter.

The result is a local optimum of *this* model. The two ways it can be wrong
are both measurable: the model can differ from ``Vehicle.step`` (``--check``
compares them directly) and the solver can find a local rather than global
optimum (start it from different initial lines and compare).

    python tools/mintime.py --check                # model vs Vehicle.step
    python tools/mintime.py --circuit Monza        # solve, save, plot
    python tools/mintime.py --circuit Monza --grip 0.97 --tag g97 --warm
                                                   # a plan a tracker can hold
    python tools/mintime.py --circuit Monza --tag g97 --drive
                                                   # drive it in Vehicle.step,
                                                   # compare with the RL policy
    python tools/mintime.py --circuit Monza --kerb # allow the kerb (two wheels)

``--grip 1.0`` is the true limit of the model; driven by a tracker it has no
grip left to correct the smallest error with, so the plan the game should
drive is a fraction under it (0.97 measured clean on Monza and Melbourne).

Output: ``assets/racelines/<Circuit>_mintime[_kerb][_<tag>].npz`` and a plot
beside it; ``--drive`` adds ``..._compare.png``.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import casadi as ca
import numpy as np

from game import config as cfg
from game import raceline
from game.trackdata import load_track
from game.vehicle import Vehicle, _pacejka_b

# --------------------------------------------------------------------------
# the car, as smooth equations
# --------------------------------------------------------------------------
M = cfg.CAR_MASS
LF, LR = cfg.CG_TO_FRONT, cfg.CG_TO_REAR
L = LF + LR
G = cfg.GRAVITY
IZ = cfg.YAW_INERTIA
MU = cfg.TYRE_GRIP
MU_R = MU * cfg.REAR_GRIP_BIAS
CF, CR = cfg.CORNER_STIFFNESS_FRONT, cfg.CORNER_STIFFNESS_REAR
PC, PE = cfg.PACEJKA_C, cfg.PACEJKA_E
PB = _pacejka_b(PC, PE)
BB = cfg.BRAKE_BIAS_FRONT
W_F = (LR / L) * M * G                 # static axle loads
W_R = (LF / L) * M * G
YAW_DAMP = 0.6                         # Vehicle.step: yaw_rate *= exp(-0.6 dt)
IDLE_SOFT = 300.0                      # N of brake over which idle drag phases in

#: State and control order, and the scales the solver works in.
X_NAMES = ("n", "xi", "vx", "vy", "r", "delta")
U_NAMES = ("ddelta", "Fd", "Fb")
XS = np.array([5.0, 0.2, 50.0, 2.0, 1.0, 0.1])
US = np.array([1.0, 1.0e4, 1.0e4])


def _smax(a, b, eps):
    return 0.5 * (a + b + ca.sqrt((a - b) ** 2 + eps * eps))


def _smin(a, b, eps):
    return 0.5 * (a + b - ca.sqrt((a - b) ** 2 + eps * eps))


def _tyre(slip, stiff, mu_eff, load):
    """Vehicle.tyre_force with the friction circle already folded into mu_eff."""
    cap = mu_eff * load
    b = PB * stiff / mu_eff                 # B = PB / peak_slip, peak = mu/stiff
    ba = -b * slip
    return cap * ca.sin(PC * ca.atan(ba - PE * (ba - ca.atan(ba))))


def car_terms(x, u, mu_scale=1.0):
    """Everything the dynamics and the limits need, in physical units."""
    n, xi, vx, vy, r, delta = (x[i] for i in range(6))
    dd, fd, fb = u[0], u[1], u[2]
    mu, mu_r = MU * mu_scale, MU_R * mu_scale

    df = cfg.DOWNFORCE_COEFF * vx * vx
    drag = cfg.DRAG_COEFF * vx * vx
    idle = cfg.IDLE_DRAG * ca.tanh(fb / IDLE_SOFT)
    f_long = fd - fb - drag - cfg.ROLL_RESIST * vx - idle
    ax = f_long / M

    transfer = cfg.WEIGHT_TRANSFER * ax * M * cfg.CG_HEIGHT / L
    fz_f = W_F - transfer + df * cfg.DOWNFORCE_FRONT_BIAS
    fz_r = W_R + transfer + df * (1.0 - cfg.DOWNFORCE_FRONT_BIAS)

    fx_f = fb * BB
    fx_r = fb * (1.0 - BB) + fd
    used_f = fx_f / (mu * fz_f)
    used_r = fx_r / (mu_r * fz_r)
    grip_f = mu * ca.sqrt(1.0 - used_f * used_f)
    grip_r = mu_r * ca.sqrt(1.0 - used_r * used_r)

    a_f = ca.atan((vy + r * LF) / vx) - delta
    a_r = ca.atan((vy - r * LR) / vx)
    lat_f = _tyre(a_f, CF, grip_f, fz_f)
    lat_r = _tyre(a_r, CR, grip_r, fz_r)

    a_lat = (lat_f * ca.cos(delta) + lat_r) / M
    dvx = ax + vy * r
    dvy = a_lat - vx * r
    dr = (lat_f * ca.cos(delta) * LF - lat_r * LR) / IZ - YAW_DAMP * r
    if cfg.ESC_ENABLED and cfg.STEER_ASSIST:
        # Vehicle._stability_control: yaw beyond 1.15x what the steering asks
        # for (itself capped by the grip) is bled off at ESC_GAIN per second.
        # It matters exactly where a line is decided -- through a chicane the
        # wheel swings through straight ahead while the car is still rotating
        # the old way, and ESC takes some of that rotation out.
        wanted = ca.sqrt((vx * ca.tan(delta) / L) ** 2 + 1e-6)
        grip_cap = mu * (M * G + df) / (M * vx)
        allowed = cfg.ESC_DEADBAND * _smin(wanted, grip_cap, 1e-3)
        r_abs = ca.sqrt(r * r + 1e-6)
        excess = _smax(r_abs - allowed, 0.0, 1e-3)
        dr = dr - cfg.ESC_GAIN * excess * r / r_abs
    return dict(vx=vx, vy=vy, r=r, delta=delta, n=n, xi=xi, fd=fd, fb=fb,
                dd=dd, df=df, fz_f=fz_f, fz_r=fz_r, fx_f=fx_f, fx_r=fx_r,
                used_f=used_f, used_r=used_r, a_f=a_f, a_r=a_r, lat_f=lat_f,
                lat_r=lat_r, dvx=dvx, dvy=dvy, dr=dr, ax=ax, mu=mu, mu_r=mu_r)


def steer_limit(speed, mu):
    """Vehicle.steer_limit, smooth enough for the solver above 10 m/s."""
    t = (speed - cfg.STEER_MARGIN_KNEE) / (cfg.STEER_MARGIN_FULL
                                           - cfg.STEER_MARGIN_KNEE)
    t = ca.fmin(ca.fmax(t, 0.0), 1.0)
    margin = cfg.STEER_LIMIT_MARGIN + cfg.STEER_MARGIN_LOW_EXTRA * (
        1.0 - t * t * (3.0 - 2.0 * t))
    v2 = speed * speed
    df = cfg.DOWNFORCE_COEFF * v2
    a_max = mu * (M * G + df) / M
    load_f = W_F + df * cfg.DOWNFORCE_FRONT_BIAS
    load_r = W_R + df * (1.0 - cfg.DOWNFORCE_FRONT_BIAS)
    d_geom = L * a_max / v2
    d_slip = (M * a_max / L) * (LR / (CF * load_f) - LF / (CR * load_r))
    d = (d_geom + ca.fmax(d_slip, 0.0)) * margin
    return ca.fmin(ca.fmax(d, cfg.STEER_LIMIT_FLOOR), cfg.MAX_STEER)


def limits(t):
    """Inequalities g <= 0 that keep every driver aid from stepping in.

    Each is scaled to order one so IPOPT weighs them evenly.
    """
    g = []
    # Steering: inside the lock the game maps a full stick to...
    speed = ca.sqrt(t["vx"] ** 2 + t["vy"] ** 2)
    lim = steer_limit(speed, t["mu"])
    g.append((t["delta"] - lim) / 0.1)
    g.append((-t["delta"] - lim) / 0.1)
    # ...and below the slip at which the steering assist starts cutting it.
    slip_cap = cfg.ASSIST_SLIP_ALLOW * t["mu"] / CF
    g.append((t["a_f"] ** 2 - slip_cap ** 2) / slip_cap ** 2)

    # Engine: force ceiling and power.
    g.append((t["fd"] - cfg.ENGINE_FORCE_MAX) / 1e4)
    g.append((t["fd"] * t["vx"] - cfg.ENGINE_POWER) / 1e5)

    # Traction control: the drive force it lets through is 0.9 x the rear
    # grip not already spent cornering (budget uses mu, not the rear-biased
    # mu_r), cut further once the rear slips past TC_SLIP_DEG.
    cap_r = t["mu"] * t["fz_r"]
    budget = _smax(cap_r ** 2 - t["lat_r"] ** 2, 0.0, 2.0e6)
    slip_deg = ca.sqrt(t["a_r"] ** 2 + 1e-6) * (180.0 / math.pi)
    slide = (slip_deg - cfg.TC_SLIP_DEG) / (cfg.TC_SLIP_FULL_DEG
                                            - cfg.TC_SLIP_DEG)
    slide = _smin(_smax(slide, 0.0, 0.02), 1.0, 0.02)
    keep = 1.0 - slide * (1.0 - cfg.TC_MIN_POWER)
    g.append((t["fd"] ** 2 - (cfg.TC_SAFETY * keep) ** 2 * budget) / 1e8)

    # ABS: per axle, the brake force is held under 0.92 x max(spare grip,
    # floor share of the axle's grip). The floor uses the *smallest* share the
    # game ever grants, so the cap is one ABS always honours.
    cap_f = t["mu"] * t["fz_f"]
    room_f = _smax(cap_f ** 2 - t["lat_f"] ** 2,
                   (cfg.ABS_BRAKE_SHARE_MIN * cap_f) ** 2, 1.0e6)
    room_r = _smax(cap_r ** 2 - t["lat_r"] ** 2,
                   (cfg.ABS_BRAKE_SHARE_MIN * cap_r) ** 2, 1.0e6)
    g.append(((t["fb"] * BB / cfg.ABS_SAFETY) ** 2 - room_f) / 1e8)
    g.append(((t["fb"] * (1.0 - BB) / cfg.ABS_SAFETY) ** 2 - room_r) / 1e8)

    # Friction-circle domain: never ask an axle for more than it has.
    g.append(t["used_f"] - 0.99)
    g.append(t["used_r"] - 0.99)
    return g


def wheel_offsets(n, xi):
    """Lateral offsets of the four contact patches, (FR, FL, RR, RL)."""
    h = cfg.WHEEL_HALF_TRACK
    out = []
    for along in (LF, -LR):
        for side in (1.0, -1.0):
            out.append(n + along * ca.sin(xi) + side * h * ca.cos(xi))
    return out


# --------------------------------------------------------------------------
# the reference curve
# --------------------------------------------------------------------------
class RefCurve:
    """A lightly smoothed copy of the centreline, as the solver's frame.

    The raw samples carry kinks -- Monza's first chicane reads as a 9 m radius
    from one sample to the next -- and the curvilinear transform divides by
    ``1 - n * kappa``, which a phantom kink pushes towards zero. So the frame
    is the centreline smoothed over a few samples, and the white lines are
    carried across into it exactly: ``e`` is where the raw centreline sits in
    the smoothed frame, and the edges are ``e +- width``.
    """

    def __init__(self, track, sigma_m: float = 7.0):
        c = track.center
        n = track.count
        step = float(np.median(track.seg_len))
        k = max(1, int(round(3.0 * sigma_m / step)))
        off = np.arange(-k, k + 1)
        w = np.exp(-0.5 * (off * step / sigma_m) ** 2)
        w /= w.sum()
        sm = np.zeros_like(c)
        for o, wi in zip(off, w):
            sm += wi * np.roll(c, -o, axis=0)
        self.center = sm
        nxt = np.roll(sm, -1, axis=0)
        prv = np.roll(sm, 1, axis=0)
        tan = nxt - prv
        tan /= np.linalg.norm(tan, axis=1, keepdims=True)
        self.tangent = tan
        self.normal = np.stack([tan[:, 1], -tan[:, 0]], axis=1)
        self.theta = np.arctan2(tan[:, 0], tan[:, 1])
        self.h = np.linalg.norm(nxt - sm, axis=1)            # interval lengths
        dth = (np.roll(self.theta, -1) - self.theta + math.pi) % (2 * math.pi) - math.pi
        self.kappa = dth / self.h                              # per interval
        self.s = np.concatenate([[0.0], np.cumsum(self.h)[:-1]])
        self.length = float(self.h.sum())
        # Raw centreline in this frame, and the white lines either side of it.
        self.e = np.einsum("ij,ij->i", track.center - sm, self.normal)
        self.edge_r = self.e + track.w_right
        self.edge_l = self.e - track.w_left
        self.count = n

    def xy(self, n_off):
        return self.center + self.normal * np.asarray(n_off)[:, None]


# --------------------------------------------------------------------------
# collocation
# --------------------------------------------------------------------------
def _collocation(d: int = 3):
    tau = np.append(0.0, ca.collocation_points(d, "legendre"))
    C = np.zeros((d + 1, d + 1))
    D = np.zeros(d + 1)
    B = np.zeros(d + 1)
    for j in range(d + 1):
        p = np.poly1d([1.0])
        for r in range(d + 1):
            if r != j:
                p *= np.poly1d([1.0, -tau[r]]) / (tau[j] - tau[r])
        D[j] = p(1.0)
        dp = np.polyder(p)
        for r in range(d + 1):
            C[j, r] = dp(tau[r])
        B[j] = np.polyint(p)(1.0)
    return tau, C, D, B


def spatial_rhs(x, u, kappa, mu_scale=1.0):
    """d(state)/ds, plus dt/ds."""
    t = car_terms(x, u, mu_scale)
    n, xi, vx, vy = t["n"], t["xi"], t["vx"], t["vy"]
    s_dot = (vx * ca.cos(xi) - vy * ca.sin(xi)) / (1.0 - n * kappa)
    dn = vx * ca.sin(xi) + vy * ca.cos(xi)
    dxi = t["r"] - kappa * s_dot
    dts = 1.0 / s_dot
    f = ca.vertcat(dn, dxi, t["dvx"], t["dvy"], t["dr"], t["dd"]) * dts
    return f, dts


def solve(track, kerb: bool = False, sigma_m: float = 7.0,
          wall_margin: float = 0.05, init=None, max_iter: int = 3000,
          reg_steer: float = 2e-4, reg_force: float = 2e-3,
          reg_both: float = 2e-2, verbose: bool = True, log_file=None,
          ipopt_opts=None, grip: float = 1.0):
    """``grip`` is the share of the tyre's friction the plan may use.

    1.0 is the true optimum, with every corner taken exactly at the limit --
    which leaves a driver no grip at all to correct the smallest error, and
    the chicanes are where that bites first. A plan at, say, 0.97 is a few
    tenths slower and leaves the tracker 3% of the tyre to steer with.
    """
    ref = RefCurve(track, sigma_m)
    N = ref.count
    nx, nu = 6, 3
    d = 3
    tau, C, D, B = _collocation(d)
    xs, us = ca.DM(XS), ca.DM(US)

    # How far past the white line a wheel may go. Without the kerb every
    # wheel stays on the asphalt. With it the outside pair may ride the kerb
    # -- legal, since the game's rule is that ANY wheel inside the line keeps
    # the car on the track -- at the kerb's grip, but never onto the grass.
    reach = cfg.KERB_WIDTH - wall_margin if kerb else -wall_margin

    def grip_scale(n, xi, white_l, white_r):
        """Surface.grip's mean over the four patches, smoothed at the line,
        times the share of it the plan is allowed."""
        if not kerb:
            return grip
        loss = 1.0 - cfg.KERB_GRIP_SCALE
        sharp = 0.08                        # metres over which a patch crosses
        g = 0
        for w in wheel_offsets(n, xi):
            over = (1.0 / (1.0 + ca.exp(-(w - white_r) / sharp))
                    + 1.0 / (1.0 + ca.exp(-(white_l - w) / sharp)))
            g = g + 1.0 - loss * over
        return grip * g / 4.0

    # --- one interval, written once and mapped over the lap ---------------
    Xk = ca.MX.sym("Xk", nx)
    Xc = ca.MX.sym("Xc", nx, d)
    Uk = ca.MX.sym("Uk", nu)
    kap = ca.MX.sym("kap")
    h = ca.MX.sym("h")
    wl = ca.MX.sym("wl")                    # white lines, in the ref frame
    wr = ca.MX.sym("wr")
    res = []
    dt = 0
    for j in range(1, d + 1):
        xp = C[0, j] * Xk
        for r in range(d):
            xp = xp + C[r + 1, j] * Xc[:, r]
        xj = Xc[:, j - 1] * xs
        f, dts = spatial_rhs(xj, Uk * us, kap,
                             grip_scale(xj[0], xj[1], wl, wr))
        res.append(h * f / xs - xp)
        dt = dt + B[j] * h * dts
    xend = D[0] * Xk
    for r in range(d):
        xend = xend + D[r + 1] * Xc[:, r]
    # Expanded to scalar (SX) form per interval: the exact Hessian is the
    # per-iteration cost and this makes it several times cheaper, while the
    # lap-sized problem stays a light MX graph of N mapped calls. (Expanding
    # the WHOLE problem instead blew the process's memory and starved MUMPS.)
    F = ca.Function("interval", [Xk, Xc, Uk, kap, h, wl, wr],
                    [ca.vertcat(*res), xend, dt]).expand()
    Fm = F.map(N, "thread", 8)

    # --- path limits at every node ----------------------------------------
    Xn = ca.MX.sym("Xn", nx)
    Un = ca.MX.sym("Un", nu)
    xn = Xn * xs
    tn = car_terms(xn, Un * us, grip_scale(xn[0], xn[1], wl, wr))
    gp = limits(tn)
    for w in wheel_offsets(tn["n"], tn["xi"]):
        gp.append(w - (wr + reach))
        gp.append((wl - reach) - w)
    P = ca.Function("path", [Xn, Un, wl, wr], [ca.vertcat(*gp)]).expand()
    Pm = P.map(N, "thread", 8)

    hi_v = ref.edge_r + reach
    lo_v = ref.edge_l - reach
    wl_dm = ca.DM(ref.edge_l).T
    wr_dm = ca.DM(ref.edge_r).T

    opti = ca.Opti()
    X = opti.variable(nx, N)
    XC = opti.variable(nx, N * d)
    U = opti.variable(nu, N)
    Xnext = ca.horzcat(X[:, 1:], X[:, 0])        # periodic: the lap closes

    kap_dm = ca.DM(ref.kappa).T
    h_dm = ca.DM(ref.h).T
    # The mapped call takes interval k's collocation states as columns
    # k*d .. k*d+d-1 of XC.
    resid, xend, dts = Fm(X, XC, U, kap_dm, h_dm, wl_dm, wr_dm)
    opti.subject_to(ca.vec(resid) == 0)
    opti.subject_to(ca.vec(xend - Xnext) == 0)
    g_path = Pm(X, U, wl_dm, wr_dm)
    opti.subject_to(ca.vec(g_path) <= 0)

    # simple bounds
    opti.subject_to(opti.bounded(-1.2 / XS[1], X[1, :], 1.2 / XS[1]))
    opti.subject_to(opti.bounded(5.0 / XS[2], X[2, :], 120.0 / XS[2]))
    opti.subject_to(opti.bounded(-cfg.MAX_STEER / XS[5], X[5, :],
                                 cfg.MAX_STEER / XS[5]))
    opti.subject_to(opti.bounded(-cfg.STEER_RACK_RATE / US[0], U[0, :],
                                 cfg.STEER_RACK_RATE / US[0]))
    opti.subject_to(opti.bounded(0.0, U[1, :], cfg.ENGINE_FORCE_MAX / US[1]))
    opti.subject_to(opti.bounded(0.0, U[2, :], cfg.BRAKE_FORCE / US[2]))

    lap = ca.sum2(dts)
    Unext = ca.horzcat(U[:, 1:], U[:, 0])
    dU = Unext - U
    reg = (reg_steer * ca.sum2(U[0, :] ** 2 * h_dm)
           + reg_force * ca.sumsqr(dU[1:, :])
           + reg_both * ca.sum2(U[1, :] * U[2, :]))
    opti.minimize(lap + reg)

    # --- initial guess ------------------------------------------------------
    x0, u0 = init if init is not None else initial_guess(track, ref, hi_v, lo_v)
    opti.set_initial(X, x0 / XS[:, None])
    opti.set_initial(XC, np.repeat(x0, d, axis=1) / XS[:, None])
    opti.set_initial(U, u0 / US[:, None])

    # A NaN anywhere in the starting point sends IPOPT straight into a
    # restoration phase it cannot leave; say where instead.
    for name, expr in (("dynamics", ca.vec(resid)), ("limits", ca.vec(g_path)),
                       ("dt", ca.vec(dts))):
        val = np.asarray(opti.value(expr, opti.initial())).ravel()
        bad = ~np.isfinite(val)
        if bad.any():
            print(f"initial guess: {bad.sum()} non-finite {name} entries "
                  f"(first at {np.flatnonzero(bad)[:8]})")
        elif verbose:
            print(f"initial guess: {name} max |.| {np.abs(val).max():.3g}")

    opts = {"ipopt.max_iter": max_iter, "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-4, "ipopt.mu_strategy": "adaptive",
            # MUMPS reserves 10x its own estimate up front by default, and on
            # this laptop that one large allocation fails intermittently
            # ("out of memory" with gigabytes free). Starting smaller lets
            # IPOPT grow it only if a factorisation actually needs it.
            "ipopt.mumps_mem_percent": 300,
            # IPOPT's own console output is C-buffered and is lost if the
            # process dies; a file is written as it goes.
            **({"ipopt.output_file": str(log_file),
                "ipopt.file_print_level": 5} if log_file else {}),
            "ipopt.print_level": 5 if verbose else 0,
            "ipopt.print_frequency_iter": 10, "print_time": verbose}
    opts.update(ipopt_opts or {})
    opti.solver("ipopt", opts)
    t0 = time.perf_counter()
    try:
        sol = opti.solve()
        status = "solved"
        get = sol.value
    except RuntimeError as exc:                   # max_iter / infeasible
        status = f"not converged: {exc}"
        get = opti.debug.value
    wall = time.perf_counter() - t0
    xv = np.asarray(get(X)) * XS[:, None]
    uv = np.asarray(get(U)) * US[:, None]
    dtv = np.asarray(get(dts)).ravel()
    return dict(ref=ref, x=xv, u=uv, dt=dtv, lap=float(dtv.sum()),
                status=status, wall=wall, lo=lo_v, hi=hi_v)


def initial_guess(track, ref, hi_v, lo_v):
    """Min-curvature line inside the limits, at the analytic speed profile."""
    off = raceline.min_curvature(track)
    # min_curvature's offsets are on the raw centreline; carry them into the
    # smoothed frame and pull them inside the all-wheels-on-asphalt corridor.
    n0 = off + ref.e
    room = cfg.WHEEL_HALF_TRACK + 0.3
    n0 = np.clip(n0, lo_v + room, hi_v - room)
    pts = ref.xy(n0)
    seg, kap, rad = raceline.line_geometry(pts)
    v0 = raceline.speed_profile(seg, kap, rad, 1.0)
    d = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    heading = np.arctan2(d[:, 0], d[:, 1])
    xi0 = (heading - ref.theta + math.pi) % (2 * math.pi) - math.pi
    r0 = kap * v0
    delta0 = L * kap
    acc = (np.roll(v0, -1) ** 2 - v0 ** 2) / (2.0 * np.maximum(seg, 1e-3))
    resist = cfg.DRAG_COEFF * v0 ** 2 + cfg.ROLL_RESIST * v0
    fd0 = np.clip(M * acc + resist, 0.0, cfg.ENGINE_FORCE_MAX)
    fb0 = np.clip(-(M * acc + resist), 0.0, cfg.BRAKE_FORCE)
    x0 = np.vstack([n0, xi0, v0, np.zeros_like(v0), r0, delta0])
    u0 = np.vstack([np.zeros_like(v0), fd0, fb0])
    return x0, u0


def throttle_for(fd: float, vx: float, fz_r: float, lat_r: float,
                 slip_r: float) -> float:
    """The pedal that makes the game's engine deliver *fd* newtons.

    The game scales the throttle onto whichever ceiling is lowest -- engine
    force, power, or what traction control lets through -- so the same force
    takes more pedal when TC is holding the ceiling down.
    """
    if fd <= 0.0:
        return 0.0
    ceiling = cfg.ENGINE_FORCE_MAX
    if vx > 1.0:
        ceiling = min(ceiling, cfg.ENGINE_POWER / vx)
    if cfg.TRACTION_CONTROL:
        traction = MU * fz_r
        tc = cfg.TC_SAFETY * math.sqrt(max(traction * traction - lat_r * lat_r, 0.0))
        slide = (abs(math.degrees(slip_r)) - cfg.TC_SLIP_DEG) / max(
            cfg.TC_SLIP_FULL_DEG - cfg.TC_SLIP_DEG, 1e-6)
        tc *= 1.0 - min(max(slide, 0.0), 1.0) * (1.0 - cfg.TC_MIN_POWER)
        ceiling = min(ceiling, tc)
    return min(fd / max(ceiling, 1e-6), 1.0)


# --------------------------------------------------------------------------
# model check against the game's integrator
# --------------------------------------------------------------------------
def check_model(circuit: str = "Monza"):
    """Compare d(vx, vy, r)/dt from the equations above with Vehicle.step.

    The game car is put into exactly the state, primed with the per-axle
    longitudinal forces and acceleration it would have carried in from the
    previous step, and stepped by a tiny dt; the finite difference is its
    derivative. Steering is held at delta (analog input = delta / lock).
    """
    from game.surface import Surface
    from game.vehicle import Controls

    track = load_track(circuit)
    surf = Surface(track)
    xs_ = ca.MX.sym("x", 6)
    us_ = ca.MX.sym("u", 3)
    tt = car_terms(xs_, us_)
    f = ca.Function("f", [xs_, us_], [ca.vertcat(tt["dvx"], tt["dvy"], tt["dr"],
                                                 tt["ax"], tt["a_f"], tt["a_r"],
                                                 tt["fz_r"], tt["lat_r"])])
    lim_f = ca.Function("lim", [xs_, us_], [ca.vertcat(*limits(tt))])
    rng = np.random.default_rng(1)
    i = 40                                        # a straight, on the asphalt
    worst = np.zeros(3)
    rows = 0
    for trial in range(400):
        vx = rng.uniform(15.0, 85.0)
        vy = rng.uniform(-2.0, 2.0)
        r = rng.uniform(-0.8, 0.8)
        delta = rng.uniform(-0.08, 0.08)
        mode = rng.integers(3)
        fd = rng.uniform(0.0, 8000.0) if mode == 0 else 0.0
        fb = rng.uniform(500.0, 18000.0) if mode == 1 else 0.0
        x = np.array([0.0, 0.0, vx, vy, r, delta])
        u = np.array([0.0, fd, fb])
        if np.any(np.asarray(lim_f(x, u)).ravel() > -1e-3):
            continue                              # an aid would intervene
        out = np.asarray(f(x, u)).ravel()
        if abs(out[4]) > 0.2 or abs(out[5]) > 0.2:
            continue                              # past the region the assist allows

        v = Vehicle()
        v.frozen = False
        yaw = math.atan2(track.tangent[i, 0], track.tangent[i, 1])
        v.place(track.center[i], yaw)
        fwd = np.array([math.sin(yaw), math.cos(yaw)])
        right = np.array([math.cos(yaw), -math.sin(yaw)])
        v.vel = fwd * vx + right * vy
        v.yaw_rate = r
        v.steer_angle = delta
        v._long_accel = out[3]
        v._f_long_front = fb * BB
        v._f_long_rear = fb * (1.0 - BB) + fd
        v.tele.slip_front = out[4]
        surf.hint = i
        speed = math.hypot(vx, vy)
        lock = v.steer_limit(speed, MU)
        ctl = Controls(throttle=throttle_for(fd, vx, out[6], out[7], out[5]),
                       brake=fb / cfg.BRAKE_FORCE,
                       steer=delta / lock, analog_steer=True)
        # Small enough that the game's own step order (yaw advanced with the
        # NEW yaw rate) adds nothing measurable to the lateral derivative.
        dt = 1e-6
        v.step(ctl, dt, surf)
        # Body-frame rates: the velocity read in the car's NEW frame against
        # the old, which is what vx_dot = a_long + vy r describes.
        fwd1 = np.array([math.sin(v.yaw), math.cos(v.yaw)])
        right1 = np.array([math.cos(v.yaw), -math.sin(v.yaw)])
        game = np.array([(float(np.dot(v.vel, fwd1)) - vx) / dt,
                         (float(np.dot(v.vel, right1)) - vy) / dt,
                         (v.yaw_rate - r) / dt])
        mod = out[:3].copy()
        # Idle drag in the game is all-or-nothing on the throttle; the model
        # phases it in over a few hundred N of brake. Compare like with like.
        if fd == 0.0:
            mod[0] -= cfg.IDLE_DRAG * (1.0 - math.tanh(fb / IDLE_SOFT)) / M
        err = np.abs(game - mod) / (np.abs(game) + np.array([1.0, 1.0, 0.2]))
        worst = np.maximum(worst, err)
        rows += 1
        if trial < 6 or err.max() > 0.02:
            print(f"vx {vx:5.1f} vy {vy:+5.2f} r {r:+5.2f} d {delta:+.3f} "
                  f"Fd {fd:6.0f} Fb {fb:6.0f} | game {game.round(3)} "
                  f"model {mod.round(3)} | rel err {err.round(4)}")
    print(f"{rows} states compared; worst relative error "
          f"(dvx, dvy, dr) = {worst.round(4)}")


# --------------------------------------------------------------------------
def save(track, res, path: Path):
    ref = res["ref"]
    x, u = res["x"], res["u"]
    xy = ref.xy(x[0])
    yaw = ref.theta + x[1]
    t = np.concatenate([[0.0], np.cumsum(res["dt"])[:-1]])
    # Offsets relative to the RAW centreline, which is what the game's
    # Surface and every other tool measure against.
    n_raw = x[0] - ref.e
    np.savez(path, circuit=track.name, lap_time=res["lap"], s=ref.s,
             t=t, xy=xy, yaw=yaw, n_raw=n_raw,
             n=x[0], xi=x[1], vx=x[2], vy=x[3], r=x[4], delta=x[5],
             ddelta=u[0], Fd=u[1], Fb=u[2], kappa_ref=ref.kappa,
             ref_center=ref.center, ref_normal=ref.normal)


def plot(track, res, path: Path, compare=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    ref = res["ref"]
    x = res["x"]
    xy = ref.xy(x[0])
    v = np.hypot(x[2], x[3]) * 3.6
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(1, 2, 1)
    left = track.center - track.normal * track.w_left[:, None]
    right = track.center + track.normal * track.w_right[:, None]
    for e in (left, right):
        ax.plot(*np.vstack([e, e[:1]]).T, color="0.6", lw=0.6)
    seg = np.stack([xy, np.roll(xy, -1, axis=0)], axis=1)
    lc = LineCollection(seg, cmap="turbo", linewidths=1.6)
    lc.set_array(v)
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, label="km/h", shrink=0.6)
    ax.set_aspect("equal")
    ax.set_title(f"{track.name}: minimum-time lap {res['lap']:.2f} s")
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(ref.s, v, lw=1.0, label=f"min-time {res['lap']:.2f} s")
    if compare is not None:
        ax2.plot(compare["s"], compare["v"], lw=0.8, alpha=0.8,
                 label=compare["label"])
    ax2.set_ylabel("km/h")
    ax2.legend()
    ax3 = fig.add_subplot(2, 2, 4, sharex=ax2)
    ax3.plot(ref.s, x[0] - ref.e, lw=1.0, label="min-time offset")
    ax3.plot(ref.s, track.w_right, color="0.5", lw=0.6)
    ax3.plot(ref.s, -track.w_left, color="0.5", lw=0.6)
    if compare is not None:
        ax3.plot(compare["s"], compare["n"], lw=0.8, alpha=0.8,
                 label=compare["label"])
    ax3.set_ylabel("offset from centre (m, +right)")
    ax3.set_xlabel("distance (m)")
    ax3.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


class _LapClock:
    """Flying-lap timing the way RaceEnv arms it: half a lap, then the line."""

    def __init__(self, n):
        self.n = n
        self.armed = False
        self.start = None
        self.laps = []

    def tick(self, i: int, t: float) -> bool:
        if 0.4 * self.n <= i <= 0.6 * self.n:
            self.armed = True
        elif self.armed and i < 0.1 * self.n:
            self.armed = False
            if self.start is not None:
                self.laps.append(t - self.start)
            self.start = t
            return True
        return False


def drive(track, laps: int = 3, path=None, dt: float = 1.0 / 60.0):
    """Drive the solved lap in the game's own physics, from a standing start.

    Returns the flying laps and, for the last one, a per-tick trace.
    """
    from game.mintime_driver import MinTimeDriver
    from game.surface import Surface

    surf = Surface(track)
    drv = MinTimeDriver(track, surf, path)
    v = Vehicle()
    v.frozen = False
    pos, yaw = track.start_pose()
    v.place(pos, yaw)
    clock = _LapClock(track.count)
    t = 0.0
    trace = []
    lap_trace = []
    off_ticks = kerb_ticks = 0
    off_lap, worst_lap = [], []
    worst = 0.0
    while len(clock.laps) < laps and t < 200.0 * (laps + 1):
        ctl = drv.controls(v)
        v.step(ctl, dt, surf)
        t += dt
        i, off = surf.progress(v.pos)
        if clock.tick(i, t):
            if lap_trace:
                trace = lap_trace
                off_lap.append(off_ticks)
                worst_lap.append(worst)
            lap_trace = []
            off_ticks = kerb_ticks = 0
            worst = 0.0
        if not v.on_track:
            off_ticks += 1
        if v.grip_scale < 1.0:
            kerb_ticks += 1
        worst = max(worst, abs(drv.lat_err))
        lap_trace.append((track.arclen[i], v.speed * 3.6, off, drv.lat_err,
                          ctl.throttle, ctl.brake, v.on_track, v.grip_scale))
    # Entry 0 of the per-lap lists is the standing-start out-lap.
    return dict(laps=clock.laps, off=off_lap[1:], worst_lat=worst_lap[1:],
                trace=np.array(trace, dtype=float))


def rl_flying_lap(track):
    """The deployed RL policy's flying lap, traced the same way as drive()."""
    from game import rlpolicy
    from game.rlenv import RaceEnv

    env = RaceEnv(track.name, randomise_start=False)
    env.episode_seconds = 400.0
    data = np.load(rlpolicy.policy_path(track.name))
    w = {k: data[k] for k in data.files if "." in k}
    mean, std = data["obs_mean"], data["obs_std"]
    qemb = rlpolicy.iqn_embedding(w)
    obs = env.reset_grid(0.0)
    clock = _LapClock(track.count)
    t = 0.0
    rows, cur = [], []
    while len(clock.laps) < 2 and t < 400.0:
        q = rlpolicy.iqn_q(w, rlpolicy.normalise(obs[None], mean, std),
                           qemb=qemb)[0]
        a = int(np.argmax(q))
        obs, _, done, info = env.step(a)
        t = env.lap_time
        i = env._index()
        v = env.vehicle
        off = float(np.dot(v.pos - track.center[i], track.normal[i]))
        if clock.tick(i, t):
            if clock.start is not None and cur:
                rows = cur
            cur = []
        pedal = rlpolicy.ACTIONS[a][1]
        cur.append((track.arclen[i], v.speed * 3.6, off, 0.0,
                    max(pedal, 0.0), max(-pedal, 0.0), v.on_track, 1.0))
        if done:
            break
    return dict(laps=clock.laps, trace=np.array(rows, dtype=float),
                best_lap=env.best_lap_time, off_steps=env.off_steps)


def compare_plot(track, res_path: Path, drove, rl, out: Path):
    """Speed and line, solved vs driven vs RL, plus the two chicanes close up."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sol = np.load(res_path)
    s = track.arclen
    v_sol = np.hypot(sol["vx"], sol["vy"]) * 3.6
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(s, v_sol, lw=1.2, label=f"solved {float(sol['lap_time']):.2f} s")
    if drove is not None and len(drove["trace"]):
        tr = drove["trace"]
        ax.plot(tr[:, 0], tr[:, 1], lw=0.8,
                label=f"solved, driven in game {drove['laps'][-1]:.2f} s")
    if rl is not None and len(rl["trace"]):
        tr = rl["trace"]
        lap = rl["laps"][-1] if rl["laps"] else float("nan")
        ax.plot(tr[:, 0], tr[:, 1], lw=0.8, alpha=0.8,
                label=f"RL (deployed) {lap:.2f} s")
    ax.set_xlim(0, track.length)
    ax.set_ylabel("km/h")
    ax.legend(loc="lower right")
    ax.set_title(f"{track.name}: speed along the lap")

    ax = fig.add_subplot(gs[1, :], sharex=fig.axes[0])
    ax.plot(s, sol["n_raw"], lw=1.2, label="solved")
    if rl is not None and len(rl["trace"]):
        ax.plot(rl["trace"][:, 0], rl["trace"][:, 2], lw=0.8, alpha=0.8,
                label="RL")
    ax.plot(s, track.w_right, color="0.5", lw=0.6)
    ax.plot(s, -track.w_left, color="0.5", lw=0.6)
    ax.set_ylabel("offset (m, +right)")
    ax.set_xlabel("distance (m)")
    ax.legend(loc="lower right")

    # The tightest direction changes: where kappa swings hardest in 60 m.
    kap = track.curvature
    win = max(3, int(60.0 / float(np.median(track.seg_len))))
    swing = np.array([np.ptp(np.take(kap, range(i, i + win), mode="wrap"))
                      for i in range(track.count)])
    picks = []
    for i in np.argsort(-swing):
        if all(min(abs(i - p), track.count - abs(i - p)) > 3 * win for p in picks):
            picks.append(int(i))
        if len(picks) == 3:
            break
    left = track.center - track.normal * track.w_left[:, None]
    right = track.center + track.normal * track.w_right[:, None]
    for c, i0 in enumerate(sorted(picks)):
        ax = fig.add_subplot(gs[2, c])
        idx = np.arange(i0 - 2 * win, i0 + 3 * win) % track.count
        ax.plot(*left[idx].T, color="0.4", lw=0.8)
        ax.plot(*right[idx].T, color="0.4", lw=0.8)
        ax.plot(*sol["xy"][idx].T, lw=1.6, label="solved")
        if rl is not None and len(rl["trace"]):
            tr = rl["trace"]
            sel = np.isin(np.searchsorted(s, tr[:, 0]), idx)
            pts = (track.center[np.searchsorted(s, tr[sel, 0])]
                   + track.normal[np.searchsorted(s, tr[sel, 0])]
                   * tr[sel, 2][:, None])
            ax.plot(*pts.T, lw=1.0, alpha=0.8, label="RL")
        ax.set_aspect("equal")
        ax.set_title(f"around {s[i0]:.0f} m")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--circuit", default="Monza")
    ap.add_argument("--check", action="store_true",
                    help="compare the model with Vehicle.step and exit")
    ap.add_argument("--kerb", action="store_true")
    ap.add_argument("--sigma", type=float, default=7.0,
                    help="metres the solver's reference frame is smoothed over")
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--log", default=None,
                    help="write IPOPT's iteration log to this file")
    ap.add_argument("--drive", action="store_true",
                    help="drive the saved solution in the game's physics, "
                         "compare with the deployed RL policy, and exit")
    ap.add_argument("--grip", type=float, default=1.0,
                    help="share of tyre grip the plan may use (1.0 = the "
                         "true limit; < 1 leaves the tracker room to correct)")
    ap.add_argument("--tag", default="",
                    help="suffix for the output file, e.g. g97")
    ap.add_argument("--warm", action="store_true",
                    help="start from the saved <Circuit>_mintime.npz instead "
                         "of the min-curvature line")
    ap.add_argument("--warm-from", default=None, metavar="TAG",
                    help="start from <Circuit>_mintime_<TAG>.npz -- a nearby "
                         "grip converges where the full-grip plan does not")
    args = ap.parse_args(argv)
    if args.check:
        check_model(args.circuit)
        return
    track = load_track(args.circuit)
    base = cfg.RACELINE_DIR / f"{track.name}_mintime.npz"
    suffix = ("_kerb" if args.kerb else "") + (f"_{args.tag}" if args.tag else "")
    out = base.with_name(f"{track.name}_mintime{suffix}.npz")
    if args.drive:
        sol = np.load(out)
        drove = drive(track, path=out)
        print(f"solved lap {float(sol['lap_time']):.3f} s; driven in game: "
              + ", ".join(f"{t:.3f}" for t in drove["laps"])
              + f" s; off-track ticks per flying lap {drove['off']}; "
              f"worst line error per lap "
              f"{[round(w, 2) for w in drove['worst_lat']]} m")
        rl = None
        from game import rlpolicy
        if rlpolicy.available(track.name):
            rl = rl_flying_lap(track)
            print(f"RL deployed: flying laps {[round(x, 3) for x in rl['laps']]}"
                  f" (best clean {rl['best_lap']:.3f} s, off steps "
                  f"{rl['off_steps']})")
        compare_plot(track, out, drove, rl,
                     out.with_name(out.stem + "_compare.png"))
        return
    init = None
    if args.warm or args.warm_from:
        w = np.load(base.with_name(f"{track.name}_mintime_{args.warm_from}.npz")
                    if args.warm_from else base)
        init = (np.vstack([w[k] for k in X_NAMES]),
                np.vstack([w[k] for k in ("ddelta", "Fd", "Fb")]))
    res = solve(track, kerb=args.kerb, sigma_m=args.sigma, init=init,
                max_iter=args.max_iter, log_file=args.log, grip=args.grip)
    print(f"{track.name}: {res['status']}  lap {res['lap']:.3f} s  "
          f"(solve {res['wall']:.0f} s)")
    if res["status"] != "solved":
        # Never let an unconverged iterate stand in for the answer.
        out = out.with_name(out.stem + "_failed.npz")
    save(track, res, out)
    plot(track, res, out.with_suffix(".png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
