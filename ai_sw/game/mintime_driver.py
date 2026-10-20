"""Drive the minimum-time lap from ``tools/mintime.py``.

The solver hands over, at every centreline sample, where the car should be,
which way it should point, how fast it should go, and the wheel angle and
pedal forces that got it there in the model. Those inputs played back blind
do not survive a lap: replayed with no feedback the car leaves the line at the
first chicane, ~10 s in -- and it does so even when the replay integrates the
solver's OWN equations, so it is not model error. A car on the limit is
unstable; any small difference grows until it is metres. So this is a
tracker: the solved inputs are the feed-forward, and a small feedback term
closes whatever gap opens between the planned and the actual car.

Feed it a plan solved a little under the grip limit (``mintime.py --grip
0.97``): a plan at exactly the limit leaves no tyre to correct with, and on
Monza it ran 5 m wide where the 0.97 plan stayed within 1.2 m of its line.

Three layers, so the race driver can reuse the lower two:

* ``Plan`` -- the solved lap as arrays, and where on it a point is.
* ``PlanFollower`` -- the tracker. Follows the plan's line, or the line
  shifted sideways by an offset (to pass, to make room), at the plan's speed
  or under a speed cap (to follow a slower car).
* ``MinTimeDriver`` -- the plan alone, no traffic: same ``controls(vehicle)``
  call as ``Autopilot`` and ``RLDriver``.
"""
from __future__ import annotations

import math

import numpy as np

from . import config, drivenet, raceai
from .vehicle import Controls, Vehicle


def path_file(circuit: str, tag: str = ""):
    suffix = f"_{tag}" if tag else ""
    return config.RACELINE_DIR / f"{circuit}_mintime{suffix}.npz"


def available(circuit: str, tag: str = "") -> bool:
    return path_file(circuit, tag).exists()


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def steer_per_curvature(speed: float) -> float:
    """Wheel angle per unit path curvature in steady cornering: the geometric
    L plus the slip the tyres must run (Autopilot's relation)."""
    cfg = config
    L, m = cfg.WHEELBASE, cfg.CAR_MASS
    df = cfg.DOWNFORCE_COEFF * speed * speed
    load_f = (cfg.CG_TO_REAR / L) * m * cfg.GRAVITY + df * cfg.DOWNFORCE_FRONT_BIAS
    load_r = (cfg.CG_TO_FRONT / L) * m * cfg.GRAVITY + df * (1 - cfg.DOWNFORCE_FRONT_BIAS)
    return L + (m * speed * speed / L) * (
        cfg.CG_TO_REAR / (cfg.CORNER_STIFFNESS_FRONT * load_f)
        - cfg.CG_TO_FRONT / (cfg.CORNER_STIFFNESS_REAR * load_r))


class Plan:
    """A solved lap, indexed by centreline sample (node k sits beside
    centreline sample k), with the geometry to find a car on it."""

    def __init__(self, path):
        data = np.load(path, allow_pickle=False)
        self.path = path
        self.xy = data["xy"]
        self.yaw = data["yaw"]
        self.v = np.hypot(data["vx"], data["vy"])
        self.r = data["r"]
        self.delta = data["delta"]
        self.fd = data["Fd"]
        self.fb = data["Fb"]
        #: Where the line sits across the road: metres right of the centreline.
        self.n_raw = data["n_raw"]
        self.lap_time = float(data["lap_time"])
        n = len(self.xy)
        d = np.roll(self.xy, -1, axis=0) - self.xy
        self.seg = np.hypot(d[:, 0], d[:, 1])
        self.tan = d / np.maximum(self.seg, 1e-9)[:, None]
        self.nrm = np.stack([self.tan[:, 1], -self.tan[:, 0]], axis=1)
        # Curvature of the line itself (+ = turning right, towards +nrm), from
        # its own segment headings rather than the car's yaw, which carries
        # the sideslip.
        head = np.arctan2(self.tan[:, 0], self.tan[:, 1])
        k = (np.roll(head, -1) - head + math.pi) % (2 * math.pi) - math.pi
        k /= np.maximum(self.seg, 1e-9)
        self.kappa = 0.25 * np.roll(k, 1) + 0.5 * k + 0.25 * np.roll(k, -1)
        self.n = n
        # Plain-list copies for the per-tick lookups: indexing a numpy array
        # one element at a time costs more than the arithmetic around it.
        self.L = {name: getattr(self, name).tolist()
                  for name in ("v", "r", "delta", "fd", "fb", "yaw", "kappa",
                               "n_raw", "seg")}
        self._xy_l = self.xy.tolist()
        self._tan_l = self.tan.tolist()

    def locate(self, pos, i: int) -> tuple[int, float]:
        """(node, fraction to the next) of the nearest plan point, searched
        around centreline sample *i*."""
        px, py = float(pos[0]), float(pos[1])
        xy, tan, seg = self._xy_l, self._tan_l, self.L["seg"]
        n = self.n
        best_d, best_k, best_f = math.inf, 0, 0.0
        for k in (i - 2, i - 1, i, i + 1, i + 2):
            k %= n
            x, y = xy[k]
            tx, ty = tan[k]
            sg = seg[k] if seg[k] > 1e-9 else 1e-9
            f = ((px - x) * tx + (py - y) * ty) / sg
            fc = 0.0 if f < 0.0 else 1.0 if f > 1.0 else f
            dx = px - (x + tx * sg * fc)
            dy = py - (y + ty * sg * fc)
            d = dx * dx + dy * dy
            if d < best_d:
                best_d, best_k, best_f = d, k, fc
        return best_k, best_f

    def at(self, arr, k: int, f: float, wrap_angle: bool = False):
        a, b = arr[k], arr[(k + 1) % self.n]
        if wrap_angle:
            return a + _wrap(b - a) * f
        return a + (b - a) * f

    def ahead(self, k: int, f: float, metres: float) -> tuple[int, float]:
        seg = self.L["seg"]
        rest = seg[k] * (1.0 - f)
        if metres <= rest:
            return k, f + metres / max(seg[k], 1e-9)
        metres -= rest
        k = (k + 1) % self.n
        while metres > seg[k]:
            metres -= seg[k]
            k = (k + 1) % self.n
        return k, metres / max(seg[k], 1e-9)


class PlanFollower:
    #: Seconds the planned wheel angle is read ahead of the car. Zero, and it
    #: has to be: the plan's wheel angle already moves no faster than the
    #: rack can (the solver's steering rate is bounded by STEER_RACK_RATE),
    #: so the game's wheel reaches it within the tick. Reading it 0.05 s early
    #: turned the car in early at every direction change -- 1-2 m of line
    #: error through each chicane and a 2 m run wide at Lesmo.
    STEER_PREVIEW = 0.0
    #: Steering feedback works on ONE error: where the car will be relative
    #: to the line a short time ahead if it holds its heading -- lateral and
    #: heading error combined in the ratio the speed calls for, instead of two
    #: gains that suit a straight and fight each other in a chicane. The
    #: correction is the curvature that closes that error over the same
    #: distance, turned into wheel angle with the car's own steady-state
    #: steer-per-curvature (slip included, as in Autopilot).
    LOOK_TIME = 0.45       # s of travel to the point the error is read at
    LOOK_MIN = 8.0         # m, floor on that distance at low speed
    K_YAW = 0.02           # rad of wheel per rad/s of yaw-rate error
    K_SPEED = 2.0          # (m/s^2) per (m/s) of speed error
    #: A plan near the limit has no grip spare to steer back onto the line
    #: with, so a growing line error is answered with speed as well: the
    #: target drops this many m/s per metre of predicted error past the
    #: dead band. Inside the band (normal tracking) it costs nothing.
    ERR_DEADBAND = 0.5     # m
    ERR_SLOW = 1.5         # (m/s) per m

    def __init__(self, plan: Plan):
        self.plan = plan
        #: Signed metres from the line being followed at the last call
        #: (+right), and the previewed error the feedback acted on.
        self.lat_err = 0.0
        self.err = 0.0
        #: The road (a ``trackdata`` track), for the network's view of the white
        #: lines; set by whoever builds the follower for racing.
        self.track = None
        #: A network that drives instead (``drivenet.DriveNet``) or, in training,
        #: an object with ``needs_teacher`` that is handed this follower's own
        #: answer to learn from. None: the hand-built follower below.
        self.drive = None
        self._drv_n = 0
        self._drv_a = (0.0, 0.0)
        #: For a network that also decides (onenet): the decision observation,
        #: rebuilt on a decision tick and held in between.
        self._dec_obs = np.zeros(raceai.OBS_DIM, np.float32)

    def controls(self, vehicle: Vehicle, k: int, f: float,
                 offset: float = 0.0, d_off: float = 0.0, dd_off: float = 0.0,
                 v_cap: float = math.inf, pace: float = 1.0,
                 hold_speed: bool = False, flat_out: bool = False,
                 v_min: float = 0.0, free=None) -> Controls:
        """Follow the line ``offset`` metres right of the plan.

        ``d_off``/``dd_off`` are the offset's first and second derivative
        along the line (a lane change in progress); ``v_cap`` a speed the car
        must not exceed (traffic, an offset line's tighter radius); ``pace``
        scales the plan's own speeds (a driver who commits less).
        ``hold_speed`` ignores the plan's speed and keeps the current one -- a
        missed braking point.

        ``free`` is what a *free* network is given instead of the above, which
        have had the traffic rules applied: ``(offset, d_off, dd_off, v_cap,
        percept)`` with the lane as it was asked for, only a yellow flag's
        speed limit, and ``percept()`` for the cars around (drivenet.perceive).
        Hand-built and rule-bound networks ignore it.
        """
        drv = self.drive
        if drv is not None and not getattr(drv, "free", False):
            free = None
        if drv is not None and not drv.needs_teacher:
            return self._net_controls(drv, vehicle, k, f, offset, d_off, dd_off, v_cap,
                                      pace, hold_speed, flat_out, v_min, free)
        cfg = config
        pl = self.plan
        Lp = pl.L
        px, py = float(vehicle.pos[0]), float(vehicle.pos[1])
        vx, vy = float(vehicle.vel[0]), float(vehicle.vel[1])
        speed = max(math.hypot(vx, vy), 1.0)

        x, y = pl._xy_l[k]
        tx, ty = pl._tan_l[k]
        nx, ny = ty, -tx                       # the plan's normal (+ = right)
        sg = Lp["seg"][k] * f
        lat = (px - (x + tx * sg + nx * offset)) * nx \
            + (py - (y + ty * sg + ny * offset)) * ny
        self.lat_err = lat
        yaw_ref = pl.at(Lp["yaw"], k, f, wrap_angle=True) + math.atan(d_off)
        head = _wrap(vehicle.yaw - yaw_ref)

        if self.STEER_PREVIEW:
            kp, fp = pl.ahead(k, f, speed * self.STEER_PREVIEW)
        else:
            kp, fp = k, f
        delta_ff = pl.at(Lp["delta"], kp, fp)
        kap = pl.at(Lp["kappa"], k, f)
        v_at = pl.at(Lp["v"], k, f)
        # The plan's wheel angle is for the plan's speed, and most of it at
        # speed is tyre slip, which goes as v^2. A car going slower than
        # planned (queueing behind another, recovering) needs less of it --
        # given the full angle it turns in too sharply and weaves.
        spc = steer_per_curvature(speed)
        delta_ff += kap * (spc - steer_per_curvature(v_at))
        if offset or dd_off:
            # A line beside the plan is the plan's corner at a different
            # radius, plus whatever bend a lane change adds.
            kap_off = kap / max(1.0 - kap * offset, 0.3) + dd_off
            delta_ff += (kap_off - kap) * spc
        r_ref = pl.at(Lp["r"], k, f)
        look = max(self.LOOK_MIN, self.LOOK_TIME * speed)
        err = lat + look * math.sin(head)
        self.err = err
        kappa_fix = -2.0 * err / (look * look)
        delta = (delta_ff + kappa_fix * spc
                 - self.K_YAW * (vehicle.yaw_rate - r_ref))
        lock = max(vehicle.steer_limit(speed, cfg.TYRE_GRIP), 1e-4)
        steer = min(max(delta / lock, -1.0), 1.0)

        fd = pl.at(Lp["fd"], k, f)
        v_plan = speed if hold_speed else v_at * pace
        if flat_out and math.isfinite(v_cap):
            # Where the plan is flat out its speed is the ENGINE's limit, not
            # the tyres': a car with less drag than the plan assumed (in a
            # tow, with DRS) may go faster -- up to ``v_cap``, which the race
            # driver sets to what it can still brake from in time.
            ceiling = min(cfg.ENGINE_FORCE_MAX, cfg.ENGINE_POWER / max(v_at, 1.0))
            if fd >= 0.95 * ceiling:
                v_plan = max(v_plan, v_cap)
        v_ref = min(v_plan, v_cap) - self.ERR_SLOW * max(
            abs(err) - self.ERR_DEADBAND, 0.0)
        # ...but never to a standstill when asked to keep moving: a car
        # recovering from metres off its line would otherwise slow for the
        # error until it stopped, and never close it.
        v_ref = max(v_ref, min(v_min, v_cap))
        # The plan's forces are the feed-forward while the car is asked to
        # drive the plan's speed profile -- scaled by pace squared, since the
        # profile's accelerations (v dv/ds) scale that way. Under a cap, or
        # holding speed, it is a plain speed loop with the resistance the car
        # must overcome to hold it. Dropping the feed-forward for ANY pace
        # other than exactly 1 sent cars into the first chicane braking late.
        plan_ff = (fd - pl.at(Lp["fb"], k, f)) * pace * pace
        hold = (cfg.DRAG_COEFF * vehicle.drag_scale * speed * speed
                + cfg.ROLL_RESIST * speed)
        if hold_speed:
            ff = hold
        elif v_cap >= v_plan:
            ff = plan_ff
        else:
            # Capped below the plan (a car ahead): never brake LESS than the
            # plan would here -- the car ahead is braking at least that hard
            # for the same corner, and a speed loop alone reacts a second late.
            ff = min(plan_ff, hold)
        force = ff + cfg.CAR_MASS * self.K_SPEED * (v_ref - speed)
        t = vehicle.tele
        if force < ff and t.load_front > 0.0:
            # Braking harder than the plan does here (a car ahead, a cap) may
            # use only the friction the tyres are not already spending on
            # turning: mid-corner that is very little, and taking more from
            # the circle than is there turns the car wide instead of slowing it.
            grip = cfg.TYRE_GRIP * (t.load_front + t.load_rear)
            lat = abs(t.lat_front * math.cos(vehicle.steer_angle) + t.lat_rear)
            spare = math.sqrt(max(grip * grip - lat * lat, 0.0))
            force = max(force, min(ff, -spare))
        if force >= 0.0:
            ceiling = cfg.ENGINE_FORCE_MAX
            if speed > 1.0:
                ceiling = min(ceiling,
                              cfg.ENGINE_POWER * vehicle.power_scale / speed)
            # Traction control caps the ceiling further mid-corner; reading
            # it back off the tyres the car just ran keeps the pedal honest.
            tc = cfg.TC_SAFETY * math.sqrt(max(
                (cfg.TYRE_GRIP * t.load_rear) ** 2 - t.lat_rear ** 2, 0.0))
            if t.load_rear > 0.0:
                ceiling = min(ceiling, max(tc, 1.0))
            ctl = Controls(throttle=min(force / ceiling, 1.0),
                           brake=0.0, steer=steer, analog_steer=True)
        else:
            ctl = Controls(throttle=0.0,
                           brake=min(-force / cfg.BRAKE_FORCE, 1.0),
                           steer=steer, analog_steer=True)
        if drv is not None:
            # Training: this follower's answer is the teacher's; the object
            # decides what is actually driven (its own, or this one).
            if self._drv_n <= 0:
                self._drv_a = drv.act(
                    self._features(vehicle, k, f, offset, d_off, dd_off, v_cap, pace,
                                   hold_speed, flat_out, v_min, free),
                    (ctl.steer, ctl.throttle - ctl.brake))
                self._drv_n = drivenet.REPEAT
            self._drv_n -= 1
            return drivenet.to_controls(self._drv_a, Controls)
        return ctl

    def _features(self, vehicle, k, f, offset, d_off, dd_off, v_cap, pace, hold_speed,
                  flat_out, v_min, free):
        """The network's input: from the orders as given, or -- for a free
        network -- from the lane as asked for and what is around."""
        percept = None
        if free is not None:
            offset, d_off, dd_off, v_cap, look = free
            percept = look()
        elif getattr(self.drive, "free", False):
            percept = drivenet.NO_CARS          # alone: nothing in view
        x = drivenet.features(self, vehicle, k, f, offset, d_off, dd_off, v_cap, pace,
                              hold_speed, flat_out, v_min, self._drv_a, percept)
        if getattr(self.drive, "one", False):
            x = np.concatenate([x, self._dec_obs])
        return x

    def _net_controls(self, drv, vehicle, k, f, offset, d_off, dd_off, v_cap, pace,
                      hold_speed, flat_out, v_min, free=None):
        """The network drives: a new decision every ``REPEAT`` ticks, held between."""
        if self._drv_n <= 0:
            self._drv_a = drv.act(self._features(vehicle, k, f, offset, d_off, dd_off,
                                                 v_cap, pace, hold_speed, flat_out, v_min,
                                                 free))
            self._drv_n = drivenet.REPEAT
        self._drv_n -= 1
        return drivenet.to_controls(self._drv_a, Controls)


class MinTimeDriver:
    """The plan alone, no traffic."""

    def __init__(self, track, surface, path=None, pace: float = 1.0):
        self.track = track
        self.surface = surface
        #: Multiplier on the plan's speeds (a difficulty level's driver).
        self.pace = pace
        self.plan = Plan(path or path_file(track.name))
        self.follower = PlanFollower(self.plan)
        self.lap_time = self.plan.lap_time
        self._k = 0

    @property
    def lat_err(self) -> float:
        return self.follower.lat_err

    def controls(self, vehicle: Vehicle) -> Controls:
        i, _ = self.surface.progress(vehicle.pos)
        k, f = self.plan.locate(vehicle.pos, i)
        self._k = k
        return self.follower.controls(vehicle, k, f, pace=self.pace)
