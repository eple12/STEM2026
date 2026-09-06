"""A simple physics-aware centreline driver.

This exists to exercise the vehicle model without a human at the keyboard, and
it is deliberately written against the same :class:`Controls` interface a real
AI driver will use later -- pure-pursuit steering plus a lookahead speed limit
derived from the grip available, which is the standard scripted-opponent
recipe.
"""
from __future__ import annotations

import math

import numpy as np

from . import config
from .trackdata import Track
from .vehicle import Controls, Vehicle


#: The planner's hand-picked constants, in one place so they can be searched
#: rather than guessed. Each was a reasonable number written while getting the
#: car round the lap at all; none of them was ever optimised, and together they
#: leave the car running at about 88% of its own speed profile.
#:
#: Names are multipliers or metres, not opaque weights, so a learned value can
#: be read back as a statement about the car -- "it brakes at 0.71 of the pad's
#: capability while turning" -- instead of a number nobody can sanity check.
TUNING = dict(
    corner_window=28.0,      # m of track an apex governs
    swing_penalty=14.0,      # how hard a curvature reversal is punished
    brake_sweep=0.62,        # share of the brake the profile plans with
    plan_decel=0.72,         # share of the brake the lookahead assumes
    brake_margin=12.0,       # m of "done braking" wanted before turn-in
    horizon_pad=40.0,        # m added to the braking lookahead
    aim_time=0.35,           # s ahead the curvature feedforward reads
    fb_heading=0.55,         # feedback: heading error
    fb_cross=0.40,           # feedback: cross-track error
    fb_cross_rate=0.85,      # feedback: cross-track rate
    brake_steer_bleed=0.75,  # how much lock bleeds the brake off
    throttle_steer_bleed=0.55,
)


class Autopilot:
    def __init__(self, track: Track, surface, pace: float = 0.88,
                 brake_margin: float | None = None, line=None, tuning=None,
                 speed_scale=None):
        self.track = track
        self.surface = surface
        # What the car aims at. The centreline is the fallback, not the plan:
        # a corner taken across the full width has a far larger radius than the
        # same corner down the middle, and this planner used to leave all of
        # that on the table. `line` supplies the same fields under the same
        # names, so pointing at it is a one-line swap rather than a second
        # geometry threaded through every calculation.
        self.geo = track if line is None else line
        self.pace = pace                    # 0..1, fraction of grip to use
        self.tune = dict(TUNING)
        if tuning:
            self.tune.update(tuning)
        if brake_margin is not None:
            self.tune["brake_margin"] = brake_margin
        self.brake_margin = self.tune["brake_margin"]
        self.v_profile = self._build_profile()
        # A learned per-sample multiplier on the planned speed. The analytic
        # profile is quasi-steady and deliberately cautious -- it takes the
        # tightest radius over a window and penalises curvature reversals --
        # which is right on average and wrong in particular places. This is
        # where the search says "here I will carry more" or "here I am braking
        # too early", one corner at a time, instead of through a single global
        # constant that has to suit every corner at once.
        if speed_scale is not None:
            self.v_profile = np.clip(self.v_profile * speed_scale,
                                     6.0, config.MAX_SPEED)

    # -- how fast can we go through a corner of radius R? -------------
    def _corner_speed(self, radius: float) -> float:
        """Highest speed that holds *radius*, limited by whichever axle lets go
        first and by the downforce available at that speed.

        Note this is not simply ``a = mu * (W + downforce) / m``. That figure
        assumes both axles reach their peak together, which only happens if the
        aero balance matches the static weight split. Ours is deliberately
        nose-light (42% front), so at speed the front runs out first and the
        real limit is set by the yaw balance:

            lat_f * lf = lat_r * lr   =>   a_max = mu * load_f * L / (m * lr)

        Using the optimistic figure makes the planner ask for ~5% more grip
        than exists -- enough to run wide out of every fast corner.
        """
        cfg = config
        mu = cfg.TYRE_GRIP * self.pace
        L, m, R = cfg.WHEELBASE, cfg.CAR_MASS, max(radius, 1.0)

        w_f = (cfg.CG_TO_REAR / L) * m * cfg.GRAVITY
        w_r = (cfg.CG_TO_FRONT / L) * m * cfg.GRAVITY
        cl_f = cfg.DOWNFORCE_COEFF * cfg.DOWNFORCE_FRONT_BIAS
        cl_r = cfg.DOWNFORCE_COEFF * (1.0 - cfg.DOWNFORCE_FRONT_BIAS)

        best = cfg.MAX_SPEED
        for w0, cl, arm in ((w_f, cl_f, cfg.CG_TO_REAR),
                            (w_r, cl_r, cfg.CG_TO_FRONT)):
            # v^2 / R = mu * (w0 + cl v^2) * L / (m * arm)
            k = mu * L / (m * arm)
            denom = 1.0 / R - k * cl
            if denom <= 0.0:      # downforce alone carries this corner
                continue
            best = min(best, math.sqrt(max(k * w0 / denom, 1.0)))
        return min(best, cfg.MAX_SPEED)

    # -- offline velocity profile --------------------------------------
    def _build_profile(self) -> np.ndarray:
        """Maximum speed at every point, respecting grip *and* the fact that
        you can only change speed so fast.

        Point-wise corner speeds are not enough: a chicane asks the car to
        reverse direction between two apexes, and the transient (yaw inertia,
        steering rack, slip build-up) costs grip that a static radius does not
        show. So the profile is smoothed over a transition window, penalised
        where curvature reverses, then swept backwards for braking and forwards
        for acceleration -- the standard racing-line velocity pass.
        """
        t = self.geo
        n = t.count
        cfg = config

        # widen each corner's influence so an apex governs its approach
        win = max(4, int(self.tune["corner_window"]
                         / max(float(np.median(t.seg_len)), 1e-3)))
        radius = t.curv_radius.copy()
        for k in range(1, win + 1):
            radius = np.minimum(radius, np.roll(t.curv_radius, k))
            radius = np.minimum(radius, np.roll(t.curv_radius, -k))

        v = np.array([self._corner_speed(float(r)) for r in radius])

        # chicane penalty: how much the curvature swings across the window
        swing = np.zeros(n)
        for k in range(1, win + 1):
            swing = np.maximum(swing, np.abs(np.roll(t.curvature, -k) - t.curvature))
        v *= 1.0 / (1.0 + self.tune["swing_penalty"] * swing)

        # backward pass: be slow enough to brake down to whatever comes next.
        # Only part of the brake is usable because the friction circle spends
        # the rest on turning.
        a_brake = self.tune["brake_sweep"] * cfg.BRAKE_FORCE / cfg.CAR_MASS
        for _ in range(2):                       # twice, to wrap the loop
            for i in range(n - 1, -1, -1):
                ds = float(t.seg_len[i])
                v[i] = min(v[i], math.sqrt(v[(i + 1) % n] ** 2 + 2 * a_brake * ds))

        # forward pass: you cannot gain speed faster than the engine allows
        a_drive = cfg.ENGINE_FORCE_MAX / cfg.CAR_MASS
        for _ in range(2):
            for i in range(n):
                ds = float(t.seg_len[i - 1])
                v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * a_drive * ds))

        return np.clip(v, 6.0, cfg.MAX_SPEED)

    def controls(self, v: Vehicle) -> Controls:
        t = self.geo
        n = t.count
        # The index still comes from the centreline -- the line is sampled at
        # the same indices, so both agree on where "here" is.
        i, _ = self.surface.progress(v.pos)
        speed = max(v.speed, 1.0)

        # --- steering: curvature feedforward + error feedback ----------
        # Pure pursuit alone cuts chicanes -- a lookahead long enough to be
        # stable at 250 km/h is longer than a 20 m radius corner, so the aim
        # point sits past the apex and the car drives straight over it.
        # Instead: feed forward the steering the corner actually needs, and use
        # feedback only to correct the error.
        seg = max(float(t.seg_len[i]), 1e-3)
        prev = (i + max(1, int(self.tune["aim_time"] * speed / seg))) % n
        kappa = float(t.curvature[prev])

        # steady-state angle for that curvature, including the slip the tyres
        # must run to make the force (the same relation as Vehicle.steer_limit)
        a_lat = kappa * speed * speed
        cfg = config
        L, m = cfg.WHEELBASE, cfg.CAR_MASS
        df = cfg.DOWNFORCE_COEFF * speed * speed
        load_f = (cfg.CG_TO_REAR / L) * m * cfg.GRAVITY + df * cfg.DOWNFORCE_FRONT_BIAS
        load_r = (cfg.CG_TO_FRONT / L) * m * cfg.GRAVITY + df * (1 - cfg.DOWNFORCE_FRONT_BIAS)
        delta_ff = L * kappa + (m * a_lat / L) * (
            cfg.CG_TO_REAR / (cfg.CORNER_STIFFNESS_FRONT * max(load_f, 1.0))
            - cfg.CG_TO_FRONT / (cfg.CORNER_STIFFNESS_REAR * max(load_r, 1.0)))

        # feedback: heading error, cross-track error, and its rate
        heading_err = (math.atan2(t.tangent[i, 0], t.tangent[i, 1]) - v.yaw
                       + math.pi) % (2 * math.pi) - math.pi
        cross = float(np.dot(v.pos - t.center[i], t.normal[i]))
        cross_rate = float(np.dot(v.vel, t.normal[i]))
        delta_fb = (self.tune["fb_heading"] * heading_err
                    - math.atan((self.tune["fb_cross"] * cross
                                 + self.tune["fb_cross_rate"] * cross_rate)
                                / max(speed, 12.0)))

        limit = max(v.steer_limit(speed, config.TYRE_GRIP), 1e-4)
        # Cap the correction so it can never swamp the feedforward. At 250 km/h
        # the whole usable lock is about 6 degrees, so an untamed cross-track
        # term saturates the steering on its own and the car weaves.
        delta_fb = float(np.clip(delta_fb, -0.6 * limit, 0.6 * limit))

        # Convert to stick travel using the *same* mapping the car applies,
        # then refuse to go past the grip-optimal angle -- the driver's stick
        # runs 1.35x beyond it, and lock past the peak only deepens understeer.
        ceiling = 1.0 / config.STEER_LIMIT_MARGIN
        steer = float(np.clip((delta_ff + delta_fb) / limit, -ceiling, ceiling))

        # --- brake for the tightest corner inside braking distance ----
        # Plan with less than the full brake so there is always headroom to
        # catch up if the estimate was optimistic.
        decel = self.tune["plan_decel"] * config.BRAKE_FORCE / config.CAR_MASS
        horizon_m = self.tune["horizon_pad"] + speed * speed / (2.0 * decel)
        seg = max(float(t.seg_len[i]), 1e-3)
        span = int(np.clip(horizon_m / seg, 6, n // 2))

        # Start at 0, not 2: the corner the car is *already in* constrains it
        # just as much as the one ahead.
        window = (i + np.arange(0, span)) % n
        dists = t.arclen[window] - t.arclen[i]
        dists = np.where(dists < 0.0, dists + t.length, dists)   # wrap the loop
        # Aim to be at corner speed slightly *before* turn-in. The friction
        # circle means brake pressure and cornering grip come out of the same
        # budget, so arriving still braking is arriving unable to steer.
        dists = np.maximum(dists - self.brake_margin, 0.0)

        # Speed we may carry *now* and still be slow enough at each point:
        #   v_allowed^2 = v_profile^2 + 2 * a * d
        allowed = np.sqrt(self.v_profile[window] ** 2 + 2.0 * decel * dists)
        v_target = float(min(allowed.min(), config.MAX_SPEED))

        if speed > v_target:
            # Commit. Feathering the brake here is what makes a planner miss
            # its own braking point: the profile assumed `decel`, so anything
            # less falls further behind every step.
            over = (speed - v_target) / max(2.0, 0.05 * speed)
            brake = float(np.clip(over, 0.40, 1.0))
            # ...but bleed it off as lock goes on, or the fronts have nothing
            # left to turn with.
            brake *= 1.0 - self.tune["brake_steer_bleed"] * min(
                1.0, abs(steer) / ceiling)
            return Controls(throttle=0.0, brake=brake, steer=steer,
                            analog_steer=True)

        throttle = 1.0 if speed < v_target * 0.98 else 0.30
        # ...and likewise don't stand on it mid-corner.
        throttle *= 1.0 - self.tune["throttle_steer_bleed"] * min(
            1.0, abs(steer) / ceiling)
        return Controls(throttle=throttle, brake=0.0, steer=steer,
                        analog_steer=True)
