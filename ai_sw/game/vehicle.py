"""Dynamic bicycle-model vehicle. Pure math, no rendering.

This is the two-axle *dynamic* single-track model, not the kinematic one:
lateral tyre forces are generated from slip angles and axle load, yaw is
driven by the resulting torque against the car's yaw inertia. That means the
car cannot change direction instantly -- grip has to build up, the nose takes
time to come round, and pushing past the friction limit understeers or slides.

References
----------
* Marco Monster, "Car Physics for Games" (the canonical write-up)
* spacejack/carphysics2d (a clean implementation of the same model)
* f1tenth_gym / CommonRoad single-track model (parameter magnitudes)

Convention: yaw = 0 points along +z; yaw increases turning towards +x, which
matches ``atan2(tangent.x, tangent.z)`` used by the track.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np

from . import config


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@functools.lru_cache(maxsize=8)
def _pacejka_b(c: float, e: float) -> float:
    """Value of ``B * peak_slip`` that puts the Magic Formula's peak exactly at
    the intended slip angle.

    The peak is where ``C * atan(phi(Ba)) == pi/2``, i.e. where the inner term
    ``phi(x) = (1-E)x + E atan(x)`` reaches ``tan(pi/2C)``. With E = 0 that
    inverts trivially, but for any other E it has to be solved -- assuming it
    does not is what produced a curve with no peak at all, rising forever.
    """
    target = math.tan(math.pi / (2.0 * c))
    if abs(e) < 1e-9:
        return target

    def phi(x: float) -> float:
        return (1.0 - e) * x + e * math.atan(x)

    lo, hi = 0.0, 1.0
    while phi(hi) < target and hi < 1e6:
        hi *= 2.0
    for _ in range(80):                      # bisection: robust, runs once
        mid = 0.5 * (lo + hi)
        if phi(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _fwd(yaw: float) -> np.ndarray:
    return np.array([math.sin(yaw), math.cos(yaw)])


def _right(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw), -math.sin(yaw)])


@dataclass
class Controls:
    throttle: float = 0.0   # 0..1
    brake: float = 0.0      # 0..1
    steer: float = 0.0      # -1..1 raw input (+1 = right)
    handbrake: bool = False
    # A keyboard is a binary device, so its steering is rate-limited into
    # something a hand could do. A wheel, a gamepad axis or an AI driver is
    # already analog and should not be filtered a second time -- doing so adds
    # half a second of lag and makes a controller chase its own tail.
    analog_steer: bool = False
    #: Back up: brake from a standstill is held to reverse. Without it the
    #: car parked by the low-speed cutoff below never gets going -- a plain
    #: brake held at rest must stay at rest (the grid, a stop).
    reverse: bool = False


@dataclass
class Telemetry:
    """Read-only per-step outputs for the HUD, effects and (later) the AI."""
    slip_front: float = 0.0
    slip_rear: float = 0.0
    load_front: float = 0.0
    load_rear: float = 0.0
    lat_front: float = 0.0
    lat_rear: float = 0.0
    saturated_front: bool = False
    saturated_rear: bool = False
    long_accel: float = 0.0
    lat_accel: float = 0.0
    engine_force: float = 0.0
    downforce: float = 0.0


@dataclass
class Vehicle:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    yaw: float = 0.0
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))  # world xz
    yaw_rate: float = 0.0

    steer_input: float = 0.0    # smoothed -1..1, the "virtual wheel"
    steer_angle: float = 0.0    # actual front wheel angle, radians
    _long_accel: float = 0.0    # kept between steps for weight transfer
    # longitudinal force carried per axle from the previous step, used for the
    # friction circle (this step's value isn't known until after the tyres)
    _f_long_front: float = 0.0
    _f_long_rear: float = 0.0

    on_track: bool = True
    grip_scale: float = 1.0
    hit_wall: bool = False
    frozen: bool = True
    tc_cut: float = 0.0         # 0..1, how hard traction control is cutting
    esc_cut: float = 0.0        # 0..1, how hard stability control is damping yaw
    traction_control: bool = config.TRACTION_CONTROL
    abs_enabled: bool = config.ABS_ENABLED
    steer_assist: bool = config.STEER_ASSIST
    #: Per-car multipliers for a field of different cars: engine power (a
    #: team's car), and aerodynamic drag (set each step by the race for a car
    #: in another's slipstream). 1.0 is the car every other tool assumes.
    power_scale: float = 1.0
    drag_scale: float = 1.0

    # State as it was before the most recent step, so the renderer can
    # interpolate. A fixed-step loop runs 3 or 4 times per frame depending on
    # where the accumulator lands, and drawing the raw state makes the car jump
    # by a different amount each frame -- visible judder even at a solid 60 fps.
    prev_pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    prev_yaw: float = 0.0

    tele: Telemetry = field(default_factory=Telemetry)

    # -- reset -----------------------------------------------------------
    def place(self, pos_xz, yaw: float) -> None:
        self.pos = np.asarray(pos_xz, dtype=float).copy()
        self.yaw = float(yaw)
        self.prev_pos = self.pos.copy()
        self.prev_yaw = self.yaw
        self.vel = np.zeros(2)
        #: What the road is doing under the car: its height above the plane,
        #: its camber angle, and that angle projected onto the car's own
        #: lateral axis. Read by the car body, the camera and the HUD; all
        #: three would otherwise have to find the nearest sample again.
        self.track = None
        self.surface_y = 0.0
        self.surface_bank = 0.0
        self.surface_roll = 0.0
        self.yaw_rate = 0.0
        self.steer_input = 0.0
        self.steer_angle = 0.0
        self._long_accel = 0.0
        self._lat_accel = 0.0
        self.hit_wall = False

    # -- convenience ---------------------------------------------------
    @property
    def speed(self) -> float:
        return math.hypot(float(self.vel[0]), float(self.vel[1]))

    @property
    def forward_speed(self) -> float:
        return (float(self.vel[0]) * math.sin(self.yaw)
                + float(self.vel[1]) * math.cos(self.yaw))

    @property
    def slip_angle(self) -> float:
        """Chassis sideslip -- angle between where it points and where it goes."""
        v = self.speed
        if v < 1.0:
            return 0.0
        lat = (float(self.vel[0]) * math.cos(self.yaw)
               - float(self.vel[1]) * math.sin(self.yaw))
        return math.atan2(lat, abs(self.forward_speed) + 1e-6)

    @property
    def long_accel(self) -> float:
        """Longitudinal acceleration, m/s^2 (car frame). Last physics tick."""
        return float(self._long_accel)

    @property
    def lat_accel(self) -> float:
        """Lateral acceleration, m/s^2 (car frame). Last physics tick."""
        return float(self._lat_accel)

    # -- tyre model -------------------------------------------------------
    @staticmethod
    def tyre_force(slip: float, stiffness: float, mu: float, load: float) -> float:
        """Lateral force from a slip angle, via Pacejka's Magic Formula.

            F = D sin(C atan(B a - E (B a - atan(B a))))

        B is chosen so the peak sits at ``mu / stiffness`` -- exactly where the
        older linear-and-clamp curve peaked -- so the shape changes but every
        formula that reasons about the peak keeps working.

        The difference that matters is past the peak: this returns *less* force
        as slip grows, instead of holding flat for ever. That is what gives a
        slide a distinct feel and makes the limit findable.
        """
        cfg = config
        cap = mu * max(load, 1.0)
        if not cfg.PACEJKA:
            return _clamp(-stiffness * slip, -mu, mu) * max(load, 1.0)

        peak_slip = mu / max(stiffness, 1e-6)
        c, e = cfg.PACEJKA_C, cfg.PACEJKA_E
        b = _pacejka_b(c, e) / max(peak_slip, 1e-6)
        ba = b * -slip                      # sign: force opposes the slip
        return cap * math.sin(c * math.atan(ba - e * (ba - math.atan(ba))))

    # -- friction circle -------------------------------------------------
    @staticmethod
    def _combined(f_long: float, mu: float, load: float) -> float:
        """Fraction of lateral grip left after spending some on accel/braking.

        A tyre has one friction budget, not two: the resultant of longitudinal
        and lateral force lives inside a circle of radius mu*Fz. Spend most of
        it stopping and there is little left to turn with -- which is why you
        cannot brake at full and take an apex at the same time, and why
        picking up the throttle mid-corner pushes the car wide.
        """
        cap = mu * max(load, 1.0)
        used = min(0.995, abs(f_long) / cap)
        return math.sqrt(max(0.0, 1.0 - used * used))

    # -- steering actuation ---------------------------------------------
    @staticmethod
    def steer_limit(speed: float, mu: float, margin: float | None = None) -> float:
        """Largest front-wheel angle the tyres can actually use at this speed.

        The steady-state cornering relation for a bicycle model is

            delta = L / R  +  (alpha_f - alpha_r)

        i.e. the geometric angle to follow the arc, *plus* the slip the tyres
        must run to generate the force. Only the first term shrinks as v^2, so
        at speed the slip term dominates -- leaving it out (as a naive L/R
        limit does) starves the car of steering and caps it around 1 g.

        Substituting the peak the axles can hold, a_max = mu * (W + downforce)
        / m, gives the angle past which the front simply washes out.
        """
        cfg = config
        if margin is None:
            # Speed-dependent margin: generous below the knee, gone by
            # STEER_MARGIN_FULL. Same smoothstep as the input rate, so the car
            # has one falloff rather than two that disagree.
            t = _clamp((speed - cfg.STEER_MARGIN_KNEE)
                       / max(cfg.STEER_MARGIN_FULL - cfg.STEER_MARGIN_KNEE, 1e-6),
                       0.0, 1.0)
            margin = cfg.STEER_LIMIT_MARGIN + cfg.STEER_MARGIN_LOW_EXTRA * (
                1.0 - t * t * (3.0 - 2.0 * t))
        if speed < cfg.STEER_LIMIT_FREE_SPEED:
            return cfg.MAX_STEER

        v2 = speed * speed
        downforce = cfg.DOWNFORCE_COEFF * v2
        load = cfg.CAR_MASS * cfg.GRAVITY + downforce
        a_max = mu * load / cfg.CAR_MASS

        L, m = cfg.WHEELBASE, cfg.CAR_MASS
        load_f = (cfg.CG_TO_REAR / L) * cfg.CAR_MASS * cfg.GRAVITY \
            + downforce * cfg.DOWNFORCE_FRONT_BIAS
        load_r = (cfg.CG_TO_FRONT / L) * cfg.CAR_MASS * cfg.GRAVITY \
            + downforce * (1.0 - cfg.DOWNFORCE_FRONT_BIAS)

        d_geom = L * a_max / v2
        d_slip = (m * a_max / L) * (
            cfg.CG_TO_REAR / (cfg.CORNER_STIFFNESS_FRONT * max(load_f, 1.0))
            - cfg.CG_TO_FRONT / (cfg.CORNER_STIFFNESS_REAR * max(load_r, 1.0)))
        d = d_geom + max(d_slip, 0.0)

        return _clamp(d * margin, cfg.STEER_LIMIT_FLOOR, cfg.MAX_STEER)

    def _update_steering(self, steer_in: float, dt: float, speed: float,
                         mu: float = config.TYRE_GRIP,
                         analog: bool = False) -> None:
        cfg = config
        # 1. Rate-limit the driver input. A keyboard is a digital device; this
        #    turns a tap into a progressive wheel movement instead of a snap.
        target = _clamp(steer_in, -1.0, 1.0)
        # Wind the input on more gently at speed: the same key press should be
        # a deft nudge at 250 km/h and a quick flick in a slow corner. Held
        # flat below the knee, then a smoothstep so the dulling arrives as a
        # ramp you drive into rather than from the first km/h -- see
        # STEER_RATE_KNEE.
        t = _clamp((speed - cfg.STEER_RATE_KNEE)
                   / max(cfg.STEER_RATE_FULL - cfg.STEER_RATE_KNEE, 1e-6), 0.0, 1.0)
        speed_rate = cfg.STEER_INPUT_RATE * (
            1.0 - cfg.STEER_RATE_SPEED_DROP * t * t * (3.0 - 2.0 * t))

        if analog:
            self.steer_input = target
        elif abs(target) > 1e-3:
            rate = speed_rate
            # allow a quicker change when reversing direction (catching a slide)
            if target * self.steer_input < 0:
                rate = speed_rate + config.STEER_RETURN_RATE
            self.steer_input += _clamp(target - self.steer_input,
                                       -rate * dt, rate * dt)
        else:
            back = config.STEER_RETURN_RATE * dt
            if abs(self.steer_input) <= back:
                self.steer_input = 0.0
            else:
                self.steer_input -= math.copysign(back, self.steer_input)
        self.steer_input = _clamp(self.steer_input, -1.0, 1.0)

        # 2. Map the stick onto the lock the tyres can use at this speed, so
        #    the whole input range stays useful instead of saturating instantly.
        wanted = self.steer_limit(speed, mu) * self.steer_input

        # 3. Steering assist: never command more front slip than the tyre can
        #    convert into grip. A keyboard is always at full deflection, so
        #    without this the front sits permanently past its peak and the car
        #    washes out whenever a direction key is held.
        if self.steer_assist and speed > cfg.STEER_LIMIT_FREE_SPEED:
            peak = mu / cfg.CORNER_STIFFNESS_FRONT      # slip angle of max grip
            slip_f = self.tele.slip_front
            # What the front slip would be with the wheels straight. If the
            # current steering makes it *smaller*, the driver is correcting a
            # slide and the assist must stay out of the way -- otherwise it
            # cancels the very input that recovers the car.
            straight = slip_f + math.copysign(1.0, self.forward_speed or 1.0) \
                * self.steer_angle
            correcting = (cfg.ASSIST_SKIP_WHEN_CORRECTING
                          and abs(slip_f) < abs(straight))
            excess = abs(slip_f) - peak * cfg.ASSIST_SLIP_ALLOW
            if excess > 0.0 and not correcting:
                ease = min(1.0, excess / max(peak * 0.6, 1e-6))
                wanted *= 1.0 - cfg.ASSIST_STRENGTH * ease

        # 3. The rack itself has a finite speed.
        d = _clamp(wanted - self.steer_angle,
                   -config.STEER_RACK_RATE * dt, config.STEER_RACK_RATE * dt)
        self.steer_angle += d

    # -- stability control ------------------------------------------------
    def _stability_control(self, v_long: float, dt: float, mu: float) -> float:
        """Bleed off yaw the steering never asked for.

        Without it, holding a correction through a slide just swaps the car
        into a steady slide the other way -- the yaw has no reason to decay, so
        the slide runs on and on. Real cars have ESC for exactly this.

        Only *excess* yaw is damped. Understeer is left untouched: adding yaw
        the tyres are not producing would be inventing grip.
        """
        cfg = config
        if not (cfg.ESC_ENABLED and self.steer_assist):
            return 0.0
        vx = abs(v_long)
        if vx < cfg.STEER_LIMIT_FREE_SPEED:
            return 0.0

        # yaw rate the steering is asking for, capped by what grip allows
        wanted = v_long * math.tan(self.steer_angle) / cfg.WHEELBASE
        load = cfg.CAR_MASS * cfg.GRAVITY + cfg.DOWNFORCE_COEFF * vx * vx
        grip_cap = mu * load / (cfg.CAR_MASS * vx)
        wanted = _clamp(wanted, -grip_cap, grip_cap)

        limit = abs(wanted) * cfg.ESC_DEADBAND
        excess = abs(self.yaw_rate) - limit
        if excess <= 0.0:
            return 0.0
        damp = min(1.0, cfg.ESC_GAIN * dt)
        self.yaw_rate -= math.copysign(excess * damp, self.yaw_rate)
        return min(1.0, excess / max(abs(self.yaw_rate) + excess, 1e-6))

    # -- integration -------------------------------------------------
    def step(self, ctl: Controls, dt: float, surface) -> None:
        # Not a copy: nothing modifies ``pos`` in place (every update below,
        # and in contact.py, assigns a new array), so the old one can be kept.
        self.prev_pos = self.pos
        self.prev_yaw = self.yaw
        if self.frozen:
            self.vel = np.zeros(2)
            self.yaw_rate = 0.0
            self._update_steering(ctl.steer, dt, 0.0, analog=ctl.analog_steer)
            return

        cfg = config
        # Plain floats throughout: this runs for twenty cars at every physics
        # step, and two-element numpy arrays cost more in call overhead than
        # the arithmetic they hold. fwd = (sy, cy), right = (cy, -sy);
        # (wx, wz) is the world velocity.
        sy, cy = math.sin(self.yaw), math.cos(self.yaw)
        wx, wz = float(self.vel[0]), float(self.vel[1])
        v_long = wx * sy + wz * cy
        v_lat = wx * cy - wz * sy
        speed = math.hypot(wx, wz)

        # -- surface -------------------------------------------------
        # Yaw goes in so the surface is sampled under the wheels rather than
        # under the centre of mass -- see Surface.grip.
        self.on_track, self.grip_scale = surface.grip(self.pos, self.yaw)
        mu = cfg.TYRE_GRIP * self.grip_scale

        self._update_steering(ctl.steer, dt, speed, mu, ctl.analog_steer)

        # -- aero ----------------------------------------------------
        v2 = v_long * v_long
        downforce = cfg.DOWNFORCE_COEFF * v2
        drag = cfg.DRAG_COEFF * self.drag_scale * v_long * abs(v_long)

        # -- axle loads (static + longitudinal transfer + downforce) --
        W = cfg.CAR_MASS * cfg.GRAVITY
        ratio_front = cfg.CG_TO_REAR / cfg.WHEELBASE
        ratio_rear = cfg.CG_TO_FRONT / cfg.WHEELBASE
        transfer = cfg.WEIGHT_TRANSFER * self._long_accel * cfg.CAR_MASS \
            * cfg.CG_HEIGHT / cfg.WHEELBASE
        load_front = max(0.0, ratio_front * W - transfer
                         + downforce * cfg.DOWNFORCE_FRONT_BIAS)
        load_rear = max(0.0, ratio_rear * W + transfer
                        + downforce * (1.0 - cfg.DOWNFORCE_FRONT_BIAS))

        # -- slip angles ---------------------------------------------
        # Each axle's velocity includes the rotation of the car about its CG.
        # Blend from the kinematic model at a crawl to the dynamic one at speed
        # (see BLEND_SPEED_LO/HI). w = 0 kinematic, w = 1 fully dynamic.
        #
        # Keyed on TOTAL speed, not the forward component. Sideways in a big
        # slide the forward component collapses even though the car is still
        # travelling fast -- keying on it scaled the tyre forces to almost
        # nothing exactly when the car was sliding, so it went dead and the
        # slide never scrubbed off. The blend exists because slip angles are
        # ill-defined at low *speed*, which is a property of the whole velocity.
        w = _clamp((speed - cfg.BLEND_SPEED_LO)
                   / max(cfg.BLEND_SPEED_HI - cfg.BLEND_SPEED_LO, 1e-6), 0.0, 1.0)
        kinematic = w <= 0.0
        # Denominator of the slip angles. Floored so atan2 stays well behaved
        # when the car is nearly sideways; at that point the slip angle is ~90
        # degrees either way, so the floor changes nothing that matters.
        vx = max(abs(v_long), 0.5)

        if kinematic:
            slip_f = slip_r = 0.0
            lat_f = lat_r = 0.0
            self.yaw_rate = ((v_long / cfg.WHEELBASE) * math.tan(self.steer_angle)
                             if abs(v_long) > 0.05 else 0.0)
        else:
            sign = math.copysign(1.0, v_long)
            slip_f = math.atan2(v_lat + self.yaw_rate * cfg.CG_TO_FRONT, vx) \
                - sign * self.steer_angle
            slip_r = math.atan2(v_lat - self.yaw_rate * cfg.CG_TO_REAR, vx)

            mu_r = mu * cfg.REAR_GRIP_BIAS
            grip_f = mu * self._combined(self._f_long_front, mu, load_front)
            grip_r = mu_r * self._combined(self._f_long_rear, mu_r, load_rear)
            if ctl.handbrake:
                grip_r *= cfg.HANDBRAKE_GRIP_SCALE

            lat_f = self.tyre_force(slip_f, cfg.CORNER_STIFFNESS_FRONT,
                                    grip_f, load_front) * w
            lat_r = self.tyre_force(slip_r, cfg.CORNER_STIFFNESS_REAR,
                                    grip_r, load_rear) * w

        # -- longitudinal forces -------------------------------------
        # Power-limited engine: constant force off the line, then P/v.
        if ctl.throttle > 0.0:
            f_engine = cfg.ENGINE_FORCE_MAX
            if v_long > 1.0:
                f_engine = min(f_engine,
                               cfg.ENGINE_POWER * self.power_scale / v_long)

            # Traction limit at the driven (rear) axle; grass eats grip too.
            traction = mu * load_rear
            if self.traction_control and not ctl.handbrake:
                # Spend only the friction the rear axle is not already using to
                # corner. Rearranging the friction circle,
                #     F_long_max = sqrt((mu*Fz)^2 - F_lat^2)
                # which is self-tuning: in a straight line F_lat is ~0 and the
                # full launch grip is available, while mid-corner it collapses
                # and stops power-on snap oversteer.
                budget = traction * traction - lat_r * lat_r
                traction = cfg.TC_SAFETY * math.sqrt(max(budget, 0.0))
                # ...and back off further once the rear is actually sliding.
                slide = (abs(math.degrees(slip_r)) - cfg.TC_SLIP_DEG) / \
                    max(cfg.TC_SLIP_FULL_DEG - cfg.TC_SLIP_DEG, 1e-6)
                self.tc_cut = _clamp(slide, 0.0, 1.0) * (1.0 - cfg.TC_MIN_POWER)
                traction *= 1.0 - self.tc_cut
            else:
                self.tc_cut = 0.0
            f_engine = min(f_engine, traction)
            f_engine *= ctl.throttle
        elif v_long > -6.0 and ctl.brake > 0.0 and v_long < 0.5:
            f_engine = -cfg.REVERSE_FORCE * ctl.brake   # reverse
            self.tc_cut = 0.0
        else:
            f_engine = 0.0
            self.tc_cut = 0.0

        f_brake = 0.0
        if ctl.brake > 0.0 and v_long > 0.5:
            f_brake = ctl.brake * cfg.BRAKE_FORCE
        if ctl.handbrake:
            f_brake += cfg.HANDBRAKE_FORCE

        if self.abs_enabled and not ctl.handbrake:
            # ABS, the mirror image of the traction control above. Full braking
            # otherwise spends the front axle's whole friction budget, leaving
            # ~10% for cornering, so the car will not turn while slowing.
            #
            # The floor matters as much as the ceiling: taking only what is
            # left after the lateral demand drives brake force to exactly zero
            # once a tyre approaches its cornering limit, so braking into a
            # corner stopped decelerating the car entirely and it ran wide.
            #
            # How big that floor is decides how the car trail-brakes, so it is
            # taken from the driver's own inputs rather than fixed: the share
            # of the two pedals-plus-wheel demand that is brake. Full brake and
            # a light steer reserves nearly everything for slowing; a dab of
            # brake at the apex reserves nearly nothing and lets the tyre
            # corner. Intent, not measured force -- the point is to honour what
            # the driver asked for, and the friction circle below still caps
            # whatever that works out to.
            want = ctl.brake + abs(ctl.steer)
            brake_share = cfg.ABS_BRAKE_SHARE_MIN
            if want > 1e-6:
                brake_share += (cfg.ABS_BRAKE_SHARE_MAX - cfg.ABS_BRAKE_SHARE_MIN) \
                    * (ctl.brake / want)

            def axle_limit(load: float, lat: float) -> float:
                cap = mu * max(load, 1.0)
                spare = math.sqrt(max(cap * cap - min(abs(lat), cap) ** 2, 0.0))
                return max(spare, brake_share * cap) * cfg.ABS_SAFETY

            f_brake = min(
                f_brake,
                axle_limit(load_front, lat_f) / max(cfg.BRAKE_BIAS_FRONT, 1e-6),
                axle_limit(load_rear, lat_r) / max(1.0 - cfg.BRAKE_BIAS_FRONT, 1e-6))
        else:
            f_brake = min(f_brake, mu * (load_front + load_rear))

        f_long = f_engine - math.copysign(f_brake, v_long) - drag \
            - cfg.ROLL_RESIST * v_long
        if ctl.throttle <= 0.0 and abs(v_long) > 0.5:
            f_long -= math.copysign(cfg.IDLE_DRAG, v_long)

        # Split the longitudinal effort per axle for next step's friction
        # circle: brakes are front-biased, drive is at the rear.
        self._f_long_front = f_brake * cfg.BRAKE_BIAS_FRONT
        self._f_long_rear = (f_brake * (1.0 - cfg.BRAKE_BIAS_FRONT)
                             + abs(f_engine))

        # -- accelerations -------------------------------------------
        a_long = f_long / cfg.CAR_MASS
        a_lat = (lat_f * math.cos(self.steer_angle) + lat_r) / cfg.CAR_MASS
        self._long_accel = a_long
        self._lat_accel = a_lat

        # Integrate in WORLD space. Carrying (v_long, v_lat) across the yaw
        # update instead would rotate the velocity vector along with the body,
        # silently supplying centripetal acceleration that no tyre had to earn
        # -- the car would corner at any g you asked for. Accelerating in world
        # coordinates keeps the turn rate honest: the path only bends as hard
        # as the lateral force actually bends it.
        wx += (sy * a_long + cy * a_lat) * dt
        wz += (cy * a_long - sy * a_lat) * dt
        # Banking, and this one term is the whole of it. On a cambered road
        # gravity has a component along the surface, pointing down the slope,
        # and on the outside of a banked corner "down the slope" is towards
        # the apex -- so some of the centripetal force comes from the planet
        # instead of from the tyres, which is the entire reason circuits are
        # banked. Added in world space beside the tyre forces rather than
        # folded into the grip model: it is an acceleration the car gets for
        # free, not extra grip, and the difference shows the moment you lift.
        # Kept so the renderer can ask the road questions without carrying a
        # Surface of its own -- see Car.sync, which samples it at the
        # *interpolated* position rather than at this step's.
        self.track = surface.track
        bank, nrm, surf_y = surface.camber(self.pos)
        self.surface_y = surf_y
        self.surface_bank = bank
        nrx, nrz = float(nrm[0]), float(nrm[1])
        self.surface_roll = bank * (nrx * cy - nrz * sy)
        if bank:
            g = cfg.GRAVITY * math.sin(bank) * dt
            wx += nrx * g
            wz += nrz * g

        if kinematic:
            # ...except at a crawl, where we steer the velocity directly.
            v_long = wx * sy + wz * cy
            v_lat = (wx * cy - wz * sy) * math.exp(-12.0 * dt)
            wx = sy * v_long + cy * v_lat
            wz = cy * v_long - sy * v_lat

        # -- yaw ------------------------------------------------------
        if not kinematic:
            torque = (lat_f * math.cos(self.steer_angle) * cfg.CG_TO_FRONT
                      - lat_r * cfg.CG_TO_REAR)
            self.yaw_rate += (torque / cfg.YAW_INERTIA) * dt
            # bleed off residual spin so the car settles instead of pirouetting
            self.yaw_rate *= math.exp(-0.6 * dt)
            self.esc_cut = self._stability_control(v_long, dt, mu)
        self.yaw += self.yaw_rate * dt

        if (math.hypot(wx, wz) < cfg.LOW_SPEED_CUTOFF and ctl.throttle <= 0.0
                and not (ctl.reverse and ctl.brake > 0.0)):
            wx = wz = 0.0
            self.yaw_rate = 0.0

        self.vel = np.array((wx, wz))
        self.pos = np.array((float(self.pos[0]) + wx * dt,
                             float(self.pos[1]) + wz * dt))

        # -- wall ------------------------------------------------------
        # An impulse at the contact point, not at the centre of mass. A corner
        # clipped on the way past a barrier has to spin the car; a square hit
        # into the same barrier must not. The difference is entirely in the
        # arm, so both fall out of one calculation.
        corrected, wn, arm = surface.resolve_body(self.pos, self.yaw)
        self.hit_wall = corrected is not None
        if corrected is not None:
            self.pos = corrected
            # How the contact point moves per unit of yaw rate: d(arm)/d(yaw)
            # in this frame, where +yaw turns towards `right`.
            spin = np.array([arm[1], -arm[0]])
            v_c = self.vel + self.yaw_rate * spin
            vn = float(np.dot(v_c, wn))
            if vn < 0.0:
                # Effective mass at the contact: the body resists along the
                # normal both by its mass and, through the arm, by its yaw
                # inertia. Ignoring the second term makes a corner hit as
                # abrupt as a flat one.
                sn = float(np.dot(spin, wn))
                jn = -(1.0 + cfg.WALL_RESTITUTION) * vn / (
                    1.0 / cfg.CAR_MASS + sn * sn / cfg.YAW_INERTIA)
                self.vel += (jn / cfg.CAR_MASS) * wn
                self.yaw_rate += jn * sn / cfg.YAW_INERTIA

                # Scrub along the barrier, capped by the normal impulse, so a
                # graze costs little and a square hit costs a lot.
                tang = np.array([-wn[1], wn[0]])
                v_c = self.vel + self.yaw_rate * spin
                st = float(np.dot(spin, tang))
                jt = -float(np.dot(v_c, tang)) / (
                    1.0 / cfg.CAR_MASS + st * st / cfg.YAW_INERTIA)
                cap = cfg.WALL_FRICTION * jn
                jt = max(-cap, min(cap, jt))
                self.vel += (jt / cfg.CAR_MASS) * tang
                self.yaw_rate += jt * st / cfg.YAW_INERTIA

        # -- telemetry -------------------------------------------------
        t = self.tele
        t.slip_front, t.slip_rear = slip_f, slip_r
        t.load_front, t.load_rear = load_front, load_rear
        t.lat_front, t.lat_rear = lat_f, lat_r
        t.saturated_front = abs(cfg.CORNER_STIFFNESS_FRONT * slip_f) > mu * 0.97
        t.saturated_rear = abs(cfg.CORNER_STIFFNESS_REAR * slip_r) > mu * 0.97
        t.long_accel, t.lat_accel = a_long, a_lat
        t.engine_force = f_engine
        t.downforce = downforce
