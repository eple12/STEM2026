"""Batched twin of ``game.vehicle.Vehicle``.

The same dynamic single-track model -- Pacejka lateral forces from slip
angles and axle load, yaw driven by the resulting torque -- with every
scalar promoted to a ``(B,)`` tensor and every branch to a ``torch.where``.
Line-for-line the CPU file is the reference; the comments there explain the
physics, the ones here only flag where batching changes the shape of the
code.
"""
from __future__ import annotations

import math

import torch

from .trackgpu import config
from game.vehicle import _pacejka_b                        # noqa: E402

_PACEJKA_B = _pacejka_b(config.PACEJKA_C, config.PACEJKA_E)


def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def _copysign(mag, ref):
    """``math.copysign`` with the tensor convention that +0 keeps the sign."""
    return torch.where(ref < 0, -mag, mag)


def steer_limit(speed, mu):
    """Largest front-wheel angle the tyres can use at this speed."""
    cfg = config
    t = ((speed - cfg.STEER_MARGIN_KNEE)
         / max(cfg.STEER_MARGIN_FULL - cfg.STEER_MARGIN_KNEE, 1e-6)).clamp(0, 1)
    margin = cfg.STEER_LIMIT_MARGIN + cfg.STEER_MARGIN_LOW_EXTRA * (
        1.0 - _smoothstep(t))

    v2 = (speed * speed).clamp(min=1e-9)
    downforce = cfg.DOWNFORCE_COEFF * v2
    load = cfg.CAR_MASS * cfg.GRAVITY + downforce
    a_max = mu * load / cfg.CAR_MASS

    L, m = cfg.WHEELBASE, cfg.CAR_MASS
    load_f = (cfg.CG_TO_REAR / L) * m * cfg.GRAVITY \
        + downforce * cfg.DOWNFORCE_FRONT_BIAS
    load_r = (cfg.CG_TO_FRONT / L) * m * cfg.GRAVITY \
        + downforce * (1.0 - cfg.DOWNFORCE_FRONT_BIAS)

    d_geom = L * a_max / v2
    d_slip = (m * a_max / L) * (
        cfg.CG_TO_REAR / (cfg.CORNER_STIFFNESS_FRONT * load_f.clamp(min=1.0))
        - cfg.CG_TO_FRONT / (cfg.CORNER_STIFFNESS_REAR * load_r.clamp(min=1.0)))
    d = d_geom + d_slip.clamp(min=0.0)
    out = (d * margin).clamp(cfg.STEER_LIMIT_FLOOR, cfg.MAX_STEER)
    return torch.where(speed < cfg.STEER_LIMIT_FREE_SPEED,
                       torch.full_like(out, cfg.MAX_STEER), out)


def tyre_force(slip, stiffness: float, mu, load):
    """Pacejka's Magic Formula, peak pinned at ``mu / stiffness``."""
    cfg = config
    cap = mu * load.clamp(min=1.0)
    peak_slip = (mu / max(stiffness, 1e-6)).clamp(min=1e-6)
    b = _PACEJKA_B / peak_slip
    ba = b * -slip
    c, e = cfg.PACEJKA_C, cfg.PACEJKA_E
    return cap * torch.sin(c * torch.atan(ba - e * (ba - torch.atan(ba))))


def _combined(f_long, mu, load):
    """Share of lateral grip left after spending some on accel/braking."""
    cap = mu * load.clamp(min=1.0)
    used = (f_long.abs() / cap).clamp(max=0.995)
    return torch.sqrt((1.0 - used * used).clamp(min=0.0))


class VecVehicle:
    """B cars, one tensor each."""

    def __init__(self, n: int, device):
        f = lambda: torch.zeros(n, device=device)           # noqa: E731
        self.n = n
        self.device = device
        self.pos = torch.zeros(n, 2, device=device)
        self.vel = torch.zeros(n, 2, device=device)
        self.yaw = f()
        self.yaw_rate = f()
        self.steer_input = f()
        self.steer_angle = f()
        self._long_accel = f()
        self._lat_accel = f()
        self._f_long_front = f()
        self._f_long_rear = f()
        self.grip_scale = torch.ones(n, device=device)
        self.on_track = torch.ones(n, dtype=torch.bool, device=device)
        self.hit_wall = torch.zeros(n, dtype=torch.bool, device=device)
        self.slip_front = f()
        self.slip_rear = f()

    @property
    def speed(self):
        return torch.linalg.norm(self.vel, dim=-1)

    @property
    def forward_speed(self):
        fwd = torch.stack([torch.sin(self.yaw), torch.cos(self.yaw)], -1)
        return (self.vel * fwd).sum(-1)

    def place(self, mask, pos, yaw, vel):
        """``Vehicle()`` + ``place()`` + the velocity ``RaceEnv._place`` sets,
        applied to the masked rows only. Every other piece of state is reset
        too, because the scalar version builds a whole new ``Vehicle``."""
        m = mask
        self.pos = torch.where(m[:, None], pos, self.pos)
        self.vel = torch.where(m[:, None], vel, self.vel)
        self.yaw = torch.where(m, yaw, self.yaw)
        for name in ("yaw_rate", "steer_input", "steer_angle", "_long_accel",
                     "_lat_accel", "_f_long_front", "_f_long_rear",
                     "slip_front", "slip_rear"):
            setattr(self, name, torch.where(m, torch.zeros_like(self.yaw),
                                            getattr(self, name)))
        self.grip_scale = torch.where(m, torch.ones_like(self.grip_scale),
                                      self.grip_scale)
        self.on_track = self.on_track | m
        self.hit_wall = self.hit_wall & ~m

    # ------------------------------------------------------------------
    def _update_steering(self, steer_in, dt, speed, mu):
        """The rate-limited rack, the speed-scaled lock and the slip assist.

        ``analog`` is always False here: the discrete action set feeds the
        same held-keypress steering the CPU run trains against.
        """
        cfg = config
        target = steer_in.clamp(-1.0, 1.0)
        t = ((speed - cfg.STEER_RATE_KNEE)
             / max(cfg.STEER_RATE_FULL - cfg.STEER_RATE_KNEE, 1e-6)).clamp(0, 1)
        speed_rate = cfg.STEER_INPUT_RATE * (
            1.0 - cfg.STEER_RATE_SPEED_DROP * _smoothstep(t))

        rate = torch.where(target * self.steer_input < 0,
                           speed_rate + cfg.STEER_RETURN_RATE, speed_rate)
        wound = self.steer_input + (target - self.steer_input).clamp(
            -rate * dt, rate * dt)

        back = cfg.STEER_RETURN_RATE * dt
        centred = torch.where(
            self.steer_input.abs() <= back,
            torch.zeros_like(self.steer_input),
            self.steer_input - _copysign(
                torch.full_like(self.steer_input, back), self.steer_input))

        self.steer_input = torch.where(target.abs() > 1e-3, wound,
                                       centred).clamp(-1.0, 1.0)

        wanted = steer_limit(speed, mu) * self.steer_input

        # Steering assist: never command more front slip than the tyre can
        # convert into grip -- unless the driver is catching a slide, where
        # the assist has to stay out of the way.
        if cfg.STEER_ASSIST:
            peak = mu / cfg.CORNER_STIFFNESS_FRONT
            slip_f = self.slip_front
            fs = self.forward_speed
            sign = torch.where(fs < 0, -torch.ones_like(fs),
                               torch.ones_like(fs))
            straight = slip_f + sign * self.steer_angle
            correcting = (slip_f.abs() < straight.abs())
            if not cfg.ASSIST_SKIP_WHEN_CORRECTING:
                correcting = torch.zeros_like(correcting)
            excess = slip_f.abs() - peak * cfg.ASSIST_SLIP_ALLOW
            ease = (excess / (peak * 0.6).clamp(min=1e-6)).clamp(max=1.0)
            scaled = wanted * (1.0 - cfg.ASSIST_STRENGTH * ease)
            active = ((speed > cfg.STEER_LIMIT_FREE_SPEED)
                      & (excess > 0.0) & ~correcting)
            wanted = torch.where(active, scaled, wanted)

        d = (wanted - self.steer_angle).clamp(-cfg.STEER_RACK_RATE * dt,
                                              cfg.STEER_RACK_RATE * dt)
        self.steer_angle = self.steer_angle + d

    def _stability_control(self, v_long, dt, mu, kinematic):
        """Bleed off yaw the steering never asked for. Excess only."""
        cfg = config
        if not (cfg.ESC_ENABLED and cfg.STEER_ASSIST):
            return
        vx = v_long.abs()
        wanted = v_long * torch.tan(self.steer_angle) / cfg.WHEELBASE
        load = cfg.CAR_MASS * cfg.GRAVITY + cfg.DOWNFORCE_COEFF * vx * vx
        grip_cap = mu * load / (cfg.CAR_MASS * vx.clamp(min=1e-6))
        wanted = torch.max(torch.min(wanted, grip_cap), -grip_cap)

        limit = wanted.abs() * cfg.ESC_DEADBAND
        excess = self.yaw_rate.abs() - limit
        damp = min(1.0, cfg.ESC_GAIN * dt)
        damped = self.yaw_rate - _copysign(excess * damp, self.yaw_rate)
        active = (~kinematic) & (vx >= cfg.STEER_LIMIT_FREE_SPEED) \
            & (excess > 0.0)
        self.yaw_rate = torch.where(active, damped, self.yaw_rate)

    # ------------------------------------------------------------------
    def step(self, steer, throttle, brake, dt: float, surf):
        cfg = config
        fwd = torch.stack([torch.sin(self.yaw), torch.cos(self.yaw)], -1)
        right = torch.stack([torch.cos(self.yaw), -torch.sin(self.yaw)], -1)
        v_long = (self.vel * fwd).sum(-1)
        v_lat = (self.vel * right).sum(-1)
        speed = self.speed

        # -- surface, sampled under the wheels --------------------------
        self.on_track, self.grip_scale = surf.grip(self.pos, self.yaw)
        mu = cfg.TYRE_GRIP * self.grip_scale

        self._update_steering(steer, dt, speed, mu)

        # -- aero -------------------------------------------------------
        v2 = v_long * v_long
        downforce = cfg.DOWNFORCE_COEFF * v2
        drag = cfg.DRAG_COEFF * v_long * v_long.abs()

        # -- axle loads (static + transfer + downforce) -----------------
        W = cfg.CAR_MASS * cfg.GRAVITY
        ratio_front = cfg.CG_TO_REAR / cfg.WHEELBASE
        ratio_rear = cfg.CG_TO_FRONT / cfg.WHEELBASE
        transfer = (cfg.WEIGHT_TRANSFER * self._long_accel * cfg.CAR_MASS
                    * cfg.CG_HEIGHT / cfg.WHEELBASE)
        load_front = (ratio_front * W - transfer
                      + downforce * cfg.DOWNFORCE_FRONT_BIAS).clamp(min=0.0)
        load_rear = (ratio_rear * W + transfer
                     + downforce * (1.0 - cfg.DOWNFORCE_FRONT_BIAS)).clamp(min=0.0)

        # -- slip angles, blended out of the dynamic model at a crawl ---
        w = ((speed - cfg.BLEND_SPEED_LO)
             / max(cfg.BLEND_SPEED_HI - cfg.BLEND_SPEED_LO, 1e-6)).clamp(0, 1)
        kinematic = w <= 0.0
        vx = v_long.abs().clamp(min=0.5)
        sign = torch.where(v_long < 0, -torch.ones_like(v_long),
                           torch.ones_like(v_long))

        slip_f = torch.atan2(v_lat + self.yaw_rate * cfg.CG_TO_FRONT, vx) \
            - sign * self.steer_angle
        slip_r = torch.atan2(v_lat - self.yaw_rate * cfg.CG_TO_REAR, vx)

        mu_r = mu * cfg.REAR_GRIP_BIAS
        grip_f = mu * _combined(self._f_long_front, mu, load_front)
        grip_r = mu_r * _combined(self._f_long_rear, mu_r, load_rear)
        lat_f = tyre_force(slip_f, cfg.CORNER_STIFFNESS_FRONT,
                           grip_f, load_front) * w
        lat_r = tyre_force(slip_r, cfg.CORNER_STIFFNESS_REAR,
                           grip_r, load_rear) * w

        zero = torch.zeros_like(slip_f)
        slip_f = torch.where(kinematic, zero, slip_f)
        slip_r = torch.where(kinematic, zero, slip_r)
        lat_f = torch.where(kinematic, zero, lat_f)
        lat_r = torch.where(kinematic, zero, lat_r)
        yr_kin = torch.where(v_long.abs() > 0.05,
                             (v_long / cfg.WHEELBASE)
                             * torch.tan(self.steer_angle), zero)
        self.yaw_rate = torch.where(kinematic, yr_kin, self.yaw_rate)

        # -- longitudinal: power-limited engine, traction control -------
        f_eng_pow = torch.where(
            v_long > 1.0,
            torch.minimum(torch.full_like(v_long, cfg.ENGINE_FORCE_MAX),
                          cfg.ENGINE_POWER / v_long.clamp(min=1e-6)),
            torch.full_like(v_long, cfg.ENGINE_FORCE_MAX))
        traction = mu * load_rear
        budget = traction * traction - lat_r * lat_r
        traction = cfg.TC_SAFETY * torch.sqrt(budget.clamp(min=0.0))
        slide = ((slip_r * 180.0 / math.pi).abs() - cfg.TC_SLIP_DEG) / \
            max(cfg.TC_SLIP_FULL_DEG - cfg.TC_SLIP_DEG, 1e-6)
        tc_cut = slide.clamp(0.0, 1.0) * (1.0 - cfg.TC_MIN_POWER)
        traction = traction * (1.0 - tc_cut)
        f_drive = torch.minimum(f_eng_pow, traction) * throttle

        f_reverse = -cfg.REVERSE_FORCE * brake
        reversing = (throttle <= 0.0) & (v_long > -6.0) & (brake > 0.0) \
            & (v_long < 0.5)
        f_engine = torch.where(throttle > 0.0, f_drive,
                               torch.where(reversing, f_reverse,
                                           torch.zeros_like(f_drive)))

        f_brake = torch.where((brake > 0.0) & (v_long > 0.5),
                              brake * cfg.BRAKE_FORCE,
                              torch.zeros_like(brake))

        # ABS: leave the axle whatever friction the lateral demand is not
        # using, but never less than the driver's own brake share of it.
        want = brake + steer.abs()
        share = torch.where(
            want > 1e-6,
            cfg.ABS_BRAKE_SHARE_MIN
            + (cfg.ABS_BRAKE_SHARE_MAX - cfg.ABS_BRAKE_SHARE_MIN)
            * (brake / want.clamp(min=1e-6)),
            torch.full_like(want, cfg.ABS_BRAKE_SHARE_MIN))

        def axle_limit(load, lat):
            cap = mu * load.clamp(min=1.0)
            spare = torch.sqrt(
                (cap * cap - torch.minimum(lat.abs(), cap) ** 2).clamp(min=0.0))
            return torch.maximum(spare, share * cap) * cfg.ABS_SAFETY

        f_brake = torch.minimum(
            f_brake,
            torch.minimum(
                axle_limit(load_front, lat_f) / max(cfg.BRAKE_BIAS_FRONT, 1e-6),
                axle_limit(load_rear, lat_r)
                / max(1.0 - cfg.BRAKE_BIAS_FRONT, 1e-6)))

        f_long = f_engine - _copysign(f_brake, v_long) - drag \
            - cfg.ROLL_RESIST * v_long
        idle = _copysign(torch.full_like(v_long, cfg.IDLE_DRAG), v_long)
        f_long = f_long - torch.where((throttle <= 0.0) & (v_long.abs() > 0.5),
                                      idle, torch.zeros_like(idle))

        self._f_long_front = f_brake * cfg.BRAKE_BIAS_FRONT
        self._f_long_rear = (f_brake * (1.0 - cfg.BRAKE_BIAS_FRONT)
                             + f_engine.abs())

        # -- accelerations, integrated in WORLD space -------------------
        a_long = f_long / cfg.CAR_MASS
        a_lat = (lat_f * torch.cos(self.steer_angle) + lat_r) / cfg.CAR_MASS
        self._long_accel = a_long
        self._lat_accel = a_lat

        self.vel = self.vel + (fwd * a_long[:, None]
                               + right * a_lat[:, None]) * dt

        # Banking is off (config.BANKING_ENABLED False) so camber adds no
        # acceleration -- but the scalar version's camber() call advances the
        # search hint, and the hint sequence has to match.
        surf.refresh(self.pos)

        # ...except at a crawl, where the velocity is steered directly.
        v_long_k = (self.vel * fwd).sum(-1)
        v_lat_k = (self.vel * right).sum(-1) * math.exp(-12.0 * dt)
        vel_kin = fwd * v_long_k[:, None] + right * v_lat_k[:, None]
        self.vel = torch.where(kinematic[:, None], vel_kin, self.vel)

        # -- yaw --------------------------------------------------------
        torque = (lat_f * torch.cos(self.steer_angle) * cfg.CG_TO_FRONT
                  - lat_r * cfg.CG_TO_REAR)
        spun = (self.yaw_rate + (torque / cfg.YAW_INERTIA) * dt) \
            * math.exp(-0.6 * dt)
        self.yaw_rate = torch.where(kinematic, self.yaw_rate, spun)
        self._stability_control(v_long, dt, mu, kinematic)
        self.yaw = self.yaw + self.yaw_rate * dt

        stopped = (self.speed < cfg.LOW_SPEED_CUTOFF) & (throttle <= 0.0)
        self.vel = torch.where(stopped[:, None], torch.zeros_like(self.vel),
                               self.vel)
        self.yaw_rate = torch.where(stopped, torch.zeros_like(self.yaw_rate),
                                    self.yaw_rate)

        self.pos = self.pos + self.vel * dt

        # -- wall: an impulse at the contact point, not at the CG -------
        hit, corrected, wn, arm = surf.resolve_body(self.pos, self.yaw)
        self.hit_wall = hit
        self.pos = torch.where(hit[:, None], corrected, self.pos)

        spin = torch.stack([arm[:, 1], -arm[:, 0]], -1)
        v_c = self.vel + self.yaw_rate[:, None] * spin
        vn = (v_c * wn).sum(-1)
        sn = (spin * wn).sum(-1)
        jn = -(1.0 + cfg.WALL_RESTITUTION) * vn / (
            1.0 / cfg.CAR_MASS + sn * sn / cfg.YAW_INERTIA)
        impact = hit & (vn < 0.0)
        jn = torch.where(impact, jn, torch.zeros_like(jn))
        self.vel = self.vel + (jn / cfg.CAR_MASS)[:, None] * wn
        self.yaw_rate = self.yaw_rate + jn * sn / cfg.YAW_INERTIA

        # Scrub along the barrier, capped by the normal impulse.
        tang = torch.stack([-wn[:, 1], wn[:, 0]], -1)
        v_c = self.vel + self.yaw_rate[:, None] * spin
        st = (spin * tang).sum(-1)
        jt = -(v_c * tang).sum(-1) / (
            1.0 / cfg.CAR_MASS + st * st / cfg.YAW_INERTIA)
        cap = cfg.WALL_FRICTION * jn
        jt = torch.max(torch.min(jt, cap), -cap)
        jt = torch.where(impact, jt, torch.zeros_like(jt))
        self.vel = self.vel + (jt / cfg.CAR_MASS)[:, None] * tang
        self.yaw_rate = self.yaw_rate + jt * st / cfg.YAW_INERTIA

        self.slip_front, self.slip_rear = slip_f, slip_r
