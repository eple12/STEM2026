"""The learned driver's hands and feet: a network that steers and works the pedals.

A race driver (``racecraft.RaceDriver``) decides WHERE to be and HOW FAST to
go -- a line offset and its slope, a speed cap, a pace, a few flags -- and
something has to turn that into a wheel angle and pedal positions sixty times a
second. That was ``PlanFollower`` (mintime_driver.py): a hand-built
feed-forward from the solver's plan plus one feedback term. This is the same
job done by a network, trained to drive like it and then past it (see
``tools/dagger_drive.py`` and README, "레이스 AI").

What the network is given is only what a driver sees and what a plan says:

* the car -- speed, yaw rate, slip, wheel angle, how much of the tyres' grip is
  already spent, whether it is on the road, its last move;
* the line -- how far off it the car is and which way it points, and the
  plan's own curvature, speed and pedal forces at ``AHEAD`` metres on (the
  minimum-time solution is the reference, so it is an INPUT, not a rule that
  runs);
* the road -- how much room is left to each white line, now and ahead, which is
  what lets it keep to the road instead of the line when the line and the road
  disagree (a lane change, a car alongside);
* the order it is driving to -- the line offset, its slope and curvature, the
  speed cap, the pace, hold and flat-out flags.

None of it is circuit-specific, so one set of weights drives every circuit at
every plan grip. The network decides every ``REPEAT`` ticks (30 Hz) and the
decision is held between; inference is numpy only, like ``rlpolicy``.
"""
from __future__ import annotations

import math

import numpy as np

from . import config

#: Physics ticks one decision is held for.
REPEAT = 2
#: Metres ahead the line is read at.
AHEAD = (0.0, 10.0, 25.0, 45.0, 70.0, 100.0, 150.0, 220.0)
FORCE_AHEAD = (10.0, 25.0, 50.0, 100.0)
EDGE_AHEAD = (0.0, 30.0)

NAMES = (
    ("lat", "head", "yaw_rate", "yaw_err", "speed", "wheel", "lock", "slip", "plan_wheel",
     "grip_use", "front_load", "on_track", "grip_scale", "v_plan", "v_cap", "plan_force",
     "offset", "d_off", "dd_off", "hold", "flat", "v_min", "last_steer", "last_pedal")
    + tuple(f"kappa{int(d)}" for d in AHEAD)
    + tuple(f"dv{int(d)}" for d in AHEAD)
    + tuple(f"force{int(d)}" for d in FORCE_AHEAD)
    + tuple(f"room_{s}{int(d)}" for d in EDGE_AHEAD for s in ("r", "l")))
OBS_DIM = len(NAMES)
ACT_DIM = 2


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def features(fol, vehicle, k, f, offset, d_off, dd_off, v_cap, pace, hold_speed,
             flat_out, v_min, last) -> np.ndarray:
    """The network's input (see NAMES). ``fol`` is the PlanFollower the driver
    runs: it brings the plan and, as ``fol.track``, the road."""
    cfg = config
    pl = fol.plan
    Lp = pl.L
    at = pl.at
    px, py = float(vehicle.pos[0]), float(vehicle.pos[1])
    vx, vy = float(vehicle.vel[0]), float(vehicle.vel[1])
    speed = max(math.hypot(vx, vy), 1.0)
    x, y = pl._xy_l[k]
    tx, ty = pl._tan_l[k]
    nx, ny = ty, -tx
    sg = Lp["seg"][k] * f
    lat = (px - (x + tx * sg + nx * offset)) * nx + (py - (y + ty * sg + ny * offset)) * ny
    head = _wrap(vehicle.yaw - (at(Lp["yaw"], k, f, wrap_angle=True) + math.atan(d_off)))
    r_ref = at(Lp["r"], k, f)
    lock = max(vehicle.steer_limit(speed, cfg.TYRE_GRIP), 1e-4)
    t = vehicle.tele
    load = t.load_front + t.load_rear
    mu = cfg.TYRE_GRIP * vehicle.grip_scale
    use = 0.0
    if load > 0.0:
        use = abs(t.lat_front * math.cos(vehicle.steer_angle) + t.lat_rear) / max(mu * load, 1.0)
    fwd = vx * math.sin(vehicle.yaw) + vy * math.cos(vehicle.yaw)
    side = vx * math.cos(vehicle.yaw) - vy * math.sin(vehicle.yaw)
    v_at = at(Lp["v"], k, f)
    ff = (at(Lp["fd"], k, f) - at(Lp["fb"], k, f)) * pace * pace / cfg.BRAKE_FORCE

    out = np.empty(OBS_DIM, np.float32)
    out[0] = _clip(lat / 3.0, -2.0, 2.0)
    out[1] = _clip(head / 0.5, -2.0, 2.0)
    out[2] = _clip(vehicle.yaw_rate / 1.0, -2.0, 2.0)
    out[3] = _clip((vehicle.yaw_rate - r_ref) / 0.5, -2.0, 2.0)
    out[4] = speed / 80.0
    out[5] = _clip(vehicle.steer_angle / lock, -2.0, 2.0)
    out[6] = _clip(lock * 5.0, 0.0, 3.0)
    out[7] = _clip(math.atan2(side, max(abs(fwd), 1.0)) * 5.0, -2.0, 2.0)
    out[8] = _clip(at(Lp["delta"], k, f) / lock, -2.0, 2.0)
    out[9] = _clip(use, 0.0, 1.5)
    out[10] = t.load_front / max(load, 1.0)
    out[11] = 1.0 if vehicle.on_track else 0.0
    out[12] = vehicle.grip_scale
    out[13] = _clip((v_at * pace - speed) / 15.0, -2.0, 2.0)
    out[14] = _clip((v_cap - speed) / 20.0, -1.0, 2.0)
    out[15] = _clip(ff, -1.0, 1.0)
    out[16] = _clip(offset / 6.0, -2.0, 2.0)
    out[17] = _clip(d_off * 5.0, -2.0, 2.0)
    out[18] = _clip(dd_off * 100.0, -2.0, 2.0)
    out[19] = 1.0 if hold_speed else 0.0
    out[20] = 1.0 if flat_out else 0.0
    out[21] = _clip(v_min / 10.0, 0.0, 2.0)
    out[22] = _clip(float(last[0]), -1.0, 1.0)
    out[23] = _clip(float(last[1]), -1.0, 1.0)
    j = 24
    for m in AHEAD:
        kp, fp = pl.ahead(k, f, m) if m else (k, f)
        kap = at(Lp["kappa"], kp, fp)
        out[j] = _clip(kap / max(1.0 - kap * offset, 0.3) * 100.0
                       + (dd_off * 100.0 if m == 0.0 else 0.0), -3.0, 3.0)
        out[j + len(AHEAD)] = _clip((at(Lp["v"], kp, fp) * pace - speed) / 30.0, -2.0, 2.0)
        j += 1
    j += len(AHEAD)
    for m in FORCE_AHEAD:
        kp, fp = pl.ahead(k, f, m)
        out[j] = _clip((at(Lp["fd"], kp, fp) - at(Lp["fb"], kp, fp)) * pace * pace
                       / cfg.BRAKE_FORCE, -1.0, 1.0)
        j += 1
    # Room to the white lines: the car's centre against the line plus its offset,
    # less half the body.
    n_car = at(Lp["n_raw"], k, f) + offset + lat
    tr = fol.track
    for m in EDGE_AHEAD:
        kp, fp = pl.ahead(k, f, m) if m else (k, f)
        n_here = at(Lp["n_raw"], kp, fp) + offset + lat if m else n_car
        out[j] = _clip((tr.w_right[kp] - n_here - cfg.BODY_HALF_WIDTH) / 6.0, -2.0, 2.0)
        out[j + 1] = _clip((tr.w_left[kp] + n_here - cfg.BODY_HALF_WIDTH) / 6.0, -2.0, 2.0)
        j += 2
    return out


def to_controls(a, Controls):
    steer = _clip(float(a[0]), -1.0, 1.0)
    p = _clip(float(a[1]), -1.0, 1.0)
    return Controls(throttle=p if p > 0.0 else 0.0, brake=-p if p < 0.0 else 0.0,
                    steer=steer, analog_steer=True)


class DriveNet:
    """The deployed network: numpy MLP, weights in an .npz (W0.., b0.., layers)."""

    #: The follower need not compute its own answer alongside.
    needs_teacher = False

    def __init__(self, w: dict):
        self.n = int(w["layers"])
        self.W = [np.asarray(w[f"W{i}"], np.float32) for i in range(self.n)]
        self.b = [np.asarray(w[f"b{i}"], np.float32) for i in range(self.n)]
        if self.W[0].shape[0] != OBS_DIM or self.W[-1].shape[1] != ACT_DIM:
            raise ValueError("drivenet weights do not match the observation")

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls({k: z[k] for k in z.files})

    def act(self, x: np.ndarray, teacher=None) -> np.ndarray:
        h = x
        for i in range(self.n):
            h = h @ self.W[i] + self.b[i]
            if i < self.n - 1:
                h = np.maximum(h, 0.0)
        return h
