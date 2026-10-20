"""The player's driving assists: auto steering, auto pedals, lane choice.

For somebody new to racing. The follower that drives the AI (``PlanFollower``,
mintime_driver.py) keeps the player's car on the solved racing line, shifted
sideways by a lane the player chooses with A / D, and it can work the pedals as
well, at the pace of the line and behind any car ahead. With both on, the whole
of driving is picking a lane.

* **Auto steering** -- the follower's wheel angle replaces the player's, until
  the player steers: A or D held is the player's own wheel, whole, and the
  follower waits. Let go, and it takes the car over again from where it is -- it
  follows the line it finds itself on rather than swinging back to the racing
  line (Q takes it back there) -- but no further out than the white line, the
  centre on it, so a wheel is always still inside.
* **Auto pedals** -- the follower's throttle and brake replace the player's, until
  the player works a pedal: W or S held is the player's own pedals.
  It drives exactly as an AI of the chosen difficulty does: the plan and the
  pace of a mid-grid driver of that level (``teams.skill_for``), so Novice is
  slow and careful and Legend is flat out; capped to what a car ahead in the
  lane leaves room for.

Either can be on alone. Settings only: nothing here can be changed mid-race.
"""
from __future__ import annotations

import math

from . import config, settings, teams
from .mintime_driver import Plan, PlanFollower, path_file
from .vehicle import Controls

#: How close the car's centre may get to a white line when the follower takes
#: over: on it. Past that, all four wheels are out and the car has left the track.
EDGE_ROOM = 0.0
REACH = 30.0
#: How briskly the car follows the line being slid (rad/s).
OMEGA = 5.0
def _grips(circuit: str) -> tuple:
    return tuple(g for g in teams.PLAN_GRIPS if path_file(circuit, f"g{round(g * 100):02d}").exists())


def available(circuit: str) -> bool:
    return bool(_grips(circuit))


def plan_for(circuit: str, level: int):
    """(plan path, pace) of a mid-grid driver of this difficulty -- what the
    AI at that level drives."""
    lv = teams.DIFFICULTY.get(level, teams.DIFFICULTY[teams.DEFAULT_LEVEL])
    grip = lv.top_grip - lv.spread * 0.5
    tag, pace = teams.plan_tag(grip, _grips(circuit))
    return path_file(circuit, tag), pace


class Assist:
    def __init__(self, track, surface, level: int = teams.DEFAULT_LEVEL):
        self.track = track
        self.surface = surface
        path, self.pace = plan_for(track.name, level)
        self.plan = Plan(path)
        self.follow = PlanFollower(self.plan)
        self.follow.track = track
        self.steer = settings.current.auto_steer
        self.pedals = settings.current.auto_pedals
        self.target = 0.0          # the lane chosen (m right of the line)
        self.offset = 0.0          # where the car is steering for now
        self.rate = 0.0
        self._i = 0
        self._k = 0

    # -- lane ---------------------------------------------------------------
    def center(self) -> None:
        self.target = 0.0

    def _room(self, k: int) -> tuple[float, float]:
        """(left limit, right limit) of the lane offset, now and a little way on."""
        pl, tr = self.plan, self.track
        lo, hi = -REACH, REACH
        for m in (0.0, 30.0, 70.0):
            kk, _ = pl.ahead(k, 0.0, m) if m else (k, 0.0)
            n = float(pl.n_raw[kk])
            lo = max(lo, -tr.w_left[kk] + EDGE_ROOM - n)
            hi = min(hi, tr.w_right[kk] - EDGE_ROOM - n)
        return min(lo, 0.0), max(hi, 0.0)

    # -- per tick -----------------------------------------------------------
    def controls(self, vehicle, manual: Controls, dt: float, cap: float = math.inf) -> Controls:
        i, _ = self.surface.progress(vehicle.pos)
        k, f = self.plan.locate(vehicle.pos, i)
        self._k = k
        lo, hi = self._room(k)
        want = self.target = min(max(self.target, lo), hi)
        speed = max(vehicle.speed, 5.0)
        acc = OMEGA * OMEGA * (want - self.offset) - 2.0 * OMEGA * self.rate
        self.rate += acc * dt
        self.offset += self.rate * dt
        ctl = self.follow.controls(vehicle, k, f, offset=self.offset,
                                   d_off=self.rate / speed, dd_off=acc / (speed * speed),
                                   v_cap=cap, pace=self.pace if self.pedals else 1.0)
        out = Controls(throttle=manual.throttle, brake=manual.brake, steer=manual.steer,
                       handbrake=manual.handbrake)
        if self.steer:
            if manual.steer:
                # The player's own wheel. The follower keeps the line it is on
                # (where the car is now, kept inside the road) for when it takes
                # over again.
                here = min(max(self.offset + self.follow.lat_err, lo), hi)
                self.target = self.offset = here
                self.rate = 0.0
            else:
                out.steer = ctl.steer
                out.analog_steer = True
        if self.pedals and not (manual.throttle or manual.brake):
            out.throttle, out.brake = ctl.throttle, ctl.brake
        return out

    def view(self) -> tuple[float, float]:
        """(racing line, tracked line) as fractions across the road at the car,
        0 = left white line, 1 = right: what the HUD's box draws."""
        k = self._k
        wl, wr = float(self.track.w_left[k]), float(self.track.w_right[k])
        n = float(self.plan.n_raw[k])
        span = max(wl + wr, 1e-6)
        return (n + wl) / span, (n + self.target + wl) / span

    @property
    def lane_text(self) -> str:
        """Metres off the racing line, + to the right."""
        return "ON THE LINE" if abs(self.target) < 0.25 else f"{self.target:+.1f} m"


#: Following a car: the deceleration the player's car is allowed to need to
#: stop behind it (m/s^2), the centre-to-centre distance it stops at (a car's
#: length and a few metres) and the time gap kept on top, at the other car's speed.
FOLLOW_DECEL = 9.0
STANDOFF = config.CAR_BODY_LENGTH + 4.0
HEADWAY = 0.4


def leader_cap(vehicle, others) -> float:
    """Speed (m/s) from which the car can still stop behind the nearest car ahead
    in its lane; ``inf`` if there is none. *others*: vehicles of the cars around.

    The speed that braking at ``FOLLOW_DECEL`` would bring to the leader's speed
    over the room between them -- so on the grid, behind a car that has not
    moved, the car may still accelerate until the room is nearly used up (a cap
    of "the leader's speed" held it on the brake until the car ahead was away)."""
    sy, cy = math.sin(vehicle.yaw), math.cos(vehicle.yaw)
    px, pz = float(vehicle.pos[0]), float(vehicle.pos[1])
    best = None
    for o in others:
        dx, dz = float(o.pos[0]) - px, float(o.pos[1]) - pz
        ahead = dx * sy + dz * cy
        side = dx * cy - dz * sy
        if 2.0 < ahead < 120.0 and abs(side) < 2.8 and (best is None or ahead < best[0]):
            best = (ahead, max(float(o.vel[0]) * sy + float(o.vel[1]) * cy, 0.0))
    if best is None:
        return math.inf
    gap, v = best
    room = max(gap - STANDOFF - HEADWAY * v, 0.0)
    return math.sqrt(v * v + 2.0 * FOLLOW_DECEL * room)
