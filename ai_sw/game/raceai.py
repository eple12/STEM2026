"""The learned decision layer of a race driver: what it sees and what it can do.

A race driver (``racecraft.RaceDriver``) is a minimum-time plan with a layer of
rules on top that decides, every plan tick (about 7.5 Hz), WHERE across the road
to drive and how hard to push -- passing, defending, queueing, getting out of
the way. That layer is what looks un-human, and it is the one thing here a
policy can replace: the plan under it stays the plan, and so do the safety
rules (the room rule, the leader cap, yellow flags, recovery).

* **Actions** -- discrete, as in ``rlpolicy`` and for the same reason (an argmax
  over value estimates has no exploration noise baked into the greedy policy):
  a lane, as metres off the plan's line, times a pace (``LANES`` are the rule
  layer's own lanes, racecraft.LANES; ``PACES`` a lift, the plan's pace, and a
  small push); and three that are about a pass -- begin one on the inside of the
  coming corner, begin one on the outside, or ``HOLD`` what is going on. A pass
  is not a lane: its geometry (alongside the car ahead, settled by the braking
  zone's middle, the room rule keeping the two apart) is the rule layer's
  machinery and stays so; the policy decides WHEN to start one and whether to
  give it up (any lane action while one is under way does).

* **Observation** -- everything in the driver's own frame, so one policy can
  drive any circuit: where it is on the road, how fast against the plan, the
  road ahead (curvature and the speed the plan asks for at five distances,
  the distance to the next braking zone) and the nearest cars -- ahead and
  behind -- as (gap, lateral offset, closing speed). No absolute positions, no
  circuit identity. ``OBS_NAMES`` documents every column; the network and the
  environment must agree on it exactly, so this is the only place it is built.

Inference is numpy-only (see ``rlpolicy``): the game never imports torch.
"""
from __future__ import annotations

import math

import numpy as np

from . import config

#: Metres off the plan's line a lane sits (+ = right), and the paces.
LANES = (-6.0, -3.5, -1.75, 0.0, 1.75, 3.5, 6.0)
PACES = (0.94, 1.0, 1.03)
N_LANES, N_PACES = len(LANES), len(PACES)
ATTACK_IN = N_LANES * N_PACES
ATTACK_OUT = ATTACK_IN + 1
HOLD = ATTACK_IN + 2
N_ACTIONS = HOLD + 1
#: The action that is "stay on the line at the plan's pace".
DEFAULT_ACTION = LANES.index(0.0) * N_PACES + PACES.index(1.0)

#: Distances ahead the road is sampled at (m).
LOOK = (20.0, 60.0, 120.0, 200.0, 300.0)
#: Cars looked at, and the window they are taken from (m behind, m ahead).
N_NEAR = 6
NEAR_BEHIND = 80.0
NEAR_AHEAD = 200.0

_EGO = ("lane", "target", "room_l", "room_r", "speed", "pace_ratio", "accel",
        "to_brake", "held", "yellow", "aggression", "width") \
    + tuple(f"kappa{int(d)}" for d in LOOK) + tuple(f"dv{int(d)}" for d in LOOK) \
    + ("lane_age", "lead_edge", "zone_dist", "zone_drop", "straight",
       "attack", "can_in", "can_out")
_NEAR = ("gap", "lat", "closing", "stopped", "valid")
OBS_NAMES = _EGO + tuple(f"n{i}_{f}" for i in range(N_NEAR) for f in _NEAR)
OBS_DIM = len(OBS_NAMES)


def decode(a: int) -> tuple[float, float]:
    """(lane offset in metres, pace multiplier) of a lane action *a*; the pass
    actions are the line at the plan's pace."""
    if a >= ATTACK_IN:
        return 0.0, 1.0
    return LANES[a // N_PACES], PACES[a % N_PACES]


def encode(lane: float, pace: float = 1.0) -> int:
    """The action nearest to a lane and a pace."""
    li = int(np.argmin([abs(lane - x) for x in LANES]))
    pi = int(np.argmin([abs(pace - x) for x in PACES]))
    return li * N_PACES + pi


def _clip(x: float, lim: float) -> float:
    return -lim if x < -lim else lim if x > lim else x


def observe(driver, me, field, out: np.ndarray | None = None) -> np.ndarray:
    """The observation of *driver* (whose own view is *me*) among *field*
    (every car's ``CarView``). float32, length ``OBS_DIM``."""
    fr = driver.frame
    plan = driver.plan
    tr = driver.track
    v = np.zeros(OBS_DIM, np.float32) if out is None else out
    k = fr.node(me.s)
    d_now = driver._offset_at(me.s)[0]
    lo = -float(tr.w_left[k])
    hi = float(tr.w_right[k])
    v_plan = float(plan.v[k]) * driver.pace_lap
    n_look = len(LOOK)
    v[0] = _clip(d_now / 6.0, 2.0)
    v[1] = _clip(driver.target / 6.0, 2.0)
    v[2] = _clip((me.n - lo) / 10.0, 2.0)
    v[3] = _clip((hi - me.n) / 10.0, 2.0)
    v[4] = me.v / 100.0
    v[5] = _clip((me.v / max(v_plan, 1.0) - 1.0) * 5.0, 2.0)
    v[6] = _clip(me.a / 10.0, 2.0)
    v[7] = min(float(driver.to_brake[k]), 400.0) / 400.0
    v[8] = _clip(driver.held_t / 2.0, 1.0)
    v[9] = 1.0 if driver.yellow else 0.0
    v[10] = driver.skill.aggression
    v[11] = (hi - lo) / 20.0
    for j, dist in enumerate(LOOK):
        kj = fr.node((me.s + dist) % fr.L)
        v[12 + j] = _clip(float(plan.kappa[kj]) * 100.0, 3.0)
        v[12 + n_look + j] = _clip((float(plan.v[kj]) * driver.pace_lap - me.v) / 30.0, 2.0)
    # A pass lasts seconds (pull out, draw alongside, settle by the apex): what
    # lets a policy carry one through is how long ago it committed to its lane
    # (``lane_age``) -- not the rule layer's own attack flags, which a learned
    # driver never sets -- plus what sets one up: the car ahead's pace against
    # this car's and the next place worth passing.
    o = 12 + 2 * n_look
    v[o] = min((driver.ticks - driver.lane_tick) * (1.0 / 60.0), 6.0) / 6.0
    edge = 0.0
    if driver.leader is not None and driver.leader in driver.lead_pace:
        edge = (driver.own_pace - driver.lead_pace[driver.leader]) * 100.0
    v[o + 1] = _clip(edge, 3.0)
    zones = driver.attack_zones
    if zones:
        arcs = getattr(driver, "_rl_zone_arcs", None)
        if arcs is None:
            arcs = driver._rl_zone_arcs = np.asarray([fr.arc[z] for z in zones])
            driver._rl_zone_drop = np.asarray(
                [float(plan.v[z]) - float(plan.v[driver.zone_apex[z]]) for z in zones])
        ahead = (arcs - float(fr.arc[k])) % fr.L
        j = int(np.argmin(ahead))
        v[o + 2] = min(float(ahead[j]), 600.0) / 400.0
        v[o + 3] = _clip(float(driver._rl_zone_drop[j]) / 50.0, 2.0)
    else:
        v[o + 2] = 1.5
        v[o + 3] = 0.0
    v[o + 4] = 1.0 if abs(float(plan.kappa[k])) < 3e-3 else 0.0
    # A pass under way (the rule layer's machinery runs it for the policy too),
    # and whether one could be started this instant, on either side.
    v[o + 5] = 1.0 if driver.attack is not None else 0.0
    can_in, can_out = driver.attack_options(me, field)
    v[o + 6] = 1.0 if can_in else 0.0
    v[o + 7] = 1.0 if can_out else 0.0
    base = len(_EGO)
    near = []
    for o in field:
        if o.idx == me.idx or o.ghost:
            continue
        gap = fr.ds(me.s, o.s)
        if -NEAR_BEHIND < gap < NEAR_AHEAD:
            near.append((abs(gap), gap, o))
    near.sort(key=lambda r: r[0])
    for i in range(N_NEAR):
        b = base + i * len(_NEAR)
        if i < len(near):
            _, gap, o = near[i]
            v[b] = _clip(gap / 100.0, 2.0)
            v[b + 1] = _clip((o.n - me.n) / 6.0, 1.5)
            v[b + 2] = _clip((o.v - me.v) / 30.0, 2.0)
            v[b + 3] = 0.0 if o.racing else 1.0
            v[b + 4] = 1.0
        else:
            v[b:b + len(_NEAR)] = 0.0
    return v


def apply_action(driver, me, a: int, field) -> None:
    """Make *driver* do action *a* from its next tick on. A lane action moves
    to the lane (clipped to the road, as the rule layer's lanes are) and takes
    the pace -- and gives up a pass under way; ATTACK_IN / ATTACK_OUT begin one
    if one can be begun (``attack_options``), else do nothing; HOLD changes
    nothing."""
    a = int(a)
    if a == HOLD:
        return
    if a >= ATTACK_IN:
        if driver.attack is None:
            driver.start_attack(me, field, inside=(a == ATTACK_IN))
        return
    lane, pace = decode(a)
    if driver.attack is not None:
        driver._end_attack(me, cooldown=2.0)
    driver.pace_mult = pace
    k = driver.frame.node(me.s)
    driver._set_lane(me.s, driver._clip_offset(k, lane) if lane else 0.0, me.v)


_ATTACK_COL = OBS_NAMES.index("attack")


def rule_action(driver, me, pace_ratio: float = 1.0, before=None) -> int:
    """The action nearest to what the rule layer has just decided -- the label
    a policy is taught from (behaviour cloning). *before* is the observation the
    decision was made from: a pass the rules were already running is HOLD, one
    they began this tick is ATTACK_IN / ATTACK_OUT, and otherwise the label is
    the lane they hold and the pace they take."""
    was = before is not None and before[_ATTACK_COL] > 0.5
    if driver.attack is not None:
        if was:
            return HOLD
        return ATTACK_IN if driver.attack_inside else ATTACK_OUT
    return encode(driver.target, pace_ratio)


class Policy:
    """A trained network behind ``RaceDriver.policy``. Greedy argmax over the
    action values; numpy only. Weights come from an ``.npz`` written by
    tools/train_raceai.py (not there yet: until it is, a driver uses the rules)."""

    def __init__(self, weights: dict):
        self.w = weights
        self._obs = np.zeros(OBS_DIM, np.float32)

    def values(self, obs: np.ndarray) -> np.ndarray:
        x = (obs - self.w["mean"]) / self.w["std"]
        for i in range(int(self.w["layers"])):
            x = x @ self.w[f"W{i}"] + self.w[f"b{i}"]
            if i < int(self.w["layers"]) - 1:
                x = np.maximum(x, 0.0)
        return x

    def __call__(self, driver, me, field) -> None:
        observe(driver, me, field, self._obs)
        apply_action(driver, me, int(np.argmax(self.values(self._obs))), field)

    @classmethod
    def load(cls, path) -> "Policy":
        z = np.load(path)
        return cls({k: z[k] for k in z.files})
