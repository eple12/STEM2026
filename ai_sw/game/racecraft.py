"""A race driver: the minimum-time plan, plus everyone else on the track.

Alone, a car should drive its solved plan (``mintime_driver``). In a field it
cannot always: someone is in the way, someone is alongside, it got hit and is
facing the barrier. This module decides, every tenth of a second, WHERE across
the road to drive and how fast -- and hands that to the same ``PlanFollower``
the solo driver uses, as a sideways offset from the plan and a speed cap.

The layers, each measured in ``tools/race_sim.py`` before it stayed:

* **Avoidance (every 0.1 s).** A handful of lanes -- "the plan's line shifted
  N metres sideways" -- are rolled forward three seconds along this driver's
  speed profile (the plan's speeds, cut where a lane's own curvature, corner
  or lane change, needs it), against every nearby car rolled forward along
  the reference plan at its own pace. Running into a car ALONGSIDE is a
  squeeze and all but forbidden; running into one AHEAD within 1.2 s makes
  the lane a queue. The plan's line is strongly preferred: this layer keeps
  cars apart, it does not try to pass.
* **Room (every tick).** With a car overlapping alongside, the lane may not
  come within a body width of it -- the hard version of the squeeze rule.
* **Following (every tick).** Never closer than it can brake to the speed
  the car ahead will have a moment later; and, close behind it, no faster at
  each point than it went there (so a slower car is matched BEFORE its slower
  corner, not by braking mid-corner with the grip already spent turning).
* **Passing.** Faster than the car ahead (by its observed pace): tuck into
  its tow down the straight -- the field gives DRS within a second -- pull out
  beside it before a heavy braking zone and draw past; ahead by the braking
  zone's settle point, take the line; not, tuck back in. A car alongside
  into a corner with its nose ahead is yielded to.
* **Incidents.** A car stopped or recovering ahead brings a yellow flag:
  lift, no passing. The car itself stops, turns (reversing off a barrier if
  it must), waits beyond the kerb -- not on the racing line -- for a gap in
  the traffic, and merges; the field ghosts it until it is clear (field.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import config
from .mintime_driver import Plan, PlanFollower, _wrap
from .vehicle import Controls, Vehicle

# --------------------------------------------------------------------------
# tuning
# --------------------------------------------------------------------------
#: Lanes considered, metres sideways from the plan's line (+ = right).
LANES = (-6.0, -4.0, -2.5, 0.0, 2.5, 4.0, 6.0)
#: Planning horizon and its sampling.
HORIZON = 3.0
PLAN_DT = 0.15
#: Re-plan every this many physics ticks.
PLAN_EVERY = 8
#: Look round at the other cars (room, following, yellow flags) every this
#: many ticks; the tracker under it runs every tick. At 60 Hz that is a
#: decision every 33 ms -- well inside a driver's reaction time -- for half
#: the cost of twenty cars all looking every tick.
THINK_EVERY = 2
#: How much road ahead the speed profiles are built over.
LOOK_NODES = 90
#: Two cars "touch" in the prediction when they are closer than this along
#: the track and across it (body length / width plus a margin).
GAP_ALONG = config.CAR_BODY_LENGTH + 1.5
GAP_ACROSS = config.CAR_BODY_WIDTH + 0.6
#: Wheel centres kept this far inside the white line by a lane.
EDGE_ROOM = config.WHEEL_HALF_TRACK + 0.25
#: Decelerations the planner assumes it can count on (m/s^2) -- less on the
#: opening lap, when following a car through a braking zone it shares with
#: nineteen others.
A_BRAKE = 20.0
A_BRAKE_OPENING = 15.0
A_ACCEL = 9.0
#: How hard a follower slows to match the car ahead's speed at a point.
A_FOLLOW = 8.0
#: Seconds ahead the car in front's speed is anticipated (its braking).
LEAD_LOOK = 0.8
#: How much slower (speed ratio) the car ahead must have been to be worth
#: attacking without having been held up by it, and how fast that estimate
#: forgets (per tick).
PACE_EDGE = 0.0025
PACE_EMA = 0.995
#: Speed lost (m/s) from a braking zone's start to its apex for it to be a
#: place to pass.
ATTACK_DROP = 18.0
#: Pull out of the tow when this close behind (m of clear road), or when the
#: braking zone is this near (m).
PULL_OUT_GAP = 10.0
PULL_OUT_DIST = 200.0
#: Only this far or more from the next braking zone does a stalking car close
#: up to tow distance.
STALK_CLEAR = 150.0
#: Metres a nose must be ahead to have the position. Alongside into a corner
#: with another car's nose this far ahead, a car yields: lifts and falls in
#: behind, as a racing driver would, rather than turning in on it.
PASS_MARGIN = 2.0
#: Below this speed (m/s) a car does not give way to one alongside it.
YIELD_MIN_V = 12.0
#: Share of its own plan's speed a following car may be held to by the speed
#: the car ahead took a point at (see _leader_cap).
TRACE_FLOOR = 0.7
#: A missed braking point is held for 0.35 s -- or, one time in LATE_BIG_P,
#: LATE_BIG_T: past the corner, into the run-off or into someone.
LATE_BIG_P = 0.3
LATE_BIG_T = 0.9
#: Metres over which a settled attack is assumed to rejoin the line.
RETURN_LEN = 60.0
#: Yellow flag: a car stopped or recovering this far ahead slows the field to
#: this share of its pace, and nobody starts a pass.
YELLOW_RANGE = 350.0
#: Pace through a yellow zone, and past the stricken car itself (the last
#: YELLOW_NEAR metres before it, and alongside). 0.90 was a lift nobody could
#: see from the cockpit; a yellow is meant to be visibly driven to. 0.75 / 0.50
#: was visible but dragged the player, who may not pass, round at a crawl.
YELLOW_PACE = 0.85
YELLOW_NEAR_PACE = 0.65
YELLOW_NEAR = 100.0
#: Inside a zone the AI holds this share of config.YELLOW_SPEED_KMH (under
#: racecontrol's tolerance), and to be there on entering it brakes along a
#: YELLOW_DECEL (m/s^2) profile from as far back as YELLOW_BRAKE_LOOK metres --
#: from 300 km/h to 100 takes about 200 m, far more than YELLOW_LEAD.
YELLOW_AI_SHARE = 0.94
YELLOW_DECEL = 18.0
YELLOW_BRAKE_LOOK = 320.0
#: Metres before a yellow zone the AI lifts, so it is slowed on entering it.
YELLOW_LEAD = 60.0
#: Under a yellow, any racing car this close ahead is followed, not passed.
YELLOW_HOLD = 60.0

#: A recovering car waits for a gap in the traffic; after RECOVER_EASE_T
#: seconds it settles for a smaller one, and after RECOVER_FORCE_T it merges.
RECOVER_EASE_T = 3.0
RECOVER_FORCE_T = 22.0
#: Seconds each phase of a recovery may take before it is retried, how many
#: retries before the marshals step in, and the most a recovery may take.
RECOVER_PHASE_T = {"stop": 6.0, "align": 8.0, "rejoin": 20.0, "merge": 10.0}
RECOVER_TRIES = 3
#: How far up the road (m) a recovering car aims, to get to the place it waits.
REJOIN_AHEAD = 14.0
RECOVER_MARSHAL_T = 30.0
#: Pace while alongside another car, passing it.
ATTACK_PACE = 1.0
#: Score weights: metres of progress a lane must promise to be worth leaving
#: the line for, per metre of offset; and to be worth changing lanes at all.
W_OFFSET = 2.5
W_SWITCH = 3.0
W_SQUEEZE = 400.0
#: A car ahead in a lane only makes the planner look elsewhere when the
#: lanes would meet within this many seconds.
IMMINENT = 1.2
#: The planning horizon's time steps, and how fast a car off the line is
#: assumed to drift back onto it over them.
_TAUS = np.arange(1, int(HORIZON / PLAN_DT) + 1) * PLAN_DT
#: Seconds a car's predicted path is reused for (TrackFrame.predict), and
#: the finer time grid it is kept on so it can be read at any age.
PRED_KEEP = 0.1
_TGRID = np.arange(0, int(round((HORIZON + PRED_KEEP) / 0.05)) + 2) * 0.05
_TAU_IDX = np.rint(_TAUS / 0.05).astype(int)
_SOON = _TAUS <= IMMINENT


@dataclass
class Skill:
    """Who is driving. Everything that makes one AI different from another."""
    plan: str = "g97"           # solved plan this driver drives (mintime tag)
    pace: float = 1.0           # multiplier on the plan's speeds
    consistency: float = 0.003  # std-dev of the per-lap pace variation
    mistakes: float = 0.0       # chance per lap of one late-braking error
    reaction: float = 0.20      # seconds from lights out to throttle
    aggression: float = 0.5     # 0..1: eagerness to pass, closeness to follow


@dataclass
class CarView:
    """What every car shows the others, refreshed once per physics tick."""
    idx: int
    s: float          # metres along the centreline, 0..L
    n: float          # metres right of the centreline
    v: float          # speed, m/s
    dev: float        # metres right of the reference plan's line
    ratio: float      # its speed over the reference plan's speed here
    racing: bool      # False while stopped, recovering or not yet started
    #: The speed it had at every centreline sample it last passed (NaN where
    #: it has not been), shared, not copied: a car following it can match
    #: the speed it actually took each corner at.
    trace: object = None
    #: Rejoining after an incident: passes through other cars, and racing cars
    #: drive as if it were not there (bar the yellow flag it brings).
    ghost: bool = False
    #: Its acceleration along its own heading, m/s^2 (negative: braking).
    a: float = 0.0


class TrackFrame:
    """Centreline arc length and lateral offset for any point, and the
    reference plan (the field's own fastest) every car is predicted with."""

    def __init__(self, track, ref: Plan):
        self.track = track
        self.ref = ref
        self.L = track.length
        self.arc = track.arclen
        self.count = track.count
        # node() runs a few dozen times per car per tick: a metre-bucket table
        # and a short walk replace a numpy searchsorted (and its call cost).
        self._arc_l = self.arc.tolist() + [self.L]
        self._bucket = (np.searchsorted(self.arc, np.arange(int(self.L) + 2),
                                        side="right") - 1).tolist()
        self._center_l = track.center.tolist()
        self._tan_l = track.tangent.tolist()
        #: Physics tick and its length, set by the field; predictions are
        #: cached against them.
        self.tick = 0
        self.dt = 1.0 / 60.0
        #: The stewards' yellow zones, (s from, s to), set by the field each
        #: step; None when nobody is keeping them (the tools' bare drivers).
        self.yellow = None
        #: Where the stricken cars are (lap distance), with the zones.
        self.stricken_s = ()
        self._pred: dict = {}
        #: How often drivers re-plan and look round (ticks). The field
        #: stretches both when it is running behind (see Field.overload);
        #: plan_every must stay a multiple of think_every.
        self.plan_every = PLAN_EVERY
        self.think_every = THINK_EVERY

    def locate(self, surface, pos) -> tuple[int, float, float]:
        """(sample, s, n) of a position."""
        i, n = surface.progress(pos)
        cx, cz = self._center_l[i]
        tx, tz = self._tan_l[i]
        s = (self._arc_l[i] + (float(pos[0]) - cx) * tx
             + (float(pos[1]) - cz) * tz) % self.L
        return i, s, n

    def ds(self, a: float, b: float) -> float:
        """b - a along the lap, wrapped to (-L/2, L/2]."""
        return (b - a + 0.5 * self.L) % self.L - 0.5 * self.L

    def node(self, s: float) -> int:
        s %= self.L
        k = self._bucket[int(s)]
        arc = self._arc_l
        while arc[k + 1] <= s:
            k += 1
        return k % self.count

    def predict(self, o: "CarView"):
        """Where car *o* will be over the planning horizon: metres ahead of
        where it is now, and across the road, at each PLAN_DT step -- rolled
        forward along the reference plan at the pace it is showing, drifting
        back towards the line. The same for every car that asks, so it is
        worked out once per car, not once per car per planner -- and reused
        for PRED_KEEP seconds, read further along by its age, since the
        planners are staggered over the ticks and would otherwise each want
        a fresh one."""
        now = self.tick * self.dt
        hit = self._pred.get(o.idx)
        if hit is not None:
            t0, racing, rel_ext, no_ext, last = hit
            age = now - t0
            if racing == o.racing and 0.0 <= age <= PRED_KEEP:
                if last[0] != self.tick:
                    if age == 0.0:
                        rel, no = rel_ext[_TAU_IDX], no_ext[_TAU_IDX]
                    else:
                        x = _TAUS + age
                        rel = np.interp(x, _TGRID, rel_ext) - np.interp(age, _TGRID, rel_ext)
                        no = np.interp(x, _TGRID, no_ext)
                    last[:] = [self.tick, rel, no]
                return last[1], last[2]
        ref = self.ref
        N = self.count
        ko = self.node(o.s)
        on = (ko + np.arange(LOOK_NODES)) % N
        oseg = self.track.seg_len[on]
        os_rel = np.concatenate(([0.0], np.cumsum(oseg[:-1])))
        if o.racing:
            ov_lim = ref.v[on] * max(min(o.ratio, 1.05), 0.0)
            ov, _ = _profile(np.maximum(ov_lim, 1.0), oseg, max(o.v, 0.5))
        else:
            ov = np.full(LOOK_NODES, max(o.v, 0.5))
        ot = _times(ov, oseg)
        rel_ext = np.interp(_TGRID, ot, os_rel)
        no_ext = np.interp(rel_ext, os_rel, ref.n_raw[on]) + o.dev * np.exp(-_TGRID / 6.0)
        rel, no = rel_ext[_TAU_IDX], no_ext[_TAU_IDX]
        self._pred[o.idx] = (now, o.racing, rel_ext, no_ext, [self.tick, rel, no])
        return rel, no


def _profile(v_lim, seg, v0, a_brake=A_BRAKE, a_accel=A_ACCEL):
    """Speeds along a run of nodes from v0: under v_lim everywhere, braking
    for what is coming, accelerating no faster than the car can.

    Both passes are closed-form running minima rather than loops:
    v_i^2 <= min_{j>=i} (v_lim_j^2 + 2a(s_j - s_i))  and the mirror image
    forward from v0.
    """
    s = np.concatenate(([0.0], np.cumsum(seg[:-1])))
    back = np.minimum.accumulate((v_lim ** 2 + 2 * a_brake * s)[::-1])[::-1]
    v2 = back - 2 * a_brake * s
    fwd = 2 * a_accel * s + np.minimum.accumulate(
        np.concatenate(([v0 * v0], (v2 - 2 * a_accel * s)[1:])))
    return np.sqrt(np.clip(np.minimum(v2, fwd), 1.0, None)), s


#: Lateral acceleration a line may always count on, m/s^2 -- what a straight
#: offers a lane change (the car manages ~4 g at speed; this is well inside).
A_LAT_FLOOR = 30.0


def _second_diff(off, s):
    """d2(off)/ds2 along each row, as np.gradient applied twice would give it
    (to within the end points), at a fraction of the call overhead."""
    h = np.diff(s)
    d1m = np.diff(off, axis=1) / h[None, :]           # slopes between nodes
    d1 = np.empty_like(off)
    d1[:, 1:-1] = 0.5 * (d1m[:, 1:] + d1m[:, :-1])
    d1[:, 0] = d1m[:, 0]
    d1[:, -1] = d1m[:, -1]
    d2m = np.diff(d1, axis=1) / h[None, :]
    d2 = np.empty_like(off)
    d2[:, 1:-1] = 0.5 * (d2m[:, 1:] + d2m[:, :-1])
    d2[:, 0] = d2m[:, 0]
    d2[:, -1] = d2m[:, -1]
    return d2


def _path_speed(v_plan, kap, off, s_rel):
    """Speed limit along lines beside the plan.

    The plan's speed at each point is the speed its own curvature allows.
    A line ``off`` metres to the side has a different curvature -- the
    corner at another radius, PLUS the bend of any lane change along it
    (the offset's second derivative) -- and the lateral grip the plan was
    using there is what that curvature must be taken with. A 4 m lane
    change over 60 m bends the path about as much as Lesmo does, so a move
    made mid-corner must cost speed or the car leaves the road.
    """
    off = np.atleast_2d(off)
    d2 = _second_diff(off, s_rel)
    k_eff = kap[None, :] / np.clip(1.0 - kap[None, :] * off, 0.3, None) + d2
    a_lat = np.maximum(v_plan ** 2 * np.abs(kap), A_LAT_FLOOR)
    v = np.sqrt(a_lat[None, :] / np.maximum(np.abs(k_eff), 1e-6))
    return np.minimum(v_plan[None, :], v)


def _times(v, seg):
    vm = np.maximum(0.5 * (v[:-1] + v[1:]), 1.0)
    return np.concatenate(([0.0], np.cumsum(seg[:-1] / vm)))


def _profile_rows(v_lim, s, v0, a_brake=A_BRAKE, a_accel=A_ACCEL):
    """_profile for several speed limits at once (one per row), all from v0
    over the same run of nodes (cumulative distances *s*)."""
    back = np.minimum.accumulate((v_lim ** 2 + 2 * a_brake * s)[:, ::-1],
                                 axis=1)[:, ::-1]
    v2 = back - 2 * a_brake * s
    head = v2 - 2 * a_accel * s
    head[:, 0] = v0 * v0
    fwd = 2 * a_accel * s + np.minimum.accumulate(head, axis=1)
    return np.sqrt(np.clip(np.minimum(v2, fwd), 1.0, None))


def _times_rows(v, seg):
    vm = np.maximum(0.5 * (v[:, :-1] + v[:, 1:]), 1.0)
    t = np.zeros_like(v)
    np.cumsum(seg[None, :-1] / vm, axis=1, out=t[:, 1:])
    return t


class RaceDriver:
    def __init__(self, track, frame: TrackFrame, plan: Plan, skill: Skill,
                 rng: np.random.Generator):
        self.track = track
        self.frame = frame
        self.plan = plan
        self.follow = PlanFollower(plan)
        self.follow.track = track
        self.skill = skill
        self.rng = rng
        self.idx = -1
        # Staggered, so the field's re-plans spread over the ticks between
        # them instead of all landing on the same one.
        self.ticks = int(rng.integers(PLAN_EVERY))
        self._ocap = None
        #: Caps decided on the last thinking tick (THINK_EVERY), and the
        #: braking envelope per pace.
        self._held = None
        self._is_held = False
        self._env: dict = {}
        self.mode = "grid"
        #: The lane: offset from the plan line moving from d0 (at s0) to d1
        #: (at s1), cosine-blended, held at d1 after.
        self.lane = (0.0, 0.0, 1.0, 0.0)
        self.target = 0.0
        self.pace_lap = skill.pace
        self.lap_seen = -1
        self.late_at = None          # node where this lap's mistake happens
        self.late_left = 0.0
        self.leader = None           # car being followed, for the record
        # recovery state
        self.rec_phase = ""
        self.rec_t = 0.0
        self.stuck_t = 0.0
        self.off_t = 0.0
        self.slow_t = 0.0
        self.reverse_t = 0.0
        self.recoveries = 0
        #: Rejoin guarantee (see _recover_watchdog): a phase that has run out
        #: of time is retried with an unsticking manoeuvre, and after
        #: RECOVER_TRIES of those, or RECOVER_MARSHAL_T in all, the marshals
        #: put the car back on the road (the field does it: ``marshal``).
        self.marshal = False
        self.stuck_n = 0
        self._goto_rev = 0.0
        self._goto_stuck = 0.0
        self._unstick_t = 0.0
        self._unstick_dir = 1.0
        self._ph_t = 0.0
        self._ph_last = ""
        self.rec_cause = ""
        self.rec_from = ""
        self.mistakes_made = 0
        # Braking zones (where the plan starts braking hard), for mistakes and
        # for choosing where to attack: each with the apex it brakes for.
        fb = plan.fb
        hard = fb > 0.5 * config.BRAKE_FORCE
        self.brake_zones = np.flatnonzero(hard & ~np.roll(hard, 1))
        n = plan.n
        #: Metres from each centreline sample to the next braking zone.
        arc = frame.arc
        zs = np.sort(self.brake_zones)
        nxt = np.searchsorted(zs, np.arange(n), side="left")
        z_next = zs[nxt % len(zs)] if len(zs) else np.zeros(n, int)
        self.to_brake = ((arc[z_next] - arc) % frame.L) if len(zs) \
            else np.full(n, np.inf)
        self.zone_apex = {}
        #: Zones worth attacking into: real stops, where braking later is
        #: worth metres. A dive into a fast corner just costs the inside line's
        #: tighter radius, with nothing to gain on the brakes.
        self.attack_zones = []
        #: Share of the way from a zone's braking point to its apex by which a
        #: pass must be done -- earlier into a chicane, whose first part is no
        #: place to be side by side.
        self.zone_settle = {}
        for z in self.brake_zones:
            win = (z + np.arange(60)) % n
            apex = int(win[int(np.argmin(plan.v[win]))])
            self.zone_apex[int(z)] = apex
            span = (z + np.arange(int((apex - z) % n) + 16)) % n
            k_span = plan.kappa[span]
            k_span = k_span[np.abs(k_span) > 4e-3]
            chicane = len(k_span) and (k_span.max() > 0) and (k_span.min() < 0)
            self.zone_settle[int(z)] = 0.6 if chicane else 0.9
            if plan.v[z] - plan.v[apex] > ATTACK_DROP:
                self.attack_zones.append(int(z))
        # attacking
        self.held_t = 0.0            # seconds held up behind a slower car
        self.attack = None           # (target idx, side, apex s) while passing
        self.attack_inside = True    # ...and whether that is the inside of the corner
        self.cooldown = 0.0
        self.attacks = 0
        self.passes_made = 0
        self.yellow = False
        self.abs_lane = None         # (s start, offset start, n) while diving inside
        #: What this driver's plan is worth against the reference plan (mean
        #: speed ratio), and what the car ahead has been showing, smoothed.
        self.own_pace = float(np.mean(plan.v / np.maximum(frame.ref.v, 1.0))) \
            * skill.pace
        self.lead_pace: dict = {}
        #: The decision layer can be replaced (raceai.py). ``policy(driver, me,
        #: field)`` is called on this driver's plan ticks instead of the rule
        #: based _plan / _attack_lane: it sets the lane (``_set_lane``) and
        #: ``pace_mult``. The safety layers under it -- the room rule, the
        #: leader cap, the yellow flag, recovery -- stay the rules' own.
        self.policy = None
        self.pace_mult = 1.0
        #: Tick of the last lane decision (a lane change, or the start of a dive
        #: down the inside): how long ago the car committed to where it is.
        self.lane_tick = 0
        #: ``watch(driver, me, field, pace_ratio, before)``, called after each
        #: plan tick with what this driver decided (``before``: whatever
        #: ``watch.before(driver, me, field)`` returned ahead of the decision): how a policy is taught to do
        #: what the rules do (behaviour cloning), and how the rules are
        #: measured. Nothing changes for a driver without either.
        self.watch = None

    # -- the lane --------------------------------------------------------
    def _lane_at(self, s: float) -> tuple[float, float, float]:
        s0, d0, s1, d1 = self.lane
        if d0 == d1:
            return d1, 0.0, 0.0
        span = max(self.frame.ds(s0, s1), 1e-3)
        x = self.frame.ds(s0, s) / span
        if x <= 0.0:
            return d0, 0.0, 0.0
        if x >= 1.0:
            return d1, 0.0, 0.0
        c = math.cos(math.pi * x)
        d = d0 + (d1 - d0) * 0.5 * (1.0 - c)
        dd = (d1 - d0) * 0.5 * math.pi * math.sin(math.pi * x) / span
        ddd = (d1 - d0) * 0.5 * math.pi ** 2 * c / span ** 2
        return d, dd, ddd

    # An attack does not follow the plan's line at an offset: it dives down
    # the INSIDE of the corner at a fixed place across the road, arriving at
    # the apex where the plan's line arrives too. (A constant offset from the
    # plan is off the road either at the apex, inside, or on the approach,
    # outside -- the line sweeps from one edge to the other.)
    ABS_BLEND = 70.0          # m (at least) over which the car moves across onto it

    def _abs_offset(self, s: float, n_abs: float) -> float:
        k = self.frame.node(s)
        f = (self.frame.ds(self.frame.arc[k], s)
             / max(float(self.track.seg_len[k]), 1e-3))
        n_line = self.plan.at(self.plan.n_raw, k, min(max(f, 0.0), 1.0))
        return n_abs - float(n_line)

    def _offset_at(self, s: float) -> tuple[float, float, float]:
        """(offset from the plan's line, its first and second derivative)."""
        if self.abs_lane is None:
            return self._lane_at(s)
        s0, n0, n_abs, blend = self.abs_lane

        # The move across is made in ROAD terms -- from where the car was to
        # the inside -- not as an offset from the line: the line itself swings
        # across the road through a chicane, and blending offsets from it sent
        # the car nine metres the wrong way and back.
        def off(x):
            w = min(max(self.frame.ds(s0, x) / blend, 0.0), 1.0)
            w = 0.5 * (1.0 - math.cos(math.pi * w))
            return self._abs_offset(x, n0 + (n_abs - n0) * w)
        h = 2.5
        a, b, c = off(s - h), off(s), off(s + h)
        return b, (c - a) / (2 * h), (c - 2 * b + a) / (h * h)

    def _offset_profile(self, s0_abs: float, s_rel: np.ndarray,
                        nodes: np.ndarray) -> np.ndarray:
        """The offset along the run ahead (vectorised _offset_at)."""
        if self.abs_lane is None:
            return self._lane_profile(s0_abs + s_rel)
        s0, n0, n_abs, blend = self.abs_lane
        return self._attack_profile(s0_abs, s_rel, nodes, s0, n0, n_abs, blend)

    def _attack_profile(self, s0_abs, s_rel, nodes, s0, n0, n_abs, blend):
        """An attack's offset along the run ahead, AS FAR AS SPEED IS
        CONCERNED: out to the side, and back onto the line after the point
        the pass is settled by -- because either way, that is what happens
        there. Assuming the side lane ran on through the corner had the
        attacker braking for a corner it would never take on that line, the
        moment it pulled out of the tow."""
        L = self.frame.L
        x = ((s0_abs + s_rel - s0 + 0.5 * L) % L - 0.5 * L) / blend
        w = 0.5 * (1.0 - np.cos(np.pi * np.clip(x, 0.0, 1.0)))
        off = (n0 + (n_abs - n0) * w) - self.plan.n_raw[nodes]
        if self.attack is not None:
            settle = self.attack[2]
            y = ((s0_abs + s_rel - settle + 0.5 * L) % L - 0.5 * L) / RETURN_LEN
            off = off * 0.5 * (1.0 + np.cos(np.pi * np.clip(y, 0.0, 1.0)))
        return off

    def _set_lane(self, s: float, target: float, speed: float):
        if abs(target - self.lane[3]) < 1e-6:
            return              # already on (or moving to) this lane
        d_now, _, _ = self._lane_at(s)
        length = min(max(1.1 * speed, 30.0), 90.0)
        self.lane = (s, d_now, (s + length) % self.frame.L, target)
        self.target = target
        self.lane_tick = self.ticks

    def _clip_offset(self, k: int, d: float) -> float:
        """Keep a lane's wheels inside the white lines where it runs. The
        plan's own line is never clipped: it was solved to its own (tighter)
        margin and pulling it in would cost time at every apex."""
        if d == 0.0:
            return 0.0
        n = self.plan.n_raw[k]
        lo = min(-self.track.w_left[k] + EDGE_ROOM - n, 0.0)
        hi = max(self.track.w_right[k] - EDGE_ROOM - n, 0.0)
        return float(min(max(d, lo), hi))

    # -- planning -------------------------------------------------------------
    def _plan(self, me: CarView, field: list[CarView]):
        fr = self.frame
        pl = self.plan
        N = fr.count
        k0 = fr.node(me.s)
        nodes = (k0 + np.arange(LOOK_NODES)) % N
        seg = self.track.seg_len[nodes]
        s_rel = np.concatenate(([0.0], np.cumsum(seg[:-1])))
        kap = pl.kappa[nodes]
        v_plan = pl.v[nodes] * self.pace_lap
        lo = np.minimum(-self.track.w_left[nodes] + EDGE_ROOM - pl.n_raw[nodes], 0.0)
        hi = np.maximum(self.track.w_right[nodes] - EDGE_ROOM - pl.n_raw[nodes], 0.0)
        taus = _TAUS

        d_now = self._offset_at(me.s)[0]
        attack_n = self._attack_lane(me, field)
        cands = sorted(set(LANES) | {round(self.target, 2)})
        C = len(cands)
        # Each lane's offset along the run ahead: a cosine move from where the
        # car is now to the lane over the usual lane-change length.
        length = min(max(1.1 * me.v, 30.0), 90.0)
        x = np.clip(s_rel / length, 0.0, 1.0)
        blend = 0.5 * (1.0 - np.cos(np.pi * x))
        off = d_now + (np.asarray(cands)[:, None] - d_now) * blend[None, :]
        if attack_n is not None:
            # One more row: the dive down the inside, at its place across the
            # road rather than at an offset from the line -- continuing the
            # move already under way if there is one.
            if self.abs_lane is not None and abs(self.abs_lane[2] - attack_n) < 1e-6:
                att = self._offset_profile(me.s, s_rel, nodes)
            else:
                att = self._attack_profile(me.s, s_rel, nodes, me.s, me.n,
                                           attack_n,
                                           max(self.ABS_BLEND, 2.0 * me.v))
            off = np.vstack([off, att[None, :]])
            C += 1
        off = np.clip(off, lo[None, :], hi[None, :])
        v_lim = _path_speed(v_plan, kap, off, s_rel)

        s_me = np.empty((C, len(taus)))
        n_me = np.empty((C, len(taus)))
        t_all = _times_rows(_profile_rows(v_lim, s_rel, max(me.v, 1.0)), seg)
        line = pl.n_raw[nodes]
        for c in range(C):
            s_at = np.interp(taus, t_all[c], s_rel)
            s_me[c] = s_at
            n_me[c] = np.interp(s_at, s_rel, line + off[c])

        # Everyone nearby, rolled forward along the reference plan at the
        # pace they are showing, drifting back towards its line.
        progress = s_me[:, -1].copy()
        squeeze = np.zeros(C)
        gaps, rels, nos, vs = [], [], [], []
        for o in field:
            if o.idx == me.idx or o.ghost:
                continue
            gap0 = fr.ds(me.s, o.s)
            if gap0 < -GAP_ALONG or gap0 > 160.0:
                continue
            # Where across the road: back towards the line, but slowly -- a
            # car off the line now is more likely still off it in a second
            # than not (TrackFrame.predict).
            rel, no = fr.predict(o)
            gaps.append(gap0)
            rels.append(rel)
            nos.append(no)
            vs.append(o.v)
        if gaps:
            # Every lane against every car at every step, at once: (C, M, T).
            g0 = np.asarray(gaps)
            so = g0[:, None] + np.asarray(rels)
            no = np.asarray(nos)
            close = (np.abs(s_me[:, None, :] - so[None, :, :]) < GAP_ALONG) & \
                (np.abs(n_me[:, None, :] - no[None, :, :]) < GAP_ACROSS)
            ahead = g0 > GAP_ALONG * 0.8
            if ahead.any():
                # Ahead: this lane means queueing behind it -- which only
                # counts against the lane when the queueing is imminent.
                # Further back the per-tick leader cap does the following, and
                # passing is the attack's job: leaving the line to "get round"
                # a car still half a second up the road only threw away the
                # corner (it was costing a faster car more than it gained).
                hit = close[:, ahead][:, :, _SOON].any(axis=2)          # (C, Ma)
                if hit.any():
                    headway = 0.55 - 0.3 * self.skill.aggression
                    cap = so[ahead, -1] - (GAP_ALONG + headway
                                           * np.maximum(np.asarray(vs)[ahead], 10.0))
                    capped = np.where(hit, cap[None, :], np.inf).min(axis=1)
                    progress = np.minimum(progress, capped)
            side = ~ahead
            if side.any():
                # Alongside: moving into it is a squeeze.
                cs = close[:, side]                                      # (C, Ms, T)
                any_ = cs.any(axis=2)
                first = np.argmax(cs, axis=2)
                T = len(taus)
                squeeze += np.where(any_, 1.0 + (T - first) / T, 0.0).sum(axis=1)

        if attack_n is not None:
            if squeeze[-1] > 0.0:
                # The gap closed (someone alongside, or the car being passed
                # moved across): give it up rather than lean on anyone.
                self._end_attack(me, cooldown=2.5)
            else:
                if self.abs_lane is None or abs(self.abs_lane[2] - attack_n) > 1e-6:
                    self.abs_lane = (me.s, me.n, attack_n, max(self.ABS_BLEND, 2.0 * me.v))
                    self.lane_tick = self.ticks
                return
        rel = slice(0, len(cands))
        keen = 0.6 + 0.8 * self.skill.aggression
        score = (keen * progress[rel]
                 - W_OFFSET * np.abs(np.asarray(cands))
                 - W_SWITCH * (np.abs(np.asarray(cands) - self.target) > 1e-6)
                 - W_SQUEEZE * squeeze[rel])
        best = int(np.argmax(score))
        target = float(cands[best])
        self._set_lane(me.s, self._clip_offset(k0, target) if target else 0.0, me.v)

    def _policy_tick(self, me: CarView, field: list[CarView]):
        """A plan tick of a driver whose decisions are a policy's: a pass it
        began carries on (and ends) by the rule layer's own machinery, then the
        policy decides -- continue it, give it up, or something else."""
        if self.attack is not None:
            n_abs = self._attack_continue(me, field)
            if n_abs is not None and (self.abs_lane is None
                                      or abs(self.abs_lane[2] - n_abs) > 1e-6):
                self.abs_lane = (me.s, me.n, n_abs, max(self.ABS_BLEND, 2.0 * me.v))
        self.policy(self, me, field)

    # -- passing -------------------------------------------------------------
    #: Sideways distance kept from the car being passed: a body width plus room.
    ATTACK_SEP = GAP_ACROSS + 0.4
    #: Seconds of being held up before a pass is tried, and how close (s) the
    #: car ahead must be.
    HELD_TRIGGER = 0.35
    ATTACK_GAP = 0.8
    #: The pull-out happens this far before the braking zone (m): late
    #: enough to have had the tow down the straight, early enough to be
    #: alongside by the braking point.
    ATTACK_FROM = 60.0
    ATTACK_TO = 400.0

    def _end_attack(self, me: CarView, cooldown: float):
        """Back to the plan's line from wherever the attempt left the car."""
        if self.abs_lane is not None:
            d = self._offset_at(me.s)[0]
            self.abs_lane = None
            # Back to the line slowly: this usually happens on a corner's
            # exit, with the line running away to the outside, and the usual
            # quick lane change there bent the path past the tyres.
            length = max(150.0, 2.0 * me.v)
            self.lane = (me.s, d, (me.s + length) % self.frame.L, 0.0)
            self.target = 0.0
        self.attack = None
        self.cooldown = cooldown
        self.held_t = 0.0

    def _attack_lane(self, me: CarView, field: list[CarView]):
        """Where across the road to pass, or None.

        The pass this racing allows is the one F1's is built around: the
        faster car closes up in the tow of a slower one down a straight,
        pulls out beside it -- a car's width over, on the side of the next
        corner's inside where there is room -- and with DRS (see field.py)
        draws past before the braking zone. Ahead by the braking point: back
        onto the line in front. Not: brake at its own point and tuck back in
        behind. No dive up the inside into the corner itself -- tried, and on
        this car the inside line's tighter radius cost more on the brakes than
        any pass could gain, and through chicanes it put cars off.
        """
        fr = self.frame
        views = {o.idx: o for o in field}
        if self.attack is not None:
            return self._attack_continue(me, field)
        if self.cooldown > 0.0 or self.leader is None or self.yellow:
            return None
        o = views.get(self.leader)
        if o is None or not o.racing:
            return None
        # Worth a go when held up, or when the car ahead has been plainly
        # slower than this one could be (its observed pace against the
        # reference plan, against this driver's own) -- being held only ever
        # registered in the braking zones, and had decayed again by the end
        # of every straight, where a pass actually starts.
        slower = self.lead_pace.get(o.idx, 1.0) < self.own_pace - PACE_EDGE
        if self.held_t < self.HELD_TRIGGER and not slower:
            return None
        gap = fr.ds(me.s, o.s)
        if gap <= 0.0 or gap > self.ATTACK_GAP * max(me.v, 10.0):
            return None
        # The next braking zone, and the corner it is for -- and only from a
        # straight: pulling out mid-corner to set up the next one is how the
        # first version of this put cars off the road at Lesmo.
        k = fr.node(me.s)
        if abs(float(self.plan.kappa[k])) > 3e-3:
            return None
        best = self._attack_zone(k)
        if best is None:
            return None
        # Stay in the tow until it has done its work -- right up behind, or
        # the braking zone near -- then pull out.
        if gap - GAP_ALONG > PULL_OUT_GAP and best[0] > PULL_OUT_DIST:
            return None
        return self._attack_begin(o, best[1], (None,))

    def _attack_zone(self, k: int):
        """(metres, zone) of the next braking zone worth passing into, within
        the window a pull-out is made from; None if there is none."""
        fr = self.frame
        best = None
        for z in self.attack_zones:
            dist = (fr.arc[z] - fr.arc[k]) % fr.L
            if self.ATTACK_FROM < dist < self.ATTACK_TO \
                    and (best is None or dist < best[0]):
                best = (dist, int(z))
        return best

    def _attack_begin(self, o: CarView, z: int, sides, commit: bool = True):
        """Where beside car *o* to pass it for the corner of braking zone *z*,
        or None: the inside of the corner if there is room there between it and
        the white line, else the outside. *sides*: which to try -- ``(None,)`` is
        inside then outside, ``(True,)`` the inside only, ``(False,)`` the
        outside only. *commit* False only looks."""
        fr = self.frame
        apex = self.zone_apex[z]
        inside = 1.0 if self.plan.kappa[apex] > 0.0 else -1.0
        ko = fr.node(o.s)
        lo = -self.track.w_left[ko] + EDGE_ROOM
        hi = self.track.w_right[ko] - EDGE_ROOM
        # Settled by the middle of the braking zone: alongside under braking
        # is a pass the later braker can still make; past that, the corner.
        settle = float(fr.arc[z]) + self.zone_settle[z] * float(
            (fr.arc[apex] - fr.arc[z]) % fr.L)
        order = {(None,): (inside, -inside), (True,): (inside,), (False,): (-inside,)}[tuple(sides)]
        for side in order:
            n_abs = o.n + side * self.ATTACK_SEP
            if lo <= n_abs <= hi:
                if commit:
                    self.attack = (o.idx, side, settle % fr.L, n_abs)
                    self.attack_inside = side == inside
                    self.attacks += 1
                return n_abs
        return None

    def _attack_continue(self, me: CarView, field: list[CarView]):
        """A pass under way: its lane, or None once it is over -- done, given
        up, or no longer possible -- having ended it."""
        fr = self.frame
        views = {o.idx: o for o in field}
        tgt, side, brake_s, n_abs = self.attack
        o = views.get(tgt)
        if o is None or not o.racing or self.yellow:
            self._end_attack(me, cooldown=0.5)
            return None
        gap = fr.ds(me.s, o.s)
        if gap < -PASS_MARGIN:
            # Nose clearly ahead: the other car yields from here (see
            # _leader_cap), so the position is this car's.
            self._end_attack(me, cooldown=1.0)
            self.passes_made += 1
            return None
        past = fr.ds(brake_s, me.s)
        if past > 0.0 and gap > PASS_MARGIN:
            self._end_attack(me, cooldown=4.0)      # did not: tuck in
            return None
        if past > 60.0:
            # Still side by side well into the corner: settle it on the
            # road (the room rule keeps the two apart), not in the lane.
            self._end_attack(me, cooldown=3.0)
            return None
        return n_abs

    def attack_options(self, me: CarView, field: list[CarView]):
        """(inside possible, outside possible): could a pass be started right
        now -- a car close ahead, a straight, a braking zone in the window and
        room beside it? What the rule layer's own triggers (being held up, the
        tow, the cooldown) say is for whoever decides to leave out."""
        if self.attack is not None or self.leader is None or self.yellow:
            return False, False
        views = {o.idx: o for o in field}
        o = views.get(self.leader)
        if o is None or not o.racing:
            return False, False
        fr = self.frame
        gap = fr.ds(me.s, o.s)
        if gap <= 0.0 or gap > 1.6 * self.ATTACK_GAP * max(me.v, 10.0):
            return False, False
        k = fr.node(me.s)
        if abs(float(self.plan.kappa[k])) > 3e-3:
            return False, False
        best = self._attack_zone(k)
        if best is None:
            return False, False
        return (self._attack_begin(o, best[1], (True,), commit=False) is not None,
                self._attack_begin(o, best[1], (False,), commit=False) is not None)

    def start_attack(self, me: CarView, field: list[CarView], inside: bool) -> bool:
        """Begin a pass on the car ahead, on the inside or the outside of the
        coming corner, if ``attack_options`` says it can be; True if it began."""
        ins, out = self.attack_options(me, field)
        if not (ins if inside else out):
            return False
        views = {o.idx: o for o in field}
        o = views[self.leader]
        best = self._attack_zone(self.frame.node(me.s))
        n_abs = self._attack_begin(o, best[1], (inside,))
        if n_abs is None:
            return False
        self.abs_lane = (me.s, me.n, n_abs, max(self.ABS_BLEND, 2.0 * me.v))
        self.lane_tick = self.ticks
        return True

    # -- every tick ------------------------------------------------------
    def _yellow(self, me: CarView, field: list[CarView]) -> bool:
        """Under a yellow flag: inside one of the stewards' zones, or within
        YELLOW_LEAD of entering one (to have lifted by the time it starts).
        The same zones race control judges overtaking and slowing in, so
        the AI cannot be racing where it would be penalised for it."""
        zones = self.frame.yellow
        if zones is not None:
            L = self.frame.L
            for a, b in zones:
                if (me.s - a + YELLOW_LEAD) % L <= (b - a) % L + YELLOW_LEAD:
                    return True
            return False
        # No stewards: a car stopped or recovering on the road ahead.
        for o in field:
            if o.idx == me.idx or o.racing:
                continue
            gap = self.frame.ds(me.s, o.s)
            if -10.0 < gap < YELLOW_RANGE and abs(o.n) < 12.0:
                return True
        return False

    def _yellow_cap(self, me: CarView) -> float:
        """The speed a yellow flag allows here: the limit inside a zone, and
        before one the speed from which the limit can still be reached by its
        start. Unbounded when there is no zone near."""
        zones = self.frame.yellow
        if not zones:
            return float("inf")
        L = self.frame.L
        v_lim = config.YELLOW_SPEED_KMH / 3.6 * YELLOW_AI_SHARE
        best = float("inf")
        for a, b in zones:
            if (me.s - a) % L <= (b - a) % L:
                return v_lim
            ahead = (a - me.s) % L
            if ahead < YELLOW_BRAKE_LOOK:
                best = min(best, math.sqrt(v_lim * v_lim + 2.0 * YELLOW_DECEL * ahead))
        return best

    def _room(self, me: CarView, field: list[CarView], k: int, d: float) -> float:
        """Clamp the lane so it never closes on a car alongside.

        The planner already scores a squeeze out, but only ten times a
        second and against predictions; this is the hard rule, every tick,
        against where the cars actually are: with a car overlapping on one
        side, the lane may not come within a body width (plus room) of it.
        """
        # Only the cars bound it here; the white lines were already applied by
        # _clip_offset (which leaves the plan's own line alone).
        lo, hi = -math.inf, math.inf
        for o in field:
            if o.idx == me.idx or o.ghost:
                continue
            if abs(self.frame.ds(me.s, o.s)) > GAP_ALONG + 1.0:
                continue
            if o.n > me.n:
                hi = min(hi, o.n - GAP_ACROSS - 0.2)
            else:
                lo = max(lo, o.n + GAP_ACROSS + 0.2)
        n_raw = float(self.plan.n_raw[k])
        want = n_raw + d
        if lo > hi:
            # Boxed in on both sides: stay where it is.
            out = me.n - n_raw
        elif lo <= want <= hi:
            return d
        else:
            out = min(max(want, lo), hi) - n_raw
        # ...but never by leaving the road: off it, nobody is any safer.
        return self._clip_offset(k, out) if out else 0.0

    def _leader_cap(self, me: CarView, field: list[CarView], d_now: float) -> float:
        """Speed that keeps the gap to the car ahead in this lane closable."""
        fr = self.frame
        best = None
        k_me = fr.node(me.s)
        cornering = (self.to_brake[k_me] < 120.0
                     or abs(float(self.plan.kappa[k_me])) > 2e-3)
        for o in field:
            if o.idx == me.idx or o.ghost:
                continue
            gap = fr.ds(me.s, o.s)
            if gap <= 0.0 or gap > 120.0:
                continue
            ko = fr.node(o.s)
            my_n_there = self.plan.n_raw[ko] + self._offset_at(o.s)[0]
            in_lane = abs(o.n - my_n_there) <= GAP_ACROSS
            # Alongside into a corner with its nose ahead: give it the corner.
            # Not when crawling: at a walking pace two cars a car's width
            # apart can go on side by side, and yielding to each other there
            # stopped a whole grid dead in the first chicane.
            yielding = (cornering and me.v > YIELD_MIN_V
                        and PASS_MARGIN < gap < GAP_ALONG + 3.0
                        and abs(o.n - me.n) < GAP_ACROSS + 1.5 and o.racing)
            # Under a yellow, no passing: a racing car just ahead is followed
            # whatever lane it is in -- one that has moved aside to give the
            # incident room is still a car not to be overtaken. (The stricken
            # car itself is not racing, and may be passed.)
            held = self.yellow and o.racing and gap < YELLOW_HOLD
            if not (in_lane or yielding or held):
                continue
            if best is None or gap < best[0]:
                best = (gap, o)
        self.leader = None if best is None else best[1].idx
        if best is None:
            return math.inf
        gap, o = best
        if o.racing and o.v > 20.0:
            prev = self.lead_pace.get(o.idx, o.ratio)
            self.lead_pace[o.idx] = PACE_EMA * prev + (1.0 - PACE_EMA) * o.ratio
        headway = 0.55 - 0.3 * self.skill.aggression
        # The opening lap: the whole field in one braking zone, with cars that
        # brake at different points -- give it more room than a race does.
        opening = self.lap_seen <= 0
        a_brake = A_BRAKE_OPENING if opening else A_BRAKE
        if opening:
            headway += 0.12
        stalking = (self.held_t > 0.0 or self.lead_pace.get(o.idx, 1.0)
                    < self.own_pace - PACE_EDGE)
        k_me = self.frame.node(me.s)
        towing = (stalking and abs(float(self.plan.kappa[k_me])) < 2e-3
                  and self.to_brake[k_me] > STALK_CLEAR)
        if towing:
            # Faster than the car ahead, on a straight: close right up into
            # its tow -- that is where a pass starts. Not into the braking
            # zone though: thirteen metres behind a car braking from 300 km/h
            # is a rear-ending (lap one at T1 was full of them).
            headway = min(headway, 0.08)
        room = gap - GAP_ALONG
        want = headway * me.v
        # Hard limit: able to come down, in the room left, to the speed it will
        # have a moment from now -- it is about to brake for the corner ahead
        # of it, at its own (possibly earlier) braking point, and its current
        # speed says nothing about that. Lap one into T1 was a string of
        # rear-endings until this looked ahead.
        ref = self.frame.ref
        k_soon = self.frame.node(o.s + LEAD_LOOK * max(o.v, 5.0))
        v_soon = min(o.v, float(ref.v[k_soon]) * min(max(o.ratio, 0.5), 1.05)) \
            if o.racing else o.v
        if o.racing and o.a < -5.0:
            # Braking already, and harder or earlier than the reference plan
            # would: believe what it is doing over what it might do.
            v_soon = min(v_soon, max(o.v + o.a * LEAD_LOOK, 0.0))
        # ...where it will be by then, too: it slows to v_soon some forty
        # metres further up the road, not where it is now. Leaving that out
        # had a faster car braking a second early behind every slower one.
        travel = LEAD_LOOK * 0.5 * (o.v + v_soon) if o.racing else 0.0
        cap = math.sqrt(max(v_soon * v_soon
                            + 2.0 * a_brake * max(room - 1.0 + travel, 0.0), 0.0))
        cap = min(cap, math.sqrt(max(o.v * o.v + 2.0 * a_brake
                                     * max(room - 1.0, 0.0), 0.0)) + 10.0)
        if self.attack is not None and self.attack[0] == o.idx:
            # Pulling out to pass it: only "do not hit it" applies, not the
            # following gap -- dropping back to a following distance the
            # moment the move began threw away the run the tow had built.
            return max(cap, 0.0)
        # Following proper: take each point between here and it no faster than
        # it took that point itself, slowing into it gently. That speed was
        # driven on (nearly) this line moments ago, so it is always possible,
        # and matching it means slowing BEFORE the corner the car ahead is
        # slower in -- not braking in the middle of it, with every bit of
        # grip already spent turning, and running wide.
        # Only while actually close: a car a second up the road will have left
        # a corner long before this one gets there, and matching its speed
        # through it from that far back just loses the second it had.
        # Nor on a straight when closing in for a pass: there is no corner to
        # protect, and matching its speed is exactly what stops the gap from
        # ever closing.
        if o.trace is not None and room < 2.0 * want + 10.0 and not towing:
            fr = self.frame
            k0, k1 = fr.node(me.s), fr.node(o.s)
            pl_v = self.plan.v * self.pace_lap
            n_ahead = (k1 - k0) % fr.count
            if 0 < n_ahead < 40:
                nodes = (k0 + 1 + np.arange(n_ahead)) % fr.count
                v_tr = o.trace[nodes]
                ok = np.isfinite(v_tr)
                if ok.any():
                    # What the car ahead did at a point is worth matching
                    # where the CORNER made it slow, not where traffic did:
                    # copied, a crawl at one spot became every car's crawl at
                    # that spot, long after what caused it had gone -- a
                    # whole queue stopping and starting at the same place on
                    # a clear road. So no lower than most of what this car's
                    # own plan allows there.
                    v_tr = np.maximum(v_tr, TRACE_FLOOR * pl_v[nodes])
                    dist = (fr.arc[nodes[ok]] - me.s) % fr.L
                    v_now = np.sqrt(v_tr[ok] ** 2 + 2.0 * A_FOLLOW * dist)
                    cap = min(cap, float(v_now.min()) + 0.5 * (room - want))
        else:
            cap = min(cap, o.v + 0.8 * (room - want) + 2.0)
        return max(cap, 0.0)

    def _new_lap(self, lap: int):
        if lap == self.lap_seen:
            return
        self.lap_seen = lap
        sk = self.skill
        # Lap-to-lap variation only ever slower than the plan: the plan is
        # already the driver's limit, and a lap "0.3% faster" than it is a lap
        # with no grip left -- it was running cars wide out of Parabolica.
        self.pace_lap = sk.pace * (1.0 - abs(self.rng.normal(0.0, sk.consistency)))
        self.late_at = None
        if sk.mistakes > 0.0 and self.rng.random() < sk.mistakes and len(self.brake_zones):
            self.late_at = int(self.rng.choice(self.brake_zones))

    def controls(self, vehicle: Vehicle, me: CarView, field: list[CarView],
                 t_race: float, lap: int, dt: float) -> Controls:
        self.ticks += 1
        if self.mode == "grid":
            if t_race < self.skill.reaction:
                return Controls(brake=1.0)
            self.mode = "race"
            # Leave the grid slot for the line over the first stretch: the
            # lane starts where the car stands relative to its own plan.
            k = self.frame.node(me.s)
            self.lane = (me.s, me.n - self.plan.n_raw[k],
                         (me.s + 250.0) % self.frame.L, 0.0)
            self.target = 0.0
        self._new_lap(lap)
        if self.mode == "recover" or self._needs_recovery(vehicle, me, dt):
            return self._recover(vehicle, me, field, dt)

        k, f = self.plan.locate(vehicle.pos, self.frame.node(me.s))
        s0, d0, s1, d1 = self.lane
        if self.abs_lane is None and d0 != d1 and self.frame.ds(s1, me.s) >= 0.0 \
                and self.frame.ds(s0, me.s) >= 0.0:
            # The lane change is done: hold the new lane outright, so the
            # wrapped lap distance can never read it as not yet started.
            self.lane = (s1, d1, s1, d1)
        if self.ticks % self.frame.think_every and self._held is not None:
            # Between decisions: the same caps, the lane followed from where
            # the car is now. The tracker itself runs every tick.
            cap, pace, flat = self._held
            d, dd, ddd = self._offset_at(me.s)
            d = self._clip_offset(k, d)
            if self.lane[0] == self.lane[2]:
                dd = ddd = 0.0
            return self._drive(vehicle, me, k, f, d, dd, ddd, cap, pace, flat, dt)
        before = None
        if self.ticks % self.frame.plan_every == 0:
            if self.watch is not None and hasattr(self.watch, "before"):
                # What the decision is made FROM -- taken ahead of it, or it
                # contains the decision (an attack already under way).
                before = self.watch.before(self, me, field)
            if self.policy is not None:
                self._policy_tick(me, field)
            else:
                self._plan(me, field)
        self.yellow = self._yellow(me, field)
        d, dd, ddd = self._offset_at(me.s)
        d = self._clip_offset(k, d)
        d_free = d
        d = self._room(me, field, k, d)
        if d != d_free:
            # Someone alongside: hold the edge of the room left, and let the
            # next plan move off from here rather than from the old lane.
            if self.abs_lane is not None:
                self._end_attack(me, cooldown=2.5)
            dd = ddd = 0.0
            self.lane = (me.s, d, me.s, d)
            self.target = d
        cap = min(self._leader_cap(me, field, d), self._yellow_cap(me))
        pace = self.pace_lap
        if self.yellow:
            # A stricken car ahead: lift, as a marshal's yellow flag asks, so
            # an incident is driven past rather than added to -- and right
            # down past the car itself.
            fr = self.frame
            near = any(-15.0 < fr.ds(me.s, s) < YELLOW_NEAR
                       for s in fr.stricken_s)
            pace *= YELLOW_NEAR_PACE if near else YELLOW_PACE
        elif self.attack is not None:
            # Alongside another car into a corner: leave a little in hand.
            pace *= ATTACK_PACE
        rule_pace = pace / self.pace_lap
        pace *= self.pace_mult
        # Held up: the car ahead is costing real speed, not just sitting there.
        own = self.plan.at(self.plan.v, k, f) * pace
        held = self.leader is not None and cap < 0.985 * own
        if d or self.lane[1] != self.lane[3] or self.abs_lane is not None:
            # A shifted line, or one changing lanes, must also brake for its
            # own curvature.
            cap = min(cap, self._offset_cap(me))
        flat = vehicle.drag_scale < 0.99 and not self.yellow
        self._held = (cap, pace, flat)
        self._is_held = held
        if self.watch is not None and self.ticks % self.frame.plan_every == 0:
            self.watch(self, me, field, rule_pace, before)
        return self._drive(vehicle, me, k, f, d, dd, ddd, cap, pace, flat, dt)

    def _drive(self, vehicle, me, k, f, d, dd, ddd, cap, pace, flat, dt):
        """The per-tick part: timers, a missed braking point, the braking
        envelope, and the tracker."""
        if self._is_held:
            self.held_t += dt
        else:
            self.held_t = max(self.held_t - 0.5 * dt, 0.0)
        self.cooldown = max(self.cooldown - dt, 0.0)
        if self.late_at is not None:
            ahead = (self.late_at - k) % self.frame.count
            if ahead == 0 or ahead == 1:
                # Mostly a moment too long; now and then a real lock-up.
                self.late_left = (LATE_BIG_T if self.rng.random() < LATE_BIG_P
                                  else 0.35)
                self.late_at = None
                self.mistakes_made += 1
        late = self.late_left > 0.0
        if late:
            # A missed braking point: hold the speed it arrived with a moment
            # too long.
            self.late_left -= dt
        if flat:
            # In a tow or with DRS the car can out-run the plan's straight-line
            # speed; let it, as far as it can still brake for what is coming.
            cap = min(cap, self._brake_envelope(me, pace))
        return self.follow.controls(vehicle, k, f, offset=d, d_off=dd,
                                    dd_off=ddd, v_cap=cap, pace=pace,
                                    hold_speed=late, flat_out=flat)

    def _brake_envelope(self, me: CarView, pace: float) -> float:
        """Fastest speed from which every corner ahead can still be braked
        for. Only the plan's grip-limited points count: where the plan is
        flat out its speed is the engine's, and counting those capped a car
        with DRS at exactly the speed DRS was meant to take it past.

        Worked out for every node at once per pace (see _envelope), so each
        tick is one lookup: from node k, cap^2 = env[k] - 2a(s - arc_k)."""
        env = self._env.get(pace)
        if env is None:
            if len(self._env) > 8:
                self._env.clear()
            env = self._env[pace] = self._envelope(pace)
        fr = self.frame
        k0 = fr.node(me.s)
        v2 = env[k0] - 2.0 * A_BRAKE * max(fr.ds(float(fr.arc[k0]), me.s), 0.0)
        return math.sqrt(max(v2, 1.0)) if math.isfinite(v2) else math.inf

    def _envelope(self, pace: float) -> list:
        """min over the grip-limited nodes j strictly ahead of k of
        v_j^2 + 2a(arc_j - arc_k), for every k, round the lap."""
        fr = self.frame
        pl = self.plan
        ceiling = np.minimum(config.ENGINE_FORCE_MAX,
                             config.ENGINE_POWER / np.maximum(pl.v, 1.0))
        limited = pl.fd < 0.95 * ceiling
        e = np.where(limited, (pl.v * pace) ** 2, np.inf)
        S = np.concatenate([fr.arc, fr.arc + fr.L])
        A = np.concatenate([e, e]) + 2.0 * A_BRAKE * S
        run = np.minimum.accumulate(A[::-1])[::-1]       # min over j >= i
        n = fr.count
        nxt = np.append(run[1:], np.inf)                  # min over j > i
        return (nxt[:n] - 2.0 * A_BRAKE * S[:n]).tolist()

    def _lane_profile(self, s_abs: np.ndarray) -> np.ndarray:
        """The lane's offset at many points at once (vectorised _lane_at)."""
        s0, d0, s1, d1 = self.lane
        if d0 == d1:
            return np.full(len(s_abs), d1)
        L = self.frame.L
        span = max(self.frame.ds(s0, s1), 1e-3)
        x = np.clip(((s_abs - s0 + 0.5 * L) % L - 0.5 * L) / span, 0.0, 1.0)
        return d0 + (d1 - d0) * 0.5 * (1.0 - np.cos(np.pi * x))

    #: Nodes past the first that an _offset_cap window stays good for.
    OCAP_SLACK = 30

    def _offset_cap(self, me: CarView) -> float:
        """Speed now that lets the lane ahead be taken at its own curvature.

        The limit at each node ahead depends only on the lane, not on where
        the car is, so it is worked out once per lane over a window a little
        longer than the look-ahead and reused while the car drives into it;
        each tick only takes the braking envelope from where the car is."""
        fr = self.frame
        k0 = fr.node(me.s)
        s0, d0, s1, d1 = self.lane
        lane_key = (round(d1, 2),) if d0 == d1 else self.lane
        key = (lane_key, self.abs_lane, self.attack, self.pace_lap)
        c = self._ocap
        if c is None or c[0] != key or (k0 - c[1]) % fr.count > self.OCAP_SLACK:
            vlim2, s_rel = self._offset_limits(k0)
            c = self._ocap = (key, k0, vlim2, s_rel)
        _, start, vlim2, s_rel = c
        j0 = (k0 - start) % fr.count
        base = fr.ds(float(fr.arc[start]), me.s)
        dist = np.maximum(s_rel[j0:j0 + LOOK_NODES] - base, 0.0)
        back = float(np.min(vlim2[j0:j0 + LOOK_NODES] + 2.0 * A_BRAKE * dist))
        return math.sqrt(max(back, 1.0))

    def _offset_limits(self, k0: int):
        """(squared speed limit, metres from node k0) at each node of the
        window ahead, for the lane as it stands."""
        fr = self.frame
        pl = self.plan
        nodes = (k0 + np.arange(LOOK_NODES + self.OCAP_SLACK)) % fr.count
        seg = self.track.seg_len[nodes]
        s_rel = np.concatenate(([0.0], np.cumsum(seg[:-1])))
        offs = self._offset_profile(float(fr.arc[k0]), s_rel, nodes)
        lo = np.minimum(-self.track.w_left[nodes] + EDGE_ROOM - pl.n_raw[nodes], 0.0)
        hi = np.maximum(self.track.w_right[nodes] - EDGE_ROOM - pl.n_raw[nodes], 0.0)
        offs = np.clip(offs, lo, hi)
        # Where the plan is flat out its speed is the engine's, not a limit
        # for a line beside it (a car with DRS alongside goes faster); only
        # the grip-limited points, and the bend of the lane itself, count.
        v_plan = pl.v[nodes] * self.pace_lap
        ceiling = np.minimum(config.ENGINE_FORCE_MAX,
                             config.ENGINE_POWER / np.maximum(pl.v[nodes], 1.0))
        free = pl.fd[nodes] >= 0.95 * ceiling
        v_base = np.where(free, 1.3 * v_plan, v_plan)
        v_lim = _path_speed(v_base, pl.kappa[nodes], offs, s_rel)[0]
        return v_lim ** 2, s_rel

    # -- recovery ------------------------------------------------------------
    def _needs_recovery(self, vehicle: Vehicle, me: CarView, dt: float) -> bool:
        self.off_t = self.off_t + dt if not vehicle.on_track else 0.0
        # Slow because the car ahead is slow is queueing, not being stuck:
        # treating it as stuck sent whole queues into "recovery" behind an
        # accident and had them rejoin into each other.
        queued = self.leader is not None
        self.slow_t = self.slow_t + dt if (me.v < 3.0 and not queued) else 0.0
        head = self._heading_err(vehicle, me)
        why = ("off" if self.off_t > 0.6 else "slow" if self.slow_t > 1.0
               else "spun" if abs(head) > 1.2 and me.v < 30.0 else "")
        if why:
            self.rec_cause = why
            self.rec_from = ("attack" if self.attack is not None else
                             "follow" if self.leader is not None else "free")
            self.attack = None
            self.abs_lane = None
            self.mode = "recover"
            self._held = None
            self.rec_phase = "stop"
            self.rec_t = 0.0
            self.reverse_t = 0.0
            self.stuck_t = 0.0
            self.marshal = False
            self.stuck_n = 0
            self._goto_rev = 0.0
            self._goto_stuck = 0.0
            self._unstick_t = 0.0
            self._ph_t = 0.0
            self._ph_last = ""
            self.recoveries += 1
            return True
        return False

    def _heading_err(self, vehicle: Vehicle, me: CarView) -> float:
        k = self.frame.node(me.s)
        t = self.track.tangent[k]
        return _wrap(math.atan2(t[0], t[1]) - vehicle.yaw)

    def _recover(self, vehicle: Vehicle, me: CarView, field: list[CarView],
                 dt: float) -> Controls:
        self.rec_t += dt
        head = self._heading_err(vehicle, me)     # + = must turn right
        turn = 1.0 if head > 0.0 else -1.0
        unstick = self._recover_watchdog(dt, turn)
        if unstick is not None:
            return unstick
        v_long = float(np.dot(vehicle.vel, [math.sin(vehicle.yaw), math.cos(vehicle.yaw)]))

        if self.rec_phase == "stop":
            if me.v < 8.0:
                self.rec_phase = "rejoin"
            return Controls(brake=1.0, steer=0.0, analog_steer=True)

        if self.rec_phase == "align":              # only ever the watchdog's
            self.rec_phase = "rejoin"

        # Out of the way first: crawl to just beyond the kerb on whichever
        # side the car is nearer and wait there -- NOT at the edge of the
        # asphalt, which at an apex or an exit is exactly where the racing
        # line runs (a car waiting there set off a chain of cars running wide
        # avoiding it). When the traffic coming past leaves a gap, merge onto
        # the edge of the road, and race from there.
        k, f = self.plan.locate(vehicle.pos, self.frame.node(me.s))
        side = 1.0 if me.n >= 0.0 else -1.0
        edge = (self.track.w_right[k] if side > 0 else self.track.w_left[k])
        wait_n = side * (edge + config.KERB_WIDTH + 1.0)
        merge_n = side * (edge - EDGE_ROOM - 0.2)
        clear = self._traffic_clear(me, field, self.rec_t)
        if self.rec_phase == "rejoin":
            if clear and abs(me.n - wait_n) < 1.5 and abs(head) < 0.3:
                self.rec_phase = "merge"
            n_lane, cap = wait_n, 12.0
        else:                                   # merge
            if not clear and not vehicle.on_track:
                self.rec_phase = "rejoin"
            n_lane, cap = merge_n, 30.0
        if self.rec_phase == "rejoin" and (abs(me.n - wait_n) > 3.0
                                           or abs(head) > 0.6):
            # Far from the place to wait, or pointing the wrong way for the
            # lane follower to be any use: drive there like a person -- turn
            # towards it, reversing and going forward again if it is behind.
            ka = self.frame.node(me.s + REJOIN_AHEAD)
            edge_a = (self.track.w_right[ka] if side > 0 else self.track.w_left[ka])
            tgt = (self.track.center[ka]
                   + self.track.normal[ka] * (side * (edge_a + config.KERB_WIDTH + 1.0)))
            return self._goto(vehicle, float(tgt[0]), float(tgt[1]), cap, dt)
        d = n_lane - self.plan.n_raw[k]
        ctl = self.follow.controls(vehicle, k, f, offset=d, v_cap=cap,
                                   pace=0.9, v_min=4.0)
        on = vehicle.on_track and abs(head) < 0.25
        if self.rec_phase == "merge" and on and clear \
                and abs(me.n - merge_n) < 1.5:
            self.mode = "race"
            d = me.n - self.plan.n_raw[k]
            self.lane = (me.s, d, (me.s + 150.0) % self.frame.L, 0.0)
            self.target = 0.0
            self.off_t = self.slow_t = 0.0
        return ctl

    def _goto(self, vehicle: Vehicle, tx: float, tz: float, cap: float,
              dt: float) -> Controls:
        """Drive to a point, as a person who has gone off would: head for it
        at a walking pace; if it is behind the car, back away from whatever
        is in the way with the wheel the right way for the nose to come round,
        then forward again; if the car stops against something, back off it."""
        pos = vehicle.pos
        e = _wrap(math.atan2(tx - float(pos[0]), tz - float(pos[1])) - vehicle.yaw)
        v_long = float(np.dot(vehicle.vel, [math.sin(vehicle.yaw), math.cos(vehicle.yaw)]))
        if self._goto_rev > 0.0:
            self._goto_rev -= dt
            return Controls(throttle=0.0, brake=0.8, reverse=True,
                            steer=(-1.0 if e > 0.0 else 1.0)
                            if abs(e) < 2.6 else 0.0, analog_steer=True)
        if abs(e) > 1.9:
            if v_long > 1.5:
                return Controls(brake=1.0, steer=0.0, analog_steer=True)
            self._goto_rev = 1.6 if abs(e) < 2.6 else 2.5
            return Controls(brake=0.8, reverse=True, analog_steer=True)
        # Pinned against something: back off it, wheel the other way.
        self._goto_stuck = (self._goto_stuck + dt
                            if (v_long < 0.6 or vehicle.hit_wall) else 0.0)
        if self._goto_stuck > 0.7:
            self._goto_stuck = 0.0
            self._goto_rev = 1.2
        want = max(2.0, cap * min(1.0, 1.4 - abs(e)))
        steer = max(-1.0, min(1.0, 1.8 * e))
        if v_long < want:
            return Controls(throttle=0.55, steer=steer, analog_steer=True)
        return Controls(throttle=0.0, brake=0.25, steer=steer, analog_steer=True)

    def _recover_watchdog(self, dt: float, turn: float):
        """Whatever the car is doing, it gets back on the road.

        Each phase of a recovery has a time (RECOVER_PHASE_T). One that runs
        over -- a car spinning on the spot, nose in a barrier, or standing on
        the brakes with nowhere to go -- is retried with a short reverse and
        the wheel the other way; the retry changes the position, the heading
        or both, so the next attempt starts from somewhere new. After
        RECOVER_TRIES of them, or RECOVER_MARSHAL_T in all, ``marshal`` goes
        up and the field lifts the car back onto the edge of the road, as a
        marshal's crane does. Returns the controls of a reverse in progress,
        else None."""
        if self.rec_t > RECOVER_MARSHAL_T:
            self.marshal = True
        if self._unstick_t > 0.0:
            self._unstick_t -= dt
            return Controls(throttle=0.0, brake=0.8, reverse=True,
                            steer=self._unstick_dir * -turn, analog_steer=True)
        if self.rec_phase != self._ph_last:
            self._ph_last = self.rec_phase
            self._ph_t = 0.0
        self._ph_t += dt
        if self._ph_t > RECOVER_PHASE_T.get(self.rec_phase, 10.0):
            self._ph_t = 0.0
            self.stuck_n += 1
            if self.stuck_n >= RECOVER_TRIES:
                self.marshal = True
            else:
                self._unstick_t = 1.6
                self._unstick_dir = -self._unstick_dir
                if self.rec_phase == "stop":
                    self.rec_phase = "align"
        return None

    def _traffic_clear(self, me: CarView, field: list[CarView],
                       waited: float = 0.0) -> bool:
        """No car due past this spot in the next ~3 s on this side. The
        wait loosens with time -- a stream of cars never leaves a gap that
        wide, and a car that sat at the side for ever (or crawled along the
        grass) was worse than a tight merge -- and after RECOVER_FORCE_T the
        field is trusted to make room."""
        late = max(0.0, waited - RECOVER_EASE_T)
        if waited > RECOVER_FORCE_T:
            return True
        window = max(1.2, 3.0 - 0.2 * late)
        lateral = max(3.0, 5.0 - 0.25 * late)
        for o in field:
            if o.idx == me.idx or not o.racing:
                continue
            gap = self.frame.ds(o.s, me.s)            # + = it is behind me
            if 0.0 < gap < max(o.v, 10.0) * window and abs(o.n - me.n) < lateral:
                return False
        return True
