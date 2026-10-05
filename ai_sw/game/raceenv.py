"""Scenarios, a referee and a training environment for the race driver's
decision layer (``raceai``).

A full twenty-car race shows a passing or defending situation once in a long
while, which is no way to learn one or to measure one. So the situations are
built on purpose -- a car in a slower car's tow before a braking zone, two cars
side by side before it, a faster car right behind, a stranded car at the edge
of the road with traffic coming, a pack -- each a few seconds long, on any
circuit that has solved plans, from a seed:

    sc = make_scenario("Monza", "tow", seed=3)
    stats = run(sc)                    # the rules drive: the baseline
    env = RaceEnv(sc); obs = env.reset()   # a policy drives (raceai.apply_action)

The **referee** watches every tick and keeps what a steward and a spectator
would: contacts, who was penalised, passes (where on the straight or in the
braking zone they were done), time spent side by side, recoveries, and the
lane changes -- including the ones made towards a car closing from behind, which
is what defending looks like (the rule-based drivers make them only as a side
effect of avoiding it, never to cover a line).

The environment's **reward** is progress and places, minus contacts, steward
penalties, recoveries and lane-change jitter -- the stewards (racecontrol.py) are
the referee, so what they punish is what the policy learns not to do. The weights
are ``REWARD``; they are a first guess and are meant to be revised against
tools/race_metrics.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import config, grandprix, raceai
from .trackdata import load_track

DT = 1.0 / 60.0
KINDS = ("tow", "sbs", "defend", "merge", "pack")
#: What the trainer also learns from but the standard measurement leaves out:
#: ``start`` is the opening lap's first seconds -- the whole grid launching at
#: lights out into the first braking zone, where half of a race's recoveries
#: and contacts happen. Twenty cars, so it is slow to run and rarely measured.
TRAIN_KINDS = KINDS + ("start",)
N_CARS = {"tow": 2, "sbs": 2, "defend": 2, "merge": 5, "pack": 8, "start": 20}
#: Longest an episode runs (s), and how far past the corner a pass is judged (m).
MAX_T = {"tow": 40.0, "sbs": 30.0, "defend": 40.0, "merge": 28.0, "pack": 40.0,
         "start": 26.0}
PAST_ZONE = 250.0

REWARD = {
    "progress": 1.0,       # per 100 m advanced
    "place": 1.0,          # per place gained (lost: minus)
    "contact": -1.0,       # per contact above contact.HIT_IMPULSE
    "steward": -4.0,       # per warning or penalty the stewards give the car
    "penalty_s": -0.5,     # per second of time penalty
    "recovery": -3.0,      # per spin / off / stuck that needs recovering
    "strike": -0.5,        # per track-limits strike
    "switch": -0.05,       # per 3.5 m of lane change asked for
    # What a spectator calls unnatural, though no steward does:
    "queue": -0.6,         # per second held up behind a car with open road beside
    "kerb": -1.0,          # per second with two wheels or more on the kerb or beyond
}

#: The slower car's pace in the tow scene, drawn from this range (the ego drives
#: 1.0): a car 1.5% slower than the one behind it is never caught in the few
#: seconds before the corner, with or without skill, which is no scene to learn
#: or measure passing in.
TOW_LEAD = (0.94, 0.97)
#: Where the tow scene starts: metres before the braking zone, and the gap (m).
TOW_START = 450.0
TOW_GAP = (10.0, 35.0)

_TRACKS: dict = {}


def circuits() -> list[str]:
    """The circuits with solved plans, found without the menu (which needs
    ursina: a training kernel has none)."""
    return sorted(p.name for p in config.TRACK_DB.iterdir()
                  if p.is_dir() and grandprix.ready(p.name))


def track_of(circuit: str):
    if circuit not in _TRACKS:
        _TRACKS[circuit] = load_track(circuit)
    return _TRACKS[circuit]


# --------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------
@dataclass
class Scenario:
    circuit: str
    kind: str
    seed: int
    fld: object
    egos: list
    zone: int                       # node of the braking zone the scene is set before
    goal: float                     # metres of progress after which it is over
    max_t: float
    start_progress: dict = field(default_factory=dict)


def _straight_before(driver, z: int, limit: float = 600.0) -> float:
    """Metres of near-straight road before node *z*."""
    plan, fr = driver.plan, driver.frame
    n = fr.count
    k, dist = z, 0.0
    while dist < limit:
        k = (k - 1) % n
        if abs(float(plan.kappa[k])) > 3e-3:
            break
        dist += float(driver.track.seg_len[k])
    return dist


def _place(fld, e, s: float, n_off: float, speed_frac: float = 0.97,
           lateral_abs: float | None = None, yaw_off: float = 0.0,
           speed: float | None = None):
    """Put car *e* at arc length *s*, *n_off* metres right of its plan's line
    (or at *lateral_abs* metres right of the centreline), at a share of the plan's
    speed, already racing."""
    t, fr, drv = fld.track, fld.frame, e.driver
    s %= fr.L
    k = fr.node(s)
    n = float(drv.plan.n_raw[k]) + n_off if lateral_abs is None else lateral_abs
    tan = t.tangent[k]
    e.vehicle.place(t.center[k] + t.normal[k] * n, math.atan2(tan[0], tan[1]) + yaw_off)
    v = float(drv.plan.v[k]) * speed_frac * drv.skill.pace if speed is None else speed
    e.vehicle.vel = np.asarray(tan, dtype=float) * v
    e.surface.hint = k
    drv.mode = "race"
    drv.lane = (s, n - float(drv.plan.n_raw[k]), (s + 250.0) % fr.L, 0.0)
    drv.target = 0.0
    e.laps = 1                       # not the opening lap: no leniency
    e.vehicle.frozen = False
    # Race distance starts from here (TrackLimits takes its zero from its first
    # update), so the first step's progress is a step's, not the whole lap.
    e.limits.update(DT, k, e.vehicle.pos, False, 0.0, 0.0)


def make_scenario(circuit: str, kind: str, seed: int, level: int = 6,
                  ego: int | None = None) -> Scenario:
    """*ego*: which car is the one being driven or measured. Default: the first
    car of each scene, except ``merge`` where it is the first car coming up
    behind the stranded one (the stranded car's own way back onto the road is
    the rules' recovery, not a decision of this layer)."""
    rng = np.random.default_rng(seed * 1000003 + TRAIN_KINDS.index(kind))
    track = track_of(circuit)
    fld = grandprix.build(track, level, 3, seed, player=False, n_cars=N_CARS[kind])
    fr = fld.frame
    cars = fld.cars
    if kind == "start":
        # The grid as the grand prix sets it (fld.grid, in build), lights out now.
        fld.lights_out()
        for e in cars:
            e.lap_start = 0.0
            e.limits.update(DT, e.surface.hint, e.vehicle.pos, False, 0.0, 0.0)
        fld.views = fld._views()
        egos = [int(ego)] if ego is not None else [int(rng.integers(len(cars)))]
        sc = Scenario(circuit, kind, seed, fld, egos, 0, 1e12, MAX_T[kind])
        for e in cars:
            sc.start_progress[e.idx] = float(fr.arc[fr.node(float(fld.views[e.idx].s))])
        return sc
    drv0 = cars[0].driver
    # A braking zone worth passing into, with a straight leading to it.
    need = 150.0 if kind == "sbs" else (TOW_START + 30.0 if kind == "tow" else 420.0)
    zones = [z for z in drv0.attack_zones if _straight_before(drv0, z) >= need] \
        or list(drv0.attack_zones) or list(drv0.brake_zones)
    if zones:
        z = int(zones[int(rng.integers(len(zones)))])
    else:
        # No braking zone anywhere (an oval): the tightest corner stands in.
        z = int(np.argmax(np.abs(drv0.plan.kappa)))
    az = float(fr.arc[z])
    for e in cars:                                   # a clean, repeatable field
        e.driver.skill.mistakes = 0.0
        e.driver.skill.consistency = 0.0

    def pace(i, p):
        cars[i].driver.skill.pace = p
        cars[i].driver.pace_lap = p

    egos = [1 if kind == "merge" else 0]
    goal_arc = az + PAST_ZONE
    if kind == "tow":
        s_l = az - TOW_START
        _place(fld, cars[1], s_l, 0.0)
        _place(fld, cars[0], s_l - float(rng.uniform(*TOW_GAP)), 0.0)
        pace(1, float(rng.uniform(*TOW_LEAD)))
        pace(0, 1.0)
    elif kind == "sbs":
        s0 = az - 140.0
        side = 1.0 if rng.random() < 0.5 else -1.0
        _place(fld, cars[0], s0, side * 1.8)
        _place(fld, cars[1], s0 + float(rng.uniform(-4.0, 4.0)), -side * 1.8)
        pace(0, 1.0)
        pace(1, float(rng.uniform(0.99, 1.01)))
    elif kind == "defend":
        s_e = az - 300.0
        _place(fld, cars[0], s_e, 0.0)
        _place(fld, cars[1], s_e - float(rng.uniform(14.0, 30.0)), 0.0)
        pace(0, 0.995)
        pace(1, 1.015)
    elif kind == "merge":
        s_a = az - 470.0
        goal_arc = s_a + 120.0
        k = fr.node(s_a)
        _place(fld, cars[0], s_a, 0.0, lateral_abs=float(track.w_right[k]) + 2.0,
               yaw_off=math.pi / 2, speed=0.0)
        for i in range(1, 5):
            _place(fld, cars[i], s_a - 230.0 - 38.0 * i, float(rng.uniform(-1.0, 1.0)))
    else:                                             # pack
        s0 = az - 260.0
        placed = []
        for i in range(8):
            for _try in range(40):
                ds = float(rng.uniform(0.0, 75.0)) if i else 0.0
                dn = float(rng.uniform(-4.5, 4.5))
                if all(abs(ds - pd) > 12.0 or abs(dn - pn) > 3.4 for pd, pn in placed):
                    break
            placed.append((ds, dn))
            _place(fld, cars[i], s0 - ds, dn)
            pace(i, float(rng.uniform(0.985, 1.015)))
    if ego is not None:
        egos = [int(ego)]
    fld.lights_out()
    for e in cars:
        e.lap_start = 0.0
    fld.views = fld._views()
    goal = 0.0
    sc = Scenario(circuit, kind, seed, fld, egos, z, goal, MAX_T[kind])
    # Progress is only known after a tick has run; the scenario's own reference
    # is where each car starts along the lap.
    for e in cars:
        k = fr.node(float(fld.views[e.idx].s))
        sc.start_progress[e.idx] = float(fr.arc[k])
    sc.goal = (goal_arc - sc.start_progress[egos[0]]) % fr.L
    return sc


# --------------------------------------------------------------------------
# the referee
# --------------------------------------------------------------------------
class Referee:
    """What a steward and a spectator would write down, tick by tick."""

    def __init__(self, sc: Scenario):
        self.sc = sc
        self.fld = sc.fld
        n = len(self.fld.cars)
        self.passes: list[dict] = []
        self._order: dict = {}
        self.side_by_side = np.zeros(n)
        self.min_gap = np.full(n, np.inf)
        self.lane_moves = np.zeros(n)
        #: Seconds held up with open road beside, and on the kerb (two wheels+).
        self.queue_s = np.zeros(n)
        self.kerb_s = np.zeros(n)
        self.racing_s = np.zeros(n)
        self.defence_moves = 0
        self._target = [e.driver.target if e.driver else 0.0 for e in self.fld.cars]
        self.start_progress = None
        self.tick = 0
        self._zone_arcs = None

    def _zone_offset(self, s: float) -> float:
        """Metres from the nearest braking-zone start (negative: before it)."""
        drv = next(e.driver for e in self.fld.cars if e.driver is not None)
        if self._zone_arcs is None:
            self._zone_arcs = np.asarray([self.fld.frame.arc[z] for z in drv.brake_zones])
        if not len(self._zone_arcs):
            return float("nan")
        d = (s - self._zone_arcs + 0.5 * self.fld.frame.L) % self.fld.frame.L - 0.5 * self.fld.frame.L
        return float(d[int(np.argmin(np.abs(d)))])

    def update(self):
        fld = self.fld
        self.tick += 1
        views = fld.views
        cars = fld.cars
        n = len(cars)
        if self.start_progress is None:
            self.start_progress = [e.limits.progress for e in cars]
        P = np.array([e.limits.progress if e.limits.progress is not None else np.nan
                      for e in cars])
        # Lane changes: all of them, and the ones towards a car closing from behind.
        for i, e in enumerate(cars):
            d = e.driver
            if d is None:
                continue
            tgt = d.target
            if abs(tgt - self._target[i]) > 1e-6:
                self.lane_moves[i] += abs(tgt - self._target[i])
                step = tgt - self._target[i]
                if abs(step) >= 1.5 and d.mode == "race":
                    for j, o in enumerate(views):
                        if j == i or not o.racing:
                            continue
                        gap = fld.frame.ds(o.s, views[i].s)      # >0: i is ahead of o
                        if 0.0 < gap < 35.0 and o.v > views[i].v \
                                and (o.n - views[i].n) * step > 0.0:
                            self.defence_moves += 1
                            break
            self._target[i] = tgt
        if self.tick % 3:
            return
        fr = fld.frame
        for i, e in enumerate(cars):
            d = e.driver
            if d is None or not views[i].racing:
                continue
            self.racing_s[i] += 3 * DT
            if e.vehicle.grip_scale < KERB_GRIP:
                self.kerb_s[i] += 3 * DT
            if d._is_held and e.vehicle.speed > 15.0 and free_side(fld, i):
                self.queue_s[i] += 3 * DT
        for i in range(n):
            for j in range(i + 1, n):
                a, b = views[i], views[j]
                gap = fr.ds(a.s, b.s)
                lat = b.n - a.n
                if abs(gap) < config.CAR_BODY_LENGTH + 2.0:
                    self.min_gap[i] = min(self.min_gap[i], abs(lat))
                    self.min_gap[j] = min(self.min_gap[j], abs(lat))
                    if abs(gap) < config.CAR_BODY_LENGTH and abs(lat) < 3.4:
                        self.side_by_side[i] += 3 * DT
                        self.side_by_side[j] += 3 * DT
                if abs(P[i] - P[j]) <= 30.0 and a.racing and b.racing:
                    ahead = P[i] > P[j]
                    was = self._order.get((i, j))
                    if was is not None and was != ahead and abs(P[i] - P[j]) > 6.0:
                        passer, passed = (i, j) if ahead else (j, i)
                        s_at = views[passer].s
                        self.passes.append({"t": fld.t, "passer": passer, "passed": passed,
                                            "zone_off": self._zone_offset(s_at),
                                            "contact": bool(cars[passer].hits or cars[passed].hits)})
                    if was is None or abs(P[i] - P[j]) > 6.0:
                        self._order[(i, j)] = ahead

    def report(self, egos) -> dict:
        fld = self.fld
        rc = fld.rc
        out = {"t": round(fld.t, 2)}
        out["contacts"] = int(sum(e.hits for e in fld.cars) // 2)
        for key, idx in (("ego", egos[0]),):
            e = fld.cars[idx]
            rec = rc.cars[idx]
            out[f"{key}_hits"] = int(e.hits)
            out[f"{key}_penalty_s"] = float(rc.penalty(idx))
            out[f"{key}_warnings"] = int(rec.collision_warnings)
            out[f"{key}_strikes"] = int(rec.strikes)
            out[f"{key}_lane_moves_m"] = round(float(self.lane_moves[idx]), 1)
            out[f"{key}_queue_s"] = round(float(self.queue_s[idx]), 2)
            out[f"{key}_kerb_s"] = round(float(self.kerb_s[idx]), 2)
            out[f"{key}_side_by_side_s"] = round(float(self.side_by_side[idx]), 2)
            out[f"{key}_min_clearance_m"] = (None if not np.isfinite(self.min_gap[idx])
                                             else round(float(self.min_gap[idx]), 2))
            out[f"{key}_recoveries"] = int(e.driver.recoveries) if e.driver else 0
        out["recoveries"] = int(sum(e.driver.recoveries for e in fld.cars if e.driver))
        racing = max(float(self.racing_s.sum()), 1e-9)
        out["queue_pct"] = round(100.0 * float(self.queue_s.sum()) / racing, 2)
        out["kerb_pct"] = round(100.0 * float(self.kerb_s.sum()) / racing, 2)
        out["passes"] = len(self.passes)
        out["pass_zone_offsets"] = [round(p["zone_off"], 1) for p in self.passes]
        out["passes_by_ego"] = sum(1 for p in self.passes if p["passer"] == egos[0])
        out["passes_on_ego"] = sum(1 for p in self.passes if p["passed"] == egos[0])
        out["defence_moves"] = int(self.defence_moves)
        out["penalised"] = [fld.cars[c].tla for c in range(len(fld.cars)) if rc.penalty(c) > 0]
        out["steward_msgs"] = [m.text for m in rc.messages
                               if m.kind in ("pen", "warn") and "YELLOW" not in m.text][:6]
        return out


def success(kind: str, r: dict) -> bool:
    """A good outcome for the ego car, as a spectator would call it: no contact
    and no penalty (or warning), and for the passing and defending scenes the
    position -- ahead at the end of the corner; for the merge, past the stranded
    car."""
    clean = r["ego_hits"] == 0 and r["ego_penalty_s"] == 0.0 and not r["ego_warnings"]
    if kind in ("tow", "defend"):
        return clean and r["ego_rank_end"] == 1
    if kind == "merge":
        return clean and r["goal_reached"]
    return clean                                    # sbs, pack


#: What counts as riding the kerb: the mean grip over the four patches below
#: this is two wheels or more past the white line (one wheel is 0.97).
KERB_GRIP = 0.9


def free_side(fld, i: int, margin: float = 3.0, ahead: float = 45.0, behind: float = 25.0,
              across: float = 4.2) -> bool:
    """Is there open road beside car *i* -- on either side, more than *margin*
    metres to the white line and no car within *behind*/*ahead* metres along and
    *across* metres over?"""
    views, fr = fld.views, fld.frame
    me = views[i]
    tr = fld.cars[i].driver.track
    k = fr.node(me.s)
    for side in (1.0, -1.0):
        room = (tr.w_right[k] - me.n) if side > 0 else (tr.w_left[k] + me.n)
        if room <= margin:
            continue
        for j, o in enumerate(views):
            if j != i and o.racing and -behind < fr.ds(me.s, o.s) < ahead                     and 0.0 < side * (o.n - me.n) < across:
                break
        else:
            return True
    return False


def _rank(fld, idx: int) -> int:
    P = [e.limits.progress if e.limits.progress is not None else -1e9 for e in fld.cars]
    return 1 + sum(1 for p in P if p > P[idx])


def run(sc: Scenario, policy=None, all_cars: bool = False, watch=None) -> dict:
    """Run the scenario to its goal or its time limit and return the referee's
    report. *policy*: a callable ``policy(driver, me, views)`` installed on the
    ego car's driver (raceai.Policy), or None for the rules; with *all_cars* on
    every car's. *watch*: ``watch(driver, me, field, pace_ratio)`` on every driver
    (what the rules decide, for tools/raceai_collect.py)."""
    fld = sc.fld
    ego = fld.cars[sc.egos[0]]
    if policy is not None:
        for e in (fld.cars if all_cars else [ego]):
            e.driver.policy = policy
    if watch is not None:
        for e in fld.cars:
            if e.driver is not None:
                e.driver.watch = watch
    ref = Referee(sc)
    t_end = sc.max_t
    reached = False
    while fld.t < t_end:
        fld.step(DT)
        ref.update()
        p0 = ref.start_progress[ego.idx] if ref.start_progress else None
        if (p0 is not None and ego.limits.progress is not None
                and ego.limits.progress - p0 >= sc.goal):
            reached = True
            break
    rep = ref.report(sc.egos)
    rep.update(kind=sc.kind, circuit=sc.circuit, seed=sc.seed, goal_reached=reached,
               ego_rank_end=_rank(fld, ego.idx),
               ego_progress_m=round(float((ego.limits.progress or 0.0) - (ref.start_progress[ego.idx] or 0.0)), 1))
    return rep


def run_race(circuit: str, seed: int, laps: int = 3, level: int = 6,
             policy_for=None, watch=None) -> dict:
    """A whole twenty-car grand prix under the referee. *policy_for*: optional
    ``{car index: policy}`` installed on those drivers (None: all rules);
    *watch* as in ``run``."""
    track = track_of(circuit)
    fld = grandprix.build(track, level, laps, seed, player=False)
    for i, pol in (policy_for or {}).items():
        if i < len(fld.cars):
            fld.cars[i].driver.policy = pol
    if watch is not None:
        for e in fld.cars:
            e.driver.watch = watch
    sc = Scenario(circuit, "race", seed, fld, [0], 0, 0.0, 0.0)
    fld.lights_out()
    fld.views = fld._views()
    ref = Referee(sc)
    limit = 140.0 * (laps + 1)
    while not fld.finished() and fld.t < limit:
        fld.step(DT)
        ref.update()
        if fld.leader_done is not None and fld.t > fld.leader_done + 60.0:
            break
    rep = ref.report([0])
    rep.update(kind="race", circuit=circuit, seed=seed, laps=laps,
               cars=len(fld.cars),
               total_penalty_s=float(sum(fld.rc.penalty(c) for c in range(len(fld.cars)))),
               n_penalised=int(sum(1 for c in range(len(fld.cars)) if fld.rc.penalty(c) > 0)),
               hits=int(sum(e.hits for e in fld.cars)))
    return rep


# --------------------------------------------------------------------------
# the training environment
# --------------------------------------------------------------------------
def _noop_policy(driver, me, field):
    """Installed on a car the environment drives: the rule layer stays out of
    the way and the environment applies the actions itself."""


class RaceEnv:
    """A decision every ``plan_every`` physics ticks (about 7.5 Hz), for each of
    the scenario's ego cars. ``reset`` -> {idx: obs}; ``step({idx: action})`` ->
    (obs, rewards, done, info), rewards being {idx: float} with the parts in
    ``info["parts"][idx]``."""

    def __init__(self, scenario: Scenario | None = None):
        self.sc = scenario
        self.ref = None

    def reset(self, scenario: Scenario | None = None):
        if scenario is not None:
            self.sc = scenario
        sc = self.sc
        self.fld = sc.fld
        for i in sc.egos:
            d = self.fld.cars[i].driver
            d.policy = _noop_policy
            d.pace_mult = 1.0
        self.ref = Referee(sc)
        self._prev = {i: self._snapshot(i) for i in sc.egos}
        self.fld.views = self.fld._views()
        return self._obs()

    def _snapshot(self, i):
        e = self.fld.cars[i]
        rc = self.fld.rc.cars[i]
        return {"P": float(e.limits.progress or 0.0), "rank": _rank(self.fld, i),
                "hits": e.hits, "events": rc.collision_warnings + len(rc.penalties),
                "pen": self.fld.rc.penalty(i), "rec": e.driver.recoveries,
                "strikes": rc.strikes, "target": e.driver.target,
                "queue": float(self.ref.queue_s[i]), "kerb": float(self.ref.kerb_s[i])}

    def _obs(self):
        fld = self.fld
        return {i: raceai.observe(fld.cars[i].driver, fld.views[i], fld.views)
                for i in self.sc.egos}

    def step(self, actions: dict):
        fld = self.fld
        for i, a in actions.items():
            raceai.apply_action(fld.cars[i].driver, fld.views[i], a, fld.views)
        for _ in range(fld.frame.plan_every):
            fld.step(DT)
            self.ref.update()
        rewards, parts = {}, {}
        for i in self.sc.egos:
            now, prev = self._snapshot(i), self._prev[i]
            p = {
                "progress": REWARD["progress"] * (now["P"] - prev["P"]) / 100.0,
                "place": REWARD["place"] * (prev["rank"] - now["rank"]),
                "contact": REWARD["contact"] * (now["hits"] - prev["hits"]),
                "steward": REWARD["steward"] * (now["events"] - prev["events"]),
                "penalty_s": REWARD["penalty_s"] * (now["pen"] - prev["pen"]),
                "recovery": REWARD["recovery"] * (now["rec"] - prev["rec"]),
                "strike": REWARD["strike"] * (now["strikes"] - prev["strikes"]),
                "switch": REWARD["switch"] * abs(now["target"] - prev["target"]) / 3.5,
                "queue": REWARD["queue"] * (now["queue"] - prev["queue"]),
                "kerb": REWARD["kerb"] * (now["kerb"] - prev["kerb"]),
            }
            parts[i] = p
            rewards[i] = float(sum(p.values()))
            self._prev[i] = now
        ego = fld.cars[self.sc.egos[0]]
        done = fld.t >= self.sc.max_t
        if ego.limits.progress is not None and self.ref.start_progress is not None:
            done = done or (ego.limits.progress - self.ref.start_progress[ego.idx]
                            >= self.sc.goal)
        return self._obs(), rewards, done, {"parts": parts, "t": fld.t}
