"""A grand-prix field: twenty cars on one circuit, touching.

Owns everything that is about the cars TOGETHER rather than any one of them:
the grid, who is where (``CarView`` for every car, every tick), the slipstream
one car makes for the one behind, contact between bodies, the stewards
(``racecontrol.py``), and the race order, laps and finish. Each car's own
driving is its ``RaceDriver``.

A car with no driver is *external*: the player, whose physics runs in the
game at its own rate. The game hands its state in every tick
(``set_external``) and the field treats it like any other car -- seen,
slipstreamed, hit, judged -- except that it never steps it; what a contact
does to it is collected in ``Entrant.kick`` for the game to apply.

Headless on purpose -- no rendering here -- so ``tools/race_sim.py`` can run
whole races to measure the racing, and the game runs the same object in a
worker process (``fieldproc.py``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import config, settings
from .contact import HIT_IMPULSE, collide, overlap
from .racecontrol import YELLOW_GRACE_T, YELLOW_TOL, Message, RaceControl
from .racecraft import CarView, RaceDriver, TrackFrame
from .rules import TrackLimits
from .surface import Surface
from .vehicle import Controls, Vehicle

#: Grid: the pole slot's distance behind the line, the gap between slots,
#: and how far either side of the centreline the two columns stand.
GRID_FRONT = 8.0
GRID_STEP = 8.0
GRID_SIDE = 3.0
#: Slipstream: a car within this many metres behind another, overlapping it
#: across the road by this much, loses up to SLIP_DRAG of its drag.
SLIP_RANGE = 40.0
SLIP_WIDTH = 1.6
SLIP_DRAG = 0.30
#: DRS, as in F1: within this many seconds of the car ahead, on a straight
#: (radius over DRS_RADIUS), drag drops to DRS_DRAG of normal. Without it a
#: car that pulls out of the tow has the same top speed as the car it is
#: trying to pass and can never draw alongside -- which is why F1 has it.
DRS_GAP = 1.8
DRS_RADIUS = 600.0
DRS_DRAG = 0.35
#: Not in the opening lap's scramble (F1 waits two laps; races here are
#: short, so one).
DRS_FROM_LAP = 1
#: Yellow flag: the stretch of lap before (and just after) a stricken car.
YELLOW_BEFORE = 250.0
YELLOW_AFTER = 40.0
#: A car is stricken (and brings out a yellow) once it has been below
#: STRICKEN_V for STRICKEN_T -- never in the first STRICKEN_GRACE seconds
#: of the race, when the whole grid is still getting off the line.
STRICKEN_V = 3.0
STRICKEN_T = 2.0
STRICKEN_GRACE = 6.0
#: A yellow stays out at least this long, and this long after the car is
#: away again, so it is felt and not a flicker.
YELLOW_MIN_T = 15.0
YELLOW_LINGER_T = 8.0
#: A car back from a recovery may pass for this long without a yellow-flag
#: overtake being called on it.
YELLOW_REJOIN_T = 10.0
#: A trip off the road that ends within this long of a recovery is not judged.
RECOVERED_EXCUSE_T = 6.0
#: After the winner takes the flag the rest have this long (at least; or 1.4
#: of the winner's last lap) to cross the line before they are classified.
FINISH_CUTOFF_MIN_T = 60.0
#: Seconds a yellow has to have been out before a pass in it counts: a pass
#: already under way (alongside) when the flag came out is not one made
#: under it, and nobody reacts to a flag in no time at all.
YELLOW_REACT = 2.0
#: A player car that has been put back on the track (R) passes through the
#: others for at least this long, and until it is clear of them.
RESET_GHOST_T = 2.0
#: A car the marshals have put back: solid to no one for this long, and put
#: this far inside the white line.
MARSHAL_GHOST_T = 4.0
MARSHAL_EDGE = 2.0
#: Metres between the timing loops the gaps are measured at.
GAP_RES = 5.0


@dataclass
class Entrant:
    idx: int
    name: str
    team: str
    color: tuple
    vehicle: Vehicle
    surface: Surface
    driver: RaceDriver | None
    limits: TrackLimits
    tla: str = ""
    laps: int = 0                      # laps completed
    lap_start: float = 0.0
    lap_times: list = field(default_factory=list)
    finish_t: float | None = None
    hits: int = 0                      # contacts above HIT_IMPULSE
    grid_slot: int = 0
    #: Speed at each centreline sample when this car last passed it.
    trace: np.ndarray | None = None
    last_node: int = -1
    #: Recovering from an incident: no contact with anyone until it is back
    #: racing AND clear of every other car (see Field._ghosts).
    ghost: bool = False
    ghost_hold: float = 0.0
    drs: bool = False
    #: Driven from outside (the player): never stepped here.
    external: bool = False
    #: What contacts did to an external car since the game last collected
    #: it: (dx, dz, dvx, dvz, dyaw_rate).
    kick: list = field(default_factory=lambda: [0.0] * 5)
    #: Qualifying lap the grid was set by (estimated for the AI).
    quali: float = 0.0
    #: Last controls, for the brake lights.
    braking: bool = False
    prev_speed: float = 0.0
    #: Seconds spent crawling (see STRICKEN_T).
    slow_t: float = 0.0
    #: Its speed at this place on its previous lap (nan on the first).
    ref_v: float = float("nan")
    #: In a yellow zone now: seconds in it, and speed-over-last-lap seconds.
    y_t: float = 0.0
    #: ...and how much of that was above the limit, after the grace.
    y_over: float = 0.0

    @property
    def best(self):
        return min(self.lap_times) if self.lap_times else None


class Field:
    def __init__(self, track, frame: TrackFrame, entrants: list[Entrant],
                 laps: int, rc: RaceControl | None = None):
        self.track = track
        self.frame = frame
        self.cars = entrants
        self.total_laps = laps
        self.rc = rc or RaceControl(len(entrants))
        self.t = 0.0
        self.ticks = 0
        self.started = False
        self.leader_done: float | None = None
        self._cutoff_t = FINISH_CUTOFF_MIN_T
        self.views: list[CarView] = []
        self.overtakes = 0
        self._order_pairs: dict = {}
        self._ref_n = frame.ref.L["n_raw"]
        self._ref_v = frame.ref.L["v"]
        self._curv = np.abs(track.curvature)
        #: Timing loops every GAP_RES metres of race distance: the session
        #: time each car passed each one, for the gaps on the tower -- the
        #: time between two cars at the SAME point, as F1 timing measures it.
        slots = int((laps + 3) * track.length / GAP_RES) + 2
        for e in entrants:
            e.loop_t = np.full(slots, np.nan)
            e.loop_i = -1
        #: What happened, for tools/race_sim.py's analysis: contacts as
        #: ("hit", t, s, a, b, impulse, state a, state b, b's lead on a,
        #: b's offset right of a), recoveries as ("recover", t, s, car, why).
        self.events: list = []
        #: Car -> session time it began being in trouble (spun, off the road,
        #: being recovered); absent while it is not. For the stewards' idea of
        #: a contact nobody could have avoided (racecontrol.UNAVOIDABLE_T).
        self._trouble_since: dict[int, float] = {}

    # -- the start ------------------------------------------------------
    def grid(self):
        """Two staggered columns behind the line, pole on the side of the
        circuit the racing line starts from. ``self.cars`` is in grid order."""
        t = self.track
        fr = self.frame
        pole_side = 1.0 if fr.ref.n_raw[0] >= 0.0 else -1.0
        for slot, e in enumerate(self.cars):
            e.grid_slot = slot
            s = (t.length - GRID_FRONT - GRID_STEP * slot) % t.length
            k = fr.node(s)
            side = pole_side if slot % 2 == 0 else -pole_side
            pos = t.center[k] + t.normal[k] * (side * GRID_SIDE)
            yaw = math.atan2(t.tangent[k, 0], t.tangent[k, 1])
            e.vehicle.place(pos, yaw)
            e.vehicle.frozen = True
            e.surface.hint = k

    def lights_out(self):
        self.started = True
        self.t = 0.0
        for e in self.cars:
            e.vehicle.frozen = False
            e.lap_start = 0.0

    # -- the player ------------------------------------------------------
    def set_external(self, idx: int, pos, yaw: float, vel, yaw_rate: float,
                     steer_angle: float, on_track: bool, braking: bool):
        """The player's car as the game has it now, plus any contact the game
        has not applied yet (``pending``, see fieldproc)."""
        v = self.cars[idx].vehicle
        v.prev_pos, v.prev_yaw = v.pos, v.yaw
        v.pos = np.asarray(pos, dtype=float)
        v.yaw = float(yaw)
        v.vel = np.asarray(vel, dtype=float)
        v.yaw_rate = float(yaw_rate)
        v.steer_angle = float(steer_angle)
        v.on_track = bool(on_track)
        self.cars[idx].braking = braking
        self.cars[idx].surface.progress(v.pos)        # keep its hint current

    def reset_external(self, idx: int):
        """The player was put back on the track: let it through everyone."""
        e = self.cars[idx]
        e.ghost = True
        e.ghost_hold = RESET_GHOST_T

    # -- what everyone sees ----------------------------------------------
    def _views(self) -> list[CarView]:
        fr = self.frame
        ref_n, ref_v = self._ref_n, self._ref_v
        out = []
        count = fr.count
        for e in self.cars:
            i, s, n = fr.locate(e.surface, e.vehicle.pos)
            k = fr.node(s)
            v = e.vehicle.speed
            if e.trace is None:
                e.trace = np.full(count, np.nan)
            if self.started:
                step = (k - e.last_node) % count
                if step:
                    # Read before this lap overwrites it: last lap, here.
                    e.ref_v = float(e.trace[k])
                if self.rc.yellow and self.rc.yellow_at(s, fr.L):
                    pass                  # a slowed lap is no reference to slow by
                elif 0 < step < 20:
                    if step == 1:
                        e.trace[k] = v
                    else:
                        e.trace[(e.last_node + 1 + np.arange(step)) % count] = v
                elif e.last_node < 0 or step >= 20:
                    e.trace[k] = v
                e.last_node = k
            if e.driver is None:
                racing = self.started and v > 3.0
            else:
                racing = self.started and e.driver.mode == "race"
            if e.external:
                a = (v - e.prev_speed) / max(self.frame.dt, 1e-3)
            else:
                a = e.vehicle.tele.long_accel
            e.prev_speed = v
            out.append(CarView(idx=e.idx, s=s, n=n, v=v,
                               dev=n - ref_n[k],
                               ratio=v / max(ref_v[k], 1.0),
                               racing=racing, trace=e.trace, ghost=e.ghost,
                               a=a))
        return out

    def _arrays(self, views):
        """Positions of every car as arrays, and every pair's separation:
        gap[i, j] is how far j is ahead of i along the lap, lat[i, j] how
        far right of i it is."""
        L = self.frame.L
        S = np.fromiter((v.s for v in views), float, len(views))
        Nn = np.fromiter((v.n for v in views), float, len(views))
        gap = (S[None, :] - S[:, None] + 0.5 * L) % L - 0.5 * L
        lat = Nn[None, :] - Nn[:, None]
        return S, gap, lat

    def _slipstream(self, views, gap, lat):
        n = len(views)
        ok = np.fromiter(((not v.ghost) and v.racing for v in views), bool, n)
        V = np.fromiter((max(v.v, 10.0) for v in views), float, n)
        ahead = (gap > 0.0) & ok[None, :]
        np.fill_diagonal(ahead, False)
        tow = ahead & (gap < SLIP_RANGE) & (np.abs(lat) < SLIP_WIDTH)
        if not settings.current.slipstream:
            tow[:] = False
        best = np.where(tow, 1.0 - gap / SLIP_RANGE, 0.0).max(axis=1)
        fr = self.frame
        curv = self._curv
        for c, (e, me) in enumerate(zip(self.cars, views)):
            straight = curv[fr.node(me.s)] < 1.0 / DRS_RADIUS
            drs = (settings.current.drs and straight and e.laps >= DRS_FROM_LAP and me.racing
                   and bool((ahead[c] & (gap[c] < DRS_GAP * V[c])).any()))
            e.vehicle.drag_scale = (1.0 - SLIP_DRAG * float(best[c])) * \
                (DRS_DRAG if drs else 1.0)
            e.drs = drs

    def _ghosts(self, views, gap, lat, dt):
        """Rejoining cars pass through the field, as in most racing games:
        a car crawling back from the run-off across a corner full of traffic
        otherwise collects everything coming through it, and each of those
        then needs rejoining too. Solid again only once it is racing and not
        overlapping anyone -- re-solidifying inside another car would launch
        both."""
        for c, (e, me) in enumerate(zip(self.cars, views)):
            if e.driver is not None and e.driver.mode == "recover":
                e.ghost = True
                continue
            if e.ghost_hold > 0.0:
                e.ghost_hold -= dt
                e.ghost = True
                continue
            if e.ghost:
                near = np.flatnonzero((np.abs(gap[c]) < 7.0) & (np.abs(lat[c]) < 4.0))
                e.ghost = any(o != c and overlap(e.vehicle, self.cars[o].vehicle)
                              is not None for o in near)

    def _contacts(self, views, gap, lat):
        n = len(self.cars)
        ghost = np.fromiter((e.ghost for e in self.cars), bool, n)
        # Two bodies can only touch with their centres closer than twice the
        # half-diagonal of one; everything further apart skips the box test.
        reach = 2.0 * math.hypot(max(config.BODY_TO_FRONT, config.BODY_TO_REAR),
                                 config.BODY_HALF_WIDTH) + 0.5
        cand = (gap * gap + lat * lat) <= reach * reach
        cand &= ~ghost[:, None] & ~ghost[None, :]
        cand = np.triu(cand, 1)
        for a, b in zip(*np.nonzero(cand)):
            a, b = int(a), int(b)
            ea, eb = self.cars[a], self.cars[b]
            va, vb = ea.vehicle, eb.vehicle
            before = [(v.pos, v.vel, v.yaw_rate) for v in (va, vb)]
            j = collide(va, vb)
            if j <= 0.0:
                continue
            for e, (p0, v0, r0) in ((ea, before[0]), (eb, before[1])):
                if e.external:
                    v = e.vehicle
                    k = e.kick
                    k[0] += float(v.pos[0] - p0[0])
                    k[1] += float(v.pos[1] - p0[1])
                    k[2] += float(v.vel[0] - v0[0])
                    k[3] += float(v.vel[1] - v0[1])
                    k[4] += float(v.yaw_rate - r0)
            if j > HIT_IMPULSE:
                ea.hits += 1
                eb.hits += 1
                g, l = float(gap[a, b]), float(lat[a, b])
                self.events.append(("hit", self.t, views[a].s, a, b, j,
                                    self._state(a), self._state(b), g, l))
                sa = self._motion(views[a], before[0][1], a)
                sb = self._motion(views[b], before[1][1], b)
                self.rc.contact(self.t, a, b, j, sa, sb, g, l)

    def _motion(self, view, vel, c: int) -> dict:
        """A velocity along and across the track where a car is, and how long
        it has been in trouble (inf: not), for the stewards."""
        k = self.frame.node(view.s)
        tx, tz = self.frame._tan_l[k]
        vx, vz = float(vel[0]), float(vel[1])
        since = self._trouble_since.get(c)
        age = self.t - since if since is not None else math.inf
        crawl = self.cars[c].slow_t
        if crawl > 0.0:
            age = min(age, crawl)            # stopped on the road, not yet a yellow
        # The track's normal points right of its tangent: (tz, -tx).
        return {"v_along": vx * tx + vz * tz, "v_across": vx * tz - vz * tx,
                "trouble_age": age, "off_track": not self.cars[c].vehicle.on_track}

    def _state(self, c: int) -> str:
        d = self.cars[c].driver
        if d is None:
            return "ext"
        if d.mode != "race":
            return f"{d.mode}:{d.rec_phase}"
        return "attack" if d.attack is not None else (
            "follow" if d.leader is not None else "free")

    def _trouble(self, views) -> list[bool]:
        """For the stewards: which cars have spun, left the road or are
        being recovered right now."""
        out = []
        for e, me in zip(self.cars, views):
            v = e.vehicle
            if e.driver is not None and e.driver.mode == "recover":
                out.append(True)
                continue
            k = self.frame.node(me.s)
            tx, tz = self.frame._tan_l[k]
            head = abs((math.atan2(tx, tz) - v.yaw + math.pi) % (2 * math.pi) - math.pi)
            out.append((not v.on_track) or head > 1.2 or abs(v.yaw_rate) > 2.0)
        return out

    def _stricken(self, e) -> bool:
        if e.finish_t is not None:
            return False
        if self.t <= STRICKEN_GRACE:
            return False                  # the start is not an incident
        if e.driver is not None and e.driver.mode == "recover":
            return True
        return e.slow_t > STRICKEN_T

    def _call_yellow(self, e, me):
        """Race control to everyone: a yellow is out, where, and for whom."""
        sector = self.track.sector_of(self.frame.node(me.s)) + 1
        what = ("RECOVERING" if e.driver is not None and e.driver.mode == "recover"
                else "STOPPED ON TRACK")
        self.rc.messages.append(Message(
            self.t, e.idx, f"YELLOW SECTOR {sector}  ·  {e.tla} {what}  ·  MAX {config.YELLOW_SPEED_KMH:.0f} KM/H  ·  NO OVERTAKES",
            "yellow"))

    def _yellow(self, views, dt: float):
        if not settings.current.yellow_flags:
            self.rc.yellow = []
            self.frame.yellow = []
            return
        L = self.frame.L
        zones = []
        since = self.__dict__.setdefault("_yellow_since", {})
        lingering = self.__dict__.setdefault("_yellow_linger", {})
        was_out = bool(since)
        for e, me in zip(self.cars, views):
            e.slow_t = e.slow_t + dt if me.v < STRICKEN_V else 0.0
            if self._stricken(e):
                if e.idx not in since:
                    since[e.idx] = self.t
                    self._call_yellow(e, me)
                lingering.pop(e.idx, None)
                zones.append(((me.s - YELLOW_BEFORE) % L, (me.s + YELLOW_AFTER) % L))
            elif e.idx in since:
                # Moving again: the flag stays out a while longer, where it
                # was -- not carried along the road with the car.
                if e.idx not in lingering:
                    lingering[e.idx] = (self.t, ((me.s - YELLOW_BEFORE) % L,
                                                 (me.s + YELLOW_AFTER) % L))
                t0, zone = lingering[e.idx]
                if (self.t - t0 > YELLOW_LINGER_T
                        and self.t - since[e.idx] > YELLOW_MIN_T):
                    since.pop(e.idx)
                    lingering.pop(e.idx, None)
                else:
                    zones.append(zone)
        if was_out and not since:
            self.rc.messages.append(Message(self.t, -1, "TRACK CLEAR  ·  "
                                            "GREEN FLAG", "green"))
        #: When each zone (same order as rc.yellow) came out.
        self._zone_since = [since[e.idx] for e in self.cars if e.idx in since]
        self.rc.yellow = zones
        self.frame.yellow = zones
        self.frame.stricken_s = tuple((b - YELLOW_AFTER) % L for _a, b in zones)
        # Slowing for it: each car's speed through a zone against its own
        # at the same places on its lap before, judged as it leaves.
        limit = config.YELLOW_SPEED_KMH / 3.6
        for e, me in zip(self.cars, views):
            inside = (bool(zones) and e.finish_t is None and not self._stricken(e)
                      and self.rc.yellow_at(me.s, L))
            if inside:
                e.y_t += dt
                if e.y_t > YELLOW_GRACE_T and me.v > limit * YELLOW_TOL:
                    e.y_over += dt
            elif e.y_t > 0.0:
                self.rc.yellow_slow(self.t, e.idx, e.y_over, e.y_t)
                e.y_t = e.y_over = 0.0

    # -- one physics tick --------------------------------------------------
    def step(self, dt: float):
        self.ticks += 1
        self.frame.tick = self.ticks
        self.frame.dt = dt
        views = self._views()
        self.views = views
        S, gap, lat = self._arrays(views)
        if self.started:
            self._slipstream(views, gap, lat)
        for e, me in zip(self.cars, views):
            if e.external:
                continue
            if e.finish_t is not None:
                # Past the flag: cruise round out of everyone's way.
                ctl = e.driver.controls(e.vehicle, me, views, self.t,
                                        e.laps, dt)
                ctl = Controls(throttle=min(ctl.throttle, 0.3), brake=ctl.brake,
                               steer=ctl.steer, analog_steer=True)
                if e.driver.marshal:
                    self._marshal(e, me)
            else:
                before = e.driver.recoveries
                ctl = (e.driver.controls(e.vehicle, me, views, self.t, e.laps, dt)
                       if self.started else Controls(brake=1.0))
                if e.driver.recoveries != before:
                    self.events.append(("recover", self.t, me.s, e.idx,
                                        e.driver.rec_cause, e.driver.rec_from,
                                        e.hits))
                if e.driver.marshal:
                    self._marshal(e, me)
                    ctl = Controls(throttle=0.3)
            if e.driver is not None and e.driver.mode == "recover":
                self.__dict__.setdefault("_last_rec", {})[e.idx] = self.t
            e.braking = ctl.brake > 0.05
            e.vehicle.step(ctl, dt, e.surface)
        if not self.started:
            return
        self.t += dt
        self._ghosts(views, gap, lat, dt)
        self._contacts(views, gap, lat)
        self._timing(dt)
        self._yellow(views, dt)
        self.rc.opening = max(e.laps for e in self.cars) == 0
        self._trouble_now = self._trouble(views)
        for c, bad in enumerate(self._trouble_now):
            if not bad:
                self._trouble_since.pop(c, None)
            else:
                self._trouble_since.setdefault(c, self.t)
        self.rc.update(self.t, self._trouble_now,
                       [e.limits.progress if e.limits.progress is not None
                        else -1e9 for e in self.cars])

    def _marshal(self, e, me):
        """A recovery that has not worked: lift the car onto the road at the
        place it is nearest to, facing the way the race goes, and let it carry
        on. Solid to nobody for MARSHAL_GHOST_T."""
        d, t = e.driver, self.track
        k = self.frame.node(me.s)
        n = min(max(me.n, -float(t.w_left[k]) + MARSHAL_EDGE),
                float(t.w_right[k]) - MARSHAL_EDGE)
        e.vehicle.place(t.center[k] + t.normal[k] * n,
                        math.atan2(t.tangent[k, 0], t.tangent[k, 1]))
        e.surface.hint = k
        e.ghost = True
        e.ghost_hold = MARSHAL_GHOST_T
        self.events.append(("marshal", self.t, me.s, e.idx, d.rec_cause, d.rec_phase,
                            d.rec_t))
        d.mode = "race"
        d.marshal = False
        d.stuck_n = 0
        d._unstick_t = 0.0
        d.off_t = d.slow_t = 0.0
        d.attack = None
        d.abs_lane = None
        d._held = None
        d.leader = None
        d.lane = (me.s, n - float(d.plan.n_raw[k]), (me.s + 150.0) % self.frame.L, 0.0)
        d.target = 0.0

    def _timing(self, dt: float):
        L = self.track.length
        for e in self.cars:
            v = e.vehicle
            i = e.surface.hint
            e.limits.update(dt, i, v.pos, not v.on_track, self.t, v.speed)
            while e.limits.pending:
                exc = e.limits.pending.pop(0)
                # A car that had to be recovered paid for it already, and its
                # way back across the grass is not a short cut.
                if (self.t - self.__dict__.get("_last_rec", {}).get(e.idx, -1e9)
                        < RECOVERED_EXCUSE_T):
                    continue
                self.rc.excursion(e.idx, exc)
            prog = e.limits.progress
            if prog is None:
                continue
            j = int(prog // GAP_RES)
            if j > e.loop_i and j < len(e.loop_t):
                e.loop_t[max(e.loop_i + 1, 0):j + 1] = self.t
                e.loop_i = j
            done = int(math.floor(prog / L))
            if done > e.laps and e.finish_t is None:
                e.lap_times.append(self.t - e.lap_start)
                e.lap_start = self.t
                e.laps = done
                if e.laps >= self.total_laps or (
                        self.leader_done is not None):
                    e.finish_t = self.t
                    e.limits.settle(self.t)
                    while e.limits.pending:
                        self.rc.excursion(e.idx, e.limits.pending.pop(0))
                    if self.leader_done is None:
                        self.leader_done = self.t
                        #: Anyone still out after this long is classified as
                        #: they stand: a car that never gets home (stuck, or a
                        #: lap and more down) kept the result card at RUNNING.
                        self._cutoff_t = max(FINISH_CUTOFF_MIN_T,
                                             1.4 * e.lap_times[-1])
        if self.leader_done is not None and self.t - self.leader_done > self._cutoff_t:
            for e in self.cars:
                if e.finish_t is None:
                    e.finish_t = self.t
                    e.limits.settle(self.t)
                    while e.limits.pending:
                        self.rc.excursion(e.idx, e.limits.pending.pop(0))
        if self.ticks % 3 == 0:
            self._count_passes()

    def _count_passes(self):
        """A pass is two racing cars, close on track, changing order."""
        cars = self.cars
        n = len(cars)
        P = np.fromiter(((e.limits.progress if e.limits.progress is not None
                          else np.nan) for e in cars), float, n)
        racing = np.fromiter((v.racing for v in self.views), bool, n)
        close = np.abs(P[:, None] - P[None, :]) <= 30.0
        close &= racing[:, None] & racing[None, :]
        close = np.triu(close, 1)
        live = set()
        for a, b in zip(*np.nonzero(close)):
            a, b = int(a), int(b)
            live.add((a, b))
            pa, pb = P[a], P[b]
            ahead = pa > pb
            was = self._order_pairs.get((a, b))
            if was is not None and was != ahead and abs(pa - pb) > 6.0:
                self.overtakes += 1
                self._yellow_pass(a, b, ahead)
            if was is None or abs(pa - pb) > 6.0:
                self._order_pairs[(a, b)] = ahead
        for key in [k for k in self._order_pairs if k not in live]:
            del self._order_pairs[key]

    def _yellow_pass(self, a: int, b: int, a_ahead: bool):
        """A pass between racing cars a and b: taken under yellow, unless
        the car passed is the stricken one (or in trouble itself)."""
        if not self.rc.yellow:
            return
        passer, passed = (a, b) if a_ahead else (b, a)
        trouble = getattr(self, "_trouble_now", None)
        if trouble is not None and trouble[passed]:
            return
        if self._stricken(self.cars[passed]):
            return
        # A car that has only just rejoined from a recovery is not an
        # overtaker: it is the cars that were queueing it passes.
        rec = self.__dict__.get("_last_rec", {})
        if self.t - rec.get(passer, -1e9) < YELLOW_REJOIN_T:
            return
        L = self.frame.L
        for (a, b), t0 in zip(self.rc.yellow, self._zone_since):
            if self.t - t0 < YELLOW_REACT:
                continue
            for s in (self.views[passer].s, self.views[passed].s):
                if (s - a) % L <= (b - a) % L:
                    self.rc.yellow_pass(self.t, passer, passed)
                    return

    def overload(self, behind: bool):
        """Running behind real time (the game's worker says so): drivers
        re-plan and look round less often until it has caught up. Racing
        gets a touch coarser for a moment, rather than the AI falling
        behind the player's clock."""
        fr = self.frame
        if behind:
            fr.plan_every, fr.think_every = 12, 3
        else:
            from .racecraft import PLAN_EVERY, THINK_EVERY
            fr.plan_every, fr.think_every = PLAN_EVERY, THINK_EVERY

    def gap(self, behind: Entrant, ahead: Entrant) -> float | None:
        """Seconds between two cars at the point the one behind has reached:
        how long ago the one ahead passed it."""
        p = behind.limits.progress
        if p is None or p < 0.0:
            return None
        x = p / GAP_RES
        j = int(x)
        lt = ahead.loop_t
        if j + 1 >= len(lt) or not np.isfinite(lt[j]):
            return None
        t1 = lt[j + 1] if np.isfinite(lt[j + 1]) else self.t
        return max(self.t - (lt[j] + (t1 - lt[j]) * (x - j)), 0.0)

    # -- the race order --------------------------------------------------
    def total_time(self, e: Entrant) -> float | None:
        return None if e.finish_t is None else e.finish_t + self.rc.penalty(e.idx)

    def order(self) -> list[Entrant]:
        """Running order: laps and distance while racing; at the flag, laps
        then race time WITH penalties added."""
        def key(e):
            if e.finish_t is not None:
                return (0, -e.laps, self.total_time(e))
            return (1, -(e.limits.progress or -1e9), 0.0)
        return sorted(self.cars, key=key)

    def finished(self) -> bool:
        return all(e.finish_t is not None for e in self.cars)
