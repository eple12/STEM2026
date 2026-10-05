"""Race control: the stewards for a twenty-car grand prix.

The cars (``field.py``) report what happened -- a trip off the road
(``rules.Excursion``), a contact between two bodies -- and this decides what
it deserves, the way a modern F1 stewards' room does, with one aim above the
rest: **a player should never be punished for something that was not their
fault, nor for a mistake that already cost them time.**

Track limits (all four wheels past the white line):

* **Pushed off.** Any car-to-car contact in the few seconds before: nothing.
* **Lost time.** The car spun, ran through the gravel, came back slow: the
  mistake was its own penalty. Nothing.
* **Gained time.** The lap advanced further than the car drove -- a cut. A
  time penalty that always outweighs the gain: ``ceil(2 x gain) + 1`` s,
  ``GAIN_PEN`` (5 s) at the least -- the heaviest thing a track-limits trip
  can cost, as in F1; past 10 s, a gross short cut, the gain plus 10 s.
* **Ran wide, kept going.** A *strike*. Three are warnings (the third shows
  the black-and-white flag); from the fourth, each costs the level's strike
  penalty (``teams.Level.strike_pen``: 0 on Novice, where it stays a warning,
  1 s on Rookie and Amateur, 2 s above) -- always less than a cut's.

Collisions (above ``contact.HIT_IMPULSE``) are judged on the geometry the
moment they happen:

* **One car behind the other** (less than half a car overlapping): the car
  behind, closing at more than ``REAR_CLOSING``, ran into the back of the
  one in front -- its fault. Otherwise, a racing incident.
* **Alongside**: the car that was moving across into the other, clearly
  faster than the other was moving across into it -- its fault. Otherwise,
  a racing incident.

Before that, a contact nobody could have avoided is let go -- a racing
incident whoever the geometry points at (the player is told which kind):

* **No time to react.** The other car had spun, left the road, stopped or been
  recovered less than ``UNAVOIDABLE_T`` seconds before: a car arriving at
  racing speed cannot do anything about that.
* **Pushed in.** The guilty car had been hit by a *third* car less than
  ``CHAIN_T`` seconds earlier: it was a passenger, not a driver.
* **Rejoining.** The other car was off the road when they touched: the car on
  the road had the right of way.

What the guilty car gets depends on what it did to the other one: if the
victim spun, went off or needed recovering within ``CONSEQUENCE_T`` seconds,
or the hit was heavy, a 5 s penalty; a light touch with no consequence is a
warning, and only a repeat (on levels that say so) is penalised -- and on
the opening lap, where the whole field shares one braking zone, not even
that. Racing incidents get no action, and the player is told so, so a
contact never leaves them wondering whether a penalty is coming.

Yellow flags (a stricken car: stopped, or being recovered):

* **No overtaking** in the yellow zone, bar the stricken car itself. The
  car that passed is told to give the place back; if it has within
  ``GIVE_BACK_T``, nothing more, otherwise ``YELLOW_PEN``.
* **Slow down.** Through the zone a car stays below ``config.YELLOW_SPEED_KMH``.
  It has ``YELLOW_GRACE_T`` seconds from entering the zone to get down to it
  (nobody brakes from 300 km/h in a car's length); after that, more than
  ``YELLOW_OVER_T`` seconds above the limit (``YELLOW_TOL`` of tolerance) is
  failing to slow. The first time is a warning; a repeat costs ``YELLOW_PEN``
  on levels that penalise repeats.

Time penalties are added to the race time at the flag, as in F1 when there is
no pit stop to serve them at.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import config

#: Seconds after a contact during which a trip off the road is excused.
EXCUSE_T = 1.5
#: Seconds a trip has to save before it is "gaining an advantage", and the
#: least that costs. (Smaller trips are strikes, and cost
#: a strike's penalty once the free ones are used.)
GAIN_MIN = 0.20
GAIN_PEN = 5.0
#: A trip that ends slower than this share of the speed it began at, or lasts
#: longer than this, was a mistake that cost time, not a short cut.
LOST_SPEED = 0.55
LOST_T = 5.0
#: Strikes that are warnings before the strike penalty applies.
STRIKES_FREE = 3
#: Collision judgement: closing speed (m/s) for a rear-ending, and the
#: across-the-road speed (m/s) that makes a car the one moving into another.
REAR_CLOSING = 3.0
SIDE_SPEED = 1.2
SIDE_MARGIN = 0.8
#: Seconds the victim of a contact is watched for its consequences, and the
#: impulse (N*s) that is serious whatever happens next.
CONSEQUENCE_T = 3.0
HEAVY_IMPULSE = 4500.0
#: One judgement per pair of cars per incident: contacts this close together
#: are the same one.
PAIR_GAP_T = 2.0
COLLISION_PEN = 5.0
#: Yellow flags: seconds to give a place taken under yellow back, the
#: penalty, and the slowing asked for -- a car's mean speed through the zone
#: over its own on the lap before -- judged on at least YELLOW_MIN_T in it.
GIVE_BACK_T = 10.0
YELLOW_PEN = 5.0
YELLOW_MIN_T = 1.5
YELLOW_GRACE_T = 3.0
YELLOW_TOL = 1.05
YELLOW_OVER_T = 1.0
#: Contacts nobody could have avoided (see the module doc): seconds a car must
#: have been in trouble before running into it counts as the other driver's
#: fault, and seconds after a hit by a third car in which a second one is not.
UNAVOIDABLE_T = 1.5
CHAIN_T = 1.0


@dataclass
class Message:
    t: float
    car: int               # the car it is about (-1: everyone)
    text: str
    kind: str = "info"     # info | warn | pen | flag | clear
    other: int = -1        # the other car, for a contact


@dataclass
class CarRecord:
    penalty: float = 0.0
    penalties: list = field(default_factory=list)   # [(t, seconds, reason)]
    strikes: int = 0
    collision_warnings: int = 0
    yellow_warnings: int = 0
    #: Session time of this car's last contact with anyone, and with whom.
    last_contact: float = -1e9
    last_contact_with: int = -1


@dataclass
class _Pending:
    t: float
    fault: int
    victim: int
    impulse: float
    reason: str
    escalated: bool = False


class RaceControl:
    def __init__(self, n_cars: int, strike_pen: float = 5.0,
                 minor_repeat_pen: bool = True):
        self.cars = [CarRecord() for _ in range(n_cars)]
        self.strike_pen = strike_pen
        self.minor_repeat_pen = minor_repeat_pen
        self.messages: list[Message] = []
        self._pair_t: dict = {}
        self._pending: list[_Pending] = []
        #: Places taken under yellow, waiting to be given back:
        #: [time, passer, passed].
        self._yellow_passes: list[list] = []
        #: Stretches of the lap under a yellow flag: (s from, s to).
        self.yellow: list[tuple[float, float]] = []
        #: The opening lap (set by the field): light contact in the first
        #: lap's scramble is let go, as stewards do -- only one that put
        #: somebody out of the race, or a heavy hit, is acted on.
        self.opening = True

    # -- penalties ---------------------------------------------------------
    def _penalise(self, t: float, car: int, secs: float, reason: str,
                  other: int = -1):
        rec = self.cars[car]
        rec.penalty += secs
        rec.penalties.append((t, secs, reason))
        self.messages.append(Message(t, car, f"{secs:.0f}S TIME PENALTY  ·  {reason}",
                                     "pen", other))

    # -- track limits --------------------------------------------------------
    def excursion(self, car: int, exc) -> None:
        """Judge one settled trip off the track."""
        rec = self.cars[car]
        t = exc.t1
        if exc.t0 - rec.last_contact < EXCUSE_T:
            return                                     # pushed off
        gain = exc.gained_s
        if gain >= GAIN_MIN:
            secs = float(max(math.ceil(2.0 * gain) + 1, GAIN_PEN))
            if secs > 10.0:
                # A gross short cut: everything it gained back, and ten more.
                secs = float(math.ceil(gain) + 10)
            self._penalise(t, car, secs, "LEAVING THE TRACK AND GAINING AN ADVANTAGE")
            return
        if exc.v_min < LOST_SPEED * exc.v_in or exc.duration > LOST_T:
            return                                     # cost itself time
        rec.strikes += 1
        n = rec.strikes
        if n < STRIKES_FREE:
            self.messages.append(Message(t, car, f"TRACK LIMITS  ·  WARNING {n} OF "
                                         f"{STRIKES_FREE}", "warn"))
        elif n == STRIKES_FREE:
            self.messages.append(Message(t, car, "BLACK AND WHITE FLAG  ·  TRACK LIMITS",
                                         "flag"))
        elif self.strike_pen > 0.0:
            self._penalise(t, car, self.strike_pen, "TRACK LIMITS")
        else:
            self.messages.append(Message(t, car, f"TRACK LIMITS  ·  STRIKE {n}", "warn"))

    # -- collisions ------------------------------------------------------------
    def contact(self, t: float, a: int, b: int, impulse: float, sa: dict,
                sb: dict, gap: float, lat: float) -> None:
        """A hit between cars *a* and *b*. ``sa``/``sb`` are their states just
        before it: speed along the track ``v_along``, across it ``v_across``
        (+ = right). ``gap`` is how far b's centre is ahead of a's along the
        lap, ``lat`` how far right of a it is."""
        prior = {c: (self.cars[c].last_contact, self.cars[c].last_contact_with)
                 for c in (a, b)}
        for c, o in ((a, b), (b, a)):
            self.cars[c].last_contact = t
            self.cars[c].last_contact_with = o
        key = (min(a, b), max(a, b))
        if t - self._pair_t.get(key, -1e9) < PAIR_GAP_T:
            self._pair_t[key] = t
            for p in self._pending:
                if {p.fault, p.victim} == {a, b}:
                    p.impulse = max(p.impulse, impulse)
            return
        self._pair_t[key] = t
        fault, reason = self._fault(a, b, sa, sb, gap, lat)
        if fault < 0:
            self.messages.append(Message(t, a, "CONTACT  ·  RACING INCIDENT  ·  "
                                         "NO FURTHER ACTION", "clear", b))
            return
        victim = b if fault == a else a
        why = self._unavoidable(t, fault, victim, sa if fault == a else sb,
                                sb if fault == a else sa, prior[fault])
        if why:
            self.messages.append(Message(t, a, f"CONTACT  ·  {why}  ·  RACING INCIDENT  ·  "
                                         "NO FURTHER ACTION", "clear", b))
            return
        self._pending.append(_Pending(t, fault, victim, impulse, reason))

    @staticmethod
    def _unavoidable(t: float, fault: int, other: int, s_fault: dict,
                     s_other: dict, prior: tuple) -> str:
        """Why a contact the geometry blames on *fault* was not avoidable
        ('' if it was). *prior* is the guilty car's previous (time, with whom)
        contact, from before this one."""
        if s_other.get("trouble_age", math.inf) < UNAVOIDABLE_T:
            return "NO TIME TO REACT"
        t_prev, with_prev = prior
        if t - t_prev < CHAIN_T and with_prev not in (other, -1):
            return "PUSHED INTO IT"
        if s_other.get("off_track", False) and not s_fault.get("off_track", False):
            return "OTHER CAR REJOINING"
        return ""

    @staticmethod
    def _fault(a, b, sa, sb, gap, lat) -> tuple[int, str]:
        length = config.CAR_BODY_LENGTH
        if abs(gap) > 0.5 * length:
            behind, ahead = (a, b) if gap > 0.0 else (b, a)
            sbh, sah = (sa, sb) if behind == a else (sb, sa)
            if sbh["v_along"] - sah["v_along"] > REAR_CLOSING:
                return behind, "CAUSING A COLLISION"
            return -1, ""
        # Alongside: who was moving across into whom.
        side = 1.0 if lat >= 0.0 else -1.0          # b is on a's right if +
        a_in = sa["v_across"] * side
        b_in = -sb["v_across"] * side
        if max(a_in, b_in) > SIDE_SPEED and abs(a_in - b_in) > SIDE_MARGIN:
            return (a if a_in > b_in else b), "CAUSING A COLLISION"
        return -1, ""

    # -- yellow flags ------------------------------------------------------
    def yellow_pass(self, t: float, passer: int, passed: int) -> None:
        """*passer* went by *passed* (a racing car) in a yellow zone."""
        for p in self._yellow_passes:
            if p[1] == passer and p[2] == passed:
                return
        self._yellow_passes.append([t, passer, passed])
        self.messages.append(Message(t, passer, "OVERTAKING UNDER YELLOW  ·  "
                                     "GIVE THE PLACE BACK", "warn", passed))

    def yellow_slow(self, t: float, car: int, over: float, secs: float) -> None:
        """*car* has left a yellow zone after *secs* in it, of which *over*
        seconds (after the grace to get down to the limit) were above it."""
        if secs < YELLOW_MIN_T or over < YELLOW_OVER_T:
            return
        rec = self.cars[car]
        rec.yellow_warnings += 1
        if rec.yellow_warnings > 1 and self.minor_repeat_pen:
            self._penalise(t, car, YELLOW_PEN, "FAILING TO SLOW FOR YELLOW FLAGS")
        else:
            self.messages.append(Message(t, car, "FAILING TO SLOW FOR YELLOW "
                                         "FLAGS  ·  WARNING", "warn"))

    def _settle_yellow(self, t: float, trouble, progress) -> None:
        keep = []
        for p in self._yellow_passes:
            t0, passer, passed = p
            if progress is not None and progress[passed] > progress[passer]:
                self.messages.append(Message(t, passer, "PLACE GIVEN BACK  ·  "
                                             "NO FURTHER ACTION", "clear", passed))
            elif trouble[passed]:
                pass                       # it is now the stricken car
            elif t - t0 >= GIVE_BACK_T:
                self._penalise(t, passer, YELLOW_PEN,
                               "OVERTAKING UNDER YELLOW FLAGS", passed)
            else:
                keep.append(p)
        self._yellow_passes = keep

    def update(self, t: float, trouble: list[bool], progress=None) -> None:
        """Once per step: settle contacts whose consequences are known, and
        places taken under yellow. ``trouble[c]`` -- car c has spun, left
        the road or is recovering; ``progress[c]`` -- its race distance."""
        if self._yellow_passes:
            self._settle_yellow(t, trouble, progress)
        keep = []
        for p in self._pending:
            hurt = trouble[p.victim]
            if hurt or p.impulse >= HEAVY_IMPULSE:
                self._penalise(p.t, p.fault, COLLISION_PEN, p.reason, p.victim)
                continue
            if t - p.t < CONSEQUENCE_T:
                keep.append(p)
                continue
            if self.opening:
                self.messages.append(Message(t, p.fault, "CONTACT  ·  LAP 1  ·  "
                                             "NO FURTHER ACTION", "clear", p.victim))
                continue
            rec = self.cars[p.fault]
            rec.collision_warnings += 1
            if rec.collision_warnings > 1 and self.minor_repeat_pen:
                self._penalise(p.t, p.fault, COLLISION_PEN, p.reason, p.victim)
            else:
                self.messages.append(Message(t, p.fault, "WARNING  ·  " + p.reason,
                                             "warn", p.victim))
        self._pending = keep

    def penalty(self, car: int) -> float:
        return self.cars[car].penalty

    def yellow_at(self, s: float, L: float) -> bool:
        for a, b in self.yellow:
            if (s - a) % L <= (b - a) % L:
                return True
        return False
