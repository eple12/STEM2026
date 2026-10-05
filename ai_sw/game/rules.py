"""Session rules: race distance and track-limit excursions for one car.

Grand prix only. Qualifying has a simpler rule -- a lap with a moment off the
track is deleted -- and that lives with the lap timing in ``app.py``.

This module only *measures*: how far round the race each car has got, and
each trip off the road -- how much lap it covered out there, how far it
actually drove, how fast it was going in and how slow it got. Deciding what
an excursion deserves (nothing, a warning, a penalty) is race control's job
(``racecontrol.py``), because it depends on things one car's limits cannot
see: whether somebody pushed it off, and how many times it has done it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

#: Metres of lap the nearest centreline sample may move in one physics step
#: before it counts as a jump across the infield. A car covers under a metre a
#: step; anything this large is the lookup snapping to another part of the
#: circuit.
JUMP_M = 200.0


@dataclass
class Excursion:
    """One trip off the track, from the step all four wheels left the road
    to the moment the car had been back on it long enough to call it over."""
    t0: float              # session time it began
    t1: float              # ...and was settled
    lap_m: float           # metres a car would have had to drive to cover
                           # the lap this trip covered, at the offsets it
                           # was at (see TrackLimits.update)
    driven_m: float        # metres it actually drove, off and rejoining
    duration: float        # seconds from leaving to settled
    v_in: float            # speed on the way off
    v_min: float           # slowest it went while off or rejoining

    @property
    def gained_s(self) -> float:
        """Seconds the trip saved over driving the lap it covered: zero for
        running wide (the car drives MORE than the lap advances), the
        shortcut over the car's speed for a cut."""
        short = max(0.0, self.lap_m - self.driven_m)
        speed = max(self.driven_m / max(self.duration, 1e-6), 10.0)
        return short / speed


class TrackLimits:
    """Distance raced and off-track excursions for one car."""

    def __init__(self, track, rejoin: float = 1.0):
        self.track = track
        self._arc_l = track.arclen.tolist()
        self._tan_l = track.tangent.tolist()
        self._cen_l = track.center.tolist()
        self._nrm_l = track.normal.tolist()
        self._kap_l = track.curvature.tolist()
        self._wl_l = track.w_left.tolist()
        self._wr_l = track.w_right.tolist()
        self.L = float(track.length)
        #: Seconds back on the road that end an excursion: wobbling along the
        #: white line is one trip, not one per wobble.
        self.rejoin = rejoin
        #: Metres of lap covered since the lights, unwrapped across the line.
        #: The grid is behind the line, so it starts slightly negative. The
        #: race order is read off this rather than off lap counts: both cars
        #: cross the line moments after the start without it counting as a
        #: lap, and a lap-based distance put whoever was still behind it a
        #: whole lap in the lead.
        self.progress: float | None = None
        self._arc: float | None = None
        self._prev = None

        self.incidents = 0
        #: Settled excursions not yet looked at by race control.
        self.pending: list[Excursion] = []

        self._off = False              # inside an excursion
        self._clear = 0.0              # seconds back on track since it
        #: Over the whole trip -- off, and the rejoin window after, since the
        #: nearest centreline sample can stay on the outbound side of a cut
        #: the whole way across and only snap once the car is back on the
        #: road -- the distance a car would have had to DRIVE to cover the
        #: lap it covered, and the distance it did drive. The first is the
        #: lap's progress corrected for where across the road the car was:
        #: a metre of centreline is (1 - kappa*n) metres on a line n metres
        #: to its side (n held within the road's edges), so taking the
        #: inside of a bend is not a short cut and a cut across the infield
        #: is.
        self._expect = 0.0
        self._driven = 0.0
        self._dur = 0.0
        self._t0 = 0.0
        self._v_in = 0.0
        self._v_min = 0.0

    # -- per physics step -----------------------------------------------
    def update(self, dt: float, i: int, pos, off: bool, now: float,
               speed: float = 0.0):
        L = self.L
        arc = self._arc_l[i]
        px, pz = float(pos[0]), float(pos[1])
        if self.progress is None:
            self.progress = arc - L if arc > 0.5 * L else arc
            self._arc = arc
            self._prev = (px, pz)
        mx, mz = px - self._prev[0], pz - self._prev[1]
        step = math.hypot(mx, mz)
        self._prev = (px, pz)
        d = (arc - self._arc) % L                      # forward, 0..L
        if d > 0.5 * L:
            d -= L                                     # or backward
        if abs(d) > JUMP_M:
            # The nearest sample leapt: the car is cutting across to another
            # part of the circuit. The shorter way round is not evidence of
            # which way it went -- a cut past half a lap would read as the
            # car losing ground -- so ask the car. Moving with the track at
            # the sample it landed on, it went forward.
            tx, tz = self._tan_l[i]
            d = (arc - self._arc) % L
            if mx * tx + mz * tz < 0.0:
                d -= L
        self.progress += d
        self._arc = arc
        if abs(d) < JUMP_M:
            cx, cz = self._cen_l[i]
            nx, nz = self._nrm_l[i]
            n = (px - cx) * nx + (pz - cz) * nz
            # Only as far across as the road goes: taking the inside of a
            # bend on the asphalt is a shorter line, but a car out on the
            # infield is not "on a tighter line", it is cutting.
            n = min(max(n, -self._wl_l[i]), self._wr_l[i])
            k = 1.0 - self._kap_l[i] * n
            d_line = d * (0.3 if k < 0.3 else 2.0 if k > 2.0 else k)
        else:
            d_line = d

        if off:
            if not self._off:
                self._off = True
                self.incidents += 1
                self._expect = 0.0
                self._driven = 0.0
                self._dur = 0.0
                self._t0 = now
                self._v_in = speed
                self._v_min = speed
            self._clear = 0.0
            self._driven += step
            self._expect += d_line
            self._dur += dt
            self._v_min = min(self._v_min, speed)
        elif self._off:
            self._clear += dt
            self._driven += step
            self._expect += d_line
            self._dur += dt
            self._v_min = min(self._v_min, speed)
            if self._clear >= self.rejoin:
                self.settle(now)

    def settle(self, now: float):
        """Close an open excursion and hand it to race control."""
        if not self._off:
            return
        self._off = False
        self.pending.append(Excursion(self._t0, now, max(self._expect, 0.0),
                                      self._driven, self._dur, self._v_in,
                                      self._v_min))

    @property
    def in_excursion(self) -> bool:
        return self._off
