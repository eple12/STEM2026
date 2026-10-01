"""Session rules: race distance and track limits for one car.

Grand prix only. Qualifying has a simpler rule -- a lap with a moment off the
track is deleted -- and that lives with the lap timing in ``app.py``.

A penalty has to make leaving the track a bad trade *every* time, including
the deliberate cut, without being a hanging offence for a wheel over the white
line. A flat fee is both at once, so the fee is small and what grows it is
**how much of the lap was covered off the road**:

    2 s  +  0.06 s per metre of lap driven off the track
         +  2 x the seconds the shortcut saved

Running wide covers a handful of metres out of bounds before the car is back,
so it stays near the fee. Cutting a chicane means most of the chicane's
distance is covered off the road, and the charge grows with it. The last term
only bites on a real shortcut: it compares how much further along the lap the
car got against the distance it actually drove, which is the one thing a cut
cannot hide -- run wide on the outside and the car drives *more* than the lap
advances, so it pays nothing extra.
"""
from __future__ import annotations

import numpy as np

from . import config

#: Metres of lap the nearest centreline sample may move in one physics step
#: before it counts as a jump across the infield. A car covers under a metre a
#: step; anything this large is the lookup snapping to another part of the
#: circuit.
JUMP_M = 200.0


class TrackLimits:
    """Distance raced and off-track penalties for one car."""

    def __init__(self, track):
        self.track = track
        #: Metres of lap covered since the lights, unwrapped across the line.
        #: The grid is behind the line, so it starts slightly negative. The
        #: race order is read off this rather than off lap counts: both cars
        #: cross the line moments after the start without it counting as a
        #: lap, and a lap-based distance put whoever was still behind it a
        #: whole lap in the lead.
        self.progress: float | None = None
        self._arc: float | None = None
        self._prev_pos = None

        self.incidents = 0
        self.penalty = 0.0
        #: The last excursion's full cost, and the session time it was
        #: settled at -- what the HUD flashes up.
        self.last_penalty = 0.0
        self.last_settled: float | None = None

        self._off = False              # inside an excursion
        self._clear = 0.0              # seconds back on track since it
        #: Where the lap stood when the excursion began, and the ground
        #: driven **back on the track** since -- the rejoin window, and any
        #: stretch between two offs of the same excursion. The lap distance
        #: charged for is everything the excursion advanced minus that, which
        #: is the one form that survives the lookup: the nearest centreline
        #: sample can stay on the outbound side of a cut the whole way across
        #: and only snap once the car is back on the road, so a sum taken
        #: over off-track steps alone misses the entire shortcut.
        self._progress0 = 0.0
        self._on_driven = 0.0
        #: Ground driven and time taken while off the road, for the
        #: shortcut-gain term.
        self._driven = 0.0
        self._dur = 0.0
        self._fee = 0.0

    # -- per physics step -----------------------------------------------
    def update(self, dt: float, i: int, pos, off: bool, now: float):
        L = self.track.length
        arc = float(self.track.arclen[i])
        pos = np.asarray(pos, dtype=float)
        if self.progress is None:
            self.progress = arc - L if arc > 0.5 * L else arc
            self._arc = arc
            self._prev_pos = pos.copy()
        move = pos - self._prev_pos
        step = float(np.hypot(*move))
        self._prev_pos = pos.copy()
        d = (arc - self._arc) % L                      # forward, 0..L
        if d > 0.5 * L:
            d -= L                                     # or backward
        if abs(d) > JUMP_M:
            # The nearest sample leapt: the car is cutting across to another
            # part of the circuit. The shorter way round is not evidence of
            # which way it went -- a cut past half a lap would read as the
            # car losing ground, for no penalty -- so ask the car. Moving
            # with the track at the sample it landed on, it went forward.
            fwd = float(np.dot(move, self.track.tangent[i])) >= 0.0
            d = (arc - self._arc) % L
            if not fwd:
                d -= L
        self.progress += d
        self._arc = arc

        if off:
            if not self._off:
                self._off = True
                self.incidents += 1
                self._fee = config.GP_OFFTRACK_PENALTY
                self.penalty += self._fee
                self._progress0 = self.progress - d
                self._on_driven = 0.0
                self._driven = 0.0
                self._dur = 0.0
            self._clear = 0.0
            self._driven += step
            self._dur += dt
        elif self._off:
            self._clear += dt
            self._on_driven += step
            # Back on the road for long enough to call it over. Wobbling
            # along the white line is one excursion, not one fee per wobble.
            if self._clear >= config.GP_OFFTRACK_REJOIN:
                self.settle(now)

    def settle(self, now: float):
        """Close an open excursion and charge for what it gained."""
        if not self._off:
            return
        self._off = False
        # Everything the excursion advanced, less the metres driven under
        # its own steam back on the track: what the car got out of being off
        # it. Charged per metre...
        off_progress = max(0.0, self.progress - self._progress0
                           - self._on_driven)
        distance = config.GP_OFFTRACK_PER_M * off_progress
        # ...and, only for a genuine shortcut, the time it saved.
        gained_m = max(0.0, off_progress - self._driven)
        speed = max(self._driven / max(self._dur, 1e-6), 10.0)
        extra = distance + config.GP_OFFTRACK_GAIN_K * gained_m / speed
        self.penalty += extra
        self.last_penalty = self._fee + extra
        self.last_settled = now

    @property
    def in_excursion(self) -> bool:
        return self._off
