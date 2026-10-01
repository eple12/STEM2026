"""An AI opponent to race against, shown as a ghost.

It is not a replay. The ghost runs the same vehicle model as the player, on its
own copy of the surface, driven by the physics planner in ``autopilot.py`` --
so it reacts to the circuit rather than repeating a recorded line, and it works
on any track without a lap having been recorded there first.

Nothing about it is trained. The planner already computes a corner speed from
the friction circle and the yaw balance, which is a better opponent than a
learned policy would be at this scale *and* gives a difficulty knob that means
something: ``pace`` is the fraction of the tyre's grip the AI is willing to
use, so the same driver simply commits less. A trained net would have to be
retrained per circuit and could not be dialled.

Ghost, not rival: it does not collide with the player, and the player does not
collide with it. Two cars sharing a corner at an exhibition would spend the
afternoon in the barriers, and the point here is to race a lap time.
"""
from __future__ import annotations

import math

import numpy as np
from panda3d.core import TransparencyAttrib

from . import config, f1tenth, raceline, rlpolicy
from .autopilot import Autopilot
from .car import Car
from .surface import Surface
from .trackdata import Track
from .vehicle import Controls, Vehicle


def rl_line(track):
    return rlpolicy.reference_line(track)


def make_driver(track: Track, surface: Surface, pace: float | None = None):
    """(driver, planner, line, pace) -- the AI as the ghost drives it.

    Shared with tools/record_ghost.py, so the lap replayed in qualifying is
    driven by exactly the driver the grand prix races against.
    """
    # The learned line and the constants it was learned with. They come as
    # a pair: on Monza the line alone, driven with the hand-picked
    # constants, laps 141.2 s against the centreline's 126.0, while the
    # two together lap 108.3.
    line = raceline.load(track)
    learned_pace, tuning = raceline.load_tuning(track)
    if learned_pace is None:
        pace = config.GHOST_PACE if pace is None else pace
    else:
        # Difficulty scales the learned pace rather than replacing it, so
        # an easier ghost is the same driver committing less -- not a
        # different driver on a line that no longer suits it.
        scale = 1.0 if pace is None else pace / config.GHOST_PACE
        pace = learned_pace * scale
    pilot = Autopilot(track, surface, line=line, pace=pace, tuning=tuning,
                      speed_scale=raceline.load_speed(track))
    # A trained policy drives instead, where the circuit has one. It is
    # handed the same reference line the training environment used -- the
    # imported f1tenth trajectory -- because the observation is measured
    # against it, and a policy shown a different line than it learned on
    # is reading the road wrong.
    driver = pilot
    if config.GHOST_DRIVER == "rl" and rlpolicy.available(track.name):
        env_line = rl_line(track)
        driver = rlpolicy.RLDriver(
            track, env_line,
            raceline.speed_profile(env_line.seg_len, env_line.curvature,
                                   env_line.curv_radius, 1.0),
            surface)
    return driver, pilot, line, pace


def ghost_car(vehicle: Vehicle) -> Car:
    """The ghost's translucent body, for a live AI or a replayed lap."""
    car = Car(vehicle, model=config.GHOST_MODEL)
    # Translucent. The shader passes the albedo's alpha straight through,
    # and the albedo picks up p3d_ColorScale, so a colour scale is all it
    # takes -- no second material and no separate shader.
    car.set_transparency(TransparencyAttrib.M_alpha)
    car.set_color_scale(1.0, 1.0, 1.0, config.GHOST_ALPHA)
    return car


class Ghost:
    """A second car, driven by the autopilot, with per-sample lap splits."""

    #: Always on screen; a replayed lap (replay.ReplayGhost) is not.
    visible = True

    def __init__(self, track: Track, pace: float | None = None,
                 out_lap: bool = True):
        self.track = track
        # Its own Surface: the class carries a nearest-sample hint from call to
        # call, and two cars in different parts of the lap sharing one would
        # make every query start from the wrong place.
        self.surface = Surface(track)
        self.vehicle = Vehicle()
        self.vehicle.frozen = True

        pos, yaw = track.start_pose()
        # Alongside and slightly back, like the other side of a grid row, so
        # the two cars do not start inside one another.
        pos = (pos + track.normal[0] * config.GHOST_GRID_OFFSET
               - track.tangent[0] * config.GHOST_GRID_BACK)
        self.vehicle.place(pos, yaw)

        self.driver, self.pilot, self.line, self.pace = make_driver(
            track, self.surface, pace)
        self.car = ghost_car(self.vehicle)

        #: Seconds into the current lap when the ghost passed each centreline
        #: sample, and the same for the lap before it. Two laps because the
        #: player can be ahead, in which case the ghost has not reached their
        #: part of the track yet on this lap and last lap's time is the only
        #: honest comparison.
        self.splits = np.full(track.count, np.nan)
        self.prev_splits = np.full(track.count, np.nan)

        self.lap_time = 0.0
        # 0 while it is on its out lap. A grand prix has none: lap 1 is timed
        # from the lights, the same as the player's.
        self.lap_num = 0 if out_lap else 1
        #: Seconds driven since the lights, on the physics clock, and the value
        #: it had at the chequered flag. Set by the race, which knows the
        #: distance; the ghost only knows laps.
        self.race_t = 0.0
        self.finish_t: float | None = None
        self.last_t: float | None = None
        self.best_t: float | None = None
        #: Best time through each third of the lap, read off the splits when
        #: a lap completes. The HUD paints a player's sector purple only when
        #: it beats the AI's best as well as their own.
        self.best_sectors: list[float | None] = [None, None, None]
        #: This lap's completed splits and the sector being driven, kept the
        #: same way the player's are. Only the *bests* were recorded before,
        #: which is all the player's HUD needed from a rival -- but spectating
        #: shows the watched car's live sector lights, and those need the lap
        #: in progress.
        self.sectors: list[float | None] = [None, None, None]
        self.sector = 0
        self._sector_start = 0.0
        self._sector_idx = track.sector_bounds()
        self._armed = False
        self._last_i = 0
        #: The pilot's last output, kept so the spectator view can show the
        #: AI's pedals and steering on the same trace the player's use.
        self.controls = Controls()

    # -- driving --------------------------------------------------------
    def step(self, dt: float):
        """One fixed physics step. Does nothing while frozen."""
        if self.vehicle.frozen:
            return
        self.controls = self.driver.controls(self.vehicle)
        self.vehicle.step(self.controls, dt, self.surface)
        self.lap_time += dt
        self.race_t += dt

        i, _ = self.surface.progress(self.vehicle.pos)
        n = self.track.count

        crossing = False
        if 0.4 * n <= i <= 0.6 * n:
            self._armed = True
        elif self._armed and i < 0.1 * n:
            fwd = np.array([math.sin(self.vehicle.yaw),
                            math.cos(self.vehicle.yaw)])
            crossing = float(np.dot(self.vehicle.vel, fwd)) > 0.0

        # Every sample between the last one and this one is filled, not just
        # the one the car is nearest now: at 88 m/s a step covers 0.7 m while
        # the samples are 5 m apart, so writing only the current index leaves
        # most of them empty and the gap reads as unavailable half the time.
        if crossing:
            self._armed = False
            # The samples on each side of the line belong to different laps.
            # Filling straight through the crossing writes the *end* of the
            # old lap into the index the *start* of a lap also uses, and the
            # next lap then compares a player who has just crossed against a
            # ghost time of a full lap: the gap read -132 s for a moment every
            # time round.
            self._fill(self._last_i, n - 1, self.lap_time)
            self._cross_line()
            self._fill(-1, i, 0.0)
        else:
            gap = (i - self._last_i) % n
            if 0 < gap < n // 4:
                self._fill(self._last_i, self._last_i + gap, self.lap_time)
        sec = self.track.sector_of(i)
        if sec != self.sector:
            if sec == self.sector + 1 and self.lap_num >= 1:
                self.sectors[self.sector] = self.lap_time - self._sector_start
                self._sector_start = self.lap_time
            self.sector = sec
        self._last_i = i

    def _fill(self, after: int, upto: int, value: float):
        """Write *value* into the splits for samples (after, upto]."""
        n = self.track.count
        for k in range(after + 1, upto + 1):
            self.splits[k % n] = value

    def _cross_line(self):
        if self.lap_num > 0:
            self.last_t = self.lap_time
            self.best_t = (self.lap_time if self.best_t is None
                           else min(self.best_t, self.lap_time))
            b1, b2 = self._sector_idx
            s1, s2 = float(self.splits[b1]), float(self.splits[b2])
            for k, v in enumerate((s1, s2 - s1, self.lap_time - s2)):
                if math.isfinite(v) and v > 0.0:
                    b = self.best_sectors[k]
                    self.best_sectors[k] = v if b is None else min(b, v)
        if self.lap_num > 0:
            self.sectors[2] = self.lap_time - self._sector_start
        self.lap_num += 1
        self.sectors = [None, None, None]
        self.sector = 0
        self._sector_start = 0.0
        self.prev_splits, self.splits = self.splits, np.full(
            self.track.count, np.nan)
        self.lap_time = 0.0

    def sync(self, dt: float, alpha: float):
        self.car.sync(dt=dt, alpha=alpha)

    # -- the gap --------------------------------------------------------
    def delta(self, index: int, player_lap_time: float | None) -> float | None:
        """Seconds the player is behind the ghost at the player's position.

        Positive means the ghost got here sooner. Compared at the same point on
        the track rather than by counting who is in front: two cars a corner
        apart are not two corners' worth of time apart, and a gap in metres
        tells the driver nothing they can act on.
        """
        if player_lap_time is None:
            return None
        t = self.splits[index]
        if not np.isfinite(t):
            t = self.prev_splits[index]
        if not np.isfinite(t):
            return None
        return float(player_lap_time - t)

    # -- lifecycle ------------------------------------------------------
    def start(self):
        self.vehicle.frozen = False

    def freeze(self):
        self.vehicle.frozen = True

    def destroy(self):
        from .ui import destroy_tree

        destroy_tree(self.car)
