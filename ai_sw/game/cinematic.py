"""A short camera film over the circuit, played before the grid countdown.

Dropping the player straight onto the grid tells them nothing about where they
are. Ten seconds of the circuit first -- two of its real corners from the
outside, then a look down the main straight -- does, and it costs nothing but
a camera path: every shot is aimed at geometry that is already built.

Two rules hold the whole thing together.

**Nothing is ever still.** A locked-off shot of a static scene is a
photograph, and cutting between four photographs reads as a slideshow. Every
shot moves the camera and its aim point slowly and in a straight line, so
there is always parallax; the speed is well under what would read as flying.

**Framed on the circuit, not on the county.** The aerial is pitched down hard
and held low enough that the run-off fills the frame. Pulled back far enough
to see the whole lap it also sees the empty country outside it, which is the
one thing this scene cannot carry.
"""
from __future__ import annotations

import math

import numpy as np
from ursina import Vec3, camera

from . import config


def _ease(t: float) -> float:
    """Smoothstep. A shot that starts and stops dead reads as a jump cut."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


class Shot:
    """One move: camera from *a* to *b*, aim from *pa* to *pb*.

    *path* and *aim*, when given, replace the straight lines with functions of
    the eased parameter. A corner shot needs that: the outside of a bend is an
    arc, and a straight line between two points on an arc is a chord that cuts
    across everything inside it -- which at a corner is the run-off, the kerb
    and the road. Following the circuit's own samples keeps the camera outside
    the bend for the whole move instead of only at its ends.
    """

    def __init__(self, a, b, pa, pb, dur, fov=None, path=None, aim=None):
        self.a, self.b, self.pa, self.pb = (Vec3(*v) for v in (a, b, pa, pb))
        self.dur = float(dur)
        self.fov = fov or config.CAM_FOV_BASE
        self.path, self.aim = path, aim

    def apply(self, t: float) -> None:
        f = _ease(t / self.dur)
        camera.position = self.path(f) if self.path else self.a + (self.b - self.a) * f
        camera.fov = self.fov
        camera.look_at(self.aim(f) if self.aim else self.pa + (self.pb - self.pa) * f)
        # look_at leaves whatever roll the camera arrived with, and the race
        # camera leans into corners -- so the first shot of the film inherited
        # the last frame of the previous race and stood the horizon on end.
        camera.rotation_z = 0.0


def _pose(track, i: int):
    return track.center[i % track.count], track.tangent[i % track.count], \
        track.normal[i % track.count]


def _pose_at(track, fi: float):
    """Centre / tangent / normal at a *fractional* sample index.

    The corner shots walk only a few dozen samples over their whole run, so
    snapping the camera and its aim to the nearest integer sample -- which is
    what ``int(round(...))`` did -- stepped them from one centreline point to
    the next. At 50 mm of asphalt per step that reads as the camera juddering
    along the corner. Interpolating between the two bracketing samples makes
    the move continuous.
    """
    n = track.count
    i0 = int(math.floor(fi)) % n
    a = fi - math.floor(fi)
    i1 = (i0 + 1) % n
    b = 1.0 - a
    return (track.center[i0] * b + track.center[i1] * a,
            track.tangent[i0] * b + track.tangent[i1] * a,
            track.normal[i0] * b + track.normal[i1] * a)


def _lerp_at(arr, fi: float, n: int) -> float:
    i0 = int(math.floor(fi)) % n
    a = fi - math.floor(fi)
    return float(arr[i0]) * (1.0 - a) + float(arr[(i0 + 1) % n]) * a


def _corner_shot(track, i: int, dur: float, low: bool = False) -> Shot:
    """A low tracking shot along the outside barrier, looking in at the road.

    The camera rides just above the outer guardrail and moves along it through
    the corner, aimed at the centreline a little ahead of itself. Because it
    looks *inward* -- across the asphalt at the inside of the bend -- the
    treeline is behind the lens the whole time, and what fills the frame is
    the road turning away. Earlier cuts of this shot sat high and outside and
    looked along or past the barrier, which framed the forest instead.

    *low* tightens it for the tightest corner on the lap: a little lower, a
    little narrower.
    """
    from .trackmesh import turn_sign

    n_ = track.count
    side = -float(turn_sign(track)[i % n_]) or 1.0
    reach_r, reach_l = track.corridor_reach()
    # Track through the corner: from the braking zone to just past the apex.
    j0 = i - n_ * (0.030 if low else 0.038)
    j1 = i + n_ * (0.010 if low else 0.014)

    def stand(f: float):
        """A point hard against the outside barrier, *f* along the run."""
        fi = j0 + (j1 - j0) * f
        c, _t, n = _pose_at(track, fi)
        room = _lerp_at(reach_r if side > 0 else reach_l, fi, n_)
        w = _lerp_at(track.w_right if side > 0 else track.w_left, fi, n_)
        # Out at the barrier line, held a touch inside it so a car length of
        # run-off is still between the lens and the fence -- never back among
        # the trees that stand behind it.
        off = min(max(w + 2.5, room - 1.0), w + 9.0)
        return c + n * side * off

    y0, y1 = (1.9, 1.6) if low else (2.9, 2.4)

    def path(f: float) -> Vec3:
        p = stand(f)
        return Vec3(p[0], y0 + (y1 - y0) * f, p[1])

    # Aim at the centreline a short way ahead of the camera's own position, so
    # the look direction crosses the track towards the inside of the bend.
    a0 = j0 + n_ * 0.018
    a1 = j1 + n_ * 0.024

    def aim(f: float) -> Vec3:
        c, _t, _n = _pose_at(track, a0 + (a1 - a0) * f)
        return Vec3(c[0], 0.8, c[1])

    p0, p1 = path(0.0), path(1.0)
    q0, q1 = aim(0.0), aim(1.0)
    return Shot(p0, p1, q0, q1, dur, fov=44.0 if low else 50.0,
                path=path, aim=aim)


def _straight_shot(track, i: int, dur: float) -> Shot:
    """Down the main straight at grid height, creeping forward."""
    c, t, n = _pose(track, i)
    a = c - t * 70.0 + n * 6.0
    b = c - t * 40.0 + n * 4.0
    d0 = c + t * 120.0
    d1 = c + t * 150.0
    return Shot((a[0], 3.2, a[1]), (b[0], 2.4, b[1]),
                (d0[0], 2.0, d0[1]), (d1[0], 2.0, d1[1]), dur, fov=60.0)


def _aerial_shot(track, i: int, dur: float) -> Shot:
    """High and behind, descending towards the grid.

    Height is set from the road, not from the circuit's bounding box: framed
    to fit the whole lap this would also frame the empty ground outside it.
    """
    c, t, n = _pose(track, i)
    a = c - t * 210.0 + n * 90.0
    b = c - t * 120.0 + n * 40.0
    d0 = c + t * 40.0
    d1 = c + t * 10.0
    return Shot((a[0], 120.0, a[1]), (b[0], 46.0, b[1]),
                (d0[0], 0.0, d0[1]), (d1[0], 0.0, d1[1]), dur, fov=56.0)


def intro_film(track) -> list[Shot]:
    """The shot list for one circuit: its two hardest corners, then the grid.

    The corners are picked by radius rather than by hand, so a circuit that
    has never been seen before still opens on the two turns that define it.
    """
    from .scenery import _corners

    corners = _corners(track)
    picks = []
    for run, _side in corners[:4]:
        i = int(run[len(run) // 2])
        # Keep them apart: two shots of the same complex is one shot twice.
        if all(min(abs(i - j), track.count - abs(i - j)) > track.count * 0.08
               for j in picks):
            picks.append(i)
        if len(picks) == 2:
            break

    # The first pick is the tightest corner on the lap -- _corners sorts by
    # radius -- and that is the one that earns the low angle.
    film = [_corner_shot(track, k, config.INTRO_CORNER_TIME, low=(n == 0))
            for n, k in enumerate(picks)]
    film.append(_aerial_shot(track, 0, config.INTRO_AERIAL_TIME))
    film.append(_straight_shot(track, 0, config.INTRO_STRAIGHT_TIME))
    return film


class Intro:
    """Plays the film, then reports done. Skippable."""

    def __init__(self, track):
        self.shots = intro_film(track)
        self.t = 0.0
        self.k = 0
        self.done = not self.shots

    @property
    def total(self) -> float:
        return sum(s.dur for s in self.shots)

    def skip(self) -> None:
        self.done = True

    def update(self, dt: float) -> None:
        if self.done:
            return
        self.t += dt
        while self.k < len(self.shots) and self.t >= self.shots[self.k].dur:
            self.t -= self.shots[self.k].dur
            self.k += 1
        if self.k >= len(self.shots):
            self.done = True
            return
        self.shots[self.k].apply(self.t)
