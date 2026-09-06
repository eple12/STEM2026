"""Track-relative collision and surface queries used by the vehicle integrator.

Cheap and robust: everything is expressed as a lateral offset from the nearest
centreline sample, so we never touch 3D mesh collision.
"""
from __future__ import annotations

import math

import numpy as np

from . import config
from .trackdata import Track


class Surface:
    def __init__(self, track: Track):
        self.track = track
        self.hint = 0
        # The barrier chords, plus the centreline sample nearest each one, so a
        # query only has to look at the handful of segments beside the car.
        # Done once: it is a few hundred segments against a thousand samples.
        self._barrier = []
        for segs in track.barrier_lines():
            if len(segs) == 0:
                self._barrier.append((segs, np.zeros(0, dtype=int)))
                continue
            mid = segs.mean(axis=1)
            dx = mid[:, None, 0] - track.center[None, :, 0]
            dz = mid[:, None, 1] - track.center[None, :, 1]
            near = np.argmin(dx * dx + dz * dz, axis=1)
            # Keyed by distance along the lap, not by sample count. Circuits
            # do not share a sample spacing -- Shanghai's is fine enough that a
            # twelve-sample window reached about a car's length, so on its long
            # straights no barrier segment was ever in range and the car
            # stopped up to 26 m early, against nothing.
            self._barrier.append((segs, track.arclen[near]))

    def _local(self, pos_xz):
        t = self.track
        i = t.nearest_index(pos_xz, self.hint)
        self.hint = i
        offset = float(np.dot(np.asarray(pos_xz) - t.center[i], t.normal[i]))
        edge = t.w_right[i] if offset > 0 else t.w_left[i]
        return i, offset, edge

    @staticmethod
    def _surface_at(d: float, edge: float) -> float:
        """Grip multiplier for one contact patch at lateral distance *d*."""
        if d <= edge:
            return 1.0
        if d <= edge + config.KERB_WIDTH:
            return config.KERB_GRIP_SCALE
        return config.OFF_TRACK_GRIP_SCALE

    def grip(self, pos_xz, yaw: float | None = None) -> tuple[bool, float]:
        """(within track limits, grip multiplier) -- asphalt / kerb / grass.

        Tested at the four contact patches, not at the centre of mass. Two
        things follow, and both matter.

        The rule is the racing one: you are within track limits while **any**
        wheel is still inside the white line. Judging it from the centre gave
        the car a corridor a whole body-width narrower than the rule allows --
        1.21 m less each side -- which is most of the margin a chicane is
        taken with.

        And grip is the mean over the four patches rather than one lookup, so
        putting two wheels on the kerb costs half of what putting four on it
        costs. A single centre sample makes that step change all at once, which
        is what a car does when it teleports, not when it runs wide.
        """
        if yaw is None:
            _, offset, edge = self._local(pos_xz)
            d = abs(offset)
            return d <= edge + config.KERB_WIDTH, self._surface_at(d, edge)

        c = np.asarray(pos_xz, dtype=float)
        fwd = np.array([math.sin(yaw), math.cos(yaw)])
        right = np.array([math.cos(yaw), -math.sin(yaw)]) * config.WHEEL_HALF_TRACK
        self._local(c)                       # refresh the search hint
        inside = False
        total = 0.0
        for along in (fwd * config.CG_TO_FRONT, -fwd * config.CG_TO_REAR):
            for side in (right, -right):
                p = c + along + side
                i = self.track.nearest_index(p, self.hint)
                offset = float(np.dot(p - self.track.center[i],
                                      self.track.normal[i]))
                edge = float(self.track.w_right[i] if offset > 0
                             else self.track.w_left[i])
                d = abs(offset)
                inside = inside or d <= edge
                total += self._surface_at(d, edge)
        return inside, total / 4.0

    def on_track(self, pos_xz, yaw: float | None = None) -> bool:
        """Within track limits. Pass *yaw* to judge it at the wheels."""
        return self.grip(pos_xz, yaw)[0]

    def _barrier_hit(self, p, i: int, side: int):
        """(penetration in metres, outward unit normal) for one point.

        Penetration is measured against the *half-space* outside the nearest
        barrier segment's face, not against the 0.6 m slab of barrier itself. A
        slab can be tunnelled: a step at 88 m/s covers 1.5 m, so the car would
        pass clean through the barrier between two frames and end up in the
        scenery.
        """
        segs, near = self._barrier[0 if side > 0 else 1]
        if len(segs) == 0:
            return -1e9, None
        lap = max(self.track.length, 1e-6)
        gap = np.abs(near - self.track.arclen[i])
        gap = np.minimum(gap, lap - gap)
        sel = np.nonzero(gap <= config.BARRIER_QUERY_RANGE)[0]
        if len(sel) == 0:
            return -1e9, None

        a = segs[sel, 0]
        b = segs[sel, 1]
        d = b - a
        L2 = np.maximum((d * d).sum(axis=1), 1e-12)
        t = np.clip(((p - a) * d).sum(axis=1) / L2, 0.0, 1.0)
        q = a + d * t[:, None]
        k = int(np.argmin(((p - q) ** 2).sum(axis=1)))

        # Outward normal of the chosen chord, oriented away from the track.
        m = np.array([d[k][1], -d[k][0]], dtype=float)
        m /= max(float(np.hypot(m[0], m[1])), 1e-9)
        if float(np.dot(m, self.track.normal[i] * side)) < 0.0:
            m = -m
        # Signed distance past the chord's centre line, plus the half thickness
        # that puts the contact on the face the eye sees rather than the middle
        # of the rail.
        u = float(np.dot(p - a[k], m))
        return u + config.BARRIER_HALF_DEPTH, m

    def resolve_body(self, pos_xz, yaw: float):
        """Push the car's body box out of the barrier.

        Returns ``(corrected centre, inward normal, contact arm)`` or three
        Nones. The arm is the contact point relative to the centre of mass,
        which is what the integrator needs to turn a clipped front corner into
        a spin.

        Two things this deliberately does not do.

        It does not test the centre point alone -- that lets half the car bury
        itself in a barrier before anything registers, and no impact can ever
        rotate the car, because a force through the centre of mass has no
        moment. Both are the same omission: the body has extent.

        And it does not compare a lateral offset against the wall distance at
        the nearest centreline sample. That is a different curve from the one
        the barriers are built along -- a chord across a bend leaves the arc it
        was cut from -- and over four circuits the gap between them reached
        4.97 m. The car stopped dead in open runoff with the barrier still
        metres away. It now hits the segments the props are placed on.
        """
        t = self.track
        c = np.asarray(pos_xz, dtype=float)
        self._local(c)                       # refresh the search hint
        fwd = np.array([math.sin(yaw), math.cos(yaw)])
        right = np.array([math.cos(yaw), -math.sin(yaw)])
        hw = right * config.BODY_HALF_WIDTH
        nose, tail = fwd * config.BODY_TO_FRONT, -fwd * config.BODY_TO_REAR
        corners = (c + nose + hw, c + nose - hw, c + tail + hw, c + tail - hw)

        deep, hit_m = 0.0, None
        touching = []
        for p in corners:
            i = t.nearest_index(p, self.hint)
            side = 1 if float(np.dot(p - t.center[i], t.normal[i])) > 0 else -1
            pen, m = self._barrier_hit(p, i, side)
            if m is None or pen <= 0.0:
                continue
            touching.append(p)
            if pen > deep:
                deep, hit_m = pen, m
        if hit_m is None:
            return None, None, None

        wall_normal = -hit_m                 # points back towards the track
        corrected = c + wall_normal * deep
        # Contact at the centroid of whatever is actually inside the barrier:
        # two corners for a square hit, so the arm lies on the body's
        # centreline and the car does not spin; one corner for a clip, so it
        # does.
        contact = np.mean(np.asarray(touching), axis=0)
        return corrected, wall_normal, contact - c

    # -- telemetry for HUD / future AI ---------------------------------
    def progress(self, pos_xz) -> tuple[int, float]:
        """(nearest index, signed lateral offset in metres)."""
        i, offset, _ = self._local(pos_xz)
        return i, offset

    def rangefinders(self, pos_xz, yaw: float, rel_angles,
                     max_range: float = 100.0) -> np.ndarray:
        """Distance from the car to the track edge along each ray.

        The GT Sophy / GTS-SAC observation that the discrete IQN run lacked:
        rays fanned out ahead of the car, each returning how far it is to the
        wall in that direction. It is the most direct signal there is for "am I
        about to run out of room", which is what a clean lap needs.

        Approximated against the track-edge polylines (centre +- normal * width)
        rather than the barrier chords -- close enough, and it vectorises to a
        handful of microseconds. A window of edge segments around the car is
        enough; a 100 m ray down a straight still only spans ~20 samples.
        """
        t = self.track
        i = t.nearest_index(pos_xz, self.hint)
        n = t.count
        span = 1 + int(max_range / max(float(np.median(t.seg_len)), 1.0))
        idx = (i - 3 + np.arange(span + 6)) % n
        left = t.center[idx] - t.normal[idx] * t.w_left[idx][:, None]
        right = t.center[idx] + t.normal[idx] * t.w_right[idx][:, None]
        a = np.vstack([left[:-1], right[:-1]])          # segment starts
        b = np.vstack([left[1:], right[1:]])            # segment ends
        seg = b - a
        p = np.asarray(pos_xz, dtype=float)

        ang = yaw + np.asarray(rel_angles, dtype=float)         # (A,)
        d = np.stack([np.sin(ang), np.cos(ang)], axis=1)        # (A, 2)
        perp = np.stack([-d[:, 1], d[:, 0]], axis=1)            # (A, 2)
        ap = a - p                                              # (S, 2)
        denom = seg @ perp.T                                    # (S, A)
        ok = np.abs(denom) > 1e-9
        u = np.where(ok, (ap @ perp.T) / np.where(ok, denom, 1.0), -1.0)
        cross = ap[:, None, :] + u[:, :, None] * seg[:, None, :]  # (S, A, 2)
        s = (cross * d[None, :, :]).sum(-1)                     # (S, A)
        s = np.where(ok & (u >= 0.0) & (u <= 1.0) & (s > 0.0), s, np.inf)
        return np.minimum(s.min(axis=0), max_range).astype(np.float32)
