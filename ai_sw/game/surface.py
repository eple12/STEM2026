"""Track-relative collision and surface queries used by the vehicle integrator.

Cheap and robust: everything is expressed as a lateral offset from the nearest
centreline sample, so we never touch 3D mesh collision.
"""
from __future__ import annotations

import math

import numpy as np

from . import config
from .structures import box_hit, obstacles
from .trackdata import Track


class Surface:
    def __init__(self, track: Track):
        self.track = track
        self.hint = 0
        #: The legs of the gantry and the bridges (structures.obstacles).
        self._legs = obstacles(track)
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
        # Per centreline sample and side: the chords within query range, with
        # their outward normals already oriented, plus a conservative bound
        # used to skip the test outright. Everything here depended only on
        # (sample, side), yet was rebuilt for every corner of both cars at
        # every physics step -- the single largest cost in a frame. Kept on the
        # track, so the player's and the AI's Surface share one build.
        cand = getattr(track, "_barrier_candidates", None)
        if cand is None:
            cand = [self._candidates(k, s) for k, s in ((0, 1), (1, -1))]
            track._barrier_candidates = cand
        self._cand = cand
        # Plain lists for the per-step lookups (shared through the track).
        lists = getattr(track, "_surface_lists", None)
        if lists is None:
            lists = (track.center.tolist(), track.normal.tolist(),
                     track.w_right.tolist(), track.w_left.tolist(),
                     cand[0][1].tolist(), cand[1][1].tolist())
            track._surface_lists = lists
        (self._cl, self._nl, self._wr, self._wl,
         self._clear_r, self._clear_l) = lists

    def _candidates(self, k: int, side: int):
        """(chord table per sample, clear distance per sample) for one side.

        The clear distance is the least, over the candidate chords, of how far
        the sample sits inside each chord's outward face. A point closer than
        that (less the rail's half depth) to the sample is inside every one of
        those faces, so it cannot be touching any of them -- an exact early
        out, not an approximation.
        """
        t = self.track
        segs, near = self._barrier[k]
        n = t.count
        table = [None] * n
        clear = np.full(n, np.inf)
        if len(segs) == 0:
            return table, clear
        lap = max(t.length, 1e-6)
        a_all = segs[:, 0]
        d_all = segs[:, 1] - a_all
        for i in range(n):
            gap = np.abs(near - t.arclen[i])
            gap = np.minimum(gap, lap - gap)
            sel = np.nonzero(gap <= config.BARRIER_QUERY_RANGE)[0]
            if len(sel) == 0:
                continue
            a = a_all[sel]
            d = d_all[sel]
            L2 = np.maximum((d * d).sum(axis=1), 1e-12)
            m = np.stack([d[:, 1], -d[:, 0]], axis=1)
            m /= np.maximum(np.hypot(m[:, 0], m[:, 1]), 1e-9)[:, None]
            flip = (m @ (t.normal[i] * side)) < 0.0
            m[flip] = -m[flip]
            table[i] = (a, d, L2, m)
            clear[i] = float(((a - t.center[i]) * m).sum(axis=1).min())
        return table, clear

    def _local(self, pos_xz):
        i = self.track.nearest_index(pos_xz, self.hint)
        self.hint = i
        cx, cz = self._cl[i]
        nx, nz = self._nl[i]
        offset = (float(pos_xz[0]) - cx) * nx + (float(pos_xz[1]) - cz) * nz
        edge = self._wr[i] if offset > 0 else self._wl[i]
        return i, offset, edge

    @staticmethod
    def _surface_at(d: float, edge: float) -> float:
        """Grip multiplier for one contact patch at lateral distance *d*."""
        if d <= edge:
            return 1.0
        if d <= edge + config.KERB_WIDTH:
            return config.KERB_GRIP_SCALE
        return config.OFF_TRACK_GRIP_SCALE

    def camber(self, pos_xz) -> tuple[float, np.ndarray, float]:
        """(bank angle, the track normal there, surface height) at a point.

        One call, because everything that wants one of these wants the others:
        the physics needs the angle and the direction it falls in, and the car
        and the camera need the height so they sit on the road rather than
        through it.
        """
        t = self.track
        i, offset, _edge = self._local(pos_xz)
        if not config.BANKING_ENABLED:
            # A flat circuit: surface_y is exactly zero everywhere.
            return 0.0, t.normal[i], 0.0
        return (float(t.bank()[i]), t.normal[i],
                float(t.surface_y(i, offset)))

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

        cx0, cz0 = float(pos_xz[0]), float(pos_xz[1])
        sy, cy = math.sin(yaw), math.cos(yaw)
        ht = config.WHEEL_HALF_TRACK
        rx, rz = cy * ht, -sy * ht
        self._local((cx0, cz0))              # refresh the search hint
        nearest = self.track.nearest_index
        cl, nl, wr, wl = self._cl, self._nl, self._wr, self._wl
        inside = False
        total = 0.0
        for along in (config.CG_TO_FRONT, -config.CG_TO_REAR):
            ax, az = cx0 + sy * along, cz0 + cy * along
            for px, pz in ((ax + rx, az + rz), (ax - rx, az - rz)):
                i = nearest((px, pz), self.hint)
                cx, cz = cl[i]
                nx, nz = nl[i]
                offset = (px - cx) * nx + (pz - cz) * nz
                edge = wr[i] if offset > 0 else wl[i]
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
        # Chords within query range, with outward normals oriented away from
        # the track -- precomputed per sample, see _candidates.
        entry = self._cand[0 if side > 0 else 1][0][i]
        if entry is None:
            return -1e9, None
        a, d, L2, m = entry
        t = np.clip(((p - a) * d).sum(axis=1) / L2, 0.0, 1.0)
        q = a + d * t[:, None]

        # Signed distance past each chord's centre line, plus the half
        # thickness that puts the contact on the face the eye sees rather than
        # the middle of the rail.
        u = ((p - a) * m).sum(axis=1) + config.BARRIER_HALF_DEPTH

        # Only a chord the car is actually *alongside* can stop it. Measuring
        # against the infinite line instead was an invisible wall generator:
        # wherever a run ends -- and it now ends at every chicane, where the
        # barrier wraps the complex instead of threading it -- the last
        # chord's half-space carries straight on across the road, and the car
        # stops dead in the middle of the track against nothing at all. Past
        # an end the chord only reaches as far as a rounded cap, so a car out
        # in the open never meets it.
        cap = config.BARRIER_HALF_DEPTH + config.BODY_HALF_WIDTH
        along = (t > 0.0) & (t < 1.0)
        near_end = ((p - q) ** 2).sum(axis=1) <= cap * cap
        # ...and only to a believable depth. Round the outside of a bend the
        # far chords face back across the circuit, so a car on the racing line
        # is tens of metres "through" them.
        u = np.where((along | near_end) & (u <= config.BARRIER_MAX_PENETRATION),
                     u, -1e9)

        # Deepest, not nearest: a corner buried past a joint between two
        # chords is inside both, and the one it is further through is the one
        # that has to push it out.
        k = int(np.argmax(u))
        return float(u[k]), m[k]

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
        cx0, cz0 = float(pos_xz[0]), float(pos_xz[1])
        self._local((cx0, cz0))              # refresh the search hint
        sy, cy = math.sin(yaw), math.cos(yaw)
        hwx, hwz = cy * config.BODY_HALF_WIDTH, -sy * config.BODY_HALF_WIDTH
        nf, nr = config.BODY_TO_FRONT, -config.BODY_TO_REAR
        corners = ((cx0 + sy * nf + hwx, cz0 + cy * nf + hwz),
                   (cx0 + sy * nf - hwx, cz0 + cy * nf - hwz),
                   (cx0 + sy * nr + hwx, cz0 + cy * nr + hwz),
                   (cx0 + sy * nr - hwx, cz0 + cy * nr - hwz))

        deep, hit_m = 0.0, None
        touching = []
        hd = config.BARRIER_HALF_DEPTH
        cl, nl = self._cl, self._nl
        for px, pz in corners:
            i = t.nearest_index((px, pz), self.hint)
            ccx, ccz = cl[i]
            rx, rz = px - ccx, pz - ccz
            nx, nz = nl[i]
            side = 1 if rx * nx + rz * nz > 0 else -1
            # Inside every candidate chord's face by more than the rail's half
            # depth: no contact is possible, so skip the chord test.
            clear = (self._clear_r if side > 0 else self._clear_l)[i]
            if math.hypot(rx, rz) + hd < clear - 1e-6:
                continue
            p = np.array((px, pz))
            pen, m = self._barrier_hit(p, i, side)
            if m is None or pen <= 0.0:
                continue
            touching.append(p)
            if pen > deep:
                deep, hit_m = pen, m
        # The legs of the gantry and the bridges: solid, like the barrier.
        leg = None
        for box in self._legs:
            h = box_hit(cx0, cz0, yaw, nf, -nr, config.BODY_HALF_WIDTH, box)
            if h is not None and h[0] > deep and (leg is None or h[0] > leg[0]):
                leg = h
        c = np.array((cx0, cz0))
        if leg is not None:
            depth, nx, nz, px, pz = leg
            wall_normal = np.array((nx, nz))
            return c + wall_normal * depth, wall_normal, np.array((px, pz)) - c
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
