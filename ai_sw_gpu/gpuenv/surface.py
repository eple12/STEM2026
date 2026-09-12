"""Batched twin of ``game.surface.Surface``.

Same queries, same answers, B environments at a time. The search ``hint`` is
per-environment state and is advanced at exactly the same call sites as the
scalar version, so a batched rollout walks the same index sequence a serial
one does.
"""
from __future__ import annotations

import torch

from .trackgpu import config


class VecSurface:
    def __init__(self, track, n_env: int):
        self.t = track
        self.device = track.device
        self.hint = torch.zeros(n_env, dtype=torch.long, device=self.device)

    # -- nearest centreline sample -------------------------------------
    def nearest_index(self, pts, hints, window: int = 10):
        """``Track.nearest_index`` for many (point, hint) pairs at once.

        The wrapping window is the fast path; landing on its edge means the
        hint was stale, and those rows fall back to a full scan -- the same
        two-tier search the scalar version does.
        """
        t = self.t
        n = t.n
        off = torch.arange(-window, window + 1, device=self.device)
        idx = (hints[:, None] + off[None, :]) % n              # (M, 2w+1)
        d = t.c_center[idx] - pts[:, None, :]
        k = (d * d).sum(-1).argmin(1)
        best = idx.gather(1, k[:, None]).squeeze(1)

        stale = (k == 0) | (k == 2 * window)
        if bool(stale.any()):
            sel = stale.nonzero(as_tuple=True)[0]
            dd = t.c_center[None, :, :] - pts[sel][:, None, :]
            best = best.clone()
            best[sel] = (dd * dd).sum(-1).argmin(1)
        return best

    def _local(self, pos):
        """Nearest sample / signed offset / edge width, and refresh the hint."""
        t = self.t
        i = self.nearest_index(pos, self.hint)
        self.hint = i
        offset = ((pos - t.c_center[i]) * t.c_normal[i]).sum(-1)
        edge = torch.where(offset > 0, t.w_right[i], t.w_left[i])
        return i, offset, edge

    def progress(self, pos):
        i, offset, _ = self._local(pos)
        return i, offset

    def refresh(self, pos):
        """What ``camber()`` does to the hint. Banking itself is off
        (``config.BANKING_ENABLED`` is False), so there is nothing else to
        return -- but the hint advance is part of the sequence."""
        self._local(pos)

    # -- surface type ---------------------------------------------------
    @staticmethod
    def _surface_at(d, edge):
        return torch.where(
            d <= edge,
            torch.ones_like(d),
            torch.where(d <= edge + config.KERB_WIDTH,
                        torch.full_like(d, config.KERB_GRIP_SCALE),
                        torch.full_like(d, config.OFF_TRACK_GRIP_SCALE)))

    def _wheel_points(self, pos, yaw, half_track, front, rear):
        fwd = torch.stack([torch.sin(yaw), torch.cos(yaw)], -1)
        right = torch.stack([torch.cos(yaw), -torch.sin(yaw)], -1) * half_track
        pts = torch.stack([
            pos + fwd * front + right,
            pos + fwd * front - right,
            pos - fwd * rear + right,
            pos - fwd * rear - right,
        ], dim=1)                                           # (B, 4, 2)
        return pts

    def grip(self, pos, yaw):
        """(within track limits, mean grip) judged at the four contact patches.

        Within limits while ANY wheel is inside the white line, and grip is
        the mean over the four patches -- the racing rule and the partial
        kerb, exactly as the scalar version.
        """
        t = self.t
        b = pos.shape[0]
        self._local(pos)                       # refresh the search hint
        pts = self._wheel_points(pos, yaw, config.WHEEL_HALF_TRACK,
                                 config.CG_TO_FRONT, config.CG_TO_REAR)
        flat = pts.reshape(-1, 2)
        hints = self.hint[:, None].expand(b, 4).reshape(-1)
        i = self.nearest_index(flat, hints)
        offset = ((flat - t.c_center[i]) * t.c_normal[i]).sum(-1)
        edge = torch.where(offset > 0, t.w_right[i], t.w_left[i])
        d = offset.abs()
        inside = (d <= edge).reshape(b, 4).any(1)
        grip = self._surface_at(d, edge).reshape(b, 4).mean(1)
        return inside, grip

    # -- barriers -------------------------------------------------------
    def _barrier_hit(self, p, i, side):
        """(penetration, outward normal) for many points at once.

        Measured against the half-space outside the nearest chord's face --
        not against the barrier slab, which a 1.5 m step could tunnel -- and
        only for a chord the point is actually alongside (or within a rounded
        cap of an end), to a believable depth.
        """
        t = self.t
        row = torch.where(side > 0, torch.zeros_like(i), torch.ones_like(i))
        js = t.b_idx[row, i]                               # (M, K)
        msk = t.b_msk[row, i]
        a = t.b_a[js]                                      # (M, K, 2)
        d = t.b_d[js]
        l2 = t.b_l2[js]
        m = t.b_m[js]

        pa = p[:, None, :] - a
        u_t = ((pa * d).sum(-1) / l2).clamp(0.0, 1.0)
        q = a + d * u_t[..., None]

        ref = t.c_normal[i] * side[:, None].to(p.dtype)     # (M, 2)
        flip = (m * ref[:, None, :]).sum(-1) < 0.0
        m = torch.where(flip[..., None], -m, m)

        u = (pa * m).sum(-1) + config.BARRIER_HALF_DEPTH

        cap = config.BARRIER_HALF_DEPTH + config.BODY_HALF_WIDTH
        along = (u_t > 0.0) & (u_t < 1.0)
        near_end = ((p[:, None, :] - q) ** 2).sum(-1) <= cap * cap
        ok = ((along | near_end)
              & (u <= config.BARRIER_MAX_PENETRATION) & msk)
        u = torch.where(ok, u, torch.full_like(u, -1e9))

        # Deepest, not nearest: a corner buried past a joint is inside both
        # chords and the deeper one has to push it out.
        k = u.argmax(1)
        pen = u.gather(1, k[:, None]).squeeze(1)
        nrm = m.gather(1, k[:, None, None].expand(-1, 1, 2)).squeeze(1)
        return pen, nrm

    def resolve_body(self, pos, yaw):
        """Push the body box out of the barrier.

        Returns ``(hit, corrected centre, inward normal, contact arm)``. The
        arm is what turns a clipped front corner into a spin and a square hit
        into a straight stop.
        """
        t = self.t
        b = pos.shape[0]
        self._local(pos)                       # refresh the search hint
        pts = self._wheel_points(pos, yaw, config.BODY_HALF_WIDTH,
                                 config.BODY_TO_FRONT, config.BODY_TO_REAR)
        flat = pts.reshape(-1, 2)
        hints = self.hint[:, None].expand(b, 4).reshape(-1)
        i = self.nearest_index(flat, hints)
        signed = ((flat - t.c_center[i]) * t.c_normal[i]).sum(-1)
        side = torch.where(signed > 0, torch.ones_like(i), -torch.ones_like(i))
        pen, nrm = self._barrier_hit(flat, i, side)

        pen = pen.reshape(b, 4)
        nrm = nrm.reshape(b, 4, 2)
        touching = pen > 0.0
        hit = touching.any(1)

        k = pen.argmax(1)
        deep = pen.gather(1, k[:, None]).squeeze(1).clamp(min=0.0)
        hit_m = nrm.gather(1, k[:, None, None].expand(-1, 1, 2)).squeeze(1)

        wall_normal = -hit_m                   # points back towards the track
        corrected = pos + wall_normal * deep[:, None]
        cnt = touching.sum(1).clamp(min=1).to(pos.dtype)
        contact = (pts * touching[..., None]).sum(1) / cnt[:, None]
        arm = contact - pos
        return hit, corrected, wall_normal, arm
