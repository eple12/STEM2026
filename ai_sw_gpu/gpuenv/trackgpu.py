"""Static circuit data, converted once to torch tensors.

Everything here is built by the *CPU* code in ``ai_sw/game`` (trackdata,
raceline, rlpolicy) and then frozen into tensors. Geometry is therefore
identical by construction -- only the per-step dynamics are re-implemented
in a batched form, in :mod:`physics` and :mod:`vecenv`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

AI_SW = Path(__file__).resolve().parents[2] / "ai_sw"
if str(AI_SW) not in sys.path:
    sys.path.insert(0, str(AI_SW))

from game import config                                    # noqa: E402
from game import raceline as rl                            # noqa: E402
from game import rlpolicy                                  # noqa: E402
from game.trackdata import load_track                      # noqa: E402


def _t(a, device, dtype=None):
    if dtype is None:
        dtype = torch.get_default_dtype()
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)


class TrackGPU:
    """Every array the batched env reads, on one device.

    Names mirror the CPU objects they come from: ``c_*`` is the raw
    :class:`Track` (progress, widths, barrier geometry), ``l_*`` is the
    reference :class:`~game.raceline.Line` (observation geometry), ``s_*`` is
    the shaping line (the potential attractor and the source of ``v_ref``).
    """

    def __init__(self, circuit: str, device="cpu"):
        self.name = circuit
        self.device = torch.device(device)
        dev = self.device

        track = load_track(circuit)
        line = rlpolicy.reference_line(track)
        shape = rlpolicy.shaping_line(track)
        vpace = (config.RL_RL_SPEED_PACE
                 if getattr(config, "RL_TASK", "linesight") == "raceline"
                 else 1.0)
        v_ref = rl.speed_profile(shape.seg_len, shape.curvature,
                                 shape.curv_radius, vpace)

        self.cpu_track = track          # kept for the parity harness
        self.n = int(track.count)

        # -- the track itself: progress, widths, collision reference ------
        self.c_center = _t(track.center, dev)
        self.c_normal = _t(track.normal, dev)
        self.c_tangent = _t(track.tangent, dev)
        self.c_arclen = _t(track.arclen, dev)
        self.c_len = float(track.length)
        self.w_right = _t(track.w_right, dev)
        self.w_left = _t(track.w_left, dev)

        # -- the reference line: observation geometry and progress --------
        self.l_center = _t(line.center, dev)
        self.l_normal = _t(line.normal, dev)
        self.l_tangent = _t(line.tangent, dev)
        self.l_arclen = _t(line.arclen, dev)
        self.l_len = float(line.length)

        # -- the shaping line: the potential attractor --------------------
        self.s_center = _t(shape.center, dev)
        self.s_normal = _t(shape.normal, dev)

        self.v_ref = _t(v_ref, dev)
        self.look_idx = _t(rlpolicy._lookahead_indices(line), dev, torch.long)

        # Clean-lap bonus break-even: this circuit's own analytic modelled
        # lap time (same formula as raceline.lap_time()) times RL_RL_LAP_W,
        # computed once here since v_ref is a per-circuit (TrackGPU), not
        # per-environment, quantity. See the long comment above RL_RL_LAP_W
        # in config.py -- a flat BASE was silently calibrated to Monza only.
        v_seg = np.maximum(0.5 * (v_ref + np.roll(v_ref, -1)), 1e-3)
        self.model_lap = float(np.sum(shape.seg_len / v_seg))
        self.lap_base = config.RL_RL_LAP_W * self.model_lap

        self._build_barriers(track, dev)

    # ------------------------------------------------------------------
    def _build_barriers(self, track, dev):
        """The per-sample barrier neighbourhoods, padded into one table.

        ``Surface.__init__`` keys each barrier chord by the lap distance of
        the centreline sample nearest its midpoint, and ``_barrier_hit``
        then tests only the chords within ``BARRIER_QUERY_RANGE`` of the
        query point's own sample. That selection depends on nothing but the
        geometry, so it is precomputed here into a fixed ``(2, N, K)`` index
        table plus a mask -- the batched collision test is then a gather.

        Both sides' chords live in ONE segment array so the gather does not
        have to branch; the side only picks which row of the index table to
        read, exactly as ``self._barrier[0 if side > 0 else 1]`` does.
        """
        sides = track.barrier_lines()            # (right, left)
        segs_all, near_all, owner = [], [], []
        for s, segs in enumerate(sides):
            if len(segs) == 0:
                continue
            mid = segs.mean(axis=1)
            dx = mid[:, None, 0] - track.center[None, :, 0]
            dz = mid[:, None, 1] - track.center[None, :, 1]
            nearest = np.argmin(dx * dx + dz * dz, axis=1)
            segs_all.append(np.asarray(segs, dtype=float))
            near_all.append(track.arclen[nearest])
            owner.append(np.full(len(segs), s, dtype=np.int64))

        segs = np.concatenate(segs_all, axis=0)          # (M, 2, 2)
        near = np.concatenate(near_all, axis=0)          # (M,)
        owner = np.concatenate(owner, axis=0)            # (M,) 0 = right

        lap = max(track.length, 1e-6)
        gap = np.abs(near[None, :] - track.arclen[:, None])
        gap = np.minimum(gap, lap - gap)                 # (N, M)
        in_range = gap <= config.BARRIER_QUERY_RANGE

        per_side = []
        k_max = 1
        for s in (0, 1):
            sel = in_range & (owner[None, :] == s)
            k_max = max(k_max, int(sel.sum(axis=1).max()))
            per_side.append(sel)

        idx = np.zeros((2, self.n, k_max), dtype=np.int64)
        msk = np.zeros((2, self.n, k_max), dtype=bool)
        for s in (0, 1):
            for i in range(self.n):
                js = np.nonzero(per_side[s][i])[0]
                idx[s, i, :len(js)] = js
                msk[s, i, :len(js)] = True

        a = segs[:, 0, :]
        b = segs[:, 1, :]
        d = b - a
        # Outward normal of each chord, before the per-query orientation
        # flip that ``_barrier_hit`` applies against ``track.normal[i]``.
        m = np.stack([d[:, 1], -d[:, 0]], axis=1)
        m = m / np.maximum(np.hypot(m[:, 0], m[:, 1]), 1e-9)[:, None]

        self.b_a = _t(a, dev)
        self.b_d = _t(d, dev)
        self.b_l2 = _t(np.maximum((d * d).sum(axis=1), 1e-12), dev)
        self.b_m = _t(m, dev)
        self.b_idx = _t(idx, dev, torch.long)
        self.b_msk = torch.as_tensor(msk, dtype=torch.bool, device=dev)
        self.b_k = k_max
