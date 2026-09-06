"""Read the optimised racing line that ships with the f1tenth track data.

The circuits here come from ``f1tenth_racetracks``, and every folder that has a
``*_centerline.csv`` mostly also has a ``*_raceline.csv`` -- a minimum-curvature
trajectory solved by TUM's global race trajectory optimiser, with a speed
profile alongside it. Twenty-one of the twenty-three circuits have one.

    # s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2

That is a far better starting line than anything solved here: it comes from a
proper optimal-control solve rather than the bounded least-squares in
``raceline.min_curvature``, which minimises a discrete curvature estimate and
turned out to be exploitable by the very search that used it.

What transfers and what does not
--------------------------------
The **geometry** transfers. A minimum-curvature line is mostly a statement
about the shape of the track, and the shape is the same whatever drives it.

The **speed profile does not**. It was solved for a 1:10 RC car -- the widths
in the centreline files are RC widths, which is why ``load_track`` renormalises
them -- and its 8 m/s straight-line speed has nothing to say about a GT car
doing 88. The line is imported; the speed is recomputed from our own physics.
"""
from __future__ import annotations

import numpy as np

from . import config
from .trackdata import Track, _resolve_csv


def raceline_path(name: str):
    """Path to the circuit's raceline file, or None if it has none."""
    folder = config.TRACK_DB / name
    if not folder.is_dir():
        return None
    for cand in (folder / f"{name}_raceline.csv", *folder.glob("*_raceline.csv")):
        if cand.exists():
            return cand
    return None


def load_raceline(track: Track):
    """The stored line as a lateral offset per centreline sample, or None.

    Returned in *our* units and indexed by *our* samples, so it drops straight
    into ``raceline.Line`` and can be compared with anything the search
    produces. The two files are sampled independently -- 2197 raceline points
    against 1159 centreline ones on Monza -- so each of our samples takes the
    offset of the nearest point on the imported line, measured along our own
    normal.
    """
    path = raceline_path(track.name)
    if path is None:
        return None

    raw = np.loadtxt(path, delimiter=";", comments="#")
    if raw.ndim != 2 or raw.shape[1] < 3:
        return None
    scale = config.TRACK_SCALE_BY_NAME.get(track.name, config.TRACK_SCALE)
    xy = raw[:, 1:3] * scale

    # Nearest imported point to each of our samples. A few thousand against a
    # few thousand is a small enough cross product to do outright, once, and
    # the correspondence comes out monotone (index steps of 1 to 4) because
    # both files run the same way round the same loop.
    dx = xy[None, :, 0] - track.center[:, None, 0]
    dz = xy[None, :, 1] - track.center[:, None, 1]
    near = np.argmin(dx * dx + dz * dz, axis=1)
    delta = xy[near] - track.center
    offset = np.einsum("ij,ij->i", delta, track.normal)

    # Carried across as a *fraction of the track's half-width*, not as metres.
    # ``load_track`` scales the geometry by TRACK_SCALE but renormalises the
    # widths onto a real circuit's 13.5 m, and the source is an RC track that
    # would otherwise come out about 28 m wide. The imported line was solved
    # against those wide edges, so in metres it sits a mean 8.1 m off the
    # centreline where our half-width is 6.75 -- outside the track everywhere,
    # clipping to the bound at every sample, which is how a minimum-curvature
    # line arrived swinging 71 m per 100 m and unable to complete a lap.
    #
    # What a racing line actually says is "this far across the track", and
    # that is what transfers.
    src = np.loadtxt(_resolve_csv(track.name), delimiter=",", comments="#")
    if np.linalg.norm(src[0, :2] - src[-1, :2]) < 1e-3:
        src = src[:-1]
    if len(src) != track.count:
        return None
    src_r = np.maximum(src[:, 2] * scale, 1e-6)
    src_l = np.maximum(src[:, 3] * scale, 1e-6)

    frac = np.where(offset > 0.0, offset / src_r, offset / src_l)

    # Smoothed before it is mapped back. The source widths are noisy sample to
    # sample -- ``load_track`` keeps only 30% of their variation for exactly
    # that reason -- and dividing by them puts that noise straight into the
    # line. A racing line has no high-frequency content in it: anything at the
    # sample scale is measurement noise, and left in it made the imported line
    # swing 44 m per 100 m of track, three times what the tightest chicane
    # asks for.
    win = max(3, int(config.F1TENTH_SMOOTH / (track.length / track.count)) | 1)
    kern = np.hanning(win + 2)[1:-1]
    kern /= kern.sum()
    frac = np.convolve(np.r_[frac[-win:], frac, frac[:win]], kern,
                       mode="same")[win:-win]
    frac = np.clip(frac, -1.0, 1.0)
    offset = np.where(frac > 0.0, frac * track.w_right, frac * track.w_left)

    from .raceline import bounds

    lo, hi = bounds(track)
    return np.clip(offset, lo, hi)
