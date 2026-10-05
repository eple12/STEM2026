"""Mountains around the circuit -- landform only, no detail.

A circuit on an endless flat plane has no sense of place: the horizon is a
straight line and every direction looks the same. A ring of hills gives the
eye something to measure the track against and, at this sun angle, a silhouette
for the sky to sit behind.

Built as one polar grid centred on the track's bounding box, starting far
enough out that it can never intrude on the racing surface whatever shape the
circuit is. That is the cheap, safe version of "keep away from the track": for
a ring this far out the difference between distance-to-centroid and
distance-to-track is irrelevant, and the exact test would cost a full nearest
-point search per vertex.
"""
from __future__ import annotations

import numpy as np
from ursina import Entity, Mesh, scene

from . import config
from . import palette as pal


def _heights(ang: np.ndarray, rad_t: np.ndarray, seed: int = 5) -> np.ndarray:
    """Ridge height in [0, 1] over a (radial, angular) grid.

    Summed sines rather than value noise: the grid is small, the shapes only
    have to read as a skyline, and this keeps the module free of a noise
    implementation nobody else needs.
    """
    rng = np.random.default_rng(seed)
    h = np.zeros_like(ang)
    for octave, (freq, amp) in enumerate(((2.0, 1.00), (3.7, 0.55),
                                          (7.3, 0.30), (13.1, 0.16))):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        radial = 1.0 + 0.35 * np.sin(rad_t * (2.0 + octave) * np.pi + phase * 1.7)
        h += amp * radial * (0.5 + 0.5 * np.sin(ang * freq + phase))
    h /= h.max()
    # Rise over the first part of the band, then stay high: a wall of hills,
    # not a cone. Peaks are pushed outward so nothing looms at the near edge.
    ramp = np.clip(rad_t / config.MOUNTAIN_RAMP, 0.0, 1.0) ** 1.4
    return h * ramp


def ring_bounds(track) -> tuple[float, float, float, float]:
    """(centre x, centre z, inner radius, outer radius) of the mountain ring.

    Exported so the ground plane can be sized to meet the hills instead of
    ending in mid-air: a plane cut to the track's bounding box stops roughly a
    kilometre short of them, and the sky shows through the gap.
    """
    lo, hi = track.bounds()
    cx, cz = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
    # Distance from the centre to the farthest corner of the track's box: the
    # ring has to clear that, not just the box's half-width.
    reach = float(np.hypot(hi[0] - cx, hi[1] - cz))
    r0 = reach + config.MOUNTAIN_GAP
    return cx, cz, r0, r0 + config.MOUNTAIN_DEPTH


def build_mountains(track) -> Entity | None:
    """One mesh ringing the circuit. Returns the entity, or None if disabled."""
    if not getattr(config, "MOUNTAINS_ENABLED", True):
        return None
    if config.MOUNTAIN_RINGS < 2 or config.MOUNTAIN_SEGMENTS < 8:
        return None

    cx, cz, r0, r1 = ring_bounds(track)

    na, nr = config.MOUNTAIN_SEGMENTS, config.MOUNTAIN_RINGS
    ang = np.linspace(0.0, 2.0 * np.pi, na, endpoint=False)
    rad_t = np.linspace(0.0, 1.0, nr)
    A, T = np.meshgrid(ang, rad_t)                     # (nr, na)
    R = r0 + (r1 - r0) * T

    H = _heights(A, T) * config.MOUNTAIN_HEIGHT
    X = cx + np.cos(A) * R
    Z = cz + np.sin(A) * R

    verts, norms, cols, tris = [], [], [], []
    # Analytic normals from the neighbouring samples: the grid is regular, so a
    # central difference is exact enough and far cheaper than rebuilding them
    # from the triangles afterwards.
    dR = (r1 - r0) / max(nr - 1, 1)
    for j in range(nr):
        for i in range(na):
            ip, im = (i + 1) % na, (i - 1) % na
            jp, jm = min(j + 1, nr - 1), max(j - 1, 0)
            dh_da = (H[j, ip] - H[j, im]) / 2.0
            dh_dr = (H[jp, i] - H[jm, i]) / max(jp - jm, 1)
            arc = R[j, i] * (2.0 * np.pi / na)
            t_ang = np.array([-np.sin(A[j, i]) * arc, dh_da, np.cos(A[j, i]) * arc])
            t_rad = np.array([np.cos(A[j, i]) * dR, dh_dr, np.sin(A[j, i]) * dR])
            n = np.cross(t_rad, t_ang)
            n /= max(np.linalg.norm(n), 1e-9)
            if n[1] < 0:
                n = -n
            # On the grass plane, not on y=0: the ramp starts at height 0 and
            # the ground sits a few centimetres below it, which would leave a
            # lip all the way round the horizon.
            verts.append((X[j, i], H[j, i] + config.Y_GRASS, Z[j, i]))
            norms.append(tuple(n))
            # Darker in the folds, paler on the tops -- the haze will do most
            # of the work at this distance, but it stops them reading as one
            # flat cut-out on the days the sun is higher.
            # Cool and dark. Distant hills at sunset are a blue-violet
            # silhouette -- the warm haze in front of them supplies all the
            # colour they need, and a green hill this far out just reads as
            # more grass.
            f = 0.62 + 0.38 * (H[j, i] / max(config.MOUNTAIN_HEIGHT, 1e-6))
            # Wooded hills. The sky's own haze (shaders.py) turns them blue
            # with distance, as aerial perspective does; painted blue as well
            # they came out violet.
            if config.LIGHTING_PRESET == "sunset":
                cols.append(pal.rgb(int(52 * f), int(56 * f), int(84 * f)))
            else:
                cols.append(pal.rgb(int(56 * f), int(70 * f), int(50 * f)))

    for j in range(nr - 1):
        for i in range(na):
            a = j * na + i
            b = j * na + (i + 1) % na
            c = (j + 1) * na + (i + 1) % na
            d = (j + 1) * na + i
            tris += [a, b, c, a, c, d]

    return Entity(parent=scene, double_sided=True,
                  model=Mesh(vertices=verts, triangles=tris, normals=norms,
                             colors=cols, mode="triangle", static=True))
