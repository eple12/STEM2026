"""Build the visible track (Phase A: procedural meshes only)."""
from __future__ import annotations

import math

import numpy as np
from ursina import Entity, Mesh, color, scene

from . import config, terrain, textures
from . import palette as pal
from .trackdata import Track

ASPHALT_UV_LEN = 12.0     # metres per texture repeat, lengthwise
# Bigger than it was (8 m). A small tile on a plane that reaches the
# mountains repeats hundreds of times and turns to shimmer; at this size
# the texture's broad patches read as ground undulation instead.
GRASS_TILE = 13.0
RUNOFF_TILE = 9.0         # metres per repeat of the apron's grain
# textures.ground() averages 0.79, and a modulate texture can only darken.
# The band colours are mixed this much brighter so the grain lands them
# back on the tone they were picked at.
RUNOFF_GAIN = 1.27
KERB_UV_LEN = 2.0

# Layering is done with real height separation (config.Y_*), not with polygon
# offsets. That only works because the near plane is set sanely -- see
# config.CLIP_NEAR for why the depth buffer could not resolve these gaps before.


def _ring(track: Track):
    """Left/right edge points as (N+1, 2) arrays, loop closed with a seam-free
    duplicate, plus a matching cumulative-length array."""
    c, nrm = track.center, track.normal
    L = c - nrm * track.w_left[:, None]
    R = c + nrm * track.w_right[:, None]
    L = np.vstack([L, L[:1]])
    R = np.vstack([R, R[:1]])
    s = np.concatenate([track.arclen, [track.length]])
    return L, R, s


def _append_strip(verts: list, uvs: list, tris: list,
                  inner: np.ndarray, outer: np.ndarray, s: np.ndarray,
                  y_inner: float, y_outer: float, uv_len: float,
                  v_tiles: float = 1.0) -> None:
    """Triangulate a quad strip into existing buffers, offsetting the indices."""
    base = len(verts)
    m = len(inner)
    for i in range(m):
        u = s[i] / uv_len
        verts.append((inner[i, 0], y_inner, inner[i, 1]))
        verts.append((outer[i, 0], y_outer, outer[i, 1]))
        uvs.append((u, 0.0))
        uvs.append((u, v_tiles))
    for i in range(m - 1):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2
        tris += [base + a, base + b, base + c, base + a, base + c, base + d]


def _strip_mesh(inner: np.ndarray, outer: np.ndarray, s: np.ndarray,
                y_inner: float, y_outer: float, uv_len: float,
                v_tiles=1.0) -> Mesh:
    """Triangulate a quad strip between two poly-lines (already closed)."""
    verts, uvs, tris = [], [], []
    m = len(inner)
    for i in range(m):
        u = s[i] / uv_len
        verts.append((inner[i, 0], y_inner, inner[i, 1]))
        verts.append((outer[i, 0], y_outer, outer[i, 1]))
        uvs.append((u, 0.0))
        uvs.append((u, v_tiles))
    for i in range(m - 1):
        # vertices around the quad, in order: inner_i, outer_i, outer_j, inner_j
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2
        tris += [a, b, c, a, c, d]
    # Straight up. These strips are the road surface and its markings -- all
    # within a few centimetres of flat -- so the true normal is (0,1,0) to well
    # inside a degree, and stating it costs nothing. Mesh.generate_normals is
    # the alternative and its smooth path is O(n^2) over the vertex list, which
    # on a 1159-sample circuit does not finish.
    return Mesh(vertices=verts, triangles=tris, uvs=uvs,
                normals=[(0.0, 1.0, 0.0)] * len(verts), mode="triangle")


def _tint(mesh: Mesh, base, amt: float, seed: int) -> Mesh:
    """Bake a subtle per-vertex brightness jitter so flat colour isn't dead flat."""
    rng = np.random.default_rng(seed)
    n = len(mesh.vertices)
    k = 1.0 + (rng.random(n) - 0.5) * 2.0 * amt
    mesh.colors = [color.rgba(base[0] * f, base[1] * f, base[2] * f, 1.0)
                   for f in k]
    mesh.generate()
    return mesh


def _corner_mask(track: Track, dilate: int = 8) -> np.ndarray:
    m = track.curv_radius < config.KERB_CURVATURE_RADIUS
    out = m.copy()
    for k in range(1, dilate + 1):
        out |= np.roll(m, k) | np.roll(m, -k)
    return out


def turn_sign(track: Track) -> np.ndarray:
    """+1 where the circuit turns left, -1 right, at every sample.

    The outside of a left-hander is the right-hand side, so this is also the
    test for "which side of the track needs the big run-off".
    """
    k = max(2, track.count // 200)
    a = track.tangent
    b = np.roll(track.tangent, -k, axis=0)
    return np.sign(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def mottle(x: float, z: float) -> float:
    """A smooth 0.8-1.2 multiplier that varies over tens of metres.

    Per-vertex random jitter is the obvious way to break up flat ground and it
    does not work: at four rails across an apron it is higher-frequency than
    the mesh can carry, so it averages to the same flat tone. Ground reads as
    ground when the variation is *broad* -- patches of drier and greener turf
    metres across -- which is what a handful of incommensurable sines gives,
    for no memory and no texture lookup.
    """
    return 1.0 + (0.065 * math.sin(x * 0.081) + 0.055 * math.sin(z * 0.063 + 1.7)
                  + 0.045 * math.sin((x + z) * 0.037 + 0.4)
                  + 0.035 * math.sin((x - z) * 0.121 + 2.3))


def _runoff_mesh(track: Track, side: int) -> Mesh:
    """The apron between the asphalt edge and the barrier line, on one side.

    Without this the ground goes track -> grass at the white line and the
    barrier stands thirteen metres away across an empty field, which is the
    single most "unfinished" thing about the scene: a real circuit's run-off
    is paved, and paving reads as *circuit* the way grass never does.

    One vertex-coloured mesh rather than several: the surface changes with
    where you are on the lap, and blending it across a shared vertex is both
    cheaper and softer than butting three separately-coloured strips together.
    """
    n = track.count
    corner = _corner_mask(track, dilate=14)
    outside = turn_sign(track) == side
    off_r, off_l = track.wall_offsets()
    off = off_r if side > 0 else off_l
    w = track.w_right if side > 0 else track.w_left

    grass = np.array([pal.GRASS.r, pal.GRASS.g, pal.GRASS.b]) * RUNOFF_GAIN
    paved = np.array([0.355, 0.365, 0.385]) * RUNOFF_GAIN   # lighter than track
    gravel = np.array([0.560, 0.492, 0.372]) * RUNOFF_GAIN

    # Four rails across the apron: edge, then two intermediate bands, then the
    # wall. Bands are fractions of the local width, so this follows a run-off
    # that opens out at a corner instead of stopping short of the barrier.
    # The last two rails are *past* the barrier. The apron used to stop dead at
    # the wall, so a gravel trap met the grass plane on a hard line -- and a
    # gravel trap that ends in a straight edge is the one thing that says
    # "painted polygon" loudest. These carry the surface out beyond the wall
    # and fade it into the turf.
    fracs = (0.0, 0.42, 0.74, 1.0, 1.16, 1.42)
    span = np.maximum(off - w, 0.5)

    # Per-vertex brightness jitter on top of the band colours. Four rails is
    # far too coarse a mesh to carry detail on its own, so the grain comes from
    # textures.ground() through the UVs below; this only breaks up the long
    # even runs the bands would otherwise have.
    rng = np.random.default_rng(19)

    verts, tris, cols, uvs = [], [], [], []
    y = config.Y_RUNOFF
    for i in range(n + 1):
        k = i % n
        base = track.center[k] + track.normal[k] * side * w[k]
        step = track.normal[k] * side * span[k]
        u = float(track.arclen[k]) / RUNOFF_TILE
        for b, f in enumerate(fracs):
            p = base + step * f
            verts.append((p[0], y, p[1]))
            # v follows real metres across the apron, so the grain does not
            # stretch where the run-off opens out at a corner.
            uvs.append((u, float(w[k] + span[k] * f) / RUNOFF_TILE))
            if not corner[k]:
                c = grass
            elif not outside[k]:
                # Inside of a corner: a paved strip to run wide onto, then
                # straight back to grass -- there is nothing to catch there.
                c = paved if b <= 1 else grass
            elif b <= 1:
                c = paved
            elif b == 2:
                c = gravel
            else:
                # Past the wall: mix towards turf over the last two rails, so
                # the trap has a scruffy edge instead of a cut one.
                mix = (b - 2) / 3.0
                c = gravel * (1.0 - mix) + grass * mix
            j = mottle(float(p[0]), float(p[1])) + (rng.random() - 0.5) * 0.05
            cols.append(color.rgba(c[0] * j, c[1] * j, c[2] * j, 1.0))
    m = len(fracs)
    for i in range(n):
        a0 = i * m
        b0 = (i + 1) * m
        for b in range(m - 1):
            tris += [a0 + b, b0 + b, b0 + b + 1, a0 + b, b0 + b + 1, a0 + b + 1]
    return Mesh(vertices=verts, triangles=tris, colors=cols, uvs=uvs,
                normals=[(0.0, 1.0, 0.0)] * len(verts), mode="triangle",
                static=True)


def _kerb_segments(track: Track, side: int):
    """Yield (inner_pts, outer_pts, s) for contiguous corner runs on one side.
    side = +1 -> right edge, -1 -> left edge."""
    mask = _corner_mask(track)
    n = track.count
    c, nrm = track.center, track.normal
    if side > 0:
        edge = c + nrm * track.w_right[:, None]
        outer = edge + nrm * config.KERB_WIDTH
    else:
        edge = c - nrm * track.w_left[:, None]
        outer = edge - nrm * config.KERB_WIDTH

    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        idx = list(range(i, min(j + 1, n)))
        if len(idx) >= 2:
            yield edge[idx], outer[idx], track.arclen[idx]
        i = j + 1


class TrackScene:
    """Holds every track Entity so nothing gets garbage collected."""

    def __init__(self, track: Track):
        self.track = track
        self.entities: list[Entity] = []
        self._build()

    def _add(self, e: Entity):
        self.entities.append(e)
        return e

    def _build(self):
        t = self.track
        L, R, s = _ring(t)

        # --- grass ground -------------------------------------------------
        # Sized to reach the mountains rather than the track's bounding box.
        # A square whose half-side is the ring's inner radius contains the
        # whole of that circle -- the corners run on under the hills, where
        # they cannot be seen -- so the horizon is closed in every direction.
        cx, cz, r_inner, _ = terrain.ring_bounds(t)
        w = h = 2.0 * (r_inner + config.MOUNTAIN_DEPTH * config.MOUNTAIN_RAMP)
        grass = self._add(Entity(
            parent=scene, model="plane", position=(cx, config.Y_GRASS, cz),
            scale=(w, 1, h), texture=textures.grass(),
            texture_scale=(w / GRASS_TILE, h / GRASS_TILE),
        ))

        # --- run-off apron ------------------------------------------------
        # Drawn before the asphalt so it is the surface the track sits on,
        # and before the kerbs so a kerb still reads on top of paved run-off.
        ground_tex = textures.ground()
        for side in (+1, -1):
            self._add(Entity(parent=scene, model=_runoff_mesh(t, side),
                             texture=ground_tex, double_sided=True))

        # --- asphalt (flat colour + tiny vertex jitter, no repeating texture) --
        base = (pal.ASPHALT.r, pal.ASPHALT.g, pal.ASPHALT.b)
        asphalt = self._add(Entity(
            parent=scene,
            model=_tint(_strip_mesh(L, R, s, 0.0, 0.0, 1.0, 1.0), base, 0.06, 11),
            double_sided=True))

        # --- white edge lines -----------------------------------------
        # Widened from 0.30 m to 0.50 m: below roughly a pixel of screen width
        # a line stops resolving at all, and a real circuit's edge line is
        # 10-15 cm at 1:1 -- but this one has to stay readable a kilometre away.
        c, nrm = t.center, t.normal
        for side in (+1, -1):
            if side > 0:
                inner = c + nrm * (t.w_right[:, None] - 0.55)
                outer = c + nrm * (t.w_right[:, None] - 0.05)
            else:
                inner = c - nrm * (t.w_left[:, None] - 0.05)
                outer = c - nrm * (t.w_left[:, None] - 0.55)
            inner = np.vstack([inner, inner[:1]])
            outer = np.vstack([outer, outer[:1]])
            e = self._add(Entity(
                parent=scene,
                model=_strip_mesh(inner, outer, s, config.Y_LINE, config.Y_LINE,
                                  1.0, 1.0),
                color=color.rgba(0.94, 0.94, 0.94, 1.0), double_sided=True))

        # --- tar seams across the track -------------------------------
        # Regular transverse detail is the strongest optical-flow cue there is:
        # at speed these stream under the car and give the eye a beat to read.
        seams = self._add(Entity(parent=scene, model=self._seams(t),
                                 double_sided=True))

        # --- kerbs ----------------------------------------------------
        # Every corner run in ONE mesh. A circuit has a dozen or two of them
        # and they all share a texture, so an entity each is a dozen or two
        # draw calls -- and, more expensively here, a dozen or two nodes for
        # Ursina to walk and Panda to cull every single frame.
        kerb_tex = textures.kerb()
        verts, uvs, tris = [], [], []
        for side in (+1, -1):
            for inner, outer, seg in self._kerb_iter(side):
                _append_strip(verts, uvs, tris, inner, outer, seg,
                              config.Y_KERB, config.Y_KERB + 0.03,
                              KERB_UV_LEN, 1.0)
        if verts:
            self._add(Entity(
                parent=scene, texture=kerb_tex, double_sided=True,
                model=Mesh(vertices=verts, triangles=tris, uvs=uvs,
                           normals=[(0.0, 1.0, 0.0)] * len(verts),
                           mode="triangle")))

        # No wall strip here any more. It was a grey ribbon standing behind the
        # barrier models, and once the barriers followed the same line
        # (Track.wall_offsets()) and were tall enough to cover it, all it did
        # was show through wherever a straight module spanned a curve. The
        # barriers in scenery.py are the wall now; the collision test reads the
        # same offsets, so nothing about where you can drive has changed.

        # --- start / finish line -------------------------------------
        i0 = 0
        fwd = t.tangent[i0]
        nn = t.normal[i0]
        half_l = t.w_left[i0]
        half_r = t.w_right[i0]
        p = t.center[i0]
        a = p - nn * half_l
        b = p + nn * half_r
        d = fwd * 4.0
        verts = [
            (a[0], config.Y_START, a[1]), (b[0], config.Y_START, b[1]),
            (b[0] + d[0], config.Y_START, b[1] + d[1]),
            (a[0] + d[0], config.Y_START, a[1] + d[1]),
        ]
        sf = self._add(Entity(
            parent=scene,
            model=Mesh(vertices=verts, triangles=[0, 1, 2, 0, 2, 3],
                       uvs=[(0, 0), (8, 0), (8, 2), (0, 2)],
                       # Flat, like every other ground strip. Without this the
                       # start line came out a third as bright as the asphalt
                       # beside it: with no normal column Panda hands the
                       # shader whatever it likes, and N.L was 0.10 instead of
                       # the 0.38 the sun's elevation calls for.
                       normals=[(0.0, 1.0, 0.0)] * 4, mode="triangle"),
            texture=textures.checker(), double_sided=True))

    # -- transverse tar seams -----------------------------------------
    def _seams(self, t: Track, spacing: float = 9.0, width: float = 0.30):
        verts, tris, cols = [], [], []
        col = color.rgba(0.30, 0.31, 0.34, 1.0)
        targets = np.arange(0.0, t.length, spacing)
        idx = np.searchsorted(t.arclen, targets).clip(0, t.count - 1)
        for i in idx:
            c, n_, tg = t.center[i], t.normal[i], t.tangent[i]
            a = c - n_ * t.w_left[i]
            b = c + n_ * t.w_right[i]
            d = tg * width
            k = len(verts)
            y = config.Y_SEAM
            verts += [(a[0], y, a[1]), (b[0], y, b[1]),
                      (b[0] + d[0], y, b[1] + d[1]),
                      (a[0] + d[0], y, a[1] + d[1])]
            cols += [col] * 4
            tris += [k, k + 1, k + 2, k, k + 2, k + 3]
        return Mesh(vertices=verts, triangles=tris, colors=cols,
                    normals=[(0.0, 1.0, 0.0)] * len(verts),
                    mode="triangle", static=True)

    # -- small wrappers so _build stays readable -----------------------
    def _strip_entity(self, inner, outer, s, yi, yo, uv_len, v_tiles,
                      texture=None, col=None):
        e = Entity(parent=scene,
                   model=_strip_mesh(inner, outer, s, yi, yo, uv_len, v_tiles),
                   texture=texture)
        if col is not None:
            e.color = col
        return self._add(e)

    def _kerb_iter(self, side):
        return _kerb_segments(self.track, side)


def line_markers(track: Track, spacing: float = 6.0):
    """Dots on the road along the centreline and the imported racing line.

    A debug overlay. The AI's reference and the line it is being compared
    against are otherwise invisible, so where a policy actually drives -- and
    whether the reference it was given is sane -- can only be read off numbers.
    Two rows of dots put both on the track where they can be seen.

    Returns the entities, or an empty list if the circuit has no stored line.
    """
    from . import f1tenth

    out = []
    step = max(1, int(spacing / max(float(np.median(track.seg_len)), 1e-3)))
    rows = [(np.zeros(track.count), pal.rgb(90, 200, 255), config.Y_LINE + 0.02)]
    off = f1tenth.load_raceline(track)
    if off is not None:
        rows.append((off, pal.rgb(255, 120, 60), config.Y_LINE + 0.03))

    for offset, col, y in rows:
        pts = track.center + track.normal * offset[:, None]
        verts, tris, cols = [], [], []
        r = 0.55
        for p in pts[::step]:
            n = len(verts)
            verts += [(p[0] - r, y, p[1] - r), (p[0] + r, y, p[1] - r),
                      (p[0] + r, y, p[1] + r), (p[0] - r, y, p[1] + r)]
            cols += [col] * 4
            tris += [n, n + 1, n + 2, n, n + 2, n + 3]
        # Unlit: these are an instrument, not scenery. Under the sunset rig
        # the same dots came out muddy brown and teal, which is the one thing a
        # marker may not be -- hard to pick out from the asphalt.
        out.append(Entity(
            parent=scene, unlit=True,
            model=Mesh(vertices=verts, triangles=tris,
                       colors=cols, mode="triangle", static=True)))
    return out
