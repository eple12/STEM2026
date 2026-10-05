"""Build the visible track (Phase A: procedural meshes only)."""
from __future__ import annotations

import math

import numpy as np
from ursina import Entity, Mesh, Vec4, color, scene

from . import config, terrain, textures, trackdata
from . import palette as pal
from .shaders import ground_shader, material, road_shader
from .trackdata import Track, _smooth_ring

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


def _heights(y, m: int) -> np.ndarray:
    """A height per vertex, from either a number or a per-sample array.

    Every strip in here used to be flat because the circuits are flat. With
    camber they are not, and the two edges of the road are at different
    heights that change all the way round the lap -- so the one number each
    rail used to carry becomes a column.
    """
    a = np.asarray(y, dtype=float)
    return np.full(m, float(a)) if a.ndim == 0 else a


def _append_strip(verts: list, uvs: list, tris: list,
                  inner: np.ndarray, outer: np.ndarray, s: np.ndarray,
                  y_inner: float, y_outer: float, uv_len: float,
                  v_tiles: float = 1.0) -> None:
    """Triangulate a quad strip into existing buffers, offsetting the indices."""
    base = len(verts)
    m = len(inner)
    yi, yo = _heights(y_inner, m), _heights(y_outer, m)
    for i in range(m):
        u = s[i] / uv_len
        verts.append((inner[i, 0], yi[i], inner[i, 1]))
        verts.append((outer[i, 0], yo[i], outer[i, 1]))
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
    yi, yo = _heights(y_inner, m), _heights(y_outer, m)
    for i in range(m):
        u = s[i] / uv_len
        verts.append((inner[i, 0], yi[i], inner[i, 1]))
        verts.append((outer[i, 0], yo[i], outer[i, 1]))
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
    # Normals from the strip's own cross-slope. Straight up is right to
    # within a degree on a flat circuit and wrong by the bank angle on a
    # banked one, which on an eighteen-degree corner is the difference
    # between a road that catches the sun and a road that does not.
    nrm = []
    for i in range(m):
        d = outer[i] - inner[i]
        run = float(np.hypot(d[0], d[1])) or 1.0
        slope = (yo[i] - yi[i]) / run
        v = np.array([-slope * d[0] / run, 1.0, -slope * d[1] / run])
        v /= np.linalg.norm(v)
        nrm += [tuple(v), tuple(v)]
    return Mesh(vertices=verts, triangles=tris, uvs=uvs,
                normals=nrm, mode="triangle")


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


def _mix(a, b, t):
    return a * (1.0 - t) + b * t


def _ramp_v(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """_ramp over a whole array."""
    t = np.clip((x - a) / max(b - a, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def mottle_v(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """mottle() over whole arrays."""
    return 1.0 + (0.065 * np.sin(x * 0.081) + 0.055 * np.sin(z * 0.063 + 1.7)
                  + 0.045 * np.sin((x + z) * 0.037 + 0.4)
                  + 0.035 * np.sin((x - z) * 0.121 + 2.3))


def _runoff_mesh(track: Track) -> Mesh:
    """Everything inside the barrier that is not the road: one mesh, both sides.

    Without this the ground goes track -> grass at the white line and the
    barrier stands thirteen metres away across an empty field, which is the
    single most "unfinished" thing about the scene: a real circuit's run-off
    is paved, and paving reads as *circuit* the way grass never does.

    This is the **interior of the drivable region**, and the barrier is that
    region's outline, so the ground and the fence are one object seen from
    two sides and there is nothing left for them to disagree about. It used to
    be a quad strip walked out along each sample's normal, which is a
    different construction, and a strip cannot cover this region: where the
    circuit turns, the outline turns with it and opens a wedge no normal
    points into. That wedge is the triangle of grass inside the fence on the
    inside of Monza's first chicane, and a quarter of Zandvoort. Clamping the
    strip only moved the problem -- pull it in and the grass plane shows
    through, push it out and run-off is painted across the countryside.

    So the region is filled as a region (``trackdata.runoff_fill``), with its
    edge snapped onto the barrier chords, and the colour of a point is decided
    from where that point actually is: how far across the run-off, how much of
    a corner it is beside, and whether it is on the outside of that corner.
    One mesh, so two aprons can no longer be laid over the same ground either.
    """
    verts2, tris, outer = trackdata.runoff_fill(
        track, config.RUNOFF_CELL, config.BARRIER_OUTSET, config.RUNOFF_SKIRT)
    n = track.count
    # Both of these used to be booleans read per sample, and a boolean that
    # flips between two rows of the mesh is a hard edge running straight out
    # across the apron -- the green wedge where a corner's gravel met the
    # straight's grass in the space of one five-metre row. Smoothed along the
    # lap they become weights, and the surface changes over fifty metres of
    # circuit instead of over one row.
    k_smooth = max(5, n // 60)
    corner = _smooth_ring(_corner_mask(track, dilate=14).astype(float), k_smooth)
    turn = turn_sign(track)
    out_r = _smooth_ring((turn == +1).astype(float), k_smooth)
    out_l = _smooth_ring((turn == -1).astype(float), k_smooth)
    reach_r, reach_l = track.corridor_reach()

    # Which sample each vertex belongs to -- everything painted here is read
    # off the circuit at that sample.
    near = track.nearest_indices(verts2)

    rel = verts2 - track.center[near]
    lat = (rel * track.normal[near]).sum(axis=1)
    right = lat > 0.0
    dist = np.abs(lat)
    w = np.where(right, track.w_right[near], track.w_left[near])
    reach = np.where(right, reach_r[near], reach_l[near])
    outside = np.where(right, out_r[near], out_l[near])
    cw = corner[near]
    # How far across the run-off, 0 at the white line and 1 at the fence. The
    # reach is the region's own edge along that sample's normal, so a corner
    # whose run-off opens out to fifty metres gets its gravel spread over all
    # fifty rather than the thirteen a straight has.
    q = np.clip((dist - w) / np.maximum(reach - w, 1.0), 0.0, 1.0)

    grass = np.array([pal.GRASS.r, pal.GRASS.g, pal.GRASS.b]) * RUNOFF_GAIN
    paved = np.array([0.355, 0.365, 0.385]) * RUNOFF_GAIN   # lighter than track
    gravel = np.array([0.560, 0.492, 0.372]) * RUNOFF_GAIN

    # Every boundary is a blend, across the apron and along the lap: sand into
    # gravel into turf, and corner into straight. Nothing is painted past the
    # fence any more, so the trap fades to turf in its last fifth instead of
    # out in the country -- which is where a real one's grass verge is.
    # The paved shoulder runs a good way out before the turf starts. It used
    # to give up almost at the white line, and with the apron now filling the
    # whole region that put a field on both sides of the fence -- which is the
    # other half of "you cannot tell the inside of the circuit from the
    # outside". A real straight has metres of paving beyond the line before
    # the grass, and the grass is a verge, not a lawn.
    r_in = _ramp_v(q, 0.24, 0.78)
    r_tr = _ramp_v(q, 0.08, 0.50)
    r_gr = _ramp_v(q, 0.84, 1.0)
    trap = _mix(_mix(paved, gravel, r_tr[:, None]), grass, r_gr[:, None])
    inner = _mix(paved, grass, r_in[:, None])
    col = _mix(grass, _mix(inner, trap, outside[:, None]), cw[:, None])
    col = np.where(outer[:, None], grass[None, :], col)

    rng = np.random.default_rng(19)
    jit = (mottle_v(verts2[:, 0], verts2[:, 1])
           + (rng.random(len(verts2)) - 0.5) * 0.05)
    col = np.clip(col * jit[:, None], 0.0, 1.0)

    # A gentle rise away from the road. Real run-off is not a billiard table,
    # and a dead-flat plane the size of a corner's gravel trap reads as one.
    #
    # And a dive under it. The fill covers the whole region, road included --
    # cutting the asphalt out of a grid fill would only put a seam where there
    # is no seam -- so the part of it under the track has to stay clear of the
    # track: twelve millimetres is under what the depth buffer resolves at the
    # far end of a straight, and gravel flickering through the racing line a
    # kilometre away is the one artifact worse than the ones this replaces.
    # The road surface under each vertex, interpolated along the lap. Taken
    # from the nearest sample instead -- which is what surface_y alone does --
    # the camber steps at every sample boundary, and on a banked corner a
    # four-metre grid cell of apron can then step *above* the road strip
    # beside it, which is run-off lying across the kerb and the racing line.
    # One formula for "how high is the ground here", shared with everything
    # that stands on it -- see Track.ground_y. Two descriptions of the same
    # ground is how the fence came to float over it.
    y = track.ground_y(verts2)
    # The skirt's far edge lands on the plane, so the ground leaves the
    # circuit at the height the rest of the world is at instead of stopping
    # at a lip. Its colour goes with it: past the fence it is the same field.
    y = np.where(outer, config.Y_GRASS, y)

    # Straight off world x and z. The obvious mapping is track-relative --
    # distance along the lap against distance across the run-off -- and it is
    # a polar coordinate system centred on every corner: the same span of lap
    # covers a fan of ground on the outside of a bend and a wedge on the
    # inside, so the grain smears into arcs radiating from the apex. That is
    # the banding across the gravel. Worse, two vertices of one triangle can
    # land on samples half a lap apart where the circuit folds back on itself,
    # and the whole texture then streaks across that triangle in one step.
    #
    # The ground texture is a grain with no direction in it, so it has nothing
    # to gain from following the lap and everything to lose from a coordinate
    # that is not continuous. In world space it tiles evenly over the whole
    # circuit, exactly like the grass plane it meets at the fence.
    u = verts2[:, 0] / RUNOFF_TILE
    v = verts2[:, 1] / RUNOFF_TILE

    # Handed over through .tolist(): a comprehension that pulls twenty
    # thousand numpy scalars out one at a time and boxes each of them costs
    # two seconds of load on its own, and every one of those seconds is spent
    # on the conversion rather than on anything about the track.
    xyz = np.stack([verts2[:, 0], y, verts2[:, 1]], axis=1)
    return Mesh(
        vertices=list(map(tuple, xyz.tolist())),
        triangles=tris.reshape(-1).tolist(),
        colors=[color.rgba(c[0], c[1], c[2], 1.0) for c in col.tolist()],
        uvs=list(map(tuple, np.stack([u, v], axis=1).tolist())),
        normals=[(0.0, 1.0, 0.0)] * len(verts2), mode="triangle", static=True)


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
            yield edge[idx], outer[idx], track.arclen[idx], np.asarray(idx)
        i = j + 1


class TrackScene:
    """Holds every track Entity so nothing gets garbage collected."""

    def __init__(self, track: Track):
        self.track = track
        self.entities: list[Entity] = []
        self._build()

    def _add(self, e: Entity, shader=None, **inputs):
        """Keep *e*, and say which world shader it wants and with what.

        The lighting assigns both (``Sunset.apply`` reads ``world_shader`` and
        ``world_inputs``): assigning a shader in Ursina writes its defaults
        over any input already on the entity, so the inputs have to go on
        after it, not here."""
        self.entities.append(e)
        if shader is not None:
            e.world_shader = shader
        e.world_inputs = inputs
        return e

    def _ground_inputs(self):
        """Mown stripes run parallel to the main straight, the way a
        circuit's groundsmen cut them for the cameras on the grid."""
        n = self.track.normal[0]
        return dict(detail_map=textures.asphalt_detail(),
                    mow=Vec4(float(n[0]), float(n[1]), 7.0, 0.11))

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
            scale=(w, 1, h),
            texture=textures.smooth_filtering(textures.grass()),
            texture_scale=(w / GRASS_TILE, h / GRASS_TILE),
        ), ground_shader, material=material(0.92, 0.5),
            **self._ground_inputs())

        # --- run-off apron ------------------------------------------------
        # Drawn before the asphalt so it is the surface the track sits on,
        # and before the kerbs so a kerb still reads on top of paved run-off.
        self._add(Entity(parent=scene, model=_runoff_mesh(t),
                         texture=textures.smooth_filtering(textures.ground()),
                         double_sided=True),
                  ground_shader, material=material(0.90, 0.6),
                  **self._ground_inputs())

        # --- asphalt (flat colour + tiny vertex jitter, no repeating texture) --
        base = (pal.ASPHALT.r, pal.ASPHALT.g, pal.ASPHALT.b)
        # The two edges of the road are at different heights wherever the
        # circuit is cambered, and the closed ring carries the seam sample
        # twice, so the columns are closed the same way the point arrays are.
        ring = np.concatenate([np.arange(t.count), [0]])
        yL = np.concatenate([t.surface_y(np.arange(t.count), -t.w_left)])
        yR = np.concatenate([t.surface_y(np.arange(t.count), t.w_right)])
        yL = np.concatenate([yL, yL[:1]])
        yR = np.concatenate([yR, yR[:1]])
        # The road shader reads the lap position off U (metres along the lap)
        # and the position across the road off V, and lays the lap map --
        # rubber, braking marks, dust, patches -- over world-space grain.
        asphalt = self._add(Entity(
            parent=scene,
            model=_tint(_strip_mesh(L, R, s, yL, yR, 1.0, 1.0), base, 0.03, 11),
            double_sided=True),
            road_shader, material=material(0.78, 1.0),
            lap_map=textures.lap_map(t), detail_map=textures.asphalt_detail(),
            lap_length=float(t.length))

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
            li = side * ((t.w_right if side > 0 else t.w_left) - 0.55)
            lo = side * ((t.w_right if side > 0 else t.w_left) - 0.05)
            yi = t.surface_y(ring[:-1], li) + config.Y_LINE
            yo = t.surface_y(ring[:-1], lo) + config.Y_LINE
            yi = np.concatenate([yi, yi[:1]])
            yo = np.concatenate([yo, yo[:1]])
            # Paint, not light: a white line at 0.94 in linear light is a
            # strip of fluorescent tube. Worn road paint is a dull 0.8.
            e = self._add(Entity(
                parent=scene,
                model=_strip_mesh(inner, outer, s, yi, yo, 1.0, 1.0),
                color=color.rgba(0.86, 0.86, 0.85, 1.0), double_sided=True),
                ground_shader, material=material(0.62, 1.0),
                **self._ground_inputs())

        # --- tar seams across the track -------------------------------
        # Regular transverse detail is the strongest optical-flow cue there is:
        # at speed these stream under the car and give the eye a beat to read.
        seams = self._add(Entity(parent=scene, model=self._seams(t),
                                 double_sided=True),
                          ground_shader, material=material(0.70, 1.0),
                          **self._ground_inputs())

        # --- kerbs ----------------------------------------------------
        # Every corner run in ONE mesh. A circuit has a dozen or two of them
        # and they all share a texture, so an entity each is a dozen or two
        # draw calls -- and, more expensively here, a dozen or two nodes for
        # Ursina to walk and Panda to cull every single frame.
        kerb_tex = textures.smooth_filtering(textures.kerb())
        verts, uvs, tris = [], [], []
        for side in (+1, -1):
            for inner, outer, seg, idx in self._kerb_iter(side):
                w = (t.w_right if side > 0 else t.w_left)[idx]
                yi = t.surface_y(idx, side * w) + config.Y_KERB
                yo = t.surface_y(idx, side * (w + config.KERB_WIDTH))                     + config.Y_KERB + 0.03
                _append_strip(verts, uvs, tris, inner, outer, seg,
                              yi, yo, KERB_UV_LEN, 1.0)
        if verts:
            self._add(Entity(
                parent=scene, texture=kerb_tex, double_sided=True,
                model=Mesh(vertices=verts, triangles=tris, uvs=uvs,
                           normals=[(0.0, 1.0, 0.0)] * len(verts),
                           mode="triangle")),
                ground_shader, material=material(0.55, 1.0),
                **self._ground_inputs())

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
        ya = float(t.surface_y(i0, -half_l)) + config.Y_START
        yb = float(t.surface_y(i0, half_r)) + config.Y_START
        verts = [
            (a[0], ya, a[1]), (b[0], yb, b[1]),
            (b[0] + d[0], yb, b[1] + d[1]),
            (a[0] + d[0], ya, a[1] + d[1]),
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
            texture=textures.smooth_filtering(textures.checker()),
            double_sided=True),
            ground_shader, material=material(0.60, 1.0),
            **self._ground_inputs())

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
            ya = float(t.surface_y(i, -t.w_left[i])) + config.Y_SEAM
            yb = float(t.surface_y(i, t.w_right[i])) + config.Y_SEAM
            verts += [(a[0], ya, a[1]), (b[0], yb, b[1]),
                      (b[0] + d[0], yb, b[1] + d[1]),
                      (a[0] + d[0], ya, a[1] + d[1])]
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


def grid_boxes(track: Track, poses, light):
    """The starting grid painted on the road: for each car's slot -- *poses*
    is [(x, z, yaw)], exactly where the field put the cars -- a white bar
    across the slot just ahead of the nose, and a short leg back from each
    end of it, the bracket every F1 grid box is drawn with. One mesh, lit
    like the other road paint. Returns the entity (the session owns it)."""
    bar_w, bar_d, leg = 2.6, 0.22, 1.4
    verts, tris = [], []

    def quad(p0, p1, p2, p3):
        k = len(verts)
        verts.extend((p0, p1, p2, p3))
        tris.extend((k, k + 1, k + 2, k, k + 2, k + 3))

    for x, z, yaw in poses:
        fx, fz = math.sin(yaw), math.cos(yaw)
        rx, rz = fz, -fx
        cx = x + fx * (config.BODY_TO_FRONT + 0.45)
        cz = z + fz * (config.BODY_TO_FRONT + 0.45)
        h = bar_w / 2.0

        def at(a, b):
            # a metres to the right, b metres forward, from the bar's centre
            return (cx + rx * a + fx * b, cz + rz * a + fz * b)
        corners = [at(-h, 0.0), at(h, 0.0), at(h, bar_d), at(-h, bar_d),
                   at(-h, -leg), at(-h + bar_d, -leg), at(-h + bar_d, 0.0),
                   at(h - bar_d, -leg), at(h, -leg), at(h - bar_d, 0.0)]
        hy, _b, _n = track.surface_pose(corners)
        p = [(c[0], float(y) + config.Y_LINE, c[1]) for c, y in zip(corners, hy)]
        quad(p[0], p[1], p[2], p[3])          # the bar
        quad(p[4], p[5], p[6], p[0])          # left leg
        quad(p[7], p[8], p[1], p[9])          # right leg
    e = Entity(parent=scene, model=Mesh(vertices=verts, triangles=tris,
                                        normals=[(0.0, 1.0, 0.0)] * len(verts),
                                        mode="triangle", static=True),
               color=color.rgba(0.86, 0.86, 0.85, 1.0), double_sided=True)
    n = track.normal[0]
    e.world_shader = ground_shader
    e.world_inputs = dict(detail_map=textures.asphalt_detail(),
                          mow=Vec4(float(n[0]), float(n[1]), 7.0, 0.11))
    light.apply(e, casts=False, material=material(0.62, 1.0))
    return e


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
