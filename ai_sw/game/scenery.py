"""Roadside objects: what makes a strip of asphalt read as a circuit.

Sense of speed is a perception problem, not a physics one: without nearby
objects streaming past there is no optical flow, and a car at 300 km/h on an
empty green plane looks parked. But a circuit also has to *read* as a circuit,
and that is a question of what goes where, not of how much of it there is.

The layout rule here is that furniture goes where the real thing would put it:

* **Crowds see things.** Grandstands go on the main straight and on the
  outside of the corners that matter -- not evenly round the lap on both
  sides, which is what this used to do and which reads as wallpaper. Six
  clusters with open country between them give the eye something to arrive at.
* **Barriers protect from something.** Tyre walls go on the outside of
  corners, where a car that lets go actually arrives.
* **Signs are read at speed.** Distance boards count down into the braking
  zone of the corners that have one, on the side the driver is looking.
* **Buildings are at the start.** Pit garages, the pit wall, race control and
  the start gantry are all anchored to the start/finish line.

Two prop kits are in use. The Kenney racing kit (CC0, see
``assets/models/kenney/LICENSE.txt``) still supplies barriers, light posts,
cones, flags and trees. Everything specific to a circuit -- stands, tyre
walls, hoardings, debris fence, pit buildings, gantry, bridges, marshal posts,
distance boards -- is the purpose-built kit in ``blender/circuit_kit.py``,
baked to .bam by ``tools/build_blender_scenery.py``.

Every class of object is one flattened batch (see props.py), so the whole
roadside is a handful of draw calls no matter how many objects are in it.

Long runs are laid down as *stretched* copies of a part built to be stretched
(see the circuit kit's module docstring) rather than as thousands of modules:
a two-hundred-metre grandstand is one copy, and it has no seam.
"""
from __future__ import annotations

import math

import numpy as np
from ursina import Entity, Mesh, color, scene

from . import config
from . import palette as pal
from . import textures
from .props import PropLibrary, yaw_at, yaw_towards
from .trackdata import Track
from .trackmesh import mottle, turn_sign

# One face at a time, so each gets its own normal. Sharing eight corners
# between six faces is cheaper but averages the normals across them, and a box
# lit with averaged corner normals reads as a soft blob rather than a post.
# (Mesh.generate_normals is not an option either: its smoothing path is O(n^2)
# over the vertex list.)
_FACES = [
    # (corner offsets, normal, triangle order within the four corners)
    ((( -.5, 0, -.5), (.5, 0, -.5), (.5, 0, .5), (-.5, 0, .5)),
     (0, -1, 0), (0, 2, 1, 0, 3, 2)),                        # bottom
    ((( -.5, 1, -.5), (.5, 1, -.5), (.5, 1, .5), (-.5, 1, .5)),
     (0, 1, 0), (0, 1, 2, 0, 2, 3)),                          # top
    ((( -.5, 0, -.5), (.5, 0, -.5), (.5, 1, -.5), (-.5, 1, -.5)),
     (0, 0, -1), (0, 1, 2, 0, 2, 3)),                         # -z
    (((.5, 0, -.5), (.5, 0, .5), (.5, 1, .5), (.5, 1, -.5)),
     (1, 0, 0), (0, 1, 2, 0, 2, 3)),                          # +x
    (((.5, 0, .5), (-.5, 0, .5), (-.5, 1, .5), (.5, 1, .5)),
     (0, 0, 1), (0, 1, 2, 0, 2, 3)),                          # +z
    ((( -.5, 0, .5), (-.5, 0, -.5), (-.5, 1, -.5), (-.5, 1, .5)),
     (-1, 0, 0), (0, 1, 2, 0, 2, 3)),                         # -x
]


class MeshBuilder:
    """Accumulates coloured boxes into one mesh."""

    def __init__(self):
        self.verts: list[tuple] = []
        self.tris: list[int] = []
        self.cols: list = []
        self.norms: list[tuple] = []

    def box(self, cx, cz, sx, sy, sz, base_col, y0=0.0, yaw=0.0):
        """Add an axis-box with real per-face normals."""
        ca, sa = math.cos(yaw), math.sin(yaw)
        for corners, nrm, order in _FACES:
            base = len(self.verts)
            for ox, oy, oz in corners:
                px, pz = ox * sx, oz * sz
                self.verts.append((cx + px * ca + pz * sa, y0 + oy * sy,
                                   cz - px * sa + pz * ca))
                self.cols.append(base_col)
                self.norms.append((nrm[0] * ca + nrm[2] * sa, nrm[1],
                                   -nrm[0] * sa + nrm[2] * ca))
            self.tris += [base + k for k in order]

    def build(self) -> Mesh | None:
        if not self.verts:
            return None
        return Mesh(vertices=self.verts, triangles=self.tris, colors=self.cols,
                    normals=self.norms, mode="triangle", static=True)


# --- geometry helpers ----------------------------------------------------
def _sample_every(track: Track, spacing: float) -> np.ndarray:
    """Indices spaced roughly *spacing* metres apart along the centreline."""
    if track.length <= 0:
        return np.array([0])
    targets = np.arange(0.0, track.length, spacing)
    return np.searchsorted(track.arclen, targets).clip(0, track.count - 1)


def _edge(track: Track, i: int, side: int, extra: float) -> np.ndarray:
    """A point *extra* metres beyond the asphalt edge on the given side."""
    w = track.w_right[i] if side > 0 else track.w_left[i]
    return track.center[i] + track.normal[i] * side * (w + extra)


def _nearest_segment(track: Track, side: int, p):
    """(closest point on the nearest barrier chord, its outward normal)."""
    segs = track.barrier_lines()[0 if side > 0 else 1]
    if len(segs) == 0:
        return None
    a, b = segs[:, 0], segs[:, 1]
    d = b - a
    L2 = np.maximum((d * d).sum(axis=1), 1e-9)
    t = np.clip(((p - a) * d).sum(axis=1) / L2, 0.0, 1.0)
    q = a + d * t[:, None]
    k = int(((p - q) ** 2).sum(axis=1).argmin())
    m = np.array([d[k][1], -d[k][0]], dtype=float)
    m /= max(float(np.hypot(m[0], m[1])), 1e-9)
    if float(np.dot(m, q[k] - p)) < 0.0:
        m = -m
    return q[k], m


def _beyond_wall(track: Track, i: int, side: int, extra: float) -> np.ndarray:
    """A point *extra* metres outside the barrier line on the given side.

    Measured off the fence's own chords, not by walking out along the
    centreline normal by a stored radius. The wall is the outline of the
    run-off, not an offset of the road: where the outline fans out round a
    corner it stands metres past any radius, so a building "set back from the
    barrier" by the second measurement is a building the fence runs through.
    Falls back to the radius only if this side has no barrier at all.
    """
    seg = _nearest_segment(track, side, track.center[i])
    if seg is not None:
        return seg[0] + seg[1] * extra
    off_r, off_l = track.wall_ray_offsets()
    off = (off_r if side > 0 else off_l)[i]
    return track.center[i] + track.normal[i] * side * (off + extra)


def _across_span(track: Track, i: int, clear: float,
                 min_extra: float = config.CAR_BODY_WIDTH,
                 leg_frac: float = 1.0
                 ) -> tuple[np.ndarray, float]:
    """See structures.across_span: shared with the physics, which collides
    the cars with the legs of what is placed here."""
    from .structures import across_span
    return across_span(track, i, clear, min_extra, leg_frac)


def _nearest_index(track: Track, p) -> int:
    d = track.center - np.asarray(p, dtype=float)
    return int(np.argmin(d[:, 0] ** 2 + d[:, 1] ** 2))


def _outward(track: Track, i: int, side: int) -> np.ndarray:
    """Unit vector from the centreline towards *side*, at sample *i*.

    Taken from the track's own normal rather than rotated out of the chord
    direction. Rotating a chord by 90 degrees gives a perpendicular but not
    necessarily the *outward* one, and picking the wrong sign silently puts
    every grandstand on the racing line -- which is exactly what it did.
    """
    return track.normal[i] * float(side)


def _outer_side(track: Track) -> int:
    """+1 or -1: which side of the circuit is *outside* the loop.

    A closed lap has an inside and an outside, and a grandstand belongs on the
    outside -- there is nothing to build on in the middle of a circuit, and a
    stand there faces the back of the far straight. Derived from the
    centreline's signed area and the handedness of its normal rather than from
    a hard-coded side, because the two differ per circuit and per data set.
    """
    x, z = track.center[:, 0], track.center[:, 1]
    area = float(np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z))
    t, n = track.tangent, track.normal
    hand = float(np.sum(t[:, 0] * n[:, 1] - t[:, 1] * n[:, 0]))
    return -1 if area * hand > 0 else 1


def _clear(track: Track, p, need: float) -> bool:
    """True if *p* is at least *need* metres from every part of the lap.

    "Outside the barrier" is not the same as "clear of the track". A circuit
    folds back on itself, and at a chicane the wall line can be a few metres
    from the next leg -- so a twelve-metre-deep grandstand set back from the
    wall there ends up standing on the corner after it. This is the check the
    old module-by-module layout did with a separating-axis test; against the
    whole centreline it is both simpler and stricter.
    """
    d = track.center - np.asarray(p, dtype=float)
    return float(np.min(d[:, 0] ** 2 + d[:, 1] ** 2)) >= need * need


def _ground(track: Track, p) -> float:
    """Height of the ground under a roadside point.

    Height only. Rolling the props to match the camber as well was tried and
    is not worth what it costs: a batched prop is posed by heading alone, its
    roll axis after that rotation is not the one the slope was measured
    across, and the guardrail came out standing on edge. A rail is 1.25 m
    tall and its posts are placed individually -- each at its own height --
    so the run follows the bank without any of them being tilted at all.
    """
    return float(track.ground_y([p])[0])


def _outside_wall(track: Track, pts, clear: float) -> bool:
    """True if every one of *pts* is outside the fence, by *clear* metres.

    Two tests, because either one alone lets a building through. The corridor
    test says which side of the wall a point is on -- it is the same field the
    barrier contour was taken from, so it cannot disagree about that -- and the
    chord test says how far from the wall it is. A grandstand corner three
    centimetres outside the corridor passes the first and still has the fence
    running through its front wall.
    """
    q = np.atleast_2d(np.asarray(pts, dtype=float))
    if (track.corridor_distance(q) < clear).any():
        return False
    return bool((track.barrier_distance(q) >= clear).all())


def _footprint(p, u, out, w: float, d: float) -> np.ndarray:
    """The four corners of a *w* x *d* box centred on *p*, plus its edge
    midpoints -- enough samples that a chord cannot thread between them."""
    hu, ho = u * (w * 0.5), out * (d * 0.5)
    pts = []
    for a in (-1.0, 0.0, 1.0):
        for b in (-1.0, 0.0, 1.0):
            if a or b:
                pts.append(p + hu * a + ho * b)
    return np.asarray(pts, dtype=float)


def _true_runs(flags, least: int) -> list[tuple[int, int]]:
    """[start, stop) spans of consecutive True at least *least* long."""
    out, k, n = [], 0, len(flags)
    while k < n:
        if not flags[k]:
            k += 1
            continue
        j = k
        while j < n and flags[j]:
            j += 1
        if j - k >= least:
            out.append((k, j))
        k = j
    return out


def _runs(mask: np.ndarray) -> list[np.ndarray]:
    """Contiguous index runs of a circular boolean mask."""
    n = len(mask)
    if not mask.any():
        return []
    if mask.all():
        return [np.arange(n)]
    start = int(np.argmin(mask))                 # begin on a False, so no wrap
    order = (np.arange(n) + start) % n
    m = mask[order]
    out, k = [], 0
    while k < n:
        if not m[k]:
            k += 1
            continue
        j = k
        while j < n and m[j]:
            j += 1
        out.append(order[k:j])
        k = j
    return out


def _between(track: Track, s: float, s0: float, s1: float) -> bool:
    """Is arc-length *s* inside [s0, s1]? The interval may wrap the start line,
    which it does whenever anything is centred on it -- as the pits are."""
    L = track.length
    s0, s1, s = s0 % L, s1 % L, s % L
    return (s0 <= s <= s1) if s0 <= s1 else (s >= s0 or s <= s1)


def _arc_len(track: Track, run: np.ndarray) -> float:
    d = track.arclen[run[-1]] - track.arclen[run[0]]
    return float(d if d >= 0 else d + track.length)


def _corners(track: Track) -> list[tuple[np.ndarray, int]]:
    """(indices, outside_side) for each corner worth furnishing, slowest first.

    Ranked by how much of the lap they occupy and how tight they are, so a
    circuit gets its stands and its tyre walls at the places a driver actually
    thinks about rather than at every kink in the data.
    """
    mask = track.curv_radius < config.FURNITURE_CORNER_RADIUS
    sign = turn_sign(track)
    out = []
    for run in _runs(mask):
        length = _arc_len(track, run)
        if length < config.FURNITURE_CORNER_MIN_LEN:
            continue
        mid = run[len(run) // 2]
        radius = float(np.median(track.curv_radius[run]))
        out.append((run, int(sign[mid]) or 1, length / max(radius, 1.0)))
    out.sort(key=lambda r: -r[2])
    return [(run, side) for run, side, _ in out]


def _main_straight(track: Track) -> np.ndarray:
    """The run of straight containing the start line -- where the pits go."""
    straight = track.curv_radius > config.FURNITURE_CORNER_RADIUS * 2.5
    for run in _runs(straight):
        if 0 in run or (run[0] <= 0 <= run[-1]):
            return run
    # Start line inside a bend (some circuits do this): fall back to the
    # samples either side of it, so the pits still land on the start line.
    n = track.count
    half = max(8, n // 40)
    return np.array([(k) % n for k in range(-half, half)])


def _segments(track: Track, side: int):
    """Barrier chords for one side, with the centreline index of each midpoint.

    Reusing ``track.barrier_lines()`` rather than cutting a second set of runs
    means the barrier, the hoarding on top of it, the fence behind it and the
    stand behind that all follow one line -- and it is the same line the
    collision test reads.
    """
    segs = track.barrier_lines()[0 if side > 0 else 1]
    for a, b in segs:
        d = b - a
        length = float(np.hypot(d[0], d[1]))
        if length < 1e-6:
            continue
        mid = (a + b) / 2.0
        yield mid, d / length, length, _nearest_index(track, mid)


# --- build ---------------------------------------------------------------
def build_scenery(track: Track, shader=None) -> list[Entity]:
    """Build the roadside. *shader* is threaded down to every batch because it
    has to be set before the batch is flattened -- see PropLibrary.batch."""
    ents: list[Entity] = []
    lib = PropLibrary()
    lib.shader = shader
    rng = np.random.default_rng(7)

    corners = _corners(track)
    straight = _main_straight(track)
    fp = lib.footprint("grandStandCovered")
    spans = _stand_spans(track, fp[1] * config.GRANDSTAND_SCALE
                         + config.STAND_SETBACK)
    billed = _billed(track, corners, straight, spans)

    # Ground the buildings claim, as (x, z, radius). Filled in as they are
    # placed and honoured by everything scattered afterwards -- see _blocked.
    blockers: list[tuple] = []

    if getattr(config, "SPECTATOR_BANK_ENABLED", True):
        _spectator_bank(track, ents, spans)
    _barriers(track, lib, ents)
    _hoardings(track, lib, ents, billed)
    _tyre_walls(track, lib, ents, corners)
    _stands(track, lib, ents, spans, blockers)
    _pit_complex(track, lib, ents, straight, blockers)
    _start_gantry(track, lib, ents, blockers)
    _bridges(track, lib, ents, blockers)
    _distance_boards(track, lib, ents, corners)
    _posts_and_lights(track, lib, ents, spans, blockers)
    _cones(track, lib, ents)
    _forest(track, lib, ents, rng, blockers)

    # Templates are detached NodePaths; drop them now the batches hold copies.
    lib.dispose()
    return ents


# --- the barrier line ----------------------------------------------------
def _barriers(track: Track, lib: PropLibrary, ents: list[Entity]):
    """Armco on posts, with the debris fence standing behind it. Both sides,
    the whole lap.

    Replaces the Kenney barrier module, which was a solid coloured wall. A
    real circuit's boundary on a straight is a steel rail you can see over
    with a tall wire fence behind it, and that -- not a painted parapet -- is
    what makes the edge of the circuit read as the edge of a circuit.

    The rail and the fence rails are swept sections, so a run of any length is
    one stretched copy; the posts of both are placed at their own pitch,
    because a post stretched along its length is a wall.
    """
    rail_w = lib.footprint("guardrail")[0] or 4.0
    fence_w = lib.footprint("debris_fence")[0] or 4.0
    rails, posts, fences, fposts = [], [], [], []

    for side in (+1, -1):
        for mid, u, length, i in _segments(track, side):
            yaw = float(np.degrees(np.arctan2(u[0], u[1]))) + 90.0
            out = _outward(track, i, side)
            # Rail on the wall line; the fence a metre behind it, as on a real
            # circuit -- the fence is not what a car hits.
            p = mid
            q = mid + out * config.FENCE_SETBACK
            # Same problem as the hoardings, and it shows as the fence's
            # kinked top leaning out over the countryside on one side of the
            # circuit and over the track on the other.
            fyaw = yaw
            if np.dot(np.array([-u[1], u[0]]), out) < 0:
                fyaw += 180.0
            # On the surface and leaning with it. Where the run-off is
            # narrower than the camber's fade -- which is most of a straight
            # on a circuit with any camber at all -- the ground the rail
            # stands on is not the plane, and a rail bolted to the plane there
            # is a rail with daylight under one end and its foot buried at the
            # other.
            rails.append((p[0], p[1], fyaw, (length / rail_w, 1.0, 1.0)))
            fences.append((q[0], q[1], fyaw, (length / fence_w, 1.0, 1.0)))
            for pitch, pts, base in ((config.GUARDRAIL_POST_PITCH, posts, p),
                                     (config.FENCE_POST_PITCH, fposts, q)):
                n = max(2, int(round(length / pitch)))
                for t in np.linspace(-0.5, 0.5, n):
                    r = base + u * (t * length)
                    pts.append((r[0], r[1], fyaw, 1.0))

    for name, places in (("guardrail", rails), ("guardrail_post", posts),
                         ("debris_fence", fences), ("debris_post", fposts)):
        # Every height in one call rather than one call per post: the same
        # numbers, and a few thousand fewer trips through the nearest-sample
        # search -- two and a half seconds of loading on its own.
        if places:
            ys = track.ground_y(np.array([(pl[0], pl[1]) for pl in places]))
            places = [(*pl, float(y)) for pl, y in zip(places, ys)]
        e = lib.batch(name, places)
        if e is not None:
            ents.append(e)


def _hoardings(track: Track, lib: PropLibrary, ents: list[Entity], billed):
    """Advertising panels on the guardrail, in the built-up stretches only.

    Not round the whole lap, though the real thing gets close to it: a
    continuous panel on both sides for six kilometres is an opaque band that
    hides the treeline behind it, and every stretch of circuit then looks like
    every other.
    """
    names = ["hoarding_a", "hoarding_b", "hoarding_c", "hoarding_d"]
    module = lib.footprint(names[0])[0] or 4.0
    buckets: dict[str, list] = {n: [] for n in names}
    k = 0
    for side in (+1, -1):
        for mid, u, length, i in _segments(track, side):
            if length < config.HOARDING_MIN_RUN or (side, int(i)) not in billed:
                continue
            p = mid + _outward(track, i, side) * 0.15
            # +90 puts the panel's long axis along the run, but which way its
            # painted face then looks depends on the chord's direction, not on
            # which side of the circuit it is. Half of them ended up showing
            # the track their blank back.
            yaw = float(np.degrees(np.arctan2(u[0], u[1]))) + 90.0
            if np.dot(np.array([-u[1], u[0]]), _outward(track, i, side)) > 0:
                yaw += 180.0
            buckets[names[k % len(names)]].append(
                (p[0], p[1], yaw, (length / module, 1.0, 1.0)))
            k += 1
    for name, places in buckets.items():
        e = lib.batch(name, places)
        if e is not None:
            ents.append(e)


def _tyre_walls(track: Track, lib: PropLibrary, ents: list[Entity], corners):
    """Tyre walls on the outside of corners -- where a car that lets go goes."""
    module = lib.footprint("tyre_wall")[0] or 2.0
    want = {+1: set(), -1: set()}
    for run, side in corners[:config.TYRE_WALL_CORNERS]:
        want[side].update(int(k) for k in run)

    places = []
    for side in (+1, -1):
        if not want[side]:
            continue
        for mid, u, length, i in _segments(track, side):
            if i not in want[side] or length < config.TYRE_WALL_MIN_RUN:
                continue
            # Hard against the rail. A metre inboard it read as a separate
            # object the barrier passes through, which is the opposite of what
            # a tyre wall is -- it is the face of the barrier.
            p = mid - _outward(track, i, side) * 0.42
            yaw = float(np.degrees(np.arctan2(u[0], u[1]))) + 90.0
            places.append((p[0], p[1], yaw, (length / module, 1.0, 1.0)))
    e = lib.batch("tyre_wall", places)
    if e is not None:
        ents.append(e)


# --- crowds --------------------------------------------------------------
def _straights(track: Track) -> list[np.ndarray]:
    """Index runs where the circuit is straight enough to build along."""
    mask = track.curv_radius > config.FURNITURE_CORNER_RADIUS * 2.0
    return [r for r in _runs(mask)
            if _arc_len(track, r) >= config.STAND_MIN_RUN]


def _stand_spans(track: Track, depth: float):
    """(side, indices) short stretches of straight that carry a grandstand.

    Straights only. A stand is a straight building and a corner is not, so a
    row of them round a bend either steps in facets or leaves a wedge between
    every pair -- and a crowd on a straight can see further anyway. Runs are
    kept short and sparse on purpose: the treeline is doing most of the work of
    filling this circuit, and stands are the punctuation.
    """
    n = track.count
    off_r, off_l = track.wall_ray_offsets()
    spans = []
    for si, run in enumerate(_straights(track)):
        # Alternate sides down the lap, and take a slice of each straight
        # rather than all of it.
        for side in (_outer_side(track),):
            if si % 2:
                continue                     # only every other straight
            m = len(run)
            a = int(m * 0.12)
            b = int(m * min(0.95, 0.12 + config.STAND_STRAIGHT_FRACTION))
            piece = run[a:b]
            if len(piece) < 3 or _arc_len(track, piece) < config.STAND_MIN_RUN:
                continue
            off = off_r if side > 0 else off_l
            nrm = track.normal[piece] * side
            front = track.center[piece] + nrm * (off[piece]
                                                 + config.STAND_SETBACK)[:, None]
            back = track.center[piece] + nrm * (off[piece] + depth)[:, None]
            ok = ((_dist_to_track(track, front) >= config.RUNOFF_WIDTH + 3.0)
                  & (_dist_to_track(track, back) >= config.RUNOFF_WIDTH + 8.0))
            keep = np.zeros(n, dtype=bool)
            keep[piece[ok]] = True
            for sub in _runs(keep):
                if _arc_len(track, sub) >= config.STAND_MIN_RUN:
                    spans.append((side, sub))
    return spans


def _stands(track: Track, lib: PropLibrary, ents: list[Entity], spans,
            blockers):
    """Kenney grandstand modules, butted along one straight line per span.

    Three rules, and each is there because breaking it looked wrong:

    * **One heading for the whole run.** Aiming each module at the nearest
      centreline point gave every one a slightly different angle, and a row of
      buildings a degree apart from each other reads as a row that has been
      knocked askew.
    * **Stepped by exactly one width.** Anything else leaves daylight between
      neighbours or overlaps them.
    * **Never fewer than two.** A single stand on its own in a field is not a
      grandstand, it is a shed.
    """
    name = "grandStandCovered"
    gs = config.GRANDSTAND_SCALE
    w, d = (v * gs for v in lib.footprint(name))
    if w <= 0.0:
        return
    off_r, off_l = track.wall_ray_offsets()
    places = []
    for side, run in spans:
        off = off_r if side > 0 else off_l
        i0, i1 = int(run[0]), int(run[-1])
        mid = int(run[len(run) // 2])
        out = _outward(track, mid, side)
        reach = config.STAND_SETBACK + d * 0.5
        a0 = track.center[i0] + _outward(track, i0, side) * (off[i0] + reach)
        a1 = track.center[i1] + _outward(track, i1, side) * (off[i1] + reach)
        span_v = a1 - a0
        length = float(np.hypot(span_v[0], span_v[1]))
        count = int(length / w)
        if count < 2:
            continue
        u = span_v / length
        yaw = yaw_towards(-out)          # one heading for every module
        # Squared to the module's own heading, not to the span: every module
        # in a row is aimed at -out, so that -- not the line the row happens
        # to run along -- is where its walls are.
        across = np.array([-out[1], out[0]], dtype=float)
        ok = []
        for k in range(count):
            p = a0 + u * ((k + 0.5) * w)
            # A span is one straight line and the wall beside it is not, so
            # "set back from the barrier at both ends" does not mean every
            # module in between clears it -- through a kink in the fence the
            # middle of the row walked straight into the wall. Tested per
            # module, against the fence's own chords, over the whole
            # footprint rather than the centre point.
            ok.append(_clear(track, p, config.RUNOFF_WIDTH + 4.0)
                      and _outside_wall(track,
                                        _footprint(p, across, out, w, d),
                                        config.STAND_WALL_CLEAR))
        # ...and then only the runs of at least two. Dropping a module the
        # fence went through was right; leaving its neighbour standing on its
        # own in a field was not, and that is what the rule at the top of this
        # docstring is for. A survivor with nothing butted against it is a
        # shed, so it goes too.
        for a, b in _true_runs(ok, 2):
            for k in range(a, b):
                p = a0 + u * ((k + 0.5) * w)
                places.append((p[0], p[1], yaw, gs))
                blockers.append((p[0], p[1], max(w, d) * 0.62))
                # ...and the ground *in front* of it, all the way to the
                # barrier. Blocking only the footprint left the treeline free
                # to grow between the stand and the circuit, which is the one
                # place a grandstand cannot have a tree.
                for f in np.linspace(0.15, 1.0, 5):
                    q = p - out * (config.STAND_SETBACK + d * 0.5) * f
                    blockers.append((q[0], q[1], w * 0.55))
    # The seating shader draws the rows of seats (shaders.py, CROWD). Given
    # to the batch, not afterwards: flattening bakes whatever shader is in
    # force.
    from .shaders import crowd_shader
    e = lib.batch(name, places, face_forward=True, shader=crowd_shader)
    if e is not None:
        e.world_shader = crowd_shader
        ents.append(e)


def _billed(track: Track, corners, straight, spans) -> set:
    """(side, index) pairs that carry advertising: the built-up stretches."""
    n = track.count
    pad = max(4, int(n * 30.0 / max(track.length, 1.0)))
    out = set()

    def add(side, idx):
        for k in idx:
            for d in range(-pad, pad + 1):
                out.add((side, int((k + d) % n)))

    for side, run in spans:
        add(side, run)
    for side in (+1, -1):
        add(side, straight)
    for run, side in corners[:config.TYRE_WALL_CORNERS]:
        add(side, run)
    return out


# --- the forest ----------------------------------------------------------
def _forest(track: Track, lib: PropLibrary, ents: list[Entity], rng, blockers):
    """Fill everything outside the circuit with low-poly trees.

    This is what closes the world. Props scattered on an empty plane are
    objects on emptiness; a treeline dense enough to have no holes in it *is*
    the horizon, and it is the one thing that makes a circuit feel enclosed
    from every point on the lap.

    Two bands with different densities: a close, tight one that reads as a
    wall of green just behind the barrier, and a sparser one behind it for
    depth. Candidates are generated on a jittered grid along the lap -- a
    uniform random scatter clumps and leaves holes, which on a treeline shows
    as gaps you can see the sky through.
    """
    names = [f"{shape}_{tone}"
             for shape in ("tree_round", "tree_pine", "tree_bush",
                           "tree_spread", "tree_cypress")
             for tone in ("a", "b", "c")]
    # Only to place the candidates -- how far out from this leg to start
    # throwing them. Whether one may actually stand there is a separate
    # question, answered against the region below.
    off_r, off_l = track.wall_ray_offsets()
    picks: dict[str, list] = {n: [] for n in names}
    for lo, hi, pitch in config.FOREST_BANDS:
        step = max(1, int(track.count * pitch / max(track.length, 1.0)))
        rows = np.arange(0, track.count, step)
        per_row = max(1, int((hi - lo) / pitch))
        span = float(track.length) * step / max(track.count, 1)
        for side in (+1, -1):
            off = off_r if side > 0 else off_l
            base = np.repeat(track.center[rows], per_row, axis=0)
            nrm = np.repeat(track.normal[rows] * side, per_row, axis=0)
            tan = np.repeat(track.tangent[rows], per_row, axis=0)
            near = np.repeat(off[rows], per_row)
            m = len(base)
            # Distance and along-track offset both drawn per tree over the
            # whole band, rather than a tree per lane per row. Lanes are what
            # made the treeline stand in ranks: every tree in a lane sat the
            # same distance out, so the forest had a ruled edge and rows you
            # could count. The square root biases towards the near edge, which
            # is what a wood does where it meets a clearing.
            d = near + lo + (hi - lo) * np.sqrt(rng.random(m))
            jit = (rng.random(m) - 0.5) * span * 1.6
            pts = base + nrm * d[:, None] + tan * jit[:, None]
            # Tested against the drivable region itself, not against the
            # nearest sample's run-off radius. The region is the union of
            # every sample's radius, so a point can be outside the disc of the
            # sample nearest to it and still be well inside the fence -- which
            # is the whole of the ground a corner's outline fans over, and
            # exactly where the trees standing inside the barrier stood. The
            # barrier is drawn BARRIER_OUTSET outside this contour, so that
            # comes off the clearance too.
            good = (track.corridor_distance(pts)
                    >= config.FOREST_CLEAR + config.BARRIER_OUTSET)                 & ~_blocked(pts, blockers, pad=config.FOREST_PROP_CLEAR)
            pts = pts[good]
            if not len(pts):
                continue
            pick = rng.integers(0, len(names), len(pts))
            scale = rng.uniform(*config.FOREST_SCALE, len(pts))
            yaw = rng.uniform(0.0, 360.0, len(pts))
            for p, k, sc, y in zip(pts, pick, scale, yaw):
                picks[names[k]].append((p[0], p[1], y, sc))

    total = sum(len(v) for v in picks.values())
    if getattr(config, "FOREST_CARDS", True):
        from . import foliage
        made = foliage.place(track, picks, rng)
        ents.extend(made)
        print(f"scenery: {total} trees ({'impostors' if foliage.impostors() else 'painted cards'}) in {len(made)} cells")
        return
    cell = config.FOREST_CELL
    if cell <= 0:
        for name, places in picks.items():
            e = lib.batch(name, places)
            if e is not None:
                ents.append(e)
        print(f"scenery: {total} trees in {len(names)} batches")
        return

    # Grouped by neighbourhood, not by species. Fifteen batches spanning the
    # whole circuit have bounding volumes that contain the camera wherever it
    # stands, so none of them is ever culled and every tree on the lap is
    # submitted every frame. Cells give each node tight bounds.
    cells: dict[tuple, dict] = {}
    for name, places in picks.items():
        for p in places:
            key = (int(p[0] // cell), int(p[1] // cell))
            cells.setdefault(key, {}).setdefault(name, []).append(p)
    made = 0
    for groups in cells.values():
        e = lib.batch_many(groups)
        if e is not None:
            ents.append(e)
            made += 1
    print(f"scenery: {total} trees in {made} cells of {cell:.0f} m")

def _blocked(pts: np.ndarray, blockers, pad: float = 0.0) -> np.ndarray:
    """True for each point that falls inside some prop's keep-out disc.

    Scattered things -- trees, light posts -- are placed from the circuit's
    geometry and know nothing about the buildings placed from it earlier, so
    without this a treeline grows through a grandstand and a lamp post stands
    in a garage doorway. Every placement that occupies ground registers a disc
    here and everything scattered afterwards tests against them.
    """
    if not len(blockers) or not len(pts):
        return np.zeros(len(pts), dtype=bool)
    b = np.asarray(blockers, dtype=float)
    out = np.zeros(len(pts), dtype=bool)
    for a0 in range(0, len(pts), 512):
        a1 = min(a0 + 512, len(pts))
        d = np.linalg.norm(pts[a0:a1, None, :2] - b[None, :, :2], axis=2)
        out[a0:a1] = (d < (b[None, :, 2] + pad)).any(axis=1)
    return out


def _dist_to_track(track: Track, pts: np.ndarray, want_index=False):
    """Distance from each point to the nearest point of the lap, in blocks.

    With *want_index*, also which sample that was -- which is what lets a
    caller ask how wide the circuit's corridor is *there* rather than where
    the point came from.
    """
    out = np.empty(len(pts))
    idx = np.empty(len(pts), dtype=np.int64)
    for a in range(0, len(pts), 512):
        b = min(a + 512, len(pts))
        d = ((pts[a:b, None, :] - track.center[None, :, :]) ** 2).sum(axis=2)
        k = d.argmin(axis=1)
        idx[a:b] = k
        out[a:b] = np.sqrt(d[np.arange(b - a), k])
    return (out, idx) if want_index else out


def _stand_cover(track: Track, spans, side: int) -> np.ndarray:
    """Per-sample 0..1 saying how much of that spot a grandstand already fills."""
    n, L = track.count, track.length
    cover = np.zeros(n)
    for s, run in spans:
        if s == side:
            cover[run] = 1.0
    # Dilate before smoothing. Smoothing alone leaves cover at about a half at
    # the very end of a span -- which is exactly where the stand's end wall is,
    # so the bank came up through it. Growing the mask by the stand's own depth
    # first means the taper starts *past* the building and the bank is flat
    # under every part of it.
    grow = max(2, int(n * 30.0 / max(L, 1.0)))
    if cover.any():
        wide = cover.copy()
        for d in range(1, grow + 1):
            wide = np.maximum(wide, np.maximum(np.roll(cover, d),
                                               np.roll(cover, -d)))
        cover = wide
    k = max(3, int(n * 55.0 / max(L, 1.0)))
    pad = np.concatenate([cover[-k:], cover, cover[:k]])
    box = np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    return np.clip(box[k:k + n] * 1.6, 0.0, 1.0)


def _spectator_bank(track: Track, ents: list[Entity], spans):
    """A grassed embankment running the whole lap outside the barrier.

    This is the answer to "the ground beside the track is an empty plane". A
    real circuit is rarely flat to the horizon: where there is no grandstand
    there is a bank people stand on, and the bank is what closes the view.
    Flat ground gives the eye nothing between the barrier and the hills, and no
    amount of extra props fixes that -- props are objects *on* the emptiness.

    Height is scaled down by two things: a grandstand in front of it, and
    proximity to any other part of the lap. The second is not cosmetic. The
    bank reaches forty-odd metres out, which at a chicane crosses the next leg
    of the circuit -- without the clearance term the embankment comes up
    through the track surface.
    """
    if not getattr(config, "SPECTATOR_BANK_ENABLED", True):
        return

    # The bank starts just outside the fence, so it has to be measured to the
    # fence that is drawn. Against the requested radius its inner edge landed
    # *inside* the barrier wherever the wall bulged, and a strip of the grass
    # plane came up through the run-off on the wrong side of the wall.
    off_r, off_l = track.wall_ray_offsets()
    n, step = track.count, 2
    profile = [(2.0, 0.0), (9.0, 1.9), (18.0, 3.4), (28.0, 4.2),
               (36.0, 4.3), (46.0, 0.15)]
    rng = np.random.default_rng(23)

    for side in (+1, -1):
        off = off_r if side > 0 else off_l
        keep = 1.0 - _stand_cover(track, spans, side)
        idx = list(range(0, n, step)) + [0]
        rows, m = len(idx), len(profile)

        # Every vertex position first, so the clearance test is one blocked
        # distance query instead of thousands of separate ones.
        pts = np.empty((rows * m, 2))
        for r, k in enumerate(idx):
            c, nrm = track.center[k], track.normal[k] * side
            for b, (d, _h) in enumerate(profile):
                pts[r * m + b] = c + nrm * (off[k] + d)
        room = np.clip(
            (_dist_to_track(track, pts) - (config.RUNOFF_WIDTH + 7.0)) / 14.0,
            0.0, 1.0)

        verts, tris, cols, uvs, norms = [], [], [], [], []
        for r, k in enumerate(idx):
            nrm = track.normal[k] * side
            for b, (d, h) in enumerate(profile):
                p = pts[r * m + b]
                y = h * keep[k] * room[r * m + b]
                prev_y = verts[-1][1] if b else 0.0
                verts.append((p[0], y, p[1]))
                uvs.append((float(track.arclen[k]) / 16.0, (off[k] + d) / 16.0))
                slope = 0.0 if b == 0 else (y - prev_y) / max(d - profile[b - 1][0],
                                                              1e-3)
                v = np.array([-slope * nrm[0], 1.0, -slope * nrm[1]])
                v /= np.linalg.norm(v)
                norms.append((v[0], v[1], v[2]))
                shade = ((0.84 + 0.22 * (y / 4.3)) * mottle(p[0], p[1])
                         + (rng.random() - 0.5) * 0.05) * 1.27
                cols.append(color.rgba(pal.GRASS.r * shade,
                                       pal.GRASS.g * shade,
                                       pal.GRASS.b * shade, 1.0))
        for r in range(rows - 1):
            a0, b0 = r * m, (r + 1) * m
            for b in range(m - 1):
                tris += [a0 + b, b0 + b, b0 + b + 1, a0 + b, b0 + b + 1, a0 + b + 1]
        ents.append(Entity(
            parent=scene, texture=textures.ground(), double_sided=True,
            model=Mesh(vertices=verts, triangles=tris, colors=cols, uvs=uvs,
                       normals=norms, mode="triangle", static=True)))


def _pit_complex(track: Track, lib: PropLibrary, ents: list[Entity], straight,
                 blockers):
    """Garages, pit wall and race control along the main straight.

    A circuit's one piece of real architecture. It goes on the start/finish
    straight because that is the only place it ever is, and it is what tells
    you at a glance which part of the lap you are on.
    """
    side = config.PIT_SIDE
    n = track.count
    garage_w = lib.footprint("pit_garage")[0] or 12.6
    # The garage's canopy reaches 8.3 m in front of its origin, so the origin
    # has to stand that far back for the canopy edge to clear the barrier.
    depth = lib.footprint("pit_garage")[1]
    front = depth * 0.62

    # A fixed number of bays centred just before the line, not "as many as the
    # straight will hold": Monza's is 955 m long and filling it gave a
    # kilometre of continuous garage, which is three times the real thing and
    # leaves no open country on the one straight that has any.
    garage_w = max(garage_w, 1.0)
    total = garage_w * config.PIT_GARAGES
    s_from = float(track.arclen[0]) - total * 0.65
    s_to = s_from + total

    # One curve for the whole block, not a bay at a time. Placing each bay
    # off the fence beside it gave every one its own distance out and its own
    # heading, and a row of buildings that each stand where the wall happens
    # to be is a row whose neighbours overlap where the wall comes in and show
    # daylight where it goes out. A pit lane is one building.
    #
    # But not one straight line either: two hundred and fifty metres of it
    # leaves the straight it started on, and at Zandvoort the far end came
    # back over the circuit. The bays run along the centreline offset by a
    # single distance -- so the block keeps the road's own gentle curve -- and
    # are stepped along *that* curve rather than along the centreline, which
    # is what makes them butt. Step by the centreline and the outside of a
    # bend opens gaps between them and the inside overlaps them, which is the
    # same bug one level down.
    # Confined to the main straight, and stepped along the *offset* curve.
    #
    # Three separate things had to be true and none of them was. Placing each
    # bay off the fence beside it gave every bay its own distance out and its
    # own heading, so neighbours overlapped where the wall came in and showed
    # daylight where it went out -- a pit lane is one building. Running the
    # block on a single straight line instead fixed that and broke something
    # worse: two hundred and fifty metres of straight line leaves the straight
    # it started on, and at Zandvoort the far end came back over the circuit.
    # And letting the block run past the end of the straight put its first
    # bays beside the previous corner, where the run-off is forty metres wide
    # -- so the setback taken from the widest point pushed the whole lane out
    # there, and the offset curve round the inside of a 56 m corner offset by
    # 44 m folds nearly to a point, which stacked two garages on one spot.
    #
    # So: the centreline of the straight, offset by one distance, resampled at
    # exactly one garage width along that curve. Stepping along the centreline
    # instead is the same bug one level down -- the outside of a bend opens
    # gaps between the bays and the inside overlaps them.
    run = np.asarray(straight, dtype=int)
    garages, walls, towers = [], [], []
    if len(run) < 3:
        return
    nrm = track.normal[run] * side
    need = np.array([float(np.dot(_beyond_wall(track, int(i), side,
                                               config.PIT_SETBACK + front)
                                  - track.center[int(i)],
                                  _outward(track, int(i), side)))
                     for i in run])
    # Where the start line falls along the straight: the pits end just past it.
    lap = np.abs(track.arclen[run] - track.arclen[0])
    k0 = int(np.argmin(np.minimum(lap, track.length - lap)))
    # Set back by the widest the fence gets *along the bays*, not along the
    # whole straight. A straight run reaches into the corner at each end,
    # where the run-off opens out to forty metres, and taking the maximum over
    # all of it stood the pit lane out in that -- forty metres of empty ground
    # between the garages and the barrier, which is not a pit lane, it is a
    # car park.
    d0 = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(track.center[run], axis=0), axis=1))])
    win = (d0 >= d0[k0] - total * 0.65) & (d0 <= d0[k0] + total * 0.35)
    reach = float(need[win].max() if win.any() else need.max())
    # Try a few setbacks and keep the one that seats the most bays. Pushing
    # out until every bay clears sounds right and is not: where the fence
    # bulges at one end of the block, "push until it fits" walks the whole
    # lane into the middle of a field to save two garages. Scored instead, so
    # a short pit lane close to the circuit beats a long one nowhere near it,
    # and the smallest setback that ties wins.
    best = None
    for attempt in range(6):
        line = track.center[run] + nrm * (reach + attempt * 2.0)
        cum = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(line, axis=0), axis=1))])
        start = float(np.clip(cum[k0] - total * 0.65, 0.0,
                              max(cum[-1] - garage_w, 0.0)))
        want = np.arange(start + garage_w * 0.5,
                         min(start + total, cum[-1]), garage_w)
        want = want[:config.PIT_GARAGES]
        if len(want) < 2:
            break
        # The outward direction at each bay, interpolated along the same
        # curve, so a bay's heading matches the piece of road it faces.
        outs = np.stack([np.interp(want, cum, nrm[:, 0]),
                         np.interp(want, cum, nrm[:, 1])], axis=1)
        outs /= np.maximum(np.linalg.norm(outs, axis=1), 1e-9)[:, None]
        qs = [np.array([x, z]) for x, z in
              zip(np.interp(want, cum, line[:, 0]),
                  np.interp(want, cum, line[:, 1]))]
        ok = [_outside_wall(track,
                            _footprint(q, np.array([-o[1], o[0]]), o,
                                       garage_w, depth),
                            config.PIT_WALL_CLEAR)
              for q, o in zip(qs, outs)]
        runs = _true_runs(ok, 2)
        if not runs:
            continue
        lo, hi = max(runs, key=lambda r: r[1] - r[0])
        if best is None or hi - lo > best[0]:
            best = (hi - lo, qs[lo:hi], [yaw_towards(-o) for o in outs[lo:hi]])
        if all(ok):
            break
    if best is not None:
        pts, yaws = best[1], best[2]
    else:
        pts, yaws = [], []
    for q, yw in zip(pts, yaws):
        garages.append((q[0], q[1], yw))
        blockers.append((q[0], q[1], max(garage_w, depth) * 0.66))
    if pts:
        s_from = float(track.arclen[_nearest_index(track, pts[0])]) - garage_w
        s_to = float(track.arclen[_nearest_index(track, pts[-1])]) + garage_w

    # Pit wall: stretched runs along the same stretch, just outside the barrier.
    module = lib.footprint("pit_wall")[0] or 6.0
    for mid, u, length, i in _segments(track, side):
        if not _between(track, float(track.arclen[i]), s_from, s_to):
            continue
        if length < 6.0:
            continue
        p = mid + _outward(track, i, side) * 1.6
        yaw = float(np.degrees(np.arctan2(u[0], u[1]))) + 90.0
        walls.append((p[0], p[1], yaw, (length / module, 1.0, 1.0)))

    i0 = int(straight[len(straight) // 3])
    q = _beyond_wall(track, i0, side, config.PIT_SETBACK + 26.0)
    towers.append((q[0], q[1], yaw_towards(track.center[i0] - q)))
    blockers.append((q[0], q[1], 16.0))

    for name, places in (("pit_garage", garages), ("pit_wall", walls),
                         ("control_tower", towers)):
        e = lib.batch(name, places)
        if e is not None:
            ents.append(e)


def _start_gantry(track: Track, lib: PropLibrary, ents: list[Entity], blockers):
    """The gantry over the start line, stretched to span the circuit.

    Authored 18.65 m wide and scaled to land its legs just outside the barrier
    on both sides -- derived rather than fixed, because the circuits here range
    from 12 to 15 m of asphalt and a gantry with its feet on the racing line is
    worse than no gantry at all.
    """
    from .structures import GANTRY_W, gantry_place
    _i, p, span = gantry_place(track)
    # The physics' legs (structures.obstacles) assume this width.
    authored = GANTRY_W
    scale = (span / authored, 1.0, 1.0)
    # Turned to meet the cars. yaw_at is the direction of travel, and a gantry
    # aimed that way shows the grid the back of its banner and the backs of
    # its five lights -- which is the one piece of the circuit that has to be
    # read from the car.
    e = lib.batch("gantry", [(p[0], p[1], yaw_at(track, 0) + 180.0, scale)])
    if e is not None:
        ents.append(e)
    # The dark lamp housings are the baked prop, stamped once at the gantry's
    # transform and left on for the whole session.
    yaw = yaw_at(track, 0) + 180.0
    lamp = lib.batch("gantry_lamps_off", [(p[0], p[1], yaw, scale)])
    if lamp is not None:
        lamp.name = "gantry_lamps_off"
        ents.append(lamp)
    # The *lit* lamps are five independent columns, not one baked "all on"
    # mesh: the gantry counts the start in one lamp at a time, exactly as the
    # HUD does, and a single mesh cannot have four fifths of itself switched
    # off. Each column carries the same two stacked faces as the baked prop
    # (blender/circuit_kit._lamp_geometry), in the gantry's own coordinates,
    # and is named so app.py can find it in the flat scenery list.
    rig = Entity(parent=scene, name="gantry_lit",
                 position=(p[0], 0.0, p[1]), rotation_y=yaw,
                 scale=(scale[0], 1.0, 1.0))
    ents.append(rig)
    top, ht, pitch = 7.4 - 0.26, 1.30, 0.92
    # Left to right for a car facing the gantry: column 0 is the first to
    # light. The rig carries the gantry's +180 yaw, so the leftmost lamp sits
    # at +x in the rig's own frame -- hence (2 - k), not (k - 2).
    #
    # The lit face is the same colour as the HUD's start lamps (ui.LAMP_ON)
    # and is drawn unlit: a sunset-shaded red box just looks like red paint,
    # and two of those stacked on the baked dark face at the same depth was
    # the white shimmer -- z-fighting, not a lamp. Sat a few centimetres
    # proud of the housing it reads as a light that is actually on.
    for k in range(5):
        col = Entity(parent=rig, name=f"gantry_lit_{k}", enabled=False,
                     position=((2 - k) * pitch, 0.0, 1.03))
        for z0 in (top - ht + 0.12, top - ht * 0.52):
            Entity(parent=col, model="cube", unlit=True,
                   color=pal.rgb(255, 28, 18),
                   position=(0.0, z0 + ht * 0.17, 0.0),
                   scale=(pitch * 0.56, ht * 0.36, 0.06))
        ents.append(col)

    blockers.append((p[0], p[1], span * 0.6))
    flags = []
    for side in (+1, -1):
        q = _beyond_wall(track, 0, side, 1.5)
        flags.append((q[0], q[1], yaw_at(track, 0)))
    e = lib.batch("flagCheckers", flags)
    if e is not None:
        ents.append(e)


def _bridges(track: Track, lib: PropLibrary, ents: list[Entity], blockers):
    """Spectator bridges. Two of these round a lap break the skyline more than
    any amount of extra grandstand does, and they give a long straight a
    landmark to measure distance against."""
    from .structures import BRIDGE_W, bridge_places
    # The physics' towers (structures.obstacles) assume this width.
    authored = BRIDGE_W
    places = []
    for i, p, span in bridge_places(track):
        places.append((p[0], p[1], yaw_at(track, i), (span / authored, 1.0, 1.0)))
        blockers.append((p[0], p[1], span * 0.6))
    e = lib.batch("bridge", places)
    if e is not None:
        ents.append(e)


# --- signage -------------------------------------------------------------
def _marshal_posts(track: Track, lib: PropLibrary, ents: list[Entity], blockers):
    """Marshal stations at a regular pitch, alternating sides."""
    places = []
    for k, i in enumerate(_sample_every(track, config.MARSHAL_SPACING)):
        side = 1 if k % 2 == 0 else -1
        p = _beyond_wall(track, int(i), side, 2.6)
        places.append((p[0], p[1], yaw_towards(track.center[i] - p)))
        blockers.append((p[0], p[1], 5.0))
    e = lib.batch("marshal_post", places)
    if e is not None:
        ents.append(e)


def _distance_boards(track: Track, lib: PropLibrary, ents: list[Entity], corners):
    """3-2-1 boards counting down into the braking zone of the real corners.

    Bars rather than numerals -- there are no textures in this game -- but the
    job is the same: something to brake against. They stand on the outside of
    the corner, which is the side a driver is looking at on the way in, and
    face back down the track at the oncoming car.
    """
    buckets = {"board_150": [], "board_100": [], "board_50": []}
    for run, side in corners[:config.BOARD_CORNERS]:
        entry = float(track.arclen[run[0]])
        for name, back in (("board_150", 150.0), ("board_100", 100.0),
                           ("board_50", 50.0)):
            s = (entry - back) % track.length
            i = int(np.searchsorted(track.arclen, s)) % track.count
            # `side` is the *outside* of the bend -- left board for a
            # right-hander, right board for a left-hander, which is the side a
            # driver is already looking at on the way in.
            # Square across the run-off, facing back down the circuit, not
            # flat on the fence. A panel lying along the barrier is edge-on to
            # the car until the instant it is level with it, which is a board
            # you cannot read; turned to meet the driver it is legible for the
            # whole approach, which is the only thing it is for.
            #
            # Its width then runs inward from the fence rather than along it,
            # so it is stood off by its own half-width -- placed on the chord
            # it would hang half outside the circuit.
            seg = _nearest_segment(track, side, track.center[i])
            if seg is None:
                continue
            q, m = seg
            half = (lib.footprint(name)[0] or 2.3) * 0.5
            p = q - m * (half + config.BOARD_STANDOFF)
            buckets[name].append((p[0], p[1],
                                  yaw_towards(-track.tangent[i])))
    for name, places in buckets.items():
        e = lib.batch(name, places)
        if e is not None:
            ents.append(e)


def _posts_and_lights(track: Track, lib: PropLibrary, ents: list[Entity],
                      spans, blockers):
    """Marker posts on the verge, light posts behind the barrier.

    The lights skip the stand spans and anything else already standing there,
    so a lamp never ends up inside a grandstand or in a garage doorway.
    """
    idx = _sample_every(track, config.MARKER_SPACING)
    # The red-and-white marker boxes that used to stand on the verge are gone.
    # They were a strong optical-flow cue, but they are not a thing a circuit
    # has, and dotted along both verges they read as litter.

    occupied = {(s, int(k)) for s, run in spans for k in run}
    lights = []
    for k, i in enumerate(idx):
        side = 1 if k % 2 == 0 else -1
        if (side, int(i)) in occupied:
            continue
        # Behind the barrier line, as on a real circuit -- inside it they stood
        # in the run-off where a car would hit them.
        # Off the barrier chord, not off the centreline normal: the wall is
        # the outline of the run-off, so a normal offset put posts metres
        # inside it, standing in the gravel.
        seg = _nearest_segment(track, side, track.center[i])
        if seg is None:
            continue
        gap = config.LIGHTPOST_WALL_GAP
        q = seg[0] + seg[1] * gap
        # Placed off the nearest chord on this side, then checked back against
        # every chord on *both* sides. Where the circuit doubles back the
        # nearest wall to a point a metre and a half behind one barrier can be
        # the other one, and the post that looked like it was hugging the
        # fence was standing in the next corner's run-off. Out of tolerance is
        # dropped, not nudged: a gap in a row of lamps reads as a gap, a lamp
        # in the gravel reads as a bug.
        if abs(float(track.barrier_distance(q)[0]) - gap) > config.LIGHTPOST_GAP_TOL:
            continue
        # ...and on the outside of it. The chord test alone is satisfied by a
        # point the same distance *inside* the wall.
        if float(track.corridor_distance(q)[0]) < gap * 0.5:
            continue
        # Aimed by direction, not by heading-plus-90: the lamp reaches along the
        # model's own z, and a signed offset rule got that backwards on one
        # side, so half the posts lit the countryside.
        if _blocked(np.array([[q[0], q[1]]]), blockers, pad=3.0)[0]:
            continue
        lights.append((q[0], q[1], yaw_towards(q - track.center[i]),
                       config.LIGHTPOST_SCALE))
        blockers.append((q[0], q[1], 3.5))
    e = lib.batch("lightPostModern", lights)
    if e is not None:
        ents.append(e)


def _cones(track: Track, lib: PropLibrary, ents: list[Entity]):
    """Cones on the apex side through corners, as a circuit marks its limits."""
    places = []
    corner = track.curv_radius < 220.0
    for i in _sample_every(track, 9.0):
        if not corner[i]:
            continue
        side = -1 if _turns_left(track, i) else 1      # inside of the corner
        lat = side * ((track.w_right[i] if side > 0 else track.w_left[i])
                      + config.KERB_WIDTH + 0.7)
        p = _edge(track, i, side, config.KERB_WIDTH + 0.7)
        # On the road's own surface, not on the plane. These sit a metre off
        # the kerb, which on a banked corner is a metre and a half above or
        # below where the plane is -- a row of cones floating over the inside
        # of the bend, or buried in it.
        places.append((p[0], p[1], 0.0, 1.0,
                       float(track.ground_y([p])[0])))
    e = lib.batch("pylon", places)
    if e is not None:
        ents.append(e)


# --- trees ---------------------------------------------------------------
def _turns_left(track: Track, i: int) -> bool:
    n = track.count
    a = track.tangent[i]
    b = track.tangent[(i + max(2, n // 200)) % n]
    return (a[0] * b[1] - a[1] * b[0]) > 0


def _emit(ents: list[Entity], b: MeshBuilder):
    m = b.build()
    if m is not None:
        ents.append(Entity(parent=scene, model=m))
