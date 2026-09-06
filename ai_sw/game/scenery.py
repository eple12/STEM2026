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


def _beyond_wall(track: Track, i: int, side: int, extra: float) -> np.ndarray:
    """A point *extra* metres outside the barrier line on the given side."""
    off_r, off_l = track.wall_offsets()
    off = (off_r if side > 0 else off_l)[i]
    return track.center[i] + track.normal[i] * side * (off + extra)


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
            rails.append((p[0], p[1], fyaw, (length / rail_w, 1.0, 1.0)))
            fences.append((q[0], q[1], fyaw, (length / fence_w, 1.0, 1.0)))
            for pitch, pts, base in ((config.GUARDRAIL_POST_PITCH, posts, p),
                                     (config.FENCE_POST_PITCH, fposts, q)):
                n = max(2, int(round(length / pitch)))
                for t in np.linspace(-0.5, 0.5, n):
                    r = base + u * (t * length)
                    pts.append((r[0], r[1], fyaw))

    for name, places in (("guardrail", rails), ("guardrail_post", posts),
                         ("debris_fence", fences), ("debris_post", fposts)):
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
    off_r, off_l = track.wall_offsets()
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
    off_r, off_l = track.wall_offsets()
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
        for k in range(count):
            p = a0 + u * ((k + 0.5) * w)
            if not _clear(track, p, config.RUNOFF_WIDTH + 4.0):
                continue
            places.append((p[0], p[1], yaw, gs))
            blockers.append((p[0], p[1], max(w, d) * 0.62))
            # ...and the ground *in front* of it, all the way to the barrier.
            # Blocking only the footprint left the treeline free to grow
            # between the stand and the circuit, which is the one place a
            # grandstand cannot have a tree.
            for f in np.linspace(0.15, 1.0, 5):
                q = p - out * (config.STAND_SETBACK + d * 0.5) * f
                blockers.append((q[0], q[1], w * 0.55))
    e = lib.batch(name, places, face_forward=True)
    if e is not None:
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
    off_r, off_l = track.wall_offsets()
    # Per-candidate, not one global maximum. Using the widest corridor
    # anywhere on the lap -- now fifty metres, at the widest corner run-off --
    # held the treeline that far back on every straight as well, which is the
    # gap between the fence and the trees.

    corridor = np.maximum(off_r, off_l)
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
            # Cleared against the corridor of whatever leg the tree ends up
            # nearest to, not the one it was generated from. Where the circuit
            # folds -- Shanghai's hairpins -- a tree thrown off a narrow leg
            # lands beside a wide one, and a threshold taken from its own leg
            # let it through onto the other one's track surface.
            dist, near_i = _dist_to_track(track, pts, want_index=True)
            room = np.maximum(near, corridor[near_i])
            good = (dist >= room + config.FOREST_CLEAR)                 & ~_blocked(pts, blockers, pad=config.FOREST_PROP_CLEAR)
            pts = pts[good]
            if not len(pts):
                continue
            pick = rng.integers(0, len(names), len(pts))
            scale = rng.uniform(*config.FOREST_SCALE, len(pts))
            yaw = rng.uniform(0.0, 360.0, len(pts))
            for p, k, sc, y in zip(pts, pick, scale, yaw):
                picks[names[k]].append((p[0], p[1], y, sc))

    total = sum(len(v) for v in picks.values())
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

    off_r, off_l = track.wall_offsets()
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

    garages, walls, towers = [], [], []
    for k in range(config.PIT_GARAGES):
        s = (s_from + (k + 0.5) * garage_w) % track.length
        i = int(np.searchsorted(track.arclen, s)) % n
        p = _beyond_wall(track, i, side, config.PIT_SETBACK + front)
        if not _clear(track, p, config.RUNOFF_WIDTH + 6.0):
            continue
        garages.append((p[0], p[1], yaw_towards(track.center[i] - p)))
        blockers.append((p[0], p[1], max(garage_w, depth) * 0.66))

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
    off_r, off_l = track.wall_offsets()
    span = float(off_r[0] + off_l[0]) + 3.0
    authored = lib.footprint("gantry")[0] or 18.65
    p = track.center[0] + track.normal[0] * float(off_r[0] - off_l[0]) / 2.0
    e = lib.batch("gantry", [(p[0], p[1], yaw_at(track, 0),
                              (span / authored, 1.0, 1.0))])
    if e is not None:
        ents.append(e)

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
    off_r, off_l = track.wall_offsets()
    authored = lib.footprint("bridge")[0] or 25.6
    places = []
    for f in np.linspace(0.0, 1.0, config.BRIDGE_COUNT, endpoint=False)[1:]:
        i = int(np.searchsorted(track.arclen, f * track.length)) % track.count
        span = float(off_r[i] + off_l[i]) + 8.0
        p = track.center[i] + track.normal[i] * float(off_r[i] - off_l[i]) / 2.0
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
            out = _outward(track, i, side)
            # Stood off the fence towards the track, not flush with it:
            # on the fence line the panel is inside the rails.
            p = _beyond_wall(track, i, side,
                             config.FENCE_SETBACK - config.BOARD_STANDOFF)
            # Angled back down the circuit, not flat against the fence.
            # Square to the fence the board is edge-on to the car until the
            # moment it is level with it, which is a board you cannot read;
            # turned to face the oncoming driver it is legible for the whole
            # approach, which is the only thing it is for.
            aim = -track.tangent[i] * config.BOARD_AIM - out * (1.0 - config.BOARD_AIM)
            buckets[name].append((p[0], p[1], yaw_towards(aim)))
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
        q = _beyond_wall(track, i, side, 2.5)
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
        p = _edge(track, i, side, config.KERB_WIDTH + 0.7)
        places.append((p[0], p[1], 0.0))
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
