"""Circuit furniture for the FORMULA-AI game, built in Blender.

    blender -b -P blender/circuit_kit.py -- --out ai_sw/assets/circuit_kit/obj
    blender -b -P blender/circuit_kit.py -- --sheet blender/kit_sheet.png

Convention (see blendkit.py): -Y is the front, the side that faces the track;
+X right, +Z up, base on z = 0, metres. Materials carry no colour -- the name
is a paint role and ``tools/build_blender_scenery.py`` maps it to a game
colour, so a repaint never needs a re-export.

Two kinds of part, and the difference matters at placement time:

* **Stretchable** -- built with ``extrude_x``, a cross-section swept along X.
  Scaling one along X only makes it longer. Runs of tyre wall, hoarding,
  debris fence and grandstand seating are laid out as a handful of stretched
  copies rather than thousands of modules, which is the same trick the barrier
  line already uses and the reason a 6 km circuit can be lined at all.
* **Fixed** -- pit garages, the gantry, marshal posts. Placed one at a time at
  their authored size.

Parts whose name ends in a role (``hoarding``) are baked once per colour
variant by the scenery builder; that is why the panel material is called
``Panel`` and not ``Red``.
"""

import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blendkit import (TAU, box, circle_profile, clear_scene, cone,  # noqa: E402
                      cylinder_z, export_obj, extrude_x, ico, slab,
                      taper_z)

import math  # noqa: E402

R = math.radians


# ---------------------------------------------------------------- pit lane
def pit_garage():
    """One garage bay of the pit building. Repeated along the pit straight.

    16 m of frontage and 8.6 m to the roof -- a real team's garage, and big
    enough that a Formula 1 car parked in the doorway looks like it fits
    through it rather than like a toy in front of a shed.
    """
    slab("shell", -8.0, 8.0, -5.8, 5.8, 0.0, 5.4, "Concrete")
    slab("door", -5.8, 5.8, -5.95, -5.7, 0.3, 4.7, "Dark")
    slab("lintel", -8.0, 8.0, -6.05, -5.72, 4.7, 5.15, "Accent")
    slab("upper", -8.0, 8.0, -5.0, 5.8, 5.4, 8.2, "Concrete")
    slab("glazing", -7.4, 7.4, -5.16, -4.94, 6.0, 7.6, "Glass")
    slab("roof", -8.4, 8.4, -6.3, 6.3, 8.2, 8.7, "Steel")
    # Pit-lane canopy: the deep shadow under it is most of what tells you this
    # is a pit building rather than a shed.
    slab("canopy", -8.0, 8.0, -10.8, -5.7, 5.2, 5.5, "Steel")
    for x in (-6.8, 6.8):
        slab("canopy_post", x - 0.13, x + 0.13, -10.7, -10.4, 0.0, 5.2, "Steel")


def pit_wall():
    """Stretchable pit wall: concrete with a painted top rail. 6 m authored."""
    extrude_x("wall", [(-0.22, 0.0), (0.22, 0.0), (0.22, 0.92), (-0.22, 0.92)],
              -3.0, 3.0, "Concrete")
    extrude_x("rail", [(-0.27, 0.92), (0.27, 0.92), (0.27, 1.02), (-0.27, 1.02)],
              -3.0, 3.0, "Accent")


def control_tower():
    """Race control. One per circuit, on the start line -- a tall landmark is
    what stops a long straight reading as an empty corridor."""
    slab("base", -5.0, 5.0, -5.0, 5.0, 0.0, 8.0, "Concrete")
    slab("mid", -4.2, 4.2, -4.2, 4.2, 8.0, 13.0, "Concrete")
    slab("band", -4.3, 4.3, -4.32, -4.1, 9.0, 11.5, "Glass")
    slab("control", -4.8, 4.8, -4.8, 4.8, 13.0, 16.4, "Glass")
    slab("roof", -5.2, 5.2, -5.2, 5.2, 16.4, 16.9, "Steel")
    cylinder_z("mast", (0.0, 0.0, 16.9), 0.14, 4.8, "Steel", n=8)


# ---------------------------------------------------------------- overhead
def _lattice_leg(x, height, mat="Steel", half=0.42, bays=6):
    """A hollow square-section steel tower: four corner chords and an X of
    bracing on every face of every bay.

    The flat zig-zag this replaces was two posts in one plane, so from any
    angle but dead ahead it read as a ladder lying flat. A gantry leg is a
    three-dimensional truss and the giveaway is that you can see through it in
    both directions at once -- which needs the bracing on all four faces, not
    on one.
    """
    for dx in (-half, half):
        for dy in (-half, half):
            slab("chord", x + dx - 0.075, x + dx + 0.075,
                 dy - 0.075, dy + 0.075, 0.0, height, mat)
    step = height / bays
    diag = math.hypot(2 * half, step)
    ang = math.atan2(step, 2 * half) - math.pi / 2
    for k in range(bays):
        z = (k + 0.5) * step
        # The two faces across the span brace in the XZ plane (ry), the two
        # along it in the YZ plane (rx). Both diagonals of each X.
        for dy in (-half, half):
            for sign in (1, -1):
                box("brace", (x, dy, z), (diag, 0.055, 0.075), mat,
                    ry=sign * ang)
        for dx in (-half, half):
            for sign in (1, -1):
                box("brace", (x + dx, 0.0, z), (0.055, diag, 0.075), mat,
                    rx=sign * ang)
        # A horizontal ring at every bay joint ties the four chords together.
        for dy in (-half, half):
            slab("ring", x - half, x + half, dy - 0.05, dy + 0.05,
                 z + step * 0.5 - 0.05, z + step * 0.5 + 0.05, mat)
        for dx in (-half, half):
            slab("ring", x + dx - 0.05, x + dx + 0.05, -half, half,
                 z + step * 0.5 - 0.05, z + step * 0.5 + 0.05, mat)


def gantry():
    """Start/finish gantry, authored for an 18 m span.

    Placed once, scaled in X to the track width, so the legs land just outside
    the barrier line. A 10-20 per cent stretch on a square-section leg is not
    something anyone reads at racing speed.
    """
    h = 7.4
    for x in (-8.4, 8.4):
        _lattice_leg(x, h)
    # Top chord: a box girder with its own bracing, not a plain slab.
    slab("beam", -9.4, 9.4, -0.44, 0.44, h, h + 1.15, "Steel")
    slab("beam_lip", -9.5, 9.5, -0.52, 0.52, h + 1.15, h + 1.30, "Steel")
    slab("banner", -8.7, 8.7, -0.60, -0.44, h + 0.14, h + 1.02, "White")
    slab("banner_stripe", -8.7, 8.7, -0.68, -0.58, h + 0.76, h + 0.98, "Accent")

    # The five start lights: clustered in the middle with a small gap between
    # them, and taller than they are wide. Spread across the whole span, as
    # they were, they read as five separate signs rather than as the one
    # thing every viewer recognises.
    pitch, w, ht = 0.92, 0.66, 1.30
    # In front of the girder, not inside it. The beam is 0.88 m deep and the
    # cases were mixed within that depth, so from the track they were buried
    # in the thing they hang off.
    rail_y0, rail_y1 = -0.92, -0.50
    slab("light_rail", -2.6, 2.6, rail_y0 - 0.06, rail_y1 + 0.02,
         h - 0.26, h - 0.06, "Steel")
    for x in (-2.3, 2.3):
        slab("light_hanger", x - 0.06, x + 0.06, rail_y1, 0.0,
             h - 0.26, h + 0.10, "Steel")
    for k in range(5):
        x = (k - 2) * pitch
        top = h - 0.26
        slab("light_case", x - w / 2, x + w / 2, rail_y0, rail_y1,
             top - ht, top, "Dark")
    # The lamps themselves are a separate prop -- see gantry_lamps(). They are
    # the one part of the circuit that changes during a session, and a mesh
    # baked into the gantry cannot be switched off.


def _lamp_geometry():
    """The ten lamp faces, in the gantry's own coordinates."""
    h = 7.4
    pitch, w, ht = 0.92, 0.66, 1.30
    rail_y0 = -0.92
    for k in range(5):
        x = (k - 2) * pitch
        top = h - 0.26
        # Two lamps stacked in each case, which is what a real gantry carries.
        for z0 in (top - ht + 0.12, top - ht * 0.52):
            slab("lamp", x - w / 2 + 0.09, x + w / 2 - 0.09,
                 rail_y0 - 0.07, rail_y0 - 0.01, z0, z0 + ht * 0.34, "Accent")


def gantry_lamps():
    """Just the gantry's ten lamp faces, as their own prop.

    Placed on top of the gantry at the same transform and stretched with it.
    Two painted copies are baked from this one mesh -- lit and dark -- and the
    race swaps which is enabled, so the lights over the grid go out with the
    ones on the HUD instead of glowing red for the whole session.
    """
    _lamp_geometry()



def bridge():
    """Spectator bridge, authored for a 22 m span. Same stretch rule as the
    gantry. Two of these round a lap break up the skyline more than any
    amount of extra grandstand does."""
    slab("tower_l", -12.4, -9.6, -2.4, 2.4, 0.0, 7.0, "Concrete")
    slab("tower_r", 9.6, 12.4, -2.4, 2.4, 0.0, 7.0, "Concrete")
    slab("deck", -12.4, 12.4, -1.7, 1.7, 6.7, 7.2, "Concrete")
    for y in (-1.7, 1.55):
        slab("parapet", -12.4, 12.4, y, y + 0.15, 7.2, 9.1, "White")
    slab("bridge_roof", -12.8, 12.8, -2.0, 2.0, 9.1, 9.4, "Steel")
    # A band on one face only gives the span a front and a back, and which one
    # ends up facing the circuit then depends on the placement yaw. Both faces
    # carry it, so there is no wrong way round to put the bridge.
    for y0 in (-1.86, 1.72):
        slab("bridge_band", -12.4, 12.4, y0, y0 + 0.14, 7.9, 8.7, "Accent")


# ---------------------------------------------------------------- barriers
def tyre_wall():
    """Three rows of tyres with a belt strap across them. Stretchable.

    Swept circles rather than modelled tyres: the silhouette is identical at
    any distance you would ever see it from, and it survives being stretched
    over a 40 m run, which a stack of individual tyres does not.
    """
    for z in (0.33, 0.93, 1.53):
        extrude_x("tyres", circle_profile(0.0, z, 0.34, 12), -1.0, 1.0,
                  "Tyre", smooth=True)
    extrude_x("belt", [(-0.42, 0.72), (-0.30, 0.72), (-0.30, 1.14), (-0.42, 1.14)],
              -1.0, 1.0, "White")


def hoarding():
    """Trackside advertising panel, sitting on top of the barrier line.

    Authored with its base at barrier height rather than at ground level: the
    batcher places every prop at y = 0, so a part that belongs up in the air
    has to carry that offset in its own geometry.

    Kept deliberately short. At 1.35 m it stood 2.6 m over the run-off, which
    from a car whose airbox is barely a metre up is not a hoarding, it is a
    fence -- and it hid the trees, the marshals and the hills behind it.
    """
    base, h = 1.05, 0.80
    extrude_x("panel", [(-0.06, base), (0.06, base),
                        (0.06, base + h), (-0.06, base + h)],
              -2.0, 2.0, "Panel")
    extrude_x("cap", [(-0.10, base + h), (0.10, base + h),
                      (0.10, base + h + 0.10), (-0.10, base + h + 0.10)],
              -2.0, 2.0, "Steel")


# ---------------------------------------------------------------- trees
# Five silhouettes, not five sizes: a forest made of one shape scaled up and
# down still reads as one shape. Placement varies the size on top of this, so
# what these have to supply is *outline* -- a round crown, a conifer, a squat
# bush, a wide spreading canopy, a tall thin cypress. Faceted on purpose: one
# icosphere subdivision is 80 flat triangles, and the facets are what make a
# low-poly tree look drawn rather than melted.
def tree_round():
    """Broad deciduous, about 12 m. The workhorse of the treeline."""
    taper_z("trunk", (0.0, 0.0, 0.0), 0.42, 0.26, 4.6, "Trunk", verts=6)
    ico("crown_a", (0.0, 0.0, 7.4), 3.6, "Leaf", scale=(1.0, 1.0, 0.92))
    ico("crown_b", (1.7, 0.9, 6.0), 2.5, "Leaf2")
    ico("crown_c", (-1.4, -1.1, 9.2), 2.3, "Leaf")


def tree_pine():
    """Conifer, about 15 m -- the tall dark punctuation in a treeline."""
    taper_z("trunk", (0.0, 0.0, 0.0), 0.34, 0.22, 3.0, "Trunk", verts=6)
    for z, r, h, mat in ((2.4, 3.5, 4.6, "Leaf2"), (5.6, 2.8, 4.2, "Leaf"),
                         (8.4, 2.1, 3.8, "Leaf2"), (10.9, 1.3, 3.4, "Leaf")):
        cone("frond", (0.0, 0.0, z), r, r * 0.18, h, mat, verts=8)


def tree_bush():
    """Squat, about 6 m. Fills the gaps a big tree leaves at ground level."""
    taper_z("trunk", (0.0, 0.0, 0.0), 0.26, 0.18, 1.9, "Trunk", verts=6)
    ico("crown_a", (0.0, 0.0, 3.5), 2.3, "Leaf2", scale=(1.15, 1.15, 0.85))
    ico("crown_b", (1.0, -0.7, 2.6), 1.5, "Leaf")


def tree_spread():
    """Wide, flat-topped, about 10 m -- a canopy that reads across a gap."""
    taper_z("trunk", (0.0, 0.0, 0.0), 0.48, 0.30, 3.9, "Trunk", verts=6)
    ico("crown_a", (0.0, 0.0, 6.2), 4.6, "Leaf", scale=(1.0, 1.0, 0.52))
    ico("crown_b", (2.4, 1.6, 5.4), 2.6, "Leaf2", scale=(1.0, 1.0, 0.6))
    ico("crown_c", (-2.2, -1.5, 5.6), 2.3, "Leaf", scale=(1.0, 1.0, 0.6))


def tree_cypress():
    """Tall and narrow, about 13 m. Italian, and Monza is in a royal park."""
    # The canopy has to reach *below* the top of the trunk. Meeting it exactly
    # leaves a ring of cut-off trunk showing through wherever the two are a
    # facet out of line, which on a shape this narrow is everywhere.
    taper_z("trunk", (0.0, 0.0, 0.0), 0.26, 0.20, 2.2, "Trunk", verts=6)
    ico("crown", (0.0, 0.0, 6.9), 5.9, "Leaf2", scale=(0.32, 0.32, 1.0))
    ico("crown_skirt", (0.0, 0.0, 1.9), 1.15, "Leaf2")
    ico("crown_top", (0.0, 0.0, 11.9), 1.3, "Leaf")


# ---------------------------------------------------------------- barrier
def guardrail():
    """Armco: a corrugated W-beam on posts. Stretchable (posts are separate).

    Replaces the Kenney barrier module. The reference circuit's boundary is
    this and nothing else on the straights -- a steel rail low enough to see
    over, with the debris fence standing behind it -- and that reads as a
    circuit far better than a solid coloured wall does.
    """
    # The W is what makes a guardrail recognisable at a glance, and it is six
    # points: face, valley, face, and the two lips folded back.
    face, back = -0.09, 0.05
    prof = [(face, 0.52), (back, 0.60), (face, 0.68), (face, 0.755),
            (back, 0.80), (back, 0.92), (face, 0.885), (face, 0.96),
            (back, 1.02), (face, 1.10), (face, 1.02), (back, 0.86),
            (back, 0.66), (face, 0.60)]
    extrude_x("rail", prof, -2.0, 2.0, "Steel")


def guardrail_post():
    """The post an Armco rail is bolted to. Placed at its own pitch."""
    # One box. There are several thousand of these round a lap and a base
    # plate nobody can see at racing speed doubles the whole batch.
    slab("post", -0.075, 0.075, -0.06, 0.06, 0.0, 0.98, "Steel")


def debris_fence():
    """The tall fence behind the guardrail: posts, rails and a mesh panel.

    Stands on the ground rather than starting at barrier height. The part it
    replaces began 1.25 m up so it could sit on a wall module, and with the
    wall gone that reads as a fence floating in the air.
    """
    # Rails only, and five of them. A solid panel is what a chain-link mesh
    # would need to be without an alpha texture, and a four-metre opaque wall
    # round the whole circuit would hide everything this scene is being filled
    # with. Close-pitched rails and posts read as a fence and stay see-through,
    # which is the property that actually matters here.
    for z in (0.62, 1.72, 2.82, 3.92, 5.02, 5.82):
        extrude_x("rail", [(-0.05, z), (0.05, z), (0.05, z + 0.075),
                           (-0.05, z + 0.075)], -2.0, 2.0, "Steel")


def debris_post():
    """Fence post, from the ground up, with the top kinked over the run-off."""
    top, ang, L = 5.90, R(30), 1.15
    slab("post", -0.07, 0.07, -0.07, 0.07, 0.0, top, "Steel")
    # The kink leans towards the track, and its lower end sits exactly on the
    # post top. Both halves of that have caught me out: a rotated box moves
    # about its centre in *both* axes, so half the length along the tilt has
    # to be added back in each -- and with the rotation the other way round
    # the arithmetic still lands the box on the post while leaning it out over
    # the countryside, which looked fixed and was not.
    box("kink", (0.0, -0.5 * L * math.sin(ang), top + 0.5 * L * math.cos(ang)),
        (0.14, 0.14, L), "Steel", rx=ang)


# ---------------------------------------------------------------- signage
def marshal_post():
    """Marshal station: platform, shelter, and the flags on the front rail."""
    slab("platform", -1.8, 1.8, -1.3, 1.3, 0.0, 0.95, "Concrete")
    slab("back", -1.8, 1.8, 1.1, 1.3, 0.95, 2.95, "White")
    for x in (-1.8, 1.6):
        slab("side", x, x + 0.2, -1.3, 1.3, 0.95, 2.95, "White")
    slab("shelter_roof", -2.1, 2.1, -1.7, 1.6, 2.95, 3.17, "Accent")
    for x in (-1.7, 1.55):
        slab("shelter_post", x, x + 0.15, -1.35, -1.2, 0.95, 2.95, "Steel")
    slab("flag_y", -1.5, -0.7, -1.42, -1.36, 1.45, 2.15, "Yellow")
    slab("flag_r", 0.7, 1.5, -1.42, -1.36, 1.45, 2.15, "Accent")


#: Seven-segment masks. Modelled as geometry because there are no textures
#: anywhere in this game -- and "150" is what a braking board says, whereas
#: three bars is a thing a driver has to be taught to read.
_SEGMENTS = {
    "0": "abcdef", "1": "bc", "5": "afgcd",
}
#: How wide each glyph actually draws, as a fraction of its cell. A "1" is two
#: strokes down one side of its box; advanced at full cell width it sits hard
#: against its right-hand neighbour and leaves a hole on the other side, which
#: is what made the gap between 1 and 5 differ from the gap between 5 and 0.
_GLYPH_WIDTH = {"1": 0.34}


def _digit(x_mid, z0, w, h, t, glyph, mat="Dark"):
    """One seven-segment numeral, centred on *x_mid*, front face at y = -t."""
    b = 0.155 * h                                    # stroke thickness
    gw = w * _GLYPH_WIDTH.get(glyph, 1.0)
    x0 = x_mid - gw / 2
    bars = {
        "a": (x0, x0 + gw, z0 + h - b, z0 + h),
        "d": (x0, x0 + gw, z0, z0 + b),
        "g": (x0, x0 + gw, z0 + h / 2 - b / 2, z0 + h / 2 + b / 2),
        "f": (x0, x0 + b, z0 + h / 2, z0 + h),
        "b": (x0 + gw - b, x0 + gw, z0 + h / 2, z0 + h),
        "e": (x0, x0 + b, z0, z0 + h / 2),
        "c": (x0 + gw - b, x0 + gw, z0, z0 + h / 2),
    }
    # Emitted mirrored in x. The face these sit on looks down -y, and reading
    # it from out there puts the model's +x on the *left* of the page -- so
    # laid out in the obvious direction the whole block comes out as its own
    # mirror image. Negating x here flips the picture, glyph shapes and
    # reading order together, which is the only thing that was wrong with it.
    for key in _SEGMENTS[glyph]:
        a0, a1, c0, c1 = bars[key]
        slab("seg", -a1, -a0, -t - 0.05, -t, c0, c1, mat)


def _board(text):
    """Distance board on its own two posts, standing square across the run-off.

    It used to be a bare panel bolted flat to the debris fence, with two stubs
    off the back reaching to the fence line. Flat on the fence is the one
    angle a countdown board cannot be read from: it is edge-on to the car
    until the instant the car is level with it, which is a hundred metres too
    late to brake against. Turned to meet the driver it is legible for the
    whole approach, which is the only thing it is for -- and turned, it is no
    longer against the fence, so it needs legs and the stubs have nothing to
    reach to.

    Authored facing -y, panel width along x, feet at z = 0. Placed on the
    inside of the barrier by scenery._distance_boards, which stands it off by
    its own half-width so no part of it hangs outside the circuit.
    """
    dw, dh, gap = 0.50, 0.82, 0.16
    span = len(text) * dw + (len(text) - 1) * gap
    half = span / 2 + 0.24
    z0 = 1.42                        # panel bottom well above a kerb, at eye
    # Legs. Set in from the ends so the panel reads as carried rather than as
    # a gate, and square in section -- a round post is more polygons than a
    # thing this size is worth.
    for x in (-half + 0.34, half - 0.34):
        slab("board_post", x - 0.06, x + 0.06, -0.06, 0.06, 0.0, z0 + 0.22,
             "Steel")
    slab("board_foot", -half + 0.28, half - 0.28, -0.10, 0.10, 0.0, 0.10,
         "Steel")
    slab("board_face", -half, half, -0.12, -0.02,
         z0 - 0.20, z0 + dh + 0.20, "White")
    slab("board_edge", -half, half, -0.02, 0.06,
         z0 - 0.20, z0 + dh + 0.20, "Steel")
    for k, ch in enumerate(text):
        _digit(-span / 2 + dw / 2 + k * (dw + gap), z0, dw, dh, 0.12, ch)



PARTS = {
    "pit_garage": pit_garage,
    "pit_wall": pit_wall,
    "control_tower": control_tower,
    "gantry": gantry,
    "gantry_lamps": gantry_lamps,
    "bridge": bridge,
    "tyre_wall": tyre_wall,
    "hoarding": hoarding,
    "guardrail": guardrail,
    "guardrail_post": guardrail_post,
    "debris_fence": debris_fence,
    "debris_post": debris_post,
    "tree_round": tree_round,
    "tree_pine": tree_pine,
    "tree_bush": tree_bush,
    "tree_spread": tree_spread,
    "tree_cypress": tree_cypress,
    "marshal_post": marshal_post,
    "board_50": lambda: _board("50"),
    "board_100": lambda: _board("100"),
    "board_150": lambda: _board("150"),
}

#: Parts that must only ever be scaled along X (see the module docstring).
STRETCHABLE = ("pit_wall", "tyre_wall", "hoarding", "guardrail",
               "debris_fence")


# ---------------------------------------------------------------- drivers
def export_all(out_dir):
    out_dir = os.path.abspath(out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for name, build in PARTS.items():
        clear_scene()
        build()
        objs = export_obj(os.path.join(out_dir, name + ".obj"))
        tris = sum(len(o.data.polygons) for o in objs)
        print("[kit] %-14s %2d objects  %4d faces" % (name, len(objs), tris))
    print("[kit] -> " + out_dir)


def contact_sheet(path):
    """Every part laid out in a row and rendered once -- the cheapest way to
    see whether a part is the right shape and the right size before it is
    baked, batched and driven past at 300 km/h."""
    from mathutils import Vector

    clear_scene()
    x = 0.0
    spans = []
    for name, build in PARTS.items():
        before = set(bpy.context.scene.objects)
        build()
        made = [o for o in bpy.context.scene.objects if o not in before]
        lo = min(min(v.co.x for v in o.data.vertices) for o in made)
        hi = max(max(v.co.x for v in o.data.vertices) for o in made)
        for o in made:
            o.location.x += x - lo + 2.0
        spans.append((name, x - lo + 2.0))
        x += (hi - lo) + 6.0

    # Paint the roles roughly as the game will, so the sheet checks material
    # assignment as well as shape -- a part with the wrong role reads as the
    # wrong object, and that is far easier to spot in colour.
    preview = {"Concrete": (0.62, 0.61, 0.58), "Steel": (0.46, 0.48, 0.52),
               "Dark": (0.07, 0.07, 0.08), "White": (0.88, 0.88, 0.88),
               "Accent": (0.72, 0.10, 0.11), "Glass": (0.13, 0.18, 0.26),
               "Seat": (0.55, 0.16, 0.16), "Panel": (0.16, 0.32, 0.62),
               "Tyre": (0.09, 0.09, 0.10), "Yellow": (0.85, 0.70, 0.10)}
    for name, rgb in preview.items():
        mat = bpy.data.materials.get(name)
        if mat is None:
            continue
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.6

    bpy.ops.mesh.primitive_plane_add(size=900, location=(x / 2, 0, -0.02))
    ground = bpy.context.object
    gmat = bpy.data.materials.new("sheet_ground")
    gmat.use_nodes = True
    gmat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
        0.20, 0.22, 0.20, 1.0)
    ground.data.materials.append(gmat)

    # Orthographic: a perspective camera far enough back to hold 190 m of parts
    # makes the near ones tiny and the far ones tinier. Ortho gives every part
    # the same scale on the sheet, which is the point of a sheet.
    # rotation_euler.x = 90 aims an ortho camera horizontally along +y; less
    # than that tilts it down, and the drop over the camera distance has to be
    # added back to its height or the frame centres on empty ground.
    dist, tilt, aim_z = 260.0, 6.0, 11.0
    bpy.ops.object.camera_add(
        location=(x / 2, -dist, aim_z + dist * math.tan(R(tilt))))
    cam = bpy.context.object
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = x + 8.0
    cam.rotation_euler = (R(90.0 - tilt), 0.0, 0.0)
    bpy.context.scene.camera = cam

    for loc, energy in (((x * 0.35, -260.0, 220.0), 30.0),
                        ((x * 0.8, 180.0, 140.0), 9.0)):
        bpy.ops.object.light_add(type='SUN', location=loc)
        light = bpy.context.object
        light.data.energy = energy / 3.0
        light.rotation_euler = (Vector((x / 2, 0, 4)) - Vector(loc)).to_track_quat(
            '-Z', 'Y').to_euler()

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 32
    scene.render.resolution_x = 2200
    scene.render.resolution_y = int(2200 * 34.0 / (x + 8.0))
    scene.render.filepath = os.path.abspath(path)
    bpy.ops.render.render(write_still=True)
    for name, at in spans:
        print("[kit] %-14s at x=%.1f" % (name, at))
    print("[kit] sheet -> " + os.path.abspath(path))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--out" in argv:
        export_all(argv[argv.index("--out") + 1])
    if "--sheet" in argv:
        contact_sheet(argv[argv.index("--sheet") + 1])


if __name__ == "__main__":
    main()
