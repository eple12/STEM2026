"""Bake the Blender circuit kit into game-ready .bam props.

    python tools/build_blender_scenery.py

Source is ``assets/circuit_kit/obj``, written by

    blender -b -P blender/circuit_kit.py -- --out ai_sw/assets/circuit_kit/obj

Output is one .bam per prop (and per colour variant) under
``assets/models/circuit``, ready for ``PropLibrary`` to batch.

Two things this does that ``build_blender_f1.py`` deliberately does not:

* **No fitting.** A car has to land on the physics wheelbase; a grandstand is
  whatever size it was authored. Coordinates come through at 1:1, and the
  authored origin is kept exactly as it is -- ``hoarding`` sits at barrier
  height and ``fence_post`` starts a metre and a quarter up, and recentring
  either of them onto y = 0 would drop it on the floor.

* **Colour variants.** Blender materials here are paint *roles* with no colour
  of their own, so one exported mesh becomes ``hoarding_red`` and
  ``hoarding_blue`` from the same file. Repainting the circuit is an edit to
  the table below, not a re-export.

Axes: the kit is authored -y front, +x right, +z up (Blender's own front view).
The game is +z forward, +y up, +x right. Mapping (x, y, z) -> (x, z, -y) has
determinant +1, so it is a rotation and face winding survives it -- and a prop
authored facing -y comes out facing +z, which is what ``yaw_towards`` in
props.py expects of an unrotated model.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panda3d.core import Filename, NodePath

from game import config
from tools.build_blender_f1 import expand, geom_node, load_obj

SRC = config.ASSET_DIR / "circuit_kit" / "obj"
OUT = config.ASSET_DIR / "models" / "circuit"


def _c(r, g, b):
    return (r / 255.0, g / 255.0, b / 255.0, 1.0)


#: Paint role -> colour. Deliberately desaturated: this is a sunset-lit scene
#: and the shader warms everything, so anything mixed at full saturation here
#: comes out fluorescent on the track.
ROLE = {
    "Concrete": _c(176, 174, 166),
    "Steel": _c(122, 126, 134),
    "Dark": _c(26, 27, 30),
    "White": _c(224, 224, 226),
    "Accent": _c(184, 42, 40),
    "Glass": _c(46, 62, 84),
    "Tyre": _c(30, 30, 34),
    "Yellow": _c(222, 188, 44),
    "Panel": _c(60, 90, 150),
    "Trunk": _c(96, 72, 52),
    "Leaf": _c(74, 122, 56),
    "Leaf2": _c(52, 96, 46),
}

#: prop name -> {asset name: {role: colour override}}. One exported mesh, several
#: painted props. Hoardings alternate down a straight; stands alternate down a
#: run, and that alternation is most of what stops a long run reading as one
#: extruded block.
#: Three greens per tree shape. A forest of one colour reads as one object no
#: matter how many silhouettes it is made of, and the placement picks a variant
#: per tree, so the treeline gets its depth from tone as well as from outline.
_TREE_GREENS = (
    ("a", _c(84, 132, 60), _c(58, 104, 50)),
    ("b", _c(66, 112, 52), _c(44, 84, 42)),
    ("c", _c(104, 146, 68), _c(72, 116, 56)),
)
VARIANTS = {
    stem: {f"{stem}_{tag}": {"Leaf": leaf, "Leaf2": leaf2}
           for tag, leaf, leaf2 in _TREE_GREENS}
    for stem in ("tree_round", "tree_pine", "tree_bush", "tree_spread",
                 "tree_cypress")
}
VARIANTS.update({
    # The start lights, lit and out. One mesh, two paints, and the race
    # enables one of them -- a colour scale on a batched, shader-lit node is
    # not reliable, and two nodes are.
    "gantry_lamps": {
        "gantry_lamps_on": {"Accent": _c(232, 40, 32)},
        "gantry_lamps_off": {"Accent": _c(34, 22, 22)},
    },
    "hoarding": {
        "hoarding_a": {"Panel": _c(96, 124, 168)},
        "hoarding_b": {"Panel": _c(176, 84, 76)},
        "hoarding_c": {"Panel": _c(216, 214, 206)},
        "hoarding_d": {"Panel": _c(112, 148, 124)},
    },
})


def to_game(p):
    """Blender (-y front, +x right, +z up) -> game (+z forward, +y up)."""
    return np.column_stack([p[:, 0], p[:, 2], -p[:, 1]])


def build(stem: str, asset: str, overrides: dict, verbose: bool) -> int:
    verts, norms, groups = load_obj(SRC / f"{stem}.obj")
    g = to_game(verts)
    gn = to_game(norms)

    root = NodePath(asset)
    total = 0
    for mat, (tv, tn) in groups.items():
        colour = overrides.get(mat, ROLE.get(mat))
        if colour is None:
            print(f"  ! unmapped material {mat!r} on {stem} -> Steel")
            colour = ROLE["Steel"]
        v, n = expand(g, gn, tv, tn)
        if not len(v):
            continue
        root.attach_new_node(geom_node(f"{asset}__{mat}", v, n, colour))
        total += len(v) // 3

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{asset}.bam"
    if not root.write_bam_file(Filename.from_os_specific(str(dest))):
        raise SystemExit(f"could not write {dest}")
    if verbose:
        lo, hi = g.min(0), g.max(0)
        print(f"  {asset:16s} {total:4d} tris   "
              f"w {hi[0] - lo[0]:5.2f}  h {hi[1] - lo[1]:5.2f}  "
              f"d {hi[2] - lo[2]:5.2f} m   base y {lo[1]:+.2f}")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default=None, help="one part stem, e.g. gantry")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    stems = sorted(p.stem for p in SRC.glob("*.obj"))
    if not stems:
        raise SystemExit(
            f"no OBJs under {SRC}\nrun:  blender -b -P blender/circuit_kit.py "
            f"-- --out {SRC}")
    if args.part:
        stems = [s for s in stems if s == args.part]

    t = time.time()
    total = 0
    for stem in stems:
        for asset, overrides in VARIANTS.get(stem, {stem: {}}).items():
            total += build(stem, asset, overrides, not args.quiet)
    print(f"-> assets/models/circuit   {total:,} tris across "
          f"{len(list(OUT.glob('*.bam')))} props   {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
