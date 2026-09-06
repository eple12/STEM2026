"""Bake the Blender F1 car (``blender/f1_car.py``) into a game-ready .bam.

    python tools/build_blender_f1.py                # both liveries
    python tools/build_blender_f1.py --livery f1red

Source is ``assets/blender_f1/obj``: five OBJ parts written by

    blender -b blender/f1_car.py -- --game-obj ai_sw/assets/blender_f1/obj

-- ``body.obj`` plus one file per wheel, in Blender's own axes (+x forward,
+y left, +z up, metres) with the Blender material names intact.

What this does, and why:

* **Re-frame.** Blender is +x forward / +y left / +z up; the game is +z
  forward / +y up / +x right. The mapping is ``(y, z, x)``, whose determinant
  is +1 -- a proper rotation, so face winding, and therefore back-face
  culling, survives it.

* **Fit.** The game's collision box is 4.45 x 2.41 m: far stubbier than the
  5.70 x 1.82 m the model follows, because the physics car is a GT car. The
  length is fitted first (bodywork has to stay inside the box the walls are
  tested against), which then leaves the car far too narrow, so a second
  factor stretches it back out to the box width.

  The stretch is applied to the *whole* car -- bodywork, wheel placement and
  wheel geometry alike -- rather than to the body alone. Stretching only the
  body is what would leave the wishbones reaching for wheels that are no
  longer where the model put them, which is exactly the floating-strut
  problem the model was built to avoid. Everything moves together, so every
  joint the Blender script anchored stays anchored.

* **Recolour.** Blender material name -> paint role -> livery colour, flat
  vertex colours. The normals from the OBJ are kept as authored (the Blender
  script smooth-shades the body and flat-shades the wings), because the game
  lights the car with a real directional shader -- see ``app.py``'s
  ``light.apply(self.car, ...)`` -- so normals are not decoration here.

* **Wheels.** Each wheel OBJ becomes a node under the name ``Car._rig_wheels``
  looks for. The geometry stays in car coordinates and the node carries no
  transform of its own: ``_rig_wheels`` derives the hub from the node's tight
  bounds and ``wrt_reparent_to`` preserves the position, so this is both
  simpler and immune to the pivot-vs-centre mistake that arc-swings a wheel.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panda3d.core import (Filename, Geom, GeomNode, GeomTriangles,
                          GeomVertexData, GeomVertexFormat, GeomVertexWriter,
                          NodePath)

from game import config, f1car

SRC = config.ASSET_DIR / "blender_f1" / "obj"
OUT = config.ASSET_DIR / "models" / "f1"

#: Blender material name -> paint role.
ROLE = {
    "Livery": "body",
    "Carbon": "carbon",
    "Interior": "dark",
    "Rubber": "tyre",
    "Metal": "rim",
}

#: Blender wheel tag -> the node name the game rigs. The mapping below turns
#: the car through 180 degrees about the vertical, so left stays left.
WHEEL_NODE = {
    "FL": "wheelFrontLeft", "FR": "wheelFrontRight",
    "RL": "wheelBackLeft", "RR": "wheelBackRight",
}

#: Livery key -> asset name, kept distinct from f1car.py's procedural
#: f1red / f1white and from build_f1_asset.py's rb_red / rb_white.
ASSET_NAME = {"f1red": "bl_red", "f1white": "bl_white"}


def _colours(livery: str):
    liv = f1car.LIVERIES[livery]
    return {"body": liv["body"], "accent": liv["accent"],
            "carbon": f1car.CARBON, "dark": f1car.CARBON_LT,
            "tyre": f1car.TYRE, "rim": f1car.RIM}


# --- obj ---------------------------------------------------------------
def load_obj(path: Path):
    """(positions, normals, {material: (tri v-indices, tri n-indices)}).

    n-gons are fan-triangulated. Normals are kept rather than recomputed:
    the source distinguishes smooth from flat shading and that distinction is
    the difference between a curved monocoque and a faceted one.
    """
    vs, ns, gv, gn, cur = [], [], {}, {}, "default"
    with open(path, "r", errors="replace") as fh:
        for ln in fh:
            if ln.startswith("v "):
                p = ln.split()
                vs.append((float(p[1]), float(p[2]), float(p[3])))
            elif ln.startswith("vn "):
                p = ln.split()
                ns.append((float(p[1]), float(p[2]), float(p[3])))
            elif ln.startswith("usemtl "):
                cur = ln.split(None, 1)[1].strip()
            elif ln.startswith("f "):
                vi, ni = [], []
                for t in ln.split()[1:]:
                    bits = t.split("/")
                    vi.append(int(bits[0]) - 1)
                    ni.append(int(bits[2]) - 1 if len(bits) > 2 and bits[2] else 0)
                for k in range(1, len(vi) - 1):
                    gv.setdefault(cur, []).append((vi[0], vi[k], vi[k + 1]))
                    gn.setdefault(cur, []).append((ni[0], ni[k], ni[k + 1]))
    if not ns:
        ns = [(0.0, 1.0, 0.0)]
    return (np.asarray(vs, dtype=np.float64), np.asarray(ns, dtype=np.float64),
            {k: (np.asarray(gv[k], dtype=np.int64),
                 np.asarray(gn[k], dtype=np.int64)) for k in gv})


# --- transform ---------------------------------------------------------
# The Blender script lays the car out nose-at-the-origin and tail at +x, so
# the direction the car *faces* is -x, not +x. Mapping (x, y, z) -> (-y, z, -x)
# turns that into the game's +z forward and keeps left on the left. Its
# determinant is +1 -- a proper rotation, so face winding, and therefore
# back-face culling, survives it. (Getting this backwards is not subtle: the
# car drives down the track in reverse.)
def to_game(p, s, stretch, offset):
    """Blender (nose at -x, +y left, +z up) -> game (+x right, +y up, +z fwd)."""
    return np.column_stack([-p[:, 1] * s * stretch,
                            p[:, 2] * s,
                            -p[:, 0] * s]) + offset


def normals_to_game(n, stretch):
    """Same permutation, but a non-uniform scale needs the inverse transpose:
    dividing the stretched axis is what keeps a normal perpendicular to the
    surface it came from instead of leaning with it."""
    g = np.column_stack([-n[:, 1] / stretch, n[:, 2], -n[:, 0]])
    L = np.linalg.norm(g, axis=1)
    return g / np.where(L > 1e-12, L, 1.0)[:, None]


# --- panda geometry ----------------------------------------------------
def geom_node(name, v, n, c):
    fmt = GeomVertexFormat.get_v3n3c4()
    vdata = GeomVertexData(name, fmt, Geom.UH_static)
    vdata.set_num_rows(len(v))
    wv = GeomVertexWriter(vdata, "vertex")
    wn = GeomVertexWriter(vdata, "normal")
    wc = GeomVertexWriter(vdata, "color")
    for p, q in zip(v, n):
        wv.add_data3(float(p[0]), float(p[1]), float(p[2]))
        wn.add_data3(float(q[0]), float(q[1]), float(q[2]))
        wc.add_data4(float(c[0]), float(c[1]), float(c[2]), float(c[3]))
    prim = GeomTriangles(Geom.UH_static)
    prim.add_next_vertices(len(v))
    prim.close_primitive()
    geom = Geom(vdata)
    geom.add_primitive(prim)
    node = GeomNode(name)
    node.add_geom(geom)
    return node


def expand(verts, norms, tv, tn):
    """Indexed triangles -> the non-indexed corner stream Panda wants.

    Corner order is **reversed**, and this is not cosmetic: it is what decides
    which side of every face is drawn.

    An OBJ winds a triangle so that ``cross(b - a, c - a)`` points along the
    outward normal. Under Ursina's left-handed setup this renderer wants the
    opposite -- measured, not assumed: load any Kenney .glb, which renders
    correctly, and its winding disagrees with its (100 per cent outward)
    normals on every face. Emitted in source order, a model comes out with
    every face back-facing: solid from the inside, see-through from the
    outside, with the far interior wall showing through. Reversing here fixes
    the car and the whole circuit kit at one point, and leaves the normals --
    which were already outward, and drive the lighting -- untouched.
    """
    tv = np.ascontiguousarray(tv[:, ::-1])
    tn = np.ascontiguousarray(tn[:, ::-1])
    v = verts[tv.reshape(-1)]
    n = norms[tn.reshape(-1)] if len(norms) > 1 else np.tile(
        norms[0], (len(v), 1))
    return v, n


# --- build -------------------------------------------------------------
def build(livery: str, length_fit: float, width_fit: float, verbose: bool):
    cols = _colours(livery)
    names = ["body"] + list(WHEEL_NODE)
    parts = {}
    for name in names:
        path = SRC / f"{name}.obj"
        if not path.is_file():
            raise SystemExit(
                f"missing {path}\nrun:  blender -b blender/f1_car.py -- "
                f"--game-obj {SRC}")
        parts[name] = load_obj(path)

    allv = np.vstack([p[0] for p in parts.values()])
    lo, hi = allv.min(0), allv.max(0)
    model_len = float(hi[0] - lo[0])
    model_wid = float(hi[1] - lo[1])

    # Length first, then width back out to the box: see the module docstring.
    s = config.CAR_BODY_LENGTH * length_fit / model_len
    stretch = config.CAR_BODY_WIDTH * width_fit / (model_wid * s)

    # Nose (Blender's -x end, so game +z after the flip) on BODY_TO_FRONT,
    # centred laterally, and seated on the ground by the *tyres* rather than by
    # the lowest vertex anywhere -- a front wing that hangs a few millimetres
    # below the contact patch should dip into the asphalt, not lift the car off
    # it, because the contact patch is the part anyone looks at.
    tyre_bottom = min(float(parts[t][0][:, 2].min()) for t in WHEEL_NODE)
    offset = np.array([(lo[1] + hi[1]) / 2 * s * stretch,
                       -tyre_bottom * s,
                       config.BODY_TO_FRONT + lo[0] * s])

    root = NodePath("body")
    total = 0
    for name in names:
        verts, norms, groups = parts[name]
        g = to_game(verts, s, stretch, offset)
        gn = normals_to_game(norms, stretch)
        # The body hangs off the root; each wheel gets its own named node with
        # no transform of its own -- Car._rig_wheels derives the hub from the
        # geometry's bounds, so the wheel must stay in car coordinates.
        parent = root if name == "body" else root.attach_new_node(
            WHEEL_NODE[name])
        for mat, (tv, tn) in groups.items():
            v, n = expand(g, gn, tv, tn)
            if not len(v):
                continue
            role = ROLE.get(mat)
            if role is None:
                print(f"  ! unmapped material {mat!r} -> carbon")
                role = "carbon"
            parent.attach_new_node(
                geom_node(f"{name}__{mat}", v, n, cols[role]))
            total += len(v) // 3

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{ASSET_NAME[livery]}.bam"
    if not root.write_bam_file(Filename.from_os_specific(str(dest))):
        raise SystemExit(f"could not write {dest}")

    if verbose:
        fl = parts["FL"][0]
        track = abs((fl[:, 1].max() + fl[:, 1].min()) / 2 * s * stretch)
        wb = abs((parts["RL"][0][:, 0].mean() - fl[:, 0].mean()) * s)
        print(f"  fit     scale {s:.4f}   width stretch {stretch:.4f}")
        print(f"  fitted  l {model_len * s:.3f}  w {model_wid * s * stretch:.3f}"
              f"  (box {config.CAR_BODY_LENGTH} x {config.CAR_BODY_WIDTH} m)")
        print(f"  wheels  half track {track:.3f} m "
              f"(physics {config.WHEEL_HALF_TRACK:.3f})   "
              f"wheelbase {wb:.3f} m (physics {config.WHEELBASE:.3f})")
    print(f"  -> assets/models/f1/{dest.name}   {total:,} tris   "
          f"{dest.stat().st_size / 1e6:.2f} MB")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--livery", default=None)
    ap.add_argument("--length", type=float, default=1.0,
                    help="multiplier on the fitted length (1.0 = collision box)")
    ap.add_argument("--width", type=float, default=1.0,
                    help="multiplier on the fitted width (1.0 = collision box)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    for liv in ([args.livery] if args.livery else list(f1car.LIVERIES)):
        t = time.time()
        print(f"{liv}:")
        build(liv, args.length, args.width, not args.quiet)
        print(f"  {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
