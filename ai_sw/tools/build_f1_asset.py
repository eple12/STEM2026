"""Bake the supplied F1 car OBJs into a game-ready .bam, one per livery.

    python tools/build_f1_asset.py             # both liveries
    python tools/build_f1_asset.py --budget 2  # twice the triangles

``assets/f1_car_model`` holds the car as six Blender OBJ parts: 747k quads,
roughly 1.5 M triangles, with absolute Windows texture paths and no wheels.
None of that can go straight into the game, so this converts it once, offline.

* **Decimate.** Vertex clustering on a grid, with the cell size bisected per
  part until it lands inside that part's triangle budget. Clustering keeps the
  silhouette -- which is what reads at racing distance -- and it is the only
  decimation that runs in seconds over a million triangles with no extra
  dependency.

* **Recolour.** The materials are either carbon or a paint-region id
  (``material_map_red`` and friends: the model ships an id map, not a finished
  livery). Each id maps to a role and each role to a livery colour, so the
  same geometry gives the player's red car and the AI's white one. Flat vertex
  colours, no textures -- it matches the rest of the game and costs nothing at
  runtime.

* **Re-frame.** The model is +x forward at about 2.45 units per metre. The
  game is +z forward, metres, origin at the centre of gravity on the ground.
  The mapping is a proper rotation rather than a mirror, so face winding --
  and therefore back-face culling -- survives it.

* **Add wheels.** The OBJ set has none, so the procedural wheel from
  ``f1car`` is bolted on at the physics axles, under the node names that
  ``Car._rig_wheels`` looks for.
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

SRC = config.ASSET_DIR / "f1_car_model" / "3D_parts_mixer"
OUT = config.ASSET_DIR / "models" / "f1"

#: Triangles kept per part. The body and the wings carry the shape people
#: actually look at; the mirrors part is mostly winglets and the cockpit foam
#: is barely visible, so they get less.
BUDGET = {
    "fp04rb_main_body": 46000,
    "fp04rb_floor": 20000,
    "fp04rb_front_wing": 20000,
    "fp04rb_rear_wing": 16000,
    "fp04rb_mirrors": 9000,
    "fp04rb_cockpit_foam": 4000,
}

#: Material name to paint role. The material_map_* names are the model's own
#: paint-region ids, so this table is where a livery actually gets applied.
ROLE = {
    "Carbon_Fiber_(no_UV)": "carbon",
    "cockpit_foam_texture": "dark",
    "main_body_texture": "body",
    "front_wing_texture": "body",
    "rear_wing_texture": "body",
    "material_map_blue": "carbon",
    "material_map_green": "body",
    "material_map_orange": "accent",
    "material_map_red": "accent",
    "material_map_yellow": "dark",
}

#: The game's collision box is 4.45 x 2.41 m -- proportionally wider than a
#: real Formula 1 car, whose 5.6 x 2.0 m the model follows. Fitting the length
#: (so the bodywork stays inside the box the walls are tested against) then
#: leaves the wheels standing well proud of the front wing, which reads as a
#: narrow car on stilts. Stretching the width alone puts the wing tips back
#: where they belong; nothing but the mesh is affected.
WIDTH_STRETCH = 1.18

#: Livery key -> asset name. "rb" is the model's own designation (fp04rb),
#: kept distinct from the procedural f1red / f1white in f1car.py.
ASSET_NAME = {"f1red": "rb_red", "f1white": "rb_white"}


def _colours(livery: str):
    liv = f1car.LIVERIES[livery]
    return {"body": liv["body"], "accent": liv["accent"],
            "carbon": f1car.CARBON, "dark": f1car.CARBON_LT}


# --- obj ---------------------------------------------------------------
def load_obj(path: Path):
    """(positions, normals, {material: (quad v-indices, quad n-indices)})."""
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
                if len(vi) == 3:
                    vi.append(vi[2])
                    ni.append(ni[2])
                if len(vi) == 4:
                    gv.setdefault(cur, []).append(vi)
                    gn.setdefault(cur, []).append(ni)
    if not ns:
        ns = [(0.0, 1.0, 0.0)]
    return (np.asarray(vs, dtype=np.float64), np.asarray(ns, dtype=np.float64),
            {k: (np.asarray(gv[k], dtype=np.int64),
                 np.asarray(gn[k], dtype=np.int64)) for k in gv})


# --- decimation --------------------------------------------------------
def cluster(verts, norms, faces, cell):
    """Weld face corners onto a grid, keyed by position *and* normal.

    Position alone is not enough. A wing or a floor is a sheet a few
    millimetres thick, so any cell coarse enough to decimate it also welds its
    upper surface to its lower one -- the panel collapses and the mesh tears
    into confetti. Folding a coarse normal bucket into the key keeps the two
    sides apart, so a thin panel thins its detail instead of losing its shape.
    """
    mats = list(faces)
    corners_v = np.concatenate([faces[m][0].ravel() for m in mats])
    corners_n = np.concatenate([faces[m][1].ravel() for m in mats])
    pos = verts[corners_v]

    grid = np.floor(pos / cell).astype(np.int64)
    grid -= grid.min(axis=0)
    dims = grid.max(axis=0) + 1
    nb = np.clip(np.round(norms[corners_n] * 1.2), -1, 1).astype(np.int64) + 1
    code = (nb[:, 0] * 3 + nb[:, 1]) * 3 + nb[:, 2]          # 0..26
    lin = ((grid[:, 0] * dims[1] + grid[:, 1]) * dims[2] + grid[:, 2]) * 27 + code

    _, inv = np.unique(lin, return_inverse=True)
    inv = np.asarray(inv).ravel()
    n = int(inv.max()) + 1
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    rep = np.column_stack([np.bincount(inv, weights=pos[:, k], minlength=n) / cnt
                           for k in range(3)])

    out, at = {}, 0
    for m in mats:
        k = faces[m][0].size
        q = inv[at:at + k].reshape(-1, 4)
        at += k
        tris = np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
        good = ((tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2])
                & (tris[:, 0] != tris[:, 2]))
        tris = tris[good]
        if len(tris):
            out[m] = tris
    return rep, out


def decimate(verts, norms, faces, budget):
    """Bisect the grid cell until the triangle count lands under *budget*."""
    span = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    lo, hi = span * 3e-4, span * 0.4
    best = None
    for _ in range(14):
        cell = float(np.sqrt(lo * hi))
        rep, tris = cluster(verts, norms, faces, cell)
        total = sum(len(t) for t in tris.values())
        if total > budget:
            lo = cell
        else:
            best = (rep, tris, total, cell)
            hi = cell
        if hi / lo < 1.05:
            break
    if best is None:
        cell = span * 0.4
        rep, tris = cluster(verts, norms, faces, cell)
        best = (rep, tris, sum(len(t) for t in tris.values()), cell)
    return best


# --- panda geometry ----------------------------------------------------
def geom_node(name, v, n, c):
    fmt = GeomVertexFormat.get_v3n3c4()
    vdata = GeomVertexData(name, fmt, Geom.UH_static)
    vdata.set_num_rows(len(v))
    wv = GeomVertexWriter(vdata, "vertex")
    wn = GeomVertexWriter(vdata, "normal")
    wc = GeomVertexWriter(vdata, "color")
    for p, q, col in zip(v, n, c):
        wv.add_data3(float(p[0]), float(p[1]), float(p[2]))
        wn.add_data3(float(q[0]), float(q[1]), float(q[2]))
        wc.add_data4(float(col[0]), float(col[1]), float(col[2]), float(col[3]))
    prim = GeomTriangles(Geom.UH_static)
    prim.add_next_vertices(len(v))
    prim.close_primitive()
    geom = Geom(vdata)
    geom.add_primitive(prim)
    node = GeomNode(name)
    node.add_geom(geom)
    return node


def flat(verts, tris, colour):
    """Expand indexed triangles into flat-shaded vertices plus normals.

    The corners come out **reversed** relative to the source, while the normal
    keeps the direction the source winding implied. See ``expand`` in
    build_blender_f1.py for the measurement: this renderer's front face is the
    one whose winding runs *against* its outward normal, so emitting an OBJ's
    own order leaves every face back-facing and the model renders inside out.
    """
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    nrm = np.cross(b - a, c - a)
    L = np.linalg.norm(nrm, axis=1)
    keep = L > 1e-12
    a, b, c, nrm, L = a[keep], b[keep], c[keep], nrm[keep], L[keep]
    if not len(a):
        return np.zeros((0, 3)), np.zeros((0, 3)), []
    nrm = nrm / L[:, None]
    v = np.empty((len(a) * 3, 3))
    v[0::3], v[1::3], v[2::3] = a, c, b       # reversed winding
    n = np.repeat(nrm, 3, axis=0)             # normal left as the source had it
    return v, n, [colour] * len(v)


# --- build -------------------------------------------------------------
def build(livery: str, scale_budget: float):
    cols = _colours(livery)
    raw = {}
    lo = np.full(3, 1e30)
    hi = np.full(3, -1e30)
    for p in sorted(SRC.glob("*.obj")):
        verts, norms, faces = load_obj(p)
        raw[p.stem] = (verts, norms, faces)
        lo = np.minimum(lo, verts.min(0))
        hi = np.maximum(hi, verts.max(0))

    # Model space is +x forward, +y up, +z across. Game space is +z forward.
    # game = (-z, y, x) * s, then slid so the nose lands on BODY_TO_FRONT.
    s = config.CAR_BODY_LENGTH / float(hi[0] - lo[0])
    dz = config.BODY_TO_FRONT - float(hi[0]) * s

    root = NodePath("body")
    total = 0
    for stem, (verts, norms, faces) in raw.items():
        budget = int(BUDGET.get(stem, 3000) * scale_budget)
        rep, tris, kept, cell = decimate(verts, norms, faces, budget)
        g = np.column_stack([-rep[:, 2] * s * WIDTH_STRETCH,
                             rep[:, 1] * s, rep[:, 0] * s + dz])
        for mat, tri in tris.items():
            v, n, c = flat(g, tri, cols[ROLE.get(mat, "carbon")])
            if not len(v):
                continue
            root.attach_new_node(geom_node(f"{stem}__{mat}", v, n, c))
            total += len(v) // 3
        print(f"  {stem:24s} {sum(len(f[0]) for f in faces.values()):>7} quads"
              f" -> {kept:>6} tris   cell {cell:.4f}")

    for nm, x, z, w in (("wheelFrontLeft", -f1car.TRACK, f1car.ZF, f1car.W_FRONT),
                        ("wheelFrontRight", f1car.TRACK, f1car.ZF, f1car.W_FRONT),
                        ("wheelBackLeft", -f1car.TRACK, f1car.ZR, f1car.W_REAR),
                        ("wheelBackRight", f1car.TRACK, f1car.ZR, f1car.W_REAR)):
        part = f1car._wheel_part(w)
        holder = root.attach_new_node(nm)
        holder.set_pos(x, f1car.R_WHEEL, z)
        holder.attach_new_node(geom_node(nm + "_geom", part.v, part.n, part.c))
        total += len(part.v) // 3

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{ASSET_NAME[livery]}.bam"
    if not root.write_bam_file(Filename.from_os_specific(str(dest))):
        raise SystemExit(f'could not write {dest}')
    w = float(hi[2] - lo[2]) * s * WIDTH_STRETCH
    h = float(hi[1] - lo[1]) * s
    ln = float(hi[0] - lo[0]) * s
    print(f"  fitted  w {w:.3f}  h {h:.3f}  l {ln:.3f} m   "
          f"(wheel track {2 * f1car.TRACK:.3f} m)")
    print(f"  -> assets/models/f1/{dest.name}   {total:,} tris   "
          f"{dest.stat().st_size / 1e6:.1f} MB")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1.0,
                    help="multiplier on every part's triangle budget")
    ap.add_argument("--livery", default=None)
    args = ap.parse_args()
    for liv in ([args.livery] if args.livery else list(f1car.LIVERIES)):
        t = time.time()
        print(f"{liv}:")
        build(liv, args.budget)
        print(f"  {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
