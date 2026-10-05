"""Pull the car out of ``ai_sw/assets/for+race.blend`` for tools/build_blender_race.py.

    blender -b ai_sw/assets/for+race.blend -P blender/export_race_car.py -- \
        --out ai_sw/assets/blender_race/car_full.npz [--ratio 0.15]

The .blend is the car plus a studio set (camera, a 40 m floor plane, turntable
discs, a cylinder). Only the car is exported: every mesh under the ``GG`` empty,
which carries a -90 degree turn about X, so everything is written in *world*
space -- Blender's own axes (+z up, the nose towards -y, 1 unit = 1 m), with the
car standing on z = 1.0 (the empty lifts it 1 m; the tyres touch there).

Each of the five parts comes out as a flat triangle-corner stream -- positions,
the authored (smooth/flat) corner normals, a material index per triangle -- so
the baker needs no Blender and no OBJ parsing:

* ``body``          everything that does not turn with a wheel
* ``FL FR RL RR``   tyre, rim, brake disc and the outer wheel cover of one wheel

The brake-duct drum behind each front wheel (Object_27 / Object_31) stays on the
body: it steers with the wheel but must not spin with it.

``--ratio`` runs Blender's collapse decimator over every object first (never
leaving one below ``--floor`` triangles, so a tyre stays round); at 1.0 the car
is taken as modelled, about 894k triangles.
"""
import argparse
import sys

import bpy
import numpy as np

#: The studio set, not the car.
SET_DRESSING = {"Camera", "Circle", "Circle.001", "Cylinder", "Plane", "Plane.001"}
#: Objects whose material says they turn with a wheel ...
WHEEL_MATERIALS = {"tyres", "rims", "brake"}
#: ... and the one carbon part that does too: the four outer wheel covers.
WHEEL_OBJECTS = {"Object_16"}
WHEELS = ("FL", "FR", "RL", "RR")


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--floor", type=int, default=2500,
                    help="decimate no object below this many triangles")
    return ap.parse_args(sys.argv[sys.argv.index("--") + 1:])


def world_tris(obj, ratio, floor, dg):
    """(corner positions, corner normals, per-triangle material name)."""
    if ratio < 0.999:
        n = len(obj.data.polygons)
        r = max(ratio, min(1.0, floor / max(n, 1)))
        if r < 0.999:
            mod = obj.modifiers.new("dec", "DECIMATE")
            mod.decimate_type = "COLLAPSE"
            mod.ratio = r
            mod.use_collapse_triangulate = True
            bpy.context.view_layer.update()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    nv = len(me.vertices)
    co = np.empty(nv * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    tv = np.empty(nt * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", tv)
    tl = np.empty(nt * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("loops", tl)
    mi = np.empty(nt, dtype=np.int64)
    me.loop_triangles.foreach_get("material_index", mi)
    cn = np.empty(len(me.corner_normals) * 3)
    me.corner_normals.foreach_get("vector", cn)
    cn = cn.reshape(-1, 3)

    M = np.array(obj.matrix_world)
    R = M[:3, :3]
    pos = (co[tv] @ R.T) + M[:3, 3]
    nrm = cn[tl] @ np.linalg.inv(R)          # inverse transpose, row-vector form
    L = np.linalg.norm(nrm, axis=1)
    nrm = nrm / np.where(L > 1e-12, L, 1.0)[:, None]
    mats = [m.name if m else "none" for m in obj.data.materials]
    names = np.array([mats[min(i, len(mats) - 1)] for i in mi])
    ev.to_mesh_clear()
    return pos.astype(np.float32), nrm.astype(np.float32), names


def main():
    a = args()
    dg = bpy.context.evaluated_depsgraph_get()
    meshes = [o for o in bpy.data.objects
              if o.type == "MESH" and o.name not in SET_DRESSING]

    # Wheel centres from the four tyres' bounding boxes.
    centres = {}
    for o in meshes:
        if o.data.materials and o.data.materials[0].name == "tyres":
            w = np.array([o.matrix_world @ v.co for v in o.data.vertices])
            c = (w.min(0) + w.max(0)) / 2
            tag = ("F" if c[1] < 0 else "R") + ("L" if c[0] < 0 else "R")
            centres[tag] = c
    assert sorted(centres) == sorted(WHEELS), centres
    cen = np.array([centres[t] for t in WHEELS])

    parts = {k: ([], [], []) for k in ("body",) + WHEELS}
    total = before = 0
    for o in meshes:
        before += sum(len(p.vertices) - 2 for p in o.data.polygons)
        pos, nrm, names = world_tris(o, a.ratio, a.floor, dg)
        pos = pos.reshape(-1, 3, 3)
        nrm = nrm.reshape(-1, 3, 3)
        mat0 = o.data.materials[0].name if o.data.materials else ""
        if mat0 in WHEEL_MATERIALS or o.name in WHEEL_OBJECTS:
            # each triangle goes to the nearest wheel by its centroid
            cent = pos.mean(1)
            d = np.linalg.norm(cent[:, None, :] - cen[None, :, :], axis=2)
            owner = d.argmin(1)
            dest = [WHEELS[i] for i in range(4)]
            for i, key in enumerate(dest):
                sel = owner == i
                parts[key][0].append(pos[sel])
                parts[key][1].append(nrm[sel])
                parts[key][2].append(names[sel])
        else:
            parts["body"][0].append(pos)
            parts["body"][1].append(nrm)
            parts["body"][2].append(names)
        total += len(names)
        print(f"  {o.name:14s} {mat0:13s} {len(names):>7} tris")

    out = {}
    allmats = sorted({n for p in parts.values() for arr in p[2] for n in set(arr.tolist())})
    out["materials"] = np.array(allmats)
    for key, (ps, ns, ms) in parts.items():
        pos = np.concatenate(ps).reshape(-1, 3)
        nrm = np.concatenate(ns).reshape(-1, 3)
        mi = np.array([allmats.index(n) for n in np.concatenate(ms)], dtype=np.uint8)
        out[f"{key}_pos"] = pos
        out[f"{key}_nrm"] = nrm
        out[f"{key}_mat"] = mi
        print(f"PART {key:5s} {len(mi):>8} tris")
    import os
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **out)
    print(f"EXPORTED {total} tris (of {before}) -> {a.out}")


main()
