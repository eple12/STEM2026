"""Bake the modelled race car (``assets/for+race.blend``) into a game-ready .bam.

    blender -b ai_sw/assets/for+race.blend -P blender/export_race_car.py -- \
        --out ai_sw/assets/blender_race/car_full.npz            # as modelled
    blender -b ai_sw/assets/for+race.blend -P blender/export_race_car.py -- \
        --out ai_sw/assets/blender_race/car_lo.npz --ratio 0.06 --floor 1500  # 91k
    blender -b ai_sw/assets/for+race.blend -P blender/export_race_car.py -- \n        --out ai_sw/assets/blender_race/car_far.npz --ratio 0.015 --floor 200  # 17k

    python tools/build_blender_race.py                          # rc_red, rc_white
    python tools/build_blender_race.py --src car_lo --prefix rl # rl_red, rl_white
    python tools/build_blender_race.py --src car_far --prefix rl --suffix _lod   # far LOD (17k)

The sources are the parts blender/export_race_car.py writes: the car's world-space
triangles with the authored corner normals, split into ``body`` and one part per
wheel. This does what build_blender_f1.py does for the procedural car, for the
same reasons, with the differences that follow from this model:

* **Re-frame.** Blender is +z up with the nose towards -y; the game is +y up,
  +z forward, +x right. ``(x, y, z) -> (x, z, -y)`` is a proper rotation
  (determinant +1), so -- exactly as in build_blender_f1.py -- the corner order
  is then reversed to give the renderer the faces it wants.

* **Fit.** The bodywork is fitted to the collision box -- length first, then
  the width stretched to the box width -- and stood up ``--height`` (1.25) times
  taller than the length scale, because at true proportions a car stretched
  sideways to a box this wide reads as flat. The wheels are *not* stretched: a
  tyre 1.5 times too wide is the first thing that makes a car look squat. Each
  keeps one uniform scale (the height factor's, so it stays round and grows with
  the car) and only its hub is carried to where the stretched bodywork expects
  it, so every wishbone still lands on its wheel. The box is what the walls are
  tested against, so the car keeps the footprint of the one it replaces.

* **Recolour.** Blender material -> (surface tag, paint role). The tag goes in
  the geom's name, which is how ``Car.apply_materials`` picks the surface
  (``body__Livery``, ``FL__Rubber``...) -- so there is one geom per part and
  surface, not per Blender material, which keeps the draw calls down. The role
  picks the flat vertex colour. The model's own colours are not used: its
  materials are all default grey.

* **Weld.** The sources are corner streams (3 vertices per triangle). Corners
  that share position, normal and colour are merged, which is what makes a
  smooth mesh a fraction of the size in memory.
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

SRC_DIR = config.ASSET_DIR / "blender_race"
OUT = config.ASSET_DIR / "models" / "f1"

#: Blender material name -> (surface tag Car.apply_materials looks for, paint role).
MATERIAL = {
    "Material.002": ("Livery", "body"),      # the monocoque and engine cover
    "Livery": ("Livery", "body"),
    "carbon": ("Carbon", "carbon"),          # floor, wings, sidepods
    "generic": ("Carbon", "dark"),           # wishbones, pushrods
    "chrome": ("Interior", "accent"),        # the thin trim lines on the wings
    "tyres": ("Rubber", "tyre"),
    "rims": ("Metal", "rim"),
    "brake": ("Metal", "rim"),
    "exhaust": ("Metal", "rim"),
    "mirror": ("Metal", "rim"),
    "Steer": ("Interior", "dark"),
    "Glass": ("Interior", "dark"),
}
ROLES = ("body", "accent", "carbon", "dark", "tyre", "rim")

#: Wheel part -> the node name Car._rig_wheels looks for.
WHEEL_NODE = {"FL": "wheelFrontLeft", "FR": "wheelFrontRight",
              "RL": "wheelBackLeft", "RR": "wheelBackRight"}

ASSET_NAME = {"f1red": "{p}_red{x}", "f1white": "{p}_white{x}"}


def _colours(livery: str):
    liv = f1car.LIVERIES[livery]
    return {"body": liv["body"], "accent": liv["accent"],
            "carbon": f1car.CARBON, "dark": f1car.CARBON_LT,
            "tyre": f1car.TYRE, "rim": f1car.RIM}


def body_to_game(p, k, ref):
    """Blender (nose at -y, +z up) -> game (+x right, +y up, +z forward), the
    bodywork's own scales *k* = (across, up, along) about *ref* = (centre x,
    ground z, nose y)."""
    return np.column_stack([(p[:, 0] - ref[0]) * k[0],
                            (p[:, 2] - ref[1]) * k[1],
                            config.BODY_TO_FRONT + (ref[2] - p[:, 1]) * k[2]])


def normals_to_game(n, k):
    """Same permutation; each axis is divided by its scale so a normal stays
    perpendicular to the surface it came from (inverse transpose)."""
    g = np.column_stack([n[:, 0] / k[0], n[:, 2] / k[1], -n[:, 1] / k[2]])
    L = np.linalg.norm(g, axis=1)
    return g / np.where(L > 1e-12, L, 1.0)[:, None]


def weld(pos, nrm, role):
    """Corner stream -> (unique positions, normals, roles, triangle indices).

    Keyed on position to 0.01 mm, normal to 0.001 and paint role, so only
    corners that are the same point, shade the same way and are painted the
    same are merged: a crease stays a crease.
    """
    key = np.empty((len(pos), 7), dtype=np.int32)
    key[:, :3] = np.round(pos * 1e5)
    key[:, 3:6] = np.round(nrm * 1e3)
    key[:, 6] = role
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).ravel()
    return pos[first], nrm[first], role[first], inv.reshape(-1, 3)


def geom_node(name, v, n, c, tris):
    vdata = GeomVertexData(name, GeomVertexFormat.get_v3n3c4(), Geom.UH_static)
    vdata.set_num_rows(len(v))
    wv = GeomVertexWriter(vdata, "vertex")
    wn = GeomVertexWriter(vdata, "normal")
    wc = GeomVertexWriter(vdata, "color")
    for p, q, col in zip(v.tolist(), n.tolist(), c.tolist()):
        wv.add_data3(*p)
        wn.add_data3(*q)
        wc.add_data4(*col)
    prim = GeomTriangles(Geom.UH_static)
    for a, b, c3 in tris.tolist():
        prim.add_vertices(a, b, c3)
    prim.close_primitive()
    geom = Geom(vdata)
    geom.add_primitive(prim)
    node = GeomNode(name)
    node.add_geom(geom)
    return node


def tri_materials(src, mats, key):
    """The Blender material name of every triangle of a part."""
    return np.array(mats)[src[f"{key}_mat"]]


def build(livery: str, src, prefix: str, length_fit: float, width_fit: float,
          height_fit: float, suffix: str = ""):
    cols = _colours(livery)
    colour_of = np.array([cols[r] for r in ROLES], dtype=np.float64)
    mats = [str(m) for m in src["materials"]]
    parts = ["body"] + list(WHEEL_NODE)

    allp = np.vstack([src[f"{k}_pos"] for k in parts])
    lo, hi = allp.min(0), allp.max(0)
    model_len = float(hi[1] - lo[1])          # nose to tail is along y
    model_wid = float(hi[0] - lo[0])
    s = config.CAR_BODY_LENGTH * length_fit / model_len
    stretch = config.CAR_BODY_WIDTH * width_fit / (model_wid * s)
    # Bodywork: along the box length, across the box width, up by height_fit.
    # Wheels keep their own proportions: one uniform scale (the same factor
    # that lifts the bodywork, so they stay round and grow with the car), with
    # only the hub carried to where the stretched bodywork expects it.
    k = (s * stretch, s * height_fit, s)
    kw = s * height_fit

    tyres = {n: src[f"{n}_pos"][np.repeat(tri_materials(src, mats, n) == "tyres", 3)]
             for n in WHEEL_NODE}
    ground = min(float(t[:, 2].min()) for t in tyres.values())
    ref = ((lo[0] + hi[0]) / 2, ground, lo[1])
    hubs = {}
    for n, t in tyres.items():
        c = (t.min(0) + t.max(0)) / 2
        hub = body_to_game(c[None, :], k, ref)[0]
        radius = float(t[:, 2].max() - t[:, 2].min()) / 2
        hub[1] = radius * kw                  # tyre exactly on the ground
        hubs[n] = (c, hub, radius * kw)

    root = NodePath("body")
    total = 0
    for key in parts:
        pos = src[f"{key}_pos"].astype(np.float64)
        nrm = src[f"{key}_nrm"].astype(np.float64)
        if key == "body":
            g = body_to_game(pos, k, ref)
            gn = normals_to_game(nrm, k)
        else:
            c, hub, _r = hubs[key]
            d = pos - c
            g = hub + np.column_stack([d[:, 0], d[:, 2], -d[:, 1]]) * kw
            gn = normals_to_game(nrm, (1.0, 1.0, 1.0))
        names = tri_materials(src, mats, key)
        for n in sorted(set(names.tolist()) - set(MATERIAL)):
            print(f"  ! unmapped material {n!r} -> carbon")
        tag = np.array([MATERIAL.get(n, ("Carbon", "carbon"))[0] for n in names])
        role = np.array([ROLES.index(MATERIAL.get(n, ("Carbon", "carbon"))[1])
                         for n in names])
        parent = root if key == "body" else root.attach_new_node(WHEEL_NODE[key])
        for t in sorted(set(tag.tolist())):
            sel = np.flatnonzero(tag == t)
            corners = (sel[:, None] * 3 + np.arange(3)[None, :]).reshape(-1)
            v, n, r, tris = weld(g[corners], gn[corners], np.repeat(role[sel], 3))
            # reverse the corner order: see build_blender_f1.expand
            front = np.ascontiguousarray(tris[:, ::-1])
            if key != "body":
                # The wheel assembly is modelled single-sided: a tyre whose
                # inner wall is missing, a cover with no back. Seen from the
                # outside that is invisible; seen from between the wheels --
                # which is where the camera goes in a spin or an onboard --
                # the faces are culled and the road shows through the wheel.
                # So each wheel face is baked twice, the second with its
                # winding and normal reversed: from inside you get the dark
                # back of the wheel, as a real one has.
                nv = len(v)
                tris = np.vstack([front, np.ascontiguousarray(tris + nv)])
                v, n, r = np.vstack([v, v]), np.vstack([n, -n]), np.concatenate([r, r])
            else:
                tris = front
            parent.attach_new_node(geom_node(f"{key}__{t}", v, n, colour_of[r], tris))
            total += len(tris)

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{ASSET_NAME[livery].format(p=prefix, x=suffix)}.bam"
    if not root.write_bam_file(Filename.from_os_specific(str(dest))):
        raise SystemExit(f"could not write {dest}")

    track = abs(hubs["FL"][1][0])
    wb = abs(hubs["RL"][1][2] - hubs["FL"][1][2])
    print(f"  fit     length scale {s:.4f}   width x{stretch:.3f}   height x{height_fit:.3f}")
    print(f"  fitted  l {model_len * s:.3f}  w {model_wid * s * stretch:.3f}  "
          f"h {(hi[2] - ground) * k[1]:.3f}  (box {config.CAR_BODY_LENGTH} x "
          f"{config.CAR_BODY_WIDTH} m)")
    print(f"  wheels  radius {hubs['FL'][2]:.3f} front / {hubs['RL'][2]:.3f} rear "
          f"(physics {0.33})   half track {track:.3f} m (physics "
          f"{config.WHEEL_HALF_TRACK:.3f})   wheelbase {wb:.3f} m "
          f"(physics {config.WHEELBASE:.3f})")
    print(f"  -> assets/models/f1/{dest.name}   {total:,} tris   "
          f"{dest.stat().st_size / 1e6:.2f} MB")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="car_full",
                    help="npz under assets/blender_race (car_full, car_lo)")
    ap.add_argument("--prefix", default="rc",
                    help="asset name prefix: <prefix>_red.bam / <prefix>_white.bam")
    ap.add_argument("--suffix", default="",
                    help="appended to the asset name: _lod for the far-LOD mesh "
                         "(<prefix>_red_lod.bam, which game/gpfield.py looks for)")
    ap.add_argument("--livery", default=None)
    ap.add_argument("--length", type=float, default=1.0,
                    help="multiplier on the fitted length (1.0 = collision box)")
    ap.add_argument("--width", type=float, default=1.0,
                    help="multiplier on the fitted width (1.0 = collision box)")
    ap.add_argument("--height", type=float, default=1.25,
                    help="how much taller than the length scale the bodywork "
                         "stands; the wheels grow with it (1.0 = true proportions)")
    args = ap.parse_args()
    src = np.load(SRC_DIR / f"{args.src}.npz")
    for liv in ([args.livery] if args.livery else list(f1car.LIVERIES)):
        t = time.time()
        print(f"{liv}:")
        build(liv, src, args.prefix, args.length, args.width, args.height,
              args.suffix)
        print(f"  {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
