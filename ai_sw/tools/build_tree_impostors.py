"""Bake photoscanned trees into impostor cards for the roadside forest.

    python tools/build_tree_impostors.py           # render in Blender, then pack
    python tools/build_tree_impostors.py --pack    # re-pack existing renders

Source: Poly Haven (CC0) glTFs in ``assets/vendor/polyhaven/`` -- millions of
triangles each, which no game frame could draw thousands of. Each tree is
rendered once, offline, from two sides and from above, orthographic, on a
transparent background, under a soft overcast sky (so the image carries the
canopy's own occlusion but no sun -- the game lights it). The views are packed
into ``assets/foliage/tree_impostors.png`` with their sizes in metres in
``tree_impostors.json``, which is all ``game/foliage.py`` reads. The vendor
files are gitignored; the packed atlas is what is committed.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets" / "vendor" / "polyhaven"
RENDERS = ROOT / "assets" / "vendor" / "impostor_renders"
OUT = ROOT / "assets" / "foliage"
BLENDER = Path("D:/Program Files/Blender/blender.exe")

#: (glTF folder, mesh object name) -> impostor id.
TREES = {
    "island_a": ("island_tree_01", "island_tree_01_LOD0"),
    "island_b": ("island_tree_02", "island_tree_02_LOD0"),
    "fir_a": ("fir_tree_01", "fir_tree_01_a_LOD0"),
    "fir_b": ("fir_tree_01", "fir_tree_01_b_LOD0"),
    "fir_c": ("fir_tree_01", "fir_tree_01_c_LOD0"),
}
SIDE_PX = 1024          # pixel height of a side view
TOP_PX = 512


# ---------------------------------------------------------------- in Blender
def _blender_main():
    import bpy
    from mathutils import Vector

    RENDERS.mkdir(parents=True, exist_ok=True)
    meta = {}
    loaded = None
    for tid, (folder, objname) in TREES.items():
        if loaded != folder:
            bpy.ops.wm.read_factory_settings(use_empty=True)
            bpy.ops.import_scene.gltf(
                filepath=str(SRC / folder / f"{folder}_1k.gltf"))
            loaded = folder
            scene = bpy.context.scene
            scene.render.engine = "CYCLES"
            scene.cycles.samples = 48
            scene.cycles.use_denoising = True
            scene.cycles.device = "CPU"
            scene.render.film_transparent = True
            scene.view_settings.view_transform = "Standard"
            scene.view_settings.look = "None"
            scene.render.image_settings.file_format = "PNG"
            scene.render.image_settings.color_mode = "RGBA"
            world = bpy.data.worlds.new("sky")
            world.use_nodes = True
            bg = world.node_tree.nodes["Background"]
            # Overcast: light from everywhere, a little more from above.
            sky = world.node_tree.nodes.new("ShaderNodeTexGradient")
            ramp = world.node_tree.nodes.new("ShaderNodeValToRGB")
            coord = world.node_tree.nodes.new("ShaderNodeTexCoord")
            sep = world.node_tree.nodes.new("ShaderNodeSeparateXYZ")
            world.node_tree.links.new(coord.outputs["Generated"], sep.inputs[0])
            world.node_tree.links.new(sep.outputs["Z"], ramp.inputs[0])
            ramp.color_ramp.elements[0].position = 0.45
            ramp.color_ramp.elements[0].color = (0.55, 0.55, 0.55, 1)
            ramp.color_ramp.elements[1].position = 0.75
            ramp.color_ramp.elements[1].color = (1.25, 1.27, 1.30, 1)
            world.node_tree.links.new(ramp.outputs["Color"], bg.inputs["Color"])
            bg.inputs["Strength"].default_value = 1.0
            scene.world = world
            del sky
        scene = bpy.context.scene
        obj = bpy.data.objects[objname]
        for o in scene.objects:
            o.hide_render = (o.type == "MESH" and o is not obj)
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        lo = Vector((min(c.x for c in corners), min(c.y for c in corners),
                     min(c.z for c in corners)))
        hi = Vector((max(c.x for c in corners), max(c.y for c in corners),
                     max(c.z for c in corners)))
        ctr = (lo + hi) / 2
        size = hi - lo
        # Square horizontal extent, so both side views share a width and the
        # tree is centred on its own trunk axis in each.
        half_w = max(size.x, size.y) / 2 * 1.02
        h = size.z * 1.01
        cam_data = bpy.data.cameras.new("cam")
        cam_data.type = "ORTHO"
        cam = bpy.data.objects.new("cam", cam_data)
        scene.collection.objects.link(cam)
        scene.camera = cam
        views = {}
        for view, yaw in (("side0", 0.0), ("side90", 90.0)):
            a = math.radians(yaw)
            # A camera rotated (90, 0, a) looks along +y turned by a.
            f = Vector((-math.sin(a), math.cos(a), 0.0))
            cam.location = Vector((ctr.x, ctr.y, lo.z + h / 2)) - f * 100.0
            cam.rotation_euler = (math.radians(90.0), 0.0, a)
            cam_data.ortho_scale = max(2 * half_w, h)
            cam_data.clip_end = 300.0
            if h >= 2 * half_w:
                rx, ry = int(round(SIDE_PX * 2 * half_w / h)), SIDE_PX
            else:
                rx, ry = SIDE_PX, int(round(SIDE_PX * h / (2 * half_w)))
            scene.render.resolution_x, scene.render.resolution_y = rx, ry
            path = RENDERS / f"{tid}_{view}.png"
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            views[view] = path.name
        cam.location = Vector((ctr.x, ctr.y, hi.z + 50.0))
        cam.rotation_euler = (0.0, 0.0, 0.0)
        cam_data.ortho_scale = 2 * half_w
        scene.render.resolution_x = TOP_PX
        scene.render.resolution_y = TOP_PX
        path = RENDERS / f"{tid}_top.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        views["top"] = path.name
        bpy.data.objects.remove(cam)
        meta[tid] = dict(width=2 * half_w, height=h, views=views)
        print("baked", tid, round(2 * half_w, 2), round(h, 2), flush=True)
    (RENDERS / "meta.json").write_text(json.dumps(meta, indent=1))


# ---------------------------------------------------------------- packing
def _bleed(im):
    """Fill transparent texels with nearby leaf colour, so the mipmaps do
    not fringe every card in the background colour."""
    import numpy as np
    from PIL import Image, ImageFilter
    a = np.asarray(im).astype(np.float64)
    alpha = a[..., 3:4] / 255.0
    rgb = a[..., :3]
    acc = np.zeros_like(rgb)
    wsum = np.zeros_like(alpha)
    cur_rgb = rgb * alpha
    cur_a = alpha.copy()
    for r in (2, 6, 16, 40):
        pre = Image.fromarray(np.clip(cur_rgb, 0, 255).astype(np.uint8))
        pa = Image.fromarray(np.clip(cur_a[..., 0] * 255, 0, 255).astype(np.uint8))
        brgb = np.asarray(pre.filter(ImageFilter.BoxBlur(r))).astype(np.float64)
        ba = np.asarray(pa.filter(ImageFilter.BoxBlur(r))).astype(np.float64)[..., None] / 255.0
        fill = brgb / np.maximum(ba, 1e-4)
        need = (wsum < 1e-3) & (ba > 0.02)
        acc = np.where(need, fill, acc)
        wsum = np.where(need, 1.0, wsum)
    out = np.where(alpha > 0.0, rgb / np.maximum(alpha, 1e-4) * np.minimum(alpha * 4, 1)
                   + acc * (1 - np.minimum(alpha * 4, 1)), acc)
    out = np.clip(out, 0, 255)
    return Image.fromarray(np.concatenate([out, a[..., 3:4]], axis=-1).astype(np.uint8), "RGBA")


def pack():
    from PIL import Image
    meta = json.loads((RENDERS / "meta.json").read_text())
    tiles = []
    for tid, m in meta.items():
        for view, fname in m["views"].items():
            im = Image.open(RENDERS / fname).convert("RGBA")
            # Halve the side views: 512 px tall is plenty at roadside range.
            scale = 0.5 if view != "top" else 0.5
            im = im.resize((max(1, int(im.width * scale)),
                            max(1, int(im.height * scale))), Image.LANCZOS)
            tiles.append((tid, view, _bleed(im)))
    # Shelf packing into a 2048-wide sheet, tallest first.
    W = 2048
    tiles.sort(key=lambda t: -t[2].height)
    x = y = shelf = 0
    pos = {}
    pad = 4
    for tid, view, im in tiles:
        if x + im.width > W:
            x, y = 0, y + shelf + pad
            shelf = 0
        pos[(tid, view)] = (x, y)
        x += im.width + pad
        shelf = max(shelf, im.height)
    H = 1
    while H < y + shelf:
        H *= 2
    sheet = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out = {}
    for tid, view, im in tiles:
        px, py = pos[(tid, view)]
        sheet.paste(im, (px, py))
        # UVs: V = 0 at the bottom of the sheet.
        u0, u1 = (px + 0.5) / W, (px + im.width - 0.5) / W
        v1 = 1.0 - (py + 0.5) / H
        v0 = 1.0 - (py + im.height - 0.5) / H
        out.setdefault(tid, dict(width=meta[tid]["width"],
                                 height=meta[tid]["height"], uv={}))
        out[tid]["uv"][view] = [u0, v0, u1, v1]
    OUT.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT / "tree_impostors.png", optimize=True)
    (OUT / "tree_impostors.json").write_text(json.dumps(out, indent=1))
    (OUT / "LICENSE.txt").write_text(
        "tree_impostors.png is rendered from Poly Haven models (CC0 1.0):\n"
        "island_tree_01, island_tree_02, fir_tree_01 -- https://polyhaven.com\n")
    print("packed", sheet.size, "->", OUT / "tree_impostors.png")


if __name__ == "__main__":
    if "--" in sys.argv and "--blender" in sys.argv:
        _blender_main()
    elif "--pack" in sys.argv:
        pack()
    else:
        subprocess.run([str(BLENDER), "-b", "-P", str(Path(__file__).resolve()),
                        "--", "--blender"], check=True)
        pack()
