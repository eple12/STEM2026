"""Trees as textured cards: painted canopies on crossed quads.

The roadside forest used to be faceted low-poly models -- icosahedra on
sticks. Nothing else in the frame says "video game" as loudly: a real treeline
has a ragged, leafy silhouette with sky showing through it, and no flat facet
anywhere. Every racing game since the early 2000s draws distant foliage the
same way, and so does this: each tree is three vertical cards crossed at 60
degrees plus one flat card on top for the view from above, all cut out of a
painted canopy texture with an alpha test.

The atlas is painted here with numpy at load -- clumps of leaves shaded light
on top and dark underneath, layered back to front so the crown has depth,
with a trunk and the odd branch showing through below. No image files, no
licences, and it costs a fraction of a second once per process.

Geometry is cheaper than what it replaces: 14 triangles a tree, against the
hundreds a faceted model spent on facets nobody wanted. Batched per 400 m
cell, exactly as the old forest was, so the draw-call count is unchanged and
the cells still cull.
"""
from __future__ import annotations

import math

import numpy as np

from . import config
from .textures import _noise, panda_texture

CELL_W, CELL_H = 256, 512
COLS, ROWS = 4, 2

#: Atlas cell per canopy kind, and the card size in metres at scale 1.
#: (cell, width, height, height of the canopy centre as a fraction).
KINDS = {
    "round_a": (0, 9.5, 12.5, 0.60),
    "round_b": (1, 11.0, 12.0, 0.58),
    "pine_a": (2, 6.8, 15.5, 0.52),
    "pine_b": (3, 6.0, 17.0, 0.52),
    "spread": (4, 13.5, 11.5, 0.62),
    "cypress": (5, 3.8, 13.5, 0.55),
    "bush": (6, 7.0, 5.2, 0.48),
}
TOP_CELL = 7

#: Old prop names (scenery._forest) onto canopy kinds.
FROM_PROP = {
    "tree_round": ("round_a", "round_b"),
    "tree_pine": ("pine_a", "pine_b"),
    "tree_spread": ("spread",),
    "tree_cypress": ("cypress",),
    "tree_bush": ("bush",),
}

_ATLAS = None


# --- painting --------------------------------------------------------------
def _hf(rng, shape):
    return rng.random(shape)


def _disc(img, cx, cy, r, col, rng, rag=0.35, hole=0.10):
    """One clump of leaves: a ragged disc, lit from above, holes at its rim."""
    h, w = img.shape[:2]
    x0, x1 = max(0, int(cx - r - 1)), min(w, int(cx + r + 2))
    y0, y1 = max(0, int(cy - r - 1)), min(h, int(cy + r + 2))
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dx = (xs - cx) / r
    dy = (ys - cy) / r
    d = dx * dx + dy * dy
    n = rng.random(d.shape)
    edge = 1.0 - rag * n
    m = d < edge
    # Pin-holes near the rim, where light shows between leaves.
    m &= ~((d > 0.45) & (n < hole))
    if not m.any():
        return
    # Each clump is a little dome: brighter on its upper side.
    lit = np.clip(0.62 - 0.42 * dy - 0.12 * dx + 0.18 * (1.0 - d), 0.25, 1.15)
    leaf = 0.80 + 0.40 * rng.random(d.shape)
    c = np.asarray(col, dtype=np.float64)[None, None, :] * (lit * leaf)[..., None]
    reg = img[y0:y1, x0:x1]
    reg[..., :3][m] = c[m]
    reg[..., 3][m] = 255.0


def _trunk(img, cx, top, bottom, w0, col, rng, branches=3):
    h, w = img.shape[:2]
    for y in range(int(top), int(bottom)):
        t = (y - top) / max(bottom - top, 1)
        half = w0 * (0.45 + 0.55 * t) * 0.5
        x0, x1 = int(cx - half), int(cx + half + 1)
        xs = np.arange(max(0, x0), min(w, x1))
        if not len(xs):
            continue
        shade = 0.75 + 0.35 * (xs - cx) / max(half, 1) * -1.0
        shade = np.clip(shade, 0.45, 1.1) * (0.85 + 0.3 * rng.random(len(xs)))
        img[y, xs, :3] = np.asarray(col)[None, :] * shade[:, None]
        img[y, xs, 3] = 255.0
    for _ in range(branches):
        y = top + (bottom - top) * rng.uniform(0.0, 0.45)
        ln = rng.uniform(18, 40)
        ang = rng.uniform(0.5, 1.1) * rng.choice([-1, 1])
        for k in range(int(ln)):
            px = int(cx + math.sin(ang) * k)
            py = int(y - math.cos(ang) * k * 0.8)
            if 0 <= px < w and 0 <= py < h:
                img[py, max(0, px - 1):px + 2, :3] = np.asarray(col) * 0.8
                img[py, max(0, px - 1):px + 2, 3] = 255.0


def _canopy(kind, rng):
    W, H = CELL_W, CELL_H
    img = np.zeros((H, W, 4), np.float64)
    bark = (88, 74, 60)
    if kind in ("round_a", "round_b", "spread", "bush"):
        green = {"round_a": (74, 100, 42), "round_b": (66, 92, 40),
                 "spread": (80, 104, 46), "bush": (70, 96, 40)}[kind]
        cx = W * 0.5
        if kind == "bush":
            cy, rx, ry, n, rr = H * 0.70, W * 0.46, H * 0.20, 260, (10, 22)
            trunk = None
        elif kind == "spread":
            cy, rx, ry, n, rr = H * 0.36, W * 0.48, H * 0.25, 420, (12, 24)
            trunk = (cy + ry * 0.2, H, W * 0.07)
        else:
            k = 0.44 if kind == "round_a" else 0.47
            cy, rx, ry, n, rr = H * 0.40, W * k, H * 0.33, 460, (11, 22)
            trunk = (cy + ry * 0.3, H, W * 0.055)
        if trunk:
            _trunk(img, cx, trunk[0], trunk[1], trunk[2], bark, rng)
        pts = []
        while len(pts) < n:
            u, v = rng.uniform(-1, 1, 2)
            # Lumpier than an ellipse: a wobbly radius round the crown.
            a = math.atan2(v, u)
            wob = 1.0 + 0.10 * math.sin(3 * a + rng.uniform(0, 6)) \
                + 0.06 * math.sin(7 * a)
            if u * u + v * v < wob * wob:
                pts.append((cx + u * rx, cy + v * ry, rng.uniform(*rr)))
        # Back to front: the first clumps drawn are the deep, dark ones.
        order = sorted(range(n), key=lambda i: rng.random())
        for k, i in enumerate(order):
            x, y, r = pts[i]
            depth = k / n
            top = 1.0 - (y - (cy - ry)) / (2 * ry)
            shade = 0.50 + 0.45 * depth + 0.18 * top
            col = tuple(c * shade for c in green)
            _disc(img, x, y, r, col, rng)
    elif kind in ("pine_a", "pine_b", "cypress"):
        green = {"pine_a": (46, 70, 42), "pine_b": (40, 62, 40),
                 "cypress": (50, 74, 38)}[kind]
        cx = W * 0.5
        top_y, base_y = H * 0.03, H * (0.86 if kind != "cypress" else 0.92)
        _trunk(img, cx, H * 0.30, H, W * 0.05, bark, rng, branches=0)
        tiers = 16 if kind != "cypress" else 10
        for t in range(tiers):
            f = t / (tiers - 1)
            y = top_y + (base_y - top_y) * f
            if kind == "cypress":
                half = W * (0.10 + 0.26 * math.sin(min(1.0, f * 1.15) * math.pi * 0.9))
            else:
                half = W * (0.04 + 0.44 * f ** 0.95)
            n = int(14 + 50 * f)
            for k in range(n):
                u = rng.uniform(-1, 1)
                x = cx + u * half
                # Boughs droop at their tips.
                yy = y + abs(u) * H * 0.03 + rng.uniform(-6, 6)
                r = rng.uniform(7, 13) * (0.6 + 0.6 * f)
                shade = 0.55 + 0.30 * (k / n) + 0.25 * (1 - f) * 0.4
                col = tuple(c * shade for c in green)
                _disc(img, x, yy, r, col, rng, rag=0.45, hole=0.14)
    # Bleed colour into the transparent texels, or the mipmaps fringe every
    # card in black.
    a = img[..., 3] > 0
    if a.any():
        mean = img[..., :3][a].mean(axis=0)
        img[..., :3][~a] = mean * 0.8
    return img


def _top_view(rng):
    """Seen from straight above: a round blob of crowns."""
    W, H = CELL_W, CELL_H
    img = np.zeros((H, W, 4), np.float64)
    cx, cy, R = W * 0.5, H * 0.5, W * 0.46
    for k in range(380):
        a, r = rng.uniform(0, 2 * math.pi), R * math.sqrt(rng.random())
        x, y = cx + math.cos(a) * r, cy + math.sin(a) * r * (W / H) * 2
        shade = 0.55 + 0.5 * (k / 380)
        _disc(img, x, y, rng.uniform(12, 24),
              tuple(c * shade for c in (72, 98, 42)), rng)
    a = img[..., 3] > 0
    img[..., :3][~a] = img[..., :3][a].mean(axis=0) * 0.8
    return img


def atlas():
    """The canopy atlas, painted once per process."""
    global _ATLAS
    if _ATLAS is not None:
        return _ATLAS
    rng = np.random.default_rng(1234)
    sheet = np.zeros((ROWS * CELL_H, COLS * CELL_W, 4), np.float64)
    for kind, (cell, *_rest) in KINDS.items():
        r, c = divmod(cell, COLS)
        sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W] = \
            _canopy(kind, rng)
    r, c = divmod(TOP_CELL, COLS)
    sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W] = \
        _top_view(rng)
    # A touch less saturated: a sunlit canopy in linear light otherwise
    # comes out lime.
    lum = sheet[..., :3].mean(axis=-1, keepdims=True)
    sheet[..., :3] = lum + (sheet[..., :3] - lum) * 0.85
    # Fine leaf texture over everything, after the fact.
    grain = _noise(512, octaves=(64, 128, 256), seed=99)
    grain = np.tile(grain, (2, 2))[:sheet.shape[0], :sheet.shape[1]]
    sheet[..., :3] *= (0.88 + 0.24 * grain)[..., None]
    _ATLAS = panda_texture(sheet, "foliage_atlas", repeat=False, aniso=4)
    return _ATLAS


def _uv(cell):
    r, c = divmod(cell, COLS)
    u0, u1 = c / COLS, (c + 1) / COLS
    # Row 0 is the top of the sheet, which is V = 1.
    v1 = 1.0 - r / ROWS
    v0 = 1.0 - (r + 1) / ROWS
    # A half-texel inset so the bilinear filter never reads the next cell.
    du, dv = 0.5 / (COLS * CELL_W), 0.5 / (ROWS * CELL_H)
    return u0 + du, u1 - du, v0 + dv, v1 - dv


# --- geometry -------------------------------------------------------------
def _foliage_bin() -> str:
    """A cull bin between the opaque one (20) and the transparent one (30),
    sorted front to back."""
    from panda3d.core import CullBinEnums, CullBinManager
    mgr = CullBinManager.get_global_ptr()
    if mgr.find_bin("foliage") < 0:
        mgr.add_bin("foliage", CullBinEnums.BT_front_to_back, 25)
    return "foliage"


def build_cell(trees, parent=None):
    """One node for a neighbourhood of trees.

    *trees* is a list of (x, z, y, kind, yaw_deg, scale, tint). Returns an
    Ursina Entity holding a single Geom, or None for an empty list.
    """
    from panda3d.core import (Geom, GeomNode, GeomTriangles,
                              GeomVertexArrayFormat, GeomVertexData,
                              GeomVertexFormat, InternalName)
    from ursina import Entity, scene

    if not trees:
        return None
    per = 16                      # 3 crossed cards + 1 top card, 4 corners each
    nv = per * len(trees)
    data = np.zeros((nv, 12), np.float32)
    idx = np.zeros((len(trees) * 4, 6), np.uint32)
    tu0, tu1, tv0, tv1 = _uv(TOP_CELL)
    for k, (x, z, y, kind, yaw, sc, tint) in enumerate(trees):
        cell, cw, ch, cyf = KINDS[kind]
        w, h = cw * sc, ch * sc
        u0, u1, v0, v1 = _uv(cell)
        hc = h * cyf
        base = k * per
        for j in range(3):
            a = math.radians(yaw + 60.0 * j)
            ax, az = math.cos(a), math.sin(a)
            for q, (sx, sy) in enumerate(((-1, 0), (1, 0), (1, 1), (-1, 1))):
                px = x + ax * sx * w * 0.5
                pz = z + az * sx * w * 0.5
                py = y + sy * h
                # Normals out of the crown, from its middle: the card shades
                # like a rounded mass, light on top and dark underneath.
                nx, ny, nz = ax * sx * 0.9, (sy * h - hc) / h * 1.6, az * sx * 0.9
                ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
                u = u0 if sx < 0 else u1
                v = v0 if sy == 0 else v1
                data[base + 4 * j + q] = (px, py, pz, nx / ln, ny / ln, nz / ln,
                                          tint[0], tint[1], tint[2], 1.0, u, v)
            b = base + 4 * j
            idx[4 * k + j] = (b, b + 1, b + 2, b, b + 2, b + 3)
        # Top card: flat, at the crown, for anything looking down on it.
        r_top = w * 0.42
        yt = y + h * min(0.92, cyf + 0.18)
        b = base + 12
        for q, (sx, sz) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            u = tu0 if sx < 0 else tu1
            v = tv0 if sz < 0 else tv1
            data[b + q] = (x + sx * r_top, yt, z + sz * r_top, 0.0, 1.0, 0.0,
                           tint[0], tint[1], tint[2], 1.0, u, v)
        idx[4 * k + 3] = (b, b + 2, b + 1, b, b + 3, b + 2)

    arr = GeomVertexArrayFormat()
    arr.add_column(InternalName.get_vertex(), 3, Geom.NT_float32, Geom.C_point)
    arr.add_column(InternalName.get_normal(), 3, Geom.NT_float32, Geom.C_normal)
    arr.add_column(InternalName.get_color(), 4, Geom.NT_float32, Geom.C_color)
    arr.add_column(InternalName.get_texcoord(), 2, Geom.NT_float32,
                   Geom.C_texcoord)
    fmt = GeomVertexFormat.register_format(GeomVertexFormat(arr))
    vdata = GeomVertexData("trees", fmt, Geom.UH_static)
    vdata.unclean_set_num_rows(nv)
    memoryview(vdata.modify_array(0)).cast("B")[:] = data.tobytes()
    prim = GeomTriangles(Geom.UH_static)
    prim.set_index_type(Geom.NT_uint32)
    handle = prim.modify_vertices()
    handle.unclean_set_num_rows(idx.size)
    memoryview(handle).cast("B")[:] = idx.astype(np.uint32).tobytes()
    geom = Geom(vdata)
    geom.add_primitive(prim)
    node = GeomNode("trees")
    node.add_geom(geom)

    e = Entity(parent=parent if parent is not None else scene)
    e.attach_new_node(node)
    e.set_texture(atlas(), 1)
    # Both sides of every card, and with a priority that beats the baked
    # shadow camera's back-face override -- a card culled from the light's
    # side would leave half the canopy out of its shadow.
    e.set_two_sided(True, 2)
    # The tint lives in the vertex colours; no flat colour over it.
    e.set_color_off(2)
    # Drawn after everything else that is opaque, nearest cell first: the
    # forest is the most fragments on screen, most of it behind stands,
    # barriers and nearer trees, and the depth test can only throw those
    # away before shading once the things in front are already drawn.
    e.set_bin(_foliage_bin(), 0)
    from .shaders import foliage_shader
    e.world_shader = foliage_shader
    e.is_foliage = True
    return e


# --- photoscanned impostors ---------------------------------------------------
#: The baked atlas (tools/build_tree_impostors.py). When it is there, the
#: forest is drawn from it; the painted canopies above are the fallback.
IMPOSTOR_PNG = config.ASSET_DIR / "foliage" / "tree_impostors.png"
IMPOSTOR_JSON = config.ASSET_DIR / "foliage" / "tree_impostors.json"

#: Old prop names onto impostors, with a size multiplier (the island trees
#: were scanned at 3-5 m; a roadside wood wants them twice that) and how
#: much of the scenery's own random scale to keep.
FROM_PROP_IMPOSTOR = {
    "tree_round": (("island_a",), 2.1, 0.5),
    "tree_spread": (("island_a", "island_b"), 2.3, 0.5),
    "tree_bush": (("island_b",), 1.4, 0.4),
    "tree_pine": (("fir_a", "fir_b"), 0.95, 0.35),
    "tree_cypress": (("fir_a", "fir_b"), 0.80, 0.35),
}

_IMP = None


def impostors():
    """(texture, metadata) for the baked trees, or None if not built."""
    global _IMP
    if _IMP is not None:
        return _IMP or None
    if not (IMPOSTOR_PNG.is_file() and IMPOSTOR_JSON.is_file()):
        _IMP = False
        return None
    import json

    from PIL import Image
    meta = json.loads(IMPOSTOR_JSON.read_text())
    img = np.asarray(Image.open(IMPOSTOR_PNG).convert("RGBA")).astype(np.float64)
    tex = panda_texture(img, "tree_impostors", repeat=False, aniso=1)
    _IMP = (tex, meta)
    return _IMP


def build_impostor_cell(trees, tex, meta, parent=None):
    """Like ``build_cell``, from the baked photoscans: three crossed cards a
    tree -- the two baked sides and the first one mirrored -- and the top
    view laid flat at the crown. *trees* holds (x, z, y, id, yaw, scale,
    tint)."""
    from panda3d.core import (Geom, GeomNode, GeomTriangles,
                              GeomVertexArrayFormat, GeomVertexData,
                              GeomVertexFormat, InternalName)
    from ursina import Entity, scene

    if not trees:
        return None
    per = 16
    nv = per * len(trees)
    data = np.zeros((nv, 12), np.float32)
    idx = np.zeros((len(trees) * 4, 6), np.uint32)
    for k, (x, z, y, tid, yaw, sc, tint) in enumerate(trees):
        m = meta[tid]
        w, h = m["width"] * sc, m["height"] * sc
        hc = h * 0.62
        uv = m["uv"]
        base = k * per
        for j, (view, mirror) in enumerate((("side0", False), ("side90", False),
                                            ("side0", True))):
            u0, v0, u1, v1 = uv[view]
            if mirror:
                u0, u1 = u1, u0
            a = math.radians(yaw + 60.0 * j)
            ax, az = math.cos(a), math.sin(a)
            for q, (sx, sy) in enumerate(((-1, 0), (1, 0), (1, 1), (-1, 1))):
                nx, ny, nz = ax * sx * 0.9, (sy * h - hc) / h * 1.6, az * sx * 0.9
                ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
                data[base + 4 * j + q] = (
                    x + ax * sx * w * 0.5, y + sy * h, z + az * sx * w * 0.5,
                    nx / ln, ny / ln, nz / ln, tint[0], tint[1], tint[2], 1.0,
                    u0 if sx < 0 else u1, v0 if sy == 0 else v1)
            b = base + 4 * j
            idx[4 * k + j] = (b, b + 1, b + 2, b, b + 2, b + 3)
        tu0, tv0, tu1, tv1 = uv["top"]
        r_top = w * 0.5
        yt = y + h * 0.80
        b = base + 12
        c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        for q, (sx, sz) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            ox, oz = sx * r_top, sz * r_top
            data[b + q] = (x + ox * c - oz * s_, yt, z + ox * s_ + oz * c,
                           0.0, 1.0, 0.0, tint[0], tint[1], tint[2], 1.0,
                           tu0 if sx < 0 else tu1, tv0 if sz < 0 else tv1)
        idx[4 * k + 3] = (b, b + 2, b + 1, b, b + 3, b + 2)

    arr = GeomVertexArrayFormat()
    arr.add_column(InternalName.get_vertex(), 3, Geom.NT_float32, Geom.C_point)
    arr.add_column(InternalName.get_normal(), 3, Geom.NT_float32, Geom.C_normal)
    arr.add_column(InternalName.get_color(), 4, Geom.NT_float32, Geom.C_color)
    arr.add_column(InternalName.get_texcoord(), 2, Geom.NT_float32,
                   Geom.C_texcoord)
    fmt = GeomVertexFormat.register_format(GeomVertexFormat(arr))
    vdata = GeomVertexData("trees", fmt, Geom.UH_static)
    vdata.unclean_set_num_rows(nv)
    memoryview(vdata.modify_array(0)).cast("B")[:] = data.tobytes()
    prim = GeomTriangles(Geom.UH_static)
    prim.set_index_type(Geom.NT_uint32)
    handle = prim.modify_vertices()
    handle.unclean_set_num_rows(idx.size)
    memoryview(handle).cast("B")[:] = idx.astype(np.uint32).tobytes()
    geom = Geom(vdata)
    geom.add_primitive(prim)
    node = GeomNode("trees")
    node.add_geom(geom)
    e = Entity(parent=parent if parent is not None else scene)
    e.attach_new_node(node)
    e.set_texture(tex, 1)
    e.set_two_sided(True, 2)
    e.set_color_off(2)
    from .shaders import foliage_shader
    e.world_shader = foliage_shader
    e.is_foliage = True
    return e


def place(track, picks, rng):
    """Turn scenery's ``picks`` (prop name -> [(x, z, yaw, scale)]) into
    cells of card trees. Returns the list of entities."""
    imp = impostors()
    if imp is not None:
        return _place_impostors(picks, rng, *imp)
    out = []
    cell = config.FOREST_CELL if config.FOREST_CELL > 0 else 1e9
    cells: dict[tuple, list] = {}
    for name, places in picks.items():
        shape = name.rsplit("_", 1)[0]
        kinds = FROM_PROP.get(shape, ("round_a",))
        for (x, z, yaw, sc) in places:
            kind = kinds[int(rng.integers(0, len(kinds)))]
            # Card trees read bigger than the faceted ones did at the same
            # scale -- they have no hard outline to shrink inside -- so the
            # spread is pulled in a little.
            s = float(sc) ** 0.75 * rng.uniform(0.85, 1.1)
            # A per-tree tint: no two trees in a wood are the same green.
            g = rng.uniform(0.80, 1.12)
            tint = (g * rng.uniform(0.92, 1.06), g, g * rng.uniform(0.85, 1.0))
            y = config.Y_GRASS - 0.15
            key = (int(x // cell), int(z // cell))
            cells.setdefault(key, []).append((float(x), float(z), y, kind,
                                              float(yaw), s, tint))
    for trees in cells.values():
        e = build_cell(trees)
        if e is not None:
            out.append(e)
    return out


def _place_impostors(picks, rng, tex, meta):
    out = []
    cell = config.FOREST_CELL if config.FOREST_CELL > 0 else 1e9
    cells: dict[tuple, list] = {}
    for name, places in picks.items():
        shape = name.rsplit("_", 1)[0]
        ids, mult, keep = FROM_PROP_IMPOSTOR.get(shape,
                                                 FROM_PROP_IMPOSTOR["tree_round"])
        ids = [i for i in ids if i in meta] or list(meta)
        for (x, z, yaw, sc) in places:
            tid = ids[int(rng.integers(0, len(ids)))]
            s = mult * float(sc) ** keep * rng.uniform(0.88, 1.12)
            # Photographs carry their own colour; only a slight shift
            # between trees, so a wood is not one tree copied.
            g = rng.uniform(0.90, 1.08)
            tint = (g * rng.uniform(0.96, 1.04), g, g * rng.uniform(0.94, 1.02))
            key = (int(x // cell), int(z // cell))
            cells.setdefault(key, []).append((float(x), float(z),
                                              config.Y_GRASS - 0.10, tid,
                                              float(yaw), s, tint))
    for trees in cells.values():
        e = build_impostor_cell(trees, tex, meta)
        if e is not None:
            out.append(e)
    return out
