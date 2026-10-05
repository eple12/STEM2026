"""Drawing kit for the broadcast overlay: one Geom per layer, not one per label.

The old overlay was built from Ursina ``Text`` and ``Entity`` objects. Each
Text is a handful of Panda nodes with its own shader binding, and Panda pays
for every node it culls and every state change it draws, every frame, whether
or not anything on it changed. Measured, the overlay cost about 4 ms a frame
to *draw* -- more than the whole HDR camera chain -- on top of the Python that
updated it.

This draws the same kind of overlay as a few big meshes:

* ``ShapeLayer`` -- every panel, bar, lamp and dot, as coloured triangles in
  one vertex buffer. Shapes can be recoloured, moved and resized; a change
  rewrites the numpy copy, and the buffer is uploaded once per frame at most.
* ``TextLayer`` -- every label, as glyph quads cut from the font's own glyph
  page, one Geom per typeface. A label has a fixed number of glyph slots;
  setting its text rewrites those slots and nothing else.

Each layer is a single node and a single draw call. The whole in-race HUD is
five or six of them.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import config

# --- shaders ---------------------------------------------------------------
_VERT = """#version 150
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec4 p3d_Color;
in vec2 p3d_MultiTexCoord0;
out vec4 col;
out vec2 uv;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    col = p3d_Color;
    uv = p3d_MultiTexCoord0;
}
"""
_SHAPE_FRAG = """#version 150
uniform vec4 p3d_ColorScale;
in vec4 col;
in vec2 uv;
out vec4 o;
void main() { o = col * p3d_ColorScale; }
"""
_TEXT_FRAG = """#version 150
uniform sampler2D p3d_Texture0;
uniform vec4 p3d_ColorScale;
in vec4 col;
in vec2 uv;
out vec4 o;
void main() {
    float a = texture(p3d_Texture0, uv).a;
    o = vec4(col.rgb, col.a * a) * p3d_ColorScale;
}
"""
_SHADERS: dict = {}


def _shader(kind: str):
    from panda3d.core import Shader
    if kind not in _SHADERS:
        frag = _TEXT_FRAG if kind == "text" else _SHAPE_FRAG
        _SHADERS[kind] = Shader.make(Shader.SL_GLSL, _VERT, frag)
    return _SHADERS[kind]


def _format():
    from panda3d.core import (Geom, GeomVertexArrayFormat, GeomVertexFormat,
                              InternalName)
    arr = GeomVertexArrayFormat()
    arr.add_column(InternalName.get_vertex(), 3, Geom.NT_float32, Geom.C_point)
    arr.add_column(InternalName.get_color(), 4, Geom.NT_float32, Geom.C_color)
    arr.add_column(InternalName.get_texcoord(), 2, Geom.NT_float32,
                   Geom.C_texcoord)
    return GeomVertexFormat.register_format(GeomVertexFormat(arr))


_RGBA_CACHE: dict = {}


def _rgba(col):
    """A colour as an (r, g, b, a) tuple of floats. The HUD passes the same
    few palette objects every frame, so their conversions are remembered."""
    if type(col) is tuple and len(col) == 4:
        return col
    hit = _RGBA_CACHE.get(id(col))
    if hit is not None and hit[0] is col:
        return hit[1]
    if hasattr(col, "r"):
        c = (float(col.r), float(col.g), float(col.b), float(col.a))
    else:
        c = tuple(float(v) for v in col)
        c = c if len(c) == 4 else (*c, 1.0)
    if hasattr(col, "r"):
        _RGBA_CACHE[id(col)] = (col, c)
    return c


class _Mesh:
    """A vertex buffer kept as numpy and uploaded when it changes."""

    def __init__(self, name):
        self.name = name
        self.v = np.zeros((0, 9), np.float32)   # x y z  r g b a  u v
        self.tri: list[int] = []
        self.dirty = False
        self.vdata = None
        self.np = None

    def build(self, parent, kind: str, sort: int, texture=None):
        from panda3d.core import (Geom, GeomNode, GeomTriangles,
                                  GeomVertexData, TransparencyAttrib)
        if not len(self.v):
            return None
        vdata = GeomVertexData(self.name, _format(), Geom.UH_dynamic)
        vdata.unclean_set_num_rows(len(self.v))
        memoryview(vdata.modify_array(0)).cast("B")[:] = self.v.tobytes()
        prim = GeomTriangles(Geom.UH_static)
        prim.set_index_type(Geom.NT_uint32)
        idx = np.asarray(self.tri, np.uint32)
        h = prim.modify_vertices()
        h.unclean_set_num_rows(len(idx))
        memoryview(h).cast("B")[:] = idx.tobytes()
        geom = Geom(vdata)
        geom.add_primitive(prim)
        node = GeomNode(self.name)
        node.add_geom(geom)
        np_ = parent.attach_new_node(node)
        np_.set_shader(_shader(kind), 10)
        np_.set_transparency(TransparencyAttrib.M_alpha)
        np_.set_depth_test(False)
        np_.set_depth_write(False)
        np_.set_bin("fixed", sort)
        np_.set_two_sided(True)
        if texture is not None:
            np_.set_texture(texture, 10)
        self.vdata, self.np = vdata, np_
        return np_

    def flush(self):
        if self.dirty and self.vdata is not None:
            memoryview(self.vdata.modify_array(0)).cast("B")[:] = self.v.tobytes()
        self.dirty = False


# --- shapes ----------------------------------------------------------------
class Shape:
    """A handle on one shape in a ShapeLayer."""

    __slots__ = ("layer", "a", "b", "base", "col", "dx", "dy", "_shown")

    def __init__(self, layer, a, b, col):
        self.layer, self.a, self.b, self.col = layer, a, b, col
        self.base = layer.m.v[a:b, :2].copy()
        self.dx = self.dy = 0.0
        self._shown = True

    @property
    def color(self):
        return self.col

    @color.setter
    def color(self, col):
        c = _rgba(col)
        if c == self.col:
            return
        self.col = c
        self.layer.m.v[self.a:self.b, 3:7] = c
        self.layer.m.dirty = True

    def move(self, dx: float, dy: float):
        """Offset from where the shape was built."""
        if dx == self.dx and dy == self.dy:
            return
        self.dx, self.dy = dx, dy
        if self._shown:
            self.layer.m.v[self.a:self.b, 0] = self.base[:, 0] + dx
            self.layer.m.v[self.a:self.b, 1] = self.base[:, 1] + dy
            self.layer.m.dirty = True

    def stretch(self, sx: float = 1.0, sy: float = 1.0, ox=None, oy=None):
        """Scale about (ox, oy) -- by default the shape's own left/bottom."""
        b = self.base
        ox = b[:, 0].min() if ox is None else ox
        oy = b[:, 1].min() if oy is None else oy
        self.layer.m.v[self.a:self.b, 0] = ox + (b[:, 0] - ox) * sx + self.dx
        self.layer.m.v[self.a:self.b, 1] = oy + (b[:, 1] - oy) * sy + self.dy
        self.layer.m.dirty = True

    def show(self, on: bool):
        if on == self._shown:
            return
        self._shown = on
        if on:
            self.layer.m.v[self.a:self.b, 0] = self.base[:, 0] + self.dx
            self.layer.m.v[self.a:self.b, 1] = self.base[:, 1] + self.dy
        else:
            # Collapsed to a point: no area, nothing drawn, nothing culled.
            self.layer.m.v[self.a:self.b, 0:2] = self.layer.m.v[self.a, 0:2]
        self.layer.m.dirty = True

    def set_rect(self, x: float, y: float, w: float, h: float):
        """Re-lay a rectangle (built by ``ShapeLayer.rect``) to a new place
        and size -- a box that fits the text in it. The offset from ``move``
        still applies on top, so a row that slides keeps sliding."""
        b = np.array(((x, y), (x + w, y), (x + w, y + h), (x, y + h)),
                     np.float32)
        if np.array_equal(b, self.base):
            return
        self.base = b
        if self._shown:
            self.layer.m.v[self.a:self.b, 0] = b[:, 0] + self.dx
            self.layer.m.v[self.a:self.b, 1] = b[:, 1] + self.dy
            self.layer.m.dirty = True


class ShapeBatch:
    """Many shapes of one layer placed in a single array write: the track
    map's twenty dots every frame were forty separate moves and shows."""

    def __init__(self, shapes):
        self.layer = shapes[0].layer
        self.rows = np.concatenate([np.arange(s.a, s.b) for s in shapes])
        self.counts = np.array([s.b - s.a for s in shapes])
        self.base = np.concatenate([s.base for s in shapes]).astype(np.float32)

    def place(self, xy):
        """Offset each shape by its row of *xy* ((n, 2), from where it was
        built); a NaN row hides that shape (collapsed to a point)."""
        off = np.repeat(np.asarray(xy, np.float32), self.counts, axis=0)
        hide = np.isnan(off[:, 0])
        pos = self.base + np.nan_to_num(off)
        pos[hide] = -10.0
        v = self.layer.m.v
        v[self.rows, 0] = pos[:, 0]
        v[self.rows, 1] = pos[:, 1]
        self.layer.m.dirty = True


class ShapeLayer:
    """Coloured triangles, one draw call. Add everything, then ``build``."""

    def __init__(self, name="shapes"):
        self.m = _Mesh(name)
        self._tri: list[int] = []
        self.np = None

    def _add(self, pts, tris, col, z=0.0) -> Shape:
        c = _rgba(col)
        a = len(self.m.v)
        rows = np.zeros((len(pts), 9), np.float32)
        rows[:, 0:2] = pts
        rows[:, 2] = z
        rows[:, 3:7] = c
        # Appended to the live buffer, never rebuilt from a list: a shape
        # hidden during the build has collapsed its rows in place.
        self.m.v = np.vstack([self.m.v, rows])
        self._tri += [a + t for t in tris]
        return Shape(self, a, a + len(pts), c)

    def rect(self, x, y, w, h, col, skew: float = 0.0) -> Shape:
        """Axis-aligned (or leaning) rectangle from its bottom-left corner."""
        s = skew * h
        pts = [(x, y), (x + w, y), (x + w + s, y + h), (x + s, y + h)]
        return self._add(pts, [0, 1, 2, 0, 2, 3], col)

    def rect_c(self, cx, cy, w, h, col, skew: float = 0.0) -> Shape:
        return self.rect(cx - w / 2, cy - h / 2, w, h, col, skew)

    def round_rect(self, x, y, w, h, r, col, seg: int = 4) -> Shape:
        r = min(r, w / 2, h / 2)
        pts = [(x + w / 2, y + h / 2)]
        for cx, cy, a0 in ((x + w - r, y + r, -90), (x + w - r, y + h - r, 0),
                           (x + r, y + h - r, 90), (x + r, y + r, 180)):
            for k in range(seg + 1):
                a = math.radians(a0 + 90.0 * k / seg)
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        n = len(pts) - 1
        tris = []
        for k in range(n):
            tris += [0, 1 + k, 1 + (k + 1) % n]
        return self._add(pts, tris, col)

    def tri(self, pts, col) -> Shape:
        """A triangle through three points (any winding: two-sided)."""
        return self._add(list(pts), [0, 1, 2], col)

    def disc(self, cx, cy, r, col, seg: int = 24) -> Shape:
        pts = [(cx, cy)] + [(cx + r * math.cos(2 * math.pi * k / seg),
                             cy + r * math.sin(2 * math.pi * k / seg))
                            for k in range(seg)]
        tris = []
        for k in range(seg):
            tris += [0, 1 + k, 1 + (k + 1) % seg]
        return self._add(pts, tris, col)

    def polyline(self, pts, width, col, closed=True) -> Shape:
        """A thick line as a mitred strip. For the track map."""
        p = np.asarray(pts, float)
        n = len(p)
        if closed:
            prev = np.roll(p, 1, axis=0)
            nxt = np.roll(p, -1, axis=0)
        else:
            prev = np.vstack([p[:1] * 2 - p[1:2], p[:-1]])
            nxt = np.vstack([p[1:], p[-1:] * 2 - p[-2:-1]])
        d0 = p - prev
        d1 = nxt - p
        d0 /= np.maximum(np.linalg.norm(d0, axis=1, keepdims=True), 1e-9)
        d1 /= np.maximum(np.linalg.norm(d1, axis=1, keepdims=True), 1e-9)
        t = d0 + d1
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        nrm = np.stack([-t[:, 1], t[:, 0]], axis=1)
        n0 = np.stack([-d1[:, 1], d1[:, 0]], axis=1)
        miter = 1.0 / np.clip((nrm * n0).sum(axis=1), 0.35, 1.0)
        off = nrm * (width / 2 * miter)[:, None]
        verts = np.empty((2 * n, 2))
        verts[0::2] = p + off
        verts[1::2] = p - off
        tris = []
        m = n if closed else n - 1
        for k in range(m):
            a, b = 2 * k, 2 * ((k + 1) % n)
            tris += [a, a + 1, b + 1, a, b + 1, b]
        return self._add(verts, tris, col)

    def build(self, parent, sort: int = 0):
        # m.v is already current -- _add keeps it so, and a shape hidden
        # before the build has collapsed its rows there, not in _rows.
        self.m.tri = self._tri
        self.np = self.m.build(parent, "shape", sort)
        return self.np

    def flush(self):
        self.m.flush()


# --- text ------------------------------------------------------------------
CHARSET = (" 0123456789:.,+-/()%°'·|#&!?*_=<>"
           "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

#: name -> (DynamicTextFont, {char: (l, b, r, t, u0, v0, u1, v1, advance)})
_FONTS: dict = {}


def font_file() -> Path | None:
    """The typeface file the menu picked (see ui.pick_font)."""
    from ursina import application
    from .ui import pick_font
    name = pick_font()
    if not name:
        return None
    p = Path(application.fonts_folder) / name
    return p if p.is_file() else None


#: Weights. For Bahnschrift, a variable font, these are its named instances
#: (FreeType's face index carries the instance in its high 16 bits); a font
#: with one weight per file simply serves every weight from that file.
WEIGHTS = {"light": 1, "regular": 3, "semibold": 4, "bold": 5}


def font(weight: str = "regular", ppu: int = 72):
    key = (weight, ppu)
    if key in _FONTS:
        return _FONTS[key]
    from panda3d.core import (DynamicTextFont, Filename, LVecBase4f,
                              SamplerState)
    path = font_file()
    if path is None:
        raise RuntimeError("hudkit: no font file")
    idx = WEIGHTS.get(weight, 0) << 16 if "bahnschrift" in path.name.lower() else 0
    f = DynamicTextFont(Filename.from_os_specific(str(path)), idx)
    if not f.is_valid():
        f = DynamicTextFont(Filename.from_os_specific(str(path)), 0)
    f.set_pixels_per_unit(ppu)
    f.set_page_size(1024, 1024)
    f.set_texture_margin(3)
    f.set_minfilter(SamplerState.FT_linear_mipmap_linear)
    f.set_magfilter(SamplerState.FT_linear)
    glyphs = {}
    page = None
    for ch in CHARSET:
        g = f.get_glyph(ord(ch))
        if g is None:
            continue
        d, t = LVecBase4f(), LVecBase4f()
        if g.get_quad(d, t):
            pg = g.get_page() if hasattr(g, "get_page") else None
            if page is None and pg is not None:
                page = pg
            glyphs[ch] = (d[0], d[1], d[2], d[3], t[0], t[1], t[2], t[3],
                          g.get_advance())
        else:
            glyphs[ch] = (0, 0, 0, 0, 0, 0, 0, 0, g.get_advance())
    if page is None:
        raise RuntimeError("hudkit: font produced no glyph page")
    cap = glyphs["H"][3] if "H" in glyphs else 0.7
    _FONTS[key] = (f, glyphs, page, cap)
    return _FONTS[key]


class Label:
    """A run of text in a TextLayer: fixed slots, rewritten on change."""

    __slots__ = ("layer", "font", "a", "cap", "x", "y", "size", "align", "col",
                 "text", "track", "skew", "width", "_shown", "_tw", "mono")

    def __init__(self, layer, font_key, a, cap, x, y, size, align, col,
                 track, skew, mono=False):
        self.layer, self.font, self.a, self.cap = layer, font_key, a, cap
        self.x, self.y, self.size = x, y, size
        self.align, self.col, self.track, self.skew = align, _rgba(col), track, skew
        #: Tabular figures: every digit on the same pitch, so a running
        #: number does not breathe as its width changes.
        self.mono = mono
        self.text = None
        self.width = 0.0
        self._shown = True
        self._tw = 0.0

    def set(self, text: str, col=None):
        text = str(text)
        c = self.col if col is None else _rgba(col)
        if text == self.text and c == self.col:
            return
        self.text, self.col = text, c
        self._write()

    @property
    def color(self):
        return self.col

    @color.setter
    def color(self, col):
        c = _rgba(col)
        if c != self.col:
            self.col = c
            m = self.layer.meshes[self.font]
            m.v[4 * self.a:4 * (self.a + self.cap), 3:7] = c
            m.dirty = True

    def move(self, x: float, y: float):
        if x == self.x and y == self.y:
            return
        dx, dy = x - self.x, y - self.y
        self.x, self.y = x, y
        if self.text is None:
            return
        # A move is a translation of the quads already laid out (unused
        # slots are zero-area wherever they are), not a new layout.
        m = self.layer.meshes[self.font]
        rows = m.v[4 * self.a:4 * (self.a + self.cap)]
        rows[:, 0] += dx
        rows[:, 1] += dy
        m.dirty = True

    def show(self, on: bool):
        if on == self._shown:
            return
        self._shown = on
        self._write()

    def resize(self, size: float):
        """A new cap height (a line too long for its box is set smaller)."""
        if size != self.size:
            self.size = size
            self._write()

    def _write(self):
        _f, glyphs, _page, capf = self.layer.fonts[self.font]
        m = self.layer.meshes[self.font]
        v = m.v
        a, n = 4 * self.a, 4 * self.cap
        text = self.text or ""
        if not self._shown:
            text = ""
        s = self.size / capf                  # screen units per font unit
        cx = 0.0
        quads = []
        pitch = self.layer.digit_pitch(self.font) if self.mono else 0.0
        for ch in text[:self.cap]:
            g = glyphs.get(ch) or glyphs.get("?")
            if g is None:
                continue
            l, b, r, t, u0, v0, u1, v1, adv = g
            shift = 0.0
            if pitch and ch.isdigit():
                shift = (pitch - adv) / 2.0
                adv = pitch
            if r > l:
                quads.append((cx + (l + shift) * s, b * s, cx + (r + shift) * s,
                              t * s, u0, v0, u1, v1))
            cx += adv * s + self.track
        w = max(0.0, cx - self.track)
        self.width = w
        ox = {"left": 0.0, "center": -w / 2, "right": -w}[self.align]
        k = self.skew
        # Corners gathered in plain Python and written in one assignment:
        # per-glyph slice writes cost more in numpy's call overhead than
        # everything else here together.
        rows = []
        bx = self.x + ox
        for x0, y0, x1, y1, u0, v0, u1, v1 in quads:
            X0, X1 = bx + x0, bx + x1
            Y0, Y1 = self.y + y0, self.y + y1
            rows += ((X0 + k * y0, Y0, u0, v0), (X1 + k * y0, Y0, u1, v0),
                     (X1 + k * y1, Y1, u1, v1), (X0 + k * y1, Y1, u0, v1))
        v[a:a + n, 0:3] = 0.0
        v[a:a + n, 3:7] = self.col
        if rows:
            r = np.array(rows, np.float32)
            g = len(rows)
            v[a:a + g, 0:2] = r[:, 0:2]
            v[a:a + g, 7:9] = r[:, 2:4]
        m.dirty = True


class DigitField:
    """A number on the HUD, drawn from pre-cut glyph cells.

    The digits are already images -- glyphs rasterised once onto the font's
    page -- but a ``Label`` still lays every glyph out in Python whenever the
    text changes, and a speedometer, a lap clock and twenty gaps change every
    frame. Here every character cell is fixed when the field is built: cell
    k sits at a fixed pitch from the field's right (or left) edge, and the
    quad, texture rectangle and all, for each character a cell can show is
    worked out once per typeface. Setting a value is then one table lookup
    and one array assignment for the whole field, whatever its length.

    Right-aligned by default, the way numbers line up in a column; only
    characters in ``CHARS`` are drawn (anything else is a blank cell).
    """

    CHARS = " 0123456789:.+-/"

    __slots__ = ("layer", "font", "a", "cap", "x", "y", "size", "align", "col",
                 "text", "_shown", "_codes", "_cells", "dy")

    def __init__(self, layer, font_key, a, cap, x, y, size, align, col):
        self.layer, self.font, self.a, self.cap = layer, font_key, a, cap
        self.x, self.y, self.size, self.align = x, y, size, align
        self.col = _rgba(col)
        self.text = None
        self._shown = True
        self.dy = 0.0
        self._codes = None
        self._cells = None

    def _prepare(self):
        """Cell origins, and the (4 corners x [x, y, u, v]) quad of every
        character at this size, relative to its cell."""
        table, pitch = self.layer.digit_table(self.font)
        s = self.size / self.layer.fonts[self.font][3]
        self._cells = (table * np.array([s, s, 1.0, 1.0], np.float32), pitch * s)

    def set(self, text: str, col=None):
        text = str(text)
        c = self.col if col is None else _rgba(col)
        if text == self.text and c == self.col:
            return
        recolor = c != self.col
        self.text, self.col = text, c
        self._write(recolor)

    def move(self, x: float, y: float):
        if x == self.x and y == self.y:
            return
        dx, dy = x - self.x, y - self.y
        self.x, self.y = x, y
        if self._cells is None:
            self._write(False)
            return
        m = self.layer.meshes[self.font]
        rows = m.v[4 * self.a:4 * (self.a + self.cap)]
        rows[:, 0] += dx
        rows[:, 1] += dy
        m.dirty = True

    def show(self, on: bool):
        if on != self._shown:
            self._shown = on
            self._write(False)

    def _write(self, recolor: bool = True):
        if self._cells is None:
            self._prepare()
        quads, pitch = self._cells
        m = self.layer.meshes[self.font]
        idx = self.layer.digit_index
        text = (self.text or "")[-self.cap:] if self._shown else ""
        n = len(text)
        codes = np.zeros(self.cap, np.intp)
        if n:
            codes[self.cap - n:] = [idx.get(ch, 0) for ch in text]
        if self.align == "left":
            codes = np.roll(codes, -(self.cap - n))
            x0 = self.x
        else:
            x0 = self.x - self.cap * pitch
        q = quads[codes]                                     # (cap, 4, 4)
        cell = x0 + pitch * np.arange(self.cap, dtype=np.float32)
        rows = m.v[4 * self.a:4 * (self.a + self.cap)].reshape(self.cap, 4, 9)
        rows[:, :, 0] = cell[:, None] + q[:, :, 0]
        rows[:, :, 1] = self.y + q[:, :, 1]
        rows[:, :, 7] = q[:, :, 2]
        rows[:, :, 8] = q[:, :, 3]
        if recolor or self._codes is None:
            rows[:, :, 3:7] = self.col
        self._codes = codes
        m.dirty = True

    @property
    def color(self):
        return self.col

    @color.setter
    def color(self, col):
        c = _rgba(col)
        if c != self.col:
            self.col = c
            m = self.layer.meshes[self.font]
            m.v[4 * self.a:4 * (self.a + self.cap), 3:7] = c
            m.dirty = True


class TextLayer:
    """Every label of one overlay, one Geom (one draw call) per weight."""

    def __init__(self, name="text"):
        self.name = name
        self.fonts: dict = {}
        self.meshes: dict[str, _Mesh] = {}
        self._count: dict[str, int] = {}
        self._labels: list[Label] = []
        self.root = None

    def digit_pitch(self, weight) -> float:
        glyphs = self.fonts[weight][1]
        return max(glyphs[d][8] for d in "0123456789" if d in glyphs)

    #: DigitField's character -> row of digit_table.
    digit_index = {ch: k for k, ch in enumerate(DigitField.CHARS)}

    def digit_table(self, weight):
        """((len(CHARS), 4, 4) quads in font units relative to a cell's
        origin -- x, y, u, v per corner, glyphs centred in the cell -- and
        the cell pitch). A blank or missing character is a zero-area quad."""
        cache = self.__dict__.setdefault("_digit_tables", {})
        if weight in cache:
            return cache[weight]
        glyphs = self.fonts[weight][1]
        pitch = self.digit_pitch(weight)
        tab = np.zeros((len(DigitField.CHARS), 4, 4), np.float32)
        for k, ch in enumerate(DigitField.CHARS):
            g = glyphs.get(ch)
            if g is None or not g[2] > g[0]:
                continue
            l, b, r, t, u0, v0, u1, v1, adv = g
            shift = (pitch - adv) / 2.0
            l, r = l + shift, r + shift
            tab[k] = ((l, b, u0, v0), (r, b, u1, v0), (r, t, u1, v1), (l, t, u0, v1))
        cache[weight] = (tab, pitch)
        return cache[weight]

    def digits(self, text, x, y, size, *, weight="regular", align="right",
               col=(1, 1, 1, 1), cap=8) -> "DigitField":
        """A DigitField of *cap* cells; *x* is its right edge (or left)."""
        if weight not in self.fonts:
            self.fonts[weight] = font(weight)
            self.meshes[weight] = _Mesh(f"{self.name}_{weight}")
            self._count[weight] = 0
        df = DigitField(self, weight, self._count[weight], cap, x, y, size,
                        align, col)
        self._count[weight] += cap
        df.text = str(text)
        self._labels.append(df)
        return df

    def label(self, text, x, y, size, *, weight="regular", align="left",
              col=(1, 1, 1, 1), cap=None, track=0.0, skew=0.0,
              mono=False) -> Label:
        """*size* is the cap height in screen units (1 = screen height);
        *x, y* the baseline's start (or centre / end, by *align*)."""
        if weight not in self.fonts:
            self.fonts[weight] = font(weight)
            self.meshes[weight] = _Mesh(f"{self.name}_{weight}")
            self._count[weight] = 0
        cap = max(1, cap or len(str(text)))
        lb = Label(self, weight, self._count[weight], cap, x, y, size, align,
                   col, track, skew, mono)
        self._count[weight] += cap
        lb.text = str(text)
        self._labels.append(lb)
        return lb

    def build(self, parent, sort: int = 20):
        from ursina import Entity
        self.root = Entity(parent=parent)
        for w, m in self.meshes.items():
            n = self._count[w]
            m.v = np.zeros((4 * n, 9), np.float32)
            tri = []
            for q in range(n):
                b = 4 * q
                tri += [b, b + 1, b + 2, b, b + 2, b + 3]
            m.tri = tri
        for lb in self._labels:
            lb._write()
        for w, m in self.meshes.items():
            m.build(self.root, "text", sort, texture=self.fonts[w][2])
            m.dirty = False
        return self.root

    def flush(self):
        for m in self.meshes.values():
            m.flush()
