"""The F-AI mark: drawn, not typed.

The broadcast's top-left corner carries a logo, and a logo is a shape, not a
string -- set in a font the wordmark looked like what it was, a label. This
draws an original mark in the same spirit as a motorsport series' badge:
everything leaning forward at one angle, a heavy red F whose arms run on into
speed stripes, and a white AI cut from the same parallelograms, so the whole
thing reads as one piece of geometry moving left to right.

Rendered once with Pillow at four times the size it is shown at and filtered
down, so its diagonals are smooth on a UI layer that has no antialiasing of
its own. No font is involved, so it looks the same on every machine.
"""
from __future__ import annotations

import numpy as np

#: Every vertical edge leans by this much x per unit of height.
SLANT = 0.34
RED = (225, 6, 0, 255)
WHITE = (255, 255, 255, 255)


def _poly(draw, pts, col, s, ox, oy, h):
    """Points in mark units (y up, cap height 1) -> image pixels."""
    draw.polygon([(ox + (x + SLANT * y) * s, oy + (h - y) * s) for x, y in pts],
                 fill=col)


def shapes():
    """(colour, polygon) list in mark units: cap height 1, y up, unslanted.

    The slant is applied when drawing, so every edge given here as vertical
    comes out at exactly the same lean.

    The F's top arm does not stop where an F's would: it runs on over the
    whole mark as one red bar, so the AI tucked under it reads as part of the
    same piece rather than a second word beside a letter.
    """
    out = []
    bar_y0, end = 0.78, 2.50
    # F: the stem, the long top arm, and a short middle arm. Tips are cut on
    # a steeper diagonal than the slant, so they read as sliced, not boxed.
    out.append((RED, [(0.00, 0.00), (0.27, 0.00), (0.27, 1.00), (0.00, 1.00)]))
    out.append((RED, [(0.00, bar_y0), (end - 0.10, bar_y0), (end, 1.00),
                      (0.00, 1.00)]))
    out.append((RED, [(0.00, 0.36), (0.74, 0.36), (0.80, 0.58), (0.00, 0.58)]))
    # AI, white, under the bar.
    h = 0.62
    ax = 1.00
    ap, apw = 0.42, 0.11
    inner = h * 0.82

    out.append((WHITE, [(ax, 0.00), (ax + 0.24, 0.00), (ax + ap, inner),
                        (ax + ap, h), (ax + ap - apw, h)]))
    out.append((WHITE, [(ax + ap, inner), (ax + 0.60, 0.00),
                        (ax + 0.84, 0.00), (ax + ap + apw, h), (ax + ap, h)]))

    def xl(y):
        return ax + 0.24 + (ap - 0.24) * y / inner

    def xr(y):
        return ax + 0.60 - (0.60 - ap) * y / inner
    out.append((WHITE, [(xl(0.17) - 0.02, 0.17), (xr(0.17) + 0.02, 0.17),
                        (xr(0.31) + 0.02, 0.31), (xl(0.31) - 0.02, 0.31)]))
    ix = ax + 0.98
    out.append((WHITE, [(ix, 0.00), (ix + 0.23, 0.00), (ix + 0.23, h),
                        (ix, h)]))
    return out


def width_units() -> float:
    xs = [x + SLANT * y for _, poly in shapes() for x, y in poly]
    return max(xs) - min(xs)


def render(height_px: int = 96, pad: int = 6, scale: int = 4,
           mono=None):
    """The mark as an RGBA numpy array, cap height *height_px*.

    *mono* draws every part in one colour (for the red tile on the cards).
    """
    from PIL import Image, ImageDraw

    s = height_px * scale
    w_units = width_units()
    W = int((w_units + 0.02) * s + 2 * pad * scale)
    H = int(s + 2 * pad * scale)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ox = pad * scale
    oy = pad * scale
    for col, poly in shapes():
        _poly(d, poly, mono or col, s, ox, oy, 1.0)
    img = img.resize((W // scale, H // scale), Image.LANCZOS)
    return np.asarray(img)


_TEX: dict = {}


def texture(height_px: int = 96, mono=None):
    """A mipmapped, clamped Panda texture of the mark (cached)."""
    key = (height_px, mono)
    if key not in _TEX:
        from .textures import panda_texture
        _TEX[key] = panda_texture(render(height_px, mono=mono).astype(float),
                                  f"logo_{height_px}", repeat=False, aniso=1)
    return _TEX[key]


def aspect() -> float:
    a = render(64)
    return a.shape[1] / a.shape[0]
