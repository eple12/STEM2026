"""Procedurally generated textures (Phase A: zero external assets).

All images are built with numpy + Pillow at run time and handed to Ursina as
``Texture`` objects, so nothing needs to be committed to the repo.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from ursina import Texture


def _noise(size: int, octaves=(4, 8, 16, 32), seed: int = 0) -> np.ndarray:
    """Value-noise in [0, 1] by summing upscaled random grids. **Tileable.**

    Resizing a random grid straight to *size* does not wrap: the left edge of
    the image has nothing to do with the right, so every repeat of the texture
    shows a seam and a plane covered in it reads as graph paper. That is what
    the ground did.

    The fix is to wrap-pad each octave's grid by a couple of cells before
    upscaling and then crop the middle back out. Bicubic only reaches a cell or
    two, so the padding is enough for the crop's edges to have been
    interpolated against the values that actually follow them -- which is
    exactly what makes the join invisible.
    """
    rng = np.random.default_rng(seed)
    acc = np.zeros((size, size), dtype=np.float64)
    weight = 0.0
    pad = 2
    for i, o in enumerate(octaves):
        w = 0.5 ** i
        grid = np.pad(rng.random((o, o)), pad, mode="wrap")
        cell = size / float(o)                       # pixels per grid cell
        big = int(round((o + 2 * pad) * cell))
        img = np.asarray(Image.fromarray((grid * 255).astype(np.uint8)).resize(
            (big, big), Image.BICUBIC), dtype=np.float64) / 255.0
        k = int(round(pad * cell))
        acc += w * img[k:k + size, k:k + size]
        weight += w
    return acc / weight


def _tex(arr: np.ndarray) -> Texture:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    return Texture(img)


def asphalt(size: int = 512) -> Texture:
    n = _noise(size, seed=1)
    base = 46 + 26 * n
    # faint lighter aggregate specks
    speck = (_noise(size, octaves=(64, 128), seed=2) > 0.72) * 22
    rgb = np.stack([base + speck] * 3, axis=-1)
    rgb[..., 2] += 4  # a hair blue/grey
    return _tex(rgb)


def grass(size: int = 512) -> Texture:
    """Turf.

    _noise() upscales a small random grid with BICUBIC, which is very smooth:
    stacking its default octaves gives an almost flat green, which is what this
    was. Real variation needs three separate scales with weight of their own --
    blades, clumps, and broad lighter/darker patches -- plus a per-pixel speck
    that the mipmaps average away at distance and that grains the turf under
    the car's nose.
    """
    rng = np.random.default_rng(30)
    blade = _noise(size, octaves=(96, 192), seed=14)
    clump = _noise(size, octaves=(16, 32), seed=3)
    patch = _noise(size, octaves=(3, 6), seed=4)
    speck = rng.random((size, size))
    # Weighted *away* from the low octaves, which is the opposite of what it
    # wants in isolation. Making the tile seamless stopped the hard join but
    # not the repetition: broad light and dark patches are exactly what the eye
    # recognises coming round again, and on a plane that reaches the mountains
    # they lay a chequerboard over the whole infield. Broad variation is done
    # in world space instead, by mottle() on the apron and the bank, where it
    # cannot repeat by construction; the texture only has to supply grain.
    t = 0.46 * blade + 0.42 * clump + 0.12 * patch
    t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
    g = 86 + 52 * t + 16 * (speck - 0.5)
    r = 0.44 * g + 6 + 11 * (speck - 0.5)
    b = 0.34 * g + 8
    return _tex(np.stack([r, g, b], axis=-1))


def ground(size: int = 512) -> Texture:
    """Neutral grain for the run-off apron.

    Greyscale, so it *multiplies* the strip's vertex colour instead of
    replacing it: one texture grains the paved run-off, the gravel trap and the
    verge alike, while the colour that says which is which stays in the mesh.
    Without it the apron is three flat bands of paint, and a gravel trap that
    reads as a flat tan polygon is the most plastic thing beside the track.

    Mean is about 0.82, not 1.0 -- a modulate texture cannot brighten, only
    darken, so the bands it multiplies are mixed brighter to compensate (see
    RUNOFF_GAIN in trackmesh).
    """
    rng = np.random.default_rng(11)
    fine = _noise(size, octaves=(64, 128), seed=11)
    broad = _noise(size, octaves=(3, 6), seed=12)
    speck = rng.random((size, size))
    t = 0.45 * fine + 0.35 * broad + 0.20 * speck
    t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
    v = 255.0 * (0.62 + 0.38 * t)
    return _tex(np.stack([v, v, v], axis=-1))


def kerb(size: int = 64) -> Texture:
    """Red/white blocks running along U (lengthwise); one full pair per repeat."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    half = size // 2
    arr[:, :half] = (198, 28, 28)
    arr[:, half:] = (236, 236, 236)
    arr = arr.astype(np.float64) * (0.82 + 0.18 * _noise(size, seed=5)[..., None])
    return _tex(arr)


def checker(size: int = 256, squares: int = 8) -> Texture:
    step = size // squares
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(squares):
        for j in range(squares):
            if (i + j) % 2 == 0:
                arr[i * step:(i + 1) * step, j * step:(j + 1) * step] = 245
    return _tex(arr)


def car_body(size: int = 128) -> Texture:
    n = _noise(size, octaves=(16, 32), seed=7)
    base = np.stack([210 + 20 * n, 30 + 10 * n, 40 + 10 * n], axis=-1)
    return _tex(base)
