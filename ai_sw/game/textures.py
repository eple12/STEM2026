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
    # Natural turf, not the Kelly green of a pitch in a video game: in
    # linear light a saturated green goes neon, and real grass under sun is
    # olive -- its red channel is most of its green.
    g = 84 + 40 * t + 14 * (speck - 0.5)
    r = 0.70 * g + 2 + 12 * (speck - 0.5)
    b = 0.40 * g + 4
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


# ---------------------------------------------------------------------------
# Filtered textures for the world shaders.
#
# Ursina's own Texture defaults to *nearest* filtering with no mipmaps, which
# is right for pixel art and wrong for everything a camera looks along: a
# grass plane or a road seen at a grazing angle turns into a crawling field of
# sparkle. Everything below is mipmapped and anisotropically filtered.
# ---------------------------------------------------------------------------
def panda_texture(arr: np.ndarray, name: str, repeat: bool = True,
                  aniso: int = 8):
    """A Panda texture from an (H, W, 3|4) or (H, W) array of 0..255.

    Row 0 of *arr* is the top of the image (V = 1), as with PIL.
    """
    from panda3d.core import SamplerState
    from panda3d.core import Texture as PTexture

    a = np.clip(arr, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    h, w, ch = a.shape
    tex = PTexture(name)
    fmt = PTexture.F_rgba8 if ch == 4 else PTexture.F_rgb8
    tex.setup_2d_texture(w, h, PTexture.T_unsigned_byte, fmt)
    # Panda stores BGR(A), bottom row first.
    order = [2, 1, 0, 3][:ch]
    tex.set_ram_image(np.ascontiguousarray(a[::-1, :, order]).tobytes())
    tex.set_minfilter(SamplerState.FT_linear_mipmap_linear)
    tex.set_magfilter(SamplerState.FT_linear)
    tex.set_anisotropic_degree(aniso)
    wrap = SamplerState.WM_repeat if repeat else SamplerState.WM_clamp
    tex.set_wrap_u(wrap)
    tex.set_wrap_v(wrap)
    return tex


def smooth_filtering(tex, aniso: int = 8):
    """Mipmaps and anisotropy on an Ursina texture made the default way."""
    from panda3d.core import SamplerState
    t = getattr(tex, "_texture", tex)
    t.set_minfilter(SamplerState.FT_linear_mipmap_linear)
    t.set_magfilter(SamplerState.FT_linear)
    t.set_anisotropic_degree(aniso)
    return tex


_DETAIL = None


def asphalt_detail(size: int = 512):
    """World-space grain for tarmac, greyscale, mean 0.5.

    Two things a road surface has at a metre's distance that flat paint does
    not: aggregate -- stones of a few millimetres, light and dark, which is
    what reads as "asphalt" up close -- and a broad mottle of a metre or two,
    the patching and wear that keeps a straight from looking like one plank.
    The R channel carries both; G carries a finer, more open grain the road
    shader uses for the roughness.
    """
    global _DETAIL
    if _DETAIL is not None:
        return _DETAIL
    rng = np.random.default_rng(71)
    stones = _noise(size, octaves=(96, 192, 256), seed=72)
    speck = rng.random((size, size))
    mottle = _noise(size, octaves=(2, 4, 8), seed=73)
    t = 0.40 * stones + 0.30 * speck + 0.30 * mottle
    t = (t - t.mean()) / max(t.std(), 1e-6)
    r = 0.5 + 0.16 * t
    fine = _noise(size, octaves=(64, 128, 256), seed=74)
    fine = 0.5 + 0.20 * (fine - fine.mean()) / max(fine.std(), 1e-6)
    img = np.stack([r, fine, r], axis=-1) * 255.0
    _DETAIL = panda_texture(img, "asphalt_detail")
    return _DETAIL


def _blur1d(a: np.ndarray, sigma: float, axis: int, wrap: bool) -> np.ndarray:
    """Gaussian blur along one axis; wrapped (the lap) or clamped (across)."""
    if sigma <= 0:
        return a
    k = int(max(1, round(3 * sigma)))
    x = np.arange(-k, k + 1)
    w = np.exp(-0.5 * (x / sigma) ** 2)
    w /= w.sum()
    mode = "wrap" if wrap else "edge"
    pad = [(0, 0)] * a.ndim
    pad[axis] = (k, k)
    ap = np.pad(a, pad, mode=mode)
    out = np.zeros_like(a, dtype=np.float64)
    m = a.shape[axis]
    for j, wt in enumerate(w):
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(j, j + m)
        out += wt * ap[tuple(sl)]
    return out


def lap_map(track, width: int = 4096, height: int = 64):
    """The road in lap space: U along the lap, V across it, left to right.

    A colour multiplier for the asphalt, stored at half intensity so it can
    brighten as well as darken. What it paints is what a real circuit's
    surface says about how it is driven:

    * **Rubber.** Two dark tracks a car's width apart, laid down exactly where
      the cars go. Taken from the circuit's recorded AI lap where there is one
      -- the line on the road is then literally the line the ghost drives --
      or from the imported racing line, or failing both the middle of the
      road.
    * **Braking.** Darker, heavier marks where the recorded lap was hard on
      the brakes, which is where they are on a real circuit: the approach to
      every slow corner.
    * **Dust.** Off the line the surface is untouched and paler, and the
      margins beside the white lines collect grit.
    * **Patches.** Broad changes of tone along the lap, as resurfaced
      sections are.
    """
    from . import config, replay

    n = track.count
    L = float(track.length)
    W, H = int(width), int(height)
    wl, wr = track.w_left, track.w_right
    rubber = np.zeros((H, W), np.float64)
    brake = np.zeros((H, W), np.float64)

    def splat(s, lat, i, acc, weight):
        u = (np.mod(s, L) / L * W).astype(np.int64) % W
        span = wl[i] + wr[i]
        v = ((lat + wl[i]) / np.maximum(span, 1e-3) * H).astype(np.int64)
        ok = (v >= 0) & (v < H)
        wt = np.broadcast_to(np.asarray(weight, dtype=np.float64), u.shape)
        np.add.at(acc, (v[ok], u[ok]), wt[ok])

    rec = replay.load(track.name, track)
    if rec is not None:
        fr = rec.frames.astype(np.float64)
        C = replay.C
        x, z, yaw = fr[:, C["x"]], fr[:, C["z"]], fr[:, C["yaw"]]
        idx = np.clip(fr[:, C["index"]].astype(np.int64), 0, n - 1)
        # Refine the recorded nearest sample by a step either way.
        for d in (-1, 1, -1, 1):
            j = (idx + d) % n
            dj = (track.center[j, 0] - x) ** 2 + (track.center[j, 1] - z) ** 2
            di = (track.center[idx, 0] - x) ** 2 + (track.center[idx, 1] - z) ** 2
            idx = np.where(dj < di, j, idx)
        half = config.WHEEL_HALF_TRACK
        right = np.stack([np.cos(yaw), -np.sin(yaw)], axis=1)
        br = np.clip(fr[:, C["brake"]], 0.0, 1.0)
        lat_g = np.abs(fr[:, C["lat_accel"]]) / 9.81
        # The recording is in time, so slow corners have more frames per
        # metre than straights and would come out blacker. Weight each frame
        # by the distance it stands for.
        speed = np.hypot(fr[:, C["vx"]], fr[:, C["vz"]])
        dist = speed / float(rec.hz)
        for side in (-1.0, 1.0):
            p = np.stack([x, z], axis=1) + right * (side * half)
            rel = p - track.center[idx]
            s = track.arclen[idx] + np.einsum("ij,ij->i", rel, track.tangent[idx])
            lat = np.einsum("ij,ij->i", rel, track.normal[idx])
            # A heavier deposit where the tyres are working hardest.
            splat(s, lat, idx, rubber,
                  dist * (1.0 + 0.8 * np.clip(lat_g - 1.2, 0.0, 2.0)))
            splat(s, lat, idx, brake, dist * br ** 2)
    else:
        from . import f1tenth
        off = f1tenth.load_raceline(track)
        if off is None:
            off = np.zeros(n)
        k = 8
        t = np.tile(np.linspace(0.0, 1.0, k, endpoint=False), n)
        i = np.repeat(np.arange(n), k)
        s = track.arclen[i] + t * track.seg_len[i]
        lat0 = np.repeat(off, k)
        for side in (-1.0, 1.0):
            splat(s, lat0 + side * config.WHEEL_HALF_TRACK, i, rubber, 1.0)

    # Spread: a tyre track is ~30 cm wide, and cars do not drive the same
    # centimetre every lap -- the band is a metre or so either side.
    texel_v = float(np.mean(wl + wr)) / H
    texel_u = L / W
    rubber = _blur1d(rubber, 3.0 / texel_u, 1, True)
    tight = _blur1d(rubber, 0.22 / texel_v, 0, False)
    wide = _blur1d(rubber, 1.00 / texel_v, 0, False)
    brake = _blur1d(_blur1d(brake, 2.0 / texel_u, 1, True), 0.18 / texel_v, 0,
                    False)

    def norm(a, q=99.0):
        top = np.percentile(a, q)
        return np.clip(a / max(top, 1e-12), 0.0, 1.0)

    rub = 0.55 * norm(tight) + 0.45 * norm(wide)
    brk = norm(brake, 99.3)

    v = (np.arange(H) + 0.5) / H
    edge = np.minimum(v, 1.0 - v)[:, None]            # 0 at a white line
    dust = 1.0 - np.clip(edge / 0.10, 0.0, 1.0)

    patch = _noise(256, octaves=(2, 4, 8), seed=6)[0]
    patch = np.interp(np.linspace(0, 255, W), np.arange(256), patch)
    patch = 1.0 + 0.06 * (patch - patch.mean()) / max(patch.std(), 1e-6)

    rng = np.random.default_rng(5)
    tone = patch[None, :] * (1.0 - 0.34 * rub - 0.40 * brk) \
        * (1.0 + 0.12 * dust) * (1.0 + 0.05 * (1.0 - rub))
    tone = tone + (rng.random((H, W)) - 0.5) * 0.02
    rgb = np.stack([tone * (1.0 + 0.03 * dust), tone,
                    tone * (1.0 - 0.03 * dust)], axis=-1)
    img = np.clip(rgb * 0.5 * 255.0, 0, 255)
    # Row 0 is V = 1 in panda_texture; V = 0 is the left edge.
    return panda_texture(img[::-1], f"lap_{track.name}", repeat=True)
