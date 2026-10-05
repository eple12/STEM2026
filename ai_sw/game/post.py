"""The camera: HDR scene buffer, then one full-screen pass that develops it.

The world shaders (shaders.py) write *linear* light, with no ceiling: a sunlit
white wall is around 3, the sun's highlight on bodywork 20 or more. Nothing on
a monitor can show that, and squashing it is exactly what a camera does --
which is most of what separates a rendered frame that reads as photographed
from one that reads as a game from 2005. So the scene goes into a 16-bit float
buffer, and a single pass here does what a broadcast camera chain does to it:

* **Bloom.** Light that is far brighter than the screen bleeds into its
  surroundings in a real lens. Taken from the buffer's own mipmaps rather than
  from a chain of blur passes: the GPU builds the mip pyramid for free on a
  render texture, and four bilinear taps at each of four levels is a wide,
  smooth glow in sixteen lookups with no extra buffers and no extra draws.
* **Tone mapping.** ACES (Hill's fit): the highlight roll-off and the
  saturation fall-off of film, the curve every modern racing game develops its
  frame with.
* **Grade.** A touch of contrast and saturation, cooler shadows and warmer
  highlights -- the broadcast look -- then vignette and a little grain, which
  also hides the banding an 8-bit sky gradient would otherwise show.
* **Speed.** At pace, the edges of the frame smear radially. It is where the
  eye reads speed from, and it costs
  nothing at a standstill: the taps are skipped.

  (There used to be a touch of chromatic aberration as well. It re-sampled red
  and blue from the unblurred frame on top of the blurred one, so at speed
  every edge grew a green and magenta fringe. Gone.)

The buffer is multisampled, so edges are antialiased before any of this runs.
The HUD is untouched: Ursina draws it in its own display region after this.

One draw call and one buffer on top of what was there, and all of it on the
GPU. On the machines this was tuned on the game is CPU-bound, so the frame
time does not move -- see tools/bench_fps.py --ab-post.
"""
from __future__ import annotations

import builtins

from panda3d.core import (PTA_LVecBase4f, FrameBufferProperties,
                          LVecBase4f, SamplerState, Shader, Texture, Vec4)

from . import config

_VERT = """#version 150
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 uv;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    uv = p3d_MultiTexCoord0;
}
"""

_FRAG = """#version 150
uniform sampler2D scene;
uniform vec4 grade;        // exposure, contrast, saturation, vignette
uniform vec4 bloom;        // strength, threshold, knee, unused
uniform vec4 lens;         // speed blur 0..1, unused, grain, time
uniform vec3 shadow_tint;  // multiplies the low end
uniform vec3 high_tint;    // multiplies the high end
uniform float sharpen;     // unsharp amount, when the scene is upscaled
in vec2 uv;
out vec4 o_color;

// ACES, Stephen Hill's fit of the RRT + ODT.
const mat3 ACES_IN = mat3(0.59719, 0.07600, 0.02840,
                          0.35458, 0.90834, 0.13383,
                          0.04823, 0.01566, 0.83777);
const mat3 ACES_OUT = mat3(1.60475, -0.10208, -0.00327,
                           -0.53108, 1.10813, -0.07276,
                           -0.07367, -0.00605, 1.07602);
vec3 aces(vec3 v) {
    v = ACES_IN * v;
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return clamp(ACES_OUT * (a / b), 0.0, 1.0);
}

vec3 bright(vec3 c) {
    // Soft-knee threshold, so the glow fades in rather than switching on.
    float l = max(c.r, max(c.g, c.b));
    float k = bloom.z;
    float s = clamp(l - bloom.y + k, 0.0, 2.0 * k);
    s = s * s / (4.0 * k + 1e-4);
    return c * max(s, l - bloom.y) / max(l, 1e-4);
}

float hash(vec2 p) {
    p = fract(p * vec2(443.897, 441.423));
    p += dot(p, p.yx + 19.19);
    return fract((p.x + p.y) * p.x);
}

void main() {
    vec2 size = vec2(textureSize(scene, 0));
    vec2 texel = 1.0 / size;
    vec2 d = uv - 0.5;
    float r = length(d * vec2(size.x / size.y, 1.0));

    vec3 c;
    float blur = lens.x;
    if (blur > 0.01) {
        // Radial smear, held off the centre of the frame where the eye is.
        float m = smoothstep(0.18, 0.75, r) * blur;
        vec2 step_ = d * 0.016 * m;
        c = texture(scene, uv).rgb * 0.24;
        c += texture(scene, uv - step_ * 1.0).rgb * 0.20;
        c += texture(scene, uv - step_ * 2.0).rgb * 0.18;
        c += texture(scene, uv - step_ * 3.0).rgb * 0.15;
        c += texture(scene, uv - step_ * 4.0).rgb * 0.13;
        c += texture(scene, uv - step_ * 5.0).rgb * 0.10;
    } else {
        c = texture(scene, uv).rgb;
    }
    if (sharpen > 0.0) {
        // The scene is drawn below the window's resolution (render scale)
        // and stretched up: give back some of the edge the stretch blurs.
        vec3 n = texture(scene, uv + vec2(texel.x, 0.0)).rgb
               + texture(scene, uv - vec2(texel.x, 0.0)).rgb
               + texture(scene, uv + vec2(0.0, texel.y)).rgb
               + texture(scene, uv - vec2(0.0, texel.y)).rgb;
        c = max(c + (c - n * 0.25) * sharpen, vec3(0.0));
    }

#ifdef BLOOM
    // Bloom from the mip pyramid: three levels over the same span, four
    // bilinear taps each (four levels cost ~0.4 ms more at 1080p, unseen).
    vec3 glow = vec3(0.0);
    float wsum = 0.0;
    for (int k = 0; k < 3; ++k) {
        float lod = 2.0 + float(k) * 1.7;
        vec2 o = texel * exp2(lod) * 0.75;
        vec3 s = textureLod(scene, uv + vec2(o.x, o.y), lod).rgb
               + textureLod(scene, uv + vec2(-o.x, o.y), lod).rgb
               + textureLod(scene, uv + vec2(o.x, -o.y), lod).rgb
               + textureLod(scene, uv + vec2(-o.x, -o.y), lod).rgb;
        float w = 1.0 + float(k) * 0.52;
        glow += bright(s * 0.25) * w;
        wsum += w;
    }
    c += glow / wsum * bloom.x;
#endif

    c *= grade.x;
    c = aces(c);

    // Grade, in display space: split-tone, contrast round mid-grey, saturation.
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c *= mix(shadow_tint, high_tint, smoothstep(0.05, 0.75, l));
    c = clamp((c - 0.5) * grade.y + 0.5, 0.0, 1.0);
    l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = mix(vec3(l), c, grade.z);

    // Vignette, then sRGB.
    c *= 1.0 - grade.w * smoothstep(0.35, 1.05, r);
    c = pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
    // Grain after the curve, where it is even across the range.
    c += (hash(uv * size + lens.w) - 0.5) * lens.z;
    o_color = vec4(c, 1.0);
}
"""


_SHADERS: dict = {}


def _shader(bloom: bool):
    """One Shader object per variant for the life of the process: a fresh
    one per race would be a fresh render state per race."""
    if bloom not in _SHADERS:
        head = "#version 150\n"
        frag = _FRAG.replace(head, head + ("#define BLOOM\n" if bloom else ""), 1)
        _SHADERS[bloom] = Shader.make(Shader.SL_GLSL, _VERT, frag)
    return _SHADERS[bloom]


def render_scale(w: int, h: int) -> float:
    """The fraction of the window's resolution the 3D scene is drawn at:
    whole up to RENDER_MAX_PIXELS, then whatever keeps it to that many
    (times RENDER_SCALE). The HUD is drawn over it at the window's own."""
    s = config.RENDER_SCALE
    if config.RENDER_MAX_PIXELS and w * h > 0:
        s *= min(1.0, (config.RENDER_MAX_PIXELS / float(w * h)) ** 0.5)
    return max(0.4, min(1.0, s))


def _scaled_manager():
    """A FilterManager whose scene buffer follows render_scale -- through
    window resizes too, which is when FilterManager re-asks for sizes."""
    from direct.filter.FilterManager import FilterManager

    class ScaledFilterManager(FilterManager):
        scale = 1.0

        def getScaledSize(self, mul, div, align):
            x, y = FilterManager.getScaledSize(self, mul, div, align)
            s = render_scale(self.win.get_x_size(), self.win.get_y_size())
            self.scale = s
            if s < 0.999:
                x = max(1, int(round(x * s)))
                y = max(1, int(round(y * s)))
            return x, y

    return ScaledFilterManager


class PostFX:
    """Puts the scene through an HDR buffer and develops it on the way out."""

    def __init__(self):
        base = builtins.base
        self.manager = _scaled_manager()(base.win, base.cam)
        self.tex = Texture("scene_hdr")
        self.tex.set_wrap_u(SamplerState.WM_clamp)
        self.tex.set_wrap_v(SamplerState.WM_clamp)
        # Mipmapped, so the bloom can read pre-blurred levels; the driver
        # rebuilds the chain every frame the buffer is drawn.
        bloom = config.POST_BLOOM > 0.0
        self.tex.set_minfilter(SamplerState.FT_linear_mipmap_linear if bloom
                               else SamplerState.FT_linear)
        self.tex.set_magfilter(SamplerState.FT_linear)

        fb = FrameBufferProperties()
        bits = config.POST_BITS
        if bits == 11:
            # Packed float: half the bandwidth of RGBA16F, no alpha (the
            # chain never reads it).
            fb.set_float_color(True)
            fb.set_rgba_bits(11, 11, 10, 0)
        elif bits == 16:
            fb.set_float_color(True)
            fb.set_rgba_bits(16, 16, 16, 16)
        else:
            fb.set_rgba_bits(8, 8, 8, 8)
        fb.set_depth_bits(24)
        if config.POST_MSAA:
            fb.set_multisamples(config.POST_MSAA)
        self.quad = self.manager.renderSceneInto(colortex=self.tex, fbprops=fb)
        if self.quad is None and config.POST_MSAA:
            # No multisampled float target on this driver: plain float.
            fb.set_multisamples(0)
            self.quad = self.manager.renderSceneInto(colortex=self.tex,
                                                     fbprops=fb)
        self.ok = self.quad is not None
        if not self.ok:
            print("post: no HDR buffer; drawing straight to the window")
            return
        self.quad.set_shader(_shader(bloom))
        self.quad.set_shader_input("scene", self.tex)
        g = config.POST_GRADE
        self.quad.set_shader_input("grade", Vec4(g["exposure"], g["contrast"],
                                                 g["saturation"], g["vignette"]))
        self.quad.set_shader_input("bloom", Vec4(config.POST_BLOOM, config.POST_BLOOM_THRESHOLD,
                                                 config.POST_BLOOM_KNEE, 0.0))
        self.quad.set_shader_input("shadow_tint", g["shadow_tint"])
        self.quad.set_shader_input("high_tint", g["high_tint"])
        self.quad.set_shader_input("sharpen", 0.0)
        self._apply_sharpen()
        # Written in place every frame -- a PTA, not a fresh Vec4, so the
        # quad's render state is not rebuilt each time (see lighting.py).
        self._lens = PTA_LVecBase4f.empty_array(1)
        self._lens[0] = LVecBase4f(0.0, 0.0, config.POST_GRAIN, 0.0)
        self.quad.set_shader_input("lens", self._lens)
        self._t = 0.0

    def _apply_sharpen(self):
        s = self.manager.scale
        self._sharp_for = s
        self.quad.set_shader_input(
            "sharpen", config.RENDER_SHARPEN if s < 0.999 else 0.0)

    def update(self, speed_frac: float, dt: float):
        """Per frame: how much speed to put into the lens."""
        if not self.ok:
            return
        if self.manager.scale != self._sharp_for:
            self._apply_sharpen()           # the window was resized
        self._t = (self._t + 17.0 * dt) % 1000.0
        s = max(0.0, (speed_frac - config.POST_SPEED_FROM)
                / max(1e-6, 1.0 - config.POST_SPEED_FROM))
        blur = config.POST_SPEED_BLUR * s * s
        self._lens[0] = LVecBase4f(blur, 0.0, config.POST_GRAIN, self._t)

    def destroy(self):
        if getattr(self, "manager", None) is not None:
            self.manager.cleanup()
            # FilterManager listens for window-event and cleanup() does not
            # stop it, so the messenger kept every manager -- one per race --
            # alive for the life of the process.
            self.manager.ignoreAll()
            self.manager = None
            self.quad = None


# --- one camera, so one of these at most ------------------------------------
_FX: PostFX | None = None


def enable() -> bool:
    """Develop the 3D view through the HDR chain. Idempotent.

    Only while a circuit is on screen: the menu and the loading card are flat
    UI over the window's clear colour, and that colour put through a tone
    curve is not the colour that was picked for it.
    """
    global _FX
    from ursina import scene
    if config.POST_ENABLED and _FX is None:
        _FX = PostFX()
        if not _FX.ok:
            _FX.destroy()
            _FX = None
    # The world shaders write linear HDR for the chain, or develop their own
    # output when there is no chain to do it.
    scene.set_shader_input("direct_out", 0.0 if _FX is not None else 1.0)
    return _FX is not None


def disable():
    global _FX
    if _FX is not None:
        _FX.destroy()
        _FX = None


def update(speed_frac: float, dt: float):
    if _FX is not None:
        _FX.update(speed_frac, dt)
