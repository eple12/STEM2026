"""World shaders: physically based sunlight, sky reflections, shadows, haze.

Everything here works in **linear light** and writes HDR values into the
camera's float buffer (post.py develops them). That one decision is what most
of the realism rests on: colours picked in sRGB are converted to linear before
any lighting happens, the sun is several times brighter than the sky instead
of both being squeezed under 1.0, and highlights are allowed to be as bright
as they really are and are rolled off by the tone curve rather than clipped.

The model, per pixel:

    diffuse   albedo * (sun * N.L * shadow + sky ambient + ground bounce)
    specular  GGX microfacet highlight of the sun, Schlick Fresnel
    reflect   the analytic sky in the reflected direction, Fresnel-weighted,
              blurred towards the ambient as the surface gets rougher
    coat      car paint: a second, near-mirror layer over the base
    haze      distance fog towards the sky's own horizon colour, brighter
              looking into the sun

The reflection is the part a racing game lives on. Bodywork is read almost
entirely from what it reflects -- the bright band of the horizon sliding over
the sidepods as the car turns -- and a car without it is a painted toy however
good its model is. The sky that is reflected is the same function the sky dome
draws (``SKY_GLSL``), so the two can never disagree.

Variants are built from one source with ``#define``s rather than with uniform
branches: an alpha-tested shader cannot use early depth rejection on most
GPUs, and only foliage needs the test.

    world   everything by default
    foliage alpha-tested cards, light through the leaves
    road    the asphalt: lap-space rubber/dust map times world-space grain
    ground  grass and run-off: mown stripes on anything green

``sunset_shader`` is kept as the name of the default variant, because every
caller in the codebase imports it under that name.
"""
from ursina import Vec2, Vec3, Vec4
from ursina.shader import Shader

# ---------------------------------------------------------------------------
# The sky, as a function. Shared by the dome and by every reflection.
# ---------------------------------------------------------------------------
SKY_GLSL = """
uniform vec3 sky_zenith;       // linear radiance straight up
uniform vec3 sky_horizon;      // ...and at the horizon
uniform vec3 sky_ground;       // the world below the horizon, as reflected
uniform vec3 sun_vec;          // world direction TOWARDS the sun, normalised
uniform vec3 sun_color;        // direct beam irradiance, linear
uniform vec3 glow_color;       // forward-scattered light round the sun
uniform float glow_strength;

vec3 sky_radiance(vec3 d) {
    float y = d.y;
    // Rayleigh-ish: a thin bright band at the horizon, deepening overhead.
    float t = pow(clamp(y, 0.0, 1.0), 0.42);
    vec3 c = mix(sky_horizon, sky_zenith, t);
    // Mie: the haze round the sun. Two lobes -- a tight bright one and a
    // broad one that warms that whole side of the sky.
    float mu = max(dot(d, sun_vec), 0.0);
    float horizon_boost = 1.0 + 1.5 * (1.0 - clamp(y * 4.0, 0.0, 1.0));
    c += glow_color * glow_strength * (pow(mu, 6.0) * 0.45 * horizon_boost
                                       + pow(mu, 48.0) * 1.6);
    // Below the horizon: the ground, fading in over a few degrees so a
    // reflection of the horizon is a soft line rather than a hard one.
    c = mix(c, sky_ground, smoothstep(0.0, -0.08, y));
    return c;
}
"""

# ---------------------------------------------------------------------------
_VERT = """#version 150

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat3 p3d_NormalMatrix;

in vec4 vertex;
in vec3 normal;
in vec4 p3d_Color;
in vec2 p3d_MultiTexCoord0;

uniform vec2 texture_scale;
uniform vec2 texture_offset;

out vec2 texcoords;
out vec2 raw_uv;
out vec4 vertex_color;
out vec3 view_position;
out vec3 view_normal;
out vec3 world_position;
out vec3 world_normal;
#ifdef CAR
out vec3 obj_pos;
out vec3 obj_normal;
#endif

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * vertex;
#ifdef CAR
    obj_pos = vertex.xyz;
    obj_normal = normal;
#endif
    view_position = vec3(p3d_ModelViewMatrix * vertex);
    view_normal = normalize(p3d_NormalMatrix * normal);
    world_position = (p3d_ModelMatrix * vertex).xyz;
    world_normal = normalize(mat3(p3d_ModelMatrix) * normal);
    texcoords = (p3d_MultiTexCoord0 * texture_scale) + texture_offset;
    raw_uv = p3d_MultiTexCoord0;
    vertex_color = p3d_Color;
}
"""

_FRAG = """#version 150
uniform struct {
    vec4 position;
} p3d_LightSource[1];

uniform sampler2D p3d_Texture0;
uniform vec4 p3d_ColorScale;
uniform mat4 p3d_ViewMatrixInverse;

in vec2 texcoords;
in vec2 raw_uv;
in vec4 vertex_color;
in vec3 view_position;
in vec3 view_normal;
in vec3 world_position;
in vec3 world_normal;

""" + SKY_GLSL + """
uniform vec3 ambient_sky;      // irradiance from the sky dome, upward faces
uniform vec3 ambient_ground;   // ...bounced off the ground, downward faces
uniform float sun_wrap;
uniform float env_strength;    // global scale on sky reflections

// Per entity: roughness, specular level (F0 / 0.04), clearcoat, unused.
uniform vec4 material;

uniform vec4 haze_color;       // rgb unused (the sky supplies it), a = max density
uniform float haze_start;
uniform float haze_end;
uniform float direct_out;      // 1: no post chain, develop the colour here

// The car's own shadow map: drawn every frame from a camera that rides with
// the car (lighting.CarShadow), holding nothing but the car.
uniform sampler2DShadow car_map;
// World -> the car map's clip space, from the shadow camera itself
// (setShaderInput("carcam", its NodePath)): Panda works it out at draw time
// from the same frame's scene graph the map was drawn from. Written by hand
// into an array instead, it ran a frame ahead of the drawing once cull and
// draw went to threads of their own -- the shadow lagged and shook.
uniform mat4 trans_world_to_clip_of_carcam;
uniform float shadow_bias;
uniform float shadow_blur;
uniform int shadow_samples;
uniform float shadow_strength;
uniform vec4 wspos_carcam;    // the car (its shadow camera), in world space
uniform float shadow_fade_start;
uniform float car_normal_offset;

// The other cars' shadows (lighting.FieldShadow): a wider map round the car
// on camera, faded out towards its edge.
uniform sampler2DShadow field_map;
uniform mat4 trans_world_to_clip_of_fieldcam;
uniform float field_on;
uniform float field_bias;
uniform float field_blur;
uniform float field_normal_offset;

uniform sampler2DShadow bake_map;
uniform vec3 bake_o;
uniform vec3 bake_ex;
uniform vec3 bake_ey;
uniform vec3 bake_ez;
uniform float bake_normal_offset;
uniform float bake_ready;
uniform float bake_bias;
uniform float bake_blur;
uniform int bake_samples;

#ifdef ROAD
uniform sampler2D lap_map;     // u: fraction of the lap, v: across the road
uniform sampler2D detail_map;  // world-space grain, tiles every few metres
uniform float lap_length;
#endif
#ifdef GROUND
uniform sampler2D detail_map;
uniform vec4 mow;              // stripe direction xz, stripe width, depth
#endif
#ifdef FOLIAGE
uniform float alpha_cut;
#endif
#ifdef CROWD
float h21(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}
#endif
#ifdef CAR
// Which part of the car this geom is, and the car's colours. The livery is
// painted here, per pixel, from the position on the car: a baked vertex
// colour per part cannot draw a crisp line across a panel, and a livery is
// nothing but crisp lines.
uniform float part;            // 0 other, 1 paint, 2 carbon, 3 tyre, 4 rim, 5 helmet
uniform vec3 livery_a;         // primary (team colour), sRGB; x < 0: the
                               // model's own paint
uniform vec3 livery_b;         // secondary, sRGB
uniform vec3 livery_c;         // accent, sRGB
in vec3 obj_pos;
in vec3 obj_normal;
float aa_step(float edge, float x) {
    float w = fwidth(x) * 0.8 + 1e-5;
    return smoothstep(edge - w, edge + w, x);
}
float band(float x, float a, float b) {
    return aa_step(a, x) * (1.0 - aa_step(b, x));
}
float weave(vec3 p, vec3 n) {
    // 2x2 twill at 1.4 cm, on whichever plane the face is closest to, and
    // faded out before it is small enough to shimmer.
    vec2 q = abs(n.y) > 0.6 ? p.xz : (abs(n.x) > 0.6 ? p.zy : p.xy);
    q /= 0.014;
    float fade = 1.0 - smoothstep(0.25, 0.6, max(fwidth(q.x), fwidth(q.y)));
    float t = mod(floor(q.x) + floor(q.y * 0.5) * 2.0, 4.0) < 2.0 ? 1.0 : 0.0;
    float along = fract(t > 0.5 ? q.x : q.y);
    float w = 0.86 + 0.28 * t * (0.6 + 0.4 * sin(along * 3.14159));
    return mix(1.0, w, fade);
}
#endif

out vec4 fragment_color;

const float PI = 3.14159265;

float sample_car(vec3 nw) {
    vec3 wp = world_position + nw * car_normal_offset;
    vec3 c0 = (trans_world_to_clip_of_carcam * vec4(wp, 1.0)).xyz * 0.5 + 0.5;
    if (c0.x < 0.0 || c0.x > 1.0 || c0.y < 0.0 || c0.y > 1.0) return 1.0;
    c0.z -= shadow_bias;
    float total = 0.0;
    float n = float(shadow_samples);
    float half_blur = shadow_blur * 0.5;
    for (int x = 0; x < shadow_samples; ++x) {
        for (int y = 0; y < shadow_samples; ++y) {
            vec3 c = c0;
            c.x += (float(x) + 0.5) * shadow_blur / n - half_blur;
            c.y += (float(y) + 0.5) * shadow_blur / n - half_blur;
            total += texture(car_map, c);
        }
    }
    return total / (n * n);
}

float sample_field(vec3 nw) {
    vec3 wp = world_position + nw * field_normal_offset;
    if (field_on < 0.5) return 1.0;
    vec3 c0 = (trans_world_to_clip_of_fieldcam * vec4(wp, 1.0)).xyz * 0.5 + 0.5;
    vec2 e = abs(c0.xy - 0.5);
    float edge = max(e.x, e.y);
    if (edge > 0.5) return 1.0;
    c0.z -= field_bias;
    float h = field_blur * 0.25;
    float s = texture(field_map, c0 + vec3(-h, -h, 0.0))
            + texture(field_map, c0 + vec3( h, -h, 0.0))
            + texture(field_map, c0 + vec3(-h,  h, 0.0))
            + texture(field_map, c0 + vec3( h,  h, 0.0));
    // Fade out over the outer part of the film rather than cut off at it.
    return mix(s * 0.25, 1.0, smoothstep(0.36, 0.48, edge));
}

float sample_bake(vec3 nw) {
    vec3 wp = world_position + nw * bake_normal_offset;
    vec3 coord = bake_o + bake_ex * wp.x + bake_ey * wp.y + bake_ez * wp.z;
    coord.z += bake_bias;
    float total = 0.0;
    float n = float(bake_samples);
    float half_blur = bake_blur * 0.5;
    for (int x = 0; x < bake_samples; ++x) {
        for (int y = 0; y < bake_samples; ++y) {
            vec3 c = coord;
            c.x += float(x) * bake_blur / n - half_blur;
            c.y += float(y) * bake_blur / n - half_blur;
            total += texture(bake_map, c);
        }
    }
    return total / (n * n);
}

vec3 to_linear(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }

// GGX / Trowbridge-Reitz with the Smith-Schlick visibility term.
float ggx(float ndh, float a) {
    float a2 = a * a;
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 1e-6);
}
float vis(float ndl, float ndv, float a) {
    float k = a * 0.5;
    return 0.25 / ((ndl * (1.0 - k) + k) * (ndv * (1.0 - k) + k) + 1e-5);
}
vec3 fresnel(vec3 f0, float vdh) {
    return f0 + (1.0 - f0) * pow(1.0 - vdh, 5.0);
}
vec3 fresnel_rough(vec3 f0, float ndv, float rough) {
    return f0 + (max(vec3(1.0 - rough), f0) - f0) * pow(1.0 - ndv, 5.0);
}

void main() {
    vec4 base = texture(p3d_Texture0, texcoords) * p3d_ColorScale * vertex_color;
#ifdef FOLIAGE
    // Mipmapping averages a leafy edge towards half-transparent, and an
    // alpha test then eats it: trees thin out and vanish with distance.
    // Sharpen the alpha by how far down the mip chain this pixel reads.
    vec2 tsz = vec2(textureSize(p3d_Texture0, 0));
    vec2 ddx = dFdx(texcoords * tsz);
    vec2 ddy = dFdy(texcoords * tsz);
    float lod = max(0.0, 0.5 * log2(max(dot(ddx, ddx), dot(ddy, ddy))));
    base.a *= 1.0 + lod * 0.30;
    // The flat card on top of a tree is for looking down on it. Seen from
    // the side it is a sliver smeared across the crown: fade it out as the
    // view goes edge-on.
    if (abs(world_normal.y) > 0.95) {
        vec3 vv = normalize(p3d_ViewMatrixInverse[3].xyz - world_position);
        base.a *= smoothstep(0.25, 0.55, abs(vv.y));
    }
    if (base.a < alpha_cut) discard;
    base.a = 1.0;
#endif
    vec3 albedo = to_linear(base.rgb);
    float rough = clamp(material.x, 0.03, 1.0);
    float spec_level = material.y;
#ifdef CROWD
    {
        // Grandstand seating. The stand model's seat faces are one flat red,
        // which reads as a painted ramp; real terracing is rows of moulded
        // seats -- a lit seat back, the shadowed step in front of the next
        // row, a hairline between each seat -- with the odd faded or
        // replaced one. Drawn per pixel from the face's own position, and
        // settled to its average tone before the rows get small enough to
        // shimmer.
        vec3 sc = base.rgb;
        float seat = step(0.42, sc.r) * step(sc.g * 1.6, sc.r)
                   * step(sc.b * 1.6, sc.r);
        if (seat > 0.5) {
            vec3 nwc = normalize(world_normal);
            vec3 h = cross(vec3(0.0, 1.0, 0.0), nwc);
            vec2 uv2;
            if (length(h) < 0.3) {
                uv2 = world_position.xz / vec2(0.50, 0.80);
            } else {
                h = normalize(h);
                uv2 = vec2(dot(world_position, h) / 0.50,
                           world_position.y / 0.42);
            }
            vec2 cell = floor(uv2);
            vec2 f = fract(uv2);
            float back = smoothstep(0.10, 0.22, f.y) * (1.0 - smoothstep(0.78, 0.92, f.y));
            float gap = smoothstep(0.0, 0.06, f.x) * (1.0 - smoothstep(0.94, 1.0, f.x));
            float shade = mix(0.42, 1.0, back) * mix(0.75, 1.0, gap);
            // A seat back is curved: lighter at the top edge.
            shade *= 0.88 + 0.22 * smoothstep(0.3, 0.75, f.y);
            float worn = 0.9 + 0.2 * h21(cell);
            vec3 c = sc * 0.82 * shade * worn;
            float fw = max(fwidth(uv2.x), fwidth(uv2.y));
            c = mix(c, sc * 0.62, smoothstep(0.35, 0.9, fw));
            albedo = to_linear(c);
            rough = mix(0.45, 0.85, smoothstep(0.35, 0.9, fw));
        }
    }
#endif

    vec3 N = normalize(view_normal);
    vec3 nw = normalize(world_normal);
#ifdef FOLIAGE
    // The card normals point out of the crown; bend them towards the eye so
    // the middle of a card faces the viewer the way the front of a real
    // canopy does.
    N = normalize(N + normalize(-view_position) * 0.55);
#endif
    vec3 cam_world = p3d_ViewMatrixInverse[3].xyz;
    vec3 Vw = normalize(cam_world - world_position);

#ifdef ROAD
    // Lap space: rubber down the racing line, dust off it, patches along it.
    // Stored at half intensity, so it can brighten as well as darken.
    vec3 lapc = texture(lap_map, vec2(raw_uv.x / lap_length, raw_uv.y)).rgb * 2.0;
    float grain = texture(detail_map, world_position.xz * 0.21).r;
    float fine = texture(detail_map, world_position.xz * 1.37).r;
    albedo *= lapc * (0.70 + 0.60 * grain) * (0.86 + 0.28 * fine);
    // Rubbered-in tarmac is smoother; the dusty margins are rougher.
    float rubber = clamp((1.0 - lapc.g) * 2.5, 0.0, 1.0);
    rough = clamp(rough - 0.22 * rubber + 0.10 * (fine - 0.5), 0.25, 1.0);
#endif
#ifdef GROUND
    float g = texture(detail_map, world_position.xz * 0.17).r;
    albedo *= 0.80 + 0.40 * g;
    // Mown stripes, only on what is green, faded out before they alias.
    float green = clamp((base.g - max(base.r, base.b)) * 6.0, 0.0, 1.0);
    float dist = length(view_position);
    float along = dot(world_position.xz, mow.xy) / mow.z;
    float band = smoothstep(-0.15, 0.15, sin(along * PI));
    float fade = 1.0 - smoothstep(120.0, 420.0, dist);
    albedo *= 1.0 + mow.w * (band - 0.5) * 2.0 * green * fade;
#endif

#ifdef CAR
    float coat_k = 1.0;
    {
        vec3 p = obj_pos;
        vec3 on = normalize(obj_normal);
        const vec3 CARBON = vec3(0.012, 0.012, 0.014);
        if (part > 0.5 && part < 1.5) {
            // Paint. Black carbon below the sidepods' undercut and on the
            // wing endplates; the secondary colour over the engine cover and
            // the nose tip; a thin accent line sweeping down the flank.
            float lower = 1.0 - aa_step(0.20 + 0.025 * p.z, p.y);
            float plates = aa_step(0.98, abs(p.x))
                         * max(aa_step(1.45, p.z), 1.0 - aa_step(-1.85, p.z));
            float cover = aa_step(0.62 - 0.07 * p.z, p.y)
                        * (1.0 - aa_step(0.25, p.z)) * aa_step(-2.0, p.z);
            float nose = aa_step(1.70, p.z) * (1.0 - aa_step(0.95, abs(p.x)));
            float line_ = band(p.y - (0.43 + 0.075 * p.z), -0.013, 0.013)
                        * aa_step(-1.6, p.z) * (1.0 - aa_step(1.2, p.z))
                        * aa_step(0.22, abs(p.x));
            vec3 c = livery_a.x < 0.0 ? albedo : to_linear(livery_a);
            c = mix(c, to_linear(livery_b), max(cover, nose));
            c = mix(c, to_linear(livery_c), line_);
            float carb = max(lower, plates);
            c = mix(c, CARBON * weave(p, on), carb);
            albedo = c;
            coat_k = 1.0 - 0.45 * carb;
            rough = mix(rough, 0.32, carb);
        } else if (part > 1.5 && part < 2.5) {
            albedo = CARBON * 1.6 * weave(p, on);
        } else if (part > 2.5 && part < 3.5) {
            // Tyre: matt tread, a slightly satin sidewall.
            float side = step(0.55, abs(on.x));
            albedo = vec3(0.022, 0.021, 0.020);
            rough = mix(0.92, 0.66, side);
        } else if (part > 3.5 && part < 4.5) {
            albedo = vec3(0.035, 0.036, 0.040);
        } else if (part > 4.5) {
            // Helmet, in the sphere's own space: a dark visor across the
            // front, the accent colour with a stripe over the crown.
            float visor = aa_step(0.10, p.z) * band(p.y, -0.04, 0.15);
            float stripe = band(p.x, -0.07, 0.07) * aa_step(0.0, p.y);
            vec3 c = mix(to_linear(livery_c), to_linear(livery_b), stripe);
            albedo = mix(c, vec3(0.01, 0.012, 0.016), visor);
            rough = mix(rough, 0.06, visor);
        }
    }
#endif

    // --- sun -------------------------------------------------------------
    vec3 L = normalize(p3d_LightSource[0].position.xyz
                       - view_position * p3d_LightSource[0].position.w);
    vec3 V = normalize(-view_position);
    float ndl_raw = dot(N, L);
    float ndl = clamp(ndl_raw, 0.0, 1.0);
    float ndv = clamp(abs(dot(N, V)), 1e-3, 1.0);
    float wrap = clamp((ndl_raw + sun_wrap) / (1.0 + sun_wrap), 0.0, 1.0);

    // The car's shadow only near the car; everything else is the baked map.
#ifdef FOLIAGE
    // Trees are a wall of alpha-tested cards, many deep: the most fragments
    // on screen, and almost never under a car. The baked map alone.
    float near_s = 1.0;
#else
    float sd = length(world_position.xz - wspos_carcam.xz);
    float near_s = (sd < shadow_fade_start) ? sample_car(nw) : 1.0;
    near_s = min(near_s, sample_field(nw));
#endif
    float s = (bake_ready > 0.5) ? min(near_s, sample_bake(nw)) : near_s;
    float shadow = mix(1.0, s, shadow_strength);

    // --- ambient ---------------------------------------------------------
    float up = clamp(nw.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 ambient = mix(ambient_ground, ambient_sky, up);
    // The sky is brighter on the sun's side; vertical faces turned to it
    // pick that up, which keeps shaded walls from being one flat value.
    vec2 flat_n = nw.xz;
    float horiz = length(flat_n);
    if (horiz > 1e-4) {
        float toward = max(dot(flat_n / horiz, normalize(sun_vec.xz)), 0.0);
        ambient += glow_color * (toward * horiz * glow_strength * 0.35);
    }

#ifdef FOLIAGE
    // Leaves pass light. Seen against the sun a canopy glows at its edges.
    float back = pow(clamp(dot(-Vw, sun_vec), 0.0, 1.0), 3.0);
    vec3 trans = albedo * sun_color * back * 0.55 * shadow;
    // A canopy is lit through and between its own leaves: even its shaded
    // side is never the black a solid object's would be.
    float leaf_wrap = clamp((ndl_raw + 0.6) / 1.6, 0.0, 1.0);
    vec3 diffuse = albedo * (sun_color * mix(wrap, leaf_wrap, 0.6) * shadow
                             + ambient * 1.45 + sun_color * 0.10) + trans;
#else
    vec3 diffuse = albedo * (sun_color * wrap * shadow + ambient);
#endif

    // --- specular --------------------------------------------------------
    vec3 f0 = vec3(0.04 * spec_level);
    float a = rough * rough;
#ifdef FOLIAGE
    // A leaf card is rough (0.85) and lit through: its gloss and its sky
    // reflection came to a couple of percent of the colour, and the trees
    // are the most fragments on screen (~1.3 ms of a 1080p frame). Diffuse
    // and the light through the leaves only.
    vec3 lit = diffuse;
#else
    vec3 H = normalize(L + V);
    float ndh = clamp(dot(N, H), 0.0, 1.0);
    float vdh = clamp(dot(V, H), 0.0, 1.0);
    vec3 spec = ggx(ndh, a) * vis(ndl, ndv, a) * fresnel(f0, vdh)
                * sun_color * ndl * shadow * PI;

    // Reflection of the sky. Rough surfaces reflect a blurred sky, which is
    // the ambient; smooth ones the sky itself. Occluded where the sun is
    // shadowed, weakly -- a cheap stand-in for the world in the way.
    vec3 Rw = reflect(-Vw, nw);
    vec3 env = mix(sky_radiance(Rw), mix(ambient_ground, ambient_sky,
                                         clamp(Rw.y * 0.5 + 0.5, 0.0, 1.0)),
                   clamp(rough * 1.15, 0.0, 1.0));
    vec3 fenv = fresnel_rough(f0, ndv, rough);
    vec3 refl = env * fenv * env_strength * mix(0.55, 1.0, shadow);
    vec3 lit = diffuse * (1.0 - fenv) + spec + refl;

    // Clearcoat: a lacquer over the base, sharp and Fresnel-weighted.
    float coat = material.z;
#ifdef CAR
    coat *= coat_k;
#endif
    if (coat > 0.0) {
        float ac = 0.035 * 0.035;
        vec3 fc = fresnel(vec3(0.04), vdh);
        vec3 coat_spec = ggx(ndh, ac) * vis(ndl, ndv, ac) * fc * sun_color
                         * ndl * shadow * PI;
        float fcv = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
        vec3 coat_env = sky_radiance(Rw) * fcv * env_strength
                        * mix(0.55, 1.0, shadow);
        lit = lit * (1.0 - fcv * coat) + (coat_spec + coat_env) * coat;
    }
#endif

    // --- haze ------------------------------------------------------------
    float d = length(view_position);
    float t = clamp((d - haze_start) / max(haze_end - haze_start, 1.0), 0.0, 1.0);
    vec3 view_dir = -Vw;
    vec3 haze = sky_radiance(normalize(vec3(view_dir.x, max(view_dir.y, 0.02),
                                            view_dir.z)));
    lit = mix(lit, haze, (1.0 - exp(-t * 3.0)) / (1.0 - exp(-3.0)) * haze_color.a);

    if (direct_out > 0.5) {
        // No camera chain: a plain filmic curve and the display gamma here.
        lit = lit / (lit + 0.7) * 1.2;
        lit = pow(clamp(lit, 0.0, 1.0), vec3(1.0 / 2.2));
    }
    fragment_color = vec4(lit, base.a);
}
"""


def _variant(name: str, *defines: str) -> Shader:
    head = "".join(f"#define {d}\n" for d in defines)
    frag = _FRAG.replace("#version 150\n", "#version 150\n" + head, 1)
    vert = _VERT.replace("#version 150\n", "#version 150\n" + head, 1)
    defaults = {
        # ONLY what genuinely differs per entity lives here.
        #
        # Ursina writes every default_input onto the entity when the shader is
        # assigned, and a shader input set on an entity overrides one inherited
        # from an ancestor. So anything listed here can no longer be driven
        # from ``scene`` -- lighting.Sunset sets all of those on ``scene``.
        'texture_scale': Vec2(1, 1),
        'texture_offset': Vec2(0, 0),
        'material': Vec4(0.80, 1.0, 0.0, 0.0),
    }
    if "FOLIAGE" in defines:
        defaults['alpha_cut'] = 0.5
        defaults['material'] = Vec4(0.85, 0.6, 0.0, 0.0)
    if "CAR" in defines:
        defaults['part'] = 0.0
    return Shader(language=Shader.GLSL, name=name, vertex=vert, fragment=frag,
                  default_input=defaults)


world_shader = _variant("world_shader")
foliage_shader = _variant("foliage_shader", "FOLIAGE")
road_shader = _variant("road_shader", "ROAD")
ground_shader = _variant("ground_shader", "GROUND")
car_shader = _variant("car_shader", "CAR")
crowd_shader = _variant("crowd_shader", "CROWD")

#: The name every caller imports. Kept so the roadside, the cars and the
#: mountains keep working untouched.
sunset_shader = world_shader


def material(roughness: float, specular: float = 1.0, clearcoat: float = 0.0):
    """Per-entity material, as the shader's ``material`` uniform."""
    return Vec4(roughness, specular, clearcoat, 0.0)


# ---------------------------------------------------------------------------
# The sky dome.
# ---------------------------------------------------------------------------
_SKY_VERT = """#version 150
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat4 p3d_ViewMatrixInverse;
in vec4 vertex;
out vec3 dir;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * vertex;
    // The dome rides on the camera, so the direction to a vertex from the
    // camera is the view ray through it.
    dir = (p3d_ModelMatrix * vertex).xyz - p3d_ViewMatrixInverse[3].xyz;
}
"""

_SKY_FRAG = """#version 150
in vec3 dir;
out vec4 fragment_color;
""" + SKY_GLSL + """
uniform sampler2D cloud_map;
uniform vec4 clouds;           // coverage, softness, scale, brightness
uniform vec2 cloud_drift;
uniform float sun_disc;        // angular radius, cos
uniform float direct_out;

float density(vec2 p) {
    float a = texture(cloud_map, p).r;
    float b = texture(cloud_map, p * 2.7 + vec2(0.31, 0.77)).g;
    float n = a * 0.72 + b * 0.38;
    return smoothstep(clouds.x, clouds.x + clouds.y, n);
}

void main() {
    vec3 d = normalize(dir);
    vec3 c = sky_radiance(d);

    // The sun itself: a hot disc with a little limb darkening. Far over 1,
    // so the bloom builds its own glare round it.
    float mu = dot(d, sun_vec);
    float disc = smoothstep(sun_disc - 0.00006, sun_disc + 0.00002, mu);
    c += sun_color * 30.0 * disc;

    // Clouds: a layer at altitude, so they foreshorten into the horizon the
    // way real ones do. Lit from the sun's side by looking a little way
    // towards it -- more cloud there means this point is in its shadow.
    if (d.y > 0.0) {
        vec2 p = d.xz / (d.y + 0.06) * clouds.z + cloud_drift;
        float m = density(p);
        if (m > 0.001) {
            vec2 toward = normalize(sun_vec.xz + 1e-4) * 0.035;
            float occ = density(p + toward) * 0.6 + density(p + toward * 2.0) * 0.4;
            float lit = 1.0 - 0.62 * occ;
            vec3 cloud = sun_color * (0.20 + 0.30 * lit) * clouds.w
                         + sky_zenith * 1.1 + sky_horizon * 0.35;
            // Silver lining towards the sun.
            cloud += glow_color * pow(max(mu, 0.0), 10.0) * 1.6 * (1.0 - m);
            // Into the haze at the horizon.
            float fade = smoothstep(0.0, 0.22, d.y);
            c = mix(c, cloud, m * fade * 0.94);
        }
    }
    if (direct_out > 0.5) {
        c = c / (c + 0.7) * 1.2;
        c = pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
    }
    fragment_color = vec4(c, 1.0);
}
"""

sky_shader = Shader(language=Shader.GLSL, name="sky_shader",
                    vertex=_SKY_VERT, fragment=_SKY_FRAG,
                    default_input={})

# Deliberately no ``continuous_input`` anywhere here. Ursina applies those by
# looping over every entity and calling set_shader_input once each, every
# frame -- with the car's wheels and hubs that came to 117 calls a frame,
# about 2 ms, plus the render-state churn it fed into Panda's state garbage
# collector. Panda composes shader inputs down the scene graph, so lighting.py
# sets the per-frame ones on ``scene`` instead: two calls, not two hundred.
