"""A sunset lighting model: warm sun, cool sky, shadows, haze.

Ursina's own ``lit_with_shadows_shader`` was the starting point, but it has no
ambient term of its own -- it tints the albedo by the light colour and calls
that the fill. At midday with a white sun that passes; at sunset with a deep
orange one, every surface goes orange including the ones facing away from it,
which is exactly backwards. In real low light the direct beam is warm *because*
the short wavelengths have been scattered out of it, and those wavelengths are
what is left lighting everything else -- so shadows and away-facing surfaces go
blue. Getting that one relationship right is most of what makes an evening
scene read as an evening scene.

So the model here is:

    colour = albedo * (sky ambient + bounce) + albedo * sun * N.L * shadow
             + specular
    then mixed toward the horizon haze with distance

with the ambient split by which way the surface points (hemispheric): up gets
the blue sky, down gets warm light bounced off the ground. It is a cheap stand
in for global illumination and costs two mixes.
"""
from ursina import Vec2, Vec3, color
from ursina.shader import Shader

sunset_shader = Shader(language=Shader.GLSL, name='sunset_shader', vertex='''#version 150
uniform struct {
    vec4 position;
    vec3 color;
    vec3 attenuation;
    vec3 spotDirection;
    float spotCosCutoff;
    float spotExponent;
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
} p3d_LightSource[1];

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
out vec4 vertex_color;
out vec3 view_position;
out vec3 view_normal;
out vec3 world_position;
out vec3 world_normal;
out vec4 shadow_coord;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * vertex;
    view_position = vec3(p3d_ModelViewMatrix * vertex);
    view_normal = normalize(p3d_NormalMatrix * normal);
    world_position = (p3d_ModelMatrix * vertex).xyz;
    world_normal = normalize(mat3(p3d_ModelMatrix) * normal);
    shadow_coord = p3d_LightSource[0].shadowViewMatrix * vec4(view_position, 1);
    texcoords = (p3d_MultiTexCoord0 * texture_scale) + texture_offset;
    vertex_color = p3d_Color;
}
''', fragment='''#version 150
uniform struct {
    vec4 position;
    vec3 color;
    vec3 attenuation;
    vec3 spotDirection;
    float spotCosCutoff;
    float spotExponent;
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
} p3d_LightSource[1];

uniform sampler2D p3d_Texture0;
uniform vec4 p3d_ColorScale;

in vec2 texcoords;
in vec4 vertex_color;
in vec3 view_position;
in vec3 view_normal;
in vec3 world_position;
in vec3 world_normal;
in vec4 shadow_coord;

uniform vec3 sky_color;        // ambient from above
uniform vec3 ground_color;     // warm bounce from below
uniform vec3 glow_color;       // the bright part of the sky, around the sun
uniform vec3 sun_dir_world;    // horizontal direction TOWARDS the sun
uniform float glow_strength;
uniform vec3 sun_color;        // direct beam, already scaled for intensity
uniform float sun_wrap;        // softens the terminator; sunset light wraps
uniform float spec_strength;
uniform float spec_power;

uniform vec4 haze_color;       // rgb = horizon glow, a = maximum density
uniform float haze_start;
uniform float haze_end;

uniform float shadow_bias;
uniform float shadow_blur;
uniform int shadow_samples;
uniform float shadow_strength;
uniform vec3 shadow_center;    // where the shadow map is currently focused
uniform float shadow_fade_start;
uniform float shadow_fade_end;

// The circuit-wide map, rendered once. Passed as plain uniforms rather than as
// a second p3d_LightSource: Panda orders that array by its own light sort, so
// which index is the near map and which the baked one is not something this
// shader can safely assume.
uniform sampler2DShadow bake_map;
// The baked map's projection, as the four world-space vectors that build it
// rather than as a matrix. A matrix would be fewer uniforms, but which way
// round GLSL and Panda agree to multiply it is not something this code can
// check from the inside: both orders produce coordinates that look plausible
// -- in range, varying smoothly -- and only one of them samples the right
// texel. These come straight out of the transform Panda itself applies, so
// there is no convention left to get wrong.
uniform vec3 bake_o;
uniform vec3 bake_ex;
uniform vec3 bake_ey;
uniform vec3 bake_ez;
uniform float bake_normal_offset;
uniform float bake_ready;
uniform float bake_bias;
uniform float bake_blur;
uniform int bake_samples;

out vec4 fragment_color;

float sample_shadow() {
    vec4 coord = shadow_coord;
    coord.z += shadow_bias;
    float total = 0.0;
    float n = float(shadow_samples);
    float half_blur = shadow_blur * 0.5;
    for (int x = 0; x < shadow_samples; ++x) {
        for (int y = 0; y < shadow_samples; ++y) {
            vec4 c = coord;
            c.x += float(x) * shadow_blur / n - half_blur;
            c.y += float(y) * shadow_blur / n - half_blur;
            total += textureProj(p3d_LightSource[0].shadowMap, c);
        }
    }
    return total / (n * n);
}

float sample_bake() {
    // Look the surface up from a point pushed out along its own normal, by
    // rather more than one texel of the baked map. This is what stops a
    // grandstand shadowing itself: at 0.74 m per texel the stored depth for a
    // sloped seat block is the depth of somewhere up to a texel away, and a
    // pure depth bias big enough to cover that also lifts every shadow off
    // the thing casting it. Offsetting along the normal moves the lookup
    // sideways, where the error actually is, so the depth slack can stay
    // small and contact shadows stay put.
    vec3 wp = world_position + normalize(world_normal) * bake_normal_offset;
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

void main() {
    vec4 albedo = texture(p3d_Texture0, texcoords) * p3d_ColorScale * vertex_color;

    vec3 N = normalize(view_normal);
    vec3 L = normalize(p3d_LightSource[0].position.xyz
                       - view_position * p3d_LightSource[0].position.w);
    vec3 V = normalize(-view_position);

    // Wrapped diffuse. A hard N.L terminator is a midday look; near sunset the
    // beam grazes everything and the falloff round a curved surface is long.
    float ndl = dot(N, L);
    float diffuse = clamp((ndl + sun_wrap) / (1.0 + sun_wrap), 0.0, 1.0);

    // One shadow map cannot cover a 6 km circuit at a useful resolution, so it
    // follows the car. Everything outside it samples garbage -- and would come
    // out fully shadowed -- so the shadow term is handed over to the baked
    // map at the edge of the covered area.
    //
    // Both maps hold the roadside, so the handover changes the resolution of
    // the shadow and not what is in it. Near the car the following map is
    // eleven times finer and is the only one sharp enough for a roof's
    // shadow on the steps under it; past its edge the baked map is the only
    // one that reaches at all, and at that distance its half-metre texels are
    // well under a pixel.
    float sd = length(world_position.xz - shadow_center.xz);
    float cover = 1.0 - smoothstep(shadow_fade_start, shadow_fade_end, sd);
    // Skip the following map's nine taps wherever it does not reach. On a
    // straight most of the screen is past its edge, so this pays for a good
    // part of the baked lookup that replaces it there.
    float near_s = (cover > 0.001) ? sample_shadow() : 1.0;
    // Fade the following map out to "lit" past its edge, where it samples
    // garbage, and then take the *darker* of the two rather than crossfading.
    //
    // Crossfading was wrong as soon as the two maps stopped holding the same
    // casters. They do not: the roadside is baked and taken out of the
    // per-frame pass (config.BAKE_REPLACES_LIVE), so the following map holds
    // only the cars. Weighting towards it near the camera therefore weighted
    // towards a map with no trees, no fences and no grandstands in it -- and
    // every static shadow faded out as you approached it, which is exactly
    // backwards. A point is in shadow if *either* map says so, at every
    // distance, and that also removes the seam the blend used to draw.
    near_s = mix(1.0, near_s, cover);
    float s = (bake_ready > 0.5) ? min(near_s, sample_bake()) : near_s;
    float shadow = mix(1.0, s, shadow_strength);
    // Hemispheric ambient: sky above, ground bounce below.
    vec3 nw = normalize(world_normal);
    float up = clamp(nw.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 ambient = mix(ground_color, sky_color, up);

    // ...plus the fact that a sunset sky is nowhere near uniform: it is far
    // brighter around the sun. Purely hemispheric ambient gives every vertical
    // face the same value whichever way it points, so an object the beam does
    // not reach -- the shaded side of a marquee, the seats of a grandstand --
    // renders as one flat colour with no faces in it. This adds the horizon
    // glow in proportion to how much a face turns towards it.
    vec2 flat_n = vec2(nw.x, nw.z);
    float horiz = length(flat_n);
    if (horiz > 1e-4) {
        float toward = max(dot(flat_n / horiz, sun_dir_world.xz), 0.0);
        ambient += glow_color * (toward * horiz * glow_strength);
    }

    vec3 lit = albedo.rgb * (ambient + sun_color * diffuse * shadow);

    // A low sun puts a long specular sheen down asphalt and along bodywork;
    // without it an evening scene looks like a dimmed noon.
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), spec_power) * spec_strength;
    lit += sun_color * spec * shadow * step(0.0, ndl);

    // Haze. Squared so the near field stays clear and the far field sinks into
    // the glow, which is what sells the depth of a low-sun scene.
    // The camera sits at the origin of view space, so the distance to it is
    // just the length of the view-space position. This used to come in as a
    // uniform and that was a bug waiting to happen: it silently stayed at the
    // world origin, so the haze measured distance from (0,0,0) and Monza's
    // first chicane -- 940 m from it -- rendered under 95% haze, a flat orange
    // wash over the car and the road. A value the shader can derive should not
    // be plumbed in from outside.
    float d = length(view_position);
    float t = clamp((d - haze_start) / max(haze_end - haze_start, 1.0), 0.0, 1.0);
    lit = mix(lit, haze_color.rgb, t * t * haze_color.a);

    fragment_color = vec4(lit, albedo.a);
}
''', default_input={
    # ONLY what genuinely differs per entity lives here.
    #
    # Ursina writes every default_input onto the entity when the shader is
    # assigned, and a shader input set on an entity overrides one inherited
    # from an ancestor. So anything listed here can no longer be driven from
    # ``scene`` -- which is exactly what went wrong: camera_world_position and
    # shadow_center stayed at the origin, so the haze measured distance from
    # the world origin (Monza's first chicane washed out orange) and the shadow
    # coverage fade killed shadows everywhere but near it. The lighting colours
    # were being ignored the same way, silently, because the defaults happened
    # to look similar. lighting.Sunset sets all of those on ``scene`` instead.
    'texture_scale': Vec2(1, 1),
    'texture_offset': Vec2(0, 0),
    'spec_strength': 0.10,
    'spec_power': 24.0,
})

# Deliberately no ``continuous_input``. Ursina applies those by looping over
# every entity and calling set_shader_input once each, every frame -- with the
# car's wheels and hubs that came to 117 calls a frame, about 2 ms, plus the
# render-state churn it fed into Panda's state garbage collector. Panda
# composes shader inputs down the scene graph, so lighting.py sets the
# per-frame ones on ``scene`` instead: two calls, not two hundred.
