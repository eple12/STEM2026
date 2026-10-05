"""Golden-hour sun, sky and shadows.

One directional light, one shadow map, and a sky whose colours the shader is
told about so the two agree. The whole look rests on a single relationship:
near sunset the beam is warm *because* the blue has been scattered out of it,
and that scattered blue is what lights everything the beam misses. Warm key,
cool fill. Get that backwards and it reads as an orange filter over noon.

The shadow map follows the car. A single map stretched over a 6 km circuit
gives about three metres per texel, which cannot draw a car; focused on a 130 m
box around it, it gives six centimetres.
"""
from __future__ import annotations

import math

import numpy as np
from panda3d.core import (Camera, FrameBufferProperties, GraphicsOutput,
                          CullFaceAttrib, GraphicsPipe, Mat4,
                          OrthographicLens, Point3,
                          RenderState, SamplerState, Texture, WindowProperties)
from ursina import Entity, Vec2, Vec3, Vec4, camera, color, scene
from ursina.lights import DirectionalLight

from . import config
from .shaders import material as make_material
from .shaders import sky_shader, sunset_shader


_CLOUDS = None


def _cloud_texture(size: int = 512):
    """Two tileable cloud fields in R and G: broad billows and their detail.

    Built once per process -- it is the same sky over every circuit -- and
    mipmapped, because the layer is foreshortened into the horizon and an
    unfiltered noise texture there is a band of sparkle.
    """
    global _CLOUDS
    if _CLOUDS is not None:
        return _CLOUDS
    from panda3d.core import Texture as PTexture

    from .textures import _noise
    broad = _noise(size, octaves=(3, 6, 12, 24, 48), seed=41)
    detail = _noise(size, octaves=(8, 16, 32, 64, 128), seed=42)
    broad = (broad - broad.min()) / max(np.ptp(broad), 1e-6)
    detail = (detail - detail.min()) / max(np.ptp(detail), 1e-6)
    img = np.zeros((size, size, 3), np.uint8)
    img[..., 0] = (broad * 255).astype(np.uint8)
    img[..., 1] = (detail * 255).astype(np.uint8)
    tex = PTexture("clouds")
    tex.setup_2d_texture(size, size, PTexture.T_unsigned_byte,
                         PTexture.F_rgb8)
    # Panda wants BGR, bottom row first.
    tex.set_ram_image(np.ascontiguousarray(img[::-1, :, ::-1]).tobytes())
    tex.set_minfilter(SamplerState.FT_linear_mipmap_linear)
    tex.set_magfilter(SamplerState.FT_linear)
    tex.set_anisotropic_degree(4)
    _CLOUDS = tex
    return tex


def _blank_shadow_map():
    """A 1x1 depth texture that shadows nothing, for use before the real one.

    It has to be a *shadow* sampler -- comparison filtering and all -- or the
    declared sampler2DShadow will not accept it.
    """
    tex = Texture('bake_seed')
    tex.setup_2d_texture(1, 1, Texture.T_unsigned_byte,
                         Texture.F_depth_component)
    tex.make_ram_image()
    tex.modify_ram_image()[0] = 255          # farthest depth: nothing occludes
    tex.set_minfilter(SamplerState.FT_shadow)
    tex.set_magfilter(SamplerState.FT_shadow)
    return tex


#: Clip space [-1, 1] -> texture space [0, 1], in Panda's row-vector order.
_DEPTH_BIAS = Mat4(0.5, 0.0, 0.0, 0.0,
                   0.0, 0.5, 0.0, 0.0,
                   0.0, 0.0, 0.5, 0.0,
                   0.5, 0.5, 0.5, 1.0)


class Sunset:
    """Owns the sun, the sky and the shader uniforms that tie them together."""

    def __init__(self, track=None):
        # Ursina's DirectionalLight calls render.setLight on construction and
        # nothing takes it off again, so a light outlives the entity that owned
        # it. The shader reads p3d_LightSource[0], which is then still the sun
        # from the *first* race of the session -- pointing the wrong way, with
        # a shadow map focused on a circuit that is no longer loaded. Clear the
        # slate before claiming it.
        import builtins
        builtins.render.clear_light()

        self.sky = _SkyDome()

        # Direction only: the light's own shadow buffer is never created
        # (no shader names its shadowMap). The car has its own map, CarShadow,
        # and the static world is baked.
        self.sun = DirectionalLight(shadows=False)
        # Panda's own light colour is unused: this shader reads the light for
        # its direction and shadow map only, and takes the beam colour from a
        # uniform, so warmth and intensity are tuned in one place.
        self.sun.color = color.white

        el = math.radians(config.SUN_ELEVATION)
        az = math.radians(self._azimuth(track))
        # Direction the light travels: down, and along the azimuth.
        self._dir = Vec3(math.cos(el) * math.sin(az), -math.sin(el),
                         math.cos(el) * math.cos(az))
        self.sun.look_at(self._dir)
        self.car_shadow = CarShadow(self.sun)
        self.field_shadow = FieldShadow(self.sun)

        # The baked map gets its own buffer and camera rather than a second
        # DirectionalLight. Two reasons. Panda only brings a light's shadow
        # buffer into existence when a shader asks for that light's
        # `shadowMap`, so a light nothing references never renders anything;
        # and a second light would join `p3d_LightSource`, whose order Panda
        # decides, putting the existing shader's assumption that [0] is the
        # sun at the mercy of a sort this code does not control.
        self._bake_buf = None
        self._bake_cam = None
        self._bake_texel = 0.0
        self._bake_track = track if config.BAKE_SHADOWS else None

        sun_vec = Vec3(-self._dir.x, -self._dir.y, -self._dir.z).normalized()
        self._uniforms = {
            # The sky, as both the dome and every reflection read it.
            'sky_zenith': Vec3(*config.SKY_ZENITH),
            'sky_horizon': Vec3(*config.SKY_HORIZON),
            'sky_ground': Vec3(*config.SKY_GROUND),
            'sun_vec': sun_vec,
            'sun_color': Vec3(*config.LIGHT_SUN),
            'glow_color': Vec3(*config.LIGHT_GLOW),
            'glow_strength': config.LIGHT_GLOW_STRENGTH,
            # Irradiance for the diffuse ambient: sky above, bounce below.
            'ambient_sky': Vec3(*config.LIGHT_SKY),
            'ambient_ground': Vec3(*config.LIGHT_BOUNCE),
            'sun_wrap': config.LIGHT_SUN_WRAP,
            'env_strength': config.LIGHT_ENV,
            'haze_color': color.rgba(1.0, 1.0, 1.0, config.HAZE_DENSITY),
            'haze_start': config.HAZE_START,
            'haze_end': config.HAZE_END,
            # The camera chain develops the frame (post.py); until it says it
            # is there, the shaders develop their own output.
            'direct_out': 1.0,
            'cloud_map': _cloud_texture(),
            'clouds': Vec4(*config.CLOUD_SHAPE),
            'cloud_drift': Vec2(0.0, 0.0),
            'sun_disc': math.cos(math.radians(config.SUN_DISC_DEG)),
            'car_map': self.car_shadow.texture(),
            # Names without underscores: Panda splits trans_x_to_y_of_<name>
            # on them.
            'carcam': self.car_shadow.cam,
            'shadow_bias': self.car_shadow.bias,
            'shadow_blur': config.CAR_SHADOW_SOFT_M / config.CAR_SHADOW_FILM,
            'shadow_samples': config.SHADOW_SAMPLES,
            'shadow_fade_start': config.CAR_SHADOW_FILM * 0.75,
            'car_normal_offset': config.CAR_SHADOW_NORMAL_M,
            'field_map': self.field_shadow.texture(),
            'fieldcam': self.field_shadow.cam,
            'field_on': 0.0,
            'field_bias': self.field_shadow.bias,
            'field_blur': config.FIELD_SHADOW_SOFT_M / config.FIELD_SHADOW_FILM,
            'field_normal_offset': config.FIELD_SHADOW_NORMAL_M,
            'shadow_strength': 1.0,
            # A PTA, not a Vec3: it is written in place every frame by
            # follow(). Assigning a new value to a shader input on the scene
            # root makes a new ShaderAttrib there, which invalidates the
            # composed render state of every node underneath -- all of them
            # were recomposed each frame and the old states left for Panda's
            # state garbage collector. Mutating the array changes the uniform
            # and nothing else.
            # Off until bake() has something to sample. The shader skips the
            # lookup entirely while this is zero, so the sampler being unbound
            # in the meantime costs nothing and reads nothing.
            'bake_ready': 0.0,
            # Panda validates every declared uniform at draw time, whether or
            # not the shader's branch reaches it, so both of these have to
            # exist from the first frame -- including the two frames bake()
            # itself renders in order to bring the real map into being.
            'bake_map': _blank_shadow_map(),
            'bake_o': Vec3.zero,
            'bake_ex': Vec3.zero,
            'bake_ey': Vec3.zero,
            'bake_ez': Vec3.zero,
            'bake_normal_offset': 0.0,
            'bake_bias': 0.0,
            'bake_blur': config.BAKE_BLUR,
            'bake_samples': config.BAKE_SAMPLES,
        }
        self._lit: list[Entity] = []
        # Shared uniforms live on the scene root and are inherited by every
        # entity under it, so they cost one call rather than one per entity.
        for k, v in self._uniforms.items():
            scene.set_shader_input(k, v)

    @staticmethod
    def _azimuth(track) -> float:
        """Where to put the sun, in degrees, given the circuit.

        Across the start/finish line from the paddock side and raked along it.
        A fixed compass bearing cannot work for all 23: on half of them it ends
        up behind the main grandstand, whose shadow at this elevation is longer
        than the track is wide, and the whole straight goes dark.
        """
        if track is None:
            return config.SUN_AZIMUTH
        tan = track.tangent[0]
        nrm = track.normal[0]
        rake = math.radians(config.SUN_RAKE)
        # SUN_SIDE picks which side of the circuit the sun sits on. It matters
        # more than it sounds: whatever is on the sun side throws its shadow
        # across the track, and the paddock marquees are close enough to the
        # edge to blanket the racing line if they are the ones lit from behind.
        # d is the direction the light TRAVELS, so the sun sits on the opposite
        # side from where d points -- hence the minus. Getting this backwards
        # put the sun over the paddock while the constant said "grandstands".
        d = -nrm * (config.SUN_SIDE * math.cos(rake)) + tan * math.sin(rake)
        return math.degrees(math.atan2(float(d[0]), float(d[1])))

    def _make_bake_target(self, track):
        """An offscreen depth buffer and an orthographic camera over the lap.

        The film is square and axis-aligned to the beam, so it has to clear
        the circuit's *diagonal*, not its width -- the track lies at whatever
        angle to the sun the circuit happens to give it. Near is negative for
        the same reason the following map's is: geometry behind the camera
        still has to cast.
        """
        import builtins

        base = builtins.base
        res = int(config.BAKE_RESOLUTION)
        lo, hi = track.bounds()
        cx, cz = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
        fb = FrameBufferProperties()
        fb.set_rgb_color(False)
        fb.set_depth_bits(24)
        buf = base.graphicsEngine.make_output(
            base.pipe, 'bake_shadow', -1000, fb,
            WindowProperties.size(res, res),
            GraphicsPipe.BF_refuse_window, base.win.get_gsg(), base.win)
        if buf is None:
            return None, None, None

        tex = Texture('bake_map')
        tex.set_format(Texture.F_depth_component)
        buf.add_render_texture(tex, GraphicsOutput.RTM_bind_or_copy,
                               GraphicsOutput.RTP_depth)
        # Comparison filtering, or the declared sampler2DShadow will not take
        # it and every lookup comes back as a plain depth value.
        tex.set_minfilter(SamplerState.FT_shadow)
        tex.set_magfilter(SamplerState.FT_shadow)
        tex.set_wrap_u(SamplerState.WM_border_color)
        tex.set_wrap_v(SamplerState.WM_border_color)
        tex.set_border_color((1.0, 1.0, 1.0, 1.0))   # outside the film: lit
        buf.set_clear_depth_active(True)
        buf.set_clear_depth(1.0)

        lens = OrthographicLens()
        cam = Camera('bake_cam', lens)
        cam.set_camera_mask(self.BAKE_MASK)
        # Store the *back* faces. The front face of a caster is the surface
        # that then has to test against it, and at this texel size it loses
        # that test against itself all over the object. Recording the far side
        # instead puts the whole thickness of the object between the two, and
        # the props here are closed solids, so nothing leaks. The override
        # beats the cull state flatten_strong baked into the geoms.
        if config.BAKE_BACKFACE:
            cam.set_initial_state(RenderState.make(
                CullFaceAttrib.make(CullFaceAttrib.M_cull_counter_clockwise), 1))
        cam_np = builtins.render.attach_new_node(cam)
        cam_np.set_pos(cx, config.SHADOW_HEIGHT, cz)
        # Copy the sun's orientation rather than deriving it again: this world
        # is y-up and Panda's look_at is not, and one of the two would be
        # wrong.
        cam_np.set_quat(self.sun.get_quat(builtins.render))

        # Fit the film to where the circuit actually lands in the beam's own
        # frame, rather than to a square big enough for any orientation. A
        # square of the diagonal wastes most of its area -- Monza filled about
        # a third of it -- and every texel thrown away that way is texel size
        # added, which is what shadow acne is made of.
        view = builtins.render.get_transform(cam_np).get_mat()
        pts = [view.xform_point(Point3(x, y, z))
               for x in (lo[0] - config.BAKE_MARGIN, hi[0] + config.BAKE_MARGIN)
               for z in (lo[1] - config.BAKE_MARGIN, hi[1] + config.BAKE_MARGIN)
               for y in (0.0, config.BAKE_TOP)]
        # Which camera-space axis is depth was measured, not assumed. Panda's
        # documented camera frame looks along +y with +z up, which says the
        # film is x/z and the depth is y -- and that is what this did. It is
        # not what comes out: normalising the projected depth against two
        # different near/far settings gave the same underlying value both
        # times, and that value tracked camera *z*, not y. Reading it as y put
        # part of the circuit outside the depth range -- a reference depth
        # below zero passes every comparison, so those places kept their
        # shadow from the following map and lost it in the baked one, which is
        # exactly what the last corner at Monza looked like -- and it sized
        # the film from the depth extent, wasting two thirds of the texels.
        x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
        y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
        fx, fz = x1 - x0, y1 - y0
        # Depth range: deliberately generous, and then checked. Deriving it
        # from the same camera-space component the film uses looked right and
        # was not -- the depth Panda's projection actually produces did not
        # match that axis, so points well inside the fitted range came out at
        # a normalised depth just below zero. A reference depth below zero
        # passes every comparison, which is why the ground kept its shadow
        # from the following map and lost it in the baked one, at exactly the
        # places the fit was tightest. Half a millimetre of 24-bit precision
        # is not worth guessing a convention for.
        d0, d1 = min(p[2] for p in pts), max(p[2] for p in pts)
        near, far = d0 - 50.0, d1 + 50.0
        lens.set_film_size(fx, fz)
        # Offset, not a doubled half-extent: the camera sits at the middle of
        # the track's bounding box, which is not the middle of where that box
        # lands in the beam's frame. Sizing the film to twice the larger half
        # made it bigger than the square it replaced.
        lens.set_film_offset((x0 + x1) * 0.5, (y0 + y1) * 0.5)
        lens.set_near_far(near, far)
        # The bias is quoted in metres and converted here, because normalised
        # depth means nothing without knowing the span it is normalised over.
        scene.set_shader_input('bake_bias',
                               config.BAKE_SLACK / max(far - near, 1e-6))
        self._bake_texel = max(fx, fz) / float(res)
        scene.set_shader_input(
            'bake_normal_offset', self._bake_texel * config.BAKE_NORMAL_OFFSET)
        buf.make_display_region(0, 1, 0, 1).set_camera(cam_np)
        return buf, cam_np, lens

    def bake(self):
        """Draw the static roadside into the baked map, then switch it off.

        Call once, after every entity has been through ``apply`` -- the masks
        decide what lands in which map, and they have to be set first.
        """
        import builtins

        if self._bake_track is None:
            return False
        buf, cam_np, lens = self._make_bake_target(self._bake_track)
        if buf is None:
            print('lighting: could not make the baked shadow buffer; '
                  'static shadows stay live')
            self._bake_track = None
            return False
        self._bake_buf, self._bake_cam = buf, cam_np

        # Draw it. Once.
        builtins.base.graphicsEngine.render_frame()

        # World -> light -> clip -> [0, 1] texture space, handed to the
        # shader as the four vectors that rebuild it. Measuring the basis by
        # transforming the origin and the three unit points means the shader
        # reproduces exactly what Panda's own xform_point does, with no
        # question about which side of the matrix a vector goes on.
        mat = (builtins.render.get_transform(cam_np).get_mat()
               * lens.get_projection_mat() * _DEPTH_BIAS)
        o = mat.xform_point(Point3(0, 0, 0))
        basis = [mat.xform_point(Point3(*p)) - o
                 for p in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        # Verify, rather than assume: every point of the circuit has to land
        # inside the map's depth range, or its shadow lookup is meaningless.
        chk = [mat.xform_point(Point3(float(x), y, float(z)))
               for x, z in self._bake_track.center[::11]
               for y in (0.0, config.BAKE_TOP)]
        d0 = min(q[2] for q in chk)
        d1 = max(q[2] for q in chk)
        if d0 < 0.02 or d1 > 0.98:
            print(f'lighting: baked depth range {d0:.3f}..{d1:.3f} is out of '
                  f'bounds; static shadows past the following map will be wrong')

        scene.set_shader_input('bake_map', buf.get_texture())
        scene.set_shader_input('bake_o', Vec3(o[0], o[1], o[2]))
        for name, v in zip(('bake_ex', 'bake_ey', 'bake_ez'), basis):
            scene.set_shader_input(name, Vec3(v[0], v[1], v[2]))
        scene.set_shader_input('bake_ready', 1.0)
        # Nothing in it moves and neither does the sun, so it never needs
        # drawing again. That is the whole point.
        buf.set_active(False)
        print(f'lighting: baked static shadows at {config.BAKE_RESOLUTION}, '
              f'{self._bake_texel:.2f} m per texel')
        return True

    # -- applying ------------------------------------------------------
    #: The shadow camera is given this mask by Ursina's DirectionalLight, so
    #: hiding an entity from it takes that entity out of the shadow pass.
    SHADOW_MASK = 0b0001
    #: The baked map's camera gets this one instead, so the two lights can be
    #: given different sets of casters.
    BAKE_MASK = 0b0010

    def apply(self, *entities, spec_strength=None, spec_power=None, casts=True,
              baked=False, material=None, shader=None):
        """Give an entity (and its children) the sunset shader.

        ``casts=False`` keeps the entity out of both shadow maps. The road
        surface needs it: a flat receiver that also casts writes its own depth
        and then fails its own comparison, so the entire asphalt strip came out
        black while the grass beside it -- same normal, same shader, but not a
        caster of anything that could reach it -- stayed lit. Ground layers only
        ever need to *receive*.

        ``baked=True`` sends it to the circuit-wide map that is drawn once
        instead of the one that follows the car. Use it for anything that
        cannot move. The two are exclusive: an entity in both would be drawn
        into a depth buffer every frame *and* be baked, which is the cost of
        the first with none of the saving.
        """
        into_bake = baked and self._bake_track is not None
        # By default the baked map is an *extension* of the following one, not
        # a replacement: static casters go in both, so the near field keeps
        # its 7 cm texels. Making the bake the only source of static shadows
        # saves the roadside's share of the shadow pass -- measured at 1.0 to
        # 1.7 ms of a 17 ms frame -- but every static shadow then comes from a
        # 0.74 m texel, which dulls the grandstands with their own acne and
        # needs enough depth slack to lift shadows off their casters.
        drop_live = into_bake and config.BAKE_REPLACES_LIVE
        if material is None and spec_power is not None:
            # The old Blinn-Phong pair, read as a microfacet material: the
            # exponent maps onto roughness the standard way, and the strength
            # onto the reflectance at normal incidence.
            rough = math.sqrt(2.0 / (float(spec_power) + 2.0))
            level = (spec_strength if spec_strength is not None else 0.1) / 0.1
            material = make_material(rough, level)
        for e in entities:
            if e is None:
                continue
            for target in self._walk(e):
                if not casts or drop_live:
                    target.hide(self.SHADOW_MASK)
                if not into_bake:
                    target.hide(self.BAKE_MASK)
                # A variant the entity asked for (the road, the grass)
                # unless the caller insists on one.
                target.shader = (shader or getattr(target, "world_shader", None)
                                 or sunset_shader)
                # Only what actually differs per entity is set per entity --
                # and after the shader, whose defaults would overwrite it.
                for k, v in getattr(target, "world_inputs", {}).items():
                    target.set_shader_input(k, v)
                if material is not None:
                    target.set_shader_input('material', material)
                self._lit.append(target)

    @staticmethod
    def _walk(e):
        if getattr(e, "no_light", False):
            # Decals that draw themselves (the car's contact shadow).
            e.hide(Sunset.SHADOW_MASK | Sunset.BAKE_MASK)
            return
        yield e
        for child in getattr(e, 'children', ()):
            yield from Sunset._walk(child)

    # -- per frame -----------------------------------------------------
    def follow(self, car):
        """Put the car's shadow camera on the car, as drawn this frame.

        Called after the car has been posed from the interpolated physics
        state, so the shadow and the car are always the same instant.
        """
        p = car.world_position
        self.car_shadow.follow(p)

    def cast(self, entity):
        """*entity* (and everything under it) casts into the car map."""
        self.car_shadow.cast(entity)

    def set_active(self, on: bool):
        self.car_shadow.set_active(on)

    def destroy(self):
        import builtins

        from .ui import destroy_tree as _destroy

        ge = builtins.base.graphicsEngine
        # Panda creates a shadow-casting light's depth buffer inside the GSG on
        # the first frame that renders it, and destroying the Ursina entity
        # that owns the light does not take it back. One race is one 2048
        # buffer left behind; an exhibition machine going menu-race-menu all
        # day accumulates them until it runs out of memory. Nothing in the
        # scene graph shows this -- the child counts stay flat -- so it only
        # turns up if you count the engine's buffers.
        sun_buf = self.sun._light.get_shadow_buffer(builtins.base.win.get_gsg())
        if sun_buf is not None:
            ge.remove_window(sun_buf)
        self.car_shadow.destroy()
        self.field_shadow.destroy()

        builtins.render.clear_light()
        _destroy(self.sky)
        _destroy(self.sun)
        if self._bake_buf is not None:
            ge.remove_window(self._bake_buf)
            self._bake_buf = None
        if self._bake_cam is not None:
            self._bake_cam.remove_node()
            self._bake_cam = None
        scene.set_shader_input('bake_ready', 0.0)
        self._lit.clear()


class _SkyDome(Entity):
    """The sky: a dome on the camera, drawn first, shaded analytically.

    It replaces Ursina's textured ``Sky``. A painted gradient cannot know
    where the sun is, and the moment the bodywork reflects the sky the two had
    better agree -- so the dome evaluates the same ``sky_radiance`` the world
    shaders reflect, plus the sun's disc and a cloud layer.
    """

    def __init__(self):
        from panda3d.core import PTA_LVecBase2f
        # The model's radius is 1: it is scaled to just inside the far plane
        # every frame, as Ursina's own Sky does -- any bigger and the far
        # plane clips the whole dome away.
        super().__init__(parent=scene, model="sky_dome", double_sided=True,
                         scale=camera.clip_plane_far * 0.8)
        self.shader = sky_shader
        self.set_bin("background", 0)
        self.set_depth_write(False)
        self.set_light_off()
        # Not a shadow caster, and not in either shadow pass.
        self.hide(Sunset.SHADOW_MASK | Sunset.BAKE_MASK)
        self._drift = PTA_LVecBase2f.empty_array(1)
        self.set_shader_input("cloud_drift", self._drift)
        self._t = 0.0

    def update(self):
        # Ursina calls this every frame, the intro film included -- the race
        # loop's own per-frame hook does not run during it.
        import time as _time
        self.position = camera.world_position
        far = camera.clip_plane_far * 0.8
        if abs(self.scale_x - far) > 1.0:
            self.scale = far
        # Clouds creep across the sky; a still cloud layer is a painting.
        t = _time.perf_counter()
        self._drift[0] = Vec2(t * config.CLOUD_DRIFT[0], t * config.CLOUD_DRIFT[1])


class CarShadow:
    """A shadow map for the hero car alone, carried along with it.

    The general-purpose map used to follow the car across the world: it was
    re-centred on the car's *physics* position, which runs up to a step
    behind the interpolated pose the car is drawn at, and either slid by
    fractions of a texel (edges crawl) or snapped by whole ones (the shadow
    steps). Both read as a shadow that lags and shivers.

    This one is a camera fixed to the car's drawn position and pointed along
    the sun. It sees only what has been given to ``cast`` -- the player's car
    -- so its 9 m film puts 4 mm in a texel, and since it moves rigidly with
    the car, the car lands on exactly the same texels every frame: the only
    thing that can change the shadow is the car actually turning against the
    sun. One small depth pass of one car a frame.
    """

    MASK = 0b0100

    def __init__(self, sun):
        import builtins

        base = builtins.base
        res = int(config.CAR_SHADOW_RES)
        fb = FrameBufferProperties()
        fb.set_rgb_color(False)
        fb.set_depth_bits(24)
        self.buf = base.graphicsEngine.make_output(
            base.pipe, "car_shadow", -900, fb, WindowProperties.size(res, res),
            GraphicsPipe.BF_refuse_window, base.win.get_gsg(), base.win)
        self.tex = Texture("car_shadow")
        if self.buf is not None:
            self.tex.set_format(Texture.F_depth_component)
            self.buf.add_render_texture(self.tex, GraphicsOutput.RTM_bind_or_copy,
                                        GraphicsOutput.RTP_depth)
            self.buf.set_clear_depth_active(True)
            self.buf.set_clear_depth(1.0)
        else:
            print("lighting: no car shadow buffer; the car casts no shadow")
            self.tex = _blank_shadow_map()
        self.tex.set_minfilter(SamplerState.FT_shadow)
        self.tex.set_magfilter(SamplerState.FT_shadow)
        self.tex.set_wrap_u(SamplerState.WM_border_color)
        self.tex.set_wrap_v(SamplerState.WM_border_color)
        self.tex.set_border_color((1.0, 1.0, 1.0, 1.0))

        self.lens = OrthographicLens()
        film, depth = config.CAR_SHADOW_FILM, config.CAR_SHADOW_DEPTH
        self.lens.set_film_size(film, film)
        self.lens.set_near_far(-depth, depth)
        cam = Camera("car_shadow_cam", self.lens)
        cam.set_camera_mask(self.MASK)
        # Thin plates (wings, endplates) are single-sided: draw both faces,
        # or a wing lit from behind casts nothing.
        cam.set_initial_state(RenderState.make(
            CullFaceAttrib.make(CullFaceAttrib.M_cull_none), 1))
        self.cam = builtins.render.attach_new_node(cam)
        self.cam.set_quat(sun.get_quat(builtins.render))
        if self.buf is not None:
            self.buf.make_display_region(0, 1, 0, 1).set_camera(self.cam)
        # Nothing is drawn into this map unless it is shown through on
        # purpose (cast): the whole scene is hidden from its camera.
        builtins.render.hide(self.MASK)
        self.bias = config.CAR_SHADOW_BIAS_M / (2.0 * depth)
        self.follow(Vec3(0, 0, 0))

    def texture(self):
        return self.tex

    def cast(self, entity):
        entity.show_through(self.MASK)
        for d in entity.find_all_matches("**/contact_shadow"):
            d.hide(self.MASK)

    def follow(self, p):
        # Only the camera moves: the shaders read its matrix from the scene
        # graph at draw time (trans_world_to_clip_of_carcam), so the lookup
        # always matches the frame the map was drawn in.
        import builtins
        self.cam.set_pos(builtins.render, p.x, p.y + 0.5, p.z)

    def set_active(self, on: bool):
        if self.buf is not None:
            self.buf.set_active(on)

    def destroy(self):
        import builtins
        if self.buf is not None:
            builtins.base.graphicsEngine.remove_window(self.buf)
            self.buf = None
        self.cam.remove_node()
        builtins.render.show(self.MASK)


_DEPTH_VERT = """#version 150
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
void main() { gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex; }
"""
_DEPTH_FRAG = """#version 150
out vec4 o;
void main() { o = vec4(1.0); }
"""


class FieldShadow:
    """The other cars' shadows: one map, wide, round the car on camera.

    They used to go into the hero car's map (``CarShadow``), whose 9 m film
    is sized for one car: a car alongside threw a shadow, one a length away
    did not, and they flicked on and off as the gaps changed. This map is
    ``FIELD_SHADOW_FILM`` across, pushed ahead of the followed car along the
    view (where the cars that matter are), and faded out at its edge in the
    shader rather than cut off.

    It draws a scene of its own: one stand-in per car, the car's far-LOD
    copy merged into a single depth-only mesh and posed from the real car
    every frame -- twenty draws, not the 300-odd of the cars themselves.
    The camera is moved in whole texels of its own film, so a shadow cast
    by a car standing still stays on the same texels while the map moves
    (no crawl along the edges).
    """

    def __init__(self, sun):
        import builtins

        from panda3d.core import NodePath, Shader as PShader

        base = builtins.base
        res = int(config.FIELD_SHADOW_RES)
        self.res = res
        fb = FrameBufferProperties()
        fb.set_rgb_color(False)
        fb.set_depth_bits(24)
        self.buf = base.graphicsEngine.make_output(
            base.pipe, "field_shadow", -899, fb, WindowProperties.size(res, res),
            GraphicsPipe.BF_refuse_window, base.win.get_gsg(), base.win)
        self.tex = Texture("field_shadow")
        if self.buf is not None:
            self.tex.set_format(Texture.F_depth_component)
            self.buf.add_render_texture(self.tex, GraphicsOutput.RTM_bind_or_copy,
                                        GraphicsOutput.RTP_depth)
            self.buf.set_clear_depth_active(True)
            self.buf.set_clear_depth(1.0)
            self.buf.set_active(False)
        else:
            print("lighting: no field shadow buffer; other cars cast no shadow")
            self.tex = _blank_shadow_map()
        self.tex.set_minfilter(SamplerState.FT_shadow)
        self.tex.set_magfilter(SamplerState.FT_shadow)
        self.tex.set_wrap_u(SamplerState.WM_border_color)
        self.tex.set_wrap_v(SamplerState.WM_border_color)
        self.tex.set_border_color((1.0, 1.0, 1.0, 1.0))

        self.root = NodePath("field_shadow_scene")
        self.root.set_shader(PShader.make(PShader.SL_GLSL, _DEPTH_VERT,
                                          _DEPTH_FRAG), 10)
        self.root.set_two_sided(True)
        self.film = film = config.FIELD_SHADOW_FILM
        depth = config.FIELD_SHADOW_DEPTH
        self.lens = OrthographicLens()
        self.lens.set_film_size(film, film)
        self.lens.set_near_far(-depth, depth)
        cam = Camera("field_shadow_cam", self.lens)
        cam.set_scene(self.root)
        self.cam = self.root.attach_new_node(cam)
        q = sun.get_quat(builtins.render)
        self.cam.set_quat(q)
        self._right = q.get_right()
        self._up = q.get_up()
        self._fwd = q.get_forward()
        if self.buf is not None:
            self.buf.make_display_region(0, 1, 0, 1).set_camera(self.cam)
        self.bias = config.FIELD_SHADOW_BIAS_M / (2.0 * depth)
        self.stand_ins: dict = {}

    def texture(self):
        return self.tex

    @staticmethod
    def _switch(on: bool):
        # Read by the shaders as field_on: with no cars in the map there is
        # nothing to look up -- and what the map last held (another session)
        # must not show through.
        from ursina import scene
        scene.set_shader_input("field_on", 1.0 if on else 0.0)

    def add(self, key, model):
        """A stand-in for one car, from a copy of its geometry (any
        NodePath: its own pose is ignored, the car's is copied each frame)."""
        from panda3d.core import NodePath
        np_ = NodePath("stand_in")
        for gn in model.find_all_matches("**/+GeomNode"):
            c = NodePath(gn.node().make_copy())
            node = c.node()
            for i in range(node.get_num_geoms()):
                node.set_geom_state(i, RenderState.make_empty())
            c.set_state(RenderState.make_empty())
            for k in list(node.get_python_tag_keys()):
                node.clear_python_tag(k)
            c.reparent_to(np_)
            c.set_transform(gn.get_transform(model))
        np_.flatten_strong()
        np_.reparent_to(self.root)
        if not self.stand_ins:
            self._switch(True)
        self.stand_ins[key] = np_
        if self.buf is not None:
            self.buf.set_active(True)

    def pose(self, key, car, shown: bool = True):
        s = self.stand_ins.get(key)
        if s is None:
            return
        if not shown:
            if not s.is_hidden():
                s.hide()
            return
        if s.is_hidden():
            s.show()
        import builtins
        s.set_mat(car.get_mat(builtins.render))

    def clear(self):
        for s in self.stand_ins.values():
            s.remove_node()
        if self.stand_ins:
            self._switch(False)
        self.stand_ins.clear()
        if self.buf is not None:
            self.buf.set_active(False)

    def follow(self, p, view_xz=None):
        """Centre the film on *p* (pushed ahead along *view_xz*), in whole
        texels of the film so the cars' shadows do not crawl."""
        if not self.stand_ins:
            return
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        if view_xz is not None:
            vx, vz = view_xz
            n = math.hypot(vx, vz)
            if n > 1e-6:
                a = config.FIELD_SHADOW_AHEAD / n
                x += vx * a
                z += vz * a
        pt = Vec3(x, y, z)
        texel = self.film / self.res
        r = round(pt.dot(self._right) / texel) * texel
        u = round(pt.dot(self._up) / texel) * texel
        f = pt.dot(self._fwd)
        pos = self._right * r + self._up * u + self._fwd * f
        self.cam.set_pos(pos)

    def destroy(self):
        import builtins
        self.clear()
        if self.buf is not None:
            builtins.base.graphicsEngine.remove_window(self.buf)
            self.buf = None
        self.root.remove_node()
