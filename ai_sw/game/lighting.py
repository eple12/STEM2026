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
                          CullFaceAttrib, GraphicsPipe, LVecBase3f, Mat4,
                          OrthographicLens, Point3, PTA_LVecBase3f,
                          RenderState, SamplerState, Texture, WindowProperties)
from ursina import Entity, Vec3, camera, color, scene
from ursina.lights import DirectionalLight
from ursina.prefabs.sky import Sky

from . import config
from .shaders import sunset_shader


def _sky_texture(size: int = 256):
    """A vertical gradient: deep blue overhead down to the horizon glow.

    Deliberately uniform round the compass rather than painting a sun disc into
    it. The dome's UV mapping would have to be established before a disc could
    be lined up with the actual light direction, and a sun in the wrong part of
    the sky is worse than no sun at all -- the shadows would point somewhere
    else and the eye reads that immediately.
    """
    from PIL import Image
    from ursina import Texture

    top = np.array(config.SKY_ZENITH, dtype=float)
    mid = np.array(config.SKY_MID, dtype=float)
    low = np.array(config.SKY_HORIZON, dtype=float)

    v = np.linspace(0.0, 1.0, size)[:, None]          # 0 = horizon, 1 = zenith
    # Two ramps: horizon -> mid is fast (the glow is a narrow band), mid ->
    # zenith is slow. A single lerp gives a flat, poster-like sky.
    t1 = np.clip(v / config.SKY_GLOW_BAND, 0.0, 1.0) ** 0.65
    t2 = np.clip((v - config.SKY_GLOW_BAND) / (1.0 - config.SKY_GLOW_BAND), 0.0, 1.0)
    band = low + (mid - low) * t1
    rgb = band + (top - band) * (t2 ** 1.15)

    img = np.repeat(rgb[::-1][:, None, :], 8, axis=1)   # V=0 at the image top
    return Texture(Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8),
                                   mode="RGB"))


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

        self.sky = Sky(texture=_sky_texture(), color=color.white)

        self.sun = DirectionalLight(shadow_map_resolution=config.SHADOW_RESOLUTION,
                                    shadows=True)
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
        self._focus_lens()

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

        self._shadow_center = PTA_LVecBase3f.empty_array(1)
        self._uniforms = {
            'sky_color': Vec3(*config.LIGHT_SKY),
            'ground_color': Vec3(*config.LIGHT_BOUNCE),
            'sun_color': Vec3(*config.LIGHT_SUN),
            'sun_wrap': config.LIGHT_SUN_WRAP,
            'glow_color': Vec3(*config.LIGHT_GLOW),
            'glow_strength': config.LIGHT_GLOW_STRENGTH,
            # Horizontal direction towards the sun (the beam travels the other
            # way), so a face can tell whether it is looking into the glow.
            'sun_dir_world': Vec3(-self._dir.x, 0.0, -self._dir.z).normalized(),
            'haze_color': color.rgba(*config.HAZE_COLOR, config.HAZE_DENSITY),
            'haze_start': config.HAZE_START,
            'haze_end': config.HAZE_END,
            'shadow_bias': config.SHADOW_BIAS,
            'shadow_blur': config.SHADOW_BLUR,
            'shadow_samples': config.SHADOW_SAMPLES,
            'shadow_fade_start': config.SHADOW_AREA * config.SHADOW_FADE_START,
            'shadow_fade_end': config.SHADOW_AREA * config.SHADOW_FADE_END,
            'shadow_strength': 1.0,
            # A PTA, not a Vec3: it is written in place every frame by
            # follow(). Assigning a new value to a shader input on the scene
            # root makes a new ShaderAttrib there, which invalidates the
            # composed render state of every node underneath -- all of them
            # were recomposed each frame and the old states left for Panda's
            # state garbage collector. Mutating the array changes the uniform
            # and nothing else.
            'shadow_center': self._shadow_center,
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

    def _focus_lens(self):
        """Point the shadow camera at a box around the origin, not the scene.

        ``DirectionalLight.update_bounds`` fits the film to an entity's extent,
        which for a whole circuit is exactly the useless case. Setting the lens
        by hand, and moving the light with the car, keeps the texels where they
        can be seen. Near is negative so geometry behind the light's own node
        still casts.
        """
        lens = self.sun._light.get_lens()
        lens.set_film_size(config.SHADOW_AREA, config.SHADOW_AREA)
        lens.set_film_offset(0, 0)
        lens.set_near_far(-config.SHADOW_DEPTH, config.SHADOW_DEPTH)

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
              baked=False):
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
        for e in entities:
            if e is None:
                continue
            for target in self._walk(e):
                if not casts or drop_live:
                    target.hide(self.SHADOW_MASK)
                if not into_bake:
                    target.hide(self.BAKE_MASK)
                target.shader = sunset_shader
                # Only what actually differs per entity is set per entity.
                if spec_strength is not None:
                    target.set_shader_input('spec_strength', spec_strength)
                if spec_power is not None:
                    target.set_shader_input('spec_power', spec_power)
                self._lit.append(target)

    @staticmethod
    def _walk(e):
        yield e
        for child in getattr(e, 'children', ()):
            yield from Sunset._walk(child)

    # -- per frame -----------------------------------------------------
    def follow(self, x: float, z: float):
        """Move the shadow map's focus to (x, z), and tell the shader where.

        One shader input on the scene root, not one per lit entity: that was
        the single most expensive thing in the frame outside the draw call.
        And written into the array it was bound with rather than re-set, so
        the scene's render state does not change -- see ``_uniforms``.
        """
        self.sun.position = Vec3(x, config.SHADOW_HEIGHT, z)
        self._shadow_center[0] = LVecBase3f(x, 0.0, z)

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
