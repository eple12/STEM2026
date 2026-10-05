"""The circuit itself: road, roadside, mountains, sky and sun.

Everything here depends only on which circuit it is -- nothing about the
session, the cars or the player -- and it is by far the slowest thing to build:
the scenery alone is most of a race's loading time. So it is built once per
circuit and kept. Going back to the menu hides it rather than tearing it down,
and the next session on the same circuit (the same grand prix again, or a
restart from the pause menu) picks it straight back up. Picking a different
circuit is what finally destroys it.
"""
from __future__ import annotations

import builtins

from ursina import window

from . import config
from . import palette as pal
from . import post
from .lighting import Sunset
from .scenery import build_scenery
from .surface import Surface
from .terrain import build_mountains
from .trackdata import load_track
from .trackmesh import TrackScene, line_markers
from .ui import destroy_tree


class World:
    def __init__(self, track_name: str, progress=None):
        step = progress if progress is not None else (lambda: None)
        self.name = track_name
        self.track = load_track(track_name)
        self.light = Sunset(self.track)
        step()
        #: The player's surface. Its only state is a nearest-sample hint,
        #: which ``reset`` puts back on the grid.
        self.surface = Surface(self.track)
        self.scene = TrackScene(self.track)
        step()
        from .shaders import sunset_shader
        self.scenery = build_scenery(self.track, shader=sunset_shader)
        step()
        self.mountains = build_mountains(self.track)
        step()
        # Debug: the centreline in blue, the imported racing line in orange.
        self.markers = (line_markers(self.track) if config.SHOW_LINE_MARKERS
                        else [])

        # The track surface receives shadows but must not cast them.
        self.light.apply(*self.scene.entities, casts=False)
        # The roadside never moves, so its shadows are drawn once into the
        # baked map rather than redrawn into the following one every frame.
        self.light.apply(*self.scenery, baked=True)
        # The range is kilometres away, well outside the shadow film, so it
        # only has to receive.
        self.light.apply(self.mountains, casts=False)
        # After every mask is set, and only once: this is what makes the
        # roadside's shadows static geometry rather than per-frame work.
        step()
        self.light.bake()
        step()

        # The two painted copies of the start lights, picked back out of the
        # flat scenery list. Either may be missing on an old asset bake, and a
        # race without start lights is better than a race that will not start.
        self.lamps = {e.name: e for e in self.scenery
                      if getattr(e, "name", "").startswith("gantry_lamps_")}
        # The five lit lamp columns on the gantry, left to right.
        self.lit_lamps = sorted(
            (e for e in self.scenery
             if getattr(e, "name", "").startswith("gantry_lit_")),
            key=lambda e: e.name)
        self.shown = True
        self._was: list = []

    def _entities(self):
        yield from self.scene.entities
        yield from self.scenery
        if self.mountains is not None:
            yield self.mountains
        yield from self.markers
        yield self.light.sky

    def _sun_buffer(self):
        return self.light.sun._light.get_shadow_buffer(
            builtins.base.win.get_gsg())

    def reset(self):
        """Ready for a new session: the grid, and the start lights dark."""
        self.surface.hint = 0
        for e in self.lit_lamps:
            e.enabled = False

    def show(self):
        # Behind the sky dome, so only ever seen for a frame; the horizon's
        # colour, clamped, since the sky values are linear and run over 1.
        window.color = pal.rgb(*[min(255, int(255 * v ** (1 / 2.2)))
                                 for v in config.SKY_HORIZON])
        # The camera chain is on while a circuit is on screen. Idempotent, so
        # the first show (straight after building) switches it on as well.
        post.enable()
        if self.shown:
            return
        self.shown = True
        # Put back exactly what was on: some of the roadside starts disabled
        # on purpose, and switching everything on would light it up.
        for e, on in self._was:
            e.enabled = on
        self._was = []
        self.light.set_active(True)

    def hide(self):
        """Off the screen, and out of the frame: nothing here is drawn or
        shadow-rendered while the menu is up."""
        if not self.shown:
            return
        self.shown = False
        post.disable()
        self._was = [(e, e.enabled) for e in self._entities()]
        for e, _ in self._was:
            e.enabled = False
        # The sun stays set on the scene root -- the menu is all 2D and never
        # reads it -- but the car's depth pass is switched off, or it would go
        # on rendering an empty scene into its map every frame.
        self.light.set_active(False)

    def forget(self):
        """Drop the light's references to entities a finished session built."""
        self.light._lit = [e for e in self.light._lit if not e.is_empty()]

    def destroy(self):
        """Ursina has no scene-clearing call, so anything created here has to
        be given back by hand -- anything missed stays in the scene graph."""
        post.disable()
        self.light.destroy()
        if self.mountains is not None:
            destroy_tree(self.mountains)
            self.mountains = None
        for e in self.markers:
            destroy_tree(e)
        self.markers = []
        for e in self.scenery:
            destroy_tree(e)
        for e in self.scene.entities:
            destroy_tree(e)
        self.scenery = []
        self.scene.entities = []
        self.lamps = {}
        self.lit_lamps = []
