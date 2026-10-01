"""A loading card, so the menu <-> race jump is a transition, not a freeze.

Ursina builds the whole circuit -- track mesh, scenery, mountains, baked
shadows, the ghost's policy -- on a single frame inside ``Game.__init__``,
which locks the window for a second or two. This drops a card in from the top
first and lets it render, then runs the heavy build on a later frame, holds a
beat so the finished scene is behind the card, and slides the card up and off.

Both moves are slides, not cross-fades: Ursina will not fade an opaque quad or
a TextNode by alpha, and a slide has nothing to go wrong. The transition owns
the update loop while it is alive -- ``app.update`` routes to ``Loading.tick``
instead of the game, and input is swallowed.
"""
from __future__ import annotations

from ursina import Entity, Text, camera, time

from . import palette as pal
from .ui import (GREY, INK, RED, WHITE, destroy_tree, pick_font, skew_quad,
                 spaced)


class Loading:
    ENTER = 0.38         # seconds to drop the card in from the top
    SHOW_FRAMES = 2      # frames it renders in place before the heavy build
    HOLD = 0.45          # seconds the built scene sits under the card
    WIPE = 0.42          # seconds to slide the card up and off
    TRAVEL = 1.4         # units of vertical travel; > one screen height

    def __init__(self, title: str, build):
        self._build = build
        self._frame = 0
        self._t = 0.0
        self._phase = "enter"      # enter -> show -> build -> hold -> wipe -> done
        self.done = False
        font = pick_font()

        # A single root, well in front of everything else on camera.ui (the
        # menu's text is at z=-0.1, a pause card's text reaches ~-0.35), so the
        # card fully covers whatever is being torn down behind it. Starts one
        # full travel above centre and eases down.
        self.root = Entity(parent=camera.ui, z=-0.45, y=self.TRAVEL)

        Entity(parent=self.root, model="quad", color=INK, scale=(4, 1.5),
               position=(0, 0, 0.02))
        # The wordmark, the same leaning tile the menu and HUD use.
        Entity(parent=self.root, model=skew_quad(0.052, 0.125), color=RED,
               position=(-0.243, 0.010, -0.02))
        Text("FORMULA-AI", parent=self.root, font=font, scale=1.7,
             origin=(-0.5, 0), position=(-0.205, 0.012, -0.03), color=WHITE)
        Text(spaced("loading"), parent=self.root, font=font, scale=0.72,
             origin=(-0.5, 0), position=(-0.202, -0.030, -0.03), color=GREY)
        Text(title.upper(), parent=self.root, font=font, scale=0.9,
             origin=(1, 0), position=(0.30, 0.012, -0.03), color=RED)

        # A sliver that sweeps the track's width the whole time the card is up.
        # Track bar spans [-0.31, 0.31]; the sweep is 0.16 wide, so its centre
        # travels [-0.23, 0.23] and stays flush inside both ends.
        self._sweep_x = 0.23
        Entity(parent=self.root, model="quad", color=pal.rgb(38, 38, 50),
               scale=(0.62, 0.005), position=(0, -0.075, -0.01))
        self._sweep = Entity(parent=self.root, model="quad", color=RED,
                             scale=(0.16, 0.005),
                             position=(-self._sweep_x, -0.075, -0.02))

    def pump(self, seconds: float = 0.045):
        """Advance the sweep and draw one frame, from inside the heavy build.

        The build is a single blocking call on the main thread, so no frame is
        drawn while it runs and the bar sits still for the second or two it
        takes -- which reads as a hang. Panda can be asked to render a frame
        on its own, without re-entering the task manager or Ursina's update,
        so the card can be pumped between stages of the build. The seconds are
        nominal: time.dt is meaningless here because no real frame boundary
        has passed.
        """
        import builtins

        self._sweep.x += seconds * 0.85
        if self._sweep.x > self._sweep_x:
            self._sweep.x = -self._sweep_x
        try:
            builtins.base.graphicsEngine.render_frame()
        except Exception:      # pragma: no cover -- headless, or no window yet
            pass

    def tick(self):
        dt = min(time.dt, 0.05)
        self._sweep.x += dt * 0.85
        if self._sweep.x > self._sweep_x:
            self._sweep.x = -self._sweep_x

        if self._phase == "enter":
            self._t += dt
            f = min(1.0, self._t / self.ENTER)
            # ease-out: fast in, settles onto the mark
            self.root.y = self.TRAVEL * (1.0 - f) * (1.0 - f)
            if self._t >= self.ENTER:
                self.root.y = 0.0
                self._phase = "show"
        elif self._phase == "show":
            self._frame += 1
            if self._frame >= self.SHOW_FRAMES:
                self._phase = "build"
        elif self._phase == "build":
            self._build(self.pump)        # the heavy frame, pumping as it goes
            self._phase = "hold"
            self._t = 0.0
        elif self._phase == "hold":
            self._t += dt
            if self._t >= self.HOLD:
                self._phase = "wipe"
                self._t = 0.0
        elif self._phase == "wipe":
            self._t += dt
            f = min(1.0, self._t / self.WIPE)
            # ease-in: accelerates off the top like a shutter
            self.root.y = self.TRAVEL * (f * f)
            if self._t >= self.WIPE:
                self.done = True

    def destroy(self):
        destroy_tree(self.root)
        self.root = None
