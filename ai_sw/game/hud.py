"""On-screen display: an F1 broadcast overlay, laid out the way F1 TV lays one out.

Top-left, the session bar -- wordmark tile, lap counter, circuit, session
clock -- with the **timing tower** hanging under it: one row per car, position
block, team-colour bar, three-letter name, gap, best lap. Top-right, the
**lap-time block**: the running lap with its three sector lights beneath it,
then last and best. Bottom-centre, the **onboard telemetry widget**: shift
lights across the top, speed, gear, throttle and brake bars, steering. The
track map sits bottom-right. Status flags and the spectator lower-third come
up bottom-left. On the grid, the five-lamp **start gantry** counts the race in.

Layout note: on an aspect-ratio change Ursina rescales ``e.x`` for every
*direct child* of ``camera.ui`` (see window.update_aspect_ratio). Anything made
of several pieces must therefore live inside a single container entity, or the
pieces drift apart from each other. So everything here hangs off one
``self.root`` at x = 0, which is also what makes ``destroy()`` a single call
when the race hands back to the menu.

Panels use GLASS rather than the menu's opaque PANEL: a broadcast overlay sits
on top of a moving picture and has to let the track show through.
"""
from __future__ import annotations

import math

from ursina import Circle, Entity, Mesh, Text, Vec3, camera, destroy, window

from . import config
from . import palette as pal
from .trackdata import Track
from .ui import (AMBER, GLASS, GLASS_HI, GREEN, GREY, GREY_DIM, INK, KEYCAP,
                 LAMP_OFF, LAMP_ON, LED_BLUE, LED_GREEN, LED_OFF, LED_RED,
                 PANEL_HI, PURPLE, RED, SLANT_DEG, TEAM_AI, TEAM_YOU, WHITE,
                 YELLOW, caption, gear_of, lap_time, pick_font, skew_quad,
                 spaced)

#: Sector-light colour by status. "live" is the sector being driven now.
SECTOR_COL = {"purple": PURPLE, "green": GREEN, "yellow": YELLOW,
              "live": pal.rgb(120, 96, 40), "off": pal.rgb(40, 40, 52)}

#: The key legend along the bottom edge, as (key, what it does).
KEYS = (("W", "THROTTLE"), ("S", "BRAKE"), ("A/D", "STEER"),
        ("SPACE", "HANDBRAKE"), ("R", "RESET"), ("C", "CAMERA"),
        ("G", "WATCH AI"), ("T", "AIDS"), ("M", "MUTE"), ("ESC", "PAUSE"))

N_LEDS = 12


class _QuadBatch:
    """Collects the HUD's static plates into a single mesh.

    Every plate, accent and rule used to be its own Entity, each a node for
    Ursina to walk and Panda to cull and draw every frame. None of them ever
    move. Merged, they are one draw call and one node, and the depth order is
    preserved because z is baked per vertex exactly as it was per entity.
    """

    def __init__(self, ox: float = 0.0, oy: float = 0.0):
        self.ox, self.oy = ox, oy
        self.v: list = []
        self.t: list = []
        self.c: list = []

    def add(self, x, y, z, w, h, angle, col):
        s = math.tan(math.radians(angle)) * h / 2
        corners = ((-w / 2 + s, h / 2), (w / 2 + s, h / 2),
                   (w / 2 - s, -h / 2), (-w / 2 - s, -h / 2))
        n = len(self.v)
        for px, py in corners:
            self.v.append((x + px, y + py, z))
            self.c.append(col)
        # Counter-clockwise on screen, same winding as skew_quad.
        self.t += [n, n + 2, n + 1, n, n + 3, n + 2]

    def build(self, parent):
        if not self.v:
            return None
        from panda3d.core import TransparencyAttrib
        e = Entity(parent=parent,
                   model=Mesh(vertices=self.v, triangles=self.t,
                              colors=self.c, mode="triangle", static=True))
        # Vertex alpha carries the glass panels; without this it is ignored.
        e.setTransparency(TransparencyAttrib.MAlpha)
        return e


class HUD:
    # Tower geometry, shared with the result card so the two tables read as
    # the same table.
    ROW_H = 0.058
    TOWER_W = 0.400
    COL_POS, COL_BAR, COL_NAME, COL_GAP, COL_BEST = 0.026, 0.0555, 0.072, 0.275, 0.388

    def __init__(self, track: Track, total_laps: int):
        self.track = track
        self.total_laps = total_laps
        # The UI plane spans x in [-aspect/2, +aspect/2], so a panel pinned at a
        # fixed x is only in the right place at one aspect ratio. Panels are
        # placed relative to the edge they belong to, and the whole HUD is
        # rebuilt if the ratio changes, because the plates are baked into one
        # merged mesh in root coordinates and cannot be moved afterwards.
        self._aspect = float(window.aspect_ratio)
        self.edge = self._aspect / 2.0
        self.margin = config.HUD_MARGIN
        self.circuit, self.full_name, self.country = caption(track.name)
        self.font = pick_font()
        self.root = Entity(parent=camera.ui)
        self._rpm = 0.0
        self._shown = -1
        self._lit = -1
        #: Static text -- every label that never changes -- is collected here
        #: during the build and then flattened into one node. Ursina gives each
        #: Text its own node, its own draw call and its own shader bind; the
        #: HUD had ~55 of them sitting idle every frame. Merged, they are one.
        self._static_txt: list = []

        self._batch = _QuadBatch()
        # Collect static text only from the widgets that are on screen for the
        # whole race. The flag, spectator strip and the two end cards are
        # toggled and mostly hidden, so their labels cost nothing where they
        # sit and must not be baked into an always-visible node.
        self._collect_static = True
        self._build_session_bar()
        self._build_tower()
        self._build_laptime()
        self._build_telemetry()
        self._build_minimap(track)
        self._collect_static = False
        self._build_flag()
        if config.SHOW_KEY_HINTS:
            self._build_keys()
        self._batch.build(self.root)
        self._freeze_static_text()
        self._build_lights()
        self._build_spectator()
        self._build_finish()
        self._build_pause()

    # -- primitives -----------------------------------------------------
    def _txt(self, s, *, size=1.0, col=WHITE, pos=(0, 0), origin=(-0.5, 0),
             parent=None, z=-0.1, static=True):
        t = Text(s, parent=parent if parent is not None else self.root,
                 font=self.font, scale=size, color=col, origin=origin,
                 position=(pos[0], pos[1], z))
        if static and self._collect_static:
            self._static_txt.append(t)
        return t

    def _freeze_static_text(self):
        """Bake every collected static label into one merged, static node.

        Ursina attaches a live ``TextNode`` per label -- its own scene-graph
        node, its own draw call, its own shader bind, re-evaluated every frame
        for text that never changes. ``TextNode.generate()`` turns each into
        plain glyph geometry; reparented under one holder and flattened, the
        whole set -- same font page, same shader -- collapses to a single
        Geom. Nothing about what is drawn changes.
        """
        from panda3d.core import TransparencyAttrib
        from ursina import destroy
        from ursina.shaders.text_shader import text_shader

        holder = Entity(parent=self.root, name="hud_static_text")
        shdr = getattr(text_shader, "_shader", None)
        if shdr is not None:
            holder.setShader(shdr)
            for k, v in text_shader.default_input.items():
                holder.setShaderInput(k, v)
        holder.setTransparency(TransparencyAttrib.MAlpha)

        for t in self._static_txt:
            for tnp in list(getattr(t, "text_nodes", [])):
                try:
                    geom = tnp.node().generate()
                except Exception:
                    continue
                np = holder.attachNewNode(geom)
                np.setMat(tnp.getMat(self.root))
            destroy(t)
        self._static_txt = []
        holder.flattenStrong()

    def _root_xy(self, e) -> tuple[float, float]:
        """Offset of an anchor entity from the HUD root, however deep it sits.

        The batch bakes plates in root coordinates, so it needs the anchor's
        position *relative to the root*, not to its own parent. Reading just
        ``parent.x`` put the tower's position blocks -- anchored to a row
        inside the tower -- at the row's local offset from the screen centre:
        a pair of grey boxes floating over the car.
        """
        x = y = 0.0
        while e is not None and e is not self.root:
            x += float(e.x)
            y += float(e.y)
            e = e.parent
        return x, y

    def _plate(self, parent, w, h, pos, col=GLASS, angle=0.0):
        """A static panel. angle=0 is a rectangle; the session bar leans."""
        z = pos[2] if len(pos) > 2 else 0.1
        px, py = self._root_xy(parent)
        self._batch.add(px - self._batch.ox + pos[0],
                        py - self._batch.oy + pos[1], z, w, h, angle, col)

    def _rect(self, parent, w, h, pos, col, z=0.06):
        """A static axis-aligned block, batched with the plates."""
        px, py = self._root_xy(parent)
        self._batch.add(px - self._batch.ox + pos[0],
                        py - self._batch.oy + pos[1], z, w, h, 0.0, col)

    @staticmethod
    def _quad(parent, w, h, pos, col, z=0.03, origin=(0, 0)):
        """A block whose colour or size changes at runtime: its own entity."""
        return Entity(parent=parent, model="quad", color=col, origin=origin,
                      scale=(w, h), position=(pos[0], pos[1], z))

    # -- session bar (top-left) ------------------------------------------
    def _build_session_bar(self):
        """One leaning strip: wordmark, lap, circuit, clock, edge to edge.

        Every segment is a parallelogram of the same height and slant, laid
        end to end, so the red tile runs straight into the lap counter the way
        a broadcast lower-third does -- no seam, no gap.
        """
        H = 0.068
        s = math.tan(math.radians(SLANT_DEG)) * H / 2   # the lean, in x
        b = Entity(parent=self.root,
                   position=(-self.edge + self.margin + s, 0.455, 0))
        x = 0.0
        # The wordmark tile, in the slot the broadcast keeps for the F1 logo.
        # Short form here -- the full name does not fit the tile at a legible
        # size, and the menu still carries it in full.
        w = 0.096
        self._plate(b, w, H, (x + w / 2, 0, 0.1), col=RED, angle=SLANT_DEG)
        # Drawn several times at sub-pixel offsets: this font has no bold
        # weight, and a wordmark is the one piece of type on screen that has to
        # carry at a glance.
        for dx, dy in ((0.0, 0.0), (0.0009, 0.0), (-0.0009, 0.0),
                       (0.0, 0.0007), (0.0, -0.0007)):
            self._txt("F-AI", size=1.25, pos=(x + w / 2 + dx, -0.001 + dy),
                      origin=(0, 0), parent=b)
        x += w
        # Lap counter.
        w = 0.165
        self._plate(b, w, H, (x + w / 2, 0, 0.1), angle=SLANT_DEG)
        self._txt(spaced("lap"), size=0.56, col=GREY, pos=(x + 0.020, 0.0),
                  parent=b)
        self.lap = self._txt("1/3", size=1.55, pos=(x + w - 0.024, -0.003),
                             origin=(0.5, 0), parent=b, static=False)
        x += w
        # Circuit.
        self._plate(b, 0.012, H * 0.62, (x, 0, 0.05), col=GLASS_HI,
                    angle=SLANT_DEG)
        w = 0.268
        self._plate(b, w, H, (x + w / 2, 0, 0.1), angle=SLANT_DEG)
        self._txt(self.circuit, size=1.12, pos=(x + 0.020, 0.011), parent=b)
        sub = self.country if self.full_name.upper() == self.circuit \
            else f"{self.country}  ·  {self.full_name.upper()}"
        self._txt(sub, size=0.50, col=GREY, pos=(x + 0.017, -0.016), parent=b)
        x += w
        # Session clock.
        self._plate(b, 0.012, H * 0.62, (x, 0, 0.05), col=GLASS_HI,
                    angle=SLANT_DEG)
        w = 0.150
        self._plate(b, w, H, (x + w / 2, 0, 0.1), angle=SLANT_DEG)
        self._txt(spaced("session"), size=0.50, col=GREY,
                  pos=(x + 0.020, 0.014), parent=b)
        self.clock = self._txt("00:00", size=1.0, pos=(x + 0.020, -0.014),
                               parent=b, static=False)
        self._bar_bottom = 0.455 - H / 2

    # -- timing tower (left) ----------------------------------------------
    def _build_tower(self):
        W, RH = self.TOWER_W, self.ROW_H
        top = self._bar_bottom - 0.012
        t = Entity(parent=self.root, position=(-self.edge + self.margin, top, 0))
        # Column header.
        hh = 0.026
        self._plate(t, W, hh, (W / 2, -hh / 2, 0.1), col=pal.rgb(12, 12, 18, 232))
        y = -hh / 2
        for label, x, org in ((spaced("pos"), self.COL_POS, (0, 0)),
                              (spaced("driver"), self.COL_NAME, (-0.5, 0)),
                              (spaced("gap"), self.COL_GAP, (0.5, 0)),
                              (spaced("best lap"), self.COL_BEST, (0.5, 0))):
            self._txt(label, size=0.46, col=GREY_DIM, pos=(x, y), origin=org,
                      parent=t)
        self.rows = []
        for r in range(2):
            cy = -hh - RH / 2 - r * (RH + 0.003)
            row = Entity(parent=t, position=(0, cy, 0))
            bg = self._quad(row, W, RH, (W / 2, 0), GLASS, z=0.08)
            self._rect(row, 0.052, RH, (self.COL_POS, 0), PANEL_HI, z=0.07)
            pos = self._txt("", size=1.3, pos=(self.COL_POS, -0.002),
                            origin=(0, 0), parent=row, static=False)
            bar = self._quad(row, 0.007, RH, (self.COL_BAR, 0), TEAM_YOU, z=0.05)
            tla = self._txt("", size=1.08, pos=(self.COL_NAME, 0.009), parent=row,
                            static=False)
            name = self._txt("", size=0.46, col=GREY, pos=(self.COL_NAME, -0.015),
                             parent=row, static=False)
            gap = self._txt("", size=0.92, pos=(self.COL_GAP, -0.001),
                            origin=(0.5, 0), parent=row, static=False)
            best = self._txt("", size=0.78, pos=(self.COL_BEST, -0.001),
                             origin=(0.5, 0), parent=row, static=False)
            self.rows.append(dict(row=row, bg=bg, pos=pos, bar=bar, tla=tla,
                                  name=name, gap=gap, best=best))

    # -- lap-time block (top-right) ---------------------------------------
    def _build_laptime(self):
        W = 0.340
        top = 0.455 + 0.034
        r = Entity(parent=self.root,
                   position=(self.edge - self.margin - W, top, 0))
        self._plate(r, W, 0.200, (W / 2, -0.100, 0.1))
        # Header strip with a red tag, the way the onboard "LAP TIME" graphic
        # is captioned.
        self._rect(r, W, 0.026, (W / 2, -0.013), pal.rgb(12, 12, 18, 232), z=0.08)
        self._rect(r, 0.118, 0.026, (0.059, -0.013), RED, z=0.07)
        self._txt(spaced("lap time"), size=0.50, pos=(0.059, -0.013),
                  origin=(0, 0), parent=r)
        # Current, big and amber, the sector lights beneath it.
        self._txt(spaced("current"), size=0.52, col=GREY, pos=(0.016, -0.059),
                  parent=r)
        self.cur = self._txt(lap_time(None), size=1.5, col=AMBER,
                             pos=(W - 0.016, -0.061), origin=(0.5, 0), parent=r,
                             static=False)
        self.sectors = []
        bw = (W - 0.032 - 0.016) / 3
        for k in range(3):
            cx = 0.016 + bw / 2 + k * (bw + 0.008)
            self.sectors.append(self._quad(r, bw, 0.011, (cx, -0.104),
                                           SECTOR_COL["off"], z=0.05))
            self._txt(f"S{k + 1}", size=0.40, col=GREY_DIM,
                      pos=(cx - bw / 2, -0.118), parent=r)
        self._rect(r, W - 0.032, 0.0015, (W / 2, -0.127), GLASS_HI, z=0.07)
        self._txt(spaced("last"), size=0.52, col=GREY, pos=(0.016, -0.147),
                  parent=r)
        self.last = self._txt(lap_time(None), size=1.0, pos=(W - 0.016, -0.147),
                              origin=(0.5, 0), parent=r, static=False)
        self._txt(spaced("best"), size=0.52, col=GREY, pos=(0.016, -0.181),
                  parent=r)
        self.best = self._txt(lap_time(None), size=1.0, col=PURPLE,
                              pos=(W - 0.016, -0.181), origin=(0.5, 0), parent=r,
                              static=False)

    # -- onboard telemetry (bottom-centre) --------------------------------
    BAR_H = 0.096            # throttle / brake bar travel

    def _build_telemetry(self):
        W, H = 0.400, 0.150
        s = Entity(parent=self.root, position=(0, -0.375, 0))
        self._plate(s, W, H, (0, 0, 0.1))
        self._rect(s, W, 0.004, (0, H / 2 - 0.002), RED, z=0.07)

        # Shift lights across the top, lit left to right with the revs.
        self.leds = []
        pitch = 0.019
        for k in range(N_LEDS):
            x = (k - (N_LEDS - 1) / 2) * pitch
            self.leds.append(self._quad(s, 0.015, 0.008, (x, 0.058), LED_OFF, z=0.04))
        self._led_col = ([LED_GREEN] * 4 + [LED_RED] * 4 + [LED_BLUE] * 4)

        # Speed: one Text per digit on a fixed pitch. Bahnschrift's figures are
        # not tabular -- a single Text changes width on almost every update and
        # the number visibly breathes. Each digit centred in its own slot
        # cannot. The slots are laid out around x = 0 for however many digits
        # are in use, so the block only re-centres crossing 10 or 100.
        self.digits = [Text("", parent=s, font=self.font, scale=2.7, color=WHITE,
                            origin=(0, 0), position=(0, 0.004, -0.1))
                       for _ in range(3)]
        self._txt(spaced("km/h"), size=0.50, col=GREY, pos=(0, -0.041),
                  origin=(0, 0), parent=s)
        # Steering, a travelling block on a bar.
        self._rect(s, 0.130, 0.008, (0, -0.062), GLASS_HI, z=0.06)
        self._rect(s, 0.002, 0.014, (0, -0.062), GREY_DIM, z=0.05)
        self.st_dot = self._quad(s, 0.010, 0.014, (0, -0.062), WHITE, z=0.03)

        # Throttle and brake, vertical bars filling upward from a shared floor.
        self._rect(s, 0.0015, 0.100, (-0.100, -0.006), GLASS_HI, z=0.06)
        for name, col, x, lab in (("p_thr", GREEN, -0.160, "thr"),
                                  ("p_brk", RED, -0.130, "brk")):
            self._rect(s, 0.020, self.BAR_H, (x, -0.006), pal.rgb(58, 60, 76, 200),
                       z=0.06)
            setattr(self, name, self._quad(s, 0.020, 0.001, (x, -0.054), col,
                                           z=0.03, origin=(0, -0.5)))
            self._txt(spaced(lab), size=0.36, col=GREY_DIM, pos=(x, -0.066),
                      origin=(0, 0), parent=s)

        # Gear, with the driver-aid and slip badges beside it.
        self._rect(s, 0.0015, 0.100, (0.092, -0.006), GLASS_HI, z=0.06)
        self.gear = self._txt("N", size=2.1, pos=(0.126, 0.008), origin=(0, 0),
                              parent=s, static=False)
        self._txt(spaced("gear"), size=0.44, col=GREY, pos=(0.126, -0.041),
                  origin=(0, 0), parent=s)
        self.aid_txt = self._txt("", size=0.44, pos=(0.172, 0.018), origin=(0, 0),
                                 parent=s, static=False)
        self.slip_txt = self._txt("", size=0.44, col=AMBER, pos=(0.172, -0.014),
                                  origin=(0, 0), parent=s, static=False)

    # -- track map (bottom-right) -----------------------------------------
    def _build_minimap(self, track: Track):
        lo, hi = track.bounds()
        span = float(max(hi[0] - lo[0], hi[1] - lo[1])) or 1.0
        self._mm_scale = 0.18 / span
        self._mm_cx = (hi[0] + lo[0]) / 2
        self._mm_cz = (hi[1] + lo[1]) / 2

        W, H = 0.270, 0.250
        m = Entity(parent=self.root,
                   position=(self.edge - self.margin - W / 2, -0.325, 0))
        self.minimap = m
        self._plate(m, W, H, (0, 0, 0.1))
        self._rect(m, W, 0.004, (0, H / 2 - 0.002), RED, z=0.07)
        # Legend, bottom-left corner of the map.
        for k, (lab, col) in enumerate(((" YOU", TEAM_YOU), (" AI", TEAM_AI))):
            x = -W / 2 + 0.020 + k * 0.058
            Entity(parent=m, model="circle", scale=0.010, color=col,
                   position=(x, -H / 2 + 0.018, -0.02))
            self._txt(lab, size=0.44, col=GREY, pos=(x + 0.006, -H / 2 + 0.018),
                      parent=m)

        # Decimated. The centreline has over a thousand samples; drawn into a
        # widget 0.2 units across that is a thick line segment per one-fifth of
        # a pixel. ~180 points traces the same shape.
        step = max(1, len(track.center) // config.MINIMAP_POINTS)
        pts = [Vec3(*self._to_mm(p), 0) for p in track.center[::step]]
        pts.append(pts[0])
        Entity(parent=m, model=Mesh(vertices=pts, mode="line", thickness=3),
               color=WHITE, z=-0.01, y=0.008)
        s, t = track.center[0], track.normal[0]
        half = span * 0.035
        Entity(parent=m, color=RED, z=-0.02, y=0.008,
               model=Mesh(vertices=[Vec3(*self._to_mm(s - t * half), 0),
                                    Vec3(*self._to_mm(s + t * half), 0)],
                          mode="line", thickness=6))
        # The AI under the player's dot rather than over it: when the two are
        # together the one you are driving is the one you need to see.
        self.mm_ghost = Entity(parent=m, model="circle", scale=0.017,
                               color=TEAM_AI, position=(0, 0, -0.03),
                               enabled=False)
        self.mm_dot = Entity(parent=m, model="circle", scale=0.020,
                             color=TEAM_YOU, position=(0, 0, -0.04))

    def _to_mm(self, xz):
        """Track world (x, z) -> minimap-local (u, v)."""
        u = (float(xz[0]) - self._mm_cx) * self._mm_scale
        v = (float(xz[1]) - self._mm_cz) * self._mm_scale
        return float(u), float(v)

    # -- flags, lower-third, start lights, keys ----------------------------
    def _build_flag(self):
        """A marshal's flag: OFF TRACK on yellow, WRONG WAY on red."""
        f = Entity(parent=self.root,
                   position=(-self.edge + self.margin + 0.130, -0.410, 0),
                   enabled=False)
        self.flag = f
        self.flag_plate = Entity(parent=f, model=skew_quad(0.240, 0.050),
                                 color=YELLOW, position=(0, 0, 0.05))
        self.flag_txt = self._txt("", size=0.86, col=INK, pos=(0.004, 0),
                                  origin=(0, 0), parent=f)

    def _build_spectator(self):
        """The lower-third that comes up when the camera is on the AI."""
        c = Entity(parent=self.root,
                   position=(-self.edge + self.margin, -0.410, 0), enabled=False)
        self.spectator = c
        self._batch = _QuadBatch(c.x, c.y)
        self._plate(c, 0.070, 0.072, (0.035, 0, 0.1), col=RED)
        self._txt("AI", size=1.35, pos=(0.035, -0.002), origin=(0, 0), parent=c)
        self._plate(c, 0.330, 0.072, (0.070 + 0.165, 0, 0.1))
        self._txt("AI DRIVER", size=1.1, pos=(0.086, 0.011), parent=c)
        self._txt(spaced("onboard") + "   ·   " + spaced("trained policy"),
                  size=0.46, col=GREY, pos=(0.086, -0.017), parent=c)
        Entity(parent=c, model="circle", scale=0.011, color=RED,
               position=(0.382, 0.011, -0.02))
        self._txt(spaced("live"), size=0.44, col=RED, pos=(0.366, 0.011),
                  origin=(0.5, 0), parent=c)
        self._batch.build(c)
        self._batch = _QuadBatch()

    def _build_lights(self):
        """Five round lamps on a gantry. They come on one by one and go out
        together.

        Each lamp is its own ``Circle`` mesh, and it is recoloured with
        ``model.setColorScale`` (see ``_lamp``), not ``entity.color``:
        assigning ``.color`` on a nested ``Circle`` renders grey -- only a
        colour scale straight on the mesh node takes.
        """
        self._gantry_y0 = 0.150
        g = Entity(parent=self.root, position=(0, self._gantry_y0, 0),
                   enabled=False)
        self.gantry = g
        Entity(parent=g, model="quad", color=pal.rgb(12, 12, 18),
               scale=(0.470, 0.120), position=(0, 0, 0.03))
        Entity(parent=g, model="quad", color=RED,
               scale=(0.470, 0.005), position=(0, 0.060, 0.02))
        self.lamps = []
        for k in range(5):
            x = (k - 2) * 0.088
            Entity(parent=g, model=Circle(resolution=28, mode="ngon"),
                   color=pal.rgb(34, 34, 42), scale=0.088, position=(x, 0, 0.01))
            lamp = Entity(parent=g, model=Circle(resolution=28, mode="ngon"),
                          scale=0.070, position=(x, 0, 0.0))
            self._lamp(lamp, LAMP_OFF)
            self.lamps.append(lamp)

    @staticmethod
    def _lamp(lamp, col):
        """Colour one lamp. A colour scale on the mesh node -- ``.color`` on a
        nested Circle does not render."""
        m = lamp.model
        m.setColorScaleOff()
        m.setColorScale(col)
        lamp._lit_col = col

    def _build_keys(self):
        """The key legend, as keycaps rather than a line of text."""
        y = -0.481
        self._rect(self.root, 4.0, 0.038, (0, y), pal.rgb(12, 12, 18, 228), z=0.2)
        # Estimated widths; they only set the spacing.
        cw = 0.0072
        items = []
        for key, label in KEYS:
            tw = 0.012 + 0.0080 * len(key)
            items.append((key, label, tw, cw * len(label)))
        total = sum(tw + 0.008 + lw for _, _, tw, lw in items) + 0.030 * (len(items) - 1)
        x = -total / 2
        for key, label, tw, lw in items:
            self._rect(self.root, tw, 0.024, (x + tw / 2, y), KEYCAP, z=0.15)
            self._txt(key, size=0.50, pos=(x + tw / 2, y), origin=(0, 0))
            self._txt(label, size=0.50, col=GREY, pos=(x + tw + 0.008, y))
            x += tw + 0.008 + lw + 0.030

    # -- cards ----------------------------------------------------------
    def _build_finish(self):
        """Result card: the classification, built once and hidden."""
        f = Entity(parent=self.root, position=(0, 0.03, -0.2), enabled=False)
        self.finish = f
        W, H = 0.820, 0.390
        self._batch = _QuadBatch(f.x, f.y)
        # Opaque: a result card is a statement, not an overlay.
        self._plate(f, W, H, (0, 0, 0.1), col=INK)
        self._rect(f, W, 0.030, (0, H / 2 - 0.015), pal.rgb(12, 12, 18), z=0.08)
        self._rect(f, 0.150, 0.030, (-W / 2 + 0.075, H / 2 - 0.015), RED, z=0.07)
        self._txt(spaced("race result"), size=0.52, pos=(-W / 2 + 0.075, H / 2 - 0.015),
                  origin=(0, 0), parent=f)
        self._txt(self.circuit, size=1.7, pos=(-W / 2 + 0.036, 0.112), parent=f)
        self._txt(f"{self.country}  ·  {self.total_laps} LAPS", size=0.52, col=GREY,
                  pos=(-W / 2 + 0.036, 0.076), parent=f)
        # Classification, on the tower's columns, spread to the wider card.
        x0 = -W / 2 + 0.036
        cols = dict(pos=x0 + 0.026, bar=x0 + 0.0555, name=x0 + 0.072,
                    best=x0 + 0.470, gap=x0 + 0.720)
        hy = 0.040
        for label, x, org in ((spaced("pos"), cols["pos"], (0, 0)),
                              (spaced("driver"), cols["name"], (-0.5, 0)),
                              (spaced("best lap"), cols["best"], (0.5, 0)),
                              (spaced("gap"), cols["gap"], (0.5, 0))):
            self._txt(label, size=0.46, col=GREY_DIM, pos=(x, hy), origin=org,
                      parent=f)
        self.fin_rows = []
        RH = 0.062
        for r in range(2):
            cy = hy - 0.020 - RH / 2 - r * (RH + 0.004)
            row = Entity(parent=f, position=(0, cy, 0))
            bg = self._quad(row, W - 0.060, RH, (0, 0), PANEL_HI, z=0.06)
            pos = self._txt("", size=1.3, pos=(cols["pos"], -0.002), origin=(0, 0),
                            parent=row)
            bar = self._quad(row, 0.007, RH, (cols["bar"], 0), TEAM_YOU, z=0.04)
            tla = self._txt("", size=1.1, pos=(cols["name"], 0.010), parent=row)
            name = self._txt("", size=0.46, col=GREY, pos=(cols["name"], -0.015),
                             parent=row)
            best = self._txt("", size=1.05, pos=(cols["best"], -0.001),
                             origin=(0.5, 0), parent=row)
            gap = self._txt("", size=0.90, pos=(cols["gap"], -0.001),
                            origin=(0.5, 0), parent=row)
            self.fin_rows.append(dict(bg=bg, pos=pos, bar=bar, tla=tla, name=name,
                                      best=best, gap=gap))
        self._rect(f, W - 0.060, 0.0015, (0, -H / 2 + 0.058), PANEL_HI, z=0.05)
        self._txt(spaced("enter") + "   or   " + spaced("esc")
                  + "        BACK TO MENU", size=0.60, col=GREY,
                  pos=(x0, -H / 2 + 0.032), parent=f)
        self._batch.build(f)
        self._batch = _QuadBatch()

    def _build_pause(self):
        p = Entity(parent=self.root, position=(0, 0.03, -0.25), enabled=False)
        self.pause = p
        W, H = 0.560, 0.250
        self._batch = _QuadBatch(p.x, p.y)
        self._plate(p, W, H, (0, 0, 0.1), col=INK)
        self._rect(p, W, 0.030, (0, H / 2 - 0.015), pal.rgb(12, 12, 18), z=0.08)
        self._rect(p, 0.120, 0.030, (-W / 2 + 0.060, H / 2 - 0.015), RED, z=0.07)
        self._txt(spaced("paused"), size=0.52, pos=(-W / 2 + 0.060, H / 2 - 0.015),
                  origin=(0, 0), parent=p)
        self._txt(self.circuit, size=1.7, pos=(-W / 2 + 0.030, 0.040), parent=p)
        self._txt(f"{self.country}  ·  {self.total_laps} LAPS", size=0.52, col=GREY,
                  pos=(-W / 2 + 0.030, 0.004), parent=p)
        self._rect(p, W - 0.060, 0.0015, (0, -0.030), PANEL_HI, z=0.05)
        self._txt(spaced("esc") + "        RESUME", size=0.70,
                  pos=(-W / 2 + 0.030, -0.060), parent=p)
        self._txt(spaced("enter") + "      EXIT TO MENU", size=0.70, col=GREY,
                  pos=(-W / 2 + 0.030, -0.098), parent=p)
        self._batch.build(p)
        self._batch = _QuadBatch()

    # -- per-frame update ------------------------------------------
    @staticmethod
    def _set(entity, value: str):
        """Assigning Text.text rebuilds the glyph mesh, which costs about as
        much as the whole physics step. Most of these strings are identical
        frame to frame, so only write the ones that actually changed."""
        if entity.text != value:
            entity.text = value

    @staticmethod
    def _col(entity, col):
        if entity.color != col:
            entity.color = col

    DIGIT_PITCH = 0.039          # the widest glyph at scale 2.7 (0.0144 * 2.7)

    def _set_speed(self, kmh: float):
        text = f"{max(0.0, kmh):0.0f}"[-3:]
        n = len(text)
        if n != self._shown:      # only re-centre when the digit count changes
            self._shown = n
            for i, d in enumerate(self.digits):
                d.x = (i - (n - 1) / 2) * self.DIGIT_PITCH
        for i, d in enumerate(self.digits):
            self._set(d, text[i] if i < n else "")

    def _set_row(self, row, entry, leader: bool, opaque: bool = False):
        self._set(row["pos"], str(entry["pos"]))
        self._col(row["bar"], entry["col"])
        self._set(row["tla"], entry["tla"])
        self._set(row["name"], entry["name"])
        self._set(row["best"], lap_time(entry["best"]))
        self._col(row["best"], PURPLE if entry.get("purple") else WHITE)
        self._set(row["gap"], entry["gap"])
        self._col(row["gap"], WHITE if leader else GREY)
        if opaque:
            self._col(row["bg"], pal.rgb(48, 48, 64) if leader else PANEL_HI)
        else:
            self._col(row["bg"], GLASS_HI if leader else GLASS)

    def stale(self) -> bool:
        """True when the window's shape has changed under the layout.

        The panels are anchored to the screen edges at build time and the
        plates are baked into one merged mesh, so a new ratio means a rebuild
        rather than a nudge. Cheap enough: it happens when someone presses F.
        """
        return abs(float(window.aspect_ratio) - self._aspect) > 1e-3

    def update(self, *, speed_kmh, speed_frac, lap, cur_t, last_t, best_t,
               session_t, sectors, standings, lights=-1, gantry_dy=0.0,
               flag="", spectating=False, car_xz=None, ghost_xz=None,
               throttle=0.0, brake=0.0, steer=0.0, slip=0.0, tc_cut=0.0,
               esc_cut=0.0, tc_off=False, dt=1.0 / 60.0, finished=False,
               paused=False):
        # -- session bar ---------------------------------------------
        # Past the last lap the car is on an in-lap, not on lap 4 of 3, and
        # clamping to "3/3" made a car still circulating look like it was on
        # its final tour for ever.
        self._set(self.lap, "OUT" if lap == 0
                  else "FIN" if lap > self.total_laps
                  else f"{lap}/{self.total_laps}")
        m, s = divmod(int(session_t), 60)
        self._set(self.clock, f"{m:02d}:{s:02d}")

        # -- lap time + sectors --------------------------------------
        self._set(self.cur, lap_time(cur_t))
        self._set(self.last, lap_time(last_t))
        self._set(self.best, lap_time(best_t))
        # Green when the lap just completed *is* the best -- the timing-tower
        # cue for a personal best that has not yet been beaten.
        self._col(self.last, GREEN if last_t is not None and best_t is not None
                  and abs(last_t - best_t) < 1e-6 else WHITE)
        for block, (_, status) in zip(self.sectors, sectors):
            self._col(block, SECTOR_COL.get(status, SECTOR_COL["off"]))

        # -- tower ----------------------------------------------------
        for k, row in enumerate(self.rows):
            if k < len(standings):
                row["row"].enabled = True
                self._set_row(row, standings[k], leader=k == 0)
            else:
                row["row"].enabled = False

        # -- telemetry ----------------------------------------------
        self._set_speed(speed_kmh)
        gear, within = gear_of(speed_frac)
        self._set(self.gear, str(gear) if speed_kmh > 2.0 else "N")
        # Smoothed the way the engine note is, so the strip does not flicker
        # on the gear change.
        target = 0.30 + 0.70 * within
        self._rpm += (target - self._rpm) * min(1.0, 9.0 * dt)
        lit = int(self._rpm * N_LEDS + 0.5)
        if lit != self._lit:
            self._lit = lit
            for k, led in enumerate(self.leds):
                self._col(led, self._led_col[k] if k < lit else LED_OFF)
        self.p_thr.scale_y = max(0.001, self.BAR_H * throttle)
        self.p_brk.scale_y = max(0.001, self.BAR_H * brake)
        self.st_dot.x = steer * 0.060

        deg = math.degrees(slip)
        if deg > 6.0:
            self._set(self.slip_txt, f"SLIP {deg:0.0f}°")
            self._col(self.slip_txt, AMBER if deg < 14 else RED)
        else:
            self._set(self.slip_txt, "")
        if tc_off:
            self._set(self.aid_txt, "AIDS OFF")
            self._col(self.aid_txt, RED)
        elif esc_cut > 0.02:
            # "ESP", not "ESC": ESC is the pause key, and an indicator sharing
            # a name with a key is a needless thing to explain to a visitor.
            self._set(self.aid_txt, "ESP")
            self._col(self.aid_txt, GREEN)
        else:
            self._set(self.aid_txt, "TC" if tc_cut > 0.05 else "")
            self._col(self.aid_txt, AMBER)

        # -- map ------------------------------------------------------
        if ghost_xz is None:
            if self.mm_ghost.enabled:
                self.mm_ghost.enabled = False
        else:
            if not self.mm_ghost.enabled:
                self.mm_ghost.enabled = True
            u, v = self._to_mm(ghost_xz)
            self.mm_ghost.position = (u, v + 0.008, -0.03)
        if car_xz is not None:
            u, v = self._to_mm(car_xz)
            self.mm_dot.position = Vec3(u, v + 0.008, -0.04)

        # -- start lights -------------------------------------------
        # ``lights`` is the lamp count 0-5, or -1 when the gantry is off
        # screen. ``gantry_dy`` slides it down into place and later up and away.
        show = lights >= 0
        if self.gantry.enabled != show:
            self.gantry.enabled = show
        if show:
            self.gantry.y = self._gantry_y0 + gantry_dy
            for k, lamp in enumerate(self.lamps):
                col = LAMP_ON if k < lights else LAMP_OFF
                if getattr(lamp, "_lit_col", None) != col:
                    self._lamp(lamp, col)

        # -- flag / lower-third (one slot, bottom-left) ----------------
        if self.spectator.enabled != spectating:
            self.spectator.enabled = spectating
        show_flag = bool(flag) and not spectating
        if self.flag.enabled != show_flag:
            self.flag.enabled = show_flag
        if show_flag:
            self._set(self.flag_txt, flag)
            red = flag == "WRONG WAY"
            self._col(self.flag_plate, RED if red else YELLOW)
            self._col(self.flag_txt, WHITE if red else INK)

        # -- cards ----------------------------------------------------
        if self.finish.enabled != finished:
            self.finish.enabled = finished
        if finished:
            for k, row in enumerate(self.fin_rows):
                if k < len(standings):
                    self._set_row(row, standings[k], leader=k == 0, opaque=True)
        if self.pause.enabled != paused:
            self.pause.enabled = paused

    def destroy(self):
        destroy(self.root)
        self.root = None
