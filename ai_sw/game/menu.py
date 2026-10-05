"""Start menu: title card and circuit select, styled after an F1 broadcast.

The look leans on three things the real thing does: a near-black ground with a
single saturated red, angular shapes slanted the same way everywhere, and type
that is condensed, uppercase and widely tracked. Everything is drawn from
primitives, so there is no image asset to keep in sync.

Circuit statistics are derived from the same geometry the game drives, not
looked up -- so the length shown is the length you actually lap. Turn count
comes from the curvature profile and lands within a turn or two of the official
figure (Monza 10 vs 11, Austin 20 vs 20); it is a description of the model, not
a claim about the real circuit.
"""
from __future__ import annotations

import numpy as np
from ursina import Entity, Mesh, Text, Vec3, camera

from . import config, teams
from . import palette as pal
from .trackdata import Track, load_track

from .ui import destroy_tree
from .ui import (CIRCUITS, GREY, GREY_DIM, INK, PANEL, PANEL_HI, PURPLE, RED,
                 WHITE, caption, lap_time, pick_font, skew_quad, spaced)

ROWS = 9                          # circuits visible at once

# --- helpers -------------------------------------------------------------
def available_circuits() -> list[str]:
    """Circuit folders that actually hold data, ordered as CIRCUITS lists them."""
    if not config.TRACK_DB.is_dir():
        return []
    on_disk = {p.name for p in config.TRACK_DB.iterdir() if p.is_dir()}
    known = [n for n in CIRCUITS if n in on_disk]
    return known + sorted(on_disk - set(CIRCUITS))





def circuit_stats(track: Track, thresh: float = 400.0, min_run: int = 3):
    """(turns, longest straight in metres) read off the curvature profile."""
    corner = np.abs(track.curv_radius) < thresh
    straights = np.flatnonzero(~corner)
    if len(straights) == 0:
        return 0, 0.0
    # Start the scan on a straight, or a corner that happens to span the
    # start/finish line gets counted once on each side.
    corner = np.roll(corner, -straights[0])
    seg = np.roll(track.seg_len, -straights[0])

    turns = run = 0
    best = cur = 0.0
    for c, length in zip(corner, seg):
        if c:
            run += 1
            cur = 0.0
        else:
            if run >= min_run:
                turns += 1
            run = 0
            cur += float(length)
            best = max(best, cur)
    if run >= min_run:
        turns += 1
    return turns, best


QUALI, GRAND_PRIX = "quali", "gp"

#: (key, title, the one line of facts under it) per session, in menu order.
MODES = (
    (QUALI, "QUALIFYING", "OUT LAP  +  {laps} TIMED LAPS"),
    (GRAND_PRIX, "GRAND PRIX", "{laps} LAPS  ·  20 CARS"),
)

#: The keys, as the controls panel lists them: (keys, what they do).
CONTROLS = (
    ("W / UP", "THROTTLE"), ("S / DOWN", "BRAKE"), ("A D / LEFT RIGHT", "STEER"),
    ("SPACE", "HANDBRAKE"), ("R", "BACK ON TRACK"),
    ("C", "CAMERA"), ("X", "LOOK BACK"), ("G / SHIFT+G", "WATCH CARS"),
    ("TAB", "INTERVAL / LEADER"), ("T", "DRIVER AIDS"),
    ("H", "HIDE HUD"), ("M", "MUTE"), ("ESC", "PAUSE"),
)

#: The chosen difficulty box while the session cards have the keys: still
#: red, but not the live red of the row being changed.
RED_IDLE = pal.rgb(120, 16, 12)


def _base_screen(menu, subtitle: str):
    """Ground, wordmark and footer rule shared by both menu screens."""
    Entity(parent=menu.root, model="quad", color=INK,
           scale=(4.0, 1.4), position=(0, 0, 0.5))
    Entity(parent=menu.root, model=skew_quad(0.045, 0.115),
           color=RED, position=(-0.80, 0.395, 0.2))
    menu._txt("FORMULA-AI", size=3.4, pos=(-0.755, 0.395))
    menu._txt(spaced(subtitle), size=0.85, col=GREY, pos=(-0.752, 0.315))
    Entity(parent=menu.root, model="quad", color=PANEL_HI,
           scale=(1.78, 0.0035), position=(0, -0.452, 0.2))


class ModeMenu:
    """The main menu: qualifying or grand prix, the difficulty, and the
    controls. Calls ``on_pick(mode)``; ``on_level(level)`` on every change
    of difficulty.

    Two rows take the keys: the session cards (A/D) and the difficulty
    boxes (A/D once W/S has moved down to them). ENTER goes on from either.
    """

    CARD_W, CARD_H = 0.78, 0.205
    CARD_Y = 0.110
    #: The difficulty panel: top edge and height, and the boxes in it.
    LV_TOP, LV_H = -0.030, 0.190
    BOX_H, BOX_GAP = 0.112, 0.012
    #: The controls panel's top edge, and both panels' width.
    KEYS_TOP = -0.245
    PANEL_W = 1.62

    def __init__(self, on_pick, on_quit, laps: int = config.TOTAL_LAPS,
                 initial: str | None = None, level: int = teams.DEFAULT_LEVEL,
                 on_level=None):
        self.on_pick = on_pick
        self.on_quit = on_quit
        self.on_level = on_level or (lambda lv: None)
        self.font = pick_font()
        keys = [m[0] for m in MODES]
        self.sel = keys.index(initial) if initial in keys else 0
        self.levels = sorted(teams.DIFFICULTY)
        self.level = level if level in teams.DIFFICULTY else teams.DEFAULT_LEVEL
        self.row = 0                 # 0: session cards, 1: difficulty
        self.root = Entity(parent=camera.ui)
        _base_screen(self, "racing")

        Entity(parent=self.root, model="quad", color=PANEL_HI,
               scale=(1.60, 0.0035), position=(0, 0.275, 0.2))
        Entity(parent=self.root, model="quad", color=RED,
               scale=(0.30, 0.006), position=(-0.65, 0.275, 0.15))
        self._txt(spaced("main menu"), size=0.8, col=GREY, pos=(-0.80, 0.245))

        H = self.CARD_H
        self.cards = []
        for k, (_key, title, detail) in enumerate(MODES):
            cx = -0.40 + k * 0.82
            c = Entity(parent=self.root, position=(cx, self.CARD_Y, 0.1))
            bg = Entity(parent=c, model="quad", color=PANEL,
                        scale=(self.CARD_W, H), position=(0, 0, 0.08))
            band = Entity(parent=c, model="quad", color=RED,
                          scale=(self.CARD_W, 0.010),
                          position=(0, H / 2 - 0.005, 0.06))
            tag = Entity(parent=c, model=skew_quad(0.075, 0.040), color=PANEL_HI,
                         position=(-self.CARD_W / 2 + 0.070, H / 2 - 0.042, 0.05))
            x0 = -self.CARD_W / 2 + 0.040
            idx = Text(f"{k + 1:02d}", parent=c, font=self.font, scale=0.9,
                       color=GREY, origin=(0, 0),
                       position=(-self.CARD_W / 2 + 0.070, H / 2 - 0.042, -0.1))
            name = Text(title, parent=c, font=self.font, scale=2.4, color=WHITE,
                        origin=(-0.5, 0), position=(x0, H / 2 - 0.108, -0.1))
            # INK, not PANEL_HI: the selected card's ground *is* PANEL_HI.
            Entity(parent=c, model="quad", color=INK,
                   scale=(self.CARD_W - 0.080, 0.0025),
                   position=(0, -H / 2 + 0.050, 0.05))
            info = Text(detail.format(laps=laps), parent=c, font=self.font,
                        scale=0.72, color=GREY_DIM, origin=(-0.5, 0),
                        position=(x0, -H / 2 + 0.026, -0.1))
            self.cards.append(dict(bg=bg, band=band, tag=tag, idx=idx,
                                   name=name, info=info))

        self._build_level()
        self._build_controls()
        self._txt("A / D  CHOOSE          W / S  SESSION  ·  DIFFICULTY"
                  "          ENTER  CONTINUE          ESC  QUIT",
                  size=0.72, col=GREY, pos=(-0.80, -0.482))
        self._refresh()

    # -- difficulty --------------------------------------------------------
    def _build_level(self):
        """A panel of six boxes, one per level, each carrying its number and
        name. The chosen one is red, and the ones below it keep a red foot,
        so the row reads as a gauge filled up to the level."""
        top, H, W = self.LV_TOP, self.LV_H, self.PANEL_W
        Entity(parent=self.root, model="quad", color=PANEL,
               scale=(W, H), position=(0.0, top - H / 2, 0.3))
        self.lv_cursor = Entity(parent=self.root, model="quad", color=RED,
                                scale=(0.008, H),
                                position=(-W / 2 - 0.004, top - H / 2, 0.25))
        self.lv_head = self._txt(spaced("difficulty"), size=0.72, col=GREY_DIM,
                                 pos=(-0.79, top - 0.026))
        n = len(self.levels)
        inner = W - 0.060
        bw = (inner - (n - 1) * self.BOX_GAP) / n
        bh = self.BOX_H
        cy = top - 0.054 - bh / 2
        self.lv_boxes = []
        for k, lv in enumerate(self.levels):
            x = -inner / 2 + k * (bw + self.BOX_GAP)
            box = Entity(parent=self.root, model="quad", color=PANEL_HI,
                         scale=(bw, bh), position=(x + bw / 2, cy, 0.2))
            foot = Entity(parent=self.root, model="quad", color=RED,
                          scale=(bw, 0.006),
                          position=(x + bw / 2, cy - bh / 2 + 0.003, 0.15))
            num = Text(str(lv), parent=self.root, font=self.font, scale=2.2,
                       color=GREY, origin=(-0.5, 0),
                       position=(x + 0.020, cy + 0.020, -0.1))
            name = Text(teams.DIFFICULTY[lv].name.upper(), parent=self.root,
                        font=self.font, scale=0.86, color=GREY,
                        origin=(-0.5, 0), position=(x + 0.022, cy - 0.032, -0.1))
            self.lv_boxes.append(dict(box=box, foot=foot, num=num, name=name))

    # -- controls ------------------------------------------------------------
    def _build_controls(self):
        top = self.KEYS_TOP
        Entity(parent=self.root, model="quad", color=PANEL,
               scale=(self.PANEL_W, 0.170), position=(0.0, top - 0.085, 0.3))
        self._txt(spaced("controls"), size=0.72, col=GREY_DIM,
                  pos=(-0.79, top - 0.026))
        per_row = 5
        for k, (keys, what) in enumerate(CONTROLS):
            r, c = divmod(k, per_row)
            x = -0.79 + c * 0.322
            y = top - 0.066 - r * 0.036
            w = 0.012 * len(keys) + 0.020
            Entity(parent=self.root, model="quad", color=PANEL_HI,
                   scale=(w, 0.028), position=(x + w / 2, y, 0.2))
            self._txt(keys, size=0.62, col=WHITE, pos=(x + w / 2, y),
                      origin=(0, 0))
            self._txt(what, size=0.66, col=GREY, pos=(x + w + 0.014, y))

    def _txt(self, s, *, size=1.0, col=WHITE, pos=(0, 0), origin=(-0.5, 0), z=-0.1):
        return Text(s, parent=self.root, font=self.font, scale=size, color=col,
                    origin=origin, position=(pos[0], pos[1], z))

    def _refresh(self):
        for k, c in enumerate(self.cards):
            on = k == self.sel
            focus = on and self.row == 0
            c["bg"].color = PANEL_HI if on else PANEL
            c["band"].enabled = on
            c["tag"].color = RED if focus else PANEL_HI
            c["idx"].color = WHITE if on else GREY_DIM
            c["name"].color = WHITE if on else GREY
            c["info"].color = RED if on else GREY_DIM
        focus = self.row == 1
        live = RED if focus else RED_IDLE
        for b, lv in zip(self.lv_boxes, self.levels):
            on = lv == self.level
            below = lv < self.level
            b["box"].color = live if on else PANEL_HI
            b["foot"].enabled = below
            b["foot"].color = live
            b["num"].color = WHITE if on or below else GREY_DIM
            b["name"].color = WHITE if on else GREY if below else GREY_DIM
        self.lv_cursor.enabled = focus
        self.lv_head.color = WHITE if focus else GREY_DIM

    def on_key(self, key: str):
        n = len(MODES)
        if key in ("down arrow", "s", "up arrow", "w"):
            self.row = 1 - self.row
            self._refresh()
        elif key in ("right arrow", "d", "left arrow", "a",
                     "right arrow hold", "d hold", "left arrow hold", "a hold"):
            step = 1 if key.startswith(("right", "d")) else -1
            if self.row == 0:
                if key.endswith("hold"):
                    return
                self.sel = (self.sel + step) % n
            else:
                k = self.levels.index(self.level)
                self.level = self.levels[min(max(k + step, 0), len(self.levels) - 1)]
                self.on_level(self.level)
            self._refresh()
        elif key in ("enter", "space"):
            self.on_pick(MODES[self.sel][0])
        elif key == "escape":
            self.on_quit()

    def destroy(self):
        destroy_tree(self.root)
        self.root = None


# --- the menu ------------------------------------------------------------
class StartMenu:
    """Title card plus circuit list. Calls ``on_start(name)`` when chosen."""

    def __init__(self, names: list[str], on_start, on_quit, initial: str | None = None,
                 on_back=None, mode: str | None = None,
                 laps: int = config.TOTAL_LAPS, level: int = teams.DEFAULT_LEVEL,
                 quali: dict | None = None, on_watch=None,
                 grid="quali", on_grid=None):
        self.names = names
        #: Grand prix start: "quali" or a slot 1..20 (A / D), kept in the
        #: session through ``on_grid``.
        self.grid = grid
        self.on_grid = on_grid
        self.on_start = on_start
        #: G: watch this session instead of driving it.
        self.on_watch = on_watch
        self.on_quit = on_quit
        # ESC goes back to the main menu when there is one to go back to.
        self.on_back = on_back
        self.mode = mode
        self.laps = laps
        self.level = level
        #: (circuit, level) -> the player's qualifying lap this session.
        self.quali = quali or {}
        self.font = pick_font()
        self.sel = names.index(initial) if initial in names else 0
        self.top = 0                      # first visible row
        self._cache: dict[str, Track] = {}
        self._outline: Entity | None = None

        # One container for everything. Ursina rescales the x of every *direct*
        # child of camera.ui when the aspect ratio changes, so anything built as
        # several top-level pieces drifts apart in fullscreen -- the minimap hit
        # exactly this. A single root at x = 0 is immune.
        self.root = Entity(parent=camera.ui)
        self._build_ground()
        self._build_header()
        self._build_list()
        self._build_panel()
        self._build_footer()
        self._refresh()

    # -- construction ---------------------------------------------------
    def _txt(self, s, *, size=1.0, col=WHITE, pos=(0, 0), origin=(-0.5, 0),
             parent=None, z=-0.1):
        return Text(s, parent=parent or self.root, font=self.font,
                    scale=size, color=col, origin=origin,
                    position=(pos[0], pos[1], z))

    def _build_ground(self):
        Entity(parent=self.root, model="quad", color=INK,
               scale=(4.0, 1.4), position=(0, 0, 0.5))
        # A low-contrast slab behind the list. Big areas stay rectangular --
        # in real broadcast graphics it is the accents that are angled, not the
        # backgrounds, and skewing everything just looks unstable.
        # Top edge sits just *below* the header rule at y = 0.255, so the rule
        # separates the title from the list instead of cutting across the slab.
        # (top 0.240, bottom -0.410 -> height 0.650, centre -0.085)
        Entity(parent=self.root, model="quad", color=PANEL,
               scale=(0.79, 0.650), position=(-0.435, -0.085, 0.42))

    def _build_header(self):
        # Red flash + wordmark
        Entity(parent=self.root, model=skew_quad(0.045, 0.115),
               color=RED, position=(-0.80, 0.395, 0.2))
        self._txt("FORMULA-AI", size=3.4, pos=(-0.755, 0.395))
        self._txt(spaced("racing"), size=0.85, col=GREY,
                  pos=(-0.752, 0.315))

        # The rule belongs to the list column; running it the full width made
        # it slice through the preview panel.
        Entity(parent=self.root, model="quad", color=PANEL_HI,
               scale=(0.79, 0.0035), position=(-0.435, 0.255, 0.2))
        Entity(parent=self.root, model="quad", color=RED,
               scale=(0.30, 0.006), position=(-0.74, 0.255, 0.15))
        title = dict((m[0], m[1]) for m in MODES).get(self.mode)
        lvl = teams.DIFFICULTY.get(self.level)
        self._txt(spaced("circuit select")
                  + ("" if title is None else "   ·   " + spaced(title))
                  + ("" if lvl is None else "   ·   " + spaced(lvl.name)),
                  size=0.8, col=GREY, pos=(-0.80, 0.212))
        self._count = self._txt("", size=0.8, col=GREY_DIM,
                                pos=(-0.055, 0.212), origin=(0.5, 0))

    def _build_list(self):
        self.rows = []
        for i in range(ROWS):
            y = 0.145 - i * 0.0625
            row = Entity(parent=self.root, position=(0, y, 0.1))
            fill = Entity(parent=row, model=skew_quad(0.70, 0.052),
                          color=RED, position=(-0.44, 0, 0.06), enabled=False)
            idx = Text("", parent=row, font=self.font, scale=0.82, color=GREY_DIM,
                       origin=(-0.5, 0), position=(-0.775, 0, -0.1))
            name = Text("", parent=row, font=self.font, scale=1.05, color=WHITE,
                        origin=(-0.5, 0), position=(-0.715, 0, -0.1))
            code = Text("", parent=row, font=self.font, scale=0.78, color=GREY_DIM,
                        origin=(0.5, 0), position=(-0.115, 0, -0.1))
            rule = Entity(parent=row, model="quad", color=PANEL_HI,
                          scale=(0.68, 0.0015), position=(-0.445, -0.031, 0.05))
            self.rows.append((row, fill, idx, name, code, rule))

        # 23 circuits into ROWS slots, so say where in the list you are. A
        # scrollbar shows position *and* how much is left; the up/down carets
        # it replaces only ever said "there is more".
        bar_h = ROWS * 0.0625
        self._bar_top = 0.145 + 0.031
        Entity(parent=self.root, model="quad", color=PANEL_HI,
               scale=(0.004, bar_h), position=(-0.068, self._bar_top - bar_h / 2, 0.05))
        self.thumb = Entity(parent=self.root, model="quad", color=RED,
                            scale=(0.004, bar_h * ROWS / max(len(self.names), 1)),
                            position=(-0.068, 0, 0.04))
        self._bar_h = bar_h

    def _build_panel(self):
        px = 0.47
        self.panel = Entity(parent=self.root, position=(px, -0.02, 0.05))
        p = self.panel
        Entity(parent=p, model="quad", color=PANEL, scale=(0.64, 0.72),
               position=(0, 0, 0.06))
        Entity(parent=p, model=skew_quad(0.22, 0.010), color=RED,
               position=(-0.208, 0.352, 0.04))
        Entity(parent=p, model="quad", color=PANEL_HI, scale=(0.64, 0.002),
               position=(0, 0.352, 0.05))

        self.p_code = Text("", parent=p, font=self.font, scale=0.8, color=RED,
                           origin=(-0.5, 0), position=(-0.30, 0.305, -0.1))
        self.p_name = Text("", parent=p, font=self.font, scale=1.35, color=WHITE,
                           origin=(-0.5, 0), position=(-0.302, 0.255, -0.1))
        self.p_full = Text("", parent=p, font=self.font, scale=0.72, color=GREY,
                           origin=(-0.5, 0), position=(-0.300, 0.212, -0.1))
        # What this session holds on this circuit: the ghost lap to beat in
        # qualifying (or that there is none yet), the distance in a race.
        self.p_mode = Text("", parent=p, font=self.font, scale=0.72, color=GREY,
                           origin=(0.5, 0), position=(0.300, 0.305, -0.1))

        # Grand prix: where the player starts (A / D).
        self.p_grid = Text("", parent=p, font=self.font, scale=0.95, color=WHITE,
                           origin=(-0.5, 0), position=(-0.300, 0.165, -0.1))

        # Where the track outline gets drawn each time the selection moves.
        self.map_anchor = Entity(parent=p, position=(0, -0.055, -0.05))

        labels = (spaced("length"), spaced("turns"), spaced("longest straight"))
        self.stat_val = []
        for i, lab in enumerate(labels):
            x = -0.295 + i * 0.205
            Text(lab, parent=p, font=self.font, scale=0.6, color=GREY_DIM,
                 origin=(-0.5, 0), position=(x, -0.272, -0.1))
            self.stat_val.append(
                Text("", parent=p, font=self.font, scale=1.1, color=WHITE,
                     origin=(-0.5, 0), position=(x - 0.004, -0.315, -0.1)))

    def _build_footer(self):
        Entity(parent=self.root, model="quad", color=PANEL_HI,
               scale=(1.78, 0.0035), position=(0, -0.452, 0.2))
        watch = ("WATCH POLE LAP" if self.mode == QUALI else "WATCH AI RACE")
        self._txt("W / S   SELECT          ENTER   START"
                  + ("          A / D   START POSITION"
                     if self.mode == GRAND_PRIX else "")
                  + (f"          G   {watch}" if self.on_watch else "")
                  + "          ESC   " + ("BACK" if self.on_back else "QUIT"),
                  size=0.72, col=GREY, pos=(-0.80, -0.482))

    # -- state ----------------------------------------------------------
    def _track(self, name: str) -> Track | None:
        if name not in self._cache:
            try:
                self._cache[name] = load_track(name)
            except Exception as exc:            # a malformed folder on disk
                print(f"menu: cannot load {name}: {exc}")
                return None
        return self._cache[name]

    def _refresh(self):
        n = len(self.names)
        # Keep the cursor near the middle of the window while there is list
        # left on both sides.
        self.top = max(0, min(self.sel - ROWS // 2, n - ROWS))
        self._count.text = f"{self.sel + 1:02d} / {n:02d}"

        # Thumb spans the visible fraction and slides over the scrolled range.
        frac = min(ROWS / n, 1.0)
        self.thumb.scale_y = self._bar_h * frac
        travel = self._bar_h * (1.0 - frac)
        pos = self.top / max(n - ROWS, 1) if n > ROWS else 0.0
        self.thumb.y = self._bar_top - self.thumb.scale_y / 2 - travel * pos

        for i, (row, fill, idx, name, code, rule) in enumerate(self.rows):
            j = self.top + i
            if j >= n:
                row.enabled = False
                continue
            row.enabled = True
            key = self.names[j]
            label, _full, cc = caption(key)
            chosen = j == self.sel
            fill.enabled = chosen
            idx.text = f"{j + 1:02d}"
            idx.color = INK if chosen else GREY_DIM
            name.text = label
            name.color = WHITE if chosen else GREY
            code.text = cc
            code.color = INK if chosen else GREY_DIM
            rule.enabled = not chosen
            # Nudge the selected row along its own slant, so the highlight
            # reads as a moving band rather than a jumping box.
            for e in (idx, name, code):
                e.x = e.x + (0.012 if chosen else 0.0) - getattr(e, "_nudge", 0.0)
            for e in (idx, name, code):
                e._nudge = 0.012 if chosen else 0.0

        self._show(self.names[self.sel])

    def _show(self, key: str):
        label, full, cc = caption(key)
        self.p_code.text = cc
        self.p_name.text = label
        # Suppress the subtitle when it only restates the label.
        self.p_full.text = "" if full.upper() == label else full
        from . import grandprix
        solved = grandprix.ready(key)
        if self.mode == QUALI:
            from .replay import lap_path, lap_time as ghost_lap_time
            t = ghost_lap_time(key) if not solved else None
            if solved:
                path = lap_path(key, self.level)
                if path.is_file():
                    from .replay import load
                    rec = load(key, None, self.level)
                    t = rec.lap_time if rec is not None else None
                self.p_mode.text = (f"POLE  ·  LEVEL {self.level}   {lap_time(t)}"
                                    if t is not None else
                                    f"20 DRIVERS  ·  LEVEL {self.level}")
                self.p_mode.color = PURPLE if t is not None else GREY
            else:
                self.p_mode.text = (f"GHOST LAP   {lap_time(t)}" if t is not None
                                    else "NO GHOST LAP  ·  SOLO RUNS")
                self.p_mode.color = PURPLE if t is not None else GREY_DIM
        self.p_grid.text = ""
        if self.mode == GRAND_PRIX:
            if solved:
                self.p_grid.text = "START   " + self._grid_text(key)
                self.p_mode.text = f"{self.laps} LAPS  ·  20 CARS"
                self.p_mode.color = GREY
            else:
                self.p_mode.text = f"{self.laps} LAPS  ·  VS 1 AI  (FIELD NOT SOLVED)"
                self.p_mode.color = GREY_DIM
        elif self.mode != QUALI:
            self.p_mode.text = ""

        track = self._track(key)
        if self._outline is not None:
            destroy_tree(self._outline)
            self._outline = None
        if track is None:
            for v in self.stat_val:
                v.text = "--"
            return

        turns, straight = circuit_stats(track)
        self.stat_val[0].text = f"{track.length / 1000:.3f} km"
        self.stat_val[1].text = f"{turns}"
        self.stat_val[2].text = f"{straight:.0f} m"

        lo, hi = track.bounds()
        span = float(max(hi[0] - lo[0], hi[1] - lo[1])) or 1.0
        k = 0.35 / span
        cx, cz = (hi[0] + lo[0]) / 2, (hi[1] + lo[1]) / 2

        def to_uv(p):
            return Vec3((float(p[0]) - cx) * k, (float(p[1]) - cz) * k, 0)

        pts = [to_uv(p) for p in track.center]
        pts.append(pts[0])
        self._outline = Entity(parent=self.map_anchor)
        Entity(parent=self._outline, color=WHITE,
               model=Mesh(vertices=pts, mode="line", thickness=4))
        # Start/finish, drawn across the track rather than as a dot on it.
        # Scaled off the map span, not the track width -- a 13 m line on a
        # 5.8 km circuit shrinks to about one pixel.
        s, t = track.center[0], track.normal[0]
        half = span * 0.035
        # In front of the outline (smaller z is nearer), not level with it:
        # at the same depth the two lines were drawn in whatever order the
        # depth test happened to settle, so the start line went over the
        # curve on some circuits and under it on others.
        Entity(parent=self._outline, color=RED, z=-0.02,
               model=Mesh(vertices=[to_uv(s - t * half), to_uv(s + t * half)],
                          mode="line", thickness=9))

    def _grid_text(self, key: str) -> str:
        """The grand prix start as the panel says it: < P7 > for a chosen
        slot, or the qualifying lap's time when that is what is selected."""
        q = self.quali.get((key, self.level))
        if self.grid == "quali":
            if q is not None:
                return f"[ LAST QUALIFYING {lap_time(q)} ]"
            return f"[ QUALIFYING: NONE  ·  P{config.GP_PLAYER_GRID} ]"
        return f"[ P{int(self.grid)} ]" + ("   POLE" if int(self.grid) == 1 else "")

    def _step_grid(self, d: int):
        """A / D: qualifying, then pole to last, round and round."""
        opts = ["quali"] + list(range(1, 21))
        i = opts.index(self.grid) if self.grid in opts else 0
        self.grid = opts[(i + d) % len(opts)]
        if self.on_grid is not None:
            self.on_grid(self.grid)
        self._show(self.names[self.sel])

    # -- input ----------------------------------------------------------
    def on_key(self, key: str):
        n = len(self.names)
        if key in ("down arrow", "s", "down arrow hold", "s hold"):
            self.sel = (self.sel + 1) % n
            self._refresh()
        elif key in ("up arrow", "w", "up arrow hold", "w hold"):
            self.sel = (self.sel - 1) % n
            self._refresh()
        elif (self.mode == GRAND_PRIX
              and key in ("d", "right arrow", "a", "left arrow",
                          "d hold", "right arrow hold", "a hold",
                          "left arrow hold")):
            self._step_grid(1 if key.startswith(("d", "right")) else -1)
        elif key in ("enter", "space"):
            self.on_start(self.names[self.sel])
        elif key == "g" and self.on_watch is not None:
            self.on_watch(self.names[self.sel])
        elif key == "escape":
            (self.on_back or self.on_quit)()

    def destroy(self):
        if self._outline is not None:
            destroy_tree(self._outline)
            self._outline = None
        destroy_tree(self.root)
        self.root = None
