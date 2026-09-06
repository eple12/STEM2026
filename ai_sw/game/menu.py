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
from ursina import Entity, Mesh, Text, Vec3, camera, destroy

from . import config
from . import palette as pal
from .trackdata import Track, load_track

from .ui import (CIRCUITS, GREY, GREY_DIM, INK, PANEL, PANEL_HI, RED, WHITE,
                 caption, pick_font, skew_quad, spaced)

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


# --- the menu ------------------------------------------------------------
class StartMenu:
    """Title card plus circuit list. Calls ``on_start(name)`` when chosen."""

    def __init__(self, names: list[str], on_start, on_quit, initial: str | None = None):
        self.names = names
        self.on_start = on_start
        self.on_quit = on_quit
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
        self._txt(spaced("circuit select"), size=0.8, col=GREY,
                  pos=(-0.80, 0.212))
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

        # Where the track outline gets drawn each time the selection moves.
        self.map_anchor = Entity(parent=p, position=(0, -0.02, -0.05))

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
        self._txt("W / S   or   UP / DOWN      SELECT          ENTER   START"
                  "          ESC   QUIT",
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

        track = self._track(key)
        if self._outline is not None:
            destroy(self._outline)
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
        k = 0.40 / span
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
        Entity(parent=self._outline, color=RED,
               model=Mesh(vertices=[to_uv(s - t * half), to_uv(s + t * half)],
                          mode="line", thickness=9))

    # -- input ----------------------------------------------------------
    def on_key(self, key: str):
        n = len(self.names)
        if key in ("down arrow", "s", "down arrow hold", "s hold"):
            self.sel = (self.sel + 1) % n
            self._refresh()
        elif key in ("up arrow", "w", "up arrow hold", "w hold"):
            self.sel = (self.sel - 1) % n
            self._refresh()
        elif key in ("enter", "space"):
            self.on_start(self.names[self.sel])
        elif key == "escape":
            self.on_quit()

    def destroy(self):
        if self._outline is not None:
            destroy(self._outline)
            self._outline = None
        destroy(self.root)
        self.root = None
