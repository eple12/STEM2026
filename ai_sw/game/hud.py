"""On-screen display: the F1 TV broadcast package, laid out as it is on air.

Top-left, the **timing tower**: the wordmark and the lap counter (or the
session clock in qualifying) in its header, a status strip that turns yellow
under a yellow flag and chequered at the finish, and a row per car --
position, team mark, three-letter name, the interval to the car ahead (or
the gap to the leader: TAB), and beside the tower the chips a broadcast
hangs off a row: a time penalty, DRS open, a car out of the race for the
moment. Rows slide to their new places when the order changes, so a pass
can be followed on the tower as well as on the road; arrows show places
gained or lost since the start. The player's row is lit.

Bottom-right, the **driver chyron**: who is on camera, the running lap time,
the time to beat, and three sector bars in purple, green and yellow.
Bottom-centre, the **speed graphic** with the revs, the gear and the pedals.
Bottom-left, a track map with every car on it. On the grid, the five-column
**start lights** drop in; a **race control** box carries the stewards' word;
the result and pause cards are cut from the same cloth.

Colours follow the timing conventions exactly: purple is the session best,
green a personal best, yellow slower than your own best.

All of it is drawn through ``hudkit``: one mesh for every panel and bar, one
per typeface for every label -- split into what never changes and what
changes every frame, so a new speed or lap time re-uploads only the small
live buffers. Running numbers (speed, lap clock, gaps) are ``DigitField``s:
their cells are fixed when the HUD is built and a new value only swaps which
pre-rasterised glyph each cell shows.

Layout note: on an aspect-ratio change Ursina rescales ``e.x`` for every
*direct child* of ``camera.ui``. Everything here hangs off one ``self.root``
at x = 0, and the HUD is rebuilt when the ratio changes (``stale``).
"""
from __future__ import annotations

import math

import numpy as np

from ursina import Entity, camera, window

from . import config
from . import palette as pal
from .hudkit import ShapeBatch, ShapeLayer, TextLayer, _rgba
from .trackdata import Track
from .ui import (AMBER, GREEN, PURPLE, RED, TEAM_AI, TEAM_YOU, WHITE, YELLOW,
                 caption, delta_time, destroy_tree, gear_of, lap_time)

# --- broadcast palette -------------------------------------------------------
BG = pal.rgb(14, 14, 20, 232)
BG_ROW = pal.rgb(18, 18, 26, 218)
BG_ROW2 = pal.rgb(24, 24, 33, 218)
BG_HI = pal.rgb(44, 44, 58, 236)
BG_SOFT = pal.rgb(10, 10, 16, 120)
HILITE = pal.rgb(236, 236, 242, 240)
WATCH_HI = pal.rgb(150, 18, 14, 236)
RULE = pal.rgb(255, 255, 255, 22)
CLEAR = pal.rgb(0, 0, 0, 0)
INK = pal.rgb(12, 12, 16)
GREY = pal.rgb(168, 170, 180)
DIM = pal.rgb(112, 114, 126)
LAMP_OFF = pal.rgb(44, 12, 12)
LAMP_ON = pal.rgb(255, 26, 14)
LAMP_GLOW = pal.rgb(255, 30, 10, 70)
DRS_GREEN = pal.rgb(40, 200, 90)
STRIP_IDLE = pal.rgb(70, 72, 84)

#: Sector bar colour by status. "live" is the sector being driven.
SECTOR_COL = {"purple": PURPLE, "green": GREEN, "yellow": YELLOW,
              "live": pal.rgb(110, 110, 124), "off": pal.rgb(48, 48, 60)}

#: Race-control flag square and its glyph, by what the message is saying.
FLAG_COL = {"warn": YELLOW, "red": RED, "pen": AMBER, "flag": WHITE,
            "clear": pal.rgb(60, 180, 120), "info": GREY, "yellow": YELLOW,
            "green": GREEN}
FLAG_ICON = {"warn": "!", "red": "!", "pen": "+", "flag": "!", "clear": "=",
             "info": "i", "yellow": "!", "green": "="}

#: Status strip under the tower header.
STRIP_COL = {"green": GREEN, "yellow": YELLOW, "idle": STRIP_IDLE}

#: Rev lights: green, then red, then blue at the shift point.
N_LEDS = 15
LED_ON = ([pal.rgb(40, 220, 90)] * 5 + [pal.rgb(240, 40, 30)] * 5
          + [pal.rgb(70, 120, 255)] * 5)
LED_OFF = pal.rgb(40, 40, 52)

KEYS = (("W", "THROTTLE"), ("S", "BRAKE"), ("A/D", "STEER"),
        ("SPACE", "HANDBRAKE"), ("R", "RESET"), ("C", "CAMERA"),
        ("X", "LOOK BACK"), ("G", "WATCH NEXT CAR"), ("T", "AIDS"),
        ("TAB", "GAPS"), ("M", "MUTE"), ("H", "HUD"), ("ESC", "PAUSE"))


def pen_tag(seconds: float) -> str:
    """A penalty as a tower tag: seconds, with a decimal while it is small
    enough for one to matter."""
    return f"+{seconds:.0f}S" if abs(seconds - round(seconds)) < 0.05 else (
        f"+{seconds:.1f}S" if seconds < 20 else f"+{seconds:.0f}S")


def tenths(t: float | None) -> str:
    """A running lap time the way the chyron shows it: to the tenth."""
    if t is None:
        return "-:--.-"
    m, s = divmod(max(0.0, t), 60.0)
    return f"{int(m):d}:{int(s):02d}.{int((s % 1.0) * 10):d}"


def _rgb01(col) -> tuple:
    """A (r, g, b) 0..1 team colour as an opaque RGBA tuple."""
    return (float(col[0]), float(col[1]), float(col[2]), 1.0)


class _Pair:
    """Two shapes shown and hidden together."""

    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a, self.b = a, b

    def show(self, on: bool):
        self.a.show(on)
        self.b.show(on)


class HUD:
    TOWER_W = 0.205
    ROW_H = 0.0245
    HEAD_H = 0.066
    CHY_W = 0.400
    LOW_H = 0.098            # height of the chyron
    SPD_W, SPD_H = 0.400, 0.126
    MAP_W, MAP_H = 0.230, 0.180
    #: Seconds a row takes to slide most of the way to its new place.
    SLIDE_RATE = 12.0
    #: Tower chips: padding either side of the text, and the gap to the
    #: tower and between two chips.
    CHIP_PAD = 0.0055
    CHIP_GAP = 0.003
    #: Race-control box: narrowest it gets, and the widest it may grow to
    #: (also held clear of the tower and its chips).
    FLAG_W_MIN, FLAG_W_MAX = 0.40, 1.10
    #: Its slide: how far up it starts and goes (off the top of the screen),
    #: and the seconds the slide takes.
    FLAG_HIDE_DY, FLAG_SLIDE_T = 0.10, 0.22

    def __init__(self, track: Track, total_laps: int, mode: str = "gp",
                 rows: int = 20):
        self.track = track
        self.total_laps = total_laps
        self.quali = mode == "quali"
        self.n_rows = rows
        self._aspect = float(window.aspect_ratio)
        self.edge = self._aspect / 2.0
        self.margin = max(config.HUD_MARGIN, 0.026)
        self.circuit, self.full_name, self.country = caption(track.name)
        self.root = Entity(parent=camera.ui)

        # What never changes after the build, and what changes every frame:
        # a new value re-uploads only the live buffers.
        self.S = ShapeLayer("hud_static")
        self.D = ShapeLayer("hud_live")
        self.T = TextLayer("hud_text")
        self.L = TextLayer("hud_text_live")
        self._rpm = 0.0
        self._lit = -1
        self._last_seen = None
        self._flash_until = -1.0
        self._flash_col = WHITE

        self._build_tower()
        self._build_chyron()
        self._build_speed()
        if config.SHOW_KEY_HINTS:
            self._build_keys()
        self.S.build(self.root, sort=0)
        self.D.build(self.root, sort=1)
        self.T.build(self.root, sort=20)
        self.L.build(self.root, sort=21)

        self._build_minimap(track)
        self._build_lights()
        self._build_flag()
        self._build_yellow_frame()
        self._build_spectator()
        self._build_finish()
        self._build_pause()

    # -- tower (top-left) ------------------------------------------------------
    def _build_tower(self):
        S, D, T, L = self.S, self.D, self.T, self.L
        W, RH = self.TOWER_W, self.ROW_H
        x0 = -self.edge + self.margin
        top = 0.5 - self.margin
        hh = self.HEAD_H
        self._tx0 = x0
        S.rect(x0, top - hh, W, hh, BG)
        # The mark where the broadcast keeps its logo (logo.py).
        self._logo(self.root, x0 + 0.050, top - 0.024, 0.030)
        # Qualifying is a fixed number of timed laps too, so both count laps.
        T.label("QUALIFYING" if self.quali else "LAP", x0 + 0.012,
                top - 0.057, 0.0084 if self.quali else 0.0094,
                weight="semibold", col=GREY, track=0.0012)
        self.t_head = L.label("1/3", x0 + W - 0.012, top - 0.059, 0.0170,
                              weight="bold", align="right", cap=8, mono=True)
        # Status strip: green racing, yellow under a yellow flag, a
        # chequer at the finish.
        sy = top - hh - 0.005
        self.strip = D.rect(x0, sy, W, 0.005, STRIP_IDLE)
        self.strip_chq = []
        n_chq = 16
        cw = W / n_chq
        for k in range(n_chq):
            for j in (0, 1):
                col = WHITE if (k + j) % 2 == 0 else INK
                sq = D.rect(x0 + k * cw, sy + j * 0.0025, cw, 0.0025, col)
                sq.show(False)
                self.strip_chq.append(sq)
        # Column caption: what the right-hand column is showing.
        my = sy - 0.017
        S.rect(x0, my, W, 0.017, BG)
        self.t_mode = L.label("INTERVAL", x0 + W - 0.010, my + 0.0055, 0.0068,
                              weight="semibold", align="right", col=DIM, cap=8,
                              track=0.0010)
        self._rows_top = my
        # Slots: the position numbers stay put; the cars move between them.
        self.slots = []
        for k in range(self.n_rows):
            yb = my - (k + 1) * RH
            S.rect(x0, yb, W, RH, BG_ROW if k % 2 == 0 else BG_ROW2)
            S.rect(x0, yb, 0.026, RH, RED if k == 0 else BG)
            lab = T.label(str(k + 1), x0 + 0.013, yb + RH / 2 - 0.0053, 0.0106,
                          weight="bold", align="center", cap=2)
            self.slots.append(lab)
        S.rect(x0, my - self.n_rows * RH - 0.0015, W, 0.0015, RULE)
        # Car rows, all built in the first slot and moved down to theirs.
        y0 = my - RH
        cy = y0 + RH / 2
        # Chips hang off the row's right edge (see _layout_chips).
        cx0 = x0 + W + self.CHIP_GAP
        self._chip_x0 = cx0
        self._chip_h = RH - 0.007
        self.rows = []
        for r in range(self.n_rows):
            row = dict(
                hl=D.rect(x0 + 0.026, y0, W - 0.026, RH, HILITE),
                bar=D.rect(x0 + 0.030, y0 + RH * 0.2, 0.0034, RH * 0.6, TEAM_YOU),
                fast=D.disc(x0 + 0.0865, cy, 0.0030, PURPLE, seg=10),
                up=D.tri([(x0 + 0.0925, cy - 0.0030), (x0 + 0.0985, cy - 0.0030),
                          (x0 + 0.0955, cy + 0.0034)], GREEN),
                dn=D.tri([(x0 + 0.0925, cy + 0.0030), (x0 + 0.0985, cy + 0.0030),
                          (x0 + 0.0955, cy - 0.0034)], RED),
                pen_bg=D.rect(cx0, y0 + 0.0035, 0.034, self._chip_h, AMBER),
                st_bg=D.rect(cx0, y0 + 0.0035, 0.030, self._chip_h, DRS_GREEN),
                tla=L.label("", x0 + 0.039, cy - 0.0053, 0.0106, weight="bold",
                            cap=3, track=0.0006),
                chg=L.digits("", x0 + 0.1005, cy - 0.0035, 0.0072,
                             weight="semibold", align="left", cap=2),
                gap=L.digits("", x0 + W - 0.008, cy - 0.0048, 0.0096,
                             weight="regular", cap=9),
                word=L.label("", x0 + W - 0.008, cy - 0.0043, 0.0080,
                             weight="semibold", align="right", cap=8, col=GREY,
                             track=0.0008),
                pen=L.label("", cx0 + self.CHIP_PAD, cy - 0.0040, 0.0080,
                            weight="bold", col=INK, cap=6),
                st=L.label("", cx0 + self.CHIP_PAD, cy - 0.0040, 0.0078,
                           weight="bold", col=INK, cap=4),
                key=None, dy=0.0, target=0.0, shown=False, state={},
                chip_y=y0 + 0.0035)
            for k in ("hl", "bar", "fast", "up", "dn", "pen_bg", "st_bg"):
                row[k].show(False)
            row["base"] = {k: (row[k].x, row[k].y)
                           for k in ("tla", "chg", "gap", "word", "pen", "st")}
            self.rows.append(row)
        self._row_of: dict = {}

    def _logo(self, parent, cx, cy, h, mono=None, sort=15, z=0.0):
        """The drawn F-AI mark, centred on (cx, cy), *h* tall."""
        from panda3d.core import TransparencyAttrib

        from . import logo
        tex = logo.texture(96, mono=mono)
        w = h * tex.get_x_size() / tex.get_y_size()
        e = Entity(parent=parent, model="quad", scale=(w, h),
                   position=(cx, cy, z))
        e.set_texture(tex, 10)
        e.set_transparency(TransparencyAttrib.M_alpha)
        e.set_depth_test(False)
        e.set_depth_write(False)
        e.set_bin("fixed", sort)
        return e

    # -- driver chyron (bottom-right) -------------------------------------------
    def _build_chyron(self):
        S, D, L = self.S, self.D, self.L
        W, H = self.CHY_W, self.LOW_H
        # In the bottom-right corner, out of the way of the telemetry, which
        # sits in the middle where the eye already is.
        x0 = self.edge - self.margin - W
        y0 = -0.5 + self.margin
        name_h, time_h = 0.032, 0.050
        sec_h = H - name_h - time_h
        yn = y0 + sec_h + time_h
        self._chy = (x0, y0, W, H)
        # Name strip: a white position block, the team mark, the name.
        S.rect(x0, yn, W, name_h, BG)
        S.rect(x0, yn, 0.034, name_h, WHITE)
        self.c_bar = D.rect(x0 + 0.034, yn, 0.005, name_h, TEAM_YOU)
        self.c_pos = L.label("1", x0 + 0.017, yn + name_h / 2 - 0.0072,
                             0.0144, weight="bold", align="center",
                             col=INK, cap=2)
        self.c_name = L.label("PLAYER", x0 + 0.049, yn + name_h / 2 - 0.0068,
                              0.0136, weight="bold", cap=14, track=0.0008)
        self.c_lap = L.label("", x0 + W - 0.012, yn + name_h / 2 - 0.0054,
                             0.0105, weight="semibold", align="right", col=GREY,
                             cap=14, track=0.0010)
        # Time row: the running time, big; the time to beat at the right.
        yt = y0 + sec_h
        S.rect(x0, yt, W, time_h, BG_ROW)
        self.c_time = L.digits("", x0 + 0.016 + 8 * 0.0, yt + 0.013, 0.0270,
                               weight="bold", align="left", cap=9)
        self.c_time_word = L.label("", x0 + 0.016, yt + 0.013, 0.0230,
                                   weight="bold", cap=8)
        self.c_delta_bg = D.rect(x0 + 0.172, yt + 0.013, 0.078, 0.024, GREEN)
        self.c_delta = L.digits("", x0 + 0.244, yt + 0.0195, 0.0118,
                                weight="bold", col=WHITE, cap=8)
        self.c_target = L.digits("", x0 + W - 0.014, yt + 0.027, 0.0150,
                                 weight="semibold", cap=9)
        self.c_target_who = L.label("", x0 + W - 0.014, yt + 0.010, 0.0090,
                                    weight="semibold", align="right", col=GREY,
                                    cap=18, track=0.0010)
        # Sector bars.
        S.rect(x0, y0, W, sec_h, BG_ROW)
        gap = 0.005
        bw = (W - 0.032 - 2 * gap) / 3
        self.c_sectors = []
        for k in range(3):
            self.c_sectors.append(D.rect(x0 + 0.016 + k * (bw + gap),
                                         y0 + sec_h / 2 - 0.003, bw, 0.006,
                                         SECTOR_COL["off"]))

    # -- telemetry (bottom-centre) ---------------------------------------------
    def _build_speed(self):
        """Speed, gear, revs and pedals, big, in the middle of the bottom
        edge -- right under the car, where a glance from the road finds it."""
        S, D, T, L = self.S, self.D, self.T, self.L
        W, H = self.SPD_W, self.SPD_H
        x0, x1 = -W / 2, W / 2
        y0 = -0.5 + self.margin
        head = 0.028
        S.rect(x0, y0 + H - head, W, head, BG)
        self.s_bar = D.rect(x0, y0 + H - head, 0.006, head, TEAM_YOU)
        self.s_name = L.label("PLAYER", x0 + 0.018, y0 + H - head / 2 - 0.0058,
                              0.0116, weight="bold", cap=14, track=0.0012)
        self.s_aid = L.label("", x1 - 0.014, y0 + H - head / 2 - 0.0055,
                             0.0108, weight="bold", align="right", col=AMBER,
                             cap=10, track=0.0010)
        self.s_slip = L.label("", x1 - 0.100, y0 + H - head / 2 - 0.0055,
                              0.0108, weight="bold", align="right", col=AMBER,
                              cap=10, track=0.0010)
        body = H - head
        S.rect(x0, y0, W, body, BG_ROW)
        # Rev lights across the top of the body.
        lw = (W - 0.030) / N_LEDS
        self.leds = [D.rect(x0 + 0.015 + k * lw + 0.0010, y0 + body - 0.014,
                            lw - 0.0020, 0.0075, LED_OFF)
                     for k in range(N_LEDS)]
        # Throttle and brake, filling upwards, with their labels under them.
        self.bar_h = body - 0.040
        self._bar_y = y0 + 0.020
        for name, col, x, lab in (("p_thr", GREEN, x0 + 0.016, "THR"),
                                  ("p_brk", RED, x0 + 0.044, "BRK")):
            S.rect(x, self._bar_y, 0.013, self.bar_h, pal.rgb(54, 56, 70, 220))
            setattr(self, name, D.rect(x, self._bar_y, 0.013, self.bar_h, col))
            T.label(lab, x + 0.0065, y0 + 0.008, 0.0066, weight="semibold",
                    align="center", col=DIM, track=0.0004)
        # Speed, in fixed digit cells so it does not breathe; the unit beside.
        self.s_speed = L.digits("0", x0 + 0.232, y0 + 0.022, 0.054,
                                weight="bold", cap=3)
        T.label("KM/H", x0 + 0.240, y0 + 0.022, 0.0120, weight="semibold",
                col=GREY, track=0.0016)
        # Gear, in its own block.
        gw = 0.070
        gx = x1 - 0.014 - gw
        S.rect(gx, y0 + 0.010, gw, body - 0.030, BG_HI)
        self.s_gear = L.label("N", gx + gw / 2, y0 + 0.024, 0.040,
                              weight="bold", align="center", cap=1)
        T.label("GEAR", gx + gw / 2, y0 + 0.013, 0.0070, weight="semibold",
                align="center", col=DIM, track=0.0010)

    def _build_keys(self):
        line = "     ".join(f"{k} {v}" for k, v in KEYS)
        self.T.label(line, 0.0, 0.5 - self.margin - 0.012, 0.0090,
                     weight="semibold", align="center", col=GREY,
                     track=0.0008)

    # -- track map (bottom-left) -------------------------------------------------
    def _build_minimap(self, track: Track):
        lo, hi = track.bounds()
        W, H = self.MAP_W, self.MAP_H
        span = float(max((hi[0] - lo[0]) / W, (hi[1] - lo[1]) / H)) or 1.0
        self._mm_scale = 0.84 / span
        self._mm_cx = (hi[0] + lo[0]) / 2
        self._mm_cz = (hi[1] + lo[1]) / 2
        x0 = -self.edge + self.margin
        y0 = -0.5 + self.margin
        self._mm_o = (x0 + W / 2, y0 + H / 2)
        m = Entity(parent=self.root)
        self.minimap = m
        S = ShapeLayer("map_shapes")
        S.round_rect(x0, y0, W, H, 0.010, BG_SOFT)
        step = max(1, len(track.center) // config.MINIMAP_POINTS)
        pts = [self._to_mm(p) for p in track.center[::step]]
        S.polyline(pts, 0.0085, pal.rgb(8, 8, 12, 210))
        S.polyline(pts, 0.0034, pal.rgb(235, 235, 240))
        # Start/finish, across the line.
        s, n = track.center[0], track.normal[0]
        a = self._to_mm(s - n * 30.0)
        b = self._to_mm(s + n * 30.0)
        S.polyline([a, b], 0.0040, RED, closed=False)
        S.build(m, sort=4)
        # The yellow-flag stretches, lit over the track line: one short
        # piece per ~20 m of lap, shown where a zone covers it.
        Y = ShapeLayer("map_yellow")
        k = max(1, int(round(20.0 / max(float(np.median(track.seg_len)), 0.5))))
        n = len(track.center)
        self._mm_yel = []
        spans = []
        for i in range(0, n, k):
            j = min(i + k, n) % n
            mid = float(track.arclen[i]) + 0.5 * k * float(np.median(track.seg_len))
            spans.append((self._to_mm(track.center[i]),
                          self._to_mm(track.center[j]), mid % float(track.length)))
        # A dark band under a wider yellow one: the cars' dots are drawn over
        # the track and used to hide a narrow highlight. All the dark first,
        # so one piece's band never cuts the next one's yellow.
        dark = [Y.polyline([a, b], 0.0150, pal.rgb(8, 8, 12, 235), closed=False)
                for a, b, _m in spans]
        for (a, b, mid), d in zip(spans, dark):
            seg = _Pair(d, Y.polyline([a, b], 0.0105, YELLOW, closed=False))
            seg.show(False)
            self._mm_yel.append((mid, seg))
        Y.build(m, sort=5)
        self.mm_Y = Y
        self._mm_zones = ()
        self._mm_L = float(track.length)
        # The field's dots: one layer, moved every frame; the player on top.
        F = ShapeLayer("map_dots")
        self.mm_others = []
        for _ in range(max(self.n_rows, 2)):
            ring = F.disc(0, 0, 0.0070, INK, seg=10)
            dot = F.disc(0, 0, 0.0050, TEAM_AI, seg=10)
            ring.show(False)
            dot.show(False)
            self.mm_others.append((ring, dot))
        self.mm_ring = F.disc(0, 0, 0.0110, INK, seg=16)
        self.mm_dot = F.disc(0, 0, 0.0080, TEAM_YOU, seg=16)
        F.build(m, sort=6)
        self.mm_S = F
        # Ring then dot, car by car: placed in one write (ShapeBatch).
        self.mm_batch = ShapeBatch([s for pair in self.mm_others for s in pair])
        self._mm_xy = np.full((2 * len(self.mm_others), 2), np.nan, np.float32)

    def _to_mm(self, xz):
        u = (float(xz[0]) - self._mm_cx) * self._mm_scale + self._mm_o[0]
        v = (float(xz[1]) - self._mm_cz) * self._mm_scale + self._mm_o[1]
        return (u, v)

    # -- start lights -----------------------------------------------------------
    def _build_lights(self):
        """Five columns, two lamps each, on black: the broadcast's own start
        graphic. They come on column by column and go out together."""
        self._gantry_y0 = 0.0
        g = Entity(parent=self.root, enabled=False)
        self.gantry = g
        S = ShapeLayer("lights")
        uw, uh, gap = 0.060, 0.124, 0.010
        total = 5 * uw + 4 * gap
        top = 0.5 - self.margin - 0.050
        self.lamps = []
        for k in range(5):
            x = -total / 2 + k * (uw + gap)
            S.round_rect(x, top - uh, uw, uh, 0.008, pal.rgb(8, 8, 10, 242))
            col = []
            for j in (0, 1):
                cy = top - uh / 2 + (0.028 if j == 0 else -0.028)
                glow = S.disc(x + uw / 2, cy, 0.030, LAMP_GLOW, seg=20)
                glow.show(False)
                lamp = S.disc(x + uw / 2, cy, 0.0205, LAMP_OFF, seg=24)
                col.append((glow, lamp))
            self.lamps.append(col)
        S.build(g, sort=8)
        self.g_S = S
        self._lit_n = -1

    # -- in a yellow sector: the screen's edge pulses --------------------------
    YF_T = 0.016            # frame thickness (screen height = 1)

    def _build_yellow_frame(self):
        f = Entity(parent=self.root, enabled=False)
        self.yframe = f
        S = ShapeLayer("yframe")
        e, h, t = self.edge, 0.5, self.YF_T
        self._yf = [S.rect(-e, h - t, 2 * e, t, YELLOW),
                    S.rect(-e, -h, 2 * e, t, YELLOW),
                    S.rect(-e, -h + t, t, 2 * h - 2 * t, YELLOW),
                    S.rect(e - t, -h + t, t, 2 * h - 2 * t, YELLOW)]
        S.build(f, sort=2)
        self.yf_S = S
        self._yf_t = 0.0
        self._yf_on = False

    def _animate_yellow_frame(self, on: bool, dt: float):
        if not on:
            if self._yf_on:
                self._yf_on = False
                self.yframe.enabled = False
            return
        if not self._yf_on:
            self._yf_on = True
            self.yframe.enabled = True
        self._yf_t += dt
        a = 0.42 + 0.30 * math.sin(self._yf_t * 2.0 * math.pi * 1.8)
        r, g, b, _a = _rgba(YELLOW)
        for sh in self._yf:
            sh.color = (r, g, b, a)
        self.yf_S.flush()

    # -- race control (top-centre) ----------------------------------------------
    def _build_flag(self):
        """The stewards' box, top centre. It is as wide as what it is saying
        (_fit_flag): the longest messages name a car and a reason, and a
        fixed box either clipped them or sat half-empty for "OFF TRACK"."""
        f = Entity(parent=self.root, enabled=False)
        self.flag = f
        S = ShapeLayer("flag")
        W, H = self.FLAG_W_MIN, 0.058
        self._flag_h = H
        self._flag_y0 = y0 = 0.5 - self.margin - H - 0.004
        # Wide enough to clear the tower and its chips either side.
        clear = self.edge - self.margin - self.TOWER_W - 0.095
        self._flag_wmax = max(self.FLAG_W_MIN, min(self.FLAG_W_MAX, 2 * clear))
        x0 = -W / 2
        self.flag_bg = S.rect(x0, y0, W, H, BG)
        self.flag_sq = S.rect(x0, y0, H, H, YELLOW)
        self.flag_rule = S.rect(x0 + H, y0 + H - 0.0016, W - H, 0.0016, RULE)
        S.build(f, sort=10)
        self.flag_S = S
        T = TextLayer("flag_text")
        self._flag_txt_size = 0.0136
        self.flag_head = T.label("RACE CONTROL", x0 + H + 0.014, y0 + H - 0.019,
                                 0.0090, weight="semibold", col=GREY,
                                 track=0.0016, cap=34)
        self.flag_txt = T.label("", x0 + H + 0.014, y0 + 0.012,
                                self._flag_txt_size, weight="bold", cap=96,
                                track=0.0006)
        self.flag_icon = T.label("!", x0 + H / 2, y0 + H / 2 - 0.011, 0.022,
                                 weight="bold", align="center", col=INK, cap=1)
        T.build(f, sort=30)
        self.flag_T = T
        self._flag_key = None
        self._flag_dy = self.FLAG_HIDE_DY

    def _set_flag(self, key):
        flag, style, head = key
        self._flag_key = key
        self.flag_head.set(head)
        self.flag_txt.set(flag)
        self._fit_flag()
        self.flag_sq.color = FLAG_COL.get(style, YELLOW)
        self.flag_icon.set(FLAG_ICON.get(style, "!"))
        self.flag_S.flush()
        self.flag_T.flush()

    def _animate_flag(self, want, dt: float):
        """The race-control box drops in from above the screen and goes
        back up when it is done; a new message sends the old one up first
        and comes down with the new one. A change to the heading alone (the
        queue's "N MORE" counting down) is made in place."""
        hide = self.FLAG_HIDE_DY
        shown = self._flag_dy < hide and self.flag.enabled
        cur = self._flag_key
        if want is not None and not shown:
            self._set_flag(want)                       # in from the top
            self.flag.enabled = True
            target = 0.0
        elif want is None:
            target = hide
        elif cur is not None and want[:2] == cur[:2]:
            if want[2] != cur[2]:
                self._set_flag(want)
            target = 0.0
        else:
            target = hide                              # out, then the new one
            if self._flag_dy >= hide - 1e-4:
                self._set_flag(want)
                target = 0.0
        step = hide / self.FLAG_SLIDE_T * dt
        d = target - self._flag_dy
        self._flag_dy += max(-step, min(step, d))
        # Eased: fast to start, settling at either end.
        f = self._flag_dy / hide
        self.flag.y = hide * (f * f * (3.0 - 2.0 * f))
        if self._flag_dy >= hide - 1e-4 and want is None and self.flag.enabled:
            self.flag.enabled = False
            self._flag_key = None

    def _fit_flag(self):
        """Size the race-control box to its text, centred; a line too long
        even for the widest box is set smaller rather than run out of it."""
        H, y0 = self._flag_h, self._flag_y0
        pad_l, pad_r = 0.014, 0.020
        room = self._flag_wmax - H - pad_l - pad_r
        lab = self.flag_txt
        lab.resize(self._flag_txt_size)
        if lab.width > room:
            lab.resize(self._flag_txt_size * room / lab.width)
        text_w = max(lab.width, self.flag_head.width)
        W = min(self._flag_wmax, max(self.FLAG_W_MIN, H + pad_l + text_w + pad_r))
        x0 = -W / 2
        self.flag_bg.set_rect(x0, y0, W, H)
        self.flag_sq.set_rect(x0, y0, H, H)
        self.flag_rule.set_rect(x0 + H, y0 + H - 0.0016, W - H, 0.0016)
        self.flag_head.move(x0 + H + pad_l, self.flag_head.y)
        lab.move(x0 + H + pad_l, lab.y)
        self.flag_icon.move(x0 + H / 2, self.flag_icon.y)

    # -- spectator tag ----------------------------------------------------------
    def _build_spectator(self):
        """The red tag over the chyron: whose onboard this is (G), and that
        the camera is looking back (X). As wide as what it says."""
        c = Entity(parent=self.root, enabled=False)
        self.spectator = c
        x0, y0, W, H = self._chy
        S = ShapeLayer("spect")
        self._spect_y = y0 + H + 0.006
        self.spect_bg = S.rect(x0, self._spect_y, 0.190, 0.024, RED)
        S.disc(x0 + 0.014, y0 + H + 0.018, 0.0036, WHITE, seg=12)
        S.build(c, sort=10)
        self.spect_S = S
        T = TextLayer("spect_text")
        self._spect_size = 0.0098
        self.spect_lab = T.label("", x0 + 0.024, y0 + H + 0.0133,
                                 self._spect_size, weight="bold",
                                 track=0.0012, cap=64)
        T.build(c, sort=30)
        self.spect_T = T
        self._spect_text = None

    def _fit_spectator(self, text: str):
        x0, _y0, W, _H = self._chy
        lab = self.spect_lab
        lab.set(text)
        lab.resize(self._spect_size)
        room = W - 0.024 - 0.012
        if lab.width > room:
            lab.resize(self._spect_size * room / lab.width)
        self.spect_bg.set_rect(x0, self._spect_y, 0.024 + lab.width + 0.012, 0.024)
        self.spect_S.flush()
        self.spect_T.flush()

    # -- cards ------------------------------------------------------------------
    def _card_header(self, S, T, x0, top, W, title, parent, sort):
        S.rect(x0, top - 0.048, W, 0.048, INK)
        S.rect(x0, top - 0.048, 0.110, 0.048, RED)
        self._logo(parent, x0 + 0.055, top - 0.024, 0.027,
                   mono=(255, 255, 255, 255), sort=sort)
        T.label(title, x0 + 0.126, top - 0.032, 0.0165, weight="bold",
                track=0.0016)

    def _build_finish(self):
        """Result card: the classification, built once and hidden."""
        f = Entity(parent=self.root, z=-0.2, enabled=False)
        self.finish = f
        S, D, T = ShapeLayer("fin"), ShapeLayer("fin_live"), TextLayer("fin_text")
        n = self.n_rows
        RH = 0.0300 if n > 10 else 0.050
        W = 0.800
        H = 0.170 + n * RH
        x0, top = -W / 2, H / 2
        S.rect(x0, top - H, W, H, pal.rgb(14, 14, 20, 246))
        self._card_header(S, T, x0, top, W,
                          "QUALIFYING RESULT" if self.quali else "RACE RESULT",
                          f, 42)
        T.label(self.full_name.upper(), x0 + W - 0.018, top - 0.031, 0.0110,
                weight="semibold", align="right", col=GREY, track=0.0012)
        laps = f"{self.total_laps} LAP" + ("S" if self.total_laps != 1 else "")
        T.label(f"{self.country}  ·  {laps}", x0 + 0.018,
                top - 0.072, 0.0100, weight="semibold", col=GREY,
                track=0.0012)
        cols = dict(pos=x0 + 0.036, bar=x0 + 0.060, tla=x0 + 0.072,
                    name=x0 + 0.130, team=x0 + 0.300, best=x0 + 0.560,
                    gap=x0 + W - 0.024)
        hy = top - 0.100
        for lab, x, al in (("POS", cols["pos"], "center"),
                           ("DRIVER", cols["tla"], "left"),
                           ("TEAM", cols["team"], "left"),
                           ("BEST LAP", cols["best"], "right"),
                           ("GAP" if self.quali else "TIME / GAP",
                            cols["gap"], "right")):
            T.label(lab, x, hy, 0.0080, weight="semibold", align=al, col=DIM,
                    track=0.0014)
        self.fin_rows = []
        y = hy - 0.008
        size = 0.0110 if n > 10 else 0.016
        for r in range(n):
            yb = y - RH
            bg = D.rect(x0 + 0.018, yb + 0.001, W - 0.036, RH - 0.002,
                        BG_ROW if r % 2 == 0 else BG_ROW2)
            bar = D.rect(cols["bar"], yb + RH * 0.2, 0.0045, RH * 0.6, TEAM_YOU)
            pen_bg = D.rect(cols["best"] + 0.012, yb + 0.005, 0.050, RH - 0.010,
                            AMBER)
            pen_bg.show(False)
            cy = yb + RH / 2
            self.fin_rows.append(dict(
                bg=bg, bar=bar, pen_bg=pen_bg,
                pos=T.label("", cols["pos"], cy - size * 0.5, size, weight="bold",
                            align="center", cap=2),
                tla=T.label("", cols["tla"], cy - size * 0.5, size, weight="bold",
                            cap=3),
                name=T.label("", cols["name"], cy - size * 0.42, size * 0.85,
                             weight="semibold", col=GREY, cap=16),
                team=T.label("", cols["team"], cy - size * 0.42, size * 0.85,
                             weight="semibold", col=GREY, cap=16),
                best=T.label("", cols["best"], cy - size * 0.45, size * 0.92,
                             weight="semibold", align="right", cap=10,
                             mono=True),
                gap=T.label("", cols["gap"], cy - size * 0.45, size * 0.92,
                            weight="bold", align="right", cap=12, mono=True),
                pen=T.label("", cols["best"] + 0.037, cy - size * 0.38,
                            size * 0.75, weight="bold", align="center", col=INK,
                            cap=5)))
            y = yb
        S.rect(x0 + 0.018, top - H + 0.040, W - 0.036, 0.0015, RULE)
        T.label("ENTER  /  ESC     BACK TO MENU", x0 + 0.018, top - H + 0.016,
                0.0096, weight="semibold", col=GREY, track=0.0014)
        S.build(f, sort=40)
        D.build(f, sort=41)
        T.build(f, sort=45)
        self.fin_S, self.fin_D, self.fin_T = S, D, T
        self._fin_key = None

    def _build_pause(self):
        """The pause card: a short menu, picked with W/S and ENTER."""
        self.pause_items = (("resume", "RESUME"),
                            *(() if self.quali else (("restart", "RESTART RACE"),)),
                            ("exit", "EXIT TO MENU"))
        n = len(self.pause_items)
        W = 0.500
        H = 0.205 + 0.042 * n
        x0, top = -W / 2, H / 2 + 0.03
        p = Entity(parent=self.root, z=-0.25, enabled=False)
        self.pause = p
        S, T = ShapeLayer("pause"), TextLayer("pause_text")
        S.rect(x0, top - H, W, H, pal.rgb(14, 14, 20, 246))
        self._card_header(S, T, x0, top, W, "PAUSED", p, 52)
        T.label(self.circuit, x0 + 0.024, top - 0.098, 0.026, weight="bold",
                track=0.0010)
        session = (f"{'QUALIFYING' if self.quali else 'GRAND PRIX'}  ·  "
                   f"{self.total_laps} LAPS")
        T.label(f"{self.full_name.upper()}  ·  {session}", x0 + 0.024,
                top - 0.124, 0.0098, weight="semibold", col=GREY,
                track=0.0012)
        S.rect(x0 + 0.024, top - 0.146, W - 0.048, 0.0015, RULE)
        self._pause_y = []
        self.pause_rows = []
        self._pause_hl = []
        for k, (_, label) in enumerate(self.pause_items):
            y = top - 0.188 - 0.042 * k
            self._pause_y.append(y)
            hl = S.rect(x0 + 0.024, y - 0.012, W - 0.048, 0.036, BG_HI)
            cur = S.rect(x0 + 0.024, y - 0.012, 0.006, 0.036, RED)
            hl.show(False)
            cur.show(False)
            self._pause_hl.append((hl, cur))
            self.pause_rows.append(T.label(label, x0 + 0.044, y - 0.0005,
                                           0.0150, weight="bold", col=GREY,
                                           track=0.0012))
        T.label("W/S  SELECT     ENTER  CONFIRM     ESC  RESUME     P  HIDE",
                x0 + 0.024, top - H + 0.020, 0.0088, weight="semibold",
                col=DIM, track=0.0012)
        S.build(p, sort=50)
        T.build(p, sort=55)
        self.pause_S, self.pause_T = S, T
        self._pause_sel = -1

    def _set_pause_sel(self, sel: int):
        if sel == self._pause_sel:
            return
        self._pause_sel = sel
        for k, row in enumerate(self.pause_rows):
            on = k == sel
            row.color = WHITE if on else GREY
            self._pause_hl[k][0].show(on)
            self._pause_hl[k][1].show(on)
        self.pause_S.flush()
        self.pause_T.flush()

    # -- per-frame update -----------------------------------------------------
    def stale(self) -> bool:
        """True when the window's shape has changed under the layout."""
        return abs(float(window.aspect_ratio) - self._aspect) > 1e-3

    def _place_row(self, row, dy: float):
        """Move a car row's pieces to *dy* below the first slot."""
        if dy == row["dy"] and row.get("_placed"):
            return
        row["dy"] = dy
        row["_placed"] = True
        for k in ("hl", "bar", "fast", "up", "dn", "pen_bg", "st_bg"):
            row[k].move(0.0, dy)
        for k, (bx, by) in row["base"].items():
            row[k].move(bx, by + dy)

    def _set_row(self, row, e, leader: bool):
        st = row["state"]
        player = bool(e.get("player"))
        col = e.get("col", TEAM_AI)
        if st.get("col") != col:
            st["col"] = col
            row["bar"].color = col
        # The player's row is lit white; the car being watched (G), red,
        # the colour of the onboard card that names it.
        hl = "me" if player else "watch" if e.get("watched") else ""
        if st.get("hl") != hl:
            st["hl"] = hl
            row["hl"].show(bool(hl))
            if hl:
                row["hl"].color = HILITE if player else WATCH_HI
            row["tla"].color = INK if player else WHITE
        row["tla"].set(e["tla"])
        gap = e.get("gap", "")
        word = bool(e.get("word"))
        gcol = (INK if player else
                (PURPLE if (self.quali and leader and e.get("purple")) else
                 WHITE if leader or hl or gap.startswith("+") else GREY))
        if word:
            row["gap"].set("")
            row["word"].set(gap, INK if player else GREY)
        else:
            row["word"].set("")
            row["gap"].set(gap, gcol)
        row["fast"].show(bool(e.get("purple")) and not self.quali)
        ch = int(e.get("change", 0) or 0)
        row["up"].show(ch > 0)
        row["dn"].show(ch < 0)
        row["chg"].set(str(min(abs(ch), 99)) if ch else "",
                       GREEN if ch > 0 else RED)
        pen = e.get("pen") or 0.0
        pen_txt = pen_tag(pen) if pen > 0.05 else ""
        status = e.get("status", "")
        if st.get("chips") != (pen_txt, status):
            st["chips"] = (pen_txt, status)
            row["pen_bg"].show(bool(pen_txt))
            row["pen"].set(pen_txt)
            row["st_bg"].show(bool(status))
            if status:
                # A car out of the race for now is what brought the yellow
                # out (it is the stricken car): its chip is the yellow flag.
                row["st_bg"].color = (DRS_GREEN if status == "DRS" else
                                      YELLOW if status in ("OUT", "STOP")
                                      else WHITE)
            row["st"].set(status)
            self._layout_chips(row)

    def _layout_chips(self, row):
        """The chips hang off the row's right edge, each as wide as its text,
        in a fixed order -- penalty, then DRS or OUT. One alone sits against
        the tower; a second is pushed out beside the first."""
        x = self._chip_x0
        dy = row["dy"] or 0.0
        for bg, k in ((row["pen_bg"], "pen"), (row["st_bg"], "st")):
            lab = row[k]
            if not lab.text:
                continue
            w = lab.width + 2 * self.CHIP_PAD
            bg.set_rect(x, row["chip_y"], w, self._chip_h)
            by = row["base"][k][1]
            row["base"][k] = (x + self.CHIP_PAD, by)
            lab.move(x + self.CHIP_PAD, by + dy)
            x += w + self.CHIP_GAP

    def _update_tower(self, standings, dt: float, gap_mode: str):
        RH = self.ROW_H
        live = set()
        for k, e in enumerate(standings[:self.n_rows]):
            key = e.get("key", e["tla"])
            live.add(key)
            r = self._row_of.get(key)
            if r is None:
                # A car the tower has not shown yet takes a free row, placed
                # straight into its slot (no slide in from the top).
                used = set(self._row_of.values())
                r = next(j for j in range(len(self.rows)) if j not in used)
                self._row_of[key] = r
                self.rows[r]["dy"] = None
                self.rows[r]["_placed"] = False
            row = self.rows[r]
            target = -k * RH
            row["target"] = target
            if not row["shown"]:
                row["shown"] = True
                row["bar"].show(True)
                self._place_row(row, target)
            # The same entry as last frame (the tower is rebuilt ten times a
            # second, not sixty) needs nothing rewritten.
            if row.get("entry") is not e:
                row["entry"] = e
                self._set_row(row, e, leader=k == 0)
        for key in [k for k in self._row_of if k not in live]:
            row = self.rows[self._row_of.pop(key)]
            row["shown"] = False
            for k in ("hl", "bar", "fast", "up", "dn", "pen_bg", "st_bg"):
                row[k].show(False)
            for k in ("tla", "chg", "gap", "word", "pen", "st"):
                row[k].set("")
            row["state"] = {}
            row["entry"] = None
        # Slide rows towards their slots.
        a = min(1.0, self.SLIDE_RATE * dt)
        for row in self.rows:
            if not row["shown"] or row["dy"] is None:
                continue
            d = row["target"] - row["dy"]
            if abs(d) > 1e-5:
                self._place_row(row, row["target"] if abs(d) < 0.0006
                                else row["dy"] + d * a)
        self.t_mode.set("TIME" if self.quali else
                        "LEADER" if gap_mode == "leader" else "INTERVAL")

    def update(self, *, speed_kmh, speed_frac, lap, cur_t, last_t, best_t,
               session_t, sectors, standings, lights=-1, gantry_dy=0.0,
               flag="", spectating=False, car_xz=None, ghost_xz=None,
               throttle=0.0, brake=0.0, steer=0.0, slip=0.0, tc_cut=0.0,
               esc_cut=0.0, tc_off=False, dt=1.0 / 60.0, finished=False,
               paused=False, pause_sel=0, delta=None, cur_invalid=False,
               last_invalid=False, flag_style="warn", flag_head="RACE CONTROL",
               field_dots=None, strip="idle", gap_mode="interval",
               me_name="PLAYER", me_col=TEAM_YOU, me_pos=None,
               target_label=None, target_t=None, spect_label=None,
               results=None, yellow_zones=(), yellow_here=False):
        # -- tower --------------------------------------------------------------
        self.t_head.set("FINISH" if lap > self.total_laps
                        else "OUT" if lap == 0 and self.quali
                        else f"{lap}/{self.total_laps}")
        self._update_tower(standings, dt, gap_mode)
        chq = strip == "chequered"
        if getattr(self, "_strip", None) != strip:
            self._strip = strip
            for sq in self.strip_chq:
                sq.show(chq)
            self.strip.show(not chq)
            if not chq:
                self.strip.color = STRIP_COL.get(strip, STRIP_IDLE)

        # -- chyron -------------------------------------------------------------
        me = None
        for e in standings:
            if bool(e.get("player")) != bool(spectating):
                me = e
                break
        if spectating and spect_label is not None:
            me = next((e for e in standings if e.get("watched")), me)
        self.c_bar.color = me_col
        self.c_name.set(me_name)
        self.c_pos.set(str(me_pos if me_pos is not None
                           else me["pos"] if me else 1))
        if self.quali and lap == 0:
            self.c_lap.set("OUT LAP")
        else:
            self.c_lap.set("FINISHED" if lap > self.total_laps
                           else f"LAP {lap}/{self.total_laps}")

        # A lap just completed holds on screen in full, in its colour, the way
        # the broadcast freezes the chyron as the car crosses the line.
        if last_t is not None and last_t != self._last_seen:
            if self._last_seen is not None or lap > 1:
                session_best = min((e["best"] for e in standings
                                    if e.get("best") is not None), default=None)
                if last_invalid:
                    self._flash_col = RED
                elif best_t is not None and abs(last_t - best_t) < 1e-6:
                    self._flash_col = (PURPLE if session_best is not None
                                       and last_t <= session_best + 1e-6
                                       else GREEN)
                else:
                    self._flash_col = YELLOW
                self._flash_until = session_t + 3.0
            self._last_seen = last_t
        if finished:
            self._chy_time(lap_time(last_t), WHITE)
        elif lap == 0 and self.quali:
            self._chy_time(None, GREY, word="OUT LAP")
        elif session_t < self._flash_until and last_t is not None:
            self._chy_time(lap_time(last_t), self._flash_col)
        else:
            self._chy_time(tenths(cur_t), RED if cur_invalid else WHITE)

        if delta is not None and not finished:
            self.c_delta_bg.show(True)
            self.c_delta_bg.color = GREEN if delta < 0 else RED
            self.c_delta.set(delta_time(delta))
        else:
            self.c_delta_bg.show(False)
            self.c_delta.set("")
        if target_t is not None or target_label is not None:
            self.c_target.set(lap_time(target_t) if target_t is not None else "",
                              WHITE)
            self.c_target_who.set(target_label or "")
        else:
            self.c_target.set(lap_time(best_t),
                              PURPLE if me and me.get("purple") else WHITE)
            self.c_target_who.set("BEST LAP")
        for bar, (_, status) in zip(self.c_sectors, sectors):
            bar.color = SECTOR_COL.get(status, SECTOR_COL["off"])

        # -- speed graphic --------------------------------------------------
        self.s_bar.color = me_col
        self.s_name.set(me_name)
        self.s_speed.set(f"{max(0.0, speed_kmh):0.0f}"[-3:])
        gear, within = gear_of(speed_frac)
        self.s_gear.set(str(gear) if speed_kmh > 2.0 else "N")
        target = 0.25 + 0.75 * within if speed_kmh > 2.0 else 0.0
        self._rpm += (target - self._rpm) * min(1.0, 9.0 * dt)
        lit = int(self._rpm * N_LEDS + 0.5)
        if lit != self._lit:
            self._lit = lit
            for k, led in enumerate(self.leds):
                led.color = LED_ON[k] if k < lit else LED_OFF
        self.p_thr.stretch(1.0, max(0.002, throttle), oy=self._bar_y)
        self.p_brk.stretch(1.0, max(0.002, brake), oy=self._bar_y)
        deg = math.degrees(slip)
        if deg > 6.0:
            self.s_slip.set(f"SLIP {deg:0.0f}°", AMBER if deg < 14 else RED)
        else:
            self.s_slip.set("")
        if tc_off:
            self.s_aid.set("AIDS OFF", RED)
        elif esc_cut > 0.02:
            self.s_aid.set("ESP", GREEN)
        elif tc_cut > 0.05:
            self.s_aid.set("TC", AMBER)
        else:
            self.s_aid.set("")

        # -- map ------------------------------------------------------------
        dots = field_dots or ([] if ghost_xz is None else [(ghost_xz, TEAM_AI)])
        xy = self._mm_xy
        xy[:] = np.nan
        n = min(len(dots), len(self.mm_others))
        if n:
            p = np.array([d[0] for d in dots[:n]], np.float32)
            p[:, 0] = (p[:, 0] - self._mm_cx) * self._mm_scale + self._mm_o[0]
            p[:, 1] = (p[:, 1] - self._mm_cz) * self._mm_scale + self._mm_o[1]
            xy[0:2 * n:2] = p
            xy[1:2 * n:2] = p
            for k in range(n):
                self.mm_others[k][1].color = dots[k][1]
        self.mm_batch.place(xy)
        self._animate_yellow_frame(bool(yellow_here), dt)
        zones = tuple(tuple(z) for z in yellow_zones)
        if zones != self._mm_zones:
            self._mm_zones = zones
            L = self._mm_L
            for mid, seg in self._mm_yel:
                seg.show(any((mid - a) % L <= (b - a) % L for a, b in zones))
            self.mm_Y.flush()
        if car_xz is not None:
            u, v = self._to_mm(car_xz)
            self.mm_dot.move(u, v)
            self.mm_ring.move(u, v)

        # -- start lights -----------------------------------------------------
        show = lights >= 0
        if self.gantry.enabled != show:
            self.gantry.enabled = show
        if show:
            self.gantry.y = self._gantry_y0 + gantry_dy
            if lights != self._lit_n:
                self._lit_n = lights
                for k, col in enumerate(self.lamps):
                    on = k < lights
                    for glow, lamp in col:
                        lamp.color = LAMP_ON if on else LAMP_OFF
                        glow.show(on)
                self.g_S.flush()

        # -- race control / spectator ------------------------------------------
        tag = spect_label or ""
        if self.spectator.enabled != bool(tag):
            self.spectator.enabled = bool(tag)
        if tag and tag != self._spect_text:
            self._spect_text = tag
            self._fit_spectator(tag)
        show_flag = bool(flag) and not spectating
        self._animate_flag((flag, flag_style, flag_head) if show_flag else None,
                           dt)

        # -- cards --------------------------------------------------------------
        if self.finish.enabled != finished:
            self.finish.enabled = finished
        if finished:
            self._update_results(results if results is not None else standings)
        if self.pause.enabled != paused:
            self.pause.enabled = paused
        if paused:
            self._set_pause_sel(pause_sel)

        self.D.flush()
        self.L.flush()
        self.mm_S.flush()

    def _chy_time(self, digits, col, word=None):
        if word is not None:
            self.c_time.set("")
            self.c_time_word.set(word, col)
        else:
            self.c_time_word.set("")
            self.c_time.set(digits, col)

    def _update_results(self, rows):
        key = tuple((e.get("tla"), e.get("gap"), e.get("pen"), e.get("best"))
                    for e in rows)
        if key == self._fin_key:
            return
        self._fin_key = key
        for k, row in enumerate(self.fin_rows):
            on = k < len(rows)
            row["bg"].show(on)
            row["bar"].show(on)
            if not on:
                for f in ("pos", "tla", "name", "team", "best", "gap", "pen"):
                    row[f].set("")
                row["pen_bg"].show(False)
                continue
            e = rows[k]
            row["pos"].set(str(e["pos"]))
            row["bar"].color = e["col"]
            row["tla"].set(e["tla"])
            row["name"].set(e.get("name", "").upper())
            row["team"].set(e.get("team", "").upper())
            row["best"].set(lap_time(e.get("best")),
                            PURPLE if e.get("purple") else WHITE)
            row["gap"].set(e.get("result", e.get("gap", "")),
                           WHITE if k == 0 else GREY)
            pen = e.get("pen") or 0.0
            row["pen_bg"].show(pen > 0.05)
            row["pen"].set(pen_tag(pen) if pen > 0.05 else "")
            row["bg"].color = BG_HI if e.get("player") else (
                BG_ROW if k % 2 == 0 else BG_ROW2)
        self.fin_D.flush()
        self.fin_T.flush()

    def destroy(self):
        destroy_tree(self.root)
        self.root = None
