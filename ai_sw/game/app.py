"""FORMULA-AI racing prototype — human-driven, Phase A procedural assets, no AI yet."""
from __future__ import annotations

import argparse
import math
import random

import numpy as np
from ursina import Ursina, Vec3, camera, held_keys, time, window

from . import config, grandprix, teams
from . import palette as pal
from . import post
from .car import Car, lerp_pose
from .freecam import FreeCam
from .ghost import Ghost
from .enginesound import EngineSound
from .hud import HUD
from .cinematic import Intro
from .menu import GRAND_PRIX, QUALI
from .replay import ReplayGhost, load as load_ghost_lap
from .fieldproc import C as SNAP_COLS
from .racecontrol import RaceControl
from .rules import TrackLimits
from .ui import TEAM_AI, TEAM_YOU, lap_time
from .vehicle import Controls, Vehicle
from .world import World

DT = 1.0 / config.PHYSICS_HZ
COUNTDOWN, RACING, FINISHED, PAUSED, INTRO = range(5)


def _col4(rgb) -> tuple:
    """A team's (r, g, b) 0..1 as the opaque RGBA the HUD takes."""
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)


class Game:
    def __init__(self, track_name: str, laps: int, mute: bool = False,
                 on_exit=None, progress=None, mode: str = GRAND_PRIX,
                 world: World | None = None, intro: bool = True,
                 on_restart=None, level: int = teams.DEFAULT_LEVEL,
                 player_time: float | None = None, spectate: bool = False,
                 player_grid: int | None = None):
        #: Qualifying: solo hot laps against a replay of the AI's best lap.
        #: Grand prix: a race against the live AI, timed from lights out.
        self.mode = mode
        self.quali = mode == QUALI
        #: Watching, not driving (G on the circuit list): a grand prix of
        #: twenty AI drivers, or the pole lap of qualifying, lap after lap.
        #: The player's car is hidden and parked; ESC leaves as ever.
        self.spectate = spectate
        #: Difficulty (teams.DIFFICULTY): how fast the AI is, and how strict
        #: race control is.
        self.level = level if level in teams.DIFFICULTY else teams.DEFAULT_LEVEL
        # Called between the heavy stages below so the loading card can draw a
        # frame and its bar keeps moving. A no-op when there is no card (the
        # tools and the --track launch build straight into a live window).
        step = progress if progress is not None else (lambda: None)
        #: The grand prix field: nineteen AI cars in a worker process
        #: (fieldproc.py), started first so it builds while the circuit does.
        #: Circuits without solved plans fall back to the single AI ghost.
        self.field = None
        self.gp = None
        self._snap = None
        #: Contact's push on the player's car not yet applied (_ease_kick).
        self._kick_pos = np.zeros(2)
        self._gap_mode = "interval"
        self._watch_idx: int | None = None
        #: The cars G has stepped through since leaving the player's car.
        self._watch_hist: list[int] = []
        #: Race control's messages: waiting their turn, [mine, text, kind,
        #: posted at], and the one on show, [mine, text, kind, seconds left].
        self._rc_queue: list = []
        self._rc_cur: list | None = None
        self._rc_clock = 0.0
        self._rc_dt = 0.0
        self._tower_cache = None
        self._tower_t = -1.0
        if (not self.quali and config.GP_FIELD
                and grandprix.ready(track_name)):
            from .fieldproc import FieldClient
            self.field = FieldClient(track_name, self.level, laps,
                                     seed=random.randrange(1 << 30),
                                     player_time=player_time,
                                     spectate=spectate,
                                     player_grid=player_grid)
        # ``on_exit`` takes the session back to the menu. Owned here rather
        # than by the caller because the race is what knows when it is over.
        self.on_exit = on_exit
        #: Grand prix: the pause card's RESTART RACE.
        self.on_restart = on_restart
        # The circuit -- road, roadside, sky and sun -- is built once and kept
        # across sessions on it (see world.py); only a different circuit, or
        # none handed in, builds a new one.
        if world is None or world.name != track_name:
            world = World(track_name, progress=step)
        self.world = world
        world.reset()
        world.show()
        self.track = world.track
        self.light = world.light
        self.surface = world.surface
        self.scene = world.scene
        self.scenery = world.scenery

        self.vehicle = Vehicle()
        self.vehicle.frozen = True
        pos, yaw = self.track.start_pose()
        self.vehicle.place(pos, yaw)
        self.car = Car(self.vehicle, model=config.PLAYER_MODEL)
        step()

        # The opponent. In a race, its own vehicle, its own surface, its own
        # driver. In qualifying, a recording of that driver's best clean lap
        # -- or nobody, on a circuit that has not had one recorded yet.
        #: Qualifying: the AI's times at this level, fastest first, for the
        #: tower -- (tla, name, team colour, time, team) -- and the pole lap
        #: as the ghost to beat.
        self.quali_field = None
        self.quali_scale = 1.0
        if self.quali:
            from . import replay
            rec = replay.level_lap(track_name, self.level, self.track,
                                   progress=step)
            if rec is None:
                rec = load_ghost_lap(track_name, self.track)
            self.ghost = ReplayGhost(self.track, rec) if rec is not None else None
            if grandprix.ready(track_name):
                table = grandprix.quali_table(track_name, self.level)
                if rec is not None and table:
                    # The table is an estimate; the ghost is the pole lap
                    # actually driven. Scale the one onto the other.
                    self.quali_scale = rec.lap_time / table[0].quali
                self.quali_field = [
                    (s.driver.tla, s.driver.name, s.team.color,
                     s.quali * self.quali_scale, s.team.name) for s in table]
        elif self.field is not None:
            self.ghost = None
        else:
            self.ghost = (Ghost(self.track, out_lap=False)
                          if config.GHOST_ENABLED else None)
        step()

        # The cars get the sunset shader too, with a sharper, stronger
        # highlight than the scenery: bodywork is the one surface here that
        # is actually glossy.
        from .shaders import car_shader
        self.light.apply(self.car, spec_strength=0.55, spec_power=64.0,
                         shader=car_shader)
        self.car.apply_materials(config.PLAYER_MODEL)
        # The hero car alone goes into the car's own shadow map.
        self.light.cast(self.car)
        # A ghost that threw a shadow would not be a ghost, and the shadow
        # would be the one part of it that looked solid.
        if self.ghost is not None:
            self.light.apply(self.ghost.car, spec_strength=0.55,
                             spec_power=64.0, casts=False, shader=car_shader)
            self.ghost.car.apply_materials(config.GHOST_MODEL)
        step()
        if self.field is not None:
            self._join_field(step)

        # One tower row per car in the session: 19 when watching (the
        # player's seat is not raced), 20 otherwise.
        if self.field is not None:
            self._hud_rows = len(self.car_info)
        elif self.quali_field:
            self._hud_rows = len(self.quali_field) + (0 if spectate else 1)
        else:
            self._hud_rows = 2
        self.hud = HUD(self.track, laps, mode, rows=self._hud_rows)
        #: Snapshot columns, and each AI car's dot colour on the map.
        self._C = SNAP_COLS
        self._dot_col = {}
        if self.gp is not None:
            self._dot_col = {k: _col4(self.car_info[k]["color"])
                             for k in self.gp.proxies}
        self._rc_seen = 0
        from .effects import Effects
        self.fx = Effects()
        # Built now but kept dark: the loading card is still up, and the HUD
        # should arrive with the start lights, not flash in behind the fade.
        # The first ``update`` reveals it (unless something has cleared the
        # flag, e.g. bench_fps's --no-hud).
        self.hud.root.enabled = False
        self._reveal_hud = True
        #: H hides the whole overlay -- pause card included -- for a clean
        #: view of the car. Kept here, not on the HUD, so it survives the HUD
        #: being rebuilt.
        self.hud_hidden = False
        #: The highlighted row on the pause card, and whether P has put the
        #: card away to look at the frozen frame with the rest of the HUD up.
        self._pause_sel = 0
        self.pause_card_hidden = False
        self.total_laps = laps

        # Which camera a session opens on -- not index 0; see CAM_DEFAULT.
        self.cam_idx = (config.CAM_MODES.index(config.CAM_DEFAULT)
                        if config.CAM_DEFAULT in config.CAM_MODES else 0)
        self.freecam = FreeCam()
        self._cooldown = None          # autopilot that drives the in-lap
        self._final_rows = None        # standings frozen at the flag
        self.finish_t = 0.0
        self._cam_aim = Vec3(0, 0, 0)
        self._alpha = 1.0
        # Spectator: when set, the camera and the HUD follow the AI instead of
        # the player. Toggled with G. Useful at an exhibition -- "watch how the
        # AI takes this corner" -- and the player's car keeps driving under
        # their input the whole time.
        self._watch_ghost = False
        if spectate and (self.quali or self.field is None):
            # Qualifying's pole lap, or the single AI of a circuit without a
            # field: there is one car to watch.
            self._watch_ghost = True
        #: X: the camera turned round to look behind the car it follows.
        self._look_back = False
        self._init_camera()

        self.sound = EngineSound(config.ASSET_DIR)
        from .impactsound import ImpactSound
        self.impacts = ImpactSound(config.ASSET_DIR)
        self._scraping = False
        self._prev_hits = None
        self.muted = mute

        # The five lit lamp columns on the gantry, left to right. They come on
        # one at a time through the countdown, mirroring the HUD lamps exactly.
        self._lit_lamps = world.lit_lamps

        # session state. The film runs first and holds the start clock at zero
        # until it is done, so the countdown timeline below is untouched by it.
        # A restart goes straight back to the grid: the film is for arriving.
        self.intro = (Intro(self.track) if config.INTRO_ENABLED and intro
                      else None)
        if self.intro is not None and self.intro.done:
            self.intro = None
        self.state = INTRO if self.intro is not None else COUNTDOWN
        self._resume_state = self.state
        # Seconds since the loading card lifted. The whole start sequence --
        # gantry drop, five lamps, held pause, lights out, gantry rise -- is
        # driven off this one clock; the pause length is randomised the way a
        # real start is so it cannot be counted. The timeline marks below are
        # cumulative offsets into it.
        self.start_t = 0.0
        self._hold = random.uniform(config.START_HOLD_MIN, config.START_HOLD_MAX)
        self._T_drop0 = config.START_WAIT
        self._T_drop1 = self._T_drop0 + config.START_GANTRY_DROP
        self._T_build1 = self._T_drop1 + config.START_LIGHTS_BUILD
        self._T_hold1 = self._T_build1 + self._hold
        self._T_out = self._T_hold1 + config.START_LIGHTS_OUT   # car released here
        self._T_rise1 = self._T_out + config.START_GANTRY_RISE
        if self.quali:
            # No start in qualifying: the car is released the moment the film
            # ends, and the gantry never comes down.
            self.start_t = self._T_rise1 + 1.0
        self.session_time = 0.0
        self.lap_start = 0.0
        # Qualifying starts on lap 0, the out lap: the HUD shows it as OUT and
        # nothing is timed until the car crosses the line for the first time.
        # A grand prix is timed from lights out, so it starts on lap 1.
        self.lap_num = 0 if self.quali else 1
        #: Qualifying: the lap in progress has had a moment off the track,
        #: and whether the last completed lap had.
        self.lap_invalid = False
        self.last_invalid = False
        #: Grand prix: distance raced and penalties for each car, the lap
        #: number at which each takes the flag (brought forward to "next time
        #: over the line" once the other car has finished, so a lapped car is
        #: not left to run its full distance), and the player's race time.
        self.limits = TrackLimits(self.track, config.GP_OFFTRACK_REJOIN)
        self.ai_limits = TrackLimits(self.track, config.GP_OFFTRACK_REJOIN)
        #: Stewards for the single-ghost race (car 0 the player, 1 the AI);
        #: a race with the field has its own, in the worker.
        lvl = teams.DIFFICULTY[self.level]
        self.rc = RaceControl(2, lvl.strike_pen, lvl.minor_repeat_pen)
        self.rc.opening = False
        self._you_finish_lap = laps + 1
        self._ai_finish_lap = laps + 1
        self.finish_race_t: float | None = None
        self.finish_laps = 0
        self._last_spect = False
        self.last_t = None
        self.best_t = None
        # Sector timing: the lap in thirds by distance. The lights under the
        # running lap time go purple / green / yellow the way a timing tower's
        # do, so a driver knows *where* on the lap they gained or lost.
        self.sector = 0
        self.sector_start = 0.0
        self.sectors: list[float | None] = [None, None, None]
        self.best_sectors: list[float | None] = [None, None, None]
        self.armed = False
        self._accum = 0.0
        self._ctl = Controls()
        #: The first live frame re-snaps the camera. The loading card holds the
        #: update loop for ~1 s while the race is built, so ``time.dt`` on the
        #: frame it hands back is that whole gap -- long enough for the damped
        #: camera to lurch and settle. One hard snap kills it.
        self._cam_warm = False
        # Everything built so far lives for the whole session: collect once
        # and freeze it out of the cyclic GC, which otherwise walks all of it
        # on every full collection -- frames of 10-20 ms, now and then, in
        # the middle of a race. Unfrozen again in destroy().
        if self.spectate:
            self.car.enabled = False
            self.vehicle.frozen = True
        import gc
        gc.collect()
        gc.freeze()

    # -- the field ----------------------------------------------------
    def _join_field(self, step):
        """Wait for the worker's field (pumping the loading card), put the
        player in their grid slot and the other nineteen cars on screen. On
        any failure, race the single AI ghost instead."""
        cl = self.field
        if not cl.wait_ready(timeout=90.0, pump=step):
            print("grand prix field unavailable:", cl.error)
            cl.close()
            self.field = None
            if config.GHOST_ENABLED:
                from .shaders import car_shader
                self.ghost = Ghost(self.track, out_lap=False)
                self.light.apply(self.ghost.car, spec_strength=0.55,
                                 spec_power=64.0, casts=False,
                                 shader=car_shader)
                self.ghost.car.apply_materials(config.GHOST_MODEL)
            return
        from .gpfield import GPCars
        if cl.player is not None:
            me = cl.info[cl.player]
            self.vehicle.place(np.asarray(me["pos"], float), float(me["yaw"]))
            self.surface.progress(self.vehicle.pos)
            self.car.sync()
        self.car_info = {c["idx"]: c for c in cl.info}
        self.gp = GPCars(self.track, cl.info, cl.player, self.light,
                         progress=step)
        if cl.player is None:
            # Spectating: open on the pole sitter.
            self._watch_idx = min(cl.info, key=lambda c: c["grid"])["idx"]
        # The grid boxes, painted where the field actually put the cars.
        from .trackmesh import grid_boxes
        self._grid_paint = grid_boxes(
            self.track, [(float(c["pos"][0]), float(c["pos"][1]),
                          float(c["yaw"])) for c in cl.info], self.light)

    def _field_frame(self, dt: float):
        """Once a frame in a grand prix with a field: tell the worker where
        the player is, take its newest snapshot, apply what contact did to
        the player's car, and pose the other cars."""
        cl = self.field
        racing = self.state in (RACING, FINISHED)
        cl.tick(self.session_time if racing else 0.0, self.vehicle,
                self._ctl.brake > 0.05)
        snap, kick = cl.poll()
        if cl.error and snap is None and self._snap is None:
            return
        if snap is not None:
            self._snap = snap
            if kick is not None and np.any(np.abs(kick) > 1e-9):
                # The change of speed lands at once -- that is the hit. The
                # push apart is fed in over the next few physics steps
                # (_ease_kick): put in whole, the car jumped a hand's breadth
                # sideways in one frame and read as a teleport.
                v = self.vehicle
                self._kick_pos = self._kick_pos + kick[0:2]
                v.vel = v.vel + kick[2:4]
                v.yaw_rate += float(kick[4])
                self.impacts.hit(float(np.hypot(kick[2], kick[3])))
            self._race_control(snap)
            self._contact_sounds(snap)
            from .fieldproc import C
            if cl.player is not None:
                me = snap["rows"][cl.player]
                done = np.isfinite(me[C["finish_t"]])
            else:
                # Watching: the result once everyone is home, or a minute
                # after the winner if someone is circulating a lap down.
                fin = snap["rows"][:, C["finish_t"]]
                lead = snap["leader_done"]
                done = (bool(np.all(np.isfinite(fin))) or
                        (lead is not None and snap["t"] - lead > 60.0))
            if self.state == RACING and done:
                self._finish_field()
        t_render = self.session_time - (1.0 - self._alpha) * DT
        self.gp.update(self._snap, t_render, dt, camera.world_position,
                       watch=self._watch_idx)

    #: Other cars throw floor sparks within this distance of the camera (m).
    FX_RANGE = 70.0

    def _fx_others(self):
        if self.gp is None or self._snap is None:
            return ()
        cam = camera.world_position
        r2 = self.FX_RANGE * self.FX_RANGE
        ghost = self._snap["rows"][:, self._C["ghost"]]
        out = []
        for k, v in self.gp.proxies.items():
            dx, dz = float(v.pos[0]) - cam.x, float(v.pos[1]) - cam.z
            if dx * dx + dz * dz < r2 and ghost[k] < 0.5:
                out.append((k, v, self.gp.cars[k]))
        return out

    def _contact_sounds(self, snap):
        """Other cars hitting each other: heard if near the camera, fainter
        with distance. A contact counts once for the pair (both cars' hit
        counters go up together)."""
        C = self._C
        hits = snap["rows"][:, C["hits"]].copy()
        prev = self._prev_hits
        self._prev_hits = hits
        if prev is None or len(prev) != len(hits):
            return
        cam = camera.world_position
        done = []
        for k in np.flatnonzero(hits > prev):
            if k == self.field.player:
                continue                      # the kick has already sounded
            r = snap["rows"][k]
            x, z = float(r[C["x"]]), float(r[C["z"]])
            if any((x - a) ** 2 + (z - b) ** 2 < 36.0 for a, b in done):
                continue
            done.append((x, z))
            d = math.hypot(x - cam.x, z - cam.z)
            self.impacts.hit(3.0, gain=1.0 / (1.0 + (d / 25.0) ** 2))

    #: Time constant of the push apart after a contact, seconds.
    KICK_EASE_T = 0.04

    def _ease_kick(self):
        """Feed one physics step's share of a contact's push into the car."""
        k = self._kick_pos
        if abs(k[0]) + abs(k[1]) < 1e-5:
            return
        share = 1.0 - math.exp(-DT / self.KICK_EASE_T)
        d = k * share
        self.vehicle.pos = self.vehicle.pos + d
        self._kick_pos = k - d

    def _finish_field(self):
        """The field says the player has taken the flag."""
        self.state = FINISHED
        self.finish_t = self.session_time
        self.finish_race_t = self.session_time
        from .autopilot import Autopilot
        self._cooldown = Autopilot(self.track, self.surface)

    def _race_control(self, snap):
        """Stewards' messages for the HUD: everything about the player, and
        penalties for anyone, each up for a few seconds."""
        me = self.field.player
        for t, car, text, kind, other in snap["msgs"]:
            if kind in ("yellow", "green"):
                # Flags are for everyone, and come before other cars' news:
                # the text already says where and who.
                self._rc_push(True, text, kind)
                continue
            mine = car == me or other == me
            if not mine and kind not in ("pen", "flag"):
                continue
            if mine:
                who = "" if car == me else f"{self.car_info[car]['tla']}  ·  "
            else:
                who = f"{self.car_info[car]['tla']}  ·  "
            self._rc_push(mine, who + text, kind)

    # -- race control's messages, one after another ---------------------------
    def _rc_push(self, mine: bool, text: str, kind: str):
        self._rc_queue.append([mine, text, kind, self.session_time])

    def _rc_current(self):
        """The message to show: the one on show until it has had its time,
        then the next -- the player's own first, oldest first. Messages
        about other cars that have waited RC_STALE_T are dropped; the
        player's always get their turn."""
        cur = self._rc_cur
        if cur is not None and cur[3] > 0.0:
            return cur
        now = self.session_time
        q = self._rc_queue
        q[:] = [m for m in q if m[0] or now - m[3] < config.RC_STALE_T]
        if not q:
            self._rc_cur = None
            return None
        k = next((j for j, m in enumerate(q) if m[0]), 0)
        mine, text, kind, _t = q.pop(k)
        hold = config.RC_MESSAGE_T if mine else config.RC_MESSAGE_T * 0.6
        if q:
            # A queue behind it: the longer it is, the shorter each turn,
            # down to what can still be read.
            hold = max(config.RC_MESSAGE_MIN_T, hold / (1.0 + 0.5 * len(q)))
        self._rc_cur = [mine, text, kind, hold]
        return self._rc_cur

    def _rc_spend(self, shown: bool):
        """Run the clock of the message on show -- only while it is shown,
        so one covered by OFF TRACK or a yellow flag is not lost under it."""
        if shown and self._rc_cur is not None:
            self._rc_cur[3] -= self._rc_dt

    # -- camera -----------------------------------------------------
    def _init_camera(self):
        camera.fov = config.CAM_FOV_BASE
        # A fresh race must not inherit the previous one's corner lean: camera
        # is a singleton and nothing else zeroes its roll.
        camera.rotation = Vec3(0, 0, 0)
        #: The corner lean (roll, degrees), kept here rather than read back
        #: off the camera: see _aim_camera.
        self._lean = 0.0
        # See config.CLIP_NEAR: the near plane, not the far one, is what sets
        # depth precision. Ursina's default 0.1 could not separate a road
        # marking from the asphalt beyond ~200 m.
        camera.clip_plane_near = self._clip_near()
        camera.clip_plane_far = config.CLIP_FAR
        self._snap_camera()

    def _clip_near(self) -> float:
        return (config.CLIP_NEAR_ONBOARD
                if config.CAM_MODES[self.cam_idx] == "onboard"
                else config.CLIP_NEAR)

    def _cam_offset(self):
        mode = config.CAM_MODES[self.cam_idx]
        off = {
            "onboard": config.CAM_ONBOARD_OFFSET,
            "chase": config.CAM_CHASE_OFFSET,
            "hood": config.CAM_HOOD_OFFSET,
            "far": config.CAM_FAR_OFFSET,
        }[mode]
        if mode in config.CAM_FRAME_LOCK_MODES:
            k = self._frame_scale()
            off = (off[0] * k, off[1] * k, off[2] * k)
        if mode == "chase":
            # Climb with speed, from the reference speed upwards: see
            # CAM_CHASE_RISE. Added after the frame lock so it is a real
            # rise, not something the lock scales away again.
            v = self._subject()
            span = max(1e-6, config.MAX_SPEED - config.CAM_CHASE_RISE_FROM)
            t = min(1.0, max(0.0, v.speed - config.CAM_CHASE_RISE_FROM) / span)
            off = (off[0], off[1] + config.CAM_CHASE_RISE * t, off[2])
        if self._look_back and mode != "onboard":
            # The chase cameras swing round to the front of the car and look
            # back past it. The onboard stays on the airbox and turns round
            # there: F1's own rear-facing T-cam.
            off = (-off[0], off[1], -off[2])
        return off

    def _frame_scale(self) -> float:
        """How far to pull the chase back to hold the car's size on screen.

        The FOV opens with speed, which shrinks everything in frame -- the
        car included, by 23% between a standstill and 80 km/h. That is the
        one thing in the shot that should not change size: it is the subject,
        and at rest it grew until the telemetry widget covered it. Scaling
        the whole offset by the ratio of the tangents cancels it exactly,
        and scaling the *whole* offset rather than the distance alone keeps
        the angle the car is seen from as well.
        """
        ref = math.tan(math.radians(config.CAM_FRAME_REF_FOV / 2.0))
        now = math.tan(math.radians(max(float(camera.fov), 1.0) / 2.0))
        k = (ref / now) ** config.CAM_FRAME_LOCK
        lo, hi = config.CAM_FRAME_SCALE
        return min(hi, max(lo, k))

    def _subject(self):
        """The vehicle the camera and HUD are following -- the player, or the
        AI while spectating. One accessor so every consumer agrees on it."""
        if self._spectating():
            if self._watch_idx is not None:
                return self.gp.proxies[self._watch_idx]
            return self.ghost.vehicle
        return self.vehicle

    def _render_pose(self):
        """Interpolated (x, z, yaw). The camera must use the same pose as the
        car, or it re-introduces exactly the judder interpolation removes."""
        return lerp_pose(self._subject(), self._alpha)

    def _cam_target_pos(self):
        ox, oy, oz = self._cam_offset()
        px, pz, yaw = self._render_pose()
        cos, sin = math.cos(yaw), math.sin(yaw)
        wx = ox * cos + oz * sin
        wz = -ox * sin + oz * cos
        # Lifted onto the road, and sampled where the camera actually is
        # rather than where the last physics step left the car -- the same
        # reason Car.sync does it that way, and the same stutter if it does
        # not.
        h, _b, _n = self.track.surface_pose([(px + wx, pz + wz)])
        return Vec3(px + wx, oy + float(h[0]), pz + wz)

    def _cam_aim_point(self) -> Vec3:
        """Look ahead down the road, not at the car's nose."""
        v = self._subject()
        px, pz, yaw = self._render_pose()
        # lead further with speed so fast corners open up before you reach them
        lead = config.CAM_LOOKAHEAD * (0.6 + 0.9 * min(1.0, v.speed / config.MAX_SPEED))
        # The onboard sits on the airbox and looks over the halo, so it aims
        # lower than a camera behind the car: aimed at the same height it is
        # mounted at, the bodywork drops out of the frame and the shot is
        # indistinguishable from a floating one.
        mode = config.CAM_MODES[self.cam_idx]
        if mode == "onboard":
            y = config.CAM_ONBOARD_AIM_Y
        elif mode == "chase":
            # The lead above shrinks to 0.6x standing still, which tips the
            # camera up and drops the car down the frame. Aim lower by the
            # same measure and the car holds its place at any speed.
            t = min(1.0, v.speed / config.CAM_CHASE_AIM_SPEED)
            y = (config.CAM_CHASE_AIM_LOW
                 + (config.CAM_CHASE_AIM_Y - config.CAM_CHASE_AIM_LOW) * t)
        else:
            y = 1.35
        if self._look_back:
            lead = -lead
        return Vec3(px + math.sin(yaw) * lead, y, pz + math.cos(yaw) * lead)

    def _snap_camera(self):
        # The onboard needs a closer near plane than the rest (see
        # config.CLIP_NEAR_ONBOARD), and every camera change comes past here.
        near = self._clip_near()
        if abs(float(camera.clip_plane_near) - near) > 1e-6:
            camera.clip_plane_near = near
        camera.position = self._cam_target_pos()
        self._cam_aim = self._cam_aim_point()
        self._aim_camera()

    def _aim_camera(self):
        """Point the camera at ``_cam_aim``, upright, then roll it by the
        lean.

        Ursina's look_at keeps whatever roll the camera already had (it
        aims with the camera's own up), and reading ``rotation_z`` back to
        ease the lean read about 180 degrees the moment the view turned
        round -- the rear view, a cut to another car -- so every cut
        spun the picture over and slowly back. The lean lives in
        ``_lean`` now and the aim is always taken against world up."""
        camera.lookAt(self._cam_aim, Vec3(0, 1, 0))
        if self._lean:
            camera.setR(camera.getR() + self._lean)

    def _update_camera(self, dt: float):
        v = self._subject()
        mode = config.CAM_MODES[self.cam_idx]
        frac = min(1.0, v.speed / config.MAX_SPEED)

        # --- position: lag behind the target so acceleration is felt ----
        target = self._cam_target_pos()
        if mode in ("hood", "onboard"):
            # Both are bolted to the car: a camera that lags its own mounting
            # point is a camera that has come loose.
            camera.position = target
        else:
            t = min(1.0, config.CAM_POS_LERP * dt)
            pos = camera.position + (target - camera.position) * t
            # keep the trail bounded (see CAM_MAX_LAG)
            lag = pos - target
            if lag.length() > config.CAM_MAX_LAG:
                pos = target + lag.normalized() * config.CAM_MAX_LAG
            camera.position = pos

        # --- FOV opens with speed: the periphery stretches --------------
        want_fov = config.CAM_FOV_BASE + config.CAM_FOV_GAIN * (frac ** 1.4)
        if mode == "onboard":
            # Wide at any speed: at the base FOV the nose drops out of the
            # bottom of the frame, so a parked car showed no car at all.
            want_fov = max(want_fov, config.CAM_ONBOARD_FOV_MIN)
        camera.fov += (want_fov - camera.fov) * min(1.0, 3.0 * dt)

        # --- shake, scaled by speed and roughened off-track -------------
        amp = config.CAM_SHAKE * frac
        if not v.on_track:
            amp *= config.CAM_SHAKE_OFFTRACK
        if amp > 1e-4:
            dy = random.uniform(-amp, amp) * 0.7
            if mode == "onboard":
                # Up only. The onboard clears the airbox by 0.42 m and its
                # near plane eats 0.35 of that, so a downward shake at speed
                # is what cut the bodywork open. Shaking upwards instead
                # cannot close that gap, and at these amplitudes the eye
                # cannot tell which way a jitter went.
                dy = abs(dy)
            camera.position += Vec3(random.uniform(-amp, amp), dy,
                                    random.uniform(-amp, amp))

        # --- aim, also damped, so the view swings through corners -------
        want = self._cam_aim_point()
        t = min(1.0, config.CAM_AIM_LERP * dt)
        self._cam_aim = self._cam_aim + (want - self._cam_aim) * t
        # lean into the corner a touch -- lerped, not snapped, and held flat
        # until the car is actually moving, so entering a race does not roll.
        target_lean = (0.0 if self.state == COUNTDOWN
                       else -math.degrees(v.yaw_rate) * config.CAM_LEAN * frac)
        if self._look_back:
            target_lean = -target_lean      # the corner is the other way round
        self._lean += (target_lean - self._lean) * min(1.0, 6.0 * dt)
        self._aim_camera()

    # -- input ----------------------------------------------------
    def read_controls(self) -> Controls:
        if self.state == COUNTDOWN:
            return Controls()
        if self.freecam.enabled:
            # WASD is flying the camera; it must not also be driving. The
            # physics keeps running, so the car coasts while you look around.
            return Controls()
        if self.state == FINISHED:
            # The race is over but the car is not a statue: hand it to the
            # autopilot and let it roll down. Freezing on the line stopped it
            # dead mid-corner, which is the one thing a racing car never does.
            return self._cooldown_controls()
        up = held_keys["w"] or held_keys["up arrow"]
        down = held_keys["s"] or held_keys["down arrow"]
        steer = (held_keys["d"] or held_keys["right arrow"]) - \
                (held_keys["a"] or held_keys["left arrow"])
        return Controls(throttle=float(up), brake=float(down),
                        steer=float(steer), handbrake=bool(held_keys["space"]))

    def _cooldown_controls(self) -> Controls:
        """The autopilot drives the car from the flag onwards, and keeps going.

        A slack throttle that decayed to nothing was the first attempt, and it
        parks the car in the middle of the circuit a quarter of a lap later --
        which is the same "stopped dead" this was meant to fix, just delayed.
        A real in-lap is a lap: the car carries on round at a cooled pace until
        the session is left.
        """
        if self._cooldown is None:
            return Controls()
        c = self._cooldown.controls(self.vehicle)
        return Controls(throttle=c.throttle * config.COOLDOWN_PACE,
                        brake=c.brake, steer=c.steer, handbrake=False)

    def destroy(self, keep_world: bool = True):
        """Tear the session down: the cars, the HUD, the sound.

        The circuit is hidden and kept for the next session on it, unless
        ``keep_world`` is False. Ursina has no scene-clearing call, so anything
        created here has to be given back by hand -- and anything missed stays
        in the scene graph, invisible but still drawn.
        """
        import gc

        from .ui import destroy_tree

        gc.unfreeze()
        self.sound.stop()
        self.impacts.stop()
        if self.field is not None:
            self.field.close()
            self.field = None
        if self.gp is not None:
            self.gp.destroy()
            self.gp = None
        if getattr(self, "_grid_paint", None) is not None:
            destroy_tree(self._grid_paint)
            self._grid_paint = None
        if self.ghost is not None:
            self.ghost.destroy()
            self.ghost = None
        self.hud.destroy()
        self.fx.destroy()
        destroy_tree(self.car)
        self.world.forget()
        if keep_world:
            self.world.hide()
        else:
            self.world.destroy()

    def on_key(self, key: str):
        if key == "f":
            on = self.freecam.toggle()
            # The car is hidden in the bonnet view; flying away from it with
            # that still in force would leave an invisible car on the track.
            self.car.hull.enabled = (
                on or config.CAM_MODES[self.cam_idx] != "hood")
            if not on:
                self._snap_camera()
            return
        if self.freecam.enabled and self.freecam.on_key(key):
            return
        if key == "h":
            self.hud_hidden = not self.hud_hidden
            if self.state != INTRO:
                self.hud.root.enabled = not self.hud_hidden
            return
        if self.state == PAUSED and key == "p":
            self.pause_card_hidden = not self.pause_card_hidden
            return
        if self.state == PAUSED and self._pause_key(key):
            return
        if key == "c":
            self.cam_idx = (self.cam_idx + 1) % len(config.CAM_MODES)
            # The bonnet camera sits inside the car, which would clip messily
            # against the near plane, so hide the car there -- what every game
            # does for an in-car view.
            self.car.hull.enabled = config.CAM_MODES[self.cam_idx] != "hood"
            self._snap_camera()
        elif key == "tab":
            # The tower's right-hand column: interval to the car ahead, or
            # gap to the leader.
            self._gap_mode = "leader" if self._gap_mode == "interval" else "interval"
            self._tower_t = -1.0
        elif key == "x":
            # Look back: the camera turns round on the car it is following.
            self._look_back = not self._look_back
            self._snap_camera()
        elif key == "g" and self.spectate and self.gp is None:
            pass                              # one car to watch: nothing to cycle
        elif key == "g" and self.gp is not None:
            # Watch the car ahead, then the one ahead of that, and on round
            # the running order until it comes back to you; SHIFT+G steps
            # back the way it came (see _next_watch). Its onboard is the
            # chase, as for the ghost.
            back = any(held_keys[k] for k in ("shift", "left shift",
                                              "right shift"))
            self._watch_idx = self._next_watch(back)
            self._tower_t = -1.0
            if (self._watch_idx is not None
                    and config.CAM_MODES[self.cam_idx] in ("hood", "onboard")):
                self.cam_idx = config.CAM_MODES.index("chase")
            self.car.hull.enabled = (
                self._watch_idx is not None
                or config.CAM_MODES[self.cam_idx] != "hood")
            self._snap_camera()
        elif key == "g" and self.ghost is not None:
            self._watch_ghost = not self._watch_ghost
            # The two on-car views follow a car from inside or on top of it,
            # which for the ghost means inside a translucent shell -- drop to
            # the far chase, and put the player's own car back on screen so it
            # can be seen from the new angle.
            if (self._watch_ghost
                    and config.CAM_MODES[self.cam_idx] in ("hood", "onboard")):
                self.cam_idx = config.CAM_MODES.index("chase")
            self.car.hull.enabled = (
                self._watch_ghost
                or config.CAM_MODES[self.cam_idx] != "hood")
            self._snap_camera()
        elif key == "r" and self.spectate:
            pass                              # no car of ours to reset
        elif key == "r":
            # A recovery is a teleport; whatever lap it happens on is no lap.
            if self.quali and self.lap_num >= 1:
                self.lap_invalid = True
            self._reset_to_track()
            if self.field is not None:
                # ...and in a race, one that passes through the other cars
                # until it is clear of them, so it cannot land on anyone.
                self.field.event("reset")
        elif key == "m":
            self.muted = not self.muted
        elif key == "t":
            # One switch for all the driver aids: at an exhibition nobody wants
            # to reason about TC vs ABS vs steering assist separately.
            v = self.vehicle
            on = not v.traction_control
            v.traction_control = v.abs_enabled = v.steer_assist = on
        elif self.state == INTRO:
            # Any key at all skips the film. Somebody on their fifth lap of the
            # evening should not have to remember which one.
            self.intro.skip()
        elif key == "escape":
            if self.state == PAUSED:
                self._resume()
            elif self.state == FINISHED:
                self._leave()
            else:
                # Pause rather than quit: ESC mid-race used to dump you back to
                # the menu with no way to say "I didn't mean that".
                self._resume_state = self.state
                self.state = PAUSED
                self.vehicle.frozen = True
                self._pause_sel = 0
                self.pause_card_hidden = False
        elif key == "enter" and self.state == FINISHED:
            self._leave()

    def _pause_key(self, key: str) -> bool:
        """W/S and ENTER on the pause card. True if the key was used."""
        if self.pause_card_hidden:
            # Nothing to pick from a card that is not on screen: an ENTER
            # here must not quit a race the player cannot see the menu of.
            return key in ("down arrow", "s", "up arrow", "w", "enter",
                           "down arrow hold", "s hold", "up arrow hold",
                           "w hold")
        items = self.hud.pause_items
        if key in ("down arrow", "s", "down arrow hold", "s hold"):
            self._pause_sel = (self._pause_sel + 1) % len(items)
        elif key in ("up arrow", "w", "up arrow hold", "w hold"):
            self._pause_sel = (self._pause_sel - 1) % len(items)
        elif key == "enter":
            action = items[self._pause_sel][0]
            if action == "resume":
                self._resume()
            elif action == "restart" and self.on_restart is not None:
                self.on_restart()
            elif action == "exit":
                self._leave()
        else:
            return False
        return True

    def _resume(self):
        self.state = self._resume_state
        self.vehicle.frozen = self.state == COUNTDOWN

    def _leave(self):
        if self.on_exit is not None:
            self.on_exit()

    def _reset_to_track(self):
        i = self.surface.hint
        pos = self.track.center[i].copy()
        yaw = math.atan2(self.track.tangent[i, 0], self.track.tangent[i, 1])
        self.vehicle.place(pos, yaw)
        self._snap_camera()

    def _start_lights(self) -> int:
        """Lamp count 0-5 for the gantry, or -1 when it is not on screen.

        Hidden until the gantry drops in, then 0 while it drops, then a lamp
        per fifth of ``START_LIGHTS_BUILD``, a steady five through the pause,
        then 0 (all dark -- the go signal) while the gantry rides back up.
        """
        t = self.start_t
        if t < self._T_drop0 or t >= self._T_rise1:
            return -1
        if t < self._T_drop1 or t >= self._T_out:
            return 0
        if t >= self._T_build1:
            return 5
        return min(5, 1 + int((t - self._T_drop1)
                              / (config.START_LIGHTS_BUILD / 5.0)))

    def _gantry_dy(self) -> float:
        """Vertical offset on the gantry's rest position: it eases down from
        above, holds, then whips up and away once the lights are out."""
        t = self.start_t
        if t < self._T_drop1:
            f = max(0.0, (t - self._T_drop0) / config.START_GANTRY_DROP)
            return 0.80 * (1.0 - f) * (1.0 - f)          # ease-out, dropping in
        if t < self._T_out:
            return 0.0
        f = min(1.0, (t - self._T_out) / config.START_GANTRY_RISE)
        return 0.95 * (f * f)                            # ease-in, lifting away

    def _follow_field_shadow(self):
        """The other cars' shadow map: round the car on camera, pushed ahead
        along the view."""
        if self.gp is None:
            return
        followed = (self.gp.cars[self._watch_idx]
                    if self._watch_idx is not None and self._spectating()
                    else self.car)
        f = camera.forward
        self.light.field_shadow.follow(followed.world_position, (f.x, f.z))

    # -- main loop ------------------------------------------------
    def update(self):
        # The loading card built the HUD dark; bring it up now, with the lights.
        if self._reveal_hud:
            self._reveal_hud = False
            self.hud.root.enabled = not self.hud_hidden
        dt = min(time.dt, 0.05)
        if self.state == INTRO:
            # Nothing else runs: no clock, no physics, no engine. The HUD is
            # hidden rather than dimmed -- a lap counter over an establishing
            # shot of a circuit nobody has driven yet is furniture.
            self.hud.root.enabled = False
            self.intro.update(dt)
            # The cars stand on the grid already: their shadow cameras go
            # there too, not to the world origin (the start line) they were
            # built at.
            self.light.follow(self.car)
            self._follow_field_shadow()
            self.sound.update(0.0, config.MAX_SPEED, 0.0, dt, muted=True)
            if self.intro.done:
                self.state = COUNTDOWN
                self._resume_state = COUNTDOWN
                self.hud.root.enabled = not self.hud_hidden
                self._snap_camera()
            return
        if not self._cam_warm:
            self._cam_warm = True
            self._snap_camera()
        if self.state == PAUSED:
            # Everything stops: no physics, no clock, no engine note. The HUD
            # still draws, so the frozen frame stays behind the pause card.
            self.sound.update(0.0, config.MAX_SPEED, 0.0, dt, muted=True)
            self.impacts.scrape(0.0, False, dt)
            self._draw_hud(self.surface.hint)
            return
        # The start clock runs through the countdown and a little past it, so
        # the gantry finishes riding up while the race is already on.
        if self.start_t < self._T_rise1 + 0.1:
            self.start_t += dt
        # The lamps on the gantry follow the ones on the HUD exactly: column k
        # comes on once the count has reached k + 1, and all five drop together
        # at lights-out (_start_lights() back to 0 -- the go signal).
        n_lit = max(self._start_lights(), 0)
        for k, e in enumerate(self._lit_lamps):
            want = k < n_lit
            if e.enabled != want:
                e.enabled = want
        if self.state == COUNTDOWN:
            if self.start_t >= self._T_out:          # lights out -> go
                self.state = RACING
                self.vehicle.frozen = self.spectate
                if self.ghost is not None:
                    self.ghost.start()
                if self.field is not None:
                    self.field.event("go")
                self.session_time = 0.0
                self.lap_start = 0.0

        ctl = self.read_controls()
        self._ctl = ctl
        scraped = False
        self._accum = min(self._accum + dt, 0.1)
        while self._accum >= DT:
            self._ease_kick()
            was_on_wall = self.vehicle.hit_wall
            v0 = self.vehicle.vel
            self.vehicle.step(ctl, DT, self.surface)
            if self.vehicle.hit_wall:
                scraped = True
                if not was_on_wall:
                    # Into the barrier (or a gantry leg): as loud as the
                    # speed it took off the car.
                    self.impacts.hit(float(np.hypot(*(self.vehicle.vel - v0))))
            if self.ghost is not None:
                self.ghost.step(DT)
            self._accum -= DT
            if self.state != COUNTDOWN:
                # The session clock runs on physics steps, the clock the AI's
                # times are counted on, and the line and the track limits are
                # judged every step: a race decided by hundredths cannot be
                # timed to whichever frame happened to notice the crossing.
                self.session_time += DT
                self._tick(DT)
        # how far past the last physics step this frame lands
        self._alpha = self._accum / DT
        if self.field is not None:
            self._field_frame(dt)
        if (self.spectate and self.quali and self.ghost is not None
                and self.state == RACING and not self.ghost.visible):
            # The pole lap again, from the line, as soon as it is done.
            self.ghost.launch()

        i, _ = self.surface.progress(self.vehicle.pos)
        # The replay ghost can vanish while it is being watched; the camera
        # goes back to the player in one cut rather than a swoop across the
        # circuit.
        spect = self._spectating()
        if spect != self._last_spect:
            self._last_spect = spect
            self._snap_camera()

        self.car.sync(braking=ctl.brake > 0.05, dt=dt, alpha=self._alpha)
        self.fx.update(dt, self.vehicle, self.car, ctl, others=self._fx_others())
        if self.ghost is not None:
            self.ghost.sync(dt, self._alpha)
        if self._watch_idx is not None and not self._spectating():
            self._watch_idx = None
            self._watch_hist.clear()
        # After car.sync: the shadow camera sits where the car is drawn.
        self.light.follow(self.car)
        # The free camera drives `camera` itself; the chase rig must not fight
        # it for the same transform on the same frame.
        if self.freecam.enabled:
            self.freecam.update(dt)
        else:
            self._update_camera(dt)
        self._follow_field_shadow()
        # Spectating means watching a car, and a car you are behind sounds
        # like itself, not like the one you left on the other side of the
        # circuit.
        heard = self._subject()
        if not self._spectating():
            throttle = ctl.throttle
        elif self._watch_idx is not None:
            # A field car's throttle is not in the snapshot; its acceleration
            # says enough for the engine note.
            throttle = 1.0 if heard.long_accel > -2.0 else 0.0
        else:
            throttle = self.ghost.controls.throttle
        self.impacts.muted = self.muted
        self.impacts.scrape(self.vehicle.speed, scraped, dt)
        self.sound.update(heard.speed, config.MAX_SPEED,
                          throttle, dt, muted=self.muted)
        post.update(min(1.0, heard.speed / config.MAX_SPEED), dt)
        self._draw_hud(i)

    def _tick(self, dt: float):
        """Timing and the rules, once per physics step once the session is on."""
        v = self.vehicle
        i, _ = self.surface.progress(v.pos)
        off = not v.on_track
        if self.quali:
            # All four wheels past the white line, for a single step, and the
            # lap is gone. Checked before the line too, so an off on the step
            # that crosses it is charged to the lap it ends.
            if off and self.state == RACING and self.lap_num >= 1:
                self.lap_invalid = True
            self._lap_logic(i)
            return

        self._lap_logic(i)
        if self.field is not None:
            # Track limits, contact and the flag are the field's (the
            # stewards there see every car, the player's included).
            return
        if self.finish_race_t is None:
            self.limits.update(dt, i, v.pos, off, self.session_time, v.speed)
            self._legacy_rc()
        g = self.ghost
        if g is None or g.vehicle.frozen or g.finish_t is not None:
            return
        self.ai_limits.update(dt, g._last_i, g.vehicle.pos,
                              not g.vehicle.on_track, self.session_time,
                              g.vehicle.speed)
        if g.lap_num >= self._ai_finish_lap:
            g.finish_t = g.race_t
            g.finish_laps = g.lap_num - 1
            self.ai_limits.settle(self.session_time)
            if self.finish_race_t is None:
                # The flag is out: the player finishes next time over the
                # line, however many laps down.
                self._you_finish_lap = min(self._you_finish_lap,
                                           self.lap_num + 1)
        self._legacy_rc()

    def _lap_logic(self, i: int):
        n = self.track.count
        if self.state != RACING:
            return
        if 0.4 * n <= i <= 0.6 * n:
            self.armed = True
        if self.armed and i < 0.1 * n:
            fwd = np.array([math.sin(self.vehicle.yaw), math.cos(self.vehicle.yaw)])
            if np.dot(self.vehicle.vel, fwd) > 0:
                self._complete_lap()
                self.armed = False
        sec = self.track.sector_of(i)
        if sec != self.sector:
            # Only a forward crossing into the next sector closes the one
            # before it; the lap crossing (3 -> 1) is handled with the lap, and
            # a car reversing over a board records nothing.
            if sec == self.sector + 1 and self.lap_num >= 1:
                self._close_sector()
            self.sector = sec

    def _close_sector(self):
        t = self.session_time - self.sector_start
        k = self.sector
        self.sectors[k] = t
        # A deleted lap sets no bests, sector bests included.
        if not (self.quali and self.lap_invalid):
            b = self.best_sectors[k]
            self.best_sectors[k] = t if b is None else min(b, t)
        self.sector_start = self.session_time

    def _start_flying_lap(self):
        """Qualifying: a new lap begins at the line, and so does the ghost's."""
        self.lap_invalid = not self.vehicle.on_track
        if self.ghost is not None:
            self.ghost.launch()

    def _complete_lap(self):
        if self.lap_num == 0:
            # End of the out lap. Nothing to record: this lap started from a
            # standing start, and timing it would put the launch in the
            # results.
            self.lap_num = 1
            self.lap_start = self.session_time
            self.sector_start = self.session_time
            self.sectors = [None, None, None]
            self._start_flying_lap()
            return
        t = self.session_time - self.lap_start
        self.last_t = t
        self.last_invalid = self.quali and self.lap_invalid
        if not self.last_invalid:
            self.best_t = t if self.best_t is None else min(self.best_t, t)
            if self.quali:
                self._quali_record(self.best_t)
        if self.sector == 2:
            self._close_sector()
        self.sectors = [None, None, None]
        self.sector_start = self.session_time
        self.lap_num += 1
        self.lap_start = self.session_time
        if self.quali:
            if self.lap_num > self.total_laps:
                self._finish_quali()
            else:
                self._start_flying_lap()
            return
        if self.field is not None:
            return                  # the field says when the flag is out
        if self.lap_num >= self._you_finish_lap:
            self.state = FINISHED
            self.finish_t = self.session_time
            self.finish_race_t = self.session_time
            self.finish_laps = self.lap_num - 1
            self.limits.settle(self.session_time)
            self._legacy_rc()
            g = self.ghost
            if g is not None and g.finish_t is None:
                self._ai_finish_lap = min(self._ai_finish_lap, g.lap_num + 1)
            # Not frozen. The result card comes up over a car that is still
            # moving, which is what the end of a race looks like; the
            # autopilot in _cooldown_controls does the driving from here.
            from .autopilot import Autopilot
            self._cooldown = Autopilot(self.track, self.surface)

    def _finish_quali(self):
        """The last timed lap is in: the chequered flag, the qualifying
        result over the in-lap, and the autopilot driving it."""
        self.state = FINISHED
        self.finish_t = self.session_time
        from .autopilot import Autopilot
        self._cooldown = Autopilot(self.track, self.surface)

    def _quali_record(self, best: float):
        """Keep the player's qualifying lap for a grand prix here at this
        level: the grid is set by it. Stored in the units the AI's grid
        times are estimated in (the tower shows them scaled onto the pole
        lap actually driven, by quali_scale), so the two compare."""
        key = (self.track.name, self.level)
        est = best / max(self.quali_scale, 1e-6)
        old = SESSION["quali"].get(key)
        if old is None or est < old:
            SESSION["quali"][key] = est

    def _spectating(self) -> bool:
        if self._watch_idx is not None:
            return self.gp is not None and self._watch_idx in self.gp.proxies
        return (self._watch_ghost and self.ghost is not None
                and self.ghost.visible)

    def _draw_hud(self, i: int):
        if self.hud.stale():
            # The window changed shape -- fullscreen, most likely. The HUD
            # anchors to the screen edges when it is built, so it is rebuilt.
            self.hud.destroy()
            self.hud = HUD(self.track, self.total_laps, self.mode,
                           rows=self._hud_rows)
            self.hud.root.enabled = not self.hud_hidden
            self._tower_t = -1.0
        # While spectating, the readouts follow the watched car -- its speed,
        # its pedals -- so the trace on screen matches the car on screen. The
        # lap clock and the gap stay the player's: those are what the player
        # is racing.
        spectating = self._spectating()
        watch = self._watch_idx if spectating else None
        g = self.ghost
        if watch is not None:
            v = self.gp.proxies[watch]
            braking = bool(self._snap["rows"][watch][self._C["braking"]]) \
                if self._snap is not None else False
            ctl = Controls(throttle=0.0 if braking else
                           (1.0 if v.long_accel > -2.0 else 0.0),
                           brake=1.0 if braking else 0.0)
        elif spectating:
            v, ctl = g.vehicle, g.controls
        else:
            v, ctl = self.vehicle, self._ctl

        racing = self.state == RACING
        flag, style, head = self._flag(i, racing, spectating)
        cur_t = None if self.state == COUNTDOWN else self.session_time - self.lap_start
        # Qualifying's live delta to the ghost lap. The recording covers the
        # whole lap, so it reads even after the ghost has gone.
        delta = None
        if (self.quali and g is not None and not spectating and racing
                and self.lap_num >= 1):
            delta = g.delta(i, cur_t)
        # Lap number and lap times follow a watched ghost as well: its lap
        # is not the player's. A field car's are the player's own (the
        # tower carries the rest).
        ghost_watch = spectating and watch is None
        show_lap = g.lap_num if ghost_watch else self.lap_num
        show_cur = (g.lap_time if ghost_watch and self.state != COUNTDOWN
                    else cur_t)
        show_last = g.last_t if ghost_watch else self.last_t
        show_best = g.best_t if ghost_watch else self.best_t
        if self.spectate and watch is not None and self._snap is not None:
            # Watching the field: the chyron is the watched car's lap.
            r = self._snap["rows"][watch]
            C = self._C
            show_lap = int(r[C["laps"]]) + 1
            show_cur = (None if self.state == COUNTDOWN
                        else self.session_time - float(r[C["lap_start"]]))
            show_last = float(r[C["last"]]) if np.isfinite(r[C["last"]]) else None
            show_best = float(r[C["best"]]) if np.isfinite(r[C["best"]]) else None

        if self.field is not None:
            standings = self._field_tower()
            results = self._field_results() if self.state == FINISHED else None
        elif self.quali:
            standings = self._quali_standings()
            results = None
        else:
            standings = self._standings(i, cur_t)
            results = None
        me_name, me_col, me_pos = "PLAYER", TEAM_YOU, None
        spect_label = None
        if watch is not None:
            info = self.car_info[watch]
            me_name = info["name"].upper()
            me_col = _col4(info["color"])
            me_pos = next((e["pos"] for e in standings if e.get("key") == watch),
                          None)
            spect_label = f"ONBOARD  ·  {info['tla']}  ·  {info['team'].upper()}"
        elif spectating:
            me_name, me_col = ("GHOST LAP" if self.quali else "AI DRIVER"), TEAM_AI
            spect_label = "ONBOARD  ·  " + ("GHOST LAP" if self.quali else "AI")
        if self._look_back and not self.freecam.enabled:
            spect_label = (spect_label + "  ·  REAR VIEW") if spect_label \
                else "REAR VIEW"
        if self.field is not None and watch is None:
            me_pos = next((e["pos"] for e in standings if e.get("player")), None)
        target_label = target_t = None
        if self.quali and self.quali_field and not spectating:
            # The time to beat is pole: the ghost lap.
            target_t = g.best_t if g is not None else self.quali_field[0][3]
            target_label = f"POLE  ·  {self.quali_field[0][0]}"
        elif self.quali and g is not None and not spectating:
            target_t, target_label = g.best_t, "AI  ·  TARGET"

        zones = (self._snap["yellow"] if self.field is not None
                 and self._snap is not None else ())
        dots = None
        if self.gp is not None:
            dots = [(p.pos, self._dot_col[k]) for k, p in self.gp.proxies.items()]
        self.hud.update(
            speed_kmh=v.speed * 3.6,
            speed_frac=min(1.0, v.speed / config.MAX_SPEED),
            lap=show_lap,
            cur_t=show_cur if self.state != FINISHED else show_last,
            last_t=show_last,
            best_t=show_best,
            session_t=self.session_time,
            sectors=self._sector_lights(spectating and watch is None),
            standings=standings,
            lights=self._start_lights(),
            gantry_dy=self._gantry_dy(),
            flag=flag,
            flag_style=style,
            flag_head=head,
            delta=delta,
            cur_invalid=(self.quali and not spectating and self.lap_invalid
                         and self.lap_num >= 1),
            last_invalid=self.quali and not spectating and self.last_invalid,
            spectating=spectating,
            # Always the player's dot for the player and the others' for the
            # others. Feeding the watched car in as `car_xz` swapped the
            # colours over the moment you pressed G.
            car_xz=self.vehicle.pos,
            ghost_xz=None if g is None or not g.visible else g.vehicle.pos,
            field_dots=dots,
            throttle=ctl.throttle,
            brake=ctl.brake,
            steer=v.steer_input,
            slip=abs(v.slip_angle),
            tc_cut=v.tc_cut,
            esc_cut=v.esc_cut,
            tc_off=not v.traction_control,
            dt=min(time.dt, 0.05),
            finished=self.state == FINISHED,
            paused=self.state == PAUSED and not self.pause_card_hidden,
            pause_sel=self._pause_sel,
            strip=self._strip_state(),
            gap_mode=self._gap_mode,
            me_name=me_name, me_col=me_col, me_pos=me_pos,
            target_label=target_label, target_t=target_t,
            spect_label=spect_label,
            results=results,
            yellow_zones=zones,
            yellow_here=self.state == RACING and self._yellow_here(),
        )

    def _flag(self, i: int, racing: bool, spectating: bool):
        """(text, style, heading) for the race-control box."""
        # Session seconds since the last frame: a message's clock runs only
        # on the frames it is actually on screen (_rc_spend).
        now = self.session_time
        self._rc_dt = max(0.0, now - self._rc_clock)
        self._rc_clock = now
        if (spectating and not self.spectate) or not racing:
            return "", "warn", "RACE CONTROL"
        v = self.vehicle
        if self.spectate:
            pass                    # no car of ours: the stewards' news only
        elif np.dot(v.vel, self.track.tangent[i]) < -3:
            return "WRONG WAY", "red", "RACE CONTROL"
        elif not v.on_track:
            return "OFF TRACK", "warn", "TRACK LIMITS"
        if self.quali:
            if self.lap_invalid and self.lap_num >= 1:
                return "LAP INVALID", "red", "TRACK LIMITS"
            return "", "warn", "RACE CONTROL"
        # The stewards: the newest message about the player first, else the
        # newest about anyone.
        cur = self._rc_current()
        yellow = self._yellow_here()
        if cur is not None and (cur[0] or not yellow):
            # The player's own news first, then the yellow flag they are
            # in, then everyone else's.
            self._rc_spend(True)
            more = len(self._rc_queue)
            head = "RACE CONTROL" + (f"  ·  {more} MORE" if more else "")
            return cur[1], cur[2], head
        self._rc_spend(False)
        if yellow:
            return (f"YELLOW FLAG  ·  KEEP BELOW {config.YELLOW_SPEED_KMH:.0f} KM/H"
                    "  ·  NO OVERTAKES", "warn", "RACE CONTROL")
        return "", "warn", "RACE CONTROL"

    def _yellow_here(self) -> bool:
        snap = self._snap
        if snap is None or not snap["yellow"]:
            return False
        car = (self.field.player if self.field.player is not None
               else self._watch_idx)
        if car is None:
            return False
        s = float(snap["rows"][car][self._C["s"]])
        L = self.track.length
        return any((s - a) % L <= (b - a) % L for a, b in snap["yellow"])

    def _strip_state(self) -> str:
        if self.quali:
            return {RACING: "green", FINISHED: "chequered"}.get(self.state, "idle")
        if self.field is not None and self._snap is not None:
            if self._snap["leader_done"] is not None:
                return "chequered"
            if self._snap["yellow"]:
                return "yellow"
        elif self.state == FINISHED:
            return "chequered"
        return "green" if self.state in (RACING, FINISHED) else "idle"

    def _legacy_rc(self):
        """Settled excursions to the stewards, in the single-ghost race."""
        for car, lim in ((0, self.limits), (1, self.ai_limits)):
            while lim.pending:
                self.rc.excursion(car, lim.pending.pop(0))
        new = self.rc.messages[self._rc_seen:]
        self._rc_seen = len(self.rc.messages)
        for m in new:
            if m.car != 0 and m.kind not in ("pen", "flag"):
                continue
            who = "" if m.car == 0 else "AI  ·  "
            self._rc_push(m.car == 0, who + m.text, m.kind)

    def _next_watch(self, back: bool) -> int | None:
        """G: the car one place ahead of the one being watched (of the
        player, to begin with), skipping the player, the leader wrapping to
        the last car -- and home once that comes round to a car already
        seen, which a full lap of the order does. SHIFT+G retraces the
        cars seen, back to the player.

        Kept as a history rather than counted from the order each press,
        so a pass between presses cannot make the cycle land on home (or
        skip a car) halfway round."""
        hist = self._watch_hist
        snap = self._snap
        if self.field.player is None:
            # Spectating: round and round the order, there is no home.
            if snap is None:
                return self._watch_idx
            rows = snap["rows"]
            order = sorted(range(len(rows)), key=lambda k: rows[k][self._C["pos"]])
            cur = self._watch_idx if self._watch_idx in order else order[0]
            j = order.index(cur)
            # Down the order (P1, P2, P3 ...); SHIFT+G back up it.
            return order[(j + (-1 if back else 1)) % len(order)]
        if back:
            if hist:
                hist.pop()
            return hist[-1] if hist else None
        if snap is None:
            return None
        rows = snap["rows"]
        P = self._C["pos"]
        me = self.field.player
        order = sorted(range(len(rows)), key=lambda k: rows[k][P])
        cur = hist[-1] if hist else me
        j = order.index(cur)
        nxt = order[(j - 1) % len(order)]
        if nxt == me:
            nxt = order[(j - 2) % len(order)]
        if nxt == me or nxt in hist:
            hist.clear()
            return None
        hist.append(nxt)
        return nxt

    def _sector_lights(self, spectating: bool = False):
        """(time, status) per sector for the lap in progress.

        Purple beats everyone's best for that sector, the AI's included; green
        beats only your own; yellow is slower than your own best. The sector
        being driven is "live"; the ones still to come are off.
        """
        g = self.ghost
        ai = g.best_sectors if g is not None else [None] * 3
        # Spectating shows the watched car's sectors, compared against the
        # other car's bests -- the same relationship, both ways round.
        own = g.sectors if spectating else self.sectors
        own_best = g.best_sectors if spectating else self.best_sectors
        rival = self.best_sectors if spectating else ai
        live_k = g.sector if spectating else self.sector
        lap_ok = (g.lap_num if spectating else self.lap_num) >= 1
        out = []
        timed = self.state != COUNTDOWN and lap_ok
        for k in range(3):
            t = own[k]
            if t is not None:
                mine = own_best[k]
                pb = mine is None or t <= mine + 1e-6
                overall = pb and (rival[k] is None or t <= rival[k] + 1e-6)
                status = "purple" if overall else ("green" if pb else "yellow")
            elif timed and k == live_k:
                status = "live"
            else:
                status = "off"
            out.append((t, status))
        return out

    # -- the tower ------------------------------------------------------
    @staticmethod
    def _gap_text(g: float) -> str:
        if g < 60.0:
            return f"+{g:.3f}"
        m, s = divmod(g, 60.0)
        return f"+{int(m)}:{s:06.3f}"

    def _field_tower(self):
        """Tower rows from the field: the running order, intervals (or gaps
        to the leader), penalties, DRS, cars out of the race for now, and
        places gained since the start. Rebuilt ten times a second -- timing
        loops, not a stopwatch -- and the rows slide between."""
        import time as _time
        now = _time.perf_counter()
        if self._tower_cache is not None and now - self._tower_t < 0.1:
            return self._tower_cache
        self._tower_t = now
        C = self._C
        snap = self._snap
        cl = self.field
        info = self.car_info
        L = self.track.length
        out = []
        if snap is None:
            for c in sorted(info.values(), key=lambda c: c["grid"]):
                out.append(dict(key=c["idx"], pos=c["grid"] + 1, tla=c["tla"],
                                col=TEAM_YOU if c["external"] else _col4(c["color"]),
                                player=c["external"], gap="", word=True,
                                name=c["name"], team=c["team"]))
            self._tower_cache = out
            return out
        rows = snap["rows"]
        order = sorted(range(len(rows)), key=lambda k: rows[k][C["pos"]])
        bests = rows[:, C["best"]]
        fin = np.isfinite(bests)
        session_best = float(bests[fin].min()) if fin.any() else None
        started = snap["started"]
        lead = rows[order[0]]
        for p, k in enumerate(order):
            r = rows[k]
            c = info[k]
            player = k == cl.player
            gap, word = "", False
            if not started:
                gap, word = "", True
            elif np.isfinite(r[C["finish_t"]]) and np.isfinite(lead[C["finish_t"]]):
                if p == 0:
                    gap = lap_time(r[C["total"]])
                elif lead[C["laps"]] - r[C["laps"]] >= 1:
                    d = int(lead[C["laps"]] - r[C["laps"]])
                    gap, word = f"+{d} LAP" + ("S" if d > 1 else ""), True
                else:
                    gap = self._gap_text(r[C["total"]] - lead[C["total"]])
            elif p == 0:
                gap, word = "LEADER", True
            else:
                behind_m = lead[C["progress"]] - r[C["progress"]]
                if behind_m >= L and self._gap_mode == "leader":
                    d = int(behind_m // L)
                    gap, word = f"+{d} LAP" + ("S" if d > 1 else ""), True
                else:
                    g = r[C["gap"] if self._gap_mode == "leader" else C["interval"]]
                    gap = self._gap_text(g) if np.isfinite(g) else ""
            status = ("OUT" if r[C["recover"]] > 0.5 else
                      "STOP" if r[C["stricken"]] > 0.5 else
                      "DRS" if r[C["drs"]] > 0.5 else "")
            best = float(r[C["best"]]) if np.isfinite(r[C["best"]]) else None
            out.append(dict(
                key=k, pos=p + 1, tla=c["tla"], name=c["name"], team=c["team"],
                col=TEAM_YOU if player else _col4(c["color"]), player=player,
                gap=gap, word=word, pen=float(r[C["penalty"]]), status=status,
                purple=(best is not None and session_best is not None
                        and abs(best - session_best) < 1e-6),
                change=(c["grid"] + 1) - (p + 1) if started else 0,
                best=best, watched=k == self._watch_idx))
        self._tower_cache = out
        return out

    def _field_results(self):
        """The classification card: laps, then race time with penalties."""
        C = self._C
        snap = self._snap
        if snap is None:
            return []
        rows = snap["rows"]
        info = self.car_info
        order = sorted(range(len(rows)), key=lambda k: rows[k][C["pos"]])
        lead = rows[order[0]]
        bests = rows[:, C["best"]]
        fin = np.isfinite(bests)
        session_best = float(bests[fin].min()) if fin.any() else None
        out = []
        for p, k in enumerate(order):
            r = rows[k]
            c = info[k]
            done = np.isfinite(r[C["finish_t"]])
            if not done:
                res = "RUNNING"
            elif p == 0:
                res = lap_time(r[C["total"]])
            elif lead[C["laps"]] - r[C["laps"]] >= 1:
                d = int(lead[C["laps"]] - r[C["laps"]])
                res = f"+{d} LAP" + ("S" if d > 1 else "")
            else:
                res = self._gap_text(r[C["total"]] - lead[C["total"]])
            best = float(r[C["best"]]) if np.isfinite(r[C["best"]]) else None
            out.append(dict(pos=p + 1, tla=c["tla"], name=c["name"],
                            team=c["team"], player=c["external"],
                            col=TEAM_YOU if c["external"] else _col4(c["color"]),
                            best=best, result=res, pen=float(r[C["penalty"]]),
                            purple=(best is not None and session_best is not None
                                    and abs(best - session_best) < 1e-6)))
        return out

    def _standings(self, i: int, cur_t):
        """Tower rows against the single AI ghost, leader first. Position is
        by distance covered, the way a race is scored; the gap is the time
        between the two cars at the same point on the track.
        """
        # The result is the result. Both cars keep circulating after the
        # flag, so left live the gap went on counting.
        if self.state == FINISHED and self._final_rows is not None:
            return self._final_rows
        if self.state == FINISHED:
            return self._classification()

        L = self.track.length
        you = dict(key="you", tla="YOU", name="PLAYER", col=TEAM_YOU,
                   best=self.best_t, player=True, pen=self.rc.penalty(0),
                   dist=self.limits.progress or 0.0)
        rows = [you]
        g = self.ghost
        if g is not None:
            rows.append(dict(key="ai", tla="AI", name="AI DRIVER", col=TEAM_AI,
                             best=g.best_t, player=False,
                             pen=self.rc.penalty(1),
                             dist=self.ai_limits.progress or 0.0))
        # Before the start both cars have covered nothing, so sorting by
        # distance is a coin toss. The player starts on pole.
        if self.state != COUNTDOWN:
            rows.sort(key=lambda r: -r["dist"])
        self._mark_purple(rows)
        for p, r in enumerate(rows):
            if p == 0:
                r["gap"], r["word"] = "LEADER", True
                continue
            laps_down = int((rows[0]["dist"] - r["dist"]) // L)
            delta = (g.delta(i, cur_t)
                     if g is not None and self.state != COUNTDOWN else None)
            if laps_down >= 1:
                r["gap"] = f"+{laps_down} LAP" + ("S" if laps_down > 1 else "")
                r["word"] = True
            elif delta is not None:
                r["gap"] = f"+{abs(delta):.3f}"
            else:
                r["gap"] = ""
        return rows

    @staticmethod
    def _mark_purple(rows):
        bests = [r["best"] for r in rows if r["best"] is not None]
        session_best = min(bests) if bests else None
        for p, r in enumerate(rows):
            r["pos"] = p + 1
            r["purple"] = (r["best"] is not None and session_best is not None
                           and abs(r["best"] - session_best) < 1e-6)

    def _quali_standings(self):
        """Qualifying order: best valid lap, no time last -- against the
        AI's times at this level when the circuit has them."""
        me_team = teams.TEAMS[teams.PLAYER_SEAT[0]].name
        rows = [] if self.spectate else [
            dict(key="you", tla="YOU", name="PLAYER", col=TEAM_YOU,
                 best=self.best_t, player=True, team=me_team)]
        if self.quali_field:
            for k, (tla, name, col, t, team) in enumerate(self.quali_field):
                rows.append(dict(key=f"q{k}", tla=tla, name=name,
                                 col=_col4(col), best=t, player=False,
                                 team=team))
        elif self.ghost is not None:
            rows.append(dict(key="ai", tla="AI", name="GHOST LAP", col=TEAM_AI,
                             best=self.ghost.best_t, player=False))
        rows.sort(key=lambda r: (r["best"] is None, r["best"] or 0.0))
        self._mark_purple(rows)
        for p, r in enumerate(rows):
            if r["best"] is None:
                r["gap"], r["word"] = "NO TIME", True
            elif p == 0:
                r["gap"] = lap_time(r["best"])
            else:
                r["gap"] = f"+{r['best'] - rows[0]['best']:.3f}"
            # The result card: pole, then the gap to it.
            r["result"] = ("POLE" if p == 0 and r["best"] is not None
                           else r["gap"])
        return rows

    def _classification(self):
        """Grand prix result against the ghost: laps completed, then race
        time plus penalties. Live until both have taken the flag, frozen
        from then on."""
        def total(t, car):
            return None if t is None else t + self.rc.penalty(car)

        rows = [dict(key="you", tla="YOU", name="PLAYER", col=TEAM_YOU,
                     best=self.best_t, player=True, pen=self.rc.penalty(0),
                     done=self.finish_race_t is not None,
                     laps=self.finish_laps,
                     total=total(self.finish_race_t, 0),
                     dist=self.limits.progress or 0.0)]
        g = self.ghost
        if g is not None:
            rows.append(dict(key="ai", tla="AI", name="AI DRIVER", col=TEAM_AI,
                             best=g.best_t, player=False,
                             pen=self.rc.penalty(1),
                             done=g.finish_t is not None,
                             laps=getattr(g, "finish_laps", 0),
                             total=total(g.finish_t, 1),
                             dist=self.ai_limits.progress or 0.0))
        rows.sort(key=lambda r: (0, -r["laps"], r["total"]) if r["done"]
                  else (1, -r["dist"], 0.0))
        self._mark_purple(rows)
        lead = rows[0]
        for p, r in enumerate(rows):
            if not r["done"]:
                r["gap"], r["word"] = "RUNNING", True
            elif p == 0:
                r["gap"] = lap_time(r["total"])
            elif r["laps"] < lead["laps"]:
                d = lead["laps"] - r["laps"]
                r["gap"], r["word"] = f"+{d} LAP" + ("S" if d > 1 else ""), True
            else:
                r["gap"] = f"+{r['total'] - lead['total']:.3f}"
            r["result"] = r["gap"]
        if all(r["done"] for r in rows):
            self._final_rows = rows
        return rows


# ---------------------------------------------------------------------------
GAME: Game | None = None


MENU = None

#: The last circuit built, kept (hidden) while the menu is up so that going
#: back to it -- or restarting -- does not rebuild it. See world.py.
WORLD = None

#: The loading card, while one is up. It owns the update loop and swallows
#: input until its build has run and it has faded out.
TRANSITION = None


def update():
    global TRANSITION
    from .recorder import RECORDER
    RECORDER.tick()
    if TRANSITION is not None:
        TRANSITION.tick()
        if TRANSITION.done:
            TRANSITION.destroy()
            TRANSITION = None
        return
    if GAME is not None:
        GAME.update()


def input(key):  # noqa: A001  (ursina hook name)
    if key == "f11":
        # Everywhere, the menu and the loading card included.
        toggle_fullscreen()
        return
    if key == config.REC_KEY:
        from .recorder import RECORDER
        RECORDER.toggle()
        return
    # The menu and the race are never both up, so one hook can serve both.
    if TRANSITION is not None:
        return
    if MENU is not None:
        MENU.on_key(key)
    elif GAME is not None:
        GAME.on_key(key)


#: The window's place and size before going fullscreen, to go back to.
_WINDOWED = None


def monitor_at(x: float, y: float, monitors):
    """The monitor containing the point (x, y), or the nearest one."""
    for m in monitors:
        if m.x <= x < m.x + m.width and m.y <= y < m.y + m.height:
            return m

    def gap(m):
        dx = max(m.x - x, 0.0, x - (m.x + m.width))
        dy = max(m.y - y, 0.0, y - (m.y + m.height))
        return dx * dx + dy * dy
    return min(monitors, key=gap) if monitors else None


def toggle_fullscreen():
    """Borderless fullscreen on whichever monitor the window is on.

    Ursina's ``window.fullscreen`` always sizes the window to the *primary*
    monitor and moves it there, so on a second screen it either jumped to the
    first one or came out the wrong size. This picks the monitor under the
    window's centre instead, asking the OS each time so a screen plugged in
    after launch counts too.
    """
    import builtins

    from panda3d.core import WindowProperties

    global _WINDOWED
    win = builtins.base.win
    cur = win.get_properties()
    want = WindowProperties()
    if _WINDOWED is None:
        try:
            from screeninfo import get_monitors
            monitors = get_monitors()
        except Exception:
            monitors = list(window.monitors or [])
        ox, oy = cur.get_x_origin(), cur.get_y_origin()
        w, h = cur.get_x_size(), cur.get_y_size()
        mon = monitor_at(ox + w / 2, oy + h / 2, monitors)
        if mon is None:
            return
        _WINDOWED = (ox, oy, w, h, cur.get_undecorated())
        want.set_undecorated(True)
        want.set_origin(mon.x, mon.y)
        want.set_size(mon.width, mon.height)
    else:
        ox, oy, w, h, undecorated = _WINDOWED
        _WINDOWED = None
        want.set_undecorated(undecorated)
        want.set_origin(ox, oy)
        want.set_size(w, h)
    win.request_properties(want)


# Set once by main(); the menu and the race hand control back and forth and
# both need the same session options.
SESSION: dict = {"laps": config.TOTAL_LAPS, "mute": False, "track": None,
                 "mode": None, "level": teams.DEFAULT_LEVEL,
                 # (circuit, level) -> the player's best qualifying lap, in
                 # the field's estimate units (see Game._quali_record), which
                 # sets their grid slot in a grand prix there.
                 "quali": {},
                 # Where the player starts a grand prix: "quali" (their
                 # qualifying lap, if there is one) or a grid slot, 1 = pole.
                 "grid": "quali"}


def start_grid(track_name: str, level: int):
    """(player_time, player_grid) for the grand prix about to be built."""
    g = SESSION["grid"]
    if g == "quali":
        return SESSION["quali"].get((track_name, level)), None
    return None, int(g)


def _transition(title: str, build):
    """Put a loading card up and hand it the update loop until it is done."""
    global TRANSITION
    from .loading import Loading
    # Kill the outgoing screen's overlay right away -- a pause card's text sits
    # closer to the camera than most of the HUD and would otherwise read
    # through the loading card until the build tears the game down.
    if GAME is not None:
        GAME.hud.root.enabled = False
    TRANSITION = Loading(title, build)


def _build_race(track_name: str, laps: int, mute: bool, progress=None,
                mode: str | None = None, intro: bool = True,
                spectate: bool = False):
    """Tear the menu down and build the race. The heavy frame -- called behind
    the loading card by ``_start_race``, or straight away by the tools and the
    ``--track`` launch, which have no menu to transition from."""
    global GAME, MENU, WORLD
    if MENU is not None:
        MENU.destroy()
        MENU = None
    if WORLD is not None and WORLD.name != track_name:
        # One circuit in memory at a time.
        WORLD.destroy()
        WORLD = None
    mode = mode or SESSION["mode"] or GRAND_PRIX
    window.title = f"FORMULA-AI — {track_name}"
    SESSION["track"] = track_name
    SESSION["mode"] = mode
    SESSION["spectate"] = spectate
    level = SESSION["level"]
    GAME = Game(track_name, laps, mute=mute, on_exit=_back_to_menu,
                progress=progress, mode=mode, world=WORLD, intro=intro,
                on_restart=_restart_race, level=level,
                player_time=start_grid(track_name, level)[0],
                player_grid=start_grid(track_name, level)[1],
                spectate=spectate)
    WORLD = GAME.world


def _start_race(track_name: str, laps: int, mute: bool, mode: str,
                spectate: bool = False):
    """Menu -> race (or, *spectate*, the race to watch), through a loading
    card."""
    from .ui import caption
    _transition(caption(track_name)[0],
                lambda pump: _build_race(track_name, laps, mute, pump, mode,
                                         spectate=spectate))


def _restart_race():
    """Pause card -> the same grand prix from the grid, through a loading
    card. The circuit is kept, so this rebuilds only the session."""
    from .ui import caption
    track, mode = SESSION["track"], SESSION["mode"]
    laps, mute = GAME.total_laps, GAME.muted

    def build(pump):
        global GAME
        GAME.destroy()
        GAME = None
        _build_race(track, laps, mute, pump, mode, intro=False,
                    spectate=SESSION.get("spectate", False))

    _transition(caption(track)[0], build)


def _set_menu(menu):
    global MENU
    if MENU is not None:
        MENU.destroy()
    MENU = menu


def _show_modes():
    """The main menu: qualifying or grand prix."""
    from ursina import application

    from .menu import ModeMenu
    _set_menu(None)
    _set_menu(ModeMenu(on_pick=_pick_mode, on_quit=application.quit,
                       laps=SESSION["laps"], initial=SESSION["mode"],
                       level=SESSION["level"], on_level=_set_level))


def _set_level(level: int):
    SESSION["level"] = level


def _pick_mode(mode: str):
    SESSION["mode"] = mode
    _show_circuits()


def _show_circuits():
    """Circuit select for the chosen session; ESC goes back to the main menu."""
    from ursina import application

    from .menu import StartMenu, available_circuits
    _set_menu(None)
    _set_menu(StartMenu(
        available_circuits(),
        on_start=lambda n: _start_race(n, SESSION["laps"], SESSION["mute"],
                                       SESSION["mode"]),
        on_watch=lambda n: _start_race(n, SESSION["laps"], SESSION["mute"],
                                       SESSION["mode"], spectate=True),
        on_quit=application.quit,
        on_back=_show_modes,
        initial=SESSION["track"],
        mode=SESSION["mode"],
        laps=SESSION["laps"],
        level=SESSION["level"],
        quali=SESSION["quali"],
        grid=SESSION["grid"],
        on_grid=lambda g: SESSION.update(grid=g)))


def _build_menu(progress=None):
    """Tear the race down and put the menu back: the circuit list of the
    session just run, on the circuit just raced -- or the main menu, at
    launch."""
    global GAME

    if GAME is not None:
        GAME.destroy()
        GAME = None

    window.color = pal.rgb(21, 21, 30)
    window.title = "FORMULA-AI"
    if SESSION["mode"] is None:
        _show_modes()
    else:
        _show_circuits()


def _back_to_menu():
    """Race -> menu, through a loading card."""
    _transition("MAIN MENU", _build_menu)


def main(argv=None):
    p = argparse.ArgumentParser(description="FORMULA-AI racing prototype")
    p.add_argument("--track", default=None,
                   help="skip the menu and go straight to this circuit "
                        "(folder name under f1tenth_racetracks-main, e.g. Monza)")
    p.add_argument("--laps", type=int, default=config.TOTAL_LAPS)
    p.add_argument("--level", type=int, default=teams.DEFAULT_LEVEL,
                   choices=sorted(teams.DIFFICULTY),
                   help="AI difficulty, 1 (novice) to 6 (legend)")
    p.add_argument("--mode", choices=(QUALI, GRAND_PRIX), default=None,
                   help="skip the main menu into this session "
                        "(with --track, default gp)")
    p.add_argument("--fullscreen", action="store_true", default=config.FULLSCREEN)
    p.add_argument("--mute", action="store_true")
    p.add_argument("--selftest", type=float, default=0.0,
                   help="run for N seconds with an autopilot, then quit")
    args = p.parse_args(argv)

    from . import frameloop
    frameloop.prepare()
    app = Ursina(title="FORMULA-AI", size=config.WINDOW_SIZE,
                 fullscreen=args.fullscreen, vsync=True,
                 development_mode=False)
    frameloop.install(app)
    # One typeface for the whole program, set before any Text exists. Doing it
    # here rather than per widget means the HUD matches the menu without the
    # HUD having to know the menu exists.
    from ursina import Text as _Text

    from .menu import pick_font
    chosen_font = pick_font()
    if chosen_font:
        _Text.default_font = chosen_font

    # ursina looks up ``update`` / ``input`` on the __main__ module every frame.
    import __main__
    __main__.update = update
    __main__.input = input

    SESSION.update(laps=args.laps, mute=args.mute, mode=args.mode,
                   track=args.track or config.DEFAULT_TRACK, level=args.level)

    # --track (and --selftest, which needs a circuit up front) skips the menu,
    # so it also skips the loading card and builds the race straight away.
    if args.track or args.selftest > 0:
        _build_race(SESSION["track"], args.laps, args.mute)
        if args.selftest > 0:
            _install_selftest(args.selftest)
    else:
        from .menu import available_circuits
        if not available_circuits():
            raise SystemExit(f"no circuit data under {config.TRACK_DB}")
        _build_menu()

    app.run()


def _install_selftest(seconds: float):
    from ursina import application, invoke

    def _shot():
        try:
            from panda3d.core import Filename
            base.win.saveScreenshot(Filename.fromOsSpecific(  # noqa: F821
                str(config.ASSET_DIR.parent / "selftest.png")))
        except Exception as exc:  # pragma: no cover
            print("screenshot failed:", exc)

    from .autopilot import Autopilot
    pilot = Autopilot(GAME.track, GAME.surface)
    GAME.read_controls = lambda: pilot.controls(GAME.vehicle)
    GAME.muted = True
    invoke(_shot, delay=max(0.5, seconds - 0.6))
    invoke(application.quit, delay=seconds)


if __name__ == "__main__":
    main()
