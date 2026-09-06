"""FORMULA-AI racing prototype — human-driven, Phase A procedural assets, no AI yet."""
from __future__ import annotations

import argparse
import math
import random

import numpy as np
from ursina import Ursina, Vec3, camera, held_keys, time, window

from . import config
from . import palette as pal
from .car import Car, lerp_pose
from .freecam import FreeCam
from .ghost import Ghost
from .enginesound import EngineSound
from .hud import HUD
from .lighting import Sunset
from .scenery import build_scenery
from .surface import Surface
from .terrain import build_mountains
from .trackdata import load_track
from .trackmesh import TrackScene, line_markers
from .vehicle import Controls, Vehicle

DT = 1.0 / config.PHYSICS_HZ
COUNTDOWN, RACING, FINISHED, PAUSED = range(4)


class Game:
    def __init__(self, track_name: str, laps: int, mute: bool = False,
                 on_exit=None, progress=None):
        # Called between the heavy stages below so the loading card can draw a
        # frame and its bar keeps moving. A no-op when there is no card (the
        # tools and the --track launch build straight into a live window).
        step = progress if progress is not None else (lambda: None)
        # ``on_exit`` takes the session back to the menu. Owned here rather
        # than by the caller because the race is what knows when it is over.
        self.on_exit = on_exit
        # Sky, sun and shadows belong to the race, not to the app: leaving them
        # behind on the way back to the menu would paint over the menu's dark
        # ground. Scene fog is gone -- the shader does its own haze, so Panda's
        # would be a second, disagreeing one.
        window.color = pal.rgb(*[int(v * 255) for v in config.SKY_HORIZON])
        self.track = load_track(track_name)
        self.light = Sunset(self.track)

        step()
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

        self.vehicle = Vehicle()
        self.vehicle.frozen = True
        pos, yaw = self.track.start_pose()
        self.vehicle.place(pos, yaw)
        self.car = Car(self.vehicle, model=config.PLAYER_MODEL)
        step()

        # The opponent. Its own vehicle, its own surface, its own driver.
        self.ghost = Ghost(self.track) if config.GHOST_ENABLED else None
        step()

        # Everything in the world gets the sunset shader. The car gets a
        # sharper, stronger highlight than the scenery: bodywork is the one
        # surface here that is actually glossy.
        # The track surface receives shadows but must not cast them.
        self.light.apply(*self.scene.entities, casts=False)
        # The roadside never moves, so its shadows are drawn once into the
        # baked map rather than redrawn into the following one every frame.
        self.light.apply(*self.scenery, baked=True)
        # The range is kilometres away, well outside the shadow film, so it
        # only has to receive.
        self.light.apply(self.mountains, casts=False)
        self.light.apply(self.car, spec_strength=0.55, spec_power=64.0)
        # A ghost that threw a shadow would not be a ghost, and the shadow
        # would be the one part of it that looked solid.
        if self.ghost is not None:
            self.light.apply(self.ghost.car, spec_strength=0.55,
                             spec_power=64.0, casts=False)
        # After every mask is set, and only once: this is what makes the
        # roadside's shadows static geometry rather than per-frame work.
        step()
        self.light.bake()
        step()

        self.hud = HUD(self.track, laps)
        # Built now but kept dark: the loading card is still up, and the HUD
        # should arrive with the start lights, not flash in behind the fade.
        # The first ``update`` reveals it (unless something has cleared the
        # flag, e.g. bench_fps's --no-hud).
        self.hud.root.enabled = False
        self._reveal_hud = True
        self.total_laps = laps

        self.cam_idx = 0
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
        self._init_camera()

        self.sound = EngineSound(config.ASSET_DIR)
        self.muted = mute

        # session state
        self.state = COUNTDOWN
        self._resume_state = COUNTDOWN
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
        self.session_time = 0.0
        self.lap_start = 0.0
        # Lap 0 is the out lap. The HUD shows it as OUT and nothing is timed
        # until the car crosses the line for the first time.
        self.lap_num = 0 if config.OUT_LAP else 1
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

    # -- camera -----------------------------------------------------
    def _init_camera(self):
        camera.fov = config.CAM_FOV_BASE
        # A fresh race must not inherit the previous one's corner lean: camera
        # is a singleton and nothing else zeroes its roll.
        camera.rotation = Vec3(0, 0, 0)
        # See config.CLIP_NEAR: the near plane, not the far one, is what sets
        # depth precision. Ursina's default 0.1 could not separate a road
        # marking from the asphalt beyond ~200 m.
        camera.clip_plane_near = config.CLIP_NEAR
        camera.clip_plane_far = config.CLIP_FAR
        self._snap_camera()

    def _cam_offset(self):
        return {
            "chase": config.CAM_CHASE_OFFSET,
            "hood": config.CAM_HOOD_OFFSET,
            "far": config.CAM_FAR_OFFSET,
        }[config.CAM_MODES[self.cam_idx]]

    def _subject(self):
        """The vehicle the camera and HUD are following -- the player, or the
        AI while spectating. One accessor so every consumer agrees on it."""
        if self._watch_ghost and self.ghost is not None:
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
        return Vec3(px + wx, oy, pz + wz)

    def _cam_aim_point(self) -> Vec3:
        """Look ahead down the road, not at the car's nose."""
        v = self._subject()
        px, pz, yaw = self._render_pose()
        # lead further with speed so fast corners open up before you reach them
        lead = config.CAM_LOOKAHEAD * (0.6 + 0.9 * min(1.0, v.speed / config.MAX_SPEED))
        return Vec3(px + math.sin(yaw) * lead, 1.35, pz + math.cos(yaw) * lead)

    def _snap_camera(self):
        camera.position = self._cam_target_pos()
        self._cam_aim = self._cam_aim_point()
        camera.look_at(self._cam_aim)

    def _update_camera(self, dt: float):
        v = self._subject()
        mode = config.CAM_MODES[self.cam_idx]
        frac = min(1.0, v.speed / config.MAX_SPEED)

        # --- position: lag behind the target so acceleration is felt ----
        target = self._cam_target_pos()
        if mode == "hood":
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
        camera.fov += (want_fov - camera.fov) * min(1.0, 3.0 * dt)

        # --- shake, scaled by speed and roughened off-track -------------
        amp = config.CAM_SHAKE * frac
        if not v.on_track:
            amp *= config.CAM_SHAKE_OFFTRACK
        if amp > 1e-4:
            camera.position += Vec3(random.uniform(-amp, amp),
                                    random.uniform(-amp, amp) * 0.7,
                                    random.uniform(-amp, amp))

        # --- aim, also damped, so the view swings through corners -------
        want = self._cam_aim_point()
        t = min(1.0, config.CAM_AIM_LERP * dt)
        self._cam_aim = self._cam_aim + (want - self._cam_aim) * t
        camera.look_at(self._cam_aim)
        # lean into the corner a touch -- lerped, not snapped, and held flat
        # until the car is actually moving, so entering a race does not roll.
        target_lean = (0.0 if self.state == COUNTDOWN
                       else -math.degrees(v.yaw_rate) * config.CAM_LEAN * frac)
        camera.rotation_z += (target_lean - camera.rotation_z) * min(1.0, 6.0 * dt)

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

    def destroy(self):
        """Tear the whole race down so a new one can be built on another track.

        Ursina has no scene-clearing call, so anything created here has to be
        given back by hand -- and anything missed stays in the scene graph,
        invisible but still drawn. Every piece below is therefore held as an
        attribute for exactly this reason.
        """
        from ursina import destroy as _destroy

        self.sound.stop()
        if self.ghost is not None:
            self.ghost.destroy()
            self.ghost = None
        self.light.destroy()
        self.hud.destroy()
        _destroy(self.car)
        if self.mountains is not None:
            _destroy(self.mountains)
            self.mountains = None
        for e in self.markers:
            _destroy(e)
        self.markers = []
        for e in self.scenery:
            _destroy(e)
        for e in self.scene.entities:
            _destroy(e)
        self.scenery = []
        self.scene.entities = []

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
        if key == "c":
            self.cam_idx = (self.cam_idx + 1) % len(config.CAM_MODES)
            # The bonnet camera sits inside the car, which would clip messily
            # against the near plane, so hide the car there -- what every game
            # does for an in-car view.
            self.car.hull.enabled = config.CAM_MODES[self.cam_idx] != "hood"
            self._snap_camera()
        elif key == "g" and self.ghost is not None:
            self._watch_ghost = not self._watch_ghost
            # Hood view follows a car from inside it, which for the ghost means
            # inside a translucent shell -- drop to chase, and put the player's
            # own car back on screen so it can be seen from the new angle.
            if self._watch_ghost and config.CAM_MODES[self.cam_idx] == "hood":
                self.cam_idx = 0
            self.car.hull.enabled = (
                self._watch_ghost
                or config.CAM_MODES[self.cam_idx] != "hood")
            self._snap_camera()
        elif key == "r":
            self._reset_to_track()
        elif key == "m":
            self.muted = not self.muted
        elif key == "t":
            # One switch for all the driver aids: at an exhibition nobody wants
            # to reason about TC vs ABS vs steering assist separately.
            v = self.vehicle
            on = not v.traction_control
            v.traction_control = v.abs_enabled = v.steer_assist = on
        elif key == "f":
            window.fullscreen = not window.fullscreen
        elif key == "escape":
            if self.state == PAUSED:
                self.state = self._resume_state
                self.vehicle.frozen = self.state == COUNTDOWN
            elif self.state == FINISHED:
                self._leave()
            else:
                # Pause rather than quit: ESC mid-race used to dump you back to
                # the menu with no way to say "I didn't mean that".
                self._resume_state = self.state
                self.state = PAUSED
                self.vehicle.frozen = True
        elif key == "enter" and self.state in (PAUSED, FINISHED):
            self._leave()

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

    # -- main loop ------------------------------------------------
    def update(self):
        # The loading card built the HUD dark; bring it up now, with the lights.
        if self._reveal_hud:
            self._reveal_hud = False
            self.hud.root.enabled = True
        dt = min(time.dt, 0.05)
        if not self._cam_warm:
            self._cam_warm = True
            self._snap_camera()
        if self.state == PAUSED:
            # Everything stops: no physics, no clock, no engine note. The HUD
            # still draws, so the frozen frame stays behind the pause card.
            self.sound.update(0.0, config.MAX_SPEED, 0.0, dt, muted=True)
            self._draw_hud(self.surface.hint)
            return
        # The start clock runs through the countdown and a little past it, so
        # the gantry finishes riding up while the race is already on.
        if self.start_t < self._T_rise1 + 0.1:
            self.start_t += dt
        if self.state == COUNTDOWN:
            if self.start_t >= self._T_out:          # lights out -> go
                self.state = RACING
                self.vehicle.frozen = False
                if self.ghost is not None:
                    self.ghost.start()
                self.session_time = 0.0
                self.lap_start = 0.0
        else:
            self.session_time += dt

        ctl = self.read_controls()
        self._ctl = ctl
        self._accum = min(self._accum + dt, 0.1)
        while self._accum >= DT:
            self.vehicle.step(ctl, DT, self.surface)
            if self.ghost is not None:
                self.ghost.step(DT)
            self._accum -= DT
        # how far past the last physics step this frame lands
        self._alpha = self._accum / DT

        i, _ = self.surface.progress(self.vehicle.pos)
        self._lap_logic(i)

        self.car.sync(braking=ctl.brake > 0.05, dt=dt, alpha=self._alpha)
        if self.ghost is not None:
            self.ghost.sync(dt, self._alpha)
        self.light.follow(float(self.vehicle.pos[0]), float(self.vehicle.pos[1]))
        # The free camera drives `camera` itself; the chase rig must not fight
        # it for the same transform on the same frame.
        if self.freecam.enabled:
            self.freecam.update(dt)
        else:
            self._update_camera(dt)
        # Spectating means watching a car, and a car you are behind sounds
        # like itself, not like the one you left on the other side of the
        # circuit.
        heard = (self.ghost.vehicle if self._spectating() else self.vehicle)
        heard_ctl = (self.ghost.controls if self._spectating() else ctl)
        self.sound.update(heard.speed, config.MAX_SPEED,
                          heard_ctl.throttle, dt, muted=self.muted)
        self._draw_hud(i)

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
        b = self.best_sectors[k]
        self.best_sectors[k] = t if b is None else min(b, t)
        self.sector_start = self.session_time

    def _complete_lap(self):
        if self.lap_num == 0:
            # End of the out lap. Nothing to record: this lap started from a
            # standing start, and timing it would put the launch in the
            # results.
            self.lap_num = 1
            self.lap_start = self.session_time
            self.sector_start = self.session_time
            self.sectors = [None, None, None]
            return
        t = self.session_time - self.lap_start
        self.last_t = t
        self.best_t = t if self.best_t is None else min(self.best_t, t)
        if self.sector == 2:
            self._close_sector()
        self.sectors = [None, None, None]
        self.sector_start = self.session_time
        self.lap_num += 1
        self.lap_start = self.session_time
        if self.lap_num > self.total_laps:
            self.state = FINISHED
            self.finish_t = self.session_time
            # Not frozen. The result card comes up over a car that is still
            # moving, which is what the end of a race looks like; the
            # autopilot in _cooldown_controls does the driving from here.
            from .autopilot import Autopilot
            self._cooldown = Autopilot(self.track, self.surface)

    def _spectating(self) -> bool:
        return self._watch_ghost and self.ghost is not None

    def _draw_hud(self, i: int):
        if self.hud.stale():
            # The window changed shape -- fullscreen, most likely. The HUD
            # anchors to the screen edges when it is built, so it is rebuilt.
            self.hud.destroy()
            self.hud = HUD(self.track, self.total_laps)
        # While spectating, the readouts follow the AI too -- its speed, its
        # pedals -- so the trace on screen matches the car on screen. The lap
        # clock and the gap stay the player's: those are what the player is
        # racing, and the point of watching the AI is to see how it is beating
        # them.
        spectating = self._spectating()
        g = self.ghost
        v = g.vehicle if spectating else self.vehicle
        ctl = g.controls if spectating else self._ctl
        # Its index too, not the player's: the wrong-way and off-track tests
        # below compare a velocity against the tangent at a point on the lap,
        # and using the player's point while watching the AI compares two cars
        # that may be half a circuit apart.
        here = int(g._last_i) if spectating else i

        flag = ""
        if self.state == RACING and np.dot(v.vel, self.track.tangent[here]) < -3:
            flag = "WRONG WAY"
        elif not v.on_track and self.state == RACING:
            flag = "OFF TRACK"

        cur_t = None if self.state == COUNTDOWN else self.session_time - self.lap_start
        # Lap number and lap times follow the car being watched as well. The
        # point of spectating is to see how the AI is doing it, and its lap is
        # not the player's -- the two are rarely even on the same tour.
        show_lap = g.lap_num if spectating else self.lap_num
        show_cur = (g.lap_time if spectating and self.state != COUNTDOWN
                    else cur_t)
        show_last = g.last_t if spectating else self.last_t
        show_best = g.best_t if spectating else self.best_t
        self.hud.update(
            speed_kmh=v.speed * 3.6,
            speed_frac=min(1.0, v.speed / config.MAX_SPEED),
            lap=show_lap,
            cur_t=show_cur if self.state != FINISHED else show_last,
            last_t=show_last,
            best_t=show_best,
            session_t=self.session_time,
            sectors=self._sector_lights(spectating),
            standings=self._standings(i, cur_t),
            lights=self._start_lights(),
            gantry_dy=self._gantry_dy(),
            flag=flag,
            spectating=spectating,
            # Always the player's dot for the player and the AI's for the
            # AI. Feeding the watched car in as `car_xz` swapped the colours
            # over the moment you pressed G, so the map said the player had
            # teleported to wherever the AI was.
            car_xz=self.vehicle.pos,
            ghost_xz=None if g is None else g.vehicle.pos,
            throttle=ctl.throttle,
            brake=ctl.brake,
            steer=v.steer_input,
            slip=abs(v.slip_angle),
            tc_cut=v.tc_cut,
            esc_cut=v.esc_cut,
            tc_off=not v.traction_control,
            dt=min(time.dt, 0.05),
            finished=self.state == FINISHED,
            paused=self.state == PAUSED,
        )

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

    def _standings(self, i: int, cur_t):
        """Tower rows, leader first. Position is by distance covered, the way
        a race is scored; the gap is the time between the two cars at the
        same point on the track, which is what the trailing driver can act on.
        """
        from .ui import TEAM_AI, TEAM_YOU

        # The result is the result. Both cars keep circulating after the flag,
        # so left live the gap went on counting -- and the finishing margin,
        # the one number the whole race was for, drifted away while the card
        # showing it was on screen.
        if self.state == FINISHED and self._final_rows is not None:
            return self._final_rows

        L = self.track.length
        you = dict(tla="YOU", name="PLAYER", col=TEAM_YOU, best=self.best_t,
                   player=True,
                   dist=self.lap_num * L + float(self.track.arclen[i]))
        rows = [you]
        g = self.ghost
        if g is not None:
            gi = g._last_i
            rows.append(dict(tla="AI", name="AI DRIVER", col=TEAM_AI,
                             best=g.best_t, player=False,
                             dist=g.lap_num * L + float(self.track.arclen[gi])))
        # Before the start both cars have covered nothing, so sorting by
        # distance is a coin toss that flips frame to frame and swapped the
        # order on the grid. The player starts on pole.
        if self.state != COUNTDOWN:
            rows.sort(key=lambda r: -r["dist"])
        bests = [r["best"] for r in rows if r["best"] is not None]
        session_best = min(bests) if bests else None
        for p, r in enumerate(rows):
            r["pos"] = p + 1
            r["purple"] = (r["best"] is not None and session_best is not None
                           and abs(r["best"] - session_best) < 1e-6)
            if p == 0:
                r["gap"] = "LEADER"
                continue
            laps_down = int((rows[0]["dist"] - r["dist"]) // L)
            delta = (g.delta(i, cur_t)
                     if g is not None and self.state != COUNTDOWN else None)
            if laps_down >= 1:
                r["gap"] = f"+{laps_down} LAP" + ("S" if laps_down > 1 else "")
            elif delta is not None:
                r["gap"] = f"+{abs(delta):.3f}"
            else:
                r["gap"] = "--.---"
        if self.state == FINISHED:
            self._final_rows = rows
        return rows


# ---------------------------------------------------------------------------
GAME: Game | None = None


MENU = None

#: The loading card, while one is up. It owns the update loop and swallows
#: input until its build has run and it has faded out.
TRANSITION = None


def update():
    global TRANSITION
    if TRANSITION is not None:
        TRANSITION.tick()
        if TRANSITION.done:
            TRANSITION.destroy()
            TRANSITION = None
        return
    if GAME is not None:
        GAME.update()


def input(key):  # noqa: A001  (ursina hook name)
    # The menu and the race are never both up, so one hook can serve both.
    if TRANSITION is not None:
        return
    if MENU is not None:
        MENU.on_key(key)
    elif GAME is not None:
        GAME.on_key(key)


# Set once by main(); the menu and the race hand control back and forth and
# both need the same session options.
SESSION: dict = {"laps": config.TOTAL_LAPS, "mute": False, "track": None}


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


def _build_race(track_name: str, laps: int, mute: bool, progress=None):
    """Tear the menu down and build the race. The heavy frame -- called behind
    the loading card by ``_start_race``, or straight away by the tools and the
    ``--track`` launch, which have no menu to transition from."""
    global GAME, MENU
    if MENU is not None:
        MENU.destroy()
        MENU = None
    window.title = f"FORMULA-AI — {track_name}"
    SESSION["track"] = track_name
    GAME = Game(track_name, laps, mute=mute, on_exit=_back_to_menu,
                progress=progress)


def _start_race(track_name: str, laps: int, mute: bool):
    """Menu -> race, through a loading card."""
    from .ui import caption
    _transition(caption(track_name)[0],
                lambda pump: _build_race(track_name, laps, mute, pump))


def _build_menu(progress=None):
    """Tear the race down and put the menu back, on the circuit just raced."""
    global GAME, MENU
    from ursina import application

    from .menu import StartMenu, available_circuits

    if GAME is not None:
        GAME.destroy()
        GAME = None

    window.color = pal.rgb(21, 21, 30)
    window.title = "FORMULA-AI"
    MENU = StartMenu(
        available_circuits(),
        on_start=lambda n: _start_race(n, SESSION["laps"], SESSION["mute"]),
        on_quit=application.quit,
        initial=SESSION["track"])


def _back_to_menu():
    """Race -> menu, through a loading card."""
    _transition("MAIN MENU", _build_menu)


def main(argv=None):
    p = argparse.ArgumentParser(description="FORMULA-AI racing prototype")
    p.add_argument("--track", default=None,
                   help="skip the menu and go straight to this circuit "
                        "(folder name under f1tenth_racetracks-main, e.g. Monza)")
    p.add_argument("--laps", type=int, default=config.TOTAL_LAPS)
    p.add_argument("--fullscreen", action="store_true", default=config.FULLSCREEN)
    p.add_argument("--mute", action="store_true")
    p.add_argument("--selftest", type=float, default=0.0,
                   help="run for N seconds with an autopilot, then quit")
    args = p.parse_args(argv)

    app = Ursina(title="FORMULA-AI", size=config.WINDOW_SIZE,
                 fullscreen=args.fullscreen, vsync=True,
                 development_mode=False)
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

    SESSION.update(laps=args.laps, mute=args.mute,
                   track=args.track or config.DEFAULT_TRACK)

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
