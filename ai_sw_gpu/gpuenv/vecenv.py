"""Batched twin of ``game.rlenv.RaceEnv``.

B independent episodes advanced together. Reward, recovery, lap arming and
the potential shaping are the same expressions as the scalar version, with
the branches turned into masks. ``step`` deliberately does NOT auto-reset:
the CPU trainer stores the pre-reset observation as the n-step successor and
resets afterwards, so the caller drives ``reset_done`` the same way.
"""
from __future__ import annotations

import math

import torch

from .physics import VecVehicle
from .surface import VecSurface
from .trackgpu import TrackGPU, config

from game import rlpolicy                                  # noqa: E402
from game.rlenv import (EPISODE_SECONDS,                       # noqa: E402
                        EPISODE_SECONDS_BY_CIRCUIT,
                        OFF_TRACK_PATIENCE, STALL_PATIENCE, STALL_SPEED)

ACTION_REPEAT = rlpolicy.ACTION_REPEAT
N_ACTIONS = rlpolicy.N_ACTIONS
N_PREV_ACTIONS = rlpolicy.N_PREV_ACTIONS
DEFAULT_ACTION = rlpolicy.DEFAULT_ACTION
OBS_DIM = rlpolicy.OBS_DIM
N_LOOK = rlpolicy.N_LOOK
_RACELINE_TASK = getattr(config, "RL_TASK", "linesight") == "raceline"


class VecRaceEnv:
    def __init__(self, circuit: str, n_env: int, dt: float = 1.0 / 60.0,
                 seed: int = 0, randomise_start: bool = True,
                 start_at_line: float = 0.15, device="cpu",
                 track: TrackGPU | None = None):
        self.t = track if track is not None else TrackGPU(circuit, device)
        self.device = self.t.device
        self.n = n_env
        self.dt = dt
        self.randomise_start = randomise_start
        self.start_at_line = start_at_line
        self.episode_seconds = EPISODE_SECONDS_BY_CIRCUIT.get(
            circuit, EPISODE_SECONDS)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

        acts = torch.tensor(rlpolicy.ACTIONS, device=self.device,
                            dtype=torch.get_default_dtype())
        self.a_steer = acts[:, 0].contiguous()
        self.a_throttle = acts[:, 1].clamp(min=0.0).contiguous()
        self.a_brake = (-acts[:, 1]).clamp(min=0.0).contiguous()

        self.vehicle = VecVehicle(n_env, self.device)
        self.surface = VecSurface(self.t, n_env)
        self.obs_dim = OBS_DIM
        self.act_dim = N_ACTIONS
        self.frontier = torch.zeros(n_env, device=self.device)
        self._reset_counters(torch.ones(n_env, dtype=torch.bool,
                                        device=self.device))

    # -- helpers --------------------------------------------------------
    def _f(self):
        return torch.zeros(self.n, device=self.device)

    def _reset_counters(self, m):
        dev, n = self.device, self.n
        if not hasattr(self, "steps"):
            self.steps = torch.zeros(n, dtype=torch.long, device=dev)
            self.off_time = self._f()
            self.stall_time = self._f()
            self.lap_time = self._f()
            self.progress = self._f()
            self.start_s = self._f()
            self.laps = torch.zeros(n, dtype=torch.long, device=dev)
            self.off_steps = torch.zeros(n, dtype=torch.long, device=dev)
            self.wall_steps = torch.zeros(n, dtype=torch.long, device=dev)
            self.recoveries = torch.zeros(n, dtype=torch.long, device=dev)
            self.speed_sum = self._f()
            self._last_s = self._f()
            self._armed = torch.zeros(n, dtype=torch.bool, device=dev)
            self.prev_actions = torch.full((n, N_PREV_ACTIONS), DEFAULT_ACTION,
                                           dtype=torch.long, device=dev)
            self.lap_off_steps = torch.zeros(n, dtype=torch.long, device=dev)
            self.lap_start_time = self._f()
            self.last_lap_time = self._f()
            self.best_lap_time = self._f()
            self._lap_rec0 = torch.zeros(n, dtype=torch.long, device=dev)
            return

        zl = torch.zeros_like(self.steps)
        zf = torch.zeros_like(self.lap_time)
        for name in ("steps", "laps", "off_steps", "wall_steps", "recoveries",
                     "lap_off_steps", "_lap_rec0"):
            setattr(self, name, torch.where(m, zl, getattr(self, name)))
        for name in ("off_time", "stall_time", "lap_time", "progress",
                     "start_s", "speed_sum", "_last_s", "lap_start_time",
                     "last_lap_time", "best_lap_time"):
            setattr(self, name, torch.where(m, zf, getattr(self, name)))
        self._armed = self._armed & ~m
        self.prev_actions = torch.where(
            m[:, None], torch.full_like(self.prev_actions, DEFAULT_ACTION),
            self.prev_actions)

    def _index(self):
        i, _ = self.surface.progress(self.vehicle.pos)
        return i

    def _potential(self, i):
        t = self.t
        v = self.vehicle
        # Attractor: distance from the shaping line (the raceline when
        # RL_SHAPE_TO_RACELINE is on). The only term that says *where* to drive.
        a_off = ((v.pos - t.s_center[i]) * t.s_normal[i]).sum(-1)
        phi = -config.RL_LINE_K * a_off.abs().clamp(config.RL_LINE_LO,
                                                    config.RL_LINE_HI)
        # The edge / limit terms stay on the centreline, where the widths live.
        off_signed = ((v.pos - t.l_center[i]) * t.l_normal[i]).sum(-1)
        off = off_signed.abs()
        edge = torch.where(off_signed > 0.0, t.w_right[i], t.w_left[i])
        phi = phi + config.RL_EDGE_K * (edge - off).clamp(0.0,
                                                          config.RL_EDGE_MARGIN)
        phi = phi - config.RL_EDGE_OUT_K * (off - edge).clamp(
            0.0, config.RL_EDGE_OUT_CAP)
        return phi

    # -- observation ----------------------------------------------------
    def observe(self):
        t, v = self.t, self.vehicle
        i = self._index()
        fwd = torch.stack([torch.sin(v.yaw), torch.cos(v.yaw)], -1)
        right = torch.stack([torch.cos(v.yaw), -torch.sin(v.yaw)], -1)

        cross = ((v.pos - t.l_center[i]) * t.l_normal[i]).sum(-1)
        line_yaw = torch.atan2(t.l_tangent[i, 0], t.l_tangent[i, 1])
        heading_err = (line_yaw - v.yaw + math.pi) % (2 * math.pi) - math.pi
        half = torch.where(cross > 0, t.w_right[i], t.w_left[i])
        phase = 2.0 * math.pi * t.l_arclen[i] / max(t.l_len, 1e-6)

        head = [
            torch.sin(phase), torch.cos(phase),
            (v.vel * fwd).sum(-1) / config.MAX_SPEED,
            (v.vel * right).sum(-1) / 20.0,
            v.yaw_rate / 3.0,
            cross / half.clamp(min=1e-3),
            heading_err / 0.6,
        ]
        mid = (self.prev_actions.to(v.yaw.dtype) - (N_ACTIONS - 1) / 2.0) \
            / ((N_ACTIONS - 1) / 2.0)
        remain = ((t.l_len - t.l_arclen[i]) / t.l_len)[:, None]

        js = t.look_idx[i]                                  # (B, N_LOOK)
        d = t.l_center[js] - v.pos[:, None, :]              # (B, N_LOOK, 2)
        lat = (d * right[:, None, :]).sum(-1) / 60.0
        lon = (d * fwd[:, None, :]).sum(-1) / 120.0
        vtg = t.v_ref[js] / config.MAX_SPEED

        return torch.cat([torch.stack(head, -1), mid, remain,
                          lat, lon, vtg], dim=-1)

    # -- placement ------------------------------------------------------
    def _place(self, m, i, speed_frac):
        """``RaceEnv._place`` for the masked rows."""
        t = self.t
        yaw = torch.atan2(t.l_tangent[i, 0], t.l_tangent[i, 1])
        speed = t.v_ref[i] * speed_frac
        vel = torch.stack([torch.sin(yaw), torch.cos(yaw)], -1) \
            * speed[:, None]
        self.vehicle.place(m, t.l_center[i], yaw, vel)
        self.surface.hint = torch.where(m, i, self.surface.hint)
        self._last_s = torch.where(m, t.l_arclen[i], self._last_s)
        self.off_time = torch.where(m, torch.zeros_like(self.off_time),
                                    self.off_time)
        self.stall_time = torch.where(m, torch.zeros_like(self.stall_time),
                                      self.stall_time)

    def reset_done(self, m):
        """``RaceEnv.reset`` for the masked rows: a grid start
        ``start_at_line`` of the time, otherwise scattered round the lap so
        every corner is in the buffer."""
        if not bool(m.any()):
            return self.observe()
        self._reset_counters(m)
        r = torch.rand(self.n, generator=self.gen, device=self.device)
        scatter = r >= self.start_at_line
        i_rand = torch.randint(self.t.n, (self.n,), generator=self.gen,
                               device=self.device)
        i = torch.where(scatter, i_rand, torch.zeros_like(i_rand))
        u = torch.rand(self.n, generator=self.gen, device=self.device)
        # Scattered starts roll in at speed; a grid start covers the launch
        # itself, so 0.0 is in range on purpose. And it needs to come up
        # often enough to be learned, not just tolerated -- see
        # RL_LAUNCH_STOP_FRAC in the CPU rlenv.reset(), which this mirrors:
        # a plain uniform(0, 0.95) draw put a true near-standing launch in
        # only about 1 in 20 grid starts.
        stop_pick = torch.rand(self.n, generator=self.gen, device=self.device)
        stop_bias = stop_pick < config.RL_LAUNCH_STOP_FRAC
        grid_frac = torch.where(stop_bias, u * config.RL_LAUNCH_STOP_MAX,
                                u * 0.95)
        frac = torch.where(scatter, 0.55 + 0.40 * u, grid_frac)
        self._place(m, i, frac)
        self.start_s = torch.where(m, self._last_s, self.start_s)
        return self.observe()

    def reset_grid(self, speed_frac):
        """Deterministic grid start at a chosen launch speed, for evaluation."""
        m = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self._reset_counters(m)
        i = torch.zeros(self.n, dtype=torch.long, device=self.device)
        if not torch.is_tensor(speed_frac):
            speed_frac = torch.full((self.n,), float(speed_frac),
                                    device=self.device)
        self._place(m, i, speed_frac)
        self.start_s = self._last_s.clone()
        return self.observe()

    # -- the loop -------------------------------------------------------
    def step(self, action, off_cost_scale: float = 1.0, pace_mult: float = 1.0):
        t, v = self.t, self.vehicle
        dt = self.dt
        steer = self.a_steer[action]
        throttle = self.a_throttle[action]
        brake = self.a_brake[action]

        phi0 = self._potential(self._index())

        zf = self._f()
        ds_total = zf.clone()
        hit_wall = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        ke_at_wall = zf.clone()
        off_secs = zf.clone()
        off_dist_speed = zf.clone()

        for _ in range(ACTION_REPEAT):
            v.step(steer, throttle, brake, dt, self.surface)
            self.lap_time = self.lap_time + dt
            i = self._index()
            s = t.l_arclen[i]
            step_ds = (s - self._last_s + t.l_len * 1.5) % t.l_len \
                - t.l_len * 0.5
            self._last_s = s
            speed = v.speed
            self.speed_sum = self.speed_sum + speed

            hit_wall = hit_wall | v.hit_wall
            ke_at_wall = torch.maximum(
                ke_at_wall, torch.where(v.hit_wall, speed * speed, zf))

            off = ~v.on_track
            self.off_steps = self.off_steps + off.long()
            self.lap_off_steps = self.lap_off_steps + off.long()
            self.off_time = torch.where(off, self.off_time + dt, zf)
            off_secs = off_secs + torch.where(off, torch.full_like(zf, dt), zf)
            off_dist_speed = off_dist_speed + torch.where(off, speed, zf)
            # Progress off the asphalt earns nothing: a wide line is a
            # metre-for-metre loss, not free speed.
            ds_total = ds_total + torch.where(off, zf, step_ds)

            self.stall_time = torch.where(speed < STALL_SPEED,
                                          self.stall_time + dt, zf)

        self.steps = self.steps + 1
        self.progress = self.progress + ds_total
        self.frontier = torch.minimum(
            torch.maximum(self.frontier, self.start_s + self.progress),
            torch.full_like(self.frontier, t.l_len))
        self.prev_actions = torch.cat(
            [self.prev_actions[:, 1:], action[:, None]], dim=1)

        i = self._index()
        n = t.n

        # -- lap arming and the clean-lap bonus -------------------------
        in_arm = (i >= 0.4 * n) & (i <= 0.6 * n)
        cross = (~in_arm) & self._armed & (i < 0.1 * n) & (ds_total > 0)
        lap_s = self.lap_time - self.lap_start_time
        closed = cross & (self.lap_start_time > 0.0)
        clean = closed & (self.lap_off_steps <= config.RL_RL_LAP_OFF_TOL) \
            & (self.recoveries == self._lap_rec0)

        lap_bonus = zf
        if _RACELINE_TASK:
            lap_bonus = torch.where(
                clean,
                (config.RL_RL_LAP_BASE
                 - config.RL_RL_LAP_W * lap_s).clamp(min=0.0), zf)
        better = clean & ((self.best_lap_time == 0.0)
                          | (lap_s < self.best_lap_time))
        self.best_lap_time = torch.where(better, lap_s, self.best_lap_time)
        self.last_lap_time = torch.where(closed, lap_s, self.last_lap_time)
        self.laps = self.laps + cross.long()
        self._armed = torch.where(in_arm, torch.ones_like(self._armed),
                                  self._armed & ~cross)
        self.lap_start_time = torch.where(cross, self.lap_time,
                                          self.lap_start_time)
        self.lap_off_steps = torch.where(cross,
                                         torch.zeros_like(self.lap_off_steps),
                                         self.lap_off_steps)
        self._lap_rec0 = torch.where(cross, self.recoveries, self._lap_rec0)
        self.wall_steps = self.wall_steps + hit_wall.long()

        # -- reward: progress, minus time -------------------------------
        reward = config.RL_PROGRESS_W * ds_total \
            - config.RL_TIME_W * (ACTION_REPEAT * dt)

        if _RACELINE_TASK:
            vtgt = t.v_ref[i].clamp(min=1.0)
            spd_err = ((v.speed - vtgt) / config.RL_RL_SPEED_SIG).clamp(max=0.0)
            reward = reward + config.RL_RL_SPEED * torch.exp(-spd_err * spd_err)
            reward = reward + config.RL_RL_SPEED_LIN * (
                v.speed / config.MAX_SPEED).clamp(max=config.RL_RL_SPEED_LIN_CAP)
            reward = reward + lap_bonus

        # -- off-course fee, per second of a wheel off the asphalt ------
        reward = reward - (config.RL_OFF_TRACK_COST * off_cost_scale
                           * off_dist_speed * dt)

        # -- a mistake is recovered, not terminal, but it is charged for -
        back = ds_total < -2.0
        stalled = self.stall_time > STALL_PATIENCE
        off_long = self.off_time > OFF_TRACK_PATIENCE
        # Precedence, as in the scalar chain: wall, off track, stalled, back.
        r_wall = hit_wall
        r_off = (~r_wall) & off_long
        r_stall = (~r_wall) & (~r_off) & stalled
        r_back = (~r_wall) & (~r_off) & (~r_stall) & back
        reason = r_wall | r_off | r_stall | r_back

        entry = torch.where(hit_wall, torch.sqrt(ke_at_wall.clamp(min=0.0)),
                            v.speed)
        charge = config.RL_RECOVER_COST + config.RL_RECOVER_SPEED_COST * (
            entry / config.MAX_SPEED).clamp(max=1.0)
        charged = reason & ~r_stall
        reward = reward - torch.where(charged, charge, zf)

        # -- potential shaping, skipped across a recovery ---------------
        reward = reward + torch.where(reason, zf, self._potential(i) - phi0)

        if bool(reason.any()):
            self.recoveries = self.recoveries + reason.long()
            frac = torch.where(r_stall, torch.full_like(zf, 0.45),
                               torch.full_like(zf, config.RL_RECOVER_FRAC))
            self._place(reason, i, frac)

        diverged = ~torch.isfinite(v.pos).all(dim=-1)
        truncated = self.lap_time >= self.episode_seconds
        done = truncated | diverged
        reward = torch.where(diverged, zf, reward)
        return self.observe(), reward, done, {"truncated": truncated,
                                              "diverged": diverged}
