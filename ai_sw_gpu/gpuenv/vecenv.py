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
                 track: TrackGPU | None = None,
                 focus_window: tuple[float, float] | None = None,
                 adaptive_reset: bool = False,
                 lap_tol: int | None = None,
                 hard_bank: int = 0, hard_lookback: int = 40,
                 hard_horizon: int = 120):
        self.t = track if track is not None else TrackGPU(circuit, device)
        #: Off-track steps a lap may have and still count as a clean lap for
        #: best_lap_time. None = config.RL_RL_LAP_OFF_TOL, the training reward's
        #: (lenient) tolerance; the eval env passes the strict PERFECT one so
        #: its reported lap time means a genuinely clean lap.
        self.lap_tol = config.RL_RL_LAP_OFF_TOL if lap_tol is None else lap_tol
        #: Failure-state replay (config.RL_HARD_*): a ring of the last
        #: hard_lookback+1 decision-start states of every env, and a bank of
        #: the states that preceded a failure. Off unless hard_bank > 0; the
        #: trainer flips hard_active once the policy is nearly clean.
        self._hard_cap = int(hard_bank)
        self.hard_active = False
        self.hard_lookback = int(hard_lookback)
        #: Decisions a banked-state episode runs before it is truncated: the
        #: lookback (approach) plus the corner itself plus a little exit.
        self.hard_horizon = int(hard_horizon)
        self._hard_slot = torch.zeros(n_env, dtype=torch.bool, device=self.t.device)
        self._short = torch.zeros(n_env, dtype=torch.bool, device=self.t.device)
        self.hard_resets = 0
        #: Multiplier on config.RL_OFF_EVENT_COST, set each iteration by the
        #: trainer's ramp (1.0 when unused, as in eval and the CPU game env).
        self.off_event_scale = 1.0
        self._bank_n = 0
        self._bank_ptr = 0
        self.device = self.t.device
        self.n = n_env
        self.dt = dt
        self.randomise_start = randomise_start
        self.start_at_line = start_at_line
        #: See RaceEnv.focus_window (rlenv.py) -- same fraction-of-lap
        #: window, restricting scattered resets for sector-focused training.
        #: Mutually exclusive with adaptive_reset (below); the caller's job
        #: to only set one.
        self.focus_window = focus_window
        #: Adaptive difficulty-weighted resets (config.RL_ADAPTIVE_*): scattered
        #: resets are drawn from config.RL_ADAPTIVE_BUCKETS equal-arclength
        #: bins, weighted toward whichever bins the policy is currently
        #: failing (going off-track) in most, tracked live from real
        #: telemetry in step(). See the long comment in config.py.
        self.adaptive_reset = adaptive_reset
        if adaptive_reset:
            nb = config.RL_ADAPTIVE_BUCKETS
            self.n_buckets = nb
            self.bucket_fail_ema = torch.zeros(nb, dtype=torch.float64,
                                               device=self.device)
            self._bucket_of_sample = (
                torch.arange(self.t.n, device=self.device) * nb
                // max(self.t.n, 1)).clamp(max=nb - 1)
        # See RaceEnv's mirror of this in rlenv.py for why a circuit missing
        # from EPISODE_SECONDS_BY_CIRCUIT is auto-sized from its OWN
        # model_lap (already computed on TrackGPU) rather than falling back
        # to EPISODE_SECONDS ("sized for Monza"), and why this starts at the
        # SHORT (~1.076x, one lap) ratio rather than LONG_EPISODE_RATIO --
        # the training loop widens it itself, once, on first PERFECT.
        self.episode_seconds = EPISODE_SECONDS_BY_CIRCUIT.get(
            circuit, 1.076 * self.t.model_lap)
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
        if self._hard_cap > 0:
            D = 16 + N_PREV_ACTIONS
            R = self.hard_lookback + 1
            self._hist = torch.zeros(R, n_env, D, device=self.device)
            self._hist_ptr = 0
            self._hard_cool = torch.zeros(n_env, dtype=torch.long,
                                          device=self.device)
            self._bank = torch.zeros(self._hard_cap, D, device=self.device)
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
            self._prev_off = torch.zeros(n, dtype=torch.bool, device=dev)
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
        self._prev_off = self._prev_off & ~m
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
        # No target-speed tail here -- see rlpolicy.observe()'s matching
        # 2026-09-14 comment (OBS_DIM 59 -> 43): must stay byte-for-byte the
        # same shape as the CPU observation or parity.py fails immediately.

        return torch.cat([torch.stack(head, -1), mid, remain,
                          lat, lon], dim=-1)

    # -- failure-state replay -------------------------------------------
    def _snapshot(self, i):
        v = self.vehicle
        cols = [v.pos, v.vel] + [x[:, None] for x in (
            v.yaw, v.yaw_rate, v.steer_input, v.steer_angle, v._long_accel,
            v._lat_accel, v._f_long_front, v._f_long_rear, v.slip_front,
            v.slip_rear, v.grip_scale)]
        cols += [i.to(v.yaw.dtype)[:, None], self.prev_actions.to(v.yaw.dtype)]
        return torch.cat(cols, -1)

    def set_hard(self, frac: float):
        """Dedicate the last ``frac`` of the env slots to failure-state
        episodes and switch failure-state replay on."""
        n_hard = int(round(frac * self.n))
        self._hard_slot = torch.arange(self.n, device=self.device)             >= (self.n - n_hard)
        self.hard_active = True

    def _restore(self, m, S):
        """Put the masked rows back into the saved states S (n, D)."""
        v, t = self.vehicle, self.t
        mm = m[:, None]
        v.pos = torch.where(mm, S[:, 0:2], v.pos)
        v.vel = torch.where(mm, S[:, 2:4], v.vel)
        for k, name in enumerate(("yaw", "yaw_rate", "steer_input",
                                  "steer_angle", "_long_accel", "_lat_accel",
                                  "_f_long_front", "_f_long_rear",
                                  "slip_front", "slip_rear", "grip_scale")):
            setattr(v, name, torch.where(m, S[:, 4 + k], getattr(v, name)))
        v.on_track = v.on_track | m
        v.hit_wall = v.hit_wall & ~m
        idx = S[:, 15].long().clamp(0, t.n - 1)
        self.surface.hint = torch.where(m, idx, self.surface.hint)
        self._last_s = torch.where(m, t.l_arclen[idx], self._last_s)
        self.prev_actions = torch.where(mm, S[:, 16:].long(), self.prev_actions)
        zf = torch.zeros_like(self.off_time)
        self.off_time = torch.where(m, zf, self.off_time)
        self.stall_time = torch.where(m, zf, self.stall_time)

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
        if self.adaptive_reset:
            nb = self.n_buckets
            fe = self.bucket_fail_ema
            total = fe.sum()
            norm = torch.where(total > 1e-8, fe / total.clamp(min=1e-8),
                               torch.full_like(fe, 1.0 / nb))
            weights = config.RL_ADAPTIVE_FLOOR / nb \
                + (1.0 - config.RL_ADAPTIVE_FLOOR) * norm
            bkt = torch.multinomial(weights, self.n, replacement=True,
                                    generator=self.gen)
            per_bucket = max(self.t.n // nb, 1)
            lo_idx = bkt * per_bucket
            span_idx = torch.where(bkt == nb - 1,
                                   self.t.n - lo_idx,
                                   torch.full_like(lo_idx, per_bucket))
            uw = torch.rand(self.n, generator=self.gen, device=self.device)
            i_rand = (lo_idx + (uw * span_idx.to(uw.dtype)).long()) \
                .clamp(max=self.t.n - 1)
        elif self.focus_window is not None:
            lo, hi = self.focus_window
            span = (hi - lo) % 1.0 or 1.0
            uw = torch.rand(self.n, generator=self.gen, device=self.device)
            i_rand = ((lo + uw * span) % 1.0 * self.t.n).long()
        else:
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
        self._short = self._short & ~m
        if (self._hard_cap > 0 and self.hard_active
                and self._bank_n >= config.RL_HARD_MIN):
            hm = m & self._hard_slot
            if bool(hm.any()):
                kk = torch.randint(0, self._bank_n, (self.n,),
                                   generator=self.gen, device=self.device)
                self._restore(hm, self._bank[kk])
                self.start_s = torch.where(hm, self._last_s, self.start_s)
                self._short = self._short | hm
                self.hard_resets += int(hm.sum())
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
    def step(self, action, off_cost_scale: float = 1.0, pace_mult: float = 1.0,
             lap_w_mult: float = 1.0):
        t, v = self.t, self.vehicle
        dt = self.dt
        steer = self.a_steer[action]
        throttle = self.a_throttle[action]
        brake = self.a_brake[action]

        i0 = self._index()
        phi0 = self._potential(i0)
        if self._hard_cap > 0:
            R = self.hard_lookback + 1
            self._hist[self._hist_ptr % R] = self._snapshot(i0)
            self._hist_ptr += 1

        zf = self._f()
        ds_total = zf.clone()
        hit_wall = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        ke_at_wall = zf.clone()
        off_secs = zf.clone()
        off_events = zf.clone()
        off_dist_speed = zf.clone()
        wall_dist_speed = zf.clone()
        if self.adaptive_reset:
            bucket_off_sum = torch.zeros(self.n_buckets, dtype=torch.float64,
                                         device=self.device)
            bucket_visit_sum = torch.zeros(self.n_buckets, dtype=torch.float64,
                                           device=self.device)

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

            if config.RL_EDGE_WALL_COST > 0.0:
                ws = ((v.pos - t.l_center[i]) * t.l_normal[i]).sum(-1)
                we = torch.where(ws > 0.0, t.w_right[i], t.w_left[i])
                wd = ((ws.abs() - we) - config.RL_EDGE_WALL_START).clamp(min=0.0)
                wall_dist_speed = wall_dist_speed + (
                    wd / config.RL_EDGE_WALL_WIDTH).clamp(
                        max=config.RL_EDGE_WALL_CAP) ** 2 * speed
            off = ~v.on_track
            off_events = off_events + (off & ~self._prev_off).to(off_events.dtype)
            self._prev_off = off
            self.off_steps = self.off_steps + off.long()
            self.lap_off_steps = self.lap_off_steps + off.long()
            self.off_time = torch.where(off, self.off_time + dt, zf)
            off_secs = off_secs + torch.where(off, torch.full_like(zf, dt), zf)
            off_dist_speed = off_dist_speed + torch.where(off, speed, zf)
            if self.adaptive_reset:
                bkt = self._bucket_of_sample[i]
                bucket_visit_sum.scatter_add_(
                    0, bkt, torch.ones(self.n, dtype=torch.float64,
                                       device=self.device))
                bucket_off_sum.scatter_add_(0, bkt, off.to(torch.float64))
            # Progress off the asphalt earns nothing: a wide line is a
            # metre-for-metre loss, not free speed.
            ds_total = ds_total + torch.where(off, zf, step_ds)

            self.stall_time = torch.where(speed < STALL_SPEED,
                                          self.stall_time + dt, zf)

        if self.adaptive_reset:
            visited = bucket_visit_sum > 0
            rate = torch.where(visited, bucket_off_sum / bucket_visit_sum.clamp(min=1.0),
                               self.bucket_fail_ema)
            decay = config.RL_ADAPTIVE_EMA
            self.bucket_fail_ema = torch.where(
                visited, self.bucket_fail_ema * decay + rate * (1.0 - decay),
                self.bucket_fail_ema)

        self.steps = self.steps + 1
        if self._hard_cap > 0:
            K = self.hard_lookback
            self._hard_cool = (self._hard_cool - 1).clamp(min=0)
            if self.hard_active:
                # Only NORMAL slots feed the bank: a hard slot fails at the
                # very states it was handed (and under exploration noise), and
                # banking those would turn the bank into a loop of its own
                # failures instead of the policy's natural ones.
                ev = (((off_secs > 0.0) | hit_wall) & (self.steps > K)
                      & (self._hard_cool == 0) & ~self._hard_slot)
                sel = ev.nonzero(as_tuple=True)[0]
                if sel.numel() > 0:
                    # the slot about to be overwritten holds the state K
                    # decisions ago -- the approach, before the mistake
                    old = self._hist[self._hist_ptr % (K + 1)][sel]
                    slots = (self._bank_ptr + torch.arange(
                        sel.numel(), device=self.device)) % self._hard_cap
                    self._bank[slots] = old
                    self._bank_ptr = (self._bank_ptr + sel.numel())                         % self._hard_cap
                    was = self._bank_n
                    self._bank_n = min(self._hard_cap,
                                       self._bank_n + sel.numel())
                    if was < config.RL_HARD_MIN <= self._bank_n:
                        # the bank just became usable: cut the hard slots'
                        # current (long) episodes so they restart from it
                        self._short = self._short | self._hard_slot
                    self._hard_cool[sel] = K
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
        clean = closed & (self.lap_off_steps <= self.lap_tol) \
            & (self.recoveries == self._lap_rec0)

        lap_bonus = zf
        if _RACELINE_TASK:
            w_eff = config.RL_RL_LAP_W * lap_w_mult
            lap_bonus = torch.where(
                clean,
                (w_eff * (t.model_lap - lap_s)).clamp(min=0.0), zf)
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
        if config.RL_OFF_EVENT_COST > 0.0:
            reward = reward - (config.RL_OFF_EVENT_COST * self.off_event_scale
                               * off_events)
        if config.RL_EDGE_WALL_COST > 0.0:
            reward = reward - config.RL_EDGE_WALL_COST * wall_dist_speed * dt

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
        limit = torch.where(
            self._short,
            torch.full_like(self.lap_time,
                            self.hard_horizon * ACTION_REPEAT * self.dt),
            torch.full_like(self.lap_time, float(self.episode_seconds)))
        truncated = self.lap_time >= limit
        done = truncated | diverged
        reward = torch.where(diverged, zf, reward)
        return self.observe(), reward, done, {"truncated": truncated,
                                              "diverged": diverged}
