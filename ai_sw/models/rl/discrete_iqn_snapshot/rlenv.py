"""The driving task as a reinforcement-learning environment, Linesight style.

This is a port of the reward and episode design from the Linesight Trackmania
project, which is the strongest published RL driver there is.

* **Discrete actions.** Nine of them (see ``rlpolicy.ACTIONS``), chosen by
  argmax over a distributional value network. Every continuous attempt here
  failed the same way -- a Gaussian policy is optimised *including its own
  exploration noise*, so the best noisy policy is slow and the mean inherits
  it. Discrete + argmax breaks that coupling.

* **Reward = progress - time.** Per step: metres advanced along the centreline
  times a small constant, minus a fixed time penalty. Nothing else. Over a
  fixed-length episode the time penalty is very nearly constant, so the return
  is dominated by distance covered -- which is average speed, which is lap
  time. There is no speed bonus, no corner term, no line multiplier: those
  were all proxies and each one leaked.

* **Line discipline is potential-based.** ``F = Phi(s') - Phi(s)`` with
  ``Phi = -k * clip(|offset from centre|, 2, 25)``. Potential-based shaping
  provably does not change the optimal policy (Ng, Harada & Russell 1999): it
  pulls the *learning* toward the centre of the road without being able to
  make a slower line score better.

* **A mistake does not end the episode.** Hitting a barrier, sliding off, or
  stopping puts the car back on the centreline where it is, slowly, and the
  clock keeps running. Ending on a crash would let the policy *escape* the
  time penalty by crashing early; recovering makes a mistake cost exactly what
  it costs in a real race -- the seconds to get going again.
"""
from __future__ import annotations

import math

import numpy as np

from . import config
from . import raceline as rl
from . import rlpolicy
from .rlpolicy import (ACTION_REPEAT, DEFAULT_ACTION, N_ACTIONS, N_PREV_ACTIONS,
                       OBS_DIM, observe)
from .surface import Surface
from .trackdata import Track, load_track
from .vehicle import Vehicle

#: Wall-clock seconds per episode. Truncation only -- the episode never ends
#: early. Long enough that a grid start is a whole Monza lap (~102 s at the
#: target pace) plus margin, so ``reach`` reads directly as lap progress and
#: the final sector is trained from grid starts, not only the scattered ones.
EPISODE_SECONDS = 130.0

#: Seconds continuously off the track before the car is recovered onto the
#: line. A brief kerb clip or a two-wheel excursion is racing; a full second
#: with every wheel on the grass is a mistake.
OFF_TRACK_PATIENCE = 0.7

#: Below this speed (m/s) for this many seconds, the car is stuck and is
#: recovered. Not a failure -- the lost seconds are the cost.
STALL_SPEED = 4.0
STALL_PATIENCE = 2.0


class RaceEnv:
    """One circuit, one car. Gym-like without the dependency. ``step`` takes a
    discrete action index and internally holds it for ``ACTION_REPEAT``
    physics ticks, the way Linesight holds a keypress for 50 ms."""

    def __init__(self, circuit: str, dt: float = 1.0 / 60.0,
                 seed: int = 0, randomise_start: bool = True,
                 start_at_line: float = 0.15):
        self.track: Track = load_track(circuit)
        self.surface = Surface(self.track)
        self.vehicle = Vehicle()
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.randomise_start = randomise_start
        #: Fraction of resets placed on the grid. The rest start at a random
        #: point on the lap so every corner is in the buffer regardless of
        #: which part the policy is currently good at -- the value network
        #: needs to have seen the far side of a corner to know it is worth
        #: braking for.
        self.start_at_line = start_at_line

        self.line = rlpolicy.reference_line(self.track)
        self.v_ref = rl.speed_profile(self.line.seg_len, self.line.curvature,
                                      self.line.curv_radius, 1.0)
        self.look_idx = rlpolicy._lookahead_indices(self.line)

        self.obs_dim, self.act_dim = OBS_DIM, N_ACTIONS
        #: Furthest raw progress any episode in this env has reached, for the
        #: log only.
        self.frontier = 0.0
        self._reset_counters()

    # -- helpers --------------------------------------------------------
    def _reset_counters(self):
        self.steps = 0
        self.off_time = 0.0
        self.stall_time = 0.0
        self.lap_time = 0.0
        self.progress = 0.0
        self.start_s = 0.0
        self.laps = 0
        self.off_steps = 0
        self.wall_steps = 0
        self.recoveries = 0
        self.speed_sum = 0.0
        self._last_s = 0.0
        self._armed = False
        self.prev_actions = [DEFAULT_ACTION] * N_PREV_ACTIONS

    def _index(self):
        i, _ = self.surface.progress(self.vehicle.pos)
        return i

    def _potential(self, i: int) -> float:
        off = abs(float(np.dot(self.vehicle.pos - self.line.center[i],
                               self.line.normal[i])))
        return -config.RL_LINE_K * min(max(off, config.RL_LINE_LO),
                                       config.RL_LINE_HI)

    def observe(self) -> np.ndarray:
        return observe(self.vehicle, self.track, self.line, self.v_ref,
                       self._index(), self.prev_actions, self.look_idx)

    # -- the loop -----------------------------------------------------------
    def _place(self, i: int, speed_frac: float):
        g = self.line
        yaw = math.atan2(g.tangent[i, 0], g.tangent[i, 1])
        self.vehicle = Vehicle()
        self.vehicle.frozen = False
        self.vehicle.place(g.center[i], yaw)
        speed = float(self.v_ref[i]) * speed_frac
        self.vehicle.vel = np.array([math.sin(yaw), math.cos(yaw)]) * speed
        self.vehicle.yaw_rate = 0.0
        self.surface.hint = i
        self._last_s = float(g.arclen[i])
        self.off_time = 0.0
        self.stall_time = 0.0

    def reset(self) -> np.ndarray:
        self._reset_counters()
        if (self.randomise_start
                and self.rng.random() >= self.start_at_line):
            i = int(self.rng.integers(self.track.count))
        else:
            i = 0
        self._place(i, float(self.rng.uniform(0.55, 0.95)))
        self.start_s = self._last_s
        return self.observe()

    def _recover(self, i: int):
        """Back onto the centreline here, slow, episode still running."""
        self._place(i, config.RL_RECOVER_FRAC)
        self.recoveries += 1

    def step(self, action_idx) -> tuple[np.ndarray, float, bool, dict]:
        a = int(action_idx)
        ctl = rlpolicy.controls_for(a)
        v = self.vehicle

        phi0 = self._potential(self._index())
        ds_total = 0.0
        hit_wall = False
        for _ in range(ACTION_REPEAT):
            v.step(ctl, self.dt, self.surface)
            self.lap_time += self.dt
            if not np.isfinite(v.pos).all():
                return (np.zeros(self.obs_dim, np.float32), 0.0, True,
                        {"reason": "diverged", "truncated": False})
            i = self._index()
            s = float(self.line.arclen[i])
            step_ds = (s - self._last_s + self.line.length * 1.5) \
                % self.line.length - self.line.length * 0.5
            self._last_s = s
            ds_total += step_ds
            self.speed_sum += v.speed
            hit_wall = hit_wall or v.hit_wall
            if not v.on_track:
                self.off_steps += 1
                self.off_time += self.dt
            else:
                self.off_time = 0.0
            if v.speed < STALL_SPEED:
                self.stall_time += self.dt
            else:
                self.stall_time = 0.0

        self.steps += 1
        self.progress += ds_total
        self.frontier = min(max(self.frontier,
                                self.start_s + self.progress),
                            self.line.length)
        self.prev_actions = self.prev_actions[1:] + [a]

        i = self._index()
        n = self.track.count
        if 0.4 * n <= i <= 0.6 * n:
            self._armed = True
        elif self._armed and i < 0.1 * n and ds_total > 0:
            self._armed = False
            self.laps += 1
        if hit_wall:
            self.wall_steps += 1

        # ---- reward: progress, minus time -----------------------------
        reward = (config.RL_PROGRESS_W * ds_total
                  - config.RL_TIME_W * (ACTION_REPEAT * self.dt))

        # ---- a mistake is recovered, not terminal ----------------------
        reason = ""
        back = ds_total < -2.0
        if hit_wall:
            reason = "wall"
        elif self.off_time > OFF_TRACK_PATIENCE:
            reason = "off track"
        elif self.stall_time > STALL_PATIENCE:
            reason = "stalled"
        elif back:
            reason = "wrong way"
        if reason:
            self._recover(i)

        # ---- potential shaping, skipped across a recovery so the teleport
        #      onto the line is not itself a reward -----------------------
        reward += 0.0 if reason else (self._potential(i) - phi0)

        done = self.lap_time >= EPISODE_SECONDS
        return self.observe(), float(reward), done, {
            "reason": reason or ("time" if done else ""),
            "ds": ds_total, "truncated": done}
