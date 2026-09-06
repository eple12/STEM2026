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
                       OBS_DIM, OBS_DIM_SAC, SAC_ACT_DIM, observe,
                       observe_sac)
from .surface import Surface
from .trackdata import Track, load_track
from .vehicle import Controls, Vehicle

#: Wall-clock seconds per episode. Truncation only -- the episode never ends
#: early. Long enough that a grid start is a whole Monza lap (~102 s at the
#: target pace) plus margin, so ``reach`` reads directly as lap progress and
#: the final sector is trained from grid starts, not only the scattered ones.
EPISODE_SECONDS = 130.0

#: Seconds continuously off the track before the car is recovered onto the
#: line. Zero tolerance (tried once) converged too slowly and never drove
#: ``rec`` to 0 during training -- every ordinary kerb wobble mid-exploration
#: was instantly a full recovery, which is a lot of noise to learn through.
#: Back to a real, if short, grace window: a brief excursion still costs the
#: per-second off-track fee below the whole time, it just is not *also*
#: teleported and fined the recovery charge unless it actually lingers.
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
                 start_at_line: float = 0.15, continuous: bool = False):
        #: Continuous mode swaps the discrete IQN action/observation for the
        #: SAC pair -- a [steer, pedal] vector and the rangefinder-based
        #: observation from the Gran Turismo Sport SAC paper.
        self.continuous = continuous
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

        if continuous:
            self.obs_dim, self.act_dim = OBS_DIM_SAC, SAC_ACT_DIM
        else:
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
        self.prev_cont = np.zeros(2, np.float32)

    def _index(self):
        i, _ = self.surface.progress(self.vehicle.pos)
        return i

    def _potential(self, i: int) -> float:
        off_signed = float(np.dot(self.vehicle.pos - self.line.center[i],
                                  self.line.normal[i]))
        off = abs(off_signed)
        phi = -config.RL_LINE_K * min(max(off, config.RL_LINE_LO),
                                      config.RL_LINE_HI)
        # Width-aware room to the white line on the side the car is leaning.
        # Positive while inside, drops to 0 at the line; the potential rises
        # with it so approaching the edge is a downhill step and pulling back
        # is uphill -- a learning gradient exactly where the centreline term
        # has gone flat.
        edge = float(self.track.w_right[i] if off_signed > 0.0
                     else self.track.w_left[i])
        phi += config.RL_EDGE_K * min(max(edge - off, 0.0), config.RL_EDGE_MARGIN)
        return phi

    def observe(self) -> np.ndarray:
        i = self._index()
        if self.continuous:
            return observe_sac(self.vehicle, self.track, self.line,
                               self.v_ref, i, self.prev_cont, self.surface)
        return observe(self.vehicle, self.track, self.line, self.v_ref,
                       i, self.prev_actions, self.look_idx)

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
            speed_frac = float(self.rng.uniform(0.55, 0.95))
        else:
            i = 0
            # The grid start has to cover the launch itself, not just an
            # already-rolling car: the real race always begins from a dead
            # stop and accelerates through the countdown, a state v4 never
            # trained on -- and it grazed the wall on exactly that lap while
            # every synthetic (already-at-speed) start it was scored against
            # stayed clean. 0.0 is in range on purpose.
            speed_frac = float(self.rng.uniform(0.0, 0.95))
        self._place(i, speed_frac)
        self.start_s = self._last_s
        return self.observe()

    def reset_grid(self, speed_frac: float = 0.0) -> np.ndarray:
        """A deterministic grid start at a chosen launch speed, for evaluation.

        No RNG involved, so calling this with the same argument always plays
        out the same lap -- which is the point: ``0.0`` is exactly what the
        game's own standing start is (``Vehicle.place`` zeroes velocity), so
        this is what should decide the ``*_best`` checkpoint, not an average
        over ``reset()``'s randomised launch speed.
        """
        self._reset_counters()
        self._place(0, speed_frac)
        self.start_s = self._last_s
        return self.observe()

    def _recover(self, i: int):
        """Back onto the centreline here, slow, episode still running."""
        self._place(i, config.RL_RECOVER_FRAC)
        self.recoveries += 1

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        v = self.vehicle
        if self.continuous:
            a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            steer, pedal = float(a[0]), float(a[1])
            ctl = Controls(throttle=max(pedal, 0.0), brake=max(-pedal, 0.0),
                           steer=steer, analog_steer=True)
        else:
            a = int(action)
            ctl = rlpolicy.controls_for(a)

        phi0 = self._potential(self._index())
        ds_total = 0.0
        hit_wall = False
        ke_at_wall = 0.0
        off_secs = 0.0
        off_dist_speed = 0.0
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
            self.speed_sum += v.speed
            if v.hit_wall:
                hit_wall = True
                ke_at_wall = max(ke_at_wall, v.speed * v.speed)
            if not v.on_track:
                self.off_steps += 1
                self.off_time += self.dt
                off_secs += self.dt
                off_dist_speed += v.speed
                # Progress off the asphalt earns nothing, discrete or not. The
                # discrete run used to credit it, and the policy learned the
                # runoff was free speed -- it would send a corner wide and take
                # the metres. Now a wide line is a metre-for-metre loss.
            else:
                self.off_time = 0.0
                ds_total += step_ds
            if v.speed < STALL_SPEED:
                self.stall_time += self.dt
            else:
                self.stall_time = 0.0

        self.steps += 1
        self.progress += ds_total
        self.frontier = min(max(self.frontier,
                                self.start_s + self.progress),
                            self.line.length)
        if self.continuous:
            self.prev_cont = a
        else:
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
        # Continuous (SAC) drops the flat time penalty -- GT Sport SAC has none;
        # the discount already makes a faster policy score higher, and the flat
        # penalty made every step negative and flattened the throttle gradient
        # into a do-nothing collapse. Discrete (IQN) keeps it: its near-1
        # discount needs the explicit "faster is better".
        reward = config.RL_PROGRESS_W * ds_total
        if not self.continuous:
            reward -= config.RL_TIME_W * (ACTION_REPEAT * self.dt)

        # ---- off-course penalty, per second of a wheel off the asphalt,
        #      scaled by speed. GT Sophy's off_course_penalty. Now on for the
        #      discrete run too: with progress no longer credited off the
        #      asphalt (above), this is what turns "run wide" from merely
        #      un-rewarded into a net loss, so the policy stops using the
        #      runoff as line and keeps all four wheels inside the white.
        if off_secs > 0.0:
            cost_w = (config.RL_OFF_COURSE_COST if self.continuous
                      else config.RL_OFF_TRACK_COST)
            reward -= cost_w * off_dist_speed * self.dt

        # ---- a mistake is recovered, not terminal, but it is charged for.
        # The discrete run drove clean at its peak but drifted back to ~2 wall
        # contacts a lap because a crash only cost the *time* of the recovery
        # (respawn at a fraction of speed). Now "crash, respawn, carry on" is
        # made strictly worse than not crashing: a flat charge plus one scaled
        # by the speed carried in, so braking for a corner beats sending it and
        # bouncing off. Progress off the asphalt already earns nothing, so a
        # slide is loss on both sides.
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
            # Charge for the *reckless* recoveries -- into a wall, off the
            # track, turned around. A stall is the opposite mistake (too
            # timid) and charging it as well drove the first attempt into a
            # brake-stall-recover doom loop, so it is only recovered, not
            # fined, and it gets more speed back so it does not re-stall on
            # the spot.
            if reason == "stalled":
                self._place(i, 0.45)
                self.recoveries += 1
            else:
                entry = math.sqrt(ke_at_wall) if hit_wall else self.vehicle.speed
                reward -= (config.RL_RECOVER_COST
                           + config.RL_RECOVER_SPEED_COST
                           * min(entry / config.MAX_SPEED, 1.0))
                self._recover(i)

        # ---- potential shaping, skipped across a recovery so the teleport
        #      onto the line is not itself a reward -----------------------
        reward += 0.0 if reason else (self._potential(i) - phi0)

        done = self.lap_time >= EPISODE_SECONDS
        return self.observe(), float(reward), done, {
            "reason": reason or ("time" if done else ""),
            "ds": ds_total, "truncated": done}
