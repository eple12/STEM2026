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

#: ...and this has to grow with the circuit. The clean-lap bonus only fires
#: when an episode contains TWO finish-line crossings, so the window has to
#: hold a whole lap on top of wherever the car happened to start. At Monza
#: (driven lap ~94 s) 130 s leaves the last ~28 % of the lap able to close
#: one; hand the same 130 s to Spa, which is 21 % longer, and only the last
#: ~12 % can, which quietly halves the reward the v26 recipe is built on.
#: Sized at ~1.38x the DRIVEN lap, estimated as 0.78x the modelled lap from
#: ``speed_profile`` (the ratio measured at Monza: 94 s driven vs 120.8 s
#: modelled) -- i.e. ~1.076x the modelled lap. Monza stays at exactly 130 s
#: so every earlier run remains comparable, and Spa keeps the value it was
#: actually trained and evaluated at (160 vs the 160.3 the formula below
#: would now give it -- close enough that this is about not disturbing
#: history, not a correction).
#:
#: 2026-09-13: found the hard way on Silverstone (no entry here, so it fell
#: back to the flat EPISODE_SECONDS -- 130 s, silently Monza-shaped): a
#: window sized for barely ONE driven lap plus closing margin means a grid
#: start reaches the final sector with only ~12-19 s left in the episode --
#: nowhere near enough exposure to "already at race pace, arriving at this
#: corner mid-lap" for the network to actually learn it. The trainer's own
#: eval, run in that same short window, reported PERFECT because it never
#: got far enough to test the part that was undertrained -- a genuine flying
#: lap attempt (a full, uninterrupted loop) then goes off exactly there.
#: RaceEnv/VecRaceEnv's fallback for a circuit missing here (any new one --
#: Silverstone, say, no manual step needed) is now sized for TWO full driven
#: laps plus the same closing margin as before, not one: (2 + 0.38) x 0.78 =
#: ~1.856x that circuit's own model_lap. A cleanly-driving policy gets a full
#: second lap, at pace, run in from every corner it just took -- not just
#: enough room to limp across the line once. Spa and Monza are left at their
#: historical 1-lap-sized values below (already trained and deployed at
#: those numbers; retraining either at the new size is a separate decision,
#: not implied by fixing the fallback new circuits get).
EPISODE_SECONDS_BY_CIRCUIT = {
    "Spa": 160.0,          # modelled 149.0 s -> driven ~116 s
    "Monza": 130.0,        # pinned to EPISODE_SECONDS's own historical value
}

#: The POST-PERFECT episode length, as a ratio of model_lap: (2 driven laps +
#: the same closing margin as the 1-lap ratio above) x 0.78 = ~1.856. Not
#: applied at construction -- the training loop sets
#: ``env.episode_seconds = LONG_EPISODE_RATIO * env.model_lap`` itself, once,
#: the first time an eval comes back tier-0 (PERFECT). See the long comment
#: where RaceEnv.__init__ sets the initial (short) episode_seconds for why
#: this has to be conditional on actually reaching PERFECT rather than a
#: flat multiplier from the start: applied unconditionally it halves how
#: often ANY reset happens (grid or scattered) for the entire run, since
#: termination here is truncation-only, which undertrains exactly the
#: reset-frequency-dependent behaviours (standing starts, corner-exposure
#: diversity) a still-converging policy needs most.
LONG_EPISODE_RATIO = 1.856

#: train_iqn_gpu.py climbs through these one entry at a time (widen_stage in
#: harvest()), each unlocked by a fresh PERFECT at the current one, ending
#: at LONG_EPISODE_RATIO. 2026-09-15: tried staged -- (1.3, 1.6,
#: LONG_EPISODE_RATIO), a 3-step climb instead of one jump -- on the theory
#: that a smaller per-step difficulty increase would make each individual
#: re-clear more likely. Result (Monza) read as a small, possibly-noise
#: improvement over the one-shot jump, not clearly worth the extra
#: complexity -- reverted to a single entry (one jump, same as before
#: staging existed) so the curriculum surface stays at just "widen" while a
#: reward-side fix (RL_RL_LAP_OFF_TOL, config.py) and a robustness gate on
#: the trigger itself (--widen-after-perfects, train_iqn_gpu.py) are tried
#: instead. Multi-entry still works if revisited -- nothing else needed
#: changing to go back to staged, just this tuple.
EPISODE_WIDEN_STAGES = (LONG_EPISODE_RATIO,)

#: Seconds continuously off the track before the car is recovered onto the
#: line. Zero tolerance (tried once) converged too slowly and never drove
#: ``rec`` to 0 during training -- every ordinary kerb wobble mid-exploration
#: was instantly a full recovery, which is a lot of noise to learn through.
#: Back to a real, if short, grace window: a brief excursion still costs the
#: per-second off-track fee below the whole time, it just is not *also*
#: teleported and fined the recovery charge unless it actually lingers.
#: v14: back to 0.45 (the v10/v11 value; v13 tried v5's 0.7 and ran wide).
OFF_TRACK_PATIENCE = 0.45

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
                 start_at_line: float = 0.15, continuous: bool = False,
                 focus_window: tuple[float, float] | None = None,
                 adaptive_reset: bool = False):
        #: Continuous mode swaps the discrete IQN action/observation for the
        #: SAC pair -- a [steer, pedal] vector and the rangefinder-based
        #: observation from the Gran Turismo Sport SAC paper.
        self.continuous = continuous
        self.track: Track = load_track(circuit)
        #: Per-env so a longer circuit gets a longer window; the trace tools
        #: raise it on the instance to watch several laps. Set for real once
        #: model_lap is known, below -- this placeholder only matters if
        #: something reads it before then, which nothing does.
        self.episode_seconds = EPISODE_SECONDS_BY_CIRCUIT.get(
            circuit, EPISODE_SECONDS)
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
        #: Multiplier on config.RL_OFF_EVENT_COST (the trainer's ramp); 1.0 in the game.
        self.off_event_scale = 1.0
        #: Restricts the SCATTERED (non-grid) reset index to a fraction-of-
        #: lap window (lo, hi), lo/hi wrapping past 1.0 for a window that
        #: crosses the finish line. None = anywhere on the lap (default).
        #: For sector-focused training: bias the buffer toward one hard
        #: sector without special-casing any one circuit -- fractions carry
        #: over to any track, unlike a hardcoded index range. Give the
        #: window some margin past the sector's own boundary on each side
        #: (the caller's job) so the policy sees a realistic spread of
        #: arrival states at the seam rather than only the sector's own
        #: interior -- training a sector on a single fixed entry state and
        #: splicing it to a neighbour trained the same way is where the
        #: seam mismatch this is meant to avoid actually comes from.
        #: Grid starts (start_at_line) are untouched, so full-lap coherence
        #: (the clean-lap bonus, whole-lap credit assignment) keeps getting
        #: *some* signal even while most resets concentrate on the sector.
        self.focus_window = focus_window
        #: See config.RL_ADAPTIVE_* -- mirrors VecRaceEnv.adaptive_reset.
        #: Mutually exclusive with focus_window (the caller's job).
        self.adaptive_reset = adaptive_reset
        if adaptive_reset:
            nb = config.RL_ADAPTIVE_BUCKETS
            self.n_buckets = nb
            self.bucket_fail_ema = np.zeros(nb, dtype=np.float64)

        self.line = rlpolicy.reference_line(self.track)
        # Progress and the lookahead geometry stay on the centreline. The
        # potential attractor and the reference-speed profile follow
        # shape_line, which is the optimised raceline (pulled inside the white
        # line) when RL_SHAPE_TO_RACELINE is on, else the centreline again.
        self.shape_line = rlpolicy.shaping_line(self.track)
        # v24: v_ref is the profile at the BASELINE pace; the training
        # curriculum scales it up over time via ``pace_mult`` (see step()).
        # RL_RL_SPEED_PACE is that baseline -- attainable, so the policy
        # converges clean against it fast (v19-style) before the ramp bites.
        _vpace = (config.RL_RL_SPEED_PACE
                  if getattr(config, "RL_TASK", "linesight") == "raceline"
                  else 1.0)
        self.v_ref = rl.speed_profile(self.shape_line.seg_len,
                                      self.shape_line.curvature,
                                      self.shape_line.curv_radius, _vpace)
        # The clean-lap bonus's break-even point: this circuit's own analytic
        # modelled lap time (same "time over each segment at the mean of its
        # end speeds" raceline.lap_time() uses), times RL_RL_LAP_W. See the
        # long comment above RL_RL_LAP_W in config.py for why this has to be
        # per-circuit rather than a flat constant.
        v_seg = np.maximum(0.5 * (self.v_ref + np.roll(self.v_ref, -1)), 1e-3)
        self.model_lap = float(np.sum(self.shape_line.seg_len / v_seg))
        self.lap_base = config.RL_RL_LAP_W * self.model_lap
        # EPISODE_SECONDS_BY_CIRCUIT was a hand-tuned override per circuit --
        # exactly the silent single-track calibration RL_RL_LAP_W's own
        # comment already flags for lap_base, just for this constant instead.
        # A circuit that never gets an entry falls back to EPISODE_SECONDS
        # (130 s, Monza-shaped), which is too short for a longer lap.
        if circuit not in EPISODE_SECONDS_BY_CIRCUIT:
            self.episode_seconds = 1.076 * self.model_lap
        # 2026-09-13, Silverstone: widening the window UNCONDITIONALLY to
        # ~1.856x model_lap (2 driven laps + closing margin, instead of 1)
        # so a clean policy could prove a full second lap fixed the flying-
        # lap-goes-off-track symptom, but broke standing-start and early-
        # straight throttle commitment: termination here is truncation-only
        # (a recovery doesn't end an episode), so doubling episode_seconds
        # exactly halves how often ANY reset happens -- including the
        # RL_LAUNCH_STOP_FRAC-biased grid starts standing-start behaviour is
        # learned from, and the scattered starts that spread corner exposure
        # around the lap. Over the same fixed decision budget that is half
        # as much exposure to both, for a benefit (proving a full clean
        # second lap) that only a policy good enough to attempt doesn't need
        # yet -- for most of training it is pure downside.
        #
        # Fix: keep episode_seconds at the SHORT (1-lap) value set above for
        # exposure diversity through the exploration/convergence phase, and
        # only widen it once the policy has actually proven it deserves the
        # harder test -- the training loop flips a RaceEnv's/VecRaceEnv's
        # episode_seconds to LONG_EPISODE_RATIO * model_lap itself, the
        # first time an eval comes back genuinely PERFECT (tier 0), not on a
        # decision-count schedule. Exposed as a ratio, not inlined in the
        # trainer, so it stays in one place next to the short-side formula
        # above it.
        self._pace_mult = 1.0
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
        self._prev_off = False
        self.wall_steps = 0
        self.recoveries = 0
        self.speed_sum = 0.0
        self._last_s = 0.0
        self._armed = False
        self.prev_actions = [DEFAULT_ACTION] * N_PREV_ACTIONS
        self.prev_cont = np.zeros(2, np.float32)
        # v17 raceline task: off-track steps within the current lap, and the
        # lap-clock reading at the last finish crossing, for the clean lap
        # bonus. last_lap_time is the most recent completed lap (0 = none yet).
        self.lap_off_steps = 0
        self.lap_start_time = 0.0
        self.last_lap_time = 0.0
        self.best_lap_time = 0.0
        self._lap_rec0 = 0          # recoveries count at the last finish crossing

    def _index(self):
        i, _ = self.surface.progress(self.vehicle.pos)
        return i

    def _potential(self, i: int) -> float:
        # Attractor term: distance from shape_line (the raceline, or the
        # centreline if that is off). This is the only term that says *where*
        # to drive; pointing it at the fast line is what makes the shaping a
        # tailwind rather than a pull back to centre.
        a_off = float(np.dot(self.vehicle.pos - self.shape_line.center[i],
                             self.shape_line.normal[i]))
        phi = -config.RL_LINE_K * min(max(abs(a_off), config.RL_LINE_LO),
                                      config.RL_LINE_HI)
        # The edge / limit terms below stay on the centreline, where the track
        # widths are defined.
        off_signed = float(np.dot(self.vehicle.pos - self.line.center[i],
                                  self.line.normal[i]))
        off = abs(off_signed)
        # Width-aware room to the white line on the side the car is leaning.
        # Positive while inside, drops to 0 at the line; the potential rises
        # with it so approaching the edge is a downhill step and pulling back
        # is uphill -- a learning gradient exactly where the centreline term
        # has gone flat.
        edge = float(self.track.w_right[i] if off_signed > 0.0
                     else self.track.w_left[i])
        phi += config.RL_EDGE_K * min(max(edge - off, 0.0), config.RL_EDGE_MARGIN)
        # Past the white line the term above is flat, so it stops pulling
        # exactly where a track-limit step happens. Keep Phi falling with how
        # far out the car is, capped. State-only, so still optimum-preserving.
        phi -= config.RL_EDGE_OUT_K * min(max(off - edge, 0.0),
                                          config.RL_EDGE_OUT_CAP)
        return phi

    def observe(self) -> np.ndarray:
        i = self._index()
        if self.continuous:
            return observe_sac(self.vehicle, self.track, self.line,
                               self.v_ref, i, self.prev_cont, self.surface)
        return observe(self.vehicle, self.track, self.line, self.v_ref,
                       i, self.prev_actions, self.look_idx, self.surface)

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
            n = self.track.count
            if self.adaptive_reset:
                nb = self.n_buckets
                fe = self.bucket_fail_ema
                total = fe.sum()
                norm = fe / total if total > 1e-8 else np.full(nb, 1.0 / nb)
                weights = (config.RL_ADAPTIVE_FLOOR / nb
                          + (1.0 - config.RL_ADAPTIVE_FLOOR) * norm)
                weights /= weights.sum()
                bkt = int(self.rng.choice(nb, p=weights))
                per_bucket = max(n // nb, 1)
                lo_idx = bkt * per_bucket
                span_idx = n - lo_idx if bkt == nb - 1 else per_bucket
                i = min(lo_idx + int(self.rng.random() * span_idx), n - 1)
            elif self.focus_window is not None:
                lo, hi = self.focus_window
                span = (hi - lo) % 1.0 or 1.0
                i = int((lo + self.rng.random() * span) % 1.0 * n)
            else:
                i = int(self.rng.integers(n))
            speed_frac = float(self.rng.uniform(0.55, 0.95))
        else:
            i = 0
            # The grid start has to cover the launch itself, not just an
            # already-rolling car: the real race always begins from a dead
            # stop and accelerates through the countdown, a state v4 never
            # trained on -- and it grazed the wall on exactly that lap while
            # every synthetic (already-at-speed) start it was scored against
            # stayed clean. 0.0 is in range on purpose.
            #
            # But it was barely IN range: uniform(0, 0.95) puts a true
            # near-standing launch (speed_frac < ~0.05) in about 1 in 20 grid
            # starts, and grid starts are themselves only start_at_line of
            # all resets (~20%) -- under 1% of everything the network ever
            # sees. A launch-specific behaviour (commit to full throttle
            # immediately) gets correspondingly little gradient signal.
            # RL_LAUNCH_STOP_FRAC of grid starts now draw from a narrow
            # near-zero band instead, so standing starts are seen often
            # enough to be learned rather than merely tolerated.
            if self.rng.random() < config.RL_LAUNCH_STOP_FRAC:
                speed_frac = float(self.rng.uniform(0.0, config.RL_LAUNCH_STOP_MAX))
            else:
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

    def step(self, action, off_cost_scale: float = 1.0,
             pace_mult: float = 1.0, lap_w_mult: float = 1.0
             ) -> tuple[np.ndarray, float, bool, dict]:
        # ``off_cost_scale`` is the training curriculum's multiplier on the
        # per-second off-track fee (config.RL_OFF_TRACK_COST_RAMP). 1.0 for the
        # game and for eval; train_iqn.py passes the ramped value.
        # ``pace_mult`` (v24) is the training curriculum's multiplier on the
        # dense-speed-reward target profile: it ramps 1.0 -> ~1.28 over training
        # so the policy first converges clean against an attainable target (as
        # v19 did, fast) and then is continuously pulled toward a faster line
        # instead of parking at one comfortable speed (v19's 181 km/h stall).
        # 1.0 for the game and for eval.
        self._pace_mult = float(pace_mult)
        v = self.vehicle
        act_jerk = 0.0
        if self.continuous:
            a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            act_jerk = float(np.sum((a - self.prev_cont) ** 2))
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
        off_depth_speed = 0.0        # sum of (metres past the white line) * speed
        wall_dist_speed = 0.0        # soft-barrier accumulator (RL_EDGE_WALL_*)
        off_events = 0               # excursions started this decision (RL_OFF_EVENT_COST)
        if self.adaptive_reset:
            bucket_off = np.zeros(self.n_buckets)
            bucket_visit = np.zeros(self.n_buckets)
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
            if self.adaptive_reset:
                bkt = min(i * self.n_buckets // self.track.count,
                         self.n_buckets - 1)
                bucket_visit[bkt] += 1.0
                if not v.on_track:
                    bucket_off[bkt] += 1.0
            if v.hit_wall:
                hit_wall = True
                ke_at_wall = max(ke_at_wall, v.speed * v.speed)
            if config.RL_EDGE_WALL_COST > 0.0:
                ws = float(np.dot(v.pos - self.line.center[i],
                                  self.line.normal[i]))
                we = float(self.track.w_right[i] if ws > 0.0
                           else self.track.w_left[i])
                wd = max((abs(ws) - we) - config.RL_EDGE_WALL_START, 0.0)
                wall_dist_speed += min(wd / config.RL_EDGE_WALL_WIDTH,
                                       config.RL_EDGE_WALL_CAP) ** 2 * v.speed
            if not v.on_track:
                if not self._prev_off:
                    off_events += 1
                self._prev_off = True
                self.off_steps += 1
                self.lap_off_steps += 1
                self.off_time += self.dt
                off_secs += self.dt
                off_dist_speed += v.speed
                off_signed = float(np.dot(v.pos - self.line.center[i],
                                          self.line.normal[i]))
                edge = float(self.track.w_right[i] if off_signed > 0.0
                             else self.track.w_left[i])
                off_depth_speed += max(abs(off_signed) - edge, 0.0) * v.speed
                # Progress off the asphalt earns nothing, discrete or not. The
                # discrete run used to credit it, and the policy learned the
                # runoff was free speed -- it would send a corner wide and take
                # the metres. Now a wide line is a metre-for-metre loss.
            else:
                self._prev_off = False
                self.off_time = 0.0
                ds_total += step_ds
            if v.speed < STALL_SPEED:
                self.stall_time += self.dt
            else:
                self.stall_time = 0.0

        if self.adaptive_reset:
            visited = bucket_visit > 0
            rate = np.where(visited, bucket_off / np.maximum(bucket_visit, 1.0),
                            self.bucket_fail_ema)
            decay = config.RL_ADAPTIVE_EMA
            self.bucket_fail_ema = np.where(
                visited, self.bucket_fail_ema * decay + rate * (1.0 - decay),
                self.bucket_fail_ema)

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
        lap_bonus = 0.0
        if 0.4 * n <= i <= 0.6 * n:
            self._armed = True
        elif self._armed and i < 0.1 * n and ds_total > 0:
            self._armed = False
            self.laps += 1
            # v17: a completed lap. The first crossing of the episode only
            # starts the clock (the run into it is a partial lap); every
            # crossing after that closes a real, whole lap.
            if self.lap_start_time > 0.0:
                lap_s = self.lap_time - self.lap_start_time
                self.last_lap_time = lap_s
                # 2026-09-15: tried replacing this binary threshold with a
                # continuous off_decay (RL_LAP_OFF_DECAY_CAP), on the theory
                # that a hard cutoff anywhere creates the same all-or-
                # nothing-bet problem the threshold itself was already
                # flagged for once (see RL_RL_LAP_OFF_TOL's own history).
                # Reverted the same day: two Monza runs (binary 0-tolerance
                # vs continuous decay) produced near-identical training
                # trajectories (eval reach within 100 m of each other at
                # matching iterations) -- the lap bonus fires once per lap,
                # sparse next to the dense per-step progress/off-track
                # terms, so reshaping it alone was not reaching the policy's
                # moment-to-moment decisions either way. A dense per-step
                # alternative (RL_DIRTY_LAP_PROGRESS_MULT) was tried next
                # and diverged worse, not better. No confirmed replacement
                # for this simple version exists, so back to it -- this is
                # the exact mechanism behind the current Monza record
                # (91.30 s, Monza_v26k_scratch), confirmed working.
                clean = (self.lap_off_steps <= config.RL_RL_LAP_OFF_TOL
                         and self.recoveries == self._lap_rec0)
                if clean:
                    if self.best_lap_time == 0.0 or lap_s < self.best_lap_time:
                        self.best_lap_time = lap_s
                    if getattr(config, "RL_TASK", "linesight") == "raceline":
                        w_eff = config.RL_RL_LAP_W * lap_w_mult
                        lap_bonus = max(
                            0.0, w_eff * (self.model_lap - lap_s))
            self.lap_start_time = self.lap_time
            self.lap_off_steps = 0
            self._lap_rec0 = self.recoveries
        if hit_wall:
            self.wall_steps += 1

        # ---- reward: progress, minus time -----------------------------
        # Continuous (SAC) drops the flat time penalty -- GT Sport SAC has none;
        # the discount already makes a faster policy score higher, and the flat
        # penalty made every step negative and flattened the throttle gradient
        # into a do-nothing collapse. Discrete (IQN) keeps it: its near-1
        # discount needs the explicit "faster is better".
        # 2026-09-15: tried multiplying this by RL_DIRTY_LAP_PROGRESS_MULT
        # once a lap had gone off-track even once (a dense, every-decision
        # version of the sparse lap-bonus reshaping above, on the theory
        # that the dense signal actually reaches the policy) -- reverted
        # the same day, it diverged worse rather than better. Reverting to
        # the exact reward this project's Monza record (91.30 s,
        # Monza_v26k_scratch) was produced under, pending a confirmed
        # replacement rather than another guess.
        reward = config.RL_PROGRESS_W * ds_total
        if not self.continuous:
            reward -= config.RL_TIME_W * (ACTION_REPEAT * self.dt)
        else:
            reward -= config.RL_SAC_TIME_W * (ACTION_REPEAT * self.dt)

        # ---- raceline task: dense speed reward + clean-lap bonus.
        # v25b: measurement killed the two-sided bell. v_ref (speed_profile on
        # the precomputed raceline, with 0.62x brake force, a 28 m min-radius
        # window and a chicane-swing cut) is NOT a real limit -- the fast v18
        # policy drives ABOVE it on 51% of the lap and at ~1.3x (p90 2.6x) in
        # corners, all perfectly clean. A bell that DROPS above v_ref was
        # therefore penalising v18-competitive cornering -- the reason every
        # bell version (v19, v24) topped out ~10-25 km/h under v18.
        #  (1) ONE-SIDED-BELOW bell: full dense "you are under the profile,
        #      get up to it" signal below v_ref (this is the convergence-speed
        #      part), FLAT above it -- never a penalty for legit speed.
        #  (2) small PURE-LINEAR pull normalised by a FIXED speed (MAX_SPEED),
        #      not v_ref -- so it has no per-corner shape, just "a bit faster
        #      is a bit better, everywhere, equally". The actual corner
        #      discipline is the off-track / recovery penalty below (v18's
        #      mechanism, proven).
        if getattr(config, "RL_TASK", "linesight") == "raceline" \
                and not self.continuous:
            vtgt = max(float(self.v_ref[i]), 1.0)
            spd_err = min(0.0, (self.vehicle.speed - vtgt)
                          / config.RL_RL_SPEED_SIG)
            reward += config.RL_RL_SPEED * math.exp(-spd_err * spd_err)
            reward += config.RL_RL_SPEED_LIN * min(
                self.vehicle.speed / config.MAX_SPEED,
                config.RL_RL_SPEED_LIN_CAP)
            reward += lap_bonus

        # ---- off-course penalty, per second of a wheel off the asphalt,
        #      scaled by speed. GT Sophy's off_course_penalty. Now on for the
        #      discrete run too: with progress no longer credited off the
        #      asphalt (above), this is what turns "run wide" from merely
        #      un-rewarded into a net loss, so the policy stops using the
        #      runoff as line and keeps all four wheels inside the white.
        if off_secs > 0.0:
            if self.continuous:
                # v15: a flat per-second-off fee plus one that grows with how
                # far past the white line the car is. Near the line it is
                # almost free (the racing line uses every centimetre); a real
                # excursion is a steep, escalating loss. This is the term that
                # has to make PERFECT beat fast-but-wide for the SAC run.
                reward -= (config.RL_SAC_OFF_BASE * off_dist_speed
                           + config.RL_SAC_OFF_DEPTH * off_depth_speed) * self.dt
            else:
                reward -= (config.RL_OFF_TRACK_COST * off_cost_scale
                           * off_dist_speed * self.dt)

        if config.RL_EDGE_WALL_COST > 0.0:
            reward -= config.RL_EDGE_WALL_COST * wall_dist_speed * self.dt
        if config.RL_OFF_EVENT_COST > 0.0 and off_events:
            reward -= (config.RL_OFF_EVENT_COST * self.off_event_scale
                       * off_events)

        # ---- action-smoothness penalty (continuous only): ||a_t - a_{t-1}||^2.
        # Without it the SAC actor chatters the steering -- a fast oscillation
        # that averages to the right angle but unsettles the car and reads as a
        # nervous driver. GT Sophy carries the same term.
        if self.continuous:
            reward -= config.RL_SAC_ACT_SMOOTH * act_jerk

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
        terminal = False
        if reason and self.continuous:
            # v15c: a mistake ENDS the episode for the continuous run -- no
            # teleport. The recover mechanism buried the twin-Q critic under
            # non-Markovian "action -> teleport -> random state" transitions
            # and it flatlined. A clean terminal with a speed-scaled charge is
            # standard RL and SAC handles it. The policy learns to not crash,
            # not to crash-and-carry-on -- which is what a PERFECT lap needs.
            entry = math.sqrt(ke_at_wall) if hit_wall else self.vehicle.speed
            reward -= (config.RL_SAC_CRASH_COST
                       + config.RL_SAC_CRASH_SPEED_COST
                       * min(entry / config.MAX_SPEED, 1.0))
            terminal = True
        elif reason:
            # Discrete run: recovered, not terminal, but charged for.
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

        truncated = self.lap_time >= self.episode_seconds
        done = truncated or terminal
        return self.observe(), float(reward), done, {
            "reason": reason or ("time" if truncated else ""),
            "ds": ds_total, "truncated": truncated, "terminal": terminal}
