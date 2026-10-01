"""The observation and the trained network, shared by training and the game.

This follows the Linesight recipe (the Trackmania RL project): a **discrete**
action set and a **distributional value** network (IQN), rather than a
continuous policy. The reason is the one every continuous run here ran into --
with a Gaussian policy the exploration noise is part of the objective, so the
best *noisy* policy is a cautious one and the mean policy inherits that. A
discrete action chosen by argmax over Q-values has no such coupling: the greedy
policy can sit right on the limit while epsilon-greedy does the exploring
separately.

``observe`` lives here because the environment and the car must agree on it
exactly. A network trained on one layout of numbers and driven with another is
not a worse driver, it is a different one, and nothing about the failure says
so.

The network is a handful of small matrices; inference is numpy, so the game
never imports torch. Training exports the weights and the observation
statistics into one ``.npz`` and this reads them back.
"""
from __future__ import annotations

import math

import numpy as np

from . import config, f1tenth, raceline
from .vehicle import Controls

# --------------------------------------------------------------------------
# discrete actions. The first pass used three steer levels and three pedal
# levels (nine actions); it beat the scripted planner but plateaued because a
# binary brake cannot trail off into an apex -- it slams full, overshoots slow,
# releases, carries too much, slams again, and the trace shows that oscillation
# at every corner. Five steer levels and a light-brake level give it the
# in-between positions to modulate with. Steering is still fed with
# ``analog_steer=False`` so a held level ramps the way a key press does.
#
# v13: back to five levels (the v5 action set), on request -- testing whether
# Seven levels: the half-steps between 0 and 0.5 are where the policy kept
# nicking the white line -- it had to average two coarse levels across the
# ACTION_REPEAT window to hold a shallow angle. v17 trains from scratch, so it
# is not locked to a 20-action checkpoint.
#
# 2026-09-13: added a THIRD brake gradation (-0.7, "medium"). The v26 family
# (EDGE_MARGIN=0, several off-track/lap-bonus reward variants) all converged
# to PERFECT in the 9250-9275 m / ~122.6-123.0 s range regardless of the
# reward tuning tried, but every one of them showed the SAME four braking-
# zone speed deficits versus the older, narrower-line v26d run (60-110 km/h
# slower entering the same corners) -- the reward experiments moved other
# things but never touched that. This is the same failure this file's own
# comment above already names for a BINARY brake ("slams full, overshoots
# slow, releases, carries too much, slams again") -- two brake levels
# (light/full) still leaves a real gap between "barely slowing" and "hard
# stop" for trail-braking into a corner. Testing whether a third rung closes
# it. Changes N_ACTIONS (28->35), so this is a from-scratch run, not a
# warm-start -- the network's action head is a different size.
#
# 2026-09-13: the 3-brake run (v26g) DID fix corner entry -- 20-50 km/h
# faster into all four problem braking zones, closing most of the gap to
# v26d -- but lost it straight back out on the exit of those same corners
# (27-50 km/h slower), for a wash on lap time. Checked whether that exit
# loss was the same caution relocating (the pattern seen when EDGE_MARGIN
# went to 0): it is NOT -- cross-track offset on the slow exits was 5.1 m
# (v26g) vs 0.5 m (v26e4), i.e. v26g ran WIDER there, not more central, so
# it isn't fear of the edge. More likely: throttle is still binary (0 or
# 1.0) same as it was before the brake fix, so there is no way to roll
# progressively back onto the power while steering angle is still bleeding
# off through the exit arc -- exactly the "binary pedal can't trail into an
# apex" failure this file already diagnosed for brake, just on the other
# pedal. Added a half-throttle rung to test the same fix on exit that
# worked on entry. Changes N_ACTIONS again (35->42): another from-scratch
# run.
STEER_LEVELS = (-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0)
PEDAL_LEVELS = (1.0, 0.5, 0.0, -0.4, -0.7, -1.0)  # full/half throttle, coast, light/medium/full brake
ACTIONS = [(s, p) for p in PEDAL_LEVELS for s in STEER_LEVELS]
N_ACTIONS = len(ACTIONS)                 # 42  (7 steer x 6 pedal)
#: The action a fresh history is padded with -- straight and on the throttle.
DEFAULT_ACTION = ACTIONS.index((0.0, 1.0))

#: How many past actions the network sees. Lets it know it is mid-corner.
N_PREV_ACTIONS = 3

#: Physics ticks one chosen action is held for. Linesight holds a keypress for
#: 50 ms (five 10 ms engine steps); at 60 Hz three ticks is the same 50 ms,
#: and it keeps the decision rate -- and the episode length in decisions --
#: sane.
ACTION_REPEAT = 3

#: Arc distances ahead, in metres, at which the centreline is sampled and
#: handed to the network as points in the car's own frame. Dense near the car
#: (line precision), sparse far away (seeing a corner coming). This raw
#: geometry is Linesight's "zone centers" -- the network plans against the
#: shape of the road rather than against a scalar curvature. Used by the
#: "lookahead" observation.
LOOKAHEAD_M = (6.0, 13.0, 22.0, 33.0, 47.0, 64.0, 85.0, 112.0, 146.0, 188.0,
               240.0, 305.0, 385.0, 485.0, 610.0, 765.0)
N_LOOK = len(LOOKAHEAD_M)

#: Rangefinder fan for the "rangefinder" observation: rays cast from the car,
#: each returning distance to the *white line* in that direction -- the direct
#: "how much room do I have, and where does it run out" signal the lookahead
#: points only imply. Shared with observe_sac.
RANGE_ANGLES = tuple(np.linspace(-1.9, 1.9, 13).tolist())
N_RANGE = len(RANGE_ANGLES)
RANGE_MAX = 100.0
#: Curvature samples ahead, in seconds at the current speed (so a fast car
#: looks further down the road), for both the rangefinder obs and SAC.
CURV_AHEAD_S = (0.4, 0.8, 1.2, 1.7, 2.3, 3.0, 4.0, 5.2)
N_CURV = len(CURV_AHEAD_S)
#: Reference-speed samples ahead, same time-scaling -- "how fast should I be
#: going a second / two / four from now", the brake-for-the-corner cue that
#: the SAC obs leaves to curvature alone.
VREF_AHEAD_S = (0.6, 1.4, 2.6, 4.2)
N_VREF = len(VREF_AHEAD_S)

#: Which observation the discrete IQN uses. "rangefinder" -> white-line
#: rangefinders + curvature + reference speed ahead (36-dim). "lookahead" ->
#: centreline points in car frame, geometry only (43-dim).
RL_OBS = getattr(config, "RL_OBS", "lookahead")
_RANGEFINDER_OBS = RL_OBS == "rangefinder"

if _RANGEFINDER_OBS:
    #: 2 phase + 5 car + N_PREV_ACTIONS + 1 finish + N_RANGE + N_CURV + N_VREF
    OBS_DIM = 2 + 5 + N_PREV_ACTIONS + 1 + N_RANGE + N_CURV + N_VREF
else:
    # 2026-09-14: dropped the per-lookahead-point target speed (was a 3rd
    # N_LOOK block here, v_ref[js]/MAX_SPEED) -- with RL_SHAPE_TO_RACELINE
    # off (see its own comment in config.py) v_ref is honest again, but
    # handing its value to the network as an INPUT is still telling it how
    # fast to go rather than letting lap-time reward alone decide that --
    # exactly the crutch a line/pace-discovery run should not have. What is
    # left is pure track shape: where the road goes, not how fast to take
    # it. Changes OBS_DIM (59 -> 43), so this is a from-scratch run, not a
    # warm-start -- the network's input layer is a different size.
    #: 2 phase + 5 car + N_PREV_ACTIONS + 1 finish + 2*N_LOOK
    OBS_DIM = 2 + 5 + N_PREV_ACTIONS + 1 + 2 * N_LOOK
ACT_DIM = N_ACTIONS

#: IQN network sizes, mirroring Linesight scaled down (no image head).
FLOAT_HIDDEN = 256
HEAD_HIDDEN = 256
IQN_EMBED = 64
IQN_K = 32                              # quantiles averaged at inference (Linesight)


def controls_for(action_idx: int) -> Controls:
    steer, pedal = ACTIONS[int(action_idx)]
    return Controls(throttle=max(pedal, 0.0), brake=max(-pedal, 0.0),
                    steer=steer, analog_steer=False)


# --------------------------------------------------------------------------
def reference_line(track):
    """The centreline, sampled and lightly smoothed. Progress and the
    observation geometry are measured against this -- the plain centre of the
    track -- regardless of RL_SHAPE_TO_RACELINE."""
    return raceline.Line(track, np.zeros(track.count),
                         smooth=config.RL_CURVATURE_SMOOTH)


def shaping_line(track):
    """The attractor for the potential term and the source of the reference
    speed profile. This project's own optimised raceline
    (assets/racelines/<Circuit>.npy) pulled inward by RL_RACELINE_SAFETY so the
    whole car body stays inside the white line, or the centreline if none is
    saved or RL_SHAPE_TO_RACELINE is off. The edge/limit terms still read the
    centreline, so a raceline that grazes a kerb cannot license an off-track
    step here."""
    if getattr(config, "RL_SHAPE_TO_RACELINE", False):
        saved = raceline.load(track)
        if saved is not None and len(saved.offset) == track.count:
            off = np.asarray(saved.offset, dtype=float)
            edge = np.where(off > 0.0, track.w_right, track.w_left)
            room = np.maximum(
                edge - config.BODY_HALF_WIDTH - config.RL_RACELINE_SAFETY, 0.0)
            off = np.clip(off, -room, room)
            return raceline.Line(track, off, smooth=config.RL_CURVATURE_SMOOTH)
    return reference_line(track)


def _lookahead_indices(line):
    """Sample indices at LOOKAHEAD_M metres ahead of every sample, precomputed
    once per line (a (count, N_LOOK) table)."""
    arc = line.arclen
    n = len(arc)
    tgt = (arc[:, None] + np.asarray(LOOKAHEAD_M)[None, :]) % line.length
    return np.searchsorted(arc, tgt).clip(0, n - 1)


def _ahead_index(line, i: int, metres: float) -> int:
    arc = line.arclen
    return int(np.searchsorted(arc, (arc[i] + metres) % line.length)
               % len(line.center))


def observe(vehicle, track, line, v_ref, i: int, prev_actions,
            look_idx=None, surface=None) -> np.ndarray:
    """Track-relative view of the world, plus lap phase and recent inputs.

    Everything is in the car's frame or measured against the line, so the
    numbers mean the same thing at every corner. Lap phase (sin/cos) is the
    deliberate exception: the brief is to master *this* circuit, and a network
    that knows where it is can brake "here, at this corner" instead of
    re-deriving it from geometry every lap.

    The tail of the vector is one of two shapes (see ``RL_OBS``):
    "rangefinder" -- white-line distances in a fan of directions, curvature
    ahead, reference speed ahead; "lookahead" -- centreline points in the car
    frame only, no speed -- track shape is the only hint given about how
    fast to go, so the pace itself is the policy's own to find (see the
    long comment where N_LOOK's slice of OBS_DIM is defined).
    """
    fwd = np.array([math.sin(vehicle.yaw), math.cos(vehicle.yaw)])
    right = np.array([math.cos(vehicle.yaw), -math.sin(vehicle.yaw)])

    cross = float(np.dot(vehicle.pos - line.center[i], line.normal[i]))
    line_yaw = math.atan2(line.tangent[i, 0], line.tangent[i, 1])
    heading_err = (line_yaw - vehicle.yaw + math.pi) % (2 * math.pi) - math.pi
    half = float(track.w_right[i] if cross > 0 else track.w_left[i])
    phase = 2.0 * math.pi * line.arclen[i] / max(line.length, 1e-6)

    obs = [
        math.sin(phase), math.cos(phase),
        float(np.dot(vehicle.vel, fwd)) / config.MAX_SPEED,
        float(np.dot(vehicle.vel, right)) / 20.0,
        vehicle.yaw_rate / 3.0,
        cross / max(half, 1e-3),
        heading_err / 0.6,
    ]
    for a in prev_actions:
        obs.append((a - (N_ACTIONS - 1) / 2.0) / ((N_ACTIONS - 1) / 2.0))
    obs.append((line.length - line.arclen[i]) / line.length)

    if _RANGEFINDER_OBS:
        rng = surface.rangefinders(vehicle.pos, vehicle.yaw, RANGE_ANGLES,
                                   RANGE_MAX) / RANGE_MAX
        obs.extend(rng.tolist())
        speed = max(float(vehicle.speed), 1.0)
        for sec in CURV_AHEAD_S:
            j = _ahead_index(line, i, float(np.clip(sec * speed, 5.0, 350.0)))
            obs.append(float(line.curvature[j]) * 60.0)
        for sec in VREF_AHEAD_S:
            j = _ahead_index(line, i, float(np.clip(sec * speed, 5.0, 400.0)))
            obs.append(float(v_ref[j]) / config.MAX_SPEED)
    else:
        if look_idx is None:
            look_idx = _lookahead_indices(line)
        js = look_idx[i]
        d = line.center[js] - vehicle.pos                   # (N_LOOK, 2)
        obs.extend((d @ right / 60.0).tolist())             # lateral, car frame
        obs.extend((d @ fwd / 120.0).tolist())              # forward, car frame
    return np.asarray(obs, dtype=np.float32)


# --------------------------------------------------------------------------
def _leaky(x, s=0.01):
    return np.where(x > 0.0, x, x * s)


def iqn_embedding(w, k: int = IQN_K):
    """The quantile embedding. Inference uses fixed quantiles, so this is a
    constant of the weights -- worth computing once rather than per decision."""
    tau = (np.linspace(0.5 / k, 1.0 - 0.5 / k, k)).astype(np.float32)  # (k,)
    ar = np.arange(1, IQN_EMBED + 1, dtype=np.float32)
    phi = np.cos(ar[None, :] * math.pi * tau[:, None])    # (k, IQN_EMBED)
    return _leaky(phi @ w["iqn.w"] + w["iqn.b"])          # (k, FLOAT_HIDDEN)


def iqn_q(w, obs_norm, k: int = IQN_K, qemb=None):
    """Mean Q-value per action for a batch of normalised observations.

    ``obs_norm`` is (B, OBS_DIM). Returns (B, N_ACTIONS). Mirrors the torch
    IQN_Network forward: float features, a per-quantile cosine embedding
    mixed in by Hadamard product, then a duelling V/A split. Pass ``qemb``
    from ``iqn_embedding`` to skip rebuilding it.
    """
    b = obs_norm.shape[0]
    h = _leaky(obs_norm @ w["ff0.w"] + w["ff0.b"])
    h = _leaky(h @ w["ff1.w"] + w["ff1.b"])               # (B, FLOAT_HIDDEN)

    if qemb is None:
        qemb = iqn_embedding(w, k)

    if b == 1:
        # The game's case. Kept 2-D so every product is a plain BLAS call
        # rather than numpy's stacked-matmul loop.
        mixed = h[0][None, :] * qemb                      # (k, FLOAT_HIDDEN)
        a = _leaky(mixed @ w["A0.w"] + w["A0.b"])
        a = a @ w["A1.w"] + w["A1.b"]                     # (k, N_ACTIONS)
        v = _leaky(mixed @ w["V0.w"] + w["V0.b"])
        v = v @ w["V1.w"] + w["V1.b"]                     # (k, 1)
        q = v + a - a.mean(axis=-1, keepdims=True)
        return q.mean(axis=0)[None, :]

    mixed = h[:, None, :] * qemb[None, :, :]              # (B, k, FLOAT_HIDDEN)
    a = _leaky(mixed @ w["A0.w"] + w["A0.b"])
    a = a @ w["A1.w"] + w["A1.b"]                         # (B, k, N_ACTIONS)
    v = _leaky(mixed @ w["V0.w"] + w["V0.b"])
    v = v @ w["V1.w"] + w["V1.b"]                         # (B, k, 1)
    q = v + a - a.mean(axis=-1, keepdims=True)
    return q.mean(axis=1)                                 # (B, N_ACTIONS)


def normalise(obs, mean, std):
    return np.clip((obs - mean) / np.maximum(std, 1e-4), -10.0, 10.0)


def policy_path(circuit: str):
    return config.RL_POLICY / f"{circuit}.npz"


def available(circuit: str) -> bool:
    return policy_path(circuit).exists()


#: m/s below which the AI's grid launch is scripted rather than learned.
GRID_LAUNCH_SPEED = 8.0


class RLDriver:
    """A trained IQN policy wrapped to look like ``Autopilot`` -- same
    ``controls(vehicle)`` call, so the ghost does not care who is driving."""

    def __init__(self, track, line, v_ref, surface):
        data = np.load(policy_path(track.name), allow_pickle=False)
        self.w = {k: data[k] for k in data.files if "." in k}
        self.mean = data["obs_mean"]
        self.std = data["obs_std"]
        self.track = track
        self.line = line
        self.v_ref = v_ref
        self.surface = surface
        self.look_idx = _lookahead_indices(line)
        self.qemb = iqn_embedding(self.w)
        self.prev = [DEFAULT_ACTION] * N_PREV_ACTIONS
        self._held = DEFAULT_ACTION
        self._ticks = 0
        self._on_grid = True

    def controls(self, vehicle) -> Controls:
        # Hold each decision for ACTION_REPEAT frames, the way training did --
        # the network never saw itself choosing at 60 Hz.
        if self._ticks % ACTION_REPEAT == 0:
            i, _ = self.surface.progress(vehicle.pos)
            # The game's grid sits ~6 m BEHIND the line, i.e. at the last
            # centreline samples, where the "lap remaining" observation reads
            # ~0 -- but training only ever launched from index 0 (reading
            # 1.0). Standing still in that never-seen state, some policies
            # (BrandsHatch) brake-and-steer into a permanent stall. Until the
            # car is properly into the lap, report the grid as the start line.
            #
            # The grand prix grid also puts the AI car 3.2 m to the side of
            # the centreline (Ghost.__init__), another state training never
            # launched from: BrandsHatch's policy read it as "stand still" and
            # sat on the grass with no throttle. So the launch itself is
            # scripted -- full throttle, straight -- until the car is rolling
            # at a speed the policy has seen from every kind of start.
            n = self.track.count
            launching = False
            if self._on_grid:
                if i > 0.9 * n:
                    i = 0
                elif i >= 0.05 * n:
                    self._on_grid = False
                launching = (self._on_grid
                             and float(vehicle.speed) < GRID_LAUNCH_SPEED)
            if launching:
                self._held = DEFAULT_ACTION
                self._ticks += 1
                return controls_for(self._held)
            obs = observe(vehicle, self.track, self.line, self.v_ref, i,
                          self.prev, self.look_idx, self.surface)
            q = iqn_q(self.w, normalise(obs[None], self.mean, self.std),
                      qemb=self.qemb)[0]
            self._held = int(np.argmax(q))
            self.prev = self.prev[1:] + [self._held]
        self._ticks += 1
        return controls_for(self._held)


# ==========================================================================
# Continuous control -- Soft Actor-Critic, after Fuchs et al. "Super-Human
# Performance in Gran Turismo Sport" (the SAC predecessor of Sony's Sophy).
#
# The discrete IQN driver beat the scripted planner but topped out at ~1-2
# wall contacts a lap and could not trail-brake with a bang-bang pedal. SAC
# gives a continuous, entropy-regularised policy: exploration is a tunable
# temperature rather than baked into the objective, and the actor can put the
# pedal exactly where it wants it.
#
# The v15 observation (48-dim). Three things IQN's lookahead-points obs and
# the first SAC obs each half-had, together:
#  * rangefinders to the WHITE LINE -- "how much room, and where does it run
#    out" -- fanned denser ahead than to the sides;
#  * the line SHAPE ahead as points in the car frame (a lean 3, not IQN's 16)
#    -- planning, the piece the pure-rangefinder IQN run (v14) lost and drove
#    170 km/h for it;
#  * the friction-circle state -- slip angle, real steer angle, and both
#    accelerations -- so the actor can feel the limit rather than infer it
#    from a single lateral-velocity number (GT Sophy carries all of these).
# Plus reference speed ahead as an explicit brake cue, which the paper leaves
# to curvature alone.
# ==========================================================================

#: Rangefinder fan for the SAC obs: denser straight ahead (line precision),
#: sparser to the sides (just "is the edge close"). Radians, car-relative.
SAC_RANGE_ANGLES = (-1.50, -1.00, -0.65, -0.40, -0.22, -0.10, 0.0,
                    0.10, 0.22, 0.40, 0.65, 1.00, 1.50)
N_SAC_RANGE = len(SAC_RANGE_ANGLES)
SAC_RANGE_MAX = 110.0
#: Curvature look-ahead, seconds at the current speed.
SAC_CURV_S = (0.5, 1.0, 1.7, 2.5, 3.4, 4.4, 5.5)
N_SAC_CURV = len(SAC_CURV_S)
#: Reference-speed look-ahead, seconds -- the brake-for-the-corner cue.
SAC_VREF_S = (0.4, 1.0, 1.8, 2.8, 4.0)
N_SAC_VREF = len(SAC_VREF_S)
#: Line-shape look-ahead: a few centreline points, seconds ahead, handed to
#: the net as (lateral, forward) in the car frame.
SAC_SHAPE_S = (1.2, 2.8, 5.0)
N_SAC_SHAPE = len(SAC_SHAPE_S)

#: 2 phase + 4 vel/rot + 2 accel + 2 line-err + 2 prev-act + 2 status
#: + N_SAC_RANGE + N_SAC_CURV + N_SAC_VREF + 2*N_SAC_SHAPE
OBS_DIM_SAC = (2 + 4 + 2 + 2 + 2 + 2
               + N_SAC_RANGE + N_SAC_CURV + N_SAC_VREF + 2 * N_SAC_SHAPE)
SAC_ACT_DIM = 2
SAC_HIDDEN = 256


def observe_sac(vehicle, track, line, v_ref, i, prev_action, surface):
    fwd = np.array([math.sin(vehicle.yaw), math.cos(vehicle.yaw)])
    right = np.array([math.cos(vehicle.yaw), -math.sin(vehicle.yaw)])
    cross = float(np.dot(vehicle.pos - line.center[i], line.normal[i]))
    line_yaw = math.atan2(line.tangent[i, 0], line.tangent[i, 1])
    heading_err = (line_yaw - vehicle.yaw + math.pi) % (2 * math.pi) - math.pi
    half = float(track.w_right[i] if cross > 0 else track.w_left[i])
    phase = 2.0 * math.pi * line.arclen[i] / max(line.length, 1e-6)

    rng = (surface.rangefinders(vehicle.pos, vehicle.yaw, SAC_RANGE_ANGLES,
                                SAC_RANGE_MAX) / SAC_RANGE_MAX)

    speed = max(float(vehicle.speed), 1.0)

    cur = [float(line.curvature[_ahead_index(line, i,
           float(np.clip(s * speed, 5.0, 350.0)))]) * 60.0
           for s in SAC_CURV_S]
    vrf = [float(v_ref[_ahead_index(line, i,
           float(np.clip(s * speed, 5.0, 420.0)))]) / config.MAX_SPEED
           for s in SAC_VREF_S]
    shape = []
    for s in SAC_SHAPE_S:
        j = _ahead_index(line, i, float(np.clip(s * speed, 6.0, 500.0)))
        d = line.center[j] - vehicle.pos
        shape.append(float(d @ right) / 60.0)
        shape.append(float(d @ fwd) / 120.0)

    obs = [
        math.sin(phase), math.cos(phase),
        float(np.dot(vehicle.vel, fwd)) / config.MAX_SPEED,
        float(np.dot(vehicle.vel, right)) / 20.0,
        vehicle.yaw_rate / 3.0,
        vehicle.slip_angle / 0.5,
        vehicle.long_accel / 30.0,
        vehicle.lat_accel / 30.0,
        cross / max(half, 1e-3),
        heading_err / 0.6,
        float(prev_action[0]), float(prev_action[1]),
        1.0 if vehicle.hit_wall else 0.0,
        (line.length - line.arclen[i]) / line.length,
        *rng.tolist(),
        *cur,
        *vrf,
        *shape,
    ]
    return np.asarray(obs, dtype=np.float32)


def sac_forward(w, obs_norm):
    """Returns (mean, log_std), each (B, 2), for the squashed-Gaussian actor."""
    x = obs_norm
    x = np.tanh(x @ w["pi0.w"] + w["pi0.b"])
    x = np.tanh(x @ w["pi1.w"] + w["pi1.b"])
    mean = x @ w["pi_mu.w"] + w["pi_mu.b"]
    log_std = np.clip(x @ w["pi_ls.w"] + w["pi_ls.b"], -5.0, 2.0)
    return mean, log_std


def sac_action(w, obs_norm, rng=None, deterministic=False):
    mean, log_std = sac_forward(w, obs_norm)
    if deterministic or rng is None:
        return np.tanh(mean)
    u = mean + np.exp(log_std) * rng.standard_normal(mean.shape)
    return np.tanh(u)


class SACDriver:
    """A trained SAC actor, greedy (mean action), wrapped as ``Autopilot``."""

    def __init__(self, track, line, v_ref, surface):
        data = np.load(policy_path(track.name), allow_pickle=False)
        self.w = {k: data[k] for k in data.files if "." in k}
        self.mean = data["obs_mean"]
        self.std = data["obs_std"]
        self.track, self.line, self.v_ref, self.surface = \
            track, line, v_ref, surface
        self.prev = np.zeros(2, np.float32)
        self._held = Controls()
        self._ticks = 0

    def controls(self, vehicle) -> Controls:
        if self._ticks % ACTION_REPEAT == 0:
            i, _ = self.surface.progress(vehicle.pos)
            obs = observe_sac(vehicle, self.track, self.line, self.v_ref, i,
                              self.prev, self.surface)
            a = sac_action(self.w, normalise(obs[None], self.mean,
                                             self.std), deterministic=True)[0]
            self.prev = a.astype(np.float32)
            steer, pedal = float(a[0]), float(a[1])
            self._held = Controls(throttle=max(pedal, 0.0),
                                  brake=max(-pedal, 0.0), steer=steer,
                                  analog_steer=True)
        self._ticks += 1
        return self._held


def sac_available(circuit: str) -> bool:
    p = policy_path(circuit)
    if not p.exists():
        return False
    try:
        return "pi_mu.w" in np.load(p, allow_pickle=False).files
    except Exception:
        return False
