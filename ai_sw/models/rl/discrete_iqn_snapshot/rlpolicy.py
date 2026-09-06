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
STEER_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)
PEDAL_LEVELS = (1.0, 0.0, -0.4, -1.0)    # throttle / coast / light brake / brake
ACTIONS = [(s, p) for p in PEDAL_LEVELS for s in STEER_LEVELS]
N_ACTIONS = len(ACTIONS)                 # 20
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
#: shape of the road rather than against a scalar curvature.
LOOKAHEAD_M = (6.0, 13.0, 22.0, 33.0, 47.0, 64.0, 85.0, 112.0, 146.0, 188.0,
               240.0, 305.0, 385.0, 485.0, 610.0, 765.0)
N_LOOK = len(LOOKAHEAD_M)

#: 2 lap-phase + 5 car state + N_PREV_ACTIONS + 1 finish + 3*N_LOOK
OBS_DIM = 2 + 5 + N_PREV_ACTIONS + 1 + 3 * N_LOOK
ACT_DIM = N_ACTIONS

#: IQN network sizes, mirroring Linesight scaled down (no image head).
FLOAT_HIDDEN = 256
HEAD_HIDDEN = 256
IQN_EMBED = 64
IQN_K = 8                               # quantiles averaged at inference


def controls_for(action_idx: int) -> Controls:
    steer, pedal = ACTIONS[int(action_idx)]
    return Controls(throttle=max(pedal, 0.0), brake=max(-pedal, 0.0),
                    steer=steer, analog_steer=False)


# --------------------------------------------------------------------------
def reference_line(track):
    """The centreline, sampled and lightly smoothed. Linesight rewards and
    describes progress along the plain centre of the track and lets the policy
    find its own line; the potential-based line term only discourages straying
    far from centre, it does not prescribe a racing line."""
    return raceline.Line(track, np.zeros(track.count),
                         smooth=config.RL_CURVATURE_SMOOTH)


def _lookahead_indices(line):
    """Sample indices at LOOKAHEAD_M metres ahead of every sample, precomputed
    once per line (a (count, N_LOOK) table)."""
    arc = line.arclen
    n = len(arc)
    tgt = (arc[:, None] + np.asarray(LOOKAHEAD_M)[None, :]) % line.length
    return np.searchsorted(arc, tgt).clip(0, n - 1)


def observe(vehicle, track, line, v_ref, i: int, prev_actions,
            look_idx=None) -> np.ndarray:
    """Track-relative view of the world, plus lap phase and recent inputs.

    Everything is in the car's frame or measured against the line, so the
    numbers mean the same thing at every corner. Lap phase (sin/cos) is the
    deliberate exception: the brief is to master *this* circuit, and a network
    that knows where it is can brake "here, at this corner" instead of
    re-deriving it from geometry every lap.
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
    remain = (line.length - line.arclen[i]) / line.length
    obs.append(remain)

    if look_idx is None:
        look_idx = _lookahead_indices(line)
    js = look_idx[i]
    d = line.center[js] - vehicle.pos                       # (N_LOOK, 2)
    obs.extend((d @ right / 60.0).tolist())                 # lateral, car frame
    obs.extend((d @ fwd / 120.0).tolist())                  # forward, car frame
    obs.extend((v_ref[js] / config.MAX_SPEED).tolist())     # target speed there
    return np.asarray(obs, dtype=np.float32)


# --------------------------------------------------------------------------
def _leaky(x, s=0.01):
    return np.where(x > 0.0, x, x * s)


def iqn_q(w, obs_norm, k: int = IQN_K):
    """Mean Q-value per action for a batch of normalised observations.

    ``obs_norm`` is (B, OBS_DIM). Returns (B, N_ACTIONS). Mirrors the torch
    IQN_Network forward: float features, a per-quantile cosine embedding
    mixed in by Hadamard product, then a duelling V/A split.
    """
    b = obs_norm.shape[0]
    h = _leaky(obs_norm @ w["ff0.w"] + w["ff0.b"])
    h = _leaky(h @ w["ff1.w"] + w["ff1.b"])               # (B, FLOAT_HIDDEN)

    tau = (np.linspace(0.5 / k, 1.0 - 0.5 / k, k)).astype(np.float32)  # (k,)
    ar = np.arange(1, IQN_EMBED + 1, dtype=np.float32)
    phi = np.cos(ar[None, :] * math.pi * tau[:, None])    # (k, IQN_EMBED)
    qemb = _leaky(phi @ w["iqn.w"] + w["iqn.b"])          # (k, FLOAT_HIDDEN)

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
        self.prev = [DEFAULT_ACTION] * N_PREV_ACTIONS
        self._held = DEFAULT_ACTION
        self._ticks = 0

    def controls(self, vehicle) -> Controls:
        # Hold each decision for ACTION_REPEAT frames, the way training did --
        # the network never saw itself choosing at 60 Hz.
        if self._ticks % ACTION_REPEAT == 0:
            i, _ = self.surface.progress(vehicle.pos)
            obs = observe(vehicle, self.track, self.line, self.v_ref, i,
                          self.prev, self.look_idx)
            q = iqn_q(self.w, normalise(obs[None], self.mean, self.std))[0]
            self._held = int(np.argmax(q))
            self.prev = self.prev[1:] + [self._held]
        self._ticks += 1
        return controls_for(self._held)
