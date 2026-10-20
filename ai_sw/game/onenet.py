"""One network that both decides and drives.

The cars so far run two networks: a decision policy (``raceai.Policy``: which lane,
what pace, start a pass -- every 0.13 s) and a driver (``drivenet.DriveNet``: wheel
and pedals -- every 0.033 s). Here the same two jobs are one network with one
trunk and two heads:

* the **fast head** (2 outputs) is the driver's: steering and pedal;
* the **slow head** (``raceai.N_ACTIONS`` outputs) is the decision policy's: it
  is read only on a decision tick (``OnePolicy``), as the policy's greedy action.

Its input is both observations side by side -- the driver's (``drivenet``) and the
decision policy's (``raceai``). The decision half is rebuilt on a decision tick
and held in between; the driver half is rebuilt every time the net is asked.

It is taught by ``tools/dagger_one.py`` from the two networks it replaces.
Inference is numpy only.
"""
from __future__ import annotations

import numpy as np

from . import drivenet, raceai

N_DEC = raceai.N_ACTIONS
IN_DIM = drivenet.OBS_DIM + raceai.OBS_DIM
OUT_DIM = drivenet.ACT_DIM + N_DEC


class OneNet:
    """The deployed network. Weights (see ``tools/dagger_one.py``): the trunk
    ``W0 b0 W1 b1`` (ReLU), the fast head ``Wf bf`` and the slow head
    ``Ws0 bs0`` (ReLU) ``Ws1 bs1``."""

    #: The follower need not compute its own answer alongside; it sees the cars
    #: around, so the traffic rules do not bind it (see drivenet).
    needs_teacher = False
    free = True
    one = True

    def __init__(self, w: dict):
        self.w = {k: np.asarray(w[k], np.float32)
                  for k in ("W0", "b0", "W1", "b1", "Wf", "bf", "Ws0", "bs0", "Ws1", "bs1")}
        if self.w["W0"].shape[0] != IN_DIM or self.w["Wf"].shape[1] != drivenet.ACT_DIM \
                or self.w["Ws1"].shape[1] != N_DEC:
            raise ValueError("onenet weights do not match the observations")

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls({k: z[k] for k in z.files})

    def forward(self, x: np.ndarray) -> np.ndarray:
        """``OUT_DIM`` numbers: steer and pedal, then the decision logits."""
        w = self.w
        h = np.maximum(x @ w["W0"] + w["b0"], 0.0)
        h = np.maximum(h @ w["W1"] + w["b1"], 0.0)
        s = np.maximum(h @ w["Ws0"] + w["bs0"], 0.0)
        return np.concatenate([h @ w["Wf"] + w["bf"], s @ w["Ws1"] + w["bs1"]])

    def act(self, x: np.ndarray, teacher=None) -> np.ndarray:
        return self.forward(x)[:drivenet.ACT_DIM]

    def decide(self, x: np.ndarray) -> int:
        """The decision policy's greedy action."""
        return int(np.argmax(self.forward(x)[drivenet.ACT_DIM:]))


class OnePolicy:
    """The decision half, installed as ``RaceDriver.policy``: on a decision tick
    the net is asked for the lane and pace (and whether to start a pass) and the
    answer is applied as the decision policy's would be."""

    def __init__(self, net):
        self.net = net

    def __call__(self, driver, me, field) -> None:
        driver.follow._dec_obs = raceai.observe(driver, me, field)
        raceai.apply_action(driver, me, self.net.decide(driver.one_input(me, field)), field)
