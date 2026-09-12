"""Batched (GPU) re-implementation of the ai_sw driving environment.

The CPU original in ``ai_sw/game`` is untouched and stays authoritative:
static geometry is built by it and frozen into tensors here, and
``tests/parity.py`` checks that a batched rollout reproduces a scalar one
step for step.
"""
from .trackgpu import TrackGPU          # noqa: F401
from .vecenv import VecRaceEnv          # noqa: F401
