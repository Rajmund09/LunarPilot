"""
LunarPilot — Policies
======================
Callable wrappers: obs -> action.  All use explicit seeds; no unseeded sampling.

Policy types:
  RandomPolicy        — uniform random, own RNG (not env.action_space.sample())
  NullPolicy          — always action 0 (do nothing)
  GymHeuristicPolicy  — Gymnasium's built-in LunarLander heuristic
  SB3Policy           — wraps a saved SB3 model (deterministic by default)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class RandomPolicy:
    """Uniform random policy with its own seeded RNG.

    Does NOT use env.action_space.sample() to avoid implicit global state.
    """

    def __init__(self, n_actions: int, seed: int = 0):
        self._n = n_actions
        self._rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray) -> int:
        return int(self._rng.integers(0, self._n))


class NullPolicy:
    """Always fires action 0 (do nothing / engine off)."""

    def __call__(self, obs: np.ndarray) -> int:
        return 0


class GymHeuristicPolicy:
    """Wraps Gymnasium's built-in LunarLander heuristic.

    Raises ImportError loudly if the heuristic is not available
    (rather than silently falling back), per Rule 6.
    """

    def __init__(self, env: Any):
        try:
            from gymnasium.envs.box2d.lunar_lander import heuristic
        except ImportError as exc:
            raise ImportError(
                "GymHeuristicPolicy requires gymnasium.envs.box2d.lunar_lander.heuristic, "
                "which is not available in this installation."
            ) from exc
        self._heuristic = heuristic
        self._env = env

    def __call__(self, obs: np.ndarray) -> int:
        return int(self._heuristic(self._env, obs))


class SB3Policy:
    """Wraps a Stable-Baselines3 model loaded from disk.

    Parameters
    ----------
    model_path  : path to the .zip model file
    deterministic: if True, use deterministic action selection
    model_cls   : SB3 algorithm class (default: PPO)
    """

    def __init__(
        self,
        model_path: str | Path,
        deterministic: bool = True,
        model_cls: Any = None,
    ):
        if model_cls is None:
            from stable_baselines3 import PPO
            model_cls = PPO
        self._model = model_cls.load(str(model_path))
        self._deterministic = deterministic

    def __call__(self, obs: np.ndarray) -> int:
        action, _ = self._model.predict(obs, deterministic=self._deterministic)
        return int(action)
