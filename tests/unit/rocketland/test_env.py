"""
Tests for RocketLand2DEnv Gymnasium Compliance (Appendix G):
- check_env passes
- Bad actions rejected (NaN, inf, wrong shape)
- Reset returns observation in space
- Step returns valid transitions
"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rocketland.env import RocketLand2DEnv


def test_gym_check_env():
    env = RocketLand2DEnv()
    # Runs standard gymnasium checks
    check_env(env.unwrapped)
    env.close()


def test_bad_action_rejection():
    env = RocketLand2DEnv()
    env.reset(seed=42)

    # Wrong shape
    with pytest.raises(ValueError):
        env.step(np.array([1.0, 0.0], dtype=np.float32))

    # NaN in action
    with pytest.raises(ValueError):
        env.step(np.array([np.nan, 0.0, 0.0], dtype=np.float32))

    # Inf in action
    with pytest.raises(ValueError):
        env.step(np.array([1.0, np.inf, 0.0], dtype=np.float32))

    env.close()


def test_observation_space_bounds():
    env = RocketLand2DEnv()
    obs, info = env.reset(seed=123)
    assert env.observation_space.contains(obs)

    for _ in range(10):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        assert env.observation_space.contains(obs)
        if term or trunc:
            break
    env.close()
