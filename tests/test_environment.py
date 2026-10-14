"""
Unit tests for LunarLander environment setup and dynamics.
"""

import pytest
import gymnasium as gym
import numpy as np


def get_test_env_id():
    try:
        gym.spec("LunarLander-v3")
        return "LunarLander-v3"
    except Exception:
        return "LunarLander-v2"


def test_environment_creation():
    env_id = get_test_env_id()
    env = gym.make(env_id)
    assert env is not None
    env.close()


def test_observation_space():
    env_id = get_test_env_id()
    env = gym.make(env_id)
    obs, info = env.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (8,)
    assert obs.dtype in (np.float32, np.float64)
    assert np.all(np.isfinite(obs))
    env.close()


def test_action_space():
    env_id = get_test_env_id()
    env = gym.make(env_id)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 4
    env.close()


def test_step_dynamics():
    env_id = get_test_env_id()
    env = gym.make(env_id)
    obs, info = env.reset(seed=123)

    for action in range(4):
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs.shape == (8,)
        assert np.all(np.isfinite(next_obs))
        assert isinstance(reward, (float, np.floating))
        assert np.isfinite(reward)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
    env.close()


def test_deterministic_seed():
    env_id = get_test_env_id()
    env1 = gym.make(env_id)
    env2 = gym.make(env_id)

    obs1, _ = env1.reset(seed=777)
    obs2, _ = env2.reset(seed=777)

    np.testing.assert_allclose(obs1, obs2, err_msg="Seeds must produce identical initial states")
    env1.close()
    env2.close()
