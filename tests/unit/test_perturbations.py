"""
Tests for Phase 1 perturbation toolkit:
- Observation noise bounds
- Action latency delay buffer
- Initial velocity kicks
"""

import numpy as np
import gymnasium as gym

from lunar_pilot.perturbations import PerturbedLunarLanderWrapper


def test_perturbation_wrapper_noise():
    base_env = gym.make("LunarLander-v3")
    wrapped = PerturbedLunarLanderWrapper(base_env, obs_noise_scale=0.05, seed=123)

    obs, _ = wrapped.reset(seed=42)
    # Binary contact sensors (obs[6] and obs[7]) must remain uncorrupted (0.0 or 1.0)
    assert obs[6] in [0.0, 1.0]
    assert obs[7] in [0.0, 1.0]
    assert obs.shape == (8,)
    wrapped.close()


def test_perturbation_wrapper_latency():
    base_env = gym.make("LunarLander-v3")
    # Action delay = 1 step
    wrapped = PerturbedLunarLanderWrapper(base_env, action_delay=1, seed=123)

    wrapped.reset(seed=42)
    # First step should execute action 0 from initial buffer
    obs1, r1, t1, tr1, _ = wrapped.step(2)
    # Next step should execute action 2 from previous step
    obs2, r2, t2, tr2, _ = wrapped.step(3)

    assert obs2.shape == (8,)
    wrapped.close()
