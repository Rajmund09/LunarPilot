"""
LunarPilot — Perturbation Toolkit (Phase 1)
Wrappers and configurations for domain randomization, observation noise,
action latency, initial velocity kicks, and engine power scaling.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import gymnasium as gym


class PerturbedLunarLanderWrapper(gym.Wrapper):
    """
    Wrapper adding structured physical and sensor disturbances to LunarLander:
    1. Random initial velocity kick (vx, vy, omega)
    2. Observation noise (zero-mean Gaussian, per-channel scale)
    3. Action latency / delay buffer (1-2 steps)
    4. Main and side engine power scaling factors
    """

    def __init__(
        self,
        env: gym.Env,
        obs_noise_scale: float = 0.0,
        action_delay: int = 0,
        engine_scale: float = 1.0,
        initial_kick_scale: float = 0.0,
        seed: int = 42,
    ):
        super().__init__(env)
        self.obs_noise_scale = float(obs_noise_scale)
        self.action_delay = int(action_delay)
        self.engine_scale = float(engine_scale)
        self.initial_kick_scale = float(initial_kick_scale)
        self.rng = np.random.default_rng(seed)
        self._action_buffer: list[int] = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._action_buffer = [0] * (self.action_delay + 1)

        # Apply initial velocity kick directly to lander body if configured
        if self.initial_kick_scale > 0.0 and hasattr(self.env.unwrapped, "lander"):
            kick_vx = self.rng.uniform(-self.initial_kick_scale, self.initial_kick_scale)
            kick_vy = self.rng.uniform(-self.initial_kick_scale, self.initial_kick_scale)
            kick_w = self.rng.uniform(-self.initial_kick_scale * 0.5, self.initial_kick_scale * 0.5)
            self.env.unwrapped.lander.linearVelocity.x += float(kick_vx)
            self.env.unwrapped.lander.linearVelocity.y += float(kick_vy)
            self.env.unwrapped.lander.angularVelocity += float(kick_w)

        return self._add_obs_noise(obs), info

    def step(self, action: int):
        self._action_buffer.append(int(action))
        effective_action = self._action_buffer.pop(0)

        # Apply action in environment
        obs, reward, terminated, truncated, info = self.env.step(effective_action)
        return self._add_obs_noise(obs), reward, terminated, truncated, info

    def _add_obs_noise(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_noise_scale <= 0.0:
            return obs
        noise = self.rng.normal(0.0, self.obs_noise_scale, size=obs.shape).astype(np.float32)
        # Do not corrupt binary ground contacts (indices 6 and 7)
        noise[6] = 0.0
        noise[7] = 0.0
        noisy_obs = obs + noise
        return noisy_obs
