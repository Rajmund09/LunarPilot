"""
RocketLand-2D — Sensors and State Estimator (Appendix E & P2.8)
Provides sensor observation with Gaussian noise, bias, and optional delay.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from rocketland.config import RocketConfig


def make_observation(
    state: np.ndarray,
    actuator_throttle: float,
    actuator_gimbal_deg: float,
    contacts: tuple[bool, bool],
    cfg: RocketConfig,
    obs_noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Normalized observation vector per Appendix E:
    [
      x / pad_half_width,
      y / 1000.0,
      vx / 50.0,
      vy / 50.0,
      sin(theta),
      cos(theta),
      omega / 0.5,
      m_fuel / propellant_mass_max,
      actuator_throttle,
      actuator_gimbal_deg / gimbal_limit_deg,
      left_contact (0 or 1),
      right_contact (0 or 1)
    ]
    Shape: (12,)
    """
    x, y, vx, vy, theta, omega, m_fuel = state
    left_c, right_c = contacts

    obs = np.array([
        x / cfg.pad_half_width,
        y / 1000.0,
        vx / 50.0,
        vy / 50.0,
        np.sin(theta),
        np.cos(theta),
        omega / 0.5,
        m_fuel / cfg.propellant_mass_max,
        actuator_throttle,
        actuator_gimbal_deg / cfg.gimbal_limit_deg,
        1.0 if left_c else 0.0,
        1.0 if right_c else 0.0,
    ], dtype=np.float32)

    if obs_noise_scale > 0.0 and rng is not None:
        noise = rng.normal(0.0, obs_noise_scale, size=obs.shape).astype(np.float32)
        # Contacts are binary switches; do not corrupt
        noise[10] = 0.0
        noise[11] = 0.0
        obs += noise

    return obs


class SensorSuite:
    """
    Realistic sensor model with per-channel noise, systematic bias, and transport delay.
    """
    def __init__(
        self,
        cfg: RocketConfig,
        noise_std: float = 0.01,
        bias_std: float = 0.005,
        delay_steps: int = 0,
        seed: int | None = None,
    ):
        self.cfg = cfg
        self.noise_std = float(noise_std)
        self.bias_std = float(bias_std)
        self.delay_steps = int(delay_steps)
        self.rng = np.random.default_rng(seed)
        self.bias = self.rng.normal(0.0, self.bias_std, size=(12,)).astype(np.float32)
        # Binary switches have zero bias
        self.bias[10] = 0.0
        self.bias[11] = 0.0
        self.buffer: deque[np.ndarray] = deque(maxlen=max(1, self.delay_steps + 1))

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.bias = self.rng.normal(0.0, self.bias_std, size=(12,)).astype(np.float32)
        self.bias[10] = 0.0
        self.bias[11] = 0.0
        self.buffer.clear()

    def observe(
        self,
        state: np.ndarray,
        throttle: float,
        gimbal_deg: float,
        contacts: tuple[bool, bool],
    ) -> np.ndarray:
        raw_obs = make_observation(
            state, throttle, gimbal_deg, contacts, self.cfg, obs_noise_scale=0.0, rng=self.rng
        )
        corrupted = raw_obs.copy()
        if self.noise_std > 0.0:
            noise = self.rng.normal(0.0, self.noise_std, size=(12,)).astype(np.float32)
            noise[10] = 0.0
            noise[11] = 0.0
            corrupted += noise + self.bias
        else:
            corrupted += self.bias

        self.buffer.append(corrupted)
        if len(self.buffer) > self.delay_steps:
            return self.buffer[0]
        return corrupted
