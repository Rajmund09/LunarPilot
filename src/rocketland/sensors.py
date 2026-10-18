"""
RocketLand-2D — Sensors and State Estimator (Appendix E & P2.8)
Provides sensor observation with Gaussian noise, bias, and optional delay.
"""

from __future__ import annotations
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
