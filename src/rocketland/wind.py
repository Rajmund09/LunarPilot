"""
RocketLand-2D — Wind and Atmospheric Turbulence (P2.8)
Implements:
- Altitude-dependent mean wind boundary layer: v_wind(y) = v_base * log(max(1, y)/z0) or power-law
- Ornstein-Uhlenbeck stochastic wind gusts:
  dw = -theta * (w - w_mean) * dt + sigma * sqrt(dt) * N(0, 1)
"""

from __future__ import annotations

import numpy as np


class OrnsteinUhlenbeckWind:
    def __init__(
        self,
        base_wind: float = 0.0,
        gust_std: float = 1.0,
        theta: float = 0.15,
        dt: float = 0.05,
        scale_height: float = 100.0,
        seed: int | None = None,
    ):
        self.base_wind = float(base_wind)
        self.gust_std = float(gust_std)
        self.theta = float(theta)
        self.dt = float(dt)
        self.scale_height = float(scale_height)
        self.rng = np.random.default_rng(seed)
        self.gust = 0.0

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.gust = 0.0

    def step(self, y: float, dt: float | None = None) -> float:
        """Updates and returns instantaneous horizontal wind speed at altitude y (m)."""
        dt_eff = float(dt) if dt is not None else self.dt
        noise = self.rng.normal(0.0, 1.0)
        dw = -self.theta * self.gust * dt_eff + self.gust_std * np.sqrt(dt_eff) * noise
        self.gust += dw

        # Altitude boundary layer profile: wind increases with altitude up to scale height
        alt_factor = float(np.clip(y / max(10.0, self.scale_height), 0.1, 1.5))
        total_wind = self.base_wind * alt_factor + self.gust
        return float(total_wind)
