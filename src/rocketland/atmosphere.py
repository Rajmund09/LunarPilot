"""
RocketLand-2D — Atmosphere Model
Exponential barometric density profile:
rho(y) = rho0 * exp(-y / H)
"""

from __future__ import annotations
import numpy as np


def air_density(y: float | np.ndarray, rho0: float = 1.225, scale_height: float = 8500.0) -> float | np.ndarray:
    """Return air density at altitude y (in meters). Clamped below y=0."""
    alt = np.maximum(y, 0.0)
    return rho0 * np.exp(-alt / scale_height)
