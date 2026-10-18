"""
RocketLand-2D — Vehicle & Simulation Configuration
Loads vehicle.json and provides dataclass with verified physical parameters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RocketConfig:
    dry_mass: float = 25000.0
    propellant_mass_min: float = 2500.0
    propellant_mass_max: float = 4000.0
    length: float = 40.0
    diameter: float = 3.7
    thrust_max: float = 845000.0
    throttle_min: float = 0.4
    isp: float = 282.0
    g0: float = 9.80665
    g: float = 9.81
    gimbal_limit_deg: float = 5.0
    gimbal_rate_limit_deg_s: float = 20.0
    throttle_rate_limit_s: float = 2.0
    throttle_tau: float = 0.1
    rcs_max_torque: float = 50000.0
    rho0: float = 1.225
    scale_height: float = 8500.0
    cd_axial: float = 0.8
    cd_lateral: float = 1.5
    pad_half_width: float = 15.0
    corridor_x_limit: float = 500.0
    max_altitude: float = 2500.0
    crash_velocity_limit: float = 5.0
    touchdown_vy_limit: float = 2.0
    touchdown_vx_limit: float = 1.0
    touchdown_tilt_limit_deg: float = 5.0
    tip_limit_deg: float = 30.0
    contact_k: float = 3000000.0
    contact_c: float = 80000.0
    coulomb_mu: float = 0.6
    dt_flight: float = 0.02
    dt_control: float = 0.1
    dt_contact: float = 0.002

    @classmethod
    def from_json(cls, path: str | Path | None = None) -> RocketConfig:
        if path is None:
            path = Path("configs/rocketland/vehicle.json")
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
