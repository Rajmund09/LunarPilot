"""
RocketLand-2D — Dynamics Core (Appendix C)
Pure functions with zero gym dependencies:
- derivatives(state, controls, params, wind) -> dstate
- rk4_step(state, controls, params, dt, wind) -> new_state
"""

from __future__ import annotations
import numpy as np

from rocketland.atmosphere import air_density
from rocketland.config import RocketConfig


def derivatives(
    state: np.ndarray,
    controls: np.ndarray,
    cfg: RocketConfig,
    wind_vx: float = 0.0,
) -> np.ndarray:
    """
    Computes time-derivatives of the 7D state:
    state = [x, y, vx, vy, theta, omega, m_fuel]
    controls = [thrust_N, gimbal_rad, rcs_torque_Nm]
    Returns dstate/dt.
    """
    x, y, vx, vy, theta, omega, m_fuel = state
    thrust_N, gimbal_rad, rcs_torque_Nm = controls

    m_fuel_eff = max(0.0, float(m_fuel))
    total_mass = cfg.dry_mass + m_fuel_eff
    inertia = total_mass * (cfg.length ** 2) / 12.0

    # If fuel is exhausted, engine produces 0 thrust
    effective_thrust = thrust_N if m_fuel_eff > 0.0 else 0.0

    # Thrust vector angle from vertical: theta + delta
    thrust_angle = theta + gimbal_rad
    f_thrust_x = -effective_thrust * np.sin(thrust_angle)
    f_thrust_y = effective_thrust * np.cos(thrust_angle)

    # Aerodynamic drag in body-frame
    v_rel_x = vx - wind_vx
    v_rel_y = vy
    v_rel_mag = np.hypot(v_rel_x, v_rel_y)

    # Unit body axis u = (-sin(theta), cos(theta)), normal n = (cos(theta), sin(theta))
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    v_axial = -v_rel_x * sin_t + v_rel_y * cos_t
    v_lateral = v_rel_x * cos_t + v_rel_y * sin_t

    rho = air_density(y, rho0=cfg.rho0, scale_height=cfg.scale_height)
    area_axial = np.pi * ((cfg.diameter / 2.0) ** 2)
    area_lateral = cfg.length * cfg.diameter

    drag_axial = -0.5 * rho * v_rel_mag * cfg.cd_axial * area_axial * v_axial
    drag_lateral = -0.5 * rho * v_rel_mag * cfg.cd_lateral * area_lateral * v_lateral

    # Convert drag back to world frame
    f_drag_x = -drag_axial * sin_t + drag_lateral * cos_t
    f_drag_y = drag_axial * cos_t + drag_lateral * sin_t

    # Translation equations
    ax = (f_thrust_x + f_drag_x) / total_mass
    ay = (f_thrust_y + f_drag_y) / total_mass - cfg.g

    # Rotational torque about CG:
    # Engine is at distance r_e = length / 2 along body axis
    # Torque = -T * r_e * sin(gimbal) + tau_RCS
    r_e = cfg.length / 2.0
    torque_thrust = -effective_thrust * r_e * np.sin(gimbal_rad)
    alpha = (torque_thrust + rcs_torque_Nm) / inertia

    # Fuel mass depletion rate: dm/dt = -T / (Isp * g0)
    dm_dt = -(effective_thrust / (cfg.isp * cfg.g0)) if m_fuel_eff > 0.0 else 0.0

    return np.array([vx, vy, ax, ay, omega, alpha, dm_dt], dtype=np.float64)


def rk4_step(
    state: np.ndarray,
    controls: np.ndarray,
    cfg: RocketConfig,
    dt: float,
    wind_vx: float = 0.0,
) -> np.ndarray:
    """4th-order Runge-Kutta numerical integration step."""
    k1 = derivatives(state, controls, cfg, wind_vx)
    k2 = derivatives(state + 0.5 * dt * k1, controls, cfg, wind_vx)
    k3 = derivatives(state + 0.5 * dt * k2, controls, cfg, wind_vx)
    k4 = derivatives(state + dt * k3, controls, cfg, wind_vx)

    new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    # Fuel cannot be negative
    new_state[6] = max(0.0, new_state[6])
    return new_state
