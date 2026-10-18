"""
RocketLand-2D — Contact Dynamics Model (Appendix D)
- Feet kinematics from body pose and dimensions
- Normal spring-damper contact with unilateral constraint (feet never pull down)
- Coulomb friction with continuous tanh smoothing
- Tip and crash detection
"""

from __future__ import annotations

import numpy as np

from rocketland.config import RocketConfig


def foot_positions(x: float, y: float, theta: float, cfg: RocketConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes global coordinates of left and right landing feet:
    p_i = CG + R(theta) * (+/- w, -h)
    where h = length / 2, w = diameter * 1.5 (extended landing legs span)
    """
    h = cfg.length / 2.0
    w = cfg.diameter * 1.5

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Left foot: local (-w, -h)
    left_x = x + (-w * cos_t - (-h) * sin_t)
    left_y = y + (-w * sin_t + (-h) * cos_t)

    # Right foot: local (+w, -h)
    right_x = x + (w * cos_t - (-h) * sin_t)
    right_y = y + (w * sin_t + (-h) * cos_t)

    return np.array([left_x, left_y]), np.array([right_x, right_y])


def foot_velocities(
    vx: float,
    vy: float,
    omega: float,
    theta: float,
    cfg: RocketConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes global linear velocities of the two feet from rigid body kinematics."""
    h = cfg.length / 2.0
    w = cfg.diameter * 1.5
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # r_left = (-w*cos - (-h)*sin, -w*sin + (-h)*cos)
    rx_left = -w * cos_t + h * sin_t
    ry_left = -w * sin_t - h * cos_t

    rx_right = w * cos_t + h * sin_t
    ry_right = w * sin_t - h * cos_t

    # v_foot = v_CG + omega x r -> vx_foot = vx - omega * ry, vy_foot = vy + omega * rx
    v_left = np.array([vx - omega * ry_left, vy + omega * rx_left])
    v_right = np.array([vx - omega * ry_right, vy + omega * rx_right])

    return v_left, v_right


def contact_forces(
    state: np.ndarray,
    cfg: RocketConfig,
    v_eps: float = 0.05,
) -> tuple[np.ndarray, float, tuple[bool, bool]]:
    """
    Computes total external contact force and torque exerted on the vehicle:
    Returns (F_contact [Fx, Fy], Torque_contact, (left_contact, right_contact))
    """
    x, y, vx, vy, theta, omega, _ = state
    p_left, p_right = foot_positions(x, y, theta, cfg)
    v_left, v_right = foot_velocities(vx, vy, omega, theta, cfg)

    total_fx = 0.0
    total_fy = 0.0
    total_torque = 0.0
    contacts = [False, False]

    for idx, (p, v) in enumerate([(p_left, v_left), (p_right, v_right)]):
        pen = -p[1]  # Penetration depth below y = 0
        if pen > 0.0:
            contacts[idx] = True
            v_n = v[1]  # Normal velocity (positive upwards)
            v_t = v[0]  # Tangential velocity

            # Unilateral spring-damper: N = max(0, k*pen - c*v_n)
            normal_force = max(0.0, cfg.contact_k * pen - cfg.contact_c * v_n)

            # Coulomb friction: Ft = -mu * N * tanh(vt / v_eps)
            friction_force = -cfg.coulomb_mu * normal_force * np.tanh(v_t / v_eps)

            total_fx += friction_force
            total_fy += normal_force

            # Torque about CG: r_x * Fy - r_y * Fx
            rx = p[0] - x
            ry = p[1] - y
            torque = rx * normal_force - ry * friction_force
            total_torque += torque

    return np.array([total_fx, total_fy]), total_torque, (contacts[0], contacts[1])
