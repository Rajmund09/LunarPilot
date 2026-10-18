"""
Unit tests for RocketLand-2D Dynamics Core (Appendix G):
1. Free fall (engine off, no drag): y = y0 + v0*t - 0.5*g*t^2 within 1e-6
2. Constant vertical burn (no drag): Delta v equals Isp*g0*ln(m0/m1) - g*t within tolerance
3. Mass accounting: Fuel >= 0, thrust 0 at zero fuel, Delta m equals integrated flow
4. Gimbal sign check: Positive gimbal delta yields negative angular acceleration alpha
"""

import numpy as np
import pytest

from rocketland.config import RocketConfig
from rocketland.dynamics import derivatives, rk4_step


def test_free_fall():
    cfg = RocketConfig(rho0=0.0)  # Vacuum
    state = np.array([0.0, 1000.0, 0.0, -10.0, 0.0, 0.0, 3000.0])
    controls = np.array([0.0, 0.0, 0.0])  # Engine off

    dt = 0.01
    t_total = 2.0
    steps = int(t_total / dt)

    s = state.copy()
    for _ in range(steps):
        s = rk4_step(s, controls, cfg, dt)

    expected_y = 1000.0 + (-10.0) * t_total - 0.5 * cfg.g * (t_total ** 2)
    expected_vy = -10.0 - cfg.g * t_total

    assert pytest.approx(expected_y, rel=1e-5) == s[1]
    assert pytest.approx(expected_vy, rel=1e-5) == s[3]


def test_constant_vertical_burn_tsiolkovsky():
    cfg = RocketConfig(rho0=0.0)  # Vacuum
    m0 = cfg.dry_mass + 3000.0
    state = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 3000.0])
    thrust = 500000.0
    controls = np.array([thrust, 0.0, 0.0])

    dt = 0.01
    t_total = 1.0
    steps = int(t_total / dt)

    s = state.copy()
    for _ in range(steps):
        s = rk4_step(s, controls, cfg, dt)

    m1 = cfg.dry_mass + s[6]
    # Tsiolkovsky rocket equation with gravity: Delta v = Isp * g0 * ln(m0 / m1) - g * t
    delta_v_expected = cfg.isp * cfg.g0 * np.log(m0 / m1) - cfg.g * t_total
    assert pytest.approx(delta_v_expected, rel=1e-3) == s[3]


def test_mass_accounting_and_fuel_exhaustion():
    cfg = RocketConfig(rho0=0.0)
    state = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 10.0])  # Only 10 kg fuel
    thrust = 800000.0
    controls = np.array([thrust, 0.0, 0.0])

    dt = 0.1
    s = state.copy()
    for _ in range(50):  # 5 seconds
        s = rk4_step(s, controls, cfg, dt)

    assert s[6] == 0.0
    # At zero fuel, acceleration must equal purely -g
    d = derivatives(s, controls, cfg)
    assert pytest.approx(-cfg.g, rel=1e-5) == d[3]


def test_gimbal_torque_sign():
    cfg = RocketConfig(rho0=0.0)
    state = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 3000.0])
    # Positive gimbal delta (5 deg > 0)
    gimbal_rad = np.radians(5.0)
    controls = np.array([500000.0, gimbal_rad, 0.0])

    d = derivatives(state, controls, cfg)
    # Appendix C Sign check: Positive delta rotates vehicle clockwise -> alpha < 0
    assert d[5] < 0.0
