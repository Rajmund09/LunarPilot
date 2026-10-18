"""
RocketLand-2D — Analytical Suicide Burn Controller (P2.6, Appendix C)
- Ignition triggered when altitude <= stopping_distance * margin
- Stopping distance computed dynamically:
  stopping_dist = vy^2 / (2 * (a_max_net))
  where a_max_net = (T_max / m) - g
- PD controller drives attitude tilt towards 0 (or lateral error compensation)
- Throttle cuts off upon touchdown
"""

from __future__ import annotations
import numpy as np

from rocketland.config import RocketConfig


class SuicideBurnController:
    def __init__(self, cfg: RocketConfig, margin: float = 1.05):
        self.cfg = cfg
        self.margin = margin
        self.kp_theta = 4.0
        self.kd_theta = 2.0
        self.kp_x = 0.05
        self.kd_x = 0.1

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Takes normalized observation:
        obs[0]: x / pad_half_width
        obs[1]: y / 1000.0
        obs[2]: vx / 50.0
        obs[3]: vy / 50.0
        obs[4]: sin(theta)
        obs[5]: cos(theta)
        obs[6]: omega / 0.5
        obs[7]: fuel ratio
        obs[10]: left leg contact
        obs[11]: right leg contact

        Returns raw continuous action Box(-1, 1, shape=(3,)):
        [a0_throttle, a1_gimbal, a2_rcs]
        """
        # Unpack physical state from normalized observation
        x = float(obs[0]) * self.cfg.pad_half_width
        y = float(obs[1]) * 1000.0
        vx = float(obs[2]) * 50.0
        vy = float(obs[3]) * 50.0
        sin_t = float(obs[4])
        cos_t = float(obs[5])
        theta = float(np.arctan2(sin_t, cos_t))
        omega = float(obs[6]) * 0.5
        fuel = float(obs[7]) * self.cfg.propellant_mass_max
        left_contact = bool(obs[10] > 0.5)
        right_contact = bool(obs[11] > 0.5)

        # 1. Engine cut off on ground contact
        if left_contact or right_contact or y < 0.2:
            return np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        # Foot altitude relative to ground
        h_geom = self.cfg.length / 2.0
        w_geom = self.cfg.diameter * 1.5
        h_foot = y - h_geom * np.cos(theta) - w_geom * abs(np.sin(theta))

        if h_foot <= 0.01:
            return np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        # 2. Hoverslam glide-slope parabolic velocity tracking
        # Target touchdown velocity is -0.7 m/s (well within [0, 2.0 m/s] limit)
        a_glide = 8.8
        v_target = -np.sqrt(2.0 * a_glide * max(0.001, h_foot) + 0.49)
        v_err = vy - v_target

        # Proportional-derivative throttle command
        a_cmd = a_glide - 4.0 * v_err
        m_total = self.cfg.dry_mass + fuel
        thrust_req = m_total * (a_cmd + self.cfg.g)
        thr_frac = thrust_req / self.cfg.thrust_max

        # Free fall check: ignite when reaching stopping window or lower than 120m
        a_net = (self.cfg.thrust_max * 0.75 / m_total) - self.cfg.g
        burn_dist = (vy ** 2) / (2.0 * max(0.5, a_net))

        if h_foot > burn_dist * 1.10 and h_foot > 120.0:
            a_throttle = -1.0
        else:
            a_throttle = float(-1.0 + np.clip((thr_frac - self.cfg.throttle_min) / (1.0 - self.cfg.throttle_min), 0.0, 1.0) * 2.0)
            # Never shut down engine completely during final 12m flare to avoid free-fall impact
            if h_foot < 12.0 and vy < -0.6:
                a_throttle = max(-0.85, a_throttle)

        # 3. Attitude guidance: tilt to cancel horizontal drift, upright rollout near pad
        if h_foot < 20.0:
            theta_target = 0.0
        else:
            vx_target = float(np.clip(-0.15 * x, -8.0, 8.0))
            theta_target = float(np.clip(0.04 * (vx - vx_target), -np.radians(4.0), np.radians(4.0)))

        theta_error = theta - theta_target
        gimbal_cmd = 12.0 * theta_error + 6.0 * omega
        a_gimbal = float(np.clip(gimbal_cmd, -1.0, 1.0))
        rcs_cmd = -(12.0 * theta_error + 6.0 * omega)
        a_rcs = float(np.clip(rcs_cmd, -1.0, 1.0))

        return np.array([a_throttle, a_gimbal, a_rcs], dtype=np.float32)

