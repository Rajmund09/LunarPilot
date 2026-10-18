"""
RocketLand-2D — Analytical Suicide Burn & Trajectory Guidance Controller (P2.6, Appendix C)
- Ignition triggered when altitude <= stopping_distance * margin (with lateral offset margin)
- Unpowered descent: aerodynamic glide bias towards landing pad
- Powered hoverslam: closed-loop parabolic glide-slope deceleration tracking
- Terminal guidance: progressive lateral velocity nulling and upright touchdown lockout
- Attitude control: critically damped TVC gimbal and RCS reaction thrusters
- Throttle cuts off upon touchdown
"""

from __future__ import annotations
import numpy as np

from rocketland.config import RocketConfig


class SuicideBurnController:
    def __init__(self, cfg: RocketConfig, margin: float = 1.15):
        self.cfg = cfg
        self.margin = margin
        self.ignited = False

    def reset(self):
        self.ignited = False

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

        if h_foot <= 0.02:
            return np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        m_total = self.cfg.dry_mass + fuel

        # 2. Hoverslam Ignition & Vertical Throttle Guidance
        a_net = (self.cfg.thrust_max * 0.72 / m_total) - self.cfg.g
        burn_dist = (vy ** 2) / (2.0 * max(0.5, a_net))

        lat_err = abs(x) + abs(vx) * 2.0
        margin = self.margin + min(0.25, 0.01 * lat_err)
        ignite_alt = max(110.0, burn_dist * margin)

        if not self.ignited:
            if h_foot <= ignite_alt:
                self.ignited = True

        if not self.ignited:
            # Unpowered glide: aerodynamic body tilt guides trajectory towards pad center
            a_throttle = -1.0
            glide_tilt_deg = -np.clip(0.12 * x + 0.25 * vx, -3.5, 3.5)
            theta_target = np.radians(glide_tilt_deg)
        else:
            # Powered hoverslam parabolic deceleration profile
            a_glide = 8.2
            v_target = -np.sqrt(2.0 * a_glide * max(0.001, h_foot) + 0.40)
            v_err = vy - v_target
            a_cmd = a_glide - 4.0 * v_err
            cos_tilt = max(0.7, np.cos(theta))
            thrust_req = (m_total * (a_cmd + self.cfg.g)) / cos_tilt
            thr_frac = thrust_req / self.cfg.thrust_max
            a_throttle = float(-1.0 + np.clip((thr_frac - self.cfg.throttle_min) / (1.0 - self.cfg.throttle_min), 0.0, 1.0) * 2.0)
            if h_foot < 12.0 and vy < -0.5:
                a_throttle = max(-0.85, a_throttle)

            # Lateral velocity tracking with progressive flare limits
            if abs(x) < 2.5:
                vx_target = 0.0
            elif h_foot > 60.0:
                vx_target = float(np.clip(-0.35 * x, -3.0, 3.0))
            elif h_foot > 20.0:
                vx_target = float(np.clip(-0.25 * x, -1.8, 1.8))
            else:
                vx_target = float(np.clip(-0.15 * x, -0.6, 0.6))

            if h_foot > 60.0:
                max_tilt = np.radians(4.5)
            elif h_foot > 20.0:
                max_tilt = np.radians(3.2)
            elif h_foot > 6.0:
                max_tilt = np.radians(1.8)
            elif h_foot > 1.2:
                max_tilt = np.radians(0.8)
            else:
                max_tilt = 0.0

            vx_err = vx - vx_target
            theta_target = float(np.clip(0.055 * vx_err, -max_tilt, max_tilt))

        # 3. Attitude control: critically damped TVC gimbal and RCS reaction jets
        theta_error = theta - theta_target
        gimbal_cmd = 5.5 * theta_error + 4.0 * omega
        a_gimbal = float(np.clip(gimbal_cmd, -1.0, 1.0))
        rcs_cmd = -(5.0 * theta_error + 4.0 * omega)
        a_rcs = float(np.clip(rcs_cmd, -1.0, 1.0))

        return np.array([a_throttle, a_gimbal, a_rcs], dtype=np.float32)
