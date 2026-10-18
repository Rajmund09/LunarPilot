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
    def __init__(
        self,
        cfg: RocketConfig,
        margin: float = 1.15,
        bias_x: float = 0.0,
        gain_lat: float = 1.0,
        extra_ignite: float = 0.0,
        kd_boost: float = 0.0,
    ):
        self.cfg = cfg
        self.margin = margin
        self.bias_x = float(bias_x)
        self.gain_lat = float(gain_lat)
        self.extra_ignite = float(extra_ignite)
        self.kd_boost = float(kd_boost)
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
        raw_x = float(obs[0]) * self.cfg.pad_half_width
        x = raw_x - self.bias_x  # Guidance target offset with learned adaptation bias
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

        # Foot altitude relative to ground (exact geometry)
        h_geom = self.cfg.length / 2.0
        w_geom = self.cfg.diameter * 1.5
        h_foot = float(y - h_geom * np.cos(theta) - w_geom * abs(np.sin(theta)))

        if h_foot <= 0.04:
            return np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        m_total = self.cfg.dry_mass + fuel

        # 2. Hoverslam Ignition & Vertical Throttle Guidance
        a_net = (self.cfg.thrust_max * 0.72 / m_total) - self.cfg.g
        burn_dist = (vy ** 2) / (2.0 * max(0.5, a_net))

        # Lateral divert margin: provides altitude and burn time needed to traverse large horizontal offsets
        lat_err = abs(x) + abs(vx) * 2.0
        extra_h = min(85.0, 3.8 * lat_err) + self.extra_ignite
        ignite_alt = max(115.0 + extra_h, burn_dist * (self.margin + 0.03) + extra_h)

        if not self.ignited and h_foot <= ignite_alt:
            self.ignited = True

        if not self.ignited:
            # High-altitude unpowered aerodynamic glide banking
            max_glide = min(16.0, 4.0 + 0.5 * abs(x))
            glide_tilt_deg = -np.clip((0.30 * x + 0.50 * vx) * self.gain_lat, -max_glide, max_glide)
            theta_target = np.radians(glide_tilt_deg)
            a_throttle = -1.0
        else:
            # Powered hoverslam parabolic deceleration profile
            a_glide = 8.2
            v_target = -np.sqrt(2.0 * a_glide * max(0.001, h_foot) + 0.16)
            v_err = vy - v_target
            a_cmd = a_glide - 4.5 * v_err
            cos_tilt = max(0.7, np.cos(theta))
            thrust_req = (m_total * (a_cmd + self.cfg.g)) / cos_tilt
            thr_frac = thrust_req / self.cfg.thrust_max
            a_throttle = float(-1.0 + np.clip((thr_frac - self.cfg.throttle_min) / (1.0 - self.cfg.throttle_min), 0.0, 1.0) * 2.0)
            if h_foot < 12.0 and vy < -0.5:
                a_throttle = max(-0.85, a_throttle)

            # Continuous progressive lateral velocity tracking
            if h_foot > 50.0:
                vx_target = float(np.clip(-0.45 * x * self.gain_lat, -4.5, 4.5))
                max_tilt = np.radians(14.0)
            elif h_foot > 20.0:
                vx_target = float(np.clip(-0.35 * x * self.gain_lat, -2.5, 2.5))
                max_tilt = np.radians(5.0)
            elif h_foot > 8.0:
                vx_target = float(np.clip(-0.25 * x, -0.8, 0.8))
                max_tilt = np.radians(1.2)
            elif h_foot > 2.5:
                vx_target = float(np.clip(-0.10 * x, -0.25, 0.25))
                max_tilt = np.radians(0.30)
            else:
                vx_target = 0.0
                max_tilt = 0.0

            vx_err = vx - vx_target
            theta_target = float(np.clip(0.065 * vx_err, -max_tilt, max_tilt))

        # 3. Attitude control: stiff critically damped TVC gimbal and RCS reaction jets
        theta_error = theta - theta_target
        if h_foot < 25.0:
            kp_g, kd_g = 10.0, 7.2 + self.kd_boost
            kp_rcs, kd_rcs = 11.0, 7.8 + self.kd_boost
        else:
            kp_g, kd_g = 6.0, 4.2
            kp_rcs, kd_rcs = 6.5, 4.5

        gimbal_cmd = kp_g * theta_error + kd_g * omega
        # Direct lateral TVC trim in terminal descent (2m <= h < 15m) to cancel horizontal drift
        if 2.0 <= h_foot < 15.0:
            gimbal_cmd -= 0.18 * vx

        a_gimbal = float(np.clip(gimbal_cmd, -1.0, 1.0))
        rcs_cmd = -(kp_rcs * theta_error + kd_rcs * omega)
        a_rcs = float(np.clip(rcs_cmd, -1.0, 1.0))

        return np.array([a_throttle, a_gimbal, a_rcs], dtype=np.float32)
