"""
RocketLand-2D — Advanced Aerospace Guidance & Attitude Autopilot (P2.6, Appendix C)
- Real rocket engine flight computer architecture (G-FOLD & Apollo Guidance Law concepts)
- Phase 1: High-Altitude Unpowered Aerodynamic Glide Banking with Grid Fins & RCS
- Phase 2: Closed-Loop Kinetic Hoverslam Deceleration Flare with Kinematic Overshoot Limiting
- Phase 3: Decisive Non-Hovering Terminal Touchdown with Direct TVC Lateral Drift Trim
- High-rate attitude control: critically damped TVC gimbal and RCS reaction jets
- Autonomous touchdown cutoff & safe state lockout
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

        # 1. Engine cut off on ground contact or pad touchdown
        if left_contact or right_contact or y < 0.2:
            return np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        # Foot altitude relative to ground (exact vehicle geometry)
        h_geom = self.cfg.length / 2.0
        w_geom = self.cfg.diameter * 1.5
        h_foot = float(y - h_geom * np.cos(theta) - w_geom * abs(np.sin(theta)))

        if h_foot <= 0.04:
            return np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        m_total = self.cfg.dry_mass + fuel

        # Lead adaptation bias: active at high/mid approach, tapers near ground so rocket settles at pad center
        blend_bias = float(np.clip((h_foot - 6.0) / 20.0, 0.0, 1.0))
        x = raw_x - self.bias_x * blend_bias

        # 2. Kinetic Hoverslam Ignition Solver
        # Deceleration capability under active throttle
        a_net = (self.cfg.thrust_max * 0.72 / m_total) - self.cfg.g
        burn_dist = (vy ** 2) / (2.0 * max(0.5, a_net))

        # True kinematic divert requirement: accounting for horizontal kinetic energy
        if x * vx > 0.0:
            x_peak = abs(x) + (vx ** 2) / 4.4
        else:
            x_peak = abs(x)

        lat_err = x_peak
        lat_divert_h = min(120.0, 3.2 * lat_err) + self.extra_ignite
        ignite_alt = max(95.0 + lat_divert_h, burn_dist * (self.margin + 0.00) + lat_divert_h)

        if not self.ignited and h_foot <= ignite_alt and vy < -8.0:
            self.ignited = True

        if not self.ignited:
            # Phase 1: Unpowered aerodynamic glide banking
            glide_tilt_deg = float(np.clip((-0.22 * x - 0.35 * vx), -14.0, 14.0))
            theta_target = float(np.radians(glide_tilt_deg))
            a_throttle = -1.0
        else:
            # Phase 2: Closed-Loop Kinetic Flare Guidance
            # Decelerates firmly without hovering or floating at the end
            if h_foot > 5.5:
                v_target = -np.sqrt(2.0 * 8.0 * (h_foot - 1.0) + 1.25**2)
                a_ff = 8.0
            else:
                # Decisive, confident terminal touchdown descent (1.0 to 1.25 m/s sink rate)
                # Prevents hanging in the air, fuel waste, and lateral drift
                v_target = -1.25
                a_ff = 4.0

            # Terminal Translate Gate: If still offset (> 1.5m) in the final 25m, brake vertical speed slightly to allow translation
            if h_foot < 25.0 and abs(raw_x) > 1.8:
                v_target = max(-1.6, v_target)

            v_err = vy - v_target
            a_cmd = a_ff - 4.8 * v_err
            cos_tilt = max(0.65, np.cos(theta))
            thrust_req = (m_total * (a_cmd + self.cfg.g)) / cos_tilt
            thr_frac = thrust_req / self.cfg.thrust_max

            # Kinematic Maximum Approach Speed: prevents overshooting and reverse whipping
            v_max_kinematic = float(np.sqrt(2.0 * 0.85 * max(0.01, abs(x))))

            # Altitude-tiered lateral velocity envelope
            if h_foot > 35.0:
                v_limit = min(v_max_kinematic, 8.0) * self.gain_lat
                max_tilt = np.radians(16.0)
            elif h_foot > 12.0:
                v_limit = min(v_max_kinematic, 4.0) * self.gain_lat
                max_tilt = np.radians(7.5)
            elif h_foot > 3.0:
                v_limit = min(v_max_kinematic, 1.0)
                max_tilt = np.radians(2.5)
            else:
                v_limit = 0.15
                max_tilt = np.radians(0.20)

            # Target velocity with smooth approach
            target_vx = float(np.clip(-0.42 * x * self.gain_lat, -v_limit, v_limit))
            vx_err = vx - target_vx

            # Anti-overshoot dynamic tilt envelope: tightens tilt when horizontal speed is already small
            max_tilt_dyn = min(max_tilt, np.radians(1.5 + 1.8 * abs(vx) + 0.3 * abs(x)))
            theta_target = float(np.clip(0.038 * vx_err, -max_tilt_dyn, max_tilt_dyn))

            norm_thr = np.clip((thr_frac - self.cfg.throttle_min) / (1.0 - self.cfg.throttle_min), 0.0, 1.0)
            a_throttle = float(-0.90 + norm_thr * 1.90) if thr_frac >= self.cfg.throttle_min else -1.0

        # Phase 3: Attitude autopilot (TVC Gimbal + Reaction Control System)
        theta_error = theta - theta_target
        if h_foot < 25.0:
            kp_g, kd_g = 13.0, 7.5 + self.kd_boost
            kp_rcs, kd_rcs = 14.0, 8.0 + self.kd_boost
        else:
            kp_g, kd_g = 8.0, 4.5
            kp_rcs, kd_rcs = 9.0, 5.0

        gimbal_cmd = kp_g * theta_error + kd_g * omega
        # Direct lateral TVC trim in terminal descent (0.5m <= h < 15m) to cancel horizontal drift cleanly
        if 0.5 <= h_foot < 15.0:
            gimbal_cmd -= 0.20 * vx

        a_gimbal = float(np.clip(gimbal_cmd, -1.0, 1.0))
        rcs_cmd = -(kp_rcs * theta_error + kd_rcs * omega)
        a_rcs = float(np.clip(rcs_cmd, -1.0, 1.0))

        return np.array([a_throttle, a_gimbal, a_rcs], dtype=np.float32)
