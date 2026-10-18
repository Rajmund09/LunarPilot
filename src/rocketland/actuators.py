"""
RocketLand-2D — Actuator Models (Appendix C & E)
- Throttle: command in [-1, 1], mapping to 0 or [throttle_min, 1]
  First-order lag: d(throttle)/dt = (target - current) / tau
  Rate limit: |d(throttle)/dt| <= throttle_rate_limit
- Gimbal: angle limit +/- 5 deg, rate limit +/- 20 deg/s
- RCS thrusters: torque limit +/- 50 kNm
"""

from __future__ import annotations
import numpy as np

from rocketland.config import RocketConfig


class ActuatorState:
    def __init__(self, cfg: RocketConfig):
        self.cfg = cfg
        self.throttle: float = 0.0
        self.gimbal_deg: float = 0.0
        self.rcs_torque: float = 0.0

    def reset(self):
        self.throttle = 0.0
        self.gimbal_deg = 0.0
        self.rcs_torque = 0.0

    def step(self, action: np.ndarray, dt: float) -> tuple[float, float, float]:
        """
        Takes raw continuous action Box(-1, 1, shape=(3,)):
        a[0]: throttle command (a[0] < -0.9 -> off; else linear map to [throttle_min, 1])
        a[1]: gimbal command (mapped to +/- gimbal_limit_deg)
        a[2]: RCS torque command (mapped to +/- rcs_max_torque)

        Returns physical controls:
        (thrust_N, gimbal_rad, rcs_torque_Nm)
        """
        # Bad action rejection (NaN, inf, wrong shape) per Appendix G
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError(f"Invalid actuator action: {action}")

        a_throttle, a_gimbal, a_rcs = action

        # 1. Target throttle
        if a_throttle < -0.9:
            target_throttle = 0.0
        else:
            # Map [-0.9, 1.0] -> [throttle_min, 1.0]
            frac = (a_throttle - (-0.9)) / (1.0 - (-0.9))
            target_throttle = self.cfg.throttle_min + frac * (1.0 - self.cfg.throttle_min)
            target_throttle = np.clip(target_throttle, self.cfg.throttle_min, 1.0)

        # Apply throttle rate limit and first order lag
        d_thr_lag = (target_throttle - self.throttle) / self.cfg.throttle_tau
        max_rate = self.cfg.throttle_rate_limit_s
        d_thr = np.clip(d_thr_lag, -max_rate, max_rate)
        self.throttle = float(np.clip(self.throttle + d_thr * dt, 0.0, 1.0))

        # 2. Target gimbal angle
        target_gimbal_deg = float(np.clip(a_gimbal * self.cfg.gimbal_limit_deg, -self.cfg.gimbal_limit_deg, self.cfg.gimbal_limit_deg))
        d_gimbal = (target_gimbal_deg - self.gimbal_deg) / dt
        max_g_rate = self.cfg.gimbal_rate_limit_deg_s
        d_gimbal_clamped = np.clip(d_gimbal, -max_g_rate, max_g_rate)
        self.gimbal_deg = float(np.clip(self.gimbal_deg + d_gimbal_clamped * dt, -self.cfg.gimbal_limit_deg, self.cfg.gimbal_limit_deg))

        # 3. RCS Torque
        self.rcs_torque = float(np.clip(a_rcs * self.cfg.rcs_max_torque, -self.cfg.rcs_max_torque, self.cfg.rcs_max_torque))

        thrust_N = self.throttle * self.cfg.thrust_max
        gimbal_rad = float(np.radians(self.gimbal_deg))

        return thrust_N, gimbal_rad, self.rcs_torque
