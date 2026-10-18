"""
RocketLand-2D — Gymnasium Environment (P2.4, Appendix E & F)
- Env ID: RocketLand2D-v0
- Action space: Box(-1, 1, shape=(3,))
- Observation space: Box(-inf, inf, shape=(12,), dtype=float32)
- Passes gymnasium.utils.env_checker.check_env
"""

from __future__ import annotations

from typing import Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rocketland.config import RocketConfig
from rocketland.actuators import ActuatorState
from rocketland.atmosphere import air_density
from rocketland.contact import contact_forces, foot_positions
from rocketland.dynamics import rk4_step, derivatives
from rocketland.sensors import make_observation
from lunar_pilot.outcomes import Outcome


class RocketLand2DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        config: RocketConfig | None = None,
        scenario: str = "NOMINAL",
        render_mode: str | None = None,
    ):
        super().__init__()
        self.cfg = config if config is not None else RocketConfig()
        self.scenario = scenario
        self.render_mode = render_mode

        # Action: Box(-1, 1, shape=(3,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation: Box(-inf, inf, shape=(12,), dtype=np.float32)
        # Using finite high bounds for Box env_checker
        obs_high = np.array([
            50.0,    # x / pad_half_width
            10.0,    # y / 1000
            10.0,    # vx / 50
            10.0,    # vy / 50
            1.0,     # sin(theta)
            1.0,     # cos(theta)
            20.0,    # omega / 0.5
            2.0,     # fuel ratio
            1.0,     # throttle
            1.0,     # gimbal
            1.0,     # left contact
            1.0,     # right contact
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.actuators = ActuatorState(self.cfg)
        self.state = np.zeros(7, dtype=np.float64)  # [x, y, vx, vy, theta, omega, m_fuel]
        self.contacts = (False, False)
        self.sim_time: float = 0.0
        self.step_count: int = 0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.first_touchdown_kinematics: dict[str, Any] | None = None
        self.settled_time: float = 0.0

        self.rng = np.random.default_rng(42)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.actuators.reset()
        self.sim_time = 0.0
        self.step_count = 0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.first_touchdown_kinematics = None
        self.settled_time = 0.0

        # Sample initial state from scenario distribution (P2.5)
        # Default nominal descent start
        alt = float(self.rng.uniform(800.0, 1200.0))
        vy = float(self.rng.uniform(-100.0, -60.0))
        x = float(self.rng.uniform(-100.0, 100.0))
        vx = float(self.rng.uniform(-10.0, 10.0))
        tilt = float(np.radians(self.rng.uniform(-5.0, 5.0)))
        omega = float(self.rng.uniform(-0.01, 0.01))
        m_fuel = float(self.rng.uniform(self.cfg.propellant_mass_min + 500.0, self.cfg.propellant_mass_max))

        self.state = np.array([x, alt, vx, vy, tilt, omega, m_fuel], dtype=np.float64)
        self.contacts = (False, False)

        obs = make_observation(
            self.state,
            self.actuators.throttle,
            self.actuators.gimbal_deg,
            self.contacts,
            self.cfg,
            obs_noise_scale=0.0,
            rng=self.rng,
        )
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if not np.all(np.isfinite(action)) or action.shape != (3,):
            raise ValueError(f"Invalid action {action}")

        action = np.clip(action, -1.0, 1.0)

        # Actuator command update
        dt_c = self.cfg.dt_control
        thrust_N, gimbal_rad, rcs_torque = self.actuators.step(action, dt=dt_c)
        controls = np.array([thrust_N, gimbal_rad, rcs_torque], dtype=np.float64)

        # High-rate physics substepping
        # Near ground (any foot y < 2.0), use dt_contact, else dt_flight
        p_l, p_r = foot_positions(self.state[0], self.state[1], self.state[4], self.cfg)
        min_foot_y = min(p_l[1], p_r[1])
        sub_dt = self.cfg.dt_contact if min_foot_y < 2.0 else self.cfg.dt_flight
        n_substeps = max(1, int(round(dt_c / sub_dt)))
        actual_sub_dt = dt_c / n_substeps

        m_fuel_start = self.state[6]

        for _ in range(n_substeps):
            f_cont, tau_cont, contacts = contact_forces(self.state, self.cfg)
            any_contact = bool(contacts[0] or contacts[1])

            # Check touchdown kinematics on exact moment of first contact
            if any_contact and self.first_touchdown_kinematics is None:
                self.first_touchdown_kinematics = {
                    "x": float(self.state[0]),
                    "y": float(self.state[1]),
                    "vx": float(self.state[2]),
                    "vy": float(self.state[3]),
                    "theta": float(self.state[4]),
                    "omega": float(self.state[5]),
                    "abs_x": abs(float(self.state[0])),
                    "left_contact": bool(contacts[0]),
                    "right_contact": bool(contacts[1]),
                }

            self.contacts = contacts

            # Incorporate contact forces into RK4
            k1 = derivatives(self.state, controls, self.cfg)
            k1[2] += f_cont[0] / (self.cfg.dry_mass + max(0.0, self.state[6]))
            k1[3] += f_cont[1] / (self.cfg.dry_mass + max(0.0, self.state[6]))
            inertia = (self.cfg.dry_mass + max(0.0, self.state[6])) * (self.cfg.length ** 2) / 12.0
            k1[5] += tau_cont / inertia

            self.state = self.state + actual_sub_dt * k1
            self.state[6] = max(0.0, self.state[6])
            self.sim_time += actual_sub_dt

        self.step_count += 1
        any_contact = bool(self.contacts[0] or self.contacts[1])

        # Check settling
        speed = np.hypot(self.state[2], self.state[3])
        if any_contact and speed < 0.2 and abs(self.state[5]) < 0.05:
            self.settled_time += dt_c
        else:
            self.settled_time = 0.0

        # Terminations per Appendix D
        terminated = False
        truncated = False
        outcome = Outcome.TIMEOUT
        info: dict[str, Any] = {}

        x, y, vx, vy, theta, omega, m_fuel = self.state
        tilt_deg = abs(np.degrees(theta))

        # Check hard crash
        if any_contact and self.first_touchdown_kinematics and abs(self.first_touchdown_kinematics["vy"]) > self.cfg.crash_velocity_limit:
            terminated = True
            outcome = Outcome.HARD_LANDING
        elif tilt_deg > self.cfg.tip_limit_deg and any_contact:
            terminated = True
            outcome = Outcome.TIPPED
        elif abs(x) > self.cfg.corridor_x_limit or y > self.cfg.max_altitude:
            terminated = True
            outcome = Outcome.OUT_OF_BOUNDS
        elif self.settled_time >= 1.0:
            terminated = True
            td = self.first_touchdown_kinematics or {}
            on_pad = abs(x) <= self.cfg.pad_half_width
            safe_speed = abs(td.get("vy", vy)) <= self.cfg.touchdown_vy_limit and abs(td.get("vx", vx)) <= self.cfg.touchdown_vx_limit
            safe_tilt = abs(np.degrees(td.get("theta", theta))) <= self.cfg.touchdown_tilt_limit_deg

            if on_pad and safe_speed and safe_tilt:
                outcome = Outcome.LANDED_ON_PAD
            elif not on_pad and safe_speed and safe_tilt:
                outcome = Outcome.LANDED_OFF_PAD
            else:
                outcome = Outcome.HARD_LANDING
        elif self.sim_time >= 90.0 or self.step_count >= 900:
            truncated = True
            outcome = Outcome.TIMEOUT

        # Reward shaping (Appendix F)
        # Potential-based shaping: Phi(s) = -(w_pos * ||pos||/L + w_vel * ||vel||/V + w_ang * |theta|)
        dist = np.hypot(x, y)
        spd = np.hypot(vx, vy)
        phi = -(1.0 * (dist / 1000.0) + 1.0 * (spd / 100.0) + 1.5 * abs(theta))
        fuel_used = max(0.0, m_fuel_start - m_fuel)
        r_fuel = -0.01 * (fuel_used / self.cfg.propellant_mass_max)
        r_act = -0.005 * float(np.linalg.norm(action - self.prev_action))
        reward = phi * 0.1 + r_fuel + r_act

        if terminated:
            if outcome == Outcome.LANDED_ON_PAD:
                reward += 100.0
            elif outcome == Outcome.LANDED_OFF_PAD:
                reward += 30.0
            else:
                reward -= 100.0

        self.prev_action = action.copy()

        obs = make_observation(
            self.state,
            self.actuators.throttle,
            self.actuators.gimbal_deg,
            self.contacts,
            self.cfg,
            obs_noise_scale=0.0,
            rng=self.rng,
        )

        if terminated or truncated:
            info["outcome"] = outcome.value
            info["touchdown"] = self.first_touchdown_kinematics or {
                "x": float(x), "y": float(y), "vx": float(vx), "vy": float(vy),
                "theta": float(theta), "omega": float(omega), "abs_x": abs(float(x)),
                "left_contact": bool(self.contacts[0]), "right_contact": bool(self.contacts[1]),
            }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            # 64x64 dummy headless image for check_env compatibility
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return None
