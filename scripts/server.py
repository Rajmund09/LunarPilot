"""
LunarPilot / RocketLand — Integrated Autonomous Simulation Server

Serves:
- Mode 1: 'falcon' — RocketLand-2D Falcon-9 Drone Ship Autonomous Suicide Burn & Landing
  * Real RK4 aerodynamic physics, de Laval Merlin 1D vacuum engine (845 kN),
  * Dynamic TVC gimbal tilt (-5° to +5°), continuous throttle (40%-100%),
  * Grid fin and RCS nitrogen roll control, mass depletion, spring-damper deck contact.
- Mode 2: 'artemis' — LunarPilot LunarLander-v3 Autonomous Deep RL Agent
  * 500k Master PPO policy and live online self-training engine on lunar regolith.
"""

import json
import os
import random
import sys
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Set UTF-8 encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add src to path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lunar_pilot.utils import get_supported_env_id
from lunar_pilot.outcomes import classify_lunarlander_episode, Outcome
from rocketland.config import RocketConfig
from rocketland.env import RocketLand2DEnv
from rocketland.sensors import make_observation
from rocketland.controllers.suicide_burn import SuicideBurnController

# --- Preload Models & Environments ---
ENV_ID = get_supported_env_id("LunarLander-v3")
BEST_MODEL_PATH = "models/best/best_model.zip"
if not os.path.exists(BEST_MODEL_PATH):
    BEST_MODEL_PATH = "models/final_model.zip"

CHECKPOINT_200K = "models/checkpoints/ppo_lunar_200000_steps.zip"
CHECKPOINT_100K = "models/checkpoints/ppo_lunar_100000_steps.zip"

print(f"[*] Preloading 500k Expert Policy from: {BEST_MODEL_PATH}")
EXPERT_MODEL = PPO.load(BEST_MODEL_PATH)


class LunarCeilingWrapper(gym.Wrapper):
    """Prevents lander from flying away into space by penalizing ascent above drop entry."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if obs[1] > 1.45 and obs[3] > 0.05:
            terminated = True
            reward = -100.0
            info["abort_reason"] = "LOST_IN_SPACE"
        return obs, reward, terminated, truncated, info


TRAINING_ENV = LunarCeilingWrapper(gym.make(ENV_ID))
if os.path.exists(BEST_MODEL_PATH):
    TRAINING_MODEL = PPO.load(BEST_MODEL_PATH, env=TRAINING_ENV)
    CURRENT_STAGE = "expert"
else:
    TRAINING_MODEL = PPO("MlpPolicy", TRAINING_ENV, n_steps=256, batch_size=64, n_epochs=4, learning_rate=5e-4, verbose=0)
    CURRENT_STAGE = "scratch"

# RocketLand Configuration & Controller
ROCKET_CONFIG = RocketConfig.from_json("configs/rocketland/vehicle.json")
SUICIDE_CONTROLLER = SuicideBurnController(ROCKET_CONFIG)

ACTIVE_VEHICLE = "falcon"  # Default is the high-fidelity Falcon-9 simulation!
ACTIVE_LUNAR_MODE = "expert"

SESSION_STATS = {
    "total_attempts": 0,
    "success_count": 0,
    "crash_count": 0,
    "timeout_count": 0,
    "recent_flights": [],
}


# --- Mode 1: RocketLand-2D Falcon-9 Drone Ship Simulation ---
def simulate_rocket_episode(alt=None, x_offset=None, vx=None, vy=None, wind=0.0, seed=None):
    """Simulates Falcon-9 descent onto an autonomous spaceport drone ship."""
    global SESSION_STATS

    if seed is None:
        seed = random.randint(1000, 999999)

    env = RocketLand2DEnv(ROCKET_CONFIG)
    obs, info = env.reset(seed=seed)

    # Custom initial state override if requested
    init_alt = float(alt) if alt is not None else float(env.state[1])
    init_x = float(x_offset) if x_offset is not None else float(env.state[0])
    init_vx = float(vx) if vx is not None else float(env.state[2])
    init_vy = float(vy) if vy is not None else float(env.state[3])
    init_fuel = float(env.state[6])

    # Re-apply state
    env.state = np.array([init_x, init_alt, init_vx, init_vy, 0.0, 0.0, init_fuel], dtype=np.float64)
    obs = make_observation(env.state, env.actuators.throttle, env.actuators.gimbal_deg, env.contacts, ROCKET_CONFIG, rng=env.rng)

    steps_data = []
    done = False
    step = 0
    dt_c = ROCKET_CONFIG.dt_control

    while not done and step < 800:
        action = SUICIDE_CONTROLLER(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        x, y, v_x, v_y, theta, omega, m_fuel = env.state
        tilt_deg = float(np.degrees(theta))
        h_geom = ROCKET_CONFIG.length / 2.0
        w_geom = ROCKET_CONFIG.diameter * 1.5
        h_foot = float(y - h_geom * np.cos(theta) - w_geom * abs(np.sin(theta)))
        legs_deployed = bool(h_foot < 45.0)

        # Actuators
        throttle_pct = float(env.actuators.throttle)
        thrust_kn = float(throttle_pct * ROCKET_CONFIG.thrust_max / 1000.0) if throttle_pct > 0.0 else 0.0
        gimbal_deg = float(env.actuators.gimbal_deg)
        rcs_knm = float(action[2] * ROCKET_CONFIG.rcs_max_torque / 1000.0)

        # Dynamic Flight Phase
        if done:
            phase = "TOUCHDOWN & ROLLOUT"
        elif h_foot < 15.0:
            phase = "TERMINAL HOVERSLAM FLARE"
        elif throttle_pct > 0.1:
            phase = "SUICIDE BURN RETRO-IGNITION"
        elif abs(action[1]) > 0.3 or abs(action[2]) > 0.3:
            phase = "GRID FIN & RCS STEERING"
        else:
            phase = "HIGH-SPEED DESCENT COAST"

        steps_data.append({
            "step": step,
            "t": round(step * dt_c, 2),
            "x": round(float(x), 2),
            "y": round(float(y), 2),
            "h_foot": round(max(0.0, h_foot), 2),
            "vx": round(float(v_x), 2),
            "vy": round(float(v_y), 2),
            "theta": round(float(theta), 4),
            "tilt_deg": round(tilt_deg, 2),
            "omega_deg": round(float(np.degrees(omega)), 2),
            "throttle": round(throttle_pct, 3),
            "thrust_kn": round(thrust_kn, 1),
            "gimbal_deg": round(gimbal_deg, 2),
            "rcs_knm": round(rcs_knm, 2),
            "fuel": round(float(m_fuel), 1),
            "left_contact": bool(env.contacts[0]),
            "right_contact": bool(env.contacts[1]),
            "legs_deployed": legs_deployed,
            "phase": phase,
        })

        obs = next_obs
        step += 1

    outcome_str = info.get("outcome", "TIMEOUT")
    td = info.get("touchdown", {})
    abs_x = td.get("abs_x", abs(float(env.state[0])))
    vy_final = td.get("vy", float(env.state[3]))

    if outcome_str == Outcome.LANDED_ON_PAD.value or outcome_str == "LANDED_ON_PAD":
        outcome_label = "TOUCHDOWN (Safe Landing on Droneship!)"
        is_success = True
        is_crash = False
    elif outcome_str == Outcome.LANDED_OFF_PAD.value or outcome_str == "LANDED_OFF_PAD":
        outcome_label = "TOUCHDOWN (Off-Pad Oceanic Landing)"
        is_success = False
        is_crash = False
    elif "HARD_LANDING" in outcome_str or "CRASH" in outcome_str or "TIPPED" in outcome_str:
        outcome_label = f"IMPACT: {outcome_str.replace('_', ' ')}"
        is_success = False
        is_crash = True
    else:
        outcome_label = "MISSION TIMEOUT"
        is_success = False
        is_crash = False

    SESSION_STATS["total_attempts"] += 1
    if is_success:
        SESSION_STATS["success_count"] += 1
    elif is_crash:
        SESSION_STATS["crash_count"] += 1
    else:
        SESSION_STATS["timeout_count"] += 1

    flight_summary = {
        "attempt": SESSION_STATS["total_attempts"],
        "vehicle": "Falcon 9 (RocketLand-2D)",
        "outcome": outcome_label,
        "steps": step,
        "final_vy": round(float(vy_final), 2),
        "dist_center": round(float(abs_x), 1),
        "fuel_remaining": round(float(env.state[6]), 1),
    }
    SESSION_STATS["recent_flights"].insert(0, flight_summary)
    if len(SESSION_STATS["recent_flights"]) > 25:
        SESSION_STATS["recent_flights"].pop()

    return {
        "episode_id": f"FLIGHT-{SESSION_STATS['total_attempts']}",
        "vehicle": "falcon",
        "vehicle_name": "Falcon 9 Orbital Booster",
        "environment": "droneship",
        "title": f"Flight #{SESSION_STATS['total_attempts']} — {outcome_label.split('(')[0].strip()}",
        "outcome": outcome_label,
        "is_success": is_success,
        "is_crash": is_crash,
        "total_steps": step,
        "duration_s": round(step * dt_c, 2),
        "touchdown": {
            "vy": round(float(vy_final), 2),
            "vx": round(float(td.get("vx", env.state[2])), 2),
            "abs_x": round(float(abs_x), 2),
            "tilt_deg": round(float(np.degrees(td.get("theta", env.state[4]))), 2),
            "fuel_remaining_kg": round(float(env.state[6]), 1),
        },
        "steps": steps_data,
        "session_stats": SESSION_STATS,
    }


# --- Mode 2: LunarPilot Artemis Moon Landing Simulation ---
def simulate_lunar_episode(target_x=None, target_y=None, tilt_deg=None, seed=None):
    """Simulates Artemis lunar lander using Box2D PPO master model."""
    global SESSION_STATS, TRAINING_MODEL, EXPERT_MODEL, ACTIVE_LUNAR_MODE

    env = gym.make(ENV_ID)
    if seed is None:
        seed = random.randint(1000, 999999)
    obs, info = env.reset(seed=seed)
    u = env.unwrapped

    w_scale = 600.0 / 30.0 / 2.0
    h_scale = 400.0 / 30.0 / 2.0

    if target_x is not None or target_y is not None or tilt_deg is not None:
        tx = target_x if target_x is not None else 0.0
        ty = target_y if target_y is not None else 1.2
        t_angle = tilt_deg if tilt_deg is not None else 0.0

        box2d_x = tx * w_scale + w_scale
        box2d_y = ty * h_scale + (u.helipad_y + 18.0 / 30.0)
        u.lander.position = (float(box2d_x), float(box2d_y))
        u.lander.angle = float(np.radians(t_angle))
        u.lander.linearVelocity = (0.0, 0.0)
        u.lander.angularVelocity = 0.0

    steps_data = []
    cumulative_reward = 0.0
    done = False
    step = 0
    model = TRAINING_MODEL if ACTIVE_LUNAR_MODE == "self_train" else EXPERT_MODEL

    while not done and step < 1000:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        next_obs, reward, terminated, truncated, info = env.step(action)

        x, y, vx, vy, angle, ang_vel, left_leg, right_leg = [float(v) for v in next_obs]
        if y > 1.45 and vy > 0.05 and step > 10:
            terminated = True
            reward = -100.0

        cumulative_reward += reward
        done = terminated or truncated

        # Mappings
        throttle = 1.0 if action == 2 else 0.0
        gimbal = 0.0
        rcs = 1.0 if action == 1 else (-1.0 if action == 3 else 0.0)

        steps_data.append({
            "step": step,
            "t": round(step * 0.02, 2),
            "x": round(x * 50.0, 2),
            "y": round(y * 100.0, 2),
            "h_foot": round(max(0.0, y * 100.0 - 5.0), 2),
            "vx": round(vx * 10.0, 2),
            "vy": round(vy * 10.0, 2),
            "theta": round(angle, 4),
            "tilt_deg": round(float(np.degrees(angle)), 2),
            "omega_deg": round(float(np.degrees(ang_vel)), 2),
            "throttle": throttle,
            "thrust_kn": 45.0 if throttle > 0.5 else 0.0,
            "gimbal_deg": 0.0,
            "rcs_knm": rcs * 5.0,
            "fuel": round(max(0.0, 1000.0 - step * 1.5), 1),
            "left_contact": bool(left_leg),
            "right_contact": bool(right_leg),
            "legs_deployed": True,
            "action": action,
            "reward": round(float(reward), 2),
            "cumulative_reward": round(float(cumulative_reward), 2),
            "phase": "TOUCHDOWN" if done else ("DESCENT THRUST" if action == 2 else "ATTITUDE ROLL"),
        })

        obs = next_obs
        step += 1

    try:
        detected_outcome, _ = classify_lunarlander_episode(env, terminated, truncated, obs)
        if detected_outcome == Outcome.LANDED_ON_PAD:
            outcome_label = "TOUCHDOWN (Safe Lunar Landing!)"
            is_success = True
            is_crash = False
        elif detected_outcome == Outcome.LANDED_OFF_PAD:
            outcome_label = "TOUCHDOWN (Off-Pad Regolith)"
            is_success = False
            is_crash = False
        elif detected_outcome == Outcome.CRASH:
            outcome_label = "CRASHED ON LUNAR SURFACE"
            is_success = False
            is_crash = True
        else:
            outcome_label = "MISSION TIMEOUT"
            is_success = False
            is_crash = False
    except Exception:
        is_success = cumulative_reward >= 80 and (bool(left_leg) or bool(right_leg))
        is_crash = done and cumulative_reward <= -80
        outcome_label = "TOUCHDOWN (Safe Landing!)" if is_success else ("CRASHED (-100 Penalty)" if is_crash else "OUT OF TIME")

    SESSION_STATS["total_attempts"] += 1
    if is_success:
        SESSION_STATS["success_count"] += 1
    elif is_crash:
        SESSION_STATS["crash_count"] += 1
    else:
        SESSION_STATS["timeout_count"] += 1

    flight_summary = {
        "attempt": SESSION_STATS["total_attempts"],
        "vehicle": "Artemis Lunar Lander (PPO)",
        "outcome": outcome_label,
        "steps": step,
        "reward": round(float(cumulative_reward), 1),
    }
    SESSION_STATS["recent_flights"].insert(0, flight_summary)
    if len(SESSION_STATS["recent_flights"]) > 25:
        SESSION_STATS["recent_flights"].pop()

    env.close()

    return {
        "episode_id": f"LUNAR-{SESSION_STATS['total_attempts']}",
        "vehicle": "artemis",
        "vehicle_name": "Artemis Human Landing System",
        "environment": "moon",
        "title": f"Lunar Flight #{SESSION_STATS['total_attempts']} ({round(cumulative_reward, 1)} pts)",
        "outcome": outcome_label,
        "is_success": is_success,
        "is_crash": is_crash,
        "total_steps": step,
        "duration_s": round(step * 0.02, 2),
        "final_reward": round(float(cumulative_reward), 2),
        "touchdown": {
            "vy": round(float(vy * 10.0), 2),
            "vx": round(float(vx * 10.0), 2),
            "abs_x": round(float(abs(x * 50.0)), 2),
            "tilt_deg": round(float(np.degrees(angle)), 2),
            "fuel_remaining_kg": round(max(0.0, 1000.0 - step * 1.5), 1),
        },
        "steps": steps_data,
        "session_stats": SESSION_STATS,
    }


# --- Request Handler ---
class LunarPilotRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        global ACTIVE_VEHICLE, ACTIVE_LUNAR_MODE, TRAINING_MODEL, CURRENT_STAGE, SESSION_STATS
        parsed = urllib.parse.urlparse(self.path)

        # 1. Next Flight Dispatcher
        if parsed.path == "/api/next_flight":
            query = urllib.parse.parse_qs(parsed.query)
            vehicle = query.get("vehicle", [ACTIVE_VEHICLE])[0]
            if vehicle in ("falcon", "artemis"):
                ACTIVE_VEHICLE = vehicle

            if ACTIVE_VEHICLE == "falcon":
                # Randomized landing burn approach parameters (260-350m for 12-16s landing burn)
                rx = round(random.uniform(-10.0, 10.0), 1)
                ralt = round(random.uniform(260.0, 360.0), 0)
                rvx = round(random.uniform(-1.5, 1.5), 1)
                rvy = round(random.uniform(-45.0, -32.0), 1)
                result = simulate_rocket_episode(alt=ralt, x_offset=rx, vx=rvx, vy=rvy)
            else:
                if ACTIVE_LUNAR_MODE == "self_train":
                    TRAINING_MODEL.learn(total_timesteps=256, reset_num_timesteps=False, progress_bar=False)
                rx = round(random.uniform(-0.5, 0.5), 2)
                ry = round(random.uniform(1.15, 1.35), 2)
                ra = round(random.uniform(-20.0, 20.0), 1)
                result = simulate_lunar_episode(target_x=rx, target_y=ry, tilt_deg=ra)

            self._send_json(result)
            return

        # 2. Simulate Falcon Rocket with custom parameters
        if parsed.path == "/api/simulate_rocket":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                alt = float(query.get("alt", [850.0])[0])
                x = float(query.get("x", [0.0])[0])
                vx = float(query.get("vx", [0.0])[0])
                vy = float(query.get("vy", [-65.0])[0])
                wind = float(query.get("wind", [0.0])[0])
            except ValueError:
                alt, x, vx, vy, wind = 850.0, 0.0, 0.0, -65.0, 0.0

            result = simulate_rocket_episode(alt=alt, x_offset=x, vx=vx, vy=vy, wind=wind)
            self._send_json(result)
            return

        # 3. Simulate Lunar Artemis Lander
        if parsed.path == "/api/simulate_lunar":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                x = float(query.get("x", [0.0])[0])
                y = float(query.get("y", [1.2])[0])
                angle = float(query.get("angle", [0.0])[0])
            except ValueError:
                x, y, angle = 0.0, 1.2, 0.0

            result = simulate_lunar_episode(target_x=x, target_y=y, tilt_deg=angle)
            self._send_json(result)
            return

        # 4. Set Vehicle Type
        if parsed.path == "/api/set_vehicle":
            query = urllib.parse.parse_qs(parsed.query)
            vehicle = query.get("vehicle", ["falcon"])[0]
            if vehicle in ("falcon", "artemis"):
                ACTIVE_VEHICLE = vehicle
            self._send_json({"status": "ok", "vehicle": ACTIVE_VEHICLE})
            return

        # 5. Fetch Session Stats
        if parsed.path == "/api/stats":
            self._send_json({
                "vehicle": ACTIVE_VEHICLE,
                "stats": SESSION_STATS,
            })
            return

        # Default: Serve static files
        return super().do_GET()

    def _send_json(self, data):
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response_bytes)


def run_server(port=8080):
    server = HTTPServer(("0.0.0.0", port), LunarPilotRequestHandler)
    print(f"[OK] LunarPilot & RocketLand Autonomous Engine running on http://localhost:{port}/viewer/")
    server.serve_forever()


if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
