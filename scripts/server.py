"""
LunarPilot - Integrated Web Server & Continuous Autonomous Self-Training Engine

Key Features:
- Space Ceiling Constraint (y > 1.45 aborts with -100 penalty to prevent flying up into deep space)
- Multi-Stage Self-Training:
    * 'cadet' (200k): Actively descends toward lunar surface, makes real mistakes, experiences negative marks and touchdowns
    * 'beginner' (100k): Early balance and hover training
    * 'scratch' (0k): Raw random policy with space ceiling abort
    * 'expert' (500k): Fully converged master policy
- Continuous Online PPO learning updates on every attempt
- Full telemetry streaming for 3D visualizer
"""

import json
import os
import random
import sys
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from lunar_pilot.utils import get_supported_env_id

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
        # obs[1] is y coordinate, obs[3] is vertical velocity vy
        if obs[1] > 1.45 and obs[3] > 0.05:
            terminated = True
            reward = -100.0  # Mission failure: lost in deep space
            info["abort_reason"] = "LOST_IN_SPACE"
        return obs, reward, terminated, truncated, info


TRAINING_ENV = LunarCeilingWrapper(gym.make(ENV_ID))

# Default live training model starts from best trained model
if os.path.exists(BEST_MODEL_PATH):
    print(f"[*] Initializing Live Agent from Best Model: {BEST_MODEL_PATH}")
    TRAINING_MODEL = PPO.load(BEST_MODEL_PATH, env=TRAINING_ENV)
    CURRENT_STAGE = "expert"
elif os.path.exists(CHECKPOINT_200K):
    print(f"[*] Initializing Live Agent from Checkpoint: {CHECKPOINT_200K}")
    TRAINING_MODEL = PPO.load(CHECKPOINT_200K, env=TRAINING_ENV)
    CURRENT_STAGE = "cadet"
else:
    print("[*] Initializing fresh training agent")
    TRAINING_MODEL = PPO("MlpPolicy", TRAINING_ENV, n_steps=256, batch_size=64, n_epochs=4, learning_rate=5e-4, verbose=0)
    CURRENT_STAGE = "scratch"

ACTIVE_MODE = "expert"

SESSION_STATS = {
    "total_attempts": 0,
    "success_count": 0,
    "crash_count": 0,
    "timeout_count": 0,
    "trained_timesteps": 500000 if CURRENT_STAGE == "expert" else 0,
    "all_rewards": [],
    "recent_flights": [],
}


def simulate_episode(model, target_x=None, target_y=None, tilt_deg=None, seed=None):
    """Run an episode in Box2D with altitude ceiling protection and return step telemetry."""
    global SESSION_STATS

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
    lost_in_space = False

    while not done and step < 1000:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        next_obs, reward, terminated, truncated, info = env.step(action)

        # Enforce mission ceiling: abort if rocket tries to flee up into outer space
        x, y, vx, vy, angle, ang_vel, left_leg, right_leg = [float(v) for v in next_obs]
        if y > 1.45 and vy > 0.05 and step > 10:
            terminated = True
            reward = -100.0
            lost_in_space = True

        cumulative_reward += reward
        done = terminated or truncated

        fuel_cost = -0.30 if action == 2 else (-0.03 if action in (1, 3) else 0.0)

        terminal_penalty = 0.0
        if done:
            if lost_in_space:
                terminal_penalty = -100.0
            elif terminated and cumulative_reward <= -80:
                terminal_penalty = -100.0
            elif terminated and cumulative_reward >= 80:
                terminal_penalty = 100.0

        steps_data.append({
            "step": step,
            "x": round(x, 4),
            "y": round(y, 4),
            "vx": round(vx, 4),
            "vy": round(vy, 4),
            "angle": round(angle, 4),
            "angular_velocity": round(ang_vel, 4),
            "left_leg": bool(left_leg),
            "right_leg": bool(right_leg),
            "action": action,
            "reward": round(float(reward), 2),
            "fuel_cost": fuel_cost,
            "terminal_penalty": terminal_penalty,
            "cumulative_reward": round(float(cumulative_reward), 2),
            "lost_in_space": lost_in_space,
        })

        obs = next_obs
        step += 1

    from lunar_pilot.outcomes import classify_lunarlander_episode, Outcome

    x, y, vx, vy, angle, ang_vel, left_leg, right_leg = [float(v) for v in obs]

    if lost_in_space:
        outcome = "ABORT: LOST IN SPACE (-100 Penalty)"
        is_success = False
        is_crash = True
    else:
        try:
            detected_outcome, _ = classify_lunarlander_episode(env, terminated, truncated, obs)
            if detected_outcome == Outcome.LANDED_ON_PAD:
                outcome = "TOUCHDOWN (Safe Landing!)"
                is_success = True
                is_crash = False
            elif detected_outcome == Outcome.LANDED_OFF_PAD:
                outcome = "TOUCHDOWN (Off-Pad)"
                is_success = False
                is_crash = False
            elif detected_outcome == Outcome.CRASH:
                outcome = "CRASHED (-100 Penalty)"
                is_success = False
                is_crash = True
            elif detected_outcome == Outcome.OUT_OF_BOUNDS:
                outcome = "OUT OF BOUNDS"
                is_success = False
                is_crash = True
            else:
                outcome = "OUT OF TIME"
                is_success = False
                is_crash = False
        except Exception:
            is_success = cumulative_reward >= 80 and (bool(left_leg) or bool(right_leg))
            is_crash = done and cumulative_reward <= -80
            outcome = "TOUCHDOWN (Safe Landing!)" if is_success else ("CRASHED (-100 Penalty)" if is_crash else "OUT OF TIME")

    SESSION_STATS["total_attempts"] += 1
    SESSION_STATS["all_rewards"].append(round(cumulative_reward, 1))
    if is_success:
        SESSION_STATS["success_count"] += 1
    elif is_crash:
        SESSION_STATS["crash_count"] += 1
    else:
        SESSION_STATS["timeout_count"] += 1

    avg_rew = round(float(np.mean(SESSION_STATS["all_rewards"])), 1)
    agent_label = f"Self-Training ({CURRENT_STAGE.capitalize()})" if ACTIVE_MODE == "self_train" else "500k PPO Expert"

    flight_summary = {
        "attempt": SESSION_STATS["total_attempts"],
        "outcome": outcome,
        "reward": round(float(cumulative_reward), 1),
        "steps": step,
        "mode": ACTIVE_MODE,
        "stage": CURRENT_STAGE,
        "trained_timesteps": SESSION_STATS["trained_timesteps"],
    }
    SESSION_STATS["recent_flights"].insert(0, flight_summary)
    if len(SESSION_STATS["recent_flights"]) > 25:
        SESSION_STATS["recent_flights"].pop()

    env.close()

    return {
        "episode_id": f"ATTEMPT-{SESSION_STATS['total_attempts']}",
        "title": f"Attempt #{SESSION_STATS['total_attempts']} ({outcome.split()[0]}: {round(cumulative_reward, 1)} pts)",
        "agent_type": agent_label,
        "outcome": outcome,
        "total_steps": step,
        "final_reward": round(float(cumulative_reward), 2),
        "final_position": {"x": round(x, 4), "y": round(y, 4)},
        "steps": steps_data,
        "session_stats": {
            "attempts": SESSION_STATS["total_attempts"],
            "successes": SESSION_STATS["success_count"],
            "crashes": SESSION_STATS["crash_count"],
            "avg_reward": avg_rew,
            "stage": CURRENT_STAGE,
            "trained_timesteps": SESSION_STATS["trained_timesteps"],
            "mode": ACTIVE_MODE,
            "recent_flights": SESSION_STATS["recent_flights"],
        }
    }


class LunarPilotRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        global EXPERT_MODEL, TRAINING_MODEL, ACTIVE_MODE, SESSION_STATS, CURRENT_STAGE
        parsed = urllib.parse.urlparse(self.path)

        # 1. Drive continuous loop with live learning
        if parsed.path == "/api/next_flight":
            if ACTIVE_MODE == "self_train":
                # Online PPO learning update (256 timesteps in ~0.15s)
                TRAINING_MODEL.learn(total_timesteps=256, reset_num_timesteps=False, progress_bar=False)
                SESSION_STATS["trained_timesteps"] += 256

                # Slight random initial perturbation to test learning robustness
                rx = round(random.uniform(-0.4, 0.4), 2)
                ry = round(random.uniform(1.15, 1.30), 2)
                ra = round(random.uniform(-15.0, 15.0), 1)
                result = simulate_episode(TRAINING_MODEL, target_x=rx, target_y=ry, tilt_deg=ra)
            else:
                # 500k Expert evaluation with randomized initial conditions
                rx = round(random.uniform(-0.65, 0.65), 2)
                ry = round(random.uniform(1.10, 1.35), 2)
                ra = round(random.uniform(-25.0, 25.0), 1)
                result = simulate_episode(EXPERT_MODEL, target_x=rx, target_y=ry, tilt_deg=ra)

            self._send_json(result)
            return

        # 2. Custom Spawn Simulation
        if parsed.path == "/api/simulate":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                x = float(query.get("x", [0.0])[0])
                y = float(query.get("y", [1.2])[0])
                angle = float(query.get("angle", [0.0])[0])
            except ValueError:
                x, y, angle = 0.0, 1.2, 0.0

            model_to_use = TRAINING_MODEL if ACTIVE_MODE == "self_train" else EXPERT_MODEL
            result = simulate_episode(model_to_use, target_x=x, target_y=y, tilt_deg=angle)
            self._send_json(result)
            return

        # 3. Set Active Learning Stage (cadet, beginner, scratch, expert)
        if parsed.path == "/api/set_stage":
            query = urllib.parse.parse_qs(parsed.query)
            stage = query.get("stage", ["cadet"])[0]

            if stage == "expert":
                ACTIVE_MODE = "expert"
                CURRENT_STAGE = "expert"
            elif stage == "cadet" and os.path.exists(CHECKPOINT_200K):
                TRAINING_MODEL = PPO.load(CHECKPOINT_200K, env=TRAINING_ENV)
                ACTIVE_MODE = "self_train"
                CURRENT_STAGE = "cadet"
                SESSION_STATS["trained_timesteps"] = 200000
            elif stage == "beginner" and os.path.exists(CHECKPOINT_100K):
                TRAINING_MODEL = PPO.load(CHECKPOINT_100K, env=TRAINING_ENV)
                ACTIVE_MODE = "self_train"
                CURRENT_STAGE = "beginner"
                SESSION_STATS["trained_timesteps"] = 100000
            elif stage == "scratch":
                TRAINING_MODEL = PPO("MlpPolicy", TRAINING_ENV, n_steps=256, batch_size=64, n_epochs=4, learning_rate=5e-4, verbose=0)
                ACTIVE_MODE = "self_train"
                CURRENT_STAGE = "scratch"
                SESSION_STATS["trained_timesteps"] = 0

            result = simulate_episode(TRAINING_MODEL if ACTIVE_MODE == "self_train" else EXPERT_MODEL)
            self._send_json(result)
            return

        # 4. Reset Training (Maps to scratch stage)
        if parsed.path in ("/api/reset_training", "/api/reset_scratch"):
            TRAINING_MODEL = PPO("MlpPolicy", TRAINING_ENV, n_steps=256, batch_size=64, n_epochs=4, learning_rate=5e-4, verbose=0)
            ACTIVE_MODE = "self_train"
            CURRENT_STAGE = "scratch"
            SESSION_STATS["trained_timesteps"] = 0
            result = simulate_episode(TRAINING_MODEL)
            self._send_json(result)
            return

        # 4. Mode Switch
        if parsed.path == "/api/set_mode":
            query = urllib.parse.parse_qs(parsed.query)
            mode = query.get("mode", ["self_train"])[0]
            if mode in ("self_train", "expert"):
                ACTIVE_MODE = mode
            self._send_json({"status": "ok", "mode": ACTIVE_MODE})
            return

        # 5. Fetch Session Stats
        if parsed.path == "/api/stats":
            self._send_json({
                "mode": ACTIVE_MODE,
                "stage": CURRENT_STAGE,
                "stats": SESSION_STATS,
            })
            return

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
    print(f"[OK] LunarPilot Autonomous Server running on http://localhost:{port}/viewer/")
    server.serve_forever()


if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
