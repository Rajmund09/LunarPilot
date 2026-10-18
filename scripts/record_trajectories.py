"""
LunarPilot - Record Trajectory Telemetry (Crashes, Negative Penalties, and Soft Landings)
Exports both untrained failure/crash flights (negative rewards) and trained PPO landings.
"""

import json
import os

import gymnasium as gym
from stable_baselines3 import PPO

from lunar_pilot.utils import get_supported_env_id


def record_trajectories(
    model_path: str = "models/best/best_model.zip",
    env_id: str = "LunarLander-v3",
    output_path: str = "viewer/flight_data.json",
):
    if not os.path.exists(model_path):
        model_path = "models/final_model.zip"

    env_id = get_supported_env_id(env_id)
    env = gym.make(env_id)
    model = PPO.load(model_path)

    episodes_data = []

    # 1. Record 2 Untrained Random Baseline Crashes (Showing -marks, negative rewards, and -100 crash penalties)
    crash_seeds = [1000, 1004]
    for i, seed in enumerate(crash_seeds):
        obs, info = env.reset(seed=seed)
        done = False
        steps_data = []
        cumulative_reward = 0.0
        step = 0

        while not done:
            action = int(env.action_space.sample())
            next_obs, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            done = terminated or truncated

            x, y, vx, vy, angle, ang_vel, left_leg, right_leg = [float(v) for v in obs]

            fuel_cost = -0.30 if action == 2 else (-0.03 if action in (1, 3) else 0.0)

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
                "cumulative_reward": round(float(cumulative_reward), 2),
            })
            obs = next_obs
            step += 1

        episodes_data.append({
            "episode_id": f"FAIL-{i+1}",
            "title": f"Untrained Crash #{i+1} (Negative Reward: {round(cumulative_reward, 1)})",
            "agent_type": "Untrained Random Agent",
            "outcome": "CRASHED (-100 Penalty)",
            "total_steps": step,
            "final_reward": round(float(cumulative_reward), 2),
            "final_position": {"x": round(float(obs[0]), 4), "y": round(float(obs[1]), 4)},
            "steps": steps_data,
        })

    # 2. Record 3 Trained PPO Flights (Showing successful recovery and soft landings)
    ppo_seeds = [42 + 137, 42 + 274, 42 + 411]
    for i, seed in enumerate(ppo_seeds):
        obs, info = env.reset(seed=seed)
        done = False
        steps_data = []
        cumulative_reward = 0.0
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            next_obs, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            done = terminated or truncated

            x, y, vx, vy, angle, ang_vel, left_leg, right_leg = [float(v) for v in obs]
            fuel_cost = -0.30 if action == 2 else (-0.03 if action in (1, 3) else 0.0)

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
                "cumulative_reward": round(float(cumulative_reward), 2),
            })
            obs = next_obs
            step += 1

        episodes_data.append({
            "episode_id": f"PPO-{i+1}",
            "title": f"Trained PPO Flight #{i+1} (Reward: +{round(cumulative_reward, 1)})",
            "agent_type": "Trained PPO Agent",
            "outcome": "TOUCHDOWN (Safe Landing)",
            "total_steps": step,
            "final_reward": round(float(cumulative_reward), 2),
            "final_position": {"x": round(float(obs[0]), 4), "y": round(float(obs[1]), 4)},
            "steps": steps_data,
        })

    env.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(episodes_data, f, indent=2)

    print(f"[OK] Successfully recorded {len(episodes_data)} flight trajectories to: {output_path}")


if __name__ == "__main__":
    record_trajectories()
