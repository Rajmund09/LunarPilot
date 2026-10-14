"""
LunarPilot - Agent Demonstration & Telemetry Visualizer
Renders the trained PPO agent landing in real-time, displays real-time telemetry,
and prints detailed episode performance summaries.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import time
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from lunar_pilot.utils import get_supported_env_id


ACTION_NAMES = {
    0: "Do Nothing",
    1: "Fire Left Engine  (Rotate Right)",
    2: "Fire Main Engine  (Thrust Up)",
    3: "Fire Right Engine (Rotate Left)",
}


def play(
    model_path: str = "models/best/best_model.zip",
    env_id: str = "LunarLander-v3",
    num_episodes: int = 5,
    fps: int = 50,
    deterministic: bool = True,
):
    print("=" * 60)
    print("[*] LunarPilot - Agent Demonstration & Live Playback")
    print(f"Loading Model: {model_path}")
    print(f"Target Episodes: {num_episodes} | Target FPS: {fps}")
    print("=" * 60)

    if not os.path.exists(model_path):
        fallback = "models/final_model.zip"
        if os.path.exists(fallback):
            print(f"[!] Warning: Model at '{model_path}' not found, falling back to '{fallback}'")
            model_path = fallback
        else:
            print(f"[FAIL] Error: Model file not found at {model_path}")
            print("Please run training first: python -m lunar_pilot.train (or python train.py)")
            sys.exit(1)

    env_id = get_supported_env_id(env_id)

    try:
        env = gym.make(env_id, render_mode="human")
    except Exception as e:
        print(f"[!] Warning: Human rendering failed: {e}. Falling back without display window...")
        env = gym.make(env_id)

    model = PPO.load(model_path)
    delay = 1.0 / fps

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset(seed=int(time.time()) % 10000 + episode)
        total_reward = 0.0
        step = 0
        done = False

        print(f"\n--- Episode {episode}/{num_episodes} Commencing ---")

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated

            # Live telemetry output every 25 steps or on termination
            if step % 25 == 0 or done:
                x_pos, y_pos, vx, vy, angle, ang_vel, left_leg, right_leg = obs
                print(
                    f"Step {step:03d} | Action: {action} ({ACTION_NAMES[action]:<17}) | "
                    f"Pos: ({x_pos:+.2f}, {y_pos:+.2f}) | Vy: {vy:+.2f} | "
                    f"Legs: [{'L' if left_leg else ' '}{'R' if right_leg else ' '}] | "
                    f"Reward: {total_reward:+.1f}"
                )

            time.sleep(delay)

        left_leg = bool(obs[6])
        right_leg = bool(obs[7])
        outcome = "TOUCHDOWN (Safe Landing!)" if (total_reward >= 100 and (left_leg or right_leg)) else (
            "CRASHED" if (terminated and total_reward <= -100) else "TIMED OUT / DRIFTED"
        )

        print(f"--- Episode {episode} Finished ---")
        print(f"Outcome       : {outcome}")
        print(f"Total Steps   : {step}")
        print(f"Total Reward  : {total_reward:.2f}")

    env.close()
    print("\n[OK] Demonstration completed safely.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Demo for LunarPilot PPO Agent")
    parser.add_argument("--model", type=str, default="models/best/best_model.zip", help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to visualize")
    parser.add_argument("--fps", type=int, default=50, help="Frames per second")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions stochastically")
    args = parser.parse_args()

    play(
        model_path=args.model,
        num_episodes=args.episodes,
        fps=args.fps,
        deterministic=not args.stochastic,
    )
