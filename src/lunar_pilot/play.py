"""
LunarPilot - Agent Demonstration & Telemetry Visualizer
Renders the trained PPO agent or random baseline in real-time, displays live telemetry,
and prints detailed episode performance summaries.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import os
import time

import gymnasium as gym
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
    random_policy: bool = False,
):
    print("=" * 60)
    print("[*] LunarPilot - Agent Demonstration & Live Playback")
    print(f"Target Episodes: {num_episodes} | Target FPS: {fps}")

    model = None
    if not random_policy:
        if not os.path.exists(model_path):
            fallback = "models/final_model.zip"
            if os.path.exists(fallback):
                print(f"[!] Model at '{model_path}' not found, falling back to '{fallback}'")
                model_path = fallback
            else:
                print(f"[!] No trained model found at '{model_path}'.")
                print("    Automatically switching to >> RANDOM BASELINE AGENT << for live demonstration!")
                print("    (To visualize trained PPO, run training first: python train.py)")
                random_policy = True

    if not random_policy:
        print(f"Loading Trained Model: {model_path}")
        model = PPO.load(model_path)
    else:
        print("Policy Mode: Random Exploration Baseline")
    print("=" * 60)

    env_id = get_supported_env_id(env_id)

    try:
        env = gym.make(env_id, render_mode="human")
    except Exception as e:
        print(f"[!] Human rendering window could not open: {e}. Falling back to headless simulation...")
        env = gym.make(env_id)

    delay = 1.0 / fps

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset(seed=int(time.time()) % 10000 + episode)
        total_reward = 0.0
        step = 0
        done = False

        print(f"\n--- Episode {episode}/{num_episodes} Commencing ---")

        while not done:
            if model is not None and not random_policy:
                action, _ = model.predict(obs, deterministic=deterministic)
                action = int(action)
            else:
                action = int(env.action_space.sample())

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
    parser.add_argument("--random", action="store_true", help="Visualize random policy")
    args = parser.parse_args()

    play(
        model_path=args.model,
        num_episodes=args.episodes,
        fps=args.fps,
        deterministic=not args.stochastic,
        random_policy=args.random,
    )
