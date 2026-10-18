"""
LunarPilot - Baseline Benchmark (Random Policy)
Evaluates 100 episodes using random uniform action selection to establish an empirical baseline.
Records rewards, episode lengths, crash rates, and landing success rates.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import os

import gymnasium as gym
import numpy as np


def run_baseline(num_episodes=100, env_id="LunarLander-v3", output_path="results/metrics/baseline_metrics.json"):
    print("=" * 60)
    print("[*] LunarPilot: Running Random Policy Baseline Benchmark")
    print(f"Target Environment: {env_id} | Total Episodes: {num_episodes}")
    print("=" * 60)

    try:
        env = gym.make(env_id)
    except Exception:
        env_id = "LunarLander-v2"
        env = gym.make(env_id)

    episode_rewards = []
    episode_lengths = []
    crashes = 0
    successful_landings = 0

    for episode in range(num_episodes):
        obs, info = env.reset(seed=1000 + episode)
        total_reward = 0.0
        steps = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                episode_rewards.append(total_reward)
                episode_lengths.append(steps)

                # In LunarLander:
                # Terminal crash gives -100 penalty
                # Successful landing gives +100 bonus (total reward typically >= 200, both legs touching ground)
                # obs[6] = left leg contact, obs[7] = right leg contact
                left_contact = bool(obs[6])
                right_contact = bool(obs[7])

                if terminated and total_reward <= -100:
                    crashes += 1
                elif terminated and total_reward >= 100 and (left_contact or right_contact):
                    successful_landings += 1
                break

    env.close()

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    min_reward = float(np.min(episode_rewards))
    max_reward = float(np.max(episode_rewards))
    mean_length = float(np.mean(episode_lengths))
    crash_rate = float(crashes / num_episodes)
    success_rate = float(successful_landings / num_episodes)

    metrics = {
        "policy": "Random Policy",
        "environment": env_id,
        "episodes": num_episodes,
        "mean_reward": round(mean_reward, 2),
        "std_reward": round(std_reward, 2),
        "min_reward": round(min_reward, 2),
        "max_reward": round(max_reward, 2),
        "mean_episode_length": round(mean_length, 2),
        "crash_rate": round(crash_rate, 4),
        "success_rate": round(success_rate, 4),
        "raw_rewards": [round(float(r), 2) for r in episode_rewards],
        "raw_lengths": episode_lengths,
    }

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- Empirical Baseline Results ---")
    print(f"Episodes Evaluated : {num_episodes}")
    print(f"Mean Reward        : {metrics['mean_reward']} +/- {metrics['std_reward']}")
    print(f"Reward Range       : [{metrics['min_reward']}, {metrics['max_reward']}]")
    print(f"Mean Episode Length: {metrics['mean_episode_length']} steps")
    print(f"Crash Rate         : {metrics['crash_rate'] * 100:.1f}%")
    print(f"Landing Success    : {metrics['success_rate'] * 100:.1f}%")
    print(f"Results saved to   : {output_path}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    run_baseline()
