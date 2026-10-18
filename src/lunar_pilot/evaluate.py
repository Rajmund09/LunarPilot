"""
LunarPilot - Scientific Evaluation Pipeline
Evaluates a trained policy across N episodes (default 100) using rigorous metrics:
mean reward, reward variance, landing success rate, crash rate, fuel efficiency,
and compares with the empirical random baseline.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import json
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from lunar_pilot.utils import get_supported_env_id, set_seed


def evaluate_policy(
    model_path: str = "models/best/best_model.zip",
    env_id: str = "LunarLander-v3",
    n_episodes: int = 100,
    seed: int = 5000,
    deterministic: bool = True,
    output_path: str = "results/metrics/ppo_evaluation_metrics.json",
):
    print("=" * 60)
    print("[*] LunarPilot - Scientific Policy Evaluation")
    print(f"Model Path        : {model_path}")
    print(f"Number of Episodes: {n_episodes}")
    print(f"Deterministic     : {deterministic}")
    print("=" * 60)

    if not os.path.exists(model_path):
        fallback = "models/final_model.zip"
        if os.path.exists(fallback):
            print(f"[!] Warning: Model at '{model_path}' not found, falling back to '{fallback}'")
            model_path = fallback
        else:
            raise FileNotFoundError(f"No trained model found at {model_path} or {fallback}.")

    env_id = get_supported_env_id(env_id)
    env = gym.make(env_id)
    model = PPO.load(model_path)

    set_seed(seed)

    rewards = []
    lengths = []
    crashes = 0
    successes = 0
    main_engine_fires = 0
    side_engine_fires = 0
    final_drifts = []
    final_vy_list = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_len = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)

            # Track fuel usage proxy
            if action == 2:
                main_engine_fires += 1
            elif action in (1, 3):
                side_engine_fires += 1

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            done = terminated or truncated

            if done:
                rewards.append(ep_reward)
                lengths.append(ep_len)
                final_drifts.append(abs(float(obs[0])))
                final_vy_list.append(float(obs[3]))

                left_contact = bool(obs[6])
                right_contact = bool(obs[7])

                # LunarLander termination logic:
                # Terminal crash gives -100 penalty
                # Successful landing gives +100 bonus (total reward >= 100 with ground contact)
                if terminated and ep_reward <= -100:
                    crashes += 1
                elif (terminated or truncated) and ep_reward >= 100 and (left_contact or right_contact):
                    successes += 1

    env.close()

    total_actions = sum(lengths)
    results = {
        "model_path": model_path,
        "environment": env_id,
        "episodes": n_episodes,
        "deterministic": deterministic,
        "mean_reward": round(float(np.mean(rewards)), 2),
        "std_reward": round(float(np.std(rewards)), 2),
        "min_reward": round(float(np.min(rewards)), 2),
        "max_reward": round(float(np.max(rewards)), 2),
        "median_reward": round(float(np.median(rewards)), 2),
        "mean_episode_length": round(float(np.mean(lengths)), 2),
        "success_rate": round(float(successes / n_episodes), 4),
        "crash_rate": round(float(crashes / n_episodes), 4),
        "main_engine_usage_pct": round(float(main_engine_fires / max(1, total_actions)), 4),
        "side_engine_usage_pct": round(float(side_engine_fires / max(1, total_actions)), 4),
        "mean_final_abs_drift": round(float(np.mean(final_drifts)), 4),
        "mean_final_vy": round(float(np.mean(final_vy_list)), 4),
        "raw_rewards": [round(float(r), 2) for r in rewards],
        "raw_lengths": lengths,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"[*] PPO EVALUATION RESULTS ({n_episodes} Episodes)")
    print("=" * 60)
    print(f"Mean Episode Reward   : {results['mean_reward']} +/- {results['std_reward']}")
    print(f"Median Reward         : {results['median_reward']}")
    print(f"Reward Range          : [{results['min_reward']}, {results['max_reward']}]")
    print(f"Landing Success Rate  : {results['success_rate'] * 100:.1f}%")
    print(f"Crash Rate            : {results['crash_rate'] * 100:.1f}%")
    print(f"Mean Episode Length   : {results['mean_episode_length']} steps")
    print(f"Main Engine Usage     : {results['main_engine_usage_pct'] * 100:.1f}%")
    print(f"Side Engine Usage     : {results['side_engine_usage_pct'] * 100:.1f}%")
    print(f"Mean Final Lat Drift  : {results['mean_final_abs_drift']}")
    print(f"Results Saved To      : {output_path}")

    # Check and display comparison with baseline if available
    baseline_path = "results/metrics/baseline_metrics.json"
    if os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        print("\n" + "-" * 60)
        print(f"{'Metric':<25} | {'Random Baseline':<16} | {'Trained PPO':<16}")
        print("-" * 60)
        print(f"{'Mean Reward':<25} | {baseline.get('mean_reward', 'N/A'):<16} | {results['mean_reward']:<16}")
        print(f"{'Reward Std Dev':<25} | {baseline.get('std_reward', 'N/A'):<16} | {results['std_reward']:<16}")
        print(f"{'Success Rate':<25} | {baseline.get('success_rate', 0)*100:.1f}%{'':<10} | {results['success_rate']*100:.1f}%")
        print(f"{'Crash Rate':<25} | {baseline.get('crash_rate', 0)*100:.1f}%{'':<10} | {results['crash_rate']*100:.1f}%")
        print(f"{'Mean Episode Length':<25} | {baseline.get('mean_episode_length', 'N/A'):<16} | {results['mean_episode_length']:<16}")
        print("-" * 60)

    print("=" * 60)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LunarPilot Policy")
    parser.add_argument("--model", type=str, default="models/best/best_model.zip", help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=5000, help="Random seed for evaluation")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions instead of deterministic")
    parser.add_argument("--output", type=str, default="results/metrics/ppo_evaluation_metrics.json", help="Output path")
    args = parser.parse_args()

    evaluate_policy(
        model_path=args.model,
        n_episodes=args.episodes,
        seed=args.seed,
        deterministic=not args.stochastic,
        output_path=args.output,
    )
