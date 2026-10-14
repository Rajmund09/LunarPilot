"""
LunarPilot - Publication-Quality Performance Visualization
Generates high-resolution figures from evaluation logs and comparative benchmarks:
1. Multi-metric Learning Curves (Mean Reward, Success Rate, Crash Rate, Episode Length)
2. Empirical Comparison: Random Baseline vs Trained PPO
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Set aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_learning_curves(
    eval_history_path: str = "logs/evaluations/eval_history.json",
    output_dir: str = "results/figures",
):
    """Plot training progression across evaluation timesteps."""
    if not os.path.exists(eval_history_path):
        print(f"⚠️ Evaluation history not found at '{eval_history_path}'. Skipping learning curve plot.")
        return

    with open(eval_history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        print("⚠️ Evaluation history is empty.")
        return

    timesteps = [entry["timestep"] for entry in history]
    mean_rewards = [entry["mean_reward"] for entry in history]
    std_rewards = [entry.get("std_reward", 0.0) for entry in history]
    success_rates = [entry.get("success_rate", 0.0) * 100 for entry in history]
    crash_rates = [entry.get("crash_rate", 0.0) * 100 for entry in history]
    lengths = [entry.get("mean_length", 0.0) for entry in history]

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # 1. Mean Reward Curve
    ax1 = axes[0, 0]
    ax1.plot(timesteps, mean_rewards, color="#1f77b4", linewidth=2.2, label="PPO Mean Reward")
    lower = np.array(mean_rewards) - np.array(std_rewards)
    upper = np.array(mean_rewards) + np.array(std_rewards)
    ax1.fill_between(timesteps, lower, upper, color="#1f77b4", alpha=0.18, label="±1 Std Dev")
    ax1.axhline(200, color="#2ca02c", linestyle="--", linewidth=1.5, label="Solved Threshold (+200)")
    ax1.axhline(0, color="#7f7f7f", linestyle=":", linewidth=1)
    ax1.set_title("Evaluation Episode Reward over Training", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Environment Timesteps")
    ax1.set_ylabel("Mean Reward")
    ax1.legend(loc="lower right")

    # 2. Landing Success Rate
    ax2 = axes[0, 1]
    ax2.plot(timesteps, success_rates, color="#2ca02c", linewidth=2.2, marker="o", markersize=4)
    ax2.set_title("Landing Success Rate (%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Environment Timesteps")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_ylim(-5, 105)

    # 3. Crash Rate
    ax3 = axes[1, 0]
    ax3.plot(timesteps, crash_rates, color="#d62728", linewidth=2.2, marker="s", markersize=4)
    ax3.set_title("Crash Rate (%)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Environment Timesteps")
    ax3.set_ylabel("Crash Rate (%)")
    ax3.set_ylim(-5, 105)

    # 4. Episode Length (Steps)
    ax4 = axes[1, 1]
    ax4.plot(timesteps, lengths, color="#9467bd", linewidth=2.2)
    ax4.set_title("Mean Episode Length (Steps)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Environment Timesteps")
    ax4.set_ylabel("Steps per Episode")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "learning_curves.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Saved learning curves figure to: {save_path}")


def plot_baseline_vs_ppo(
    baseline_path: str = "results/metrics/baseline_metrics.json",
    ppo_path: str = "results/metrics/ppo_evaluation_metrics.json",
    output_dir: str = "results/figures",
):
    """Generate comparative charts between Random Baseline and Trained PPO."""
    if not (os.path.exists(baseline_path) and os.path.exists(ppo_path)):
        print("⚠️ Both baseline and PPO metrics must exist to plot comparison.")
        return

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    with open(ppo_path, "r", encoding="utf-8") as f:
        ppo = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=300)

    # Subplot 1: Reward Distribution Boxplot
    raw_b = baseline.get("raw_rewards", [])
    raw_p = ppo.get("raw_rewards", [])

    box = ax1.boxplot(
        [raw_b, raw_p],
        patch_artist=True,
        labels=["Random Baseline", "Trained PPO Agent"],
        widths=0.45,
    )
    colors = ["#ff7f0e", "#1f77b4"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.axhline(200, color="#2ca02c", linestyle="--", linewidth=1.4, label="Solved Goal (+200)")
    ax1.set_title("Reward Distribution (100 Episodes)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Episode Reward")
    ax1.legend(loc="upper left")

    # Subplot 2: Success Rate vs Crash Rate
    categories = ["Landing Success Rate", "Crash Rate"]
    b_rates = [baseline.get("success_rate", 0) * 100, baseline.get("crash_rate", 0) * 100]
    p_rates = [ppo.get("success_rate", 0) * 100, ppo.get("crash_rate", 0) * 100]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax2.bar(x - width / 2, b_rates, width, label="Random Baseline", color="#ff7f0e", alpha=0.85)
    rects2 = ax2.bar(x + width / 2, p_rates, width, label="Trained PPO Agent", color="#1f77b4", alpha=0.85)

    ax2.set_title("Outcome Probabilities (%)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, 110)
    ax2.legend()

    # Add labels on bars
    for rect in rects1:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "baseline_vs_ppo.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Saved baseline vs PPO comparison figure to: {save_path}")


def generate_all_plots():
    plot_learning_curves()
    plot_baseline_vs_ppo()


if __name__ == "__main__":
    generate_all_plots()
