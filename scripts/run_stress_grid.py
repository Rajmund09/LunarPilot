"""
Stress grid evaluation across wind power x observation noise scales,
generating heatmap figures in results/figures/ and quantitative metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from lunar_pilot.evaluation import run_episodes, summarize
from lunar_pilot.perturbations import PerturbedLunarLanderWrapper
from lunar_pilot.policies import SB3Policy


def run_stress_grid():
    model_path = Path("models/best/best_robust_model.zip")
    if not model_path.exists():
        model_path = Path("models/best/best_model.zip")
    policy = SB3Policy(model_path)

    # 4 x 4 grid: wind power vs observation noise
    wind_levels = [0.0, 5.0, 10.0, 15.0]
    noise_levels = [0.0, 0.02, 0.05, 0.08]

    # Evaluate each cell on 50 test seeds from test range (20000-20049)
    test_seeds = list(range(20000, 20050))
    grid_rates = np.zeros((len(wind_levels), len(noise_levels)))

    print("[*] Running Robustness Stress Grid Evaluation...")
    for i, w in enumerate(wind_levels):
        for j, n in enumerate(noise_levels):
            def env_factory():
                base = gym.make("LunarLander-v3", enable_wind=(w > 0.0), wind_power=w, turbulence_power=w * 0.1)
                return PerturbedLunarLanderWrapper(base, obs_noise_scale=n, seed=42)

            records = run_episodes(
                policy,
                env_factory,
                test_seeds,
                episodes_dir="results/episodes",
                run_id=f"grid_w{w}_n{n}",
            )
            summary = summarize(records)
            rate = summary["outcome_rates"]["LANDED_ON_PAD"]["rate"]
            grid_rates[i, j] = rate
            print(f"Wind: {w:4.1f} | Noise: {n:4.2f} -> Success: {rate*100:5.1f}%")

    # Generate Heatmap figure
    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.matshow(grid_rates * 100, cmap="viridis", vmin=0, vmax=100)
    fig.colorbar(cax, label="Landing Success Rate (%)")

    ax.set_xticks(range(len(noise_levels)))
    ax.set_yticks(range(len(wind_levels)))
    ax.set_xticklabels([f"{n:.2f}" for n in noise_levels])
    ax.set_yticklabels([f"{w:.1f}" for w in wind_levels])
    ax.set_xlabel("Observation Noise (std)")
    ax.set_ylabel("Wind Power")
    ax.set_title("PPO Robustness Stress Grid: Success Rate (%)", pad=20)

    for i in range(len(wind_levels)):
        for j in range(len(noise_levels)):
            val = grid_rates[i, j] * 100
            color = "white" if val < 50 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontweight="bold")

    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "stress_grid_heatmap.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] Heatmap saved to {fig_path}")

    # Evaluate HELD_OUT_2X on full 1000 test seeds for Gate G1
    print("\n[*] Evaluating on HELD_OUT_2X scenario on 1000 test seeds (Gate G1)...")
    full_test_seeds = list(range(20000, 21000))
    def held_out_env_factory():
        base = gym.make("LunarLander-v3", enable_wind=True, wind_power=10.0, turbulence_power=1.0)
        return PerturbedLunarLanderWrapper(base, obs_noise_scale=0.04, action_delay=1, initial_kick_scale=0.5, seed=42)

    held_out_records = run_episodes(
        policy,
        held_out_env_factory,
        full_test_seeds,
        episodes_dir="results/episodes",
        run_id="held_out_2x_eval",
    )
    held_out_summary = summarize(held_out_records)
    landed_rate = held_out_summary["outcome_rates"]["LANDED_ON_PAD"]["rate"]
    ci95 = held_out_summary["outcome_rates"]["LANDED_ON_PAD"]["ci95"]

    print(f"\n[G1 Result] HELD_OUT_2X Success Rate: {landed_rate*100:.2f}% (Wilson CI: [{ci95[0]*100:.2f}%, {ci95[1]*100:.2f}%])")

    out_metrics = {
        "schema_version": 1,
        "kind": "robustness_evaluation",
        "scenario": "HELD_OUT_2X",
        "n_episodes": len(full_test_seeds),
        "outcome_counts": held_out_summary["outcome_counts"],
        "outcome_rates": held_out_summary["outcome_rates"],
        "reward": held_out_summary["reward"],
        "grid": {
            "wind_levels": wind_levels,
            "noise_levels": noise_levels,
            "success_matrix": grid_rates.tolist(),
        },
    }
    with open("results/metrics/robustness_g1.json", "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)


if __name__ == "__main__":
    run_stress_grid()
