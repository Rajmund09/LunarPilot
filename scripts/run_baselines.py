"""
Script to evaluate baselines (Random, Null, Heuristic) on the 1000 test seeds (20000-20999)
and save to results/metrics/lander_baselines.json with full provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym

from lunar_pilot.evaluation import run_episodes, summarize
from lunar_pilot.policies import GymHeuristicPolicy, NullPolicy, RandomPolicy
from lunar_pilot.provenance import make_provenance


def run_baselines():
    with open("configs/seeds.json", "r", encoding="utf-8") as f:
        seeds_cfg = json.load(f)

    test_range = seeds_cfg["test"]["range"]
    test_seeds = list(range(test_range[0], test_range[1] + 1))
    print(f"[*] Running baselines on {len(test_seeds)} test seeds: {test_range[0]} - {test_range[1]}")

    def env_factory():
        return gym.make("LunarLander-v3")

    # 1. Null Policy
    print("\n--- Running Null Policy ---")
    null_policy = NullPolicy()
    null_records = run_episodes(
        null_policy,
        env_factory,
        seeds=test_seeds,
        episodes_dir="results/episodes",
        run_id="lander_null_baseline",
    )
    null_summary = summarize(null_records)
    print(f"Null Policy: {null_summary['outcome_counts']}")
    print(f"Mean reward: {null_summary['reward']['mean']:.2f}")

    # 2. Random Policy
    print("\n--- Running Random Policy (Seed=42) ---")
    random_policy = RandomPolicy(n_actions=4, seed=42)
    rand_records = run_episodes(
        random_policy,
        env_factory,
        seeds=test_seeds,
        episodes_dir="results/episodes",
        run_id="lander_random_baseline",
    )
    rand_summary = summarize(rand_records)
    print(f"Random Policy: {rand_summary['outcome_counts']}")
    print(f"Mean reward: {rand_summary['reward']['mean']:.2f}")

    # 3. Gymnasium Built-in Heuristic Policy
    print("\n--- Running Heuristic Policy ---")
    eval_env = env_factory()
    heuristic_policy = GymHeuristicPolicy(eval_env)
    heur_records = run_episodes(
        heuristic_policy,
        env_factory,
        seeds=test_seeds,
        episodes_dir="results/episodes",
        run_id="lander_heuristic_baseline",
    )
    eval_env.close()
    heur_summary = summarize(heur_records)
    print(f"Heuristic Policy: {heur_summary['outcome_counts']}")
    print(f"Mean reward: {heur_summary['reward']['mean']:.2f}")

    out_data = {
        "schema_version": 1,
        "kind": "baselines",
        "env_id": "LunarLander-v3",
        "seeds": {
            "range": test_range,
            "count": len(test_seeds),
        },
        "baselines": {
            "null": {
                "policy": {"name": "NullPolicy"},
                "summary": null_summary,
                "episodes_file": "results/episodes/lander_null_baseline.jsonl",
            },
            "random": {
                "policy": {"name": "RandomPolicy", "seed": 42},
                "summary": rand_summary,
                "episodes_file": "results/episodes/lander_random_baseline.jsonl",
            },
            "heuristic": {
                "policy": {"name": "GymHeuristicPolicy"},
                "summary": heur_summary,
                "episodes_file": "results/episodes/lander_heuristic_baseline.jsonl",
            },
        },
        "provenance": make_provenance(config=seeds_cfg, seeds=test_seeds),
    }

    out_path = Path("results/metrics/lander_baselines.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"\n[OK] Baselines saved to {out_path}")


if __name__ == "__main__":
    run_baselines()
