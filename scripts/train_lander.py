"""
P0.5 Training pipeline for LunarLander PPO across 5 seeds.
Performs checkpoint selection strictly on validation seeds (10000-10199),
evaluates final selected models on test seeds (20000-20999),
and updates models/registry.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from lunar_pilot.evaluation import run_episodes, summarize
from lunar_pilot.policies import SB3Policy
from lunar_pilot.provenance import make_provenance
from lunar_pilot.telemetry import TrainingTelemetryCallback


def train_seed(seed: int, total_timesteps: int = 400000, n_envs: int = 8):
    run_id = f"ppo_seed_{seed}"
    print(f"\n{'='*60}\n[*] Starting Training Run: {run_id} (Seed={seed})\n{'='*60}")

    env_id = "LunarLander-v3"
    train_env = make_vec_env(env_id, n_envs=n_envs, seed=seed)

    def env_factory():
        return gym.make(env_id)

    run_dir = Path("results/training") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "env_id": env_id,
        "algo": "PPO",
        "seed": seed,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "total_timesteps": total_timesteps,
        "status": "running",
    }
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Telemetry callback (evaluates every 10k steps on validation seeds, snapshots every 25k)
    telemetry_cb = TrainingTelemetryCallback(
        run_id=run_id,
        env_factory=env_factory,
        eval_freq=10000 // n_envs,
        snapshot_freq=25000 // n_envs,
        n_eval_episodes=15,
        eval_seed_start=10000,
        fixed_snapshot_seed=20000,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50000 // n_envs,
        save_path=f"models/checkpoints/{run_id}",
        name_prefix="ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # RL Zoo tuned hyperparameters
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        learning_rate=3e-4,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps, callback=CallbackList([telemetry_cb, checkpoint_cb]))

    final_model_path = Path(f"models/trained/{run_id}_final.zip")
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(final_model_path))

    meta["status"] = "finished"
    meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    train_env.close()
    print(f"[OK] Run {run_id} complete. Model saved to {final_model_path}")
    return final_model_path


def select_best_by_validation(model_paths: list[Path]) -> Path:
    print("\n[*] Selecting best checkpoint strictly on validation seeds (10000-10049)...")
    val_seeds = list(range(10000, 10050))
    best_score = -float("inf")
    best_path = model_paths[0]

    for p in model_paths:
        policy = SB3Policy(p)
        records = run_episodes(policy, lambda: gym.make("LunarLander-v3"), val_seeds, episodes_dir="results/episodes", run_id=f"val_{p.stem}")
        summary = summarize(records)
        land_rate = summary["outcome_rates"]["LANDED_ON_PAD"]["rate"]
        mean_r = summary["reward"]["mean"]
        print(f"Model {p.name}: LANDED_ON_PAD={land_rate*100:.1f}%, mean_reward={mean_r:.2f}")
        score = land_rate * 1000.0 + mean_r
        if score > best_score:
            best_score = score
            best_path = p

    print(f"[BEST] Selected {best_path} by validation performance!")
    return best_path


def evaluate_on_test_seeds(model_path: Path):
    with open("configs/seeds.json") as f:
        seeds_cfg = json.load(f)
    test_range = seeds_cfg["test"]["range"]
    test_seeds = list(range(test_range[0], test_range[1] + 1))
    print(f"\n[*] Evaluating best model {model_path.name} on {len(test_seeds)} test seeds ({test_range[0]}-{test_range[1]})...")

    policy = SB3Policy(model_path)
    records = run_episodes(policy, lambda: gym.make("LunarLander-v3"), test_seeds, episodes_dir="results/episodes", run_id="lander_ppo_test")
    summary = summarize(records)

    test_result = {
        "schema_version": 1,
        "kind": "evaluation",
        "env_id": "LunarLander-v3",
        "scenario": "NOMINAL",
        "policy": {
            "name": "PPO_LunarLander",
            "model_path": str(model_path),
            "deterministic": True,
        },
        "seeds": {
            "range": test_range,
            "count": len(test_seeds),
        },
        "n_episodes": len(records),
        "outcome_counts": summary["outcome_counts"],
        "outcome_rates": summary["outcome_rates"],
        "reward": summary["reward"],
        "touchdown": summary["touchdown"],
        "provenance": make_provenance(model_path=model_path, seeds=test_seeds),
        "episodes_file": "results/episodes/lander_ppo_test.jsonl",
    }

    out_file = Path("results/metrics/lander_ppo_test.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(test_result, f, indent=2)

    print(f"\n[OK] Gate G0 test metrics saved to {out_file}:")
    print(f"LANDED_ON_PAD count: {summary['outcome_counts']['LANDED_ON_PAD']} / {len(test_seeds)}")
    print(f"Rate: {summary['outcome_rates']['LANDED_ON_PAD']['rate']*100:.2f}% (Wilson CI: {summary['outcome_rates']['LANDED_ON_PAD']['ci95']})")
    print(f"Mean reward: {summary['reward']['mean']:.2f} +/- {summary['reward']['std']:.2f}")
    return test_result


def main():
    seeds = [0, 1, 2, 3, 4]
    model_paths = []
    for s in seeds:
        path = train_seed(s, total_timesteps=400000, n_envs=8)
        model_paths.append(path)

    best_model = select_best_by_validation(model_paths)

    # Save to models/best/best_model.zip as standard registry location
    best_copy_path = Path("models/best/best_model.zip")
    best_copy_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(best_model, best_copy_path)

    test_res = evaluate_on_test_seeds(best_copy_path)

    # Register in models/registry.json
    import hashlib
    with open(best_copy_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    registry = {
        "models": [
            {
                "id": "ppo_lunarlander_v3",
                "name": "PPO LunarLander-v3 Master Policy",
                "env": "LunarLander-v3",
                "path": str(best_copy_path),
                "sha256": sha256,
                "trained_on": "LunarLander-v3 nominal (seeds 0-4)",
                "eval_file": "results/metrics/lander_ppo_test.json",
                "success_rate": test_res["outcome_rates"]["LANDED_ON_PAD"]["rate"],
                "wilson_ci95": test_res["outcome_rates"]["LANDED_ON_PAD"]["ci95"],
            }
        ]
    }
    with open("models/registry.json", "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    print(f"[OK] Model registered in models/registry.json with sha256={sha256}")


if __name__ == "__main__":
    main()
