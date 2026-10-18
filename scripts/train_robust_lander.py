"""
P1.2 Domain-randomized PPO training for LunarLander.
Trains with structured disturbances (wind, observation noise, action delay, velocity kick)
configured in configs/scenarios.json (TRAIN_DISTURB).
Validates on validation seeds (10000-10049) under disturbance.
Registers robust model in models/registry.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import hashlib

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from lunar_pilot.evaluation import run_episodes, summarize
from lunar_pilot.perturbations import PerturbedLunarLanderWrapper
from lunar_pilot.policies import SB3Policy
from lunar_pilot.provenance import make_provenance
from lunar_pilot.telemetry import TrainingTelemetryCallback


def make_perturbed_env(seed: int, scenario_cfg: dict):
    base = gym.make(
        "LunarLander-v3",
        enable_wind=scenario_cfg.get("enable_wind", False),
        wind_power=scenario_cfg.get("wind_power", 0.0),
        turbulence_power=scenario_cfg.get("turbulence_power", 0.0),
    )
    return PerturbedLunarLanderWrapper(
        base,
        obs_noise_scale=scenario_cfg.get("obs_noise_scale", 0.0),
        action_delay=scenario_cfg.get("action_delay", 0),
        engine_scale=scenario_cfg.get("engine_scale", 1.0),
        initial_kick_scale=scenario_cfg.get("initial_kick_scale", 0.0),
        seed=seed,
    )


def train_robust_seed(seed: int, total_timesteps: int = 500000, n_envs: int = 8):
    run_id = f"ppo_robust_seed_{seed}"
    print(f"\n{'='*60}\n[*] Starting Robust Training Run: {run_id} (Seed={seed})\n{'='*60}")

    with open("configs/scenarios.json") as f:
        scenarios = json.load(f)
    train_disturb_cfg = scenarios["TRAIN_DISTURB"]

    def make_single_env():
        return make_perturbed_env(seed, train_disturb_cfg)

    # Use make_vec_env with callable factory
    train_env = make_vec_env(make_single_env, n_envs=n_envs, seed=seed)

    run_dir = Path("results/training") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "env_id": "LunarLander-v3-Robust",
        "algo": "PPO",
        "seed": seed,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "total_timesteps": total_timesteps,
        "status": "running",
    }
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    telemetry_cb = TrainingTelemetryCallback(
        run_id=run_id,
        env_factory=make_single_env,
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


def select_best_robust(model_paths: list[Path]) -> Path:
    print("\n[*] Selecting best robust model on validation seeds (10000-10049) under TRAIN_DISTURB...")
    with open("configs/scenarios.json") as f:
        scenarios = json.load(f)
    train_disturb_cfg = scenarios["TRAIN_DISTURB"]

    val_seeds = list(range(10000, 10050))
    best_score = -float("inf")
    best_path = model_paths[0]

    for p in model_paths:
        policy = SB3Policy(p)
        records = run_episodes(
            policy,
            lambda: make_perturbed_env(42, train_disturb_cfg),
            val_seeds,
            episodes_dir="results/episodes",
            run_id=f"val_robust_{p.stem}",
        )
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


def register_robust_model(best_path: Path):
    dest_path = Path("models/best/best_robust_model.zip")
    dest_path.write_bytes(best_path.read_bytes())
    sha = hashlib.sha256(dest_path.read_bytes()).hexdigest()

    reg_path = Path("models/registry.json")
    if reg_path.exists():
        with open(reg_path, encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {"models": []}

    # Add or update robust entry
    registry["models"] = [m for m in registry.get("models", []) if m.get("id") != "ppo_lunar_robust"]
    registry["models"].append({
        "id": "ppo_lunar_robust",
        "path": str(dest_path),
        "sha256": sha,
        "env": "LunarLander-v3",
        "trained_on": "TRAIN_DISTURB",
        "eval_file": "results/metrics/robustness_g1.json",
        "registered_utc": datetime.now(timezone.utc).isoformat(),
    })
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    print(f"[OK] Registered robust model in {reg_path} (SHA: {sha[:16]}...)")
    return dest_path


if __name__ == "__main__":
    trained = []
    # Train across seeds 0 and 1
    for s in [0, 1]:
        trained.append(train_robust_seed(seed=s, total_timesteps=400000, n_envs=8))
    best_robust = select_best_robust(trained)
    register_robust_model(best_robust)
