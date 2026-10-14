"""
LunarPilot - PPO Policy Training Pipeline
Orchestrates reproducible training of PPO agent on Gymnasium LunarLander with
parallel environments, checkpoints, TensorBoard logging, and evaluation callbacks.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import json
import argparse
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from lunar_pilot.utils import load_config, set_seed, get_supported_env_id, DetailedEvalCallback


def train(config_path: str = "configs/default.json"):
    print("=" * 60)
    print("[*] LunarPilot - PPO Training Pipeline")
    print(f"Loading configuration from: {config_path}")
    print("=" * 60)

    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    env_id = get_supported_env_id(config.get("environment", "LunarLander-v3"))
    print(f"Target Environment: {env_id}")
    print(f"Parallel Envs     : {config.get('n_envs', 4)}")
    print(f"Total Timesteps   : {config.get('total_timesteps', 500000):,}")
    print(f"Learning Rate     : {config.get('learning_rate', 3e-4)}")

    # Create directories
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("models/best", exist_ok=True)
    os.makedirs("logs/tensorboard", exist_ok=True)
    os.makedirs("logs/evaluations", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # Vectorized training environments
    train_env = make_vec_env(
        env_id,
        n_envs=config.get("n_envs", 4),
        seed=config.get("seed", 42),
    )

    # Dedicated single evaluation environment
    eval_env = gym.make(env_id)

    # TensorBoard run identifier
    run_name = f"PPO_{env_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tensorboard_log_dir = os.path.join(config.get("log_dir", "logs"), "tensorboard")

    # Callbacks
    eval_callback = DetailedEvalCallback(
        eval_env=eval_env,
        eval_freq=config.get("evaluation_interval", 10000) // config.get("n_envs", 4),
        n_eval_episodes=config.get("evaluation_episodes", 10),
        log_dir=os.path.join(config.get("log_dir", "logs"), "evaluations"),
        best_model_save_path=os.path.join(config.get("model_save_dir", "models"), "best"),
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config.get("checkpoint_interval", 50000) // config.get("n_envs", 4),
        save_path=os.path.join(config.get("model_save_dir", "models"), "checkpoints"),
        name_prefix="ppo_lunar",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    callback_list = CallbackList([eval_callback, checkpoint_callback])

    # Initialize PPO Model
    model = PPO(
        policy=config.get("policy", "MlpPolicy"),
        env=train_env,
        learning_rate=config.get("learning_rate", 3e-4),
        n_steps=config.get("n_steps", 2048),
        batch_size=config.get("batch_size", 64),
        n_epochs=config.get("n_epochs", 10),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        clip_range=config.get("clip_range", 0.2),
        ent_coef=config.get("ent_coef", 0.01),
        vf_coef=config.get("vf_coef", 0.5),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        tensorboard_log=tensorboard_log_dir,
        verbose=1,
        seed=config.get("seed", 42),
    )

    print("\nPolicy Architecture:")
    print(model.policy)

    print("\nStarting training loop...")
    start_time = datetime.now()
    try:
        model.learn(
            total_timesteps=config.get("total_timesteps", 500000),
            callback=callback_list,
            tb_log_name=run_name,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user. Saving current checkpoint...")

    duration = datetime.now() - start_time
    print(f"\nTraining completed in {duration}")

    # Save final model
    final_model_path = os.path.join(config.get("model_save_dir", "models"), "final_model.zip")
    model.save(final_model_path)
    print(f"[OK] Final model saved to: {final_model_path}")

    # Save summary metadata
    summary = {
        "timestamp": datetime.now().isoformat(),
        "run_name": run_name,
        "environment": env_id,
        "algorithm": "PPO",
        "total_timesteps": config.get("total_timesteps", 500000),
        "duration_seconds": duration.total_seconds(),
        "final_model_path": final_model_path,
        "best_model_path": os.path.join(config.get("model_save_dir", "models"), "best", "best_model.zip"),
        "config": config,
    }
    with open("results/metrics/training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Clean shutdown
    train_env.close()
    eval_env.close()
    print("[OK] All environments safely closed.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LunarPilot PPO Agent")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to config file")
    args = parser.parse_args()
    train(config_path=args.config)
