"""
LunarPilot - Utility Functions and Custom RL Callbacks
Includes custom evaluation callbacks, metric tracking, seed fixing, and environment helpers.
"""

import os
import json
import random
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


def load_config(config_path: str = "configs/default.json") -> dict:
    """Load JSON experiment configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def set_seed(seed: int = 42):
    """Seed Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_supported_env_id(preferred_id: str = "LunarLander-v3") -> str:
    """Check if preferred environment is available, fallback if needed."""
    try:
        gym.spec(preferred_id)
        return preferred_id
    except gym.error.NamespaceNotFound:
        return "LunarLander-v2"
    except Exception:
        return preferred_id


class DetailedEvalCallback(BaseCallback):
    """
    Custom callback that evaluates the agent periodically, computes detailed
    metrics (reward, landing success rate, crash rate, episode length), logs to
    TensorBoard, and records history for plotting.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        log_dir: str = "logs/evaluations",
        best_model_save_path: str = "models/best",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_dir = log_dir
        self.best_model_save_path = best_model_save_path
        self.best_mean_reward = -float("inf")
        self.evaluation_history = []

        os.makedirs(self.log_dir, exist_ok=True)
        if self.best_model_save_path:
            os.makedirs(self.best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            rewards = []
            lengths = []
            crashes = 0
            successes = 0

            for ep in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset(seed=2000 + ep)
                done = False
                ep_reward = 0.0
                ep_len = 0

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    action = int(action)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    ep_reward += reward
                    ep_len += 1
                    done = terminated or truncated

                    if done:
                        # obs[6] = left leg contact, obs[7] = right leg contact
                        left_contact = bool(obs[6])
                        right_contact = bool(obs[7])
                        rewards.append(ep_reward)
                        lengths.append(ep_len)

                        if terminated and ep_reward <= -100:
                            crashes += 1
                        elif (terminated or truncated) and ep_reward >= 100 and (left_contact or right_contact):
                            successes += 1

            mean_reward = float(np.mean(rewards))
            std_reward = float(np.std(rewards))
            mean_length = float(np.mean(lengths))
            crash_rate = float(crashes / self.n_eval_episodes)
            success_rate = float(successes / self.n_eval_episodes)

            # Record in TensorBoard
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/mean_ep_length", mean_length)
            self.logger.record("eval/crash_rate", crash_rate)
            self.logger.record("eval/success_rate", success_rate)

            eval_entry = {
                "timestep": self.num_timesteps,
                "mean_reward": round(mean_reward, 2),
                "std_reward": round(std_reward, 2),
                "mean_length": round(mean_length, 2),
                "crash_rate": round(crash_rate, 4),
                "success_rate": round(success_rate, 4),
            }
            self.evaluation_history.append(eval_entry)

            # Save evaluation history json
            history_path = os.path.join(self.log_dir, "eval_history.json")
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self.evaluation_history, f, indent=2)

            if self.verbose > 0:
                print(
                    f"\n[Eval @ Step {self.num_timesteps}] "
                    f"Reward: {mean_reward:.2f} +/- {std_reward:.2f} | "
                    f"Success: {success_rate * 100:.1f}% | "
                    f"Crash: {crash_rate * 100:.1f}% | "
                    f"Length: {mean_length:.1f}"
                )

            # Check if this is the best model so far
            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print(
                        f"[BEST] New best model found! ({self.best_mean_reward:.2f} -> {mean_reward:.2f}). Saving checkpoint..."
                    )
                self.best_mean_reward = mean_reward
                if self.best_model_save_path:
                    save_path = os.path.join(self.best_model_save_path, "best_model.zip")
                    self.model.save(save_path)

        return True
