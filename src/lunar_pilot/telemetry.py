"""
LunarPilot - Training Telemetry Callback
Appends real-time metrics to results/training/<run_id>/metrics.jsonl and saves
periodic fixed-seed evaluation trajectories to results/training/<run_id>/snapshots/step_<N>.json.
Reusable for both LunarLander and RocketLand.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from lunar_pilot.outcomes import OutcomeWrapper


class TrainingTelemetryCallback(BaseCallback):
    """
    Writes training telemetry (metrics.jsonl) and checkpoint replay snapshots (step_<N>.json)
    per Appendix L.
    """

    def __init__(
        self,
        run_id: str,
        env_factory: Callable[[], Any],
        eval_freq: int = 10000,
        snapshot_freq: int = 25000,
        n_eval_episodes: int = 10,
        eval_seed_start: int = 10000,  # Uses validation seeds!
        fixed_snapshot_seed: int = 20000,
        output_dir: str | Path = "results/training",
        curriculum_stage: str = "A",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.run_id = run_id
        self.env_factory = env_factory
        self.eval_freq = eval_freq
        self.snapshot_freq = snapshot_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_seed_start = eval_seed_start
        self.fixed_snapshot_seed = fixed_snapshot_seed
        self.curriculum_stage = curriculum_stage

        self.run_dir = Path(output_dir) / run_id
        self.snapshots_dir = self.run_dir / "snapshots"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.meta_file = self.run_dir / "meta.json"
        self._last_time = datetime.now(timezone.utc)
        self._last_step = 0

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            now = datetime.now(timezone.utc)
            step_delta = self.num_timesteps - self._last_step
            time_delta = (now - self._last_time).total_seconds()
            fps = float(step_delta / time_delta) if time_delta > 0 else 0.0
            self._last_time = now
            self._last_step = self.num_timesteps

            # Run evaluation on validation seeds
            eval_rewards = []
            eval_lens = []
            outcome_counts: dict[str, int] = {}

            for ep in range(self.n_eval_episodes):
                seed = self.eval_seed_start + ep
                raw_env = self.env_factory()
                env = OutcomeWrapper(raw_env)
                obs, _ = env.reset(seed=seed)
                done = False
                ep_rew = 0.0
                ep_len = 0
                outcome = "UNKNOWN"

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_rew += reward
                    ep_len += 1
                    done = terminated or truncated
                    if done:
                        outcome = info.get("outcome", "UNKNOWN")
                env.close()

                eval_rewards.append(ep_rew)
                eval_lens.append(ep_len)
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

            n = max(1, len(eval_rewards))
            outcome_rates = {k: round(v / n, 4) for k, v in outcome_counts.items()}

            # Extract SB3 train diagnostics
            entropy = float(self.logger.name_to_value.get("train/entropy_loss", 0.0))
            value_loss = float(self.logger.name_to_value.get("train/value_loss", 0.0))
            policy_loss = float(self.logger.name_to_value.get("train/policy_gradient_loss", 0.0))
            approx_kl = float(self.logger.name_to_value.get("train/approx_kl", 0.0))
            explained_var = float(self.logger.name_to_value.get("train/explained_variance", 0.0))

            metric_row = {
                "step": self.num_timesteps,
                "time_utc": now.isoformat(),
                "fps": round(fps, 1),
                "ep_reward_mean": round(float(np.mean(eval_rewards)), 2),
                "ep_len_mean": round(float(np.mean(eval_lens)), 1),
                "outcome_rates": outcome_rates,
                "entropy": round(entropy, 4),
                "value_loss": round(value_loss, 4),
                "policy_loss": round(policy_loss, 4),
                "approx_kl": round(approx_kl, 5),
                "explained_variance": round(explained_var, 4),
                "curriculum_stage": self.curriculum_stage,
            }

            with open(self.metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(metric_row) + "\n")
                f.flush()

            if self.verbose > 0:
                print(
                    f"[{self.run_id} @ {self.num_timesteps}] "
                    f"R_mean: {metric_row['ep_reward_mean']} | "
                    f"Outcomes: {outcome_rates} | FPS: {metric_row['fps']}"
                )

        # Snapshot check
        if self.snapshot_freq > 0 and (self.n_calls % self.snapshot_freq == 0 or self.n_calls == 1):
            self._save_snapshot(self.num_timesteps)

        return True

    def _save_snapshot(self, step: int):
        raw_env = self.env_factory()
        env = OutcomeWrapper(raw_env)
        obs, _ = env.reset(seed=self.fixed_snapshot_seed)
        done = False
        trajectory = []
        ep_rew = 0.0
        frame_idx = 0
        outcome = "UNKNOWN"
        touchdown = {}

        while not done and frame_idx < 1000:
            action, _ = self.model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                act_val = int(action.item()) if action.ndim == 0 else [float(a) for a in action]
            else:
                act_val = int(action)
            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_rew += reward
            done = terminated or truncated

            trajectory.append({
                "step": frame_idx,
                "state": [round(float(s), 4) for s in obs],
                "action": act_val,
                "reward": round(float(reward), 4),
            })
            obs = next_obs
            frame_idx += 1
            if done:
                outcome = info.get("outcome", "UNKNOWN")
                touchdown = info.get("touchdown", {})

        env.close()

        snapshot_data = {
            "step": step,
            "seed": self.fixed_snapshot_seed,
            "outcome": outcome,
            "touchdown": touchdown,
            "total_reward": round(float(ep_rew), 2),
            "trajectory": trajectory,
        }

        snap_file = self.snapshots_dir / f"step_{step}.json"
        with open(snap_file, "w", encoding="utf-8") as f:
            json.dump(snapshot_data, f, indent=2)
