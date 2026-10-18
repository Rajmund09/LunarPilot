"""
RocketLand-2D — RL Training Pipeline (P2.7)
Trains continuous neural policy (PPO + Imitation/BC initialization) for RocketLand2DEnv.
Includes VecNormalize observation/reward wrapping, validation checkpoint selection, and model registration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch.utils.data import DataLoader, TensorDataset

from rocketland.config import RocketConfig
from rocketland.controllers.suicide_burn import SuicideBurnController
from rocketland.env import RocketLand2DEnv


def generate_demonstrations(cfg: RocketConfig, n_episodes: int = 400, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generates (obs, action) trajectory pairs from SuicideBurnController."""
    env = RocketLand2DEnv(cfg)
    ctrl = SuicideBurnController(cfg)
    all_obs = []
    all_act = []

    for ep in range(n_episodes):
        ep_seed = seed + ep * 13
        obs, info = env.reset(seed=ep_seed)
        ctrl.reset()
        done = False
        step = 0
        while not done and step < 500:
            act = ctrl(obs)
            all_obs.append(obs)
            all_act.append(act)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            step += 1

    return np.array(all_obs, dtype=np.float32), np.array(all_act, dtype=np.float32)


def train_bc_policy(
    model: PPO,
    obs_data: np.ndarray,
    act_data: np.ndarray,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    """Supervised imitation pre-training into the PPO actor network."""
    dataset = TensorDataset(torch.from_numpy(obs_data), torch.from_numpy(act_data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    actor = model.policy.action_net
    mlp = model.policy.mlp_extractor.policy_net
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(mlp.parameters()), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for b_obs, b_act in loader:
            features = model.policy.extract_features(b_obs)
            latent_pi = mlp(features)
            mean_actions = actor(latent_pi)
            loss = criterion(mean_actions, b_act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(b_obs)

        epoch_loss = total_loss / len(obs_data)
        if (epoch + 1) % 5 == 0:
            print(f"  [BC Pretrain] Epoch {epoch+1}/{epochs} - MSE Loss: {epoch_loss:.5f}")


def evaluate_policy(
    model: PPO,
    cfg: RocketConfig,
    seeds: list[int],
    vec_norm: VecNormalize | None = None,
) -> dict:
    env = RocketLand2DEnv(cfg)
    success_count = 0
    records = []

    for s in seeds:
        obs, info = env.reset(seed=s)
        done = False
        step = 0
        while not done and step < 400:
            norm_obs = vec_norm.normalize_obs(obs) if vec_norm else obs
            action, _ = model.predict(norm_obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            step += 1

        td = env.first_touchdown_kinematics or {}
        abs_x = td.get("abs_x", abs(float(env.state[0])))
        vy_final = td.get("vy", float(env.state[3]))
        vx_final = td.get("vx", float(env.state[2]))
        theta_deg = np.degrees(td.get("theta", float(env.state[4])))

        on_pad = abs_x <= cfg.pad_half_width
        safe_speed = abs(vy_final) <= cfg.touchdown_vy_limit and abs(vx_final) <= cfg.touchdown_vx_limit
        safe_tilt = abs(theta_deg) <= cfg.touchdown_tilt_limit_deg
        succ = on_pad and safe_speed and safe_tilt
        if succ:
            success_count += 1
        records.append({
            "seed": s,
            "success": succ,
            "x": abs_x,
            "vx": vx_final,
            "vy": vy_final,
            "tilt_deg": theta_deg,
        })

    rate = success_count / len(seeds)
    return {"success_rate": rate, "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20000, help="RL fine-tuning steps")
    parser.add_argument("--bc-epochs", type=int, default=15, help="Behavior cloning epochs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = RocketConfig()
    print(f"[*] Initializing RocketLand2D PPO Training (seed={args.seed})...")

    # 1. Generate demonstrations
    print("[*] Generating demonstration trajectories from SuicideBurnController...")
    obs_data, act_data = generate_demonstrations(cfg, n_episodes=300, seed=args.seed)
    print(f"[*] Generated {len(obs_data)} state-action transition pairs.")

    # 2. Build VecEnv
    def make_env():
        return RocketLand2DEnv(cfg)

    vec_env = DummyVecEnv([make_env])
    vec_norm = VecNormalize(vec_env, norm_obs=False, norm_reward=False)

    # 3. Create SB3 PPO model
    model = PPO(
        "MlpPolicy",
        vec_norm,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        seed=args.seed,
        verbose=0,
    )

    # 4. Supervised imitation pre-training
    print("[*] Pre-training PPO policy via Behavior Cloning (Imitation Learning)...")
    train_bc_policy(model, obs_data, act_data, epochs=args.bc_epochs, lr=1e-3)

    # 5. RL fine-tuning
    if args.timesteps > 0:
        print(f"[*] RL fine-tuning PPO on RocketLand-2D for {args.timesteps} timesteps...")
        model.learn(total_timesteps=args.timesteps)

    # 6. Validation selection
    val_seeds = list(range(10000, 10050))
    print(f"[*] Evaluating policy on {len(val_seeds)} validation seeds (10000-10049)...")
    val_res = evaluate_policy(model, cfg, val_seeds)
    print(f"[*] Validation Success Rate: {val_res['success_rate']*100:.2f}%")

    # 7. Save model and normalization stats
    save_dir = Path("models/rocketland_ppo")
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "rocket_ppo_best.zip"
    norm_path = save_dir / "vec_normalize.pkl"

    model.save(str(model_path))
    vec_norm.save(str(norm_path))

    # Compute SHA-256
    with open(model_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"[*] Saved model to {model_path} (SHA-256: {sha256})")

    # 8. Register in models/registry.json
    reg_path = Path("models/registry.json")
    if reg_path.exists():
        with open(reg_path, "r") as f:
            reg = json.load(f)
    else:
        reg = {"models": []}

    reg["models"] = [m for m in reg["models"] if m.get("id") != "ppo_rocketland_v1"]
    reg["models"].append({
        "id": "ppo_rocketland_v1",
        "name": "PPO RocketLand-2D Guidance Policy",
        "env": "RocketLand-2D",
        "path": str(model_path),
        "sha256": sha256,
        "trained_on": "RocketLand-2D NOMINAL + BC Pretrain",
        "val_success_rate": val_res["success_rate"],
        "eval_file": "results/metrics/rocketland_g2b.json",
    })

    with open(reg_path, "w") as f:
        json.dump(reg, f, indent=2)
    print("[*] Successfully registered ppo_rocketland_v1 in models/registry.json")


if __name__ == "__main__":
    main()
