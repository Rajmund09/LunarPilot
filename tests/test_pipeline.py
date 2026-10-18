"""
Integration tests for configuration, model initialization, and serialization.
"""

import os

import gymnasium as gym
from stable_baselines3 import PPO

from lunar_pilot.utils import get_supported_env_id, load_config, set_seed


def test_config_loading():
    config = load_config("configs/default.json")
    assert "environment" in config
    assert "algorithm" in config
    assert config["algorithm"] == "PPO"
    assert config["learning_rate"] > 0
    assert config["gamma"] > 0 and config["gamma"] <= 1.0
    assert config["n_steps"] > 0
    assert config["batch_size"] > 0


def test_model_initialization_and_forward_pass():
    set_seed(42)
    env_id = get_supported_env_id("LunarLander-v3")
    env = gym.make(env_id)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        learning_rate=3e-4,
        verbose=0,
    )

    obs, _ = env.reset(seed=42)
    action, _ = model.predict(obs, deterministic=True)

    assert int(action) in [0, 1, 2, 3]
    env.close()


def test_short_training_and_saving(tmp_path):
    set_seed(42)
    env_id = get_supported_env_id("LunarLander-v3")
    env = gym.make(env_id)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        learning_rate=3e-4,
        verbose=0,
    )

    # Train for small step budget
    model.learn(total_timesteps=128)

    save_path = tmp_path / "test_model.zip"
    model.save(str(save_path))
    assert os.path.exists(save_path)

    loaded_model = PPO.load(str(save_path))
    obs, _ = env.reset(seed=999)
    action, _ = loaded_model.predict(obs, deterministic=True)
    assert int(action) in [0, 1, 2, 3]

    env.close()
