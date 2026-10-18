"""
LunarPilot - Environment Verification Script
Verifies that Gymnasium and Box2D simulator bindings are installed and functioning properly.
"""

import sys

# Configure UTF-8 stdout on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import gymnasium as gym
import numpy as np


def verify():
    print("=" * 60)
    print("[*] LunarPilot Environment Verification")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Gymnasium Version: {gym.__version__}")
    print("=" * 60)

    # Determine environment id (prefer LunarLander-v3, fallback to v2 if needed)
    env_id = "LunarLander-v3"
    try:
        gym.spec(env_id)
    except gym.error.Error:
        env_id = "LunarLander-v2"
    except Exception:
        pass

    print(f"\n[1/4] Instantiating environment '{env_id}'...")
    try:
        env = gym.make(env_id)
    except Exception as e:
        print(f"[FAIL] Failed to instantiate '{env_id}': {e}")
        print("\nTroubleshooting tips:")
        print("  - Ensure 'gymnasium[box2d]' is installed.")
        print("  - On Windows, install swig via conda/pip or ensure Visual C++ Build Tools exist.")
        sys.exit(1)

    print(f"[OK] Environment successfully instantiated!")

    print("\n[2/4] Inspecting Spaces...")
    obs, info = env.reset(seed=42)
    print(f"  - Observation Shape: {obs.shape} (Expected: (8,))")
    print(f"  - Observation Dtype: {obs.dtype}")
    print(f"  - Initial Observation Sample: {np.round(obs, 4)}")
    print(f"  - Observation Space: {env.observation_space}")
    print(f"  - Action Space: {env.action_space} (Expected: Discrete(4))")

    assert obs.shape == (8,), f"Unexpected observation shape: {obs.shape}"
    assert env.action_space.n == 4, f"Unexpected action space size: {env.action_space.n}"

    print("\n[3/4] Running 100 random action steps...")
    terminated_count = 0
    truncated_count = 0
    total_steps = 0

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        if terminated or truncated:
            if terminated:
                terminated_count += 1
            if truncated:
                truncated_count += 1
            obs, info = env.reset()

    print(f"  - Executed {total_steps} steps smoothly.")
    print(f"  - Terminated episodes: {terminated_count}, Truncated episodes: {truncated_count}")

    print("\n[4/4] Verifying headless rgb_array rendering...")
    render_env = gym.make(env_id, render_mode="rgb_array")
    render_env.reset(seed=42)
    frame = render_env.render()
    assert isinstance(frame, np.ndarray), f"Expected render frame ndarray, got {type(frame)}"
    assert frame.ndim == 3 and frame.shape[2] == 3, f"Unexpected render frame shape: {frame.shape}"
    render_env.close()
    print(f"[OK] Headless rgb_array render verified! Frame shape: {frame.shape}")

    print("\n[Clean shutdown...]")
    env.close()
    print("[OK] Environment closed safely.")

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL CHECKS PASSED! The environment is ready for training.")
    print("=" * 60)


if __name__ == "__main__":
    verify()
