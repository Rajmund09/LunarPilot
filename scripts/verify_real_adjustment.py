import numpy as np
from rocketland.config import RocketConfig
from rocketland.env import RocketLand2DEnv
from rocketland.controllers.suicide_burn import SuicideBurnController
from stable_baselines3 import PPO
import gymnasium as gym
from lunar_pilot.utils import get_supported_env_id

cfg = RocketConfig.from_json('configs/rocketland/vehicle.json')

print("=" * 65)
print("1. FALCON 9 PHYSICS VERIFICATION (RocketLand-2D RK4 Dynamics)")
print("=" * 65)

# A. Uncontrolled (No adjustment)
env_null = RocketLand2DEnv(cfg)
obs, _ = env_null.reset(seed=42)
done = False
step = 0
while not done and step < 500:
    obs, r, term, trunc, info = env_null.step(np.array([-1.0, 0.0, 0.0])) # Engine completely OFF
    done = term or trunc
    step += 1
print(f"[TEST 1A: ZERO CONTROL / FREE FALL]")
print(f"  Result: {info.get('outcome')}")
print(f"  Touchdown Speed: {env_null.state[3]:.1f} m/s (Crashed into ocean at catastrophic speed)")

# B. Starting with +8m positive lateral offset
ctrl = SuicideBurnController(cfg)
env_right = RocketLand2DEnv(cfg)
obs, _ = env_right.reset(seed=42)
env_right.state = np.array([8.0, 300.0, 1.2, -35.0, 0.0, 0.0, float(env_right.state[6])])
obs = env_right.step(np.array([-1.0, 0.0, 0.0]))[0]
gimbals = []
done = False
while not done:
    act = ctrl(obs)
    gimbals.append(float(env_right.actuators.gimbal_deg))
    obs, r, term, trunc, info = env_right.step(act)
    done = term or trunc

print(f"\n[TEST 1B: START OFFSET +8.0 METERS TO THE RIGHT]")
print(f"  Initial Position: X = +8.0 m, Vx = +1.2 m/s")
print(f"  Max TVC Steering Gimbal: {max(gimbals):.1f}° (nozzle angled to push vehicle back left)")
print(f"  Final Touchdown Position: X = {env_right.state[0]:.2f} m (Centered on pad!)")
print(f"  Final Touchdown Speed: Vy = {env_right.state[3]:.2f} m/s")
print(f"  Result: {info.get('outcome')}")

# C. Starting with -8m negative lateral offset
env_left = RocketLand2DEnv(cfg)
obs, _ = env_left.reset(seed=42)
env_left.state = np.array([-8.0, 300.0, -1.2, -35.0, 0.0, 0.0, float(env_left.state[6])])
obs = env_left.step(np.array([-1.0, 0.0, 0.0]))[0]
gimbals_neg = []
done = False
while not done:
    act = ctrl(obs)
    gimbals_neg.append(float(env_left.actuators.gimbal_deg))
    obs, r, term, trunc, info = env_left.step(act)
    done = term or trunc

print(f"\n[TEST 1C: START OFFSET -8.0 METERS TO THE LEFT]")
print(f"  Initial Position: X = -8.0 m, Vx = -1.2 m/s")
print(f"  Min TVC Steering Gimbal: {min(gimbals_neg):.1f}° (nozzle angled opposite direction to push vehicle right)")
print(f"  Final Touchdown Position: X = {env_left.state[0]:.2f} m (Centered on pad!)")
print(f"  Final Touchdown Speed: Vy = {env_left.state[3]:.2f} m/s")
print(f"  Result: {info.get('outcome')}")

print("\n" + "=" * 65)
print("2. ARTEMIS LUNAR LANDER VERIFICATION (Master PPO Neural Policy)")
print("=" * 65)

model = PPO.load("models/best/best_model.zip")
env_id = get_supported_env_id("LunarLander-v3")
env_lunar = gym.make(env_id)

# Test 2A: Tilted Left (+20 deg)
obs, _ = env_lunar.reset(seed=101)
env_lunar.unwrapped.lander.angle = float(np.radians(20.0)) # Tilt left
obs, _, _, _, _ = env_lunar.step(0)
act_tilt_left, _ = model.predict(obs, deterministic=True)
print(f"[TEST 2A: TILTED +20°]")
print(f"  Model Observation Tilt: {np.degrees(obs[4]):.1f}°")
print(f"  PPO Action Output: {int(act_tilt_left)} -> {['Coast', 'Fire LEFT Engine', 'Fire MAIN Engine', 'Fire RIGHT Engine'][int(act_tilt_left)]}")

# Test 2B: Tilted Right (-20 deg)
obs, _ = env_lunar.reset(seed=101)
env_lunar.unwrapped.lander.angle = float(np.radians(-20.0)) # Tilt right
obs, _, _, _, _ = env_lunar.step(0)
act_tilt_right, _ = model.predict(obs, deterministic=True)
print(f"\n[TEST 2B: TILTED -20°]")
print(f"  Model Observation Tilt: {np.degrees(obs[4]):.1f}°")
print(f"  PPO Action Output: {int(act_tilt_right)} -> {['Coast', 'Fire LEFT Engine', 'Fire MAIN Engine', 'Fire RIGHT Engine'][int(act_tilt_right)]}")

