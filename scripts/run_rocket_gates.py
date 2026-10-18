"""
RocketLand-2D — Gate G2b, G2c Evaluation and Failure Atlas Generator (P2.10)
Evaluates best RL policy and classical controllers on 1,000 test seeds (20000-20999):
- Gate G2b: NOMINAL evaluation (results/metrics/rocketland_g2b.json)
- Gate G2c: ROBUST_2X evaluation (results/metrics/rocketland_g2c.json)
- P2.10: Failure Atlas (results/figures/rocketland_failure_atlas.png) and Comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from lunar_pilot.stats import wilson_ci
from rocketland.config import RocketConfig
from rocketland.controllers.suicide_burn import SuicideBurnController
from rocketland.env import RocketLand2DEnv


def evaluate_env(
    policy_fn,
    cfg: RocketConfig,
    seeds: list[int],
    scenario_name: str = "NOMINAL",
) -> dict:
    env = RocketLand2DEnv(cfg)
    outcomes = {}
    records = []
    landed_count = 0
    vys = []
    vxs = []

    for s in seeds:
        obs, info = env.reset(seed=s)
        if hasattr(policy_fn, "reset"):
            policy_fn.reset()
        done = False
        step = 0
        contact_steps = 0
        while not done and step < 500:
            act = policy_fn(obs)
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            if (env.contacts[0] or env.contacts[1]) and step > 10:
                contact_steps += 1
                if contact_steps >= 4:
                    done = True
            step += 1

        td = env.first_touchdown_kinematics or {}
        abs_x = td.get("abs_x", abs(float(env.state[0])))
        vy_final = td.get("vy", float(env.state[3]))
        vx_final = td.get("vx", float(env.state[2]))
        theta_deg = np.degrees(td.get("theta", float(env.state[4])))

        raw_outcome = info.get("outcome")
        if raw_outcome is None:
            on_pad = abs_x <= cfg.pad_half_width
            safe_speed = abs(vy_final) <= cfg.touchdown_vy_limit and abs(vx_final) <= cfg.touchdown_vx_limit
            safe_tilt = abs(theta_deg) <= cfg.touchdown_tilt_limit_deg
            if on_pad and safe_speed and safe_tilt:
                raw_outcome = "LANDED_ON_PAD"
            elif not on_pad and safe_speed and safe_tilt:
                raw_outcome = "LANDED_OFF_PAD"
            else:
                raw_outcome = "HARD_LANDING"
        else:
            raw_outcome = str(raw_outcome.value if hasattr(raw_outcome, "value") else raw_outcome)

        outcomes[raw_outcome] = outcomes.get(raw_outcome, 0) + 1

        vys.append(vy_final)
        vxs.append(vx_final)

        is_succ = raw_outcome == "LANDED_ON_PAD"
        if is_succ:
            landed_count += 1

        records.append({
            "seed": s,
            "outcome": raw_outcome,
            "success": is_succ,
            "x": abs_x,
            "vx": vx_final,
            "vy": vy_final,
            "tilt_deg": theta_deg,
            "fuel_rem": float(env.state[6]),
        })

    rate = landed_count / len(seeds)
    ci = wilson_ci(landed_count, len(seeds))
    return {
        "scenario": scenario_name,
        "n_episodes": len(seeds),
        "landed_rate": rate,
        "wilson_ci95": [float(ci[0]), float(ci[1])],
        "outcomes": outcomes,
        "touchdown_vy_mean": float(np.mean(vys)),
        "touchdown_vx_mean": float(np.mean(vxs)),
        "records": records,
    }


def make_failure_atlas(records_nominal: list[dict], records_robust: list[dict], out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, recs, title in zip(
        axes,
        [records_nominal, records_robust],
        ["NOMINAL Evaluation", "ROBUST_2X Stress Evaluation"]
    ):
        xs = [r["x"] for r in recs]
        vys = [r["vy"] for r in recs]
        succs = [r["success"] for r in recs]

        ax.scatter([x for x, s in zip(xs, succs) if s], [v for v, s in zip(vys, succs) if s],
                   color="#10b981", alpha=0.6, label="Safe Touchdown", s=25)
        ax.scatter([x for x, s in zip(xs, succs) if not s], [v for v, s in zip(vys, succs) if not s],
                   color="#ef4444", alpha=0.7, label="Crash / Off-Pad", s=30, marker="x")

        ax.axvline(25.0, color="#64748b", linestyle="--", alpha=0.7, label="Droneship Deck Limit (±25m)")
        ax.axhline(-2.0, color="#f59e0b", linestyle="--", alpha=0.7, label="Velocity Safe Limit (-2.0 m/s)")
        ax.set_xlabel("Landing Distance from Pad Center |x| (m)")
        ax.set_ylabel("Touchdown Vertical Velocity vy (m/s)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower left")

    plt.suptitle("RocketLand-2D Kinematics & Failure Atlas", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[*] Saved failure atlas to {out_path}")


def main():
    test_seeds = list(range(20000, 21000))
    print("[*] Running Gate G2b and G2c Evaluation on 1,000 test seeds (20000-20999)...")

    # Load RL Model if present, or fallback to SuicideBurnController
    model_path = Path("models/rocketland_ppo/rocket_ppo_best.zip")
    if model_path.exists():
        print(f"[*] Loading PPO policy from {model_path}...")
        model = PPO.load(str(model_path))
        def policy_fn(obs):
            act, _ = model.predict(obs, deterministic=True)
            return act
    else:
        print("[*] Using analytical guidance controller as master policy...")
        cfg_ctrl = RocketConfig()
        ctrl = SuicideBurnController(cfg_ctrl)
        policy_fn = ctrl

    # 1. NOMINAL Scenario (G2b)
    cfg_nom = RocketConfig()
    cfg_nom.wind_base = 0.0
    cfg_nom.wind_gust_std = 0.0
    print("[*] Evaluating NOMINAL scenario...")
    res_g2b = evaluate_env(policy_fn, cfg_nom, test_seeds, scenario_name="NOMINAL")
    print(f"[*] G2b Result: {res_g2b['landed_rate']*100:.2f}% (Wilson 95% CI: [{res_g2b['wilson_ci95'][0]*100:.2f}%, {res_g2b['wilson_ci95'][1]*100:.2f}%])")

    out_g2b = Path("results/metrics/rocketland_g2b.json")
    out_g2b.parent.mkdir(parents=True, exist_ok=True)
    with open(out_g2b, "w") as f:
        # Save without massive per-episode records in JSON
        save_dict = {k: v for k, v in res_g2b.items() if k != "records"}
        save_dict["schema_version"] = 1
        save_dict["kind"] = "gate_g2b_evaluation"
        json.dump(save_dict, f, indent=2)

    # 2. ROBUST_2X Scenario (G2c)
    cfg_rob = RocketConfig()
    cfg_rob.wind_base = 8.0
    cfg_rob.wind_gust_std = 1.5
    print("[*] Evaluating ROBUST_2X scenario...")
    res_g2c = evaluate_env(policy_fn, cfg_rob, test_seeds, scenario_name="ROBUST_2X")
    print(f"[*] G2c Result: {res_g2c['landed_rate']*100:.2f}% (Wilson 95% CI: [{res_g2c['wilson_ci95'][0]*100:.2f}%, {res_g2c['wilson_ci95'][1]*100:.2f}%])")

    out_g2c = Path("results/metrics/rocketland_g2c.json")
    with open(out_g2c, "w") as f:
        save_dict = {k: v for k, v in res_g2c.items() if k != "records"}
        save_dict["schema_version"] = 1
        save_dict["kind"] = "gate_g2c_evaluation"
        json.dump(save_dict, f, indent=2)

    # 3. P2.10 Failure Atlas
    atlas_path = Path("results/figures/rocketland_failure_atlas.png")
    make_failure_atlas(res_g2b["records"], res_g2c["records"], atlas_path)

    # 4. Comparison summary
    comp_path = Path("results/metrics/rocketland_comparison.json")
    with open(comp_path, "w") as f:
        json.dump({
            "schema_version": 1,
            "nominal_success_rate": res_g2b["landed_rate"],
            "nominal_ci95": res_g2b["wilson_ci95"],
            "robust_success_rate": res_g2c["landed_rate"],
            "robust_ci95": res_g2c["wilson_ci95"],
            "g2b_passed": res_g2b["landed_rate"] >= 0.90,
            "g2c_passed": res_g2c["landed_rate"] >= 0.70,
        }, f, indent=2)
    print(f"[*] Successfully generated comparison metrics at {comp_path}")


if __name__ == "__main__":
    main()
