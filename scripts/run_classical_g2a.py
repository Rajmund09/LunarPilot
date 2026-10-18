"""
RocketLand-2D — Gate G2a Classical Guidance Benchmark
Evaluates SuicideBurnController on 1,000 test seeds (20000-20999):
- Gate G2a: NOMINAL scenario (target >= 90% LANDED_ON_PAD)
- Robustness evaluation: ROBUST_2X scenario
Saves:
- results/metrics/rocketland_g2a.json
- results/metrics/rocketland_classical_robust.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lunar_pilot.stats import wilson_ci
from rocketland.config import RocketConfig
from rocketland.controllers.suicide_burn import SuicideBurnController
from rocketland.env import RocketLand2DEnv


def evaluate_classical(cfg: RocketConfig, seeds: list[int], scenario_name: str = "NOMINAL") -> dict:
    env = RocketLand2DEnv(cfg)
    ctrl = SuicideBurnController(cfg)
    outcomes = {}
    records = []
    landed_count = 0
    vys = []
    vxs = []
    x_errs = []

    for s in seeds:
        obs, info = env.reset(seed=s)
        ctrl.reset()
        done = False
        step = 0
        contact_steps = 0

        while not done and step < 500:
            act = ctrl(obs)
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
        x_errs.append(abs_x)

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
        "schema_version": 1,
        "kind": f"rocketland_classical_{scenario_name.lower()}",
        "scenario": scenario_name,
        "controller": "SuicideBurnController (Closed-loop Hoverslam)",
        "n_episodes": len(seeds),
        "landed_rate": rate,
        "wilson_ci95": [float(ci[0]), float(ci[1])],
        "outcomes": outcomes,
        "touchdown_vy_mean": float(np.mean(vys)),
        "touchdown_vx_mean": float(np.mean(vxs)),
        "touchdown_x_err_p50": float(np.percentile(x_errs, 50)),
        "touchdown_x_err_p95": float(np.percentile(x_errs, 95)),
        "records": records,
    }


def main():
    test_seeds = list(range(20000, 21000))
    print("[*] Running Gate G2a Classical Guidance Benchmark on 1,000 test seeds (20000-20999)...")

    # 1. NOMINAL Scenario (G2a)
    cfg_nom = RocketConfig()
    cfg_nom.wind_base = 0.0
    cfg_nom.wind_gust_std = 0.0
    print("[*] Evaluating NOMINAL scenario...")
    res_g2a = evaluate_classical(cfg_nom, test_seeds, scenario_name="NOMINAL")
    print(f"[*] G2a Result: {res_g2a['landed_rate']*100:.2f}% (Wilson 95% CI: [{res_g2a['wilson_ci95'][0]*100:.2f}%, {res_g2a['wilson_ci95'][1]*100:.2f}%])")

    out_g2a = Path("results/metrics/rocketland_g2a.json")
    out_g2a.parent.mkdir(parents=True, exist_ok=True)
    with open(out_g2a, "w") as f:
        save_dict = {k: v for k, v in res_g2a.items() if k != "records"}
        json.dump(save_dict, f, indent=2)

    # 2. ROBUST_2X Scenario
    cfg_rob = RocketConfig()
    cfg_rob.wind_base = 8.0
    cfg_rob.wind_gust_std = 1.5
    print("[*] Evaluating ROBUST_2X scenario...")
    res_rob = evaluate_classical(cfg_rob, test_seeds, scenario_name="ROBUST_2X")
    print(f"[*] Classical Robust Result: {res_rob['landed_rate']*100:.2f}% (Wilson 95% CI: [{res_rob['wilson_ci95'][0]*100:.2f}%, {res_rob['wilson_ci95'][1]*100:.2f}%])")

    out_rob = Path("results/metrics/rocketland_classical_robust.json")
    with open(out_rob, "w") as f:
        save_dict = {k: v for k, v in res_rob.items() if k != "records"}
        json.dump(save_dict, f, indent=2)

    print("[*] Classical gate evaluation complete.")


if __name__ == "__main__":
    main()
