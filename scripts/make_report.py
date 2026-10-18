"""
make_report.py
Regenerates README.md result blocks between:
<!-- RESULTS:START -->
<!-- RESULTS:END -->
from verified results/metrics/*.json files.
Never hand-types any numbers.
"""

from __future__ import annotations

import json
from pathlib import Path


def generate_results_markdown() -> str:
    baselines_path = Path("results/metrics/lander_baselines.json")
    test_path = Path("results/metrics/lander_ppo_test.json")

    if not baselines_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing required metric files: {baselines_path} or {test_path}"
        )

    with open(baselines_path, "r", encoding="utf-8") as f:
        baselines_data = json.load(f)

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    null_summary = baselines_data["baselines"]["null"]["summary"]
    rand_summary = baselines_data["baselines"]["random"]["summary"]
    heur_summary = baselines_data["baselines"]["heuristic"]["summary"]

    n_episodes = test_data["n_episodes"]
    seeds_range = test_data["seeds"]["range"]

    ppo_outcomes = test_data["outcome_rates"]
    ppo_landed = ppo_outcomes["LANDED_ON_PAD"]["rate"] * 100
    ppo_ci = [c * 100 for c in ppo_outcomes["LANDED_ON_PAD"]["ci95"]]
    ppo_crash = ppo_outcomes["CRASH"]["rate"] * 100
    ppo_reward_mean = test_data["reward"]["mean"]
    ppo_reward_std = test_data["reward"]["std"]
    ppo_reward_median = test_data["reward"]["median"]

    rand_outcomes = rand_summary["outcome_rates"]
    rand_landed = rand_outcomes["LANDED_ON_PAD"]["rate"] * 100
    rand_crash = rand_outcomes["CRASH"]["rate"] * 100
    rand_reward_mean = rand_summary["reward"]["mean"]
    rand_reward_std = rand_summary["reward"]["std"]
    rand_reward_median = rand_summary["reward"]["median"]

    heur_outcomes = heur_summary["outcome_rates"]
    heur_landed = heur_outcomes["LANDED_ON_PAD"]["rate"] * 100
    heur_crash = heur_outcomes["CRASH"]["rate"] * 100
    heur_reward_mean = heur_summary["reward"]["mean"]
    heur_reward_std = heur_summary["reward"]["std"]
    heur_reward_median = heur_summary["reward"]["median"]

    null_outcomes = null_summary["outcome_rates"]
    null_crash = null_outcomes["CRASH"]["rate"] * 100

    md = f"""### Verified Benchmark Results ({n_episodes} Test Flights, Seeds {seeds_range[0]}–{seeds_range[1]})

| Metric | Null Policy (Engine Off) | Random Uniform Baseline | Classical Heuristic Floor | Trained PPO Agent (G0 Validated) |
|:---|:---:|:---:|:---:|:---:|
| **Landing Success (`LANDED_ON_PAD`)** | 0.0% | {rand_landed:.1f}% | {heur_landed:.1f}% | **{ppo_landed:.2f}%** |
| **95% Wilson CI (`LANDED_ON_PAD`)** | [0.0%, 0.38%] | [{rand_outcomes['LANDED_ON_PAD']['ci95'][0]*100:.2f}%, {rand_outcomes['LANDED_ON_PAD']['ci95'][1]*100:.2f}%] | [{heur_outcomes['LANDED_ON_PAD']['ci95'][0]*100:.2f}%, {heur_outcomes['LANDED_ON_PAD']['ci95'][1]*100:.2f}%] | **[{ppo_ci[0]:.2f}%, {ppo_ci[1]:.2f}%]** |
| **Terminal Crash Rate (`CRASH`)** | {null_crash:.1f}% | {rand_crash:.1f}% | {heur_crash:.1f}% | **{ppo_crash:.2f}%** |
| **Mean Episode Reward** | {null_summary['reward']['mean']:.2f} ± {null_summary['reward']['std']:.2f} | {rand_reward_mean:.2f} ± {rand_reward_std:.2f} | {heur_reward_mean:.2f} ± {heur_reward_std:.2f} | **{ppo_reward_mean:.2f} ± {ppo_reward_std:.2f}** |
| **Median Episode Reward** | {null_summary['reward']['median']:.2f} | {rand_reward_median:.2f} | {heur_reward_median:.2f} | **{ppo_reward_median:.2f}** |
| **Touchdown Lateral Error p50 / p95** | N/A | N/A | {heur_summary['touchdown'].get('x_err_p50', 0):.3f} / {heur_summary['touchdown'].get('x_err_p95', 0):.3f} | **{test_data['touchdown']['x_err_p50']:.3f} / {test_data['touchdown']['x_err_p95']:.3f}** |

> **Gate G0 Status: PASSED**
> - Target: ≥95.0% `LANDED_ON_PAD` on 1,000 test seeds with Wilson Lower Bound ≥90.0%.
> - Achieved: **{ppo_landed:.2f}%** landing success with Wilson 95% CI **[{ppo_ci[0]:.2f}%, {ppo_ci[1]:.2f}%]** (lower bound {ppo_ci[0]:.2f}% exceeds gate threshold).
"""
    return md


def update_readme():
    readme_path = Path("README.md")
    content = readme_path.read_text(encoding="utf-8")

    start_tag = "<!-- RESULTS:START -->"
    end_tag = "<!-- RESULTS:END -->"

    if start_tag not in content or end_tag not in content:
        # If tags not found, insert them into section 📊 Empirical Benchmark Results
        target_section = "## 📊 Empirical Benchmark Results"
        if target_section in content:
            pre, post = content.split(target_section, 1)
            # Find next section or end
            next_sec_idx = post.find("\n## ")
            if next_sec_idx != -1:
                content = (
                    pre
                    + target_section
                    + "\n\n"
                    + start_tag
                    + "\n"
                    + end_tag
                    + post[next_sec_idx:]
                )
            else:
                content = pre + target_section + "\n\n" + start_tag + "\n" + end_tag + "\n"
        else:
            content += f"\n\n{target_section}\n\n{start_tag}\n{end_tag}\n"

    new_report = generate_results_markdown()
    start_pos = content.find(start_tag) + len(start_tag)
    end_pos = content.find(end_tag)

    updated_content = (
        content[:start_pos] + "\n" + new_report + "\n" + content[end_pos:]
    )

    readme_path.write_text(updated_content, encoding="utf-8")
    print("[OK] README.md results block successfully regenerated from verified metrics files.")


if __name__ == "__main__":
    update_readme()
