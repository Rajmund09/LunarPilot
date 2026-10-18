"""
LunarPilot — Evaluation harness
=================================
run_episodes: run a policy on a list of seeds, save per-episode JSONL.
summarize:    outcome rates (Wilson CI), reward stats, touchdown stats.

The harness is env-agnostic — it reads info['outcome'] set by OutcomeWrapper
(or the RocketLand equivalent).  It works for both LunarLander and RocketLand.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from lunar_pilot.outcomes import Outcome, OutcomeWrapper
from lunar_pilot.stats import bootstrap_mean_ci, wilson_ci


def run_episodes(
    policy_fn: Callable[[np.ndarray], int],
    env_factory: Callable[[], object],  # callable that returns a gym.Env
    seeds: Sequence[int],
    episodes_dir: str | Path = "results/episodes",
    run_id: str | None = None,
) -> list[dict]:
    """Run *one episode per seed* and return a list of per-episode records.

    Each record is also appended to a JSONL file under episodes_dir.

    Parameters
    ----------
    policy_fn    : obs -> action callable
    env_factory  : callable returning a fresh unwrapped gym.Env (plain, not VecEnv)
    seeds        : explicit list of integer seeds
    episodes_dir : directory for the JSONL output file
    run_id       : identifier string used in the output filename; auto-generated if None

    Returns
    -------
    list of dicts, one per episode.
    """

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    out_dir = Path(episodes_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{run_id}.jsonl"

    records: list[dict] = []

    with open(jsonl_path, "w", encoding="utf-8") as fout:
        for seed in seeds:
            env_raw = env_factory()
            env = OutcomeWrapper(env_raw)

            obs, _ = env.reset(seed=int(seed))
            ep_reward = 0.0
            ep_len = 0
            action_counts: dict[int, int] = {}
            outcome: str | None = None
            touchdown: dict | None = None
            done = False

            while not done:
                action = policy_fn(obs)
                action_counts[action] = action_counts.get(action, 0) + 1
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_len += 1
                done = terminated or truncated

                if done:
                    outcome = info.get("outcome")
                    touchdown = info.get("touchdown")

            env.close()

            if outcome is None:
                raise RuntimeError(
                    f"run_episodes: episode with seed={seed} finished without "
                    "info['outcome']. Ensure OutcomeWrapper is in the env chain."
                )

            record = {
                "seed": seed,
                "outcome": outcome,
                "reward": round(float(ep_reward), 4),
                "ep_len": ep_len,
                "action_counts": action_counts,
                "touchdown": touchdown,
            }
            records.append(record)
            fout.write(json.dumps(record) + "\n")
            fout.flush()  # never leave a partial line for a live reader

    return records


def summarize(
    records: list[dict],
    all_outcome_labels: list[str] | None = None,
) -> dict:
    """Compute outcome rates (Wilson CI), reward stats, touchdown stats.

    Parameters
    ----------
    records            : list of per-episode records from run_episodes
    all_outcome_labels : if provided, all listed outcomes appear in counts
                         even if 0 (ensures no silent omissions)

    Returns
    -------
    summary dict matching Appendix J schema.
    """
    if not records:
        raise ValueError("summarize: records must not be empty")

    n = len(records)

    # Use all Outcome enum labels if not specified
    if all_outcome_labels is None:
        all_outcome_labels = [o.value for o in Outcome]

    outcome_counts: dict[str, int] = {label: 0 for label in all_outcome_labels}
    for rec in records:
        lbl = rec["outcome"]
        if lbl not in outcome_counts:
            raise RuntimeError(
                f"summarize: unknown outcome label '{lbl}' in record. "
                "Add it to Outcome enum or all_outcome_labels."
            )
        outcome_counts[lbl] += 1

    # Verify all episodes are classified (no unclassified bucket)
    total_classified = sum(outcome_counts.values())
    if total_classified != n:
        raise RuntimeError(
            f"summarize: {n} records but only {total_classified} classified. "
            "This should never happen — check OutcomeWrapper."
        )

    outcome_rates: dict[str, dict] = {}
    for label, count in outcome_counts.items():
        lo, hi = wilson_ci(count, n)
        outcome_rates[label] = {
            "count": count,
            "rate": round(count / n, 6),
            "ci95": [round(lo, 6), round(hi, 6)],
        }

    rewards = [rec["reward"] for rec in records]
    reward_arr = np.array(rewards)
    r_lo, r_hi = bootstrap_mean_ci(rewards, seed=0)

    # Touchdown stats (only from episodes with a non-None touchdown)
    vy_vals = [rec["touchdown"]["vy"] for rec in records if rec.get("touchdown")]
    x_err_vals = [rec["touchdown"]["abs_x"] for rec in records if rec.get("touchdown")]
    touchdown_stats: dict = {}
    if vy_vals:
        touchdown_stats = {
            "vy_p50": float(np.percentile(vy_vals, 50)),
            "vy_p95": float(np.percentile(vy_vals, 95)),
            "x_err_p50": float(np.percentile(x_err_vals, 50)),
            "x_err_p95": float(np.percentile(x_err_vals, 95)),
        }

    return {
        "n_episodes": n,
        "outcome_counts": outcome_counts,
        "outcome_rates": outcome_rates,
        "reward": {
            "mean": float(np.mean(reward_arr)),
            "median": float(np.median(reward_arr)),
            "std": float(np.std(reward_arr)),
            "min": float(np.min(reward_arr)),
            "max": float(np.max(reward_arr)),
            "mean_ci95": [r_lo, r_hi],
        },
        "touchdown": touchdown_stats,
    }
