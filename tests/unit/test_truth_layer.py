"""
Tests for outcomes.py, stats.py, provenance.py, policies.py, evaluation.py
Covering test matrix requirements from AGENT_PLAN Appendix G.
"""

import gymnasium as gym
import numpy as np
import pytest

from lunar_pilot.evaluation import run_episodes, summarize
from lunar_pilot.outcomes import Outcome, _pad_half_width_in_obs, classify_lunarlander_episode
from lunar_pilot.policies import GymHeuristicPolicy, NullPolicy, RandomPolicy
from lunar_pilot.provenance import make_provenance
from lunar_pilot.stats import bootstrap_mean_ci, wilson_ci, wilson_lower


class DummyBody:
    def __init__(self, awake=True):
        self.awake = awake


class DummyEnv:
    def __init__(self, game_over=False, awake=True, helipad_x1=8.0, helipad_x2=12.0):
        self.game_over = game_over
        self.lander = DummyBody(awake=awake)
        self.helipad_x1 = helipad_x1
        self.helipad_x2 = helipad_x2
        self.unwrapped = self


def test_wilson_ci_math_edge_cases():
    # n = 0 raises ValueError
    with pytest.raises(ValueError, match="n must be > 0"):
        wilson_ci(0, 0)

    # k < 0 or k > n raises
    with pytest.raises(ValueError):
        wilson_ci(-1, 10)
    with pytest.raises(ValueError):
        wilson_ci(11, 10)

    # k = 0
    lo, hi = wilson_ci(0, 100)
    assert lo == 0.0
    assert 0.0 < hi < 0.1

    # k = n
    lo, hi = wilson_ci(100, 100)
    assert hi == 1.0
    assert 0.9 < lo < 1.0

    # n = 1
    lo, hi = wilson_ci(1, 1)
    assert 0.0 <= lo <= 1.0
    assert hi == 1.0

    assert wilson_lower(10, 100) == wilson_ci(10, 100)[0]


def test_bootstrap_mean_ci():
    with pytest.raises(ValueError):
        bootstrap_mean_ci([])

    data = [10.0, 10.0, 10.0]
    lo, hi = bootstrap_mean_ci(data)
    assert lo == 10.0
    assert hi == 10.0


def test_outcome_classifier_branches():
    # Branch 1: Truncation
    env = DummyEnv(game_over=False, awake=True)
    obs = np.zeros(8, dtype=np.float32)
    outcome, td = classify_lunarlander_episode(env, terminated=False, truncated=True, last_obs=obs)
    assert outcome == Outcome.TIMEOUT

    # Branch 2: Crash (game_over)
    env = DummyEnv(game_over=True, awake=True)
    outcome, td = classify_lunarlander_episode(env, terminated=True, truncated=False, last_obs=obs)
    assert outcome == Outcome.CRASH

    # Branch 3: Out of bounds (|x| >= 1.0)
    env = DummyEnv(game_over=False, awake=True)
    obs_oob = np.zeros(8, dtype=np.float32)
    obs_oob[0] = 1.05
    outcome, td = classify_lunarlander_episode(env, terminated=True, truncated=False, last_obs=obs_oob)
    assert outcome == Outcome.OUT_OF_BOUNDS

    # Branch 4: Landed on pad (both legs, inside pad half width)
    env = DummyEnv(game_over=False, awake=False)
    obs_landed = np.zeros(8, dtype=np.float32)
    obs_landed[0] = 0.05
    obs_landed[6] = 1.0
    obs_landed[7] = 1.0
    outcome, td = classify_lunarlander_episode(env, terminated=True, truncated=False, last_obs=obs_landed)
    assert outcome == Outcome.LANDED_ON_PAD

    # Landed off pad (outside pad or one leg)
    obs_off = np.zeros(8, dtype=np.float32)
    obs_off[0] = 0.35  # outside 0.2 pad half-width
    obs_off[6] = 1.0
    obs_off[7] = 1.0
    outcome, td = classify_lunarlander_episode(env, terminated=True, truncated=False, last_obs=obs_off)
    assert outcome == Outcome.LANDED_OFF_PAD

    # Branch 5: Unhandled state raises loudly
    env_unhandled = DummyEnv(game_over=False, awake=True)
    with pytest.raises(RuntimeError, match="no branch matched"):
        classify_lunarlander_episode(env_unhandled, terminated=False, truncated=False, last_obs=obs)


def test_pad_half_width_calculation():
    env = DummyEnv(helipad_x1=8.0, helipad_x2=12.0)
    hw = _pad_half_width_in_obs(env)
    assert abs(hw - 0.2) < 1e-5


def test_gymnasium_lunarlander_attributes_exist():
    real_env = gym.make("LunarLander-v3")
    uw = real_env.unwrapped
    real_env.reset(seed=42)
    assert hasattr(uw, "game_over"), "Required attribute 'game_over' missing from LunarLander"
    assert hasattr(uw, "lander"), "Required attribute 'lander' missing from LunarLander"
    assert hasattr(uw.lander, "awake"), "Required attribute 'lander.awake' missing from LunarLander"
    assert hasattr(uw, "helipad_x1"), "Required attribute 'helipad_x1' missing from LunarLander"
    assert hasattr(uw, "helipad_x2"), "Required attribute 'helipad_x2' missing from LunarLander"
    assert hasattr(uw, "legs"), "Required attribute 'legs' missing from LunarLander"
    real_env.close()


def test_policies():
    rand_pol = RandomPolicy(n_actions=4, seed=42)
    obs = np.zeros(8, dtype=np.float32)
    a1 = rand_pol(obs)
    assert 0 <= a1 <= 3

    null_pol = NullPolicy()
    assert null_pol(obs) == 0

    real_env = gym.make("LunarLander-v3")
    heur_pol = GymHeuristicPolicy(real_env)
    obs, _ = real_env.reset(seed=42)
    a_heur = heur_pol(obs)
    assert a_heur in [0, 1, 2, 3]
    real_env.close()


def test_provenance():
    prov = make_provenance(config={"lr": 0.001}, seeds=[1, 2, 3])
    assert "git_commit" in prov
    assert "packages" in prov
    assert "gymnasium" in prov["packages"]
    assert "timestamp_utc" in prov


def test_evaluation_harness_roundtrip(tmp_path):
    def env_factory():
        return gym.make("LunarLander-v3")

    null_pol = NullPolicy()
    # 5 test episodes with null policy
    records = run_episodes(
        null_pol,
        env_factory,
        seeds=[20000, 20001, 20002],
        episodes_dir=tmp_path,
        run_id="test_run",
    )
    assert len(records) == 3
    for r in records:
        assert "outcome" in r
        assert "touchdown" in r
        assert r["outcome"] in [Outcome.CRASH.value, Outcome.OUT_OF_BOUNDS.value]

    summary = summarize(records)
    assert summary["n_episodes"] == 3
    assert sum(summary["outcome_counts"].values()) == 3
    assert "reward" in summary
    assert "touchdown" in summary
