# STATUS.md — LunarPilot → RocketLand Execution Status
**Format:** per AGENT_PLAN Appendix K. Updated after every commit.  
**Last updated:** 2026-10-17

| Task | Status | Evidence (paths) | Gate result | Notes |
|------|--------|------------------|-------------|-------|
| **P0.0** Audit | **DONE** | `docs/AUDIT.md` | — | 21 issues catalogued; all false claims listed with evidence |
| **P0.1** Packaging & hygiene | **DONE** | `pyproject.toml`, `requirements.lock`, `verify_environment.py` | — | Lockfile pinned, ruff/black configs, `gym.error.Error` handling, headless rgb_array render verified |
| **P0.2** Outcome classification | **DONE** | `src/lunar_pilot/outcomes.py`, `tests/unit/test_truth_layer.py` | — | `Outcome` enum, `classify_lunarlander_episode` with 5 non-reward event branches, `OutcomeWrapper` |
| **P0.3** Statistics & evaluation harness | **DONE** | `src/lunar_pilot/stats.py`, `src/lunar_pilot/evaluation.py`, `src/lunar_pilot/policies.py`, `src/lunar_pilot/provenance.py` | — | Wilson CI, bootstrap mean CI, `run_episodes` JSONL, env-agnostic `summarize`, reproducible policies |
| **P0.4** Baselines on test seeds | **DONE** | `results/metrics/lander_baselines.json`, `scripts/run_baselines.py` | — | Evaluated Null, Random, and Heuristic policies on 1000 test seeds (20000-20999) |
| **P0.5** Train PPO properly | **DONE** | `scripts/train_lander.py`, `models/best/best_model.zip`, `models/registry.json`, `results/training/` | — | 5 training seeds, validation selection on seeds 10000-10049, telemetry callback, snapshots saved |
| **P0.6** Generated documentation | **DONE** | `scripts/make_report.py`, `README.md` | — | Deterministic regeneration of README results block between markers |
| **P0.7** Fix false claims | **DONE** | `README.md`, `IMPLEMENTATION_PLAN.md`, `docs/AUDIT.md` | — | Fixed gravity, bonus condition, continuous env id, clone url, and removed all unverified metrics |
| **G0** Gate | **DONE** | `results/metrics/lander_ppo_test.json` | **PASS** | **95.50%** `LANDED_ON_PAD` on 1,000 test seeds (seeds 20000–20999), Wilson 95% CI **[94.03%, 96.62%]** (LB ≥ 90%) |
| **P1.1** Perturbation toolkit | TODO | — | — | Wind/turbulence, obs noise, action latency, engine-power scaling |
| **P1.2** Domain-randomized training | TODO | — | — | Train with perturbations |
| **P1.3** Stress-grid evaluation | TODO | — | — | Grid eval; heatmaps to `results/figures/` |
| **G1** Gate | TODO | — | PENDING | ≥85% LANDED_ON_PAD on HELD_OUT_2X |
| **P2.1** Dynamics core | TODO | — | — | `dynamics.py`, `atmosphere.py`, `config.py`; pure functions; RK4 |
| **P2.2** Contact model | TODO | — | — | `contact.py`; foot-spring; Coulomb friction; substeps |
| **P2.3** Actuators | TODO | — | — | `actuators.py`; throttle lag; gimbal; RCS |
| **P2.4** Gymnasium env | TODO | — | — | `RocketLand2D-v0`; check_env; rgb_array |
| **P2.5** Start distribution & feasibility | TODO | — | — | Feasible fraction published |
| **P2.6** Classical controllers | TODO | — | — | null; suicide_burn |
| **G2a** Gate | TODO | — | PENDING | suicide_burn ≥90% LANDED_ON_PAD on NOMINAL |
| **P2.7** RL training (rocket) | TODO | — | — | PPO+SAC; VecNormalize; curriculum; 5 seeds |
| **P2.8** Disturbances & realism | TODO | — | — | Wind OU; sensors; estimator; param randomization; moving pad |
| **P2.9** MPC / convex guidance | SKIPPED | — | — | Time-boxed; logged here not silently dropped |
| **P2.10** Comparison report | TODO | — | — | All controllers on identical seeds; failure atlas |
| **G2b** Gate | TODO | — | PENDING | Best RL ≥95% LANDED_ON_PAD on NOMINAL |
| **G2c** Gate | TODO | — | PENDING | Best RL ≥85% on ROBUST_2X |
| **P3.1** Backend (FastAPI) | TODO | — | — | All endpoints per Appendix H; Pydantic; cancellation |
| **P3.2** Frontend (React+Vite) | TODO | — | — | Canvas; telemetry plots; all controls per Appendix I |
| **P3.3** Honesty checks | TODO | — | — | No fixture data in prod bundle; Compare table = backend |
| **P3.4** Training dashboard | TODO | — | — | Read-only; live metrics.jsonl stream; checkpoint replay |
| **G3** Gate | TODO | — | PENDING | Every Appendix I control has passing Playwright test |
| **P4** Release | TODO | — | — | CI; Docker; README rewrite; MODEL_CARD.md; demo GIF; v1.0.0 tag |
| **G4** Final acceptance | TODO | — | PENDING | All items in Section 9 checklist |
