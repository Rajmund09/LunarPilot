<USER_REQUEST>
# AGENT_PLAN.md — LunarPilot → RocketLand: Execution Plan for the IDE Agent

**Audience:** the coding agent working inside this repository.
**Goal:** turn this repo from a LunarLander tutorial project with unverified claims into a *real, tested, honest* autonomous rocket-landing simulation, training pipeline, evaluation suite, and working web UI.
**Read this entire file before touching code. Then execute tasks in order.**

---

## 0. Kickoff protocol (do this first)

1. Read this file fully.
2. Read the whole repo, including `src/lunar_pilot/*`, `configs/`, `tests/`, `models/`, `results/`, and any UI folder.
3. Create `docs/AUDIT.md` (task P0.0). Do **not** change any code before the audit exists.
4. Create `STATUS.md` (format in Appendix K) and `docs/DECISIONS.md`.
5. Work task by task: **tests first or alongside, never after**.

---

## 1. Non-negotiable rules

1. **No fake numbers.** Every metric in README, docs, UI, or reports must come from a results file produced by code you ran. Never type a metric by hand. Never "round up" to meet a gate.
2. **Gates are fixed.** Never lower a threshold or redefine success to pass a gate. If a gate fails: diagnose, make up to 3 documented fix attempts (each logged in `STATUS.md` with evidence), then mark the task `BLOCKED` with the evidence and continue with tasks that do not depend on it. Report honestly.
3. **No untested claims about libraries.** Gymnasium/SB3 attribute names below (for example `game_over`, `lander.awake`, `helipad_x1`) are *expected*, not guaranteed. Verify against the installed version, and add a test that fails loudly if one is missing.
4. **Determinism.** All randomness flows from explicit seeds (`np.random.default_rng(seed)`). Never use `action_space.sample()` in evaluation code.
5. **Seed discipline.** Three disjoint seed ranges, defined once in `configs/seeds.json`:
   - train: free choice
   - validation: `10000–10199` (checkpoint selection, tuning)
   - test: `20000–20999` (final reporting only; **never** tune on these)
6. **No silent failure.** No `except Exception: pass` in library code. No "unclassified" outcome bucket. Unknown states raise.
7. **No stubs in merged code.** No `TODO`, `pass`-bodies, or placeholder returns. If something is deferred, it goes in `STATUS.md` as `TODO`, not in code.
8. **Traceability.** Every results JSON includes provenance (Appendix J). Every task ends with a commit `Pxx.y: <summary>` and a `STATUS.md` update linking evidence files.
9. **Decision protocol.** If something is ambiguous, pick the simplest option consistent with this document, log it in `docs/DECISIONS.md` (decision, alternatives, reason), and continue. Do not stall.
10. **Honest fidelity claim.** The rocket sim is "Fidelity Level 2": a believable 2D control-research simulator. Never describe it as a real Falcon 9 simulation. State this in `docs/MODEL_CARD.md` and in the UI footer.
11. **Single dependency source.** `pyproject.toml` is the truth; a lockfile (`uv.lock` or `pip-compile` output) pins versions; `requirements.txt` is generated from it, not hand-edited.

**Definition of Done (every task):** code merged, tests pass (`pytest -q`), lint passes, `STATUS.md` updated with evidence paths, commit made.

---

## 2. Target repository layout

```
LunarPilot/
├── AGENT_PLAN.md
├── STATUS.md
├── configs/
│   ├── default.json              # existing LunarLander PPO config (fixed)
│   ├── seeds.json                # train/val/test seed ranges
│   └── rocketland/
│       ├── vehicle.json          # vehicle + engine parameters
│       ├── scenarios.json        # NOMINAL, ROBUST_1X, ROBUST_2X, curriculum stages
│       └── ppo.json / sac.json
├── docs/
│   ├── AUDIT.md  DECISIONS.md  PARAMETERS.md  MODEL_CARD.md
├── models/
│   └── registry.json             # id, path, sha256, env, trained-on, eval-file
├── results/
│   ├── metrics/  figures/  episodes/   # JSON, PNG, JSONL
├── scripts/
│   ├── make_report.py            # regenerates README result blocks
│   ├── run_eval.py  train_lander.py  train_rocket.py  make_demo.py
├── src/
│   ├── lunar_pilot/              # existing package (kept, fixed)
│   │   ├── outcomes.py  stats.py  provenance.py  evaluation.py  policies.py
│   └── rocketland/
│       ├── config.py  dynamics.py  contact.py  actuators.py  atmosphere.py
│       ├── wind.py  sensors.py  estimator.py  env.py  rendering.py
│       └── controllers/  (null.py, suicide_burn.py, mpc.py [stretch])
│   └── server/                   # FastAPI backend
├── web/                          # React + Vite + TypeScript frontend
├── tests/
│   ├── unit/  integration/  e2e/
└── .github/workflows/ci.yml
```

Keep the existing root shims (`train.py`, `evaluate.py`, `play.py`, `plot_results.py`), but make them thin and working.

---

## 3. Phases and gates at a glance

| Phase | Purpose | Gate |
|---|---|---|
| **P0** Truth layer | Correct metrics, honest baselines, real PPO on LunarLander | **G0:** ≥95% `LANDED_ON_PAD` on 1000 test seeds, Wilson lower bound ≥90% |
| **P1** Robustness protocol | Perturbation harness, domain randomization | **G1:** ≥85% success at 2× training disturbances |
| **P2** RocketLand-2D | Real physics env, controllers, RL | **G2a/b/c** (see P2) |
| **P3** Product layer | API + frontend where every button works | **G3:** every UI control has a passing automated test |
| **P4** Release | CI, docs, model card, demo | **G4:** final acceptance checklist (Section 9) |

---

## PHASE 0 — Truth layer

### P0.0 Audit
Produce `docs/AUDIT.md` answering, with evidence (command outputs, file/line references):
- Does `src/lunar_pilot/train.py` actually train PPO with the config in `configs/default.json`? Which hyperparameters are used?
- Does a trained model exist? Where is it, and what are its SHA-256, training steps, and seed?
- How does `evaluate.py` classify crash and success? (Known suspect pattern: `total_reward <= -100` for crash and `total_reward >= 100` for success. These use cumulative reward, not the terminal event.)
- Does the TensorBoard callback compute success/crash with the same flawed logic?
- Do tests exist and pass? What do they actually assert?
- Are README/PLAN numbers reproducible from files in `results/`? Which are hand-typed?
- List every false or unverifiable claim. Known suspects to verify and fix:
  - README/PLAN gravity "-9.81 lunar" (env default is -10; the Moon is ~1.62)
  - "+100 bonus between the flags" (env awards +100 on coming to rest anywhere)
  - success defined with "terminal velocity constraints" and a "zero crash flag" that the code does not implement
  - "locked dependencies" while `requirements.txt` uses `>=`
  - `LunarLanderContinuous-v3` (in Gymnasium 1.x it is `LunarLander-v3` with `continuous=True`; verify)
  - placeholder clone URL in README
  - `verify_environment.py` fallback catches `NamespaceNotFound` (wrong exception for a missing version; catch `gym.error.Error`)
  - baseline uses unseeded `action_space.sample()`; baseline seeds (1000+) differ from eval seeds (5000)
  - 81% success + 0% crash leaves 19% unclassified
- **Whether the claimed PPO results reproduce.** Re-run the *existing* evaluation on the existing model and record the output before changing anything.

### P0.1 Packaging and hygiene
- Single dependency source (`pyproject.toml`), lockfile, generated `requirements.txt`. Revisit the `numpy<2` pin: remove it if the installed Gymnasium/SB3/Box2D stack works with NumPy 2, otherwise document why.
- Clean-venv install test: `pip install -e .[dev]` then `pytest -q` must pass.
- Fix README placeholders. Fix `verify_environment.py` (generic `gym.error.Error`, add a headless `rgb_array` render check).
- Add `ruff` (or flake8) and `black` configs; CI will enforce them.

### P0.2 Outcome classification (`src/lunar_pilot/outcomes.py`) — see Appendix A
- Implement the shared `Outcome` enum and `classify_lunarlander_episode`.
- Implement `OutcomeWrapper(gym.Wrapper)` that, on the **final step** of an episode, writes `info["outcome"]` (string value) and `info["touchdown"]` (dict of final kinematics).
- Tests per Appendix G.

### P0.3 Statistics and evaluation harness
- `stats.py`: Wilson interval (Appendix B), bootstrap CI for means.
- `evaluation.py`: `run_episodes(policy_fn, env_factory, seeds)` → list of per-episode records saved as JSONL under `results/episodes/`; `summarize(records)` → outcome rates with Wilson CIs, reward mean/median/std, touchdown statistics, action usage.
- `policies.py`: callables `obs -> action`: `RandomPolicy(seed)` (own RNG), `NullPolicy`, `GymHeuristicPolicy` (Gymnasium's built-in `heuristic`; verify it exists), `SB3Policy(path, deterministic=True)`.
- The same harness must later serve RocketLand (it reads `info["outcome"]`, so it is env-agnostic).
- `provenance.py`: git commit (and dirty flag), package versions, config hash, seed-list hash, model SHA-256, timestamp, Python version.

### P0.4 Baselines on identical test seeds
Run random, null, and heuristic on the 1000 test seeds. Save to `results/metrics/lander_baselines.json`. The heuristic is your classical floor. Report it.

### P0.5 Train PPO properly
- Start from the RL Zoo PPO LunarLander config (approximately: 16 envs, `n_steps=1024`, `batch_size=64`, `n_epochs=4`, `gamma=0.999`, `gae_lambda=0.98`, `ent_coef=0.01`, 1M timesteps). Verify exact values against the Zoo if you can; otherwise log your values as assumptions.
- Replace the eval/success callback so success/crash metrics come from `info["outcome"]`.
- Train **5 seeds**. Select each seed's checkpoint on validation seeds only. Report all 5 on test seeds (mean ± spread). Register the best-by-validation model in `models/registry.json`.
- Limit hyperparameter search to a small, logged budget; never tune on test seeds.

### P0.6 Generated documentation
- `scripts/make_report.py` rewrites only the blocks between `<!-- RESULTS:START -->` and `<!-- RESULTS:END -->` in `README.md` from `results/metrics/*.json`.
- Remove every hardcoded metric from README and IMPLEMENTATION_PLAN. CI re-runs the script and fails on diff.

### P0.7 Fix false claims
Correct every item listed in the audit. Leave nothing unexplained.

### Gate G0
`results/metrics/lander_ppo_test.json` shows ≥95% `LANDED_ON_PAD` on the 1000 test seeds with Wilson lower bound ≥90%. If PPO cannot reach it after the allowed attempts, mark `BLOCKED`, publish the true numbers, and continue (P2 does not depend on G0).

---

## PHASE 1 — Robustness protocol (on LunarLander)

### P1.1 Perturbation toolkit
- Use the env's wind/turbulence options if present (verify names in the installed version) and add a wrapper for: random initial velocity kick, observation noise (Gaussian, per-channel scale), action latency (1–2 step delay), engine-power scaling.
- Define scenarios in config: `NOMINAL`, `TRAIN_DISTURB`, `HELD_OUT_2X` (2× training magnitudes, never seen in training).

### P1.2 Domain-randomized training
- Train with randomized perturbations (curriculum optional). Save separately registered models.

### P1.3 Stress-grid evaluation
- Evaluate over a grid (for example wind × noise) on test seeds. Generate heatmaps of success rate with CIs into `results/figures/`.

### Gate G1
≥85% `LANDED_ON_PAD` on `HELD_OUT_2X` (Wilson lower bound reported). Record the true number regardless.

---

## PHASE 2 — RocketLand-2D (the real simulation)

All numeric parameters live in `configs/rocketland/vehicle.json`. They are **starting estimates**, not facts about Falcon 9. Record each parameter's basis (assumed / public-source / derived) in `docs/PARAMETERS.md`. Equations are in Appendix C; contact in Appendix D; spaces in Appendix E; reward in Appendix F.

### P2.1 Dynamics core (`dynamics.py`, `atmosphere.py`, `config.py`)
- Pure functions, no Gym dependency: `derivatives(state, controls, params, wind) -> dstate`, `rk4_step(...)`.
- Exponential atmosphere `ρ(y) = ρ0·exp(−y/H)`.
- Body-frame drag with separate axial and lateral `Cd·A`.
- Mass depletion `ṁ = −T/(Isp·g0)`; thrust zero when fuel is zero; inertia `I = m·L²/12` (documented simplification).
- Tests: Appendix G, "Dynamics".

### P2.2 Contact model (`contact.py`)
- Foot-spring contact with Coulomb friction (Appendix D). Contact substeps near the ground.
- Tests: Appendix G, "Contact".

### P2.3 Actuators (`actuators.py`)
- Throttle: `0` or `[min_throttle, 1]`, first-order lag, rate limit. Gimbal: angle limit, rate limit. RCS: torque limit.
- Tests: clipping, rate limits, lag step response, bad action rejection (NaN, inf, wrong shape).

### P2.4 Gymnasium environment (`env.py`)
- `RocketLand2D-v0`, registered, continuous `Box` action and observation (Appendix E).
- Reset samples from the scenario's start distribution using the supplied seed.
- On the final step, `info["outcome"]` and `info["touchdown"]` are set (same schema as P0.2).
- `render()` with `rgb_array` and a `human` mode; no windowing required for tests.
- Pass `gymnasium.utils.env_checker.check_env` and SB3 `check_env`.

### P2.5 Start distribution and feasibility filter
- Start ranges (initial): altitude 800–1500 m, `vy` −150…−80 m/s, lateral offset ±300 m, `vx` ±30 m/s, tilt ±10°, `ω` ±0.02 rad/s, fuel 2500–4000 kg.
- Feasibility filter: a start is *feasible* if a reference solver can land it (use the suicide-burn controller first; later MPC). Publish the feasible fraction in `results/metrics/feasibility.json`. Training and the gates use feasible starts only. Never silently drop starts; log how many were rejected and why.

### P2.6 Classical controllers (`controllers/`)
- `null`: engine off. Must `CRASH`/`HARD_LANDING`/`OUT_OF_BOUNDS` ~100%.
- `suicide_burn`: ignition when altitude ≤ stopping distance × margin, where stopping distance uses the *current* net deceleration capability (thrust, mass, gravity). A PD controller holds attitude through gimbal. A simple lateral guidance command sets the target tilt. Cut off the engine at touchdown.
- **Gate G2a:** `suicide_burn` achieves ≥90% `LANDED_ON_PAD` on `NOMINAL` feasible starts (1000 test seeds, CI reported). If it does not, the environment or controller has a bug; fix that before any RL. If still failing after 3 attempts: `BLOCKED`, with a diagnosis of whether the scenario is physically too hard.

### P2.7 RL training (`scripts/train_rocket.py`)
- SB3 PPO (continuous Gaussian policy) and SAC; compare.
- `VecNormalize` for observations and rewards; **save and load the normalization statistics with each model** (registry must reference them).
- Curriculum (Stage A → B → C), promotion when success ≥90% over the last 500 evaluation episodes of the current stage. Stages in `scenarios.json`.
- 5 training seeds per algorithm; validation-seed checkpoint selection; test-seed final evaluation.
- Measure env throughput first. If under ~3k env steps/s per core, profile and accelerate (Numba or vectorization) before long runs. Budget guideline: 2–5M steps per run.
- Optional hybrid (record as separate models): imitation from `suicide_burn` (behavior cloning) then RL fine-tuning; or RL residual on top of `suicide_burn`.

### P2.8 Disturbances and realism (Stage C and held-out)
- Wind: mean wind plus Ornstein–Uhlenbeck gusts; altitude-dependent.
- Sensors (`sensors.py`): Gaussian noise per channel, bias, update delay; optional estimator (`estimator.py`, EKF or complementary filter) so the policy sees *estimated* state in Stage C. Log estimator error.
- Parameter randomization: mass ±5%, Isp ±3%, thrust ±3%, drag coefficients ±20%, throttle lag ±20%.
- Moving pad: horizontal drift and small tilt; success judged relative to pad velocity.
- Scenarios: `NOMINAL`, `ROBUST_1X` (training magnitudes), `ROBUST_2X` (never seen in training).

### P2.9 MPC / convex guidance (stretch)
- Use `cvxpy` for a powered-descent guidance problem with relaxed minimum-throttle handling. Compare against `suicide_burn` and RL. Skip if time-boxed; log it as `SKIPPED` in `STATUS.md`, not silently.

### P2.10 Comparison report
- Evaluate all controllers/policies on **identical test seeds** per scenario. Generate:
  - outcome table with Wilson CIs
  - touchdown velocity and landing-error distributions (p50, p95)
  - fuel use
  - a failure atlas: scatter of start conditions colored by outcome
- All generated into `results/` and injected into the README by `make_report.py`.

### Gates for Phase 2
- **G2a:** classical controller as above.
- **G2b:** best RL policy ≥95% `LANDED_ON_PAD` on `NOMINAL` (1000 test seeds, Wilson lower bound ≥90%).
- **G2c:** best RL policy ≥85% on `ROBUST_2X`.
- RL is **not** required to beat the classical controller; report the comparison honestly either way.

---

## PHASE 3 — Product layer (backend and frontend that actually work)

### P3.1 Backend (`src/server`, FastAPI)
Implement the contract in Appendix H. Requirements:
- The server computes everything. It uses the **same** `evaluation` and `policies` code paths as the CLI.
- Policies come from `models/registry.json`; if a model file is missing or its SHA-256 mismatches, the policy is listed as `available: false` with a reason. Never substitute a fake.
- Pydantic validation on every input; consistent error body `{ "error": { "code": str, "message": str } }`.
- Runs are cancellable; batches report progress; resource limits (max episodes per batch, max concurrent runs).
- Tests: `tests/integration/test_api.py` covering every endpoint, validation errors, cancellation, missing-model behavior, WebSocket streaming order, and determinism (same request → same trajectory hash).

### P3.2 Frontend (`web/`, React + Vite + TypeScript)
- Canvas renderer for the rocket, pad, and trajectory; telemetry plots (altitude, velocities, tilt, throttle, gimbal, fuel).
- Controls (see Appendix I): environment select, policy select, scenario preset, seed, wind/noise sliders, Run, Pause/Resume, Step, Reset, Replay, speed, Batch Run, Compare, Export (JSON and CSV), Cancel.
- **The UI renders only data returned by the backend.** No hardcoded metrics, no fixture data in the production bundle (fixtures live only under `web/tests/`).
- Every control has: enabled/disabled logic, loading state, error state, and a stable `data-testid`.
- Unit tests (Vitest) for state logic; end-to-end tests (Playwright) per Appendix I.
- Footer displays the fidelity statement and the model SHA for the selected policy.

### P3.3 Honesty checks
- A test fails if the production bundle imports from `web/tests/` or contains a known fixture marker.
- A test checks that the "Compare" table values equal the backend's results JSON for the same request.

### Gate G3
Every control in Appendix I has a passing Playwright test, and `pytest` plus `vitest` pass in CI.

---

## PHASE 4 — Release

- **CI** (`.github/workflows/ci.yml`): lint, type check (where practical), fast tests, `make_report.py` diff check, frontend tests, Playwright. Slow training tests excluded from CI but runnable locally with a marker.
- **Docker**: one image for the backend plus built frontend; documented run command.
- **README rewrite**: honest overview, fidelity statement, quickstart that works from a clean clone, results block generated from files, limitations section.
- **`docs/MODEL_CARD.md`**: intended use, fidelity level, validity envelope (the start-state and disturbance ranges actually tested), known failure modes (from the failure atlas), parameter provenance, reproduction commands, seeds, model SHAs.
- **Demo**: `scripts/make_demo.py` records a GIF/MP4 from a real run, with seed and policy printed in the file name.
- Tag `v1.0.0`.

---

## 9. Final acceptance checklist (G4)

- [ ] Clean-clone install and `pytest -q` pass.
- [ ] No hand-typed metric anywhere; `make_report.py` produces zero diff.
- [ ] `STATUS.md` shows every task `DONE`, `BLOCKED` (with evidence), or `SKIPPED` (with reason). No silent gaps.
- [ ] Gates G0, G1, G2a/b/c, G3 each show PASS or an honestly documented FAIL with true numbers.
- [ ] Every results JSON has provenance; every registered model has a SHA-256 and matching evaluation file.
- [ ] No unclassified episodes in any evaluation (outcome counts sum to N).
- [ ] UI: every control tested; nothing works by hardcoded data.
- [ ] `MODEL_CARD.md` states fidelity level, validity envelope, and failure modes.
- [ ] Docs are internally consistent: parameter values in docs match `configs/`; numbers match `results/`.

---

# APPENDICES

## Appendix A — Outcome classification

**Enum (shared):** `LANDED_ON_PAD`, `LANDED_OFF_PAD`, `HARD_LANDING`, `TIPPED`, `CRASH`, `OUT_OF_BOUNDS`, `TIMEOUT`.

**LunarLander classifier** `classify_lunarlander_episode(env, terminated, truncated, last_obs)`; first match wins:
1. `truncated and not terminated` → `TIMEOUT`
2. `env.unwrapped.game_over` → `CRASH`
3. `abs(last_obs[0]) >= 1.0` → `OUT_OF_BOUNDS`
4. `terminated and not env.unwrapped.lander.awake` (came to rest):
   - both legs in contact and `|x|` within pad half-width → `LANDED_ON_PAD`
   - otherwise → `LANDED_OFF_PAD`
5. anything else → raise `RuntimeError` with full context.

Read pad bounds from the env (`helipad_x1`, `helipad_x2`, converted to observation coordinates; verify the conversion) instead of hardcoding 0.2. Classify on the exact terminal step, before reset, using a plain (non-vector) env in evaluation. In SB3 `VecEnv` training, terminal info is available via `info["terminal_observation"]` and the `OutcomeWrapper`'s `info["outcome"]`.

**Touchdown record:** final `x, y, vx, vy, θ, ω`, `|x|`, leg contacts, episode length, counts of each action.

## Appendix B — Wilson interval

For `k` successes of `n`, `p = k/n`, `z = 1.96`:
```
center = (p + z²/(2n)) / (1 + z²/n)
half   = z · sqrt(p(1−p)/n + z²/(4n²)) / (1 + z²/n)
CI     = [center − half, center + half]   (clip to [0,1])
```
Edge cases: `n = 0` raises; `k = 0` and `k = n` stay within [0, 1]; `n = 1` works.

## Appendix C — RocketLand-2D physics

**Frame:** x right, y up. `θ` = tilt from vertical, CCW positive. Body axis `u = (−sinθ, cosθ)`. Gimbal `δ`: CCW rotation of the thrust vector relative to the body axis.

```
Thrust vector direction angle from vertical: θ + δ
Thrust force:   F_T = T · ( −sin(θ+δ),  cos(θ+δ) )
Translation:    m·ẍ = −T·sin(θ+δ) + F_drag,x + F_contact,x
                m·ÿ =  T·cos(θ+δ) − m·g + F_drag,y + F_contact,y
Rotation:       θ̇ = ω
                I·ω̇ = −T·r_e·sin(δ) + τ_RCS + τ_aero + τ_contact
                (r_e = distance from CG to engine along body axis)
Mass:           ṁ = −T / (Isp·g0)        (T = 0 when propellant = 0)
Thrust:         T = throttle·T_max,  throttle ∈ {0} ∪ [throttle_min, 1], through a first-order lag
Drag:           F_drag = −½·ρ(y)·|v_rel|·( Cd_a·A_a·v_rel,axial,  Cd_l·A_l·v_rel,lateral ) in body frame,
                v_rel = v − v_wind; convert back to the world frame
Atmosphere:     ρ(y) = ρ0·exp(−y/H),  ρ0 ≈ 1.225 kg/m³, H ≈ 8500 m
```
Sign check (must be verified by a test, not trusted): torque from the engine about the CG is `−T·r_e·sin(δ)`. A positive `δ` rotates the vehicle clockwise.

**Starting estimates (config; mark as assumed in PARAMETERS.md):**
- dry mass ≈ 25,000 kg; propellant 2,500–4,000 kg; length ≈ 40 m; diameter ≈ 3.7 m
- `T_max` ≈ 845 kN; `Isp` ≈ 282 s; `throttle_min` ≈ 0.4; gimbal limit ±5°
- `g = 9.81`, `g0 = 9.80665`
- Key behavior: minimum thrust (≈338 kN) exceeds vehicle weight (≈245–290 kN), so the vehicle **cannot hover** and must execute a suicide burn.

**Integration:** RK4; physics `dt` ≈ 0.02 s in flight; control step 0.05–0.1 s (hold controls constant between decisions). Near the ground (any foot within ~1 m of contact) substep to ~0.001–0.002 s for contact stability.

## Appendix D — Contact model

- Feet at `p_i = CG + R(θ)·(±w, −h)` where `R(θ)·(bx,by) = (bx·cosθ − by·sinθ, bx·sinθ + by·cosθ)`; velocity from rigid-body kinematics.
- Normal force per foot: `N = max(0, k·pen − c·v_n)`; clamp so the foot never pulls the vehicle down.
- Friction: `F_t = −μ·N·tanh(v_t / v_eps)`.
- Apply forces at the feet; torque about the CG.
- Body or nose touching the ground, or `|θ|` beyond a tip limit (config, e.g. 30° at settle), → `TIPPED`.
- First-contact speed above `v_crash` (config, e.g. 5 m/s) → `HARD_LANDING`.
- Success: touchdown kinematics (recorded at first contact) within limits (`|vy| ≤ 2 m/s`, `|vx| ≤ 1 m/s`, `|θ| ≤ 5°`) **and** the vehicle settles (all speeds below small thresholds for 1.0 s while in contact) **on the pad** (half-width configurable, start 15 m) → `LANDED_ON_PAD`. Settled off the pad → `LANDED_OFF_PAD`. Settled with touchdown limits violated → `HARD_LANDING`.
- Episode limit: e.g. 90 s of simulated time → `TIMEOUT`; `OUT_OF_BOUNDS` for leaving a configured corridor.

## Appendix E — Spaces

**Action:** `Box(-1, 1, shape=(3,))`
- `a0` throttle command: `a0 < −0.9` → engine off; otherwise map linearly onto `[throttle_min, 1]`.
- `a1` gimbal: maps to `±gimbal_limit`.
- `a2` RCS torque: maps to `±τ_RCS_max`.

**Observation (float32, normalized by fixed scales from config):** pad-relative `x`, altitude `y`, `vx`, `vy`, `sinθ`, `cosθ`, `ω`, fuel fraction, previous throttle, previous gimbal, left/right leg contact; plus pad velocity and tilt when the moving pad is enabled. In Stage C the position/velocity/attitude entries come from the sensor+estimator chain, not ground truth. Observation bounds are tested by fuzzing.

## Appendix F — Reward (potential-based shaping)

```
Φ(s)   = −( w_pos·‖Δpos‖/L_pos + w_vel·‖v_rel‖/V_ref + w_ang·|θ| )
r_t    = γ·Φ(s') − Φ(s) − c_fuel·(Δfuel / fuel_0) − c_act·‖a_t − a_{t−1}‖
terminal: + R_land · quality    for LANDED_ON_PAD  (quality ∈ [0.5, 1] from touchdown margins)
          − R_fail              for every failure outcome
```
All weights live in config and are tuned **only on validation seeds**. Shaping must not make a failure outcome more attractive than a landing; add a test that scripted failure trajectories score below scripted successful ones.

## Appendix G — Test matrix (minimum)

**Outcomes and statistics**

| Test | Proves |
|---|---|
| Null policy over 200 seeds is ~100% failure outcomes (`CRASH`/`OUT_OF_BOUNDS`) | Classifier sees crashes |
| `GymHeuristicPolicy` is mostly `LANDED_ON_PAD` | Classifier recognizes landings |
| Outcome counts sum to N in every evaluation | No unclassified episodes |
| Fake env objects force each classifier branch incl. `TIMEOUT` | Every branch reachable |
| Required Gymnasium attributes exist (fails loudly otherwise) | API assumptions verified |
| Wilson at `k=0`, `k=n`, `n=1`, `n=0` (raises) | Math edge cases |
| Same seeds → identical JSONL twice | Reproducibility |
| README result block equals `make_report.py` output | Traceable docs |

**Dynamics**

| Test | Check |
|---|---|
| Free fall, engine off, no drag | `y = y0 + v0·t − ½·g·t²` within 1e-6 |
| Constant vertical burn, no drag | Δv equals `Isp·g0·ln(m0/m1) − g·t` within tolerance |
| Mass accounting | Fuel ≥ 0; thrust 0 at zero fuel; Δmass equals integrated flow |
| Gimbal sign | Positive `δ` yields negative angular acceleration |
| Convergence | Halving `dt` changes the trajectory less than a stated tolerance |
| Energy sanity (no thrust, no drag) | Total mechanical energy conserved within tolerance |
| Monotonic sensitivity | More drag, less mass, or less thrust moves outcomes the expected direction |

**Contact**

| Test | Check |
|---|---|
| Rest on level ground | Settles; penetration steady; no jitter beyond tolerance |
| Tilted drop | Tips past the limit → `TIPPED` |
| Friction slide | Horizontal speed decays per Coulomb friction |
| High-speed contact | → `HARD_LANDING` |
| Energy not created in contact | Total energy never increases |

**Environment**

| Test | Check |
|---|---|
| `check_env` (Gymnasium and SB3) | Passes |
| Bad actions (NaN, inf, wrong shape) | Rejected with clear errors |
| Fuzz 10k random steps | Observations finite, float32, within declared bounds |
| Determinism | Same seed and actions → identical trajectory hash |
| Scripted starts reach each outcome | Every outcome branch exercised |
| Golden trajectory regression | Detects silent physics changes |
| Reward ordering | Scripted failures score below scripted successes |
| Throughput | Measured and recorded in `results/metrics/throughput.json` |

## Appendix H — API contract (FastAPI)

| Method and path | Purpose | Notes |
|---|---|---|
| `GET /api/health` | Liveness | Returns version, git commit |
| `GET /api/policies` | List policies | Each: `id, name, env, kind, available, reason?, model_sha?` |
| `GET /api/scenarios` | Scenario presets | Includes ranges for sliders |
| `POST /api/runs` | Start a single run | Body: `env, policy_id, seed, scenario, overrides` → `{run_id}` |
| `WS /api/runs/{id}/stream` | Telemetry stream | Messages `{type:"frame", t, state, action, reward}` then `{type:"done", outcome, touchdown, metrics}` |
| `GET /api/runs/{id}` | Full trajectory + summary | For replay and export |
| `DELETE /api/runs/{id}` | Cancel | Idempotent |
| `POST /api/batches` | Monte Carlo batch | Body: `policy_ids[], n_episodes, seed_start, scenario` → `{batch_id}`; capped by config |
| `GET /api/batches/{id}` | Progress and summary | Outcome rates with Wilson CIs |
| `GET /api/results` | Generated results JSON | The exact files the README is built from |

Errors: `{ "error": { "code", "message" } }` with 4xx for client mistakes. Same request with the same seed → identical trajectory (hash returned in the summary).

## Appendix I — Frontend control map (each needs an e2e test)

| Control (`data-testid`) | Action | Expected result | Disabled when |
|---|---|---|---|
| `env-select` | Choose environment | Policy and scenario lists reload from API | Loading |
| `policy-select` | Choose policy | Model SHA shown; unavailable policies marked | No policies |
| `scenario-select` | Load preset | Sliders update to preset values | — |
| `seed-input` | Set seed | Invalid input shows validation error | — |
| `wind-slider`, `noise-slider` | Override disturbance | Included in the next run request | Preset locks them |
| `run-btn` | `POST /api/runs`, open stream | Canvas animates from streamed frames | Invalid form or run active |
| `pause-btn` / `resume-btn` | Pause rendering | Frame index freezes/resumes | No active run |
| `step-btn` | Advance one frame | Frame index +1 | Not paused |
| `cancel-btn` | `DELETE` run | Stream stops; status "cancelled" | No active run |
| `replay-btn` | Replay finished run | Same frames; telemetry identical | No finished run |
| `speed-select` | Playback speed | Frame rate changes | — |
| `batch-btn` | `POST /api/batches` | Progress bar, then outcome table with CIs | Invalid form |
| `compare-btn` | Batch for multiple policies | Side-by-side table matching `/api/results` format | <2 policies chosen |
| `export-json-btn`, `export-csv-btn` | Download trajectory | File contents equal API data | No finished run |
| Error states | Backend 4xx/5xx, network drop | Visible error message; app stays usable | — |

## Appendix J — Results JSON provenance (required keys)

```json
{
  "schema_version": 1,
  "kind": "evaluation",
  "env_id": "...",
  "scenario": "...",
  "policy": {"name": "...", "model_sha256": "...", "deterministic": true},
  "seeds": {"range": [20000, 20999], "hash": "..."},
  "n_episodes": 1000,
  "outcome_counts": {"LANDED_ON_PAD": 0, "...": 0},
  "outcome_rates": {"LANDED_ON_PAD": {"rate": 0.0, "ci95": [0.0, 0.0]}},
  "reward": {"mean": 0.0, "median": 0.0, "std": 0.0},
  "touchdown": {"vy_p50": 0.0, "vy_p95": 0.0, "x_err_p50": 0.0, "x_err_p95": 0.0},
  "provenance": {
    "git_commit": "...", "git_dirty": false, "python": "...",
    "packages": {"gymnasium": "...", "stable_baselines3": "...", "numpy": "..."},
    "config_hash": "...", "timestamp_utc": "..."
  },
  "episodes_file": "results/episodes/....jsonl"
}
```

## Appendix K — `STATUS.md` format

```
| Task | Status (TODO/IN_PROGRESS/DONE/BLOCKED/SKIPPED) | Evidence (paths) | Gate result | Notes |
```
Rules: one row per task ID in this document; `DONE` requires an evidence path; `BLOCKED` requires the diagnosis and the three attempts; `SKIPPED` requires a reason. Update the file in the same commit as the work.

---

## When you are unsure

1. Re-read the rules in Section 1.
2. Prefer the smallest change that keeps tests green and claims verifiable.
3. If a result looks too good, assume a measurement bug first, and try to break it (null policy, scrambled seeds, wrong-pad test) before believing it.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-10-17T18:57:51+05:30.
</ADDITIONAL_METADATA>
<USER_SETTINGS_CHANGE>
The user changed setting `Model Selection` from None to Claude Sonnet 4.6 (Thinking). No need to comment on this change if the user doesn't ask about it. If reporting what model you are, please use a human readable name instead of the exact string.
</USER_SETTINGS_CHANGE>