# docs/DECISIONS.md — Architecture & Design Decisions Log
**Format:** per AGENT_PLAN Section 1, Rule 9.  
Every ambiguity resolved by the agent is logged here with the decision, alternatives considered, and the reason for choosing this path.

---

## D-001 — Outcome classifier: use `lander.awake` not reward threshold
**Date:** 2026-10-17  
**Context:** The audit (P0.0) confirmed `game_over`, `lander.awake`, `helipad_x1`, and `helipad_x2` all exist in Gymnasium 1.4.0. AGENT_PLAN Appendix A specifies the algorithm.  
**Decision:** Implement `classify_lunarlander_episode` exactly per Appendix A using these attributes. Raise `RuntimeError` if no branch matches (Rule 6: no silent failure).  
**Alternatives:**
- Keep reward-threshold classifier (rejected: provably wrong, leaves 19% unclassified)
- Use only `game_over` (rejected: does not distinguish LANDED_ON_PAD vs LANDED_OFF_PAD)

**Reason:** Appendix A is clear, the attributes exist, and correctness requires event-based classification.

---

## D-002 — Pad-bounds conversion from world space to observation space
**Date:** 2026-10-17  
**Context:** `helipad_x1 = 8.0`, `helipad_x2 = 12.0` in Box2D world units. The observation `obs[0]` is in normalized space where the pad center = 0. Must convert to determine "within pad half-width".  
**Decision:** Read the conversion from Gymnasium source (VIEWPORT_W, SCALE). Verified: the env normalizes `x` as `(body.position.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2)`. With VIEWPORT_W=600, SCALE=30: center = 600/30/2 = 10. Half-width in world coords = (12-8)/2 = 2. In obs coords: `2 / (600/30/2) = 2/10 = 0.2`. So `|obs[0]| <= 0.2` means on-pad. Add a test to verify this matches the env's own helipad attributes at runtime.  
**Alternatives:**
- Hardcode 0.2 (rejected: brittle if env changes)
- Compute dynamically from helipad_x1/x2 at every call (chosen: runtime-safe)

**Reason:** Compute from actual attributes; add a test to catch any future env change.

---

## D-003 — Seed ranges in `configs/seeds.json`
**Date:** 2026-10-17  
**Context:** AGENT_PLAN specifies: train = free, validation = 10000–10199, test = 20000–20999.  
**Decision:** Adopt exactly these ranges. Training seeds will use 0–9999 range (5 seeds: 0, 1, 2, 3, 4 for clear labelling).  
**Alternatives:**
- Use arbitrary training seeds (rejected: less readable)
- Use higher training seeds (rejected: overlaps val/test)

**Reason:** Simplest choice consistent with AGENT_PLAN.

---

## D-004 — `numpy<2` pin: retain for now
**Date:** 2026-10-17  
**Context:** numpy 1.26.4 is installed; the pin is `>=1.24.0,<2.0.0`. Gymnasium 1.4.0 and SB3 2.9.0 are the installed versions.  
**Decision:** Retain the `<2.0.0` cap until a clean-venv install test with numpy 2.x is run. Document as an open investigation item, not remove blindly. If numpy 2 works, remove the cap and update the lockfile.  
**Alternatives:**
- Remove immediately (risk: may break Box2D SWIG bindings; untested)
- Leave as-is forever (rejected: unnecessary restriction if unnecessary)

**Reason:** Rule 3 — do not make untested claims about libraries.

---

## D-005 — evaluation harness: plain (non-vector) env in evaluation
**Date:** 2026-10-17  
**Context:** AGENT_PLAN Appendix A: "Classify on the exact terminal step, before reset, using a plain (non-vector) env in evaluation."  
**Decision:** All evaluation uses `gym.make()` directly, not SB3 `VecEnv`. Episode loop exits immediately on `terminated or truncated`, reads `env.unwrapped` attributes before reset.  
**Alternatives:**
- Use VecEnv for speed (rejected: `terminal_observation` path obscures the unwrapped env's post-terminal state; risk of incorrect classification)

**Reason:** Correctness over speed in evaluation; SB3 VecEnv training is separate from evaluation.

---

## D-006 — P2.9 MPC/convex guidance: SKIPPED
**Date:** 2026-10-17  
**Context:** AGENT_PLAN P2.9 marked as "stretch". Implementing `cvxpy`-based powered descent guidance is a substantial engineering task that requires solving and testing a constrained optimal control problem.  
**Decision:** Mark as SKIPPED in STATUS.md. Logged here explicitly so the skip is documented, not silent.  
**Reason:** AGENT_PLAN explicitly labels it stretch; the non-stretch path (suicide_burn + RL) is the primary requirement.

---

## D-007 — `gymnasium.error.Error` as the catch-all exception
**Date:** 2026-10-17  
**Context:** `verify_environment.py` uses `gym.error.NamespaceNotFound` as the env-version fallback exception. The audit shows this is wrong; the correct exception for a missing *version* is `VersionNotFound`. For maximum safety across future Gymnasium versions, use the base `gymnasium.error.Error`.  
**Decision:** Replace `NamespaceNotFound` with `gymnasium.error.Error` in `verify_environment.py` and `utils.py::get_supported_env_id`. Add `VersionNotFound` as a documented first-match for clarity.  
**Alternatives:**
- Use `VersionNotFound` only (narrower; misses other registration errors)
- Use bare `Exception` (too broad; swallows unrelated errors like ImportError)

**Reason:** `gymnasium.error.Error` is the documented base for registration errors; safe and specific enough.
