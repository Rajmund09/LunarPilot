"""
LunarPilot — Outcome Classification
====================================
Shared outcome enum and classifiers for LunarLander.  All classification
uses terminal *events* (env attributes), never cumulative reward thresholds.

Decision D-001: use game_over / lander.awake / pad-bounds, not reward.
Decision D-002: pad half-width computed from helipad_x1/x2 at runtime.
"""

from __future__ import annotations

import enum
from typing import Any

import gymnasium as gym
import numpy as np


class Outcome(enum.Enum):
    """Canonical outcome labels, shared across LunarLander and RocketLand."""

    LANDED_ON_PAD = "LANDED_ON_PAD"
    LANDED_OFF_PAD = "LANDED_OFF_PAD"
    HARD_LANDING = "HARD_LANDING"
    TIPPED = "TIPPED"
    CRASH = "CRASH"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    TIMEOUT = "TIMEOUT"


# ---------------------------------------------------------------------------
# Pad-half-width helper
# ---------------------------------------------------------------------------

def _pad_half_width_in_obs(env_unwrapped: Any) -> float:
    """Return pad half-width in normalised observation x-coordinates.

    Gymnasium LunarLander-v3 normalises x as:
        obs[0] = (world_x - center_x) / center_x
    where center_x = VIEWPORT_W / SCALE / 2 = 600 / 30 / 2 = 10.

    helipad_x1=8, helipad_x2=12  →  half-width world=2  →  obs=0.2.

    We compute this from the env attributes so a future env change is caught
    by the runtime assertion added in tests.
    """
    x1 = float(env_unwrapped.helipad_x1)
    x2 = float(env_unwrapped.helipad_x2)
    # centre of the viewport in world coords; the obs normalisation divides by this
    # VIEWPORT_W=600, SCALE=30 → 10.0
    world_center = (x1 + x2) / 2.0  # midpoint of pad (should be 10.0)
    half_width_world = (x2 - x1) / 2.0
    # obs_x = (world_x - world_center) / world_center
    # so pad edge in obs coords = half_width_world / world_center
    half_width_obs = half_width_world / world_center
    return half_width_obs


# ---------------------------------------------------------------------------
# LunarLander classifier (Appendix A)
# ---------------------------------------------------------------------------

def classify_lunarlander_episode(
    env: gym.Env,
    terminated: bool,
    truncated: bool,
    last_obs: np.ndarray,
) -> tuple[Outcome, dict]:
    """Classify the final step of a LunarLander episode.

    Must be called *before* env.reset(), on the terminal observation.
    Uses a plain (non-vector) env so env.unwrapped attributes are accessible.

    Returns (Outcome, touchdown_record).
    Raises RuntimeError if no branch matches (no silent failures, Rule 6).

    Branch order (first match wins, per Appendix A):
      1. truncated and not terminated  → TIMEOUT
      2. env.unwrapped.game_over       → CRASH
      3. |obs[0]| >= 1.0               → OUT_OF_BOUNDS
      4. terminated and not lander.awake (came to rest):
           both legs + |obs[0]| within pad half-width → LANDED_ON_PAD
           otherwise                                   → LANDED_OFF_PAD
      5. anything else                 → RuntimeError
    """
    uw = env.unwrapped

    # --- build touchdown record from the terminal observation ---
    obs = np.asarray(last_obs, dtype=np.float32)
    x, y, vx, vy, theta, omega = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3]), float(obs[4]), float(obs[5])
    left_contact = bool(obs[6] > 0.5)
    right_contact = bool(obs[7] > 0.5)
    touchdown: dict = {
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "theta": theta,
        "omega": omega,
        "abs_x": abs(x),
        "left_contact": left_contact,
        "right_contact": right_contact,
    }

    # Branch 1: truncation (episode time limit hit, never terminated)
    if truncated and not terminated:
        return Outcome.TIMEOUT, touchdown

    # Branch 2: game_over (crash)
    if uw.game_over:
        return Outcome.CRASH, touchdown

    # Branch 3: out of bounds (normalised x >= 1.0 means lander left the visible area)
    if abs(x) >= 1.0:
        return Outcome.OUT_OF_BOUNDS, touchdown

    # Branch 4: lander came to rest (not awake)
    if terminated and not uw.lander.awake:
        pad_hw = _pad_half_width_in_obs(uw)
        both_legs = left_contact and right_contact
        on_pad = both_legs and (abs(x) <= pad_hw)
        if on_pad:
            return Outcome.LANDED_ON_PAD, touchdown
        else:
            return Outcome.LANDED_OFF_PAD, touchdown

    # Branch 5: unhandled state — raise loudly
    raise RuntimeError(
        f"classify_lunarlander_episode: no branch matched.\n"
        f"  terminated={terminated}, truncated={truncated}, game_over={uw.game_over}, "
        f"lander.awake={uw.lander.awake}, obs[0]={x:.4f}, obs[6]={obs[6]:.2f}, obs[7]={obs[7]:.2f}"
    )


# ---------------------------------------------------------------------------
# OutcomeWrapper
# ---------------------------------------------------------------------------

class OutcomeWrapper(gym.Wrapper):
    """Gymnasium wrapper that writes info['outcome'] and info['touchdown'] on the
    final step of every episode.

    Works in plain env evaluation.  In SB3 VecEnv training the terminal_observation
    is available via info['terminal_observation']; the VecEnv training path reads
    info['outcome'] injected here.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            outcome, touchdown = classify_lunarlander_episode(
                self.env, terminated, truncated, obs
            )
            info["outcome"] = outcome.value
            info["touchdown"] = touchdown
        return obs, reward, terminated, truncated, info
