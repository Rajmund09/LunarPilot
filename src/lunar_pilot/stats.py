"""
LunarPilot — Statistics utilities
===================================
Wilson score interval (Appendix B) and bootstrap CI for means.
Pure functions; no Gymnasium dependency.
"""

from __future__ import annotations

import math
from typing import Sequence

# ---------------------------------------------------------------------------
# Wilson score interval (Appendix B)
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score confidence interval for a proportion.

    Parameters
    ----------
    k : int   — number of successes
    n : int   — total observations
    z : float — z-score (default 1.96 for 95%)

    Returns
    -------
    (lower, upper) both clipped to [0, 1].

    Raises
    ------
    ValueError — if n == 0.
    """
    if n == 0:
        raise ValueError("wilson_ci: n must be > 0")
    if k < 0 or k > n:
        raise ValueError(f"wilson_ci: k={k} out of range [0, {n}]")

    p = k / n
    z2 = z * z
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    margin = z * math.sqrt(max(0.0, p * (1 - p) / n + z2 / (4 * n * n))) / denominator
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return lo, hi


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """Return only the lower bound of the Wilson interval."""
    return wilson_ci(k, n, z)[0]


# ---------------------------------------------------------------------------
# Bootstrap CI for means
# ---------------------------------------------------------------------------

def bootstrap_mean_ci(
    values: Sequence[float],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean.

    Parameters
    ----------
    values      : sample data
    n_bootstrap : number of bootstrap replicates
    confidence  : coverage (e.g. 0.95 for 95% CI)
    seed        : RNG seed for reproducibility

    Returns
    -------
    (lower, upper)

    Raises
    ------
    ValueError — if values is empty.
    """
    import numpy as np  # imported here to keep stats.py importable without numpy

    vals = list(values)
    if len(vals) == 0:
        raise ValueError("bootstrap_mean_ci: values must not be empty")

    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    means = np.array(
        [rng.choice(arr, size=n, replace=True).mean() for _ in range(n_bootstrap)]
    )
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return lo, hi
