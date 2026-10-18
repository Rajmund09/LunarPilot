"""
LunarPilot — Provenance
========================
Captures git commit, package versions, config/seed hashes, model SHA-256,
and timestamp into a provenance dict for inclusion in every results JSON.

Decision: use subprocess for git rather than a git library to avoid adding
a dependency just for provenance.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_commit() -> tuple[str, bool]:
    """Return (commit_sha, is_dirty).  Returns ('unknown', True) on failure."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = len(status) > 0
        return sha, dirty
    except Exception:
        return "unknown", True


def _pkg_version(pkg: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(pkg)
    except Exception:
        return "unknown"


def _sha256_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return "file_not_found"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_json(obj: Any) -> str:
    """Deterministic SHA-256 of a JSON-serialisable object."""
    encoded = json.dumps(obj, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _hash_seed_list(seeds: list[int]) -> str:
    return _hash_json(seeds)


def make_provenance(
    config: dict | None = None,
    seeds: list[int] | None = None,
    model_path: str | Path | None = None,
    extra_packages: list[str] | None = None,
) -> dict:
    """Build a provenance dict for inclusion in results JSON (Appendix J).

    Parameters
    ----------
    config       : experiment config dict (will be hashed)
    seeds        : list of evaluation seeds (will be hashed)
    model_path   : path to model file (SHA-256 computed)
    extra_packages: additional package names to record versions for
    """
    commit, dirty = _git_commit()

    packages = {
        "gymnasium": _pkg_version("gymnasium"),
        "stable_baselines3": _pkg_version("stable-baselines3"),
        "numpy": _pkg_version("numpy"),
        "torch": _pkg_version("torch"),
    }
    for pkg in (extra_packages or []):
        packages[pkg] = _pkg_version(pkg)

    prov: dict = {
        "git_commit": commit,
        "git_dirty": dirty,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "packages": packages,
        "config_hash": _hash_json(config) if config is not None else None,
        "seed_list_hash": _hash_seed_list(seeds) if seeds is not None else None,
        "model_sha256": _sha256_file(model_path) if model_path is not None else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return prov
