"""Adaptive weighting for the Nexus scoring engine."""
from __future__ import annotations

from typing import Dict


DEFAULT_WEIGHTS: Dict[str, float] = {
    "trend": 0.35,
    "momentum": 0.30,
    "quality": 0.20,
    "risk": 0.15,
}

REGIME_PROFILES: Dict[str, Dict[str, float]] = {
    "expansion": {"trend": 0.40, "momentum": 0.30, "quality": 0.20, "risk": 0.10},
    "inflation": {"trend": 0.25, "momentum": 0.25, "quality": 0.30, "risk": 0.20},
    "stress": {"trend": 0.10, "momentum": 0.10, "quality": 0.30, "risk": 0.50},
    "recovery": {"trend": 0.25, "momentum": 0.35, "quality": 0.20, "risk": 0.20},
}


def _normalise(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(value for value in weights.values() if value is not None)
    if total <= 0:
        return {key: 0.0 for key in DEFAULT_WEIGHTS}
    return {key: round(float(value) / total, 6) for key, value in weights.items()}


def compute_weights(regime: str) -> Dict[str, float]:
    """Return component weights for the provided macro ``regime``."""

    profile = REGIME_PROFILES.get(regime.lower().strip(), DEFAULT_WEIGHTS)
    merged = {**DEFAULT_WEIGHTS, **profile}
    return _normalise({key: merged.get(key, 0.0) for key in DEFAULT_WEIGHTS})


__all__ = ["DEFAULT_WEIGHTS", "REGIME_PROFILES", "compute_weights"]
