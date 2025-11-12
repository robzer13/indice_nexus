"""Aggregate bias scores into global behavioural metrics."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from .bias_bank import BiasDefinition
from .schemas import BiasSignal


def _normalise_weights(definitions: Sequence[BiasDefinition]) -> Tuple[float, dict[str, float]]:
    weights = {definition.bias_id: max(definition.weight, 0.0) for definition in definitions}
    total = sum(weights.values()) or 1.0
    for key in weights:
        weights[key] /= total
    return total, weights


def aggregate_scores(
    signals: Iterable[BiasSignal],
    definitions: Sequence[BiasDefinition],
    *,
    threshold: float = 30.0,
    max_influence: float = 0.25,
) -> tuple[float, float, list[str]]:
    """Return (score, confidence_adjustment, tags)."""

    _, weights = _normalise_weights(definitions)
    weighted = 0.0
    used: List[str] = []
    for signal in signals:
        weight = weights.get(signal.bias_id, 0.0)
        weighted += signal.score * weight
        if signal.score >= 0.5:
            used.append(signal.bias_id)
    behavioural_score = float(max(0.0, min(1.0, weighted))) * 100.0

    if behavioural_score <= threshold:
        delta = threshold - behavioural_score
        adjustment = min(max_influence, (delta / max(threshold, 1.0)) * max_influence)
    elif behavioural_score <= 60.0:
        span = 60.0 - threshold
        ratio = (behavioural_score - threshold) / max(span, 1.0)
        adjustment = -min(max_influence * 0.4, ratio * max_influence * 0.4)
    else:
        ratio = (behavioural_score - 60.0) / 40.0
        adjustment = -min(max_influence, ratio * max_influence)

    tags = [f"bias:{bias_id}" for bias_id in used[:3]]
    if behavioural_score >= 70:
        tags.append("alert:high")
    elif behavioural_score >= 40:
        tags.append("monitor:medium")
    else:
        tags.append("status:calm")

    return behavioural_score, float(adjustment), tags


__all__ = ["aggregate_scores"]
