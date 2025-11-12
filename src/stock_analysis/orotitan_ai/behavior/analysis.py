"""High-level orchestration for the OroTitan behavioural layer."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Sequence

from .bias_bank import get_bias_definitions
from .coach import build_recommendations
from .indicators import compute_indicators
from .persistence import BehaviourStore, PersistenceMode
from .rules import evaluate_biases
from .scoring import aggregate_scores
from .schemas import BehaviorAnalysis, BiasSignal

LOGGER = logging.getLogger("orotitan.behavior")

_DEFAULT_THRESHOLD = 30.0
_DEFAULT_MAX_INFLUENCE = 0.25


def _normalise_context(context: Mapping[str, Any] | None) -> Dict[str, Any]:
    if context is None:
        return {}
    if isinstance(context, dict):
        return dict(context)
    return dict(context.items())


def _limit_biases(signals: Iterable[BiasSignal], limit: int = 3) -> list[BiasSignal]:
    ordered = sorted(signals, key=lambda item: item.score, reverse=True)
    return ordered[:limit]


def analyze_behavior(
    tickers: Sequence[str],
    context: Mapping[str, Any] | None = None,
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    max_influence: float = _DEFAULT_MAX_INFLUENCE,
    persist: PersistenceMode = "jsonl",
) -> BehaviorAnalysis:
    """Return a behavioural profile for the supplied tickers."""

    payload = _normalise_context(context)
    indicators = compute_indicators(tickers, payload)
    bias_signals = evaluate_biases(indicators)
    definitions = get_bias_definitions()
    behavioural_score, adjustment, tags = aggregate_scores(
        bias_signals,
        definitions,
        threshold=threshold,
        max_influence=max_influence,
    )
    top_biases = _limit_biases(bias_signals)
    recommendations = build_recommendations(top_biases)
    metadata: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tickers": list(tickers),
        "indicator_count": len(indicators),
    }
    analysis = BehaviorAnalysis(
        behavioral_score=behavioural_score,
        top_biases=top_biases,
        recommendations=recommendations,
        confidence_adjustment=adjustment,
        tags=tags,
        metadata=metadata,
    )

    if persist != "none":
        try:
            store = BehaviourStore(mode=persist)
            store.append(tickers, analysis, payload)
        except Exception as exc:  # pragma: no cover - persistence must not break analysis
            LOGGER.warning("Behaviour persistence failed", exc_info=exc)

    LOGGER.info(
        "Behavioural analysis computed",
        extra={
            "tickers": ",".join(tickers),
            "score": round(behavioural_score, 2),
            "adjustment": round(adjustment, 4),
            "biases": [signal.bias_id for signal in top_biases],
        },
    )
    return analysis


__all__ = ["analyze_behavior"]
