"""Decision policies for the OroTitan AI layer."""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, TYPE_CHECKING

try:  # pragma: no cover - optional dependency
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - numpy may be unavailable in tests
    _np = None

if TYPE_CHECKING:  # pragma: no cover - typing helper
    import numpy as np

    VectorLike = np.ndarray
else:
    VectorLike = Sequence[float]

import pandas as pd

from .behavior import analyze_behavior, BehaviorAnalysis

logger = logging.getLogger(__name__)

if _np is not None:
    _RNG = _np.random.default_rng(42)
else:  # pragma: no cover - fallback deterministic RNG
    _RNG = random.Random(42)
_ACTION_THRESHOLD = 0.2


@dataclass(slots=True)
class Decision:
    """Structured decision output for downstream consumers."""

    date: pd.Timestamp
    ticker: str
    score: float
    action: str
    size: float
    confidence: float
    rationale: str
    factors: Dict[str, float]
    behavior: BehaviorAnalysis | None = None


def _normalise_weights(nexus_weights: Dict[str, float]) -> Dict[str, float]:
    if not nexus_weights:
        return {"trend": 0.25, "momentum": 0.25, "fundamental": 0.25, "behavioral": 0.25}
    total = float(sum(max(value, 0.0) for value in nexus_weights.values()))
    if total <= 0.0:
        return {key: 0.25 for key in ("trend", "momentum", "fundamental", "behavioral")}
    return {key: max(value, 0.0) / total for key, value in nexus_weights.items()}


def decide_for_date(
    ticker: str,
    date: pd.Timestamp,
    vectors: VectorLike,
    nexus_weights: Dict[str, float],
    risk_budget: float = 0.1,
    temperature: float = 0.0,
    *,
    enable_behavior: bool = False,
    behavior_context: Mapping[str, object] | None = None,
    behavior_threshold: float = 30.0,
    max_behavior_influence: float = 0.25,
    behavior_persist: str = "jsonl",
) -> Decision:
    """Generate a deterministic action with optional exploratory noise."""
    if _np is not None:
        array = _np.asarray(vectors, dtype=float)
        if array.ndim == 2:
            if array.shape[0] != 1:
                raise ValueError("vectors must be 1-D or a single-row 2-D array")
            vector_1d = array.flatten()
        elif array.ndim == 1:
            vector_1d = array
        else:
            raise ValueError("vectors must be 1-D or a single-row 2-D array")
        vector_1d = _np.nan_to_num(vector_1d, copy=False)
        if vector_1d.size == 0:
            raise ValueError("vectors array is empty")
        values: Sequence[float] = vector_1d.tolist()
    else:
        if isinstance(vectors, Sequence):
            if vectors and isinstance(vectors[0], Sequence):
                rows = list(vectors)
                if len(rows) != 1:
                    raise ValueError("vectors must be 1-D or a single-row 2-D array")
                seq = list(rows[0])
            else:
                seq = list(vectors)
        else:  # pragma: no cover - generic iterable fallback
            seq = list(vectors)
        if not seq:
            raise ValueError("vectors array is empty")
        values = [0.0 if value != value else float(value) for value in seq]

    weights = _normalise_weights(nexus_weights)
    mean_value = (
        float(_np.asarray(values).mean()) if _np is not None else sum(values) / len(values)
    )
    base_score = float(_np.tanh(mean_value)) if _np is not None else math.tanh(mean_value)
    weighted_score = base_score * float(sum(weights.values()))

    noise = 0.0
    if temperature > 0:
        if _np is not None:
            noise = float(_RNG.normal(0.0, temperature * 0.05))
        else:
            noise = float(_RNG.gauss(0.0, temperature * 0.05))
    adjusted_raw = weighted_score + noise
    adjusted_score = (
        float(_np.clip(adjusted_raw, -1.0, 1.0)) if _np is not None else float(max(-1.0, min(1.0, adjusted_raw)))
    )

    clipped = (
        float(_np.clip(adjusted_score, -1.0, 1.0))
        if _np is not None
        else float(max(-1.0, min(1.0, adjusted_score)))
    )
    size = clipped * max(risk_budget, 0.0)

    if adjusted_score > _ACTION_THRESHOLD:
        action = "BUY"
    elif adjusted_score < -_ACTION_THRESHOLD:
        action = "SELL"
    else:
        action = "HOLD"

    confidence = float(
        1.0 / (1.0 + (_np.exp if _np is not None else math.exp)(-abs(adjusted_score) * 4.0))
    )
    behavior_analysis: BehaviorAnalysis | None = None
    if enable_behavior:
        behavior_analysis = analyze_behavior(
            [ticker],
            behavior_context,
            threshold=behavior_threshold,
            max_influence=max_behavior_influence,
            persist=behavior_persist,  # type: ignore[arg-type]
        )
        confidence *= float(max(0.0, min(1.0, 1.0 + behavior_analysis.confidence_adjustment)))
        logger.info(
            "Behavioural adjustment applied",
            extra={
                "ticker": ticker,
                "score": round(behavior_analysis.behavioral_score, 2),
                "adjustment": round(behavior_analysis.confidence_adjustment, 4),
            },
        )

    factors = {key: float(base_score * weight) for key, weight in weights.items()}
    top_factor = max(factors, key=lambda name: abs(factors[name])) if factors else "trend"
    rationale = (
        "Action decided due to {factor} emphasis under current regime; risk budget {rb:.2f}, "
        "confidence {conf:.2f}."
    ).format(factor=top_factor, rb=risk_budget, conf=confidence)

    logger.info(
        "OroTitan decision generated",
        extra={
            "ticker": ticker,
            "date": str(date),
            "base_score": round(base_score, 4),
            "adjusted_score": round(adjusted_score, 4),
            "action": action,
            "size": round(size, 4),
        },
    )

    return Decision(
        date=date,
        ticker=ticker,
        score=adjusted_score,
        action=action,
        size=size,
        confidence=confidence,
        rationale=rationale,
        factors=factors,
        behavior=behavior_analysis,
    )

