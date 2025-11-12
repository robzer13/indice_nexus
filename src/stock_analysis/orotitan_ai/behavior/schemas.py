"""Typed payloads used by the behavioural layer."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover
    from ...tests.pydantic_stub import BaseModel, Field  # type: ignore


class BehaviorIndicator(BaseModel):
    name: str
    value: float
    window: Optional[int] = None
    note: Optional[str] = None


class BiasSignal(BaseModel):
    bias_id: str
    score: float
    indicators: List[BehaviorIndicator]
    rationale: str


class BehaviorAnalysis(BaseModel):
    behavioral_score: float
    top_biases: List[BiasSignal]
    recommendations: List[str]
    confidence_adjustment: float
    tags: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BehaviorPersistRecord(BaseModel):
    timestamp: datetime
    tickers: List[str]
    analysis: BehaviorAnalysis
    context: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "BehaviorIndicator",
    "BiasSignal",
    "BehaviorAnalysis",
    "BehaviorPersistRecord",
]
