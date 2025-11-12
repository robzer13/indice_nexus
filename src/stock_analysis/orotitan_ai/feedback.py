"""Feedback handling and lightweight state persistence for OroTitan AI.""" 
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_STATE_PATH = Path("out/state/orotitan_ai.json")
_DEFAULT_MODULES = ("trend", "momentum", "fundamental", "behavioral")


@dataclass(slots=True)
class FeedbackItem:
    """Individual feedback entry with realised outcome information."""

    date: pd.Timestamp
    ticker: str
    pnl: float
    horizon_days: int
    user_note: str | None = None


def _initial_weights() -> Dict[str, float]:
    return {module: 1.0 / len(_DEFAULT_MODULES) for module in _DEFAULT_MODULES}


def _normalise(weights: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(max(value, 0.0) for value in weights.values()))
    if total <= 0.0:
        return _initial_weights()
    return {key: max(value, 0.0) / total for key, value in weights.items()}


def update_weights(
    current: Dict[str, float],
    feedback: Iterable[FeedbackItem],
    lr: float = 0.05,
    decay: float = 0.99,
) -> Dict[str, float]:
    """Update module weights using a simple reward-based adaptive scheme."""
    weights = {module: current.get(module, 0.25) for module in _DEFAULT_MODULES}
    for item in feedback:
        signal = math.tanh(item.pnl)
        if signal >= 0:
            focus = {"trend", "momentum"}
        else:
            focus = {"fundamental", "behavioral"}
        for module in weights:
            weights[module] *= decay
            if module in focus:
                weights[module] += lr * abs(signal)
            else:
                weights[module] += lr * (0.5 * abs(signal))
    return _normalise(weights)


def apply_feedback(
    current: Dict[str, float],
    feedback: Iterable[FeedbackItem],
    *,
    lr: float = 0.05,
    decay: float = 0.99,
) -> Dict[str, float]:
    """Wrapper exposed for external modules to adapt weights."""

    return update_weights(current, feedback, lr=lr, decay=decay)


def load_state(path: Path | None = None) -> Dict[str, Any]:
    """Load persisted OroTitan AI state if it exists."""
    target = path or _DEFAULT_STATE_PATH
    if not target.exists():
        logger.info("No OroTitan state file found at %s", target)
        return {"weights": _initial_weights(), "meta": {"version": "v5", "notes": "new"}}
    with target.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    weights = data.get("weights", _initial_weights())
    return {"weights": _normalise(weights), "meta": data.get("meta", {}), "as_of": data.get("as_of")}


def save_state(state: Dict[str, Any], path: Path | None = None) -> Path:
    """Persist updated OroTitan AI state atomically."""
    target = path or _DEFAULT_STATE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(".tmp")
    payload = json.dumps(state, indent=2, sort_keys=True)
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
    tmp_path.replace(target)
    logger.info("OroTitan state saved", extra={"path": str(target)})
    return target

