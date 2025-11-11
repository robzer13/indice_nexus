"""Scoring utilities combining technical and fundamental metrics."""
from __future__ import annotations

import logging
import math
from typing import Dict, Iterable, List, Mapping

import pandas as pd

from .indicators import compute_macd, compute_moving_averages, compute_rsi

LOGGER = logging.getLogger(__name__)

MAX_COMPONENTS = {"trend": 40.0, "momentum": 30.0, "quality": 20.0, "risk": 10.0}
DEFAULT_WEIGHTS = MAX_COMPONENTS.copy()


def _coerce_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def compute_volatility(
    frame: pd.DataFrame,
    *,
    price_column: str = "Close",
    window: int = 20,
) -> pd.DataFrame:
    """Return a copy of ``frame`` augmented with simple return volatility."""

    if window <= 0:
        raise ValueError("Volatility window must be a positive integer")
    if price_column not in frame.columns:
        raise KeyError(f"Column {price_column!r} not present in DataFrame")

    prices = frame.copy(deep=True)
    returns = prices[price_column].pct_change()
    prices["RET"] = returns
    prices[f"VOL{window}"] = returns.rolling(window).std()
    return prices


def _latest_value(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    series = frame[column]
    if getattr(series, "empty", True):
        return None
    try:
        value = series.iloc[-1]
    except Exception:
        return None
    return _coerce_float(value)


def score_trend(frame: pd.DataFrame, *, price_column: str = "Close") -> float:
    """Score the long-term trend alignment of the latest data point."""

    if getattr(frame, "empty", True):
        return 0.0

    components: List[float] = []

    close_value = _latest_value(frame, price_column)
    sma20 = _latest_value(frame, "SMA20")
    sma50 = _latest_value(frame, "SMA50")
    sma200 = _latest_value(frame, "SMA200")
    ema21 = _latest_value(frame, "EMA21")

    if (
        close_value is not None
        and sma20 is not None
        and sma50 is not None
        and sma200 is not None
        and close_value > sma20 > sma50 > sma200
    ):
        components.append(16.0)
    else:
        components.append(0.0)

    components.append(8.0 if close_value is not None and sma200 is not None and close_value > sma200 else 0.0)
    components.append(8.0 if close_value is not None and sma50 is not None and close_value > sma50 else 0.0)
    components.append(8.0 if close_value is not None and ema21 is not None and close_value > ema21 else 0.0)

    return float(sum(components))


def _map_linear(value: float, *, start: float, end: float, start_score: float, end_score: float) -> float:
    if value <= start:
        return start_score
    if value >= end:
        return end_score
    span = end - start
    if span == 0:
        return end_score
    slope = (end_score - start_score) / span
    return start_score + slope * (value - start)


def score_momentum(frame: pd.DataFrame, *, price_column: str = "Close") -> float:
    """Score the momentum of the latest data point using RSI and MACD."""

    if getattr(frame, "empty", True):
        return 0.0

    rsi_raw = _latest_value(frame, "RSI14")
    macd = _latest_value(frame, "MACD")
    macd_signal = _latest_value(frame, "MACD_signal")
    macd_hist = _latest_value(frame, "MACD_hist")

    rsi_score = 0.0
    if rsi_raw is not None:
        if rsi_raw <= 30.0:
            rsi_score = 0.0
        elif rsi_raw >= 70.0:
            rsi_score = 20.0
        elif rsi_raw <= 50.0:
            rsi_score = _map_linear(rsi_raw, start=30.0, end=50.0, start_score=0.0, end_score=10.0)
        else:
            rsi_score = _map_linear(rsi_raw, start=50.0, end=70.0, start_score=10.0, end_score=20.0)

    macd_score = 0.0
    if macd_hist is not None and macd_hist > 0:
        macd_score += 5.0
    if (
        macd is not None
        and macd_signal is not None
        and macd > macd_signal
    ):
        macd_score += 5.0

    return float(min(30.0, rsi_score + macd_score))


def _quality_component(fundamentals: Dict[str, object]) -> tuple[float, List[str]]:
    score = 0.0
    notes: List[str] = []

    pe_ratio = fundamentals.get("pe_ratio") if isinstance(fundamentals, dict) else None
    pe_value = _coerce_float(pe_ratio)
    if pe_value is None:
        notes.append("no fundamentals.pe_ratio")
    elif 10.0 <= pe_value <= 30.0:
        score += 6.0
    elif 5.0 <= pe_value < 10.0 or 30.0 < pe_value <= 40.0:
        score += 3.0

    margin = fundamentals.get("net_margin_pct") if isinstance(fundamentals, dict) else None
    margin_value = _coerce_float(margin)
    if margin_value is None:
        notes.append("no fundamentals.net_margin_pct")
    elif margin_value >= 10.0:
        score += 6.0
    elif 3.0 <= margin_value < 10.0:
        score += 3.0

    debt = fundamentals.get("debt_to_equity") if isinstance(fundamentals, dict) else None
    debt_value = _coerce_float(debt)
    if debt_value is None:
        notes.append("no fundamentals.debt_to_equity")
    elif 0.0 <= debt_value <= 1.5:
        score += 4.0
    elif 1.5 < debt_value <= 2.5:
        score += 2.0

    dividend = fundamentals.get("dividend_yield_pct") if isinstance(fundamentals, dict) else None
    dividend_value = _coerce_float(dividend)
    if dividend_value is None:
        notes.append("no fundamentals.dividend_yield_pct")
    elif 1.0 <= dividend_value <= 6.0:
        score += 4.0
    elif 0.0 <= dividend_value < 1.0 or 6.0 < dividend_value <= 8.0:
        score += 2.0

    return score, notes


def score_quality(fundamentals: Dict[str, object]) -> float:
    """Return the quality score derived from the fundamentals dictionary."""

    score, _ = _quality_component(fundamentals)
    return float(min(20.0, max(0.0, score)))


def score_risk(frame: pd.DataFrame) -> float:
    """Score the volatility risk using the most recent ``VOL20`` value."""

    if getattr(frame, "empty", True):
        return 0.0

    column = "VOL20"
    if column not in frame.columns:
        return 0.0

    latest = _latest_value(frame, column)
    if latest is None:
        return 0.0

    if latest <= 0.01:
        return 10.0
    if latest <= 0.02:
        return 7.0
    if latest <= 0.03:
        return 4.0
    if latest <= 0.05:
        return 2.0
    return 0.0


def _ensure_indicators(
    frame: pd.DataFrame,
    *,
    price_column: str,
) -> pd.DataFrame:
    """Ensure trend/momentum columns are present, recomputing if required."""

    working = frame
    try:
        missing_ma = any(
            column not in working.columns
            for column in ["EMA21", "SMA20", "SMA50", "SMA200"]
        )
    except AttributeError:
        missing_ma = True

    if missing_ma:
        try:
            working = compute_moving_averages(working, price_column=price_column)
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.warning("Unable to compute moving averages", exc_info=exc)

    if f"RSI14" not in working.columns:
        try:
            working = compute_rsi(working, price_column=price_column, period=14)
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.warning("Unable to compute RSI", exc_info=exc)

    macd_columns = {"MACD", "MACD_signal", "MACD_hist"}
    if not macd_columns.issubset(set(working.columns)):
        try:
            working = compute_macd(working, price_column=price_column)
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.warning("Unable to compute MACD", exc_info=exc)

    if "VOL20" not in working.columns:
        try:
            working = compute_volatility(working, price_column=price_column, window=20)
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.warning("Unable to compute volatility", exc_info=exc)

    return working


def compute_score_bundle(
    frame: pd.DataFrame,
    fundamentals: Dict[str, object] | None,
    *,
    price_column: str = "Close",
    weights: Mapping[str, float] | None = None,
) -> Dict[str, object]:
    """Compute the composite scoring bundle for the provided ticker data."""

    notes: List[str] = []
    fundamentals = fundamentals or {}

    required_columns: Dict[str, Iterable[str]] = {
        "trend": [price_column, "SMA20", "SMA50", "SMA200", "EMA21"],
        "momentum": ["RSI14", "MACD", "MACD_signal", "MACD_hist"],
        "risk": ["VOL20"],
    }

    if frame is None or getattr(frame, "empty", True):
        notes.append("missing: price data")
        working = pd.DataFrame()
        as_of = ""
    else:
        working = frame.copy(deep=True)
        if price_column not in working.columns:
            notes.append(f"missing: {price_column}")
        for columns in required_columns.values():
            for column in columns:
                if column not in working.columns:
                    notes.append(f"missing: {column}")
        working = _ensure_indicators(working, price_column=price_column)
        as_of_raw = working.index[-1]
        if hasattr(as_of_raw, "isoformat"):
            as_of = as_of_raw.isoformat()
        else:
            as_of = str(as_of_raw)

    trend_score = score_trend(working, price_column=price_column)
    momentum_score = score_momentum(working, price_column=price_column)
    risk_score = score_risk(working)
    quality_score, quality_notes = _quality_component(fundamentals)
    quality_score = float(min(20.0, max(0.0, quality_score)))
    notes.extend(quality_notes)

    for label, columns in required_columns.items():
        for column in columns:
            if column not in working.columns:
                notes.append(f"missing: {column}")
                continue
            value = _coerce_float(working[column].iloc[-1])
            if value is None:
                notes.append(f"nan: {column}")

    component_map = {
        "trend": trend_score,
        "momentum": momentum_score,
        "quality": quality_score,
        "risk": risk_score,
    }

    weights_map = {**DEFAULT_WEIGHTS}
    if weights:
        for key, value in weights.items():
            if key in weights_map:
                weights_map[key] = float(value)

    weighted_total = 0.0
    for key, raw_value in component_map.items():
        max_value = MAX_COMPONENTS.get(key, 1.0)
        weight = weights_map.get(key, 0.0)
        normalised = (raw_value / max_value) if max_value else 0.0
        weighted_total += normalised * weight

    weight_sum = sum(weights_map.values())
    total = (weighted_total / weight_sum) * 100.0 if weight_sum else 0.0
    total = max(0.0, min(100.0, round(total, 2)))

    return {
        "score": total,
        "trend": round(trend_score, 2),
        "momentum": round(momentum_score, 2),
        "quality": round(quality_score, 2),
        "risk": round(risk_score, 2),
        "as_of": as_of,
        "notes": sorted(set(notes)),
    }


__all__ = [
    "compute_score_bundle",
    "compute_volatility",
    "score_momentum",
    "score_quality",
    "score_risk",
    "score_trend",
]
