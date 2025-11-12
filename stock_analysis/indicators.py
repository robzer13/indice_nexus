"""Pure-Python technical indicator implementations."""
from __future__ import annotations

import logging
import math
from typing import Iterable, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _ensure_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Value {value!r} cannot be converted to float") from exc


def _calculate_ema(values: Sequence[object], window: int, *, alpha: float | None = None) -> list[float]:
    if window <= 0:
        raise ValueError("EMA window must be a positive integer")

    if alpha is None:
        alpha = 2.0 / (window + 1)

    ema_value: float | None = None
    seen = 0
    results: list[float] = []

    for raw in values:
        if _is_missing(raw):
            results.append(math.nan)
            continue

        seen += 1
        numeric = _ensure_float(raw)
        if ema_value is None:
            ema_value = numeric
        else:
            ema_value = (numeric * alpha) + (ema_value * (1.0 - alpha))

        if seen < window:
            results.append(math.nan)
        else:
            results.append(ema_value)

    return results


def _calculate_wilder_rma(values: Sequence[object], window: int) -> list[float]:
    """Return Wilder's smoothed moving average for the provided ``values``."""

    if window <= 0:
        raise ValueError("Wilder window must be a positive integer")

    results: list[float] = []
    buffer: list[float] = []
    previous: float | None = None

    for raw in values:
        if _is_missing(raw):
            results.append(math.nan)
            continue

        numeric = _ensure_float(raw)
        buffer.append(numeric)

        if previous is None:
            if len(buffer) < window:
                results.append(math.nan)
                continue
            initial = sum(buffer[-window:]) / window
            previous = initial
            results.append(initial)
        else:
            previous = ((previous * (window - 1)) + numeric) / window
            results.append(previous)

    return results


def _calculate_sma(values: Sequence[object], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("SMA window must be a positive integer")

    results: list[float] = []
    buffer: list[float] = []

    for raw in values:
        if _is_missing(raw):
            results.append(math.nan)
            buffer.clear()
            continue

        numeric = _ensure_float(raw)
        buffer.append(numeric)
        if len(buffer) < window:
            results.append(math.nan)
            continue

        if len(buffer) > window:
            buffer.pop(0)

        results.append(sum(buffer) / window)

    return results


def compute_moving_averages(
    frame: pd.DataFrame,
    *,
    price_column: str = "Close",
    windows_ema: Iterable[int] = (7, 9, 20, 21),
    windows_sma: Iterable[int] = (20, 50, 100, 200),
) -> pd.DataFrame:
    """Return a new DataFrame with EMA/SMA columns computed from ``price_column``."""

    if price_column not in frame.columns:
        raise KeyError(f"Column {price_column!r} not present in DataFrame")

    prices = frame.copy(deep=True)
    series = list(prices[price_column].tolist())

    for window in windows_ema:
        column_name = f"EMA{window}"
        LOGGER.info("Computing EMA", extra={"window": window, "price_column": price_column})
        prices[column_name] = _calculate_ema(series, int(window))

    for window in windows_sma:
        column_name = f"SMA{window}"
        LOGGER.info("Computing SMA", extra={"window": window, "price_column": price_column})
        prices[column_name] = _calculate_sma(series, int(window))

    return prices


def compute_rsi(
    frame: pd.DataFrame,
    *,
    price_column: str = "Close",
    period: int = 14,
    method: str = "wilder",
) -> pd.DataFrame:
    """Return a copy of ``frame`` augmented with a Relative Strength Index column."""

    if method.lower() != "wilder":
        raise ValueError("Only Wilder RSI method is supported")

    if price_column not in frame.columns:
        raise KeyError(f"Column {price_column!r} not present in DataFrame")

    if period <= 0:
        raise ValueError("RSI period must be a positive integer")

    LOGGER.info(
        "Computing RSI",
        extra={"price_column": price_column, "period": period, "method": method},
    )

    prices = frame.copy(deep=True)
    values = list(prices[price_column].tolist())

    deltas: list[float] = [math.nan]
    for index in range(1, len(values)):
        current = values[index]
        previous = values[index - 1]

        if _is_missing(current) or _is_missing(previous):
            deltas.append(math.nan)
            continue

        deltas.append(_ensure_float(current) - _ensure_float(previous))

    gains: list[float] = []
    losses: list[float] = []
    for delta in deltas:
        if _is_missing(delta):
            gains.append(math.nan)
            losses.append(math.nan)
        elif delta > 0:
            gains.append(delta)
            losses.append(0.0)
        elif delta < 0:
            gains.append(0.0)
            losses.append(abs(delta))
        else:
            gains.append(0.0)
            losses.append(0.0)

    avg_gains = _calculate_wilder_rma(gains, period)
    avg_losses = _calculate_wilder_rma(losses, period)

    rsi_values: list[float] = []
    for gain, loss in zip(avg_gains, avg_losses):
        if _is_missing(gain) or _is_missing(loss):
            rsi_values.append(math.nan)
            continue

        gain_val = _ensure_float(gain)
        loss_val = _ensure_float(loss)

        if loss_val == 0 and gain_val == 0:
            rsi_values.append(50.0)
            continue
        if loss_val == 0:
            rsi_values.append(100.0)
            continue
        if gain_val == 0:
            rsi_values.append(0.0)
            continue

        rs = gain_val / loss_val
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi_values.append(rsi)

    column_name = f"RSI{period}"
    prices[column_name] = rsi_values
    return prices


def compute_macd(
    frame: pd.DataFrame,
    *,
    price_column: str = "Close",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Return a copy of ``frame`` augmented with MACD related columns."""

    if price_column not in frame.columns:
        raise KeyError(f"Column {price_column!r} not present in DataFrame")

    if not all(value > 0 for value in (fast, slow, signal)):
        raise ValueError("MACD periods must be positive integers")

    LOGGER.info(
        "Computing MACD",
        extra={"price_column": price_column, "fast": fast, "slow": slow, "signal": signal},
    )

    prices = frame.copy(deep=True)
    values = list(prices[price_column].tolist())

    fast_column = f"EMA{fast}"
    slow_column = f"EMA{slow}"

    if fast_column in prices.columns:
        ema_fast = list(prices[fast_column].tolist())
    else:
        ema_fast = _calculate_ema(values, int(fast))
        prices[fast_column] = ema_fast

    if slow_column in prices.columns:
        ema_slow = list(prices[slow_column].tolist())
    else:
        ema_slow = _calculate_ema(values, int(slow))
        prices[slow_column] = ema_slow

    macd_values: list[float] = []
    for fast_value, slow_value in zip(ema_fast, ema_slow):
        if _is_missing(fast_value) or _is_missing(slow_value):
            macd_values.append(math.nan)
        else:
            macd_values.append(_ensure_float(fast_value) - _ensure_float(slow_value))

    prices["MACD"] = macd_values

    signal_values = _calculate_ema(macd_values, int(signal))
    prices["MACD_signal"] = signal_values

    histogram: list[float] = []
    for macd_value, signal_value in zip(macd_values, signal_values):
        if _is_missing(macd_value) or _is_missing(signal_value):
            histogram.append(math.nan)
        else:
            histogram.append(_ensure_float(macd_value) - _ensure_float(signal_value))

    prices["MACD_hist"] = histogram

    return prices


__all__ = ["compute_moving_averages", "compute_rsi", "compute_macd"]
