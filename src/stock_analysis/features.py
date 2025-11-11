"""Feature engineering utilities for the machine-learning pipeline."""
from __future__ import annotations

import math
import pandas as pd

from .indicators import compute_rsi


def _select_price_column(frame: pd.DataFrame, fallback: str = "Close") -> str:
    if "Adj Close" in frame.columns:
        return "Adj Close"
    if fallback in frame.columns:
        return fallback
    raise KeyError(f"Required price column '{fallback}' absent from dataframe")


def add_ta_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Enrich ``frame`` with technical factors used by the ML pipeline."""

    if frame is None or getattr(frame, "empty", True):
        raise ValueError("Input dataframe must not be empty")

    price_column = _select_price_column(frame)
    working = frame.copy()

    price = working[price_column]
    features = pd.DataFrame(index=working.index)

    ret_1d = price.pct_change()
    features["ret_1d"] = ret_1d

    vol_20 = ret_1d.rolling(window=20, min_periods=20).std(ddof=0) * math.sqrt(252.0)
    features["vol_20"] = vol_20

    if "SMA200" in working.columns:
        sma200 = working["SMA200"]
    else:
        sma200 = price.rolling(window=200, min_periods=200).mean()
    features["sma200_gap"] = price.divide(sma200).subtract(1.0)

    enriched = compute_rsi(working, price_column=price_column, period=14)
    rsi_column = "RSI14"
    features["rsi_14"] = enriched.get(rsi_column)

    features["mom_21"] = price.pct_change(21)

    return features


def make_label_future_ret(frame: pd.DataFrame, horizon: int = 5, thr: float = 0.0) -> pd.Series:
    """Return binary labels indicating whether the future return exceeds ``thr``."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    price_column = _select_price_column(frame)
    price = frame[price_column]

    future = price.shift(-horizon)
    future_ret = future.divide(price).subtract(1.0)

    labels = pd.Series(float("nan"), index=price.index, dtype="float64")
    valid_mask = future_ret.notna()
    labels.loc[valid_mask] = (future_ret.loc[valid_mask] > thr).astype(float)
    labels.name = f"future_ret_{horizon}d"
    return labels


__all__ = ["add_ta_features", "make_label_future_ret"]
