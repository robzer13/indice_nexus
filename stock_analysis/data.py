"""Data acquisition helpers built on top of yfinance."""
from __future__ import annotations

import logging
import math
from typing import Dict
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)

PARIS_TZ = ZoneInfo("Europe/Paris")
UTC_TZ = ZoneInfo("UTC")
CANONICAL_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def fetch_price_history(
    ticker: str,
    *,
    period: str = "1y",
    interval: str = "1d",
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Download historical price data for ``ticker`` using yfinance.

    The resulting frame is guaranteed to expose the canonical OHLCV columns, uses a
    timezone-aware ``DatetimeIndex`` in Europe/Paris, is sorted chronologically, and
    carries metadata in ``DataFrame.attrs`` describing the request.
    """

    LOGGER.info(
        "Downloading historical data",
        extra={"ticker": ticker, "period": period, "interval": interval},
    )

    try:
        history = yf.Ticker(ticker).history(
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
        )
    except Exception as exc:  # pragma: no cover - network errors are runtime only
        LOGGER.error("Failed to download data", exc_info=exc, extra={"ticker": ticker})
        raise

    if history is None or getattr(history, "empty", True):
        raise ValueError(
            f"No historical data returned for {ticker!r} with period={period!r} interval={interval!r}."
        )

    # Ensure we work on a copy because we mutate shape and metadata.
    cleaned = history.copy()

    if not isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")

    # Drop rows that failed to coerce to datetime before timezone handling.
    cleaned = cleaned[cleaned.index.notna()]

    if cleaned.index.tz is None:
        cleaned.index = cleaned.index.tz_localize(UTC_TZ)

    cleaned.index = cleaned.index.tz_convert(PARIS_TZ)
    cleaned.sort_index(inplace=True)

    # Remove duplicate timestamps keeping the last observation.
    cleaned = _drop_duplicate_indices(cleaned)

    for column in CANONICAL_PRICE_COLUMNS:
        if column not in cleaned.columns:
            LOGGER.warning("Column missing from dataset", extra={"column": column, "ticker": ticker})
            cleaned[column] = pd.Series(
                [float("nan")] * len(cleaned), index=cleaned.index, dtype="float64"
            )

    # Drop rows that are entirely missing across our canonical fields.
    cleaned.dropna(axis=0, how="all", subset=CANONICAL_PRICE_COLUMNS, inplace=True)

    if cleaned.empty:
        raise ValueError(
            f"Historical data for {ticker!r} contains no valid rows after cleaning with period={period!r} interval={interval!r}."
        )

    cleaned.attrs = {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "auto_adjust": auto_adjust,
        "source": "yfinance",
    }

    return cleaned


def _drop_duplicate_indices(frame: pd.DataFrame) -> pd.DataFrame:
    index = list(frame.index)
    last_positions: Dict[object, int] = {}
    for position, label in enumerate(index):
        last_positions[label] = position

    keep_mask = [pos == last_positions[label] for pos, label in enumerate(index)]
    duplicate_count = len(index) - sum(1 for keep in keep_mask if keep)

    if duplicate_count:
        LOGGER.warning(
            "Duplicate timestamps detected and dropped",
            extra={"removed": duplicate_count},
        )
        frame = frame[keep_mask]

    return frame


def _row_is_all_na(frame: pd.DataFrame, position: int) -> bool:
    for column in frame.columns:
        value = frame[column].iloc[position]
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        return False
    return True


def quality_report(
    frame: pd.DataFrame,
    *,
    price_column: str = "Close",
    gap_threshold_pct: float = 5.0,
) -> Dict[str, object]:
    """Analyse data quality issues on ``frame`` and return aggregated metrics."""

    if price_column not in frame.columns:
        raise KeyError(f"Column {price_column!r} not present in DataFrame")

    working = frame.copy(deep=True)
    working.sort_index(inplace=True)

    na_rows_all = sum(_row_is_all_na(working, idx) for idx in range(len(working.index)))
    if na_rows_all:
        LOGGER.info("Dropping fully empty rows", extra={"count": na_rows_all})
        keep_mask = [not _row_is_all_na(working, idx) for idx in range(len(working.index))]
        working = working[keep_mask]

    timezone_status = "other"
    if isinstance(working.index, pd.DatetimeIndex):
        tzinfo = working.index.tz
        if tzinfo is None:
            LOGGER.info("Localising naive index to UTC prior to Paris conversion")
            working.index = working.index.tz_localize(UTC_TZ)
            tzinfo = working.index.tz
        if tzinfo is not None and tzinfo.key != PARIS_TZ.key:  # type: ignore[attr-defined]
            LOGGER.info("Converting index timezone", extra={"from": tzinfo, "to": PARIS_TZ})
            working.index = working.index.tz_convert(PARIS_TZ)
        if working.index.tz and getattr(working.index.tz, "key", None) == PARIS_TZ.key:
            timezone_status = "Europe/Paris"
    else:
        LOGGER.warning("Quality report expects a DatetimeIndex", extra={"type": type(working.index)})

    before_dedup = len(working.index)
    working = _drop_duplicate_indices(working)
    duplicates_removed = before_dedup - len(working.index)

    ohlc_anomalies = 0
    if {"High", "Low", "Open", price_column}.issubset(set(working.columns)):
        highs = working["High"].tolist()
        lows = working["Low"].tolist()
        opens = working["Open"].tolist()
        closes = working[price_column].tolist()
        for high, low, open_value, close_value in zip(highs, lows, opens, closes):
            if _is_invalid_ohlc(open_value, close_value, high, low):
                ohlc_anomalies += 1

    gaps_count = 0
    if {"Open", price_column}.issubset(set(working.columns)):
        opens = working["Open"].tolist()
        closes = working[price_column].tolist()
        previous_close: float | None = None
        for open_value, close_value in zip(opens, closes):
            if previous_close is not None and not _is_missing(open_value) and not _is_missing(previous_close):
                prev = _ensure_float(previous_close)
                current_open = _ensure_float(open_value)
                if prev != 0:
                    pct_change = abs((current_open - prev) / prev) * 100.0
                    if pct_change > gap_threshold_pct:
                        gaps_count += 1
            if not _is_missing(close_value):
                previous_close = _ensure_float(close_value)
            else:
                previous_close = None

    report = {
        "duplicates": {"count": duplicates_removed},
        "ohlc_anomalies": {"count": ohlc_anomalies},
        "gaps": {"count": gaps_count, "threshold_pct": float(gap_threshold_pct)},
        "na_rows_all": {"count": na_rows_all},
        "timezone": timezone_status,
    }

    return report


def _is_invalid_ohlc(open_value: object, close_value: object, high_value: object, low_value: object) -> bool:
    values = []
    for item in (open_value, close_value, high_value, low_value):
        if _is_missing(item):
            return False
        values.append(_ensure_float(item))

    open_numeric, close_numeric, high_numeric, low_numeric = values
    if high_numeric < low_numeric:
        return True
    return not (low_numeric <= open_numeric <= high_numeric and low_numeric <= close_numeric <= high_numeric)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _ensure_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Value {value!r} cannot be converted to float") from exc


__all__ = ["CANONICAL_PRICE_COLUMNS", "fetch_price_history", "quality_report"]
