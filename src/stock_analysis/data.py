"""Data acquisition helpers built on top of yfinance."""
from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from typing import Dict, Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from .cache import load_cached_prices, store_cached_prices

LOGGER = logging.getLogger(__name__)

PARIS_TZ = ZoneInfo("Europe/Paris")
UTC_TZ = ZoneInfo("UTC")
CANONICAL_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
STOCK_DL_WORKERS = int(os.getenv("STOCK_DL_WORKERS", "8"))


def _download_history(
    ticker: str,
    *,
    period: str,
    interval: str,
    auto_adjust: bool,
) -> pd.DataFrame:
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

    return history


def finalize_prices(
    frame: pd.DataFrame,
    *,
    ticker: str,
    period: str,
    interval: str,
    auto_adjust: bool,
) -> pd.DataFrame:
    """Normalise OHLCV columns and timezone metadata for ``frame``."""

    cleaned = frame.copy()

    if not isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned.index = pd.to_datetime(cleaned.index, errors="coerce")

    cleaned = cleaned[cleaned.index.notna()]

    if cleaned.index.tz is None:
        cleaned.index = cleaned.index.tz_localize(UTC_TZ)

    cleaned.index = cleaned.index.tz_convert(PARIS_TZ)
    cleaned.sort_index(inplace=True)

    cleaned = _drop_duplicate_indices(cleaned)

    for column in CANONICAL_PRICE_COLUMNS:
        if column not in cleaned.columns:
            LOGGER.warning("Column missing from dataset", extra={"column": column, "ticker": ticker})
            cleaned[column] = pd.Series(
                [float("nan")] * len(cleaned), index=cleaned.index, dtype="float64"
            )

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


def fetch_price_history(
    ticker: str,
    *,
    period: str = "1y",
    interval: str = "1d",
    auto_adjust: bool = False,
    use_cache: bool = True,
    cache_ttl_seconds: int | None = None,
) -> pd.DataFrame:
    """Download historical price data for ``ticker`` using yfinance with caching."""

    if use_cache:
        cached = load_cached_prices(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            ttl_seconds=cache_ttl_seconds,
        )
        if cached is not None:
            LOGGER.info("Cache hit for price history", extra={"ticker": ticker})
            return finalize_prices(
                cached,
                ticker=ticker,
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
            )

    raw = _download_history(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
    )
    cleaned = finalize_prices(
        raw,
        ticker=ticker,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
    )

    if use_cache:
        try:
            store_cached_prices(
                cleaned,
                ticker=ticker,
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
            )
        except Exception:  # pragma: no cover - cache IO errors should not break main flow
            LOGGER.warning("Unable to persist dataset to cache", extra={"ticker": ticker})

    return cleaned


def download_many(
    tickers: Iterable[str],
    *,
    period: str = "1y",
    interval: str = "1d",
    auto_adjust: bool = False,
    use_cache: bool = True,
    cache_ttl_seconds: int | None = None,
) -> Dict[str, pd.DataFrame]:
    """Download multiple tickers concurrently returning a mapping of ticker->DataFrame."""

    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return {}

    LOGGER.info(
        "Downloading tickers concurrently",
        extra={"count": len(tickers), "workers": STOCK_DL_WORKERS},
    )

    results: Dict[str, pd.DataFrame] = {}
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=STOCK_DL_WORKERS) as executor:
        future_map = {
            executor.submit(
                fetch_price_history,
                ticker,
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
                use_cache=use_cache,
                cache_ttl_seconds=cache_ttl_seconds,
            ): ticker
            for ticker in tickers
        }

        for future in as_completed(future_map):
            ticker = future_map[future]
            try:
                frame = future.result()
            except Exception as exc:  # pragma: no cover - runtime only
                LOGGER.error(
                    "Failed to download ticker", extra={"ticker": ticker, "error": str(exc)}
                )
                continue
            results[ticker] = frame

    duration_ms = (perf_counter() - start) * 1000.0
    LOGGER.info("Download batch completed", extra={"duration_ms": round(duration_ms, 2)})
    return results


def align_many(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Align multiple price frames on a shared date index."""

    if not frames:
        return {}

    indices = set()
    for frame in frames.values():
        indices.update(frame.index)

    full_index = sorted(indices)
    aligned: Dict[str, pd.DataFrame] = {}
    for ticker, frame in frames.items():
        aligned_frame = frame.reindex(full_index)
        aligned[ticker] = aligned_frame

    return aligned


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


__all__ = [
    "CANONICAL_PRICE_COLUMNS",
    "STOCK_DL_WORKERS",
    "align_many",
    "download_many",
    "fetch_price_history",
    "finalize_prices",
    "quality_report",
]
