"""Macro regime detection with caching and offline fallbacks."""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import time
from typing import Dict, Iterable, Optional

import pandas as pd

from .data import fetch_price_history

LOGGER = logging.getLogger(__name__)


@dataclass
class RegimeThresholds:
    """Thresholds used for the macro regime classification heuristics."""

    stress_vix: float = 28.0
    credit_stress: float = 2.0
    inflation_cpi: float = 4.0
    high_rates: float = 3.0
    inversion_spread: float = -0.25


@dataclass
class MacroSnapshot:
    """Container for macro indicators used by the regime detector."""

    date: pd.Timestamp
    vix: float | None = None
    cpi_yoy: float | None = None
    rate_10y: float | None = None
    rate_2y: float | None = None
    credit_spread: float | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "date": self.date.isoformat() if hasattr(self.date, "isoformat") else str(self.date),
            "vix": self.vix,
            "cpi_yoy": self.cpi_yoy,
            "rate_10y": self.rate_10y,
            "rate_2y": self.rate_2y,
            "credit_spread": self.credit_spread,
        }


@dataclass
class RegimeAssessment:
    """Result of the macro regime evaluation."""

    regime: str
    snapshot: MacroSnapshot

    def to_dict(self) -> Dict[str, object]:
        payload = self.snapshot.to_dict()
        payload["regime"] = self.regime
        return payload


MACRO_SYMBOLS = {
    "vix": "^VIX",
    "cpi_yoy": "CPIAUCSL",
    "rate_10y": "^TNX",
    "rate_2y": "^IRX",
    "credit_spread": "BAMLH0A0HYM2",
}

FALLBACK_VALUES = {
    "vix": 18.0,
    "cpi_yoy": 3.0,
    "rate_10y": 2.0,
    "rate_2y": 1.5,
    "credit_spread": 1.0,
}

CACHE_TTL_SECONDS = int(os.getenv("MACRO_CACHE_TTL", str(6 * 60 * 60)))
CACHE_DIR = Path(os.getenv("MACRO_CACHE_DIR", ".cache/macro"))


def _fallback_series(symbol: str) -> pd.Series:
    """Return a deterministic fallback series for offline usage."""

    now = datetime.now(timezone.utc)
    try:
        start = pd.Timestamp(now - timedelta(days=364))
    except Exception:  # pragma: no cover - defensive
        start = now - timedelta(days=364)
    index = pd.date_range(start=start, periods=365, freq="D", tz="UTC")
    base = float(FALLBACK_VALUES.get(symbol, 1.0))
    data = [base for _ in index]
    series = pd.Series(data, index=index)
    if hasattr(series, "name"):
        try:
            series.name = symbol
        except Exception:  # pragma: no cover - defensive
            pass
    return series


def _normalise_series(series: pd.Series) -> pd.Series:
    """Normalise a time-series index to UTC and ensure chronological order."""

    index = getattr(series, "index", None)
    if index is not None and hasattr(index, "tz"):
        tzinfo = index.tz
        if tzinfo is None and hasattr(index, "tz_localize"):
            try:
                series.index = index.tz_localize("UTC")
                index = series.index
            except Exception:
                pass
        elif tzinfo is not None and hasattr(index, "tz_convert"):
            try:
                series.index = index.tz_convert("UTC")
                index = series.index
            except Exception:
                pass
    sorter = getattr(series, "sort_index", None)
    if callable(sorter):
        try:
            return sorter()
        except Exception:
            pass
    if index is not None:
        raw_values = getattr(series, "_data", None)
        if raw_values is None:
            getter = getattr(series, "values", None)
            if callable(getter):
                try:
                    raw_values = list(getter())
                except Exception:
                    raw_values = None
        if raw_values is None:
            try:
                raw_values = list(series)  # type: ignore[arg-type]
            except Exception:
                raw_values = []
        pairs = sorted(zip(index, raw_values), key=lambda item: item[0])
        sorted_values = [value for _, value in pairs]
        sorted_index_raw = [label for label, _ in pairs]
        try:
            sorted_index = index.__class__(sorted_index_raw)
        except Exception:
            sorted_index = sorted_index_raw
        normalised = pd.Series(sorted_values, index=sorted_index)
        if hasattr(series, "name") and hasattr(normalised, "name"):
            try:
                normalised.name = series.name  # type: ignore[attr-defined]
            except Exception:
                pass
        return normalised
    return series


def _ensure_utc_timestamp(value: object) -> pd.Timestamp:
    """Return a UTC-normalised pandas Timestamp from any datetime-like input."""

    try:
        ts = pd.Timestamp(value)
    except Exception:  # pragma: no cover - defensive fallback
        ts = pd.Timestamp(datetime.now(timezone.utc))

    if hasattr(ts, "to_pydatetime"):
        dt = ts.to_pydatetime()
    elif isinstance(ts, datetime):
        dt = ts
    else:  # pragma: no cover - extremely rare representations
        dt = datetime.fromisoformat(str(ts))

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return pd.Timestamp(dt)


class MacroDataProvider:
    """Fetch and cache macro time-series used for regime detection."""

    def __init__(self, *, cache_dir: Path | None = None, ttl_seconds: int | None = None) -> None:
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = CACHE_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        self._memory: Dict[str, pd.Series] = {}

    def _cache_path(self, symbol: str) -> Path:
        safe = symbol.replace("/", "_").replace("^", "_")
        return self.cache_dir / f"{safe}.pkl"

    def _load_cached(self, symbol: str) -> pd.Series | None:
        path = self._cache_path(symbol)
        if not path.exists():
            return None
        if self.ttl_seconds and self.ttl_seconds > 0:
            age = time() - path.stat().st_mtime
            if age > self.ttl_seconds:
                return None
        try:
            series = pd.read_pickle(path)
        except Exception:
            return None
        return _normalise_series(series)

    def _store_cached(self, symbol: str, series: pd.Series) -> None:
        path = self._cache_path(symbol)
        try:
            _normalise_series(series).to_pickle(path)
        except Exception:  # pragma: no cover - cache failures should be silent
            LOGGER.debug("Unable to persist macro series", extra={"symbol": symbol})

    def _download_series(self, symbol: str) -> pd.Series:
        ticker = MACRO_SYMBOLS.get(symbol, symbol)
        try:
            frame = fetch_price_history(
                ticker,
                period="5y",
                interval="1d",
                auto_adjust=False,
                use_cache=True,
                cache_ttl_seconds=self.ttl_seconds,
            )
            column = "Close"
            if column not in frame.columns and len(frame.columns):
                column = frame.columns[0]
            series = frame[column].copy()
        except Exception:  # pragma: no cover - runtime/network only
            LOGGER.warning("Macro download failed, using fallback", extra={"symbol": symbol})
            series = _fallback_series(symbol)
        return _normalise_series(series)

    def get_series(self, symbol: str) -> pd.Series:
        if symbol in self._memory:
            return self._memory[symbol]
        cached = self._load_cached(symbol)
        if cached is not None:
            self._memory[symbol] = cached
            return cached
        series = self._download_series(symbol)
        self._memory[symbol] = series
        self._store_cached(symbol, series)
        return series

    def snapshot(self, date: pd.Timestamp) -> MacroSnapshot:
        timestamp = _ensure_utc_timestamp(date)
        values: Dict[str, float | None] = {}
        for key in MACRO_SYMBOLS:
            series = self.get_series(key)
            try:
                value = float(series.asof(timestamp))
            except Exception:
                value = None
            if value is None or (isinstance(value, float) and math.isnan(value)):
                value = float(FALLBACK_VALUES.get(key, float("nan")))
            values[key] = value
        return MacroSnapshot(
            date=timestamp,
            vix=values.get("vix"),
            cpi_yoy=values.get("cpi_yoy"),
            rate_10y=values.get("rate_10y"),
            rate_2y=values.get("rate_2y"),
            credit_spread=values.get("credit_spread"),
        )


def classify_snapshot(snapshot: MacroSnapshot, *, thresholds: RegimeThresholds | None = None) -> str:
    """Classify a macro snapshot into one of the Nexus regimes."""

    thresholds = thresholds or RegimeThresholds()
    spread = None
    if snapshot.rate_10y is not None and snapshot.rate_2y is not None:
        spread = snapshot.rate_10y - snapshot.rate_2y

    if (
        (snapshot.vix is not None and snapshot.vix >= thresholds.stress_vix)
        or (
            snapshot.credit_spread is not None
            and snapshot.credit_spread >= thresholds.credit_stress
        )
    ):
        return "Stress"

    if (
        snapshot.cpi_yoy is not None
        and snapshot.cpi_yoy >= thresholds.inflation_cpi
        and (
            snapshot.rate_10y is None
            or snapshot.rate_10y >= thresholds.high_rates
        )
    ):
        return "Inflation"

    if spread is not None and spread <= thresholds.inversion_spread:
        return "Recovery"

    return "Expansion"


def evaluate_regime(
    date: pd.Timestamp,
    *,
    provider: MacroDataProvider | None = None,
    thresholds: RegimeThresholds | None = None,
) -> RegimeAssessment:
    """Return a :class:`RegimeAssessment` for the requested date."""

    provider = provider or MacroDataProvider()
    snapshot = provider.snapshot(date)
    regime = classify_snapshot(snapshot, thresholds=thresholds)
    return RegimeAssessment(regime=regime, snapshot=snapshot)


def detect_regime(
    date: pd.Timestamp,
    *,
    provider: MacroDataProvider | None = None,
    thresholds: RegimeThresholds | None = None,
) -> str:
    """Return the macro regime label for ``date`` using cached indicators."""

    assessment = evaluate_regime(date, provider=provider, thresholds=thresholds)
    return assessment.regime


def infer_regime_series(
    index: Iterable[pd.Timestamp],
    *,
    provider: MacroDataProvider | None = None,
    thresholds: RegimeThresholds | None = None,
) -> pd.Series:
    """Return a labelled Series matching the provided index."""

    provider = provider or MacroDataProvider()
    labels = [
        classify_snapshot(provider.snapshot(timestamp), thresholds=thresholds)
        for timestamp in index
    ]
    return pd.Series(labels, index=index, name="regime")


__all__ = [
    "MacroDataProvider",
    "MacroSnapshot",
    "RegimeAssessment",
    "RegimeThresholds",
    "classify_snapshot",
    "detect_regime",
    "evaluate_regime",
    "infer_regime_series",
]
