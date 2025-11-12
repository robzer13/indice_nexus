"""Local on-disk caching helpers for price downloads."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(os.getenv("STOCK_CACHE_DIR", ".cache"))
DEFAULT_TTL_SECONDS = int(os.getenv("STOCK_CACHE_TTL", "3600"))


def _ensure_cache_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def _build_cache_path(
    ticker: str,
    *,
    period: str,
    interval: str,
    auto_adjust: bool,
    directory: Optional[Path] = None,
) -> Path:
    safe_ticker = ticker.replace("/", "_").replace("\\", "_")
    suffix = "adj" if auto_adjust else "raw"
    directory = directory or DEFAULT_CACHE_DIR
    _ensure_cache_dir(directory)
    filename = f"{safe_ticker}_{period}_{interval}_{suffix}.parquet"
    return directory / filename


def load_cached_prices(
    ticker: str,
    *,
    period: str,
    interval: str,
    auto_adjust: bool = False,
    ttl_seconds: Optional[int] = None,
    directory: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    ttl_seconds = DEFAULT_TTL_SECONDS if ttl_seconds is None else ttl_seconds
    cache_path = _build_cache_path(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        directory=directory,
    )

    if not cache_path.exists():
        return None

    age = time.time() - cache_path.stat().st_mtime
    if age > ttl_seconds:
        LOGGER.info(
            "Cached dataset expired",
            extra={"ticker": ticker, "period": period, "interval": interval, "age_s": age},
        )
        return None

    try:
        frame = pd.read_parquet(cache_path)
    except Exception as exc:  # pragma: no cover - IO errors are runtime specific
        LOGGER.warning("Failed to read cached dataset", exc_info=exc, extra={"path": str(cache_path)})
        return None

    frame.attrs.setdefault("cache_path", str(cache_path))
    return frame


def store_cached_prices(
    frame: pd.DataFrame,
    *,
    ticker: str,
    period: str,
    interval: str,
    auto_adjust: bool = False,
    directory: Optional[Path] = None,
) -> Path:
    cache_path = _build_cache_path(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        directory=directory,
    )
    try:
        frame.to_parquet(cache_path, engine="pyarrow")
    except Exception as exc:  # pragma: no cover - optional dependency at runtime
        LOGGER.error("Failed to write cache file", exc_info=exc, extra={"path": str(cache_path)})
        raise

    LOGGER.info(
        "Cached dataset written",
        extra={"ticker": ticker, "period": period, "interval": interval, "path": str(cache_path)},
    )
    return cache_path


__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_TTL_SECONDS",
    "load_cached_prices",
    "store_cached_prices",
]
