"""High level orchestration for the analysis workflow."""
from __future__ import annotations

import logging
from time import perf_counter
from typing import Dict, Iterable, Mapping

from .data import align_many, download_many, fetch_price_history, quality_report
from .fundamentals import fetch_fundamentals
from .indicators import compute_macd, compute_moving_averages, compute_rsi
from .scoring import compute_score_bundle, compute_volatility

LOGGER = logging.getLogger(__name__)

DEFAULT_TICKERS = ["ASML.AS", "TTE.PA", "MC.PA"]


def analyze_tickers(
    tickers: Iterable[str],
    *,
    period: str = "2y",
    interval: str = "1d",
    auto_adjust: bool = False,
    price_column: str = "Close",
    gap_threshold_pct: float = 5.0,
    use_cache: bool = True,
    cache_ttl_seconds: int | None = None,
    score_weights: Mapping[str, float] | None = None,
    regime: str | None = None,
) -> Dict[str, Dict[str, object]]:
    """Run the complete analysis pipeline for the provided tickers."""

    results: Dict[str, Dict[str, object]] = {}

    ticker_list = list(dict.fromkeys(tickers))
    price_frames = download_many(
        ticker_list,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        use_cache=use_cache,
        cache_ttl_seconds=cache_ttl_seconds,
    )
    price_frames = align_many(price_frames)

    for ticker in ticker_list:
        LOGGER.info(
            "Analysing ticker",
            extra={"ticker": ticker, "period": period, "interval": interval},
        )
        try:
            fetch_start = perf_counter()
            prices = price_frames.get(ticker)
            if prices is None:
                prices = fetch_price_history(
                    ticker,
                    period=period,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    use_cache=use_cache,
                    cache_ttl_seconds=cache_ttl_seconds,
                )
            fetch_duration = (perf_counter() - fetch_start) * 1000.0
            LOGGER.info(
                "Downloaded price history",
                extra={
                    "ticker": ticker,
                    "rows": len(prices),
                    "duration_ms": round(fetch_duration, 2),
                },
            )

            ma_start = perf_counter()
            enriched = compute_moving_averages(prices, price_column=price_column)
            enriched = compute_rsi(enriched, price_column=price_column)
            enriched = compute_macd(enriched, price_column=price_column)
            enriched = compute_volatility(enriched, price_column=price_column, window=20)
            ma_duration = (perf_counter() - ma_start) * 1000.0
            LOGGER.info(
                "Computed indicators",
                extra={"ticker": ticker, "duration_ms": round(ma_duration, 2)},
            )

            fundamentals_start = perf_counter()
            fundamentals = fetch_fundamentals(ticker)
            fundamentals_duration = (perf_counter() - fundamentals_start) * 1000.0
            LOGGER.info(
                "Fetched fundamentals",
                extra={"ticker": ticker, "duration_ms": round(fundamentals_duration, 2)},
            )

            quality_start = perf_counter()
            quality = quality_report(
                enriched,
                price_column=price_column,
                gap_threshold_pct=gap_threshold_pct,
            )
            quality_duration = (perf_counter() - quality_start) * 1000.0
            LOGGER.info(
                "Generated quality report",
                extra={"ticker": ticker, "duration_ms": round(quality_duration, 2)},
            )

            score_start = perf_counter()
            score_bundle = compute_score_bundle(
                enriched,
                fundamentals,
                price_column=price_column,
                weights=score_weights,
                regime=regime,
            )
            score_duration = (perf_counter() - score_start) * 1000.0
            LOGGER.info(
                "Computed scoring bundle",
                extra={"ticker": ticker, "duration_ms": round(score_duration, 2)},
            )
        except Exception as exc:  # pragma: no cover - runtime/network failures
            LOGGER.error(
                "Analysis failed for ticker",
                extra={"ticker": ticker},
                exc_info=exc,
            )
            continue

        results[ticker] = {
            "prices": enriched,
            "fundamentals": fundamentals,
            "quality": quality,
            "score": score_bundle,
            "meta": {
                "ticker": ticker,
                "period": period,
                "interval": interval,
                "price_column": price_column,
                "gap_threshold_pct": gap_threshold_pct,
                "regime": regime,
            },
        }

    return results


def fetch_benchmark(
    symbol: str = "^FCHI",
    *,
    period: str = "2y",
    interval: str = "1d",
    price_column: str = "Close",
    auto_adjust: bool = False,
) -> object:
    """Fetch a benchmark price series aligned with the standard data contract."""

    frame = fetch_price_history(symbol, period=period, interval=interval, auto_adjust=auto_adjust)
    column = price_column if price_column in frame.columns else "Close"
    series = frame[column].copy()
    series.attrs = dict(getattr(frame, "attrs", {}))
    return series


__all__ = ["DEFAULT_TICKERS", "analyze_tickers", "fetch_benchmark"]
