"""Script to fetch historical data for tickers and compute moving averages.

This module previously focused on the LVMH ticker (``MC.PA``). It now provides
helpers to compute indicators for multiple tickers so that we can easily test
the workflow with LVMH, ASML and TotalEnergies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import pandas as pd
import yfinance as yf


@dataclass
class MovingAverageConfig:
    """Configuration for moving averages to compute."""

    ema_windows: Iterable[int]
    sma_windows: Iterable[int]


DEFAULT_TICKER = "MC.PA"  # Euronext Paris ticker for LVMH
DEFAULT_TICKERS: Sequence[str] = (
    "ASML.AS",  # ASML traded on Euronext Amsterdam
    "TTE.PA",  # TotalEnergies traded on Euronext Paris
    DEFAULT_TICKER,
)
DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"
DEFAULT_CONFIG = MovingAverageConfig(
    ema_windows=(7, 9, 20, 21),
    sma_windows=(20, 50, 100, 200),
)


def fetch_price_history(
    ticker: str = DEFAULT_TICKER,
    *,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Return the historical price data for the provided ticker.

    Parameters
    ----------
    ticker:
        Symbol to fetch. Defaults to ``MC.PA`` for LVMH.
    period:
        Window of historical data to download. ``"1y"`` by default.
    interval:
        Temporal resolution of the data. ``"1d"`` by default.
    auto_adjust:
        Whether to request prices adjusted for dividends and splits.
    """

    ticker_client = yf.Ticker(ticker)
    history = ticker_client.history(
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
    )

    if history.empty:
        raise ValueError(
            "No historical data was returned. Check the ticker symbol or parameters."
        )

    return history


def add_moving_averages(
    data: pd.DataFrame,
    *,
    ema_windows: Iterable[int],
    sma_windows: Iterable[int],
    price_column: str = "Close",
) -> pd.DataFrame:
    """Compute EMA and SMA columns for the provided DataFrame.

    Parameters
    ----------
    data:
        Historical price DataFrame as returned by :func:`fetch_price_history`.
    ema_windows:
        Iterable of window sizes for exponential moving averages.
    sma_windows:
        Iterable of window sizes for simple moving averages.
    price_column:
        Name of the column to use for the calculations.
    """

    if price_column not in data.columns:
        raise KeyError(
            f"Column '{price_column}' is missing from the input data: {data.columns!r}"
        )

    enriched = data.copy()

    for window in ema_windows:
        enriched[f"EMA_{window}"] = enriched[price_column].ewm(span=window, adjust=False).mean()

    for window in sma_windows:
        enriched[f"SMA_{window}"] = enriched[price_column].rolling(window=window, min_periods=window).mean()

    return enriched


def compute_indicators(
    ticker: str = DEFAULT_TICKER,
    *,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    auto_adjust: bool = False,
    config: MovingAverageConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Fetch the historical data and append EMA/SMA indicators.

    This is a convenience wrapper that combines :func:`fetch_price_history`
    and :func:`add_moving_averages` using the default configuration.
    """

    history = fetch_price_history(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
    )

    enriched_history = add_moving_averages(
        history,
        ema_windows=config.ema_windows,
        sma_windows=config.sma_windows,
    )

    return enriched_history


def compute_indicators_for_tickers(
    tickers: Sequence[str],
    *,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    auto_adjust: bool = False,
    config: MovingAverageConfig = DEFAULT_CONFIG,
) -> Dict[str, pd.DataFrame]:
    """Compute moving-average indicators for a list of tickers.

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols (e.g. ``["ASML.AS", "TTE.PA", "MC.PA"]``).
    period, interval, auto_adjust, config:
        Same semantics as :func:`compute_indicators`.
    """

    results: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        results[ticker] = compute_indicators(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            config=config,
        )

    return results


if __name__ == "__main__":
    # Example usage: fetch one year of daily data for the three reference tickers.
    for ticker, frame in compute_indicators_for_tickers(DEFAULT_TICKERS).items():
        print(f"Ticker: {ticker}")
        print("Computed columns:", [col for col in frame.columns if "MA" in col])
        print(frame.tail())
        print("-" * 80)
