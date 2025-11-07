"""Script to fetch historical data for tickers and compute indicators.

This module previously focused on the LVMH ticker (``MC.PA``). It now provides
helpers to compute moving-average and fundamental indicators for multiple
tickers so that we can easily test the workflow with LVMH, ASML and
TotalEnergies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd
import yfinance as yf


@dataclass
class FundamentalMetrics:
    """Container for fundamental ratios derived from financial statements."""

    eps: Optional[float]
    pe_ratio: Optional[float]
    net_margin: Optional[float]
    debt_to_equity: Optional[float]
    dividend_yield_pct: Optional[float]


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


def _latest_value(frame: pd.DataFrame, row_label: str) -> Optional[float]:
    """Return the most recent non-null value for ``row_label`` in ``frame``."""

    if frame.empty or row_label not in frame.index:
        return None

    series = frame.loc[row_label].dropna()
    if series.empty:
        return None

    return float(series.iloc[0])


def fetch_fundamental_metrics(
    ticker: str = DEFAULT_TICKER,
    *,
    price: Optional[float] = None,
) -> FundamentalMetrics:
    """Fetch fundamental indicators such as EPS, P/E, margin, leverage, dividend yield."""

    ticker_client = yf.Ticker(ticker)

    if price is None:
        recent_history = ticker_client.history(period="5d", interval="1d")
        if recent_history.empty:
            price = None
        else:
            price = float(recent_history["Close"].iloc[-1])

    shares_outstanding = ticker_client.shares_outstanding
    if not shares_outstanding:
        shares_outstanding = ticker_client.info.get("sharesOutstanding")

    income_statement = ticker_client.financials
    balance_sheet = ticker_client.balance_sheet

    net_income = _latest_value(income_statement, "Net Income")
    revenue = _latest_value(income_statement, "Total Revenue")
    total_debt = _latest_value(balance_sheet, "Total Debt")
    equity = _latest_value(balance_sheet, "Total Stockholder Equity")

    if net_income is None or not shares_outstanding:
        eps = None
    else:
        eps = net_income / shares_outstanding

    if eps and eps != 0 and price:
        pe_ratio = price / eps
    else:
        pe_ratio = None

    if net_income is not None and revenue not in (None, 0):
        net_margin = (net_income / revenue) * 100
    else:
        net_margin = None

    if total_debt is not None and equity not in (None, 0):
        debt_to_equity = total_debt / equity
    else:
        debt_to_equity = None

    dividends = ticker_client.dividends
    if dividends.empty or not price:
        dividend_yield_pct = ticker_client.info.get("dividendYield")
        if dividend_yield_pct is not None:
            dividend_yield_pct *= 100
    else:
        last_date = dividends.index.max()
        if pd.isna(last_date):
            dividend_yield_pct = None
        else:
            window_start = last_date - pd.DateOffset(years=1)
            annual_dividend = dividends.loc[dividends.index > window_start].sum()
            dividend_yield_pct = (annual_dividend / price) * 100 if price else None

    return FundamentalMetrics(
        eps=eps,
        pe_ratio=pe_ratio,
        net_margin=net_margin,
        debt_to_equity=debt_to_equity,
        dividend_yield_pct=dividend_yield_pct,
    )


def compute_indicators_for_tickers(
    tickers: Sequence[str],
    *,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    auto_adjust: bool = False,
    config: MovingAverageConfig = DEFAULT_CONFIG,
) -> Dict[str, Dict[str, object]]:
    """Compute moving-average and fundamental indicators for a list of tickers.

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols (e.g. ``["ASML.AS", "TTE.PA", "MC.PA"]``).
    period, interval, auto_adjust, config:
        Same semantics as :func:`compute_indicators`.
    """

    results: Dict[str, Dict[str, object]] = {}
    for ticker in tickers:
        prices = compute_indicators(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            config=config,
        )
        last_close = float(prices["Close"].iloc[-1]) if not prices.empty else None
        fundamentals = fetch_fundamental_metrics(ticker, price=last_close)

        results[ticker] = {
            "prices": prices,
            "fundamentals": fundamentals,
        }

    return results


if __name__ == "__main__":
    # Example usage: fetch one year of daily data for the three reference tickers.
    for ticker, payload in compute_indicators_for_tickers(DEFAULT_TICKERS).items():
        print(f"Ticker: {ticker}")
        prices = payload["prices"]
        print("Computed columns:", [col for col in prices.columns if "MA" in col])
        print(prices.tail())
        fundamentals: FundamentalMetrics = payload["fundamentals"]
        print("Fundamentals:")
        print(
            f"  EPS: {fundamentals.eps}\n"
            f"  P/E Ratio: {fundamentals.pe_ratio}\n"
            f"  Net Margin: {fundamentals.net_margin}\n"
            f"  Debt-to-Equity: {fundamentals.debt_to_equity}\n"
            f"  Dividend Yield (%): {fundamentals.dividend_yield_pct}"
        )
        print("-" * 80)
