"""Fundamental ratios sourced from Yahoo Finance via yfinance."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)
PARIS_TZ = ZoneInfo("Europe/Paris")


def _first_available(candidates: Iterable[Tuple[str, Any]]) -> tuple[float | None, str | None]:
    for label, value in candidates:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        return numeric, label
    return None, None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    try:
        return pd.isna(value)
    except Exception:  # pragma: no cover - fallback for exotic objects
        return False


def _latest_value(frame: Any, row_label: str) -> float | None:
    if frame is None or getattr(frame, "empty", False):
        return None

    try:
        series = frame.loc[row_label]
    except KeyError:
        return None

    if isinstance(series, pd.Series):
        ordered = series.dropna()
        if ordered.empty:
            return None
        return float(ordered.iloc[0])

    if isinstance(series, (list, tuple)):
        for value in series:
            if not _is_missing(value):
                return float(value)
        return None

    try:
        iterator = iter(series.values())
    except AttributeError:  # pragma: no cover - unexpected structure
        return None

    for value in iterator:
        if not _is_missing(value):
            return float(value)
    return None


def _compute_dividend_yield(ticker_client: Any, last_price: float | None) -> tuple[float | None, str | None]:
    fast_info = getattr(ticker_client, "fast_info", {}) or {}
    info = getattr(ticker_client, "info", {}) or {}

    if "dividend_yield" in fast_info and fast_info["dividend_yield"] is not None:
        try:
            return float(fast_info["dividend_yield"]) * 100.0, "fast_info.dividend_yield"
        except (TypeError, ValueError):
            LOGGER.warning("Invalid dividend_yield in fast_info", extra={"ticker": ticker_client.ticker})

    if info.get("dividendYield") is not None:
        try:
            return float(info["dividendYield"]) * 100.0, "info.dividendYield"
        except (TypeError, ValueError):
            LOGGER.warning("Invalid dividendYield in info", extra={"ticker": ticker_client.ticker})

    dividends = getattr(ticker_client, "dividends", None)
    if dividends is None or getattr(dividends, "empty", True) or not last_price:
        return None, None

    window_start = datetime.now(PARIS_TZ) - timedelta(days=365)
    try:
        series = dividends[dividends.index >= window_start]
    except Exception:  # pragma: no cover - pandas variations
        return None, None

    if series.empty:
        return None, None

    annual_dividend = float(series.sum())
    if not annual_dividend:
        return None, None

    return (annual_dividend / float(last_price)) * 100.0, "derived.dividends"


def fetch_fundamentals(ticker: str) -> Dict[str, Any]:
    """Return fundamental ratios for ``ticker`` with provenance metadata."""

    LOGGER.info("Fetching fundamental metrics", extra={"ticker": ticker})
    client = yf.Ticker(ticker)
    client.ticker = ticker  # makes logging extras consistent in tests

    fast_info = getattr(client, "fast_info", {}) or {}
    info = getattr(client, "info", {}) or {}

    shares, shares_source = _first_available(
        [
            ("fast_info.shares", fast_info.get("shares")),
            ("info.sharesOutstanding", info.get("sharesOutstanding")),
            ("client.shares_outstanding", getattr(client, "shares_outstanding", None)),
        ]
    )

    income_statement = getattr(client, "financials", None)
    balance_sheet = getattr(client, "balance_sheet", None)

    net_income = _latest_value(income_statement, "Net Income")
    revenue = _latest_value(income_statement, "Total Revenue")
    total_debt = _latest_value(balance_sheet, "Total Debt")
    equity = _latest_value(balance_sheet, "Total Stockholder Equity")

    eps, eps_source = _first_available(
        [
            ("fast_info.eps", fast_info.get("eps")),
            ("info.trailingEps", info.get("trailingEps")),
            ("info.epsTrailingTwelveMonths", info.get("epsTrailingTwelveMonths")),
        ]
    )

    if eps is None and net_income is not None and shares:
        eps = net_income / shares
        eps_source = "derived.net_income_per_share"

    if eps is None:
        LOGGER.warning("EPS unavailable", extra={"ticker": ticker})

    price, price_source = _first_available(
        [
            ("fast_info.last_price", fast_info.get("last_price")),
            ("fast_info.lastPrice", fast_info.get("lastPrice")),
            ("info.currentPrice", info.get("currentPrice")),
            ("info.regularMarketPrice", info.get("regularMarketPrice")),
        ]
    )

    pe_ratio, pe_source = _first_available(
        [
            ("fast_info.pe_ratio", fast_info.get("pe_ratio")),
            ("info.trailingPE", info.get("trailingPE")),
        ]
    )

    if pe_ratio is None and eps and eps > 0 and price:
        pe_ratio = price / eps
        pe_source = "derived.price_over_eps"
    elif pe_ratio is None and eps and eps <= 0:
        LOGGER.warning("EPS non-positive; cannot compute P/E", extra={"ticker": ticker})
    elif pe_ratio is None:
        LOGGER.warning("P/E unavailable", extra={"ticker": ticker})

    net_margin_pct = None
    if net_income is not None and revenue not in (None, 0):
        net_margin_pct = (net_income / revenue) * 100.0
    else:
        LOGGER.warning("Net margin unavailable", extra={"ticker": ticker})

    debt_to_equity = None
    if total_debt is not None and equity not in (None, 0):
        debt_to_equity = total_debt / equity
    else:
        LOGGER.warning("Debt to equity unavailable", extra={"ticker": ticker})

    dividend_yield_pct, dividend_source = _compute_dividend_yield(client, price)
    if dividend_yield_pct is None:
        LOGGER.warning("Dividend yield unavailable; defaulting to 0", extra={"ticker": ticker})
        dividend_yield_pct = 0.0

    result = {
        "eps": eps,
        "pe_ratio": pe_ratio,
        "net_margin_pct": net_margin_pct,
        "debt_to_equity": debt_to_equity,
        "dividend_yield_pct": dividend_yield_pct,
        "as_of": datetime.now(PARIS_TZ),
        "source_fields": {
            "shares": shares_source,
            "eps": eps_source,
            "pe_ratio": pe_source,
            "net_margin_pct": "financials.Net Income/Total Revenue",
            "debt_to_equity": "balance_sheet.Total Debt/Total Stockholder Equity",
            "dividend_yield_pct": dividend_source,
            "price": price_source,
        },
    }

    return result


__all__ = ["fetch_fundamentals"]
