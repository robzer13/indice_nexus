"""FastAPI application exposing price, feature, and ML insights."""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .analyzer import analyze_tickers
from .data import fetch_price_history
from .features import add_ta_features, make_label_future_ret
from .indicators import compute_macd, compute_moving_averages, compute_rsi
from .ml_pipeline import FEATURES, confusion, sharpe_sim, time_cv, walk_forward_signals

LOGGER = logging.getLogger("stock_analysis.api")

app = FastAPI(title="Indice Nexus API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _frame_to_records(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> List[Dict[str, object]]:
    target_columns = list(columns) if columns is not None else list(getattr(frame, "columns", []))
    if not target_columns:
        target_columns = list(getattr(frame, "columns", []))
    records: List[Dict[str, object]] = []
    for timestamp, row in frame.iterrows():
        payload = {column: getattr(row, column, None) for column in target_columns}
        ts_value = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
        payload["timestamp"] = ts_value
        records.append(payload)
    return records


def _series_to_records(series: pd.Series, value_key: str) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    index = getattr(series, "index", [])
    values = list(series)
    for timestamp, value in zip(index, values):
        ts_value = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
        entries.append({"timestamp": ts_value, value_key: value})
    return entries


def _prepare_prices(
    ticker: str,
    *,
    period: str,
    interval: str,
    price_column: str = "Close",
) -> pd.DataFrame:
    frame = fetch_price_history(ticker, period=period, interval=interval)
    enriched = compute_moving_averages(frame, price_column=price_column)
    enriched = compute_rsi(enriched, price_column=price_column, period=14)
    enriched = compute_macd(enriched, price_column=price_column)
    return enriched


@app.get("/health")
def health() -> Dict[str, object]:
    return {"status": "ok", "version": __version__}


@app.get("/prices/{ticker}")
def get_prices(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    price_column: str = "Close",
) -> Dict[str, object]:
    try:
        frame = _prepare_prices(ticker, period=period, interval=interval, price_column=price_column)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime/network failures
        LOGGER.error("Failed to download prices", exc_info=exc, extra={"ticker": ticker})
        raise HTTPException(status_code=500, detail="Unable to download prices") from exc

    return {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "price_column": price_column,
        "data": _frame_to_records(frame),
    }


@app.get("/features/{ticker}")
def get_features(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    price_column: str = "Close",
) -> Dict[str, object]:
    try:
        frame = _prepare_prices(ticker, period=period, interval=interval, price_column=price_column)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        LOGGER.error("Failed to prepare features", exc_info=exc, extra={"ticker": ticker})
        raise HTTPException(status_code=500, detail="Unable to compute features") from exc

    try:
        features = add_ta_features(frame)
    except Exception as exc:  # pragma: no cover - computation failure
        LOGGER.error("Unable to compute technical features", exc_info=exc, extra={"ticker": ticker})
        raise HTTPException(status_code=500, detail="Unable to compute technical features") from exc

    combined = frame.copy()
    for column in getattr(features, "columns", []):
        combined[column] = features[column]

    return {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "price_column": price_column,
        "data": _frame_to_records(combined),
    }


@app.get("/ml/{ticker}")
def get_ml_summary(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    model: str = "xgb",
    horizon: int = 5,
    threshold: float = 0.0,
    retrain: int = 60,
    proba_threshold: float = 0.55,
    price_column: str = "Close",
) -> Dict[str, object]:
    try:
        frame = _prepare_prices(ticker, period=period, interval=interval, price_column=price_column)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        LOGGER.error("Failed to prepare ML dataset", exc_info=exc, extra={"ticker": ticker})
        raise HTTPException(status_code=500, detail="Unable to prepare ML dataset") from exc

    try:
        feature_frame = add_ta_features(frame)
        label_series = make_label_future_ret(frame, horizon=horizon, thr=threshold)
    except Exception as exc:  # pragma: no cover - computation failure
        LOGGER.error("Unable to compute ML artefacts", exc_info=exc, extra={"ticker": ticker})
        raise HTTPException(status_code=500, detail="Unable to compute ML features") from exc

    features = (
        feature_frame[FEATURES]
        .replace([float("inf"), float("-inf")], float("nan"))
        .dropna()
    )
    label_series = label_series.loc[features.index].dropna()
    if len(features) == 0 or len(label_series) == 0:
        raise HTTPException(status_code=404, detail="Insufficient data for ML evaluation")

    try:
        auc_mean, auc_std = time_cv(features, label_series, model_kind=model, splits=5)
        warmup = max(10, min(len(features), 200))
        proba_series, signal_series = walk_forward_signals(
            features,
            label_series,
            retrain_every=max(1, retrain),
            warmup=warmup,
            model_kind=model,
            proba_threshold=proba_threshold,
        )
        price_series = frame["Adj Close"] if "Adj Close" in frame.columns else frame[price_column]
        aligned_prices = price_series.loc[signal_series.index]
        sharpe = sharpe_sim(aligned_prices, signal_series)
        cm = confusion(
            label_series.loc[proba_series.dropna().index],
            proba_series.dropna(),
            thr=0.5,
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - ML runtime errors
        LOGGER.error("ML evaluation failed", exc_info=exc, extra={"ticker": ticker})
        raise HTTPException(status_code=500, detail="ML evaluation failed") from exc

    return {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "model": model,
        "horizon": horizon,
        "threshold": threshold,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "sharpe": sharpe,
        "confusion": cm,
        "probabilities": _series_to_records(proba_series.dropna(), "proba"),
        "signals": _series_to_records(signal_series.dropna(), "signal"),
    }


@app.get("/report/{ticker}")
def get_report(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    price_column: str = "Close",
) -> Dict[str, object]:
    try:
        results = analyze_tickers(
            [ticker],
            period=period,
            interval=interval,
            price_column=price_column,
            use_cache=True,
        )
    except Exception as exc:  # pragma: no cover - runtime/network failures
        LOGGER.error("Analysis failed", exc_info=exc, extra={"ticker": ticker})
        raise HTTPException(status_code=500, detail="Unable to compute report") from exc

    payload = results.get(ticker)
    if not payload:
        raise HTTPException(status_code=404, detail=f"No data available for {ticker}")

    prices = payload.get("prices")
    fundamentals = payload.get("fundamentals")
    quality = payload.get("quality")
    score = payload.get("score")
    ml_bundle = payload.get("ml") if isinstance(payload.get("ml"), dict) else None

    latest = prices.index[-1] if hasattr(prices, "index") and len(prices.index) else None
    close_value = None
    if hasattr(prices, "columns") and price_column in prices.columns:
        close_value = prices[price_column].iloc[-1]

    return {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "price_column": price_column,
        "latest": latest.isoformat() if hasattr(latest, "isoformat") else str(latest),
        "close": close_value,
        "score": score,
        "quality": quality,
        "fundamentals": fundamentals,
        "ml": ml_bundle,
    }


__all__ = ["app"]
