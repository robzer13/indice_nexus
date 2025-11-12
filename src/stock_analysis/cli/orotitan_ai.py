"""CLI dispatcher for OroTitan AI operations."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import pandas as pd

from ..data import fetch_price_history
from ..features import add_ta_features
from ..indicators import compute_macd, compute_moving_averages, compute_rsi
from ..orotitan_ai import (
    Decision,
    FeedbackItem,
    apply_feedback,
    build_feature_dict,
    encode_inputs,
    render_markdown_report,
)
from ..orotitan_ai.decision_engine import decide_for_date
from ..orotitan_ai.feedback import load_state, save_state
from ..report_orotitan import render_orotitan_report
from ..regimes import MacroDataProvider, evaluate_regime, infer_regime_series
from ..weighting import compute_weights

LOGGER = logging.getLogger("stock_analysis.cli.orotitan")
_DEFAULT_LIMIT = 250


@dataclass(slots=True)
class OroTitanRun:
    """Structured payload returned by the OroTitan pipeline."""

    decisions: List[Decision]
    embeddings: Dict[str, Dict[pd.Timestamp, List[float]]]
    weights: Dict[str, float]
    regime: str
    as_of: pd.Timestamp
    kpis: Dict[str, float]
    behavior_contexts: Dict[str, Dict[str, object]]


def _vector_to_list(vector: Any) -> List[float]:
    if hasattr(vector, "tolist"):
        raw = vector.tolist()
        if isinstance(raw, float):
            return [float(raw)]
        return [float(value) for value in raw]
    return [float(value) for value in vector]


def _prepare_prices(
    ticker: str,
    *,
    period: str,
    interval: str,
    price_column: str,
) -> pd.DataFrame:
    frame = fetch_price_history(ticker, period=period, interval=interval)
    enriched = compute_moving_averages(frame, price_column=price_column)
    enriched = compute_rsi(enriched, price_column=price_column, period=14)
    enriched = compute_macd(enriched, price_column=price_column)
    return enriched


def _load_feedback_items(events: Iterable[Mapping[str, Any]]) -> List[FeedbackItem]:
    items: List[FeedbackItem] = []
    now = pd.Timestamp(datetime.now(timezone.utc))
    for payload in events:
        ticker = str(payload.get("ticker", "")).strip()
        if not ticker:
            continue
        outcome = str(payload.get("outcome", "neutral")).lower()
        note = payload.get("note")
        try:
            weight = float(payload.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        pnl: float
        if outcome == "hit":
            pnl = abs(weight)
        elif outcome == "miss":
            pnl = -abs(weight)
        else:
            pnl = 0.0
        horizon = int(payload.get("horizon_days", 5) or 5)
        date_value = payload.get("date")
        if date_value:
            try:
                timestamp = pd.Timestamp(date_value)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize("UTC")
            except Exception:
                timestamp = now
        else:
            timestamp = now
        items.append(
            FeedbackItem(
                date=timestamp,
                ticker=ticker,
                pnl=pnl,
                horizon_days=horizon,
                user_note=str(note) if note is not None else None,
            )
        )
    return items


def safe_json_loads(payload: str | None) -> Any:
    if payload is None:
        return None
    text = payload.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid JSON payload: {exc}") from exc


def _derive_behavior_context(ticker: str, prices: pd.DataFrame) -> Dict[str, float]:
    returns = prices["Close"].pct_change().dropna()
    volatility = float(returns.std() or 0.0)
    avg_return = float(returns.tail(min(20, len(returns))).mean() or 0.0)
    loss_bias = float(max(0.0, -(returns[returns < 0].mean() or 0.0)))
    turnover = min(1.0, len(returns) / 252)
    context = {
        "turnover_ratio": turnover,
        "position_size_variance": min(1.0, abs(volatility) * 12.0),
        "loss_hold_bias": min(1.0, loss_bias * 20.0),
        "drawdown_streak": min(1.0, float((returns < 0).astype(int).rolling(5).sum().max() or 0) / 5.0),
        "plan_deviation": min(1.0, abs(avg_return) * 40.0),
        "win_loss_asymmetry": min(1.0, abs(returns.gt(0).mean() - returns.lt(0).mean())),
        "concentration_ratio": 0.5,
        "chasing_intensity": min(1.0, float((returns.tail(10) > returns.tail(10).mean()).mean())),
    }
    return context


def _normalise_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    positive = {str(key): float(value) for key, value in weights.items() if float(value) > 0}
    if not positive:
        return {"trend": 0.25, "momentum": 0.25, "fundamental": 0.25, "behavioral": 0.25}
    total = sum(positive.values())
    return {key: value / total for key, value in positive.items()}


def _combine_weights(
    *,
    state_weights: Mapping[str, float] | None,
    regime_weights: Mapping[str, float] | None,
    computed: Mapping[str, float] | None,
) -> Dict[str, float]:
    if regime_weights:
        return _normalise_weights(regime_weights)
    if state_weights:
        baseline = dict(state_weights)
        if computed:
            merged: MutableMapping[str, float] = baseline.copy()
            for key, value in computed.items():
                merged[key] = (baseline.get(key, 0.0) + float(value)) / 2.0
            return _normalise_weights(merged)
        return _normalise_weights(baseline)
    if computed:
        return _normalise_weights(computed)
    return _normalise_weights({})


def run_pipeline(
    tickers: Sequence[str],
    *,
    period: str = "1y",
    interval: str = "1d",
    price_column: str = "Close",
    risk_budget: float = 0.1,
    temperature: float = 0.0,
    regime_override: str | None = None,
    regime_weights: Mapping[str, float] | None = None,
    limit: int = _DEFAULT_LIMIT,
    behavior_enabled: bool = False,
    behavior_threshold: float = 30.0,
    behavior_mode: str = "jsonl",
    behavior_max_influence: float = 0.25,
) -> OroTitanRun:
    provider = MacroDataProvider()
    datasets: Dict[str, dict[str, pd.Series | pd.DataFrame]] = {}
    last_dates: Dict[str, pd.Timestamp] = {}
    behavior_contexts: Dict[str, Dict[str, object]] = {}
    for ticker in tickers:
        try:
            prices = _prepare_prices(
                ticker,
                period=period,
                interval=interval,
                price_column=price_column,
            )
        except Exception as exc:
            LOGGER.error("Failed to prepare prices", exc_info=exc, extra={"ticker": ticker})
            continue
        try:
            features = add_ta_features(prices)
        except Exception as exc:
            LOGGER.error("Feature engineering failed", exc_info=exc, extra={"ticker": ticker})
            continue
        regimes = infer_regime_series(prices.index, provider=provider)
        feature_dict = build_feature_dict(prices, features, regimes)
        datasets[ticker] = feature_dict
        last_dates[ticker] = prices.index[-1]
        if behavior_enabled:
            behavior_contexts[ticker] = {
                "indicator_overrides": _derive_behavior_context(ticker, prices),
                "period_days": float(len(prices)),
            }

    if not datasets:
        raise ValueError("No datasets prepared for OroTitan AI")

    as_of = max(last_dates.values())
    regime_label = regime_override or evaluate_regime(as_of).regime

    state = load_state()
    weights = _combine_weights(
        state_weights=state.get("weights"),
        regime_weights=regime_weights,
        computed=compute_weights(regime_label),
    )

    embeddings: Dict[str, Dict[pd.Timestamp, List[float]]] = {}
    for ticker, feature_dict in datasets.items():
        available_dates = list(feature_dict["prices"].index)
        if limit and len(available_dates) > limit:
            selected = available_dates[-limit:]
        else:
            selected = available_dates
        encoded = encode_inputs(feature_dict, dates=selected)
        embeddings[ticker] = {
            pd.Timestamp(ts): _vector_to_list(vector) for ts, vector in encoded.items()
        }

    decisions: List[Decision] = []
    for ticker, encoded in embeddings.items():
        if not encoded:
            continue
        last_ts = max(encoded)
        vector = encoded[last_ts]
        decision = decide_for_date(
            ticker=ticker,
            date=last_ts,
            vectors=vector,
            nexus_weights=weights,
            risk_budget=risk_budget,
            temperature=temperature,
            enable_behavior=behavior_enabled,
            behavior_context=behavior_contexts.get(ticker),
            behavior_threshold=behavior_threshold,
            max_behavior_influence=behavior_max_influence,
            behavior_persist=behavior_mode,
        )
        decisions.append(decision)

    kpis = _build_kpis(decisions, as_of)
    return OroTitanRun(
        decisions=decisions,
        embeddings=embeddings,
        weights=weights,
        regime=regime_label,
        as_of=as_of,
        kpis=kpis,
        behavior_contexts=behavior_contexts,
    )


def _build_kpis(decisions: Sequence[Decision], as_of: pd.Timestamp) -> Dict[str, float]:
    if not decisions:
        return {"count": 0, "as_of": float(pd.Timestamp(as_of).timestamp())}
    count = len(decisions)
    avg_score = sum(decision.score for decision in decisions) / count
    buy_ratio = sum(1 for decision in decisions if decision.action == "BUY") / count
    sell_ratio = sum(1 for decision in decisions if decision.action == "SELL") / count
    return {
        "count": float(count),
        "avg_score": float(avg_score),
        "buy_ratio": float(buy_ratio),
        "sell_ratio": float(sell_ratio),
        "as_of": float(pd.Timestamp(as_of).timestamp()),
    }


def apply_feedback_events(
    weights: Mapping[str, float],
    events: Iterable[Mapping[str, Any]],
) -> tuple[Dict[str, float], List[str]]:
    items = _load_feedback_items(events)
    if not items:
        return dict(weights), []
    updated = apply_feedback(dict(weights), items)
    notes = [item.user_note or "" for item in items if item.user_note]
    return updated, notes


def _format_table(decisions: Sequence[Decision], regime: str) -> str:
    if not decisions:
        return "No decisions generated."
    headers = ["Ticker", "Regime", "Decision", "Score", "Confidence", "Headline"]
    rows = []
    for decision in decisions:
        headline = decision.rationale.split(".")[0]
        rows.append(
            [
                decision.ticker,
                regime,
                decision.action,
                f"{decision.score:.2f}",
                f"{decision.confidence:.2f}",
                headline,
            ]
        )
    col_widths = [max(len(str(row[idx])) for row in [headers] + rows) for idx in range(len(headers))]
    line = " | ".join(
        header.ljust(col_widths[idx]) for idx, header in enumerate(headers)
    )
    separator = "-+-".join("-" * width for width in col_widths)
    body = [
        " | ".join(str(row[idx]).ljust(col_widths[idx]) for idx in range(len(headers)))
        for row in rows
    ]
    return "\n".join([line, separator, *body])


def _decisions_to_json(decisions: Sequence[Decision], regime: str) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for decision in decisions:
        record = asdict(decision)
        record["date"] = decision.date.isoformat()
        record["regime"] = regime
        behavior = record.pop("behavior", None)
        if behavior is not None and decision.behavior is not None:
            record["behavior"] = decision.behavior.dict()
        payload.append(record)
    return payload


def run_from_args(args: Any) -> int:
    started = perf_counter()
    tickers: List[str] = []
    if args.tickers:
        tickers.extend([token.strip() for token in str(args.tickers).replace(",", " ").split() if token.strip()])
    if getattr(args, "tickers_file", None):
        path = Path(args.tickers_file)
        if not path.exists():
            raise FileNotFoundError(f"Tickers file not found: {path}")
        tickers.extend(
            [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        )
    tickers = sorted(set(tickers))
    if not tickers:
        raise ValueError("No tickers provided for OroTitan AI")

    regime_weights = safe_json_loads(getattr(args, "regime_weights", None))
    if getattr(args, "regime_file", None):
        data = safe_json_loads(Path(args.regime_file).read_text(encoding="utf-8"))
        if isinstance(data, Mapping):
            regime_weights = data

    behavior_enabled = bool(getattr(args, "behavior", False))
    behavior_threshold = float(getattr(args, "behavior_threshold", 30) or 30)
    behavior_mode = str(getattr(args, "behavior_persist", "jsonl") or "jsonl").lower()
    if behavior_mode not in {"jsonl", "sqlite", "none"}:
        raise ValueError("behavior-persist must be jsonl, sqlite or none")
    behavior_max = 0.25

    pipeline = run_pipeline(
        tickers,
        period=getattr(args, "period", "1y") or "1y",
        interval=getattr(args, "interval", "1d") or "1d",
        price_column=getattr(args, "price_column", "Close") or "Close",
        risk_budget=float(getattr(args, "risk_budget", 0.1) or 0.1),
        temperature=float(getattr(args, "temperature", 0.0) or 0.0),
        regime_override=getattr(args, "regime", None),
        regime_weights=regime_weights if isinstance(regime_weights, Mapping) else None,
        behavior_enabled=behavior_enabled,
        behavior_threshold=behavior_threshold,
        behavior_mode=behavior_mode,
        behavior_max_influence=behavior_max,
    )

    events: List[Mapping[str, Any]] = []
    feedback_file = getattr(args, "feedback_file", None)
    if feedback_file:
        path = Path(feedback_file)
        if path.exists():
            events = safe_json_loads(path.read_text(encoding="utf-8")) or []
            if not isinstance(events, list):
                raise ValueError("Feedback file must contain a JSON list")
        else:
            LOGGER.warning("Feedback file not found", extra={"path": str(path)})
    if events:
        updated_weights, notes = apply_feedback_events(pipeline.weights, events)
        pipeline.weights = updated_weights
        if getattr(args, "save_state", False) and not getattr(args, "dry_run", False):
            state_payload = {
                "as_of": pipeline.as_of.isoformat(),
                "weights": pipeline.weights,
                "meta": {"version": "v5", "source": "cli"},
            }
            save_state(state_payload)
        LOGGER.info("Feedback applied", extra={"count": len(events), "notes": notes})

    task = getattr(args, "ai_task", "full")
    dry_run = getattr(args, "dry_run", False)
    json_enabled = bool(getattr(args, "json", False))

    if events:
        updated_weights, notes = apply_feedback_events(pipeline.weights, events)
        pipeline.weights = updated_weights
        if pipeline.decisions:
            refreshed: List[Decision] = []
            for ticker, encoded in pipeline.embeddings.items():
                if not encoded:
                    continue
                last_ts = max(encoded)
                vector = encoded[last_ts]
                refreshed.append(
                    decide_for_date(
                        ticker=ticker,
                        date=last_ts,
                        vectors=vector,
                        nexus_weights=pipeline.weights,
                        risk_budget=float(getattr(args, "risk_budget", 0.1) or 0.1),
                        temperature=float(getattr(args, "temperature", 0.0) or 0.0),
                        enable_behavior=behavior_enabled,
                        behavior_context=pipeline.behavior_contexts.get(ticker),
                        behavior_threshold=behavior_threshold,
                        max_behavior_influence=behavior_max,
                        behavior_persist=behavior_mode,
                    )
                )
            pipeline.decisions = refreshed
            pipeline.kpis = _build_kpis(pipeline.decisions, pipeline.as_of)
        if getattr(args, "save_state", False) and not getattr(args, "dry_run", False):
            state_payload = {
                "as_of": pipeline.as_of.isoformat(),
                "weights": pipeline.weights,
                "meta": {"version": "v5", "source": "cli"},
            }
            save_state(state_payload)
        LOGGER.info("Feedback applied", extra={"count": len(events), "notes": notes})

    if task in {"decide", "full", "report"}:
        table = _format_table(pipeline.decisions, pipeline.regime)
        print(table)

    result_payload: Dict[str, Any] = {
        "decisions": _decisions_to_json(pipeline.decisions, pipeline.regime),
        "weights": pipeline.weights,
        "regime": pipeline.regime,
        "kpis": pipeline.kpis,
        "run": {
            "started": datetime.utcnow().isoformat() + "Z",
            "duration_s": round(perf_counter() - started, 4),
        },
    }

    if behavior_enabled and pipeline.decisions:
        behavior_snapshot = [
            decision.behavior.dict()
            for decision in pipeline.decisions
            if decision.behavior is not None
        ]
        result_payload["behavior"] = behavior_snapshot
        if getattr(args, "behavior_json", False):
            print(json.dumps({"behavior": behavior_snapshot}, indent=2))

    if task in {"embed", "full"}:
        embeddings_serialised: Dict[str, Dict[str, List[float]]] = {}
        for ticker, mapping in pipeline.embeddings.items():
            embeddings_serialised[ticker] = {ts.isoformat(): vector for ts, vector in mapping.items()}
        result_payload["embeddings"] = embeddings_serialised
        if task == "embed":
            LOGGER.info("Generated embeddings", extra={"tickers": len(embeddings_serialised)})

    report_paths: Dict[str, str] | None = None
    if task in {"report", "full"} and pipeline.decisions:
        report_dir = Path(getattr(args, "report_dir", "reports"))
        report_override = getattr(args, "report_out", None)
        target_path: Path | None = None
        if report_override:
            target_path = Path(report_override).expanduser().resolve()
            report_dir = target_path.parent
        if not dry_run:
            paths = render_orotitan_report(
                pipeline.decisions,
                pipeline.weights,
                pipeline.kpis,
                report_dir,
                title="OroTitan Cognitive Report",
            )
            report_paths = {key: str(value) for key, value in paths.items()}
            if target_path is not None:
                md_src = Path(paths["md"])
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(md_src.read_text(encoding="utf-8"), encoding="utf-8")
                report_paths["override_md"] = str(target_path)
            LOGGER.info("OroTitan report saved", extra=report_paths)
        else:
            markdown_body = render_markdown_report(
                pipeline.decisions,
                pipeline.weights,
                pipeline.kpis,
                title="OroTitan Cognitive Report",
            )
            print(markdown_body.splitlines()[0])
    if report_paths:
        result_payload["report"] = report_paths

    if json_enabled:
        print(json.dumps(result_payload, indent=2))

    LOGGER.info(
        "OroTitan run completed",
        extra={
            "tickers": len(tickers),
            "decisions": len(pipeline.decisions),
            "task": task,
        },
    )
    return 0


__all__ = [
    "OroTitanRun",
    "apply_feedback_events",
    "load_state",
    "render_markdown_report",
    "render_orotitan_report",
    "run_from_args",
    "run_pipeline",
    "safe_json_loads",
]
