"""Derive behavioural indicators from trading context."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

import pandas as pd

from .schemas import BehaviorIndicator

_INDICATOR_DEFAULTS: Dict[str, float] = {
    "turnover_ratio": 0.15,
    "position_size_variance": 0.1,
    "loss_hold_bias": 0.1,
    "drawdown_streak": 0.1,
    "add_to_losers": 0.05,
    "chasing_intensity": 0.05,
    "plan_deviation": 0.05,
    "win_loss_asymmetry": 0.1,
    "concentration_ratio": 0.15,
    "volatility_shift": 0.1,
}

_REQUIRED_COLUMNS = {
    "pnl",
    "size",
    "holding_days",
}


def _safe_frame(trades: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(trades))
    if frame.empty:
        return frame
    for column in _REQUIRED_COLUMNS:
        if column not in frame:
            frame[column] = 0.0
    if "added_to_loser" not in frame:
        frame["added_to_loser"] = False
    if "chasing" not in frame:
        frame["chasing"] = False
    if "planned_exit" not in frame:
        frame["planned_exit"] = frame.get("holding_days", 0.0)
    if "actual_exit" not in frame:
        frame["actual_exit"] = frame.get("holding_days", 0.0)
    if "sector" not in frame:
        frame["sector"] = "unknown"
    return frame


def _compute_turnover(trades: pd.DataFrame, window_days: float) -> float:
    if trades.empty:
        return _INDICATOR_DEFAULTS["turnover_ratio"]
    window = max(window_days, 1.0)
    return float(min(1.0, len(trades) / window))


def _position_size_variance(trades: pd.DataFrame) -> float:
    if trades.empty:
        return _INDICATOR_DEFAULTS["position_size_variance"]
    std = float(trades["size"].std(ddof=0) or 0.0)
    mean = float(abs(trades["size"].mean()) or 1.0)
    return float(min(1.0, std / mean))


def _loss_hold_bias(trades: pd.DataFrame) -> float:
    if trades.empty:
        return _INDICATOR_DEFAULTS["loss_hold_bias"]
    losers = trades[trades["pnl"] < 0]
    winners = trades[trades["pnl"] > 0]
    if losers.empty or winners.empty:
        return _INDICATOR_DEFAULTS["loss_hold_bias"]
    loser_hold = float(losers["holding_days"].mean())
    winner_hold = float(winners["holding_days"].mean() or 1.0)
    delta = max(0.0, loser_hold - winner_hold)
    return float(min(1.0, delta / (winner_hold + 1e-9)))


def _drawdown_streak(trades: pd.DataFrame) -> float:
    if trades.empty:
        return _INDICATOR_DEFAULTS["drawdown_streak"]
    streak = 0
    best = 0
    for value in trades["pnl"]:
        if float(value) < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return float(min(1.0, best / 10.0))


def _add_to_losers(trades: pd.DataFrame) -> float:
    if trades.empty:
        return _INDICATOR_DEFAULTS["add_to_losers"]
    losses = trades[trades["pnl"] < 0]
    if losses.empty:
        return 0.0
    ratio = float(losses["added_to_loser"].astype(bool).mean())
    return float(min(1.0, ratio))


def _chasing(trades: pd.DataFrame) -> float:
    if trades.empty:
        return _INDICATOR_DEFAULTS["chasing_intensity"]
    ratio = float(trades["chasing"].astype(bool).mean())
    return float(min(1.0, ratio))


def _plan_deviation(trades: pd.DataFrame) -> float:
    if trades.empty:
        return _INDICATOR_DEFAULTS["plan_deviation"]
    diff = (trades["actual_exit"] - trades["planned_exit"]).abs()
    if diff.empty:
        return _INDICATOR_DEFAULTS["plan_deviation"]
    return float(min(1.0, diff.mean() / max(trades["planned_exit"].mean(), 1.0)))


def _win_loss_asymmetry(trades: pd.DataFrame) -> float:
    if trades.empty:
        return _INDICATOR_DEFAULTS["win_loss_asymmetry"]
    wins = float((trades["pnl"] > 0).mean())
    losses = float((trades["pnl"] < 0).mean())
    return float(min(1.0, abs(wins - losses)))


def _concentration(trades: pd.DataFrame, tickers: Sequence[str]) -> float:
    counts: MutableMapping[str, int] = defaultdict(int)
    for ticker in tickers:
        counts[ticker] = 0
    for ticker in trades.get("ticker", []):
        counts[str(ticker)] += 1
    total = sum(counts.values())
    if total <= 0:
        return _INDICATOR_DEFAULTS["concentration_ratio"]
    max_ratio = max(count / total for count in counts.values() if total)
    return float(min(1.0, max_ratio * len(counts)))


def _volatility_shift(context: Mapping[str, float]) -> float:
    recent = float(context.get("recent_vol", 0.0))
    baseline = float(context.get("baseline_vol", 0.0))
    if baseline <= 0:
        return _INDICATOR_DEFAULTS["volatility_shift"]
    ratio = abs(recent - baseline) / baseline
    return float(min(1.0, ratio))


def compute_indicators(
    tickers: Sequence[str],
    context: Mapping[str, object],
) -> Dict[str, BehaviorIndicator]:
    """Generate normalised behavioural indicators."""

    overrides = context.get("indicator_overrides")
    indicators: Dict[str, BehaviorIndicator] = {}
    if isinstance(overrides, Mapping):
        for name, value in overrides.items():
            indicators[name] = BehaviorIndicator(name=name, value=float(value))

    trades_source = context.get("trades", {})
    frames = []
    if isinstance(trades_source, Mapping):
        for ticker in tickers:
            dataset = trades_source.get(ticker)
            if dataset is None:
                continue
            if isinstance(dataset, pd.DataFrame):
                frame = dataset.copy()
            else:
                frame = _safe_frame(dataset if isinstance(dataset, Iterable) else [])
            if not frame.empty:
                frame = frame.copy()
                frame["ticker"] = ticker
                frames.append(frame)
    if frames:
        trades = pd.concat(frames, ignore_index=True)
    else:
        trades = pd.DataFrame(columns=list(_REQUIRED_COLUMNS))

    period_days = float(context.get("period_days", max(len(trades), 1)))
    if "turnover_ratio" not in indicators:
        indicators["turnover_ratio"] = BehaviorIndicator(
            name="turnover_ratio", value=_compute_turnover(trades, period_days)
        )
    if "position_size_variance" not in indicators:
        indicators["position_size_variance"] = BehaviorIndicator(
            name="position_size_variance", value=_position_size_variance(trades)
        )
    if "loss_hold_bias" not in indicators:
        indicators["loss_hold_bias"] = BehaviorIndicator(
            name="loss_hold_bias", value=_loss_hold_bias(trades)
        )
    if "drawdown_streak" not in indicators:
        indicators["drawdown_streak"] = BehaviorIndicator(
            name="drawdown_streak", value=_drawdown_streak(trades)
        )
    if "add_to_losers" not in indicators:
        indicators["add_to_losers"] = BehaviorIndicator(
            name="add_to_losers", value=_add_to_losers(trades)
        )
    if "chasing_intensity" not in indicators:
        indicators["chasing_intensity"] = BehaviorIndicator(
            name="chasing_intensity", value=_chasing(trades)
        )
    if "plan_deviation" not in indicators:
        indicators["plan_deviation"] = BehaviorIndicator(
            name="plan_deviation", value=_plan_deviation(trades)
        )
    if "win_loss_asymmetry" not in indicators:
        indicators["win_loss_asymmetry"] = BehaviorIndicator(
            name="win_loss_asymmetry", value=_win_loss_asymmetry(trades)
        )
    if "concentration_ratio" not in indicators:
        indicators["concentration_ratio"] = BehaviorIndicator(
            name="concentration_ratio", value=_concentration(trades, tickers)
        )
    vol_context = context.get("volatility", {})
    if isinstance(vol_context, Mapping) and "volatility_shift" not in indicators:
        indicators["volatility_shift"] = BehaviorIndicator(
            name="volatility_shift", value=_volatility_shift(vol_context)
        )
    elif "volatility_shift" not in indicators:
        indicators["volatility_shift"] = BehaviorIndicator(
            name="volatility_shift", value=_INDICATOR_DEFAULTS["volatility_shift"]
        )

    return indicators


__all__ = ["compute_indicators"]
