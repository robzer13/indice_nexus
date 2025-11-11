"""Utilities for summarising and reporting backtest results."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Iterable, List

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _ensure_series(series: pd.Series | Iterable[float] | None) -> pd.Series:
    if series is None:
        raise ValueError("equity series is required")
    if isinstance(series, pd.Series):
        return series
    return pd.Series(list(series))


def _frame_to_records(frame: object) -> List[Dict[str, object]]:
    if frame is None:
        return []
    if hasattr(frame, "to_dict"):
        try:
            records = frame.to_dict(orient="records")  # type: ignore[attr-defined]
        except TypeError:
            records = frame.to_dict()  # type: ignore[attr-defined]
            if isinstance(records, list):
                return [row if isinstance(row, dict) else {} for row in records]
            return []
        else:
            return [row if isinstance(row, dict) else {} for row in records]
    columns = list(getattr(frame, "columns", []))
    length = len(frame) if hasattr(frame, "__len__") else 0
    data: List[Dict[str, object]] = []
    for position in range(length):
        row = {}
        for column in columns:
            column_data = getattr(frame, "_data", {}).get(column)
            if column_data is not None and position < len(column_data):
                row[column] = column_data[position]
            else:
                row[column] = None
        data.append(row)
    return data


def compute_drawdown(equity: pd.Series | Iterable[float]) -> pd.DataFrame:
    """Compute running drawdown statistics for an equity curve."""

    equity_series = _ensure_series(equity)
    if equity_series.empty:
        return pd.DataFrame(columns=["equity", "peak", "dd", "dd_abs"], index=equity_series.index)

    peaks: List[float] = []
    dd_pct: List[float] = []
    dd_abs: List[float] = []

    peak_value = None
    for value in equity_series.tolist():
        if value is None:
            peaks.append(float("nan"))
            dd_pct.append(float("nan"))
            dd_abs.append(float("nan"))
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float("nan")
        if numeric != numeric:
            peaks.append(float("nan"))
            dd_pct.append(float("nan"))
            dd_abs.append(float("nan"))
            continue
        if peak_value is None or numeric > peak_value:
            peak_value = numeric
        peaks.append(peak_value)
        if peak_value == 0:
            dd_pct.append(0.0)
            dd_abs.append(0.0)
        else:
            dd_abs_value = numeric - peak_value
            dd_abs.append(dd_abs_value)
            dd_pct.append(numeric / peak_value - 1.0)

    frame = pd.DataFrame(
        {"equity": equity_series.tolist(), "peak": peaks, "dd": dd_pct, "dd_abs": dd_abs},
        index=equity_series.index,
    )
    return frame


_DEF_METRIC_KEYS = [
    "CAGR",
    "Vol",
    "Sharpe",
    "MaxDD",
    "Calmar",
    "HitRate",
    "AvgWin",
    "AvgLoss",
    "Payoff",
    "ExposurePct",
]


def _format_datetime(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return "" if value is None else str(value)


def summarize_backtest(backtest: Dict[str, object]) -> Dict[str, object]:
    """Return a flattened dictionary of the key backtest metrics."""

    metrics = backtest.get("metrics") if isinstance(backtest, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}

    equity = backtest.get("equity") if isinstance(backtest, dict) else None
    try:
        equity_series = _ensure_series(equity)
    except Exception:
        equity_series = pd.Series(dtype=float)

    start = equity_series.index[0] if len(equity_series) else None
    end = equity_series.index[-1] if len(equity_series) else None

    trades_obj = backtest.get("trades") if isinstance(backtest, dict) else None
    if isinstance(trades_obj, pd.DataFrame):
        trades_count = len(trades_obj)
    else:
        trades_count = len(trades_obj) if isinstance(trades_obj, list) else 0

    summary = {key: metrics.get(key) for key in _DEF_METRIC_KEYS}
    summary.update(
        {
            "Trades": trades_count,
            "Start": _format_datetime(start),
            "End": _format_datetime(end),
        }
    )
    return summary


def attach_benchmark(
    equity: pd.Series | Iterable[float],
    *,
    benchmark_df: pd.Series | Iterable[float],
    label: str = "Benchmark",
) -> pd.DataFrame:
    """Align strategy equity with a benchmark series and rebase to 1.0."""

    equity_series = _ensure_series(equity)
    benchmark_series = _ensure_series(benchmark_df)
    if equity_series.empty or benchmark_series.empty:
        raise ValueError("equity and benchmark must be non-empty series")

    equity_index = list(equity_series.index)
    benchmark_index = list(benchmark_series.index)
    benchmark_lookup = {idx: value for idx, value in zip(benchmark_index, benchmark_series.tolist())}
    common_index: List[object] = []
    equity_values: List[float] = []
    benchmark_values: List[float] = []
    for idx, value in zip(equity_index, equity_series.tolist()):
        if idx in benchmark_lookup:
            common_index.append(idx)
            equity_values.append(value)
            benchmark_values.append(benchmark_lookup[idx])
    if not common_index:
        raise ValueError("No overlapping dates between equity and benchmark")

    try:
        sorted_index = sorted(common_index)
    except Exception:  # pragma: no cover - fallback for non sortable labels
        sorted_index = list(common_index)

    order_lookup = {idx: position for position, idx in enumerate(sorted_index)}
    sorted_equity = [None] * len(sorted_index)
    sorted_bench = [None] * len(sorted_index)
    for idx, value in zip(common_index, equity_values):
        sorted_equity[order_lookup[idx]] = value
    for idx, value in zip(common_index, benchmark_values):
        sorted_bench[order_lookup[idx]] = value

    equity_aligned = pd.Series(sorted_equity, index=sorted_index)
    benchmark_aligned = pd.Series(sorted_bench, index=sorted_index)

    start_equity = equity_aligned.iloc[0]
    start_bench = benchmark_aligned.iloc[0]

    def _safe_rebase(series, start):
        values = []
        for value in series.tolist():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = float("nan")
            if numeric != numeric or start in {None, 0}:
                values.append(float("nan"))
            else:
                values.append(numeric / float(start))
        return values

    data = {
        "Strategy": _safe_rebase(equity_aligned, start_equity),
        label: _safe_rebase(benchmark_aligned, start_bench),
    }
    return pd.DataFrame(data, index=equity_aligned.index)


_SCORE_LEVELS = {
    "low": (0, 40),
    "medium": (40, 70),
}


def _score_text(score: float | None) -> str:
    if score is None:
        return "faible"
    if score < _SCORE_LEVELS["low"][1]:
        return "faible"
    if score < _SCORE_LEVELS["medium"][1]:
        return "modéré"
    return "élevé"


def _format_trade(trade: Dict[str, object]) -> str:
    ticker = trade.get("ticker", "?")
    entry = _format_datetime(trade.get("entry_date")) or "?"
    exit_date = _format_datetime(trade.get("exit_date")) or "?"
    pnl = trade.get("ret")
    try:
        pnl_value = float(pnl)
    except (TypeError, ValueError):
        pnl_value = float("nan")
    pnl_str = "n/a" if pnl_value != pnl_value else f"{pnl_value * 100:.2f}%"
    return f"{ticker} : {entry} → {exit_date} ({pnl_str})"


def render_bt_markdown(
    backtest: Dict[str, object],
    *,
    benchmark_name: str | None,
    kpis: Dict[str, object],
    dd: pd.DataFrame,
    charts: Dict[str, str],
) -> str:
    """Render a Markdown representation of the backtest results."""

    lines: List[str] = []
    params = backtest.get("params") if isinstance(backtest, dict) else {}
    strategy = params.get("strategy") if isinstance(params, dict) else None
    title = f"### Résultats du backtest ({strategy})" if strategy else "### Résultats du backtest"
    lines.append(title)
    equity = backtest.get("equity") if isinstance(backtest, dict) else None
    try:
        equity_series = _ensure_series(equity)
    except Exception:
        equity_series = pd.Series(dtype=float)

    if len(equity_series):
        start = _format_datetime(equity_series.index[0])
        end = _format_datetime(equity_series.index[-1])
        lines.append(f"Période simulée : {start} → {end}.")
    else:
        lines.append("Période simulée indisponible.")

    lines.append("")
    lines.append("KPIs principaux :")
    lines.append("")
    headers = ["Metric", "Value"]
    rows = []
    for key in [
        "CAGR",
        "Vol",
        "Sharpe",
        "MaxDD",
        "Calmar",
        "HitRate",
        "AvgWin",
        "AvgLoss",
        "Payoff",
        "ExposurePct",
        "Trades",
    ]:
        rows.append((key, kpis.get(key, "n/a")))
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for key, value in rows:
        lines.append(f"| {key} | {value} |")

    lines.append("")
    equity_label = "Courbe equity"
    if benchmark_name:
        equity_label += f" vs {benchmark_name}"
    chart_equity = charts.get("equity")
    if chart_equity:
        lines.append(f"![{equity_label}]({chart_equity})")
        lines.append("")

    chart_dd = charts.get("drawdown")
    if chart_dd:
        lines.append(f"![Drawdown]({chart_dd})")
        lines.append("")

    chart_heatmap = charts.get("exposure")
    if chart_heatmap:
        lines.append(f"![Exposition]({chart_heatmap})")
        lines.append("")

    trades_obj = backtest.get("trades") if isinstance(backtest, dict) else None
    trade_rows = [row for row in _frame_to_records(trades_obj) if isinstance(row, dict)]

    def _ret_key(trade: Dict[str, object]) -> float:
        value = trade.get("ret")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return numeric

    top_trades = sorted(trade_rows, key=_ret_key, reverse=True)[:3]
    bottom_trades = sorted(trade_rows, key=_ret_key)[:3]

    if top_trades:
        lines.append("Top trades :")
        for trade in top_trades:
            lines.append(f"- {_format_trade(trade)}")
        lines.append("")

    if bottom_trades:
        lines.append("Pires trades :")
        for trade in bottom_trades:
            lines.append(f"- {_format_trade(trade)}")
        lines.append("")

    return "\n".join(lines).strip()


__all__ = [
    "compute_drawdown",
    "summarize_backtest",
    "attach_benchmark",
    "render_bt_markdown",
]
