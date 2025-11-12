"""Plotting helpers dedicated to backtest visualisations."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

try:  # pragma: no cover - optional dependency
    import matplotlib  # type: ignore
except Exception:  # pragma: no cover - matplotlib absent
    matplotlib = None  # type: ignore
    plt = None  # type: ignore
else:  # pragma: no cover - backend selection only executed when available
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore

LOGGER = logging.getLogger(__name__)


def _extract_series(df, column: str) -> List[float]:
    if column not in getattr(df, "columns", []):
        return []
    series = df[column]
    if hasattr(series, "tolist"):
        values = series.tolist()
    else:  # pragma: no cover - defensive path
        values = list(series)
    cleaned: List[float] = []
    for value in values:
        if value is None:
            cleaned.append(float("nan"))
            continue
        try:
            cleaned.append(float(value))
        except (TypeError, ValueError):
            cleaned.append(float("nan"))
    return cleaned


def plot_equity_with_benchmark(df, *, title: str | None = None):
    """Plot strategy equity (and optional benchmark) rebased at 1.0."""

    if plt is None:
        raise RuntimeError("matplotlib est requis pour générer les graphiques de backtest")
    if getattr(df, "empty", True):
        raise ValueError("Frame vide, impossible de tracer la courbe d'équité")

    dates: Iterable = getattr(df, "index", [])
    figure, ax = plt.subplots(figsize=(10, 4))
    columns = [column for column in getattr(df, "columns", []) if column]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, column in enumerate(columns):
        series_values = _extract_series(df, column)
        if not series_values:
            continue
        color = palette[idx % len(palette)]
        ax.plot(dates, series_values, label=column, linewidth=1.4, color=color)

    ax.set_ylabel("Valeur rebasée")
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left", frameon=False)
    figure.autofmt_xdate(rotation=30)
    figure.tight_layout()
    return figure


def plot_drawdown(dd, *, title: str | None = None):
    """Plot drawdown percentage over time."""

    if plt is None:
        raise RuntimeError("matplotlib est requis pour générer les graphiques de backtest")
    if getattr(dd, "empty", True):
        raise ValueError("Frame vide, impossible de tracer le drawdown")

    dates: Iterable = getattr(dd, "index", [])
    dd_values = _extract_series(dd, "dd")
    figure, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(dates, dd_values, 0.0, color="#d62728", alpha=0.3)
    ax.plot(dates, dd_values, color="#d62728", linewidth=1.0)
    min_dd = min((value for value in dd_values if value == value), default=0.0)
    heading = title or f"MaxDD = {min_dd:.2%}" if min_dd == min_dd else "Drawdown"
    ax.set_title(heading)
    ax.set_ylabel("Drawdown")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    figure.autofmt_xdate(rotation=30)
    figure.tight_layout()
    return figure


def plot_exposure_heatmap(positions):
    """Plot a heatmap of exposure weights per ticker."""

    if plt is None:
        raise RuntimeError("matplotlib est requis pour générer les graphiques de backtest")
    if getattr(positions, "empty", True):
        raise ValueError("Frame vide, impossible de tracer la heatmap d'exposition")

    tickers = getattr(positions, "columns", [])
    dates: Iterable = getattr(positions, "index", [])
    matrix: List[List[float]] = []
    for date_idx in range(len(positions)):
        row: List[float] = []
        for ticker in tickers:
            series = positions[ticker]
            if hasattr(series, "tolist"):
                value = series.tolist()[date_idx]
            else:  # pragma: no cover - fallback
                value = series[date_idx]
            try:
                row.append(float(value))
            except (TypeError, ValueError):
                row.append(float("nan"))
        matrix.append(row)

    figure, ax = plt.subplots(figsize=(10, 4))
    cax = ax.imshow(matrix, aspect="auto", cmap="Blues", origin="lower")
    ax.set_yticks(range(len(dates)))
    ax.set_yticklabels([str(label) for label in dates])
    ax.set_xticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_xlabel("Tickers")
    ax.set_ylabel("Dates")
    ax.set_title("Exposition relative")
    figure.colorbar(cax, ax=ax, orientation="vertical", label="Poids")
    figure.tight_layout()
    return figure


def save_figure(fig, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight")
    LOGGER.info("Backtest chart saved", extra={"path": str(destination)})


__all__ = [
    "plot_equity_with_benchmark",
    "plot_drawdown",
    "plot_exposure_heatmap",
    "save_figure",
]
