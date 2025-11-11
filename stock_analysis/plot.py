"""Plotting utilities for price and indicator visualisations."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

try:  # pragma: no cover - optional dependency
    import matplotlib  # type: ignore
except Exception:  # pragma: no cover - matplotlib absent
    matplotlib = None  # type: ignore
    plt = None  # type: ignore
else:  # pragma: no cover - backend configuration only executed when available
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore

LOGGER = logging.getLogger(__name__)

_PRICE_SERIES = ["Close", "SMA20", "SMA50", "SMA200", "EMA21"]


def _extract_series(df, column: str) -> List[float]:
    if column not in getattr(df, "columns", []):
        return []
    series = df[column]
    if hasattr(series, "tolist"):
        values = series.tolist()
    else:  # pragma: no cover - defensive fallback for unexpected types
        values = list(series)
    filtered = []
    for value in values:
        if value is None:
            filtered.append(float("nan"))
        else:
            try:
                filtered.append(float(value))
            except (TypeError, ValueError):
                filtered.append(float("nan"))
    return filtered


def plot_ticker(df, *, price_column: str = "Close"):
    """Create a matplotlib figure summarising price action and MACD."""

    if plt is None:
        raise RuntimeError("matplotlib est requis pour générer des graphiques")

    if getattr(df, "empty", True):
        raise ValueError("DataFrame is empty, impossible de tracer un graphique")

    dates: Iterable = getattr(df, "index", [])
    title_ticker = getattr(getattr(df, "attrs", {}), "get", lambda *_: "")("ticker")
    title_ticker = title_ticker or "Ticker"

    try:
        last_date = dates[-1]
    except Exception:  # pragma: no cover - defensive fallback
        last_date = "?"

    price_values = _extract_series(df, price_column)
    if not price_values:
        raise ValueError(f"Colonne {price_column!r} absente pour le tracé")

    fig, (ax_price, ax_macd) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(10, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_price.plot(dates, price_values, label=price_column, color="#1f77b4", linewidth=1.5)

    for column, color in zip(_PRICE_SERIES[1:], ["#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]):
        series_values = _extract_series(df, column)
        if series_values:
            ax_price.plot(dates, series_values, label=column, linewidth=1.0, color=color)

    ax_price.set_ylabel("Prix")
    last_close = price_values[-1] if price_values else float("nan")
    ax_price.set_title(f"{title_ticker} — {last_date} (price={last_close:.2f})")
    handles, labels = ax_price.get_legend_handles_labels()
    if handles:
        ax_price.legend(loc="upper left", frameon=False)

    hist_values = _extract_series(df, "MACD_hist")
    if hist_values:
        ax_macd.bar(dates, hist_values, label="MACD_hist", color="#17becf", alpha=0.6)
    line_colors = {"MACD": "#d62728", "MACD_signal": "#7f7f7f"}
    for column in ("MACD", "MACD_signal"):
        series_values = _extract_series(df, column)
        if series_values:
            ax_macd.plot(dates, series_values, label=column, linewidth=1.0, color=line_colors[column])
    ax_macd.axhline(0.0, color="#333333", linewidth=0.8)
    ax_macd.set_ylabel("MACD")
    handles, labels = ax_macd.get_legend_handles_labels()
    if handles:
        ax_macd.legend(loc="upper left", frameon=False)

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


def save_figure(fig, path: str | Path) -> None:
    """Persist the provided ``fig`` to disk."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight")
    LOGGER.info("Chart saved", extra={"path": str(destination)})


__all__ = ["plot_ticker", "save_figure"]

