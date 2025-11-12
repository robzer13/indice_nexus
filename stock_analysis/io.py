"""Persistence helpers for writing analysis outputs."""
from __future__ import annotations

import json
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import pandas as pd

from . import __version__
from .data import CANONICAL_PRICE_COLUMNS

LOGGER = logging.getLogger(__name__)

_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_ticker_name(ticker: str) -> str:
    safe = ticker.replace(os.sep, "_")
    safe = safe.replace("/", "_")
    safe = safe.replace(":", "_")
    safe = safe.replace(" ", "_")
    return safe


def _prepare_prices_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` that guarantees a stable column layout."""

    prepared = frame.copy()
    row_count = len(prepared)

    missing_columns: List[str] = []
    for column in CANONICAL_PRICE_COLUMNS:
        if column not in prepared.columns:
            missing_columns.append(column)
            prepared[column] = [getattr(pd, "NA", None)] * row_count

    if missing_columns:
        LOGGER.warning("Injecting missing canonical columns", extra={"columns": missing_columns})

    indicator_columns: List[str] = [
        column
        for column in prepared.columns
        if column not in CANONICAL_PRICE_COLUMNS
    ]

    ordered_columns: List[str] = [
        *CANONICAL_PRICE_COLUMNS,
        *[column for column in indicator_columns if column not in CANONICAL_PRICE_COLUMNS],
    ]

    data = {column: prepared[column].tolist() for column in ordered_columns if column in prepared.columns}
    for column in ordered_columns:
        if column not in data:
            data[column] = [getattr(pd, "NA", None)] * row_count

    normalised = pd.DataFrame(data, index=prepared.index, columns=ordered_columns)
    normalised.attrs = dict(getattr(prepared, "attrs", {}))
    return normalised


def _resolve_parquet_engine() -> str:
    try:
        import pyarrow  # type: ignore

        return "pyarrow"
    except Exception:  # pragma: no cover - optional dependency
        try:
            import fastparquet  # type: ignore

        except Exception as exc:  # pragma: no cover - optional dependency
            raise ValueError(
                "Saving to parquet requires either pyarrow or fastparquet to be installed."
            ) from exc
        LOGGER.warning("pyarrow non disponible, utilisation de fastparquet par repli")
        return "fastparquet"


def _write_prices(frame: pd.DataFrame, path: Path, *, format: str) -> None:
    if format == "csv":
        frame.to_csv(path, index=True, date_format=_DATE_FORMAT)
        return

    if format != "parquet":
        raise ValueError(f"Unsupported format {format!r}")

    engine = _resolve_parquet_engine()
    frame.to_parquet(path, engine=engine, index=True)


def _write_scores(frame: pd.DataFrame, path: Path, *, format: str) -> None:
    if format == "csv":
        frame.to_csv(path, index=False)
        return

    if format != "parquet":
        raise ValueError(f"Unsupported format {format!r}")

    engine = _resolve_parquet_engine()
    frame.to_parquet(path, engine=engine, index=False)


def _write_backtest_frame(frame: pd.DataFrame, path: Path, *, format: str, index: bool) -> None:
    if format == "csv":
        frame.to_csv(path, index=index, date_format=_DATE_FORMAT if index else None)
        return

    if format != "parquet":
        raise ValueError(f"Unsupported format {format!r}")

    engine = _resolve_parquet_engine()
    frame.to_parquet(path, engine=engine, index=index)


def save_analysis(
    result: Dict[str, Dict[str, object]],
    *,
    out_dir: str,
    base_name: str = "analysis",
    format: str = "parquet",
    include_prices: bool = True,
    include_fundamentals: bool = True,
    include_quality: bool = True,
    include_scores: bool = True,
    schema_version: str = "1.0.0",
    backtest: Dict[str, object] | None = None,
    include_backtest: bool = True,
) -> List[str]:
    """Persist the aggregated ``result`` to disk and return the created paths."""

    start_total = perf_counter()
    output_directory = Path(out_dir)
    _ensure_directory(output_directory)

    written_files: List[str] = []
    tickers: List[str] = []
    period = None
    interval = None
    price_column = None

    scoreboard_rows: List[Dict[str, object]] = []

    for ticker, payload in result.items():
        tickers.append(ticker)
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        period = period or meta.get("period")
        interval = interval or meta.get("interval")
        price_column = price_column or meta.get("price_column")

        safe_ticker = _safe_ticker_name(ticker)

        if include_prices and "prices" in payload:
            prices = payload["prices"]
            if getattr(prices, "empty", True):
                LOGGER.warning("Skipping empty price frame", extra={"ticker": ticker})
            else:
                prices_path = output_directory / f"{base_name}_{safe_ticker}_prices.{format}"
                price_start = perf_counter()
                prepared = _prepare_prices_frame(prices)
                _write_prices(prepared, prices_path, format=format)
                elapsed_ms = (perf_counter() - price_start) * 1000.0
                LOGGER.info(
                    "Saved price history",
                    extra={"ticker": ticker, "path": str(prices_path), "duration_ms": round(elapsed_ms, 2)},
                )
                written_files.append(str(prices_path))

        if include_fundamentals and "fundamentals" in payload:
            fundamentals_path = output_directory / f"{base_name}_{safe_ticker}_fundamentals.json"
            fundamentals_start = perf_counter()
            with fundamentals_path.open("w", encoding="utf-8") as handle:
                json.dump(payload["fundamentals"], handle, indent=2, default=str)
                handle.write("\n")
            elapsed_ms = (perf_counter() - fundamentals_start) * 1000.0
            LOGGER.info(
                "Saved fundamentals",
                extra={"ticker": ticker, "path": str(fundamentals_path), "duration_ms": round(elapsed_ms, 2)},
            )
            written_files.append(str(fundamentals_path))

        if include_quality and "quality" in payload:
            quality_path = output_directory / f"{base_name}_{safe_ticker}_quality.json"
            quality_start = perf_counter()
            with quality_path.open("w", encoding="utf-8") as handle:
                json.dump(payload["quality"], handle, indent=2, default=str)
                handle.write("\n")
            elapsed_ms = (perf_counter() - quality_start) * 1000.0
            LOGGER.info(
                "Saved quality report",
                extra={"ticker": ticker, "path": str(quality_path), "duration_ms": round(elapsed_ms, 2)},
            )
            written_files.append(str(quality_path))

        if include_scores and isinstance(payload, dict):
            bundle = payload.get("score")
            if isinstance(bundle, dict) and bundle:
                notes = bundle.get("notes") if isinstance(bundle.get("notes"), list) else []
                scoreboard_rows.append(
                    {
                        "Ticker": ticker,
                        "Score": bundle.get("score"),
                        "Trend": bundle.get("trend"),
                        "Momentum": bundle.get("momentum"),
                        "Quality": bundle.get("quality"),
                        "Risk": bundle.get("risk"),
                        "As Of": bundle.get("as_of", ""),
                        "NotesCount": len(notes),
                        "Notes": "; ".join(str(note) for note in notes),
                    }
                )

    if include_scores and scoreboard_rows:
        columns = ["Ticker", "Score", "Trend", "Momentum", "Quality", "Risk", "As Of", "NotesCount", "Notes"]
        scores_frame = pd.DataFrame({column: [row.get(column) for row in scoreboard_rows] for column in columns})
        scores_path = output_directory / f"{base_name}_scores.{format}"
        scores_start = perf_counter()
        _write_scores(scores_frame, scores_path, format=format)
        elapsed_ms = (perf_counter() - scores_start) * 1000.0
        LOGGER.info(
            "Saved score table",
            extra={"path": str(scores_path), "duration_ms": round(elapsed_ms, 2), "rows": len(scores_frame)},
        )
        written_files.append(str(scores_path))

    backtest_files: List[str] = []
    backtest_manifest: Dict[str, object] | None = None

    if include_backtest and isinstance(backtest, dict) and backtest:
        metrics = backtest.get("metrics") if isinstance(backtest.get("metrics"), dict) else {}
        params = backtest.get("params") if isinstance(backtest.get("params"), dict) else {}
        charts_meta = backtest.get("charts") if isinstance(backtest.get("charts"), dict) else {}

        equity_obj = backtest.get("equity")
        if equity_obj is not None:
            if hasattr(equity_obj, "tolist"):
                equity_values = list(equity_obj.tolist())  # type: ignore[attr-defined]
                equity_index = getattr(equity_obj, "index", list(range(len(equity_values))))
            else:
                equity_values = list(equity_obj)
                equity_index = list(range(len(equity_values)))
            equity_frame = pd.DataFrame({"equity": equity_values}, index=equity_index)
            equity_path = output_directory / f"{base_name}_equity.{format}"
            _write_backtest_frame(equity_frame, equity_path, format=format, index=True)
            written_files.append(str(equity_path))
            backtest_files.append(str(equity_path))

        trades_obj = backtest.get("trades")
        if trades_obj is not None:
            trades_frame = trades_obj if isinstance(trades_obj, pd.DataFrame) else pd.DataFrame(trades_obj)
            trades_path = output_directory / f"{base_name}_trades.{format}"
            _write_backtest_frame(trades_frame, trades_path, format=format, index=False)
            written_files.append(str(trades_path))
            backtest_files.append(str(trades_path))

        positions_obj = backtest.get("positions")
        if positions_obj is not None:
            positions_frame = positions_obj if isinstance(positions_obj, pd.DataFrame) else pd.DataFrame(positions_obj)
            positions_path = output_directory / f"{base_name}_positions.{format}"
            _write_backtest_frame(positions_frame, positions_path, format=format, index=True)
            written_files.append(str(positions_path))
            backtest_files.append(str(positions_path))

        kpis_obj = backtest.get("kpis")
        if kpis_obj is not None:
            if isinstance(kpis_obj, pd.DataFrame):
                kpis_frame = kpis_obj
            elif isinstance(kpis_obj, dict):
                kpis_frame = pd.DataFrame({key: [value] for key, value in kpis_obj.items()})
            else:
                kpis_frame = pd.DataFrame(kpis_obj)
            kpis_path = output_directory / f"{base_name}_kpis.{format}"
            _write_backtest_frame(kpis_frame, kpis_path, format=format, index=False)
            written_files.append(str(kpis_path))
            backtest_files.append(str(kpis_path))

        drawdown_obj = backtest.get("drawdown")
        if drawdown_obj is not None:
            if isinstance(drawdown_obj, pd.DataFrame):
                drawdown_frame = drawdown_obj
            elif isinstance(drawdown_obj, dict):
                drawdown_frame = pd.DataFrame({key: value for key, value in drawdown_obj.items()})
            else:
                drawdown_frame = pd.DataFrame(drawdown_obj)
            drawdown_path = output_directory / f"{base_name}_drawdown.{format}"
            _write_backtest_frame(drawdown_frame, drawdown_path, format=format, index=True)
            written_files.append(str(drawdown_path))
            backtest_files.append(str(drawdown_path))

        backtest_manifest = {
            "metrics": metrics,
            "params": params,
            "charts": charts_meta,
            "files": backtest_files,
        }

    manifest_path = output_directory / f"{base_name}_MANIFEST.json"
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": schema_version,
        "tickers": tickers,
        "period": period or "",
        "interval": interval or "",
        "price_column": price_column or "Close",
        "source": "yfinance",
        "timezone": "Europe/Paris",
        "format": format,
        "scores_included": bool(include_scores and scoreboard_rows),
        "backtest": backtest_manifest,
        "versions": {
            "package": __version__,
            "python": platform.python_version(),
            "pandas": getattr(pd, "__version__", "unknown"),
        },
        "files": written_files,
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    written_files.append(str(manifest_path))

    total_elapsed = (perf_counter() - start_total) * 1000.0
    LOGGER.info(
        "Analysis saved",
        extra={"directory": str(output_directory), "duration_ms": round(total_elapsed, 2), "files": len(written_files)},
    )

    return written_files


__all__ = ["save_analysis"]
