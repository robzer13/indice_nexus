"""Command line entry-point for running the stock analysis pipeline."""
from __future__ import annotations

import argparse
import base64
import io
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Sequence

import pandas as pd

try:  # pragma: no cover - Python 3.11+ provides tomllib
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback when unavailable
    tomllib = None  # type: ignore

from .analyzer import DEFAULT_TICKERS, analyze_tickers, fetch_benchmark
from .backtest import run_backtest
from .bt_report import (
    attach_benchmark,
    compute_drawdown,
    render_bt_markdown,
    summarize_backtest,
)
from .io import save_analysis
from .plot import plot_ticker, save_figure as save_price_figure
from .plot_bt import (
    plot_drawdown,
    plot_equity_with_benchmark,
    plot_exposure_heatmap,
    save_figure as save_bt_figure,
)
from .report import build_summary_table, render_html, render_markdown

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"
LOGGER = logging.getLogger(__name__)
CONFIG_FILE = Path("stock_analysis.toml")

DEFAULT_SETTINGS: Dict[str, object] = {
    "tickers": DEFAULT_TICKERS,
    "period": "2y",
    "interval": "1d",
    "price_column": "Close",
    "gap_threshold": 5.0,
    "out_dir": "out",
    "format": "parquet",
    "base_name": "run",
    "save": False,
    "log_level": "INFO",
    "score": False,
    "top": 10,
    "report": False,
    "html": False,
    "charts_dir": "charts",
    "include_charts": True,
    "report_title": "Stock Analysis Report",
    "bt": False,
    "strategy": "sma200_trend",
    "capital": 10_000.0,
    "cost_bps": 10.0,
    "slippage_bps": 5.0,
    "max_positions": None,
    "stop_pct": None,
    "tp_pct": None,
    "benchmark": "",
    "bt_report": False,
    "charts_bt_dir": "bt_charts",
    "include_bt_charts": True,
    "bt_title": "Backtest Results",
}

ENV_KEYS = {
    "tickers": "STOCK_ANALYSIS_TICKERS",
    "period": "STOCK_ANALYSIS_PERIOD",
    "interval": "STOCK_ANALYSIS_INTERVAL",
    "price_column": "STOCK_ANALYSIS_PRICE_COLUMN",
    "gap_threshold": "STOCK_ANALYSIS_GAP_THRESHOLD",
    "out_dir": "STOCK_ANALYSIS_OUT_DIR",
    "format": "STOCK_ANALYSIS_FORMAT",
    "base_name": "STOCK_ANALYSIS_BASE_NAME",
    "save": "STOCK_ANALYSIS_SAVE",
    "log_level": "STOCK_ANALYSIS_LOG_LEVEL",
    "score": "STOCK_ANALYSIS_SCORE",
    "top": "STOCK_ANALYSIS_TOP",
    "report": "STOCK_ANALYSIS_REPORT",
    "html": "STOCK_ANALYSIS_HTML",
    "charts_dir": "STOCK_ANALYSIS_CHARTS_DIR",
    "include_charts": "STOCK_ANALYSIS_INCLUDE_CHARTS",
    "report_title": "STOCK_ANALYSIS_REPORT_TITLE",
    "bt": "STOCK_ANALYSIS_BT",
    "strategy": "STOCK_ANALYSIS_STRATEGY",
    "capital": "STOCK_ANALYSIS_CAPITAL",
    "cost_bps": "STOCK_ANALYSIS_COST_BPS",
    "slippage_bps": "STOCK_ANALYSIS_SLIPPAGE_BPS",
    "max_positions": "STOCK_ANALYSIS_MAX_POSITIONS",
    "stop_pct": "STOCK_ANALYSIS_STOP_PCT",
    "tp_pct": "STOCK_ANALYSIS_TP_PCT",
    "benchmark": "STOCK_ANALYSIS_BENCHMARK",
    "bt_report": "STOCK_ANALYSIS_BT_REPORT",
    "charts_bt_dir": "STOCK_ANALYSIS_BT_CHARTS_DIR",
    "include_bt_charts": "STOCK_ANALYSIS_BT_CHARTS",
    "bt_title": "STOCK_ANALYSIS_BT_TITLE",
}


def _format_number(value: object, *, precision: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(numeric):
        return "n/a"
    return f"{numeric:.{precision}f}"


def _format_int(value: object) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "n/a"


def _coerce_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _format_percent(value: object, *, precision: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(numeric):
        return "n/a"
    return f"{numeric * 100.0:.{precision}f}%"


def _parse_ticker_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return []


def _safe_ticker_name(ticker: str) -> str:
    safe = ticker.replace(os.sep, "_")
    safe = safe.replace("/", "_")
    safe = safe.replace(":", "_")
    safe = safe.replace(" ", "_")
    return safe


def _load_toml_defaults(path: Path) -> Dict[str, object]:
    if tomllib is None or not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except Exception:  # pragma: no cover - configuration errors are runtime only
        LOGGER.warning("Impossible de lire %s", path)
        return {}
    defaults = data.get("defaults", {}) if isinstance(data, dict) else {}
    return defaults if isinstance(defaults, dict) else {}


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_env_defaults() -> Dict[str, object]:
    resolved: Dict[str, object] = {}
    for name, env_key in ENV_KEYS.items():
        raw = os.environ.get(env_key)
        if raw is None:
            continue
        if name == "tickers":
            resolved[name] = raw
        elif name == "gap_threshold":
            try:
                resolved[name] = float(raw)
            except ValueError:
                LOGGER.warning("Valeur d'écart invalide pour %s: %s", env_key, raw)
        elif name == "save":
            resolved[name] = _parse_bool(raw)
        elif name == "score":
            resolved[name] = _parse_bool(raw)
        elif name == "report":
            resolved[name] = _parse_bool(raw)
        elif name == "html":
            resolved[name] = _parse_bool(raw)
        elif name == "include_charts":
            resolved[name] = _parse_bool(raw)
        elif name == "bt":
            resolved[name] = _parse_bool(raw)
        elif name == "top":
            try:
                resolved[name] = int(raw)
            except ValueError:
                LOGGER.warning("Valeur top invalide pour %s: %s", env_key, raw)
        elif name == "capital":
            try:
                resolved[name] = float(raw)
            except ValueError:
                LOGGER.warning("Capital invalide pour %s: %s", env_key, raw)
        elif name in {"cost_bps", "slippage_bps", "stop_pct", "tp_pct"}:
            try:
                resolved[name] = float(raw)
            except ValueError:
                LOGGER.warning("Valeur flottante invalide pour %s: %s", env_key, raw)
        elif name == "max_positions":
            try:
                resolved[name] = int(raw)
            except ValueError:
                LOGGER.warning("max_positions invalide pour %s: %s", env_key, raw)
        else:
            resolved[name] = raw
    return resolved


def _coalesce(*values: object) -> object | None:
    for value in values:
        if value is not None:
            return value
    return None


def _resolve_settings(args: argparse.Namespace) -> Dict[str, object]:
    config_defaults = _load_toml_defaults(CONFIG_FILE)
    env_defaults = _load_env_defaults()

    def pick(name: str) -> object | None:
        return _coalesce(
            getattr(args, name, None),
            env_defaults.get(name),
            config_defaults.get(name),
            DEFAULT_SETTINGS.get(name),
        )

    tickers = _parse_ticker_list(pick("tickers")) or list(DEFAULT_SETTINGS["tickers"])
    period = str(pick("period"))
    interval = str(pick("interval"))
    price_column = str(pick("price_column"))
    gap_threshold_raw = pick("gap_threshold")
    gap_threshold = float(gap_threshold_raw) if gap_threshold_raw is not None else float(DEFAULT_SETTINGS["gap_threshold"])
    out_dir = str(pick("out_dir"))
    file_format = str(pick("format") or DEFAULT_SETTINGS["format"]).lower()
    base_name = str(pick("base_name"))
    save_value = pick("save")
    save = _parse_bool(save_value) if save_value is not None else bool(DEFAULT_SETTINGS["save"])
    log_level = str(pick("log_level") or DEFAULT_SETTINGS["log_level"]).upper()
    score_value = pick("score")
    score = _parse_bool(score_value) if score_value is not None else bool(DEFAULT_SETTINGS["score"])

    top_value = pick("top")
    try:
        top = int(top_value) if top_value is not None else int(DEFAULT_SETTINGS["top"])
    except (TypeError, ValueError):
        top = int(DEFAULT_SETTINGS["top"])

    report_value = pick("report")
    report = _parse_bool(report_value) if report_value is not None else bool(DEFAULT_SETTINGS["report"])
    html_value = pick("html")
    html = _parse_bool(html_value) if html_value is not None else bool(DEFAULT_SETTINGS["html"])
    charts_dir_value = str(pick("charts_dir") or DEFAULT_SETTINGS["charts_dir"])
    charts_dir = charts_dir_value.strip() or str(DEFAULT_SETTINGS["charts_dir"])
    include_charts_value = pick("include_charts")
    include_charts = (
        _parse_bool(include_charts_value)
        if include_charts_value is not None
        else bool(DEFAULT_SETTINGS["include_charts"])
    )
    report_title = str(pick("report_title") or DEFAULT_SETTINGS["report_title"])
    benchmark = str(pick("benchmark") or DEFAULT_SETTINGS["benchmark"])
    bt_value = pick("bt")
    bt = _parse_bool(bt_value) if bt_value is not None else bool(DEFAULT_SETTINGS["bt"])
    strategy = str(pick("strategy") or DEFAULT_SETTINGS["strategy"])

    capital_raw = pick("capital")
    capital = float(capital_raw) if capital_raw is not None else float(DEFAULT_SETTINGS["capital"])

    cost_raw = pick("cost_bps")
    cost_bps = float(cost_raw) if cost_raw is not None else float(DEFAULT_SETTINGS["cost_bps"])

    slippage_raw = pick("slippage_bps")
    slippage_bps = float(slippage_raw) if slippage_raw is not None else float(DEFAULT_SETTINGS["slippage_bps"])

    max_positions_raw = pick("max_positions")
    if max_positions_raw in {None, "", "none", "None"}:
        max_positions = None
    else:
        try:
            max_positions = int(max_positions_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            max_positions = None

    stop_raw = pick("stop_pct")
    stop_pct = float(stop_raw) if stop_raw not in {None, "", "none", "None"} else None

    tp_raw = pick("tp_pct")
    tp_pct = float(tp_raw) if tp_raw not in {None, "", "none", "None"} else None

    bt_report_value = pick("bt_report")
    bt_report = (
        _parse_bool(bt_report_value)
        if bt_report_value is not None
        else bool(DEFAULT_SETTINGS["bt_report"])
    )

    charts_bt_dir_value = str(pick("charts_bt_dir") or DEFAULT_SETTINGS["charts_bt_dir"])
    charts_bt_dir = charts_bt_dir_value.strip() or str(DEFAULT_SETTINGS["charts_bt_dir"])

    include_bt_charts_value = pick("include_bt_charts")
    include_bt_charts = (
        _parse_bool(include_bt_charts_value)
        if include_bt_charts_value is not None
        else bool(DEFAULT_SETTINGS["include_bt_charts"])
    )

    bt_title = str(pick("bt_title") or DEFAULT_SETTINGS["bt_title"])

    return {
        "tickers": tickers,
        "period": period,
        "interval": interval,
        "price_column": price_column,
        "gap_threshold": gap_threshold,
        "out_dir": out_dir,
        "format": file_format,
        "base_name": base_name,
        "save": save,
        "log_level": log_level,
        "score": score,
        "top": top,
        "report": report,
        "html": html,
        "charts_dir": charts_dir,
        "include_charts": include_charts,
        "report_title": report_title,
        "benchmark": benchmark,
        "bt": bt,
        "strategy": strategy,
        "capital": capital,
        "cost_bps": cost_bps,
        "slippage_bps": slippage_bps,
        "max_positions": max_positions,
        "stop_pct": stop_pct,
        "tp_pct": tp_pct,
        "bt_report": bt_report,
        "charts_bt_dir": charts_bt_dir,
        "include_bt_charts": include_bt_charts,
        "bt_title": bt_title,
    }


def _summarise_prices(
    prices,
    price_column: str = "Close",
    *,
    rsi_period: int = 14,
) -> Dict[str, object]:
    if getattr(prices, "empty", True):
        return {
            "timestamp": None,
            "close": "n/a",
            "rsi": "n/a",
            "macd_hist": "n/a",
            "moving_averages": "",
            "rsi_period": rsi_period,
        }

    latest_timestamp = prices.index[-1]
    ma_columns = [col for col in prices.columns if col.startswith("EMA") or col.startswith("SMA")]
    ma_snapshot_parts = []
    for column in ma_columns:
        try:
            value = prices[column].iloc[-1]
        except Exception:
            value = None
        ma_snapshot_parts.append(f"{column}={_format_number(value)}")
    ma_snapshot = ", ".join(ma_snapshot_parts)

    rsi_column = f"RSI{rsi_period}"
    if rsi_column in prices.columns:
        try:
            rsi_raw = prices[rsi_column].iloc[-1]
        except Exception:
            rsi_raw = None
        rsi_value = _format_number(rsi_raw)
    else:
        rsi_value = "n/a"

    if "MACD_hist" in prices.columns:
        try:
            macd_raw = prices["MACD_hist"].iloc[-1]
        except Exception:
            macd_raw = None
        macd_hist = _format_number(macd_raw)
    else:
        macd_hist = "n/a"

    return {
        "timestamp": latest_timestamp,
        "close": _format_number(prices[price_column].iloc[-1] if price_column in prices.columns else None),
        "rsi": rsi_value,
        "macd_hist": macd_hist,
        "moving_averages": ma_snapshot,
        "rsi_period": rsi_period,
    }


def _summarise_quality(quality: Dict[str, object]) -> str:
    duplicates = _format_int(quality.get("duplicates", {}).get("count")) if isinstance(quality, dict) else "n/a"
    ohlc = _format_int(quality.get("ohlc_anomalies", {}).get("count")) if isinstance(quality, dict) else "n/a"
    gaps = _format_int(quality.get("gaps", {}).get("count")) if isinstance(quality, dict) else "n/a"
    threshold = quality.get("gaps", {}).get("threshold_pct") if isinstance(quality, dict) else None
    threshold_display = "?"
    if threshold is not None:
        try:
            threshold_display = f"{float(threshold):g}"
        except (TypeError, ValueError):
            threshold_display = "?"
    return f"dups={duplicates}, ohlc={ohlc}, gaps={gaps}>{threshold_display}%"


def _collect_scores(results: Dict[str, Dict[str, object]]) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    for ticker, payload in results.items():
        if not isinstance(payload, dict):
            continue
        bundle = payload.get("score")
        if not isinstance(bundle, dict):
            continue
        notes = bundle.get("notes") if isinstance(bundle.get("notes"), list) else []
        rows.append(
            {
                "Ticker": ticker,
                "Score": bundle.get("score"),
                "Trend": bundle.get("trend"),
                "Momentum": bundle.get("momentum"),
                "Quality": bundle.get("quality"),
                "Risk": bundle.get("risk"),
                "As Of": bundle.get("as_of"),
                "NotesCount": len(notes),
                "Notes": notes,
            }
        )
    return rows


def _render_scoreboard(rows: list[Dict[str, object]], top: int) -> str:
    if not rows:
        return "Aucun score disponible."

    def _score_key(row: Dict[str, object]) -> float:
        try:
            return float(row.get("Score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    sorted_rows = sorted(rows, key=_score_key, reverse=True)
    display_rows = sorted_rows[: max(1, top)]

    headers = ["Ticker", "Score", "Trend", "Momentum", "Quality", "Risk", "As Of", "NotesCount"]
    widths = {
        header: max(len(header), *(len(str(row.get(header, ""))) for row in display_rows))
        for header in headers
    }

    def render_row(row: Dict[str, object]) -> str:
        cells = []
        for header in headers:
            value = row.get(header)
            if header in {"Score", "Trend", "Momentum", "Quality", "Risk"}:
                cells.append(f"{_format_number(value):>{widths[header]}}")
            else:
                cells.append(f"{str(value):<{widths[header]}}")
        return " | ".join(cells)

    header_row = " | ".join(f"{header:<{widths[header]}}" for header in headers)
    separator = "-+-".join("-" * widths[header] for header in headers)
    body = "\n".join(render_row(row) for row in display_rows)
    return f"{header_row}\n{separator}\n{body}"


def _render_summary_table(rows: list[Dict[str, object]]) -> str:
    if not rows:
        return "Aucune donnée de synthèse."

    headers = [
        "Ticker",
        "Score",
        "Trend",
        "Momentum",
        "Quality",
        "Risk",
        "AsOf",
        "Dups",
        "OHLC_Anom",
        "Gaps",
        "MissingFundamentals",
    ]

    widths = {
        header: max(len(header), *(len(str(row.get(header, ""))) for row in rows))
        for header in headers
    }

    def render_row(row: Dict[str, object]) -> str:
        cells = []
        for header in headers:
            value = row.get(header)
            if header in {"Score", "Trend", "Momentum", "Quality", "Risk"}:
                cells.append(f"{_format_number(value):>{widths[header]}}")
            else:
                cells.append(f"{str(value):<{widths[header]}}")
        return " | ".join(cells)

    header_row = " | ".join(f"{header:<{widths[header]}}" for header in headers)
    separator = "-+-".join("-" * widths[header] for header in headers)
    body = "\n".join(render_row(row) for row in rows)
    return f"{header_row}\n{separator}\n{body}"


def _rows_from_frame(frame) -> list[Dict[str, object]]:
    if frame is None or getattr(frame, "empty", True):
        return []
    columns = getattr(frame, "columns", [])
    length = len(frame)
    rows: list[Dict[str, object]] = []
    for position in range(length):
        row: Dict[str, object] = {}
        for column in columns:
            try:
                series = frame[column]
                value = series.iloc[position]
            except Exception:
                value = None
            row[str(column)] = value
        rows.append(row)
    return rows


def _format_backtest_metrics(metrics: Dict[str, object]) -> str:
    fields = [
        ("CAGR", True),
        ("Vol", True),
        ("Sharpe", False),
        ("MaxDD", True),
        ("Calmar", False),
        ("HitRate", True),
        ("AvgWin", True),
        ("AvgLoss", True),
        ("Payoff", False),
        ("ExposurePct", True),
    ]
    parts = []
    for key, is_percent in fields:
        value = metrics.get(key)
        if is_percent:
            parts.append(f"{key}={_format_percent(value)}")
        else:
            parts.append(f"{key}={_format_number(value)}")
    return ", ".join(parts)


def _format_trade_entry(row: Dict[str, object]) -> str:
    ticker = row.get("ticker", "?")
    entry = row.get("entry_date")
    exit_date = row.get("exit_date")
    entry_str = entry.isoformat() if isinstance(entry, datetime) else str(entry)
    exit_str = exit_date.isoformat() if isinstance(exit_date, datetime) else str(exit_date)
    ret_display = _format_percent(row.get("ret"))
    pnl_display = _format_number(row.get("pnl"))
    return f"{ticker} | {entry_str} -> {exit_str} | ret={ret_display} | pnl={pnl_display}"

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse technique et fondamentale de plusieurs actions.")
    parser.set_defaults(save=None, report=None, html=None, include_charts=None, bt_report=None, include_bt_charts=None)
    parser.add_argument("--tickers", help="Liste de tickers séparés par des virgules")
    parser.add_argument("--period", help="Fenêtre de téléchargement (ex: 2y)")
    parser.add_argument("--interval", help="Granularité des données (ex: 1d)")
    parser.add_argument("--price-column", dest="price_column", help="Colonne de prix à utiliser (Close ou Adj Close)")
    parser.add_argument("--gap-threshold", dest="gap_threshold", type=float, help="Seuil de gap en pourcentage")
    parser.add_argument("--out-dir", dest="out_dir", help="Répertoire de sortie pour la sauvegarde")
    parser.add_argument("--format", choices=["parquet", "csv"], help="Format de sauvegarde des prix")
    parser.add_argument("--base-name", dest="base_name", help="Préfixe des fichiers générés")
    parser.add_argument("--save", dest="save", action="store_true", help="Active la persistance des résultats")
    parser.add_argument("--no-save", dest="save", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--log-level", dest="log_level", choices=["INFO", "WARN", "ERROR"], help="Niveau de logs")
    parser.add_argument("--score", dest="score", action="store_true", help="Affiche le tableau des scores")
    parser.add_argument("--no-score", dest="score", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--top", dest="top", type=int, help="Nombre de valeurs à afficher dans le tableau des scores")
    parser.add_argument("--report", dest="report", action="store_true", help="Génère un rapport Markdown")
    parser.add_argument("--no-report", dest="report", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--html", dest="html", action="store_true", help="Génère aussi une version HTML du rapport")
    parser.add_argument("--report-title", dest="report_title", help="Titre du rapport")
    parser.add_argument("--charts-dir", dest="charts_dir", help="Sous-répertoire pour les graphiques (par défaut: charts)")
    parser.add_argument("--no-charts", dest="include_charts", action="store_false", help="Désactive les graphiques dans le rapport")
    parser.add_argument("--benchmark", dest="benchmark", help="Indice de référence pour comparaison (ex: ^FCHI)")
    parser.add_argument("--bt", dest="bt", action="store_true", help="Active le backtest EOD")
    parser.add_argument("--no-bt", dest="bt", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument(
        "--strategy",
        choices=["sma200_trend", "rsi_rebound", "macd_cross"],
        help="Stratégie de signal pour le backtest",
    )
    parser.add_argument("--capital", dest="capital", type=float, help="Capital initial du backtest")
    parser.add_argument("--cost-bps", dest="cost_bps", type=float, help="Coût en points de base par transaction")
    parser.add_argument(
        "--slippage-bps",
        dest="slippage_bps",
        type=float,
        help="Glissement en points de base appliqué au prix d'exécution",
    )
    parser.add_argument("--max-positions", dest="max_positions", type=int, help="Nombre maximal de positions simultanées")
    parser.add_argument(
        "--stop-pct",
        dest="stop_pct",
        type=float,
        help="Stop-loss en pourcentage (ex: 0.1 pour 10%)",
    )
    parser.add_argument(
        "--tp-pct",
        dest="tp_pct",
        type=float,
        help="Take-profit en pourcentage (ex: 0.2 pour 20%)",
    )
    parser.add_argument("--bt-report", dest="bt_report", action="store_true", help="Ajoute un bloc backtest au rapport")
    parser.add_argument("--no-bt-report", dest="bt_report", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--charts-bt-dir", dest="charts_bt_dir", help="Sous-répertoire pour les graphiques backtest")
    parser.add_argument("--no-bt-charts", dest="include_bt_charts", action="store_false", help="Désactive les graphiques backtest")
    parser.add_argument("--bt-title", dest="bt_title", help="Titre du bloc backtest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    settings = _resolve_settings(args)

    logging.basicConfig(level=getattr(logging, settings["log_level"], logging.INFO), format=LOG_FORMAT)

    LOGGER.info(
        "Configuration utilisée",
        extra={key: value for key, value in settings.items() if key != "tickers"},
    )

    try:
        results = analyze_tickers(
            settings["tickers"],
            period=settings["period"],
            interval=settings["interval"],
            price_column=settings["price_column"],
            gap_threshold_pct=settings["gap_threshold"],
        )
    except Exception as exc:  # pragma: no cover - runtime/network failures
        LOGGER.error(
            "Échec de l'analyse. Vérifiez votre connexion internet ou exécutez `python -m unittest` pour une validation hors-ligne.",
            exc_info=exc,
        )
        return 1

    if not results:
        LOGGER.error("Aucune analyse n'a pu être menée. Vérifiez les tickers ou la connexion réseau.")
        return 1

    report_output_dir = Path(settings["out_dir"])
    charts_dir_path: Path | None = None
    charts_dir_rel: str | None = None
    if settings["report"]:
        report_output_dir.mkdir(parents=True, exist_ok=True)
        if settings["include_charts"]:
            raw_charts_dir = Path(settings["charts_dir"])
            charts_dir_path = raw_charts_dir if raw_charts_dir.is_absolute() else report_output_dir / raw_charts_dir
            charts_dir_path.mkdir(parents=True, exist_ok=True)
            try:
                charts_dir_rel = os.path.relpath(charts_dir_path, report_output_dir)
            except ValueError:
                charts_dir_rel = str(charts_dir_path)

    score_rows = _collect_scores(results)
    generated_charts: list[str] = []
    backtest_result: Dict[str, object] | None = None
    bt_generated_charts: list[str] = []
    bt_chart_references: Dict[str, str] = {}
    backtest_kpis: Dict[str, object] = {}
    backtest_drawdown = pd.DataFrame()
    benchmark_label = settings["benchmark"].strip() or None
    benchmark_series = None

    bt_charts_dir_path: Path | None = None
    bt_charts_dir_rel: str | None = None
    if settings["bt_report"] and settings["include_bt_charts"]:
        report_output_dir.mkdir(parents=True, exist_ok=True)
        raw_bt_dir = Path(settings["charts_bt_dir"])
        bt_charts_dir_path = raw_bt_dir if raw_bt_dir.is_absolute() else report_output_dir / raw_bt_dir
        bt_charts_dir_path.mkdir(parents=True, exist_ok=True)
        if settings["report"]:
            try:
                bt_charts_dir_rel = os.path.relpath(bt_charts_dir_path, report_output_dir)
            except ValueError:
                bt_charts_dir_rel = str(bt_charts_dir_path)
        else:
            bt_charts_dir_rel = str(bt_charts_dir_path)

    for ticker, payload in results.items():
        prices = payload.get("prices")
        fundamentals = payload.get("fundamentals", {})
        quality = payload.get("quality", {})
        score_bundle = payload.get("score", {})
        report_section: Dict[str, object] | None = None
        report_notes: list[str] = []

        if settings["report"]:
            existing_report = payload.get("report") if isinstance(payload, dict) else None
            if isinstance(existing_report, dict):
                report_section = existing_report
            else:
                report_section = {}
                if isinstance(payload, dict):
                    payload["report"] = report_section
            notes_candidate = report_section.get("notes") if isinstance(report_section, dict) else []
            if isinstance(notes_candidate, list):
                report_notes = notes_candidate
            else:
                report_notes = []
                report_section["notes"] = report_notes

            if prices is not None:
                required_indicators = ["SMA20", "SMA50", "SMA200", "EMA21"]
                missing_indicators = [col for col in required_indicators if col not in getattr(prices, "columns", [])]
                if missing_indicators:
                    report_notes.append("indicateurs absents: " + ", ".join(missing_indicators))

    
        price_summary = _summarise_prices(prices, price_column=settings["price_column"])
        quality_summary = _summarise_quality(quality)

        timestamp = price_summary["timestamp"]
        formatted_timestamp = f"{timestamp:%Y-%m-%d %H:%M %Z}" if timestamp is not None else "n/a"

        LOGGER.info(
            "%s — dernière=%s (%s) RSI%s=%s MACD_hist=%s | QC %s",
            ticker,
            price_summary["close"],
            formatted_timestamp,
            price_summary["rsi_period"],
            price_summary["rsi"],
            price_summary["macd_hist"],
            quality_summary,
        )

        print("=" * 80)
        print(f"Ticker : {ticker}")
        print(
            "Dernière clôture ({ts}): {close} | RSI{period}={rsi} | MACD_hist={macd}".format(
                ts=formatted_timestamp,
                close=price_summary["close"],
                period=price_summary["rsi_period"],
                rsi=price_summary["rsi"],
                macd=price_summary["macd_hist"],
            )
        )
        if price_summary["moving_averages"]:
            print(f"Moyennes mobiles: {price_summary['moving_averages']}")
        print(
            "EPS={eps}, PER={pe}, Marge nette={margin}, D/E={de}, Rendement Div.={div} (en %)".format(
                eps=_format_number(fundamentals.get("eps")),
                pe=_format_number(fundamentals.get("pe_ratio")),
                margin=_format_number(fundamentals.get("net_margin_pct")),
                de=_format_number(fundamentals.get("debt_to_equity")),
                div=_format_number(fundamentals.get("dividend_yield_pct")),
            )
        )
        as_of = fundamentals.get("as_of")
        if as_of is not None:
            print(f"Données fondamentales au {as_of:%Y-%m-%d %H:%M %Z}")
        print("Qualité:", quality_summary, "| tz=", quality.get("timezone"))
        print("Provenance :", fundamentals.get("source_fields"))
        if isinstance(score_bundle, dict) and score_bundle:
            notes_count = len(score_bundle.get("notes", [])) if isinstance(score_bundle.get("notes"), list) else 0
            print(
                "Score global={score} | Trend={trend} Momentum={momentum} Quality={quality_score} Risk={risk} (notes={notes})".format(
                    score=_format_number(score_bundle.get("score")),
                    trend=_format_number(score_bundle.get("trend")),
                    momentum=_format_number(score_bundle.get("momentum")),
                    quality_score=_format_number(score_bundle.get("quality")),
                    risk=_format_number(score_bundle.get("risk")),
                    notes=notes_count,
                )
            )

        if settings["report"] and report_section is not None:
            if settings["include_charts"] and charts_dir_path is not None:
                if getattr(prices, "empty", True):
                    report_notes.append("graphique indisponible (données vides)")
                else:
                    chart_start = perf_counter()
                    fig = None
                    try:
                        fig = plot_ticker(prices, price_column=settings["price_column"])
                        buffer = io.BytesIO()
                        try:
                            fig.savefig(buffer, format="png", bbox_inches="tight")
                            image_bytes = buffer.getvalue()
                        finally:
                            buffer.close()

                        chart_filename = f"{settings['base_name']}_{_safe_ticker_name(ticker)}.png"
                        report_section["chart_filename"] = chart_filename
                        report_section["chart_data_url"] = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")

                        try:
                            save_price_figure(fig, charts_dir_path / chart_filename)
                            generated_charts.append(str(charts_dir_path / chart_filename))
                        except Exception as exc:  # pragma: no cover - filesystem failure
                            LOGGER.warning(
                                "Impossible d'enregistrer le graphique",
                                extra={"ticker": ticker, "path": str(charts_dir_path / chart_filename)},
                                exc_info=exc,
                            )

                        chart_duration = (perf_counter() - chart_start) * 1000.0
                        LOGGER.info(
                            "Graphique généré",
                            extra={"ticker": ticker, "duration_ms": round(chart_duration, 2)},
                        )
                    except Exception as exc:  # pragma: no cover - matplotlib/runtime failures
                        LOGGER.warning(
                            "Impossible de générer le graphique",
                            extra={"ticker": ticker},
                            exc_info=exc,
                        )
                        report_notes.append("graphique indisponible")
                    finally:
                        if "fig" in locals() and fig is not None:
                            try:
                                from matplotlib import pyplot as _plt  # type: ignore

                                _plt.close(fig)
                            except Exception:  # pragma: no cover - optional backend cleanup
                                pass

    if settings["bt"]:
        try:
            backtest_start = perf_counter()
            backtest_result = run_backtest(
                results,
                strategy=settings["strategy"],
                capital=settings["capital"],
                weight_scheme="equal",
                max_positions=settings["max_positions"],
                cost_bps=settings["cost_bps"],
                slippage_bps=settings["slippage_bps"],
                stop_loss_pct=settings["stop_pct"],
                take_profit_pct=settings["tp_pct"],
            )
            duration_ms = (perf_counter() - backtest_start) * 1000.0
            LOGGER.info(
                "Backtest exécuté",
                extra={"duration_ms": round(duration_ms, 2), "strategy": settings["strategy"]},
            )
        except Exception as exc:  # pragma: no cover - runtime failures
            LOGGER.error("Échec du backtest", exc_info=exc)
            backtest_result = None
        else:
            metrics = backtest_result.get("metrics") if isinstance(backtest_result, dict) else {}
            if not isinstance(metrics, dict):
                metrics = {}
            print("\nRésumé backtest :")
            print(_format_backtest_metrics(metrics))

            trade_rows = _rows_from_frame(backtest_result.get("trades") if isinstance(backtest_result, dict) else None)
            if trade_rows:
                def best_key(row: Dict[str, object]) -> float:
                    value = _coerce_float(row.get("ret"))
                    return value if value is not None else float("-inf")

                def worst_key(row: Dict[str, object]) -> float:
                    value = _coerce_float(row.get("ret"))
                    return value if value is not None else float("inf")

                top_trades = sorted(trade_rows, key=best_key, reverse=True)[:3]
                bottom_trades = sorted(trade_rows, key=worst_key)[:3]
                print("Top trades :")
                for row in top_trades:
                    print(" - " + _format_trade_entry(row))
                print("Pires trades :")
                for row in bottom_trades:
                    print(" - " + _format_trade_entry(row))

            try:
                backtest_kpis = summarize_backtest(backtest_result)
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.warning("Impossible de résumer le backtest", exc_info=exc)
                backtest_kpis = {}

            try:
                backtest_drawdown = compute_drawdown(backtest_result.get("equity"))  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover - runtime failures
                LOGGER.warning("Impossible de calculer le drawdown", exc_info=exc)
                backtest_drawdown = pd.DataFrame()

            equity_obj = backtest_result.get("equity") if isinstance(backtest_result, dict) else None
            if benchmark_label and benchmark_series is None:
                try:
                    benchmark_series = fetch_benchmark(
                        benchmark_label,
                        period=settings["period"],
                        interval=settings["interval"],
                        price_column=settings["price_column"],
                    )
                except Exception as exc:  # pragma: no cover - network/runtime failures
                    LOGGER.warning(
                        "Impossible de récupérer le benchmark",
                        extra={"benchmark": benchmark_label},
                        exc_info=exc,
                    )
                    benchmark_series = None

            equity_index: List[object] = []
            equity_values: List[float] = []
            if equity_obj is not None:
                if hasattr(equity_obj, "tolist"):
                    equity_values = list(equity_obj.tolist())  # type: ignore[attr-defined]
                    equity_index = list(getattr(equity_obj, "index", range(len(equity_values))))
                else:
                    equity_values = list(equity_obj)
                    equity_index = list(getattr(equity_obj, "index", range(len(equity_values))))

            equity_frame_for_plot = pd.DataFrame({"Strategy": equity_values}, index=equity_index)
            if benchmark_label and benchmark_series is not None and equity_obj is not None:
                try:
                    equity_frame_for_plot = attach_benchmark(
                        equity_obj,  # type: ignore[arg-type]
                        benchmark_df=benchmark_series,
                        label=benchmark_label,
                    )
                except Exception as exc:  # pragma: no cover - runtime failures
                    LOGGER.warning(
                        "Impossible d'aligner le benchmark",
                        extra={"benchmark": benchmark_label},
                        exc_info=exc,
                    )

            if settings["bt_report"] and settings["include_bt_charts"]:
                if not getattr(equity_frame_for_plot, "empty", True):
                    try:
                        chart_start = perf_counter()
                        fig_equity = plot_equity_with_benchmark(
                            equity_frame_for_plot,
                            title=f"{settings['bt_title']} — Equity",
                        )
                        if bt_charts_dir_path is not None:
                            chart_filename = f"{settings['base_name']}_bt_equity.png"
                            save_bt_figure(fig_equity, bt_charts_dir_path / chart_filename)
                            bt_generated_charts.append(str(bt_charts_dir_path / chart_filename))
                            if bt_charts_dir_rel:
                                reference = str(Path(bt_charts_dir_rel) / chart_filename)
                            else:
                                reference = str(bt_charts_dir_path / chart_filename)
                            bt_chart_references["equity"] = reference
                        else:
                            buffer = io.BytesIO()
                            try:
                                fig_equity.savefig(buffer, format="png", bbox_inches="tight")
                                bt_chart_references["equity"] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
                            finally:
                                buffer.close()
                        LOGGER.info(
                            "Graphique backtest généré",
                            extra={"type": "equity", "duration_ms": round((perf_counter() - chart_start) * 1000.0, 2)},
                        )
                    except Exception as exc:  # pragma: no cover - matplotlib/runtime failures
                        LOGGER.warning("Impossible de générer le graphique equity du backtest", exc_info=exc)
                    finally:
                        if "fig_equity" in locals() and fig_equity is not None:
                            try:
                                from matplotlib import pyplot as _plt  # type: ignore

                                _plt.close(fig_equity)
                            except Exception:  # pragma: no cover - optional cleanup
                                pass

                if not backtest_drawdown.empty:
                    try:
                        chart_start = perf_counter()
                        fig_dd = plot_drawdown(backtest_drawdown)
                        if bt_charts_dir_path is not None:
                            chart_filename = f"{settings['base_name']}_bt_drawdown.png"
                            save_bt_figure(fig_dd, bt_charts_dir_path / chart_filename)
                            bt_generated_charts.append(str(bt_charts_dir_path / chart_filename))
                            if bt_charts_dir_rel:
                                reference = str(Path(bt_charts_dir_rel) / chart_filename)
                            else:
                                reference = str(bt_charts_dir_path / chart_filename)
                            bt_chart_references["drawdown"] = reference
                        else:
                            buffer = io.BytesIO()
                            try:
                                fig_dd.savefig(buffer, format="png", bbox_inches="tight")
                                bt_chart_references["drawdown"] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
                            finally:
                                buffer.close()
                        LOGGER.info(
                            "Graphique backtest généré",
                            extra={"type": "drawdown", "duration_ms": round((perf_counter() - chart_start) * 1000.0, 2)},
                        )
                    except Exception as exc:  # pragma: no cover - matplotlib/runtime failures
                        LOGGER.warning("Impossible de générer le graphique drawdown", exc_info=exc)
                    finally:
                        if "fig_dd" in locals() and fig_dd is not None:
                            try:
                                from matplotlib import pyplot as _plt  # type: ignore

                                _plt.close(fig_dd)
                            except Exception:  # pragma: no cover - optional cleanup
                                pass

                positions_frame = backtest_result.get("positions") if isinstance(backtest_result, dict) else None
                if isinstance(positions_frame, pd.DataFrame) and not getattr(positions_frame, "empty", True):
                    try:
                        chart_start = perf_counter()
                        fig_heatmap = plot_exposure_heatmap(positions_frame)
                        if bt_charts_dir_path is not None:
                            chart_filename = f"{settings['base_name']}_bt_exposure.png"
                            save_bt_figure(fig_heatmap, bt_charts_dir_path / chart_filename)
                            bt_generated_charts.append(str(bt_charts_dir_path / chart_filename))
                            if bt_charts_dir_rel:
                                reference = str(Path(bt_charts_dir_rel) / chart_filename)
                            else:
                                reference = str(bt_charts_dir_path / chart_filename)
                            bt_chart_references["exposure"] = reference
                        else:
                            buffer = io.BytesIO()
                            try:
                                fig_heatmap.savefig(buffer, format="png", bbox_inches="tight")
                                bt_chart_references["exposure"] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
                            finally:
                                buffer.close()
                        LOGGER.info(
                            "Graphique backtest généré",
                            extra={"type": "exposure", "duration_ms": round((perf_counter() - chart_start) * 1000.0, 2)},
                        )
                    except Exception as exc:  # pragma: no cover - matplotlib/runtime failures
                        LOGGER.warning("Impossible de générer la heatmap d'exposition", exc_info=exc)
                    finally:
                        if "fig_heatmap" in locals() and fig_heatmap is not None:
                            try:
                                from matplotlib import pyplot as _plt  # type: ignore

                                _plt.close(fig_heatmap)
                            except Exception:  # pragma: no cover - optional cleanup
                                pass

    if backtest_result is not None:
        if backtest_kpis:
            backtest_result["kpis"] = backtest_kpis
        backtest_result["drawdown"] = backtest_drawdown
        if bt_chart_references or bt_generated_charts:
            charts_payload: Dict[str, object] = {}
            if bt_generated_charts:
                charts_payload["files"] = bt_generated_charts
            if bt_chart_references:
                charts_payload["references"] = bt_chart_references
            backtest_result["charts"] = charts_payload
        if benchmark_label:
            params = backtest_result.get("params") if isinstance(backtest_result, dict) else None
            if isinstance(params, dict):
                params.setdefault("benchmark", benchmark_label)
            elif isinstance(backtest_result, dict):
                backtest_result["params"] = {"benchmark": benchmark_label}

    if settings["save"]:
        try:
            saved_paths = save_analysis(
                results,
                out_dir=settings["out_dir"],
                base_name=settings["base_name"],
                format=settings["format"],
                backtest=backtest_result,
            )
        except Exception as exc:  # pragma: no cover - disk failures are runtime only
            LOGGER.error("Échec de la sauvegarde des résultats.", exc_info=exc)
            return 1

        print("\nFichiers écrits :")
        for path in saved_paths:
            print(f" - {path}")

    if settings["report"]:
        summary_rows = build_summary_table(results)
        print("\nRésumé synthétique :")
        print(_render_summary_table(summary_rows))

        markdown_start = perf_counter()
        try:
            markdown_report = render_markdown(
                results,
                title=settings["report_title"],
                include_charts=settings["include_charts"],
                charts_dir=charts_dir_rel if settings["include_charts"] else None,
            )
        except Exception as exc:  # pragma: no cover - markdown failures
            LOGGER.error("Échec de la génération du rapport Markdown.", exc_info=exc)
            return 1

        if settings["bt_report"] and backtest_result is not None:
            charts_for_report = bt_chart_references if settings["include_bt_charts"] else {}
            dd_for_report = backtest_drawdown if not backtest_drawdown.empty else pd.DataFrame()
            kpis_for_report = backtest_kpis or summarize_backtest(backtest_result)
            bt_block = render_bt_markdown(
                backtest_result,
                benchmark_name=benchmark_label,
                kpis=kpis_for_report,
                dd=dd_for_report,
                charts=charts_for_report,
            )
            if settings["bt_title"]:
                bt_block = f"## {settings['bt_title']}\n\n{bt_block}"
            markdown_report = markdown_report.rstrip() + "\n\n" + bt_block

        report_path = report_output_dir / f"{settings['base_name']}_report.md"
        try:
            with report_path.open("w", encoding="utf-8") as handle:
                handle.write(markdown_report)
            duration_ms = (perf_counter() - markdown_start) * 1000.0
            LOGGER.info(
                "Rapport Markdown écrit",
                extra={"path": str(report_path), "duration_ms": round(duration_ms, 2)},
            )
        except Exception as exc:  # pragma: no cover - filesystem failure
            LOGGER.error("Échec de l'écriture du rapport Markdown.", exc_info=exc)
            return 1

        print(f"\nRapport écrit : {report_path}")
        if settings["include_charts"] and generated_charts:
            print("Graphiques :")
            for chart_path in generated_charts:
                print(f" - {chart_path}")
        if settings["bt_report"] and settings["include_bt_charts"] and bt_generated_charts:
            print("Graphiques backtest :")
            for chart_path in bt_generated_charts:
                print(f" - {chart_path}")

        if settings["html"]:
            try:
                html_start = perf_counter()
                html_report = render_html(markdown_report)
                html_path = report_output_dir / f"{settings['base_name']}_report.html"
                with html_path.open("w", encoding="utf-8") as handle:
                    handle.write(html_report)
                duration_ms = (perf_counter() - html_start) * 1000.0
                LOGGER.info(
                    "Rapport HTML écrit",
                    extra={"path": str(html_path), "duration_ms": round(duration_ms, 2)},
                )
            except Exception as exc:  # pragma: no cover - html conversion failure
                LOGGER.error("Échec de la génération du rapport HTML.", exc_info=exc)
                return 1
            print(f"Rapport HTML : {html_path}")

    elif settings["bt_report"] and backtest_result is not None:
        report_output_dir.mkdir(parents=True, exist_ok=True)
        charts_for_report = bt_chart_references if settings["include_bt_charts"] else {}
        dd_for_report = backtest_drawdown if not backtest_drawdown.empty else pd.DataFrame()
        kpis_for_report = backtest_kpis or summarize_backtest(backtest_result)
        bt_block = render_bt_markdown(
            backtest_result,
            benchmark_name=benchmark_label,
            kpis=kpis_for_report,
            dd=dd_for_report,
            charts=charts_for_report,
        )
        if settings["bt_title"]:
            bt_block = f"## {settings['bt_title']}\n\n{bt_block}"
        report_path = report_output_dir / f"{settings['base_name']}_bt.md"
        try:
            with report_path.open("w", encoding="utf-8") as handle:
                handle.write(bt_block)
            LOGGER.info("Rapport backtest écrit", extra={"path": str(report_path)})
        except Exception as exc:  # pragma: no cover - filesystem failure
            LOGGER.error("Échec de l'écriture du rapport backtest.", exc_info=exc)
            return 1
        print(f"\nRapport backtest écrit : {report_path}")
        if settings["include_bt_charts"] and bt_generated_charts:
            print("Graphiques backtest :")
            for chart_path in bt_generated_charts:
                print(f" - {chart_path}")

        if settings["html"]:
            try:
                html_start = perf_counter()
                html_report = render_html(bt_block)
                html_path = report_output_dir / f"{settings['base_name']}_bt.html"
                with html_path.open("w", encoding="utf-8") as handle:
                    handle.write(html_report)
                duration_ms = (perf_counter() - html_start) * 1000.0
                LOGGER.info(
                    "Rapport backtest HTML écrit",
                    extra={"path": str(html_path), "duration_ms": round(duration_ms, 2)},
                )
            except Exception as exc:  # pragma: no cover - html conversion failure
                LOGGER.error("Échec de la génération du rapport backtest HTML.", exc_info=exc)
                return 1
            print(f"Rapport backtest HTML : {html_path}")

    if settings["score"]:
        print("\nTableau des scores (top {top}) :".format(top=settings["top"]))
        print(_render_scoreboard(score_rows, settings["top"]))

    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    raise SystemExit(main())
