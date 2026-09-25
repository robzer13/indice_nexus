"""Modular toolkit for downloading and analysing equity data."""

__version__ = "0.4.0"

from .analyzer import DEFAULT_TICKERS, analyze_tickers, fetch_benchmark
from .backtest import generate_signals, run_backtest
from .bt_report import attach_benchmark, compute_drawdown, render_bt_markdown, summarize_backtest
from .cache import load_cached_prices, store_cached_prices
from .data import CANONICAL_PRICE_COLUMNS, download_many, fetch_price_history, quality_report
from .fundamentals import fetch_fundamentals
from .indicators import compute_macd, compute_moving_averages, compute_rsi
from .io import save_analysis
from .features import add_ta_features, make_label_future_ret
from .ml_pipeline import (
    FEATURES,
    build_model,
    confusion,
    sharpe_sim,
    time_cv,
    walk_forward_signals,
)
from .plot import plot_ticker, save_figure as _save_price_figure
from .plot_bt import (
    plot_drawdown,
    plot_equity_with_benchmark,
    plot_exposure_heatmap,
    save_figure as _save_backtest_figure,
)
from .orotitan_ai import (
    Decision,
    FeedbackItem,
    apply_feedback as apply_orotitan_feedback,
    build_feature_dict,
    decide,
    encode_inputs,
    encode_snapshot,
    encode_text_optional,
    load_state,
    render_markdown_report as render_orotitan_markdown,
    save_state,
    update_weights as update_orotitan_weights,
)
from .regimes import (
    MacroDataProvider,
    MacroSnapshot,
    RegimeAssessment,
    RegimeThresholds,
    evaluate_regime,
    infer_regime_series,
)
from .report import build_summary_table, format_commentary, render_html, render_markdown
from .report_nexus import generate_nexus_report
from .report_orotitan import render_orotitan_report
from .regimes import RegimeThresholds, infer_regime
from .report import build_summary_table, format_commentary, render_html, render_markdown
from .scoring import (
    compute_score_bundle,
    compute_volatility,
    score_momentum,
    score_quality,
    score_risk,
    score_trend,
)
from .weighting import compute_weights

try:  # pragma: no cover - optional dependency
    from .scheduler import schedule_daily_run
except ModuleNotFoundError:  # pragma: no cover - APScheduler not installed
    def schedule_daily_run() -> None:
        raise RuntimeError("APScheduler must be installed to use the scheduler module")

save_price_figure = _save_price_figure
save_backtest_figure = _save_backtest_figure
save_figure = save_price_figure

__all__ = [
    "CANONICAL_PRICE_COLUMNS",
    "Decision",
    "DEFAULT_TICKERS",
    "FeedbackItem",
    "FEATURES",
    "MacroDataProvider",
    "MacroSnapshot",
    "RegimeAssessment",
    "RegimeThresholds",
    "add_ta_features",
    "analyze_tickers",
    "attach_benchmark",
    "apply_orotitan_feedback",
    "build_feature_dict",
    "build_model",
    "build_summary_table",
    "compute_drawdown",
    "compute_macd",
    "DEFAULT_TICKERS",
    "analyze_tickers",
    "attach_benchmark",
    "download_many",
    "compute_macd",
    "compute_drawdown",
    "compute_moving_averages",
    "compute_rsi",
    "compute_score_bundle",
    "compute_volatility",
    "compute_weights",
    "confusion",
    "decide",
    "download_many",
    "encode_inputs",
    "encode_snapshot",
    "encode_text_optional",
    "evaluate_regime",
    "fetch_benchmark",
    "fetch_fundamentals",
    "fetch_price_history",
    "format_commentary",
    "generate_nexus_report",
    "generate_signals",
    "infer_regime_series",
    "load_cached_prices",
    "load_state",
    "make_label_future_ret",
    "plot_drawdown",
    "plot_equity_with_benchmark",
    "plot_exposure_heatmap",
    "plot_ticker",
    "quality_report",
    "render_bt_markdown",
    "render_html",
    "render_markdown",
    "render_orotitan_markdown",
    "render_orotitan_report",
    "run_backtest",
    "save_analysis",
    "save_backtest_figure",
    "save_figure",
    "save_price_figure",
    "save_state",
    "schedule_daily_run",
    "FEATURES",
    "add_ta_features",
    "fetch_fundamentals",
    "fetch_price_history",
    "fetch_benchmark",
    "format_commentary",
    "build_summary_table",
    "build_model",
    "confusion",
    "generate_signals",
    "infer_regime",
    "make_label_future_ret",
    "load_cached_prices",
    "quality_report",
    "render_html",
    "render_markdown",
    "run_backtest",
    "plot_ticker",
    "plot_drawdown",
    "plot_equity_with_benchmark",
    "plot_exposure_heatmap",
    "score_momentum",
    "score_quality",
    "score_risk",
    "score_trend",
    "sharpe_sim",
    "store_cached_prices",
    "summarize_backtest",
    "time_cv",
    "update_orotitan_weights",
    "walk_forward_signals",
]
    "save_analysis",
    "save_figure",
    "save_price_figure",
    "save_backtest_figure",
    "schedule_daily_run",
    "store_cached_prices",
    "RegimeThresholds",
    "summarize_backtest",
    "render_bt_markdown",
    "sharpe_sim",
    "time_cv",
    "walk_forward_signals",
]

