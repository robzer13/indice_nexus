"""Streamlit dashboard for interactive exploration of the analysis pipeline."""
from __future__ import annotations

import logging
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import streamlit as st

try:  # pragma: no cover - optional dependency
    import plotly.express as px
except Exception:  # pragma: no cover - plotly not available
    px = None

from .analyzer import analyze_tickers
from .cache import DEFAULT_CACHE_DIR
from .features import add_ta_features, make_label_future_ret
from .ml_pipeline import FEATURES, confusion, sharpe_sim, time_cv, walk_forward_signals
from .report import render_html, render_markdown

LOGGER = logging.getLogger("stock_analysis.dashboard")
DEFAULT_TICKERS = ["MC.PA", "ASML", "TTE.PA"]


def _clear_cache(directory: Path = DEFAULT_CACHE_DIR) -> None:
    try:
        if directory.exists():
            shutil.rmtree(directory)
            LOGGER.info("Cache cleared", extra={"directory": str(directory)})
    except Exception as exc:  # pragma: no cover - filesystem specific
        LOGGER.warning("Unable to clear cache", exc_info=exc)


def _select_price_series(prices: pd.DataFrame, price_column: str) -> pd.Series:
    if "Adj Close" in prices.columns:
        return prices["Adj Close"]
    if price_column in prices.columns:
        return prices[price_column]
    if "Close" in prices.columns:
        return prices["Close"]
    raise KeyError("No price column available for ML evaluation")


def _compute_ml_metrics(
    prices: pd.DataFrame,
    *,
    model: str,
    horizon: int,
    threshold: float,
    retrain: int,
    proba_threshold: float,
    price_column: str,
) -> Dict[str, object] | None:
    try:
        features = add_ta_features(prices)
        labels = make_label_future_ret(prices, horizon=horizon, thr=threshold)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.error("Failed to compute ML features", exc_info=exc)
        return None

    feature_matrix = (
        features[FEATURES]
        .replace([float("inf"), float("-inf")], float("nan"))
        .dropna()
    )
    labels = labels.loc[feature_matrix.index].dropna()
    if feature_matrix.empty or labels.empty:
        return None

    try:
        auc_mean, auc_std = time_cv(feature_matrix, labels, model_kind=model, splits=5)
        warmup = max(10, min(len(feature_matrix), 200))
        proba_series, signal_series = walk_forward_signals(
            feature_matrix,
            labels,
            retrain_every=max(1, retrain),
            warmup=warmup,
            model_kind=model,
            proba_threshold=proba_threshold,
        )
        price_series = _select_price_series(prices, price_column)
        aligned_prices = price_series.loc[signal_series.index]
        sharpe = sharpe_sim(aligned_prices, signal_series)
        cm = confusion(labels.loc[proba_series.dropna().index], proba_series.dropna(), thr=0.5)
        positive_ratio = float(signal_series.dropna().mean()) if len(signal_series.dropna()) else 0.0
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("ML evaluation failed", exc_info=exc)
        return None

    return {
        "model": model,
        "horizon": horizon,
        "retrain_every": retrain,
        "proba_threshold": proba_threshold,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "sharpe_ml": sharpe,
        "confusion": cm,
        "positive_ratio": positive_ratio,
        "proba_series": proba_series,
        "signal_series": signal_series,
    }


def _render_price_chart(prices: pd.DataFrame, price_column: str) -> None:
    columns = [column for column in [price_column, "EMA7", "EMA20", "EMA200"] if column in prices.columns]
    if not columns:
        st.warning("Pas de colonnes de prix disponibles pour le graphique.")
        return
    chart_frame = prices[columns].dropna()
    if chart_frame.empty:
        st.warning("Pas de données suffisantes pour tracer le graphique de prix.")
        return
    display_frame = chart_frame.reset_index()
    display_frame.rename(columns={"index": "Date"}, inplace=True)
    if px is not None:
        figure = px.line(display_frame, x="Date", y=columns, title="Prix et moyennes mobiles")
        st.plotly_chart(figure, use_container_width=True)
    else:  # pragma: no cover - fallback path
        st.line_chart(chart_frame)


def _render_indicator_chart(prices: pd.DataFrame) -> None:
    columns = [column for column in ["RSI14", "MACD", "MACD_signal", "MACD_hist"] if column in prices.columns]
    if not columns:
        st.warning("Indicateurs indisponibles pour le graphique.")
        return
    indicator_frame = prices[columns].dropna()
    if indicator_frame.empty:
        st.warning("Pas de données suffisantes pour les indicateurs.")
        return
    display_frame = indicator_frame.reset_index()
    display_frame.rename(columns={"index": "Date"}, inplace=True)
    if px is not None:
        figure = px.line(display_frame, x="Date", y=[col for col in columns if col != "MACD_hist"], title="RSI & MACD")
        if "MACD_hist" in columns:
            hist = display_frame[["Date", "MACD_hist"]]
            hist.rename(columns={"MACD_hist": "Histogramme"}, inplace=True)
            figure.add_bar(x=hist["Date"], y=hist["Histogramme"], name="MACD_hist", opacity=0.3)
        st.plotly_chart(figure, use_container_width=True)
    else:  # pragma: no cover - fallback
        st.line_chart(indicator_frame)


def _render_score_block(score: Dict[str, object] | None) -> None:
    if not isinstance(score, dict):
        st.info("Score indisponible pour ce ticker.")
        return
    st.markdown(
        """
        **Scores**

        - Trend : {trend:.2f}
        - Momentum : {momentum:.2f}
        - Quality : {quality:.2f}
        - Risk : {risk:.2f}
        - Global : {score_value:.2f}
        """.format(
            trend=float(score.get("trend", math.nan)),
            momentum=float(score.get("momentum", math.nan)),
            quality=float(score.get("quality", math.nan)),
            risk=float(score.get("risk", math.nan)),
            score_value=float(score.get("score", math.nan)),
        )
    )


def _render_fundamentals_block(fundamentals: Dict[str, object] | None) -> None:
    if not isinstance(fundamentals, dict) or not fundamentals:
        st.info("Ratios fondamentaux indisponibles.")
        return
    rows = {
        "EPS": fundamentals.get("eps"),
        "P/E": fundamentals.get("pe_ratio"),
        "Marge nette (%)": fundamentals.get("net_margin_pct"),
        "Debt/Equity": fundamentals.get("debt_to_equity"),
        "Rendement dividende (%)": fundamentals.get("dividend_yield_pct"),
    }
    frame = pd.DataFrame({"Valeur": rows})
    st.dataframe(frame)


def _render_ml_block(ml_metrics: Dict[str, object] | None) -> None:
    if not isinstance(ml_metrics, dict):
        st.info("Aucune évaluation ML disponible pour ce ticker.")
        return
    st.markdown(
        """
        **Machine Learning**

        - Modèle : {model}
        - AUC (CV) : {auc:.3f} ± {std:.3f}
        - Sharpe simulé : {sharpe:.3f}
        - Ratio positions : {ratio:.1f}%
        """.format(
            model=ml_metrics.get("model", "n/a"),
            auc=float(ml_metrics.get("auc_mean", 0.0)),
            std=float(ml_metrics.get("auc_std", 0.0)),
            sharpe=float(ml_metrics.get("sharpe_ml", 0.0)),
            ratio=100.0 * float(ml_metrics.get("positive_ratio", 0.0)),
        )
    )
    confusion_matrix = ml_metrics.get("confusion") if isinstance(ml_metrics.get("confusion"), dict) else {}
    st.write("Confusion (seuil 0.50):", confusion_matrix)


def _render_feature_table(prices: pd.DataFrame, horizon: int, threshold: float) -> None:
    features = add_ta_features(prices)
    labels = make_label_future_ret(prices, horizon=horizon, thr=threshold)
    combined = features.copy()
    combined[labels.name] = labels
    st.dataframe(combined.tail(50))


def main() -> None:
    st.set_page_config(page_title="Indice Nexus Dashboard", layout="wide")
    st.title("Indice Nexus – Dashboard")

    with st.sidebar:
        tickers = st.multiselect("Tickers", DEFAULT_TICKERS, default=DEFAULT_TICKERS)
        period = st.selectbox("Période", ["6mo", "1y", "2y", "5y"], index=2)
        interval = st.selectbox("Intervalle", ["1d", "1wk", "1mo"], index=0)
        price_column = st.selectbox("Colonne de prix", ["Close", "Adj Close"], index=0)
        ml_model = st.selectbox("Modèle ML", ["xgb", "rf", "logreg"], index=0)
        ml_horizon = st.number_input("Horizon ML (jours)", value=5, min_value=1, step=1)
        proba_threshold = st.number_input("Seuil proba ML", value=0.55, min_value=0.0, max_value=1.0, step=0.05)
        retrain = st.number_input("Retrain (jours)", value=60, min_value=1, step=1)
        threshold = st.number_input("Seuil rendement futur", value=0.0, step=0.01, format="%.2f")
        analyse_clicked = st.button("Analyser")
        refresh_clicked = st.button("Rafraîchir le cache")

    if refresh_clicked:
        _clear_cache(Path(DEFAULT_CACHE_DIR))
        st.success("Cache supprimé. Relancez une analyse pour re-télécharger les données.")

    if not analyse_clicked:
        st.info("Sélectionnez des tickers puis cliquez sur *Analyser* pour lancer le pipeline.")
        return

    if not tickers:
        st.warning("Veuillez sélectionner au moins un ticker.")
        return

    try:
        results = analyze_tickers(
            tickers,
            period=period,
            interval=interval,
            price_column=price_column,
            use_cache=True,
        )
    except Exception as exc:  # pragma: no cover - network runtime
        LOGGER.error("Analyse échouée", exc_info=exc)
        st.error(f"Impossible de terminer l'analyse : {exc}")
        return

    report_payload: Dict[str, Dict[str, object]] = {}
    for ticker in tickers:
        payload = results.get(ticker)
        if not payload:
            st.warning(f"Aucune donnée pour {ticker}.")
            continue
        st.header(ticker)
        prices = payload.get("prices")
        if not isinstance(prices, pd.DataFrame) or prices.empty:
            st.warning("Série de prix vide pour ce ticker.")
            continue

        _render_price_chart(prices, price_column)
        _render_indicator_chart(prices)
        _render_score_block(payload.get("score"))
        _render_fundamentals_block(payload.get("fundamentals"))

        ml_metrics = _compute_ml_metrics(
            prices,
            model=ml_model,
            horizon=int(ml_horizon),
            threshold=float(threshold),
            retrain=int(retrain),
            proba_threshold=float(proba_threshold),
            price_column=price_column,
        )
        if ml_metrics:
            _render_ml_block(ml_metrics)
            payload.setdefault("ml", {}).update(ml_metrics)
        else:
            st.info("ML : données insuffisantes pour une évaluation fiable.")

        _render_feature_table(prices, int(ml_horizon), float(threshold))
        report_payload[ticker] = payload

    if not report_payload:
        st.warning("Aucune valeur n'a pu être analysée.")
        return

    try:
        markdown_report = render_markdown(report_payload, include_charts=False)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("Impossible de générer le rapport Markdown", exc_info=exc)
        markdown_report = "Rapport indisponible."

    st.download_button(
        "Télécharger le rapport Markdown",
        markdown_report,
        file_name=f"rapport_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
    )

    try:
        html_report = render_html(markdown_report)
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("Conversion HTML indisponible", exc_info=exc)
        html_report = None

    if isinstance(html_report, str) and html_report:
        st.download_button(
            "Télécharger le rapport HTML",
            html_report,
            file_name=f"rapport_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
        )


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["main"]
