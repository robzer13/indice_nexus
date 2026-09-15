from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from stock_analysis import streamlit_app
from tests import pandas_stub, streamlit_stub


def _build_dashboard_frame(rows: int = 220) -> pd.DataFrame:
    index = pandas_stub.date_range(datetime(2024, 1, 1, tzinfo=ZoneInfo("Europe/Paris")), periods=rows)
    base = list(range(rows))
    close_values = [150 + value * 0.2 for value in base]
    data = {
        "Open": close_values,
        "High": [value + 1 for value in close_values],
        "Low": [value - 1 for value in close_values],
        "Close": close_values,
        "Adj Close": close_values,
        "Volume": [5000 + value for value in base],
        "EMA7": close_values,
        "EMA20": close_values,
        "EMA200": close_values,
        "RSI14": [50.0 for _ in base],
        "MACD": [0.1 for _ in base],
        "MACD_signal": [0.05 for _ in base],
        "MACD_hist": [0.05 for _ in base],
    }
    return pd.DataFrame(data, index=index)


def test_streamlit_main_runs(monkeypatch) -> None:
    streamlit_stub.reset()
    streamlit_stub.configure(
        tickers=["MC.PA"],
        **{
            "Période": "1y",
            "Intervalle": "1d",
            "Colonne de prix": "Close",
            "Modèle ML": "xgb",
            "Horizon ML (jours)": 5,
            "Seuil proba ML": 0.55,
            "Retrain (jours)": 60,
            "Seuil rendement futur": 0.0,
            "button::Analyser": True,
        },
    )

    prices = _build_dashboard_frame()
    payload = {
        "prices": prices,
        "fundamentals": {"eps": 2.5, "pe_ratio": 18.0, "net_margin_pct": 12.0, "dividend_yield_pct": 2.0},
        "quality": {"duplicates": {"count": 0}},
        "score": {
            "score": 60.0,
            "trend": 25.0,
            "momentum": 15.0,
            "quality": 12.0,
            "risk": 8.0,
            "weights": {"trend": 0.35, "momentum": 0.3, "quality": 0.2, "risk": 0.15},
            "regime": "Expansion",
        },
        "score": {"score": 60.0, "trend": 25.0, "momentum": 15.0, "quality": 12.0, "risk": 8.0},
    }

    monkeypatch.setattr(streamlit_app, "analyze_tickers", lambda *args, **kwargs: {"MC.PA": payload})

    ml_result = {
        "model": "xgb",
        "horizon": 5,
        "retrain_every": 60,
        "proba_threshold": 0.55,
        "auc_mean": 0.61,
        "auc_std": 0.02,
        "sharpe_ml": 0.35,
        "positive_ratio": 0.4,
        "confusion": {"tn": 5, "fp": 2, "fn": 1, "tp": 6},
    }
    monkeypatch.setattr(streamlit_app, "_compute_ml_metrics", lambda *args, **kwargs: ml_result)
    monkeypatch.setattr(streamlit_app, "render_markdown", lambda *args, **kwargs: "# Rapport\n")
    monkeypatch.setattr(streamlit_app, "render_html", lambda markdown: "<h1>Rapport</h1>")

    streamlit_app.main()

    assert payload["score"]["score"] == 60.0
    assert payload.get("ml", {}).get("auc_mean") == 0.61
