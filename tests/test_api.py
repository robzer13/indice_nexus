from fastapi.testclient import TestClient

from stock_analysis.api import app as api_module
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi.testclient import TestClient

from stock_analysis import api as api_module
from tests import pandas_stub


def _build_frame(rows: int = 260) -> pd.DataFrame:
    index = pandas_stub.date_range(datetime(2024, 1, 1, tzinfo=ZoneInfo("Europe/Paris")), periods=rows)
    base = list(range(rows))
    data = {
        "Open": [100 + value * 0.1 for value in base],
        "High": [100 + value * 0.1 + 1 for value in base],
        "Low": [100 + value * 0.1 - 1 for value in base],
        "Close": [100 + value * 0.1 for value in base],
        "Adj Close": [100 + value * 0.1 for value in base],
        "Volume": [1000 + value for value in base],
    }
    frame = pd.DataFrame(data, index=index)
    return frame


def test_health_endpoint() -> None:
    client = TestClient(api_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert "version" in payload


def test_prices_endpoint(monkeypatch) -> None:
    frame = _build_frame()

    def fake_fetch(ticker: str, **_: object) -> pd.DataFrame:
        return frame

    monkeypatch.setattr(api_module, "fetch_price_history", fake_fetch)

    client = TestClient(api_module.app)
    response = client.get("/prices/MC.PA")
    payload = response.json()
    assert response.status_code == 200
    assert payload["ticker"] == "MC.PA"
    assert len(payload["data"]) == len(frame)
    assert "Close" in payload["data"][0]

    def fake_missing(ticker: str, **_: object) -> pd.DataFrame:
        raise ValueError(f"unknown ticker {ticker}")

    monkeypatch.setattr(api_module, "fetch_price_history", fake_missing)
    missing = client.get("/prices/FAKE")
    assert missing.status_code == 404


def test_ml_endpoint(monkeypatch) -> None:
    frame = _build_frame()

    def fake_fetch(ticker: str, **_: object) -> pd.DataFrame:
        return frame

    monkeypatch.setattr(api_module, "fetch_price_history", fake_fetch)

    monkeypatch.setattr(api_module, "time_cv", lambda X, y, model_kind, splits=5: (0.62, 0.03))

    def fake_walk_forward(X, y, retrain_every, warmup, model_kind, proba_threshold):
        effective_index = list(X.index)[-5:]
        proba = pd.Series([0.6] * len(effective_index), index=effective_index)
        signal = pd.Series([1.0] * len(effective_index), index=effective_index)
        return proba, signal

    monkeypatch.setattr(api_module, "walk_forward_signals", fake_walk_forward)
    monkeypatch.setattr(api_module, "sharpe_sim", lambda price, signal: 0.42)
    monkeypatch.setattr(
        api_module,
        "confusion",
        lambda y_true, y_prob, thr=0.5: {"tn": 10, "fp": 2, "fn": 1, "tp": 7},
    )

    client = TestClient(api_module.app)
    response = client.get("/ml/MC.PA?model=rf&horizon=5&threshold=0.0")
    payload = response.json()
    assert response.status_code == 200
    assert payload["model"] == "rf"
    assert payload["ticker"] == "MC.PA"
    assert payload["auc_mean"] == 0.62
    assert payload["confusion"]["tp"] == 7
    assert len(payload["signals"]) > 0


def test_report_endpoint(monkeypatch) -> None:
    dummy_prices = _build_frame(20)
    dummy_payload = {
        "prices": dummy_prices,
        "fundamentals": {"eps": 2.0, "pe_ratio": 15.0},
        "quality": {"duplicates": {"count": 0}},
        "score": {"score": 55.0, "trend": 20.0, "momentum": 15.0, "quality": 10.0, "risk": 10.0},
    }

    monkeypatch.setattr(api_module, "analyze_tickers", lambda *args, **kwargs: {"MC.PA": dummy_payload})

    client = TestClient(api_module.app)
    response = client.get("/report/MC.PA")
    payload = response.json()
    assert response.status_code == 200
    assert payload["ticker"] == "MC.PA"
    assert payload["score"]["score"] == 55.0
