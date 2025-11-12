from fastapi.testclient import TestClient

from stock_analysis.api import app as api_module


def test_behavior_analyze_endpoint() -> None:
    client = TestClient(api_module.app)
    payload = {
        "tickers": ["AAA"],
        "context": {"indicator_overrides": {"loss_hold_bias": 0.7, "drawdown_streak": 0.6}},
        "options": {"threshold": 40, "persist": "none"},
    }
    response = client.post("/ai/behavior/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "behavioral_score" in data
    assert "top_biases" in data and data["top_biases"]


def test_behavior_analyze_empty_defaults() -> None:
    client = TestClient(api_module.app)
    response = client.post("/ai/behavior/analyze", json={"tickers": ["BBB"]})
    assert response.status_code == 200
    data = response.json()
    assert data["behavioral_score"] >= 0
