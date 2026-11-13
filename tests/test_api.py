"""Offline regression tests for the current /ai/* API surface (no network)."""
from datetime import datetime, timezone
from importlib import import_module
from types import SimpleNamespace

from fastapi.testclient import TestClient

api_module = import_module("stock_analysis.api.app")


def client() -> TestClient:
    return TestClient(api_module.app)


def test_health_endpoint() -> None:
    response = client().get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    parsed = datetime.fromisoformat(payload["time"].replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


def test_behavior_endpoint_uses_explicit_context(monkeypatch) -> None:
    calls = []

    def fake_analyze(tickers, context, **kwargs):
        calls.append((tickers, context, kwargs))
        return SimpleNamespace(dict=lambda: {
            "behavioral_score": 23.0,
            "confidence_adjustment": -0.05,
            "top_biases": [{"type": "loss_aversion"}],
            "recommendations": ["Review risk limits"],
            "tags": ["behavior"],
        })

    monkeypatch.setattr(api_module, "analyze_behavior", fake_analyze)
    response = client().post("/ai/behavior/analyze", json={
        "tickers": ["MC.PA"], "context": {"source": "unit-test"},
        "options": {"threshold": 35.0, "max_influence": 0.2, "persist": "none"},
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["behavioral_score"] == 23.0
    assert payload["tags"] == ["behavior"]
    assert payload["run"]["duration_s"] >= 0
    assert calls == [(["MC.PA"], {"source": "unit-test"}, {
        "threshold": 35.0, "max_influence": 0.2, "persist": "none",
    })]


def test_embeddings_endpoint_serializes_timestamp_keys(monkeypatch) -> None:
    observed = []
    timestamp = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)

    def run_pipeline(tickers, **kwargs):
        observed.append((tickers, kwargs))
        return SimpleNamespace(embeddings={"MC.PA": {timestamp: [0.3, 0.7]}})

    monkeypatch.setattr(api_module, "_runtime_helpers", lambda: SimpleNamespace(
        run_pipeline=run_pipeline,
    ))
    response = client().post("/ai/embed", json={"tickers": ["MC.PA"]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["embeddings"]["MC.PA"][timestamp.isoformat()] == [0.3, 0.7]
    assert observed[0][0] == ["MC.PA"]
    assert observed[0][1]["risk_budget"] == 0.1


def test_decision_endpoint_converts_pipeline_records(monkeypatch) -> None:
    decision = SimpleNamespace(
        ticker="MC.PA", action="HOLD", score=56.5, rationale="Neutral",
        confidence=0.8, size=0.0, factors={"trend": 0.4}, behavior=None,
    )
    pipeline = SimpleNamespace(decisions=[decision], regime="neutral")
    monkeypatch.setattr(api_module, "_runtime_helpers", lambda: SimpleNamespace(
        run_pipeline=lambda *args, **kwargs: pipeline,
    ))
    response = client().post("/ai/decide", json={"tickers": ["MC.PA"]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["decisions"][0]["ticker"] == "MC.PA"
    assert payload["decisions"][0]["decision"] == "HOLD"
    assert payload["decisions"][0]["score"] == 56.5
    assert payload["decisions"][0]["meta"]["confidence"] == 0.8


def test_feedback_endpoint_handles_typed_events(monkeypatch) -> None:
    observed = []

    def apply_feedback(weights, events):
        observed.append((weights, events))
        return 1, ["recorded"]

    monkeypatch.setattr(api_module, "_runtime_helpers", lambda: SimpleNamespace(
        load_state=lambda: {"weights": {"trend": 1.0}},
        apply_feedback_events=apply_feedback,
    ))
    response = client().post("/ai/feedback", json={
        "events": [{"ticker": "MC.PA", "outcome": "hit", "horizon_days": 5}],
    })
    assert response.status_code == 200
    assert response.json()["updated"] == 1
    assert response.json()["notes"] == ["recorded"]
    assert observed[0][0] == {"trend": 1.0}
    assert observed[0][1][0]["ticker"] == "MC.PA"


def test_report_endpoint_uses_current_markdown_pipeline(monkeypatch) -> None:
    pipeline = SimpleNamespace(decisions=[], weights={}, kpis={"signal_count": 0})
    observed = []

    def render(decisions, weights, kpis, *, title):
        observed.append((decisions, weights, kpis, title))
        return "# " + title

    monkeypatch.setattr(api_module, "_runtime_helpers", lambda: SimpleNamespace(
        run_pipeline=lambda *args, **kwargs: pipeline,
        render_markdown_report=render,
    ))
    monkeypatch.delenv("SAVE_REPORTS", raising=False)
    response = client().post("/ai/report", json={
        "tickers": ["MC.PA"], "title": "C07 offline regression",
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["markdown"] == "# C07 offline regression"
    assert payload["saved_to"] is None
    assert observed == [([], {}, {"signal_count": 0}, "C07 offline regression")]


def test_pipeline_failure_stays_http_500(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic pipeline failure")

    monkeypatch.setattr(api_module, "_runtime_helpers", lambda: SimpleNamespace(
        run_pipeline=fail,
    ))
    response = client().post("/ai/decide", json={"tickers": ["MC.PA"]})
    assert response.status_code == 500
    assert "synthetic pipeline failure" in response.json()["detail"]
