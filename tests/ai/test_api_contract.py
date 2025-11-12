from datetime import datetime
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi.testclient import TestClient

from stock_analysis.api.app import app
from stock_analysis.cli import orotitan_ai
from stock_analysis.orotitan_ai.decision_engine import Decision


def _dummy_pipeline(tickers, **_: object) -> orotitan_ai.OroTitanRun:
    timestamp = pd.Timestamp(datetime(2024, 1, 1, tzinfo=ZoneInfo("UTC")))
    decisions = [
        Decision(
            date=timestamp,
            ticker=tickers[0],
            score=0.55,
            action="BUY",
            size=0.1,
            confidence=0.8,
            rationale="Action decided due to trend; risk budget 0.10, confidence 0.80.",
            factors={"trend": 0.4},
        )
    ]
    embeddings = {tickers[0]: {timestamp: [0.1, 0.2, 0.3]}}
    return orotitan_ai.OroTitanRun(
        decisions=decisions,
        embeddings=embeddings,
        weights={"trend": 0.6, "momentum": 0.4},
        regime="Expansion",
        as_of=timestamp,
        kpis={"count": 1.0},
        behavior_contexts={tickers[0]: {}},
    )


def test_api_decide_and_report(monkeypatch) -> None:
    monkeypatch.setattr(orotitan_ai, "run_pipeline", _dummy_pipeline)
    monkeypatch.setattr(orotitan_ai, "load_state", lambda: {"weights": {"trend": 0.5}})
    monkeypatch.setattr(orotitan_ai, "apply_feedback_events", lambda weights, events: (weights, []))
    monkeypatch.setattr(orotitan_ai, "render_markdown_report", lambda *args, **kwargs: "# Test Report")
    monkeypatch.setattr(
        orotitan_ai,
        "render_orotitan_report",
        lambda *args, **kwargs: {"md": "report.md", "html": "report.html"},
    )

    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    decide_payload = client.post("/ai/decide", json={"tickers": ["AAA"]})
    assert decide_payload.status_code == 200
    body = decide_payload.json()
    assert body["decisions"][0]["ticker"] == "AAA"
    assert body["decisions"][0]["decision"] == "BUY"

    report_payload = client.post("/ai/report", json={"tickers": ["AAA"]})
    assert report_payload.status_code == 200
    assert "# Test Report" in report_payload.json()["markdown"]
