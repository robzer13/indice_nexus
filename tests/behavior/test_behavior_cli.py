import json

import pandas as pd

from stock_analysis import __main__ as cli_main
from stock_analysis.cli import orotitan_ai
from stock_analysis.orotitan_ai.behavior.schemas import BehaviorAnalysis, BiasSignal, BehaviorIndicator
from stock_analysis.orotitan_ai.decision_engine import Decision


def test_cli_behavior_json(monkeypatch, capsys) -> None:
    timestamp = pd.Timestamp("2024-01-01", tz="UTC")
    behavior = BehaviorAnalysis(
        behavioral_score=55.0,
        top_biases=[
            BiasSignal(
                bias_id="loss_aversion",
                score=0.7,
                indicators=[BehaviorIndicator(name="loss_hold_bias", value=0.7)],
                rationale="Synthetic",
            )
        ],
        recommendations=["Limiter la taille des positions"],
        confidence_adjustment=-0.1,
        tags=["bias:loss_aversion"],
        metadata={},
    )
    decision = Decision(
        date=timestamp,
        ticker="AAA",
        score=0.6,
        action="BUY",
        size=0.12,
        confidence=0.85,
        rationale="Action decided due to trend; risk budget 0.10, confidence 0.85.",
        factors={"trend": 0.5},
        behavior=behavior,
    )
    run = orotitan_ai.OroTitanRun(
        decisions=[decision],
        embeddings={"AAA": {timestamp: [0.1, 0.2]}},
        weights={"trend": 0.7, "momentum": 0.3},
        regime="Expansion",
        as_of=timestamp,
        kpis={"count": 1.0},
        behavior_contexts={"AAA": {}},
    )

    monkeypatch.setattr(orotitan_ai, "run_pipeline", lambda *args, **kwargs: run)
    monkeypatch.setattr(orotitan_ai, "apply_feedback_events", lambda weights, events: (weights, []))

    exit_code = cli_main.main(
        [
            "--orotitan-ai",
            "--ai-task",
            "decide",
            "--tickers",
            "AAA",
            "--behavior",
            "--behavior-json",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "behavior" in captured.out
    payload = json.loads(captured.out.splitlines()[-1])
    assert payload["behavior"][0]["behavioral_score"] == 55.0
