import pandas as pd

from stock_analysis import __main__ as cli_main
from stock_analysis.cli import orotitan_ai
from stock_analysis.orotitan_ai.decision_engine import Decision


def test_cli_decide_json(monkeypatch, capsys) -> None:
    timestamp = pd.Timestamp("2024-01-01", tz="UTC")
    run = orotitan_ai.OroTitanRun(
        decisions=[
            Decision(
                date=timestamp,
                ticker="AAA",
                score=0.6,
                action="BUY",
                size=0.12,
                confidence=0.85,
                rationale="Action decided due to trend; risk budget 0.10, confidence 0.85.",
                factors={"trend": 0.5},
            )
        ],
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
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "decisions" in captured.out
    assert "AAA" in captured.out
