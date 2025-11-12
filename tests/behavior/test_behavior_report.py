from stock_analysis.orotitan_ai.behavior.schemas import BehaviorAnalysis, BiasSignal, BehaviorIndicator
from stock_analysis.orotitan_ai.decision_engine import Decision
from stock_analysis.report_orotitan import render_markdown_report

import pandas as pd


def _decision_with_behavior() -> Decision:
    timestamp = pd.Timestamp("2024-01-01", tz="UTC")
    behavior = BehaviorAnalysis(
        behavioral_score=62.0,
        top_biases=[
            BiasSignal(
                bias_id="sunk_cost",
                score=0.8,
                indicators=[BehaviorIndicator(name="add_to_losers", value=0.8)],
                rationale="Synthetic",
            )
        ],
        recommendations=["Limiter les moyennes à la baisse"],
        confidence_adjustment=-0.12,
        tags=["bias:sunk_cost"],
        metadata={},
    )
    return Decision(
        date=timestamp,
        ticker="AAA",
        score=0.2,
        action="HOLD",
        size=0.0,
        confidence=0.6,
        rationale="Action decided due to trend; risk budget 0.10, confidence 0.60.",
        factors={"trend": 0.2},
        behavior=behavior,
    )


def test_report_contains_behavioral_sections() -> None:
    decision = _decision_with_behavior()
    markdown = render_markdown_report(
        [decision],
        {"trend": 0.5, "momentum": 0.5},
        {"count": 1.0},
        title="Test Report",
    )
    assert "Behavioral Insights" in markdown
    assert "Self-Coaching Actions" in markdown
