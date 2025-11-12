from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from stock_analysis.orotitan_ai.decision_engine import Decision
from stock_analysis.report_orotitan import render_markdown_report, render_orotitan_report


def _decision(ticker: str) -> Decision:
    return Decision(
        date=pd.Timestamp(datetime(2024, 1, 1, tzinfo=ZoneInfo("UTC"))),
        ticker=ticker,
        score=0.42,
        action="BUY",
        size=0.05,
        confidence=0.8,
        rationale="Action decided due to trend; risk budget 0.10, confidence 0.80.",
        factors={"trend": 0.4, "momentum": 0.2},
    )


def test_render_markdown_contains_sections(tmp_path) -> None:
    decisions = [_decision("AAA"), _decision("BBB")]
    markdown = render_markdown_report(decisions, {"trend": 0.5, "momentum": 0.5}, {"count": 2.0})
    assert "# OroTitan" in markdown
    assert "AAA" in markdown

    paths = render_orotitan_report(
        decisions,
        {"trend": 0.5, "momentum": 0.5},
        {"count": 2.0},
        tmp_path,
        title="OroTitan Test Report",
    )
    md_content = paths["md"].read_text(encoding="utf-8")
    assert "OroTitan Test Report" in md_content
    assert "BBB" in md_content
