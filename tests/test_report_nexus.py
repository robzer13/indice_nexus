import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from stock_analysis.report_nexus import build_markdown, generate_nexus_report
from stock_analysis.regimes import MacroSnapshot, RegimeAssessment


def _sample_results() -> dict[str, dict]:
    index = pd.date_range(datetime(2024, 1, 1, 9, 0), periods=5, freq="D", tz="Europe/Paris")
    frame = pd.DataFrame(
        {
            "Open": [100 + i for i in range(5)],
            "High": [101 + i for i in range(5)],
            "Low": [99 + i for i in range(5)],
            "Close": [100 + i for i in range(5)],
            "Adj Close": [100 + i for i in range(5)],
            "Volume": [1000 + i for i in range(5)],
        },
        index=index,
    )
    score = {
        "score": 68.0,
        "trend": 32.0,
        "momentum": 20.0,
        "quality": 12.0,
        "risk": 4.0,
        "as_of": index[-1].isoformat(),
        "notes": [],
        "weights": {"trend": 0.4, "momentum": 0.3, "quality": 0.2, "risk": 0.1},
        "regime": "Expansion",
    }
    return {
        "AAA": {
            "prices": frame,
            "fundamentals": {"pe_ratio": 18.0, "net_margin_pct": 12.0, "debt_to_equity": 1.1, "dividend_yield_pct": 2.0},
            "quality": {"duplicates": {"count": 0}, "gaps": {"count": 0, "threshold_pct": 5.0}},
            "score": score,
        }
    }


def test_build_markdown_contains_sections() -> None:
    assessment = RegimeAssessment(
        regime="Expansion",
        snapshot=MacroSnapshot(
            date=pd.Timestamp.utcnow(),
            vix=18.0,
            cpi_yoy=3.0,
            rate_10y=2.5,
            rate_2y=1.5,
            credit_spread=1.0,
        ),
    )
    markdown = build_markdown(_sample_results(), assessment, {"trend": 0.4, "momentum": 0.3, "quality": 0.2, "risk": 0.1})
    assert "## Contexte macro" in markdown
    assert "## Top" in markdown
    assert "Recommandation" in markdown


def test_generate_nexus_report_creates_files() -> None:
    assessment = RegimeAssessment(
        regime="Expansion",
        snapshot=MacroSnapshot(
            date=pd.Timestamp.utcnow(),
            vix=18.0,
            cpi_yoy=3.0,
            rate_10y=2.5,
            rate_2y=1.5,
            credit_spread=1.0,
        ),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = generate_nexus_report(
            _sample_results(),
            assessment,
            {"trend": 0.4, "momentum": 0.3, "quality": 0.2, "risk": 0.1},
            output_dir=Path(tmpdir),
            include_html=True,
        )
        assert os.path.exists(paths["markdown"])
        assert paths["html"] is None or os.path.exists(paths["html"])
