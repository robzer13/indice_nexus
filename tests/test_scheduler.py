from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from stock_analysis import scheduler
from stock_analysis.regimes import MacroSnapshot, RegimeAssessment
from scripts import nexus_daily


def test_build_cli_arguments_includes_nexus_flags(monkeypatch) -> None:
    monkeypatch.setenv("NEXUS_TICKERS", "AAA")
    args = scheduler._build_cli_arguments()
    assert "--nexus-report" in args
    assert "--regime" in args


def test_nexus_daily_run_once(monkeypatch) -> None:
    timestamp = pd.Timestamp(datetime(2024, 6, 1, 9, 0), tz=ZoneInfo("Europe/Paris"))
    assessment = RegimeAssessment(
        regime="Expansion",
        snapshot=MacroSnapshot(
            date=timestamp,
            vix=18.0,
            cpi_yoy=3.0,
            rate_10y=2.5,
            rate_2y=1.5,
            credit_spread=1.0,
        ),
    )

    monkeypatch.setattr(nexus_daily, "evaluate_regime", lambda _: assessment)
    monkeypatch.setattr(nexus_daily, "compute_weights", lambda regime: {"trend": 0.4, "momentum": 0.3, "quality": 0.2, "risk": 0.1})
    monkeypatch.setattr(nexus_daily, "analyze_tickers", lambda *args, **kwargs: {"AAA": {}})
    monkeypatch.setattr(nexus_daily, "save_analysis", lambda *args, **kwargs: None)
    monkeypatch.setattr(nexus_daily, "generate_nexus_report", lambda *args, **kwargs: {"markdown": "report.md", "html": None})

    paths = nexus_daily.run_once()
    assert paths["markdown"] == "report.md"
    assert "html" in paths
