from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from stock_analysis.cli import orotitan_ai
from stock_analysis.orotitan_ai.decision_engine import Decision
from tests import pandas_stub


def _make_prices(rows: int = 128) -> pd.DataFrame:
    index = pandas_stub.date_range(
        datetime(2024, 1, 1, tzinfo=ZoneInfo("Europe/Paris")), periods=rows
    )
    base = list(range(rows))
    data = {
        "Open": [100 + value * 0.1 for value in base],
        "High": [100 + value * 0.1 + 1 for value in base],
        "Low": [100 + value * 0.1 - 1 for value in base],
        "Close": [100 + value * 0.1 for value in base],
        "Adj Close": [100 + value * 0.1 for value in base],
        "Volume": [1000 + value for value in base],
    }
    return pd.DataFrame(data, index=index)


def test_run_pipeline_generates_decisions(monkeypatch) -> None:
    prices = _make_prices()

    monkeypatch.setattr(
        orotitan_ai,
        "fetch_price_history",
        lambda ticker, **_: prices,
    )

    def fake_features(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ret_1d": 0.01,
                "vol_20": 0.02,
                "sma200_gap": 0.03,
                "rsi_14": 55.0,
                "mom_21": 0.05,
            },
            index=frame.index,
        )

    monkeypatch.setattr(orotitan_ai, "add_ta_features", fake_features)
    monkeypatch.setattr(
        orotitan_ai,
        "infer_regime_series",
        lambda index, provider=None: pd.Series(["Expansion"] * len(index), index=index),
    )

    class DummyAssessment:
        regime = "Expansion"
        snapshot = type("Snap", (), {})()

    monkeypatch.setattr(orotitan_ai, "evaluate_regime", lambda date: DummyAssessment())

    result = orotitan_ai.run_pipeline(
        ["AAA", "BBB"],
        regime_weights={"trend": 0.6, "momentum": 0.4},
        risk_budget=0.2,
    )

    assert len(result.decisions) == 2
    assert isinstance(result.decisions[0], Decision)
    assert result.weights

    # Ensure truthiness handling when passing explicit empty dict
    result_default = orotitan_ai.run_pipeline(["AAA"], regime_weights={})
    assert result_default.weights
