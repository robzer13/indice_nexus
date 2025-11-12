from datetime import datetime

import pandas as pd

from stock_analysis.regimes import (
    MacroSnapshot,
    RegimeAssessment,
    classify_snapshot,
    evaluate_regime,
    infer_regime_series,
)


class FixedProvider:
    """Simple provider returning precomputed snapshots for testing."""

    def __init__(self, mapping: dict[pd.Timestamp, MacroSnapshot]) -> None:
        self.mapping = mapping
        self.default = next(iter(mapping.values()))

    def snapshot(self, date: pd.Timestamp) -> MacroSnapshot:
        key = pd.Timestamp(date).tz_convert("UTC") if pd.Timestamp(date).tzinfo else pd.Timestamp(date, tz="UTC")
        return self.mapping.get(key, self.default)


def test_classify_stress() -> None:
    snapshot = MacroSnapshot(date=pd.Timestamp.utcnow(), vix=35.0, credit_spread=3.0)
    assert classify_snapshot(snapshot) == "Stress"


def test_classify_inflation() -> None:
    snapshot = MacroSnapshot(
        date=pd.Timestamp.utcnow(),
        vix=18.0,
        cpi_yoy=5.0,
        rate_10y=3.5,
    )
    assert classify_snapshot(snapshot) == "Inflation"


def test_classify_recovery() -> None:
    snapshot = MacroSnapshot(
        date=pd.Timestamp.utcnow(),
        vix=20.0,
        cpi_yoy=2.0,
        rate_10y=1.5,
        rate_2y=2.0,
    )
    assert classify_snapshot(snapshot) == "Recovery"


def test_classify_expansion_default() -> None:
    snapshot = MacroSnapshot(date=pd.Timestamp.utcnow(), vix=18.0, cpi_yoy=3.0, rate_10y=2.5, rate_2y=1.5)
    assert classify_snapshot(snapshot) == "Expansion"


def test_evaluate_regime_uses_provider() -> None:
    ts = pd.Timestamp(datetime(2024, 5, 1, 9, 0), tz="UTC")
    provider = FixedProvider({ts: MacroSnapshot(date=ts, vix=32.0, credit_spread=2.5)})
    assessment = evaluate_regime(ts, provider=provider)
    assert isinstance(assessment, RegimeAssessment)
    assert assessment.regime == "Stress"


def test_infer_regime_series_custom_provider() -> None:
    idx = pd.date_range(datetime(2024, 1, 1, 9, 0), periods=3, freq="D", tz="Europe/Paris")
    mapping = {
        idx[0].tz_convert("UTC"): MacroSnapshot(date=idx[0], vix=30.0),
        idx[1].tz_convert("UTC"): MacroSnapshot(date=idx[1], cpi_yoy=4.5, rate_10y=3.5),
        idx[2].tz_convert("UTC"): MacroSnapshot(date=idx[2], rate_10y=1.0, rate_2y=1.6),
    }
    provider = FixedProvider(mapping)
    series = infer_regime_series(idx, provider=provider)
    assert list(series) == ["Stress", "Inflation", "Recovery"]
