"""Fail-closed checks: missing macro observations cannot silently become Expansion."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from stock_analysis.regimes import (
    MacroDataProvider,
    MacroDataUnavailable,
    MacroSnapshot,
    RegimeAssessment,
    evaluate_regime,
    infer_regime_series,
)


def test_macro_provider_requires_verified_observation() -> None:
    instant = pd.Timestamp(datetime(2024, 4, 1, tzinfo=timezone.utc))
    with pytest.raises(MacroDataUnavailable):
        evaluate_regime(instant)


def test_verified_macro_observation_and_report_payload() -> None:
    instant = pd.Timestamp(datetime(2024, 4, 1, tzinfo=timezone.utc))
    snap = MacroSnapshot(date=instant, vix=35.0, credit_spread=3.0)
    provider = MacroDataProvider({instant: snap})
    result = evaluate_regime(instant, provider=provider)
    assert isinstance(result, RegimeAssessment)
    assert result.regime == "Stress"
    assert result.to_dict()["snapshot"]["vix"] == 35.0


def test_empty_snapshot_never_defaults_to_expansion() -> None:
    instant = pd.Timestamp(datetime(2024, 4, 1, tzinfo=timezone.utc))
    provider = MacroDataProvider({instant: MacroSnapshot(date=instant)})
    with pytest.raises(MacroDataUnavailable):
        infer_regime_series([instant], provider=provider)
