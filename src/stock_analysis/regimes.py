"""Simple macro regime heuristics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


@dataclass
class RegimeThresholds:
    stress_vix: float = 25.0
    inflation_cpi: float = 4.0


def infer_regime(
    index: Iterable[pd.Timestamp],
    *,
    vix: Optional[pd.Series] = None,
    cpi_yoy: Optional[pd.Series] = None,
    thresholds: RegimeThresholds | None = None,
) -> pd.Series:
    """Return a labelled Series using heuristic macro thresholds."""

    thresholds = thresholds or RegimeThresholds()
    index = list(index)
    labels = []
    for timestamp in index:
        vix_value = None
        if vix is not None:
            try:
                vix_value = float(vix.asof(timestamp))
            except Exception:
                vix_value = None
        cpi_value = None
        if cpi_yoy is not None:
            try:
                cpi_value = float(cpi_yoy.asof(timestamp))
            except Exception:
                cpi_value = None

        if vix_value is not None and vix_value >= thresholds.stress_vix:
            labels.append("Stress")
        elif cpi_value is not None and cpi_value >= thresholds.inflation_cpi:
            labels.append("Inflation")
        else:
            labels.append("Normal")

    return pd.Series(labels, index=index, name="regime")



class MacroDataUnavailable(RuntimeError):
    """A requested regime lacks an admitted macro snapshot; no synthetic fallback."""


@dataclass(frozen=True)
class MacroSnapshot:
    """Verified macro observations, with None for unavailable measures.

    No values are synthesized by this data structure.
    """

    date: pd.Timestamp
    vix: float | None = None
    cpi_yoy: float | None = None
    rate_10y: float | None = None
    rate_2y: float | None = None
    credit_spread: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "date": pd.Timestamp(self.date).isoformat(),
            "vix": self.vix,
            "cpi_yoy": self.cpi_yoy,
            "rate_10y": self.rate_10y,
            "rate_2y": self.rate_2y,
            "credit_spread": self.credit_spread,
        }


@dataclass(frozen=True)
class RegimeAssessment:
    regime: str
    snapshot: MacroSnapshot

    def to_dict(self) -> dict[str, object]:
        return {"regime": self.regime, "snapshot": self.snapshot.to_dict()}


class MacroDataProvider:
    """Read-only explicitly supplied observations; never generate market values."""

    def __init__(self, observations: dict[pd.Timestamp, MacroSnapshot] | None = None) -> None:
        self._observations = {
            pd.Timestamp(key): value for key, value in (observations or {}).items()
        }

    def snapshot(self, date: pd.Timestamp) -> MacroSnapshot:
        timestamp = pd.Timestamp(date)
        try:
            return self._observations[timestamp]
        except KeyError:
            raise MacroDataUnavailable(
                "Macro snapshot unavailable: supply an independently sourced observation"
            ) from None


def classify_snapshot(
    snapshot: MacroSnapshot, *, thresholds: RegimeThresholds | None = None
) -> str:
    """Apply the tested regime policy only to verified input observations.

    The existing stress/inflation thresholds are retained; the inverted
    10Y–2Y curve is the tested recovery marker when neither override applies.
    """
    policy = thresholds or RegimeThresholds()
    if all(
        getattr(snapshot, name) is None
        for name in ("vix", "cpi_yoy", "rate_10y", "rate_2y", "credit_spread")
    ):
        raise MacroDataUnavailable("Cannot classify a snapshot with no macro observations")
    if snapshot.vix is not None and snapshot.vix >= policy.stress_vix:
        return "Stress"
    if snapshot.cpi_yoy is not None and snapshot.cpi_yoy >= policy.inflation_cpi:
        return "Inflation"
    if (
        snapshot.rate_10y is not None
        and snapshot.rate_2y is not None
        and snapshot.rate_10y < snapshot.rate_2y
    ):
        return "Recovery"
    return "Expansion"


def evaluate_regime(
    date: pd.Timestamp, *, provider: object | None = None,
    thresholds: RegimeThresholds | None = None,
) -> RegimeAssessment:
    source = provider if provider is not None else MacroDataProvider()
    snapshot = source.snapshot(pd.Timestamp(date))
    return RegimeAssessment(
        regime=classify_snapshot(snapshot, thresholds=thresholds), snapshot=snapshot
    )


def infer_regime_series(
    index: Iterable[pd.Timestamp], *, provider: object | None = None,
    thresholds: RegimeThresholds | None = None,
) -> pd.Series:
    source = provider if provider is not None else MacroDataProvider()
    dates = list(index)
    return pd.Series(
        [
            classify_snapshot(source.snapshot(pd.Timestamp(date)), thresholds=thresholds)
            for date in dates
        ],
        index=dates,
        name="regime",
    )


__all__ = [
    "RegimeThresholds",
    "infer_regime",
    "MacroDataUnavailable",
    "MacroSnapshot",
    "RegimeAssessment",
    "MacroDataProvider",
    "classify_snapshot",
    "evaluate_regime",
    "infer_regime_series",
]
