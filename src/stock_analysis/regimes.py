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


__all__ = ["RegimeThresholds", "infer_regime"]
