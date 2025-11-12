import math
import unittest
from datetime import datetime

import pandas as pd

from stock_analysis.indicators import compute_macd, compute_moving_averages, compute_rsi
from stock_analysis.scoring import (
    compute_score_bundle,
    compute_volatility,
    score_quality,
    score_risk,
    score_trend,
)


def _make_frame(close_values: list[float]) -> pd.DataFrame:
    index = pd.date_range(
        start=datetime(2023, 1, 1, 9, 0),
        periods=len(close_values),
        freq="D",
        tz="Europe/Paris",
    )
    highs = [value + 2.0 for value in close_values]
    lows = [value - 2.0 for value in close_values]
    return pd.DataFrame(
        {
            "Open": close_values,
            "High": highs,
            "Low": lows,
            "Close": close_values,
            "Adj Close": close_values,
            "Volume": [1_000_000] * len(close_values),
        },
        index=index,
    )


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = compute_moving_averages(frame, price_column="Close")
    enriched = compute_rsi(enriched, price_column="Close", period=14)
    enriched = compute_macd(enriched, price_column="Close")
    enriched = compute_volatility(enriched, price_column="Close", window=20)
    return enriched


class ScoringWorkflowTests(unittest.TestCase):
    def test_uptrend_scores_high(self) -> None:
        close_values = [100.0 + index * 0.6 for index in range(240)]
        frame = _prepare(_make_frame(close_values))

        fundamentals = {
            "pe_ratio": 20.0,
            "net_margin_pct": 15.0,
            "debt_to_equity": 1.0,
            "dividend_yield_pct": 2.0,
        }

        bundle = compute_score_bundle(frame, fundamentals)

        self.assertGreaterEqual(bundle["trend"], 24.0)
        self.assertGreaterEqual(bundle["momentum"], 18.0)
        self.assertGreaterEqual(bundle["score"], 50.0)
        self.assertLessEqual(len(bundle["notes"]), 2)

    def test_downtrend_scores_low(self) -> None:
        close_values = [200.0 - index * 0.6 for index in range(240)]
        frame = _prepare(_make_frame(close_values))

        fundamentals = {
            "pe_ratio": 35.0,
            "net_margin_pct": 6.0,
            "debt_to_equity": 2.0,
            "dividend_yield_pct": 1.5,
        }

        bundle = compute_score_bundle(frame, fundamentals)

        self.assertLessEqual(bundle["trend"], 8.0)
        self.assertLessEqual(bundle["momentum"], 10.0)
        self.assertLessEqual(bundle["score"], 35.0)

    def test_high_volatility_penalises_risk(self) -> None:
        pattern = [100.0, 120.0, 90.0, 130.0, 85.0, 125.0]
        close_values: list[float] = []
        while len(close_values) < 240:
            close_values.extend(pattern)
        close_values = close_values[:240]

        frame = _prepare(_make_frame(close_values))
        fundamentals = {
            "pe_ratio": 18.0,
            "net_margin_pct": 8.0,
            "debt_to_equity": 1.8,
            "dividend_yield_pct": 0.5,
        }

        bundle = compute_score_bundle(frame, fundamentals)

        self.assertLessEqual(bundle["risk"], 2.0)
        self.assertGreaterEqual(bundle["momentum"], 10.0)

    def test_quality_scoring_ranges(self) -> None:
        fundamentals = {
            "pe_ratio": 20.0,
            "net_margin_pct": 12.0,
            "debt_to_equity": 1.0,
            "dividend_yield_pct": 2.0,
        }
        self.assertEqual(score_quality(fundamentals), 20.0)

        poor_fundamentals = {
            "pe_ratio": None,
            "net_margin_pct": None,
            "debt_to_equity": None,
            "dividend_yield_pct": None,
        }

        frame = _prepare(_make_frame([100.0 + index for index in range(240)]))
        bundle = compute_score_bundle(frame, poor_fundamentals)
        self.assertEqual(bundle["quality"], 0.0)
        self.assertTrue(any(note.startswith("no fundamentals.") for note in bundle["notes"]))

    def test_missing_columns_are_handled(self) -> None:
        close_values = [100.0 + index * 0.4 for index in range(60)]
        frame = _prepare(_make_frame(close_values))
        frame = frame.drop(columns=["SMA200", "VOL20"], errors="ignore")

        bundle = compute_score_bundle(frame, {})

        self.assertIn("missing: SMA200", bundle["notes"] or [])
        self.assertIn("missing: VOL20", bundle["notes"] or [])
        self.assertGreaterEqual(bundle["risk"], 0.0)


class ComponentScoringTests(unittest.TestCase):
    def test_trend_and_risk_basic_behaviour(self) -> None:
        close_values = [float(value) for value in range(200, 0, -1)]
        frame = _prepare(_make_frame(close_values))

        self.assertTrue(math.isclose(score_trend(frame), 0.0))
        self.assertLessEqual(score_risk(frame), 10.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
