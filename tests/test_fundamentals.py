import unittest
from unittest.mock import patch

import pandas as pd

from stock_analysis.fundamentals import fetch_fundamentals


class _RichStubTicker:
    def __init__(self) -> None:
        self.fast_info = {
            "eps": 5.0,
            "pe_ratio": 15.0,
            "last_price": 120.0,
            "dividend_yield": 0.02,
            "shares": 1_000_000,
        }
        self.info = {
            "trailingEps": 4.8,
            "sharesOutstanding": 1_000_000,
            "dividendYield": 0.02,
            "currentPrice": 121.0,
        }
        self.shares_outstanding = 1_000_000
        self.financials = pd.DataFrame(
            {pd.Timestamp("2023-12-31"): [500_000_000.0, 1_000_000_000.0]},
            index=["Net Income", "Total Revenue"],
        )
        self.balance_sheet = pd.DataFrame(
            {pd.Timestamp("2023-12-31"): [200_000_000.0, 400_000_000.0]},
            index=["Total Debt", "Total Stockholder Equity"],
        )
        self.dividends = pd.Series(
            [1.0, 1.0],
            index=pd.to_datetime(["2023-06-01", "2023-12-01"]).tz_localize("UTC"),
        )

    def history(self, **_: object) -> pd.DataFrame:  # pragma: no cover - not used
        raise AssertionError("history should not be called in this test")


class _SparseStubTicker:
    def __init__(self) -> None:
        self.fast_info = {}
        self.info = {"regularMarketPrice": 50.0}
        self.shares_outstanding = 0
        self.financials = pd.DataFrame(index=["Operating Income"], columns=[])
        self.balance_sheet = pd.DataFrame(index=["Cash"], columns=[])
        self.dividends = pd.Series(dtype="float64")

    def history(self, **_: object) -> pd.DataFrame:  # pragma: no cover - not used
        raise AssertionError("history should not be called in this test")


class FetchFundamentalsTests(unittest.TestCase):
    def test_fetch_fundamentals_with_complete_payload(self) -> None:
        with patch("stock_analysis.fundamentals.yf") as mock_yf:
            mock_yf.Ticker.return_value = _RichStubTicker()
            result = fetch_fundamentals("TST")

        self.assertAlmostEqual(result["eps"], 5.0)
        self.assertAlmostEqual(result["pe_ratio"], 15.0)
        self.assertAlmostEqual(result["net_margin_pct"], 50.0)
        self.assertAlmostEqual(result["debt_to_equity"], 0.5)
        self.assertAlmostEqual(result["dividend_yield_pct"], 2.0)
        self.assertEqual(result["as_of"].tzinfo.key, "Europe/Paris")
        self.assertEqual(result["source_fields"]["eps"], "fast_info.eps")
        self.assertEqual(result["source_fields"]["pe_ratio"], "fast_info.pe_ratio")
        self.assertEqual(result["source_fields"]["dividend_yield_pct"], "fast_info.dividend_yield")

    def test_fetch_fundamentals_handles_missing_values(self) -> None:
        with patch("stock_analysis.fundamentals.yf") as mock_yf:
            mock_yf.Ticker.return_value = _SparseStubTicker()
            with self.assertLogs("stock_analysis.fundamentals", level="WARNING") as log:
                result = fetch_fundamentals("TST")

        self.assertIsNone(result["eps"])
        self.assertIsNone(result["pe_ratio"])
        self.assertIsNone(result["net_margin_pct"])
        self.assertIsNone(result["debt_to_equity"])
        self.assertEqual(result["dividend_yield_pct"], 0.0)
        self.assertEqual(result["source_fields"]["dividend_yield_pct"], None)
        self.assertTrue(any("EPS unavailable" in entry for entry in log.output))
        self.assertTrue(any("Dividend yield unavailable" in entry for entry in log.output))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
