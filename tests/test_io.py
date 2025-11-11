import json
import os
import tempfile
import unittest
from datetime import datetime

import pandas as pd

from stock_analysis.io import save_analysis


class SaveAnalysisTests(unittest.TestCase):
    def test_save_analysis_creates_expected_files(self) -> None:
        index = [
            datetime(2023, 1, 2, 9, 0),
            datetime(2023, 1, 3, 9, 0),
        ]
        frame = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [110.0, 112.0],
                "Low": [95.0, 96.0],
                "Close": [105.0, 108.0],
                "Volume": [1000, 1100],
                "EMA7": [100.0, 101.0],
                "SMA20": [99.0, 100.0],
                "RSI14": [50.0, 55.0],
                "MACD": [0.5, 0.6],
                "MACD_signal": [0.4, 0.5],
                "MACD_hist": [0.1, 0.1],
            },
            index=index,
        )

        fundamentals = {
            "eps": 5.2,
            "pe_ratio": 18.4,
            "net_margin_pct": 12.5,
            "debt_to_equity": 0.8,
            "dividend_yield_pct": 1.5,
        }
        quality = {
            "duplicates": {"count": 0},
            "ohlc_anomalies": {"count": 0},
            "gaps": {"count": 0, "threshold_pct": 5.0},
            "na_rows_all": {"count": 0},
            "timezone": "Europe/Paris",
        }

        result = {
            "AAA": {
                "prices": frame,
                "fundamentals": fundamentals,
                "quality": quality,
                "score": {
                    "score": 70.0,
                    "trend": 24.0,
                    "momentum": 20.0,
                    "quality": 18.0,
                    "risk": 8.0,
                    "as_of": "2023-01-03T09:00:00+00:00",
                    "notes": ["missing: SMA200"],
                },
                "meta": {
                    "ticker": "AAA",
                    "period": "2y",
                    "interval": "1d",
                    "price_column": "Close",
                },
            }
        }

        backtest = {
            "equity": pd.Series([1_000.0, 1_020.0], index=index),
            "trades": pd.DataFrame(
                {
                    "ticker": ["AAA"],
                    "entry_date": [index[0]],
                    "entry_px": [100.0],
                    "exit_date": [index[1]],
                    "exit_px": [102.0],
                    "pnl": [20.0],
                    "ret": [0.02],
                    "holding_days": [1],
                    "costs": [0.0],
                }
            ),
            "positions": pd.DataFrame({"AAA": [0.0, 1.0]}, index=index),
            "metrics": {"CAGR": 0.1},
            "params": {"strategy": "sma200_trend"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            written = save_analysis(result, out_dir=tmpdir, base_name="run", format="csv", backtest=backtest)

            self.assertTrue(written)
            for path in written:
                self.assertTrue(os.path.exists(path))

            manifest_path = next(path for path in written if path.endswith("MANIFEST.json"))
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)

            self.assertEqual(manifest["schema_version"], "1.0.0")
            self.assertEqual(manifest["tickers"], ["AAA"])
            self.assertEqual(manifest["period"], "2y")
            self.assertEqual(manifest["interval"], "1d")
            self.assertEqual(manifest["price_column"], "Close")
            self.assertIn("run_AAA_prices.csv", manifest["files"][0])
            self.assertTrue(manifest["scores_included"])
            self.assertIn("backtest", manifest)
            self.assertIn("files", manifest["backtest"])

            prices_path = next(path for path in written if path.endswith("prices.csv"))
            with open(prices_path, "r", encoding="utf-8") as handle:
                header = handle.readline().strip()

            self.assertIn("Adj Close", header)
            self.assertIn("MACD_hist", header)

            fundamentals_path = next(path for path in written if path.endswith("fundamentals.json"))
            with open(fundamentals_path, "r", encoding="utf-8") as handle:
                stored_fundamentals = json.load(handle)
            self.assertEqual(stored_fundamentals["eps"], 5.2)

            scores_path = next(path for path in written if path.endswith("_scores.csv"))
            with open(scores_path, "r", encoding="utf-8") as handle:
                score_header = handle.readline().strip()
                score_row = handle.readline().strip()

            self.assertIn("Ticker", score_header)
            self.assertIn("Score", score_header)
            self.assertIn("AAA", score_row)

            self.assertTrue(any(path.endswith("_equity.csv") for path in written))
            self.assertTrue(any(path.endswith("_trades.csv") for path in written))
            self.assertTrue(any(path.endswith("_positions.csv") for path in written))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
