import os
import tempfile
import unittest
from datetime import datetime

import pandas as pd

from stock_analysis.plot import plot_ticker, save_figure
from stock_analysis.report import build_summary_table, format_commentary, render_markdown

try:  # pragma: no cover - optional dependency check
    import matplotlib  # type: ignore
except Exception:  # pragma: no cover - matplotlib absent
    MATPLOTLIB_AVAILABLE = False
else:
    MATPLOTLIB_AVAILABLE = True


class ReportModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        index = [
            datetime(2023, 1, 2, 9, 0),
            datetime(2023, 1, 3, 9, 0),
            datetime(2023, 1, 4, 9, 0),
        ]
        self.frame = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [110.0, 111.0, 112.0],
                "Low": [95.0, 96.0, 97.0],
                "Close": [104.0, 105.0, 106.0],
                "Adj Close": [104.5, 105.5, 106.5],
                "Volume": [1000, 1100, 1200],
                "SMA20": [100.0, 101.0, 102.0],
                "SMA50": [99.0, 100.5, 101.0],
                "SMA200": [95.0, 95.5, 96.0],
                "EMA21": [101.0, 102.0, 103.0],
                "RSI14": [48.0, 52.0, 58.0],
                "MACD": [0.4, 0.5, 0.6],
                "MACD_signal": [0.3, 0.4, 0.5],
                "MACD_hist": [0.1, 0.1, 0.1],
                "VOL20": [0.02, 0.02, 0.02],
            },
            index=index,
        )
        self.frame.attrs = {"ticker": "AAA"}

        self.fundamentals = {
            "eps": 5.0,
            "pe_ratio": 18.0,
            "net_margin_pct": 12.5,
            "debt_to_equity": 1.0,
            "dividend_yield_pct": 2.0,
            "as_of": datetime(2022, 12, 31, 0, 0),
        }
        self.quality = {
            "duplicates": {"count": 0},
            "ohlc_anomalies": {"count": 0},
            "gaps": {"count": 1, "threshold_pct": 5.0},
            "na_rows_all": {"count": 0},
            "timezone": "Europe/Paris",
        }
        self.score = {
            "score": 72.5,
            "trend": 28.0,
            "momentum": 20.0,
            "quality": 20.0,
            "risk": 4.5,
            "as_of": "2023-01-04T09:00:00+00:00",
            "notes": [],
            "weights": {"trend": 0.4, "momentum": 0.3, "quality": 0.2, "risk": 0.1},
            "regime": "Expansion",
        }

    def _build_results(self) -> dict[str, dict]:
        return {
            "AAA": {
                "prices": self.frame,
                "fundamentals": self.fundamentals,
                "quality": self.quality,
                "score": self.score,
                "report": {"notes": []},
            }
        }

    def test_build_summary_table_counts_missing(self) -> None:
        results = self._build_results()
        results["AAA"]["fundamentals"]["dividend_yield_pct"] = None
        rows = build_summary_table(results)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["Ticker"], "AAA")
        self.assertEqual(row["MissingFundamentals"], 1)
        self.assertEqual(row["Gaps"], 1)
        self.assertAlmostEqual(row["TrendW"], 0.4)
        self.assertAlmostEqual(row["MomentumW"], 0.3)

    def test_format_commentary_range(self) -> None:
        results = self._build_results()
        commentary = format_commentary("AAA", results["AAA"])
        sentences = [segment for segment in commentary.split(". ") if segment.strip()]
        self.assertGreaterEqual(len(sentences), 2)
        self.assertLessEqual(len(sentences), 4)
        self.assertIn("AAA", commentary)
        self.assertNotIn("buy", commentary.lower())
        self.assertNotIn("sell", commentary.lower())
        self.assertNotIn("hold", commentary.lower())

    @unittest.skipUnless(MATPLOTLIB_AVAILABLE, "matplotlib requis pour ce test")
    def test_render_markdown_and_plot(self) -> None:
        results = self._build_results()
        results["AAA"]["report"] = {"chart_filename": "AAA.png", "notes": []}
        markdown_content = render_markdown(results, title="Mon Rapport", include_charts=True, charts_dir="charts")
        self.assertIn("# Mon Rapport", markdown_content)
        self.assertIn("### AAA", markdown_content)
        self.assertIn("![AAA chart](charts/AAA.png)", markdown_content)

        fig = plot_ticker(self.frame, price_column="Close")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "aaa.png")
            save_figure(fig, path)
            self.assertTrue(os.path.exists(path))
        try:
            from matplotlib import pyplot as plt  # type: ignore

            plt.close(fig)
        except Exception:  # pragma: no cover - optional cleanup
            pass


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
