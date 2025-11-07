import unittest
from datetime import datetime, timedelta

import pandas as pd

from stock_analysis.data import quality_report


class QualityReportTests(unittest.TestCase):
    def test_quality_report_detects_anomalies(self) -> None:
        start = datetime(2023, 1, 2, 9, 0)
        index = [
            start,
            start,  # duplicate timestamp
            start + timedelta(days=1),
            start + timedelta(days=2),
            start + timedelta(days=3),
        ]

        frame = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 99.0, 120.0, None],
                "High": [110.0, 111.0, 90.0, 130.0, None],
                "Low": [95.0, 96.0, 95.0, 110.0, None],
                "Close": [105.0, 100.0, 98.0, 118.0, None],
                "Adj Close": [105.0, 100.0, 98.0, 118.0, None],
                "Volume": [1000, 1100, 900, 1200, None],
            },
            index=index,
        )

        report = quality_report(frame)

        self.assertEqual(report["duplicates"]["count"], 1)
        self.assertEqual(report["ohlc_anomalies"]["count"], 1)
        self.assertEqual(report["gaps"]["count"], 1)
        self.assertEqual(report["na_rows_all"]["count"], 1)
        self.assertEqual(report["timezone"], "Europe/Paris")
if __name__ == "__main__":  # pragma: no cover
    unittest.main()
