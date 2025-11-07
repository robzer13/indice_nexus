import math
import unittest

import pandas as pd

from stock_analysis.indicators import compute_macd, compute_moving_averages, compute_rsi


class RSIMACDTests(unittest.TestCase):
    def test_rsi_macd_columns_and_lengths(self) -> None:
        base = pd.DataFrame({"Close": [100.0 + i * 0.5 for i in range(120)]})
        with_ma = compute_moving_averages(base)
        with_rsi = compute_rsi(with_ma)
        with_macd = compute_macd(with_rsi)

        for column in ("RSI14", "MACD", "MACD_signal", "MACD_hist"):
            self.assertIn(column, with_macd.columns)

        self.assertEqual(len(with_macd), len(base))

        macd_values = with_macd["MACD"].tolist()
        signal_values = with_macd["MACD_signal"].tolist()
        hist_values = with_macd["MACD_hist"].tolist()

        self.assertTrue(all(math.isnan(value) for value in macd_values[:25]))
        self.assertTrue(all(math.isnan(value) for value in signal_values[:33]))
        self.assertTrue(all(math.isnan(value) for value in hist_values[:33]))

        valid_macd = [value for value in macd_values if not math.isnan(value)]
        self.assertTrue(valid_macd)

    def test_rsi_bounds(self) -> None:
        noisy = [100.0 + i * 0.3 + (math.sin(i / 3.0) * 2.0) for i in range(60)]
        frame = pd.DataFrame({"Close": noisy})
        result = compute_rsi(frame)
        rsi_values = [value for value in result["RSI14"].tolist() if not math.isnan(value)]
        self.assertTrue(rsi_values)
        for value in rsi_values:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 100.0)

    def test_rsi_behaviour_edge_cases(self) -> None:
        constant = pd.DataFrame({"Close": [50.0] * 40})
        increasing = pd.DataFrame({"Close": [float(i) for i in range(1, 41)]})
        decreasing = pd.DataFrame({"Close": [float(100 - i) for i in range(40)]})

        constant_rsi = compute_rsi(constant)["RSI14"].tolist()
        increasing_rsi = compute_rsi(increasing)["RSI14"].tolist()
        decreasing_rsi = compute_rsi(decreasing)["RSI14"].tolist()

        self.assertTrue(all(math.isnan(value) for value in constant_rsi[:14]))
        self.assertTrue(all(math.isnan(value) for value in increasing_rsi[:14]))
        self.assertTrue(all(math.isnan(value) for value in decreasing_rsi[:14]))

        constant_tail = [value for value in constant_rsi if not math.isnan(value)]
        increasing_tail = [value for value in increasing_rsi if not math.isnan(value)]
        decreasing_tail = [value for value in decreasing_rsi if not math.isnan(value)]

        self.assertTrue(constant_tail)
        self.assertTrue(increasing_tail)
        self.assertTrue(decreasing_tail)

        self.assertTrue(all(abs(value - 50.0) < 1e-6 for value in constant_tail))
        self.assertTrue(all(abs(value - 100.0) < 1e-6 for value in increasing_tail))
        self.assertTrue(all(abs(value - 0.0) < 1e-6 for value in decreasing_tail))
if __name__ == "__main__":  # pragma: no cover
    unittest.main()
