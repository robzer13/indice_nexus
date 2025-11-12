import math
import unittest

import pandas as pd

from stock_analysis.indicators import compute_moving_averages


class MovingAverageTests(unittest.TestCase):
    def test_simple_and_exponential_average_values(self) -> None:
        base = pd.DataFrame({"Close": [10, 11, 12, 13, 14]})
        result = compute_moving_averages(base, windows_ema=(3,), windows_sma=(3,))

        expected_sma = [math.nan, math.nan, 11.0, 12.0, 13.0]
        expected_ema = [math.nan, math.nan, 11.25, 12.125, 13.0625]

        sma = result["SMA3"].tolist()
        ema = result["EMA3"].tolist()

        for actual, expected in zip(sma, expected_sma):
            if math.isnan(expected):
                self.assertTrue(math.isnan(actual))
            else:
                self.assertAlmostEqual(actual, expected, places=9)

        for actual, expected in zip(ema, expected_ema):
            if math.isnan(expected):
                self.assertTrue(math.isnan(actual))
            else:
                self.assertAlmostEqual(actual, expected, places=9)

        # Ensure input DataFrame has not been mutated.
        self.assertEqual(list(base.columns), ["Close"])

    def test_default_windows_exist(self) -> None:
        base = pd.DataFrame({"Close": [float(i) for i in range(1, 205)]})
        result = compute_moving_averages(base)

        for window in (7, 9, 20, 21):
            self.assertIn(f"EMA{window}", result.columns)
        for window in (20, 50, 100, 200):
            self.assertIn(f"SMA{window}", result.columns)
        sma200_prefix = result["SMA200"][:199]
        self.assertTrue(all(math.isnan(value) for value in sma200_prefix))

    def test_missing_price_column_raises(self) -> None:
        frame = pd.DataFrame({"Adj Close": [1, 2, 3]})
        with self.assertRaises(KeyError):
            compute_moving_averages(frame)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
