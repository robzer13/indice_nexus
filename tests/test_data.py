import unittest
from unittest.mock import patch

import pandas as pd

from stock_analysis.data import CANONICAL_PRICE_COLUMNS, fetch_price_history


class _StubTicker:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def history(self, **_: object) -> pd.DataFrame:
        return self._frame.copy()


class FetchPriceHistoryTests(unittest.TestCase):
    def test_fetch_price_history_normalises_dataframe(self) -> None:
        index = pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02"])
        frame = pd.DataFrame(
            {
                "Open": [1.0, None, 3.0],
                "High": [2.0, None, 4.0],
                "Low": [0.5, None, 2.5],
                "Close": [1.5, None, 3.5],
                "Adj Close": [1.4, None, 3.4],
                "Volume": [1000, None, 3000],
            },
            index=index,
        )

        with patch("stock_analysis.data.yf") as mock_yf:
            mock_yf.Ticker.return_value = _StubTicker(frame)
            result = fetch_price_history("TST", period="1mo", interval="1d")

        self.assertEqual(result.index.tz.key, "Europe/Paris")
        self.assertTrue(result.index.is_monotonic_increasing)
        self.assertTrue(set(CANONICAL_PRICE_COLUMNS).issubset(result.columns))
        self.assertEqual(len(result), 2)  # fully missing row dropped
        self.assertEqual(result.attrs["ticker"], "TST")
        self.assertEqual(result.attrs["period"], "1mo")
        self.assertEqual(result.attrs["interval"], "1d")
        self.assertEqual(result.attrs["source"], "yfinance")

    def test_fetch_price_history_raises_when_empty(self) -> None:
        empty = pd.DataFrame(columns=CANONICAL_PRICE_COLUMNS)

        with patch("stock_analysis.data.yf") as mock_yf:
            mock_yf.Ticker.return_value = _StubTicker(empty)
            with self.assertRaisesRegex(ValueError, "No historical data returned"):
                fetch_price_history("TST")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
