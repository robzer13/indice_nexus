import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from stock_analysis.features import add_ta_features, make_label_future_ret


class FeatureEngineeringTests(unittest.TestCase):
    def setUp(self) -> None:
        start = datetime(2023, 1, 1, 9, 0, tzinfo=ZoneInfo("Europe/Paris"))
        dates = [start + timedelta(days=idx) for idx in range(30)]
        close_values = [100 + idx for idx in range(30)]
        data = {
            "Open": list(close_values),
            "High": list(close_values),
            "Low": list(close_values),
            "Close": list(close_values),
            "Adj Close": list(close_values),
            "Volume": [1_000] * len(close_values),
        }
        self.frame = pd.DataFrame(data, index=dates)

    def test_add_ta_features_shapes(self) -> None:
        features = add_ta_features(self.frame)
        self.assertTrue(set(["ret_1d", "vol_20", "sma200_gap", "rsi_14", "mom_21"]).issubset(features.columns))
        self.assertEqual(len(features), len(self.frame))
        self.assertTrue(features["ret_1d"].isna().iloc[0])
        self.assertTrue(features["mom_21"].tail(5).isna().all())

    def test_make_label_future_ret_alignment(self) -> None:
        labels = make_label_future_ret(self.frame, horizon=5, thr=0.01)
        self.assertEqual(len(labels), len(self.frame))
        self.assertTrue(labels.tail(5).isna().all())
        self.assertTrue(labels.dropna().isin({0.0, 1.0}).all())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
