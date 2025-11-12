import unittest

import importlib.util
import pandas as pd

from stock_analysis.ml_pipeline import confusion, sharpe_sim, time_cv, walk_forward_signals

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


@unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn not installed")
class MLPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        index = pd.date_range("2022-01-01", periods=300, freq="D", tz="Europe/Paris")
        span = len(index)
        ret = [(-0.4 + (0.8 * idx) / (span - 1)) for idx in range(span)]
        data = {
            "ret_1d": ret,
            "vol_20": [0.1 + (0.1 * idx) / (span - 1) for idx in range(span)],
            "sma200_gap": [(-0.2 + (0.4 * idx) / (span - 1)) for idx in range(span)],
            "rsi_14": [30 + (40 * idx) / (span - 1) for idx in range(span)],
            "mom_21": [(-0.3 + (0.6 * idx) / (span - 1)) for idx in range(span)],
        }
        self.X = pd.DataFrame(data, index=index)
        self.y = pd.Series((self.X["ret_1d"] > 0).astype(int).values, index=index)

    def test_time_cv_auc_bounds(self) -> None:
        auc_mean, auc_std = time_cv(self.X, self.y, model_kind="logreg", splits=3)
        self.assertGreaterEqual(auc_mean, 0.5)
        self.assertLessEqual(auc_mean, 1.0)
        self.assertGreaterEqual(auc_std, 0.0)

    def test_walk_forward_outputs(self) -> None:
        proba, signal = walk_forward_signals(
            self.X,
            self.y,
            retrain_every=30,
            warmup=80,
            model_kind="logreg",
            proba_threshold=0.55,
        )
        self.assertEqual(len(proba), len(self.X))
        self.assertEqual(len(signal), len(self.X))
        self.assertTrue(set(signal.dropna().unique()).issubset({0.0, 1.0}))

        price = pd.Series([100 + (100 * idx) / (len(self.X) - 1) for idx in range(len(self.X))], index=self.X.index)
        sharpe = sharpe_sim(price, signal)
        self.assertIsInstance(sharpe, float)

        valid_index = proba.dropna().index
        cm = confusion(self.y.loc[valid_index], proba.dropna(), 0.5)
        self.assertEqual(sum(cm.values()), len(valid_index))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
