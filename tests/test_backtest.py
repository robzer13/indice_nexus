import math
import unittest
from datetime import datetime, timedelta

import pandas as pd

from stock_analysis.backtest import run_backtest


class BacktestEngineTests(unittest.TestCase):
    def _make_index(self, length: int) -> list[datetime]:
        start = datetime(2023, 1, 2, 9, 0)
        return [start + timedelta(days=offset) for offset in range(length)]

    def _make_frame(
        self,
        opens,
        closes,
        lows,
        highs,
        sma200,
        ema21,
        rsi,
        macd,
        macd_signal,
        macd_hist,
    ) -> pd.DataFrame:
        index = self._make_index(len(opens))
        data = {
            "Open": list(opens),
            "High": list(highs),
            "Low": list(lows),
            "Close": list(closes),
            "Adj Close": list(closes),
            "Volume": [1_000] * len(opens),
            "SMA20": list(closes),
            "SMA50": list(closes),
            "SMA100": list(closes),
            "SMA200": list(sma200),
            "EMA7": list(closes),
            "EMA9": list(closes),
            "EMA20": list(closes),
            "EMA21": list(ema21),
            "RSI14": list(rsi),
            "MACD": list(macd),
            "MACD_signal": list(macd_signal),
            "MACD_hist": list(macd_hist),
        }
        return pd.DataFrame(data, index=index)

    def test_monotonic_equity_without_costs(self) -> None:
        frame = self._make_frame(
            opens=[100, 101, 102, 103, 104],
            closes=[101, 102, 103, 104, 105],
            lows=[99, 100, 101, 102, 103],
            highs=[102, 103, 104, 105, 106],
            sma200=[80, 80, 80, 80, 80],
            ema21=[90, 90, 90, 90, 90],
            rsi=[60, 60, 60, 60, 60],
            macd=[0.5, 0.6, 0.7, 0.8, 0.9],
            macd_signal=[0.1, 0.2, 0.3, 0.4, 0.5],
            macd_hist=[0.4, 0.4, 0.4, 0.4, 0.4],
        )
        results = {"AAA": {"prices": frame}}

        backtest = run_backtest(
            results,
            strategy="sma200_trend",
            capital=1_000.0,
            cost_bps=0.0,
            slippage_bps=0.0,
        )

        equity_series = backtest["equity"]
        equity_values = equity_series.tolist()
        self.assertAlmostEqual(equity_values[0], 1_000.0)
        for previous, current in zip(equity_values, equity_values[1:]):
            self.assertGreaterEqual(current, previous)

        trades = backtest["trades"]
        entry_dates = trades["entry_date"].tolist()
        self.assertEqual(entry_dates[0], frame.index[1])

    def test_transaction_costs_reduce_pnl(self) -> None:
        frame = self._make_frame(
            opens=[100, 101, 102, 103, 104],
            closes=[101, 102, 103, 104, 105],
            lows=[99, 100, 101, 102, 103],
            highs=[102, 103, 104, 105, 106],
            sma200=[80, 80, 80, 80, 80],
            ema21=[90, 90, 90, 90, 90],
            rsi=[60, 60, 60, 60, 60],
            macd=[0.5, 0.6, 0.7, 0.8, 0.9],
            macd_signal=[0.1, 0.2, 0.3, 0.4, 0.5],
            macd_hist=[0.4, 0.4, 0.4, 0.4, 0.4],
        )
        results = {"AAA": {"prices": frame}}

        reference = run_backtest(
            results,
            strategy="sma200_trend",
            capital=10_000.0,
            cost_bps=0.0,
            slippage_bps=0.0,
        )
        with_costs = run_backtest(
            results,
            strategy="sma200_trend",
            capital=10_000.0,
            cost_bps=50.0,
            slippage_bps=20.0,
        )

        self.assertLess(with_costs["equity"].tolist()[-1], reference["equity"].tolist()[-1])

    def test_rsi_rebound_strategy_generates_gain(self) -> None:
        frame = self._make_frame(
            opens=[100, 80, 85, 95, 100],
            closes=[100, 82, 88, 97, 102],
            lows=[99, 78, 82, 90, 98],
            highs=[101, 83, 90, 99, 104],
            sma200=[70, 70, 70, 70, 70],
            ema21=[75, 75, 75, 75, 75],
            rsi=[45, 25, 35, 55, 60],
            macd=[0.2, 0.2, 0.3, 0.4, 0.5],
            macd_signal=[0.1, 0.1, 0.2, 0.3, 0.4],
            macd_hist=[0.1, 0.1, 0.1, 0.1, 0.1],
        )
        results = {"BBB": {"prices": frame}}

        backtest = run_backtest(
            results,
            strategy="rsi_rebound",
            capital=5_000.0,
            cost_bps=0.0,
            slippage_bps=0.0,
        )

        pnl_values = backtest["trades"]["pnl"].tolist()
        self.assertTrue(any(value > 0 for value in pnl_values if isinstance(value, (int, float))))

    def test_stop_loss_and_take_profit_exit_next_open(self) -> None:
        stop_frame = self._make_frame(
            opens=[110, 112, 115, 95],
            closes=[111, 113, 90, 96],
            lows=[109, 110, 85, 94],
            highs=[112, 116, 118, 97],
            sma200=[100, 100, 100, 100],
            ema21=[105, 105, 105, 105],
            rsi=[60, 60, 40, 45],
            macd=[0.4, 0.5, 0.6, 0.6],
            macd_signal=[0.2, 0.2, 0.3, 0.3],
            macd_hist=[0.2, 0.3, 0.3, 0.3],
        )
        results = {"CCC": {"prices": stop_frame}}

        stop_bt = run_backtest(
            results,
            strategy="sma200_trend",
            capital=8_000.0,
            cost_bps=0.0,
            slippage_bps=0.0,
            stop_loss_pct=0.1,
        )
        stop_trade = {
            "exit_date": stop_bt["trades"]["exit_date"].tolist()[0],
            "entry_date": stop_bt["trades"]["entry_date"].tolist()[0],
        }
        self.assertEqual(stop_trade["exit_date"], stop_frame.index[3])
        self.assertLess(stop_bt["trades"]["ret"].tolist()[0], 0)

        tp_frame = self._make_frame(
            opens=[100, 102, 104, 120],
            closes=[101, 103, 121, 122],
            lows=[99, 101, 103, 119],
            highs=[102, 105, 130, 123],
            sma200=[90, 90, 90, 90],
            ema21=[95, 95, 95, 95],
            rsi=[55, 28, 35, 65],
            macd=[0.2, 0.3, 0.4, 0.5],
            macd_signal=[0.1, 0.2, 0.3, 0.4],
            macd_hist=[0.1, 0.1, 0.1, 0.1],
        )
        tp_results = {"DDD": {"prices": tp_frame}}
        tp_bt = run_backtest(
            tp_results,
            strategy="rsi_rebound",
            capital=8_000.0,
            cost_bps=0.0,
            slippage_bps=0.0,
            take_profit_pct=0.1,
        )
        tp_trade = {
            "entry": tp_bt["trades"]["entry_date"].tolist()[0],
            "exit": tp_bt["trades"]["exit_date"].tolist()[0],
        }
        self.assertEqual(tp_trade["exit"], tp_frame.index[3])
        self.assertGreater(tp_bt["trades"]["ret"].tolist()[0], 0)

    def test_max_positions_selects_best_trend(self) -> None:
        index = self._make_index(4)
        frames = {
            "AAA": self._make_frame(
                opens=[110, 111, 112, 113],
                closes=[112, 113, 114, 115],
                lows=[109, 110, 111, 112],
                highs=[113, 114, 115, 116],
                sma200=[100, 100, 100, 100],
                ema21=[105, 105, 105, 105],
                rsi=[60, 60, 60, 60],
                macd=[0.3, 0.3, 0.4, 0.4],
                macd_signal=[0.2, 0.2, 0.2, 0.2],
                macd_hist=[0.1, 0.1, 0.2, 0.2],
            ),
            "BBB": self._make_frame(
                opens=[120, 118, 117, 119],
                closes=[122, 119, 118, 120],
                lows=[119, 117, 116, 118],
                highs=[123, 120, 119, 121],
                sma200=[100, 100, 100, 100],
                ema21=[105, 105, 105, 105],
                rsi=[60, 60, 60, 60],
                macd=[0.4, 0.3, 0.3, 0.4],
                macd_signal=[0.2, 0.2, 0.2, 0.2],
                macd_hist=[0.2, 0.1, 0.1, 0.2],
            ),
            "CCC": self._make_frame(
                opens=[108, 130, 140, 150],
                closes=[110, 132, 142, 152],
                lows=[107, 129, 139, 149],
                highs=[111, 133, 143, 153],
                sma200=[100, 100, 100, 100],
                ema21=[105, 105, 105, 105],
                rsi=[60, 60, 60, 60],
                macd=[0.3, 0.5, 0.6, 0.7],
                macd_signal=[0.2, 0.3, 0.3, 0.3],
                macd_hist=[0.1, 0.2, 0.3, 0.4],
            ),
        }
        results = {ticker: {"prices": frame} for ticker, frame in frames.items()}

        backtest = run_backtest(
            results,
            strategy="sma200_trend",
            capital=9_000.0,
            cost_bps=0.0,
            slippage_bps=0.0,
            max_positions=1,
        )

        positions = backtest["positions"]
        for row_idx in range(1, len(index)):
            row = {ticker: positions[ticker].iloc[row_idx] for ticker in frames}
            active = [ticker for ticker, weight in row.items() if weight > 0.01]
            self.assertLessEqual(len(active), 1)
            if active:
                expected = max(
                    frames.keys(),
                    key=lambda ticker: frames[ticker]["Close"].iloc[row_idx - 1] / frames[ticker]["SMA200"].iloc[row_idx - 1],
                )
                self.assertEqual(active[0], expected)

    def test_metrics_match_expected_growth(self) -> None:
        periods = 101
        growth = 0.001
        opens = [100.0 * ((1 + growth) ** i) for i in range(periods)]
        closes = opens[:]
        lows = [value * 0.999 for value in opens]
        highs = [value * 1.001 for value in opens]
        sma200 = [value * 0.95 for value in opens]
        ema21 = [value * 0.97 for value in opens]
        rsi = [60.0] * periods
        macd = [0.3] * periods
        macd_signal = [0.2] * periods
        macd_hist = [0.1] * periods
        frame = self._make_frame(opens, closes, lows, highs, sma200, ema21, rsi, macd, macd_signal, macd_hist)
        results = {"EEE": {"prices": frame}}

        backtest = run_backtest(
            results,
            strategy="sma200_trend",
            capital=5_000.0,
            cost_bps=0.0,
            slippage_bps=0.0,
        )

        equity_values = backtest["equity"].tolist()
        metrics = backtest["metrics"]

        years = (len(equity_values) - 1) / 252.0
        expected_cagr = (equity_values[-1] / equity_values[0]) ** (1 / years) - 1 if years > 0 else 0.0
        self.assertAlmostEqual(metrics["CAGR"], expected_cagr, places=6)

        returns = [equity_values[i] / equity_values[i - 1] - 1 for i in range(1, len(equity_values))]
        mean_ret = sum(returns) / len(returns)
        variance = sum((value - mean_ret) ** 2 for value in returns) / len(returns)
        std_dev = math.sqrt(variance)
        expected_sharpe = (mean_ret / std_dev) * math.sqrt(252.0) if std_dev > 0 else 0.0
        self.assertAlmostEqual(metrics["Sharpe"], expected_sharpe, places=6)

        exposure_expected = (len(equity_values) - 1) / len(equity_values) * 100.0
        self.assertAlmostEqual(metrics["ExposurePct"], exposure_expected, places=6)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
