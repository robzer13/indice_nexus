import os
import tempfile
import unittest
from datetime import datetime, timedelta

import pandas as pd

from stock_analysis.bt_report import (
    attach_benchmark,
    compute_drawdown,
    render_bt_markdown,
    summarize_backtest,
)
from stock_analysis.plot_bt import (
    plot_drawdown,
    plot_equity_with_benchmark,
    plot_exposure_heatmap,
    save_figure,
)

try:  # pragma: no cover - optional dependency check
    import matplotlib  # type: ignore
except Exception:  # pragma: no cover - matplotlib absent
    MATPLOTLIB_AVAILABLE = False
else:
    MATPLOTLIB_AVAILABLE = True


class BacktestReportModuleTests(unittest.TestCase):
    def _equity_series(self) -> pd.Series:
        index = [datetime(2023, 1, 2, 9, 0) + timedelta(days=i) for i in range(5)]
        values = [1000.0, 1050.0, 980.0, 990.0, 1_050.0]
        return pd.Series(values, index=index)

    def test_compute_drawdown_structure(self) -> None:
        equity = self._equity_series()
        dd_frame = compute_drawdown(equity)
        self.assertIn("dd", dd_frame.columns)
        self.assertTrue(any(value < 0 for value in dd_frame["dd"].tolist() if isinstance(value, float)))
        self.assertTrue(all(value <= 0 or value != value for value in dd_frame["dd"].tolist()))
        self.assertIn("peak", dd_frame.columns)
        self.assertLess(min(value for value in dd_frame["dd"].tolist() if value == value), 0.0)

    def test_summarize_backtest_extracts_metrics(self) -> None:
        equity = self._equity_series()
        trades = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "entry_date": [equity.index[0], equity.index[1]],
                "entry_px": [100.0, 101.0],
                "exit_date": [equity.index[2], equity.index[3]],
                "exit_px": [102.0, 103.0],
                "pnl": [20.0, -5.0],
                "ret": [0.02, -0.005],
                "holding_days": [2, 2],
                "costs": [0.0, 0.0],
            }
        )
        metrics = {
            "CAGR": 0.12,
            "Vol": 0.15,
            "Sharpe": 0.8,
            "MaxDD": -0.1,
            "Calmar": 1.2,
            "HitRate": 0.5,
            "AvgWin": 0.05,
            "AvgLoss": -0.02,
            "Payoff": 2.5,
            "ExposurePct": 40.0,
        }
        backtest = {
            "equity": equity,
            "trades": trades,
            "metrics": metrics,
            "params": {"strategy": "demo"},
        }
        summary = summarize_backtest(backtest)
        expected_keys = {
            "CAGR",
            "Vol",
            "Sharpe",
            "MaxDD",
            "Calmar",
            "HitRate",
            "AvgWin",
            "AvgLoss",
            "Payoff",
            "ExposurePct",
            "Trades",
            "Start",
            "End",
        }
        self.assertTrue(expected_keys.issubset(summary.keys()))
        self.assertEqual(summary["Trades"], 2)
        self.assertIn("2023-01-02", summary["Start"])

    def test_attach_benchmark_rebases(self) -> None:
        equity = self._equity_series()
        benchmark_values = [5000.0, 5050.0, 4900.0, 4950.0, 5_100.0]
        benchmark = pd.Series(benchmark_values, index=equity.index)
        combined = attach_benchmark(equity, benchmark_df=benchmark, label="Bench")
        self.assertAlmostEqual(combined["Strategy"].iloc[0], 1.0)
        self.assertAlmostEqual(combined["Bench"].iloc[0], 1.0)

    @unittest.skipUnless(MATPLOTLIB_AVAILABLE, "matplotlib requis pour ce test")
    def test_plot_helpers_generate_figures(self) -> None:
        equity = self._equity_series()
        combined = pd.DataFrame({"Strategy": equity.tolist()}, index=equity.index)
        dd_frame = compute_drawdown(equity)
        positions = pd.DataFrame({"AAA": [0.0, 0.5, 1.0, 0.5, 0.0]}, index=equity.index)

        fig_equity = plot_equity_with_benchmark(combined)
        fig_dd = plot_drawdown(dd_frame)
        fig_heatmap = plot_exposure_heatmap(positions)

        with tempfile.TemporaryDirectory() as tmpdir:
            for name, fig in {
                "equity": fig_equity,
                "drawdown": fig_dd,
                "exposure": fig_heatmap,
            }.items():
                path = os.path.join(tmpdir, f"{name}.png")
                save_figure(fig, path)
                self.assertTrue(os.path.exists(path))

        try:
            from matplotlib import pyplot as plt  # type: ignore

            plt.close(fig_equity)
            plt.close(fig_dd)
            plt.close(fig_heatmap)
        except Exception:  # pragma: no cover - optional cleanup
            pass

    def test_render_markdown_contains_sections(self) -> None:
        equity = self._equity_series()
        trades = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "entry_date": [equity.index[0], equity.index[1]],
                "entry_px": [100.0, 101.0],
                "exit_date": [equity.index[2], equity.index[3]],
                "exit_px": [102.0, 103.0],
                "pnl": [20.0, -5.0],
                "ret": [0.02, -0.005],
                "holding_days": [2, 2],
                "costs": [0.0, 0.0],
            }
        )
        backtest = {
            "equity": equity,
            "trades": trades,
            "positions": pd.DataFrame({"AAA": [0.0, 1.0, 0.5]}, index=equity.index[:3]),
            "metrics": {
                "CAGR": 0.1,
                "Vol": 0.15,
                "Sharpe": 0.7,
                "MaxDD": -0.2,
                "Calmar": 0.5,
                "HitRate": 0.6,
                "AvgWin": 0.04,
                "AvgLoss": -0.02,
                "Payoff": 2.0,
                "ExposurePct": 55.0,
            },
            "params": {"strategy": "demo"},
        }
        dd_frame = compute_drawdown(equity)
        kpis = summarize_backtest(backtest)
        charts = {
            "equity": "charts/equity.png",
            "drawdown": "charts/drawdown.png",
            "exposure": "charts/exposure.png",
        }
        markdown = render_bt_markdown(
            backtest,
            benchmark_name="Bench",
            kpis=kpis,
            dd=dd_frame,
            charts=charts,
        )
        self.assertIn("Résultats du backtest", markdown)
        self.assertIn("| Metric | Value |", markdown)
        self.assertIn("![Courbe equity vs Bench](charts/equity.png)", markdown)
        self.assertIn("Top trades", markdown)
        self.assertIn("Pires trades", markdown)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
