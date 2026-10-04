import io
import io
import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import pandas as pd

import stock_analysis.__main__ as cli


class CLIRunnerTests(unittest.TestCase):
    def test_cli_runs_and_saves_outputs(self) -> None:
        index = [
            datetime(2023, 1, 2, 9, 0),
            datetime(2023, 1, 3, 9, 0),
        ]
        frame = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [110.0, 111.0],
                "Low": [95.0, 96.0],
                "Close": [104.0, 105.0],
                "Adj Close": [106.0, 107.0],
                "Volume": [1000, 1100],
                "EMA7": [100.0, 101.0],
                "SMA20": [99.0, 100.0],
                "RSI14": [48.0, 52.0],
                "MACD": [0.4, 0.5],
                "MACD_signal": [0.3, 0.4],
                "MACD_hist": [0.1, 0.1],
            },
            index=index,
        )

        fundamentals = {
            "eps": 5.0,
            "pe_ratio": 18.0,
            "net_margin_pct": 12.0,
            "debt_to_equity": 0.9,
            "dividend_yield_pct": 1.2,
            "as_of": datetime(2022, 12, 31, 0, 0),
            "source_fields": {"eps": "fast_info"},
        }
        quality = {
            "duplicates": {"count": 0},
            "ohlc_anomalies": {"count": 0},
            "gaps": {"count": 1, "threshold_pct": 5.0},
            "na_rows_all": {"count": 0},
            "timezone": "Europe/Paris",
        }

        result = {
            "AAA": {
                "prices": frame,
                "fundamentals": fundamentals,
                "quality": quality,
                "score": {
                    "score": 72.5,
                    "trend": 28.0,
                    "momentum": 24.5,
                    "quality": 15.0,
                    "risk": 5.0,
                    "as_of": "2023-01-03T09:00:00+00:00",
                    "notes": ["missing: SMA200"],
                    "weights": {"trend": 0.4, "momentum": 0.3, "quality": 0.2, "risk": 0.1},
                    "regime": "Expansion",
                },
                "meta": {
                    "ticker": "AAA",
                    "period": "2y",
                    "interval": "1d",
                    "price_column": "Adj Close",
                },
            },
            "BBB": {
                "prices": frame,
                "fundamentals": fundamentals,
                "quality": quality,
                "score": {
                    "score": 45.0,
                    "trend": 12.0,
                    "momentum": 15.0,
                    "quality": 14.0,
                    "risk": 4.0,
                    "as_of": "2023-01-03T09:00:00+00:00",
                    "notes": [],
                    "weights": {"trend": 0.3, "momentum": 0.3, "quality": 0.25, "risk": 0.15},
                    "regime": "Expansion",
                },
                "meta": {
                    "ticker": "BBB",
                    "period": "2y",
                    "interval": "1d",
                    "price_column": "Adj Close",
                },
            },
        }

        previous_env = {key: os.environ.pop(key, None) for key in cli.ENV_KEYS.values()}
        dummy_fig = _DummyFigure()
        backtest_result = {
            "equity": pd.Series([1_000.0, 1_050.0], index=index),
            "trades": pd.DataFrame(
                {
                    "ticker": ["AAA", "BBB"],
                    "entry_date": [index[0], index[0]],
                    "entry_px": [100.0, 101.0],
                    "exit_date": [index[1], index[1]],
                    "exit_px": [105.0, 100.5],
                    "pnl": [50.0, -5.0],
                    "ret": [0.05, -0.005],
                    "holding_days": [1, 1],
                    "costs": [0.0, 0.0],
                }
            ),
            "positions": pd.DataFrame({"AAA": [0.0, 1.0], "BBB": [0.0, 0.0]}, index=index),
            "metrics": {"CAGR": 0.12, "Vol": 0.1, "Sharpe": 1.2, "MaxDD": -0.02, "Calmar": 6.0, "HitRate": 0.5, "AvgWin": 0.05, "AvgLoss": -0.005, "Payoff": 10.0, "ExposurePct": 50.0},
            "params": {"strategy": "sma200_trend"},
        }

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with mock.patch.object(cli, "analyze_tickers", return_value=result), mock.patch.object(
                    cli, "plot_ticker", return_value=dummy_fig
                ), mock.patch.object(cli, "run_backtest", return_value=backtest_result), mock.patch.object(
                    cli, "fetch_benchmark", return_value=pd.Series([1000.0, 1010.0], index=index[:2])
                ):
                    buffer = io.StringIO()
                    with mock.patch("sys.stdout", buffer):
                        exit_code = cli.main(
                            [
                                "--tickers",
                                "AAA,BBB",
                                "--save",
                                "--out-dir",
                                tmpdir,
                                "--format",
                                "csv",
                                "--price-column",
                                "Adj Close",
                                "--log-level",
                                "INFO",
                                "--score",
                                "--top",
                                "1",
                                "--report",
                                "--report-title",
                                "Test Report",
                                "--html",
                                "--no-charts",
                                "--bt",
                                "--strategy",
                                "sma200_trend",
                                "--capital",
                                "10000",
                                "--cost-bps",
                                "12",
                                "--slippage-bps",
                                "5",
                                "--max-positions",
                                "2",
                                "--stop-pct",
                                "0.1",
                                "--tp-pct",
                                "0.2",
                                "--bt-report",
                                "--bt-title",
                                "Test BT",
                                "--benchmark",
                                "^FCHI",
                                "--charts-bt-dir",
                                "bt_charts",
                                "--no-bt-charts",
                            ]
                        )

                self.assertEqual(exit_code, 0)

                output = buffer.getvalue()
                self.assertIn("Ticker : AAA", output)
                self.assertIn("RSI14", output)
                self.assertIn("Qualité: dups=0, ohlc=0, gaps=1>5%", output)
                self.assertIn("Fichiers écrits", output)
                self.assertIn("Tableau des scores", output)
                self.assertIn("AAA", output)
                self.assertIn("Score global=72.50", output)
                self.assertIn("Résumé synthétique", output)
                self.assertIn("Rapport écrit", output)
                self.assertIn("Résumé backtest", output)
                self.assertIn("Top trades", output)
                self.assertIn("Pires trades", output)
                report_path = os.path.join(tmpdir, "run_report.md")
                with open(report_path, "r", encoding="utf-8") as handle:
                    report_content = handle.read()
                self.assertIn("Test BT", report_content)

                saved_files = os.listdir(tmpdir)
                self.assertTrue(any(name.endswith("_prices.csv") for name in saved_files))
                self.assertTrue(any(name.endswith("MANIFEST.json") for name in saved_files))
                self.assertTrue(any(name.endswith("_scores.csv") for name in saved_files))

                self.assertTrue(any(name.endswith("_report.md") for name in saved_files))
                self.assertTrue(any(name.endswith("_report.html") for name in saved_files))
                self.assertTrue(any(name.endswith("_equity.csv") for name in saved_files))
                self.assertTrue(any(name.endswith("_trades.csv") for name in saved_files))
                self.assertTrue(any(name.endswith("_positions.csv") for name in saved_files))
                self.assertTrue(any(name.endswith("_kpis.csv") for name in saved_files))
                self.assertTrue(any(name.endswith("_drawdown.csv") for name in saved_files))

                manifest_path = next(
                    os.path.join(tmpdir, name)
                    for name in saved_files
                    if name.endswith("MANIFEST.json")
                )
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                self.assertEqual(manifest["format"], "csv")
                self.assertIn("backtest", manifest)
                self.assertIn("files", manifest["backtest"])
                self.assertTrue(any(path.endswith("_kpis.csv") for path in manifest["backtest"]["files"]))
                self.assertIn("backtest", manifest)
                self.assertIn("files", manifest["backtest"])

                report_path = next(
                    os.path.join(tmpdir, name)
                    for name in saved_files
                    if name.endswith("_report.md")
                )
                with open(report_path, "r", encoding="utf-8") as handle:
                    report_content = handle.read()
                self.assertIn("# Test Report", report_content)
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


    def test_prepare_output_subdir_handles_nested_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "out"
            base.mkdir()
            previous_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                path, rel = cli._prepare_output_subdir(base, "charts")
                self.assertEqual(path, base / "charts")
                if rel is not None:
                    self.assertEqual(rel.replace("\\", "/"), "charts")

                path_two, rel_two = cli._prepare_output_subdir(base, "out/charts2")
                self.assertEqual(path_two, base / "charts2")
                if rel_two is not None:
                    self.assertEqual(rel_two.replace("\\", "/"), "charts2")

                external = Path(tmpdir) / "external" / "plots"
                path_three, rel_three = cli._prepare_output_subdir(base, str(external))
                self.assertEqual(path_three, external)
                self.assertIsNone(rel_three)
            finally:
                os.chdir(previous_cwd)


class _DummyFigure:
    def savefig(self, target, *args, **kwargs):
        data = b"PNG"
        if hasattr(target, "write"):
            target.write(data)
        else:
            with open(target, "wb") as handle:
                handle.write(data)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
