"""Automated Nexus-grade daily report runner."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict
from zoneinfo import ZoneInfo

from stock_analysis.__main__ import setup_logging
from stock_analysis.analyzer import analyze_tickers
from stock_analysis.io import save_analysis
from stock_analysis.regimes import evaluate_regime
from stock_analysis.report_nexus import generate_nexus_report
from stock_analysis.weighting import compute_weights

LOGGER = logging.getLogger(__name__)
PARIS_TZ = ZoneInfo("Europe/Paris")


def _resolve_tickers() -> list[str]:
    raw = os.getenv("NEXUS_TICKERS", "ASML,TTE.PA,MC.PA")
    tickers = [token.strip() for token in raw.split(",") if token.strip()]
    return tickers or ["ASML", "TTE.PA", "MC.PA"]


def run_once() -> Dict[str, str | None]:
    """Execute a full Nexus analysis and return generated report paths."""

    assessment = evaluate_regime(datetime.now(PARIS_TZ))
    weights = compute_weights(assessment.regime)

    tickers = _resolve_tickers()
    period = os.getenv("NEXUS_PERIOD", "2y")
    interval = os.getenv("NEXUS_INTERVAL", "1d")
    price_column = os.getenv("NEXUS_PRICE_COLUMN", "Close")
    out_dir = Path(os.getenv("NEXUS_OUT", "out")).expanduser()
    base_name = os.getenv("NEXUS_BASE_NAME", "nexus")
    fmt = os.getenv("NEXUS_FORMAT", "parquet")
    report_dir = out_dir / os.getenv("NEXUS_REPORT_DIR", "reports")
    report_title = os.getenv("NEXUS_REPORT_TITLE", "Nexus Market Report")
    top_n = int(os.getenv("NEXUS_REPORT_TOP", "10"))

    LOGGER.info(
        "Lancement Nexus quotidien",
        extra={"tickers": tickers, "period": period, "interval": interval, "regime": assessment.regime},
    )

    results = analyze_tickers(
        tickers,
        period=period,
        interval=interval,
        price_column=price_column,
        score_weights=weights,
        regime=assessment.regime,
    )

    regime_payload = assessment.to_dict()
    regime_payload["weights"] = weights

    save_analysis(
        results,
        out_dir=str(out_dir),
        base_name=base_name,
        format=fmt,
        regime=regime_payload,
    )

    report_paths = generate_nexus_report(
        results,
        assessment,
        weights,
        output_dir=report_dir,
        base_name="Nexus",
        title=report_title,
        top_n=top_n,
        include_html=True,
    )

    return report_paths


def main() -> int:
    log_level = os.getenv("NEXUS_LOG_LEVEL", "INFO")
    setup_logging(log_level)
    try:
        paths = run_once()
    except Exception as exc:  # pragma: no cover - runtime failures only
        LOGGER.error("Nexus daily run failed", exc_info=exc)
        return 1

    LOGGER.info("Nexus daily run complete", extra={"paths": paths})
    print("Rapports Nexus :")
    for label, path in paths.items():
        if path:
            print(f" - {label}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
