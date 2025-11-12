"""Simple APScheduler integration to relaunch the Nexus command daily."""
from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
from typing import List

try:  # pragma: no cover - optional dependency
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
except Exception:  # pragma: no cover - runtime fallback for environments sans APScheduler
    BlockingScheduler = None  # type: ignore[assignment]
    CronTrigger = None  # type: ignore[assignment]
from zoneinfo import ZoneInfo

LOGGER = logging.getLogger(__name__)

PARIS_TZ = ZoneInfo("Europe/Paris")

DEFAULT_HOUR = int(os.getenv("NEXUS_HOUR", "7"))
DEFAULT_MINUTE = int(os.getenv("NEXUS_MINUTE", "15"))


def _build_cli_arguments() -> List[str]:
    tickers = os.getenv("NEXUS_TICKERS", "ASML,TTE.PA,MC.PA")
    period = os.getenv("NEXUS_PERIOD", "2y")
    interval = os.getenv("NEXUS_INTERVAL", "1d")
    price_column = os.getenv("NEXUS_PRICE_COLUMN", "Close")
    out_dir = os.getenv("NEXUS_OUT", "out")
    fmt = os.getenv("NEXUS_FORMAT", "parquet")
    base_name = os.getenv("NEXUS_BASE_NAME", "run")
    charts_dir = os.getenv("NEXUS_CHARTS", "charts")
    additional = os.getenv("NEXUS_ADDITIONAL_ARGS", "")

    args = [
        sys.executable,
        "-m",
        "stock_analysis",
        "--tickers",
        tickers,
        "--period",
        period,
        "--interval",
        interval,
        "--price-column",
        price_column,
        "--score",
        "--report",
        "--nexus-report",
        "--regime",
        "--save",
        "--out-dir",
        out_dir,
        "--format",
        fmt,
        "--base-name",
        base_name,
        "--charts-dir",
        charts_dir,
    ]

    benchmark = os.getenv("NEXUS_BENCHMARK")
    if benchmark:
        args.extend(["--benchmark", benchmark])

    if os.getenv("NEXUS_ENABLE_BT", "0") == "1":
        args.extend(["--bt", "--bt-report"])

    if additional:
        args.extend(shlex.split(additional))

    return args


def _run_once() -> None:
    args = _build_cli_arguments()
    LOGGER.info("Launching Nexus analysis", extra={"args": args})
    process = subprocess.run(args, check=False)
    if process.returncode != 0:
        LOGGER.error("Nexus command failed", extra={"returncode": process.returncode})
    else:
        LOGGER.info("Nexus command completed", extra={"returncode": process.returncode})


def schedule_daily_run() -> None:
    if BlockingScheduler is None or CronTrigger is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "APScheduler n'est pas installé. Installez l'extra 'scheduler' ou la dépendance apscheduler pour utiliser le planificateur."
        )
    scheduler = BlockingScheduler(timezone=PARIS_TZ)
    scheduler.add_job(
        _run_once,
        CronTrigger(hour=DEFAULT_HOUR, minute=DEFAULT_MINUTE, timezone=PARIS_TZ),
        name="nexus-daily",
    )

    LOGGER.info(
        "Scheduler initialised",
        extra={
            "time": f"{DEFAULT_HOUR:02d}:{DEFAULT_MINUTE:02d}",
            "tickers": os.getenv("NEXUS_TICKERS", "ASML,TTE.PA,MC.PA"),
            "out": os.getenv("NEXUS_OUT", "out"),
        },
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):  # pragma: no cover - runtime only
        LOGGER.info("Scheduler stopped")


def main() -> None:
    schedule_daily_run()


if __name__ == "__main__":  # pragma: no cover - manual execution only
    main()


__all__ = ["schedule_daily_run", "main"]
