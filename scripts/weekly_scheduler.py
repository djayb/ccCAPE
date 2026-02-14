#!/usr/bin/env python3
"""Weekly scheduler for free-data pipeline + CC CAPE calculation."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]

WEEKDAY_MAP = {
    "MON": 0,
    "TUE": 1,
    "WED": 2,
    "THU": 3,
    "FRI": 4,
    "SAT": 5,
    "SUN": 6,
}


def log(message: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc).isoformat()}] {message}", flush=True)


def parse_weekday(value: str) -> int:
    normalized = value.strip().upper()[:3]
    if normalized not in WEEKDAY_MAP:
        allowed = ", ".join(WEEKDAY_MAP.keys())
        raise ValueError(f"Invalid weekday '{value}'. Use one of: {allowed}")
    return WEEKDAY_MAP[normalized]


def next_scheduled_run(now_local: dt.datetime, *, weekday: int, hour: int, minute: int) -> dt.datetime:
    candidate = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    days_ahead = (weekday - candidate.weekday()) % 7
    if days_ahead == 0 and candidate <= now_local:
        days_ahead = 7
    return candidate + dt.timedelta(days=days_ahead)


def run_command(command: list[str]) -> int:
    log(f"Running: {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(ROOT_DIR), check=False)
    log(f"Exit code: {completed.returncode}")
    return completed.returncode


def run_job(args: argparse.Namespace) -> tuple[int, int]:
    pipeline_cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "free_data_pipeline.py"),
        "--data-db",
        args.data_db,
        "--tracker-db",
        args.tracker_db,
        "--facts-limit",
        str(args.facts_limit),
        "--facts-stale-days",
        str(args.facts_stale_days),
        "--prices-symbol-limit",
        str(args.prices_symbol_limit),
        "--prices-rows-per-symbol",
        str(args.prices_rows_per_symbol),
        "--request-delay",
        str(args.request_delay),
    ]
    if args.facts_missing_only:
        pipeline_cmd.append("--facts-missing-only")
    if args.prices_missing_only:
        pipeline_cmd.append("--prices-missing-only")
    calc_cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "calc_cc_cape_free.py"),
        "--data-db",
        args.data_db,
        "--tracker-db",
        args.tracker_db,
        "--max-symbols",
        str(args.calc_max_symbols),
        "--min-eps-points",
        str(args.min_eps_points),
        "--lookback-years",
        str(args.lookback_years),
        "--market-cap-min-coverage",
        str(args.market_cap_min_coverage),
        "--write-markdown",
        args.calc_markdown_path,
    ]
    if args.shiller_cape is not None:
        calc_cmd.extend(["--shiller-cape", str(args.shiller_cape)])

    if args.update_tracker:
        pipeline_cmd.append("--update-tracker")
        calc_cmd.append("--update-tracker")
    else:
        pipeline_cmd.append("--no-update-tracker")
        calc_cmd.append("--no-update-tracker")

    pipeline_code = run_command(pipeline_cmd)
    calc_code = run_command(calc_cmd)

    series_code = 0
    if args.series_enabled and pipeline_code == 0 and calc_code == 0:
        series_cmd = [
            sys.executable,
            str(ROOT_DIR / "scripts" / "backfill_cc_cape_series_free.py"),
            "--data-db",
            args.data_db,
            "--tracker-db",
            args.tracker_db,
            "--series-years",
            str(args.series_years),
            "--max-symbols",
            str(args.series_max_symbols),
            "--min-eps-points",
            str(args.min_eps_points),
            "--lookback-years",
            str(args.lookback_years),
            "--market-cap-min-coverage",
            str(args.market_cap_min_coverage),
        ]
        if args.update_tracker:
            series_cmd.append("--update-tracker")
        else:
            series_cmd.append("--no-update-tracker")
        series_code = run_command(series_cmd)
        if series_code != 0:
            log("Monthly series backfill failed (non-zero exit); continuing.")

    kpi_code = 0
    if args.kpi_enabled:
        kpi_cmd = [
            sys.executable,
            str(ROOT_DIR / "scripts" / "generate_kpi_report.py"),
            "--tracker-db",
            args.tracker_db,
            "--data-db",
            args.data_db,
            "--days",
            str(args.kpi_days),
            "--out",
            args.kpi_markdown_path,
        ]
        if args.update_tracker:
            kpi_cmd.append("--update-tracker")
        else:
            kpi_cmd.append("--no-update-tracker")
        kpi_code = run_command(kpi_cmd)
    if kpi_code != 0:
        log("KPI report generation failed (non-zero exit); continuing.")
    return pipeline_code, calc_code


def sleep_until(target: dt.datetime) -> None:
    while True:
        now = dt.datetime.now(target.tzinfo)
        remaining = (target - now).total_seconds()
        if remaining <= 0:
            return
        chunk = min(300, max(1, int(remaining)))
        time.sleep(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run free-data ingestion + CC CAPE calculation weekly.")
    parser.add_argument("--data-db", default=os.getenv("TRACKER_FREE_DATA_DB", "data/free_data.db"))
    parser.add_argument("--tracker-db", default=os.getenv("TRACKER_DB", "data/internal_jira.db"))
    parser.add_argument("--timezone", default=os.getenv("WEEKLY_TIMEZONE", "America/New_York"))
    parser.add_argument("--weekday", default=os.getenv("WEEKLY_RUN_DAY", "SUN"))
    parser.add_argument("--hour", type=int, default=int(os.getenv("WEEKLY_RUN_HOUR", "9")))
    parser.add_argument("--minute", type=int, default=int(os.getenv("WEEKLY_RUN_MINUTE", "0")))
    parser.add_argument("--facts-limit", type=int, default=int(os.getenv("WEEKLY_FACTS_LIMIT", "200")))
    parser.add_argument("--facts-stale-days", type=int, default=int(os.getenv("WEEKLY_FACTS_STALE_DAYS", "30")))
    parser.add_argument(
        "--facts-missing-only",
        action=argparse.BooleanOptionalAction,
        default=(os.getenv("WEEKLY_FACTS_MISSING_ONLY", "false").lower() in {"1", "true", "yes"}),
        help="Only fetch company facts missing from the DB (useful for incremental backfills).",
    )
    parser.add_argument("--prices-symbol-limit", type=int, default=int(os.getenv("WEEKLY_PRICES_SYMBOL_LIMIT", "503")))
    parser.add_argument("--prices-rows-per-symbol", type=int, default=int(os.getenv("WEEKLY_PRICES_ROWS_PER_SYMBOL", "3650")))
    parser.add_argument(
        "--prices-missing-only",
        action=argparse.BooleanOptionalAction,
        default=(os.getenv("WEEKLY_PRICES_MISSING_ONLY", "false").lower() in {"1", "true", "yes"}),
        help="Only fetch prices for symbols missing from the DB (useful for filling gaps quickly).",
    )
    parser.add_argument("--request-delay", type=float, default=float(os.getenv("WEEKLY_REQUEST_DELAY", "0.25")))
    parser.add_argument("--calc-max-symbols", type=int, default=int(os.getenv("WEEKLY_CALC_MAX_SYMBOLS", "0")))
    parser.add_argument("--min-eps-points", type=int, default=int(os.getenv("WEEKLY_MIN_EPS_POINTS", "8")))
    parser.add_argument("--lookback-years", type=int, default=int(os.getenv("WEEKLY_LOOKBACK_YEARS", "10")))
    parser.add_argument(
        "--market-cap-min-coverage",
        type=float,
        default=float(os.getenv("WEEKLY_MARKET_CAP_MIN_COVERAGE", "0.8")),
    )
    shiller_env = os.getenv("WEEKLY_SHILLER_CAPE", "")
    parser.add_argument("--shiller-cape", type=float, default=float(shiller_env) if shiller_env else None)
    parser.add_argument("--calc-markdown-path", default=os.getenv("WEEKLY_CALC_MARKDOWN_PATH", "docs/CC_CAPE_FREE_RUN.md"))
    parser.add_argument(
        "--series-enabled",
        action=argparse.BooleanOptionalAction,
        default=(os.getenv("WEEKLY_SERIES_ENABLED", "true").lower() in {"1", "true", "yes"}),
        help="Backfill/update the monthly CC CAPE series after each scheduled run.",
    )
    parser.add_argument("--series-years", type=int, default=int(os.getenv("WEEKLY_SERIES_YEARS", "10")))
    parser.add_argument("--series-max-symbols", type=int, default=int(os.getenv("WEEKLY_SERIES_MAX_SYMBOLS", "0")))
    parser.add_argument(
        "--kpi-enabled",
        action=argparse.BooleanOptionalAction,
        default=(os.getenv("WEEKLY_KPI_ENABLED", "true").lower() in {"1", "true", "yes"}),
        help="Generate KPI baseline report after each scheduled run.",
    )
    parser.add_argument("--kpi-days", type=int, default=int(os.getenv("WEEKLY_KPI_DAYS", "30")))
    parser.add_argument("--kpi-markdown-path", default=os.getenv("WEEKLY_KPI_MARKDOWN_PATH", "docs/KPI_BASELINE.md"))
    parser.add_argument(
        "--update-tracker",
        action=argparse.BooleanOptionalAction,
        default=(os.getenv("WEEKLY_UPDATE_TRACKER", "true").lower() in {"1", "true", "yes"}),
    )
    parser.add_argument(
        "--run-on-startup",
        action=argparse.BooleanOptionalAction,
        default=(os.getenv("WEEKLY_RUN_ON_STARTUP", "true").lower() in {"1", "true", "yes"}),
    )
    parser.add_argument("--max-runs", type=int, default=int(os.getenv("WEEKLY_MAX_RUNS", "0")))
    args = parser.parse_args()

    if not (0 <= args.hour <= 23 and 0 <= args.minute <= 59):
        raise SystemExit("hour must be 0-23 and minute must be 0-59")

    timezone = ZoneInfo(args.timezone)
    weekday = parse_weekday(args.weekday)

    run_count = 0
    if args.run_on_startup:
        log("Startup run triggered.")
        run_job(args)
        run_count += 1

    while True:
        if args.max_runs > 0 and run_count >= args.max_runs:
            log(f"Reached max-runs={args.max_runs}. Exiting.")
            return 0

        now_local = dt.datetime.now(timezone)
        next_run = next_scheduled_run(now_local, weekday=weekday, hour=args.hour, minute=args.minute)
        log(f"Next run scheduled at {next_run.isoformat()}")
        sleep_until(next_run)
        log("Scheduled run triggered.")
        run_job(args)
        run_count += 1


if __name__ == "__main__":
    raise SystemExit(main())
