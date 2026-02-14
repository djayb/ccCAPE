#!/usr/bin/env python3
"""Import SimFin bulk fundamentals into `data/free_data.db` (optional).

This script supports pulling SimFin "bulk download" datasets (income annual)
and mapping a subset of columns into our `company_facts_values` table.

Notes:
- Requires a SimFin API key (free or paid depending on your needs).
- SimFin licensing applies; this repo treats SimFin as an optional adapter.

By default we import only current S&P 500 constituents (from `sp500_constituents`)
to keep the DB smaller, but you can override with `--universe all`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
from pathlib import Path
import sys
import zipfile
from typing import Any, Iterable

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from internal_jira import now_utc  # noqa: E402

from scripts._fundamentals_import import (  # noqa: E402
    connect_data_db,
    delete_existing_taxonomy_tags,
    load_latest_constituents_symbol_to_cik,
    load_symbol_overrides,
    load_sec_ticker_map_symbol_to_cik,
    normalize_symbol,
    parse_date,
    parse_float,
    resolve_cik,
    symbol_candidates,
    table_exists,
    upsert_company_fact_value,
)


SIMFIN_BULK_URL = "https://prod.simfin.com/api/bulk-download/s3"

# Column names as used by SimFin bulk downloads (semicolon-delimited CSV).
COL_TICKER = "Ticker"
COL_REPORT_DATE = "Report Date"
COL_FISCAL_YEAR = "Fiscal Year"
COL_FISCAL_PERIOD = "Fiscal Period"
COL_NET_INCOME = "Net Income"
COL_SHARES_BASIC = "Shares (Basic)"
COL_SHARES_OUTSTANDING = "Shares Outstanding"
COL_EPS_BASIC = "Earnings Per Share, Basic"
COL_EPS_DILUTED = "Earnings Per Share, Diluted"


TAGS_BY_COL = {
    COL_NET_INCOME: ("NetIncomeLoss", "USD"),
    COL_SHARES_BASIC: ("WeightedAverageNumberOfSharesOutstandingBasic", "shares"),
    COL_SHARES_OUTSTANDING: ("EntityCommonStockSharesOutstanding", "shares"),
    COL_EPS_BASIC: ("EarningsPerShareBasic", "USD/shares"),
    COL_EPS_DILUTED: ("EarningsPerShareDiluted", "USD/shares"),
}


def build_universe(conn, *, as_of_date: str | None, universe: str) -> set[str]:
    universe = (universe or "").strip().lower()
    if universe not in {"sp500", "all"}:
        raise ValueError("universe must be 'sp500' or 'all'")
    if universe == "all":
        return set()
    mapping = load_latest_constituents_symbol_to_cik(conn, as_of_date)
    return set(mapping.keys())


def match_symbol_to_universe(simfin_ticker: str, universe_symbols: set[str]) -> str | None:
    sym = normalize_symbol(simfin_ticker)
    if not sym:
        return None
    if not universe_symbols:
        return sym
    for candidate in symbol_candidates(sym):
        if candidate in universe_symbols:
            return candidate
    return None


def iter_simfin_rows_from_bytes(content: bytes) -> Iterable[dict[str, str]]:
    # Content can be a ZIP file containing CSV(s) or a raw CSV.
    if content[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise RuntimeError("SimFin zip did not contain any .csv file.")
            # Prefer the largest CSV in the zip (typical case: exactly one).
            csv_names.sort(key=lambda n: zf.getinfo(n).file_size if n in zf.namelist() else 0, reverse=True)
            with zf.open(csv_names[0]) as f:
                text = io.TextIOWrapper(f, encoding="utf-8", newline="")
                reader = csv.DictReader(text, delimiter=";")
                yield from reader
    else:
        text = io.StringIO(content.decode("utf-8", errors="replace"))
        reader = csv.DictReader(text, delimiter=";")
        yield from reader


def fetch_simfin_dataset(*, api_key: str, dataset: str, variant: str, market: str, timeout: float) -> bytes:
    params = {"dataset": dataset, "variant": variant, "market": market}
    headers = {"Authorization": f"api-key {api_key}"}
    resp = requests.get(SIMFIN_BULK_URL, params=params, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        # Response body tends to be a helpful string (e.g. invalid key / permissions).
        raise RuntimeError(f"SimFin bulk download failed: HTTP {resp.status_code}: {resp.text[:400]}")
    return resp.content


def import_simfin_income_annual(args: argparse.Namespace) -> dict[str, Any]:
    api_key = (args.api_key or "").strip()
    if not api_key:
        raise SystemExit("Missing SimFin API key. Provide --api-key or set SIMFIN_API_KEY.")

    summary: dict[str, Any] = {
        "data_db": args.data_db,
        "taxonomy": args.taxonomy,
        "dataset": "income",
        "variant": "annual",
        "market": args.market,
        "universe": args.universe,
        "only_before": args.only_before,
        "dry_run": bool(args.dry_run),
        "started_at": now_utc(),
        "rows_read": 0,
        "rows_matched_universe": 0,
        "rows_skipped_missing_id": 0,
        "rows_skipped_bad_dates": 0,
        "rows_skipped_no_values": 0,
        "facts_upserted": 0,
        "deleted_existing": 0,
    }

    only_before_date = parse_date(args.only_before) if args.only_before else None

    with connect_data_db(args.data_db) as conn:
        if not table_exists(conn, "company_facts_values"):
            raise SystemExit("Missing table company_facts_values. Run scripts/free_data_pipeline.py at least once first.")

        universe_symbols = build_universe(conn, as_of_date=args.as_of_constituents_date, universe=args.universe)
        symbol_to_cik = load_latest_constituents_symbol_to_cik(conn, args.as_of_constituents_date)
        sec_map = load_sec_ticker_map_symbol_to_cik(conn)
        overrides = load_symbol_overrides(conn)

        if args.replace_existing:
            summary["deleted_existing"] = delete_existing_taxonomy_tags(
                conn,
                taxonomy=args.taxonomy,
                tags=tuple(t for t, _u in TAGS_BY_COL.values()),
            )
            if not args.dry_run:
                conn.commit()

        if args.zip_path:
            content = Path(args.zip_path).expanduser().read_bytes()
        else:
            content = fetch_simfin_dataset(
                api_key=api_key,
                dataset="income",
                variant="annual",
                market=args.market,
                timeout=args.timeout,
            )

        for row in iter_simfin_rows_from_bytes(content):
            summary["rows_read"] += 1

            raw_ticker = row.get(COL_TICKER, "")
            symbol = match_symbol_to_universe(raw_ticker, universe_symbols)
            if symbol is None:
                continue
            summary["rows_matched_universe"] += 1

            cik = resolve_cik(symbol_to_cik, sec_map, symbol=symbol, cik_hint=None, overrides=overrides)
            if not cik:
                summary["rows_skipped_missing_id"] += 1
                continue

            end_dt = parse_date(row.get(COL_REPORT_DATE))
            if end_dt is None:
                summary["rows_skipped_bad_dates"] += 1
                continue
            if only_before_date is not None and end_dt >= only_before_date:
                continue

            end_date = end_dt.isoformat()
            start_date = (end_dt - dt.timedelta(days=365) + dt.timedelta(days=1)).isoformat()

            fiscal_year: int | None = None
            try:
                fy_raw = (row.get(COL_FISCAL_YEAR) or "").strip()
                fiscal_year = int(fy_raw) if fy_raw else end_dt.year
            except ValueError:
                fiscal_year = end_dt.year

            fiscal_period = (row.get(COL_FISCAL_PERIOD) or "FY").strip().upper() or "FY"

            facts_to_write: list[tuple[str, str, float]] = []
            for col, (tag, unit) in TAGS_BY_COL.items():
                v = parse_float(row.get(col))
                if v is None:
                    continue
                facts_to_write.append((tag, unit, v))
            if not facts_to_write:
                summary["rows_skipped_no_values"] += 1
                continue

            fetched_at = now_utc()
            filed_date = end_date
            form = f"simfin:{args.market}"

            for tag, unit, v in facts_to_write:
                accession = f"import:simfin:{tag}:{end_date}"
                if args.dry_run:
                    continue
                upsert_company_fact_value(
                    conn,
                    cik=cik,
                    taxonomy=args.taxonomy,
                    tag=tag,
                    unit=unit,
                    end_date=end_date,
                    start_date=start_date,
                    value=v,
                    accession=accession,
                    fiscal_year=fiscal_year,
                    fiscal_period=fiscal_period,
                    form=form,
                    filed_date=filed_date,
                    frame=None,
                    fetched_at=fetched_at,
                )
                summary["facts_upserted"] += 1

        if not args.dry_run:
            conn.commit()

    summary["completed_at"] = now_utc()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import SimFin bulk fundamentals into free_data.db.")
    parser.add_argument("--data-db", default="data/free_data.db", help="Path to free-data SQLite DB.")
    parser.add_argument("--taxonomy", default="simfin", help="Value for company_facts_values.taxonomy.")
    parser.add_argument("--api-key", default=os.getenv("SIMFIN_API_KEY", ""), help="SimFin API key.")
    parser.add_argument("--market", default="us", help="SimFin market code (e.g. us).")
    parser.add_argument(
        "--universe",
        default="sp500",
        choices=["sp500", "all"],
        help="Import scope. sp500 imports only current constituents; all imports everything it can map to CIK.",
    )
    parser.add_argument(
        "--as-of-constituents-date",
        default=None,
        help="Use this constituents snapshot for symbol->CIK mapping (default: latest).",
    )
    parser.add_argument("--only-before", default=None, help="Only import rows with Report Date < YYYY-MM-DD.")
    parser.add_argument("--zip-path", default=None, help="Import from a local SimFin zip/csv file instead of downloading.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        default=False,
        help="Delete existing rows for this taxonomy+tags before importing.",
    )
    parser.add_argument("--dry-run", action="store_true", default=False, help="Parse and validate input without DB writes.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = import_simfin_income_annual(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
