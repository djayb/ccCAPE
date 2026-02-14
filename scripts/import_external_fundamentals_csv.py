#!/usr/bin/env python3
"""Import external fundamentals (annual-ish) into `data/free_data.db`.

Why this exists
---------------
The free-data pipeline uses SEC XBRL "company facts" for fundamentals. That
coverage is usually limited to the modern XBRL era. If you have an additional
fundamentals dataset (e.g. licensed history, internal research extracts),
you can import it here to extend the CC CAPE time series.

Storage model
-------------
We reuse the `company_facts_values` table and distinguish sources via
`taxonomy` (e.g. `external`, `simfin`, `compustat`).

Expected CSV schema
-------------------
Required:
- `end_date` (YYYY-MM-DD) OR `fiscal_year` (YYYY)
- one of `symbol` or `cik`

Optional:
- `start_date` (YYYY-MM-DD). If missing, we approximate a 1y period ending
  on `end_date` (start_date = end_date - 365 days + 1).
- `fiscal_year`, `fiscal_period` (defaults to FY).

Fundamentals columns (all optional, import what you have):
- `net_income` (USD) -> NetIncomeLoss
- `shares_basic` (shares) -> WeightedAverageNumberOfSharesOutstandingBasic
- `shares_outstanding` (shares) -> EntityCommonStockSharesOutstanding
- `eps_basic` (USD/share) -> EarningsPerShareBasic
- `eps_diluted` (USD/share) -> EarningsPerShareDiluted
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from internal_jira import now_utc  # noqa: E402

from scripts._fundamentals_import import (  # noqa: E402
    connect_data_db,
    delete_existing_taxonomy_tags,
    load_latest_constituents_symbol_to_cik,
    load_sec_ticker_map_symbol_to_cik,
    normalize_symbol,
    parse_date,
    parse_float,
    resolve_cik,
    table_exists,
    upsert_company_fact_value,
)


TAGS = {
    "net_income": ("NetIncomeLoss", "USD"),
    "shares_basic": ("WeightedAverageNumberOfSharesOutstandingBasic", "shares"),
    "shares_outstanding": ("EntityCommonStockSharesOutstanding", "shares"),
    "eps_basic": ("EarningsPerShareBasic", "USD/shares"),
    "eps_diluted": ("EarningsPerShareDiluted", "USD/shares"),
}


def compute_dates(row: dict[str, Any]) -> tuple[str | None, str | None, int | None]:
    """Return (start_date_iso, end_date_iso, fiscal_year)."""
    end_date = parse_date(row.get("end_date") or "")
    fiscal_year: int | None = None
    if end_date is None:
        fiscal_year_value = row.get("fiscal_year") or row.get("year")
        try:
            fiscal_year = int(str(fiscal_year_value).strip()) if fiscal_year_value not in (None, "") else None
        except ValueError:
            fiscal_year = None
        if fiscal_year:
            end_date = dt.date(fiscal_year, 12, 31)
    if end_date is None:
        return None, None, None

    if fiscal_year is None:
        fiscal_year = end_date.year

    start_date = parse_date(row.get("start_date") or "")
    if start_date is None:
        # Approximate an annual-ish period.
        start_date = end_date - dt.timedelta(days=365) + dt.timedelta(days=1)

    if start_date > end_date:
        return None, None, None

    return start_date.isoformat(), end_date.isoformat(), fiscal_year


def import_csv(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.csv).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")

    summary: dict[str, Any] = {
        "csv": str(path),
        "data_db": args.data_db,
        "taxonomy": args.taxonomy,
        "dry_run": bool(args.dry_run),
        "started_at": now_utc(),
        "rows_read": 0,
        "facts_upserted": 0,
        "rows_skipped_missing_id": 0,
        "rows_skipped_bad_dates": 0,
        "rows_skipped_no_values": 0,
        "deleted_existing": 0,
    }

    with connect_data_db(args.data_db) as conn:
        if not table_exists(conn, "company_facts_values"):
            raise SystemExit("Missing table company_facts_values. Run scripts/free_data_pipeline.py at least once first.")

        symbol_to_cik = load_latest_constituents_symbol_to_cik(conn, args.as_of_constituents_date)
        sec_map = load_sec_ticker_map_symbol_to_cik(conn)

        if args.replace_existing:
            summary["deleted_existing"] = delete_existing_taxonomy_tags(conn, taxonomy=args.taxonomy, tags=tuple(t for t, _u in TAGS.values()))
            if not args.dry_run:
                conn.commit()

        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                summary["rows_read"] += 1

                symbol = normalize_symbol(row.get("symbol") or row.get("ticker") or "")
                cik = resolve_cik(symbol_to_cik, sec_map, symbol=symbol, cik_hint=row.get("cik"))
                if not cik:
                    summary["rows_skipped_missing_id"] += 1
                    continue

                start_date, end_date, fiscal_year = compute_dates(row)
                if not end_date:
                    summary["rows_skipped_bad_dates"] += 1
                    continue

                fiscal_period = (row.get("fiscal_period") or "FY").strip().upper()
                if not fiscal_period:
                    fiscal_period = "FY"

                facts_to_write: list[tuple[str, str, float]] = []
                for col, (tag, unit) in TAGS.items():
                    v = parse_float(row.get(col))
                    if v is None:
                        continue
                    facts_to_write.append((tag, unit, v))
                if not facts_to_write:
                    summary["rows_skipped_no_values"] += 1
                    continue

                fetched_at = now_utc()
                filed_date = end_date
                form = f"external:{args.taxonomy}"
                for tag, unit, v in facts_to_write:
                    accession = f"import:{args.taxonomy}:{tag}:{end_date}"
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
    parser = argparse.ArgumentParser(description="Import external fundamentals CSV into free_data.db.")
    parser.add_argument("--data-db", default="data/free_data.db", help="Path to free-data SQLite DB.")
    parser.add_argument("--csv", required=True, help="Path to CSV to import.")
    parser.add_argument("--taxonomy", default="external", help="Value for company_facts_values.taxonomy (e.g. external, simfin).")
    parser.add_argument(
        "--as-of-constituents-date",
        default=None,
        help="Use this constituents snapshot for symbol->CIK mapping (default: latest).",
    )
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
    summary = import_csv(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

