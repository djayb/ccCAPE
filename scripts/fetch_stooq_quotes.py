#!/usr/bin/env python3
"""Fetch latest Stooq quotes (bulk) and store into `daily_prices`.

Why
----
The existing pipeline fetches full daily history per symbol using the Stooq
download endpoint (`q/d/l`). That is useful for backfills, but it is request-
heavy for routine refresh.

Stooq also offers a *quote* endpoint (`q/l`) that supports requesting many
tickers at once using `+` separators (e.g. `aapl.us+msft.us`). This script
uses that endpoint to refresh the latest close for many symbols with far
fewer HTTP requests.

Notes
-----
- Prices are stored with `source='stooq'` so existing calc/backfill logic
  picks them up automatically.
- This does not replace historical backfills; it only refreshes the latest
  quote date observed by Stooq per ticker.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Iterable

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from internal_jira import now_utc  # noqa: E402

from scripts._fundamentals_import import (  # noqa: E402
    connect_data_db,
    normalize_symbol,
    table_exists,
)


STOOQ_QUOTES_URL = "https://stooq.com/q/l/"


def load_constituent_symbols(conn: sqlite3.Connection, as_of_date: str | None) -> tuple[str, list[str]]:
    if not table_exists(conn, "sp500_constituents"):
        return "", []
    if not as_of_date:
        row = conn.execute("SELECT MAX(as_of_date) AS as_of_date FROM sp500_constituents").fetchone()
        as_of_date = row["as_of_date"] if row else None
    if not as_of_date:
        return "", []
    rows = conn.execute(
        "SELECT symbol FROM sp500_constituents WHERE as_of_date = ? ORDER BY symbol",
        (as_of_date,),
    ).fetchall()
    return as_of_date, [normalize_symbol(r["symbol"]) for r in rows if normalize_symbol(r["symbol"])]


def stooq_ticker_for_symbol(symbol: str) -> str:
    # Stooq uses hyphen for class-share tickers (e.g. BRK-B, BF-B).
    return normalize_symbol(symbol).lower().replace(".", "-") + ".us"


def chunk_tickers(tickers: list[str], *, max_chunk: int, max_url_len: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    cur: list[str] = []
    # Conservative overhead allowance for URL + query params.
    base_len = len(STOOQ_QUOTES_URL) + len("?s=&f=sd2c&h&e=csv")
    for t in tickers:
        candidate = cur + [t]
        joined = "+".join(candidate)
        url_len = base_len + len(joined)
        if cur and (len(cur) >= max_chunk or url_len > max_url_len):
            chunks.append(cur)
            cur = [t]
        else:
            cur.append(t)
    if cur:
        chunks.append(cur)
    return chunks


def parse_quote_csv(body: str) -> Iterable[dict[str, str]]:
    reader = csv.DictReader(body.splitlines())
    for row in reader:
        yield {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k is not None}


def persist_price(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    price_date: str,
    close_price: float,
    fetched_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO daily_prices (symbol, price_date, close_price, source, fetched_at)
        VALUES (?, ?, ?, 'stooq', ?)
        ON CONFLICT(symbol, price_date, source)
        DO UPDATE SET
            close_price = excluded.close_price,
            fetched_at = excluded.fetched_at
        """,
        (normalize_symbol(symbol), price_date, float(close_price), fetched_at),
    )


def fetch_and_store_quotes(args: argparse.Namespace) -> dict[str, Any]:
    started_at = now_utc()
    summary: dict[str, Any] = {
        "started_at": started_at,
        "completed_at": None,
        "data_db": args.data_db,
        "as_of_constituents_date": args.as_of_constituents_date,
        "symbols_total": 0,
        "symbols_requested": 0,
        "symbols_priced": 0,
        "symbols_failed": 0,
        "failed_examples": [],
        "quotes_max_date": None,
        "quotes_min_date": None,
        "requests_made": 0,
        "status": "success",
    }

    with connect_data_db(args.data_db) as conn:
        if not table_exists(conn, "daily_prices"):
            raise SystemExit("Missing table daily_prices. Run scripts/free_data_pipeline.py at least once first.")

        as_of_date, symbols = load_constituent_symbols(conn, args.as_of_constituents_date)
        if not symbols:
            raise SystemExit("No constituents found. Run scripts/free_data_pipeline.py first.")
        summary["as_of_constituents_date"] = as_of_date

        if args.max_symbols > 0:
            symbols = symbols[: args.max_symbols]

        tickers = [stooq_ticker_for_symbol(s) for s in symbols]
        ticker_base_to_symbol = {t.split(".")[0].lower(): sym for t, sym in zip(tickers, symbols)}

        summary["symbols_total"] = len(symbols)
        summary["symbols_requested"] = len(symbols)

        chunks = chunk_tickers(tickers, max_chunk=args.chunk_size, max_url_len=args.max_url_len)

        session = requests.Session()
        failures: list[str] = []
        min_dt: dt.date | None = None
        max_dt: dt.date | None = None

        for idx, chunk in enumerate(chunks, start=1):
            # Stooq expects raw '+' separators in the query-string for multi-ticker requests.
            joined = "+".join(chunk)
            url = f"{STOOQ_QUOTES_URL}?s={joined}&f=sd2c&h&e=csv"
            resp = session.get(url, timeout=args.timeout)
            summary["requests_made"] += 1
            if resp.status_code != 200:
                # Treat as transient; record failures and continue.
                for t in chunk:
                    base = t.split(".")[0].lower()
                    failures.append(ticker_base_to_symbol.get(base, base))
                continue

            body = resp.text.strip()
            if not body:
                for t in chunk:
                    base = t.split(".")[0].lower()
                    failures.append(ticker_base_to_symbol.get(base, base))
                continue

            fetched_at = now_utc()
            for row in parse_quote_csv(body):
                sym_raw = (row.get("Symbol") or "").strip()
                date_raw = (row.get("Date") or "").strip()
                close_raw = (row.get("Close") or "").strip()

                if not sym_raw:
                    continue

                ticker_base = sym_raw.lower().strip()
                if ticker_base.endswith(".us"):
                    ticker_base = ticker_base[:-3]
                ticker_base = ticker_base.split(".")[0]

                symbol = ticker_base_to_symbol.get(ticker_base)
                if not symbol:
                    continue

                if not date_raw or date_raw.upper() == "N/D":
                    failures.append(symbol)
                    continue
                if not close_raw or close_raw.upper() == "N/D":
                    failures.append(symbol)
                    continue
                try:
                    close_val = float(close_raw)
                except ValueError:
                    failures.append(symbol)
                    continue
                if not math.isfinite(close_val) or close_val <= 0:
                    failures.append(symbol)
                    continue

                persist_price(conn, symbol=symbol, price_date=date_raw, close_price=close_val, fetched_at=fetched_at)
                summary["symbols_priced"] += 1

                try:
                    d = dt.date.fromisoformat(date_raw[:10])
                except ValueError:
                    continue
                if min_dt is None or d < min_dt:
                    min_dt = d
                if max_dt is None or d > max_dt:
                    max_dt = d

            if args.request_delay > 0 and idx < len(chunks):
                time.sleep(args.request_delay)

        conn.commit()

        # Dedup failures to canonical symbols for reporting.
        failures_norm = []
        seen = set()
        for item in failures:
            sym = normalize_symbol(item)
            if not sym or sym in seen:
                continue
            seen.add(sym)
            failures_norm.append(sym)

        summary["symbols_failed"] = len(failures_norm)
        summary["failed_examples"] = failures_norm[:25]
        summary["quotes_min_date"] = min_dt.isoformat() if min_dt else None
        summary["quotes_max_date"] = max_dt.isoformat() if max_dt else None
        if failures_norm:
            summary["status"] = "partial_failure"

        # Optional: record in ingestion_runs if available.
        if table_exists(conn, "ingestion_runs"):
            conn.execute(
                """
                INSERT INTO ingestion_runs (run_started_at, run_completed_at, step, status, details_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    started_at,
                    now_utc(),
                    "prices_quotes",
                    summary["status"],
                    json.dumps(summary, sort_keys=True),
                ),
            )
            conn.commit()

    summary["completed_at"] = now_utc()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch latest Stooq quotes in bulk and store into free_data.db.")
    parser.add_argument("--data-db", default="data/free_data.db", help="Path to free-data SQLite DB.")
    parser.add_argument("--as-of-constituents-date", default=None, help="Use this constituents snapshot (YYYY-MM-DD).")
    parser.add_argument("--max-symbols", type=int, default=0, help="Optional cap on symbols processed (0 = all).")
    parser.add_argument("--chunk-size", type=int, default=75, help="Max tickers per quote request.")
    parser.add_argument("--max-url-len", type=int, default=1800, help="Max URL length guardrail for batching.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    parser.add_argument("--request-delay", type=float, default=0.15, help="Delay between quote requests (seconds).")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = fetch_and_store_quotes(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
