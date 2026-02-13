#!/usr/bin/env python3
"""Free/public data ingestion pipeline for CC CAPE prototyping.

Sources:
- Wikipedia S&P 500 constituents page (constituent universe + CIK).
- SEC company facts endpoint (fundamentals by CIK).
- FRED CPI series (real earnings deflator input).
- Stooq daily CSV endpoint (free historical close prices).

This script stores data in a dedicated SQLite database and can
automatically post run updates into the internal tracker.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from internal_jira import (
    connect as tracker_connect,
    ensure_default_admin,
    get_epic,
    get_issue,
    get_or_create_issue,
    get_project,
    init_db as init_tracker_db,
    now_utc,
)

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_COMPANY_FACTS_URL_TMPL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
FRED_CPI_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
STOOQ_DAILY_URL_TMPL = "https://stooq.com/q/d/l/?s={symbol}.us&i=d"

DEFAULT_TAGS = (
    "NetIncomeLoss",
    "EarningsPerShareBasic",
    "EarningsPerShareDiluted",
)

DATA_SCHEMA = """
CREATE TABLE IF NOT EXISTS sp500_constituents (
    symbol TEXT NOT NULL,
    security TEXT,
    gics_sector TEXT,
    gics_sub_industry TEXT,
    headquarters TEXT,
    date_added TEXT,
    cik TEXT,
    founded TEXT,
    source_url TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (symbol, as_of_date)
);

CREATE TABLE IF NOT EXISTS sec_ticker_map (
    symbol TEXT PRIMARY KEY,
    company_name TEXT,
    cik TEXT,
    exchange TEXT,
    source_url TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cpi_observations (
    observation_date TEXT PRIMARY KEY,
    cpi_value REAL NOT NULL,
    source_url TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS company_facts_meta (
    cik TEXT PRIMARY KEY,
    entity_name TEXT,
    source_url TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS company_facts_values (
    cik TEXT NOT NULL,
    taxonomy TEXT NOT NULL,
    tag TEXT NOT NULL,
    unit TEXT NOT NULL,
    end_date TEXT,
    start_date TEXT,
    value REAL,
    accession TEXT,
    fiscal_year INTEGER,
    fiscal_period TEXT,
    form TEXT,
    filed_date TEXT,
    frame TEXT,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (cik, taxonomy, tag, unit, end_date, accession)
);

CREATE TABLE IF NOT EXISTS daily_prices (
    symbol TEXT NOT NULL,
    price_date TEXT NOT NULL,
    close_price REAL NOT NULL,
    source TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (symbol, price_date, source)
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_started_at TEXT NOT NULL,
    run_completed_at TEXT,
    step TEXT NOT NULL,
    status TEXT NOT NULL,
    details_json TEXT
);
"""


class PipelineError(Exception):
    """Raised for non-transient pipeline failures."""


def ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def connect_data_db(path: str) -> sqlite3.Connection:
    ensure_parent_dir(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_data_db(conn: sqlite3.Connection) -> None:
    conn.executescript(DATA_SCHEMA)
    conn.commit()


def record_step_run(
    conn: sqlite3.Connection,
    *,
    run_started_at: str,
    step: str,
    status: str,
    details: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO ingestion_runs (run_started_at, run_completed_at, step, status, details_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (run_started_at, now_utc(), step, status, json.dumps(details, sort_keys=True)),
    )
    conn.commit()


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def normalize_cik(cik_value: str | int | None) -> str | None:
    if cik_value is None:
        return None
    raw = str(cik_value).strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None
    return digits.lstrip("0") or "0"


def cik10(cik_value: str | int) -> str:
    normalized = normalize_cik(cik_value)
    if normalized is None:
        raise ValueError(f"Invalid CIK value: {cik_value!r}")
    return normalized.zfill(10)


def parse_wikipedia_constituents(html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="constituents")
    if table is None:
        table = soup.find("table", class_="wikitable")
    if table is None:
        raise PipelineError("Could not find S&P 500 constituents table on Wikipedia page.")

    rows = []
    tbody = table.find("tbody")
    if tbody is None:
        raise PipelineError("Constituents table missing tbody section.")

    for tr in tbody.find_all("tr"):
        cols = [cell.get_text(" ", strip=True) for cell in tr.find_all("td")]
        if len(cols) < 8:
            continue
        rows.append(
            {
                "symbol": normalize_symbol(cols[0]),
                "security": cols[1],
                "gics_sector": cols[2],
                "gics_sub_industry": cols[3],
                "headquarters": cols[4],
                "date_added": cols[5],
                "cik": normalize_cik(cols[6]) or "",
                "founded": cols[7],
            }
        )
    if not rows:
        raise PipelineError("No constituents parsed from Wikipedia table.")
    return rows


def fetch_sp500_constituents(session: requests.Session, conn: sqlite3.Connection) -> dict[str, Any]:
    response = session.get(WIKI_SP500_URL, timeout=45)
    response.raise_for_status()
    rows = parse_wikipedia_constituents(response.text)
    as_of = dt.date.today().isoformat()
    ts = now_utc()

    for row in rows:
        conn.execute(
            """
            INSERT INTO sp500_constituents
            (symbol, security, gics_sector, gics_sub_industry, headquarters, date_added,
             cik, founded, source_url, as_of_date, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, as_of_date)
            DO UPDATE SET
                security = excluded.security,
                gics_sector = excluded.gics_sector,
                gics_sub_industry = excluded.gics_sub_industry,
                headquarters = excluded.headquarters,
                date_added = excluded.date_added,
                cik = excluded.cik,
                founded = excluded.founded,
                source_url = excluded.source_url,
                ingested_at = excluded.ingested_at
            """,
            (
                row["symbol"],
                row["security"],
                row["gics_sector"],
                row["gics_sub_industry"],
                row["headquarters"],
                row["date_added"],
                row["cik"],
                row["founded"],
                WIKI_SP500_URL,
                as_of,
                ts,
            ),
        )
    conn.commit()
    return {
        "status": "success",
        "records": len(rows),
        "as_of_date": as_of,
    }


def parse_sec_ticker_map(payload: dict[str, Any]) -> list[dict[str, str]]:
    fields = payload.get("fields")
    data = payload.get("data")
    parsed: list[dict[str, str]] = []

    if isinstance(fields, list) and isinstance(data, list):
        for row in data:
            if not isinstance(row, list):
                continue
            item = dict(zip(fields, row))
            symbol = normalize_symbol(str(item.get("ticker", "")))
            if not symbol:
                continue
            parsed.append(
                {
                    "symbol": symbol,
                    "company_name": str(item.get("name", "")).strip(),
                    "cik": normalize_cik(item.get("cik")) or "",
                    "exchange": str(item.get("exchange", "")).strip(),
                }
            )
        return parsed

    for item in payload.values():
        if not isinstance(item, dict):
            continue
        symbol = normalize_symbol(str(item.get("ticker", "")))
        if not symbol:
            continue
        parsed.append(
            {
                "symbol": symbol,
                "company_name": str(item.get("title", "")).strip(),
                "cik": normalize_cik(item.get("cik_str")) or "",
                "exchange": "",
            }
        )
    return parsed


def fetch_sec_ticker_map(session: requests.Session, conn: sqlite3.Connection) -> dict[str, Any]:
    response = session.get(SEC_TICKER_MAP_URL, timeout=45)
    if response.status_code == 403:
        return {
            "status": "skipped",
            "records": 0,
            "reason": "SEC ticker map endpoint returned HTTP 403 in this environment.",
        }
    response.raise_for_status()
    payload = response.json()
    rows = parse_sec_ticker_map(payload)
    if not rows:
        raise PipelineError("SEC ticker map response parsed to zero rows.")

    ts = now_utc()
    for row in rows:
        conn.execute(
            """
            INSERT INTO sec_ticker_map (symbol, company_name, cik, exchange, source_url, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol)
            DO UPDATE SET
                company_name = excluded.company_name,
                cik = excluded.cik,
                exchange = excluded.exchange,
                source_url = excluded.source_url,
                ingested_at = excluded.ingested_at
            """,
            (
                row["symbol"],
                row["company_name"],
                row["cik"],
                row["exchange"],
                SEC_TICKER_MAP_URL,
                ts,
            ),
        )
    conn.commit()
    return {
        "status": "success",
        "records": len(rows),
    }


def load_latest_constituents(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    latest = conn.execute("SELECT MAX(as_of_date) AS as_of_date FROM sp500_constituents").fetchone()
    as_of_date = latest["as_of_date"] if latest else None
    if as_of_date is None:
        return []
    return conn.execute(
        """
        SELECT *
        FROM sp500_constituents
        WHERE as_of_date = ?
        ORDER BY symbol
        """,
        (as_of_date,),
    ).fetchall()


def resolve_cik_for_symbol(conn: sqlite3.Connection, symbol: str, wiki_cik: str) -> str | None:
    if wiki_cik:
        return normalize_cik(wiki_cik)

    query_symbols = {
        normalize_symbol(symbol),
        normalize_symbol(symbol).replace(".", "-"),
        normalize_symbol(symbol).replace("-", "."),
    }
    for candidate in query_symbols:
        row = conn.execute(
            "SELECT cik FROM sec_ticker_map WHERE symbol = ?",
            (candidate,),
        ).fetchone()
        if row and row["cik"]:
            return normalize_cik(row["cik"])
    return None


def fetch_fred_cpi_csv(session: requests.Session, conn: sqlite3.Connection) -> dict[str, Any]:
    response = session.get(FRED_CPI_CSV_URL, timeout=45)
    response.raise_for_status()

    reader = csv.DictReader(response.text.splitlines())
    ts = now_utc()
    inserted = 0
    for row in reader:
        date_value = (row.get("observation_date") or "").strip()
        cpi_raw = (row.get("CPIAUCSL") or "").strip()
        if not date_value or not cpi_raw or cpi_raw == ".":
            continue
        try:
            cpi_val = float(cpi_raw)
        except ValueError:
            continue
        conn.execute(
            """
            INSERT INTO cpi_observations (observation_date, cpi_value, source_url, ingested_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(observation_date)
            DO UPDATE SET
                cpi_value = excluded.cpi_value,
                source_url = excluded.source_url,
                ingested_at = excluded.ingested_at
            """,
            (date_value, cpi_val, FRED_CPI_CSV_URL, ts),
        )
        inserted += 1
    conn.commit()
    return {
        "status": "success",
        "records": inserted,
    }


def stooq_symbol_variants(symbol: str) -> list[str]:
    base = symbol.lower()
    variants = [
        base,
        base.replace(".", "-"),
        base.replace("-", "."),
    ]
    deduped = []
    seen = set()
    for item in variants:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def fetch_symbol_prices_stooq(
    session: requests.Session,
    *,
    symbol: str,
    max_rows: int | None,
) -> tuple[int, str | None]:
    for variant in stooq_symbol_variants(symbol):
        url = STOOQ_DAILY_URL_TMPL.format(symbol=variant)
        response = session.get(url, timeout=45)
        if response.status_code != 200:
            continue
        body = response.text.strip()
        if not body or body.startswith("No data"):
            continue
        reader = csv.DictReader(body.splitlines())
        parsed = []
        for row in reader:
            date_value = (row.get("Date") or "").strip()
            close_raw = (row.get("Close") or "").strip()
            if not date_value or not close_raw or close_raw == "0":
                continue
            try:
                close_val = float(close_raw)
            except ValueError:
                continue
            parsed.append((date_value, close_val))
        if not parsed:
            continue
        if max_rows is not None and max_rows > 0:
            parsed = parsed[-max_rows:]
        return len(parsed), body
    return 0, None


def persist_symbol_prices(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    csv_body: str,
    max_rows: int | None,
) -> int:
    reader = csv.DictReader(csv_body.splitlines())
    rows = []
    for row in reader:
        date_value = (row.get("Date") or "").strip()
        close_raw = (row.get("Close") or "").strip()
        if not date_value or not close_raw or close_raw == "0":
            continue
        try:
            close_val = float(close_raw)
        except ValueError:
            continue
        rows.append((date_value, close_val))
    if max_rows is not None and max_rows > 0:
        rows = rows[-max_rows:]

    ts = now_utc()
    for date_value, close_val in rows:
        conn.execute(
            """
            INSERT INTO daily_prices (symbol, price_date, close_price, source, fetched_at)
            VALUES (?, ?, ?, 'stooq', ?)
            ON CONFLICT(symbol, price_date, source)
            DO UPDATE SET
                close_price = excluded.close_price,
                fetched_at = excluded.fetched_at
            """,
            (normalize_symbol(symbol), date_value, close_val, ts),
        )
    return len(rows)


def fetch_prices_stooq(
    session: requests.Session,
    conn: sqlite3.Connection,
    *,
    symbols: list[str],
    symbol_limit: int,
    max_rows_per_symbol: int,
    request_delay: float,
) -> dict[str, Any]:
    selected_symbols = symbols[:symbol_limit] if symbol_limit > 0 else symbols
    inserted_rows = 0
    failures = []

    for idx, symbol in enumerate(selected_symbols, start=1):
        count, body = fetch_symbol_prices_stooq(
            session,
            symbol=symbol,
            max_rows=max_rows_per_symbol if max_rows_per_symbol > 0 else None,
        )
        if count == 0 or body is None:
            failures.append(symbol)
        else:
            inserted_rows += persist_symbol_prices(
                conn,
                symbol=symbol,
                csv_body=body,
                max_rows=max_rows_per_symbol if max_rows_per_symbol > 0 else None,
            )
        if request_delay > 0 and idx < len(selected_symbols):
            time.sleep(request_delay)

    conn.commit()
    return {
        "status": "success",
        "symbols_requested": len(selected_symbols),
        "rows_written": inserted_rows,
        "symbol_failures": len(failures),
        "failed_symbols": failures[:20],
    }


def extract_selected_tag_units(
    payload: dict[str, Any],
    selected_tags: set[str],
) -> list[tuple[str, str, str, dict[str, Any]]]:
    output = []
    facts = payload.get("facts", {})
    if not isinstance(facts, dict):
        return output
    for taxonomy, taxonomy_data in facts.items():
        if not isinstance(taxonomy_data, dict):
            continue
        for tag, tag_payload in taxonomy_data.items():
            if tag not in selected_tags:
                continue
            units = tag_payload.get("units", {})
            if not isinstance(units, dict):
                continue
            for unit_name, observations in units.items():
                if not isinstance(observations, list):
                    continue
                for obs in observations:
                    if isinstance(obs, dict):
                        output.append((taxonomy, tag, unit_name, obs))
    return output


def fetch_company_facts(
    session: requests.Session,
    conn: sqlite3.Connection,
    *,
    constituents: list[sqlite3.Row],
    cik_limit: int,
    request_delay: float,
    selected_tags: tuple[str, ...],
) -> dict[str, Any]:
    target = []
    missing_cik = []
    for row in constituents:
        cik_value = resolve_cik_for_symbol(conn, row["symbol"], row["cik"])
        if cik_value is None:
            missing_cik.append(row["symbol"])
            continue
        target.append((row["symbol"], cik_value))
    if cik_limit > 0:
        target = target[:cik_limit]

    tag_set = set(selected_tags)
    ts = now_utc()
    fetched = 0
    failed = []
    observations_written = 0

    for idx, (symbol, cik_value) in enumerate(target, start=1):
        url = SEC_COMPANY_FACTS_URL_TMPL.format(cik=cik10(cik_value))
        response = session.get(url, timeout=45)
        if response.status_code != 200:
            failed.append({"symbol": symbol, "cik": cik_value, "http_status": response.status_code})
            if request_delay > 0 and idx < len(target):
                time.sleep(request_delay)
            continue
        payload = response.json()
        conn.execute(
            """
            INSERT INTO company_facts_meta (cik, entity_name, source_url, fetched_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(cik)
            DO UPDATE SET
                entity_name = excluded.entity_name,
                source_url = excluded.source_url,
                fetched_at = excluded.fetched_at
            """,
            (normalize_cik(payload.get("cik")) or cik_value, payload.get("entityName"), url, ts),
        )

        tag_rows = extract_selected_tag_units(payload, tag_set)
        for taxonomy, tag, unit_name, obs in tag_rows:
            val = obs.get("val")
            try:
                numeric_val = float(val) if val is not None else None
            except (TypeError, ValueError):
                numeric_val = None
            conn.execute(
                """
                INSERT INTO company_facts_values
                (cik, taxonomy, tag, unit, end_date, start_date, value, accession, fiscal_year,
                 fiscal_period, form, filed_date, frame, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cik, taxonomy, tag, unit, end_date, accession)
                DO UPDATE SET
                    start_date = excluded.start_date,
                    value = excluded.value,
                    fiscal_year = excluded.fiscal_year,
                    fiscal_period = excluded.fiscal_period,
                    form = excluded.form,
                    filed_date = excluded.filed_date,
                    frame = excluded.frame,
                    fetched_at = excluded.fetched_at
                """,
                (
                    normalize_cik(payload.get("cik")) or cik_value,
                    taxonomy,
                    tag,
                    unit_name,
                    obs.get("end"),
                    obs.get("start"),
                    numeric_val,
                    obs.get("accn"),
                    obs.get("fy"),
                    obs.get("fp"),
                    obs.get("form"),
                    obs.get("filed"),
                    obs.get("frame"),
                    ts,
                ),
            )
            observations_written += 1
        fetched += 1

        if request_delay > 0 and idx < len(target):
            time.sleep(request_delay)

    conn.commit()
    return {
        "status": "success",
        "constituents_total": len(constituents),
        "cik_resolved": len(target),
        "missing_cik": len(missing_cik),
        "missing_cik_symbols": missing_cik[:25],
        "facts_fetched": fetched,
        "facts_failed": len(failed),
        "failed_examples": failed[:10],
        "observations_written": observations_written,
        "selected_tags": list(selected_tags),
    }


def run_quality_checks(conn: sqlite3.Connection) -> dict[str, Any]:
    latest_constituents = load_latest_constituents(conn)
    total_constituents = len(latest_constituents)
    missing_cik = sum(1 for row in latest_constituents if not resolve_cik_for_symbol(conn, row["symbol"], row["cik"]))

    latest_price_count = conn.execute(
        """
        SELECT COUNT(DISTINCT symbol) AS count_symbols
        FROM daily_prices
        WHERE source = 'stooq'
        """
    ).fetchone()["count_symbols"]

    facts_cik_count = conn.execute(
        "SELECT COUNT(DISTINCT cik) AS count_cik FROM company_facts_meta"
    ).fetchone()["count_cik"]

    return {
        "status": "success",
        "constituent_count": total_constituents,
        "missing_cik_count": missing_cik,
        "priced_symbol_count": int(latest_price_count or 0),
        "facts_cik_count": int(facts_cik_count or 0),
    }


def tracker_add_comment(conn: sqlite3.Connection, issue_key: str, author: str, body: str) -> bool:
    issue = get_issue(conn, issue_key)
    if issue is None:
        return False
    ts = now_utc()
    conn.execute(
        """
        INSERT INTO issue_comments (issue_id, author, body, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (issue["id"], author, body, ts),
    )
    conn.execute(
        """
        INSERT INTO issue_events (issue_id, event_type, old_value, new_value, actor, created_at)
        VALUES (?, 'comment_added', NULL, NULL, ?, ?)
        """,
        (issue["id"], author, ts),
    )
    conn.execute(
        "UPDATE issues SET updated_at = ? WHERE id = ?",
        (ts, issue["id"]),
    )
    conn.commit()
    return True


def tracker_move_status_if_needed(conn: sqlite3.Connection, issue_key: str, target_status: str, actor: str) -> bool:
    issue = get_issue(conn, issue_key)
    if issue is None:
        return False
    if issue["status"] == target_status:
        return False
    ts = now_utc()
    conn.execute(
        "UPDATE issues SET status = ?, updated_at = ? WHERE id = ?",
        (target_status, ts, issue["id"]),
    )
    conn.execute(
        """
        INSERT INTO issue_events (issue_id, event_type, old_value, new_value, actor, created_at)
        VALUES (?, 'status_changed', ?, ?, ?, ?)
        """,
        (issue["id"], issue["status"], target_status, actor, ts),
    )
    conn.commit()
    return True


def ensure_free_data_issue(conn: sqlite3.Connection) -> str | None:
    project = get_project(conn, "CAPE")
    epic = get_epic(conn, "CAPE-EP2")
    if not project or not epic:
        return None
    created_issue = get_or_create_issue(
        conn,
        project=project,
        epic=epic,
        issue_type="story",
        title="Implement free-data ingestion pipeline (SEC/FRED/Wikipedia/Stooq)",
        description=(
            "Build and operate a no-paid-data ingestion pipeline for CC CAPE prototyping. "
            "Includes constituents, company facts, CPI, and free historical prices."
        ),
        priority="p1",
        due_date=epic["target_date"] or "",
        story_points=8,
        sprint="Phase-1",
    )
    conn.commit()
    return created_issue["key"]


def update_tracker_with_summary(
    *,
    tracker_db_path: str,
    summary: dict[str, Any],
    author: str,
) -> dict[str, Any]:
    tracker_results = {
        "tracker_db": tracker_db_path,
        "comments_added": 0,
        "statuses_changed": 0,
        "tracked_issues": [],
    }
    with tracker_connect(tracker_db_path) as conn:
        init_tracker_db(conn)
        ensure_default_admin(conn)

        free_data_issue_key = ensure_free_data_issue(conn)
        if free_data_issue_key:
            tracker_results["tracked_issues"].append(free_data_issue_key)

        body = (
            f"Free-data pipeline run completed at {summary['completed_at']} UTC.\n\n"
            f"Constituents: {summary['steps']['constituents'].get('records', 0)}\n"
            f"SEC ticker map: {summary['steps']['sec_ticker_map'].get('status')} "
            f"({summary['steps']['sec_ticker_map'].get('records', 0)} rows)\n"
            f"CPI: {summary['steps']['cpi'].get('status')} "
            f"({summary['steps']['cpi'].get('records', 0)} rows)\n"
            f"Company facts: {summary['steps']['company_facts'].get('facts_fetched', 0)} fetched, "
            f"{summary['steps']['company_facts'].get('facts_failed', 0)} failed, "
            f"{summary['steps']['company_facts'].get('missing_cik', 0)} missing CIK\n"
            f"Prices: {summary['steps']['prices'].get('rows_written', 0)} rows, "
            f"{summary['steps']['prices'].get('symbol_failures', 0)} symbol failures\n"
            f"Quality checks: missing CIK {summary['steps']['quality_checks'].get('missing_cik_count', 0)}, "
            f"priced symbols {summary['steps']['quality_checks'].get('priced_symbol_count', 0)}, "
            f"facts CIKs {summary['steps']['quality_checks'].get('facts_cik_count', 0)}"
        )

        for issue_key in [free_data_issue_key, "CAPE-4", "CAPE-5", "CAPE-6"]:
            if not issue_key:
                continue
            if tracker_add_comment(conn, issue_key, author, body):
                tracker_results["comments_added"] += 1
                tracker_results["tracked_issues"].append(issue_key)

        for issue_key in ("CAPE-4", "CAPE-5"):
            if tracker_move_status_if_needed(conn, issue_key, "in_progress", author):
                tracker_results["statuses_changed"] += 1
                tracker_results["tracked_issues"].append(issue_key)

        issue_cape6 = get_issue(conn, "CAPE-6")
        if issue_cape6 and issue_cape6["status"] == "backlog":
            if tracker_move_status_if_needed(conn, "CAPE-6", "todo", author):
                tracker_results["statuses_changed"] += 1
                tracker_results["tracked_issues"].append("CAPE-6")

    tracker_results["tracked_issues"] = sorted(set(tracker_results["tracked_issues"]))
    return tracker_results


def build_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/json,text/csv,text/html;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate",
        }
    )
    return session


def run_pipeline(args: argparse.Namespace) -> int:
    run_started_at = now_utc()
    session = build_session(args.sec_user_agent)
    summary: dict[str, Any] = {
        "started_at": run_started_at,
        "completed_at": None,
        "steps": {},
    }

    with connect_data_db(args.data_db) as conn:
        init_data_db(conn)

        def run_step(step_name: str, fn):
            try:
                result = fn()
                summary["steps"][step_name] = result
                record_step_run(
                    conn,
                    run_started_at=run_started_at,
                    step=step_name,
                    status=result.get("status", "success"),
                    details=result,
                )
            except Exception as error:
                result = {
                    "status": "error",
                    "error": str(error),
                }
                summary["steps"][step_name] = result
                record_step_run(
                    conn,
                    run_started_at=run_started_at,
                    step=step_name,
                    status="error",
                    details=result,
                )

        run_step("constituents", lambda: fetch_sp500_constituents(session, conn))
        run_step("sec_ticker_map", lambda: fetch_sec_ticker_map(session, conn))
        run_step("cpi", lambda: fetch_fred_cpi_csv(session, conn))

        constituents = load_latest_constituents(conn)
        run_step(
            "company_facts",
            lambda: fetch_company_facts(
                session,
                conn,
                constituents=constituents,
                cik_limit=args.facts_limit,
                request_delay=args.request_delay,
                selected_tags=tuple(args.sec_tags),
            ),
        )
        run_step(
            "prices",
            lambda: fetch_prices_stooq(
                session,
                conn,
                symbols=[row["symbol"] for row in constituents],
                symbol_limit=args.prices_symbol_limit,
                max_rows_per_symbol=args.prices_rows_per_symbol,
                request_delay=args.request_delay,
            ),
        )
        run_step("quality_checks", lambda: run_quality_checks(conn))

        summary["completed_at"] = now_utc()
        overall_status = "success"
        for step in summary["steps"].values():
            if step.get("status") == "error":
                overall_status = "partial_failure"
                break

        record_step_run(
            conn,
            run_started_at=run_started_at,
            step="pipeline",
            status=overall_status,
            details=summary,
        )

    if args.update_tracker:
        try:
            tracker_result = update_tracker_with_summary(
                tracker_db_path=args.tracker_db,
                summary=summary,
                author=args.tracker_author,
            )
            summary["tracker"] = tracker_result
        except Exception as error:  # noqa: BLE001
            summary["tracker"] = {"status": "error", "error": str(error)}

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Free/public data ingestion pipeline for CC CAPE.")
    parser.add_argument("--data-db", default="data/free_data.db", help="Path to free-data SQLite database.")
    parser.add_argument("--tracker-db", default="data/internal_jira.db", help="Path to tracker SQLite database.")
    parser.add_argument(
        "--sec-user-agent",
        default=os.getenv("SEC_USER_AGENT", "ccCAPE/0.1 (research@localhost)"),
        help="User-Agent header for SEC requests.",
    )
    parser.add_argument("--facts-limit", type=int, default=25, help="Max number of CIKs to fetch per run.")
    parser.add_argument("--prices-symbol-limit", type=int, default=100, help="Max number of symbols for price fetch.")
    parser.add_argument(
        "--prices-rows-per-symbol",
        type=int,
        default=3650,
        help="Max daily rows per symbol written from Stooq CSV.",
    )
    parser.add_argument("--request-delay", type=float, default=0.2, help="Delay between remote requests in seconds.")
    parser.add_argument(
        "--sec-tags",
        nargs="+",
        default=list(DEFAULT_TAGS),
        help="SEC XBRL tags to persist from company facts.",
    )
    parser.add_argument(
        "--update-tracker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, post run summaries into tracker issues.",
    )
    parser.add_argument("--tracker-author", default="data-bot", help="Author name for tracker comments/events.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
