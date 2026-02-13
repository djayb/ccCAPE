#!/usr/bin/env python3
"""Compute CC CAPE from free-data pipeline outputs.

This module reads `data/free_data.db` populated by `free_data_pipeline.py`,
calculates a research-grade free-data CC CAPE estimate, persists run outputs,
and can post run summaries into the internal tracker.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import datetime as dt
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from internal_jira import (  # noqa: E402
    connect as tracker_connect,
    ensure_default_admin,
    get_epic,
    get_issue,
    get_or_create_issue,
    get_project,
    init_db as init_tracker_db,
    now_utc,
)

CALC_SCHEMA = """
CREATE TABLE IF NOT EXISTS cc_cape_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    as_of_constituents_date TEXT NOT NULL,
    latest_price_date TEXT NOT NULL,
    symbols_total INTEGER NOT NULL,
    symbols_with_price INTEGER NOT NULL,
    symbols_with_valid_cape INTEGER NOT NULL,
    min_eps_points INTEGER NOT NULL,
    lookback_years INTEGER NOT NULL,
    weighting_method TEXT NOT NULL,
    market_cap_coverage REAL NOT NULL,
    cc_cape REAL NOT NULL,
    avg_company_cape REAL NOT NULL,
    shiller_cape REAL,
    shiller_cape_date TEXT,
    cape_spread REAL,
    cc_cape_percentile REAL,
    cc_cape_zscore REAL,
    cape_spread_percentile REAL,
    cape_spread_zscore REAL,
    notes_json TEXT
);

CREATE TABLE IF NOT EXISTS cc_cape_constituent_metrics (
    run_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    cik TEXT NOT NULL,
    gics_sector TEXT,
    price_date TEXT NOT NULL,
    close_price REAL NOT NULL,
    shares_outstanding REAL,
    market_cap REAL,
    eps_tag TEXT NOT NULL,
    eps_points INTEGER NOT NULL,
    avg_real_eps REAL NOT NULL,
    company_cape REAL NOT NULL,
    weight REAL NOT NULL,
    PRIMARY KEY (run_id, symbol),
    FOREIGN KEY (run_id) REFERENCES cc_cape_runs(run_id) ON DELETE CASCADE
);
"""

EPS_TAGS = ("EarningsPerShareBasic", "EarningsPerShareDiluted")
SHARES_TAGS = (
    "EntityCommonStockSharesOutstanding",
    "CommonStockSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
)


def ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def connect_data_db(path: str) -> sqlite3.Connection:
    ensure_parent_dir(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


def init_calc_db(conn: sqlite3.Connection) -> None:
    conn.executescript(CALC_SCHEMA)
    migrate_calc_schema(conn)
    conn.commit()


def migrate_calc_schema(conn: sqlite3.Connection) -> None:
    """Best-effort schema migration for existing SQLite databases."""

    existing_cols = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(cc_cape_runs)").fetchall()
        if row and row["name"]
    }

    def add_col(name: str, col_type: str) -> None:
        if name in existing_cols:
            return
        conn.execute(f"ALTER TABLE cc_cape_runs ADD COLUMN {name} {col_type}")
        existing_cols.add(name)

    add_col("shiller_cape_date", "TEXT")
    add_col("cc_cape_percentile", "REAL")
    add_col("cc_cape_zscore", "REAL")
    add_col("cape_spread_percentile", "REAL")
    add_col("cape_spread_zscore", "REAL")


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


def parse_date(date_value: str | None) -> dt.date | None:
    if not date_value:
        return None
    try:
        return dt.date.fromisoformat(date_value[:10])
    except ValueError:
        return None


def load_latest_constituents(conn: sqlite3.Connection) -> tuple[str, list[sqlite3.Row]]:
    row = conn.execute("SELECT MAX(as_of_date) AS as_of_date FROM sp500_constituents").fetchone()
    if not row or row["as_of_date"] is None:
        return "", []
    as_of_date = row["as_of_date"]
    rows = conn.execute(
        """
        SELECT symbol, cik, gics_sector
        FROM sp500_constituents
        WHERE as_of_date = ?
        ORDER BY symbol
        """,
        (as_of_date,),
    ).fetchall()
    return as_of_date, rows


def load_latest_cpi(conn: sqlite3.Connection) -> tuple[float, str]:
    row = conn.execute(
        """
        SELECT cpi_value, observation_date
        FROM cpi_observations
        ORDER BY observation_date DESC
        LIMIT 1
        """
    ).fetchone()
    if not row:
        raise RuntimeError("No CPI observations found. Run free_data_pipeline first.")
    return float(row["cpi_value"]), row["observation_date"]


def load_cpi_series(conn: sqlite3.Connection) -> tuple[list[dt.date], list[float]]:
    rows = conn.execute(
        """
        SELECT observation_date, cpi_value
        FROM cpi_observations
        ORDER BY observation_date
        """
    ).fetchall()
    dates: list[dt.date] = []
    values: list[float] = []
    for row in rows:
        date_obj = parse_date(row["observation_date"])
        if date_obj is None:
            continue
        dates.append(date_obj)
        values.append(float(row["cpi_value"]))
    return dates, values


def cpi_for_date(target_date: dt.date, cpi_dates: list[dt.date], cpi_values: list[float]) -> float | None:
    if not cpi_dates:
        return None
    idx = bisect_right(cpi_dates, target_date) - 1
    if idx < 0:
        return None
    return cpi_values[idx]


def shiller_cape_asof(conn: sqlite3.Connection, target_date: dt.date) -> tuple[float, str] | None:
    """Return (shiller_cape, observation_date) for the latest observation on/before target_date."""
    try:
        row = conn.execute(
            """
            SELECT observation_date, shiller_cape
            FROM shiller_cape_observations
            WHERE observation_date <= ?
            ORDER BY observation_date DESC
            LIMIT 1
            """,
            (target_date.isoformat(),),
        ).fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None
    obs_date = row["observation_date"]
    return float(row["shiller_cape"]), obs_date


def empirical_percentile(values: list[float], x: float) -> float | None:
    if not values:
        return None
    count_le = sum(1 for v in values if v <= x)
    return 100.0 * (count_le / len(values))


def zscore(values: list[float], x: float) -> float | None:
    if len(values) < 2:
        return None
    stdev = statistics.stdev(values)
    if stdev <= 0:
        return None
    mean = statistics.mean(values)
    return (x - mean) / stdev


def latest_price(conn: sqlite3.Connection, symbol: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT price_date, close_price
        FROM daily_prices
        WHERE symbol = ? AND source = 'stooq'
        ORDER BY price_date DESC
        LIMIT 1
        """,
        (normalize_symbol(symbol),),
    ).fetchone()


def resolve_cik(conn: sqlite3.Connection, symbol: str, cik_hint: str | None) -> str | None:
    if cik_hint:
        normalized = normalize_cik(cik_hint)
        if normalized:
            return normalized
    symbol_candidates = {
        normalize_symbol(symbol),
        normalize_symbol(symbol).replace(".", "-"),
        normalize_symbol(symbol).replace("-", "."),
    }
    for candidate in symbol_candidates:
        row = conn.execute(
            "SELECT cik FROM sec_ticker_map WHERE symbol = ?",
            (candidate,),
        ).fetchone()
        if row and row["cik"]:
            normalized = normalize_cik(row["cik"])
            if normalized:
                return normalized
    return None


def _dedup_latest_by_end_date(rows: list[sqlite3.Row]) -> dict[str, sqlite3.Row]:
    dedup: dict[str, sqlite3.Row] = {}
    for row in rows:
        end_date = row["end_date"] or ""
        if not end_date:
            continue
        current = dedup.get(end_date)
        if current is None:
            dedup[end_date] = row
            continue
        current_filed = current["filed_date"] or ""
        candidate_filed = row["filed_date"] or ""
        if candidate_filed >= current_filed:
            dedup[end_date] = row
    return dedup


def eps_candidates(conn: sqlite3.Connection, cik: str) -> list[tuple[str, list[dict[str, Any]]]]:
    """Return candidate EPS series to try in the CAPE denominator.

    Priority is decided later (after windowing + CPI adjustment). We include:
    - reported EPS basic/diluted (if present)
    - computed EPS from NetIncomeLoss / WeightedAverageNumberOfSharesOutstandingBasic (if present)
    """

    candidates: list[tuple[str, list[dict[str, Any]]]] = []

    rows = conn.execute(
        """
        SELECT tag, end_date, filed_date, value, form, fiscal_period, unit
        FROM company_facts_values
        WHERE cik = ?
          AND tag IN ('EarningsPerShareBasic', 'EarningsPerShareDiluted')
          AND value IS NOT NULL
          AND end_date IS NOT NULL
          AND (unit LIKE 'USD%' OR unit LIKE 'usd%')
          AND (form LIKE '10-K%' OR fiscal_period = 'FY')
        ORDER BY end_date, filed_date
        """,
        (cik,),
    ).fetchall()

    if rows:
        dedup: dict[tuple[str, str], sqlite3.Row] = {}
        for row in rows:
            key = (row["tag"], row["end_date"])
            current = dedup.get(key)
            if current is None:
                dedup[key] = row
            else:
                current_filed = current["filed_date"] or ""
                candidate_filed = row["filed_date"] or ""
                if candidate_filed >= current_filed:
                    dedup[key] = row

        for tag in EPS_TAGS:
            series = [dict(r) for (t, _), r in dedup.items() if t == tag]
            series.sort(key=lambda r: r["end_date"] or "")
            if series:
                candidates.append((tag, series))

    # Fallback: compute EPS = NetIncomeLoss / WeightedAvgSharesBasic when reported EPS is missing.
    net_income_rows = conn.execute(
        """
        SELECT end_date, filed_date, value, form, fiscal_period, unit
        FROM company_facts_values
        WHERE cik = ?
          AND tag = 'NetIncomeLoss'
          AND value IS NOT NULL
          AND end_date IS NOT NULL
          AND (unit LIKE 'USD%' OR unit LIKE 'usd%')
          AND (form LIKE '10-K%' OR fiscal_period = 'FY')
        ORDER BY end_date, filed_date
        """,
        (cik,),
    ).fetchall()
    shares_rows = conn.execute(
        """
        SELECT end_date, filed_date, value, form, fiscal_period, unit
        FROM company_facts_values
        WHERE cik = ?
          AND tag = 'WeightedAverageNumberOfSharesOutstandingBasic'
          AND value IS NOT NULL
          AND value > 0
          AND end_date IS NOT NULL
          AND (form LIKE '10-K%' OR fiscal_period = 'FY')
        ORDER BY end_date, filed_date
        """,
        (cik,),
    ).fetchall()
    if net_income_rows and shares_rows:
        net_by_end = _dedup_latest_by_end_date(net_income_rows)
        shares_by_end = _dedup_latest_by_end_date(shares_rows)
        computed: list[dict[str, Any]] = []
        for end_date, net_row in net_by_end.items():
            shares_row = shares_by_end.get(end_date)
            if not shares_row:
                continue
            try:
                net = float(net_row["value"])
                shares = float(shares_row["value"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(net) or not math.isfinite(shares) or shares <= 0:
                continue
            eps = net / shares
            if not math.isfinite(eps):
                continue
            computed.append(
                {
                    "end_date": end_date,
                    "filed_date": max(net_row["filed_date"] or "", shares_row["filed_date"] or ""),
                    "value": eps,
                    "unit": "USD/shares",
                }
            )
        computed.sort(key=lambda r: r["end_date"] or "")
        if computed:
            candidates.append(("ComputedEPS(NetIncomeLoss/WeightedAvgSharesBasic)", computed))

    return candidates


def latest_shares_outstanding(conn: sqlite3.Connection, cik: str) -> float | None:
    row = conn.execute(
        """
        SELECT value
        FROM company_facts_values
        WHERE cik = ?
          AND tag IN ('EntityCommonStockSharesOutstanding',
                      'CommonStockSharesOutstanding',
                      'WeightedAverageNumberOfSharesOutstandingBasic')
          AND value IS NOT NULL
          AND value > 0
        ORDER BY COALESCE(end_date, filed_date) DESC, filed_date DESC
        LIMIT 1
        """,
        (cik,),
    ).fetchone()
    if not row:
        return None
    try:
        value = float(row["value"])
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def compute_company_metrics(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    cik: str,
    gics_sector: str | None,
    latest_cpi: float,
    cpi_dates: list[dt.date],
    cpi_values: list[float],
    min_eps_points: int,
    lookback_years: int,
) -> dict[str, Any] | None:
    price_row = latest_price(conn, symbol)
    if not price_row:
        return None
    price_date = parse_date(price_row["price_date"])
    if price_date is None:
        return None
    close_price = float(price_row["close_price"])
    if close_price <= 0:
        return None

    candidates = eps_candidates(conn, cik)
    if not candidates:
        return None

    cutoff_date = price_date - dt.timedelta(days=365 * lookback_years + 3)

    best_tag = ""
    best_real_eps_values: list[float] = []
    best_points = 0

    for candidate_tag, series_rows in candidates:
        real_eps_values: list[float] = []
        for row in series_rows:
            end_date = parse_date(str(row.get("end_date") if isinstance(row, dict) else row["end_date"]))
            if end_date is None:
                continue
            if end_date < cutoff_date or end_date > price_date:
                continue
            try:
                eps_value = float(row.get("value") if isinstance(row, dict) else row["value"])
            except (TypeError, ValueError):
                continue
            cpi_then = cpi_for_date(end_date, cpi_dates, cpi_values)
            if cpi_then is None or cpi_then <= 0:
                continue
            real_eps = eps_value * (latest_cpi / cpi_then)
            if math.isfinite(real_eps):
                real_eps_values.append(real_eps)

        if len(real_eps_values) < min_eps_points:
            continue

        # Pick the densest candidate series within the window.
        if len(real_eps_values) > best_points:
            best_tag = candidate_tag
            best_real_eps_values = real_eps_values
            best_points = len(real_eps_values)

    if best_points < min_eps_points:
        return None

    avg_real_eps = statistics.mean(best_real_eps_values)
    if not math.isfinite(avg_real_eps) or avg_real_eps <= 0:
        return None

    company_cape = close_price / avg_real_eps
    if not math.isfinite(company_cape) or company_cape <= 0:
        return None

    shares = latest_shares_outstanding(conn, cik)
    market_cap = close_price * shares if shares else None

    return {
        "symbol": normalize_symbol(symbol),
        "cik": cik,
        "gics_sector": gics_sector or "",
        "price_date": price_row["price_date"],
        "close_price": close_price,
        "shares_outstanding": shares,
        "market_cap": market_cap,
        "eps_tag": best_tag,
        "eps_points": best_points,
        "avg_real_eps": avg_real_eps,
        "company_cape": company_cape,
    }


def assign_weights(metrics: list[dict[str, Any]], market_cap_min_coverage: float) -> tuple[list[dict[str, Any]], str, float]:
    with_mcap = [m for m in metrics if m["market_cap"] is not None and m["market_cap"] > 0]
    coverage = (len(with_mcap) / len(metrics)) if metrics else 0.0

    if with_mcap and coverage >= market_cap_min_coverage:
        mcap_values = [m["market_cap"] for m in with_mcap]
        imputed = statistics.median(mcap_values)
        method = "market_cap"
        if len(with_mcap) < len(metrics):
            method = "market_cap_imputed"
        bases = []
        for metric in metrics:
            base = metric["market_cap"] if metric["market_cap"] and metric["market_cap"] > 0 else imputed
            bases.append(base)
    else:
        method = "equal_weight"
        bases = [1.0 for _ in metrics]

    total = sum(bases)
    for metric, base in zip(metrics, bases):
        metric["weight"] = (base / total) if total > 0 else 0.0
    return metrics, method, coverage


def persist_run(
    conn: sqlite3.Connection,
    *,
    as_of_constituents_date: str,
    latest_price_date: str,
    symbols_total: int,
    symbols_with_price: int,
    metrics: list[dict[str, Any]],
    min_eps_points: int,
    lookback_years: int,
    weighting_method: str,
    market_cap_coverage: float,
    cc_cape: float,
    avg_company_cape: float,
    shiller_cape: float | None,
    shiller_cape_date: str | None,
    cape_spread: float | None,
    cc_cape_percentile: float | None,
    cc_cape_zscore: float | None,
    cape_spread_percentile: float | None,
    cape_spread_zscore: float | None,
    notes: dict[str, Any],
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO cc_cape_runs
        (run_at, as_of_constituents_date, latest_price_date, symbols_total, symbols_with_price,
         symbols_with_valid_cape, min_eps_points, lookback_years, weighting_method, market_cap_coverage,
         cc_cape, avg_company_cape, shiller_cape, shiller_cape_date, cape_spread,
         cc_cape_percentile, cc_cape_zscore, cape_spread_percentile, cape_spread_zscore,
         notes_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now_utc(),
            as_of_constituents_date,
            latest_price_date,
            symbols_total,
            symbols_with_price,
            len(metrics),
            min_eps_points,
            lookback_years,
            weighting_method,
            market_cap_coverage,
            cc_cape,
            avg_company_cape,
            shiller_cape,
            shiller_cape_date,
            cape_spread,
            cc_cape_percentile,
            cc_cape_zscore,
            cape_spread_percentile,
            cape_spread_zscore,
            json.dumps(notes, sort_keys=True),
        ),
    )
    run_id = cursor.lastrowid

    for metric in metrics:
        conn.execute(
            """
            INSERT INTO cc_cape_constituent_metrics
            (run_id, symbol, cik, gics_sector, price_date, close_price, shares_outstanding, market_cap,
             eps_tag, eps_points, avg_real_eps, company_cape, weight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                metric["symbol"],
                metric["cik"],
                metric["gics_sector"],
                metric["price_date"],
                metric["close_price"],
                metric["shares_outstanding"],
                metric["market_cap"],
                metric["eps_tag"],
                metric["eps_points"],
                metric["avg_real_eps"],
                metric["company_cape"],
                metric["weight"],
            ),
        )

    conn.commit()
    return run_id


def update_run_history_stats(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    cc_cape: float,
    cape_spread: float | None,
) -> dict[str, Any]:
    cc_values = [float(row["cc_cape"]) for row in conn.execute("SELECT cc_cape FROM cc_cape_runs").fetchall()]
    cc_pct = empirical_percentile(cc_values, cc_cape)
    cc_z = zscore(cc_values, cc_cape)

    spread_pct: float | None = None
    spread_z: float | None = None
    spread_values: list[float] = []
    if cape_spread is not None:
        spread_values = [
            float(row["cape_spread"])
            for row in conn.execute("SELECT cape_spread FROM cc_cape_runs WHERE cape_spread IS NOT NULL").fetchall()
        ]
        spread_pct = empirical_percentile(spread_values, cape_spread)
        spread_z = zscore(spread_values, cape_spread)

    conn.execute(
        """
        UPDATE cc_cape_runs
        SET cc_cape_percentile = ?,
            cc_cape_zscore = ?,
            cape_spread_percentile = ?,
            cape_spread_zscore = ?
        WHERE run_id = ?
        """,
        (cc_pct, cc_z, spread_pct, spread_z, run_id),
    )
    conn.commit()

    return {
        "cc_cape_percentile": cc_pct,
        "cc_cape_zscore": cc_z,
        "cape_spread_percentile": spread_pct,
        "cape_spread_zscore": spread_z,
        "history_cc_cape_count": len(cc_values),
        "history_spread_count": len(spread_values),
    }


def write_markdown_summary(path: str, summary: dict[str, Any], metrics: list[dict[str, Any]], top_n: int) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    top_by_weight = sorted(metrics, key=lambda m: m["weight"], reverse=True)[:top_n]
    top_by_cape = sorted(metrics, key=lambda m: m["company_cape"], reverse=True)[:top_n]

    lines: list[str] = []
    lines.append("# Free-Data CC CAPE Run")
    lines.append("")
    lines.append(f"Generated: {summary['completed_at']}")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- CC CAPE: **{summary['cc_cape']:.3f}**")
    if summary.get("cc_cape_percentile") is not None:
        lines.append(f"- CC CAPE percentile (run history): **{summary['cc_cape_percentile']:.1f}%**")
    if summary.get("cc_cape_zscore") is not None:
        lines.append(f"- CC CAPE z-score (run history): **{summary['cc_cape_zscore']:.2f}**")
    if summary.get("shiller_cape") is not None:
        shiller_line = f"- Shiller CAPE: **{summary['shiller_cape']:.3f}**"
        if summary.get("shiller_cape_date"):
            shiller_line += f" (as of {summary['shiller_cape_date']})"
        lines.append(shiller_line)
    if summary.get("cape_spread") is not None:
        lines.append(f"- CAPE Spread (CC CAPE - Shiller): **{summary['cape_spread']:.3f}**")
        if summary.get("cape_spread_percentile") is not None:
            lines.append(f"- Spread percentile (run history): **{summary['cape_spread_percentile']:.1f}%**")
        if summary.get("cape_spread_zscore") is not None:
            lines.append(f"- Spread z-score (run history): **{summary['cape_spread_zscore']:.2f}**")
    lines.append(f"- Weighting: `{summary['weighting_method']}`")
    lines.append(f"- Symbols total: {summary['symbols_total']}")
    lines.append(f"- Symbols with latest price: {summary['symbols_with_price']}")
    lines.append(f"- Symbols used in CC CAPE: {summary['symbols_with_valid_cape']}")
    lines.append("")
    lines.append("## Top Weights")
    lines.append("")
    lines.append("| Symbol | Weight | Company CAPE | Avg Real EPS | Price | EPS Points |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for metric in top_by_weight:
        lines.append(
            f"| {metric['symbol']} | {metric['weight']:.4f} | {metric['company_cape']:.2f} | "
            f"{metric['avg_real_eps']:.4f} | {metric['close_price']:.2f} | {metric['eps_points']} |"
        )
    lines.append("")
    lines.append("## Highest Company CAPE")
    lines.append("")
    lines.append("| Symbol | Company CAPE | Weight | EPS Tag |")
    lines.append("| --- | --- | --- | --- |")
    for metric in top_by_cape:
        lines.append(
            f"| {metric['symbol']} | {metric['company_cape']:.2f} | {metric['weight']:.4f} | {metric['eps_tag']} |"
        )

    output.write_text("\n".join(lines), encoding="utf-8")


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
    conn.execute("UPDATE issues SET updated_at = ? WHERE id = ?", (ts, issue["id"]))
    conn.commit()
    return True


def tracker_move_status_if_needed(conn: sqlite3.Connection, issue_key: str, target_status: str, actor: str) -> bool:
    issue = get_issue(conn, issue_key)
    if issue is None or issue["status"] == target_status:
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


def ensure_calc_issue(conn: sqlite3.Connection) -> str | None:
    project = get_project(conn, "CAPE")
    epic = get_epic(conn, "CAPE-EP3")
    if not project or not epic:
        return None
    created = get_or_create_issue(
        conn,
        project=project,
        epic=epic,
        issue_type="story",
        title="Build free-data CC CAPE calculation module",
        description=(
            "Compute free-data proxy CC CAPE from SEC/FRED/Stooq inputs, persist run outputs, "
            "and publish periodic run summaries."
        ),
        priority="p1",
        due_date=epic["target_date"] or "",
        story_points=8,
        sprint="Phase-2",
    )
    conn.commit()
    return created["key"]


def update_tracker(
    *,
    tracker_db_path: str,
    summary: dict[str, Any],
    author: str,
) -> dict[str, Any]:
    tracker_info = {
        "tracker_db": tracker_db_path,
        "comments_added": 0,
        "statuses_changed": 0,
        "tracked_issues": [],
    }
    with tracker_connect(tracker_db_path) as conn:
        init_tracker_db(conn)
        ensure_default_admin(conn)
        calc_issue_key = ensure_calc_issue(conn)
        if calc_issue_key:
            tracker_info["tracked_issues"].append(calc_issue_key)

        body = (
            f"Free-data CC CAPE calculation completed at {summary['completed_at']} UTC.\n\n"
            f"CC CAPE: {summary['cc_cape']:.4f}\n"
            f"Weighting method: {summary['weighting_method']}\n"
            f"Symbols total: {summary['symbols_total']}\n"
            f"Symbols with price: {summary['symbols_with_price']}\n"
            f"Symbols used: {summary['symbols_with_valid_cape']}\n"
            f"Lookback years: {summary['lookback_years']}\n"
            f"Min EPS points: {summary['min_eps_points']}\n"
            f"Market-cap coverage: {summary['market_cap_coverage']:.3f}"
        )
        if summary.get("cc_cape_percentile") is not None:
            body += f"\nCC CAPE percentile: {summary['cc_cape_percentile']:.1f}%"
        if summary.get("cc_cape_zscore") is not None:
            body += f"\nCC CAPE z-score: {summary['cc_cape_zscore']:.2f}"
        if summary.get("shiller_cape") is not None:
            shiller = f"{summary['shiller_cape']:.4f}"
            if summary.get("shiller_cape_date"):
                shiller += f" (as of {summary['shiller_cape_date']})"
            body += f"\nShiller CAPE: {shiller}"
        if summary["cape_spread"] is not None:
            body += f"\nCAPE spread: {summary['cape_spread']:.4f}"
            if summary.get("cape_spread_percentile") is not None:
                body += f"\nSpread percentile: {summary['cape_spread_percentile']:.1f}%"
            if summary.get("cape_spread_zscore") is not None:
                body += f"\nSpread z-score: {summary['cape_spread_zscore']:.2f}"

        for issue_key in [calc_issue_key, "CAPE-8", "CAPE-9"]:
            if not issue_key:
                continue
            if tracker_add_comment(conn, issue_key, author, body):
                tracker_info["comments_added"] += 1
                tracker_info["tracked_issues"].append(issue_key)

        if tracker_move_status_if_needed(conn, "CAPE-8", "in_progress", author):
            tracker_info["statuses_changed"] += 1
            tracker_info["tracked_issues"].append("CAPE-8")
        issue_cape9 = get_issue(conn, "CAPE-9")
        if issue_cape9 and issue_cape9["status"] == "backlog":
            if tracker_move_status_if_needed(conn, "CAPE-9", "todo", author):
                tracker_info["statuses_changed"] += 1
                tracker_info["tracked_issues"].append("CAPE-9")

    tracker_info["tracked_issues"] = sorted(set(tracker_info["tracked_issues"]))
    return tracker_info


def run_calculation(args: argparse.Namespace) -> dict[str, Any]:
    with connect_data_db(args.data_db) as conn:
        init_calc_db(conn)

        as_of_date, constituents = load_latest_constituents(conn)
        if not constituents:
            raise RuntimeError("No constituents found in free-data DB. Run free_data_pipeline first.")

        if args.max_symbols > 0:
            constituents = constituents[: args.max_symbols]

        latest_cpi, latest_cpi_date = load_latest_cpi(conn)
        cpi_dates, cpi_values = load_cpi_series(conn)
        if not cpi_dates:
            raise RuntimeError("No CPI series loaded.")

        metrics: list[dict[str, Any]] = []
        symbols_with_price = 0
        skipped_missing_cik = 0
        skipped_invalid = 0

        for row in constituents:
            symbol = row["symbol"]
            cik = resolve_cik(conn, symbol, row["cik"])
            if cik is None:
                skipped_missing_cik += 1
                continue
            if latest_price(conn, symbol):
                symbols_with_price += 1
            metric = compute_company_metrics(
                conn,
                symbol=symbol,
                cik=cik,
                gics_sector=row["gics_sector"],
                latest_cpi=latest_cpi,
                cpi_dates=cpi_dates,
                cpi_values=cpi_values,
                min_eps_points=args.min_eps_points,
                lookback_years=args.lookback_years,
            )
            if metric is None:
                skipped_invalid += 1
                continue
            metrics.append(metric)

        if not metrics:
            raise RuntimeError("No valid constituent metrics were produced. Increase data coverage and rerun pipeline.")

        metrics, weighting_method, coverage = assign_weights(metrics, args.market_cap_min_coverage)

        cc_cape = sum(metric["weight"] * metric["company_cape"] for metric in metrics)
        avg_company_cape = statistics.mean(metric["company_cape"] for metric in metrics)

        latest_price_date = max(metric["price_date"] for metric in metrics)

        notes = {
            "latest_cpi_date": latest_cpi_date,
            "latest_cpi_value": latest_cpi,
            "skipped_missing_cik": skipped_missing_cik,
            "skipped_invalid_metrics": skipped_invalid,
            "symbols_requested": len(constituents),
        }

        shiller_cape = args.shiller_cape
        shiller_cape_date: str | None = None
        if shiller_cape is not None:
            notes["shiller_cape_source"] = "cli"
        else:
            target_date = parse_date(latest_price_date)
            if target_date is not None:
                lookup = shiller_cape_asof(conn, target_date)
                if lookup is not None:
                    shiller_cape, shiller_cape_date = lookup
                    notes["shiller_cape_source"] = "multpl"
                    notes["shiller_cape_observation_date"] = shiller_cape_date
                    notes["shiller_cape_target_date"] = target_date.isoformat()

        cape_spread = (cc_cape - shiller_cape) if shiller_cape is not None else None

        run_id = persist_run(
            conn,
            as_of_constituents_date=as_of_date,
            latest_price_date=latest_price_date,
            symbols_total=len(constituents),
            symbols_with_price=symbols_with_price,
            metrics=metrics,
            min_eps_points=args.min_eps_points,
            lookback_years=args.lookback_years,
            weighting_method=weighting_method,
            market_cap_coverage=coverage,
            cc_cape=cc_cape,
            avg_company_cape=avg_company_cape,
            shiller_cape=shiller_cape,
            shiller_cape_date=shiller_cape_date,
            cape_spread=cape_spread,
            cc_cape_percentile=None,
            cc_cape_zscore=None,
            cape_spread_percentile=None,
            cape_spread_zscore=None,
            notes=notes,
        )

        history_stats = update_run_history_stats(conn, run_id=run_id, cc_cape=cc_cape, cape_spread=cape_spread)

    summary = {
        "run_id": run_id,
        "completed_at": now_utc(),
        "as_of_constituents_date": as_of_date,
        "latest_price_date": latest_price_date,
        "symbols_total": len(constituents),
        "symbols_with_price": symbols_with_price,
        "symbols_with_valid_cape": len(metrics),
        "min_eps_points": args.min_eps_points,
        "lookback_years": args.lookback_years,
        "weighting_method": weighting_method,
        "market_cap_coverage": coverage,
        "cc_cape": cc_cape,
        "avg_company_cape": avg_company_cape,
        "shiller_cape": shiller_cape,
        "shiller_cape_date": shiller_cape_date,
        "cape_spread": cape_spread,
        "cc_cape_percentile": history_stats["cc_cape_percentile"],
        "cc_cape_zscore": history_stats["cc_cape_zscore"],
        "cape_spread_percentile": history_stats["cape_spread_percentile"],
        "cape_spread_zscore": history_stats["cape_spread_zscore"],
        "notes": notes,
    }

    write_markdown_summary(args.write_markdown, summary, metrics, args.print_top)

    if args.update_tracker:
        summary["tracker"] = update_tracker(
            tracker_db_path=args.tracker_db,
            summary=summary,
            author=args.tracker_author,
        )

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute free-data proxy CC CAPE.")
    parser.add_argument("--data-db", default="data/free_data.db", help="Path to free-data SQLite DB.")
    parser.add_argument("--tracker-db", default="data/internal_jira.db", help="Path to tracker SQLite DB.")
    parser.add_argument("--max-symbols", type=int, default=0, help="Optional cap on symbols processed (0 = all).")
    parser.add_argument("--min-eps-points", type=int, default=8, help="Minimum EPS observations required per company.")
    parser.add_argument("--lookback-years", type=int, default=10, help="Lookback window in years.")
    parser.add_argument(
        "--market-cap-min-coverage",
        type=float,
        default=0.8,
        help="Minimum market-cap coverage to use market-cap weighting; otherwise equal-weight fallback.",
    )
    parser.add_argument(
        "--shiller-cape",
        type=float,
        default=None,
        help="Optional benchmark Shiller CAPE. If omitted, uses latest ingested Shiller CAPE on/before latest price date.",
    )
    parser.add_argument(
        "--update-tracker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, post run summary into tracker issues.",
    )
    parser.add_argument("--tracker-author", default="calc-bot", help="Author for tracker comments/events.")
    parser.add_argument("--write-markdown", default="docs/CC_CAPE_FREE_RUN.md", help="Markdown summary output path.")
    parser.add_argument("--print-top", type=int, default=10, help="Top N constituents shown in markdown summary.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run_calculation(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
