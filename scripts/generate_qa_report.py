#!/usr/bin/env python3
"""Generate a data-quality (QA) markdown report from SQLite state.

This focuses on *what's missing* and *how far back* the CC CAPE proxy can
reasonably be computed given current data coverage.

Outputs:
- Markdown file (default: docs/QA_REPORT.md)
- Optional tracker comment to CAPE-6 (data quality checks)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from internal_jira import (  # noqa: E402
    connect as tracker_connect,
    ensure_default_admin,
    get_issue,
    init_db as init_tracker_db,
    now_utc,
)


def ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def connect_sqlite(path: str) -> sqlite3.Connection:
    ensure_parent_dir(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def parse_utc_ts(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def age_days_from_date(value: str | None, now_date: dt.date) -> int | None:
    d = parse_date(value)
    if not d:
        return None
    return (now_date - d).days


def age_days_from_ts(value: str | None, now_dt: dt.datetime) -> int | None:
    t = parse_utc_ts(value)
    if not t:
        return None
    return int((now_dt - t).total_seconds() // 86400)


def fmt_ratio(n: int, d: int) -> str:
    if d <= 0:
        return f"{n}/{d}"
    return f"{n}/{d} ({(n/d):.1%})"


def fmt_int(x: int | None) -> str:
    return "-" if x is None else str(int(x))


def fmt_date(x: str | None) -> str:
    return x or "-"


def load_latest_constituents(conn: sqlite3.Connection) -> tuple[str | None, list[sqlite3.Row]]:
    if not table_exists(conn, "sp500_constituents"):
        return None, []
    row = conn.execute("SELECT MAX(as_of_date) AS as_of_date FROM sp500_constituents").fetchone()
    as_of = row["as_of_date"] if row else None
    if not as_of:
        return None, []
    rows = conn.execute(
        """
        SELECT symbol, security, gics_sector, cik
        FROM sp500_constituents
        WHERE as_of_date = ?
        ORDER BY symbol
        """,
        (as_of,),
    ).fetchall()
    return as_of, rows


def load_latest_calc(conn: sqlite3.Connection) -> dict[str, Any] | None:
    if not table_exists(conn, "cc_cape_runs"):
        return None
    row = conn.execute("SELECT * FROM cc_cape_runs ORDER BY run_id DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def compute_price_coverage(conn: sqlite3.Connection, constituents: list[sqlite3.Row]) -> dict[str, Any]:
    symbols = [r["symbol"] for r in constituents]
    total = len(symbols)
    latest_prices = conn.execute(
        """
        SELECT symbol, MAX(price_date) AS latest_price_date
        FROM daily_prices
        WHERE source = 'stooq'
        GROUP BY symbol
        """
    ).fetchall() if table_exists(conn, "daily_prices") else []
    price_by_symbol = {r["symbol"]: r["latest_price_date"] for r in latest_prices if r["symbol"]}

    now_date = dt.datetime.now(dt.timezone.utc).date()
    with_price = 0
    stale_7 = 0
    stale_30 = 0
    missing: list[str] = []
    for sym in symbols:
        latest = price_by_symbol.get(sym)
        if not latest:
            missing.append(sym)
            continue
        with_price += 1
        age = age_days_from_date(latest, now_date)
        if age is not None and age > 7:
            stale_7 += 1
        if age is not None and age > 30:
            stale_30 += 1

    max_price_date = None
    if price_by_symbol:
        max_price_date = max((d for d in price_by_symbol.values() if d), default=None)

    return {
        "total": total,
        "with_price": with_price,
        "missing_price": total - with_price,
        "max_price_date": max_price_date,
        "stale_over_7d": stale_7,
        "stale_over_30d": stale_30,
        "missing_examples": missing[:25],
    }


def compute_facts_coverage(conn: sqlite3.Connection, constituents: list[sqlite3.Row]) -> dict[str, Any]:
    now_dt = dt.datetime.now(dt.timezone.utc)

    # Resolve CIK as: override > wiki > blank.
    overrides: dict[str, str] = {}
    if table_exists(conn, "symbol_overrides"):
        rows = conn.execute("SELECT symbol, cik FROM symbol_overrides WHERE cik IS NOT NULL AND cik != ''").fetchall()
        overrides = {r["symbol"]: str(r["cik"]).strip() for r in rows if r and r["symbol"] and r["cik"]}

    symbols_total = len(constituents)
    resolved_cik = 0
    missing_cik_symbols: list[str] = []
    ciks: set[str] = set()
    for r in constituents:
        sym = r["symbol"]
        cik = (overrides.get(sym) or r["cik"] or "").strip()
        if not cik:
            missing_cik_symbols.append(sym)
            continue
        resolved_cik += 1
        ciks.add(cik.lstrip("0") or "0")

    meta: dict[str, str] = {}
    if table_exists(conn, "company_facts_meta"):
        rows = conn.execute("SELECT cik, fetched_at FROM company_facts_meta").fetchall()
        meta = {str(r["cik"]).strip().lstrip("0") or "0": (r["fetched_at"] or "") for r in rows if r and r["cik"]}

    with_facts = 0
    stale_90 = 0
    missing_facts_cik: list[str] = []
    for cik in sorted(ciks):
        fetched_at = meta.get(cik)
        if not fetched_at:
            missing_facts_cik.append(cik)
            continue
        with_facts += 1
        age = age_days_from_ts(fetched_at, now_dt)
        if age is not None and age > 90:
            stale_90 += 1

    latest_fetched_at = None
    if meta:
        latest_fetched_at = max((ts for ts in meta.values() if ts), default=None)

    return {
        "symbols_total": symbols_total,
        "resolved_cik": resolved_cik,
        "missing_cik": symbols_total - resolved_cik,
        "missing_cik_examples": missing_cik_symbols[:25],
        "ciks_total": len(ciks),
        "ciks_with_facts": with_facts,
        "missing_facts": len(ciks) - with_facts,
        "missing_facts_examples": missing_facts_cik[:25],
        "latest_fetched_at": latest_fetched_at,
        "facts_stale_over_90d": stale_90,
    }


def compute_calc_exclusions(conn: sqlite3.Connection, run_id: int) -> dict[str, Any]:
    if not table_exists(conn, "cc_cape_constituent_exclusions"):
        return {"present": False}
    rows = conn.execute(
        """
        SELECT reason, COUNT(*) AS n
        FROM cc_cape_constituent_exclusions
        WHERE run_id = ?
        GROUP BY reason
        ORDER BY n DESC, reason
        """,
        (run_id,),
    ).fetchall()
    by_reason = {r["reason"]: int(r["n"] or 0) for r in rows if r and r["reason"]}
    examples = conn.execute(
        """
        SELECT symbol, reason
        FROM cc_cape_constituent_exclusions
        WHERE run_id = ?
        ORDER BY reason, symbol
        LIMIT 50
        """,
        (run_id,),
    ).fetchall()
    return {
        "present": True,
        "by_reason": by_reason,
        "examples": [(r["symbol"], r["reason"]) for r in examples if r and r["symbol"] and r["reason"]],
    }


def compute_series_earliest_usable(
    conn: sqlite3.Connection,
    *,
    as_of_constituents_date: str,
    coverage_threshold: float,
) -> dict[str, Any]:
    if not table_exists(conn, "cc_cape_series_monthly"):
        return {"present": False}
    rows = conn.execute(
        """
        SELECT observation_date,
               symbols_total,
               symbols_with_price,
               symbols_with_valid_cape
        FROM cc_cape_series_monthly
        WHERE as_of_constituents_date = ?
        ORDER BY observation_date
        """,
        (as_of_constituents_date,),
    ).fetchall()
    if not rows:
        return {"present": False}

    min_date = rows[0]["observation_date"]
    max_date = rows[-1]["observation_date"]
    count = len(rows)

    earliest_meeting = None
    for r in rows:
        total = int(r["symbols_total"] or 0)
        valid = int(r["symbols_with_valid_cape"] or 0)
        if total <= 0:
            continue
        if (valid / total) >= coverage_threshold:
            earliest_meeting = r["observation_date"]
            break

    return {
        "present": True,
        "count": count,
        "min_observation_date": min_date,
        "max_observation_date": max_date,
        "coverage_threshold": coverage_threshold,
        "earliest_meeting_threshold": earliest_meeting,
    }


def render_markdown(
    *,
    generated_at: str,
    as_of_constituents_date: str,
    price_cov: dict[str, Any],
    facts_cov: dict[str, Any],
    latest_calc: dict[str, Any] | None,
    calc_exclusions: dict[str, Any],
    series_stats: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# QA Report (Data Quality)")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append(f"Constituents as-of: {as_of_constituents_date}")
    lines.append("")

    lines.append("## Prices (Stooq)")
    lines.append("")
    lines.append(f"- Coverage: {fmt_ratio(price_cov['with_price'], price_cov['total'])}")
    lines.append(f"- Latest price date (max): {fmt_date(price_cov.get('max_price_date'))}")
    lines.append(f"- Stale >7d: {fmt_int(price_cov.get('stale_over_7d'))}")
    lines.append(f"- Stale >30d: {fmt_int(price_cov.get('stale_over_30d'))}")
    if price_cov.get("missing_examples"):
        lines.append(f"- Missing examples: {', '.join(price_cov['missing_examples'])}")
    lines.append("")

    lines.append("## Fundamentals (SEC Company Facts)")
    lines.append("")
    lines.append(f"- Symbols with resolved CIK: {fmt_ratio(facts_cov['resolved_cik'], facts_cov['symbols_total'])}")
    lines.append(f"- Resolved CIKs with facts fetched: {fmt_ratio(facts_cov['ciks_with_facts'], facts_cov['ciks_total'])}")
    lines.append(f"- Latest facts fetched_at (max): {fmt_date(facts_cov.get('latest_fetched_at'))}")
    lines.append(f"- Stale facts >90d (by fetched_at): {fmt_int(facts_cov.get('facts_stale_over_90d'))}")
    if facts_cov.get("missing_cik_examples"):
        lines.append(f"- Missing CIK examples: {', '.join(facts_cov['missing_cik_examples'])}")
    if facts_cov.get("missing_facts_examples"):
        lines.append(f"- Missing facts CIK examples: {', '.join(facts_cov['missing_facts_examples'])}")
    lines.append("")

    lines.append("## Latest Calculation")
    lines.append("")
    if not latest_calc:
        lines.append("- No CC CAPE runs found yet.")
    else:
        lines.append(f"- Run: {latest_calc.get('run_id')} at {latest_calc.get('run_at')}")
        lines.append(f"- Latest price date: {latest_calc.get('latest_price_date')}")
        lines.append(f"- CC CAPE: {latest_calc.get('cc_cape')}")
        lines.append(f"- Coverage (valid CAPE): {fmt_ratio(int(latest_calc.get('symbols_with_valid_cape') or 0), int(latest_calc.get('symbols_total') or 0))}")
        if calc_exclusions.get("present") and calc_exclusions.get("by_reason"):
            lines.append("")
            lines.append("Exclusions by reason:")
            for reason, n in sorted(calc_exclusions["by_reason"].items(), key=lambda kv: (-kv[1], kv[0]))[:12]:
                lines.append(f"- `{reason}`: {n}")
    lines.append("")

    lines.append("## Monthly Series Usability")
    lines.append("")
    if not series_stats.get("present"):
        lines.append("- Monthly series not available yet.")
    else:
        lines.append(f"- Observations: {series_stats.get('count')}")
        lines.append(f"- Range: {series_stats.get('min_observation_date')} to {series_stats.get('max_observation_date')}")
        lines.append(f"- Coverage threshold (valid/total): {series_stats.get('coverage_threshold'):.0%}")
        lines.append(f"- Earliest month meeting threshold: {fmt_date(series_stats.get('earliest_meeting_threshold'))}")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- This is a free-data proxy; fundamentals history is the binding constraint for deep history.")
    lines.append("- Use `scripts/manage_symbol_overrides.py` to patch edge-case CIK / ticker mappings deterministically.")
    lines.append("")
    return "\n".join(lines)


def write_markdown(path: str, content: str) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")


def update_tracker(tracker_db: str, *, author: str, report_path: str, summary: dict[str, Any]) -> None:
    with tracker_connect(tracker_db) as conn:
        init_tracker_db(conn)
        ensure_default_admin(conn)
        issue = get_issue(conn, "CAPE-6")
        if not issue:
            return
        body = (
            f"QA report updated: `{report_path}` (generated at {now_utc()} UTC).\n\n"
            f"- Prices coverage: {summary.get('prices_coverage', '-')}\n"
            f"- Facts coverage: {summary.get('facts_coverage', '-')}\n"
            f"- Latest calc coverage: {summary.get('calc_coverage', '-')}\n"
            f"- Series earliest usable: {summary.get('series_earliest_usable', '-')}"
        )
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate data-quality QA markdown report.")
    parser.add_argument("--tracker-db", default="data/internal_jira.db", help="Path to tracker SQLite DB.")
    parser.add_argument("--data-db", default="data/free_data.db", help="Path to free-data SQLite DB.")
    parser.add_argument("--out", default="docs/QA_REPORT.md", help="Markdown output path.")
    parser.add_argument(
        "--series-coverage-threshold",
        type=float,
        default=0.7,
        help="Earliest usable series month is the first with (valid/total) >= threshold.",
    )
    parser.add_argument(
        "--update-tracker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, post a short note into CAPE-6.",
    )
    parser.add_argument("--tracker-author", default="qa-bot", help="Author for tracker comments/events.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    with connect_sqlite(args.data_db) as free_conn:
        as_of, constituents = load_latest_constituents(free_conn)
        if not as_of or not constituents:
            raise SystemExit("No constituents found in free-data DB. Run scripts/free_data_pipeline.py first.")

        price_cov = compute_price_coverage(free_conn, constituents)
        facts_cov = compute_facts_coverage(free_conn, constituents)
        latest_calc = load_latest_calc(free_conn)
        calc_exclusions = compute_calc_exclusions(free_conn, int(latest_calc["run_id"])) if latest_calc else {"present": False}

        # Series stats are keyed by as_of_constituents_date from the series table (may lag constituents table).
        series_asof = None
        if table_exists(free_conn, "cc_cape_series_monthly"):
            row = free_conn.execute(
                "SELECT MAX(as_of_constituents_date) AS as_of_constituents_date FROM cc_cape_series_monthly"
            ).fetchone()
            series_asof = row["as_of_constituents_date"] if row else None

        series_stats = (
            compute_series_earliest_usable(
                free_conn,
                as_of_constituents_date=series_asof,
                coverage_threshold=float(args.series_coverage_threshold),
            )
            if series_asof
            else {"present": False}
        )

    content = render_markdown(
        generated_at=now_utc(),
        as_of_constituents_date=as_of,
        price_cov=price_cov,
        facts_cov=facts_cov,
        latest_calc=latest_calc,
        calc_exclusions=calc_exclusions,
        series_stats=series_stats,
    )
    write_markdown(args.out, content)

    summary = {
        "completed_at": now_utc(),
        "out": args.out,
        "as_of_constituents_date": as_of,
        "prices_coverage": fmt_ratio(price_cov["with_price"], price_cov["total"]),
        "facts_coverage": fmt_ratio(facts_cov["ciks_with_facts"], facts_cov["ciks_total"]),
        "calc_coverage": (
            fmt_ratio(int(latest_calc.get("symbols_with_valid_cape") or 0), int(latest_calc.get("symbols_total") or 0))
            if latest_calc
            else "-"
        ),
        "series_earliest_usable": series_stats.get("earliest_meeting_threshold") if series_stats.get("present") else "-",
    }

    if args.update_tracker:
        try:
            update_tracker(args.tracker_db, author=args.tracker_author, report_path=args.out, summary=summary)
        except Exception as exc:  # noqa: BLE001
            summary["tracker_error"] = str(exc)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

