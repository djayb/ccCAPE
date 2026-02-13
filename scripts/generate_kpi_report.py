#!/usr/bin/env python3
"""Generate a KPI baseline markdown report from SQLite state.

Inputs:
- Tracker DB (internal_jira.db): access_audit_logs
- Free-data DB (free_data.db): ingestion_runs, cc_cape_runs, cc_cape_series_monthly
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
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
    get_issue,
    init_db as init_tracker_db,
    now_utc,
)


def ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def connect_sqlite(path: str, *, readonly: bool = False) -> sqlite3.Connection:
    ensure_parent_dir(path)
    if readonly:
        p = Path(path).expanduser().resolve()
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


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


def parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def fmt_percent(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:.1f}%"


def fmt_float(x: float | None, nd: int = 4) -> str:
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def age_days_from_date(latest: str | None, now_date: dt.date) -> int | None:
    d = parse_date(latest)
    if not d:
        return None
    return (now_date - d).days


def age_days_from_ts(latest: str | None, now_dt: dt.datetime) -> int | None:
    t = parse_utc_ts(latest)
    if not t:
        return None
    return int((now_dt - t).total_seconds() // 86400)


def load_latest_pipeline(free_conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = free_conn.execute(
        """
        SELECT run_started_at, run_completed_at, status, details_json
        FROM ingestion_runs
        WHERE step = 'pipeline'
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()
    if not row:
        return None
    details = {}
    try:
        details = json.loads(row["details_json"] or "{}")
    except Exception:
        details = {}
    steps = details.get("steps", {}) if isinstance(details, dict) else {}
    quality = steps.get("quality_checks", {}) if isinstance(steps, dict) else {}
    if not isinstance(quality, dict):
        quality = {}

    now_dt = dt.datetime.now(dt.timezone.utc)
    now_date = now_dt.date()

    def fallback(name: str, query: str, key: str) -> None:
        if quality.get(name) not in (None, "", "-"):
            return
        try:
            row2 = free_conn.execute(query).fetchone()
        except sqlite3.Error:
            return
        if row2 and row2[key]:
            quality[name] = row2[key]

    fallback("latest_price_date", "SELECT MAX(price_date) AS price_date FROM daily_prices WHERE source='stooq'", "price_date")
    fallback("latest_cpi_date", "SELECT MAX(observation_date) AS observation_date FROM cpi_observations", "observation_date")
    fallback(
        "shiller_latest_observation_date",
        "SELECT MAX(observation_date) AS observation_date FROM shiller_cape_observations",
        "observation_date",
    )
    fallback("facts_cik_count", "SELECT COUNT(DISTINCT cik) AS count_cik FROM company_facts_meta", "count_cik")
    fallback(
        "priced_symbol_count",
        "SELECT COUNT(DISTINCT symbol) AS count_symbols FROM daily_prices WHERE source='stooq'",
        "count_symbols",
    )

    if "price_age_days" not in quality:
        quality["price_age_days"] = age_days_from_date(quality.get("latest_price_date"), now_date)
    if "cpi_age_days" not in quality:
        quality["cpi_age_days"] = age_days_from_date(quality.get("latest_cpi_date"), now_date)
    if "shiller_age_days" not in quality:
        quality["shiller_age_days"] = age_days_from_date(quality.get("shiller_latest_observation_date"), now_date)
    if "facts_age_days" not in quality:
        row3 = None
        try:
            row3 = free_conn.execute("SELECT MAX(fetched_at) AS fetched_at FROM company_facts_meta").fetchone()
        except sqlite3.Error:
            row3 = None
        quality["facts_age_days"] = age_days_from_ts(row3["fetched_at"] if row3 else None, now_dt)

    return {
        "run_started_at": row["run_started_at"],
        "run_completed_at": row["run_completed_at"],
        "status": row["status"],
        "quality": quality,
        "steps": steps,
    }


def pipeline_stats(free_conn: sqlite3.Connection, *, since: dt.datetime, limit: int = 500) -> dict[str, Any]:
    rows = free_conn.execute(
        """
        SELECT run_started_at, status
        FROM ingestion_runs
        WHERE step = 'pipeline'
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    total = 0
    by_status: dict[str, int] = {}
    for row in rows:
        started = parse_utc_ts(row["run_started_at"])
        if not started or started < since:
            continue
        total += 1
        st = (row["status"] or "").strip() or "unknown"
        by_status[st] = by_status.get(st, 0) + 1
    success = by_status.get("success", 0)
    success_rate = (100.0 * success / total) if total else None
    return {"total": total, "by_status": by_status, "success_rate": success_rate}


def calc_stats(free_conn: sqlite3.Connection, *, since: dt.datetime) -> dict[str, Any]:
    rows = free_conn.execute(
        """
        SELECT run_id, run_at, cc_cape, cape_spread, symbols_total, symbols_with_valid_cape
        FROM cc_cape_runs
        ORDER BY run_id DESC
        LIMIT 2000
        """
    ).fetchall()
    total = 0
    values: list[float] = []
    spreads: list[float] = []
    last = None
    for row in rows:
        t = parse_utc_ts(row["run_at"])
        if not t or t < since:
            continue
        total += 1
        try:
            values.append(float(row["cc_cape"]))
        except Exception:
            pass
        if row["cape_spread"] is not None:
            try:
                spreads.append(float(row["cape_spread"]))
            except Exception:
                pass
        if last is None:
            last = dict(row)
    return {
        "total": total,
        "cc_cape_min": min(values) if values else None,
        "cc_cape_max": max(values) if values else None,
        "cc_cape_mean": statistics.mean(values) if values else None,
        "spread_min": min(spreads) if spreads else None,
        "spread_max": max(spreads) if spreads else None,
        "spread_mean": statistics.mean(spreads) if spreads else None,
        "latest_run": last,
    }


def series_stats(free_conn: sqlite3.Connection) -> dict[str, Any] | None:
    if not table_exists(free_conn, "cc_cape_series_monthly"):
        return None
    row = free_conn.execute(
        "SELECT MAX(as_of_constituents_date) AS as_of_constituents_date FROM cc_cape_series_monthly"
    ).fetchone()
    as_of = row["as_of_constituents_date"] if row else None
    if not as_of:
        return None
    stats = free_conn.execute(
        """
        SELECT MIN(observation_date) AS min_observation_date,
               MAX(observation_date) AS max_observation_date,
               COUNT(*) AS count
        FROM cc_cape_series_monthly
        WHERE as_of_constituents_date = ?
        """,
        (as_of,),
    ).fetchone()
    latest = free_conn.execute(
        """
        SELECT observation_date, cc_cape, cape_spread
        FROM cc_cape_series_monthly
        WHERE as_of_constituents_date = ?
        ORDER BY observation_date DESC
        LIMIT 1
        """,
        (as_of,),
    ).fetchone()
    return {
        "as_of_constituents_date": as_of,
        "min_observation_date": stats["min_observation_date"] if stats else None,
        "max_observation_date": stats["max_observation_date"] if stats else None,
        "count": int(stats["count"] or 0) if stats else 0,
        "latest_observation_date": latest["observation_date"] if latest else None,
        "latest_cc_cape": float(latest["cc_cape"]) if latest and latest["cc_cape"] is not None else None,
        "latest_cape_spread": float(latest["cape_spread"]) if latest and latest["cape_spread"] is not None else None,
    }


def audit_stats(tracker_conn: sqlite3.Connection, *, since: dt.datetime) -> dict[str, Any] | None:
    if not table_exists(tracker_conn, "access_audit_logs"):
        return None
    rows = tracker_conn.execute(
        """
        SELECT occurred_at, username, path
        FROM access_audit_logs
        ORDER BY id DESC
        LIMIT 100000
        """
    ).fetchall()
    total = 0
    users: set[str] = set()
    by_prefix: dict[str, int] = {}
    for row in rows:
        t = parse_utc_ts(row["occurred_at"])
        if not t or t < since:
            continue
        total += 1
        if row["username"]:
            users.add(row["username"])
        path = row["path"] or ""
        prefix = path
        if path.startswith("/api/"):
            prefix = "/api/*"
        elif path.startswith("/metrics/"):
            prefix = "/metrics/*"
        elif path.startswith("/admin/"):
            prefix = "/admin/*"
        elif path.startswith("/issue/"):
            prefix = "/issue/*"
        by_prefix[prefix] = by_prefix.get(prefix, 0) + 1

    top = sorted(by_prefix.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {"total_requests": total, "unique_users": len(users), "top_prefixes": top}


def render_markdown(
    *,
    generated_at: str,
    pipeline: dict[str, Any] | None,
    pipeline_30d: dict[str, Any],
    calc_30d: dict[str, Any],
    series: dict[str, Any] | None,
    audit_7d: dict[str, Any] | None,
    audit_30d: dict[str, Any] | None,
) -> str:
    lines: list[str] = []
    lines.append("# KPI Baseline (Internal)")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append("")

    lines.append("## Freshness and Coverage")
    lines.append("")
    if not pipeline:
        lines.append("- Latest pipeline: -")
    else:
        qc = pipeline.get("quality", {}) if isinstance(pipeline.get("quality"), dict) else {}
        lines.append(f"- Latest pipeline status: `{pipeline.get('status')}`")
        lines.append(f"- Latest price date: {qc.get('latest_price_date', '-') } (age {qc.get('price_age_days', '-') } days)")
        lines.append(f"- Latest CPI date: {qc.get('latest_cpi_date', '-') } (age {qc.get('cpi_age_days', '-') } days)")
        lines.append(f"- Latest Shiller date: {qc.get('shiller_latest_observation_date', '-') } (age {qc.get('shiller_age_days', '-') } days)")
        lines.append(f"- Facts CIKs: {qc.get('facts_cik_count', '-') }")
        lines.append(f"- Priced symbols: {qc.get('priced_symbol_count', '-') }")
        warnings = qc.get("warnings") or []
        if warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in list(warnings)[:10]:
                lines.append(f"- {w}")

    lines.append("")
    lines.append("## Reliability")
    lines.append("")
    lines.append("### Pipeline (last 30 days)")
    lines.append("")
    lines.append(f"- Runs: {pipeline_30d.get('total', 0)}")
    lines.append(f"- Success rate: {fmt_percent(pipeline_30d.get('success_rate'))}")
    lines.append(f"- By status: {pipeline_30d.get('by_status', {})}")
    lines.append("")

    lines.append("### Calculation (last 30 days)")
    lines.append("")
    lines.append(f"- Runs: {calc_30d.get('total', 0)}")
    lines.append(f"- CC CAPE mean: {fmt_float(calc_30d.get('cc_cape_mean'), 3)} (min {fmt_float(calc_30d.get('cc_cape_min'), 3)} / max {fmt_float(calc_30d.get('cc_cape_max'), 3)})")
    lines.append(f"- Spread mean: {fmt_float(calc_30d.get('spread_mean'), 3)} (min {fmt_float(calc_30d.get('spread_min'), 3)} / max {fmt_float(calc_30d.get('spread_max'), 3)})")
    latest = calc_30d.get("latest_run") or {}
    if latest:
        lines.append(f"- Latest run: {latest.get('run_at', '-') } (run_id {latest.get('run_id', '-') })")
        lines.append(
            f"  - Coverage: {latest.get('symbols_with_valid_cape', '-') } / {latest.get('symbols_total', '-') }"
        )
    lines.append("")

    lines.append("## Monthly Series")
    lines.append("")
    if not series:
        lines.append("- Not generated yet. Run `scripts/backfill_cc_cape_series_free.py`.")
    else:
        lines.append(f"- As-of constituents: {series.get('as_of_constituents_date')}")
        lines.append(f"- Range: {series.get('min_observation_date')} to {series.get('max_observation_date')}")
        lines.append(f"- Observations: {series.get('count')}")
        lines.append(f"- Latest point: {series.get('latest_observation_date')} (CC CAPE {fmt_float(series.get('latest_cc_cape'), 3)}, spread {fmt_float(series.get('latest_cape_spread'), 3)})")
    lines.append("")

    lines.append("## Usage (Access Audit Logs)")
    lines.append("")
    if not audit_7d:
        lines.append("- Access audit logs not available.")
    else:
        lines.append(f"- Last 7 days: {audit_7d.get('total_requests', 0)} requests, {audit_7d.get('unique_users', 0)} unique users")
        if audit_7d.get("top_prefixes"):
            lines.append("")
            lines.append("| Prefix | Requests |")
            lines.append("| --- | --- |")
            for prefix, count in audit_7d["top_prefixes"]:
                lines.append(f"| `{prefix}` | {count} |")
    lines.append("")

    if audit_30d:
        lines.append(f"- Last 30 days: {audit_30d.get('total_requests', 0)} requests, {audit_30d.get('unique_users', 0)} unique users")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- This is a free-data proxy; expect missing/stale data and licensing constraints.")
    lines.append("- Price refresh can be rate-limited by Stooq.")
    lines.append("")
    return "\n".join(lines)


def write_markdown(path: str, content: str) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")


def update_tracker(tracker_db: str, *, author: str, report_path: str) -> None:
    with tracker_connect(tracker_db) as conn:
        init_tracker_db(conn)
        ensure_default_admin(conn)
        issue = get_issue(conn, "CAPE-21")
        if not issue:
            return
        body = f"KPI baseline report updated: `{report_path}` (generated at {now_utc()} UTC)."
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
    parser = argparse.ArgumentParser(description="Generate KPI baseline markdown report.")
    parser.add_argument("--tracker-db", default="data/internal_jira.db", help="Path to tracker SQLite DB.")
    parser.add_argument("--data-db", default="data/free_data.db", help="Path to free-data SQLite DB.")
    parser.add_argument("--days", type=int, default=30, help="Lookback days for reliability stats.")
    parser.add_argument("--audit-days-short", type=int, default=7, help="Short window for access audit stats.")
    parser.add_argument("--out", default="docs/KPI_BASELINE.md", help="Markdown output path.")
    parser.add_argument(
        "--update-tracker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, post a short note into CAPE-21.",
    )
    parser.add_argument("--tracker-author", default="kpi-bot", help="Author for tracker comments/events.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    now_dt = dt.datetime.now(dt.timezone.utc)
    since = now_dt - dt.timedelta(days=max(1, int(args.days)))
    audit_short_since = now_dt - dt.timedelta(days=max(1, int(args.audit_days_short)))

    with connect_sqlite(args.data_db, readonly=True) as free_conn:
        pipeline = load_latest_pipeline(free_conn)
        pipeline_30d = pipeline_stats(free_conn, since=since)
        calc_30d = calc_stats(free_conn, since=since)
        series = series_stats(free_conn)

    with connect_sqlite(args.tracker_db, readonly=True) as tracker_conn:
        # Ensure schema exists for older DBs; in readonly mode this is best-effort.
        audit_7d = audit_stats(tracker_conn, since=audit_short_since)
        audit_30d = audit_stats(tracker_conn, since=since)

    content = render_markdown(
        generated_at=now_utc(),
        pipeline=pipeline,
        pipeline_30d=pipeline_30d,
        calc_30d=calc_30d,
        series=series,
        audit_7d=audit_7d,
        audit_30d=audit_30d,
    )
    write_markdown(args.out, content)

    summary = {
        "completed_at": now_utc(),
        "out": args.out,
        "pipeline_runs": pipeline_30d["total"],
        "pipeline_success_rate": pipeline_30d["success_rate"],
        "calc_runs": calc_30d["total"],
        "series_present": bool(series),
        "audit_7d_requests": (audit_7d or {}).get("total_requests"),
        "audit_7d_unique_users": (audit_7d or {}).get("unique_users"),
    }

    if args.update_tracker:
        try:
            update_tracker(args.tracker_db, author=args.tracker_author, report_path=args.out)
        except Exception as exc:  # noqa: BLE001
            summary["tracker_error"] = str(exc)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
