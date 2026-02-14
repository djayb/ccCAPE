#!/usr/bin/env python3
"""Minimal internal web UI for the CC CAPE tracker."""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import os
from pathlib import Path
import sqlite3
import time

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from internal_jira import (
    PRIORITY_LABELS,
    ROLES,
    STATUSES,
    STATUS_LABELS,
    connect,
    create_user,
    get_issue,
    get_project,
    get_user_by_username,
    init_db,
    ensure_default_admin,
    now_utc,
    normalize_choice,
    priority_rank_sql,
    status_rank_sql,
    verify_password,
)

DB_PATH = os.environ.get("TRACKER_DB", "data/internal_jira.db")
FREE_DATA_DB_PATH = os.environ.get("FREE_DATA_DB", "data/free_data.db")
SESSION_SECRET = os.environ.get("TRACKER_SESSION_SECRET", "replace-this-secret")
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"

app = FastAPI(
    title="CC CAPE Internal Tracker",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, same_site="lax")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _serialize_cc_cape_run(row: sqlite3.Row) -> dict:
    data = dict(row)
    return {
        "run_id": data.get("run_id"),
        "run_at": data.get("run_at"),
        "as_of_constituents_date": data.get("as_of_constituents_date"),
        "latest_price_date": data.get("latest_price_date"),
        "cc_cape": data.get("cc_cape"),
        "avg_company_cape": data.get("avg_company_cape"),
        "cc_cape_percentile": data.get("cc_cape_percentile"),
        "cc_cape_zscore": data.get("cc_cape_zscore"),
        "symbols_total": data.get("symbols_total"),
        "symbols_with_price": data.get("symbols_with_price"),
        "symbols_with_valid_cape": data.get("symbols_with_valid_cape"),
        "weighting_method": data.get("weighting_method"),
        "market_cap_coverage": data.get("market_cap_coverage"),
        "lookback_years": data.get("lookback_years"),
        "min_eps_points": data.get("min_eps_points"),
        "shiller_cape": data.get("shiller_cape"),
        "shiller_cape_date": data.get("shiller_cape_date"),
        "cape_spread": data.get("cape_spread"),
        "cape_spread_percentile": data.get("cape_spread_percentile"),
        "cape_spread_zscore": data.get("cape_spread_zscore"),
    }


def _open_conn():
    conn = connect(DB_PATH)
    init_db(conn)
    ensure_default_admin(conn)
    return conn


def _open_free_data_conn() -> sqlite3.Connection | None:
    path = Path(FREE_DATA_DB_PATH)
    if not path.exists():
        return None
    # Read-only by default to avoid accidental writes from the web app.
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 2000;")
    return conn


def _current_user(request: Request, conn):
    username = request.session.get("username")
    if not username:
        return None
    user = get_user_by_username(conn, username)
    if not user:
        request.session.clear()
        return None
    if not user["active"]:
        request.session.clear()
        return None
    return user


def _require_login(request: Request, conn):
    user = _current_user(request, conn)
    if not user:
        return None, RedirectResponse("/login", status_code=303)
    return user, None


def _require_api_login(request: Request, conn):
    user = _current_user(request, conn)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def _can_edit(user) -> bool:
    return user["role"] in {"admin", "editor"}


@app.on_event("startup")
def startup() -> None:
    with _open_conn() as conn:
        init_db(conn)
        ensure_default_admin(conn)


@app.middleware("http")
async def access_audit_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = int((time.time() - start) * 1000)

    try:
        path = request.url.path
        if path.startswith("/static"):
            return response

        username = request.session.get("username")
        role = request.session.get("role")
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent", "")

        with connect(DB_PATH) as conn:
            conn.execute("PRAGMA busy_timeout = 2000;")
            conn.execute(
                """
                INSERT INTO access_audit_logs
                (occurred_at, username, role, method, path, status_code, client_ip, user_agent, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_utc(),
                    username,
                    role,
                    request.method,
                    path,
                    int(getattr(response, "status_code", 0) or 0),
                    client_ip,
                    user_agent[:500],
                    duration_ms,
                ),
            )
            conn.commit()
    except Exception:
        # Best-effort audit; never block user flows.
        pass

    return response


@app.get("/", response_class=HTMLResponse)
def root() -> RedirectResponse:
    return RedirectResponse("/board", status_code=303)


@app.get("/metrics", response_class=HTMLResponse)
def metrics_redirect() -> RedirectResponse:
    return RedirectResponse("/metrics/cc-cape", status_code=303)


def _resolve_docs_path(relpath: str) -> Path:
    """Resolve a user-provided docs path safely inside DOCS_DIR."""
    if not relpath:
        raise HTTPException(status_code=400, detail="Missing doc path.")
    if relpath.startswith(("/", "\\")):
        raise HTTPException(status_code=400, detail="Invalid doc path.")
    if ".." in relpath.replace("\\", "/").split("/"):
        raise HTTPException(status_code=400, detail="Invalid doc path.")

    root = DOCS_DIR.resolve()
    candidate = (DOCS_DIR / relpath).resolve()
    if candidate == root or root not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid doc path.")
    if candidate.suffix.lower() not in {".md", ".txt"}:
        raise HTTPException(status_code=400, detail="Unsupported doc type.")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Doc not found.")
    return candidate


@app.get("/docs", response_class=HTMLResponse)
def docs_index(request: Request):
    with _open_conn() as tracker_conn:
        user, redirect = _require_login(request, tracker_conn)
        if redirect:
            return redirect

    docs: list[dict[str, str]] = []
    if DOCS_DIR.exists():
        for path in sorted(DOCS_DIR.glob("*.md")):
            try:
                stat = path.stat()
                mtime = dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                docs.append(
                    {
                        "name": path.name,
                        "relpath": path.name,
                        "size_kb": f"{int(stat.st_size / 1024)}",
                        "mtime": mtime,
                    }
                )
            except OSError:
                continue

    return templates.TemplateResponse(
        "docs_index.html",
        {
            "request": request,
            "user": user,
            "docs": docs,
            "message": request.query_params.get("msg", ""),
            "error": request.query_params.get("err", ""),
        },
    )


@app.get("/docs/view/{doc_path:path}", response_class=HTMLResponse)
def docs_view(request: Request, doc_path: str):
    with _open_conn() as tracker_conn:
        user, redirect = _require_login(request, tracker_conn)
        if redirect:
            return redirect

    path = _resolve_docs_path(doc_path)
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        raise HTTPException(status_code=500, detail="Failed to read doc.")

    return templates.TemplateResponse(
        "docs_view.html",
        {
            "request": request,
            "user": user,
            "doc_name": path.name,
            "relpath": doc_path,
            "content": content,
        },
    )


@app.get("/docs/raw/{doc_path:path}")
def docs_raw(request: Request, doc_path: str):
    with _open_conn() as tracker_conn:
        user, redirect = _require_login(request, tracker_conn)
        if redirect:
            return redirect

    path = _resolve_docs_path(doc_path)
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        raise HTTPException(status_code=500, detail="Failed to read doc.")

    return Response(content, media_type="text/plain; charset=utf-8")


@app.get("/metrics/cc-cape", response_class=HTMLResponse)
def metrics_cc_cape(request: Request):
    with _open_conn() as tracker_conn:
        user, redirect = _require_login(request, tracker_conn)
        if redirect:
            return redirect

    free_conn = _open_free_data_conn()
    if free_conn is None:
        return templates.TemplateResponse(
            "metrics_cc_cape.html",
            {
                "request": request,
                "user": user,
                "latest_run": None,
                "runs": [],
                "latest_ingestion": None,
                "message": request.query_params.get("msg", ""),
                "error": "Free-data DB not found. Run scripts/free_data_pipeline.py first.",
            },
        )

    with free_conn:
        latest_run_row = free_conn.execute(
            """
            SELECT *
            FROM cc_cape_runs
            ORDER BY run_id DESC
            LIMIT 1
            """
        ).fetchone()
        runs_rows = free_conn.execute(
            """
            SELECT *
            FROM cc_cape_runs
            ORDER BY run_id DESC
            LIMIT 30
            """
        ).fetchall()
        ingestion = free_conn.execute(
            """
            SELECT run_started_at, run_completed_at, status, details_json
            FROM ingestion_runs
            WHERE step = 'pipeline'
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

    latest_run = dict(latest_run_row) if latest_run_row else None
    runs = [dict(row) for row in runs_rows]

    latest_run_notes: dict[str, object] = {}
    if latest_run and latest_run.get("notes_json"):
        try:
            latest_run_notes = json.loads(latest_run.get("notes_json") or "{}")
        except Exception:
            latest_run_notes = {}

    latest_ingestion = None
    if ingestion:
        try:
            latest_ingestion = {
                "run_started_at": ingestion["run_started_at"],
                "run_completed_at": ingestion["run_completed_at"],
                "status": ingestion["status"],
                "details": json.loads(ingestion["details_json"] or "{}"),
            }
        except Exception:
            latest_ingestion = {
                "run_started_at": ingestion["run_started_at"],
                "run_completed_at": ingestion["run_completed_at"],
                "status": ingestion["status"],
                "details": {},
            }

    return templates.TemplateResponse(
        "metrics_cc_cape.html",
        {
            "request": request,
            "user": user,
            "latest_run": latest_run,
            "latest_run_notes": latest_run_notes,
            "runs": runs,
            "latest_ingestion": latest_ingestion,
            "message": request.query_params.get("msg", ""),
            "error": request.query_params.get("error", ""),
        },
    )


@app.get("/metrics/health", response_class=HTMLResponse)
def metrics_health(request: Request):
    with _open_conn() as tracker_conn:
        user, redirect = _require_login(request, tracker_conn)
        if redirect:
            return redirect

    free_conn = _open_free_data_conn()
    if free_conn is None:
        return templates.TemplateResponse(
            "metrics_health.html",
            {
                "request": request,
                "user": user,
                "pipeline": None,
                "calc": None,
                "series": None,
                "message": request.query_params.get("msg", ""),
                "error": "Free-data DB not found. Run scripts/free_data_pipeline.py first.",
            },
        )

    pipeline_obj = None
    calc_obj = None
    series_obj = None

    with free_conn:
        pipeline = free_conn.execute(
            """
            SELECT run_started_at, run_completed_at, status, details_json
            FROM ingestion_runs
            WHERE step = 'pipeline'
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        if pipeline:
            try:
                details = json.loads(pipeline["details_json"] or "{}")
            except Exception:
                details = {}
            steps = details.get("steps", {}) if isinstance(details, dict) else {}
            pipeline_obj = {
                "run_started_at": pipeline["run_started_at"],
                "run_completed_at": pipeline["run_completed_at"],
                "status": pipeline["status"],
                "quality": steps.get("quality_checks", {}) if isinstance(steps, dict) else {},
            }

        calc = free_conn.execute(
            """
            SELECT *
            FROM cc_cape_runs
            ORDER BY run_id DESC
            LIMIT 1
            """
        ).fetchone()
        calc_obj = dict(calc) if calc else None

        if _table_exists(free_conn, "cc_cape_series_monthly"):
            row = free_conn.execute(
                "SELECT MAX(as_of_constituents_date) AS as_of_constituents_date FROM cc_cape_series_monthly"
            ).fetchone()
            as_of = row["as_of_constituents_date"] if row else None
            if as_of:
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
                    SELECT cc_cape
                    FROM cc_cape_series_monthly
                    WHERE as_of_constituents_date = ?
                    ORDER BY observation_date DESC
                    LIMIT 1
                    """,
                    (as_of,),
                ).fetchone()
                series_obj = {
                    "as_of_constituents_date": as_of,
                    "min_observation_date": stats["min_observation_date"] if stats else None,
                    "max_observation_date": stats["max_observation_date"] if stats else None,
                    "count": stats["count"] if stats else 0,
                    "latest_cc_cape": float(latest["cc_cape"]) if latest and latest["cc_cape"] is not None else None,
                }

    kpi_report = None
    kpi_path = DOCS_DIR / "KPI_BASELINE.md"
    if kpi_path.exists():
        try:
            stat = kpi_path.stat()
            kpi_report = {
                "relpath": kpi_path.name,
                "size_kb": f"{int(stat.st_size / 1024)}",
                "mtime": dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
        except OSError:
            kpi_report = None

    return templates.TemplateResponse(
        "metrics_health.html",
        {
            "request": request,
            "user": user,
            "pipeline": pipeline_obj,
            "calc": calc_obj,
            "series": series_obj,
            "kpi_report": kpi_report,
            "message": request.query_params.get("msg", ""),
            "error": request.query_params.get("error", ""),
        },
    )


@app.get("/metrics/gaps", response_class=HTMLResponse)
def metrics_gaps(request: Request):
    with _open_conn() as tracker_conn:
        user, redirect = _require_login(request, tracker_conn)
        if redirect:
            return redirect

    free_conn = _open_free_data_conn()
    if free_conn is None:
        return templates.TemplateResponse(
            "metrics_gaps.html",
            {
                "request": request,
                "user": user,
                "error": "Free-data DB not found. Run scripts/free_data_pipeline.py first.",
            },
        )

    def age_days(date_value: str | None) -> int | None:
        if not date_value:
            return None
        try:
            parsed = dt.date.fromisoformat(date_value[:10])
        except ValueError:
            return None
        return (dt.datetime.now(dt.timezone.utc).date() - parsed).days

    def age_days_ts(ts_value: str | None) -> int | None:
        if not ts_value:
            return None
        raw = ts_value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = dt.datetime.fromisoformat(raw)
        except ValueError:
            return None
        return (dt.datetime.now(dt.timezone.utc) - parsed).days

    with free_conn:
        as_of_row = free_conn.execute("SELECT MAX(as_of_date) AS as_of_date FROM sp500_constituents").fetchone()
        as_of_date = as_of_row["as_of_date"] if as_of_row else None
        if not as_of_date:
            return templates.TemplateResponse(
                "metrics_gaps.html",
                {
                    "request": request,
                    "user": user,
                    "error": "No constituents found. Run scripts/free_data_pipeline.py first.",
                },
            )

        constituents = free_conn.execute(
            """
            SELECT symbol, security, gics_sector, cik
            FROM sp500_constituents
            WHERE as_of_date = ?
            ORDER BY symbol
            """,
            (as_of_date,),
        ).fetchall()

        prices = free_conn.execute(
            """
            SELECT symbol, MAX(price_date) AS latest_price_date
            FROM daily_prices
            WHERE source = 'stooq'
            GROUP BY symbol
            """
        ).fetchall()
        latest_price_by_symbol = {row["symbol"]: row["latest_price_date"] for row in prices if row["symbol"]}

        facts = free_conn.execute("SELECT cik, fetched_at FROM company_facts_meta").fetchall()
        facts_by_cik = {row["cik"]: row["fetched_at"] for row in facts if row["cik"]}

        latest_run_row = None
        if _table_exists(free_conn, "cc_cape_runs"):
            latest_run_row = free_conn.execute("SELECT * FROM cc_cape_runs ORDER BY run_id DESC LIMIT 1").fetchone()

        included_symbols: set[str] = set()
        if latest_run_row and _table_exists(free_conn, "cc_cape_constituent_metrics"):
            rows = free_conn.execute(
                "SELECT symbol FROM cc_cape_constituent_metrics WHERE run_id = ?",
                (latest_run_row["run_id"],),
            ).fetchall()
            included_symbols = {r["symbol"] for r in rows if r and r["symbol"]}

        exclusion_reason_by_symbol: dict[str, str] = {}
        if latest_run_row and _table_exists(free_conn, "cc_cape_constituent_exclusions"):
            rows = free_conn.execute(
                "SELECT symbol, reason FROM cc_cape_constituent_exclusions WHERE run_id = ?",
                (latest_run_row["run_id"],),
            ).fetchall()
            exclusion_reason_by_symbol = {r["symbol"]: r["reason"] for r in rows if r and r["symbol"] and r["reason"]}

    latest_run = dict(latest_run_row) if latest_run_row else None
    exclusion_reason_by_symbol = exclusion_reason_by_symbol if latest_run_row else {}

    rows_out: list[dict[str, Any]] = []
    with_price = 0
    with_facts = 0
    included_in_latest = 0
    bucket_counts: dict[str, int] = {}

    for c in constituents:
        symbol = c["symbol"]
        cik = (c["cik"] or "").strip()
        latest_price_date = latest_price_by_symbol.get(symbol)
        facts_fetched_at = facts_by_cik.get(cik) if cik else None
        in_latest_run = bool(latest_run and symbol in included_symbols)

        if latest_price_date:
            with_price += 1
        if facts_fetched_at:
            with_facts += 1
        if in_latest_run:
            included_in_latest += 1

        calc_reason = exclusion_reason_by_symbol.get(symbol) if latest_run else None

        if in_latest_run:
            bucket = "included"
        elif calc_reason:
            bucket = calc_reason
        elif not latest_price_date:
            bucket = "missing_price"
        elif not cik:
            bucket = "missing_cik"
        elif not facts_fetched_at:
            bucket = "missing_facts"
        elif latest_run:
            bucket = "excluded_from_latest_calc"
        else:
            bucket = "not_calculated"

        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        rows_out.append(
            {
                "symbol": symbol,
                "security": c["security"],
                "gics_sector": c["gics_sector"],
                "cik": cik,
                "latest_price_date": latest_price_date,
                "price_age_days": age_days(latest_price_date),
                "facts_fetched_at": facts_fetched_at,
                "facts_age_days": age_days_ts(facts_fetched_at),
                "in_latest_run": in_latest_run,
                "calc_reason": calc_reason,
                "bucket": bucket,
            }
        )

    breakdown = [{"bucket": k, "count": bucket_counts[k]} for k in sorted(bucket_counts.keys())]
    summary = {
        "total": len(constituents),
        "with_price": with_price,
        "with_facts": with_facts,
        "included_in_latest_run": included_in_latest,
    }

    return templates.TemplateResponse(
        "metrics_gaps.html",
        {
            "request": request,
            "user": user,
            "error": "",
            "as_of_constituents_date": as_of_date,
            "latest_run": latest_run,
            "summary": summary,
            "breakdown": breakdown,
            "rows": rows_out,
        },
    )


def _get_cc_cape_run(free_conn: sqlite3.Connection, run_id: int | None) -> sqlite3.Row | None:
    if run_id is None:
        return free_conn.execute(
            """
            SELECT *
            FROM cc_cape_runs
            ORDER BY run_id DESC
            LIMIT 1
            """
        ).fetchone()
    return free_conn.execute(
        """
        SELECT *
        FROM cc_cape_runs
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()


def _csv_response(filename: str, fieldnames: list[str], rows: list[dict]) -> Response:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


@app.get("/metrics/cc-cape/contributors", response_class=HTMLResponse)
def metrics_cc_cape_contributors(
    request: Request,
    run_id: int | None = None,
    top_n: int = 25,
    sort: str = "weight",
    q: str = "",
    limit: int = 503,
):
    if top_n < 5 or top_n > 200:
        top_n = 25
    if limit < 5 or limit > 5000:
        limit = 503

    sort = (sort or "weight").strip().lower()
    order_by = "m.weight DESC, m.symbol"
    if sort == "contribution":
        order_by = "(m.weight * m.company_cape) DESC, m.symbol"
    elif sort == "cape":
        order_by = "m.company_cape DESC, m.symbol"
    elif sort == "symbol":
        order_by = "m.symbol"
    elif sort == "weight":
        order_by = "m.weight DESC, m.symbol"
    else:
        sort = "weight"
        order_by = "m.weight DESC, m.symbol"

    with _open_conn() as tracker_conn:
        user, redirect = _require_login(request, tracker_conn)
        if redirect:
            return redirect

    free_conn = _open_free_data_conn()
    if free_conn is None:
        return templates.TemplateResponse(
            "metrics_contributors.html",
            {
                "request": request,
                "user": user,
                "run": None,
                "sectors": [],
                "top_constituents": [],
                "top_contributors": [],
                "all_constituents": [],
                "sort": sort,
                "q": (q or "").strip(),
                "limit": limit,
                "message": request.query_params.get("msg", ""),
                "error": "Free-data DB not found. Run scripts/free_data_pipeline.py first.",
            },
        )

    with free_conn:
        run = _get_cc_cape_run(free_conn, run_id)
        if not run:
            return templates.TemplateResponse(
                "metrics_contributors.html",
                {
                    "request": request,
                    "user": user,
                    "run": None,
                    "sectors": [],
                    "top_constituents": [],
                    "top_contributors": [],
                    "all_constituents": [],
                    "sort": sort,
                    "q": (q or "").strip(),
                    "limit": limit,
                    "message": request.query_params.get("msg", ""),
                    "error": request.query_params.get("error", "No matching CC CAPE run found."),
                },
            )

        sectors = free_conn.execute(
            """
            SELECT COALESCE(gics_sector, '(Unknown)') AS sector,
                   COUNT(*) AS constituents,
                   SUM(weight) AS weight_sum,
                   SUM(weight * company_cape) AS contribution,
                   CASE WHEN SUM(weight) > 0 THEN SUM(weight * company_cape) / SUM(weight) END AS sector_cape
            FROM cc_cape_constituent_metrics
            WHERE run_id = ?
            GROUP BY COALESCE(gics_sector, '(Unknown)')
            ORDER BY contribution DESC
            """,
            (run["run_id"],),
        ).fetchall()

        top_constituents = free_conn.execute(
            """
            SELECT m.symbol,
                   c.security,
                   m.gics_sector,
                   m.weight,
                   m.company_cape,
                   (m.weight * m.company_cape) AS contribution
            FROM cc_cape_constituent_metrics m
            JOIN cc_cape_runs r ON r.run_id = m.run_id
            LEFT JOIN sp500_constituents c
              ON c.symbol = m.symbol AND c.as_of_date = r.as_of_constituents_date
            WHERE m.run_id = ?
            ORDER BY m.weight DESC
            LIMIT ?
            """,
            (run["run_id"], top_n),
        ).fetchall()

        top_contributors = free_conn.execute(
            """
            SELECT m.symbol,
                   c.security,
                   m.gics_sector,
                   m.weight,
                   m.company_cape,
                   (m.weight * m.company_cape) AS contribution
            FROM cc_cape_constituent_metrics m
            JOIN cc_cape_runs r ON r.run_id = m.run_id
            LEFT JOIN sp500_constituents c
              ON c.symbol = m.symbol AND c.as_of_date = r.as_of_constituents_date
            WHERE m.run_id = ?
            ORDER BY (m.weight * m.company_cape) DESC
            LIMIT ?
            """,
            (run["run_id"], top_n),
        ).fetchall()

        q_norm = (q or "").strip()
        where_extra = ""
        params: list[object] = [run["run_id"]]
        if q_norm:
            like = f"%{q_norm.replace('%', '')}%"
            where_extra = " AND (m.symbol LIKE ? OR c.security LIKE ? OR m.gics_sector LIKE ?)"
            params.extend([like, like, like])
        params.append(limit)

        all_constituents = free_conn.execute(
            f"""
            SELECT m.symbol,
                   c.security,
                   m.gics_sector,
                   m.weight,
                   m.company_cape,
                   (m.weight * m.company_cape) AS contribution,
                   m.eps_tag,
                   m.eps_points,
                   m.price_date,
                   m.close_price,
                   m.market_cap
            FROM cc_cape_constituent_metrics m
            JOIN cc_cape_runs r ON r.run_id = m.run_id
            LEFT JOIN sp500_constituents c
              ON c.symbol = m.symbol AND c.as_of_date = r.as_of_constituents_date
            WHERE m.run_id = ?
            {where_extra}
            ORDER BY {order_by}
            LIMIT ?
            """,
            params,
        ).fetchall()

    return templates.TemplateResponse(
        "metrics_contributors.html",
        {
            "request": request,
            "user": user,
            "run": run,
            "sectors": sectors,
            "top_constituents": top_constituents,
            "top_contributors": top_contributors,
            "all_constituents": all_constituents,
            "sort": sort,
            "q": q_norm,
            "limit": limit,
            "message": request.query_params.get("msg", ""),
            "error": request.query_params.get("error", ""),
        },
    )


@app.get("/metrics/cc-cape/export/constituents.csv")
def export_cc_cape_constituents_csv(request: Request, run_id: int | None = None):
    with _open_conn() as tracker_conn:
        user, redirect = _require_login(request, tracker_conn)
        if redirect:
            return redirect

    free_conn = _open_free_data_conn()
    if free_conn is None:
        raise HTTPException(status_code=404, detail="Free-data DB not found.")

    with free_conn:
        run = _get_cc_cape_run(free_conn, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="No CC CAPE runs found.")

        rows = free_conn.execute(
            """
            SELECT m.symbol,
                   c.security,
                   m.cik,
                   m.gics_sector,
                   m.price_date,
                   m.close_price,
                   m.shares_outstanding,
                   m.market_cap,
                   m.eps_tag,
                   m.eps_points,
                   m.avg_real_eps,
                   m.company_cape,
                   m.weight,
                   (m.weight * m.company_cape) AS contribution
            FROM cc_cape_constituent_metrics m
            JOIN cc_cape_runs r ON r.run_id = m.run_id
            LEFT JOIN sp500_constituents c
              ON c.symbol = m.symbol AND c.as_of_date = r.as_of_constituents_date
            WHERE m.run_id = ?
            ORDER BY m.weight DESC, m.symbol
            """,
            (run["run_id"],),
        ).fetchall()

    data = [dict(row) for row in rows]
    fieldnames = [
        "symbol",
        "security",
        "cik",
        "gics_sector",
        "price_date",
        "close_price",
        "shares_outstanding",
        "market_cap",
        "eps_tag",
        "eps_points",
        "avg_real_eps",
        "company_cape",
        "weight",
        "contribution",
    ]
    filename = f"cc_cape_constituents_run_{run['run_id']}.csv"
    return _csv_response(filename, fieldnames, data)


@app.get("/metrics/cc-cape/export/sectors.csv")
def export_cc_cape_sectors_csv(request: Request, run_id: int | None = None):
    with _open_conn() as tracker_conn:
        user, redirect = _require_login(request, tracker_conn)
        if redirect:
            return redirect

    free_conn = _open_free_data_conn()
    if free_conn is None:
        raise HTTPException(status_code=404, detail="Free-data DB not found.")

    with free_conn:
        run = _get_cc_cape_run(free_conn, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="No CC CAPE runs found.")

        rows = free_conn.execute(
            """
            SELECT COALESCE(gics_sector, '(Unknown)') AS sector,
                   COUNT(*) AS constituents,
                   SUM(weight) AS weight_sum,
                   SUM(weight * company_cape) AS contribution,
                   CASE WHEN SUM(weight) > 0 THEN SUM(weight * company_cape) / SUM(weight) END AS sector_cape
            FROM cc_cape_constituent_metrics
            WHERE run_id = ?
            GROUP BY COALESCE(gics_sector, '(Unknown)')
            ORDER BY contribution DESC
            """,
            (run["run_id"],),
        ).fetchall()

    data = [dict(row) for row in rows]
    fieldnames = ["sector", "constituents", "weight_sum", "sector_cape", "contribution"]
    filename = f"cc_cape_sectors_run_{run['run_id']}.csv"
    return _csv_response(filename, fieldnames, data)


@app.get("/metrics/cc-cape/export/series_monthly.csv")
def export_cc_cape_series_monthly_csv(
    request: Request,
    as_of_constituents_date: str | None = None,
    lookback_years: int = 10,
    min_eps_points: int = 8,
    market_cap_min_coverage: float = 0.8,
):
    with _open_conn() as tracker_conn:
        user, redirect = _require_login(request, tracker_conn)
        if redirect:
            return redirect

    free_conn = _open_free_data_conn()
    if free_conn is None:
        raise HTTPException(status_code=404, detail="Free-data DB not found.")

    mcap_permille = int(round(market_cap_min_coverage * 1000))

    with free_conn:
        if not _table_exists(free_conn, "cc_cape_series_monthly"):
            raise HTTPException(status_code=404, detail="Monthly series not found. Run backfill script first.")

        if not as_of_constituents_date:
            row = free_conn.execute(
                "SELECT MAX(as_of_constituents_date) AS as_of_constituents_date FROM cc_cape_series_monthly"
            ).fetchone()
            as_of_constituents_date = row["as_of_constituents_date"] if row else None

        if not as_of_constituents_date:
            raise HTTPException(status_code=404, detail="Monthly series not found. Run backfill script first.")

        rows = free_conn.execute(
            """
            SELECT as_of_constituents_date,
                   observation_date,
                   cc_cape,
                   avg_company_cape,
                   cc_cape_percentile,
                   cc_cape_zscore,
                   shiller_cape,
                   shiller_cape_date,
                   cape_spread,
                   cape_spread_percentile,
                   cape_spread_zscore,
                   symbols_total,
                   symbols_with_price,
                   symbols_with_valid_cape,
                   weighting_method,
                   market_cap_coverage,
                   lookback_years,
                   min_eps_points,
                   market_cap_min_coverage_permille
            FROM cc_cape_series_monthly
            WHERE as_of_constituents_date = ?
              AND lookback_years = ?
              AND min_eps_points = ?
              AND market_cap_min_coverage_permille = ?
            ORDER BY observation_date
            """,
            (as_of_constituents_date, lookback_years, min_eps_points, mcap_permille),
        ).fetchall()

    data = [dict(row) for row in rows]
    fieldnames = [
        "as_of_constituents_date",
        "observation_date",
        "cc_cape",
        "avg_company_cape",
        "cc_cape_percentile",
        "cc_cape_zscore",
        "shiller_cape",
        "shiller_cape_date",
        "cape_spread",
        "cape_spread_percentile",
        "cape_spread_zscore",
        "symbols_total",
        "symbols_with_price",
        "symbols_with_valid_cape",
        "weighting_method",
        "market_cap_coverage",
        "lookback_years",
        "min_eps_points",
        "market_cap_min_coverage_permille",
    ]
    filename = f"cc_cape_series_monthly_{as_of_constituents_date}.csv"
    return _csv_response(filename, fieldnames, data)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "error": request.query_params.get("error", ""),
            "message": request.query_params.get("msg", ""),
        },
    )


@app.post("/login", response_class=HTMLResponse)
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    with _open_conn() as conn:
        user = get_user_by_username(conn, username)
        if not user or not user["active"]:
            return RedirectResponse("/login?error=Invalid%20credentials", status_code=303)
        if not verify_password(password, user["password_salt"], user["password_hash"]):
            return RedirectResponse("/login?error=Invalid%20credentials", status_code=303)

    request.session["username"] = user["username"]
    request.session["role"] = user["role"]
    return RedirectResponse("/board", status_code=303)


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login?msg=Signed%20out", status_code=303)


@app.get("/board", response_class=HTMLResponse)
def board(request: Request, project: str | None = None):
    with _open_conn() as conn:
        user, redirect = _require_login(request, conn)
        if redirect:
            return redirect

        projects = conn.execute("SELECT key, name FROM projects ORDER BY key").fetchall()
        selected_project = (project or (projects[0]["key"] if projects else "")).upper()
        project_row = get_project(conn, selected_project) if selected_project else None

        grouped: dict[str, list] = {status: [] for status in STATUSES}
        if project_row:
            issues = conn.execute(
                f"""
                SELECT i.key, i.title, i.status, i.priority, i.assignee, i.due_date, e.key AS epic_key
                FROM issues i
                LEFT JOIN epics e ON e.id = i.epic_id
                WHERE i.project_id = ?
                ORDER BY {status_rank_sql('i.status')}, {priority_rank_sql('i.priority')}, i.key
                """,
                (project_row["id"],),
            ).fetchall()
            for issue in issues:
                grouped[issue["status"]].append(issue)

    return templates.TemplateResponse(
        "board.html",
        {
            "request": request,
            "user": user,
            "projects": projects,
            "selected_project": selected_project,
            "project_row": project_row,
            "grouped": grouped,
            "statuses": STATUSES,
            "status_labels": STATUS_LABELS,
            "priority_labels": PRIORITY_LABELS,
            "can_edit": _can_edit(user),
            "message": request.query_params.get("msg", ""),
            "error": request.query_params.get("error", ""),
        },
    )


def _issue_summary_payload(issue) -> dict:
    return {
        "key": issue["key"],
        "title": issue["title"],
        "status": issue["status"],
        "status_label": STATUS_LABELS.get(issue["status"], issue["status"]),
        "priority": issue["priority"],
        "priority_label": PRIORITY_LABELS.get(issue["priority"], issue["priority"]),
        "assignee": issue["assignee"],
        "due_date": issue["due_date"],
        "epic_key": issue["epic_key"],
    }


@app.get("/api/board")
def api_board(request: Request, project: str | None = None):
    with _open_conn() as conn:
        user = _require_api_login(request, conn)
        projects = conn.execute("SELECT key, name FROM projects ORDER BY key").fetchall()
        if not projects:
            return {
                "project": None,
                "statuses": [],
                "generated_at": now_utc(),
                "viewer_role": user["role"],
            }

        selected_project = (project or projects[0]["key"]).upper()
        project_row = get_project(conn, selected_project)
        if not project_row:
            raise HTTPException(status_code=404, detail=f"Project '{selected_project}' not found.")

        issues = conn.execute(
            f"""
            SELECT i.key, i.title, i.status, i.priority, i.assignee, i.due_date, e.key AS epic_key
            FROM issues i
            LEFT JOIN epics e ON e.id = i.epic_id
            WHERE i.project_id = ?
            ORDER BY {status_rank_sql('i.status')}, {priority_rank_sql('i.priority')}, i.key
            """,
            (project_row["id"],),
        ).fetchall()

    grouped: dict[str, list] = {status: [] for status in STATUSES}
    for issue in issues:
        grouped[issue["status"]].append(_issue_summary_payload(issue))

    return {
        "project": {"key": project_row["key"], "name": project_row["name"]},
        "statuses": [
            {
                "key": status,
                "label": STATUS_LABELS.get(status, status),
                "count": len(grouped[status]),
                "issues": grouped[status],
            }
            for status in STATUSES
        ],
        "generated_at": now_utc(),
        "viewer_role": user["role"],
    }


@app.get("/api/issues")
def api_issues(
    request: Request,
    project: str | None = None,
    status: str | None = None,
    assignee: str | None = None,
    epic: str | None = None,
    issue_type: str | None = None,
):
    with _open_conn() as conn:
        user = _require_api_login(request, conn)

        projects = conn.execute("SELECT key FROM projects ORDER BY key").fetchall()
        if not projects:
            return {"project": None, "issues": [], "count": 0, "generated_at": now_utc(), "viewer_role": user["role"]}

        selected_project = (project or projects[0]["key"]).upper()
        project_row = get_project(conn, selected_project)
        if not project_row:
            raise HTTPException(status_code=404, detail=f"Project '{selected_project}' not found.")

        query = f"""
            SELECT i.key, i.type, i.title, i.status, i.priority, i.assignee, i.story_points,
                   i.sprint, i.due_date, i.created_at, i.updated_at, e.key AS epic_key
            FROM issues i
            LEFT JOIN epics e ON e.id = i.epic_id
            WHERE i.project_id = ?
        """
        params: list[object] = [project_row["id"]]

        if status:
            try:
                status_normalized = normalize_choice(status, STATUSES, "status")
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            query += " AND i.status = ?"
            params.append(status_normalized)

        if assignee:
            query += " AND i.assignee = ?"
            params.append(assignee)

        if epic:
            epic_key = epic.upper()
            epic_row = conn.execute(
                "SELECT id FROM epics WHERE project_id = ? AND key = ?",
                (project_row["id"], epic_key),
            ).fetchone()
            if not epic_row:
                raise HTTPException(status_code=404, detail=f"Epic '{epic_key}' not found in project '{selected_project}'.")
            query += " AND i.epic_id = ?"
            params.append(epic_row["id"])

        if issue_type:
            query += " AND i.type = ?"
            params.append(issue_type.lower())

        query += f" ORDER BY {status_rank_sql('i.status')}, {priority_rank_sql('i.priority')}, i.key"
        rows = conn.execute(query, params).fetchall()

    issues = []
    for row in rows:
        item = _issue_summary_payload(row)
        item.update(
            {
                "type": row["type"],
                "story_points": row["story_points"],
                "sprint": row["sprint"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
        issues.append(item)

    return {
        "project": {"key": selected_project},
        "count": len(issues),
        "issues": issues,
        "generated_at": now_utc(),
        "viewer_role": user["role"],
    }


@app.get("/api/issues/{issue_key}")
def api_issue_detail(request: Request, issue_key: str):
    issue_key = issue_key.upper()
    with _open_conn() as conn:
        user = _require_api_login(request, conn)
        issue = get_issue(conn, issue_key)
        if not issue:
            raise HTTPException(status_code=404, detail=f"Issue '{issue_key}' not found.")

        deps = conn.execute(
            """
            SELECT i.key, i.title, i.status
            FROM issue_dependencies d
            JOIN issues i ON i.id = d.depends_on_issue_id
            WHERE d.issue_id = ?
            ORDER BY i.key
            """,
            (issue["id"],),
        ).fetchall()
        blocked_by = conn.execute(
            """
            SELECT i.key, i.title, i.status
            FROM issue_dependencies d
            JOIN issues i ON i.id = d.issue_id
            WHERE d.depends_on_issue_id = ?
            ORDER BY i.key
            """,
            (issue["id"],),
        ).fetchall()
        comments = conn.execute(
            """
            SELECT author, body, created_at
            FROM issue_comments
            WHERE issue_id = ?
            ORDER BY id DESC
            """,
            (issue["id"],),
        ).fetchall()
        events = conn.execute(
            """
            SELECT event_type, old_value, new_value, actor, created_at
            FROM issue_events
            WHERE issue_id = ?
            ORDER BY id DESC
            LIMIT 20
            """,
            (issue["id"],),
        ).fetchall()
        project_row = conn.execute("SELECT key, name FROM projects WHERE id = ?", (issue["project_id"],)).fetchone()

    return {
        "issue": {
            "key": issue["key"],
            "title": issue["title"],
            "description": issue["description"],
            "type": issue["type"],
            "status": issue["status"],
            "status_label": STATUS_LABELS.get(issue["status"], issue["status"]),
            "priority": issue["priority"],
            "priority_label": PRIORITY_LABELS.get(issue["priority"], issue["priority"]),
            "assignee": issue["assignee"],
            "story_points": issue["story_points"],
            "sprint": issue["sprint"],
            "due_date": issue["due_date"],
            "created_at": issue["created_at"],
            "updated_at": issue["updated_at"],
            "project_key": project_row["key"] if project_row else None,
            "project_name": project_row["name"] if project_row else None,
            "epic_key": issue["epic_key"],
        },
        "depends_on": [
            {"key": dep["key"], "title": dep["title"], "status": dep["status"], "status_label": STATUS_LABELS.get(dep["status"], dep["status"])}
            for dep in deps
        ],
        "blocks": [
            {
                "key": item["key"],
                "title": item["title"],
                "status": item["status"],
                "status_label": STATUS_LABELS.get(item["status"], item["status"]),
            }
            for item in blocked_by
        ],
        "comments": [{"author": row["author"], "body": row["body"], "created_at": row["created_at"]} for row in comments],
        "events": [
            {
                "event_type": row["event_type"],
                "old_value": row["old_value"],
                "new_value": row["new_value"],
                "actor": row["actor"],
                "created_at": row["created_at"],
            }
            for row in events
        ],
        "generated_at": now_utc(),
        "viewer_role": user["role"],
    }


@app.get("/api/metrics/cc-cape/latest")
def api_cc_cape_latest(request: Request):
    with _open_conn() as conn:
        user = _require_api_login(request, conn)

    free_conn = _open_free_data_conn()
    if free_conn is None:
        raise HTTPException(status_code=404, detail="Free-data DB not found.")

    with free_conn:
        row = free_conn.execute(
            """
            SELECT *
            FROM cc_cape_runs
            ORDER BY run_id DESC
            LIMIT 1
            """
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No CC CAPE runs found.")

    payload = _serialize_cc_cape_run(row)
    payload["viewer_role"] = user["role"]
    payload["generated_at"] = now_utc()
    return payload


@app.get("/api/metrics/cc-cape/runs")
def api_cc_cape_runs(request: Request, limit: int = 50):
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")

    with _open_conn() as conn:
        user = _require_api_login(request, conn)

    free_conn = _open_free_data_conn()
    if free_conn is None:
        raise HTTPException(status_code=404, detail="Free-data DB not found.")

    with free_conn:
        rows = free_conn.execute(
            """
            SELECT *
            FROM cc_cape_runs
            ORDER BY run_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return {
        "count": len(rows),
        "runs": [_serialize_cc_cape_run(r) for r in rows],
        "viewer_role": user["role"],
        "generated_at": now_utc(),
    }


@app.get("/api/metrics/cc-cape/series/monthly")
def api_cc_cape_series_monthly(
    request: Request,
    as_of_constituents_date: str | None = None,
    lookback_years: int = 10,
    min_eps_points: int = 8,
    market_cap_min_coverage: float = 0.8,
    limit: int = 240,
):
    if limit < 1 or limit > 5000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 5000")

    with _open_conn() as conn:
        user = _require_api_login(request, conn)

    free_conn = _open_free_data_conn()
    if free_conn is None:
        raise HTTPException(status_code=404, detail="Free-data DB not found.")

    mcap_permille = int(round(market_cap_min_coverage * 1000))

    with free_conn:
        if not _table_exists(free_conn, "cc_cape_series_monthly"):
            raise HTTPException(status_code=404, detail="Monthly series not found. Run backfill script first.")

        if not as_of_constituents_date:
            row = free_conn.execute(
                "SELECT MAX(as_of_constituents_date) AS as_of_constituents_date FROM cc_cape_series_monthly"
            ).fetchone()
            as_of_constituents_date = row["as_of_constituents_date"] if row else None

        if not as_of_constituents_date:
            raise HTTPException(status_code=404, detail="Monthly series not found. Run backfill script first.")

        rows = free_conn.execute(
            """
            SELECT *
            FROM (
                SELECT as_of_constituents_date,
                       observation_date,
                       cc_cape,
                       avg_company_cape,
                       cc_cape_percentile,
                       cc_cape_zscore,
                       shiller_cape,
                       shiller_cape_date,
                       cape_spread,
                       cape_spread_percentile,
                       cape_spread_zscore,
                       symbols_total,
                       symbols_with_price,
                       symbols_with_valid_cape,
                       weighting_method,
                       market_cap_coverage,
                       lookback_years,
                       min_eps_points,
                       market_cap_min_coverage_permille
                FROM cc_cape_series_monthly
                WHERE as_of_constituents_date = ?
                  AND lookback_years = ?
                  AND min_eps_points = ?
                  AND market_cap_min_coverage_permille = ?
                ORDER BY observation_date DESC
                LIMIT ?
            )
            ORDER BY observation_date
            """,
            (as_of_constituents_date, lookback_years, min_eps_points, mcap_permille, limit),
        ).fetchall()

    return {
        "as_of_constituents_date": as_of_constituents_date,
        "lookback_years": lookback_years,
        "min_eps_points": min_eps_points,
        "market_cap_min_coverage_permille": mcap_permille,
        "count": len(rows),
        "series": [dict(row) for row in rows],
        "viewer_role": user["role"],
        "generated_at": now_utc(),
    }


@app.get("/api/metrics/shiller-cape/series/monthly")
def api_shiller_cape_series_monthly(request: Request, years: int = 50, limit: int | None = None):
    if years < 1 or years > 200:
        raise HTTPException(status_code=400, detail="years must be between 1 and 200")

    if limit is None:
        # Keep a bit of headroom for partial months and source quirks.
        limit = years * 12 + 24

    if limit < 1 or limit > 5000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 5000")

    with _open_conn() as conn:
        user = _require_api_login(request, conn)

    free_conn = _open_free_data_conn()
    if free_conn is None:
        raise HTTPException(status_code=404, detail="Free-data DB not found.")

    with free_conn:
        if not _table_exists(free_conn, "shiller_cape_observations"):
            raise HTTPException(status_code=404, detail="Shiller CAPE series not found. Run free-data pipeline first.")

        rows = free_conn.execute(
            """
            SELECT observation_date, shiller_cape
            FROM (
                SELECT observation_date, shiller_cape
                FROM shiller_cape_observations
                ORDER BY observation_date DESC
                LIMIT ?
            )
            ORDER BY observation_date
            """,
            (limit,),
        ).fetchall()

    return {
        "years": years,
        "limit": limit,
        "count": len(rows),
        "series": [dict(row) for row in rows],
        "viewer_role": user["role"],
        "generated_at": now_utc(),
    }


@app.get("/api/metrics/cc-cape/constituents")
def api_cc_cape_constituents(
    request: Request,
    run_id: int | None = None,
    limit: int = 500,
    sort: str = "weight",
):
    if limit < 1 or limit > 5000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 5000")

    sort = (sort or "weight").strip().lower()
    order_by = "m.weight DESC, m.symbol"
    if sort == "contribution":
        order_by = "(m.weight * m.company_cape) DESC, m.symbol"
    elif sort == "cape":
        order_by = "m.company_cape DESC, m.symbol"
    elif sort == "symbol":
        order_by = "m.symbol"
    elif sort == "weight":
        order_by = "m.weight DESC, m.symbol"
    else:
        raise HTTPException(status_code=400, detail="sort must be one of: weight, contribution, cape, symbol")

    with _open_conn() as conn:
        user = _require_api_login(request, conn)

    free_conn = _open_free_data_conn()
    if free_conn is None:
        raise HTTPException(status_code=404, detail="Free-data DB not found.")

    with free_conn:
        run = _get_cc_cape_run(free_conn, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="No matching CC CAPE run found.")

        rows = free_conn.execute(
            f"""
            SELECT m.symbol,
                   c.security,
                   m.cik,
                   m.gics_sector,
                   m.price_date,
                   m.close_price,
                   m.shares_outstanding,
                   m.market_cap,
                   m.eps_tag,
                   m.eps_points,
                   m.avg_real_eps,
                   m.company_cape,
                   m.weight,
                   (m.weight * m.company_cape) AS contribution
            FROM cc_cape_constituent_metrics m
            JOIN cc_cape_runs r ON r.run_id = m.run_id
            LEFT JOIN sp500_constituents c
              ON c.symbol = m.symbol AND c.as_of_date = r.as_of_constituents_date
            WHERE m.run_id = ?
            ORDER BY {order_by}
            LIMIT ?
            """,
            (run["run_id"], limit),
        ).fetchall()

    return {
        "run": _serialize_cc_cape_run(run),
        "count": len(rows),
        "constituents": [dict(row) for row in rows],
        "viewer_role": user["role"],
        "generated_at": now_utc(),
    }


@app.get("/api/metrics/cc-cape/sectors")
def api_cc_cape_sectors(request: Request, run_id: int | None = None):
    with _open_conn() as conn:
        user = _require_api_login(request, conn)

    free_conn = _open_free_data_conn()
    if free_conn is None:
        raise HTTPException(status_code=404, detail="Free-data DB not found.")

    with free_conn:
        run = _get_cc_cape_run(free_conn, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="No matching CC CAPE run found.")

        rows = free_conn.execute(
            """
            SELECT COALESCE(gics_sector, '(Unknown)') AS sector,
                   COUNT(*) AS constituents,
                   SUM(weight) AS weight_sum,
                   SUM(weight * company_cape) AS contribution,
                   CASE WHEN SUM(weight) > 0 THEN SUM(weight * company_cape) / SUM(weight) END AS sector_cape
            FROM cc_cape_constituent_metrics
            WHERE run_id = ?
            GROUP BY COALESCE(gics_sector, '(Unknown)')
            ORDER BY contribution DESC
            """,
            (run["run_id"],),
        ).fetchall()

    return {
        "run": _serialize_cc_cape_run(run),
        "count": len(rows),
        "sectors": [dict(row) for row in rows],
        "viewer_role": user["role"],
        "generated_at": now_utc(),
    }


@app.get("/issue/{issue_key}", response_class=HTMLResponse)
def issue_detail(request: Request, issue_key: str):
    issue_key = issue_key.upper()
    with _open_conn() as conn:
        user, redirect = _require_login(request, conn)
        if redirect:
            return redirect

        issue = get_issue(conn, issue_key)
        if not issue:
            return RedirectResponse("/board?error=Issue%20not%20found", status_code=303)

        deps = conn.execute(
            """
            SELECT i.key, i.title, i.status
            FROM issue_dependencies d
            JOIN issues i ON i.id = d.depends_on_issue_id
            WHERE d.issue_id = ?
            ORDER BY i.key
            """,
            (issue["id"],),
        ).fetchall()

        blocked_by = conn.execute(
            """
            SELECT i.key, i.title, i.status
            FROM issue_dependencies d
            JOIN issues i ON i.id = d.issue_id
            WHERE d.depends_on_issue_id = ?
            ORDER BY i.key
            """,
            (issue["id"],),
        ).fetchall()

        comments = conn.execute(
            """
            SELECT author, body, created_at
            FROM issue_comments
            WHERE issue_id = ?
            ORDER BY id DESC
            """,
            (issue["id"],),
        ).fetchall()

        events = conn.execute(
            """
            SELECT event_type, old_value, new_value, actor, created_at
            FROM issue_events
            WHERE issue_id = ?
            ORDER BY id DESC
            LIMIT 20
            """,
            (issue["id"],),
        ).fetchall()

        project_row = conn.execute("SELECT key, name FROM projects WHERE id = ?", (issue["project_id"],)).fetchone()

    return templates.TemplateResponse(
        "issue.html",
        {
            "request": request,
            "user": user,
            "issue": issue,
            "project_row": project_row,
            "deps": deps,
            "blocked_by": blocked_by,
            "comments": comments,
            "events": events,
            "statuses": STATUSES,
            "status_labels": STATUS_LABELS,
            "priority_labels": PRIORITY_LABELS,
            "can_edit": _can_edit(user),
            "message": request.query_params.get("msg", ""),
            "error": request.query_params.get("error", ""),
        },
    )


@app.post("/issue/{issue_key}/status")
def update_issue_status(request: Request, issue_key: str, status: str = Form(...)):
    issue_key = issue_key.upper()
    with _open_conn() as conn:
        user, redirect = _require_login(request, conn)
        if redirect:
            return redirect
        if not _can_edit(user):
            return RedirectResponse(f"/issue/{issue_key}?error=Insufficient%20role", status_code=303)

        issue = get_issue(conn, issue_key)
        if not issue:
            return RedirectResponse("/board?error=Issue%20not%20found", status_code=303)

        new_status = normalize_choice(status, STATUSES, "status")
        if issue["status"] != new_status:
            ts = now_utc()
            conn.execute(
                "UPDATE issues SET status = ?, updated_at = ? WHERE id = ?",
                (new_status, ts, issue["id"]),
            )
            conn.execute(
                """
                INSERT INTO issue_events (issue_id, event_type, old_value, new_value, actor, created_at)
                VALUES (?, 'status_changed', ?, ?, ?, ?)
                """,
                (issue["id"], issue["status"], new_status, user["username"], ts),
            )
            conn.commit()

    return RedirectResponse(f"/issue/{issue_key}?msg=Status%20updated", status_code=303)


@app.post("/issue/{issue_key}/comment")
def add_issue_comment(request: Request, issue_key: str, body: str = Form(...)):
    issue_key = issue_key.upper()
    body = body.strip()
    with _open_conn() as conn:
        user, redirect = _require_login(request, conn)
        if redirect:
            return redirect
        if not _can_edit(user):
            return RedirectResponse(f"/issue/{issue_key}?error=Insufficient%20role", status_code=303)
        if not body:
            return RedirectResponse(f"/issue/{issue_key}?error=Comment%20cannot%20be%20empty", status_code=303)

        issue = get_issue(conn, issue_key)
        if not issue:
            return RedirectResponse("/board?error=Issue%20not%20found", status_code=303)

        ts = now_utc()
        conn.execute(
            "INSERT INTO issue_comments (issue_id, author, body, created_at) VALUES (?, ?, ?, ?)",
            (issue["id"], user["username"], body, ts),
        )
        conn.execute(
            """
            INSERT INTO issue_events (issue_id, event_type, old_value, new_value, actor, created_at)
            VALUES (?, 'comment_added', NULL, NULL, ?, ?)
            """,
            (issue["id"], user["username"], ts),
        )
        conn.execute(
            "UPDATE issues SET updated_at = ? WHERE id = ?",
            (ts, issue["id"]),
        )
        conn.commit()

    return RedirectResponse(f"/issue/{issue_key}?msg=Comment%20added", status_code=303)


@app.post("/issue/{issue_key}/dependency")
def add_issue_dependency(request: Request, issue_key: str, depends_on: str = Form(...)):
    issue_key = issue_key.upper()
    depends_on = depends_on.upper().strip()

    with _open_conn() as conn:
        user, redirect = _require_login(request, conn)
        if redirect:
            return redirect
        if not _can_edit(user):
            return RedirectResponse(f"/issue/{issue_key}?error=Insufficient%20role", status_code=303)

        issue = get_issue(conn, issue_key)
        dep = get_issue(conn, depends_on)
        if not issue or not dep:
            return RedirectResponse(f"/issue/{issue_key}?error=Issue%20or%20dependency%20not%20found", status_code=303)
        if issue["project_id"] != dep["project_id"]:
            return RedirectResponse(f"/issue/{issue_key}?error=Dependency%20must%20be%20in%20same%20project", status_code=303)

        ts = now_utc()
        try:
            conn.execute(
                "INSERT INTO issue_dependencies (issue_id, depends_on_issue_id, created_at) VALUES (?, ?, ?)",
                (issue["id"], dep["id"], ts),
            )
        except Exception:
            return RedirectResponse(f"/issue/{issue_key}?error=Dependency%20already%20exists", status_code=303)

        conn.execute(
            """
            INSERT INTO issue_events (issue_id, event_type, old_value, new_value, actor, created_at)
            VALUES (?, 'dependency_added', NULL, ?, ?, ?)
            """,
            (issue["id"], depends_on, user["username"], ts),
        )
        conn.execute(
            "UPDATE issues SET updated_at = ? WHERE id = ?",
            (ts, issue["id"]),
        )
        conn.commit()

    return RedirectResponse(f"/issue/{issue_key}?msg=Dependency%20added", status_code=303)


@app.post("/issue/{issue_key}/assignee")
def update_issue_assignee(request: Request, issue_key: str, assignee: str = Form("")):
    issue_key = issue_key.upper()
    assignee = assignee.strip()

    with _open_conn() as conn:
        user, redirect = _require_login(request, conn)
        if redirect:
            return redirect
        if not _can_edit(user):
            return RedirectResponse(f"/issue/{issue_key}?error=Insufficient%20role", status_code=303)

        issue = get_issue(conn, issue_key)
        if not issue:
            return RedirectResponse("/board?error=Issue%20not%20found", status_code=303)

        ts = now_utc()
        conn.execute(
            "UPDATE issues SET assignee = ?, updated_at = ? WHERE id = ?",
            (assignee if assignee else None, ts, issue["id"]),
        )
        conn.execute(
            """
            INSERT INTO issue_events (issue_id, event_type, old_value, new_value, actor, created_at)
            VALUES (?, 'assignee_changed', ?, ?, ?, ?)
            """,
            (issue["id"], issue["assignee"], assignee if assignee else None, user["username"], ts),
        )
        conn.commit()

    return RedirectResponse(f"/issue/{issue_key}?msg=Assignee%20updated", status_code=303)


@app.get("/admin/users", response_class=HTMLResponse)
def users_page(request: Request):
    with _open_conn() as conn:
        user, redirect = _require_login(request, conn)
        if redirect:
            return redirect
        if user["role"] != "admin":
            return RedirectResponse("/board?error=Admin%20role%20required", status_code=303)

        users = conn.execute(
            "SELECT username, role, active, created_at FROM users ORDER BY username"
        ).fetchall()

    return templates.TemplateResponse(
        "users.html",
        {
            "request": request,
            "user": user,
            "users": users,
            "roles": ROLES,
            "message": request.query_params.get("msg", ""),
            "error": request.query_params.get("error", ""),
        },
    )


@app.get("/admin/audit", response_class=HTMLResponse)
def audit_page(request: Request, username: str | None = None, limit: int = 200):
    if limit < 1 or limit > 2000:
        limit = 200

    with _open_conn() as conn:
        user, redirect = _require_login(request, conn)
        if redirect:
            return redirect
        if user["role"] != "admin":
            return RedirectResponse("/board?error=Admin%20role%20required", status_code=303)

        if username:
            logs = conn.execute(
                """
                SELECT occurred_at, username, role, method, path, status_code, client_ip, user_agent, duration_ms
                FROM access_audit_logs
                WHERE username = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (username, limit),
            ).fetchall()
        else:
            logs = conn.execute(
                """
                SELECT occurred_at, username, role, method, path, status_code, client_ip, user_agent, duration_ms
                FROM access_audit_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    return templates.TemplateResponse(
        "audit.html",
        {
            "request": request,
            "user": user,
            "logs": logs,
            "username_filter": username,
            "limit": limit,
            "message": request.query_params.get("msg", ""),
            "error": request.query_params.get("error", ""),
        },
    )


@app.get("/admin/analytics", response_class=HTMLResponse)
def analytics_page(request: Request, days: int = 14):
    if days < 1 or days > 180:
        days = 14

    with _open_conn() as conn:
        user, redirect = _require_login(request, conn)
        if redirect:
            return redirect
        if user["role"] != "admin":
            return RedirectResponse("/board?error=Admin%20role%20required", status_code=303)

        now = dt.datetime.now(dt.timezone.utc)
        cutoff = (now - dt.timedelta(days=days)).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        total = conn.execute(
            "SELECT COUNT(*) AS c FROM access_audit_logs WHERE occurred_at >= ?",
            (cutoff,),
        ).fetchone()["c"]
        users = conn.execute(
            "SELECT COUNT(DISTINCT username) AS c FROM access_audit_logs WHERE occurred_at >= ? AND username IS NOT NULL AND username != ''",
            (cutoff,),
        ).fetchone()["c"]
        paths = conn.execute(
            "SELECT COUNT(DISTINCT path) AS c FROM access_audit_logs WHERE occurred_at >= ?",
            (cutoff,),
        ).fetchone()["c"]
        errors = conn.execute(
            "SELECT COUNT(*) AS c FROM access_audit_logs WHERE occurred_at >= ? AND status_code >= 400",
            (cutoff,),
        ).fetchone()["c"]

        daily = conn.execute(
            """
            SELECT substr(occurred_at, 1, 10) AS day,
                   COUNT(*) AS requests,
                   COUNT(DISTINCT CASE WHEN username IS NOT NULL AND username != '' THEN username END) AS users,
                   SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) AS errors
            FROM access_audit_logs
            WHERE occurred_at >= ?
            GROUP BY substr(occurred_at, 1, 10)
            ORDER BY day DESC
            LIMIT 90
            """,
            (cutoff,),
        ).fetchall()

        top_paths = conn.execute(
            """
            SELECT path,
                   COUNT(*) AS requests
            FROM access_audit_logs
            WHERE occurred_at >= ?
            GROUP BY path
            ORDER BY requests DESC
            LIMIT 20
            """,
            (cutoff,),
        ).fetchall()

        top_users = conn.execute(
            """
            SELECT COALESCE(NULLIF(username, ''), '(anon)') AS username,
                   COUNT(*) AS requests
            FROM access_audit_logs
            WHERE occurred_at >= ?
            GROUP BY COALESCE(NULLIF(username, ''), '(anon)')
            ORDER BY requests DESC
            LIMIT 20
            """,
            (cutoff,),
        ).fetchall()

    return templates.TemplateResponse(
        "analytics.html",
        {
            "request": request,
            "user": user,
            "days": days,
            "since": cutoff[:10],
            "totals": {"total": int(total or 0), "users": int(users or 0), "paths": int(paths or 0), "errors": int(errors or 0)},
            "daily": daily,
            "top_paths": top_paths,
            "top_users": top_users,
            "message": request.query_params.get("msg", ""),
            "error": request.query_params.get("error", ""),
        },
    )


@app.post("/admin/users")
def users_create(request: Request, username: str = Form(...), password: str = Form(...), role: str = Form(...)):
    with _open_conn() as conn:
        user, redirect = _require_login(request, conn)
        if redirect:
            return redirect
        if user["role"] != "admin":
            return RedirectResponse("/board?error=Admin%20role%20required", status_code=303)

        try:
            create_user(conn, username=username, password=password, role=role)
            conn.commit()
        except Exception as error:
            msg = str(error).replace(" ", "%20")
            return RedirectResponse(f"/admin/users?error={msg}", status_code=303)

    return RedirectResponse("/admin/users?msg=User%20created", status_code=303)


@app.post("/admin/users/{username}/active")
def users_toggle_active(request: Request, username: str, active: str = Form(...)):
    with _open_conn() as conn:
        user, redirect = _require_login(request, conn)
        if redirect:
            return redirect
        if user["role"] != "admin":
            return RedirectResponse("/board?error=Admin%20role%20required", status_code=303)

        value = 1 if active.lower() == "yes" else 0
        conn.execute("UPDATE users SET active = ? WHERE username = ?", (value, username))
        conn.commit()

    return RedirectResponse("/admin/users?msg=User%20updated", status_code=303)
