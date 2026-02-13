#!/usr/bin/env python3
"""Backfill a monthly CC CAPE time series using the free-data pipeline outputs.

This computes "current constituents" CC CAPE historically by holding the
constituent set fixed to the latest ingested S&P 500 snapshot, then computing
company CAPE and CC CAPE as-of each month-end over a requested horizon.

This is a research-grade proxy:
- prices from Stooq
- EPS and shares proxies from SEC company facts
- CPI from FRED
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
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

SERIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS cc_cape_series_monthly (
    as_of_constituents_date TEXT NOT NULL,
    observation_date TEXT NOT NULL,
    run_at TEXT NOT NULL,
    symbols_total INTEGER NOT NULL,
    symbols_with_price INTEGER NOT NULL,
    symbols_with_valid_cape INTEGER NOT NULL,
    min_eps_points INTEGER NOT NULL,
    lookback_years INTEGER NOT NULL,
    weighting_method TEXT NOT NULL,
    market_cap_min_coverage_permille INTEGER NOT NULL,
    market_cap_coverage REAL NOT NULL,
    cc_cape REAL NOT NULL,
    avg_company_cape REAL NOT NULL,
    cc_cape_percentile REAL,
    cc_cape_zscore REAL,
    shiller_cape REAL,
    shiller_cape_date TEXT,
    cape_spread REAL,
    cape_spread_percentile REAL,
    cape_spread_zscore REAL,
    notes_json TEXT,
    PRIMARY KEY (
        as_of_constituents_date,
        observation_date,
        lookback_years,
        min_eps_points,
        market_cap_min_coverage_permille
    )
);
"""

EPS_TAGS = ("EarningsPerShareBasic", "EarningsPerShareDiluted")
SHARES_TAG_PRIORITY = {
    "EntityCommonStockSharesOutstanding": 1,
    "CommonStockSharesOutstanding": 2,
    "WeightedAverageNumberOfSharesOutstandingBasic": 3,
}


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


def init_series_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SERIES_SCHEMA)
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


def parse_date(date_value: str | None) -> dt.date | None:
    if not date_value:
        return None
    try:
        return dt.date.fromisoformat(date_value[:10])
    except ValueError:
        return None


def month_end(date_obj: dt.date) -> dt.date:
    next_month = (date_obj.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
    return next_month - dt.timedelta(days=1)


def add_months(date_obj: dt.date, months: int) -> dt.date:
    year = date_obj.year + (date_obj.month - 1 + months) // 12
    month = (date_obj.month - 1 + months) % 12 + 1
    day = min(date_obj.day, 28)
    return dt.date(year, month, day)


def shift_years(date_obj: dt.date, years: int) -> dt.date:
    try:
        return date_obj.replace(year=date_obj.year + years)
    except ValueError:
        # e.g. Feb 29
        return date_obj.replace(month=2, day=28, year=date_obj.year + years)


def load_latest_constituents(conn: sqlite3.Connection, as_of_date: str | None) -> tuple[str, list[sqlite3.Row]]:
    if as_of_date:
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

    row = conn.execute("SELECT MAX(as_of_date) AS as_of_date FROM sp500_constituents").fetchone()
    if not row or row["as_of_date"] is None:
        return "", []
    return load_latest_constituents(conn, row["as_of_date"])


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
    return float(row["shiller_cape"]), row["observation_date"]


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


def load_eps_series(conn: sqlite3.Connection, cik: str) -> tuple[str, list[tuple[dt.date, float, float | None]]]:
    """Return (eps_tag, [(end_date, eps_value, cpi_at_end_date_or_none), ...])."""
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
    if not rows:
        return "", []

    dedup: dict[tuple[str, str], sqlite3.Row] = {}
    for row in rows:
        key = (row["tag"], row["end_date"])
        current = dedup.get(key)
        if current is None:
            dedup[key] = row
            continue
        current_filed = current["filed_date"] or ""
        candidate_filed = row["filed_date"] or ""
        if candidate_filed >= current_filed:
            dedup[key] = row

    by_tag: dict[str, list[sqlite3.Row]] = {tag: [] for tag in EPS_TAGS}
    for row in dedup.values():
        if row["tag"] in by_tag:
            by_tag[row["tag"]].append(row)
    for tag in EPS_TAGS:
        by_tag[tag].sort(key=lambda r: r["end_date"])

    basic = by_tag["EarningsPerShareBasic"]
    diluted = by_tag["EarningsPerShareDiluted"]
    if len(basic) >= len(diluted):
        chosen_tag = "EarningsPerShareBasic"
        chosen_rows = basic
    else:
        chosen_tag = "EarningsPerShareDiluted"
        chosen_rows = diluted

    series: list[tuple[dt.date, float, float | None]] = []
    for row in chosen_rows:
        end_date = parse_date(row["end_date"])
        if end_date is None:
            continue
        try:
            value = float(row["value"])
        except (TypeError, ValueError):
            continue
        series.append((end_date, value, None))
    return chosen_tag, series


def load_shares_series(conn: sqlite3.Connection, cik: str) -> tuple[list[dt.date], list[float]]:
    rows = conn.execute(
        """
        SELECT tag,
               COALESCE(end_date, filed_date) AS obs_date,
               filed_date,
               value
        FROM company_facts_values
        WHERE cik = ?
          AND tag IN ('EntityCommonStockSharesOutstanding',
                      'CommonStockSharesOutstanding',
                      'WeightedAverageNumberOfSharesOutstandingBasic')
          AND value IS NOT NULL
          AND value > 0
          AND (end_date IS NOT NULL OR filed_date IS NOT NULL)
        ORDER BY COALESCE(end_date, filed_date), filed_date
        """,
        (cik,),
    ).fetchall()
    if not rows:
        return [], []

    dedup: dict[tuple[str, str], tuple[str, float]] = {}
    for row in rows:
        obs_date = row["obs_date"]
        tag = row["tag"]
        filed = row["filed_date"] or ""
        try:
            value = float(row["value"])
        except (TypeError, ValueError):
            continue
        key = (tag, obs_date)
        current = dedup.get(key)
        if current is None or filed >= current[0]:
            dedup[key] = (filed, value)

    by_date: dict[dt.date, list[tuple[str, float]]] = {}
    for (tag, obs_date_raw), (_filed, value) in dedup.items():
        obs_date = parse_date(obs_date_raw)
        if obs_date is None:
            continue
        by_date.setdefault(obs_date, []).append((tag, value))

    dates_sorted = sorted(by_date.keys())
    values_sorted: list[float] = []
    for d in dates_sorted:
        candidates = by_date[d]
        candidates.sort(key=lambda item: SHARES_TAG_PRIORITY.get(item[0], 99))
        values_sorted.append(candidates[0][1])
    return dates_sorted, values_sorted


def shares_asof(shares_dates: list[dt.date], shares_values: list[float], target_date: dt.date) -> float | None:
    if not shares_dates:
        return None
    idx = bisect_right(shares_dates, target_date) - 1
    if idx < 0:
        return None
    value = shares_values[idx]
    return float(value) if value and value > 0 else None


def price_asof(conn: sqlite3.Connection, symbol: str, target_date: dt.date) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT price_date, close_price
        FROM daily_prices
        WHERE symbol = ? AND source = 'stooq' AND price_date <= ?
        ORDER BY price_date DESC
        LIMIT 1
        """,
        (normalize_symbol(symbol), target_date.isoformat()),
    ).fetchone()


def assign_weights(
    metrics: list[dict[str, Any]],
    market_cap_min_coverage: float,
) -> tuple[list[dict[str, Any]], str, float]:
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


def persist_series_row(
    conn: sqlite3.Connection,
    *,
    as_of_constituents_date: str,
    observation_date: str,
    symbols_total: int,
    symbols_with_price: int,
    symbols_with_valid_cape: int,
    min_eps_points: int,
    lookback_years: int,
    weighting_method: str,
    market_cap_min_coverage_permille: int,
    market_cap_coverage: float,
    cc_cape: float,
    avg_company_cape: float,
    shiller_cape: float | None,
    shiller_cape_date: str | None,
    cape_spread: float | None,
    notes: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO cc_cape_series_monthly
        (as_of_constituents_date, observation_date, run_at, symbols_total, symbols_with_price,
         symbols_with_valid_cape, min_eps_points, lookback_years, weighting_method,
         market_cap_min_coverage_permille, market_cap_coverage, cc_cape, avg_company_cape,
         shiller_cape, shiller_cape_date, cape_spread, notes_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(as_of_constituents_date, observation_date, lookback_years, min_eps_points, market_cap_min_coverage_permille)
        DO UPDATE SET
          run_at = excluded.run_at,
          symbols_total = excluded.symbols_total,
          symbols_with_price = excluded.symbols_with_price,
          symbols_with_valid_cape = excluded.symbols_with_valid_cape,
          weighting_method = excluded.weighting_method,
          market_cap_coverage = excluded.market_cap_coverage,
          cc_cape = excluded.cc_cape,
          avg_company_cape = excluded.avg_company_cape,
          shiller_cape = excluded.shiller_cape,
          shiller_cape_date = excluded.shiller_cape_date,
          cape_spread = excluded.cape_spread,
          notes_json = excluded.notes_json
        """,
        (
            as_of_constituents_date,
            observation_date,
            now_utc(),
            symbols_total,
            symbols_with_price,
            symbols_with_valid_cape,
            min_eps_points,
            lookback_years,
            weighting_method,
            market_cap_min_coverage_permille,
            market_cap_coverage,
            cc_cape,
            avg_company_cape,
            shiller_cape,
            shiller_cape_date,
            cape_spread,
            json.dumps(notes, sort_keys=True),
        ),
    )


def compute_and_store_series(args: argparse.Namespace) -> dict[str, Any]:
    with connect_data_db(args.data_db) as conn:
        init_series_db(conn)

        as_of_date, constituents = load_latest_constituents(conn, args.as_of_constituents_date)
        if not constituents:
            raise RuntimeError("No constituents found in free-data DB. Run scripts/free_data_pipeline.py first.")

        if args.max_symbols > 0:
            constituents = constituents[: args.max_symbols]

        cpi_dates, cpi_values = load_cpi_series(conn)
        if not cpi_dates:
            raise RuntimeError("No CPI observations found in free-data DB. Run scripts/free_data_pipeline.py first.")

        max_price_row = conn.execute(
            "SELECT MAX(price_date) AS max_price_date FROM daily_prices WHERE source = 'stooq'"
        ).fetchone()
        max_price_date = parse_date(max_price_row["max_price_date"] if max_price_row else None)
        if max_price_date is None:
            raise RuntimeError("No Stooq prices found. Run scripts/free_data_pipeline.py first.")

        end_date = parse_date(args.end_date) if args.end_date else None
        if end_date is None:
            end_date = max_price_date
        # last complete month-end <= end_date
        end_obs = month_end(end_date)
        if end_obs > end_date:
            end_obs = month_end(add_months(end_date, -1))

        start_obs = month_end(shift_years(end_obs, -args.series_years))
        if start_obs > end_obs:
            raise RuntimeError("Computed start date is after end date; check --series-years/--end-date.")

        market_cap_min_coverage_permille = int(round(args.market_cap_min_coverage * 1000))
        market_cap_min_coverage = market_cap_min_coverage_permille / 1000.0

        # Preload EPS and shares series for all CIKs we can resolve.
        eps_cache: dict[str, tuple[str, list[tuple[dt.date, float]]]] = {}
        shares_cache: dict[str, tuple[list[dt.date], list[float]]] = {}

        resolved = []
        missing_cik = 0
        for row in constituents:
            symbol = row["symbol"]
            cik = resolve_cik(conn, symbol, row["cik"])
            if cik is None:
                missing_cik += 1
                continue
            resolved.append((symbol, cik, row["gics_sector"]))
            if cik not in eps_cache:
                tag, eps_series_raw = load_eps_series(conn, cik)
                eps_series = [(d, v) for (d, v, _cpi) in eps_series_raw]
                eps_cache[cik] = (tag, eps_series)
            if cik not in shares_cache:
                shares_cache[cik] = load_shares_series(conn, cik)

        # Build month-end observation list.
        obs_dates: list[dt.date] = []
        cursor = dt.date(start_obs.year, start_obs.month, 1)
        end_cursor = dt.date(end_obs.year, end_obs.month, 1)
        while cursor <= end_cursor:
            obs_dates.append(month_end(cursor))
            cursor = add_months(cursor, 1).replace(day=1)

        stored = 0
        observations = 0
        for obs_date in obs_dates:
            observations += 1
            target_cpi = cpi_for_date(obs_date, cpi_dates, cpi_values)
            if target_cpi is None:
                continue

            metrics: list[dict[str, Any]] = []
            symbols_with_price = 0
            skipped_no_price = 0
            skipped_no_eps = 0
            skipped_eps_window = 0
            skipped_nonpositive = 0
            max_price_used: dt.date | None = None

            window_start = shift_years(obs_date, -args.lookback_years)

            for symbol, cik, sector in resolved:
                price_row = price_asof(conn, symbol, obs_date)
                if not price_row:
                    skipped_no_price += 1
                    continue
                symbols_with_price += 1

                price_date = parse_date(price_row["price_date"])
                if price_date and (max_price_used is None or price_date > max_price_used):
                    max_price_used = price_date
                close_price = float(price_row["close_price"])

                eps_tag, eps_series = eps_cache.get(cik, ("", []))
                if not eps_series:
                    skipped_no_eps += 1
                    continue

                # Filter EPS points within (window_start, obs_date]
                eps_points = [(d, v) for (d, v) in eps_series if window_start <= d <= obs_date]
                if len(eps_points) < args.min_eps_points:
                    skipped_eps_window += 1
                    continue

                real_eps_values: list[float] = []
                for end_d, eps_val in eps_points:
                    cpi_end = cpi_for_date(end_d, cpi_dates, cpi_values)
                    if cpi_end is None or cpi_end <= 0:
                        continue
                    real_eps_values.append(eps_val * (target_cpi / cpi_end))
                if len(real_eps_values) < args.min_eps_points:
                    skipped_eps_window += 1
                    continue

                avg_real_eps = statistics.mean(real_eps_values)
                if not avg_real_eps or avg_real_eps <= 0:
                    skipped_nonpositive += 1
                    continue

                shares_dates, shares_values = shares_cache.get(cik, ([], []))
                shares = shares_asof(shares_dates, shares_values, obs_date)
                market_cap = close_price * shares if shares else None

                company_cape = close_price / avg_real_eps
                metrics.append(
                    {
                        "symbol": symbol,
                        "cik": cik,
                        "gics_sector": sector,
                        "price_date": price_row["price_date"],
                        "close_price": close_price,
                        "shares_outstanding": shares,
                        "market_cap": market_cap,
                        "eps_tag": eps_tag,
                        "eps_points": len(real_eps_values),
                        "avg_real_eps": avg_real_eps,
                        "company_cape": company_cape,
                    }
                )

            if not metrics:
                continue

            metrics, weighting_method, mcap_cov = assign_weights(metrics, market_cap_min_coverage)
            cc_cape = sum(m["weight"] * m["company_cape"] for m in metrics)
            avg_company_cape = statistics.mean(m["company_cape"] for m in metrics)

            shiller = shiller_cape_asof(conn, obs_date)
            shiller_cape: float | None = None
            shiller_cape_date: str | None = None
            if shiller is not None:
                shiller_cape, shiller_cape_date = shiller
            cape_spread = (cc_cape - shiller_cape) if shiller_cape is not None else None

            notes = {
                "asof_price_policy": "last_trading_day_on_or_before_observation_date",
                "observation_date": obs_date.isoformat(),
                "max_price_date_used": max_price_used.isoformat() if max_price_used else None,
                "symbols_resolved": len(resolved),
                "missing_cik": missing_cik,
                "skipped_no_price": skipped_no_price,
                "skipped_no_eps_series": skipped_no_eps,
                "skipped_eps_window": skipped_eps_window,
                "skipped_nonpositive_avg_real_eps": skipped_nonpositive,
                "market_cap_min_coverage": market_cap_min_coverage,
            }

            persist_series_row(
                conn,
                as_of_constituents_date=as_of_date,
                observation_date=obs_date.isoformat(),
                symbols_total=len(constituents),
                symbols_with_price=symbols_with_price,
                symbols_with_valid_cape=len(metrics),
                min_eps_points=args.min_eps_points,
                lookback_years=args.lookback_years,
                weighting_method=weighting_method,
                market_cap_min_coverage_permille=market_cap_min_coverage_permille,
                market_cap_coverage=mcap_cov,
                cc_cape=cc_cape,
                avg_company_cape=avg_company_cape,
                shiller_cape=shiller_cape,
                shiller_cape_date=shiller_cape_date,
                cape_spread=cape_spread,
                notes=notes,
            )
            stored += 1

        conn.commit()

        # Compute percentiles and z-scores for this series and persist.
        rows = conn.execute(
            """
            SELECT observation_date, cc_cape, cape_spread
            FROM cc_cape_series_monthly
            WHERE as_of_constituents_date = ?
              AND lookback_years = ?
              AND min_eps_points = ?
              AND market_cap_min_coverage_permille = ?
            ORDER BY observation_date
            """,
            (as_of_date, args.lookback_years, args.min_eps_points, market_cap_min_coverage_permille),
        ).fetchall()
        cc_values = [float(r["cc_cape"]) for r in rows]
        spread_values = [float(r["cape_spread"]) for r in rows if r["cape_spread"] is not None]
        cc_mean = statistics.mean(cc_values) if cc_values else None
        cc_stdev = statistics.stdev(cc_values) if len(cc_values) >= 2 else None
        sp_mean = statistics.mean(spread_values) if spread_values else None
        sp_stdev = statistics.stdev(spread_values) if len(spread_values) >= 2 else None

        for r in rows:
            cc = float(r["cc_cape"])
            cc_pct = empirical_percentile(cc_values, cc)
            cc_z = ((cc - cc_mean) / cc_stdev) if (cc_mean is not None and cc_stdev and cc_stdev > 0) else None

            spread = float(r["cape_spread"]) if r["cape_spread"] is not None else None
            sp_pct = empirical_percentile(spread_values, spread) if (spread is not None and spread_values) else None
            sp_z = (
                ((spread - sp_mean) / sp_stdev)
                if (spread is not None and sp_mean is not None and sp_stdev and sp_stdev > 0)
                else None
            )

            conn.execute(
                """
                UPDATE cc_cape_series_monthly
                SET cc_cape_percentile = ?,
                    cc_cape_zscore = ?,
                    cape_spread_percentile = ?,
                    cape_spread_zscore = ?
                WHERE as_of_constituents_date = ?
                  AND observation_date = ?
                  AND lookback_years = ?
                  AND min_eps_points = ?
                  AND market_cap_min_coverage_permille = ?
                """,
                (
                    cc_pct,
                    cc_z,
                    sp_pct,
                    sp_z,
                    as_of_date,
                    r["observation_date"],
                    args.lookback_years,
                    args.min_eps_points,
                    market_cap_min_coverage_permille,
                ),
            )
        conn.commit()

    return {
        "completed_at": now_utc(),
        "as_of_constituents_date": as_of_date,
        "start_observation_date": start_obs.isoformat(),
        "end_observation_date": end_obs.isoformat(),
        "series_years": args.series_years,
        "lookback_years": args.lookback_years,
        "min_eps_points": args.min_eps_points,
        "market_cap_min_coverage": market_cap_min_coverage,
        "observations_requested": observations,
        "observations_stored": stored,
        "symbols_total": len(constituents),
        "symbols_resolved": len(resolved),
        "missing_cik": missing_cik,
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


def update_tracker(tracker_db_path: str, summary: dict[str, Any], author: str) -> None:
    with tracker_connect(tracker_db_path) as conn:
        init_tracker_db(conn)
        ensure_default_admin(conn)

        body = (
            f"Backfilled monthly CC CAPE series at {summary['completed_at']} UTC.\n\n"
            f"As-of constituents: {summary['as_of_constituents_date']}\n"
            f"Range: {summary['start_observation_date']} to {summary['end_observation_date']}\n"
            f"Observations stored: {summary['observations_stored']} / {summary['observations_requested']}\n"
            f"Symbols total: {summary['symbols_total']}\n"
            f"Symbols resolved to CIK: {summary['symbols_resolved']} (missing {summary['missing_cik']})\n"
            f"Lookback years: {summary['lookback_years']}\n"
            f"Min EPS points: {summary['min_eps_points']}\n"
            f"Market-cap min coverage: {summary['market_cap_min_coverage']:.3f}\n\n"
            "Series table: cc_cape_series_monthly\n"
            "API: /api/metrics/cc-cape/series/monthly"
        )
        tracker_add_comment(conn, "CAPE-11", author, body)
        tracker_move_status_if_needed(conn, "CAPE-11", "in_progress", author)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill monthly CC CAPE series (free-data proxy).")
    parser.add_argument("--data-db", default="data/free_data.db", help="Path to free-data SQLite DB.")
    parser.add_argument("--tracker-db", default="data/internal_jira.db", help="Path to tracker SQLite DB.")
    parser.add_argument(
        "--as-of-constituents-date",
        default=None,
        help="Freeze current constituents to this YYYY-MM-DD snapshot (default: latest).",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Series end date YYYY-MM-DD (default: latest available price date, rounded down to last complete month-end).",
    )
    parser.add_argument("--series-years", type=int, default=10, help="Years of monthly observations to backfill.")
    parser.add_argument("--lookback-years", type=int, default=10, help="CAPE lookback window in years.")
    parser.add_argument("--min-eps-points", type=int, default=8, help="Minimum EPS observations required per company.")
    parser.add_argument(
        "--market-cap-min-coverage",
        type=float,
        default=0.8,
        help="Minimum market-cap coverage to use market-cap weighting; otherwise equal-weight fallback.",
    )
    parser.add_argument("--max-symbols", type=int, default=0, help="Optional cap on symbols processed (0 = all).")
    parser.add_argument(
        "--update-tracker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, post backfill summary into tracker issue CAPE-11.",
    )
    parser.add_argument("--tracker-author", default="backfill-bot", help="Author for tracker comments/events.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = compute_and_store_series(args)
    if args.update_tracker:
        try:
            update_tracker(args.tracker_db, summary, args.tracker_author)
        except Exception as exc:  # noqa: BLE001
            summary["tracker_error"] = str(exc)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

