"""Shared helpers for importing external fundamentals into `company_facts_values`.

We reuse the existing `company_facts_values` table to store *both* SEC XBRL
facts and any external fundamentals, distinguished by `taxonomy`.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path


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


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (name,),
    ).fetchone()
    return bool(row)


def ensure_symbol_overrides_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS symbol_overrides (
            symbol TEXT PRIMARY KEY,
            cik TEXT,
            stooq_symbol TEXT,
            notes TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def load_symbol_overrides(conn: sqlite3.Connection) -> dict[str, dict[str, str]]:
    if not table_exists(conn, "symbol_overrides"):
        return {}
    rows = conn.execute("SELECT symbol, cik, stooq_symbol FROM symbol_overrides").fetchall()
    overrides: dict[str, dict[str, str]] = {}
    for r in rows:
        sym = normalize_symbol(r["symbol"])
        if not sym:
            continue
        overrides[sym] = {
            "cik": normalize_cik(r["cik"]) or "",
            "stooq_symbol": (r["stooq_symbol"] or "").strip(),
        }
    return overrides


def normalize_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper()


def symbol_candidates(symbol: str) -> set[str]:
    sym = normalize_symbol(symbol)
    if not sym:
        return set()
    return {sym, sym.replace(".", "-"), sym.replace("-", ".")}


def normalize_cik(cik_value: str | int | None) -> str | None:
    if cik_value is None:
        return None
    raw = str(cik_value).strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None
    return digits.lstrip("0") or "0"


def parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return f
    raw = str(value).strip()
    if raw == "" or raw.lower() in {"na", "n/a", "null", "none"}:
        return None
    neg = False
    if raw.startswith("(") and raw.endswith(")"):
        neg = True
        raw = raw[1:-1].strip()
    raw = raw.replace(",", "")
    try:
        f = float(raw)
    except ValueError:
        return None
    return -f if neg else f


def load_latest_constituents_symbol_to_cik(conn: sqlite3.Connection, as_of_date: str | None = None) -> dict[str, str]:
    if not table_exists(conn, "sp500_constituents"):
        return {}
    if as_of_date is None:
        row = conn.execute("SELECT MAX(as_of_date) AS as_of_date FROM sp500_constituents").fetchone()
        as_of_date = row["as_of_date"] if row else None
    if not as_of_date:
        return {}
    rows = conn.execute(
        "SELECT symbol, cik FROM sp500_constituents WHERE as_of_date = ?",
        (as_of_date,),
    ).fetchall()
    mapping: dict[str, str] = {}
    for r in rows:
        sym = normalize_symbol(r["symbol"])
        cik = normalize_cik(r["cik"])
        if sym and cik:
            mapping[sym] = cik
    return mapping


def load_sec_ticker_map_symbol_to_cik(conn: sqlite3.Connection) -> dict[str, str]:
    if not table_exists(conn, "sec_ticker_map"):
        return {}
    rows = conn.execute("SELECT symbol, cik FROM sec_ticker_map").fetchall()
    mapping: dict[str, str] = {}
    for r in rows:
        sym = normalize_symbol(r["symbol"])
        cik = normalize_cik(r["cik"])
        if sym and cik:
            mapping[sym] = cik
    return mapping


def resolve_cik(
    symbol_to_cik: dict[str, str],
    sec_map_symbol_to_cik: dict[str, str],
    *,
    symbol: str | None,
    cik_hint: str | int | None,
    overrides: dict[str, dict[str, str]] | None = None,
) -> str | None:
    normalized = normalize_cik(cik_hint)
    if normalized:
        return normalized
    if overrides and symbol:
        override = overrides.get(normalize_symbol(symbol), {})
        normalized = normalize_cik(override.get("cik"))
        if normalized:
            return normalized
    if symbol:
        for candidate in symbol_candidates(symbol):
            cik = symbol_to_cik.get(candidate) or sec_map_symbol_to_cik.get(candidate)
            normalized = normalize_cik(cik)
            if normalized:
                return normalized
    return None


def upsert_company_fact_value(
    conn: sqlite3.Connection,
    *,
    cik: str,
    taxonomy: str,
    tag: str,
    unit: str,
    end_date: str,
    start_date: str | None,
    value: float,
    accession: str,
    fiscal_year: int | None,
    fiscal_period: str | None,
    form: str | None,
    filed_date: str | None,
    frame: str | None,
    fetched_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO company_facts_values (
            cik, taxonomy, tag, unit,
            end_date, start_date, value, accession,
            fiscal_year, fiscal_period, form, filed_date, frame,
            fetched_at
        )
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
            cik,
            taxonomy,
            tag,
            unit,
            end_date,
            start_date,
            float(value),
            accession,
            fiscal_year,
            fiscal_period,
            form,
            filed_date,
            frame,
            fetched_at,
        ),
    )


def delete_existing_taxonomy_tags(conn: sqlite3.Connection, *, taxonomy: str, tags: tuple[str, ...]) -> int:
    placeholders = ", ".join("?" for _ in tags)
    cursor = conn.execute(
        f"DELETE FROM company_facts_values WHERE taxonomy = ? AND tag IN ({placeholders})",
        (taxonomy, *tags),
    )
    return int(cursor.rowcount or 0)
