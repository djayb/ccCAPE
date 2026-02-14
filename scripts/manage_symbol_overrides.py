#!/usr/bin/env python3
"""Manage symbol overrides stored in `free_data.db`.

Overrides allow you to fix edge cases deterministically:
- Force a `symbol -> CIK` mapping (when SEC/Wikipedia mappings are missing/wrong).
- Force a `symbol -> Stooq ticker` mapping for price fetches.

Storage:
- Table: `symbol_overrides` in `data/free_data.db`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from internal_jira import now_utc  # noqa: E402


def ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def connect_db(path: str) -> sqlite3.Connection:
    ensure_parent_dir(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


def normalize_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper()


def normalize_cik(cik_value: str | int | None) -> str | None:
    if cik_value is None:
        return None
    raw = str(cik_value).strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None
    return digits.lstrip("0") or "0"


def normalize_stooq_symbol(value: str | None) -> str | None:
    if not value:
        return None
    raw = str(value).strip().lower()
    if not raw:
        return None
    if raw.endswith(".us"):
        raw = raw[:-3]
    return raw


def ensure_schema(conn: sqlite3.Connection) -> None:
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


def list_overrides(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    ensure_schema(conn)
    rows = conn.execute(
        "SELECT symbol, cik, stooq_symbol, notes, updated_at FROM symbol_overrides ORDER BY symbol"
    ).fetchall()
    return [dict(r) for r in rows]


def set_override(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    cik: str | None,
    stooq_symbol: str | None,
    notes: str | None,
) -> None:
    ensure_schema(conn)
    sym = normalize_symbol(symbol)
    if not sym:
        raise SystemExit("symbol is required")
    cik_norm = normalize_cik(cik) if cik is not None else None
    stooq_norm = normalize_stooq_symbol(stooq_symbol) if stooq_symbol is not None else None
    conn.execute(
        """
        INSERT INTO symbol_overrides (symbol, cik, stooq_symbol, notes, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(symbol)
        DO UPDATE SET
            cik = excluded.cik,
            stooq_symbol = excluded.stooq_symbol,
            notes = excluded.notes,
            updated_at = excluded.updated_at
        """,
        (sym, cik_norm or "", stooq_norm or "", (notes or "").strip(), now_utc()),
    )
    conn.commit()


def delete_override(conn: sqlite3.Connection, *, symbol: str) -> int:
    ensure_schema(conn)
    sym = normalize_symbol(symbol)
    if not sym:
        raise SystemExit("symbol is required")
    cur = conn.execute("DELETE FROM symbol_overrides WHERE symbol = ?", (sym,))
    conn.commit()
    return int(cur.rowcount or 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage ccCAPE symbol overrides.")
    parser.add_argument("--data-db", default="data/free_data.db", help="Path to free-data SQLite DB.")

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List overrides.")

    p_set = sub.add_parser("set", help="Create/update an override.")
    p_set.add_argument("--symbol", required=True, help="Constituent symbol (e.g. BRK.B).")
    p_set.add_argument("--cik", default=None, help="Override CIK (digits).")
    p_set.add_argument("--stooq-symbol", default=None, help="Override Stooq base ticker (e.g. brk-b).")
    p_set.add_argument("--notes", default=None, help="Optional notes.")

    p_del = sub.add_parser("delete", help="Delete an override.")
    p_del.add_argument("--symbol", required=True, help="Constituent symbol.")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    with connect_db(args.data_db) as conn:
        if args.cmd == "list":
            rows = list_overrides(conn)
            print(json.dumps({"count": len(rows), "overrides": rows}, indent=2, sort_keys=True))
            return 0
        if args.cmd == "set":
            set_override(
                conn,
                symbol=args.symbol,
                cik=args.cik,
                stooq_symbol=args.stooq_symbol,
                notes=args.notes,
            )
            print(json.dumps({"status": "ok", "symbol": normalize_symbol(args.symbol)}, indent=2, sort_keys=True))
            return 0
        if args.cmd == "delete":
            deleted = delete_override(conn, symbol=args.symbol)
            print(json.dumps({"status": "ok", "deleted": deleted, "symbol": normalize_symbol(args.symbol)}, indent=2, sort_keys=True))
            return 0

    raise SystemExit("unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())

