import sqlite3
import unittest


def _init_min_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS symbol_overrides (
            symbol TEXT PRIMARY KEY,
            cik TEXT,
            stooq_symbol TEXT,
            notes TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS sec_ticker_map (
            symbol TEXT PRIMARY KEY,
            cik TEXT
        );
        """
    )


class TestResolveCikOverrides(unittest.TestCase):
    def test_override_beats_hint(self) -> None:
        from scripts.calc_cc_cape_free import resolve_cik

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _init_min_tables(conn)

        conn.execute("INSERT INTO symbol_overrides (symbol, cik) VALUES ('ABC', '0000012345')")
        conn.commit()

        self.assertEqual(resolve_cik(conn, "ABC", "0000099999"), "12345")

    def test_hint_used_when_no_override(self) -> None:
        from scripts.calc_cc_cape_free import resolve_cik

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _init_min_tables(conn)

        self.assertEqual(resolve_cik(conn, "ABC", "0000012345"), "12345")

    def test_sec_map_used_with_symbol_variants(self) -> None:
        from scripts.calc_cc_cape_free import resolve_cik

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _init_min_tables(conn)

        conn.execute("INSERT INTO sec_ticker_map (symbol, cik) VALUES ('BRK-B', '0000100001')")
        conn.commit()

        self.assertEqual(resolve_cik(conn, "BRK.B", None), "100001")


if __name__ == "__main__":
    unittest.main()

