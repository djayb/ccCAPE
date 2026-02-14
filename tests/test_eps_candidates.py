import sqlite3
import unittest


def _init_company_facts_values(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
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
        """
    )


class TestEpsCandidates(unittest.TestCase):
    def test_filters_to_annual_and_dedups_by_longest_period(self) -> None:
        from scripts.calc_cc_cape_free import eps_candidates

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _init_company_facts_values(conn)

        cik = "1"

        # Quarterly EPS (should be filtered out by period_days >= 330).
        conn.execute(
            """
            INSERT INTO company_facts_values
              (cik, taxonomy, tag, unit, end_date, start_date, value, accession, fiscal_year,
               fiscal_period, form, filed_date, frame, fetched_at)
            VALUES (?, 'us-gaap', 'EarningsPerShareBasic', 'USD/shares',
                    '2020-12-31', '2020-10-01', 2.5, 'q1', 2020, 'Q4', '10-Q', '2021-01-15', NULL, '2026-01-01T00:00:00Z')
            """,
            (cik,),
        )

        # Annual EPS, shorter period.
        conn.execute(
            """
            INSERT INTO company_facts_values
              (cik, taxonomy, tag, unit, end_date, start_date, value, accession, fiscal_year,
               fiscal_period, form, filed_date, frame, fetched_at)
            VALUES (?, 'us-gaap', 'EarningsPerShareBasic', 'USD/shares',
                    '2020-12-31', '2020-01-01', 10.0, 'a1', 2020, 'FY', '10-K', '2021-02-01', NULL, '2026-01-01T00:00:00Z')
            """,
            (cik,),
        )

        # Annual-ish EPS with same end_date but longer period -> should win dedupe.
        conn.execute(
            """
            INSERT INTO company_facts_values
              (cik, taxonomy, tag, unit, end_date, start_date, value, accession, fiscal_year,
               fiscal_period, form, filed_date, frame, fetched_at)
            VALUES (?, 'us-gaap', 'EarningsPerShareBasic', 'USD/shares',
                    '2020-12-31', '2019-12-29', 9.0, 'a2', 2020, 'FY', '10-K', '2021-01-15', NULL, '2026-01-01T00:00:00Z')
            """,
            (cik,),
        )
        conn.commit()

        cands = eps_candidates(conn, cik)
        tags = [t for (t, _series) in cands]
        self.assertIn("EarningsPerShareBasic", tags)

        series = None
        for t, s in cands:
            if t == "EarningsPerShareBasic":
                series = s
                break
        self.assertIsNotNone(series)
        self.assertEqual(len(series), 1)
        self.assertAlmostEqual(float(series[0]["value"]), 9.0)

    def test_computed_eps_from_net_income_and_shares(self) -> None:
        from scripts.calc_cc_cape_free import eps_candidates

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _init_company_facts_values(conn)

        cik = "2"
        conn.execute(
            """
            INSERT INTO company_facts_values
              (cik, taxonomy, tag, unit, end_date, start_date, value, accession, fiscal_year,
               fiscal_period, form, filed_date, frame, fetched_at)
            VALUES (?, 'us-gaap', 'NetIncomeLoss', 'USD',
                    '2020-12-31', '2020-01-01', 100.0, 'ni1', 2020, 'FY', '10-K', '2021-02-01', NULL, '2026-01-01T00:00:00Z')
            """,
            (cik,),
        )
        conn.execute(
            """
            INSERT INTO company_facts_values
              (cik, taxonomy, tag, unit, end_date, start_date, value, accession, fiscal_year,
               fiscal_period, form, filed_date, frame, fetched_at)
            VALUES (?, 'us-gaap', 'WeightedAverageNumberOfSharesOutstandingBasic', 'shares',
                    '2020-12-31', '2020-01-01', 10.0, 'sh1', 2020, 'FY', '10-K', '2021-02-01', NULL, '2026-01-01T00:00:00Z')
            """,
            (cik,),
        )
        conn.commit()

        cands = eps_candidates(conn, cik)
        tag = "ComputedEPS(NetIncomeLoss/WeightedAvgSharesBasic)"
        tags = [t for (t, _series) in cands]
        self.assertIn(tag, tags)

        series = None
        for t, s in cands:
            if t == tag:
                series = s
                break
        self.assertIsNotNone(series)
        self.assertEqual(len(series), 1)
        self.assertAlmostEqual(float(series[0]["value"]), 10.0)


if __name__ == "__main__":
    unittest.main()

