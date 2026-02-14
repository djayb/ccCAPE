# Free Data Alternatives for CC CAPE

Date: 2026-02-13

## Goal

Build CC CAPE without paid market data subscriptions, understanding accuracy and licensing tradeoffs.

## What You Can Use for Free (Practical Stack)

1. Filings and fundamentals (free, official)
- SEC EDGAR APIs (`data.sec.gov`) for submissions and XBRL company facts.
- Useful endpoints:
  - `https://data.sec.gov/submissions/CIK##########.json`
  - `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`
- SEC also publishes nightly bulk ZIPs for submissions and company facts.

2. Inflation for real earnings conversion (free, official)
- FRED CPI series (`CPIAUCSL`) + FRED API key.

3. Shiller CAPE benchmark (free, unofficial)
- Multpl publishes a public Shiller PE (CAPE) history table that can be scraped for research/prototyping.
- For academic source-of-truth, Robert Shiller publishes data via Yale (often as XLS), which requires parsing and careful field mapping.

4. Current constituent list (free, unofficial)
- Wikipedia S&P 500 component table is commonly used for prototypes.
- For official production-grade membership/weights, S&P constituent data is generally licensed.

5. Price data options without paid contract
- SimFin free tier (API + bulk download), but free tier has shorter history and non-commercial constraints.
- Alpha Vantage free tier works for prototyping but request limits are low for full S&P 500 daily processing.
- Nasdaq Data Link has free and premium feeds; free feed coverage/quality differs by dataset.

Optional extension for fundamentals history:

- If you have an internal or licensed fundamentals dataset, you can import it into `company_facts_values`
  (distinguished via `taxonomy`) to extend CC CAPE history beyond SEC XBRL coverage.
- See: `docs/EXTERNAL_FUNDAMENTALS.md`

## Hard Constraint for True CC CAPE

CC CAPE needs:
- 10-year real earnings history per current constituent.
- Current market-cap weighting (or at least robust price + shares).

Without paid feeds, this is possible but operationally harder:
- SEC can cover fundamentals well.
- The main bottleneck is robust, scalable, and license-safe daily price/market-cap coverage for all constituents.

## Recommended No-Paid Implementation Path

## Phase A: Research-grade MVP (no paid data)

- Constituents: Wikipedia snapshot + change-log capture.
- Fundamentals: SEC company facts (10+ years where available).
- Inflation: FRED CPIAUCSL.
- Prices: begin with free source for prototyping (document licensing and field mapping).
- Output: weekly (not daily) CC CAPE refresh to reduce stress on free endpoints.

## Phase B: Data quality hardening

- Add reconciliation checks:
  - ticker-to-CIK mapping breaks
  - missing quarterly earnings windows
  - corporate action anomalies
- Add fallback hierarchy for missing prices.
- Keep audit trail per point-in-time calculation.

## Phase C: Decide on production posture

- If this remains internal research, free stack can be acceptable with caveats.
- If this becomes client-facing/commercial, move to licensed constituent + price history feeds.

## Licensing and Compliance Notes

- SEC automated access must follow fair-access constraints and declared user-agent rules.
- FRED requires API keys and has owner-specific restrictions for some series.
- SimFin free tier is non-commercial per its license terms.
- Official S&P constituent/weight data is typically licensed.

## Suggested Weekly Refresh Cadence (Free Stack)

1. Pull constituents snapshot.
2. Resolve CIK mapping.
3. Pull latest SEC filings / company facts deltas.
4. Refresh CPI.
5. Refresh price snapshot.
6. Recompute CC CAPE + spread.
7. Run QA checks and publish.

## Current Repo Implementation

Pipeline script:

- `scripts/free_data_pipeline.py`
- `scripts/calc_cc_cape_free.py`
- `scripts/weekly_scheduler.py`

Storage:

- `data/free_data.db`

Tracker integration:

- Automatically comments on `CAPE-4`, `CAPE-5`, `CAPE-6` and creates/updates a dedicated free-data issue under Phase 1.
- Automatically comments on `CAPE-8`, `CAPE-9` and creates/updates a dedicated free-calc issue under Phase 2.
