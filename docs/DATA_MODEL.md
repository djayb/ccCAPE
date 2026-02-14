# ccCAPE Data Model (SQLite)

Date: 2026-02-13

This document describes the **logical schema** used by ccCAPE. Source-of-truth for exact DDL lives in:

- Tracker: `internal_jira.py` (`SCHEMA`)
- Free-data ingestion: `scripts/free_data_pipeline.py` (`DATA_SCHEMA`)
- CC CAPE calc: `scripts/calc_cc_cape_free.py` (`CALC_SCHEMA` + migrations)
- Monthly series: `scripts/backfill_cc_cape_series_free.py` (`SERIES_SCHEMA`)

## 1) Tracker DB: `data/internal_jira.db`

Purpose: internal execution tracking + lightweight access telemetry.

Core tables:

- `projects`: project containers (e.g., `CAPE`)
- `epics`: roadmap epics (e.g., Phase 1/2/3)
- `issues`: work items with status/priority/sprint metadata
- `issue_dependencies`: blocking relationships
- `issue_comments`: discussion log
- `issue_events`: audit trail of status/field changes
- `users`: local auth (username, PBKDF2 hash, role, active)
- `access_audit_logs`: best-effort request logs (method/path/status/user/duration)

## 2) Free Data DB: `data/free_data.db`

Purpose: raw ingested datasets + curated computed metrics.

### Raw / ingested tables

- `sp500_constituents`
  - Key: `(symbol, as_of_date)`
  - Fields include `cik`, `gics_sector`, and `source_url`
  - Lineage: `as_of_date`, `ingested_at`, `source_url`

- `sec_ticker_map`
  - Key: `symbol`
  - Lineage: `ingested_at`, `source_url`

- `cpi_observations`
  - Key: `observation_date`
  - Lineage: `ingested_at`, `source_url`

- `shiller_cape_observations`
  - Key: `observation_date`
  - Lineage: `ingested_at`, `source_url`

- `company_facts_meta`
  - Key: `cik`
  - Lineage: `fetched_at`, `source_url`

- `company_facts_values`
  - Key: `(cik, taxonomy, tag, unit, end_date, accession)`
  - Lineage: `fetched_at` (and filing metadata like `form`, `filed_date`)
  - Notes:
    - SEC XBRL facts typically use taxonomies like `us-gaap` and `dei`
    - external / licensed fundamentals can be imported into the same table using a different `taxonomy`
      (see: `docs/EXTERNAL_FUNDAMENTALS.md`)

- `daily_prices`
  - Key: `(symbol, price_date, source)`
  - Lineage: `fetched_at`, `source`

- `ingestion_runs`
  - Step-level and pipeline-level logs of ingestion runs
  - `details_json` includes row counts and warnings

### Curated / computed tables

- `cc_cape_runs`
  - One row per CC CAPE run (headline metrics + notes)
  - Key: `run_id`
  - Includes:
    - `as_of_constituents_date`
    - `latest_price_date`
    - `cc_cape`, `avg_company_cape`
    - `shiller_cape`, `shiller_cape_date`, `cape_spread`
    - run-history percentiles / z-scores
    - `notes_json`

- `cc_cape_constituent_metrics`
  - Per-run per-constituent decomposition
  - Key: `(run_id, symbol)`
  - Includes:
    - `company_cape`, `avg_real_eps`, `eps_points`, `weight`
    - `market_cap` (proxy) and sector fields

- `cc_cape_series_monthly`
  - Monthly time series for “current constituents” backfill
  - Key: `(as_of_constituents_date, observation_date, lookback_years, min_eps_points, market_cap_min_coverage_permille)`
  - Includes:
    - `cc_cape`, `shiller_cape`, `cape_spread`
    - percentiles / z-scores within the series
    - `notes_json` with coverage counters and assumptions

## 3) Lineage and Freshness

Each ingested record set has at least one of:

- `ingested_at` / `fetched_at`
- `source_url`

The pipeline also writes a summary record:

- `ingestion_runs` where `step = 'pipeline'`

The UI uses this to display:

- last run status and step summaries
- freshness/coverage warnings (prices/CPI/Shiller/facts)
