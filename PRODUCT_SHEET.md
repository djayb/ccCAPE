# Product Sheet: Current Constituents CAPE Tracker

Version: 0.1  
Date: 2026-02-13  
Owner: Product + Quant Engineering

## 1. Product Summary

Build a tool that calculates and tracks Current Constituents CAPE (CC CAPE) for a target equity index (starting with S&P 500), then compares it to traditional Shiller CAPE and publishes the CAPE Spread over time.

Source reference: [Research Affiliates paper](https://www.researchaffiliates.com/content/dam/ra/publications/pdf/1070-current-constituents-cape.pdf)

## 2. Problem Statement

Traditional Shiller CAPE is index-level and can be affected by structural changes in index membership over time.  
The product should answer:

- What is valuation for the current set of constituents only?
- How different is that from Shiller CAPE right now and through history?
- Which constituents drive the gap?

## 3. Goals and Non-Goals

Goals:

- Produce a reliable CC CAPE time series with transparent methodology.
- Track CAPE Spread (CC CAPE minus Shiller CAPE) with weekly refresh in free-data mode (daily is a future enhancement with stronger data feeds).
- Provide decomposition by company and sector contributions.
- Expose data through dashboard + API.

Non-goals (MVP):

- Intraday valuation updates.
- Multi-asset support outside equities.
- Automated portfolio trading logic.

## 4. Users and Jobs-to-be-Done

Primary users:

- CIOs, PMs, and quant researchers.
- Advisory/research teams publishing valuation commentary.

Jobs-to-be-done:

- Check current valuation regime for an index.
- Compare CC CAPE vs Shiller CAPE before allocation decisions.
- Generate reproducible charts/tables for investment memos.

## 5. Product Scope (MVP)

MVP includes:

- Single index support: S&P 500.
- Weekly CC CAPE and CAPE Spread updates (configurable cadence).
- 10-year history backfill (monthly series for current constituents).
- Dashboard with:
  - Latest values and percentile bands.
  - History chart (CC CAPE, Shiller CAPE, Spread).
  - Top positive/negative spread contributors.
- CSV export and read-only API.

## 6. Methodology Requirements

The implementation must be auditable and versioned.

Definitions:

- Universe U_t: current constituents at date t.
- Market-cap weight w_i,t: MC_i,t / sum(MC_j,t for j in U_t).
- Real earnings: nominal earnings adjusted to current dollars using CPI.
- Company CAPE_i,t: price (or market cap) divided by 10-year average real earnings.

Aggregate metrics:

- CC_CAPE_t = sum(w_i,t * CAPE_i,t for i in U_t).
- CAPE_Spread_t = CC_CAPE_t - Shiller_CAPE_t.

Methodology policy:

- Use annual-ish EPS points (10-K / FY) from SEC XBRL company facts in free-data mode.
- Use daily prices for numerator; use a market-cap proxy (price * shares) when available.
- Freeze methodology versions (v1, v2, etc.) and store lineage metadata.

## 7. Data Requirements

Required data domains:

- Index constituents:
  - Current membership (and optional history for QA).
- Market data:
  - Daily close price, shares outstanding, market cap.
- Fundamentals:
  - Free-data mode: annual EPS from SEC XBRL company facts (10-K / FY).
  - Scale-up mode (future): vendor fundamentals (quarterly/TTM earnings, standardized definitions, restatement flags).
- Inflation:
  - CPI series for real earnings conversion.
- Benchmark:
  - Shiller CAPE series for spread comparison.

Data quality checks:

- Missing value thresholds.
- Stale records detection.
- Restatement handling (recompute affected dates).
- Corporate actions consistency checks.

## 8. Functional Requirements

FR-001 Data ingestion:

- Scheduled ETL pulls constituents, prices, fundamentals, CPI, benchmark CAPE (weekly by default in free-data mode).

FR-002 Calculation engine:

- Compute company-level CAPE and index-level CC CAPE on schedule (weekly by default in free-data mode).

FR-003 Spread analytics:

- Compute CAPE Spread and rolling z-score/percentiles.

FR-004 Decomposition:

- Attribute spread to constituent and sector buckets.

FR-005 Visualization:

- Interactive dashboard with filters and export.

FR-006 API:

- Read-only endpoints for latest values and historical series.

FR-007 Auditability:

- For each output value, store input version IDs and run timestamp.

## 9. Non-Functional Requirements

- Accuracy: reproduce a fixed historical sample within agreed tolerance.
- Reliability: scheduled runs complete successfully and publish updated metrics weekly in free-data mode; failures are surfaced with actionable status/warnings.
- Latency: weekly update completes within 60 minutes of scheduled start (typical runs should be much faster).
- Security: role-based access for admin functions.
- Observability: run logs, metrics, and alerting for data and compute failures.

## 10. Proposed Technical Architecture

Current MVP implementation (free-data mode):

- Orchestration: weekly scheduler (Docker service).
- Data processing: Python scripts.
- Storage: SQLite (`internal_jira.db`, `free_data.db`).
- API + UI: FastAPI + Jinja templates.
- Ops: health page + KPI baseline markdown report + tracker comments.

Scale-up path (future):

- Orchestration: cron/Airflow.
- Storage: Postgres/warehouse.
- Monitoring: job health checks + alert hooks.

## 11. Success Metrics

Product KPIs:

- Weekly freshness SLA met (free-data mode).
- Data completeness rate.
- API/dashboard adoption (weekly active users).
- Time to publish valuation commentary reduced.

Quant KPIs:

- Stability of spread calculations after restatements.
- Reproducibility of historical recomputations.

## 12. Risks and Mitigations

Risk: licensed constituent/fundamental data constraints.  
Mitigation: finalize data vendor and legal permissions before build phase.

Risk: methodology drift from paper intent.  
Mitigation: methodology spec + sign-off by quant owner.

Risk: negative or highly volatile earnings causing unstable constituent CAPE.  
Mitigation: explicit policy for treatment and sensitivity reporting.

Risk: survivorship and corporate action artifacts.  
Mitigation: robust entity mapping and reconciliation checks.

## 13. Open Decisions

- Exact earnings field for denominator (reported vs operating).
- Treatment of negative earnings cases.
- Refresh cadence for the intended users (weekly vs daily) given data source constraints.
- First target deployment: internal-only or client-facing.
