# Roadmap: Current Constituents CAPE Tracker

Version: 0.1  
Date: 2026-02-13  
Horizon: 16 weeks to MVP + ongoing enhancements

## 1. Delivery Principles

- Build auditable calculation first, then UI polish.
- De-risk data licensing and methodology choices early.
- Gate each phase with objective exit criteria.

## 2. Phase Plan

## Phase 0 (Weeks 1-2): Discovery and Specification

Objectives:

- Lock methodology assumptions and acceptance tolerances.
- Confirm data vendors, access, and legal use rights.

Deliverables:

- Signed methodology spec (formulas, edge-case policy, versioning).
- Data source contract table (owner, fields, refresh schedule).
- Technical design doc (pipeline, storage, API, dashboard).

Exit criteria:

- Quant lead and product sign-off.
- No unresolved blocker on required data feeds.

## Phase 1 (Weeks 3-5): Data Foundation

Objectives:

- Stand up ingestion pipelines for all required datasets.
- Implement data quality checks and lineage capture.

Deliverables:

- ETL jobs for constituents, prices/market cap, fundamentals, CPI, Shiller CAPE.
- Raw and curated database schemas.
- Data quality dashboard and alerting.

Exit criteria:

- 30-day dry run with >=99% successful pipeline runs.
- Missing/stale data checks functioning with actionable alerts.

## Phase 2 (Weeks 6-8): Calculation Engine

Objectives:

- Implement company CAPE, CC CAPE, and CAPE Spread computation.
- Add decomposition logic by constituent and sector.

Deliverables:

- Versioned compute module with deterministic outputs.
- Historical backfill for at least 10 years.
- Unit tests and reconciliation report.

Exit criteria:

- Reproducible reruns from the same inputs.
- Validation package approved by quant owner.

## Phase 3 (Weeks 9-11): API and Dashboard MVP

Objectives:

- Deliver first usable interface and read-only API.

Deliverables:

- API endpoints:
  - latest metrics
  - historical series
  - decomposition snapshot
- Dashboard pages:
  - overview (latest + percentile)
  - time series (CC CAPE vs Shiller CAPE vs Spread)
  - contributors (top names/sectors)
- CSV export.

Exit criteria:

- Internal users can retrieve and visualize all MVP outputs without manual analyst support.

## Phase 4 (Weeks 12-14): Hardening and Pilot

Objectives:

- Improve operational reliability and user readiness.

Deliverables:

- Performance tuning, runbook, and failure recovery procedures.
- Access controls and audit logs for data/API use.
- Pilot with selected users and structured feedback cycle.

Exit criteria:

- Two consecutive weeks with no P1 data incidents.
- Pilot feedback issues triaged and MVP blockers closed.

## Phase 5 (Weeks 15-16): Launch

Objectives:

- Move from pilot to production usage.

Deliverables:

- Production release checklist complete.
- Onboarding docs and short user guide.
- KPI baseline report (freshness, reliability, adoption).

Exit criteria:

- Product accepted by sponsor and transitioned to BAU support.

## 3. Milestones and Owners

M1 End Week 2:

- Methodology and data contracts locked.
- Owner: Product + Quant Lead.

M2 End Week 5:

- Production-like data pipelines stable.
- Owner: Data Engineering.

M3 End Week 8:

- Calculation engine validated.
- Owner: Quant Engineering.

M4 End Week 11:

- API + dashboard MVP demo complete.
- Owner: Backend + Frontend Engineering.

M5 End Week 16:

- Launch complete and operating model active.
- Owner: Product + Platform.

## 4. Backlog After MVP

- Multi-index support (Nasdaq 100, Russell 1000, MSCI variants).
- Scenario layer (rate/inflation shocks vs valuation states).
- Alert subscriptions (threshold and percentile triggers).
- Research notebook integration.
- External client portal and report automation.

## 5. Key Risks and Dependencies

Dependency: licensed historical constituent and fundamentals coverage quality.  
Action: complete legal/data validation in Phase 0.

Dependency: methodology decisions on negative earnings and restatements.  
Action: define explicit policy before Phase 2 code freeze.

Risk: late UI feedback causing rework.  
Action: run clickable mock walkthrough during Week 8.

Risk: production incidents during launch period.  
Action: establish runbook, alert routing, and on-call owner in Phase 4.

## 6. Acceptance Checklist for MVP

- Daily CC CAPE and Spread generated automatically.
- Historical charting available with export.
- Methodology and lineage visible for every published value.
- Reliability and freshness KPIs tracked and reported weekly.
