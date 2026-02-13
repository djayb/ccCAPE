# ccCAPE Onboarding (Internal)

Date: 2026-02-13

## What This Tool Does

ccCAPE provides:

- A lightweight internal tracker (Jira-like) to manage delivery work.
- A free-data ingestion + calculation pipeline to compute a research-grade proxy for:
  - Current Constituents CAPE (CC CAPE)
  - Shiller CAPE (benchmark proxy)
  - CAPE Spread (CC CAPE - Shiller)
- Dashboards for metrics, contributors, and ops health.

## Roles and Access

Roles:

- `viewer`: read-only
- `editor`: can update issues, add comments and dependencies
- `admin`: user management + audit logs

Default admin is `admin/admin123` unless changed. Change it immediately.

## Getting Started (Docker)

1. Edit `docker-compose.yml`:
- `TRACKER_SESSION_SECRET`
- `TRACKER_ADMIN_PASSWORD`
- `SEC_USER_AGENT` (use a real contact string)

2. Start:

```bash
cd ccCAPE
docker compose up --build -d
```

3. Open:

- `http://127.0.0.1:8000` (board)
- `http://127.0.0.1:8000/metrics/cc-cape` (metrics)
- `http://127.0.0.1:8000/metrics/health` (ops)
- `http://127.0.0.1:8000/metrics/cc-cape/contributors` (decomposition)

## Typical Workflows

### Product / PM

- Use `/board` to track roadmap items and status.
- Use `/metrics/cc-cape` to review headline CC CAPE and the spread.
- Use `/metrics/cc-cape/contributors` to export CSV for deeper analysis.

### Analyst / Quant

- Run ingestion + calc locally for controlled experiments:

```bash
python3 scripts/free_data_pipeline.py --no-update-tracker
python3 scripts/calc_cc_cape_free.py --no-update-tracker
```

- Backfill monthly time series:

```bash
python3 scripts/backfill_cc_cape_series_free.py --series-years 10 --no-update-tracker
```

### Admin

- Manage users: `/admin/users`
- Review access logs: `/admin/audit`

## Interpreting Ops Health

- `/metrics/health` summarizes the latest pipeline, calc, and monthly series state.
- If the pipeline shows warnings:
  - price coverage may be low
  - prices may be stale
  - SEC facts coverage may be incomplete

## Data Caveats (Free-Data Proxy)

- Constituent list is Wikipedia (prototype), not a licensed S&P membership feed.
- Prices are Stooq; availability and rate limits can cause missing/stale data.
- Fundamentals come from SEC XBRL facts; coverage varies by company and tag.
- Shiller CAPE benchmark comes from Multpl table scrape (research/prototyping).

For details: `docs/FREE_DATA_ALTERNATIVES.md`

