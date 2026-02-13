# ccCAPE Release Checklist (Internal)

Date: 2026-02-13

Use this checklist before declaring the tool “ready for internal production use”.

## Configuration

- `TRACKER_SESSION_SECRET` is set to a long random string
- Default admin password is changed (`TRACKER_ADMIN_PASSWORD`)
- `SEC_USER_AGENT` is a real contact string
- Docker volumes for `./data` and `./docs` are enabled

## Security

- Admin user is active and password rotated
- Non-admin users created for day-to-day access (viewer/editor)
- `/admin/audit` works and is restricted to admin

## Data + Compute

- Manual run succeeds:
  - `scripts/free_data_pipeline.py`
  - `scripts/calc_cc_cape_free.py`
- Metrics page loads:
  - `/metrics/cc-cape`
- Contributors export works:
  - `/metrics/cc-cape/export/constituents.csv`
  - `/metrics/cc-cape/export/sectors.csv`
- Monthly series backfill executed at least once (optional but recommended):
  - `scripts/backfill_cc_cape_series_free.py --series-years 10`

## Ops

- `/metrics/health` shows acceptable freshness and coverage
- Backups documented and tested:
  - `data/internal_jira.db`
  - `data/free_data.db`
- Scheduler configured to acceptable cadence and limits:
  - facts limit, request delay, symbol limits

## Known Limitations Logged

- Stooq daily hit limit can rate-limit price refresh
- Wikipedia constituents are not an official licensed feed
- Shiller CAPE benchmark is scraped from Multpl table (prototype)

## Rollback Plan

- Stop services: `docker compose down`
- Restore `data/*.db` from last known-good backup
- Restart: `docker compose up --build -d`

