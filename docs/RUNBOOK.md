# ccCAPE Runbook (Internal)

Date: 2026-02-13

This runbook covers operating the ccCAPE stack (tracker UI + free-data ingestion + CC CAPE calculation).

## Services and Data Stores

- Web tracker UI (FastAPI): `http://127.0.0.1:8000`
- Tracker DB (SQLite): `data/internal_jira.db`
- Free-data DB (SQLite): `data/free_data.db`
- Weekly scheduler (Docker service): `cccape-weekly` (runs pipeline + calc)

## Configuration (Required)

Set these in `docker-compose.yml` (or env vars if running directly):

- `TRACKER_SESSION_SECRET`: long random string
- `TRACKER_ADMIN_PASSWORD`: change from default
- `SEC_USER_AGENT`: real contact string (SEC fair-access compliance)

## Standard Operations

### Start/Stop (Docker)

```bash
cd ccCAPE
docker compose up --build -d
docker compose down
```

### Manual Pipeline Run (No Paid Data)

```bash
python3 scripts/free_data_pipeline.py --update-tracker
```

### Manual Calculation Run

```bash
python3 scripts/calc_cc_cape_free.py --update-tracker
```

### Backfill Monthly Series (One-Time / On Demand)

```bash
python3 scripts/backfill_cc_cape_series_free.py --series-years 10 --update-tracker
```

## Health Checks

Use the Ops page:

- `http://127.0.0.1:8000/metrics/health`

Key signals:

- `latest_price_date` age (staleness)
- facts coverage (`facts_cik_count`) and warnings
- latest calculation coverage (`symbols_with_valid_cape / symbols_total`)

## Common Failures and Fixes

### SEC endpoints return 403

Cause: missing/invalid User-Agent or SEC rate limiting.

Actions:

1. Set `SEC_USER_AGENT` to a real contact string.
2. Increase `--request-delay` (for example `0.5` or `1.0`).
3. Reduce `--facts-limit` for incremental backfill.

### Stooq returns “Exceeded the daily hits limit”

Cause: Stooq has an anonymous daily request limit.

Actions:

1. Re-run later (typically the next day).
2. Reduce `--prices-symbol-limit`.
3. Accept cached prices for the calculation (staleness is surfaced in quality warnings).

### SQLite “database disk image is malformed”

Actions:

1. Stop containers (`docker compose down`) to avoid further writes.
2. Backup `data/free_data.db` and `data/internal_jira.db`.
3. Run an integrity check:
   - `sqlite3 data/free_data.db "PRAGMA integrity_check;"`
4. If corrupted:
   - restore from backup, or
   - attempt export/import via `.dump` to a new DB.

### Pipeline/Calc lock contention

Mitigations implemented:

- WAL mode + `busy_timeout` in ingestion + calc scripts.

If issues persist:

- avoid concurrent runs (don’t run manual jobs while scheduler is running)
- increase `busy_timeout` or reduce parallel reads

## Backup/Restore

Minimal backup set:

- `data/internal_jira.db`
- `data/free_data.db`

Suggested approach:

1. Stop services.
2. Copy `data/*.db` to a timestamped folder.
3. Restart services.

## Security Notes

- Change the default admin password immediately.
- Treat the tracker as internal-only; it is not hardened for internet exposure.

