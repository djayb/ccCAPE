# ccCAPE

Current Constituents CAPE planning and execution workspace.

This repository includes:

- Product sheet: `PRODUCT_SHEET.md`
- Delivery roadmap: `ROADMAP.md`
- Runbook: `docs/RUNBOOK.md`
- Architecture: `docs/ARCHITECTURE.md`
- Data model: `docs/DATA_MODEL.md`
- Methodology: `docs/METHODOLOGY.md`
- Data sources: `docs/DATA_SOURCES.md`
- Onboarding: `docs/ONBOARDING.md`
- Release checklist: `docs/RELEASE_CHECKLIST.md`
- Pilot plan: `docs/PILOT_PLAN.md`
- Internal Jira-like tracker CLI: `internal_jira.py`
- Internal web tracker UI with auth/roles: `web_app.py`

## 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Initialize and Seed Tracker

```bash
python3 internal_jira.py init
python3 internal_jira.py seed-cc-cape
```

Default admin user:

- Username: `admin`
- Password: `admin123`

Change password immediately:

```bash
python3 internal_jira.py user-update --username admin --password "your-new-password"
```

## 3. Run the Web UI

```bash
TRACKER_SESSION_SECRET="replace-with-long-random-secret" \
uvicorn web_app:app --reload --host 0.0.0.0 --port 8000
```

Open:

- [http://127.0.0.1:8000](http://127.0.0.1:8000)
- [http://127.0.0.1:8000/metrics/cc-cape](http://127.0.0.1:8000/metrics/cc-cape)
- [http://127.0.0.1:8000/metrics/health](http://127.0.0.1:8000/metrics/health)
- [http://127.0.0.1:8000/metrics/cc-cape/contributors](http://127.0.0.1:8000/metrics/cc-cape/contributors)

## 4. Run With Docker

1. Set secure values in `docker-compose.yml`:

- `TRACKER_SESSION_SECRET`
- `TRACKER_ADMIN_PASSWORD`
- `SEC_USER_AGENT` (real contact string for SEC fair-access compliance)

2. Build and run:

```bash
docker compose up --build -d
```

3. Open:

- [http://127.0.0.1:8000](http://127.0.0.1:8000)

Data persistence:

- SQLite DB is mounted at `./data/internal_jira.db`.
- Export docs are mounted at `./docs/`.

Stop service:

```bash
docker compose down
```

## 5. API Usage

The API reuses session authentication from `/login`.

Example:

```bash
curl -s -c /tmp/cccape.cookies -X POST http://127.0.0.1:8000/login \
  -d "username=admin" -d "password=admin123" >/dev/null

curl -s -b /tmp/cccape.cookies "http://127.0.0.1:8000/api/board?project=CAPE"
curl -s -b /tmp/cccape.cookies "http://127.0.0.1:8000/api/issues?project=CAPE&status=in_progress"
curl -s -b /tmp/cccape.cookies "http://127.0.0.1:8000/api/issues/CAPE-1"
```

Available endpoints:

- `GET /api/board?project=CAPE`
- `GET /api/issues?project=CAPE&status=in_progress`
- `GET /api/issues/{issue_key}`
- `GET /api/metrics/cc-cape/latest`
- `GET /api/metrics/cc-cape/runs?limit=N`
- `GET /api/metrics/cc-cape/series/monthly?limit=N`
- `GET /api/metrics/cc-cape/constituents?run_id=ID&limit=N&sort=weight|contribution|cape|symbol`
- `GET /api/metrics/cc-cape/sectors?run_id=ID`

## 6. CLI Usage (Core)

```bash
python3 internal_jira.py project-list
python3 internal_jira.py epic-list --project CAPE
python3 internal_jira.py issue-list --project CAPE
python3 internal_jira.py issue-show --key CAPE-1
python3 internal_jira.py issue-move --key CAPE-1 --status in_progress --actor jean
python3 internal_jira.py comment-add --key CAPE-1 --author jean --body "Working session started"
python3 internal_jira.py export-markdown --project CAPE --out docs/INTERNAL_TRACKER_SNAPSHOT.md
```

## 7. Roles

Supported roles:

- `admin`: full access, including user management.
- `editor`: can update issues, add comments/dependencies.
- `viewer`: read-only board and issue detail.

Admin commands:

```bash
python3 internal_jira.py user-create --username pm --password "secret" --role editor
python3 internal_jira.py user-list
python3 internal_jira.py user-update --username pm --role viewer
python3 internal_jira.py user-update --username pm --active no
```

## 8. Start of Phase 0 Execution

Phase 0 execution is tracked in:

- issue updates/comments in the tracker (`CAPE-1`, `CAPE-2`, `CAPE-3`)
- `docs/PHASE0_EXECUTION_LOG.md`

No-paid-data strategy notes:

- `docs/FREE_DATA_ALTERNATIVES.md`

## 9. Free Data Ingestion Pipeline

Run end-to-end free/public ingestion:

```bash
python3 scripts/free_data_pipeline.py \
  --facts-limit 25 \
  --prices-symbol-limit 100 \
  --update-tracker
```

What it ingests:

- Wikipedia S&P 500 constituents (including CIKs)
- SEC company facts by CIK
- FRED CPI CSV (`CPIAUCSL`)
- Shiller CAPE history (Multpl Shiller PE table)
- Stooq daily price history

Outputs:

- Free-data DB: `data/free_data.db`
- Tracker comments/status updates in `data/internal_jira.db`

Run from Docker container:

```bash
docker compose exec cccape \
  python3 /app/scripts/free_data_pipeline.py \
  --facts-limit 25 \
  --prices-symbol-limit 100 \
  --update-tracker
```

## 10. Free-Data CC CAPE Calculation

Run CC CAPE calculation on ingested free data:

```bash
python3 scripts/calc_cc_cape_free.py \
  --min-eps-points 8 \
  --lookback-years 10 \
  --update-tracker
```

Outputs:

- Persisted run tables in `data/free_data.db`:
  - `cc_cape_runs`
  - `cc_cape_constituent_metrics`
- Markdown summary:
  - `docs/CC_CAPE_FREE_RUN.md`
- Tracker updates on:
  - `CAPE-8`, `CAPE-9`, and a dedicated free-calc issue under Phase 2.

Optional benchmark spread:

By default, the calculator uses the latest ingested Shiller CAPE on/before the latest price date.
You can override with an explicit value:

```bash
python3 scripts/calc_cc_cape_free.py --shiller-cape 31.5
```

## 11. Weekly Scheduler (Docker Service)

The compose stack now includes `cccape-weekly`, which runs:

1. `scripts/free_data_pipeline.py`
2. `scripts/calc_cc_cape_free.py`

Default schedule:

- Weekly on Sunday at 09:00 (`WEEKLY_TIMEZONE=America/New_York`)
- Startup run disabled (`WEEKLY_RUN_ON_STARTUP=false`)

Key scheduler settings in `docker-compose.yml`:

- `WEEKLY_RUN_DAY` (`MON`..`SUN`)
- `WEEKLY_RUN_HOUR` / `WEEKLY_RUN_MINUTE`
- `WEEKLY_FACTS_LIMIT`
- `WEEKLY_PRICES_SYMBOL_LIMIT`
- `WEEKLY_MIN_EPS_POINTS`
- `WEEKLY_UPDATE_TRACKER`

Restart scheduler after changing schedule/env:

```bash
docker compose up --build -d
```

## 12. Monthly Series Backfill (Free-Data Proxy)

Backfill a monthly CC CAPE series for "current constituents" (fixed to the latest constituents snapshot):

```bash
python3 scripts/backfill_cc_cape_series_free.py \
  --series-years 10 \
  --lookback-years 10 \
  --min-eps-points 8 \
  --update-tracker
```

This writes into `data/free_data.db`:

- `cc_cape_series_monthly`

And enables:

- `GET /api/metrics/cc-cape/series/monthly`
- CSV export: `GET /metrics/cc-cape/export/series_monthly.csv`

## 13. KPI Baseline Report

Generate a lightweight ops/adoption baseline report:

```bash
python3 scripts/generate_kpi_report.py --out docs/KPI_BASELINE.md --update-tracker
```

This is also run by the weekly scheduler by default (`WEEKLY_KPI_ENABLED=true`).
