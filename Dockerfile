FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRACKER_DB=/app/data/internal_jira.db

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app

RUN chmod +x /app/internal_jira.py \
    /app/scripts/docker-entrypoint.sh \
    /app/scripts/free_data_pipeline.py \
    /app/scripts/fetch_stooq_quotes.py \
    /app/scripts/calc_cc_cape_free.py \
    /app/scripts/backfill_cc_cape_series_free.py \
    /app/scripts/import_external_fundamentals_csv.py \
    /app/scripts/import_simfin_bulk.py \
    /app/scripts/manage_symbol_overrides.py \
    /app/scripts/generate_kpi_report.py \
    /app/scripts/weekly_scheduler.py

EXPOSE 8000

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
