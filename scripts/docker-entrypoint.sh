#!/usr/bin/env sh
set -eu

mkdir -p /app/data /app/docs

DB_PATH="${TRACKER_DB:-/app/data/internal_jira.db}"
ADMIN_USER="${TRACKER_ADMIN_USER:-admin}"
ADMIN_PASSWORD="${TRACKER_ADMIN_PASSWORD:-admin123}"
SEED_FLAG="${TRACKER_SEED:-true}"
HOST="${TRACKER_HOST:-0.0.0.0}"
PORT="${TRACKER_PORT:-8000}"

python3 /app/internal_jira.py --db "$DB_PATH" init --admin-user "$ADMIN_USER" --admin-password "$ADMIN_PASSWORD"

if [ "$SEED_FLAG" = "true" ] || [ "$SEED_FLAG" = "1" ] || [ "$SEED_FLAG" = "yes" ]; then
  python3 /app/internal_jira.py --db "$DB_PATH" seed-cc-cape
fi

exec uvicorn web_app:app --host "$HOST" --port "$PORT"
