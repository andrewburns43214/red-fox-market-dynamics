#!/usr/bin/env bash

set -euo pipefail

exec 201>/var/lock/redfox_maintenance.lock
flock -n 201 || exit 0

export TMPDIR=/opt/red-fox-market-dynamics/tmp
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"

cd /opt/red-fox-market-dynamics
LOG=/var/log/redfox_maintenance.log
PY="/opt/red-fox-market-dynamics/.venv/bin/python"

echo "===== $(date) MAINT START =====" >> "$LOG"
# Historical finals are downstream work. Give live board publication CPU and
# disk priority whenever the two cron jobs overlap.
if timeout 600 ionice -c3 nice -n 15 "$PY" main.py report_maintenance >> "$LOG" 2>&1; then
  echo "===== $(date) MAINT END status=OK =====" >> "$LOG"
else
  status=$?
  echo "===== $(date) MAINT END status=FAILED exit=$status =====" >> "$LOG"
  exit "$status"
fi
