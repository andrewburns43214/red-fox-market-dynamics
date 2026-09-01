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
"$PY" main.py report_maintenance >> "$LOG" 2>&1
echo "===== $(date) MAINT END =====" >> "$LOG"
