#!/usr/bin/env bash

set -euo pipefail

export TMPDIR=/opt/red-fox-market-dynamics/tmp
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"


cd /opt/red-fox-market-dynamics
LOG=/var/log/redfox_update.log
PY="/opt/red-fox-market-dynamics/.venv/bin/python"

echo "===== $(date) RUN START =====" >> "$LOG"

for SPORT in nfl nba mlb nhl ncaaf ncaab ufc; do
  echo "--- $(date) snapshot --sport $SPORT ---" >> "$LOG"
  "$PY" main.py snapshot --sport "$SPORT" >> "$LOG" 2>&1
  echo "--- $(date) snapshot DONE --sport $SPORT ---" >> "$LOG"
  sleep 10
done

echo "--- $(date) report ---" >> "$LOG"
"$PY" main.py report >> "$LOG" 2>&1

# publish
cp -f data/dashboard.html /var/www/redfox/index.html >> "$LOG" 2>&1

echo "===== $(date) RUN END =====" >> "$LOG"
