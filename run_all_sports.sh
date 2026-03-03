#!/usr/bin/env bash

set -euo pipefail

# ---- prevent overlapping runs ----
exec 200>/var/lock/redfox_run.lock
flock -n 200 || exit 0
# ---------------------------------


export TMPDIR=/opt/red-fox-market-dynamics/tmp
export RF_DISABLE_BASELINE_LOG=1
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"


cd /opt/red-fox-market-dynamics
LOG=/var/log/redfox_update.log
PY="/opt/red-fox-market-dynamics/.venv/bin/python"

echo "===== $(date) RUN START =====" >> "$LOG"

# Auto-detect active sports by month (starts ~1 week before season)
MONTH=$(date +%-m)
SPORTS="nba nhl ncaab ufc"
# NFL: late Aug (8) through Feb — covers week-before + playoffs
if [ "$MONTH" -ge 8 ] || [ "$MONTH" -le 2 ]; then SPORTS="nfl $SPORTS"; fi
# NCAAF: late Aug (8) through Jan — covers week-before + bowls
if [ "$MONTH" -ge 8 ] || [ "$MONTH" -le 1 ]; then SPORTS="ncaaf $SPORTS"; fi
# MLB: Apr through Oct — skip spring training
if [ "$MONTH" -ge 4 ] && [ "$MONTH" -le 10 ]; then SPORTS="mlb $SPORTS"; fi

echo "--- active sports: $SPORTS ---" >> "$LOG"
for SPORT in $SPORTS; do
  echo "--- $(date) snapshot --sport $SPORT ---" >> "$LOG"
  if "$PY" main.py snapshot --sport "$SPORT" >> "$LOG" 2>&1;
  then
  echo "--- $(date) snapshot DONE --sport $SPORT ---" >> "$LOG"
  else
    echo "--- $(date) snapshot ERROR --sport $SPORT (continuing) ---" >> "$LOG"
  fi
  sleep 3
done

echo "--- $(date) report ---" >> "$LOG"
"$PY" main.py report >> "$LOG" 2>&1

# publish (nginx serves directly from project dir)

echo "===== $(date) RUN END =====" >> "$LOG"
