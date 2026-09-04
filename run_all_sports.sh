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

# Auto-detect active sports by month+day (skips preseason)
MONTH=$(date +%-m)
DAY=$(date +%-d)
MMDD="${MONTH}$(printf '%02d' $DAY)"  # e.g. "307" for Mar 7, "1122" for Nov 22

SPORTS="ufc"  # UFC always on

# Helper: check if today is within a date range (handles year-wrap)
# Usage: in_season START_MMDD END_MMDD
in_season() {
  local s=$1 e=$2
  if [ "$s" -le "$e" ]; then
    # same-year range (e.g. Mar 26 - Nov 6)
    [ "$MMDD" -ge "$s" ] && [ "$MMDD" -le "$e" ]
  else
    # wraps around year (e.g. Sep 7 - Feb 25)
    [ "$MMDD" -ge "$s" ] || [ "$MMDD" -le "$e" ]
  fi
}

# Start each sport a bit before opening day so upcoming slates are already flowing.
# Offseason runs stay safe because the scraper simply returns no events.
# NFL: Aug 1 - Feb 25
if in_season 801 225; then SPORTS="nfl $SPORTS"; fi
# NCAAF: Aug 1 - Feb 1
if in_season 801 201; then SPORTS="ncaaf $SPORTS"; fi
# MLB: Mar 1 - Nov 15
if in_season 301 1115; then SPORTS="mlb $SPORTS"; fi
# NBA: Oct 1 - Jul 15
if in_season 1001 715; then SPORTS="nba $SPORTS"; fi
# NHL: Sep 20 - Jul 15
if in_season 920 715; then SPORTS="nhl $SPORTS"; fi
# NCAAB: Oct 1 - Apr 15
if in_season 1001 415; then SPORTS="ncaab $SPORTS"; fi

echo "--- active sports: $SPORTS ---" >> "$LOG"
SNAPSHOT_TIMEOUT_SECONDS="${REDFOX_SNAPSHOT_TIMEOUT_SECONDS:-120}"
for SPORT in $SPORTS; do
  echo "--- $(date) snapshot --sport $SPORT ---" >> "$LOG"
  if timeout "$SNAPSHOT_TIMEOUT_SECONDS" "$PY" main.py snapshot --sport "$SPORT" >> "$LOG" 2>&1;
  then
  echo "--- $(date) snapshot DONE --sport $SPORT ---" >> "$LOG"
  else
    echo "--- $(date) snapshot ERROR --sport $SPORT (continuing) ---" >> "$LOG"
  fi
  sleep 3
done

echo "--- $(date) refresh anomaly board ---" >> "$LOG"
# Production reached the former 120-second watchdog while publishing a valid
# high-volume two-sided event set.  Keep the watchdog (and atomic publisher),
# but allow measured production headroom.  Operators may lower it explicitly.
REFRESH_TIMEOUT_SECONDS="${REDFOX_REFRESH_TIMEOUT_SECONDS:-300}"
if timeout "$REFRESH_TIMEOUT_SECONDS" "$PY" refresh_anomaly_board.py >> "$LOG" 2>&1; then
  # The public board consumes anomaly_board.csv directly. Only mark the engine
  # fresh after its complete live export is atomically available to Nginx.
  "$PY" -c "
import json, os
from datetime import datetime, timezone
fp = 'data/freshness.json'
f = json.load(open(fp)) if os.path.exists(fp) else {}
f['engine_ts'] = datetime.now(timezone.utc).isoformat()
json.dump(f, open(fp, 'w'))
" >> "$LOG" 2>&1
  echo "--- $(date) refresh anomaly board DONE ---" >> "$LOG"
else
  echo "--- $(date) refresh anomaly board ERROR ---" >> "$LOG"
fi

# publish (nginx serves directly from project dir)

echo "===== $(date) RUN END =====" >> "$LOG"
