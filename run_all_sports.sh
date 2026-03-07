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
    # same-year range (e.g. Mar 26 – Nov 6)
    [ "$MMDD" -ge "$s" ] && [ "$MMDD" -le "$e" ]
  else
    # wraps around year (e.g. Sep 7 – Feb 25)
    [ "$MMDD" -ge "$s" ] || [ "$MMDD" -le "$e" ]
  fi
}

# NFL: Sep 7 – Feb 25
if in_season 907 225; then SPORTS="nfl $SPORTS"; fi
# NCAAF: Aug 24 – Feb 1
if in_season 824 201; then SPORTS="ncaaf $SPORTS"; fi
# MLB: Mar 26 – Nov 6
if in_season 326 1106; then SPORTS="mlb $SPORTS"; fi
# NBA: Oct 22 – Jul 5
if in_season 1022 705; then SPORTS="nba $SPORTS"; fi
# NHL: Oct 10 – Jul 5
if in_season 1010 705; then SPORTS="nhl $SPORTS"; fi
# NCAAB: Nov 5 – Apr 10
if in_season 1105 410; then SPORTS="ncaab $SPORTS"; fi

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


# Update dk_ts in freshness.json so dashboard shows fresh DK timestamp
"$PY" -c "
import json, os
from datetime import datetime, timezone
fp = 'data/freshness.json'
f = json.load(open(fp)) if os.path.exists(fp) else {}
f['dk_ts'] = datetime.now(timezone.utc).isoformat()
json.dump(f, open(fp, 'w'))
" >> "$LOG" 2>&1

echo "--- $(date) report ---" >> "$LOG"
"$PY" main.py report >> "$LOG" 2>&1

# publish (nginx serves directly from project dir)

echo "===== $(date) RUN END =====" >> "$LOG"
