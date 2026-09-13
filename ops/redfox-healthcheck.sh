#!/usr/bin/env bash
# Lightweight, read-only production health detection.  Designed for cron.
set -euo pipefail

ROOT="${REDFOX_ROOT:-/opt/red-fox-market-dynamics}"
DATA="$ROOT/data"
STATE_DIR="${REDFOX_HEALTH_STATE_DIR:-/var/lib/redfox-health}"
MAX_BOARD_AGE_MINUTES="${REDFOX_MAX_BOARD_AGE_MINUTES:-10}"
MAX_LIVE_RECENT_AGE_MINUTES="${REDFOX_MAX_LIVE_RECENT_AGE_MINUTES:-5}"
MAX_DISK_PERCENT="${REDFOX_MAX_DISK_PERCENT:-85}"
MAX_UPSTREAM_ERRORS="${REDFOX_MAX_UPSTREAM_ERRORS:-5}"

mkdir -p "$STATE_DIR"
issues=()
now_epoch=$(date +%s)

age_minutes() {
  local file=$1
  [[ -f "$file" ]] || { echo 999999; return; }
  echo $(( (now_epoch - $(stat -c %Y "$file")) / 60 ))
}

json_age_minutes() {
  local file=$1 key=$2
  [[ -f "$file" ]] || { echo 999999; return; }
  "$ROOT/.venv/bin/python" -c '
import json, sys
from datetime import datetime, timezone
try:
    value = json.load(open(sys.argv[1], encoding="utf-8")).get(sys.argv[2])
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    print(max(0, int((datetime.now(timezone.utc) - parsed).total_seconds() // 60)))
except Exception:
    print(999999)
' "$file" "$key"
}

board_age=$(age_minutes "$DATA/freshness.json")
board_source_age=$(json_age_minutes "$DATA/freshness.json" board_oldest_ts)
snapshot_age=$(age_minutes "$DATA/snapshots.csv")
recent_age=$(age_minutes "$DATA/live_recent.csv")
(( board_age <= MAX_BOARD_AGE_MINUTES )) || issues+=("board freshness ${board_age}m")
(( board_source_age <= MAX_BOARD_AGE_MINUTES )) || issues+=("board source freshness ${board_source_age}m")
(( snapshot_age <= MAX_BOARD_AGE_MINUTES )) || issues+=("snapshot freshness ${snapshot_age}m")
(( recent_age <= MAX_LIVE_RECENT_AGE_MINUTES )) || issues+=("live/recent freshness ${recent_age}m")

if ! "$ROOT/.venv/bin/python" "$ROOT/coverage_monitor.py" --data-dir "$DATA" > "$STATE_DIR/coverage.json" 2>&1; then
  issues+=("publication coverage alert (see $STATE_DIR/coverage.json)")
fi
if ! "$ROOT/.venv/bin/python" "$ROOT/live_score_monitor.py" --data-dir "$DATA" > "$STATE_DIR/live-score-coverage.json" 2>&1; then
  issues+=("live score coverage alert (see $STATE_DIR/live-score-coverage.json)")
fi

disk_percent=$(df -P "$ROOT" | awk 'NR==2 {gsub(/%/, "", $5); print $5}')
(( disk_percent < MAX_DISK_PERCENT )) || issues+=("disk ${disk_percent}%")

# A completed run is the runner's authoritative success marker.  Count only
# recent, explicit scrape/refresh errors; historical log lines are irrelevant.
recent_log=$(tail -n 2000 /var/log/redfox_update.log 2>/dev/null || true)
if ! grep -q 'RUN END' <<<"$recent_log"; then
  issues+=("no completed full pipeline marker")
fi
# Count consecutive failed publish outcomes, not old failures that preceded a
# successful recovery.  This keeps the status actionable after the board heals.
upstream_errors=$(awk '
  /refresh anomaly board DONE/ { failures=0; next }
  /refresh anomaly board ERROR/ { failures++ }
  END { print failures+0 }
' <<<"$recent_log")
(( upstream_errors < MAX_UPSTREAM_ERRORS )) || issues+=("repeated upstream failures ${upstream_errors}")

status_file="$STATE_DIR/status"
if ((${#issues[@]})); then
  message="ALERT: ${issues[*]}"
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$message" > "$status_file"
  logger -p daemon.warning -t redfox-health -- "$message"
  echo "$message" >&2
  exit 1
fi

score_coverage=$(tr -d '\n' < "$STATE_DIR/live-score-coverage.json" 2>/dev/null || echo unavailable)
message="OK: board=${board_age}m board_source=${board_source_age}m snapshots=${snapshot_age}m live_recent=${recent_age}m disk=${disk_percent}% upstream_errors=${upstream_errors} live_scores=${score_coverage}"
printf '%s %s\n' "$(date --iso-8601=seconds)" "$message" > "$status_file"
logger -p daemon.info -t redfox-health -- "$message"
echo "$message"
