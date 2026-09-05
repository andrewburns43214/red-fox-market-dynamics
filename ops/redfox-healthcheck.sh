#!/usr/bin/env bash
# Lightweight, read-only production health detection.  Designed for cron.
set -euo pipefail

ROOT="${REDFOX_ROOT:-/opt/red-fox-market-dynamics}"
DATA="$ROOT/data"
STATE_DIR="${REDFOX_HEALTH_STATE_DIR:-/var/lib/redfox-health}"
MAX_BOARD_AGE_MINUTES="${REDFOX_MAX_BOARD_AGE_MINUTES:-25}"
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

board_age=$(age_minutes "$DATA/freshness.json")
snapshot_age=$(age_minutes "$DATA/snapshots.csv")
recent_age=$(age_minutes "$DATA/live_recent.csv")
(( board_age <= MAX_BOARD_AGE_MINUTES )) || issues+=("board freshness ${board_age}m")
(( snapshot_age <= MAX_BOARD_AGE_MINUTES )) || issues+=("snapshot freshness ${snapshot_age}m")
(( recent_age <= MAX_LIVE_RECENT_AGE_MINUTES )) || issues+=("live/recent freshness ${recent_age}m")

disk_percent=$(df -P "$ROOT" | awk 'NR==2 {gsub(/%/, "", $5); print $5}')
(( disk_percent < MAX_DISK_PERCENT )) || issues+=("disk ${disk_percent}%")

# A completed run is the runner's authoritative success marker.  Count only
# recent, explicit scrape/refresh errors; historical log lines are irrelevant.
recent_log=$(tail -n 2000 /var/log/redfox_update.log 2>/dev/null || true)
if ! grep -q 'RUN END' <<<"$recent_log"; then
  issues+=("no completed full pipeline marker")
fi
upstream_errors=$(grep -Eic 'snapshot ERROR|refresh anomaly board ERROR|TimeoutException|SessionNotCreatedException' <<<"$recent_log" || true)
(( upstream_errors < MAX_UPSTREAM_ERRORS )) || issues+=("repeated upstream failures ${upstream_errors}")

status_file="$STATE_DIR/status"
if ((${#issues[@]})); then
  message="ALERT: ${issues[*]}"
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$message" > "$status_file"
  logger -p daemon.warning -t redfox-health -- "$message"
  echo "$message" >&2
  exit 1
fi

message="OK: board=${board_age}m snapshots=${snapshot_age}m live_recent=${recent_age}m disk=${disk_percent}% upstream_errors=${upstream_errors}"
printf '%s %s\n' "$(date --iso-8601=seconds)" "$message" > "$status_file"
logger -p daemon.info -t redfox-health -- "$message"
echo "$message"
