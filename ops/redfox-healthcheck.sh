#!/usr/bin/env bash
# Lightweight, read-only production health detection.  Designed for cron.
set -euo pipefail

ROOT="${REDFOX_ROOT:-/opt/red-fox-market-dynamics}"
DATA="$ROOT/data"
STATE_DIR="${REDFOX_HEALTH_STATE_DIR:-/var/lib/redfox-health}"
MAX_BOARD_AGE_MINUTES="${REDFOX_MAX_BOARD_AGE_MINUTES:-20}"
MAX_LIVE_RECENT_AGE_MINUTES="${REDFOX_MAX_LIVE_RECENT_AGE_MINUTES:-5}"
MAX_PROP_PAYLOAD_AGE_MINUTES="${REDFOX_MAX_PROP_PAYLOAD_AGE_MINUTES:-20}"
MAX_DISK_PERCENT="${REDFOX_MAX_DISK_PERCENT:-85}"
MAX_UPSTREAM_ERRORS="${REDFOX_MAX_UPSTREAM_ERRORS:-1}"

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
prop_age=$(age_minutes "$DATA/prop_projections.json")
(( board_age <= MAX_BOARD_AGE_MINUTES )) || issues+=("board freshness ${board_age}m")
(( snapshot_age <= MAX_BOARD_AGE_MINUTES )) || issues+=("snapshot freshness ${snapshot_age}m")
(( recent_age <= MAX_LIVE_RECENT_AGE_MINUTES )) || issues+=("live/recent freshness ${recent_age}m")
(( prop_age <= MAX_PROP_PAYLOAD_AGE_MINUTES )) || issues+=("prop payload freshness ${prop_age}m")

if ! "$ROOT/.venv/bin/python" "$ROOT/coverage_monitor.py" --data-dir "$DATA" > "$STATE_DIR/coverage.json" 2>&1; then
  issues+=("publication coverage alert (see $STATE_DIR/coverage.json)")
fi
if ! "$ROOT/.venv/bin/python" "$ROOT/live_score_monitor.py" --data-dir "$DATA" > "$STATE_DIR/live-score-coverage.json" 2>&1; then
  issues+=("live score coverage alert (see $STATE_DIR/live-score-coverage.json)")
fi
if ! "$ROOT/.venv/bin/python" "$ROOT/prop_health_monitor.py" \
    --path "$DATA/prop_projections.json" \
    --max-payload-age-minutes "$MAX_PROP_PAYLOAD_AGE_MINUTES" \
    > "$STATE_DIR/prop-health.json" 2>&1; then
  issues+=("prop projection health alert (see $STATE_DIR/prop-health.json)")
fi

disk_percent=$(df -P "$ROOT" | awk 'NR==2 {gsub(/%/, "", $5); print $5}')
(( disk_percent < MAX_DISK_PERCENT )) || issues+=("disk ${disk_percent}%")

# A completed run is the runner's authoritative success marker.  Count only
# recent, explicit scrape/refresh errors; historical log lines are irrelevant.
recent_log=$(tail -n 2000 /var/log/redfox_update.log 2>/dev/null || true)
recent_prop_log=$(tail -n 1000 /var/log/redfox-props.log 2>/dev/null || true)
last_run_marker=$(grep 'RUN END' <<<"$recent_log" | tail -n 1 || true)
if ! grep -q 'RUN END status=OK' <<<"$last_run_marker"; then
  issues+=("no completed full pipeline marker")
fi
if ! grep -q 'prop collection DONE' <<<"$recent_prop_log"; then
  issues+=("no completed prop collection marker")
fi
prop_failures=$(awk '
  /prop collection DONE/ { failures=0; next }
  /prop collection UNAVAILABLE/ { failures++; next }
  END { print failures+0 }
' <<<"$recent_prop_log")
(( prop_failures == 0 )) || issues+=("consecutive prop collection failures ${prop_failures}")

expected_revision_file="${REDFOX_EXPECTED_REVISION_FILE:-/etc/redfox-expected-revision}"
current_revision=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)
expected_revision=$(tr -d '[:space:]' < "$expected_revision_file" 2>/dev/null || true)
if [[ -z "$expected_revision" ]]; then
  issues+=("expected production revision is not recorded")
elif [[ "$current_revision" != "$expected_revision" ]]; then
  issues+=("deployed revision mismatch current=${current_revision:-unknown} expected=$expected_revision")
fi
# Count current consecutive failures for publishing and for each sport capture.
# A validation rejection exits nonzero, so a provider contract change can no
# longer look healthy while every market is being quarantined.
upstream_errors=$(awk '
  /refresh anomaly board DONE/ { publish_failures=0; next }
  /refresh anomaly board ERROR/ { publish_failures++; next }
  /snapshot DONE --sport/ {
    sport=$0; sub(/^.*--sport /, "", sport); sub(/ .*/, "", sport)
    snapshot_failures[sport]=0; next
  }
  /snapshot ERROR --sport/ {
    sport=$0; sub(/^.*--sport /, "", sport); sub(/ .*/, "", sport)
    snapshot_failures[sport]++; next
  }
  END {
    total=publish_failures+0
    for (sport in snapshot_failures) total += snapshot_failures[sport]
    print total
  }
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
message="OK: revision=${current_revision:0:12} board=${board_age}m snapshots=${snapshot_age}m live_recent=${recent_age}m props=${prop_age}m disk=${disk_percent}% upstream_errors=${upstream_errors} prop_failures=${prop_failures} live_scores=${score_coverage}"
printf '%s %s\n' "$(date --iso-8601=seconds)" "$message" > "$status_file"
logger -p daemon.info -t redfox-health -- "$message"
echo "$message"
