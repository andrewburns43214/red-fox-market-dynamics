#!/usr/bin/env bash
set -euo pipefail

cd /opt/red-fox-market-dynamics
LOG=/var/log/redfox-props.log
PY=/opt/red-fox-market-dynamics/.venv/bin/python
export PYTHONUNBUFFERED=1
export TMPDIR=/opt/red-fox-market-dynamics/tmp

# This job is launched by its own cron entry. The collector also holds its
# private process lock, so a slow run cannot overlap the following tick.
if [ -r /etc/redfox-propline.env ]; then
  set -a
  . /etc/redfox-propline.env
  set +a
fi

echo "--- $(date) prop collection START ---" >> "$LOG"
if timeout "${REDFOX_PROP_TIMEOUT_SECONDS:-120}" "$PY" prop_projection_service.py collect >> "$LOG" 2>&1; then
  echo "--- $(date) prop collection DONE ---" >> "$LOG"
else
  echo "--- $(date) prop collection UNAVAILABLE ---" >> "$LOG"
fi
