#!/usr/bin/env bash
# Fast-forward, verify, refresh, and health-check the production checkout.
set -euo pipefail

ROOT="${REDFOX_ROOT:-/opt/red-fox-market-dynamics}"
BRANCH="${REDFOX_DEPLOY_BRANCH:-main}"
PY="$ROOT/.venv/bin/python"
EXPECTED_REVISION_FILE="${REDFOX_EXPECTED_REVISION_FILE:-/etc/redfox-expected-revision}"

exec 209>/var/lock/redfox_deploy.lock
flock -n 209 || { echo "another deployment is active" >&2; exit 1; }
# Do not change code while the board or prop writers are running.
exec 200>/var/lock/redfox_run.lock
flock 200
mkdir -p "$ROOT/data/prop_projection"
exec 202>"$ROOT/data/prop_projection/collector.lock"
flock 202

cd "$ROOT"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked production files are modified; refusing to overwrite them" >&2
  exit 1
fi
if [[ "$(git branch --show-current)" != "$BRANCH" ]]; then
  echo "production checkout is not on $BRANCH" >&2
  exit 1
fi

git fetch --prune origin "$BRANCH"
git merge --ff-only "origin/$BRANCH"
revision=$(git rev-parse HEAD)

"$PY" -m pytest -q
install -m 0644 deploy/redfox-board-locations.conf /etc/nginx/redfox-board-locations.conf
nginx -t
systemctl reload nginx
printf '%s\n' "$revision" > "$EXPECTED_REVISION_FILE"

# Force a current prop pull and rebuild every active board source before the
# deployment can be declared healthy.
flock -u 202
flock -u 200
REDFOX_PROP_FORCE=1 REDFOX_PROP_TIMEOUT_SECONDS="${REDFOX_PROP_TIMEOUT_SECONDS:-180}" \
  "$ROOT/ops/run_prop_collection.sh"
"$ROOT/run_all_sports.sh"
"$ROOT/ops/redfox-healthcheck.sh"

echo "deployed $revision"
