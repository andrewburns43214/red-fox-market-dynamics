#!/usr/bin/env bash
# Post-deploy Nginx security check — run after any nginx config change
# Usage: bash check_nginx.sh

BASE="https://www.redfoxmi.com"
PASS=0
FAIL=0

check() {
    local expect=$1 url=$2 label=$3
    code=$(curl -s -o /dev/null -w "%{http_code}" "$url")
    if [ "$code" = "$expect" ]; then
        echo "  OK  $code $label"
        ((PASS++))
    else
        echo "  FAIL $code (expected $expect) $label"
        ((FAIL++))
    fi
}

echo "=== Nginx Security Check ==="

# Allowlisted UI files — should return 200
check 200 "$BASE/data/dashboard.csv"          "dashboard.csv"
check 200 "$BASE/data/results_resolved.csv"   "results_resolved.csv"
check 200 "$BASE/data/signal_ledger.csv"      "signal_ledger.csv"
check 200 "$BASE/data/snapshots.csv"          "snapshots.csv"
check 200 "$BASE/data/freshness.json"        "freshness.json"
check 200 "$BASE/data/score_history.csv"     "score_history.csv"

# Blocked internal files — should return 403
check 403 "$BASE/data/l1_sharp.csv"           "l1_sharp.csv (internal)"
check 403 "$BASE/data/l2_consensus.csv"       "l2_consensus.csv (internal)"
check 403 "$BASE/data/l1_open_registry.csv"   "l1_open_registry.csv (internal)"
check 403 "$BASE/data/decision_snapshots.csv" "decision_snapshots.csv (internal)"
check 403 "$BASE/data/"                       "/data/ directory listing"

# Every admin export must fail closed without an authenticated admin token.
check 401 "$BASE/admin-data/performance_supported_sides.csv" "supported performance (admin only)"
check 401 "$BASE/admin-data/performance_favorites.csv"       "Favorite performance (admin only)"
check 401 "$BASE/admin-data/performance_combined.csv"        "combined performance (admin only)"
check 401 "$BASE/admin-data/favorite_candidate_audit.csv"    "Favorite candidate audit (admin only)"
check 401 "$BASE/admin-data/favorite_cohort_kpis.csv"         "Favorite cohort KPIs (admin only)"
check 401 "$BASE/admin-data/red_fox_favorite_shadow.csv"      "Favorite shadow captures (admin only)"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && echo "All clear." || echo "ACTION REQUIRED: fix nginx config."
exit $FAIL
