import json
from pathlib import Path

import pandas as pd

from performance_ledger import FAVORITE_TRACKING_START_DATE, update_performance_ledger


ROOT = Path(__file__).resolve().parents[1]
BOARD = (ROOT / "site" / "board.html").read_text(encoding="utf-8")


def test_performance_summary_is_derived_from_the_favorite_ledger(tmp_path):
    rows = []
    for index, grade in enumerate(["W", "W", "L", "Push"]):
        rows.append({
            "ledger_id": str(index),
            "favorite_qualified": "yes",
            "grade": grade,
        })
    pd.DataFrame(rows).to_csv(tmp_path / "performance_ledger.csv", index=False)

    update_performance_ledger(tmp_path, frozen_sources=[], attach_results=False)

    payload = json.loads((tmp_path / "favorite_performance.json").read_text(encoding="utf-8"))
    assert payload["wins"] == 2
    assert payload["losses"] == 1
    assert payload["pushes"] == 1
    assert payload["win_rate_excluding_pushes"] == 0.666667
    assert payload["tracking_start_date"] == FAVORITE_TRACKING_START_DATE


def test_tracker_uses_one_shared_payload_for_desktop_mobile_and_guide():
    assert BOARD.count('data-favorite-performance hidden') == 2
    assert 'favorite-performance-desktop' in BOARD
    assert 'favorite-performance-mobile' in BOARD
    assert "const gradedDecisions=wins+losses;" in BOARD
    assert "winPercentage:gradedDecisions?wins/gradedDecisions*100:0" in BOARD
    assert "FAVORITE_PERFORMANCE_BOOTSTRAP = Object.freeze({wins:7,losses:3,pushes:0,tracking_start_date:'2026-09-06'})" in BOARD
    assert "let favoritePerformance=normalizeFavoritePerformance(FAVORITE_PERFORMANCE_BOOTSTRAP);" in BOARD
    assert "renderFavoritePerformance();" in BOARD
    assert "favoritePerformanceFooter()" in BOARD
    assert 'class="favorite-guide-record"' in BOARD
    assert "footer.textContent=text?`Record: ${text.record} · ${text.rate} · ${text.since}`:'';" in BOARD
    assert "10 Graded" not in BOARD


def test_aggregate_endpoint_is_board_protected_without_exposing_admin_rows():
    production = (ROOT / "deploy" / "nginx-redfox.production.conf").read_text(encoding="utf-8")
    site = (ROOT / "redfox-nginx-site.conf").read_text(encoding="utf-8")
    assert "location = /data/favorite_performance.json" in production
    assert "auth_request /_internal/board-access;" in production
    assert "favorite_performance\\.json" in site
    assert "FAVORITE_PERFORMANCE_URL = '/data/favorite_performance.json'" in BOARD
