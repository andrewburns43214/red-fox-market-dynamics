import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import build_live_recent as live
from live_score_monitor import evaluate


ROOT = Path(__file__).resolve().parents[1]
BOARD = (ROOT / "site" / "board.html").read_text(encoding="utf-8")


def test_provider_location_and_abbreviation_keys_cover_verified_cfb_mismatches():
    cases = [
        ("Western Kentucky @ Nevada", {"location": "Western Kentucky", "displayName": "Western Kentucky Hilltoppers"}, {"location": "Nevada", "displayName": "Nevada Wolf Pack"}),
        ("Mississippi Valley @ Sacramento State", {"abbreviation": "MVSU", "location": "Mississippi Valley State"}, {"location": "Sacramento State"}),
        ("Mercyhurst @ New Mexico State", {"location": "Mercyhurst"}, {"location": "New Mexico State"}),
        ("Utah Tech @ BYU", {"location": "Utah Tech"}, {"location": "BYU"}),
        ("Lamar @ Louisiana", {"location": "Lamar"}, {"location": "Louisiana"}),
        ("Northwestern State @ Louisiana Tech", {"location": "Northwestern State"}, {"location": "Louisiana Tech"}),
        ("ULM @ Mississippi State", {"abbreviation": "ULM"}, {"location": "Mississippi State"}),
    ]
    for game, away, home in cases:
        assert live.game_key(game, "ncaaf") in live.provider_game_keys(away, home, "ncaaf")


def test_score_coverage_counts_canonical_states_and_failure_stages(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "SCORE_COVERAGE_OUT", tmp_path / "live_score_coverage.json")
    now = datetime.now(timezone.utc)
    rows = pd.DataFrame([
        {"sport": "ncaaf", "game_id": "1", "game": "A @ B", "score_state": "in", "score_match_state": "matched", "score_away": "7", "score_home": "3", "score_updated_at_utc": now.isoformat()},
        {"sport": "ncaaf", "game_id": "2", "game": "C @ D", "score_state": "unknown", "score_match_state": "unmatched", "score_away": "-", "score_home": "-"},
        {"sport": "ufc", "game_id": "3", "game": "E vs F", "score_state": "unknown", "score_match_state": "unsupported", "score_away": "-", "score_home": "-"},
        {"sport": "mlb", "game_id": "4", "game": "G @ H", "score_state": "post", "score_match_state": "matched", "score_away": "2", "score_home": "1", "score_updated_at_utc": now.isoformat()},
    ])
    payload = live.write_score_coverage(rows, now)
    assert payload["active_live_games"] == 3
    assert payload["matched"] == 1
    assert payload["receiving_score"] == 1
    assert payload["unmatched"] == 1
    assert payload["provider_unavailable"] == 1
    assert payload["retention"] == {"final_hours": 10, "unresolved_hours": 8}
    monitored, issues = evaluate(tmp_path / "live_score_coverage.json")
    assert monitored == payload
    assert "unmatched 1" in issues
    assert "provider unavailable 1" in issues


def test_live_recent_controls_and_sort_modes_use_separate_canonical_state():
    assert 'id="live-sport-filters"' in BOARD
    assert 'id="live-status-filters"' in BOARD
    assert '<option value="STATUS_TIME">Status &amp; Time</option>' in BOARD
    assert '<option value="GAME_TIME">Game Time</option>' in BOARD
    assert '<option value="RANK">Pregame Market Read Rank</option>' in BOARD
    assert '<option value="SAVED">Saved First</option>' in BOARD
    assert "function liveState(game){ return String(game.first.score_state" in BOARD
    assert "if(liveState(a)==='in') return a.kickoff-b.kickoff" in BOARD
    assert "if(liveState(a)==='post') return b.completed-a.completed" in BOARD
    assert "if(liveRecentSort==='RANK') return a.bestRank-b.bestRank" in BOARD
    assert "if(liveRecentSort==='SAVED') return Number(b.saved)-Number(a.saved)" in BOARD
    assert "Math.min(...gameRows.map(boardRank))" in BOARD


def test_progressive_render_keeps_full_inventory_for_filtering_and_sorting():
    assert "let liveRecentSport='ALL', liveRecentStatus='ALL', liveRecentSort='STATUS_TIME', liveRecentVisible=18" in BOARD
    assert "const visible=games.slice(0,liveRecentVisible)" in BOARD
    assert "function showMoreLiveRecent(){ liveRecentVisible+=18; renderLiveRecent(); }" in BOARD
    assert "filteredLiveRecentGames(rows=allLiveRecent)" in BOARD
    assert "ensureGameEvents(row)" not in BOARD[BOARD.index("async function loadLiveRecent(force=false)"):BOARD.index("async function refreshLiveRecentScores()")]


def test_live_recent_count_loads_on_board_startup_and_updates_both_badges():
    assert "document.addEventListener('DOMContentLoaded',()=>{ applySignalTooltips(); refreshLiveRecentCount();" in BOARD
    assert "for(const id of ['b-live','rail-live-count'])" in BOARD
    count_refresh = BOARD[BOARD.index("async function refreshLiveRecentCount()"):BOARD.index("function _decRank")]
    assert "const rows=await loadCSV(LIVE_RECENT_URL);" in count_refresh
    assert "allLiveRecent=rows;_liveRecentLoaded=true;" in count_refresh
    assert "updateLiveRecentBadges(groupLiveRecentRows(allLiveRecent).length);" in count_refresh


def test_live_score_refresh_patches_score_fields_without_rebuilding_frozen_rows():
    refresh = BOARD[BOARD.index("async function refreshLiveRecentScores()"):BOARD.index("function liveMarketSections(rows)")]
    assert "const scoreFields=['score_away','score_home','score_status','score_state'" in refresh
    assert "score.textContent=liveScoreText(game.first)" in refresh
    assert "status.textContent=liveDisplayStatus(game.first)" in refresh
    assert "frozenLiveRecord" not in refresh
    assert "market_sides" not in refresh


def test_score_coverage_artifact_is_valid_json(tmp_path):
    path = tmp_path / "coverage.json"
    payload = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "unmatched": 0, "stale": 0, "provider_unavailable": 0}
    path.write_text(json.dumps(payload), encoding="utf-8")
    observed, issues = evaluate(path)
    assert observed == payload
    assert issues == []
