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


def test_app_state_provider_brand_matches_appalachian_state_board_identity():
    away = {
        "displayName": "App State Mountaineers",
        "location": "App State",
        "shortDisplayName": "App State",
        "abbreviation": "APP",
    }
    home = {
        "displayName": "East Carolina Pirates",
        "location": "East Carolina",
        "shortDisplayName": "East Carolina",
        "abbreviation": "ECU",
    }
    assert live.game_key("Appalachian State @ East Carolina", "ncaaf") in live.provider_game_keys(away, home, "ncaaf")


def test_southern_university_matches_espn_southern_identity():
    away = {
        "displayName": "Southern Jaguars",
        "location": "Southern",
        "shortDisplayName": "Southern",
        "abbreviation": "SOU",
    }
    home = {
        "displayName": "Houston Cougars",
        "location": "Houston",
        "shortDisplayName": "Houston",
        "abbreviation": "HOU",
    }
    assert live.game_key("Southern University @ Houston", "ncaaf") in live.provider_game_keys(away, home, "ncaaf")


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
    assert payload["provider_unavailable"] == 0
    assert payload["unsupported"] == 1
    assert payload["retention"] == {"final_hours": 10, "unresolved_hours": 8}
    monitored, issues = evaluate(tmp_path / "live_score_coverage.json")
    assert monitored == payload
    assert "unmatched 1" in issues
    assert not any("provider unavailable" in issue for issue in issues)


def test_live_recent_controls_and_sort_modes_use_frozen_canonical_state():
    assert 'id="live-sport-filters"' in BOARD
    assert 'id="live-status-filters"' in BOARD
    assert '<option value="STATUS_TIME">Status &amp; Time</option>' in BOARD
    assert '<option value="GAME_TIME">Game Time</option>' not in BOARD
    assert '<option value="RANK">Pregame Market Read Rank</option>' in BOARD
    assert '<option value="SAVED">Saved First</option>' in BOARD
    assert "function liveState(game){ return String(game.first.score_state" in BOARD
    assert "if(liveState(a)==='in') return a.kickoff-b.kickoff" in BOARD
    assert "if(liveState(a)==='post') return b.completed-a.completed" in BOARD
    assert "if(liveRecentSort==='RANK') return a.bestRank-b.bestRank" in BOARD
    assert "if(liveRecentSort==='SAVED') return Number(b.saved)-Number(a.saved)" in BOARD
    assert "Math.min(...gameRows.map(boardRank))" in BOARD
    assert "function frozenFavoriteRank(row)" in BOARD
    assert "favorite_final_market_rank" in BOARD
    assert "Math.min(...favorites.map(frozenFavoriteRank))" in BOARD


def test_live_recent_cards_present_persisted_favorite_supported_and_saved_state():
    assert "const favorites=gameRows.filter(isRedFoxFavorite);" in BOARD
    assert "favoriteMarkets" in BOARD
    assert 'class="live-game-badges"' in BOARD
    assert 'title="Red Fox Favorite pregame market:' in BOARD
    assert "const confirmed=new Set(sides.map(item=>String(item.row.supported_side||'').trim()" in BOARD
    assert "confirmed.size===1" in BOARD
    assert "item.supported?' is-supported':''" in BOARD
    assert ".live-market-side-row.is-supported { background:linear-gradient(90deg,rgba(21,134,106,.10)" in BOARD
    assert 'class="live-save"' in BOARD
    assert "window.toggleSavedLiveGame=function(gameKey,event)" in BOARD
    assert "const favoriteRows=rows.filter(isRedFoxFavorite).sort" in BOARD
    assert "[...(window._boardRows||[]),...(allLiveRecent||[])]" in BOARD


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


def test_final_pregame_state_depends_on_source_time_not_timer_offset():
    board = pd.DataFrame([{
        "sport": "mlb", "game_id": "1", "market_display": "MONEYLINE",
        "kickoff_iso": "2026-09-08T22:40:00Z",
        "state_as_of_utc": "2026-09-08T22:30:55Z",
        "supported_side": "NY Mets",
    }])
    before = live.final_pregame_states(board, datetime(2026, 9, 8, 22, 39, tzinfo=timezone.utc))
    one_minute_after = live.final_pregame_states(board, datetime(2026, 9, 8, 22, 41, tzinfo=timezone.utc))
    three_minutes_after = live.final_pregame_states(board, datetime(2026, 9, 8, 22, 43, tzinfo=timezone.utc))
    assert before.empty
    assert one_minute_after.iloc[0]["supported_side"] == three_minutes_after.iloc[0]["supported_side"] == "NY Mets"
    assert one_minute_after.iloc[0]["final_pregame_state_at_utc"] == three_minutes_after.iloc[0]["final_pregame_state_at_utc"]


def test_final_pregame_state_rejects_source_observed_after_kickoff():
    board = pd.DataFrame([{
        "kickoff_iso": "2026-09-08T22:40:00Z",
        "state_as_of_utc": "2026-09-08T22:40:01Z",
    }])
    assert live.final_pregame_states(board, datetime(2026, 9, 8, 22, 42, tzinfo=timezone.utc)).empty


def test_favorite_handoff_survives_absence_from_current_board():
    candidate = pd.DataFrame([{
        "sport": "ncaaf", "game_id": "34603696", "market_display": "SPREAD",
        "kickoff_iso": "2026-09-12T23:35:00Z", "state_as_of_utc": "2026-09-12T23:11:03Z",
        "supported_side": "Florida Atlantic +4", "red_fox_favorite": "true",
        "favorite_side": "Florida Atlantic +4",
    }])
    frozen = live.favorite_handoff_states(candidate, datetime(2026, 9, 12, 23, 36, tzinfo=timezone.utc))
    assert len(frozen) == 1
    assert frozen.iloc[0].favorite_side == "Florida Atlantic +4"
    assert frozen.iloc[0].freeze_method == "favorite_tracking_last_qualified_at_or_before_start"


def test_visibility_invalidated_favorite_cannot_reenter_from_handoff_archive():
    candidate = pd.DataFrame([{
        "sport": "ncaaf", "game_id": "34603681", "market_display": "SPREAD",
        "kickoff_iso": "2026-09-12T22:00:00Z", "state_as_of_utc": "2026-09-12T21:50:59Z",
        "supported_side": "Jacksonville State +1.5", "red_fox_favorite": "true",
        "favorite_side": "Jacksonville State +1.5", "favorite_state": "qualified",
    }])
    frozen = live.favorite_handoff_states(candidate, datetime(2026, 9, 12, 22, 1, tzinfo=timezone.utc))
    assert frozen.empty


def test_one_minute_worker_expires_started_board_only_after_freeze_window(tmp_path, monkeypatch):
    board_path = tmp_path / "anomaly_board.csv"
    monkeypatch.setattr(live, "BOARD", board_path)
    pd.DataFrame([
        {"game_id": "started", "kickoff_iso": "2026-09-12T16:00:00Z", "red_fox_favorite": "true"},
        {"game_id": "grace", "kickoff_iso": "2026-09-12T16:08:00Z", "red_fox_favorite": "false"},
        {"game_id": "future", "kickoff_iso": "2026-09-12T18:00:00Z", "red_fox_favorite": "false"},
    ]).to_csv(board_path, index=False)

    removed = live.expire_started_board_rows(datetime(2026, 9, 12, 16, 10, tzinfo=timezone.utc))

    assert removed == 1
    remaining = pd.read_csv(board_path, dtype=str, keep_default_na=False)
    assert remaining.game_id.tolist() == ["grace", "future"]
    assert remaining.loc[remaining.game_id.eq("grace"), "red_fox_favorite"].item() == "false"
