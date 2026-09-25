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


def test_live_recent_cards_highlight_only_persisted_favorite_market_and_saved_state():
    assert "const favorites=gameRows.filter(isRedFoxFavorite);" in BOARD
    assert "favoriteMarkets" in BOARD
    assert 'class="live-game-badges"' in BOARD
    assert 'title="Red Fox Favorite pregame market:' in BOARD
    assert "const favoriteIdentities=new Set(sides.map(item=>isRedFoxFavorite(item.row)" in BOARD
    assert "favoriteIdentities.size===1" in BOARD
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


def test_raw_recovery_pairs_both_sides_and_fills_missing_game_markets(tmp_path, monkeypatch):
    snapshots = pd.DataFrame([
        {"timestamp": "2026-09-12T23:20:00Z", "sport": "ncaaf", "game_id": "9", "game": "Navy @ Florida Atlantic", "side": "Florida Atlantic", "bets_pct": "20", "money_pct": "25", "open_line": "Florida Atlantic @ +180", "current_line": "Florida Atlantic @ +164", "dk_start_iso": "2026-09-12T23:35:00Z"},
        {"timestamp": "2026-09-12T23:20:00Z", "sport": "ncaaf", "game_id": "9", "game": "Navy @ Florida Atlantic", "side": "Navy", "bets_pct": "80", "money_pct": "75", "open_line": "Navy @ -218", "current_line": "Navy @ -198", "dk_start_iso": "2026-09-12T23:35:00Z"},
        {"timestamp": "2026-09-12T23:20:00Z", "sport": "ncaaf", "game_id": "9", "game": "Navy @ Florida Atlantic", "side": "Florida Atlantic +4", "bets_pct": "20", "money_pct": "19", "open_line": "Florida Atlantic +6.5 @ -105", "current_line": "Florida Atlantic +4 @ -108", "dk_start_iso": "2026-09-12T23:35:00Z"},
        {"timestamp": "2026-09-12T23:20:00Z", "sport": "ncaaf", "game_id": "9", "game": "Navy @ Florida Atlantic", "side": "Navy -4", "bets_pct": "80", "money_pct": "81", "open_line": "Navy -6.5 @ -115", "current_line": "Navy -4 @ -112", "dk_start_iso": "2026-09-12T23:35:00Z"},
    ])
    path = tmp_path / "snapshots.csv"
    snapshots.to_csv(path, index=False)
    monkeypatch.setattr(live, "SNAPSHOTS", path)
    recovered = live.bootstrap_started_records(datetime(2026, 9, 12, 23, 40, tzinfo=timezone.utc))
    assert set(recovered.market_display) == {"MONEYLINE", "SPREAD"}
    for value in recovered.market_sides:
        assert len(json.loads(value)) == 2


def test_compact_market_handoff_preserves_all_paired_markets_without_reclassifying(tmp_path):
    rows = pd.DataFrame([
        {"timestamp": "2026-09-12T23:10:00Z", "sport": "ncaaf", "game_id": "9", "game": "Navy @ Florida Atlantic", "market_display": "MONEYLINE", "side": "Florida Atlantic", "bets_pct": "22", "money_pct": "24", "open_line": "Florida Atlantic @ +180", "current_line": "Florida Atlantic @ +170", "dk_start_iso": "2026-09-12T23:35:00Z"},
        {"timestamp": "2026-09-12T23:10:00Z", "sport": "ncaaf", "game_id": "9", "game": "Navy @ Florida Atlantic", "market_display": "MONEYLINE", "side": "Navy", "bets_pct": "78", "money_pct": "76", "open_line": "Navy @ -218", "current_line": "Navy @ -205", "dk_start_iso": "2026-09-12T23:35:00Z"},
        {"timestamp": "2026-09-12T23:20:00Z", "sport": "ncaaf", "game_id": "9", "game": "Navy @ Florida Atlantic", "market_display": "MONEYLINE", "side": "Florida Atlantic", "bets_pct": "20", "money_pct": "25", "open_line": "Florida Atlantic @ +180", "current_line": "Florida Atlantic @ +164", "dk_start_iso": "2026-09-12T23:35:00Z"},
        {"timestamp": "2026-09-12T23:20:00Z", "sport": "ncaaf", "game_id": "9", "game": "Navy @ Florida Atlantic", "market_display": "MONEYLINE", "side": "Navy", "bets_pct": "80", "money_pct": "75", "open_line": "Navy @ -218", "current_line": "Navy @ -198", "dk_start_iso": "2026-09-12T23:35:00Z"},
        {"timestamp": "2026-09-12T23:20:00Z", "sport": "ncaaf", "game_id": "9", "game": "Navy @ Florida Atlantic", "market_display": "SPREAD", "side": "Florida Atlantic +4", "bets_pct": "20", "money_pct": "19", "open_line": "Florida Atlantic +6.5 @ -105", "current_line": "Florida Atlantic +4 @ -108", "dk_start_iso": "2026-09-12T23:35:00Z"},
        {"timestamp": "2026-09-12T23:20:00Z", "sport": "ncaaf", "game_id": "9", "game": "Navy @ Florida Atlantic", "market_display": "SPREAD", "side": "Navy -4", "bets_pct": "80", "money_pct": "81", "open_line": "Navy -6.5 @ -115", "current_line": "Navy -4 @ -112", "dk_start_iso": "2026-09-12T23:35:00Z"},
    ])
    path = tmp_path / "live_recent_market_candidates.csv"
    candidates = live.update_market_candidates(rows, path, as_of="2026-09-12T23:20:00Z")
    assert set(candidates.market_display) == {"MONEYLINE", "SPREAD"}
    assert candidates.red_fox_favorite.eq("false").all()
    moneyline = candidates[candidates.market_display.eq("MONEYLINE")].iloc[0]
    assert moneyline.current_line == "Florida Atlantic @ +164"
    assert len(json.loads(moneyline.market_sides)) == 2
    frozen = live.final_pregame_states(candidates, datetime(2026, 9, 12, 23, 36, tzinfo=timezone.utc))
    assert set(frozen.market_display) == {"MONEYLINE", "SPREAD"}
    assert all(len(json.loads(value)) == 2 for value in frozen.market_sides)


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


def test_favorite_handoff_rejects_a_later_explicit_prekickoff_falloff():
    candidate = pd.DataFrame([{
        "sport": "mlb", "game_id": "stale", "market_display": "MONEYLINE",
        "kickoff_iso": "2026-09-13T17:40:00Z", "state_as_of_utc": "2026-09-13T11:31:00Z",
        "red_fox_favorite": "true", "favorite_side": "HOU Astros",
    }])
    tracking = pd.DataFrame([
        {"sport": "mlb", "game_id": "stale", "market_display": "MONEYLINE", "recorded_at": "2026-09-13T11:31:00Z", "favorite_state": "qualified"},
        {"sport": "mlb", "game_id": "stale", "market_display": "MONEYLINE", "recorded_at": "2026-09-13T11:41:00Z", "favorite_state": "not_qualified"},
    ])

    frozen = live.favorite_handoff_states(
        candidate, datetime(2026, 9, 13, 17, 41, tzinfo=timezone.utc), tracking
    )

    assert frozen.empty


def test_favorite_handoff_keeps_a_final_requalification_after_an_earlier_falloff():
    candidate = pd.DataFrame([{
        "sport": "nfl", "game_id": "requalified", "market_display": "SPREAD",
        "kickoff_iso": "2026-09-13T20:25:00Z", "state_as_of_utc": "2026-09-13T20:01:00Z",
        "red_fox_favorite": "true", "favorite_side": "MIA Dolphins +3",
    }])
    tracking = pd.DataFrame([
        {"sport": "nfl", "game_id": "requalified", "market_display": "SPREAD", "recorded_at": "2026-09-13T12:55:00Z", "favorite_state": "not_qualified"},
        {"sport": "nfl", "game_id": "requalified", "market_display": "SPREAD", "recorded_at": "2026-09-13T20:01:00Z", "favorite_state": "qualified"},
    ])

    frozen = live.favorite_handoff_states(
        candidate, datetime(2026, 9, 13, 20, 26, tzinfo=timezone.utc), tracking
    )

    assert len(frozen) == 1


def test_existing_live_recent_card_loses_badge_after_later_prekickoff_falloff():
    existing = pd.DataFrame([{
        "sport": "mlb", "game_id": "stale", "market_display": "MONEYLINE",
        "kickoff_iso": "2026-09-13T17:40:00Z",
        "freeze_method": "favorite_tracking_last_qualified_at_or_before_start",
        "red_fox_favorite": "true", "favorite_side": "HOU Astros",
        "favorite_state": "qualified", "favorite_originally_qualified": "true",
    }])
    tracking = pd.DataFrame([
        {"sport": "mlb", "game_id": "stale", "market_display": "MONEYLINE", "recorded_at": "2026-09-13T11:31:00Z", "favorite_state": "qualified"},
        {"sport": "mlb", "game_id": "stale", "market_display": "MONEYLINE", "recorded_at": "2026-09-13T11:41:00Z", "favorite_state": "not_qualified"},
    ])

    cleaned = live.suppress_stale_retained_favorites(existing, tracking).iloc[0]

    assert cleaned.red_fox_favorite == "false"
    assert cleaned.favorite_state == "not_qualified"
    assert cleaned.favorite_originally_qualified == "true"
    assert "later explicit pregame falloff" in cleaned.favorite_reason


def test_existing_live_recent_card_keeps_badge_after_final_requalification():
    existing = pd.DataFrame([{
        "sport": "nfl", "game_id": "requalified", "market_display": "SPREAD",
        "kickoff_iso": "2026-09-13T20:25:00Z",
        "freeze_method": "favorite_tracking_last_qualified_at_or_before_start",
        "red_fox_favorite": "true", "favorite_side": "MIA Dolphins +3",
        "favorite_state": "qualified",
    }])
    tracking = pd.DataFrame([
        {"sport": "nfl", "game_id": "requalified", "market_display": "SPREAD", "recorded_at": "2026-09-13T12:55:00Z", "favorite_state": "not_qualified"},
        {"sport": "nfl", "game_id": "requalified", "market_display": "SPREAD", "recorded_at": "2026-09-13T20:01:00Z", "favorite_state": "qualified"},
    ])

    cleaned = live.suppress_stale_retained_favorites(existing, tracking).iloc[0]

    assert cleaned.red_fox_favorite == "true"
    assert cleaned.favorite_state == "qualified"


def test_late_invalidated_favorite_is_retained_for_audit_but_suppressed_at_handoff():
    candidate = pd.DataFrame([{
        "sport": "nfl", "game_id": "late-1", "market_display": "SPREAD",
        "kickoff_iso": "2026-09-12T23:35:00Z", "state_as_of_utc": "2026-09-12T23:34:00Z",
        "supported_side": "Arizona +9.5", "red_fox_favorite": "true",
        "favorite_side": "Arizona +9.5", "favorite_state": "qualified",
        "favorite_originally_qualified": "true", "favorite_late_invalidated": "true",
        "favorite_late_invalidated_reason": "Hard removal: the confirmed supported side flipped.",
    }])
    frozen = live.favorite_handoff_states(candidate, datetime(2026, 9, 12, 23, 36, tzinfo=timezone.utc))
    assert len(frozen) == 1
    assert frozen.iloc[0].red_fox_favorite == "false"
    assert frozen.iloc[0].favorite_state == "late_invalidated"
    assert frozen.iloc[0].favorite_originally_qualified == "true"


def test_visibility_invalidated_favorite_cannot_reenter_from_handoff_archive():
    candidate = pd.DataFrame([{
        "sport": "ncaaf", "game_id": "34603681", "market_display": "SPREAD",
        "kickoff_iso": "2026-09-12T22:00:00Z", "state_as_of_utc": "2026-09-12T21:50:59Z",
        "supported_side": "Jacksonville State +1.5", "red_fox_favorite": "true",
        "favorite_side": "Jacksonville State +1.5", "favorite_state": "qualified",
    }])
    frozen = live.favorite_handoff_states(candidate, datetime(2026, 9, 12, 22, 1, tzinfo=timezone.utc))
    assert frozen.empty


def test_rule_invalidated_mlb_favorite_is_removed_from_retained_display():
    candidate = pd.DataFrame([{
        "sport": "mlb", "game_id": "34694536", "market_display": "MONEYLINE",
        "red_fox_favorite": "true", "favorite_side": "WAS Nationals",
        "favorite_state": "qualified", "favorite_reason": "legacy qualification",
    }])
    corrected = live.apply_favorite_exclusions(candidate).iloc[0]
    assert corrected.red_fox_favorite == "false"
    assert corrected.favorite_state == "not_qualified"
    assert "five-point" in corrected.favorite_reason


def test_arizona_system_miss_is_retroactively_classified_with_explicit_provenance():
    sides = [
        {"flagged_side": "ARI Cardinals +8.5", "bets_pct": 37, "money_pct": 46, "open_line": "+10.5 (-110)", "current_line": "+8.5 (-110)", "path": "One-Way"},
        {"flagged_side": "LA Chargers -8.5", "bets_pct": 63, "money_pct": 54, "open_line": "-10.5 (-110)", "current_line": "-8.5 (-110)", "path": "One-Way"},
    ]
    frozen = pd.DataFrame([{
        "sport": "nfl", "game_id": "34118231", "market_display": "SPREAD",
        "market_sides": json.dumps(sides), "state_as_of_utc": "2026-09-13T20:24:00Z",
        "final_pregame_state_at_utc": "2026-09-13T20:24:00Z",
        "red_fox_favorite": "false", "supported_side": "",
    }])

    corrected = live.apply_classification_corrections(frozen).iloc[0]

    assert corrected.red_fox_favorite == "true"
    assert corrected.favorite_side == "ARI Cardinals +8.5"
    assert corrected.supported_side == "ARI Cardinals +8.5"
    assert corrected.favorite_state == "qualified_retroactive_system_correction"
    assert corrected.classification_correction == "true"
    assert corrected.classification_original_publication == "missed"
    assert corrected.favorite_first_qualified_at == "2026-09-13T20:24:00Z"
    assert corrected.freeze_method == "retroactive_system_correction_from_retained_pregame_state"


def test_falcons_whipsaw_recovery_is_retroactively_classified_with_explicit_provenance():
    sides = [
        {"flagged_side": "GB Packers -4.5", "bets_pct": 69, "money_pct": 65, "open_line": "GB Packers -6.5 @ -115", "current_line": "GB Packers -4.5 @ -112", "path": "Pregame snapshot"},
        {"flagged_side": "ATL Falcons +4.5", "bets_pct": 31, "money_pct": 35, "open_line": "ATL Falcons +6.5 @ -105", "current_line": "ATL Falcons +4.5 @ -108", "path": "Pregame snapshot"},
    ]
    frozen = pd.DataFrame([{
        "sport": "nfl", "game_id": "34118180", "market_display": "SPREAD",
        "market_sides": json.dumps(sides), "state_as_of_utc": "2026-09-25T00:12:20.118916+00:00",
        "final_pregame_state_at_utc": "2026-09-25T00:12:20.118916+00:00",
        "red_fox_favorite": "false", "supported_side": "",
    }])

    corrected = live.apply_classification_corrections(frozen).iloc[0]

    assert corrected.red_fox_favorite == "true"
    assert corrected.favorite_side == "ATL Falcons +4.5"
    assert corrected.favorite_rule_version == "red_fox_favorite_v3"
    assert corrected.classification_correction == "true"
    assert corrected.classification_original_publication == "missed"
    assert "whipsaw counter" in corrected.classification_correction_reason


def test_one_minute_worker_expires_started_board_only_after_freeze_window(tmp_path, monkeypatch):
    board_path = tmp_path / "anomaly_board.csv"
    monkeypatch.setattr(live, "BOARD", board_path)
    monkeypatch.setattr(live, "DATA", tmp_path)
    pd.DataFrame([
        {"game_id": "started", "kickoff_iso": "2026-09-12T16:00:00Z", "red_fox_favorite": "true"},
        {"game_id": "grace", "kickoff_iso": "2026-09-12T16:08:00Z", "red_fox_favorite": "false"},
        {"game_id": "future", "kickoff_iso": "2026-09-12T18:00:00Z", "red_fox_favorite": "false"},
    ]).to_csv(board_path, index=False)
    original_hash = __import__("hashlib").sha256(board_path.read_bytes()).hexdigest()
    (tmp_path / "publication_coverage.json").write_text(json.dumps({"board_sha256": original_hash}))

    removed = live.expire_started_board_rows(datetime(2026, 9, 12, 16, 10, tzinfo=timezone.utc))

    assert removed == 1
    remaining = pd.read_csv(board_path, dtype=str, keep_default_na=False)
    assert remaining.game_id.tolist() == ["grace", "future"]
    assert remaining.loc[remaining.game_id.eq("grace"), "red_fox_favorite"].item() == "false"
    coverage = json.loads((tmp_path / "publication_coverage.json").read_text())
    assert coverage["board_sha256"] == __import__("hashlib").sha256(board_path.read_bytes()).hexdigest()
    assert coverage["post_publish_expiry"]["removed_rows"] == 1
