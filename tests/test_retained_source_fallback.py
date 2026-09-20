import json
from pathlib import Path

import pandas as pd

from publication_coverage import PublicationCoverage
from refresh_anomaly_board import restore_retained_favorites, write_board_freshness


ROOT = Path(__file__).resolve().parents[1]
BOARD = (ROOT / "site" / "board.html").read_text(encoding="utf-8")
REFRESH = (ROOT / "refresh_anomaly_board.py").read_text(encoding="utf-8")


def test_retained_source_keeps_true_capture_age_and_explicit_state(tmp_path):
    dashboard = pd.DataFrame(
        [{"sport": "nfl", "game_id": "1", "market_display": "SPREAD", "timestamp": "2026-09-20T08:30:00Z"}]
    )
    oldest, newest, count = write_board_freshness(
        dashboard,
        data_dir=tmp_path,
        now="2026-09-20T16:00:00Z",
        source_state="RETAINED",
        source_note="provider empty",
    )
    payload = json.loads((tmp_path / "freshness.json").read_text())
    assert oldest.isoformat() == "2026-09-20T08:30:00+00:00"
    assert newest == oldest
    assert count == 1
    assert payload["board_source_state"] == "RETAINED"
    assert payload["board_source_note"] == "provider empty"
    assert payload["board_published_at"] == "2026-09-20T16:00:00+00:00"


def test_retained_mode_is_bounded_and_never_fakes_live_timestamp():
    assert 'REDFOX_RETAINED_SOURCE_MAX_AGE_MINUTES", "720"' in REFRESH
    assert 'board["data_badge"] = "RETAINED"' in REFRESH
    assert "Retained source" in BOARD
    assert "sources[0].label='Retained source'; sources[0].forced='warn'" in BOARD


def test_recent_empty_scrape_can_reuse_only_parse_failed_rows(tmp_path):
    coverage = PublicationCoverage(tmp_path, pd.Timestamp("2026-09-20T16:00:00Z"))
    coverage.store.update(
        [
            {"sport": "nfl", "game_id": "parse", "market_display": "SPREAD",
             "capture_exclusion_reason": "RAW_MARKET_PARSE_FAILED",
             "dk_start_iso": "2026-09-21T20:00:00Z"},
            {"sport": "nfl", "game_id": "identity", "market_display": "TOTAL",
             "capture_exclusion_reason": "UNRESOLVED_EVENT_IDENTITY",
             "dk_start_iso": "2026-09-21T20:00:00Z"},
        ],
        coverage.run_id,
        "TEST",
    )
    scrape = coverage.store.begin("SCRAPE", "nfl", now="2026-09-20T15:55:00Z")
    coverage.store.finish(scrape, "EMPTY_COMPLETE")
    rows = pd.DataFrame([
        {"sport": "nfl", "game_id": "parse", "market_display": "SPREAD"},
        {"sport": "nfl", "game_id": "identity", "market_display": "TOTAL"},
    ])

    empty_sports = coverage.recent_empty_sports()
    accepted = coverage.validated(rows, allow_retained_parse_failed_sports=empty_sports)

    assert empty_sports == {"nfl"}
    assert accepted.game_id.tolist() == ["parse"]
    assert coverage.retained_parse_failed_keys == {("nfl", "parse", "SPREAD")}
    assert coverage.reasons[("nfl", "identity", "TOTAL")] == "UNRESOLVED_EVENT_IDENTITY"
    board_path = tmp_path / "board.csv"
    accepted.to_csv(board_path, index=False)
    coverage.gate_ready = {("nfl", "parse", "SPREAD")}
    summary = coverage.publish(accepted, board_path, lambda probe, now: probe)
    assert summary["publication_conflicts"] == 0
    assert summary["sports"]["nfl"]["published"] == 1


def test_parse_failed_rows_stay_blocked_without_a_recent_empty_scrape(tmp_path):
    coverage = PublicationCoverage(tmp_path, pd.Timestamp("2026-09-20T16:00:00Z"))
    coverage.store.update(
        [{"sport": "nfl", "game_id": "g", "market_display": "SPREAD",
          "capture_exclusion_reason": "RAW_MARKET_PARSE_FAILED"}],
        coverage.run_id,
        "TEST",
    )
    rows = pd.DataFrame([{"sport": "nfl", "game_id": "g", "market_display": "SPREAD"}])

    assert coverage.validated(rows).empty


def test_retained_mode_restores_only_matching_pre_outage_favorite(tmp_path):
    archive = pd.DataFrame([
        {"sport": "nfl", "game_id": "keep", "market_display": "SPREAD",
         "red_fox_favorite": "true", "favorite_state": "qualified",
         "favorite_side": "NY Jets +3.5", "favorite_pathway": "low_support_contrarian",
         "favorite_late_invalidated": "true",
         "favorite_late_invalidated_reason": "Favorite unavailable: latest verified market state is 49.6 minutes old (maximum 30).",
         "candidate_recorded_at_utc": "2026-09-20T07:05:00Z"},
        {"sport": "nfl", "game_id": "mismatch", "market_display": "SPREAD",
         "red_fox_favorite": "true", "favorite_state": "qualified",
         "favorite_side": "Away +3", "candidate_recorded_at_utc": "2026-09-20T07:05:00Z"},
        {"sport": "nfl", "game_id": "historical", "market_display": "SPREAD",
         "red_fox_favorite": "true", "favorite_state": "qualified",
         "favorite_side": "Home -2", "candidate_recorded_at_utc": "2026-09-19T07:05:00Z"},
    ])
    archive.to_csv(tmp_path / "red_fox_favorite_freeze_candidates.csv", index=False)
    board = pd.DataFrame([
        {"sport": "nfl", "game_id": "keep", "market_display": "SPREAD",
         "kickoff_iso": "2026-09-20T17:00:00Z", "supported_side": "NY Jets +3.5",
         "red_fox_favorite": "false", "favorite_state": "not_qualified"},
        {"sport": "nfl", "game_id": "mismatch", "market_display": "SPREAD",
         "kickoff_iso": "2026-09-20T20:00:00Z", "supported_side": "Home -3",
         "red_fox_favorite": "false", "favorite_state": "not_qualified"},
        {"sport": "nfl", "game_id": "historical", "market_display": "SPREAD",
         "kickoff_iso": "2026-09-20T20:00:00Z", "supported_side": "Home -2",
         "red_fox_favorite": "false", "favorite_state": "not_qualified"},
    ])

    result = restore_retained_favorites(board, tmp_path, now="2026-09-20T16:30:00Z")

    assert result.loc[result.game_id.eq("keep"), "red_fox_favorite"].iloc[0] == "true"
    assert "pre-outage" in result.loc[result.game_id.eq("keep"), "favorite_reason"].iloc[0]
    assert result.loc[result.game_id.eq("mismatch"), "red_fox_favorite"].iloc[0] == "false"
    assert result.loc[result.game_id.eq("historical"), "red_fox_favorite"].iloc[0] == "false"
