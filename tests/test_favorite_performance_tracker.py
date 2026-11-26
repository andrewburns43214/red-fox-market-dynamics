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
            "sport": ["nfl", "nfl", "nba", "nfl"][index],
            "favorite_rule_version": "red_fox_favorite_20261124" if index == 2 else "",
            "game": f"Game {index}",
            "scheduled_start": f"2026-09-{index+10}T20:00:00Z",
            "side": "Away +3",
            "market": "SPREAD",
            "open_line": "+4",
            "open_price": "-108",
            "decision_line": "+3",
            "decision_price": "-110",
            "bets_pct": "0" if index == 3 else "35.0",
            "money_pct": "61" if index != 3 else "",
        })
    rows.extend([
        {"ledger_id": "excluded", "favorite_qualified": "no", "grade": "W", "sport": "nfl"},
        {"ledger_id": "pending", "favorite_qualified": "yes", "grade": "", "sport": "nfl"},
    ])
    pd.DataFrame(rows).to_csv(tmp_path / "performance_ledger.csv", index=False)

    update_performance_ledger(tmp_path, frozen_sources=[], attach_results=False)

    payload = json.loads((tmp_path / "favorite_performance.json").read_text(encoding="utf-8"))
    assert payload["wins"] == 2
    assert payload["losses"] == 1
    assert payload["pushes"] == 1
    assert payload["win_rate_excluding_pushes"] == 0.666667
    assert payload["tracking_start_date"] == FAVORITE_TRACKING_START_DATE
    assert payload["average_entry_odds"] == {
        "american": -110.0, "priced_selections": 4, "pregame_fallback_quotes": 0,
    }
    assert [row["label"] for row in payload["sports"]] == ["NFL", "NBA", "CFB", "MLB", "NCAAB", "NHL", "UFC"]
    nfl = next(row for row in payload["sports"] if row["sport"] == "nfl")
    assert nfl == {
        "sport": "nfl", "label": "NFL", "wins": 2, "losses": 0,
        "pushes": 1, "total_graded": 3, "win_rate_excluding_pushes": 1.0,
    }
    nba = next(row for row in payload["sports"] if row["sport"] == "nba")
    assert nba["wins"] == 0
    assert nba["losses"] == 1
    assert nba["win_rate_excluding_pushes"] == 0.0
    assert set(payload["logic_change_records"]) == {"ncaab", "nba", "nhl", "ncaaf", "mlb", "ufc"}
    assert payload["logic_change_records"]["nba"]["since"] == "2026-11-24"
    for sport in ("ncaaf", "mlb", "ufc"):
        assert payload["logic_change_records"][sport]["since"] == "2026-11-25"
        assert payload["logic_change_records"][sport]["total_graded"] == 0
    assert payload["logic_change_records"]["nba"]["losses"] == 0
    assert payload["logic_change_records"]["ncaab"]["total_graded"] == 0
    assert [row["game"] for row in payload["games"]] == ["Game 3", "Game 2", "Game 1", "Game 0"]
    assert [row["grade"] for row in payload["games"] if row["sport"] == "nfl"] == ["Push", "W", "W"]
    assert payload["games"][0]["decision_price"] == "-110"
    assert (payload["games"][0]["open_line"], payload["games"][0]["open_price"]) == ("+4", "-108")
    assert payload["games"][0]["bets_pct"] == "0"
    assert payload["games"][0]["money_pct"] == ""
    assert payload["games"][2]["money_pct"] == "61"
    priced = payload["priced_performance"]
    assert priced["overall"]["net_units"] == 0.818182
    assert priced["overall"]["roi"] == 0.204545
    assert priced["sports"]["nba"]["net_units"] == -1.0
    assert priced["sports"]["nba"]["roi"] == -1.0
    assert priced["sports"]["nhl"] is None
    assert "ledger_id" not in payload["games"][0]
    assert "score_provider_event_id" not in payload["games"][0]



def test_average_entry_odds_include_manual_and_audited_quotes_without_rewriting_history(tmp_path):
    common = dict(
        favorite_qualified="yes", sport="nfl", market="SPREAD", grade="W",
        scheduled_start="2026-09-13T20:25:00Z",
    )
    recovery = dict(
        ledger_id="0bbd0934a6431da48d41", side="ARI Cardinals +8.5",
        decision_line="+8.5", decision_line_source="final_pregame_fallback",
        final_pregame_state_at_utc="2026-09-13T20:16:25.192992+00:00",
    )
    rows = [
        {**common, **recovery},
        {**common, "ledger_id": "manual", "favorite_origin": "manual",
         "decision_price": "+150", "decision_line_source": "owner_selected_quote"},
        {**common, "ledger_id": "fallback", "decision_price": "-125",
         "decision_line_source": "final_pregame_fallback", "grade": "Push"},
        {**common, "ledger_id": "invalid-price", "decision_price": "bad-110"},
        {**common, **recovery, "ledger_id": "different-selection"},
        {**common, "ledger_id": "void", "decision_price": "+900", "grade": "Void"},
        {**common, "ledger_id": "pending", "decision_price": "+900", "grade": ""},
        {**common, "ledger_id": "other", "decision_price": "+900", "favorite_qualified": "no"},
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "performance_ledger.csv", index=False)

    update_performance_ledger(tmp_path, frozen_sources=[], attach_results=False)

    payload = json.loads((tmp_path / "favorite_performance.json").read_text(encoding="utf-8"))
    assert payload["total_graded"] == 5
    assert payload["average_entry_odds"] == {
        "american": 109.346405, "priced_selections": 3, "pregame_fallback_quotes": 1,
    }
    overall = payload["priced_performance"]["overall"]
    assert overall["net_units"] == 2.480392
    assert overall["roi"] == 0.826797
    assert payload["priced_performance"]["sports"]["nfl"] == overall
    ledger = pd.read_csv(tmp_path / "performance_ledger.csv", dtype=str, keep_default_na=False)
    assert ledger.loc[ledger["ledger_id"].eq(recovery["ledger_id"]), "decision_price"].item() == ""

def test_logic_change_record_counts_official_entries_from_release_not_rule_version(tmp_path):
    examples = [
        ("Prior NBA Favorite", "nba", "red_fox_favorite_basketball_v2", "2026-11-24T17:39:01-05:00", "W", "yes"),
        ("At release NBA Favorite", "nba", "red_fox_favorite_basketball_v2", "2026-11-24T17:39:02-05:00", "L", "yes"),
        ("NBA replacement", "nba", "red_fox_favorite_nba_public_neutral_v1", "2026-11-25T17:00:00Z", "W", "yes"),
        ("Louisville", "ncaab", "red_fox_favorite_20261124", "2026-11-24T23:01:48Z", "W", "yes"),
        ("Xavier", "ncaab", "red_fox_favorite_basketball_v2", "2026-11-25T15:48:00Z", "W", "yes"),
        ("Missing entry time", "ncaab", "red_fox_favorite_20261124", "", "W", "yes"),
        ("Not official", "ncaab", "red_fox_favorite_basketball_v2", "2026-11-25T17:00:00Z", "W", "no"),
        ("Not settled", "ncaab", "red_fox_favorite_basketball_v2", "2026-11-25T17:00:00Z", "", "yes"),
        ("NHL replacement", "nhl", "red_fox_favorite_nhl_genuine_crossover_v1", "2026-11-25T22:00:00Z", "Push", "yes"),
        ("Early CFB", "ncaaf", "red_fox_favorite_v8", "2026-11-25T05:19:34Z", "L", "yes"),
        ("Post-change CFB", "ncaaf", "red_fox_favorite_cfb_high_money_v1", "2026-11-25T05:19:35Z", "W", "yes"),
        ("Early MLB", "mlb", "red_fox_favorite_v5", "2026-11-25T18:13:32Z", "L", "yes"),
        ("Post-change MLB", "mlb", "red_fox_favorite_mlb_static_ml_v1", "2026-11-25T18:13:33Z", "W", "yes"),
        ("Early UFC", "ufc", "red_fox_favorite_ufc_v1", "2026-11-25T18:40:18Z", "L", "yes"),
        ("Post-change UFC", "ufc", "red_fox_favorite_ufc_v1", "2026-11-25T18:40:19Z", "W", "yes"),
    ]
    pd.DataFrame([
        dict(ledger_id=str(index), game=name, sport=sport, favorite_rule_version=version,
             favorite_first_qualified_at=entry, scheduled_start="2026-11-26T02:00:00Z",
             grade=grade, favorite_qualified=favorite)
        for index, (name, sport, version, entry, grade, favorite) in enumerate(examples)
    ]).to_csv(tmp_path / "performance_ledger.csv", index=False)

    update_performance_ledger(tmp_path, frozen_sources=[], attach_results=False)

    records = json.loads((tmp_path / "favorite_performance.json").read_text(encoding="utf8"))
    assert records["wins"] == 8
    current = records["logic_change_records"]
    assert (current["nba"]["wins"], current["nba"]["losses"]) == (1, 1)
    assert (current["ncaab"]["wins"], current["ncaab"]["total_graded"]) == (2, 2)
    assert current["nhl"]["pushes"] == 1
    for sport in ("ncaaf", "mlb", "ufc"):
        assert current[sport]["wins"] == current[sport]["total_graded"] == 1
        assert current[sport]["losses"] == 0


def test_tracker_uses_one_shared_payload_for_desktop_mobile_and_guide():
    assert BOARD.count('data-favorite-performance hidden') == 2
    assert 'favorite-performance-desktop' in BOARD
    assert 'favorite-performance-mobile' in BOARD
    assert "const gradedDecisions=wins+losses;" in BOARD
    assert "winPercentage:gradedDecisions?wins/gradedDecisions*100:0" in BOARD
    assert "FAVORITE_PERFORMANCE_BOOTSTRAP" not in BOARD
    assert "let favoritePerformance=null;" in BOARD
    assert "favoritePerformanceFooter()" in BOARD
    assert 'class="favorite-guide-record"' in BOARD
    assert "footer.innerHTML=favoritePerformanceFooterContent(text);" in BOARD
    assert 'class="favorite-guide-heading favorite-guide-sports-title">By sport</h4>' in BOARD
    assert 'class="favorite-performance-sports" aria-label="Red Fox Favorite win percentage by sport"' in BOARD
    assert "10 Graded" not in BOARD


def test_every_favorite_logo_routes_to_the_same_information_modal():
    assert "favorite=game.hasFavorite?favoriteBadge(favoriteContext):'';" in BOARD
    assert "onclick=\"event.stopPropagation();hideFavoriteTooltip();openFavoriteBadge(this)\"" in BOARD
    assert '<button class="favorite-performance-main" type="button" aria-haspopup="dialog"' in BOARD
    assert 'aria-label="Open Red Fox Favorite information and performance"' in BOARD
    assert "onclick=\"event.stopPropagation();openSignalGuide('Red Fox Favorite')\"" in BOARD
    assert "overlay.classList.toggle('favorite-guide-open',requested==='Red Fox Favorite');" in BOARD
    assert "overlay.classList.remove('open','favorite-guide-open');" in BOARD


def test_performance_widget_uses_crisp_cropped_brand_gold_frame_and_sans_record_numbers():
    assert ".favorite-performance-record,\n.favorite-performance-rate { font-family:var(--sans)!important; }" in BOARD
    assert 'class="favorite-performance-brand-crop"' in BOARD
    assert ".favorite-performance-desktop .favorite-performance-brand-crop{width:110px;height:27px;}" in BOARD
    assert ".favorite-performance-desktop .favorite-performance-brand{width:136px;left:-15px;top:-9px;}" in BOARD
    assert "object-fit:cover" not in BOARD
    assert "border:1px solid #c79532;" in BOARD
    assert ".sandbox-action-widgets .favorite-performance-main" in BOARD
    assert "background:transparent!important" in BOARD


def test_sport_percentages_are_sorted_best_to_worst_with_ungraded_last():
    assert "b.winPercentage-a.winPercentage||a.label.localeCompare(b.label)" in BOARD
    assert "if(a.winPercentage===null)" in BOARD


def test_drill_in_uses_reference_sans_face_for_all_numeric_surfaces():
    assert "#drill-overlay .game-detail-modal { --mono:var(--sans); }" in BOARD
    assert "#drill-overlay .game-detail-modal *," in BOARD
    assert "#drill-overlay .game-detail-modal .movement-svg text { font-family:var(--sans)!important; }" in BOARD


def test_aggregate_endpoint_is_board_protected_without_exposing_admin_rows():
    production = (ROOT / "deploy" / "nginx-redfox.production.conf").read_text(encoding="utf-8")
    site = (ROOT / "redfox-nginx-site.conf").read_text(encoding="utf-8")
    assert "location = /data/favorite_performance.json" in production
    assert "auth_request /_internal/board-access;" in production
    assert "favorite_performance\\.json" in site
    assert "FAVORITE_PERFORMANCE_URL = '/data/favorite_performance.json'" in BOARD
    board_locations = (ROOT / "deploy" / "redfox-board-locations.conf").read_text(encoding="utf-8")
    assert "location = /data/favorite_performance.json" in board_locations
    assert "auth_request /_internal/board-access;" in board_locations


def test_local_preview_serves_the_existing_favorite_asset():
    server = (ROOT / "serve.py").read_text(encoding="utf-8")
    assert "elif path.startswith('/assets/'):" in server
    assert "self.path = '/site' + self.path" in server
