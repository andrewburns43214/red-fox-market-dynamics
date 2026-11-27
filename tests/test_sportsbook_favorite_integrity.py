import json

import pandas as pd
import pytest

from anomaly_board import _build_history_points
from build_live_recent import _confirm_favorite_with_final_raw_market
from performance_ledger import _attach_clv, _closing_observed_at, _closing_source, _cohort_row, _tracking_lifecycle
from canonical_match import build_canonical_key_from_dk
from red_fox_favorite import (
    _american_implied_probability, _material_addition, _material_removal,
    _review_confirmed, _review_record, favorite_side_quote,
)
from tools.audit_sportsbook_integrity import build_reference_event_map, inspect_review_history


SPORT_MARKETS = [
    ("nfl", "SPREAD", 10), ("ncaaf", "SPREAD", 10), ("nba", "SPREAD", 10),
    ("ncaab", "SPREAD", 10), ("nhl", "MONEYLINE", 15),
    ("mlb", "MONEYLINE", 10), ("ufc", "MONEYLINE", 15),
]
NOW = pd.Timestamp("2026-11-24T21:00:00Z")


@pytest.mark.parametrize("sport,market,minutes", SPORT_MARKETS)
@pytest.mark.parametrize("action", ["addition", "removal"])
def test_ordinary_confirmation_requires_contiguous_matching_side_and_version(sport, market, minutes, action):
    key = (sport, "123", market)
    prior = {
        "sport": sport, "game_id": "123", "market_display": market,
        "recorded_at": (NOW - pd.Timedelta(minutes=minutes)).isoformat(),
        "source_observed_at": (NOW - pd.Timedelta(minutes=minutes)).isoformat(),
        "favorite_side": "North Team +2.5",
        "favorite_rule_version": "v2", "review_action": action, "review_state": "pending",
    }
    quote = {"flagged_side": "North Team +3", "current_line": "+3 (-110)" if market == "SPREAD" else "-110"}
    candidate = pd.Series({
        "favorite_side": "North Team +3", "favorite_rule_version": "v2",
        "market_sides": json.dumps([dict(quote, source_latest_at=NOW.isoformat())]),
    })
    assert _review_confirmed(pd.DataFrame([prior]), key, action, NOW, candidate)
    assert not _review_confirmed(pd.DataFrame([prior]), (*key[:2], "TOTAL"), action, NOW, candidate)
    assert not _review_confirmed(pd.DataFrame([prior]), (sport, "999", market), action, NOW, candidate)
    reused = candidate.copy()
    reused["market_sides"] = json.dumps([dict(quote, source_latest_at=prior["source_observed_at"])])
    reused["state_as_of_utc"] = NOW.isoformat()
    assert not _review_confirmed(pd.DataFrame([prior]), key, action, NOW, reused)
    if action == "removal":
        assert _review_confirmed(pd.DataFrame([prior]), key, action, NOW, reused, observed_row=candidate)
    no_source = pd.DataFrame([dict(prior, source_observed_at="")])
    assert not _review_confirmed(no_source, key, action, NOW, candidate)
    for mismatch in (
        {"favorite_side": "South Team -3"}, {"favorite_rule_version": "v1"},
        {"favorite_rule_version": ""}, {"review_state": "cleared"},
        {"review_action": "removal" if action == "addition" else "addition"},
    ):
        interruption = {**prior, **mismatch, "recorded_at": (NOW - pd.Timedelta(minutes=1)).isoformat()}
        assert not _review_confirmed(pd.DataFrame([prior, interruption]), key, action, NOW, candidate)


def test_review_and_entry_use_subject_side_even_if_the_board_leader_differs():
    row = pd.Series({
        "favorite_side": "North Team", "favorite_rule_version": "v2",
        "current_line": "South Team @ -105",
        "market_sides": json.dumps([
            {"flagged_side": "North Team", "current_line": "North Team @ -115",
             "source_latest_at": "2026-11-24T20:59:00Z"},
            {"flagged_side": "South Team", "current_line": "South Team @ -105"},
        ]),
    })
    assert favorite_side_quote(row) == "North Team @ -115"
    review = _review_record(row, NOW.isoformat(), "pending", "addition", "Reason")
    assert review["favorite_side"] == "North Team"
    assert review["source_observed_at"] == "2026-11-24T20:59:00Z"


def test_reprocessed_capture_does_not_restart_confirmation_interval():
    prior = {
        "sport": "nfl", "game_id": "123", "market_display": "SPREAD",
        "recorded_at": "2026-11-24T20:40:00Z", "source_observed_at": "2026-11-24T20:39:00Z",
        "favorite_side": "North Team +3", "favorite_rule_version": "v5",
        "review_action": "addition", "review_state": "pending",
    }
    records = pd.DataFrame([prior, dict(prior, recorded_at="2026-11-24T20:55:00Z")])
    current = pd.Series({
        "favorite_side": "North Team +3", "favorite_rule_version": "v5",
        "market_sides": json.dumps([{
            "flagged_side": "North Team +3", "source_latest_at": "2026-11-24T20:59:00Z",
        }]),
    })
    assert not _review_confirmed(records, ("nfl", "123", "SPREAD"), "addition", NOW, current)


def test_closing_provenance_handles_mixed_evidence_without_inventing_a_timestamp():
    row = pd.Series({"closing_line": "+3.5", "closing_price": "", "final_pregame_price": "-110"})
    side = {"source_latest_at": "2026-11-24T21:00:00Z"}
    assert _closing_source(row, "closing_line", "+3") == "retained_close_unverified"
    assert _closing_observed_at(row, "closing_line", "+3", side) == ""
    assert _closing_source(row, "closing_price", "-110") == "final_pregame_fallback"
    assert _closing_observed_at(row, "closing_price", "-110", side) == side["source_latest_at"]


def test_entry_for_new_records_is_bound_to_frozen_side_and_version():
    base = {"sport": "nba", "game_id": "123", "market_display": "MONEYLINE", "favorite_state": "qualified"}
    history = pd.DataFrame([
        dict(base, favorite_side="South Team", favorite_rule_version="v1",
             first_qualified_line="+120", recorded_at="2026-11-24T20:00:00Z", entry_observed_at="2026-11-24T19:59:00Z"),
        dict(base, favorite_side="North Team", favorite_rule_version="v2",
             first_qualified_line="-120", recorded_at="2026-11-24T20:30:00Z", entry_observed_at="2026-11-24T20:29:00Z"),
    ])
    result = _tracking_lifecycle(history, pd.Series(dict(base, favorite_side="North Team", favorite_rule_version="v2")), NOW)
    assert result["first_qualified_line"] == "-120"
    assert result["entry_observed_at"] == "2026-11-24T20:29:00Z"


@pytest.mark.parametrize("sport,market,minutes", SPORT_MARKETS)
def test_reference_mapping_requires_exact_event_and_refuses_ambiguity(sport, market, minutes):
    game = "North Team vs South Team" if sport == "ufc" else "North Team @ South Team"
    kickoff = "2026-11-25T01:00:00Z"
    board = {"sport": sport, "game_id": "123", "game": game, "kickoff_iso": "2026-11-24T20:00:00-05:00"}
    provider_key = build_canonical_key_from_dk(game, sport, kickoff)
    assert provider_key and provider_key != f"{sport}|123"
    reference = {"canonical_key": provider_key, "commence_time": kickoff}
    matches, collisions = build_reference_event_map([board], [reference])
    assert list(matches.values()) == [(sport, "123")]
    assert not collisions
    matches, collisions = build_reference_event_map([board, dict(board, game_id="456")], [reference])
    assert not matches
    assert len(collisions) == 1
    matches, _ = build_reference_event_map([board], [dict(reference, commence_time="2026-11-25T03:00:00Z")])
    assert not matches


def test_old_review_export_can_expose_but_cannot_prove_a_cross_side_promotion():
    pending = {"sport": "nfl", "game_id": "123", "market_display": "SPREAD",
               "recorded_at": "2026-11-24T20:00:00Z", "review_state": "pending",
               "review_action": "addition", "favorite_side": "North Team +3"}
    applied = dict(pending, recorded_at="2026-11-24T20:10:00Z", review_state="applied", favorite_side="South Team -3")
    result = inspect_review_history([pending, applied])
    assert result["counts"]["adjacent_pending_side_differs"] == 1
    assert result["counts"]["pending_missing_version_or_source_timestamp"] == 1
    assert result["adjacent_side_differences"][0]["gap_minutes"] == 10


@pytest.mark.parametrize("opener,current,material", [
    ("+100", "-100", False), ("-100", "+100", False),
    ("+105", "-105", False), ("+107", "-108", True),
    ("-105", "-120", True), ("+120", "+106", False),
])
def test_material_moneyline_addition_preserves_fifteen_cent_threshold(opener, current, material):
    row = pd.Series({
        "sport": "mlb", "market_display": "MONEYLINE",
        "favorite_supporting_evidence": json.dumps({
            "bets_pct": 30, "money_pct": 30, "open_line": opener, "current_line": current,
        }),
    })
    assert _material_addition(row)[0] == material
    assert _american_implied_probability(-100) == _american_implied_probability(100) == 50


@pytest.mark.parametrize("opener,current,confirmed", [
    ("-100", "+100", True), ("+100", "-100", True),
    ("+108", "-107", True), ("-107", "+108", False),
])
def test_kickoff_handoff_uses_consistent_even_money_direction(opener, current, confirmed):
    candidate = pd.Series({
        "market_display": "MONEYLINE", "favorite_side": "Team A",
        "kickoff_iso": "2026-11-24T22:00:00Z",
        "market_sides": json.dumps([{"flagged_side": "Team A", "current_line": opener}]),
    })
    raw = pd.Series({"state_as_of_utc": "2026-11-24T21:55:00Z",
                     "market_sides": json.dumps([{"flagged_side": "Team A", "current_line": current}])})
    assert (_confirm_favorite_with_final_raw_market(candidate, raw) is not None) == confirmed
    for observed, badge in (("", "CLEAN"), ("2026-11-24T18:00:00Z", "CLEAN"),
                            ("2026-11-24T22:05:00Z", "CLEAN"), ("2026-11-24T21:55:00Z", "RETAINED")):
        assert _confirm_favorite_with_final_raw_market(
            candidate, pd.Series(raw.to_dict() | {
                "state_as_of_utc": observed, "data_badge": badge,
            })
        ) is None


def test_clv_requires_verified_closing_provenance_and_excludes_retained_estimates_from_coverage():
    evidence = {
        "market": "MONEYLINE", "decision_price": "+120", "closing_price": "-100",
        "decision_quote_observed_at": "2026-11-24T20:00:00Z", "scheduled_start": "2026-11-24T22:00:00Z",
        "closing_price_observed_at": "2026-11-24T21:59:00Z", "clv": "99", "grade": "W",
        "clv_available": "yes", "clv_method": "retained_source",
    }
    result = pd.DataFrame([
        dict(evidence, closing_price_source="verified_closing_capture"),
        dict(evidence, closing_price_source="final_pregame_fallback"),
        dict(evidence, closing_price_source="verified_closing_capture", closing_price_observed_at=""),
        dict(evidence, closing_price_source="verified_closing_capture", closing_price_observed_at="2026-11-24T22:01:00Z"),
    ])
    for index, row in result.iterrows():
        _attach_clv(result, index, row)
    assert result.clv_available.tolist() == ["yes", "no", "no", "no"]
    assert result.clv.tolist() == ["4.5455", "99", "99", "99"]
    cohort = _cohort_row(result, "overall", "all", "all")
    assert cohort["clv_available"] == 1
    assert cohort["average_clv"] == pytest.approx(4.5455)


def test_unchanged_observations_are_timestamped_persistence():
    quote = {"current_line": "-105", "bets_pct": 35, "money_pct": 35}
    history = pd.DataFrame([
        dict(quote, timestamp="2026-11-24T20:00:00Z"),
        dict(quote, timestamp="2026-11-24T20:00:00Z"),
        dict(quote, timestamp="2026-11-24T20:10:00Z"),
        dict(quote, timestamp="2026-11-24T20:20:00Z"),
    ])
    points = _build_history_points(history, "MONEYLINE")
    assert len(points) == 3
    assert len({(p["value"], p["bets_pct"], p["money_pct"]) for p in points}) == 1


@pytest.mark.parametrize("sport,market,old,new,opponent,destructive", [
    ("nfl", "SPREAD", "+6.5", "+6.5", "-6.5", False),
    ("nfl", "SPREAD", "-3.5", "+6.5", "+3.5", True),
    ("nfl", "SPREAD", "+6.5", "-3.5", "-6.5", False),
    ("nba", "SPREAD", "+6.5", "+6.5", "-6.5", False),
    ("ncaab", "SPREAD", "+6.5", "+6.5", "-6.5", False),
    ("nfl", "MONEYLINE", "-100", "+100", "-115", False),
    ("mlb", "MONEYLINE", "-107", "+108", "-115", True),
    ("ncaab", "MONEYLINE", "-107", "+107", "-115", False),
    ("nba", "MONEYLINE", "+100", "-100", "+115", False),
    ("ncaaf", "SPREAD", "+6.5", "+6.5", "-6.5", False),
    ("mlb", "MONEYLINE", "+110", "-105", "-120", False),
    ("mlb", "MONEYLINE", "-105", "+110", "-115", True),
])
def test_closing_comparison_uses_team_even_when_selected_quote_crosses_zero(sport,market,old,new,opponent,destructive):
    source = pd.Series({
        "sport":sport,"game_id":"123","market_display":market,
        "favorite_side":"Colts "+old if market=="SPREAD" else "Colts",
        "flagged_side":"Giants "+opponent if market=="SPREAD" else "Giants","current_line":opponent,
        "market_sides":json.dumps([
            {"flagged_side":"Giants "+opponent if market=="SPREAD" else "Giants","current_line":opponent},
            {"flagged_side":"Colts "+old if market=="SPREAD" else "Colts","current_line":old},
        ]),
    })
    row = source.copy()
    row["supported_side"] = "Colts "+new if market=="SPREAD" else "Colts"
    row["market_sides"] = json.dumps([{"flagged_side":row["supported_side"],"current_line":new}])
    assert _material_removal(row,source)[:2] == (False,destructive)


def test_future_freeze_recovers_legacy_entry_from_first_capture_only():
    base={"sport":"nfl","game_id":"123","market_display":"SPREAD","favorite_state":"qualified",
          "favorite_side":"Colts +6.5","favorite_rule_version":"v2","first_qualified_line":"-6.5 (-115)",
          "first_qualified_at":"2026-11-24T20:00:00Z","source_observed_at":"2026-11-24T19:59:00Z"}
    history=pd.DataFrame([
        dict(base,recorded_at=base["first_qualified_at"],supporting_evidence='{"current_line":"+6.5 (-104)"}'),
        dict(base,recorded_at="2026-11-24T20:10:00Z",supporting_evidence='{"current_line":"+5.5 (-105)"}'),
    ])
    row=pd.Series(dict(base,favorite_side="Colts -3.5"))
    verified=_tracking_lifecycle(history,row,NOW,recover_entry=True)
    assert verified["first_qualified_line"]=="+6.5 (-104)"
    assert verified["entry_observed_at"]==base["source_observed_at"]
    assert _tracking_lifecycle(history,row,NOW)["first_qualified_line"]=="-6.5 (-115)"
