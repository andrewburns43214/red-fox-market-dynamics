import json
from pathlib import Path

import pandas as pd
import pytest

from red_fox_favorite import CONFIG, apply_red_fox_favorites, update_favorite_tracking
from build_live_recent import ensure_output_columns
from audit_directional_integrity import audit as audit_directional_integrity


def side(name, current, *, bets=35, money=30, reaction="Contrarian", direction="TOWARD",
         observations=4, clean=True, kpi=True, action_type="CONTRARIAN CANDIDATE",
         action_side=None, path="One-Way", line_move=1.0, price_move=3.0, context="",
         active_worsening=False, open_line=None, evidence_role="", key_number_pinned="",
         return_toward_open=False, line_dir_changes=0):
    return {
        "flagged_side": name, "bets_pct": bets, "money_pct": money,
        "open_line": open_line or current, "current_line": current, "reaction": reaction,
        "response_direction": direction, "observation_count": observations,
        "data_badge": "Clean" if clean else "Thin", "kpi_eligible": kpi,
        "action_type": action_type, "action_side": action_side or name, "path": path,
        "line_move_abs": line_move, "price_move_pct": price_move,
        "context_chips": context, "whipsaw_recovered": "Whipsaw Recovered" in context,
        "active_worsening_reversal": active_worsening,
        "evidence_role": evidence_role, "key_number_pinned": key_number_pinned,
        "return_toward_open": return_toward_open, "line_dir_changes": line_dir_changes,
    }


def market(sport, market_name, sides, *, game_id="1", game="Away @ Home", rank=1, supported_side=None):
    if supported_side is None:
        supported_side = sides[0]["flagged_side"]
    return {
        "sport": sport, "game_id": game_id, "game": game, "market_display": market_name,
        "board_rank": rank, "current_line": sides[0]["current_line"],
        "market_sides": json.dumps(sides), "supported_side": supported_side,
    }


def pair_for(candidate, opponent=None):
    opponent = opponent or side("Opponent -4.5", "-4.5 (-110)", bets=65, money=70,
                                reaction="Watch", direction="AGAINST", kpi=False,
                                action_type="OBSERVE ONLY")
    return [candidate, opponent]


def result(rows):
    return apply_red_fox_favorites(pd.DataFrame(rows), as_of="2026-09-06T20:00:00Z")


def is_favorite(frame, index=0):
    return frame.iloc[index].red_fox_favorite == "true"


@pytest.mark.parametrize("name,current", [("Dog +1", "+1 (-110)"), ("Dog +8", "+8 (-110)"),
                                            ("Small favorite -2.5", "-2.5 (-110)")])
def test_spread_contrarian_paths_and_boundaries_qualify(name, current):
    assert is_favorite(result([market("nfl", "SPREAD", pair_for(side(name, current)))]))


@pytest.mark.parametrize("current", ["PK (-110)", "+8.5 (-110)", "-9 (-110)"])
def test_spread_outside_v1_range_is_rejected(current):
    assert not is_favorite(result([market("nfl", "SPREAD", pair_for(side("Candidate " + current.split()[0], current)))]))


@pytest.mark.parametrize("sport", ["mlb", "nhl", "ufc"])
@pytest.mark.parametrize("price", [-165, 120])
def test_primary_moneyline_sports_and_boundaries_qualify(sport, price):
    candidate = side("Candidate", f"{price:+d}", price_move=3.1)
    opponent = side("Opponent", f"{-price:+d}", bets=65, money=70, reaction="Watch",
                    direction="AGAINST", kpi=False, action_type="OBSERVE ONLY")
    assert is_favorite(result([market(sport, "MONEYLINE", [candidate, opponent])]))


@pytest.mark.parametrize("price", [-166, 121, 125, 126])
def test_moneyline_outside_final_v1_range_is_rejected(price):
    candidate = side("Candidate", f"{price:+d}")
    opponent = side("Opponent", "+100", bets=65, money=70, reaction="Watch", direction="LIMITED", kpi=False)
    assert not is_favorite(result([market("mlb", "MONEYLINE", [candidate, opponent])]))


def florida_state_path_b(*, active_worsening=False, pressure_reaction="Freeze", pressure_direction="AGAINST"):
    candidate = side(
        "Florida State +3", "+3 (-115)", open_line="+3 (-110)", bets=23, money=25,
        reaction="Watch", direction="LIMITED", kpi=False, action_type="OBSERVE ONLY",
        path="Held", line_move=0, price_move=1.107, context="K3 | Whipsaw Recovered",
        active_worsening=active_worsening, evidence_role="Resistance Side", key_number_pinned="K3",
    )
    pressure = side(
        "SMU -3", "-3 (-105)", open_line="-3 (-110)", bets=81, money=75,
        reaction=pressure_reaction, direction=pressure_direction, kpi=False,
        action_type="OBSERVE ONLY", path="Held", line_move=0, price_move=1.162,
        context="K3 | Public Pressure | Whipsaw Recovered", evidence_role="Pressure Side",
        key_number_pinned="K3",
    )
    return candidate, pressure


def test_florida_state_key_three_recovered_freeze_without_confirmed_supported_side_is_not_favorite():
    candidate, pressure = florida_state_path_b()
    assert pressure["kpi_eligible"] is False
    assert pressure["action_type"] == "OBSERVE ONLY"
    assert pressure["key_number_pinned"] == "K3"
    frame = result([market("ncaaf", "SPREAD", [candidate, pressure], game_id="smu-fsu",
                           game="SMU @ Florida State", supported_side="")])
    assert not is_favorite(frame)


def test_freeze_resistance_path_qualifies_when_engine_confirms_the_fade_side():
    candidate, pressure = florida_state_path_b()
    pressure["kpi_eligible"] = True
    pressure["action_type"] = "FADE CANDIDATE"
    pressure["action_side"] = "Florida State +3"
    frame = result([market("ncaaf", "SPREAD", [candidate, pressure], game_id="smu-fsu",
                           game="SMU @ Florida State", supported_side="Florida State +3")])
    assert is_favorite(frame)
    assert frame.iloc[0].favorite_pathway == "low_support_freeze"
    assert frame.iloc[0].favorite_side == "Florida State +3"


def test_positive_secondary_moneyline_is_confirmation_only_when_team_gets_points():
    ml = market("nfl", "MONEYLINE", pair_for(side("Away", "+124")))
    spread = market("nfl", "SPREAD", pair_for(side("Away +3", "+3 (-110)")), rank=2)
    frame = result([ml, spread])
    assert not is_favorite(frame, 0)
    assert is_favorite(frame, 1)
    assert frame.iloc[1].favorite_side == "Away +3"


def test_ufc_freeze_path_is_not_favorite_eligible_but_movement_backed_contrarian_remains_eligible():
    protected = side("Underdog", "+110", bets=19, money=10, reaction="Watch", direction="LIMITED",
                     kpi=False, action_type="OBSERVE ONLY", path="Held", line_move=0, price_move=0)
    pressure = side("Favorite", "-130", bets=81, money=90, reaction="Freeze", direction="LIMITED",
                    action_type="FADE CANDIDATE", action_side="Underdog", path="Held",
                    evidence_role="Pressure Side", line_move=0, price_move=0)
    assert not is_favorite(result([market("ufc", "MONEYLINE", [protected, pressure], supported_side="Underdog")]))
    contrarian = side("Underdog", "+110", bets=19, money=10, reaction="Contrarian",
                      direction="TOWARD", price_move=3.0)
    assert is_favorite(result([market("ufc", "MONEYLINE", pair_for(contrarian), supported_side="Underdog")]))


@pytest.mark.parametrize("candidate_change,pressure_change", [
    ({"return_toward_open": True}, {}),
    ({}, {"return_toward_open": True}),
    ({"line_dir_changes": 5}, {}),
    ({}, {"line_dir_changes": 5}),
])
def test_freeze_favorite_rejects_retracement_and_excessive_price_churn(candidate_change, pressure_change):
    candidate = side("Underdog", "+110", bets=19, money=10, reaction="Watch", direction="LIMITED",
                     kpi=False, action_type="OBSERVE ONLY", path="Held", line_move=0, price_move=0,
                     evidence_role="Resistance Side", **candidate_change)
    pressure = side("Favorite", "-130", bets=81, money=90, reaction="Freeze", direction="LIMITED",
                    action_type="FADE CANDIDATE", action_side="Underdog", path="Held",
                    line_move=0, price_move=0, evidence_role="Pressure Side", **pressure_change)
    assert not is_favorite(result([market("mlb", "MONEYLINE", [candidate, pressure], supported_side="Underdog")]))


def test_freeze_favorite_requires_strong_pressure_and_persistent_resistance():
    candidate = side("Underdog", "+110", bets=19, money=10, reaction="Watch", direction="LIMITED",
                     kpi=False, action_type="OBSERVE ONLY", path="", line_move=0, price_move=2.6,
                     evidence_role="Resistance Side")
    pressure = side("Favorite", "-130", bets=81, money=90, reaction="Freeze", direction="AGAINST",
                    action_type="FADE CANDIDATE", action_side="Underdog", path="",
                    line_move=0, price_move=2.6, evidence_role="Pressure Side")
    assert is_favorite(result([market("mlb", "MONEYLINE", [candidate, pressure], supported_side="Underdog")]))
    weak_pressure = dict(pressure, bets_pct=79)
    assert not is_favorite(result([market("mlb", "MONEYLINE", [candidate, weak_pressure], supported_side="Underdog")]))
    small_move = dict(pressure, price_move_pct=2.4)
    assert not is_favorite(result([market("mlb", "MONEYLINE", [candidate, small_move], supported_side="Underdog")]))


def test_rule_tightening_keeps_v2_version_and_tracking_history_append_only(tmp_path):
    assert CONFIG.version == "red_fox_favorite_v2"
    legacy = result([market("mlb", "MONEYLINE", pair_for(side("Away", "+120")))])
    update_favorite_tracking(legacy, tmp_path, as_of="2026-09-06T20:00:00Z")
    newly_ineligible = legacy.copy()
    newly_ineligible["red_fox_favorite"] = "false"
    newly_ineligible["favorite_state"] = "not_qualified"
    update_favorite_tracking(newly_ineligible, tmp_path, as_of="2026-09-06T20:01:00Z")
    ledger = pd.read_csv(tmp_path / "red_fox_favorite_tracking.csv", dtype=str, keep_default_na=False)
    assert ledger.favorite_state.tolist() == ["qualified", "not_qualified"]
    assert ledger.favorite_rule_version.tolist() == ["red_fox_favorite_v2", "red_fox_favorite_v2"]


def test_similar_splits_without_resistance_or_protection_do_not_qualify():
    candidate, pressure = florida_state_path_b(pressure_reaction="Follow", pressure_direction="TOWARD")
    candidate["response_direction"] = "AGAINST"
    assert not is_favorite(result([market("nfl", "SPREAD", [candidate, pressure])]))


def test_path_b_active_destructive_reversal_does_not_qualify():
    candidate, pressure = florida_state_path_b(active_worsening=True)
    assert not is_favorite(result([market("ncaaf", "SPREAD", [candidate, pressure])]))


def test_path_b_material_moneyline_contradiction_does_not_qualify():
    candidate, pressure = florida_state_path_b()
    spread = market("ncaaf", "SPREAD", [candidate, pressure], game_id="smu-fsu", game="SMU @ Florida State")
    moneyline = market("ncaaf", "MONEYLINE", [
        side("Florida State", "+120", bets=23, money=25, reaction="Watch", direction="AGAINST", kpi=False, price_move=3.0),
        side("SMU", "-140", bets=77, money=75, reaction="Follow", direction="TOWARD", kpi=False, price_move=3.0),
    ], game_id="smu-fsu", game="SMU @ Florida State", rank=2)
    assert not is_favorite(result([spread, moneyline]))


def test_moderate_support_movement_without_directional_market_read_is_not_favorite():
    candidate = side("Confirming side -3", "-3 (-110)", bets=55, money=52,
                     reaction="Watch", direction="TOWARD", kpi=False, line_move=1.0)
    opponent = side("Other side +3", "+3 (-110)", bets=45, money=48,
                    reaction="Watch", direction="AGAINST", kpi=False)
    frame = result([market("ncaaf", "SPREAD", [candidate, opponent], supported_side="")])
    assert not is_favorite(frame)


def test_favorite_requires_supported_side_to_match_the_qualified_side():
    candidate = side("Away +3", "+3 (-110)")
    opponent = side("Home -3", "-3 (-110)", bets=65, money=70, reaction="Watch",
                    direction="AGAINST", kpi=False, action_type="OBSERVE ONLY")
    assert not is_favorite(result([market("nfl", "SPREAD", [candidate, opponent], supported_side="")]))
    assert not is_favorite(result([market("nfl", "SPREAD", [candidate, opponent], supported_side="Home -3")]))
    assert is_favorite(result([market("nfl", "SPREAD", [candidate, opponent], supported_side="Away +3")]))


def test_directional_integrity_audit_reports_favorite_without_green_side():
    candidate = side("Away +3", "+3 (-110)")
    opponent = side("Home -3", "-3 (-110)", bets=65, money=70, reaction="Watch",
                    direction="AGAINST", kpi=False, action_type="OBSERVE ONLY")
    legacy = pd.DataFrame([market("nfl", "SPREAD", [candidate, opponent], supported_side="")])
    legacy["red_fox_favorite"] = "true"
    legacy["favorite_side"] = "Away +3"
    report = audit_directional_integrity(legacy)
    assert report["red_fox_favorites"] == 1
    assert report["favorites_without_supported_side"] == 1
    assert report["favorites_on_different_supported_side"] == 0
    assert report["anomalies"][0]["issue"] == "Favorite has no confirmed supported side"


def test_directional_integrity_audit_accepts_an_empty_board():
    report = audit_directional_integrity(pd.DataFrame(columns=["market_display", "market_sides"]))
    assert report["published_markets"] == 0
    assert report["red_fox_favorites"] == 0
    assert not report["anomalies"]


def test_heavy_public_follow_and_nonmeaningful_move_do_not_qualify():
    heavy = side("Public side -3", "-3 (-110)", bets=75, money=80, reaction="Follow", kpi=True)
    other = side("Other +3", "+3 (-110)", bets=25, money=20, reaction="Watch", direction="AGAINST", kpi=False)
    assert not is_favorite(result([market("ncaaf", "SPREAD", [heavy, other])]))
    moderate = side("Balanced -3", "-3 (-110)", bets=55, money=55, reaction="Watch",
                    direction="TOWARD", kpi=False, line_move=0.25, price_move=1.0)
    other = side("Balanced +3", "+3 (-110)", bets=45, money=45, reaction="Watch", direction="LIMITED", kpi=False)
    assert not is_favorite(result([market("nba", "SPREAD", [moderate, other])]))


def test_secondary_moneyline_uses_spread_size_gate_and_routes_point_receivers_to_spread():
    ml = market("nfl", "MONEYLINE", pair_for(side("Away", "+120")))
    spread4 = market("nfl", "SPREAD", pair_for(side("Away +4", "+4 (-110)")), rank=2)
    routed = result([ml, spread4])
    assert not is_favorite(routed, 0)
    assert is_favorite(routed, 1)
    spread45 = market("nfl", "SPREAD", pair_for(side("Away +4.5", "+4.5 (-110)")), rank=2)
    assert not is_favorite(result([ml, spread45]), 0)


def test_secondary_moneyline_can_remain_favorite_when_team_is_laying_points():
    ml = market("ncaab", "MONEYLINE", pair_for(side("Home", "-120")))
    spread = market("ncaab", "SPREAD", pair_for(side("Home -2", "-2 (-110)")), rank=2)
    assert is_favorite(result([ml, spread]), 0)


def test_negative_money_football_favorite_requires_strong_cross_market_final_state():
    texas_ml = market("ncaaf", "MONEYLINE", [
        side("Texas", "-142", bets=39, money=45, path="Whipsaw", line_dir_changes=13,
             return_toward_open=True),
        side("Ohio State", "+105", bets=61, money=55, reaction="Watch", direction="AGAINST", kpi=False),
    ], game_id="texas", game="Ohio State @ Texas", supported_side="Texas")
    texas_spread = market("ncaaf", "SPREAD", [
        side("Texas -2.5", "-2.5 (-112)", bets=49, money=60, path="Whipsaw", line_dir_changes=13,
             return_toward_open=True),
        side("Ohio State +2.5", "+2.5 (-108)", bets=51, money=40, reaction="Watch", direction="AGAINST", kpi=False),
    ], game_id="texas", game="Ohio State @ Texas", rank=2, supported_side="Texas -2.5")
    assert not is_favorite(result([texas_ml, texas_spread]), 0)

    houston_ml = market("nfl", "MONEYLINE", [
        side("HOU Texans", "-110", bets=24, money=27),
        side("BUF Bills", "+100", bets=76, money=73, reaction="Watch", direction="AGAINST", kpi=False),
    ], game_id="houston", game="BUF Bills @ HOU Texans", supported_side="HOU Texans")
    houston_spread = market("nfl", "SPREAD", [
        side("HOU Texans -1.5", "-1.5 (+102)", bets=24, money=28),
        side("BUF Bills +1.5", "+1.5 (-122)", bets=76, money=72, reaction="Watch", direction="AGAINST", kpi=False),
    ], game_id="houston", game="BUF Bills @ HOU Texans", rank=2, supported_side="HOU Texans -1.5")
    assert is_favorite(result([houston_ml, houston_spread]), 0)


def test_tighter_moneyline_favorite_gate_does_not_change_point_taking_spreads():
    candidate = side("Appalachian State +6.5", "+6.5 (-108)", bets=41, money=19, path="Whipsaw")
    opponent = side("East Carolina -6.5", "-6.5 (-112)", bets=59, money=81,
                    reaction="Watch", direction="AGAINST", kpi=False)
    assert is_favorite(result([market("ncaaf", "SPREAD", [candidate, opponent],
                                      game="Appalachian State @ East Carolina")]))


def test_positive_secondary_moneyline_is_not_published_at_pickem():
    ml = market("nba", "MONEYLINE", pair_for(side("Away", "+105")))
    spread = market("nba", "SPREAD", pair_for(side("Away PK", "PK (-110)")), rank=2)
    frame = result([ml, spread])
    assert not is_favorite(frame, 0)
    assert not is_favorite(frame, 1)


@pytest.mark.parametrize("direction,price_move,expected,state", [
    ("TOWARD", 3.0, True, "confirmation"),
    ("LIMITED", 1.0, True, "neutral"),
    ("AGAINST", 3.0, False, ""),
])
def test_corresponding_moneyline_confirmation_neutral_and_contradiction(direction, price_move, expected, state):
    spread = market("nfl", "SPREAD", pair_for(side("Away +3", "+3 (-110)")))
    away_ml = side("Away", "+120", bets=35, money=30, reaction="Watch", direction=direction,
                   kpi=False, price_move=price_move)
    home_ml = side("Home", "-140", bets=65, money=70, reaction="Watch", direction="LIMITED", kpi=False)
    ml = market("nfl", "MONEYLINE", [away_ml, home_ml], rank=2)
    frame = result([spread, ml])
    assert is_favorite(frame, 0) is expected
    assert frame.iloc[0].favorite_cross_market_state == state


def test_corresponding_moneyline_can_confirm_outside_its_own_favorite_range():
    spread = market("ncaab", "SPREAD", pair_for(side("Away +4", "+4 (-110)")))
    ml = market("ncaab", "MONEYLINE", [
        side("Away", "+180", direction="TOWARD", price_move=3.0),
        side("Home", "-220", bets=65, money=70, reaction="Watch", direction="AGAINST", kpi=False),
    ], rank=2)
    frame = result([spread, ml])
    assert is_favorite(frame, 0)
    assert frame.iloc[0].favorite_cross_market_state == "confirmation"
    assert not is_favorite(frame, 1)


def test_path_aware_whipsaw_allows_intact_or_recovered_but_not_erased_move():
    intact = side("Away +6", "+6 (-110)", path="Whipsaw", direction="TOWARD")
    assert is_favorite(result([market("nfl", "SPREAD", pair_for(intact))]))
    recovered = side("Away +6", "+6 (-110)", path="One-Way", context="Whipsaw Recovered")
    assert is_favorite(result([market("nfl", "SPREAD", pair_for(recovered))]))
    erased = side("Away +7", "+7 (-110)", path="Whipsaw", direction="LIMITED")
    assert not is_favorite(result([market("nfl", "SPREAD", pair_for(erased))]))
    worsening = side("Away +6", "+6 (-110)", path="Whipsaw", direction="TOWARD", active_worsening=True)
    assert not is_favorite(result([market("nfl", "SPREAD", pair_for(worsening))]))


@pytest.mark.parametrize("changes", [
    {"clean": False}, {"observations": 2}, {"context": "Market Lag"},
])
def test_data_quality_history_and_market_lag_block(changes):
    candidate = side("Away +3", "+3 (-110)", **changes)
    assert not is_favorite(result([market("nfl", "SPREAD", pair_for(candidate))]))


def test_totals_are_never_favorite_eligible():
    assert not is_favorite(result([market("nfl", "TOTAL", pair_for(side("Under 45", "U 45 (-110)")))]))


def test_rank_and_row_order_are_unchanged():
    normal = market("nfl", "TOTAL", pair_for(side("Under 45", "U 45 (-110)")), game_id="1", rank=1)
    favorite = market("mlb", "MONEYLINE", pair_for(side("Away", "+120")), game_id="2", rank=8)
    frame = result([normal, favorite])
    assert frame.board_rank.tolist() == [1, 8]
    assert frame.game_id.tolist() == ["1", "2"]


def test_tracking_persists_first_qualification_and_records_subsequent_snapshots(tmp_path: Path):
    frame = result([market("mlb", "MONEYLINE", pair_for(side("Away", "+120")))])
    first = update_favorite_tracking(frame, tmp_path, as_of="2026-09-06T20:00:00Z")
    later = apply_red_fox_favorites(frame.drop(columns=[column for column in frame if column.startswith("favorite_") or column == "red_fox_favorite"]), as_of="2026-09-06T20:01:00Z")
    second = update_favorite_tracking(later, tmp_path, as_of="2026-09-06T20:01:00Z")
    assert second.iloc[0].favorite_first_qualified_at == first.iloc[0].favorite_first_qualified_at
    ledger = pd.read_csv(tmp_path / "red_fox_favorite_tracking.csv", dtype=str)
    assert len(ledger) == 2
    assert set(ledger.favorite_rule_version) == {CONFIG.version}


def test_tracking_retains_full_favorite_handoff_candidate_after_board_disappearance(tmp_path: Path):
    source = market("ncaaf", "SPREAD", pair_for(side("Florida Atlantic +4", "+4 (-108)", bets=20, money=19)),
                    game_id="34603696", game="Navy @ Florida Atlantic")
    # Keep this handoff test outside the final-hour visibility lock; final-hour
    # behavior has dedicated coverage below.
    source["kickoff_iso"] = "2026-09-13T00:35:00Z"
    source["state_as_of_utc"] = "2026-09-12T23:11:03Z"
    qualified = result([source])
    update_favorite_tracking(qualified, tmp_path, as_of="2026-09-12T23:11:50Z")
    update_favorite_tracking(pd.DataFrame(columns=qualified.columns), tmp_path, as_of="2026-09-12T23:21:43Z")
    candidates = pd.read_csv(tmp_path / "red_fox_favorite_freeze_candidates.csv", dtype=str, keep_default_na=False)
    assert len(candidates) == 1
    assert candidates.iloc[0].favorite_side == "Florida Atlantic +4"
    assert candidates.iloc[0].state_as_of_utc == "2026-09-12T23:11:03Z"


def test_tracking_records_when_a_current_market_loses_favorite_status(tmp_path: Path):
    qualified = result([market("mlb", "MONEYLINE", pair_for(side("Away", "+120")))])
    update_favorite_tracking(qualified, tmp_path, as_of="2026-09-06T20:00:00Z")
    lost = qualified.copy()
    lost["red_fox_favorite"] = "false"
    lost["favorite_state"] = "not_qualified"
    update_favorite_tracking(lost, tmp_path, as_of="2026-09-06T20:01:00Z")
    ledger = pd.read_csv(tmp_path / "red_fox_favorite_tracking.csv", dtype=str)
    assert ledger.favorite_state.tolist() == ["qualified", "not_qualified"]
    assert ledger.iloc[-1].disappearance_reason == "current qualification gates no longer satisfied"


def test_visibility_invalidated_jacksonville_state_is_not_a_favorite():
    source = market(
        "ncaaf", "SPREAD",
        pair_for(side("Jacksonville State +1.5", "+1.5 (-105)", bets=34, money=24)),
        game_id="34603681", game="Jacksonville State @ Ohio",
    )
    assert not is_favorite(result([source]))


def test_final_hour_rejects_unconfirmed_late_addition(tmp_path: Path):
    source = market("ncaaf", "SPREAD", pair_for(side("Away +3", "+3 (-110)", bets=30, money=20)))
    source["kickoff_iso"] = "2026-09-06T21:00:00Z"
    qualified = apply_red_fox_favorites(pd.DataFrame([source]), as_of="2026-09-06T20:10:00Z")
    locked = update_favorite_tracking(qualified, tmp_path, as_of="2026-09-06T20:10:00Z")
    assert locked.iloc[0].red_fox_favorite == "false"
    assert "T-60" in locked.iloc[0].favorite_reason


def test_final_hour_keeps_confirmed_favorite_from_falling_off(tmp_path: Path):
    source = market("ncaaf", "SPREAD", pair_for(side("Away +3", "+3 (-110)", bets=30, money=20)))
    source["kickoff_iso"] = "2026-09-06T21:00:00Z"
    for captured in ("2026-09-06T19:40:00Z", "2026-09-06T19:50:00Z"):
        qualified = apply_red_fox_favorites(pd.DataFrame([source]), as_of=captured)
        update_favorite_tracking(qualified, tmp_path, as_of=captured)
    lost = apply_red_fox_favorites(pd.DataFrame([source]), as_of="2026-09-06T20:10:00Z")
    lost["red_fox_favorite"] = "false"
    lost["favorite_state"] = "not_qualified"
    locked = update_favorite_tracking(lost, tmp_path, as_of="2026-09-06T20:10:00Z")
    assert locked.iloc[0].red_fox_favorite == "true"
    assert locked.iloc[0].favorite_state == "qualified"


def test_final_hour_rejects_requalification_without_two_prelock_confirmations(tmp_path: Path):
    source = market("ncaaf", "SPREAD", pair_for(side("Away +3", "+3 (-110)", bets=30, money=20)))
    source["kickoff_iso"] = "2026-09-06T21:00:00Z"
    first = apply_red_fox_favorites(pd.DataFrame([source]), as_of="2026-09-06T19:40:00Z")
    update_favorite_tracking(first, tmp_path, as_of="2026-09-06T19:40:00Z")
    lost = first.copy()
    lost["red_fox_favorite"] = "false"
    lost["favorite_state"] = "not_qualified"
    update_favorite_tracking(lost, tmp_path, as_of="2026-09-06T19:50:00Z")
    returned = apply_red_fox_favorites(pd.DataFrame([source]), as_of="2026-09-06T19:54:00Z")
    update_favorite_tracking(returned, tmp_path, as_of="2026-09-06T19:54:00Z")
    final_hour = apply_red_fox_favorites(pd.DataFrame([source]), as_of="2026-09-06T20:10:00Z")
    locked = update_favorite_tracking(final_hour, tmp_path, as_of="2026-09-06T20:10:00Z")
    assert locked.iloc[0].red_fox_favorite == "false"


def test_customer_badge_and_sort_contract_preserve_active_market_semantics():
    board = (Path(__file__).resolve().parents[1] / "site" / "board.html").read_text(encoding="utf-8")
    assert 'src="assets/red-fox-favorite.png" alt="Red Fox Favorite"' in board
    assert "const favorite=isRedFoxFavorite(r);" in board
    assert "if(isRedFoxFavorite(r)) tr.classList.add('favorite-row');" in board
    assert "selected=rows.find(row=>String(row.market_display||'').toUpperCase()===requested)||favoriteRows[0]||rows[0]" in board
    assert "const BOARD_SORTS={FAVORITES:'Red Fox Favorites'" in board
    assert "let boardSortMode='FAVORITES';" in board
    assert "liveRecentSort==='FAVORITES'" in board
    assert '<option value="FAVORITES">Red Fox Favorites</option>' in board
    assert ".favorite-plane-spacer{display:block;height:32px" in board
    assert ".favorite-plane-spacer{display:none!important}" in board
    assert ".red-fox-favorite-badge{width:108px;height:36px;max-width:calc(100% - 42px)}" in board


def test_live_recent_legacy_rows_receive_the_favorite_schema_without_backfill():
    legacy = pd.DataFrame([{"sport": "mlb", "game_id": "old", "market_display": "MONEYLINE"}])
    output = ensure_output_columns(legacy)
    assert output.iloc[0].red_fox_favorite == "false"
    for column in ("favorite_side", "favorite_pathway", "favorite_first_qualified_at",
                   "favorite_final_market_read", "favorite_snapshot_id"):
        assert column in output
        assert output.iloc[0][column] == ""
