import json
from pathlib import Path

import pandas as pd
import pytest

from red_fox_favorite import CONFIG, apply_red_fox_favorites, update_favorite_tracking
from build_live_recent import ensure_output_columns


def side(name, current, *, bets=35, money=30, reaction="Contrarian", direction="TOWARD",
         observations=4, clean=True, kpi=True, action_type="CONTRARIAN CANDIDATE",
         action_side=None, path="One-Way", line_move=1.0, price_move=3.0, context="",
         active_worsening=False):
    return {
        "flagged_side": name, "bets_pct": bets, "money_pct": money,
        "open_line": current, "current_line": current, "reaction": reaction,
        "response_direction": direction, "observation_count": observations,
        "data_badge": "Clean" if clean else "Thin", "kpi_eligible": kpi,
        "action_type": action_type, "action_side": action_side or name, "path": path,
        "line_move_abs": line_move, "price_move_pct": price_move,
        "context_chips": context, "whipsaw_recovered": "Whipsaw Recovered" in context,
        "active_worsening_reversal": active_worsening,
    }


def market(sport, market_name, sides, *, game_id="1", game="Away @ Home", rank=1):
    return {
        "sport": sport, "game_id": game_id, "game": game, "market_display": market_name,
        "board_rank": rank, "current_line": sides[0]["current_line"],
        "market_sides": json.dumps(sides),
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
@pytest.mark.parametrize("price", [-165, 125])
def test_primary_moneyline_sports_and_boundaries_qualify(sport, price):
    candidate = side("Candidate", f"{price:+d}", price_move=3.1)
    opponent = side("Opponent", f"{-price:+d}", bets=65, money=70, reaction="Watch",
                    direction="AGAINST", kpi=False, action_type="OBSERVE ONLY")
    assert is_favorite(result([market(sport, "MONEYLINE", [candidate, opponent])]))


@pytest.mark.parametrize("price", [-166, 126])
def test_moneyline_outside_final_v1_range_is_rejected(price):
    candidate = side("Candidate", f"{price:+d}")
    opponent = side("Opponent", "+100", bets=65, money=70, reaction="Watch", direction="LIMITED", kpi=False)
    assert not is_favorite(result([market("mlb", "MONEYLINE", [candidate, opponent])]))


def test_actionable_persistent_freeze_resistance_qualifies():
    candidate = side("Low side +3", "+3 (-110)", bets=20, money=15, reaction="Watch",
                     direction="LIMITED", kpi=False, action_type="OBSERVE ONLY")
    pressure = side("Public side -3", "-3 (-110)", bets=80, money=85, reaction="Freeze",
                    direction="LIMITED", kpi=True, action_type="FADE CANDIDATE",
                    action_side="Low side +3", path="Held")
    frame = result([market("nfl", "SPREAD", [candidate, pressure])])
    assert is_favorite(frame)
    assert frame.iloc[0].favorite_pathway == "low_support_freeze"


def test_descriptive_freeze_does_not_qualify():
    candidate = side("Low side +3", "+3 (-110)", bets=20, money=15, reaction="Watch", direction="LIMITED", kpi=False)
    pressure = side("Public side -3", "-3 (-110)", bets=80, money=85, reaction="Freeze",
                    direction="LIMITED", kpi=False, action_type="OBSERVE ONLY", action_side="Low side +3", path="Held")
    assert not is_favorite(result([market("nfl", "SPREAD", [candidate, pressure])]))


def test_moderate_support_follow_path_uses_existing_movement_evidence():
    candidate = side("Confirming side -3", "-3 (-110)", bets=55, money=52,
                     reaction="Watch", direction="TOWARD", kpi=False, line_move=1.0)
    opponent = side("Other side +3", "+3 (-110)", bets=45, money=48,
                    reaction="Watch", direction="AGAINST", kpi=False)
    frame = result([market("ncaaf", "SPREAD", [candidate, opponent])])
    assert is_favorite(frame)
    assert frame.iloc[0].favorite_pathway == "moderate_support_follow"


def test_heavy_public_follow_and_nonmeaningful_move_do_not_qualify():
    heavy = side("Public side -3", "-3 (-110)", bets=75, money=80, reaction="Follow", kpi=True)
    other = side("Other +3", "+3 (-110)", bets=25, money=20, reaction="Watch", direction="AGAINST", kpi=False)
    assert not is_favorite(result([market("ncaaf", "SPREAD", [heavy, other])]))
    moderate = side("Balanced -3", "-3 (-110)", bets=55, money=55, reaction="Watch",
                    direction="TOWARD", kpi=False, line_move=0.25, price_move=1.0)
    other = side("Balanced +3", "+3 (-110)", bets=45, money=45, reaction="Watch", direction="LIMITED", kpi=False)
    assert not is_favorite(result([market("nba", "SPREAD", [moderate, other])]))


def test_secondary_moneyline_requires_corresponding_spread_at_four_or_less():
    ml = market("nfl", "MONEYLINE", pair_for(side("Away", "+120")))
    spread4 = market("nfl", "SPREAD", pair_for(side("Away +4", "+4 (-110)")), rank=2)
    assert is_favorite(result([ml, spread4]), 0)
    spread45 = market("nfl", "SPREAD", pair_for(side("Away +4.5", "+4.5 (-110)")), rank=2)
    assert not is_favorite(result([ml, spread45]), 0)


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
