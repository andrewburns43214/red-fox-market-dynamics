import json
from pathlib import Path

import pandas as pd

from cross_market_integrity import apply_cross_market_integrity, evaluate_cross_market_history
from red_fox_favorite import apply_red_fox_favorites


def market_sides(market, spread_a=-3.0, ml_a=-135, *, stale=False):
    context = "Market Lag" if stale else ""
    if market == "SPREAD":
        return [
            {"flagged_side": f"Team A {spread_a:+g}", "data_badge": "Clean", "context_chips": context},
            {"flagged_side": f"Team B {-spread_a:+g}", "data_badge": "Clean", "context_chips": context},
        ]
    return [
        {"flagged_side": "Team A", "data_badge": "Clean", "context_chips": context},
        {"flagged_side": "Team B", "data_badge": "Clean", "context_chips": context},
    ]


def board_pair(spread_a=-3.0, ml_a=-135, *, stale=False):
    ml_b = 115 if ml_a < 0 else -135
    common = {"sport": "nfl", "game_id": "g1", "game": "Team A @ Team B"}
    return pd.DataFrame([
        {**common, "market_display": "SPREAD", "market_sides": json.dumps(market_sides("SPREAD", spread_a, stale=stale)),
         "board_rank": 4, "reaction": "Contrarian", "market_rationale": "Spread rationale", "anomaly_chips": "Contrarian | One-Way"},
        {**common, "market_display": "MONEYLINE", "market_sides": json.dumps(market_sides("MONEYLINE", ml_a, stale=stale)),
         "board_rank": 9, "reaction": "Watch", "market_rationale": "Moneyline rationale", "anomaly_chips": "Watch"},
        {**common, "market_display": "TOTAL", "market_sides": json.dumps([]), "board_rank": 12,
         "reaction": "Watch", "market_rationale": "Total rationale", "anomaly_chips": "Watch"},
    ])


def history_rows(times=(0, 5, 10), *, spread_a=-3.0, ml_a=-135, ml_b=115,
                 coherent_after=None, spread_start=0, ml_start=0, open_spread_a=None,
                 open_ml_a=None, open_ml_b=None):
    base = pd.Timestamp("2026-09-07T12:00:00Z")
    rows = []
    for minute in times:
        current_spread = coherent_after.get(minute, spread_a) if coherent_after else spread_a
        spread_time = base + pd.Timedelta(minutes=minute + spread_start)
        ml_time = base + pd.Timedelta(minutes=minute + ml_start)
        for team, value in (("Team A", current_spread), ("Team B", -current_spread)):
            opening = open_spread_a if team == "Team A" and open_spread_a is not None else (
                -open_spread_a if team == "Team B" and open_spread_a is not None else value
            )
            rows.append({"sport": "nfl", "game_id": "g1", "game": "Team A @ Team B", "market_display": "SPREAD",
                         "timestamp": spread_time.isoformat(), "side": f"{team} {value:+g}",
                         "open_line": f"{team} {opening:+g} @ -110", "current_line": f"{team} {value:+g} @ -110"})
        for team, odds, opening in (("Team A", ml_a, open_ml_a), ("Team B", ml_b, open_ml_b)):
            open_odds = odds if opening is None else opening
            rows.append({"sport": "nfl", "game_id": "g1", "game": "Team A @ Team B", "market_display": "MONEYLINE",
                         "timestamp": ml_time.isoformat(), "side": team,
                         "open_line": f"{team} @ {open_odds:+d}", "current_line": f"{team} @ {odds:+d}"})
    return pd.DataFrame(rows)


def favorite_board_pair():
    frame = board_pair(spread_a=3.0, ml_a=-135)
    candidate = {
        "flagged_side": "Team A +3", "bets_pct": 35, "money_pct": 30,
        "open_line": "+4 (-110)", "current_line": "+3 (-110)", "reaction": "Contrarian",
        "response_direction": "TOWARD", "observation_count": 4, "data_badge": "Clean",
        "kpi_eligible": True, "path": "One-Way", "line_move_abs": 1.0, "price_move_pct": 3.0,
        "context_chips": "", "active_worsening_reversal": False,
    }
    opponent = {
        "flagged_side": "Team B -3", "bets_pct": 65, "money_pct": 70,
        "open_line": "-4 (-110)", "current_line": "-3 (-110)", "reaction": "Watch",
        "response_direction": "AGAINST", "observation_count": 4, "data_badge": "Clean",
        "kpi_eligible": False, "path": "One-Way", "line_move_abs": 1.0, "price_move_pct": 1.0,
        "context_chips": "", "active_worsening_reversal": False,
    }
    frame.at[0, "market_sides"] = json.dumps([candidate, opponent])
    frame.at[0, "supported_side"] = "Team A +3"
    return frame


def test_synchronized_spread_and_moneyline_favorite_same_team_has_no_mismatch():
    result = apply_cross_market_integrity(board_pair(-3, -135), history_rows(spread_a=-3, ml_a=-135, ml_b=115), as_of="2026-09-07T12:10:00Z")
    assert set(result.cross_market_mismatch) == {"false"}


def test_spread_dog_with_at_least_53_percent_no_vig_moneyline_is_a_mismatch():
    result = apply_cross_market_integrity(board_pair(3, -135), history_rows(spread_a=3, ml_a=-135, ml_b=115), as_of="2026-09-07T12:10:00Z")
    assert result.iloc[0].cross_market_mismatch == "true"
    assert float(result.iloc[0].cross_market_no_vig_probability.rstrip("%")) >= 53


def test_spread_favorite_with_at_most_47_percent_no_vig_moneyline_is_a_mismatch():
    result = apply_cross_market_integrity(board_pair(-3, 115), history_rows(spread_a=-3, ml_a=115, ml_b=-135), as_of="2026-09-07T12:10:00Z")
    assert result.iloc[0].cross_market_mismatch == "true"
    assert float(result.iloc[0].cross_market_no_vig_probability.rstrip("%")) <= 47


def test_spread_under_one_and_a_half_and_near_pickem_do_not_mismatch():
    small = apply_cross_market_integrity(board_pair(1, -135), history_rows(spread_a=1, ml_a=-135, ml_b=115), as_of="2026-09-07T12:10:00Z")
    pickem = apply_cross_market_integrity(board_pair(3, -105), history_rows(spread_a=3, ml_a=-105, ml_b=-115), as_of="2026-09-07T12:10:00Z")
    assert set(small.cross_market_mismatch) == {"false"}
    assert set(pickem.cross_market_mismatch) == {"false"}


def test_observations_more_than_fifteen_minutes_apart_do_not_pair():
    history = history_rows(times=(0, 60, 120), spread_a=3, ml_a=-135, ml_b=115, ml_start=20)
    result = apply_cross_market_integrity(board_pair(3, -135), history, as_of="2026-09-07T14:20:00Z")
    assert set(result.cross_market_mismatch) == {"false"}


def test_single_conflict_does_not_confirm_but_three_observations_do():
    single = apply_cross_market_integrity(board_pair(3, -135), history_rows(times=(0,), spread_a=3, ml_a=-135, ml_b=115), as_of="2026-09-07T12:00:00Z")
    three = apply_cross_market_integrity(board_pair(3, -135), history_rows(times=(0, 5, 10), spread_a=3, ml_a=-135, ml_b=115), as_of="2026-09-07T12:10:00Z")
    assert set(single.cross_market_mismatch) == {"false"}
    assert three.iloc[0].cross_market_observation_count == "3"
    assert three.iloc[0].cross_market_mismatch == "true"


def test_two_observations_twenty_minutes_apart_confirm_by_duration():
    result = apply_cross_market_integrity(board_pair(3, -135), history_rows(times=(0, 20), spread_a=3, ml_a=-135, ml_b=115), as_of="2026-09-07T12:20:00Z")
    assert result.iloc[0].cross_market_mismatch == "true"
    assert result.iloc[0].cross_market_duration_minutes == "20"


def test_stale_corresponding_market_uses_reliability_handling_not_mismatch():
    result = apply_cross_market_integrity(board_pair(3, -135, stale=True), history_rows(spread_a=3, ml_a=-135, ml_b=115), as_of="2026-09-07T12:10:00Z")
    assert set(result.cross_market_mismatch) == {"false"}


def test_different_opener_availability_windows_do_not_create_opener_mismatch():
    history = history_rows(times=(0, 60, 120), spread_a=-3, ml_a=-135, ml_b=115, ml_start=60,
                           open_spread_a=3, open_ml_a=-135, open_ml_b=115)
    evaluation = evaluate_cross_market_history(history, as_of="2026-09-07T15:00:00Z")
    assert evaluation["opener_verified"] is False


def test_mississippi_state_minnesota_chronology_is_not_an_opener_or_current_mismatch():
    early = history_rows(times=(0, 10, 20), spread_a=2.5, ml_a=-120, ml_b=100, ml_start=1440,
                         open_spread_a=2.5, open_ml_a=-125, open_ml_b=105)
    synced = history_rows(times=(1440, 1450, 1460), spread_a=-1.5, ml_a=-120, ml_b=100,
                          open_spread_a=2.5, open_ml_a=-125, open_ml_b=105)
    history = pd.concat([early[early.market_display.eq("SPREAD")], synced], ignore_index=True)
    evaluation = evaluate_cross_market_history(history, as_of="2026-09-08T12:20:00Z")
    assert evaluation["confirmed"] is False
    assert evaluation["opener_verified"] is False


def test_confirmed_mismatch_withholds_otherwise_qualifying_favorite():
    annotated = apply_cross_market_integrity(favorite_board_pair(), history_rows(spread_a=3, ml_a=-135, ml_b=115), as_of="2026-09-07T12:10:00Z")
    result = apply_red_fox_favorites(annotated, as_of="2026-09-07T12:10:00Z")
    assert result.iloc[0].red_fox_favorite == "false"
    assert result.iloc[0].favorite_cross_market_state == "mismatch"


def test_neutral_or_missing_moneyline_does_not_block_favorite():
    neutral = apply_cross_market_integrity(favorite_board_pair(), history_rows(spread_a=3, ml_a=105, ml_b=-115), as_of="2026-09-07T12:10:00Z")
    assert apply_red_fox_favorites(neutral).iloc[0].red_fox_favorite == "true"
    spread_only = favorite_board_pair().iloc[[0]].copy()
    assert apply_red_fox_favorites(spread_only).iloc[0].red_fox_favorite == "true"


def test_resolved_mismatch_clears_and_favorite_can_requalify():
    conflict = history_rows(times=(0, 5, 10), spread_a=3, ml_a=-135, ml_b=115)
    coherent = history_rows(times=(15, 20, 25), spread_a=3, ml_a=115, ml_b=-135)
    history = pd.concat([conflict, coherent], ignore_index=True)
    annotated = apply_cross_market_integrity(favorite_board_pair(), history, as_of="2026-09-07T12:25:00Z")
    assert set(annotated.cross_market_mismatch) == {"false"}
    assert apply_red_fox_favorites(annotated).iloc[0].red_fox_favorite == "true"


def test_integrity_annotation_does_not_change_market_read_rank_or_raw_data():
    board = board_pair(3, -135)
    before = board[["market_display", "board_rank", "reaction", "anomaly_chips", "market_sides"]].copy()
    after = apply_cross_market_integrity(board, history_rows(spread_a=3, ml_a=-135, ml_b=115), as_of="2026-09-07T12:10:00Z")
    pd.testing.assert_frame_equal(before, after[before.columns])
    assert after.iloc[0].market_rationale.startswith("Spread rationale")
    assert "while the Moneyline prices" in after.iloc[0].market_rationale


def test_badge_state_is_pair_level_and_total_is_excluded():
    result = apply_cross_market_integrity(board_pair(3, -135), history_rows(spread_a=3, ml_a=-135, ml_b=115), as_of="2026-09-07T12:10:00Z")
    by_market = dict(zip(result.market_display, result.cross_market_mismatch))
    assert by_market == {"SPREAD": "true", "MONEYLINE": "true", "TOTAL": "false"}
    assert all("cross_market_mismatch" not in json.loads(value)[0] for value in result.loc[result.market_display != "TOTAL", "market_sides"])


def test_publication_wires_integrity_after_reads_and_ranking_but_before_favorites():
    source = (Path(__file__).resolve().parents[1] / "refresh_anomaly_board.py").read_text(encoding="utf-8")
    selected = source.index("board = select_market_leaders(board)")
    integrity = source.index("board = apply_cross_market_integrity(board, history", selected)
    split = source.index("board = apply_cross_market_split(board)", integrity)
    favorites = source.index("board = apply_red_fox_favorites(board", split)
    assert selected < integrity < split < favorites
    assert "*CROSS_MARKET_COLUMNS, *CROSS_MARKET_SPLIT_COLUMNS, *FAVORITE_COLUMNS" in source
    assert "cross_market_adj" not in (Path(__file__).resolve().parents[1] / "cross_market_integrity.py").read_text(encoding="utf-8")
