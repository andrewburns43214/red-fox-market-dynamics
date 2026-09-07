import json
from pathlib import Path

import pandas as pd

from cross_market_split import CROSS_MARKET_SPLIT_COLUMNS, apply_cross_market_split
from red_fox_favorite import apply_red_fox_favorites


def row(market, lean="", *, rank=1, mismatch="false", reaction="Contrarian", favorite="false", sides=None):
    default_sides = [
        {"flagged_side": "Team A +3" if market == "SPREAD" else "Team A"},
        {"flagged_side": "Team B -3" if market == "SPREAD" else "Team B"},
    ]
    return {
        "sport": "nfl", "game_id": "g1", "game": "Team A @ Team B",
        "market_display": market, "directional_lean_side": lean,
        "read_anchor_side": lean, "board_rank": rank, "reaction": reaction,
        "market_sides": json.dumps(sides or default_sides),
        "cross_market_mismatch": mismatch, "red_fox_favorite": favorite,
    }


def annotate(*rows):
    return apply_cross_market_split(pd.DataFrame(rows))


def by_market(frame, column="cross_market_split"):
    return dict(zip(frame.market_display, frame[column]))


def favorite_spread_sides():
    return [
        {
            "flagged_side": "Team A +3", "bets_pct": 35, "money_pct": 30,
            "open_line": "+4 (-110)", "current_line": "+3 (-110)", "reaction": "Contrarian",
            "response_direction": "TOWARD", "observation_count": 4, "data_badge": "Clean",
            "kpi_eligible": True, "action_type": "CONTRARIAN CANDIDATE", "action_side": "Team A +3",
            "path": "One-Way", "line_move_abs": 1.0, "price_move_pct": 3.0,
            "context_chips": "", "active_worsening_reversal": False,
        },
        {
            "flagged_side": "Team B -3", "bets_pct": 65, "money_pct": 70,
            "open_line": "-4 (-110)", "current_line": "-3 (-110)", "reaction": "Watch",
            "response_direction": "AGAINST", "observation_count": 4, "data_badge": "Clean",
            "kpi_eligible": False, "action_type": "OBSERVE ONLY", "action_side": "Team B -3",
            "path": "One-Way", "line_move_abs": 1.0, "price_move_pct": 1.0,
            "context_chips": "", "active_worsening_reversal": False,
        },
    ]


def test_same_confirmed_team_has_no_split():
    result = annotate(row("SPREAD", "Team A +3"), row("MONEYLINE", "Team A", rank=2))
    assert set(result.cross_market_split) == {"false"}


def test_different_confirmed_teams_create_pair_level_split():
    result = annotate(row("SPREAD", "Team A +3"), row("MONEYLINE", "Team B", rank=2))
    assert by_market(result) == {"SPREAD": "true", "MONEYLINE": "true"}
    assert result.iloc[0].cross_market_split_spread_team == "Team A"
    assert result.iloc[0].cross_market_split_moneyline_team == "Team B"


def test_spread_supported_and_moneyline_watch_has_no_split():
    result = annotate(row("SPREAD", "Team A +3"), row("MONEYLINE", "", rank=2, reaction="Watch"))
    assert set(result.cross_market_split) == {"false"}


def test_spread_watch_and_moneyline_supported_has_no_split():
    result = annotate(row("SPREAD", "", reaction="Watch"), row("MONEYLINE", "Team B", rank=2))
    assert set(result.cross_market_split) == {"false"}


def test_missing_corresponding_market_has_no_split():
    assert set(annotate(row("SPREAD", "Team A +3")).cross_market_split) == {"false"}
    assert set(annotate(row("MONEYLINE", "Team B")).cross_market_split) == {"false"}


def test_split_does_not_change_rank_reads_or_raw_market_data():
    board = pd.DataFrame([row("SPREAD", "Team A +3", rank=4), row("MONEYLINE", "Team B", rank=9)])
    protected = ["market_display", "board_rank", "reaction", "read_anchor_side", "directional_lean_side", "market_sides"]
    result = apply_cross_market_split(board)
    pd.testing.assert_frame_equal(board[protected], result[protected])


def test_split_column_is_not_an_automatic_favorite_blocker():
    spread = row("SPREAD", "Team A +3", favorite="true", sides=favorite_spread_sides())
    moneyline = row("MONEYLINE", "Team B", rank=2, reaction="Follow", sides=[
        {"flagged_side": "Team A", "response_direction": "LIMITED", "price_move_pct": 0},
        {"flagged_side": "Team B", "response_direction": "LIMITED", "price_move_pct": 0},
    ])
    annotated = annotate(spread, moneyline)
    assert set(annotated.cross_market_split) == {"true"}
    qualified = apply_red_fox_favorites(annotated, as_of="2026-09-07T12:00:00Z")
    assert qualified.loc[qualified.market_display.eq("SPREAD"), "red_fox_favorite"].iloc[0] == "true"


def test_confirmed_mismatch_suppresses_split():
    result = annotate(
        row("SPREAD", "Team A +3", mismatch="true"),
        row("MONEYLINE", "Team B", rank=2, mismatch="true"),
    )
    assert set(result.cross_market_split) == {"false"}


def test_total_is_never_annotated():
    result = annotate(
        row("SPREAD", "Team A +3"), row("MONEYLINE", "Team B", rank=2),
        row("TOTAL", "Over", rank=3),
    )
    assert by_market(result) == {"SPREAD": "true", "MONEYLINE": "true", "TOTAL": "false"}


def test_publication_wires_split_after_mismatch_and_before_unchanged_favorite_evaluation():
    source = (Path(__file__).resolve().parents[1] / "refresh_anomaly_board.py").read_text(encoding="utf-8")
    mismatch = source.index("board = apply_cross_market_integrity(board, history")
    split = source.index("board = apply_cross_market_split(board)", mismatch)
    favorite = source.index("board = apply_red_fox_favorites(board", split)
    assert mismatch < split < favorite
    assert "*CROSS_MARKET_COLUMNS, *CROSS_MARKET_SPLIT_COLUMNS, *FAVORITE_COLUMNS" in source
    assert CROSS_MARKET_SPLIT_COLUMNS == [
        "cross_market_split", "cross_market_split_state", "cross_market_split_explanation",
        "cross_market_split_spread_team", "cross_market_split_moneyline_team",
    ]
