from datetime import datetime, timezone

import pandas as pd

from anomaly_board import build_anomaly_outputs


def _timestamp(hour):
    return datetime(2026, 9, 1, hour, 0, tzinfo=timezone.utc).isoformat()


def _board(latest, history):
    board, _ = build_anomaly_outputs(
        pd.DataFrame([latest]),
        pd.DataFrame(history),
        pd.DataFrame(),
        as_of=_timestamp(16),
    )
    return board.iloc[0]


def test_price_only_spread_move_is_visible_without_calling_it_a_freeze():
    latest = {
        "sport": "ncaaf", "game_id": "g1", "market_display": "SPREAD", "side_key": "HOME",
        "side": "Home -19.5", "game": "Away @ Home", "canonical_key": "away @ home|ncaaf|2026-09-05",
        "bets_pct": 86, "money_pct": 15, "open_line": "Home -19.5 @ +102",
        "current_line": "Home -19.5 @ -112", "_sort_time": "2026-09-05T16:00:00Z",
    }
    history = [
        {"timestamp": _timestamp(17), "sport": "ncaaf", "game_id": "g1", "market_display": "SPREAD", "side_key": "HOME", "current_line": "Home -19.5 @ +102", "bets_pct": 86, "money_pct": 15},
        {"timestamp": _timestamp(18), "sport": "ncaaf", "game_id": "g1", "market_display": "SPREAD", "side_key": "HOME", "current_line": "Home -19.5 @ -112", "bets_pct": 86, "money_pct": 15},
    ]

    row = _board(latest, history)

    assert row["reaction"] == "Watch"
    assert row["path"] == "Juice Move"
    assert row["open_line"] == "-19.5 (+102)"
    assert row["current_line"] == "-19.5 (-112)"
    assert row["price_move_pct"] > 3


def test_capped_split_cannot_create_a_freeze():
    latest = {
        "sport": "nfl", "game_id": "g2", "market_display": "MONEYLINE", "side_key": "HOME",
        "side": "Home", "game": "Away @ Home", "canonical_key": "away @ home|nfl|2026-09-20",
        "bets_pct": 100, "money_pct": 100, "open_line": "Home @ -345",
        "current_line": "Home @ -355", "_sort_time": "2026-09-20T17:00:00Z",
    }
    history = [
        {"timestamp": _timestamp(17), "sport": "nfl", "game_id": "g2", "market_display": "MONEYLINE", "side_key": "HOME", "current_line": "Home @ -345", "bets_pct": 100, "money_pct": 100},
        {"timestamp": _timestamp(18), "sport": "nfl", "game_id": "g2", "market_display": "MONEYLINE", "side_key": "HOME", "current_line": "Home @ -355", "bets_pct": 100, "money_pct": 100},
    ]

    row = _board(latest, history)

    assert row["reaction"] == "Watch"
    assert row["path"] == "Held"
    assert "Split Cap" in row["context_chips"]
    assert "Heavy Favorite" in row["context_chips"]
    assert row["data_badge"] == "Split Risk"


def test_extreme_moneyline_uses_implied_probability_for_whipsaw_detection():
    latest = {
        "sport": "ncaaf", "game_id": "g3", "market_display": "MONEYLINE", "side_key": "HOME",
        "side": "Home", "game": "Away @ Home", "canonical_key": "away @ home|ncaaf|2026-09-05",
        "bets_pct": 60, "money_pct": 60, "open_line": "Home @ -1350",
        "current_line": "Home @ -1350", "_sort_time": "2026-09-05T16:00:00Z",
    }
    history = [
        {"timestamp": _timestamp(17), "sport": "ncaaf", "game_id": "g3", "market_display": "MONEYLINE", "side_key": "HOME", "current_line": "Home @ -1350", "bets_pct": 60, "money_pct": 60},
        {"timestamp": _timestamp(18), "sport": "ncaaf", "game_id": "g3", "market_display": "MONEYLINE", "side_key": "HOME", "current_line": "Home @ -1200", "bets_pct": 60, "money_pct": 60},
        {"timestamp": _timestamp(19), "sport": "ncaaf", "game_id": "g3", "market_display": "MONEYLINE", "side_key": "HOME", "current_line": "Home @ -1350", "bets_pct": 60, "money_pct": 60},
    ]

    row = _board(latest, history)

    assert row["path"] == "Held"
    assert row["move_abs"] == 0
    assert row["max_excursion"] < 2.5
