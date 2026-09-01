from datetime import datetime, timezone

import pandas as pd

from anomaly_action_ledger import update_action_ledger
from anomaly_action_results import rebuild_action_results
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


def test_ticket_heavy_short_favorite_is_context_not_a_freeze():
    latest = {
        "sport": "nfl", "game_id": "g4", "market_display": "MONEYLINE", "side_key": "HOME",
        "side": "Home", "game": "Away @ Home", "canonical_key": "away @ home|nfl|2026-09-13",
        "bets_pct": 90, "money_pct": 71, "open_line": "Home @ -575",
        "current_line": "Home @ -550", "_sort_time": "2026-09-13T20:25:00Z",
    }
    history = [
        {"timestamp": _timestamp(17), "sport": "nfl", "game_id": "g4", "market_display": "MONEYLINE", "side_key": "HOME", "current_line": "Home @ -575", "bets_pct": 90, "money_pct": 71},
        {"timestamp": _timestamp(18), "sport": "nfl", "game_id": "g4", "market_display": "MONEYLINE", "side_key": "HOME", "current_line": "Home @ -550", "bets_pct": 90, "money_pct": 71},
    ]

    row = _board(latest, history)

    assert row["reaction"] == "Watch"
    assert "Heavy Favorite" in row["context_chips"]
    assert "parlay risk" in row["rank_reason"].lower()


def test_freeze_focus_identifies_the_high_split_side_not_a_recommendation():
    latest = {
        "sport": "nfl", "game_id": "g5", "market_display": "SPREAD", "side_key": "HOME",
        "side": "Home -3.5", "game": "Away @ Home", "canonical_key": "away @ home|nfl|2026-09-13",
        "bets_pct": 82, "money_pct": 76, "open_line": "Home -3.5 @ -110",
        "current_line": "Home -3.5 @ -110", "_sort_time": "2026-09-13T20:25:00Z",
    }
    history = [
        {"timestamp": _timestamp(17), "sport": "nfl", "game_id": "g5", "market_display": "SPREAD", "side_key": "HOME", "current_line": "Home -3.5 @ -110", "bets_pct": 82, "money_pct": 76},
        {"timestamp": _timestamp(18), "sport": "nfl", "game_id": "g5", "market_display": "SPREAD", "side_key": "HOME", "current_line": "Home -3.5 @ -110", "bets_pct": 82, "money_pct": 76},
    ]

    row = _board(latest, history)

    assert row["reaction"] == "Freeze"
    assert row["flagged_side"] == "Home -3.5"
    assert row["focus_basis"] == "High-split side; market held"
    assert row["action_type"] == "OBSERVE ONLY"
    assert not row["kpi_eligible"]


def test_sustained_non_key_freeze_emits_the_opposing_fade_candidate():
    latest = pd.DataFrame([
        {"sport": "ncaab", "game_id": "g6", "market_display": "TOTAL", "side_key": "OVER", "side": "Over 145.5", "game": "Away @ Home", "canonical_key": "away @ home|ncaab|2026-09-13", "bets_pct": 84, "money_pct": 72, "open_line": "Over 145.5 @ -110", "current_line": "Over 145.5 @ -110", "_sort_time": "2026-09-13T20:25:00Z"},
        {"sport": "ncaab", "game_id": "g6", "market_display": "TOTAL", "side_key": "UNDER", "side": "Under 145.5", "game": "Away @ Home", "canonical_key": "away @ home|ncaab|2026-09-13", "bets_pct": 16, "money_pct": 28, "open_line": "Under 145.5 @ -110", "current_line": "Under 145.5 @ -110", "_sort_time": "2026-09-13T20:25:00Z"},
    ])
    history = []
    for hour in range(17, 21):
        history.extend([
            {"timestamp": _timestamp(hour), "sport": "ncaab", "game_id": "g6", "market_display": "TOTAL", "side_key": "OVER", "current_line": "Over 145.5 @ -110", "bets_pct": 84, "money_pct": 72},
            {"timestamp": _timestamp(hour), "sport": "ncaab", "game_id": "g6", "market_display": "TOTAL", "side_key": "UNDER", "current_line": "Under 145.5 @ -110", "bets_pct": 16, "money_pct": 28},
        ])

    board, _ = build_anomaly_outputs(latest, pd.DataFrame(history), pd.DataFrame(), as_of=_timestamp(16))
    row = board.loc[board["flagged_side"] == "Over 145.5"].iloc[0]

    assert row["action_type"] == "FADE CANDIDATE"
    assert row["action_side"] == "Under 145.5"
    assert row["action_line"] == "Under 145.5 @ -110"
    assert row["kpi_eligible"]


def test_action_ledger_preserves_one_candidate_at_its_decision_line(tmp_path):
    board = pd.DataFrame([{
        "sport": "ncaab", "game_id": "g7", "market_display": "TOTAL", "action_type": "FADE CANDIDATE",
        "action_side": "Under 145.5", "action_line": "Under 145.5 @ -110", "flagged_side": "Over 145.5",
        "current_line": "Over 145.5 @ -110", "first_anomaly_seen": _timestamp(18), "kpi_eligible": True,
    }])

    assert update_action_ledger(board, tmp_path, datetime.now(timezone.utc)) == 1
    assert update_action_ledger(board, tmp_path, datetime.now(timezone.utc)) == 0
    ledger = pd.read_csv(tmp_path / "anomaly_action_ledger.csv", dtype=str)

    assert len(ledger) == 1
    assert ledger.iloc[0]["observed_side"] == "Over 145.5"
    assert ledger.iloc[0]["action_side"] == "Under 145.5"
    assert ledger.iloc[0]["action_line"] == "Under 145.5 @ -110"


def test_action_results_grade_locked_fade_side_and_daily_kpi(tmp_path):
    ledger = pd.DataFrame([
        {"action_id": "a1", "first_anomaly_seen": _timestamp(18), "sport": "ncaab", "game_id": "g8", "game": "Away @ Home", "market_display": "SPREAD", "reaction": "Freeze", "observed_side": "Home -3.5", "action_side": "Away +3.5", "action_line": "Away +3.5 @ -110", "action_type": "FADE CANDIDATE"},
        {"action_id": "a2", "first_anomaly_seen": _timestamp(18), "sport": "ncaab", "game_id": "g9", "game": "Away @ Home", "market_display": "TOTAL", "reaction": "Freeze", "observed_side": "Over 145.5", "action_side": "Under 145.5", "action_line": "Under 145.5 @ -110", "action_type": "FADE CANDIDATE"},
    ])
    scores = pd.DataFrame([
        {"game_id": "g8", "team1": "away", "team1_score": 70, "team2": "home", "team2_score": 71},
        {"game_id": "g9", "team1": "away", "team1_score": 70, "team2": "home", "team2_score": 72},
    ])
    ledger.to_csv(tmp_path / "anomaly_action_ledger.csv", index=False)
    scores.to_csv(tmp_path / "final_scores_history.csv", index=False)

    assert rebuild_action_results(tmp_path) == 2
    results = pd.read_csv(tmp_path / "anomaly_action_results.csv", dtype=str)

    assert results["outcome"].tolist() == ["WIN", "WIN"]
    assert (tmp_path / "anomaly_action_kpi_daily.csv").exists()


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
