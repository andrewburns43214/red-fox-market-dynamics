from pathlib import Path

import pandas as pd

from anomaly_board import build_anomaly_outputs
from refresh_anomaly_board import complete_public_market_rows, filter_publication_eligible_markets, latest_synchronized_market_rows


def test_refresh_watchdog_keeps_atomic_failure_protection_with_measured_headroom():
    script = (Path(__file__).resolve().parents[1] / "run_all_sports.sh").read_text(encoding="utf-8")
    assert 'REFRESH_TIMEOUT_SECONDS="${REDFOX_REFRESH_TIMEOUT_SECONDS:-300}"' in script
    assert 'if timeout "$REFRESH_TIMEOUT_SECONDS" "$PY" refresh_anomaly_board.py' in script
    assert 'refresh anomaly board ERROR' in script


def test_rolling_football_publication_window_keeps_board_and_csv_market_sets_in_parity():
    now = "2026-09-03T12:00:00-04:00"
    rows = pd.DataFrame([
        {"sport": "nfl", "game_id": "today", "market_display": "TOTAL", "dk_start_iso": "2026-09-03T23:00:00Z"},
        {"sport": "ncaaf", "game_id": "day-seven", "market_display": "TOTAL", "dk_start_iso": "2026-09-10T23:00:00Z"},
        {"sport": "nfl", "game_id": "day-eight", "market_display": "TOTAL", "dk_start_iso": "2026-09-11T00:00:00-04:00"},
        {"sport": "nfl", "game_id": "past", "market_display": "TOTAL", "dk_start_iso": "2026-09-03T09:00:00-04:00"},
        {"sport": "mlb", "game_id": "unaffected", "market_display": "TOTAL", "dk_start_iso": "2026-09-20T23:00:00Z"},
    ])
    published = filter_publication_eligible_markets(rows, now=now)
    keys = set(published.game_id)
    assert {"today", "day-seven", "unaffected"}.issubset(keys)
    assert "day-eight" not in keys
    # Existing kickoff expiry owns removal of already-started rows; the rolling
    # calendar window itself intentionally does not redefine that gate.
    assert "past" in keys
    assert keys == set(published.loc[:, "game_id"])


def test_board_uses_latest_shared_snapshot_for_both_market_sides():
    active = pd.DataFrame([
        {"sport": "nfl", "game_id": "g1", "market_display": "TOTAL", "side_key": "Over", "timestamp": "2026-09-03T15:00:00Z", "current_line": "Over 56.5"},
        {"sport": "nfl", "game_id": "g1", "market_display": "TOTAL", "side_key": "Under", "timestamp": "2026-09-03T15:00:00Z", "current_line": "Under 56.5"},
        # A partial later scrape must not be stitched into the older Under.
        {"sport": "nfl", "game_id": "g1", "market_display": "TOTAL", "side_key": "Over", "timestamp": "2026-09-03T15:05:00Z", "current_line": "Over 55.5"},
    ])
    active["timestamp"] = pd.to_datetime(active["timestamp"], utc=True)

    result = latest_synchronized_market_rows(active)

    assert result["side_key"].tolist() == ["Over", "Under"]
    assert result["current_line"].tolist() == ["Over 56.5", "Under 56.5"]
    assert result["timestamp"].nunique() == 1


def test_board_omits_markets_without_a_complete_shared_snapshot():
    active = pd.DataFrame([
        {"sport": "nfl", "game_id": "g1", "market_display": "SPREAD", "side_key": "Away", "timestamp": "2026-09-03T15:00:00Z"},
        {"sport": "nfl", "game_id": "g1", "market_display": "SPREAD", "side_key": "Home", "timestamp": "2026-09-03T15:01:00Z"},
    ])
    active["timestamp"] = pd.to_datetime(active["timestamp"], utc=True)

    assert latest_synchronized_market_rows(active).empty


def test_public_board_omits_paired_markets_missing_customer_essential_fields():
    base = {
        "sport": "nfl", "game_id": "g1", "market_display": "SPREAD",
        "bets_pct": "50", "money_pct": "50", "open_line": "+3 (-110)",
        "current_line": "+3 (-110)", "dk_start_iso": "2026-09-05T17:00:00Z",
    }
    dashboard = pd.DataFrame([
        {**base, "side_key": "Away", "side": "Away +3"},
        {**base, "side_key": "Home", "side": "Home -3"},
        {**base, "game_id": "g2", "side_key": "Away", "side": "Other Away +3"},
        {**base, "game_id": "g2", "side_key": "Home", "side": "Other Home -3", "money_pct": ""},
    ])

    result = complete_public_market_rows(dashboard)

    assert result["game_id"].unique().tolist() == ["g1"]


def test_two_sided_market_uses_its_first_shared_history_capture_for_both_open_prices():
    """A partial early total capture cannot become one side's public opening line."""
    latest = pd.DataFrame([
        {"sport": "nfl", "game_id": "g-total", "market_display": "TOTAL", "side_key": "Over", "side": "Over 56.5", "game": "Away @ Home", "bets_pct": 70, "money_pct": 80, "open_line": "O 56.5 (-110)", "current_line": "O 55.5 (-110)", "_sort_time": "2026-09-03T15:10:00Z"},
        {"sport": "nfl", "game_id": "g-total", "market_display": "TOTAL", "side_key": "Under", "side": "Under 56.5", "game": "Away @ Home", "bets_pct": 30, "money_pct": 20, "open_line": "U 56.5 (-110)", "current_line": "U 55.5 (-110)", "_sort_time": "2026-09-03T15:10:00Z"},
    ])
    history = pd.DataFrame([
        # Over was seen early, but the market was not yet a complete two-sided capture.
        {"timestamp": "2026-09-03T15:00:00Z", "sport": "nfl", "game_id": "g-total", "market_display": "TOTAL", "side_key": "Over", "current_line": "O 57.5 (-110)", "bets_pct": 70, "money_pct": 80},
        {"timestamp": "2026-09-03T15:05:00Z", "sport": "nfl", "game_id": "g-total", "market_display": "TOTAL", "side_key": "Over", "current_line": "O 56.5 (-110)", "bets_pct": 70, "money_pct": 80},
        {"timestamp": "2026-09-03T15:05:00Z", "sport": "nfl", "game_id": "g-total", "market_display": "TOTAL", "side_key": "Under", "current_line": "U 56.5 (-110)", "bets_pct": 30, "money_pct": 20},
        {"timestamp": "2026-09-03T15:10:00Z", "sport": "nfl", "game_id": "g-total", "market_display": "TOTAL", "side_key": "Over", "current_line": "O 55.5 (-110)", "bets_pct": 70, "money_pct": 80},
        {"timestamp": "2026-09-03T15:10:00Z", "sport": "nfl", "game_id": "g-total", "market_display": "TOTAL", "side_key": "Under", "current_line": "U 55.5 (-110)", "bets_pct": 30, "money_pct": 20},
    ])

    board, _ = build_anomaly_outputs(latest, history, pd.DataFrame(), as_of="2026-09-03T14:00:00Z")
    sides = board.set_index("flagged_side")

    assert sides.loc["Over 56.5", "open_line"] == "56.5"
    assert sides.loc["Under 56.5", "open_line"] == "56.5"
