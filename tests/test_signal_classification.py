from datetime import datetime, timezone

import pandas as pd

from anomaly_board import build_anomaly_outputs


def _timestamp(hour):
    return datetime(2026, 9, 1, hour, 0, tzinfo=timezone.utc).isoformat()


def test_ticket_only_market_is_watch_not_freeze():
    latest = pd.DataFrame([
        {
            "sport": "nfl", "game_id": "g1", "market_display": "SPREAD", "side_key": "NE",
            "side": "NE -3", "game": "NYJ @ NE", "canonical_key": "nyj @ ne|nfl|2026-09-01",
            "bets_pct": 82, "money_pct": 31, "open_line": "NE -3 @ -110",
            "current_line": "NE -3 @ -110", "_sort_time": _timestamp(20),
        }
    ])
    history = pd.DataFrame([
        {"timestamp": _timestamp(17), "sport": "nfl", "game_id": "g1", "market_display": "SPREAD", "side_key": "NE", "current_line": "NE -3 @ -110", "bets_pct": 82, "money_pct": 31},
        {"timestamp": _timestamp(18), "sport": "nfl", "game_id": "g1", "market_display": "SPREAD", "side_key": "NE", "current_line": "NE -3 @ -110", "bets_pct": 82, "money_pct": 31},
    ])

    board, _ = build_anomaly_outputs(latest, history, pd.DataFrame(), as_of=_timestamp(16))

    assert board.iloc[0]["reaction"] == "Watch"
    assert board.iloc[0]["path"] == "Held"
    assert "Ticket-led" in board.iloc[0]["context_chips"]
