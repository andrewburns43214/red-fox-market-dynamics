from datetime import datetime, timezone
import unittest

import pandas as pd

from anomaly_board import build_anomaly_outputs, select_market_leaders


def _ts(hour, minute):
    return datetime(2026, 9, 1, hour, minute, tzinfo=timezone.utc).isoformat()


class TestAnomalyBoard(unittest.TestCase):
    def test_board_publishes_one_evidence_leader_per_market(self):
        board = pd.DataFrame([
            {"sport": "nfl", "game_id": "g1", "market_display": "SPREAD", "flagged_side": "SEA +3", "anomaly_sort": 3, "severity_sort": 80, "recorded_reaction": "Contrarian", "game": "NE @ SEA"},
            {"sport": "nfl", "game_id": "g1", "market_display": "SPREAD", "flagged_side": "NE -3", "anomaly_sort": 1, "severity_sort": 70, "game": "NE @ SEA"},
            {"sport": "nfl", "game_id": "g1", "market_display": "TOTAL", "flagged_side": "Over 44.5", "anomaly_sort": 2, "severity_sort": 60, "game": "NE @ SEA"},
        ])

        leaders = select_market_leaders(board)

        self.assertEqual(len(leaders), 2)
        self.assertEqual(leaders.iloc[0]["flagged_side"], "SEA +3")
        self.assertEqual(leaders.iloc[0]["board_rank"], 1)

    def test_board_rank_keeps_more_severe_like_signals_ahead_of_alphabetical_order(self):
        board = pd.DataFrame([
            {"sport": "nfl", "game_id": "g5", "market_display": "SPREAD", "flagged_side": "Alpha +3", "reaction": "Freeze", "anomaly_sort": 3, "severity_sort": 10, "game": "Alpha @ Beta"},
            {"sport": "nfl", "game_id": "g6", "market_display": "SPREAD", "flagged_side": "Zulu +3", "reaction": "Freeze", "anomaly_sort": 3, "severity_sort": 50, "game": "Zulu @ Yankee"},
        ])

        leaders = select_market_leaders(board)

        self.assertEqual(leaders.iloc[0]["flagged_side"], "Zulu +3")

    def test_contrarian_whipsaw_with_low_bets_high_money(self):
        latest = pd.DataFrame([
            {
                "sport": "nfl",
                "game_id": "g1",
                "market_display": "SPREAD",
                "side_key": "SEA",
                "side": "SEA +3",
                "game": "NE @ SEA",
                "canonical_key": "ne @ sea|nfl|2026-09-01",
                "bets_pct": 28,
                "money_pct": 31,
                "open_line": "SEA +4 @ -110",
                "current_line": "SEA +3 @ -110",
                "_sort_time": _ts(23, 15),
            },
            {
                "sport": "nfl",
                "game_id": "g1",
                "market_display": "SPREAD",
                "side_key": "NE",
                "side": "NE -3",
                "game": "NE @ SEA",
                "canonical_key": "ne @ sea|nfl|2026-09-01",
                "bets_pct": 72,
                "money_pct": 37,
                "open_line": "NE -4 @ -110",
                "current_line": "NE -3 @ -110",
                "_sort_time": _ts(23, 15),
            },
        ])
        history = pd.DataFrame([
            {"timestamp": _ts(18, 0), "sport": "nfl", "game_id": "g1", "market_display": "SPREAD", "side_key": "SEA", "current_line": "SEA +4 @ -110", "bets_pct": 28, "money_pct": 31},
            {"timestamp": _ts(19, 0), "sport": "nfl", "game_id": "g1", "market_display": "SPREAD", "side_key": "SEA", "current_line": "SEA +2.5 @ -110", "bets_pct": 29, "money_pct": 32},
            {"timestamp": _ts(20, 0), "sport": "nfl", "game_id": "g1", "market_display": "SPREAD", "side_key": "SEA", "current_line": "SEA +3 @ -110", "bets_pct": 28, "money_pct": 31},
            {"timestamp": _ts(18, 0), "sport": "nfl", "game_id": "g1", "market_display": "SPREAD", "side_key": "NE", "current_line": "NE -4 @ -110", "bets_pct": 72, "money_pct": 37},
            {"timestamp": _ts(19, 0), "sport": "nfl", "game_id": "g1", "market_display": "SPREAD", "side_key": "NE", "current_line": "NE -2.5 @ -110", "bets_pct": 71, "money_pct": 38},
            {"timestamp": _ts(20, 0), "sport": "nfl", "game_id": "g1", "market_display": "SPREAD", "side_key": "NE", "current_line": "NE -3 @ -110", "bets_pct": 72, "money_pct": 37},
        ])

        board, events = build_anomaly_outputs(latest, history, pd.DataFrame(), as_of=_ts(17, 0))

        row = board.loc[board["flagged_side"] == "SEA +3"].iloc[0]
        self.assertEqual(row["reaction"], "Contrarian")
        self.assertEqual(row["path"], "Whipsaw")
        self.assertIn("K3", row["context_chips"])
        self.assertEqual(row["data_badge"], "Clean")
        self.assertEqual(row["path_summary"], "+4 -> +2.5 -> +3")
        self.assertEqual(len(events[events["flagged_side"] == "SEA +3"]), 3)

    def test_strong_freeze_sorts_ahead_of_follow(self):
        latest = pd.DataFrame([
            {
                "sport": "ncaab",
                "game_id": "g2",
                "market_display": "TOTAL",
                "side_key": "Over",
                "side": "Over 145.5",
                "game": "Duke @ UNC",
                "canonical_key": "duke @ unc|ncaab|2026-09-01",
                "bets_pct": 82,
                "money_pct": 77,
                "open_line": "Over 145.5 @ -130",
                "current_line": "Over 145.5 @ -130",
                "_sort_time": _ts(22, 0),
            },
            {
                "sport": "ncaab",
                "game_id": "g2",
                "market_display": "TOTAL",
                "side_key": "Under",
                "side": "Under 145.5",
                "game": "Duke @ UNC",
                "canonical_key": "duke @ unc|ncaab|2026-09-01",
                "bets_pct": 18,
                "money_pct": 23,
                "open_line": "Under 145.5 @ -110",
                "current_line": "Under 145.5 @ -110",
                "_sort_time": _ts(22, 0),
            },
            {
                "sport": "nba",
                "game_id": "g3",
                "market_display": "MONEYLINE",
                "side_key": "BOS",
                "side": "BOS Celtics",
                "game": "NYK @ BOS",
                "canonical_key": "nyk @ bos|nba|2026-09-01",
                "bets_pct": 76,
                "money_pct": 74,
                "open_line": "BOS Celtics @ -130",
                "current_line": "BOS Celtics @ -150",
                "_sort_time": _ts(21, 0),
            },
            {
                "sport": "nba",
                "game_id": "g3",
                "market_display": "MONEYLINE",
                "side_key": "NYK",
                "side": "NYK Knicks",
                "game": "NYK @ BOS",
                "canonical_key": "nyk @ bos|nba|2026-09-01",
                "bets_pct": 24,
                "money_pct": 26,
                "open_line": "NYK Knicks @ +110",
                "current_line": "NYK Knicks @ +130",
                "_sort_time": _ts(21, 0),
            },
        ])
        history = pd.DataFrame([
            {"timestamp": _ts(17, 0), "sport": "ncaab", "game_id": "g2", "market_display": "TOTAL", "side_key": "Over", "current_line": "Over 145.5 @ -130", "bets_pct": 82, "money_pct": 77},
            {"timestamp": _ts(20, 0), "sport": "ncaab", "game_id": "g2", "market_display": "TOTAL", "side_key": "Over", "current_line": "Over 145.5 @ -130", "bets_pct": 82, "money_pct": 77},
            {"timestamp": _ts(17, 0), "sport": "ncaab", "game_id": "g2", "market_display": "TOTAL", "side_key": "Under", "current_line": "Under 145.5 @ -110", "bets_pct": 18, "money_pct": 23},
            {"timestamp": _ts(20, 0), "sport": "ncaab", "game_id": "g2", "market_display": "TOTAL", "side_key": "Under", "current_line": "Under 145.5 @ -110", "bets_pct": 18, "money_pct": 23},
            {"timestamp": _ts(17, 0), "sport": "nba", "game_id": "g3", "market_display": "MONEYLINE", "side_key": "BOS", "current_line": "BOS Celtics @ -130", "bets_pct": 76, "money_pct": 74},
            {"timestamp": _ts(20, 0), "sport": "nba", "game_id": "g3", "market_display": "MONEYLINE", "side_key": "BOS", "current_line": "BOS Celtics @ -150", "bets_pct": 76, "money_pct": 74},
            {"timestamp": _ts(17, 0), "sport": "nba", "game_id": "g3", "market_display": "MONEYLINE", "side_key": "NYK", "current_line": "NYK Knicks @ +110", "bets_pct": 24, "money_pct": 26},
            {"timestamp": _ts(20, 0), "sport": "nba", "game_id": "g3", "market_display": "MONEYLINE", "side_key": "NYK", "current_line": "NYK Knicks @ +130", "bets_pct": 24, "money_pct": 26},
        ])

        board, _ = build_anomaly_outputs(latest, history, pd.DataFrame(), as_of=_ts(16, 0))

        self.assertEqual(board.iloc[0]["reaction"], "Freeze")
        self.assertEqual(board.iloc[0]["flagged_side"], "Over 145.5")
        self.assertIn("Price Risk", board.iloc[0]["context_chips"])
        self.assertIn("Price Risk", board.iloc[0]["reason"])
        self.assertEqual(board.iloc[1]["reaction"], "Follow")

    def test_high_public_watch_explains_the_missing_market_response(self):
        latest = pd.DataFrame([
            {"sport": "nfl", "game_id": "g4", "market_display": "TOTAL", "side_key": "Over", "side": "Over 44.5", "game": "A @ B", "bets_pct": 82, "money_pct": 76, "open_line": "Over 44.5 @ -110", "current_line": "Over 45 @ -105", "_sort_time": _ts(22, 0)},
            {"sport": "nfl", "game_id": "g4", "market_display": "TOTAL", "side_key": "Under", "side": "Under 44.5", "game": "A @ B", "bets_pct": 18, "money_pct": 24, "open_line": "Under 44.5 @ -110", "current_line": "Under 44 @ -115", "_sort_time": _ts(22, 0)},
        ])
        history = pd.DataFrame([
            {"timestamp": _ts(18, 0), "sport": "nfl", "game_id": "g4", "market_display": "TOTAL", "side_key": "Over", "current_line": "Over 44.5 @ -110", "bets_pct": 82, "money_pct": 76},
            {"timestamp": _ts(20, 0), "sport": "nfl", "game_id": "g4", "market_display": "TOTAL", "side_key": "Over", "current_line": "Over 45 @ -105", "bets_pct": 82, "money_pct": 76},
            {"timestamp": _ts(18, 0), "sport": "nfl", "game_id": "g4", "market_display": "TOTAL", "side_key": "Under", "current_line": "Under 44.5 @ -110", "bets_pct": 18, "money_pct": 24},
            {"timestamp": _ts(20, 0), "sport": "nfl", "game_id": "g4", "market_display": "TOTAL", "side_key": "Under", "current_line": "Under 44 @ -115", "bets_pct": 18, "money_pct": 24},
        ])

        board, _ = build_anomaly_outputs(latest, history, pd.DataFrame(), as_of=_ts(17, 0))

        row = board.loc[board["flagged_side"] == "Over 44.5"].iloc[0]
        self.assertEqual(row["reaction"], "Watch")
        self.assertIn("Public Pressure", row["context_chips"])
        self.assertIn("below the confirmed signal threshold", row["reason"])

    def test_smaller_directional_move_is_visible_as_a_developing_read(self):
        latest = pd.DataFrame([
            {"sport": "nfl", "game_id": "g7", "market_display": "TOTAL", "side_key": "Over", "side": "Over 44.5", "game": "A @ B", "bets_pct": 31, "money_pct": 27, "open_line": "Over 44.5 @ -110", "current_line": "Over 45 @ -110", "_sort_time": _ts(22, 0)},
            {"sport": "nfl", "game_id": "g7", "market_display": "TOTAL", "side_key": "Under", "side": "Under 44.5", "game": "A @ B", "bets_pct": 69, "money_pct": 73, "open_line": "Under 44.5 @ -110", "current_line": "Under 44 @ -110", "_sort_time": _ts(22, 0)},
        ])
        history = pd.DataFrame([
            {"timestamp": _ts(18, 0), "sport": "nfl", "game_id": "g7", "market_display": "TOTAL", "side_key": "Over", "current_line": "Over 44.5 @ -110", "bets_pct": 31, "money_pct": 27},
            {"timestamp": _ts(20, 0), "sport": "nfl", "game_id": "g7", "market_display": "TOTAL", "side_key": "Over", "current_line": "Over 45 @ -110", "bets_pct": 31, "money_pct": 27},
            {"timestamp": _ts(18, 0), "sport": "nfl", "game_id": "g7", "market_display": "TOTAL", "side_key": "Under", "current_line": "Under 44.5 @ -110", "bets_pct": 69, "money_pct": 73},
            {"timestamp": _ts(20, 0), "sport": "nfl", "game_id": "g7", "market_display": "TOTAL", "side_key": "Under", "current_line": "Under 44 @ -110", "bets_pct": 69, "money_pct": 73},
        ])

        board, _ = build_anomaly_outputs(latest, history, pd.DataFrame(), as_of=_ts(17, 0))

        row = board.loc[board["flagged_side"] == "Over 44.5"].iloc[0]
        self.assertEqual(row["reaction"], "Watch")
        self.assertIn("Developing Read", row["context_chips"])
        self.assertIn("below the confirmed signal threshold", row["reason"])

    def test_timeline_keeps_one_latest_observation_per_timestamp(self):
        latest = pd.DataFrame([
            {"sport": "nfl", "game_id": "g8", "market_display": "TOTAL", "side_key": "Over", "side": "Over 44.5", "game": "A @ B", "bets_pct": 50, "money_pct": 50, "open_line": "Over 44.5 @ -110", "current_line": "Over 45 @ -110", "_sort_time": _ts(22, 0)},
        ])
        history = pd.DataFrame([
            {"timestamp": _ts(18, 0), "sport": "nfl", "game_id": "g8", "market_display": "TOTAL", "side_key": "Over", "current_line": "Over 44.5 @ -110", "bets_pct": 50, "money_pct": 50},
            {"timestamp": _ts(20, 0), "sport": "nfl", "game_id": "g8", "market_display": "TOTAL", "side_key": "Over", "current_line": "Over 44.5 @ -110", "bets_pct": 50, "money_pct": 50},
            {"timestamp": _ts(20, 0), "sport": "nfl", "game_id": "g8", "market_display": "TOTAL", "side_key": "Over", "current_line": "Over 45 @ -110", "bets_pct": 50, "money_pct": 50},
        ])

        _, events = build_anomaly_outputs(latest, history, pd.DataFrame(), as_of=_ts(17, 0))

        self.assertEqual(len(events), 2)
        self.assertEqual(events.iloc[-1]["line_display"], "O 45 (-110)")


if __name__ == "__main__":
    unittest.main()
