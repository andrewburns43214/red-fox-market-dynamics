from datetime import datetime, timezone
import json
import unittest

import pandas as pd

from anomaly_board import _is_late_move, _key_numbers_crossed, build_anomaly_outputs, select_market_leaders


def _ts(hour, minute):
    return datetime(2026, 9, 1, hour, minute, tzinfo=timezone.utc).isoformat()


class TestAnomalyBoard(unittest.TestCase):
    def test_late_requires_an_upcoming_kickoff_inside_its_closing_window(self):
        points = [
            {"value": 50.0, "implied_pct": None},
            {"value": 50.5, "implied_pct": None},
            {"value": 51.0, "implied_pct": None},
        ]
        self.assertTrue(_is_late_move(points, "TOTAL", 1.0, 4.0, "nfl"))
        self.assertFalse(_is_late_move(points, "TOTAL", 1.0, 7.0, "nfl"))
        self.assertFalse(_is_late_move(points, "TOTAL", 1.0, -0.1, "nfl"))

    def test_board_publishes_one_evidence_leader_per_market(self):
        board = pd.DataFrame([
            {"sport": "nfl", "game_id": "g1", "market_display": "SPREAD", "flagged_side": "SEA +3", "anomaly_sort": 3, "severity_sort": 80, "reaction": "Contrarian", "recorded_reaction": "Contrarian", "game": "NE @ SEA"},
            {"sport": "nfl", "game_id": "g1", "market_display": "SPREAD", "flagged_side": "NE -3", "anomaly_sort": 1, "severity_sort": 70, "game": "NE @ SEA"},
            {"sport": "nfl", "game_id": "g1", "market_display": "TOTAL", "flagged_side": "Over 44.5", "anomaly_sort": 2, "severity_sort": 60, "game": "NE @ SEA"},
        ])

        leaders = select_market_leaders(board)

        self.assertEqual(len(leaders), 2)
        self.assertEqual(leaders.iloc[0]["flagged_side"], "SEA +3")
        self.assertEqual(leaders.iloc[0]["board_rank"], 1)

    def test_market_leader_retains_the_exact_two_side_payload(self):
        board = pd.DataFrame([
            {"sport": "nfl", "game_id": "g1", "market_display": "SPREAD", "flagged_side": "NE +3", "bets_pct": 31, "money_pct": 15, "open_line": "NE +3 (-110)", "current_line": "NE +3.5 (-118)", "reaction": "Contrarian", "path": "One-Way", "anomaly_sort": 1, "severity_sort": 80, "game": "NE @ PIT"},
            {"sport": "nfl", "game_id": "g1", "market_display": "SPREAD", "flagged_side": "PIT -3", "bets_pct": 69, "money_pct": 85, "open_line": "PIT -3 (-110)", "current_line": "PIT -3.5 (-102)", "reaction": "Public Pressure", "anomaly_sort": 4, "severity_sort": 20, "game": "NE @ PIT"},
        ])

        leaders = select_market_leaders(board)

        self.assertEqual(len(leaders), 1)
        sides = json.loads(leaders.iloc[0]["market_sides"])
        self.assertEqual([side["flagged_side"] for side in sides], ["NE +3", "PIT -3"])
        self.assertEqual(sides[1]["current_line"], "PIT -3.5 (-102)")

    def test_canonical_market_rationale_matrix(self):
        """Every public read/context family gets a factual paired explanation."""
        cases = [
            (
                "contrarian total", "TOTAL",
                [
                    {"flagged_side": "Over 56.5", "bets_pct": 70, "money_pct": 82, "open_line": "O 56.5 (-105)", "current_line": "O 55.5 (-115)", "reaction": "Watch", "context_chips": "Public Pressure"},
                    {"flagged_side": "Under 56.5", "bets_pct": 30, "money_pct": 18, "open_line": "U 56.5 (-115)", "current_line": "U 55.5 (-105)", "reaction": "Contrarian", "context_chips": "Whipsaw"},
                ], ["Despite 70% bets / 82% money", "total fell 56.5 → 55.5", "Whipsaw risk"],
            ),
            (
                "follow moneyline", "MONEYLINE",
                [
                    {"flagged_side": "CHI Bears", "bets_pct": 79, "money_pct": 71, "open_line": "-142", "current_line": "-162", "reaction": "Follow"},
                    {"flagged_side": "CAR Panthers", "bets_pct": 21, "money_pct": 29, "open_line": "+120", "current_line": "+136", "reaction": "Watch"},
                ], ["CHI Bears has 79% bets / 71% money", "-142 → -162", "confirming the same direction"],
            ),
            (
                "freeze", "SPREAD",
                [
                    {"flagged_side": "Team A -3", "bets_pct": 86, "money_pct": 78, "open_line": "-3 (-110)", "current_line": "-3 (-110)", "reaction": "Freeze"},
                    {"flagged_side": "Team B +3", "bets_pct": 14, "money_pct": 22, "open_line": "+3 (-110)", "current_line": "+3 (-110)", "reaction": "Watch"},
                ], ["86% bets / 78% money", "near its opening number", "meaningful response"],
            ),
            (
                "juice move", "SPREAD",
                [
                    {"flagged_side": "Dodgers -1.5", "bets_pct": 74, "money_pct": 68, "open_line": "-1.5 (-105)", "current_line": "-1.5 (-135)", "reaction": "Follow", "path": "Juice Move"},
                    {"flagged_side": "Opponents +1.5", "bets_pct": 46, "money_pct": 48, "open_line": "+1.5 (-115)", "current_line": "+1.5 (+115)", "reaction": "Watch"},
                ], ["stayed -1.5", "juice moved -105 → -135"],
            ),
            (
                "watch whipsaw", "TOTAL",
                [
                    {"flagged_side": "Over 59.5", "bets_pct": 51, "money_pct": 49, "open_line": "O 59.5 (-110)", "current_line": "O 59.5 (-110)", "reaction": "Watch", "path": "Whipsaw"},
                    {"flagged_side": "Under 59.5", "bets_pct": 49, "money_pct": 51, "open_line": "U 59.5 (-110)", "current_line": "U 59.5 (-110)", "reaction": "Watch"},
                ], ["reversed direction", "neither side with sustained control"],
            ),
            (
                "low bets high money and keys", "SPREAD",
                [
                    {"flagged_side": "Albany +18.5", "bets_pct": 31, "money_pct": 81, "open_line": "+24.5 (-110)", "current_line": "+18.5 (-110)", "reaction": "Watch", "context_chips": "Low Bets / High $ | K10", "key_numbers_crossed": "K10 | K14"},
                    {"flagged_side": "Favorite -18.5", "bets_pct": 69, "money_pct": 19, "open_line": "-24.5 (-110)", "current_line": "-18.5 (-110)", "reaction": "Watch"},
                ], ["Only 31% of tickets", "+24.5 (-110) → +18.5 (-110)", "key numbers 10, 14"],
            ),
        ]
        for name, market, sides, expected in cases:
            rows = []
            for index, side in enumerate(sides):
                rows.append({"sport": "nfl", "game_id": name, "market_display": market, "game": "Away @ Home", "anomaly_sort": index + 1, "severity_sort": 20 - index, **side})
            rationale = select_market_leaders(pd.DataFrame(rows)).iloc[0]["market_rationale"]
            for fragment in expected:
                self.assertIn(fragment, rationale, msg=f"{name}: {rationale}")

    def test_all_crossed_key_numbers_are_preserved(self):
        points = [{"value": -9.5}, {"value": -14.5}]
        self.assertEqual(_key_numbers_crossed("nfl", "SPREAD", points), ["K10", "K14"])

    def test_all_crossed_key_numbers_flow_to_board_context(self):
        latest = pd.DataFrame([
            {"sport": "nfl", "game_id": "keys", "market_display": "SPREAD", "side_key": "Away", "side": "Away +9.5", "game": "Away @ Home", "bets_pct": 30, "money_pct": 20, "open_line": "Away +9.5 (-110)", "current_line": "Away +14.5 (-110)", "_sort_time": _ts(17, 0)},
            {"sport": "nfl", "game_id": "keys", "market_display": "SPREAD", "side_key": "Home", "side": "Home -9.5", "game": "Away @ Home", "bets_pct": 70, "money_pct": 80, "open_line": "Home -9.5 (-110)", "current_line": "Home -14.5 (-110)", "_sort_time": _ts(17, 0)},
        ])
        history = pd.DataFrame([
            {"timestamp": _ts(15, 0), "sport": "nfl", "game_id": "keys", "market_display": "SPREAD", "side_key": "Away", "current_line": "Away +9.5 (-110)", "bets_pct": 30, "money_pct": 20},
            {"timestamp": _ts(16, 0), "sport": "nfl", "game_id": "keys", "market_display": "SPREAD", "side_key": "Away", "current_line": "Away +14.5 (-110)", "bets_pct": 30, "money_pct": 20},
            {"timestamp": _ts(15, 0), "sport": "nfl", "game_id": "keys", "market_display": "SPREAD", "side_key": "Home", "current_line": "Home -9.5 (-110)", "bets_pct": 70, "money_pct": 80},
            {"timestamp": _ts(16, 0), "sport": "nfl", "game_id": "keys", "market_display": "SPREAD", "side_key": "Home", "current_line": "Home -14.5 (-110)", "bets_pct": 70, "money_pct": 80},
        ])
        board, _ = build_anomaly_outputs(latest, history, pd.DataFrame(), as_of=_ts(17, 0))
        self.assertIn("K10", board.iloc[0]["context_chips"])
        self.assertIn("K14", board.iloc[0]["context_chips"])

    def test_canonical_rationale_keeps_path_context_and_never_claims_a_held_line_moved(self):
        board = pd.DataFrame([
            {"sport": "nfl", "game_id": "g-path", "market_display": "MONEYLINE", "game": "Away @ Home", "flagged_side": "Away", "bets_pct": 79, "money_pct": 71, "open_line": "+295", "current_line": "+295", "reaction": "Follow", "path": "Late", "anomaly_sort": 1, "severity_sort": 80},
            {"sport": "nfl", "game_id": "g-path", "market_display": "MONEYLINE", "game": "Away @ Home", "flagged_side": "Home", "bets_pct": 21, "money_pct": 29, "open_line": "-350", "current_line": "-350", "reaction": "Watch", "anomaly_sort": 2, "severity_sort": 10},
        ])
        rationale = select_market_leaders(board).iloc[0]["market_rationale"]
        self.assertIn("held at +295", rationale)
        self.assertIn("no price movement is implied", rationale)
        self.assertIn("closing window", rationale)
        self.assertNotIn("+295 → +295", rationale)

    def test_follow_whipsaw_mentions_the_reversal_in_canonical_rationale(self):
        board = pd.DataFrame([
            {"sport": "nfl", "game_id": "g-whip", "market_display": "SPREAD", "game": "Away @ Home", "flagged_side": "Away +3", "bets_pct": 78, "money_pct": 68, "open_line": "+3 (-110)", "current_line": "+2.5 (-110)", "reaction": "Follow", "path": "Whipsaw", "anomaly_sort": 1, "severity_sort": 80},
            {"sport": "nfl", "game_id": "g-whip", "market_display": "SPREAD", "game": "Away @ Home", "flagged_side": "Home -3", "bets_pct": 22, "money_pct": 32, "open_line": "-3 (-110)", "current_line": "-2.5 (-110)", "reaction": "Watch", "anomaly_sort": 2, "severity_sort": 10},
        ])
        rationale = select_market_leaders(board).iloc[0]["market_rationale"]
        self.assertIn("path later reversed", rationale)
        self.assertIn("Whipsaw risk", rationale)

    def test_read_anchor_and_directional_lean_are_distinct_for_freeze(self):
        board = pd.DataFrame([
            {"sport": "nfl", "game_id": "freeze", "market_display": "SPREAD", "game": "Away @ Home", "flagged_side": "Away +3", "bets_pct": 80, "money_pct": 70, "open_line": "+3 (-110)", "current_line": "+3 (-110)", "reaction": "Freeze", "action_type": "OBSERVE ONLY", "action_side": "", "kpi_eligible": False, "anomaly_sort": 1, "severity_sort": 80},
            {"sport": "nfl", "game_id": "freeze", "market_display": "SPREAD", "game": "Away @ Home", "flagged_side": "Home -3", "bets_pct": 20, "money_pct": 30, "open_line": "-3 (-110)", "current_line": "-3 (-110)", "reaction": "Watch", "anomaly_sort": 2, "severity_sort": 10},
        ])
        row = select_market_leaders(board).iloc[0]
        self.assertEqual(row["read_anchor_side"], "Away +3")
        self.assertEqual(row["directional_lean_side"], "")

    def test_board_rank_keeps_more_severe_like_signals_ahead_of_alphabetical_order(self):
        board = pd.DataFrame([
            {"sport": "nfl", "game_id": "g5", "market_display": "SPREAD", "flagged_side": "Alpha +3", "reaction": "Freeze", "anomaly_sort": 3, "severity_sort": 10, "game": "Alpha @ Beta"},
            {"sport": "nfl", "game_id": "g6", "market_display": "SPREAD", "flagged_side": "Zulu +3", "reaction": "Freeze", "anomaly_sort": 3, "severity_sort": 50, "game": "Zulu @ Yankee"},
        ])

        leaders = select_market_leaders(board)

        self.assertEqual(leaders.iloc[0]["flagged_side"], "Zulu +3")

    def test_current_market_move_ranks_above_freeze_but_below_contrarian(self):
        board = pd.DataFrame([
            {"sport": "nfl", "game_id": "g1", "market_display": "SPREAD", "flagged_side": "Move +3", "reaction": "Watch", "context_chips": "Market Move", "anomaly_sort": 6.75, "severity_sort": 10, "game": "Move @ Home"},
            {"sport": "nfl", "game_id": "g2", "market_display": "SPREAD", "flagged_side": "Freeze -3", "reaction": "Freeze", "context_chips": "", "anomaly_sort": 3, "severity_sort": 50, "game": "Freeze @ Home"},
            {"sport": "nfl", "game_id": "g3", "market_display": "SPREAD", "flagged_side": "Contra +3", "reaction": "Contrarian", "context_chips": "Market Move", "anomaly_sort": 2, "severity_sort": 5, "game": "Contra @ Home"},
        ])

        leaders = select_market_leaders(board)

        self.assertEqual(leaders["flagged_side"].tolist(), ["Contra +3", "Move +3", "Freeze -3"])

    def test_price_risk_market_move_follows_a_clean_market_move(self):
        board = pd.DataFrame([
            {"sport": "nfl", "game_id": "g1", "market_display": "SPREAD", "flagged_side": "Clean +3", "reaction": "Watch", "context_chips": "Market Move", "anomaly_sort": 6.75, "severity_sort": 10, "game": "Clean @ Home"},
            {"sport": "nfl", "game_id": "g2", "market_display": "MONEYLINE", "flagged_side": "Risky dog", "reaction": "Watch", "context_chips": "Market Move | Price Risk", "anomaly_sort": 6.9, "severity_sort": 20, "game": "Risky @ Home"},
            {"sport": "nfl", "game_id": "g3", "market_display": "SPREAD", "flagged_side": "Freeze -3", "reaction": "Freeze", "context_chips": "", "anomaly_sort": 3, "severity_sort": 50, "game": "Freeze @ Home"},
        ])

        leaders = select_market_leaders(board)

        self.assertEqual(leaders["flagged_side"].tolist(), ["Clean +3", "Risky dog", "Freeze -3"])

    def test_price_risk_contrarian_follows_a_clean_contrarian(self):
        board = pd.DataFrame([
            {"sport": "ncaaf", "game_id": "g1", "market_display": "SPREAD", "flagged_side": "Risky +24", "reaction": "Contrarian", "context_chips": "Price Risk", "anomaly_sort": 0, "severity_sort": 90, "game": "Risky @ Home"},
            {"sport": "ncaaf", "game_id": "g2", "market_display": "TOTAL", "flagged_side": "Clean Under", "reaction": "Contrarian", "context_chips": "", "anomaly_sort": 2, "severity_sort": 10, "game": "Clean @ Home"},
        ])

        leaders = select_market_leaders(board)

        self.assertEqual(leaders["flagged_side"].tolist(), ["Clean Under", "Risky +24"])

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
