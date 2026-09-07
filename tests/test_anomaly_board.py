from datetime import datetime, timezone
import json
import unittest

import pandas as pd

from anomaly_board import (
    _assign_pair_evidence_roles,
    _is_late_move,
    _key_numbers_crossed,
    build_anomaly_outputs,
    select_market_leaders,
)


def _ts(hour, minute):
    return datetime(2026, 9, 1, hour, minute, tzinfo=timezone.utc).isoformat()


class TestAnomalyBoard(unittest.TestCase):
    def test_evidence_roles_follow_pressure_response_not_freeze_classification(self):
        cases = [
            ("freeze-adverse", "SPREAD", "Freeze", "AGAINST", "Held", True),
            ("freeze-limited", "TOTAL", "Freeze", "LIMITED", "One-Way", True),
            ("follow", "MONEYLINE", "Follow", "TOWARD", "One-Way", False),
            ("contrarian-pair", "SPREAD", "Watch", "AGAINST", "One-Way", True),
            ("watch-adverse", "TOTAL", "Watch", "AGAINST", "Juice Move", True),
            ("limited-favorable", "MONEYLINE", "Watch", "LIMITED", "", False),
            ("active-whipsaw", "TOTAL", "Watch", "LIMITED", "Whipsaw", False),
        ]
        rows = []
        for order, (game_id, market, reaction, response, path, _resists) in enumerate(cases):
            rows.extend([
                {
                    "sport": "nfl", "game_id": game_id, "market_display": market,
                    "flagged_side": f"Pressure {game_id}", "bets_pct": 62, "money_pct": 90,
                    "reaction": reaction, "response_direction": response, "path": path,
                    "anomaly_sort": order + 1, "severity_sort": 50,
                },
                {
                    "sport": "nfl", "game_id": game_id, "market_display": market,
                    "flagged_side": f"Opposing {game_id}", "bets_pct": 38, "money_pct": 10,
                    "reaction": "Contrarian" if game_id == "contrarian-pair" else "Watch",
                    "response_direction": "TOWARD", "path": "Held",
                    "anomaly_sort": order + 1, "severity_sort": 10,
                },
            ])
        rows.extend([
            {"sport": "nfl", "game_id": "neither", "market_display": "SPREAD", "flagged_side": "Side A", "bets_pct": 54, "money_pct": 80, "reaction": "Watch", "response_direction": "AGAINST", "anomaly_sort": 20, "severity_sort": 4},
            {"sport": "nfl", "game_id": "neither", "market_display": "SPREAD", "flagged_side": "Side B", "bets_pct": 46, "money_pct": 20, "reaction": "Watch", "response_direction": "TOWARD", "anomaly_sort": 20, "severity_sort": 3},
        ])
        source = pd.DataFrame(rows)
        classified_before = source[["reaction", "anomaly_sort", "severity_sort"]].copy()

        assigned = _assign_pair_evidence_roles(source)

        pd.testing.assert_frame_equal(
            classified_before.reset_index(drop=True),
            assigned[["reaction", "anomaly_sort", "severity_sort"]].reset_index(drop=True),
        )
        for game_id, _market, _reaction, _response, _path, resists in cases:
            pair = assigned[assigned["game_id"] == game_id]
            self.assertEqual(pair.iloc[0]["evidence_role"], "Pressure Side", msg=game_id)
            self.assertEqual(pair.iloc[1]["evidence_role"], "Resistance Side" if resists else "", msg=game_id)
        self.assertTrue((assigned.loc[assigned["game_id"] == "neither", "evidence_role"] == "").all())

    def test_role_assignment_preserves_rank_scoring_and_directional_lean_rules(self):
        board = pd.DataFrame([
            {"sport": "nfl", "game_id": "contrarian", "market_display": "SPREAD", "game": "A @ B", "flagged_side": "A +6.5", "bets_pct": 22, "money_pct": 28, "reaction": "Contrarian", "response_direction": "TOWARD", "anomaly_sort": 1, "severity_sort": 40, "score": 71},
            {"sport": "nfl", "game_id": "contrarian", "market_display": "SPREAD", "game": "A @ B", "flagged_side": "B -6.5", "bets_pct": 78, "money_pct": 72, "reaction": "Watch", "response_direction": "AGAINST", "anomaly_sort": 1, "severity_sort": 20, "score": 52},
            {"sport": "nfl", "game_id": "follow", "market_display": "MONEYLINE", "game": "C @ D", "flagged_side": "C", "bets_pct": 76, "money_pct": 81, "reaction": "Follow", "response_direction": "TOWARD", "anomaly_sort": 2, "severity_sort": 30, "score": 68},
            {"sport": "nfl", "game_id": "follow", "market_display": "MONEYLINE", "game": "C @ D", "flagged_side": "D", "bets_pct": 24, "money_pct": 19, "reaction": "Watch", "response_direction": "AGAINST", "anomaly_sort": 2, "severity_sort": 10, "score": 45},
            {"sport": "nfl", "game_id": "freeze", "market_display": "TOTAL", "game": "E @ F", "flagged_side": "Over 44", "bets_pct": 82, "money_pct": 75, "reaction": "Freeze", "response_direction": "LIMITED", "action_type": "OBSERVE ONLY", "kpi_eligible": False, "anomaly_sort": 3, "severity_sort": 50, "score": 63},
            {"sport": "nfl", "game_id": "freeze", "market_display": "TOTAL", "game": "E @ F", "flagged_side": "Under 44", "bets_pct": 18, "money_pct": 25, "reaction": "Watch", "response_direction": "LIMITED", "anomaly_sort": 3, "severity_sort": 5, "score": 44},
        ])

        leaders = select_market_leaders(board)

        self.assertEqual(leaders["game_id"].tolist(), ["contrarian", "freeze", "follow"])
        self.assertEqual(leaders["board_rank"].tolist(), [1, 2, 3])
        self.assertEqual(leaders.set_index("game_id")["directional_lean_side"].to_dict(), {
            "contrarian": "A +6.5", "freeze": "", "follow": "C",
        })
        self.assertEqual(leaders.set_index("game_id")["score"].to_dict(), {
            "contrarian": 71, "freeze": 63, "follow": 68,
        })

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
                ], ["Despite 70% bets / 82% money", "line moved 56.5 → 55.5", "Whipsaw risk"],
            ),
            (
                "follow moneyline", "MONEYLINE",
                [
                    {"flagged_side": "CHI Bears", "bets_pct": 79, "money_pct": 71, "open_line": "-142", "current_line": "-162", "reaction": "Follow"},
                    {"flagged_side": "CAR Panthers", "bets_pct": 21, "money_pct": 29, "open_line": "+120", "current_line": "+136", "reaction": "Watch"},
                ], ["CHI Bears has 79% bets / 71% money", "moneyline price moved -142 → -162", "same direction"],
            ),
            (
                "freeze", "SPREAD",
                [
                    {"flagged_side": "Team A -3", "bets_pct": 86, "money_pct": 78, "open_line": "-3 (-110)", "current_line": "-3 (-110)", "reaction": "Freeze"},
                    {"flagged_side": "Team B +3", "bets_pct": 14, "money_pct": 22, "open_line": "+3 (-110)", "current_line": "+3 (-110)", "reaction": "Watch"},
                ], ["86% bets / 78% money", "no meaningful favorable response", "resistance side and evidence anchor"],
            ),
            (
                "juice move", "SPREAD",
                [
                    {"flagged_side": "Dodgers -1.5", "bets_pct": 74, "money_pct": 68, "open_line": "-1.5 (-105)", "current_line": "-1.5 (-135)", "reaction": "Follow", "path": "Juice Move"},
                    {"flagged_side": "Opponents +1.5", "bets_pct": 46, "money_pct": 48, "open_line": "+1.5 (-115)", "current_line": "+1.5 (+115)", "reaction": "Watch"},
                ], ["price/juice moved -105 → -135"],
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

    def test_watch_rationale_anchors_the_toward_side_without_promoting_follow(self):
        board = pd.DataFrame([
            {"sport": "ncaaf", "game_id": "temple", "market_display": "SPREAD", "game": "Rhode Island @ Temple", "flagged_side": "Rhode Island +14.5", "bets_pct": 45, "money_pct": 25, "open_line": "+9.5 (-118)", "current_line": "+14.5 (-110)", "reaction": "Watch", "path": "One-Way", "anomaly_sort": 1, "severity_sort": 20},
            {"sport": "ncaaf", "game_id": "temple", "market_display": "SPREAD", "game": "Rhode Island @ Temple", "flagged_side": "Temple -14.5", "bets_pct": 55, "money_pct": 75, "open_line": "-9.5 (-102)", "current_line": "-14.5 (-110)", "reaction": "Watch", "path": "One-Way", "anomaly_sort": 2, "severity_sort": 10},
        ])
        row = select_market_leaders(board).iloc[0]
        self.assertEqual(row["supported_side"], "")
        self.assertEqual(row["directional_lean_side"], "")
        self.assertIn("Temple -14.5 moved 5.0 points from -9.5 to -14.5", row["market_rationale"])
        self.assertIn("55% bets / 75% money", row["market_rationale"])
        self.assertIn("does not meet the confirmation threshold for Follow", row["market_rationale"])

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
        self.assertEqual(row["read_anchor_side"], "Home -3")
        self.assertEqual(row["supported_side"], "")
        self.assertEqual(row["directional_lean_side"], "")

    def test_pressure_at_key_with_adverse_juice_anchors_resistance_without_confirmed_read(self):
        latest = pd.DataFrame([
            {"sport": "ncaaf", "game_id": "smu-fsu", "market_display": "SPREAD", "side_key": "SMU", "side": "SMU -3", "game": "SMU @ Florida State", "bets_pct": 78, "money_pct": 78, "open_line": "SMU -3 @ -110", "current_line": "SMU -3 @ -105", "_sort_time": _ts(20, 0)},
            {"sport": "ncaaf", "game_id": "smu-fsu", "market_display": "SPREAD", "side_key": "FSU", "side": "Florida State +3", "game": "SMU @ Florida State", "bets_pct": 22, "money_pct": 22, "open_line": "Florida State +3 @ -110", "current_line": "Florida State +3 @ -115", "_sort_time": _ts(20, 0)},
        ])
        history = []
        smu_path = ["SMU -3 @ -110", "SMU -2.5 @ -120", "SMU -3 @ -105", "SMU -3 @ -105", "SMU -3 @ -105", "SMU -3 @ -105"]
        fsu_path = ["Florida State +3 @ -110", "Florida State +2.5 @ +100", "Florida State +3 @ -115", "Florida State +3 @ -115", "Florida State +3 @ -115", "Florida State +3 @ -115"]
        for offset, (smu_line, fsu_line) in enumerate(zip(smu_path, fsu_path), start=10):
            history.extend([
                {"timestamp": _ts(offset, 0), "sport": "ncaaf", "game_id": "smu-fsu", "market_display": "SPREAD", "side_key": "SMU", "current_line": smu_line, "bets_pct": 78, "money_pct": 78},
                {"timestamp": _ts(offset, 0), "sport": "ncaaf", "game_id": "smu-fsu", "market_display": "SPREAD", "side_key": "FSU", "current_line": fsu_line, "bets_pct": 22, "money_pct": 22},
            ])

        board, _ = build_anomaly_outputs(latest, pd.DataFrame(history), pd.DataFrame(), as_of=_ts(16, 0))
        pressure = board.loc[board["flagged_side"] == "SMU -3"].iloc[0]
        resistance = board.loc[board["flagged_side"] == "Florida State +3"].iloc[0]
        leader = select_market_leaders(board).iloc[0]
        sides = {side["flagged_side"]: side for side in json.loads(leader["market_sides"])}

        self.assertEqual(pressure["reaction"], "Freeze")
        self.assertEqual(pressure["response_direction"], "AGAINST")
        self.assertTrue(pressure["whipsaw_recovered"])
        self.assertEqual(pressure["key_number_pinned"], "K3")
        self.assertEqual(pressure["action_type"], "OBSERVE ONLY")
        self.assertFalse(pressure["kpi_eligible"])
        self.assertIn("Public Pressure", pressure["context_chips"])
        self.assertEqual(sides["SMU -3"]["evidence_role"], "Pressure Side")
        self.assertEqual(sides["SMU -3"]["evidence_polarity"], "adverse")
        self.assertEqual(sides["Florida State +3"]["evidence_role"], "Resistance Side")
        self.assertEqual(leader["read_anchor_side"], "Florida State +3")
        self.assertEqual(leader["supported_side"], "")
        self.assertEqual(leader["directional_lean_side"], "")
        self.assertIn("price moved against that pressure", leader["market_rationale"])
        self.assertIn("durable reset", leader["market_rationale"])

    def test_non_actionable_freeze_does_not_mask_confirmed_contrarian_counterpart(self):
        board = pd.DataFrame([
            {"sport": "ncaaf", "game_id": "smu-fsu", "market_display": "SPREAD", "game": "SMU @ Florida State", "flagged_side": "SMU -2.5", "reaction": "Freeze", "action_type": "OBSERVE ONLY", "action_side": "", "kpi_eligible": False, "evidence_role": "Pressure Side", "anomaly_sort": 1},
            {"sport": "ncaaf", "game_id": "smu-fsu", "market_display": "SPREAD", "game": "SMU @ Florida State", "flagged_side": "Florida State +2.5", "reaction": "Contrarian", "evidence_role": "Resistance Side", "anomaly_sort": 2},
        ])
        leader = select_market_leaders(board).iloc[0]
        self.assertEqual(leader["read_anchor_side"], "Florida State +2.5")
        self.assertEqual(leader["supported_side"], "Florida State +2.5")
        self.assertEqual(leader["directional_lean_side"], "Florida State +2.5")

    def test_conflicting_directional_reads_resolve_neutral_instead_of_using_order(self):
        board = pd.DataFrame([
            {"sport": "nfl", "game_id": "conflict", "market_display": "SPREAD", "game": "Away @ Home", "flagged_side": "Away +3", "reaction": "Contrarian", "anomaly_sort": 1},
            {"sport": "nfl", "game_id": "conflict", "market_display": "SPREAD", "game": "Away @ Home", "flagged_side": "Home -3", "reaction": "Follow", "anomaly_sort": 2},
        ])
        leader = select_market_leaders(board).iloc[0]
        self.assertEqual(leader["supported_side"], "")
        self.assertEqual(leader["directional_lean_side"], "")

    def test_ranked_watch_does_not_require_a_supported_side(self):
        board = pd.DataFrame([
            {"sport": "nfl", "game_id": "watch", "market_display": "TOTAL", "game": "Away @ Home", "flagged_side": "Over 44.5", "reaction": "Watch", "anomaly_sort": 1},
            {"sport": "nfl", "game_id": "watch", "market_display": "TOTAL", "game": "Away @ Home", "flagged_side": "Under 44.5", "reaction": "Watch", "anomaly_sort": 2},
        ])
        leader = select_market_leaders(board).iloc[0]
        self.assertEqual(leader["board_rank"], 1)
        self.assertEqual(leader["supported_side"], "")

    def test_heavy_favorite_context_does_not_hide_confirmed_contrarian_explanation(self):
        board = pd.DataFrame([
            {"sport": "ncaaf", "game_id": "heavy", "market_display": "MONEYLINE", "game": "Duke @ Illinois", "flagged_side": "Duke", "bets_pct": 12, "money_pct": 18, "open_line": "+270", "current_line": "+190", "reaction": "Contrarian", "response_direction": "TOWARD", "anomaly_sort": 1},
            {"sport": "ncaaf", "game_id": "heavy", "market_display": "MONEYLINE", "game": "Duke @ Illinois", "flagged_side": "Illinois", "bets_pct": 88, "money_pct": 82, "open_line": "-340", "current_line": "-230", "reaction": "Watch", "response_direction": "AGAINST", "context_chips": "Heavy Favorite", "anomaly_sort": 2},
        ])
        leader = select_market_leaders(board).iloc[0]
        self.assertEqual(leader["supported_side"], "Duke")
        self.assertIn("toward Duke", leader["market_rationale"])
        self.assertIn("does not create the supported side", leader["market_rationale"])
        self.assertNotIn("split remains context only", leader["market_rationale"])

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
        self.assertTrue(bool(row["active_worsening_reversal"]))
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

    def test_high_public_with_only_a_subthreshold_favorable_move_is_freeze(self):
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
        self.assertEqual(row["reaction"], "Freeze")
        self.assertIn("Public Pressure", row["context_chips"])
        self.assertIn("without a meaningful favorable response", row["reason"])

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
