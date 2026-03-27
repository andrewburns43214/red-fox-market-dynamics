import csv
import unittest
from pathlib import Path
from copy import deepcopy

from scoring_reaction import classify_reaction_live, score_reaction
from tests.fixtures.scoring_fixtures import SCORING_FIXTURES


def _base_row(**overrides):
    row = {
        "sport": "ncaab",
        "market_display": "SPREAD",
        "side": "Nebraska -1.5",
        "timing_bucket": "LATE",
        "bets_pct": 71,
        "money_pct": 67,
        "open_line_val": -2.5,
        "current_line_val": -1.5,
        "prev_line_val": -1.5,
        "open_odds": -110,
        "current_odds": -112,
        "prev_odds": -112,
        "line_move_open": 1.0,
        "line_move_prev": 0.0,
        "odds_move_open": -2.0,
        "odds_move_prev": 0.0,
        "effective_move_mag": 1.0,
        "line_dir_changes": 0,
        "line_settled_ticks": 2,
        "line_last_dir": -1,
        "line_max_move": 1.0,
    }
    row.update(overrides)
    return row


class TestReactionEngine(unittest.TestCase):
    def test_live_adapter_derives_freeze_subtypes_from_market_rows(self):
        expected_by_fixture = {
            "freeze_resistance_houston_illinois": "FREEZE_RESISTANCE",
            "freeze_weak_mild_morning_pressure": "FREEZE_WEAK",
            "freeze_key_number_nfl_spread_three": "FREEZE_KEY_NUMBER",
            "freeze_balanced_real_money_dog": "FREEZE_BALANCED",
        }

        for fixture in SCORING_FIXTURES:
            name = fixture["name"]
            if name not in expected_by_fixture:
                continue
            with self.subTest(fixture=name):
                market_rows = []
                for row in fixture["market_rows"]:
                    clean = deepcopy(row)
                    clean.pop("meaningful_pressure", None)
                    clean.pop("balanced_counteraction", None)
                    clean.pop("key_number_pinned", None)
                    clean.pop("market_stale", None)
                    clean.pop("freeze_subtype_candidate", None)
                    market_rows.append(clean)
                if len(market_rows) == 1:
                    market_rows.append(self._synthetic_opposite_row(market_rows[0]))
                evaluated_row = next(
                    row for row in market_rows if row["side"] == fixture["evaluated_side"]
                )
                result = classify_reaction_live(
                    evaluated_row,
                    market_rows=market_rows,
                    evaluated_side=fixture["evaluated_side"],
                    pressure_side=fixture["pressure_side"],
                )
                self.assertEqual(
                    result["semantic_reaction_state"],
                    expected_by_fixture[name],
                )
                self.assertEqual(result["semantic_source"], "market_context")

    def _synthetic_opposite_row(self, row):
        other = deepcopy(row)
        side = str(other.get("side", ""))
        if other.get("market_display") == "SPREAD":
            if "+" in side:
                other["side"] = side.replace("+", "-", 1)
            elif "-" in side:
                other["side"] = side.replace("-", "+", 1)
            for key in ("open_line_val", "current_line_val", "prev_line_val"):
                if other.get(key) is not None:
                    other[key] = -float(other[key])
        other["bets_pct"] = max(0.0, 100.0 - float(row.get("bets_pct", 0) or 0))
        other["money_pct"] = max(0.0, 100.0 - float(row.get("money_pct", 0) or 0))
        return other

    def test_fade_state_never_recommends_same_side(self):
        result = score_reaction(_base_row())
        self.assertEqual(result["reaction_state"], "FADE")
        self.assertEqual(result["decision"], "NO_BET")
        self.assertLess(result["reaction_score"], 50.0)

    def test_follow_state_can_still_recommend_same_side(self):
        result = score_reaction(
            _base_row(
                side="Nebraska -2.5",
                open_line_val=-1.5,
                current_line_val=-2.5,
                prev_line_val=-2.5,
                line_move_open=-1.0,
            )
        )
        self.assertEqual(result["reaction_state"], "FOLLOW")
        self.assertIn(result["decision"], {"LEAN", "BET"})

    def test_low_bet_side_moving_toward_side_is_follow_not_fade(self):
        result = score_reaction(
            _base_row(
                side="Iowa +1.5",
                bets_pct=29,
                money_pct=33,
                open_line_val=2.5,
                current_line_val=1.5,
                prev_line_val=1.5,
                line_move_open=-1.0,
            )
        )
        self.assertEqual(result["reaction_state"], "FOLLOW")
        self.assertIn(result["decision"], {"LEAN", "BET"})

    def test_fixture_has_no_fade_bets(self):
        fixture = Path("tests") / "fixtures" / "reaction_semantic_fixture.csv"
        with fixture.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

        bad = []
        for row in rows:
            result = score_reaction(_base_row(**row))
            if result["reaction_state"] == "FADE" and result["decision"] != "NO_BET":
                bad.append((row["side"], result))

        self.assertFalse(bad)


if __name__ == "__main__":
    unittest.main()
