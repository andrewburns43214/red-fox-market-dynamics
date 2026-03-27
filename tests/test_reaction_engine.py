import csv
import unittest
from pathlib import Path

from scoring_reaction import score_reaction


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
