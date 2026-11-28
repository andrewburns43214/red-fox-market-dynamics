import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from team_market_history import merge_market_history, reference_quotes, sportsbook_history


def game(uid, keys=("army", "florida atlantic"), scores=(56, 24), **fields):
    return {"uid": uid, "keys": list(keys), "names": list(keys), "scores": list(scores),
            "start": "2026-10-17T16:00:00.000000+00:00", **fields}


class MarketHistoryTest(unittest.TestCase):
    def test_number_can_grade_without_being_used_for_roi(self):
        history = reference_quotes("ncaaf", game("fa"), "source",
                                   spread=((None, -2.5), (None, 2.5)), total=((59, 60), (59, 60)),
                                   money=(-146, 130))
        self.assertEqual(history["SPREAD"]["army"]["grade"], "win")
        self.assertIsNone(history["SPREAD"]["army"]["units"])
        self.assertEqual(history["TOTAL"]["under"]["grade"], "loss")
        self.assertIsNone(history["TOTAL"]["under"]["units"])
        self.assertAlmostEqual(history["MONEYLINE"]["army"]["units"], 100/146, places=5)

    def test_reference_matches_reversed_score_orientation_without_duplicating(self):
        archive = game("rf")
        external = game("public", ("florida atlantic", "army"), (24, 56))
        external["quotes"] = reference_quotes("ncaaf", external, "source", money=(130, -146))
        result = merge_market_history([archive], [[external, dict(external, uid="duplicate")]], sport="ncaaf")
        self.assertFalse(result["games"])
        self.assertEqual(result["quotes"]["rf"]["MONEYLINE"]["army"]["price"], -146)
        self.assertFalse(merge_market_history([archive], [[game("wrong", scores=(25, 24))]], sport="ncaaf")["games"])

    def test_distinct_mlb_doubleheader_game_kept_and_merged_by_score(self):
        a = game("archive")
        first = game("source1", date="2026-10-17")
        second = game("source2", scores=(3, 4), date="2026-10-17")
        result = merge_market_history([a], [[first, second], [dict(second, uid="same second game")]], sport="mlb")
        self.assertEqual([g["uid"] for g in result["games"]], ["source2"])

    def test_archive_quotes_need_paired_sides_and_verified_pregame_times(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "odds.csv"
            rows = []
            for stamp, spreads, prices in [
                ("2026-10-17T14:00:00Z", [-2.5, 2.5], [-110, -110]),
                ("2026-10-17T15:00:00Z", [-4.5, 4.5], [-105, -115]),
                ("2026-10-17T15:30:00Z", [-5.5], [-110]),
                ("2026-10-17T16:00:00Z", [-9.5, 9.5], [-110, -110]),
            ]:
                for side, line, price in zip(("army", "florida atlantic"), spreads, prices):
                    rows.append(dict(timestamp=stamp, sport="ncaaf", bookmaker="test",
                                     away_team_norm="army", home_team_norm="florida atlantic",
                                     commence_time="2026-10-17T16:00:00Z", market="SPREAD",
                                     side=side, line=line, odds_american=price))
            with path.open("w", newline="", encoding="utf-8") as file:
                out = csv.DictWriter(file, fieldnames=rows[0])
                out.writeheader()
                out.writerows(rows)
            data = {"quotes": {}, "games": []}
            sportsbook_history(path, "ncaaf", [game("archive")], data, lambda: True)
        choice = data["quotes"]["archive"]["SPREAD"]["army"]
        self.assertEqual((choice["open_line"], choice["line"], choice["price"]), (-2.5, -4.5, -105))
        self.assertEqual(choice["grade"], "win")
