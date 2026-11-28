import csv
import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from history_blocks import encode
from team_intelligence import graded_quotes, identities, nfl_merge, published_rows, season, valid_outcomes


class TeamIntelligenceTest(unittest.TestCase):
    start = "2026-11-29T21:25:00.000000+00:00"
    checkpoint = "2026-11-29T21:15:00.000000+00:00"
    observed = "2026-11-29T21:14:00.000000+00:00"

    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.addCleanup(self.db.close)
        self.db.executescript("""
            CREATE TABLE comparison_windows(sport, market, minutes, event_id, decision_at, entry_at, scheduled_at, evidence);
            CREATE TABLE outcomes(sport, event_id, source, status, provider_event_id, observed_at, digest, evidence);
        """)

    def window(self, sides, checkpoint=None, observed=None, start=None, verified_start=None):
        evidence = {"sides": sides, "verified_start": verified_start}
        self.db.execute(
            "INSERT INTO comparison_windows VALUES (?, 'SPREAD', 10, ?, ?, ?, ?, ?)",
            ("nfl", "fixture", checkpoint or self.checkpoint, observed or self.observed, start or self.start, encode(evidence)),
        )

    def test_pricing_and_grading_use_selected_side_and_latest_valid_pregame_capture(self):
        names, keys = identities("KC Chiefs @ BUF Bills", "nfl")
        match = {"vendor": "fixture", "keys": keys, "start": self.start}
        self.window([
            {"side": keys[0], "line": 3.5, "price": 110},
            {"side": keys[1], "line": -3.5, "price": -120},
        ])
        self.window([
            {"side": keys[0], "line": 7, "price": -110},
            {"side": keys[1], "line": -7, "price": 100},
        ], checkpoint=self.start, observed=self.start)
        outcome = {"matchup": " @ ".join(names), "score_away": 24, "score_home": 27}
        quotes = graded_quotes(self.db, "nfl", match, "SPREAD", ("verified", outcome))
        self.assertEqual((quotes[keys[0]]["price"], quotes[keys[0]]["grade"], quotes[keys[0]]["units"]), (110, "win", 1.1))
        self.assertEqual((quotes[keys[1]]["price"], quotes[keys[1]]["grade"], quotes[keys[1]]["units"]), (-120, "loss", -1))

    def test_incompatible_fixture_schedule_and_post_start_checkpoint_are_excluded(self):
        names, keys = identities("KC Chiefs @ BUF Bills", "nfl")
        match = {"vendor": "fixture", "keys": keys, "start": self.start}
        sides = [{"side": keys[0], "line": 3.5, "price": 110}, {"side": keys[1], "line": -3.5, "price": -120}]
        self.window(sides, verified_start=self.observed)
        self.window(sides, start="2026-11-30T21:25:00.000000+00:00")
        outcome = {"matchup": " @ ".join(names), "score_away": 24, "score_home": 27}
        self.assertEqual(graded_quotes(self.db, "nfl", match, "SPREAD", ("verified", outcome)), {})

    def test_replacement_fighter_cannot_inherit_completed_bout(self):
        names, keys = identities("JosÃ© Aldo vs Rob Font", "ufc")
        raw = {"matchup": "JosÃ© Aldo vs Marlon Vera", "winner": "josealdo", "actual_start": self.start}
        catalog = {"fixture": {"start": self.start, "keys": keys, "names": names, "season": 2026}}
        self.db.execute("INSERT INTO outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        ("ufc", "fixture", "verified_research_ufc", "verified", "bout", self.start, "hash", json.dumps(raw)))
        outcomes, conflicts, _ = valid_outcomes(self.db, "ufc", catalog, "2026-11-30T21:25:00.000000+00:00")
        self.assertEqual(outcomes, {})
        self.assertEqual(conflicts, {"fixture"})

    def test_verified_mlb_doubleheaders_remain_distinct(self):
        names, keys = identities("NY Mets @ NY Yankees", "mlb")
        catalog = {}
        for i, start in enumerate(["2026-06-12T16:00:00+00:00", "2026-06-12T22:00:00+00:00"]):
            event = "fixture" + str(i)
            raw = {"matchup": " @ ".join(names), "score_away": 1 + i, "score_home": 3}
            catalog[event] = {"start": start, "keys": keys, "names": names, "season": 2026}
            self.db.execute("INSERT INTO outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            ("mlb", event, "verified_research_mlb", "verified", "provider" + str(i), start, "hash", json.dumps(raw)))
        outcomes, conflicts, _ = valid_outcomes(self.db, "mlb", catalog, self.start)
        self.assertEqual(set(outcomes), set(catalog))
        self.assertEqual(conflicts, set())

    def test_nfl_exact_fixture_links_market_history_and_counts_one_score(self):
        names, keys = identities("KC Chiefs @ BUF Bills", "nfl")
        schedule = "2026-11-29T21:15:00.000000+00:00"
        archived = dict(uid="fixture", vendor="fixture", provider="espn", start=self.start, keys=keys, names=names, scores=[24, 27])
        external = {"games": [dict(uid="nflverse", provider="espn", start=schedule, keys=keys, names=names, scores=[24, 27],
                                   stage="REG", venue="neutral")]}
        after = "2026-11-30T21:25:00.000000+00:00"
        merged, conflicts = nfl_merge({"fixture": archived}, external, {"espn": ["fixture"]}, after)
        self.assertEqual(set(merged), {"nflverse"})
        self.assertEqual(merged["nflverse"]["market_start"], self.start)
        self.assertEqual(merged["nflverse"]["venue"], "neutral")
        self.assertEqual(conflicts, set())
        merged, _ = nfl_merge({"fixture": archived}, external, {}, after)
        self.assertNotIn("vendor", merged["nflverse"])

    def test_small_published_handoff_disagreement_uses_frozen_cutoff(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            fields = ["game_id", "game", "sport", "kickoff_iso", "market_display", "market_sides"]
            rows = [
                dict(game_id="same", game="West Virginia @ Utah", sport="ncaaf", market_display="SPREAD",
                     market_sides=json.dumps([{"flagged_side": "Utah", "current_line": "-8.5"}, {"flagged_side": "West Virginia"}])),
                dict(game_id="postponed", game="Michigan State @ Rutgers", sport="ncaaf", market_display="SPREAD",
                     market_sides="[]"),
            ]
            for filename, dates in [
                ("live_recent.csv", [self.start, self.start]),
                ("anomaly_board.csv", ["2026-11-29T21:30:00+00:00", "2026-12-04T21:25:00+00:00"]),
            ]:
                with (root / "data" / filename).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fields)
                    writer.writeheader()
                    writer.writerows(row | {"kickoff_iso": date} for row, date in zip(rows, dates))
            output = published_rows(root, "ncaaf")
            self.assertEqual(output["same"]["start"], self.start)
            self.assertEqual(output["same"]["markets"]["SPREAD"][0]["current_line"], "-8.5")
            self.assertEqual(len(output["same"]["published_starts"]), 2)
            self.assertTrue(output["postponed"]["conflict"])

    def test_crossover_calendar_keeps_basketball_in_previous_season(self):
        self.assertEqual(season("nba", "2027-02-01T02:00:00+00:00"), 2026)
        self.assertEqual(season("mlb", "2027-02-01T02:00:00+00:00"), 2027)
