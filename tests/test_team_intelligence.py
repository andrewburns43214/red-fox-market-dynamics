import csv
import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from history_blocks import encode
from team_sport_context import (
    cfb_elo, cfb_profile_context, fighter_identities, nfl_availability, nfl_efficiency,
    nfl_profile_context, wikidata_fighters,
)
from team_intelligence import clock, graded_quotes, identities, nfl_game, nfl_merge, published_rows, season, valid_outcomes


class SportingDataTest(unittest.TestCase):
    def test_cfb_ratings_require_unambiguous_fbs_identity(self):
        def rating(name, slug, identity, rank, classification="fbs"):
            return {"team": {"school": name, "slug": slug, "id": identity},
                    "fbs_rank": rank, "rating": 1500, "classification": classification}

        teams = [rating("Miami", "miami", 2, 1), rating("Miami (OH)", "miami-oh", 3, 2),
                 rating("UL Monroe", "ul-monroe", 4, 3), rating("FCS team", "fcs-team", 5, None, "fcs")]
        payload = {
            "data": {"season": 2026, "week": None, "teams": teams},
            "meta": {"as_of": "2026-11-27T11:05:50Z", "truncated": False},
            "attribution": [{"license_url": "https://creativecommons.org/licenses/by/4.0/"}],
        }
        data = cfb_elo(json.dumps(payload).encode(), 2026)
        self.assertEqual({key: row["rank"] for key, row in data["teams"].items()},
                         {"miami": 1, "miami oh": 2, "ulm": 3})
        teams.append(rating("Miami", "another-miami", 6, 1))
        with self.assertRaises(ValueError):
            cfb_elo(json.dumps(payload).encode(), 2026)

    def test_cfb_ratings_are_only_exposed_for_two_verified_upcoming_participants(self):
        now = "2026-11-28T02:00:00.000000+00:00"
        at = "2026-11-27T11:05:50.000000+00:00"
        future = "2026-11-28T19:00:00.000000+00:00"
        data = {"as_of": at, "teams": {"army": {"rank": 1, "rating": 1526.9},
                                        "rice": {"rank": 2, "rating": 1334.2}}}
        with TemporaryDirectory() as directory:
            state = Path(directory)
            cache = state / "cfb-digest-elo-2026.json"
            cache.write_text(json.dumps({"data": data}), encoding="utf-8")
            matchup = {"season": 2026, "keys": ["army", "rice"], "start": future, "markets": {"SPREAD": [{}]}}
            fixtures = {
                "upcoming": dict(matchup),
                "completed": dict(matchup, start="2026-11-27T01:00:00.000000+00:00"),
                "unknown": dict(matchup, keys=["army", "other"]),
                "unpublished": dict(matchup, markets={}),
            }
            status = cfb_profile_context(state, fixtures, now, lambda: True, False)
            self.assertEqual(status["cfb_elo"]["coverage"], 1)
            self.assertEqual(fixtures["upcoming"]["sporting"]["cfb_elo"]["at"], at)
            self.assertFalse(any("sporting" in fixtures[key] for key in ("completed", "unknown", "unpublished")))
            for ineligible_stamp in ("2026-11-24T00:00:00.000000+00:00", "2026-11-28T03:00:00.000000+00:00"):
                with self.subTest(ineligible_stamp=ineligible_stamp):
                    cache.write_text(json.dumps({"data": dict(data, as_of=ineligible_stamp)}), encoding="utf-8")
                    fixture = dict(matchup)
                    status = cfb_profile_context(state, {"game": fixture}, now, lambda: True, False)
                    self.assertEqual(status["cfb_elo"]["coverage"], 0)
                    self.assertNotIn("sporting", fixture)

    def test_efficiency_requires_unique_opponents_and_uses_attempts_sacks_and_carries(self):
        raw = (
            "game_id,team,opponent_team,season_type,attempts,sacks_suffered,carries,passing_epa,rushing_epa\n"
            "game,A,B,REG,20,2,22,2.25,-1.5\n"
            "game,B,A,REG,21,4,20,3.0,0.0\n"
            "duplicated,A,B,REG,20,2,22,1.0,1.0\n"
            "duplicated,A,B,REG,20,2,22,1.0,1.0\n"
            "duplicated,B,A,REG,20,2,22,1.0,1.0\n"
        ).encode("utf-8")
        data = nfl_efficiency(raw)
        self.assertEqual(data["game"]["A"], {"opponent":"B","plays":44,"epa":.75,
                         "dropbacks":22,"pass_epa":2.25,"rushes":22,"rush_epa":-1.5,"sacks":2})
        self.assertNotIn("duplicated",data)

    def test_practice_absence_is_preserved_separately_from_game_status(self):
        raw = (
            "season,season_type,week,team,gsis_id,full_name,position,report_primary_injury,report_status,practice_primary_injury,practice_status\n"
            "2026,REG,12,WAS,a,A Player,WR,,,Ankle,Did Not Participate In Practice\n"
            "2026,REG,12,WAS,b,B Player,CB,Calf,Out,Calf,Did Not Participate In Practice\n"
            "2026,REG,12,WAS,c,C Player,FS,,,Elbow,Full Participation in Practice\n"
        ).encode("utf-8")
        results = nfl_availability(raw)["2026:12:WAS"]
        self.assertEqual([(r["name"],r["status"]) for r in results],
                         [("B Player","Out"),("A Player","Did not practice")])

    def test_nfl_supplements_require_exact_score_fixture_mapping(self):
        with TemporaryDirectory() as directory:
            state = Path(directory)
            (state/"nfl-team-stats-2026.json").write_text(json.dumps({"data":{"game":{
                "WAS":{"opponent":"ARI","epa":2,"plays":40}, "ARI":{"opponent":"WAS","epa":-3,"plays":45},
            }}}))
            at = "2026-11-27T18:00:00.000000+00:00"
            (state/"nfl-availability-2026.json").write_text(json.dumps({"source_at":at,
                "data":{"2026:12:WAS":[{"name":"A Player","status":"Out"}]}}))
            keys = ["washington commanders","arizona cardinals"]
            start = "2026-11-29T21:25:00.000000+00:00"
            schedule = {"games":[dict(keys=keys,stage="REG",start=start,codes=["WAS","ARI"],week=12,season=2026)]}
            fixtures = {"correct":{"keys":keys,"start":start},"postponed":{"keys":keys,"start":"2026-12-02T21:25:00.000000+00:00"}}
            games = {"game":{"uid":"game","source":"nflverse","codes":["WAS","ARI"],"keys":keys},
                     "similar":{"uid":"similar","source":"nflverse","codes":["WAS","ARI"],"keys":keys}}
            nfl_profile_context(state, fixtures, games, schedule, "2026-11-28T02:00:00.000000+00:00", lambda:True, False)
            self.assertEqual(games["game"]["sporting"]["efficiency"][keys[0]]["epa"],2)
            self.assertNotIn("sporting",games["similar"])
            self.assertEqual(fixtures["correct"]["sporting"]["availability"][keys[0]][0]["status"],"Out")
            self.assertNotIn("sporting",fixtures["postponed"])

    def test_fighter_ids_and_measurements_require_unambiguous_evidence(self):
        receipts = [
            {"source":"verified_research_ufc","names":["Fighter One","Fighter Two"],"athlete_ids":["12345","23456"]},
            {"source":"verified_research_ufc","names":["Different Fighter","Fighter Three"],"athlete_ids":["12345","34567"]},
        ]
        self.assertNotIn("12345",fighter_identities(receipts).values())
        def binding(id,item,amount,unit):
            return {name:{"value":value} for name,value in zip(("espn","item","height","unit"),(id,item,amount,unit))}
        raw = json.dumps({"results":{"bindings":[binding("23456","Q1","1.80","Q11573"),binding("34567","Q2","1.75","Q11573"),
                                                   binding("34567","Q2","1.78","Q11573")]}}).encode()
        self.assertEqual(wikidata_fighters(raw),{"23456":{"item":"Q1","height_cm":180.0}})


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
        names, keys = identities("JosÃƒÂ© Aldo vs Rob Font", "ufc")
        raw = {"matchup": "JosÃƒÂ© Aldo vs Marlon Vera", "winner": "josealdo", "actual_start": self.start}
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

    def test_nfl_closing_spread_uses_feed_sign_and_only_graded_fixtures(self):
        year = clock().year
        row = {"season": str(year), "game_id": f"{year}_12_KC_BUF", "espn": "401123456",
               "gameday": f"{year}-11-29", "gametime": "16:25", "week": "12", "game_type": "REG",
               "away_team": "KC", "home_team": "BUF", "away_score": "21",
               "home_score": "24", "spread_line": "3.5"}
        ats = nfl_game(row)["ats"]
        self.assertEqual([(r["grade"], r["line"]) for r in ats], [("win", 3.5), ("loss", -3.5)])
        self.assertIsNone(nfl_game(dict(row, home_score=""))["ats"])
        self.assertIsNone(nfl_game(dict(row, espn=""))["ats"])

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

    def test_official_school_box_score_requires_exact_fixture_timing(self):
        names, keys = identities("New Mexico State @ Hawaii", "ncaaf")
        catalog = {"fixture": {"start": self.start, "keys": keys, "names": names, "season": 2026}}
        raw = {"matchup": " @ ".join(names), "score_away": 19, "score_home": 29, "event_id": "fixture",
               "fixture_scheduled_at": self.start, "research_provenance": "school_athletics_official_final",
               "url": "https://school.test/sports/football/stats/2026/new-mexico-state/boxscore/31550"}
        self.db.execute("INSERT INTO outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        ("ncaaf", "fixture", "verified_research_ncaaf", "verified", None, self.start, "hash", json.dumps(raw)))
        now = "2026-11-30T21:25:00.000000+00:00"
        results, _, _ = valid_outcomes(self.db, "ncaaf", catalog, now)
        self.assertEqual(results["fixture"]["scores"], [19, 29])
        catalog["fixture"]["start"] = self.checkpoint
        results, _, _ = valid_outcomes(self.db, "ncaaf", catalog, now)
        self.assertEqual(results, {})

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
