import os
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from team_independent_sources import (
    Feed, FeedUnavailable, cached, career_recent, event_pairs, game_row, independent_context, load,
    save, schedule, schedule_page, summarize, updated_credentials, windows,
)


NOW = datetime(2026, 11, 27, 16, tzinfo=timezone.utc)
PUBLIC_NOW = NOW.isoformat(timespec="microseconds")


def nba_game(identity, day, status="Final"):
    return {"id": identity, "season": 2026, "date": day, "status": status, "visitor_team_score": 91,
            "home_team_score": 101, "visitor_team": {"id": 1, "full_name": "Chicago Bulls"},
            "home_team": {"id": 2, "full_name": "Boston Celtics"}}


class IndependentSourcesTest(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.state = Path(folder.name)
        self.keys = self.state / "keys"
        self.keys.mkdir()
        self.key = "private-example-credential"

    def feed(self, provider, *, network=True, now=NOW):
        (self.keys / provider).write_text(self.key, encoding="utf-8")
        return Feed(self.state, provider, now, lambda: True, network, self.keys)

    def test_missing_key_is_passive(self):
        fixtures = {"fixture": {"start": "2026-11-28T02:00:00Z", "markets": {"SPREAD": [{}]}}}
        with patch("team_independent_sources.build_opener") as transport:
            result = independent_context(self.state, fixtures, "ncaab", PUBLIC_NOW,
                                         lambda: True, True, self.keys)
        self.assertEqual(result["rad-sports"]["error"], "not configured")
        self.assertFalse(transport.called)
        self.assertEqual(list(self.state.iterdir()), [self.keys])

    def test_new_shared_credential_prioritizes_only_its_stale_sports(self):
        self.feed("rad-sports")
        os.utime(self.keys / "rad-sports", (NOW.timestamp(), NOW.timestamp()))
        self.assertEqual(updated_credentials(
            {"ncaaf": {"checked": (NOW+timedelta(seconds=1)).isoformat()},
             "ncaab": {"checked": (NOW-timedelta(seconds=1)).isoformat()}}, self.keys), {"ncaab"})

    def test_shared_daily_monthly_limits_persist_across_processes(self):
        feed = self.feed("rad-sports")
        save(feed.path, {"day": "2026-11-26", "daily": 12, "month": "2026-11", "monthly": 849})
        feed = self.feed("rad-sports")
        with patch("team_independent_sources.build_opener") as opener:
            response = opener.return_value.open.return_value.__enter__.return_value
            response.read.return_value = b'{"data":[]}'
            feed.request("https://api.collegefootballdata.com/games", {})
            request = opener.return_value.open.call_args.args[0]
            self.assertEqual(request.get_header("Authorization"), "Bearer " + self.key)
            other_worker = self.feed("rad-sports")
            with self.assertRaisesRegex(FeedUnavailable, "budget deferred"):
                other_worker.request("https://api.collegebasketballdata.com/games", {})
        self.assertEqual(load(feed.path)["monthly"], 850)
        self.assertNotIn(self.key, feed.path.read_text(encoding="utf-8"))

    def test_http_backoff_keeps_last_good_cache_and_resumes(self):
        feed = self.feed("puckapi")
        path = self.state / "retry.json"
        save(path, {"data": {"latest": 42}, "at": PUBLIC_NOW, "next": 0})
        with patch("team_independent_sources.build_opener") as opener:
            opener.return_value.open.side_effect = HTTPError(
                "https://mcp.puckapi.com/v1/get_standings", 429, "Rate limit", {"Retry-After": "7200"}, None)
            url = "https://mcp.puckapi.com/v1/get_standings"
            result = cached(feed, self.state, "retry", url, {}, lambda row: row, 8, cost=2)
            self.assertEqual(result["data"], {"latest": 42})
            self.assertEqual(result["error"], "HTTP 429")
            with self.assertRaisesRegex(FeedUnavailable, "provider backoff"):
                self.feed("puckapi", now=NOW+timedelta(hours=1)).request(url, {}, cost=2)
            opener.return_value.open.side_effect = None
            opener.return_value.open.return_value.__enter__.return_value.read.return_value = b'{"latest": 43}'
            result = cached(self.feed("puckapi", now=NOW+timedelta(hours=3)), self.state, "retry",
                            url, {}, lambda row: row, 8, cost=2)
        self.assertEqual(result["data"], {"latest": 43})
        self.assertNotIn("error", result)

    def test_incremental_pages_only_publish_after_complete(self):
        feed = self.feed("balldontlie")
        feed.remaining = 1

        def first(feed, *_args):
            feed.remaining = 0
            return [nba_game(11, "2026-11-20")], 11, 100

        with patch("team_independent_sources.schedule_page", side_effect=first):
            self.assertEqual(schedule(feed, self.state, "nba", 2026)[0], [])
        with patch("team_independent_sources.schedule_page",
                   return_value=([nba_game(12, "2026-11-28", "7:00 pm ET")], None, 100)) as page:
            rows, result = schedule(self.feed("balldontlie"), self.state, "nba", 2026)
        self.assertEqual(page.call_args.args[-1], 11)
        self.assertEqual({row["id"] for row in rows}, {"11", "12"})
        self.assertEqual(result["partitions"], 1)

    def test_malformed_refresh_preserves_previous_partition(self):
        row = game_row("nba", 2026, nba_game(11, "2026-11-20"))
        save(self.state / "games-nba-2026.json",
             {"2026-11-01": {"rows": {"11": row}, "as_of": PUBLIC_NOW, "next": 0},
              **{str(start): {"next": NOW.timestamp()+86400} for start,_ in windows("nba", 2026)
                 if str(start) != "2026-11-01"}})
        with patch("team_independent_sources.schedule_page", return_value=([], None, 100)):
            rows, status = schedule(self.feed("balldontlie"), self.state, "nba", 2026)
        self.assertEqual(rows[0]["id"], "11")
        self.assertEqual(status["error"], "schema or coverage validation")

    def test_provider_contract_and_season_rollover(self):
        feed = MagicMock()
        feed.request.return_value = {"data": [], "meta": {}}
        schedule_page(feed, "mlb", 2027, date(2027, 4, 1), date(2027, 4, 30), None)
        self.assertEqual(len(feed.request.call_args.args[1]["dates[]"]), 30)
        self.assertEqual(feed.request.call_args.args[1]["seasons[]"], 2027)
        self.assertIn((date(2028, 2, 16), date(2028, 2, 29)), windows("ncaab", 2027))
        row = {"id": "2026020424", "season": "20262027", "gameType": "regular", "gameDate": "2026-11-27",
               "homeTeam": "NJD", "awayTeam": "VAN", "homeScore": 5, "awayScore": 5, "gameState": "LIVE"}
        self.assertIsNone(game_row("nhl", 2026, row)["scores"])
        self.assertEqual(game_row("nhl", 2026, dict(row, gameState="OFF", homeScore=6))["scores"], [5,6])
        feed.request.return_value = {"data":{"games":[row],"count":1}}
        self.assertEqual(schedule_page(feed,"nhl",2026,date(2026,11,1),date(2026,11,30),None)[0], [row])
        start = "2026-11-29T01:00:00Z"
        nba = dict(nba_game(5, "2026-11-28", start), datetime="2026-11-29T01:00:00.000Z", period=0)
        self.assertTrue(game_row("nba",2026,nba)["scheduled"])
        self.assertFalse(game_row("nba",2026,dict(nba,period=1))["scheduled"])

    def test_independent_overlay_verifies_fixture_and_excludes_later_results(self):
        rows = [dict(game_row("nba", 2026, sample), as_of=PUBLIC_NOW) for sample in (
            nba_game(1, "2026-11-01"), nba_game(2, "2026-11-21"), nba_game(3, "2026-11-28", "7:00 pm ET"),
            nba_game(4, "2026-11-29"),
        )]
        fixture = {"keys": rows[0]["keys"], "start": "2026-11-29T00:00:00+00:00"}
        status = {"at": PUBLIC_NOW, "partitions": 1, "expected_partitions": 9}
        self.assertEqual(summarize(rows, [fixture], PUBLIC_NOW, status, "balldontlie"), 1)
        independent = fixture["sporting"]["independent"]
        self.assertEqual(independent["teams"]["boston celtics"]["record"], [2,0,0])
        self.assertEqual(independent["teams"]["boston celtics"]["opponent_win_games"], 1)
        self.assertEqual(independent["teams"]["boston celtics"]["recent"][0], {
            "opponent": "Chicago Bulls", "opponent_key": "chicago bulls", "at": "2026-11-21",
            "scores": [101, 91], "venue": None,
        })
        fixture.pop("sporting")
        self.assertEqual(summarize(rows+[dict(rows[2], id="9")], [fixture], PUBLIC_NOW, status, "balldontlie"), 0)
        self.assertNotIn("sporting", fixture)

    def test_ufc_card_requires_live_pairing_and_only_professional_history(self):
        sides = {"fighter_a":{"id":1, "name":"Jane Doe"}, "fighter_b":{"id":2, "name":"Jill Smith"}}
        card = event_pairs({"data":{"id":7, "org":"ufc", "starts_at":"2026-12-05T21:00:00Z",
                                   "status":"announced", "card":[dict(sport="mma", status="cancelled", **sides),
                                                                dict(sport="mma", status="confirmed", **sides)]}}, "7")
        self.assertEqual(card["bouts"], [{"janedoe":"1", "jillsmith":"2"}])
        entry = {"date":"2026-01-02", "opponent_name":"Opposing Fighter", "status":"win",
                 "competition_level":"pro", "sport":"mma", "method":"Decision"}
        recent = career_recent({"data":[entry, entry, dict(entry, competition_level="amateur", date="2026-11-24")]})
        self.assertEqual(recent, [{"date":"2026-01-02", "opponent":"Opposing Fighter","result":"win","method":"Decision"}])
