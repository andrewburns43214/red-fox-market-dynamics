"""Show scheduled games when an active sport has no verified board markets."""

import csv
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


DATA = Path("data")
SCOREBOARDS = {
    "nfl": ("football/nfl", 8, {}),
    "ncaaf": ("football/college-football", 8, {"groups": 80, "limit": 500}),
    "mlb": ("baseball/mlb", 3, {}),
    "nhl": ("hockey/nhl", 3, {"limit": 500}),
    "nba": ("basketball/nba", 3, {}),
    "ncaab": ("basketball/mens-college-basketball", 3, {"groups": 50, "limit": 500}),
}


def fetch_scoreboard(url):
    request = Request(url, headers={"User-Agent": "RedFoxMarketData/1.0", "Accept": "application/json"})
    with urlopen(request, timeout=8) as response:
        return json.load(response)


def uncovered_schedule(board, coverage, now=None, fetch=fetch_scoreboard):
    now = now or datetime.now(timezone.utc)
    present = {row.get("sport", "").lower() for row in board}
    active = set(coverage.get("expected_active_sports") or ())
    today = now.astimezone(ZoneInfo("America/New_York")).date()
    games = {}
    errors = {}
    for sport in sorted(active - present):
        if sport not in SCOREBOARDS:
            continue
        path, days, params = SCOREBOARDS[sport]
        for offset in range(days + 1):
            query = urlencode({"dates": (today + timedelta(days=offset)).strftime("%Y%m%d"), **params})
            url = f"https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard?{query}"
            try:
                events = fetch(url).get("events", [])
            except (OSError, ValueError, TypeError) as error:
                errors[sport] = type(error).__name__
                continue
            for event in events:
                try:
                    start = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
                    if not now - timedelta(minutes=5) <= start <= now + timedelta(days=days):
                        continue
                    competitors = event["competitions"][0]["competitors"]
                    sides = {side["homeAway"]: side["team"]["displayName"] for side in competitors}
                    if not sides.get("away") or not sides.get("home"):
                        continue
                    item = {
                        "sport": sport,
                        "game_id": str(event["id"]),
                        "game": f'{sides["away"]} @ {sides["home"]}',
                        "kickoff_iso": start.isoformat(),
                        "status": "AWAITING_VERIFIED_MARKETS",
                    }
                    games[(sport, item["game_id"])] = item
                except (KeyError, IndexError, TypeError, ValueError):
                    continue
    return sorted(games.values(), key=lambda item: (item["kickoff_iso"], item["sport"], item["game"])), errors


def main():
    board_path = DATA / "anomaly_board.csv"
    coverage_path = DATA / "publication_coverage.json"
    if not board_path.exists() or not coverage_path.exists():
        return
    with board_path.open(newline="", encoding="utf-8") as handle:
        board = list(csv.DictReader(handle))
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    games, errors = uncovered_schedule(board, coverage)
    coverage["scheduled_without_markets"] = games
    coverage["schedule_checked_at"] = datetime.now(timezone.utc).isoformat()
    coverage["schedule_fetch_errors"] = errors
    temporary = coverage_path.with_name(".publication_coverage.schedule.tmp")
    temporary.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    os.replace(temporary, coverage_path)
    print(f"[schedule] {len(games)} scheduled games awaiting verified markets; fetch errors={errors}")


if __name__ == "__main__":
    main()
