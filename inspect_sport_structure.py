import requests

SPORT_URLS = {
    "nfl":   "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
    "nba":   "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "ncaaf": "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard",
    "nhl":   "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
    "mlb":   "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
    "ncaab": "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard",
}

DATE = "20260226"

for sport, base in SPORT_URLS.items():
    try:
        url = f"{base}?dates={DATE}&limit=50"
        data = requests.get(url, timeout=10).json()
        events = data.get("events", [])
        print(f"\n=== {sport.upper()} ===")
        print("Event count:", len(events))

        if not events:
            continue

        e = events[0]
        print(" shortName:", e.get("shortName"))

        comp = e["competitions"][0]["competitors"]
        for c in comp:
            print("  team.displayName:", c["team"]["displayName"])
            print("  team.abbreviation:", c["team"]["abbreviation"])
            print("  homeAway:", c["homeAway"])

    except Exception as e:
        print(f"{sport} ERROR:", e)
