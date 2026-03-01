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
        for e in events[:5]:
            print(" shortName:", e.get("shortName"))
    except Exception as e:
        print(f"{sport} ERROR:", e)
