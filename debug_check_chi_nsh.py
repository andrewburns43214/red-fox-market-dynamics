import requests

url = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates=20260226&limit=500"
data = requests.get(url, timeout=10).json()

names = [e["shortName"] for e in data.get("events", [])]

print("All events:")
for n in names:
    print(n)

print("\nCHI @ NSH present?:", any("CHI" in n and "NSH" in n for n in names))
