import requests, json

url = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates=20260226&limit=500"
r = requests.get(url, timeout=10)
data = r.json()

print("Event count:", len(data.get("events", [])))
print("\nSample events:")
for e in data.get("events", [])[:5]:
    print(e["shortName"])
