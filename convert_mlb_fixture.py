import pandas as pd

src = pd.read_csv("data/snapshots_mlb_test.csv")

games = {}

for _, r in src.iterrows():
    gid = r["game_id"]
    games.setdefault(gid, []).append(r)

rows = []

for gid, entries in games.items():

    if len(entries) >= 2:
        team1 = entries[0]["side"]
        team2 = entries[1]["side"]
        game_name = f"{team1} @ {team2}"
    else:
        game_name = f"{entries[0]['side']} @ Opponent"

    for r in entries:
        rows.append({
            "timestamp": r["timestamp"],
            "sport": "mlb",
            "game_id": gid,
            "game": game_name,
            "side": r["side"],          # <-- REQUIRED COLUMN
            "selection": r["side"],     # <-- dashboard logic
            "market": r["market"].lower(),
            "bet_pct": r["bets_pct"],
            "handle_pct": r["money_pct"],
            "open": "",
            "current": r["current_line"],
            "dk_start_iso": r["timestamp"]
        })

pd.DataFrame(rows).to_csv("data/snapshots.csv", index=False)
print("OK: converted to engine snapshot format (side fixed)")
