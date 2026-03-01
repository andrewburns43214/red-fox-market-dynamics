import pandas as pd

GAME = "Utah State @ San Diego State"

# Load row_state (side-level memory)
state = pd.read_csv("data/row_state.csv", dtype=str)

# Filter this game
s = state[state["game_id"].notna()].copy()

# We need to identify by game text (row_state stores sport/game_id/market/side)
# So let's load snapshots to map game_id first

snap = pd.read_csv("data/snapshots.csv", dtype=str)
gid = snap[snap["game"] == GAME]["game_id"].unique()

if len(gid) == 0:
    print("Game not found in snapshots.")
    raise SystemExit()

gid = gid[0]

print("\nGame ID:", gid)

rows = state[state["game_id"] == gid].copy()

# Numeric conversion
for col in ["last_score","peak_score","net_edge"]:
    if col in rows.columns:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")

cols = [
    "market",
    "side",
    "last_score",
    "peak_score",
    "timing_bucket",
    "strong72_now",
    "strong_streak",
    "last_decision",
    "net_edge"
]

print("\n=== Side-Level Rows ===\n")
print(rows[cols].sort_values(["market","side"]).to_string(index=False))
