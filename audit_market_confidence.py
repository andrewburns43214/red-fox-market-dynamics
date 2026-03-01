import pandas as pd

d = pd.read_csv("data/dashboard.csv")

print("\n================ MARKET CONFIDENCE AUDIT ================\n")

# --- 1. confidence must equal model_score
bad_match = d[(d["game_confidence"] - d["model_score"]).abs() > 0.0001]

print("Mismatch rows:", len(bad_match))
if len(bad_match):
    print(bad_match[[
        "game","sport_label","market_display",
        "model_score","game_confidence"
    ]].to_string(index=False))

# --- 2. confidence must stay within side bounds
bad_bounds = d[
    (d["game_confidence"] < d["min_side_score"]) |
    (d["game_confidence"] > d["max_side_score"])
]

print("\nOut-of-bounds rows:", len(bad_bounds))
if len(bad_bounds):
    print(bad_bounds[[
        "game","sport_label","market_display",
        "game_confidence","min_side_score","max_side_score"
    ]].to_string(index=False))

# --- 3. sanity: one row per game+market
dup = d.duplicated(subset=["sport","game_id","market_display"]).sum()
print("\nDuplicate game+market rows:", dup)

print("\n=========================================================\n")
