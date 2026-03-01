import pandas as pd

rs = pd.read_csv("data/row_state.csv", dtype=str)

mask = rs["sport"].str.lower() == "ncaab"
mask &= rs["game_id"].notna()

# Try searching by team name inside game_id or side
mask &= (
    rs["side"].str.contains("miss", case=False, na=False) |
    rs["game_id"].str.contains("miss", case=False, na=False)
)

print(rs.loc[mask, [
    "sport",
    "game_id",
    "market",
    "side",
    "last_score",
    "last_bucket",
    "peak_score",
    "strong_streak"
]])
