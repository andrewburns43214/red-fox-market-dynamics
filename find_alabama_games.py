import pandas as pd

rs = pd.read_csv("data/row_state.csv", dtype=str)

mask = rs["sport"].str.lower() == "ncaab"
mask &= rs["side"].str.contains("alabama", case=False, na=False)

print(rs.loc[mask, [
    "game_id",
    "market",
    "side",
    "last_score",
    "last_bucket",
    "peak_score"
]].sort_values("game_id"))
