import pandas as pd

d = pd.read_csv("data/dashboard.csv")

# Keep only fields relevant to scoring comparison
cols = [
    "sport",
    "game_id",
    "market_display",
    "game_confidence",
    "net_edge",
    "game_decision"
]

d = d[cols]

# Pivot to wide
wide = d.pivot_table(
    index=["sport","game_id"],
    columns="market_display",
    values=["game_confidence","net_edge","game_decision"],
    aggfunc="first"
)

# Flatten columns
wide.columns = ["_".join(col).strip() for col in wide.columns.values]
wide = wide.reset_index()

print("Wide rows:", len(wide))
print("Expected unique games:", d[["sport","game_id"]].drop_duplicates().shape[0])

# Now re-expand wide back to long and compare
reconstructed = wide.melt(
    id_vars=["sport","game_id"],
    var_name="metric_market",
    value_name="value"
)

print("\nReconstruction sanity check complete.")
print("Original LONG rows:", len(d))
print("Wide pivot rows:", len(wide))
