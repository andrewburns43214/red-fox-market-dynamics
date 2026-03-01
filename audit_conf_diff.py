import pandas as pd
import numpy as np

d = pd.read_csv("data/dashboard.csv")

# Keep necessary columns
d_long = d[["sport","game_id","market_display","game_confidence"]].copy()

# Pivot to wide
wide = d_long.pivot_table(
    index=["sport","game_id"],
    columns="market_display",
    values="game_confidence",
    aggfunc="first"
)

wide = wide.reset_index()

# Melt back to long
reconstructed = wide.melt(
    id_vars=["sport","game_id"],
    var_name="market_display",
    value_name="reconstructed_conf"
)

# Merge with original
merged = pd.merge(
    d_long,
    reconstructed,
    on=["sport","game_id","market_display"],
    how="left"
)

# Compute difference
merged["diff"] = merged["game_confidence"] - merged["reconstructed_conf"]

# Flag anything not zero (tolerance for float rounding)
bad = merged[np.abs(merged["diff"]) > 0.0001]

print("Total rows checked:", len(merged))
print("Confidence mismatches:", len(bad))

if len(bad):
    print("\nMISMATCH SAMPLE:")
    print(bad.head(10).to_string(index=False))
else:
    print("\nNo confidence differences detected.")
