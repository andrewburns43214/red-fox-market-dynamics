import pandas as pd

d = pd.read_csv("data/dashboard.csv", keep_default_na=False)

d["normalized_side"] = (
    d["favored_side"]
    .str.replace(r"\s[+-]?\d+\.?\d*", "", regex=True)
    .str.replace(r"(Over|Under)\s*\d+\.?\d*", r"\1", regex=True)
    .str.strip()
)

print(d[["market_display","favored_side","normalized_side"]].head(15))
print("\nUNIQUE NORMALIZED SIDES:", d["normalized_side"].nunique())
print("TOTAL ROWS:", len(d))
