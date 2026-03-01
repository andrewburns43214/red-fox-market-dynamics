import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 3000)

d = pd.read_csv("data/dashboard.csv")

# Filter to Omaha ML + one 72 total
subset = d[
    (
        (d["game_id"] == 33702858) & (d["market_display"] == "MONEYLINE")
    )
    |
    (
        (d["game"] == "North Texas @ Charlotte") & (d["market_display"] == "TOTAL")
    )
]

print("\n=== FULL DASHBOARD ROWS ===\n")
print(subset.to_string(index=False))
