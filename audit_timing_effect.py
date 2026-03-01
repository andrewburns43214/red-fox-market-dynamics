import pandas as pd

d = pd.read_csv("data/dashboard.csv")

# if timing_bucket exported
if "timing_bucket" in d.columns:
    print("\nConfidence by timing bucket:")
    print(d.groupby("timing_bucket")["game_confidence"].describe())
else:
    print("timing_bucket not exported to dashboard.csv")
