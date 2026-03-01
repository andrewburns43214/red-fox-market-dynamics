import pandas as pd

df = pd.read_csv("data/snapshots.csv")

extra = df.copy()
extra["market"] = "total"
extra["side"] = "Over 8.5"
extra["open_line"] = "O 8.5"
extra["current_line"] = "O 8.5"

df = pd.concat([df, extra], ignore_index=True)

df.to_csv("data/snapshots.csv", index=False)
print("Added TOTAL market rows")
