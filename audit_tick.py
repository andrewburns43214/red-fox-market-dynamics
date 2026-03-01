import pandas as pd

df = pd.read_csv("data/row_state.csv", dtype=str)
print("Latest tick:", df["last_seen_tick"].max())

recent = df[df["last_seen_tick"] == df["last_seen_tick"].max()]
print("Recent rows:", len(recent))
