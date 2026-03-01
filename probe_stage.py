import pandas as pd
import main

df = pd.read_csv("data/snapshots.csv", dtype=str)

mlb = df[df["sport"].str.lower()=="mlb"]
print("MLB snapshot rows:", len(mlb))

# simulate report load stage
latest = main.build_latest_dataframe(df) if hasattr(main,"build_latest_dataframe") else None
print("latest exists:", latest is not None)
