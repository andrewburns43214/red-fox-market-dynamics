import pandas as pd

dash = pd.read_csv("data/dashboard.csv")
res  = pd.read_csv("data/results_resolved.csv")

res["result"] = res["result"].astype(str).str.strip().str.upper()
resolved = res[res["result"].isin(["WIN","LOSS","PUSH"])].copy()

print("\n=== SAMPLE RESOLVED TOTAL ROW ===")
print(resolved.head(1).to_string(index=False))

gid = resolved.iloc[0]["game_id"]

print("\n=== DASHBOARD ROWS FOR SAME GAME_ID ===")
print(dash[dash["game_id"]==gid].to_string(index=False))
