import pandas as pd

res = pd.read_csv("data/results_resolved.csv")

res["result_u"] = res["result"].astype(str).str.strip().str.upper()

# pick the game_id from the sample you showed
gid = 33701386

print("\n=== ALL ROWS FOR GAME_ID ===")
print(res[res["game_id"]==gid]
      .sort_values(["market_display","side"])
      .to_string(index=False))

print("\n=== RESOLVED ROWS FOR THIS GAME_ID ===")
print(res[(res["game_id"]==gid) & 
          (res["result_u"].isin(["WIN","LOSS","PUSH"]))]
      .to_string(index=False))
