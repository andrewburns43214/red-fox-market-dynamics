import pandas as pd

GAME = "Utah State @ San Diego State"

ledger = pd.read_csv("data/signal_ledger.csv", dtype=str)

gid = ledger[ledger["game"] == GAME]["game_id"].unique()
if len(gid)==0:
    print("No ledger entries found.")
    raise SystemExit()

gid = gid[0]

l = ledger[(ledger["game_id"] == gid) & (ledger["market"]=="MONEYLINE")].copy()

print("\n=== Ledger Events For SDSU ML ===\n")
print(l[[
    "ts",
    "event",
    "from_bucket",
    "to_bucket",
    "score",
    "net_edge",
    "timing_bucket"
]].to_string(index=False))
