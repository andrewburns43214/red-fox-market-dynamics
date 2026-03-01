import pandas as pd

snaps = pd.read_csv("data/snapshots.csv", dtype=str)
ledger = pd.read_csv("data/decision_freeze_ledger.csv", dtype=str)

snaps = snaps.drop_duplicates(subset=["sport","game_id","side"], keep="last")

print("Snaps columns:", snaps.columns.tolist())
print("Ledger columns:", ledger.columns.tolist())

print("Snaps sample key:")
print(snaps[["sport","game_id","side"]].head())

print("Ledger sample key:")
print(ledger[["sport","game_id","market_display","side"]].head())
