import pandas as pd

lg = pd.read_csv("data/signal_ledger.csv", keep_default_na=False)

print("\n[SCORE IMMUTABILITY CHECK]")

if "result" in lg.columns:
    graded = lg[lg["result"].astype(str).str.strip()!=""]
else:
    graded = lg.copy()

print("Total graded rows:", len(graded))

if "score" in graded.columns:
    print("Score sample:")
    print(graded["score"].tail(10).to_list())
else:
    print("No score column found")
