import pandas as pd

lg = pd.read_csv("data/signal_ledger.csv", keep_default_na=False)

print("\n[OUTCOME FIELD CHECK]")
print("Columns:", list(lg.columns))

needed = ["result", "final_score", "win_flag"]
for c in needed:
    print(c, "present:", c in lg.columns)

print("\nSample rows with nonblank result:")
if "result" in lg.columns:
    print(lg[lg["result"].astype(str).str.strip()!=""].tail(10).to_string(index=False))
