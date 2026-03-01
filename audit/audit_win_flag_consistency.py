import pandas as pd

lg = pd.read_csv("data/signal_ledger.csv", keep_default_na=False)

print("\n[WIN FLAG CONSISTENCY]")

if "win_flag" in lg.columns and "result" in lg.columns:
    bad = lg[(lg["result"]=="WIN") & (lg["win_flag"]!="1")]
    print("WIN mismatches:", len(bad))
    bad2 = lg[(lg["result"]=="LOSS") & (lg["win_flag"]!="0")]
    print("LOSS mismatches:", len(bad2))
else:
    print("Required columns not present")
