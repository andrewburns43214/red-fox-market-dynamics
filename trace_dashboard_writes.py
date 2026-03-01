import sys, runpy
import pandas as pd

ORIG = pd.DataFrame.to_csv

def audit(df, tag):
    try:
        gc = pd.to_numeric(df.get("game_confidence"), errors="coerce")
        mx = pd.to_numeric(df.get("max_side_score"), errors="coerce")
        bad = (gc > mx) & gc.notna() & mx.notna()
    except Exception:
        bad = None

    print(f"\n===== WRITE {tag} =====")
    print("rows:", len(df))
    print("has market_display:", "market_display" in df.columns)
    if bad is not None:
        print("bad rows:", int(bad.sum()))
        if bad.sum():
            print(df.loc[bad, ["game","market_display","game_confidence","max_side_score","model_score"]])
    print("=======================\n")

def patched(self, path_or_buf=None, *a, **k):
    p = str(path_or_buf) if path_or_buf else ""
    if p.replace("\\","/").endswith("data/dashboard.csv"):
        patched.n += 1
        audit(self, patched.n)
    return ORIG(self, path_or_buf, *a, **k)

patched.n = 0
pd.DataFrame.to_csv = patched

sys.argv = ["main.py","report"]
runpy.run_path("main.py", run_name="__main__")

print("writes seen:", patched.n)
