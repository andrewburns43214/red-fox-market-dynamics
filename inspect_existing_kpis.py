import pandas as pd

for f in [
    "data/kpi_summary_overall.csv",
    "data/kpi_master_dataset.csv"
]:
    try:
        d = pd.read_csv(f)
        print(f"\nFILE: {f}")
        print("COLUMNS:")
        for c in d.columns:
            print("  ", c)
        print("ROW COUNT:", len(d))
    except Exception as e:
        print(f"\nFILE: {f} — ERROR:", e)
