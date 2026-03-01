import pandas as pd
import main

print("REPORT TEST START")

try:
    main.build_dashboard()
    print("dashboard built OK")
except Exception as e:
    print("dashboard error:", e)

d = pd.read_csv("data/dashboard.csv", dtype=str, keep_default_na=False)
print("columns:", list(d.columns))
