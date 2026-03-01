import pandas as pd
import main

# build_dashboard() writes files; we only want to inspect the in-memory shapes.
# So we call the function but then read snapshots/dashboard artifacts it writes.
# We’ll inspect dashboard.csv AND snapshots.csv to infer if side grain exists upstream.

d = pd.read_csv("data/dashboard.csv")
print("dashboard.csv rows:", len(d))
print("dashboard.csv cols:", list(d.columns))

# check for side-level fields that SHOULD exist if exported
want = ["side","model_score","confidence_score","timing_bucket","row_status"]
print("present side-level fields:", {c:(c in d.columns) for c in want})

