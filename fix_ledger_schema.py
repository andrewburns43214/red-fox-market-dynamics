import csv
from pathlib import Path

p = Path("data/signal_ledger.csv")
rows = []

with open(p, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    for r in reader:
        rows.append(r)

max_cols = max(len(r) for r in rows)

fixed = []
for r in rows:
    if len(r) < max_cols:
        r = r + [""] * (max_cols - len(r))
    fixed.append(r)

with open(p, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(fixed)

print("ledger normalized to", max_cols, "columns")
