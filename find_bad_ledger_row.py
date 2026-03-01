from pathlib import Path

p = Path("data/signal_ledger.csv")
lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()

header_cols = len(lines[0].split(","))

print("HEADER COLS:", header_cols)

for i,l in enumerate(lines[1:], start=2):
    cols = len(l.split(","))
    if cols != header_cols:
        print(f"BAD ROW at line {i}: expected {header_cols}, got {cols}")
        print(l[:300])
        break
