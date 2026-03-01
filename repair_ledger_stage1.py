from pathlib import Path

p = Path("data/signal_ledger.csv")
lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()

header = lines[0]
expected = len(header.split(","))

good = [header]
bad_count = 0

for l in lines[1:]:
    if len(l.split(",")) == expected:
        good.append(l)
    else:
        bad_count += 1

Path("data/signal_ledger.REPAIRED_STAGE1.csv").write_text("\n".join(good), encoding="utf-8")

print("GOOD ROWS:", len(good)-1)
print("REMOVED CORRUPT ROWS:", bad_count)
