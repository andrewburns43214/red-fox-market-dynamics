import csv, os

def head(path):
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        return None
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        try:
            h = next(r)
        except StopIteration:
            print(f"EMPTY: {path}")
            return []
    return h

for p in ["data/row_state.csv", "data/signal_ledger.csv", "data/dashboard.csv", "data/snapshots.csv"]:
    h = head(p)
    if h is None: 
        continue
    print("\n==", p, "==")
    print("cols:", len(h))
    print("first10:", h[:10])
    print("last10:", h[-10:])
