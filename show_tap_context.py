from pathlib import Path
s = Path("main.py").read_text(encoding="utf-8").splitlines()
# find the metrics tap line we inserted
for i, line in enumerate(s):
    if "update_row_state_and_signal_ledger(_metrics_side)" in line:
        start = max(0, i-12)
        end = min(len(s), i+6)
        print("\n".join(s[start:end]))
        break
else:
    print("tap call not found")
