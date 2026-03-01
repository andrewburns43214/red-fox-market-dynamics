from pathlib import Path

p = Path("main.py")
s = p.read_text(encoding="utf-8")

BAD = "def update_row_state_and_signal_ledger(latest):"
if BAD not in s:
    raise SystemExit("metrics function not found")

REPLACEMENT = '''
def update_row_state_and_signal_ledger(latest):
    # HARD DISABLED — metrics temporarily off
    return
'''

s = s.replace(BAD, REPLACEMENT, 1)
p.write_text(s, encoding="utf-8")
print("[ok] metrics hard-disabled")
