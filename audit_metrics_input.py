from pathlib import Path

p = Path("main.py")
s = p.read_text(encoding="utf-8")

anchor = "update_row_state_and_signal_ledger("

if anchor not in s:
    raise SystemExit("Call site not found")

s = s.replace(
    anchor,
    "print('\\n[METRICS INPUT COLUMNS]:', list(_metrics_df.columns) if '_metrics_df' in locals() else 'UNKNOWN')\n    " + anchor,
    1
)

p.write_text(s, encoding="utf-8")
print("Inserted audit print")
