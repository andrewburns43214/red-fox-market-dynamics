from pathlib import Path
import re

p = Path("main.py")
s = p.read_text(encoding="utf-8")

anchor = "update_row_state_and_signal_ledger(_metrics_df)"

if anchor not in s:
    raise SystemExit("anchor not found")

inject = '''
print("[METRICS PROBE] rows=", len(_metrics_df))
print("[METRICS PROBE] cols=", list(_metrics_df.columns)[:20])
print("[METRICS PROBE] sample=", _metrics_df.head(2).to_dict("records"))
update_row_state_and_signal_ledger(_metrics_df)
'''

s = s.replace(anchor, inject, 1)

p.write_text(s, encoding="utf-8")
print("[ok] probe injected")
