from pathlib import Path
import re

p = Path("main.py")
s = p.read_text(encoding="utf-8")

marker = "def _timing_bucket_from_minutes(mins):"

if marker not in s:
    raise SystemExit("timing function not found  abort")

# Insert a safe return right before timing function
parts = s.split(marker, 1)

before = parts[0].rstrip()
after = marker + parts[1]

# if function already returns, do nothing
if "return dash_html" in before[-400:]:
    print("build_dashboard already closed")
    raise SystemExit(0)

patched = before + "\n\n    return dash_html\n\n" + after

p.write_text(patched, encoding="utf-8")
print("[OK] build_dashboard boundary restored")
