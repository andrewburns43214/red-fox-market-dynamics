from pathlib import Path
import re

p = Path("main.py")
s = p.read_text(encoding="utf-8")

# Fix the bad indent: reduce 12 spaces -> 8 spaces for timing_health block
s = re.sub(
    r'\n {12}(# --- v1\.2 timing_health passthrough ---)',
    r'\n        \1',
    s
)

s = re.sub(
    r'\n {12}(# Preserve engine eligibility state through final rewrite stage)',
    r'\n        \1',
    s
)

s = re.sub(
    r'\n {12}(if "timing_health" not in _d\.columns:)',
    r'\n        \1',
    s
)

s = re.sub(
    r'\n {16}(_d\["timing_health"\] = "")',
    r'\n            \1',
    s
)

p.write_text(s, encoding="utf-8")
print("OK: indentation repaired")
