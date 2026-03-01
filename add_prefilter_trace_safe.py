from pathlib import Path
import re

p = Path("main.py")
s = p.read_text(encoding="utf-8")

pattern = r'^(\s*)# --- HARD FILTER'
m = re.search(pattern, s, flags=re.MULTILINE)
if not m:
    raise SystemExit("FAILED — hard filter anchor not found")

indent = m.group(1)

injection = indent + 'print("[PRE-FILTER] rows:", len(latest), "sports:", latest["sport"].value_counts().to_dict() if "sport" in latest.columns else "NO_SPORT")\n'

s = re.sub(pattern, injection + indent + '# --- HARD FILTER', s, count=1, flags=re.MULTILINE)

p.write_text(s, encoding="utf-8")
print("[OK] PRE-FILTER trace inserted safely")
