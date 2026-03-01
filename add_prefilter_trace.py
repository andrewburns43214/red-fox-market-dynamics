from pathlib import Path

p = Path("main.py")
s = p.read_text(encoding="utf-8")

# Find the hard filter anchor
ANCH = "# --- HARD FILTER"
if ANCH not in s:
    raise SystemExit("FAILED — hard filter anchor not found")

# Find the first occurrence and insert PRE-FILTER print before it
PRE = '    print("[PRE-FILTER] rows:", len(latest), "sports:", latest["sport"].value_counts().to_dict() if "sport" in latest.columns else "NO_SPORT")\n'

s = s.replace(ANCH, PRE + ANCH, 1)

p.write_text(s, encoding="utf-8")
print("[OK] PRE-FILTER trace inserted")
