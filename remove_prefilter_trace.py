from pathlib import Path

p = Path("main.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
    '    print("[PRE-FILTER] rows:", len(latest), "sports:", latest["sport"].value_counts().to_dict() if "sport" in latest.columns else "NO_SPORT")\n',
    ''
)

p.write_text(s, encoding="utf-8")
print("[OK] PRE-FILTER trace removed")
