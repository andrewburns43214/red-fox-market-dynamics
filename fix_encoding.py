from pathlib import Path

p = Path("make_mlb_snapshot.py")

# Read as UTF-16 (Windows wrote it this way)
text = p.read_text(encoding="utf-16")

# Rewrite as UTF-8 (what Python requires)
p.write_text(text, encoding="utf-8")

print("FIXED: make_mlb_snapshot.py converted to UTF-8")
