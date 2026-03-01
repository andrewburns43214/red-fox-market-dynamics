from pathlib import Path
import re

p = Path("main.py")
s = p.read_text(encoding="utf-8")

# 1) Turn the stray text into a real comment
s = s.replace("\n (always runs)", "\n        # (always runs)")

# 2) Ensure try aligns to same level as timing_bucket block
s = re.sub(r'\n {8}try:', r'\n        try:', s)

# 3) Guarantee newline before try block
s = s.replace('_d["timing_health"] = ""\n        # (always runs)\n        try:',
              '_d["timing_health"] = ""\n\n        # (always runs)\n        try:')

p.write_text(s, encoding="utf-8")
print("OK: block structure repaired")
