from pathlib import Path
import re

p = Path("main.py")
s = p.read_text(encoding="utf-8")

pattern = r"def update_final_scores_history\(\):.*?(?=\ndef )"
m = re.search(pattern, s, flags=re.S)

if not m:
    print("Function not found")
else:
    print(m.group(0))
