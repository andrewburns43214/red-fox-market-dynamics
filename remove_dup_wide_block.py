from pathlib import Path
import re

p = Path("main.py")
s = p.read_text(encoding="utf-8")

anchor = "# Base keys for one-row-per-game"

# find all anchor positions
matches = list(re.finditer(re.escape(anchor), s))
if len(matches) < 2:
    print("Did not find duplicate anchor safely. Aborting.")
    raise SystemExit(1)

start = matches[1].start()

# find end of duplicate block:
# we stop before the NEXT indentation drop or before next major section.
# safest anchor: the SECOND occurrence of 'print("[DBG wide] columns:"'
dbg_matches = list(re.finditer(r'print\("\[DBG wide\] columns:', s))

if len(dbg_matches) < 2:
    print("Did not find duplicate debug anchor safely. Aborting.")
    raise SystemExit(1)

end = dbg_matches[1].start()

print("Removing duplicate FORCE wide block...")
new_s = s[:start] + s[end:]

p.write_text(new_s, encoding="utf-8")
print("OK: duplicate wide block removed.")
