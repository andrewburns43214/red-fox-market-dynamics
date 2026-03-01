with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

old = """    # Idempotent: already canonical -- return as-is
    if s in ("TOTAL_OVER", "TOTAL_UNDER", "TOTAL_UNKNOWN"):
        return s
    if s.startswith("TEAM:"):
        return s"""

new = """    # Idempotent: already canonical -- return as-is
    # But only if it looks clean (no double-prefix like TEAM:teamteam...)
    if s in ("TOTAL_OVER", "TOTAL_UNDER", "TOTAL_UNKNOWN"):
        return s
    if s.startswith("TEAM:") and not s.startswith("TEAM:team"):
        return s
    # Strip TEAM: prefix if present before normalizing (handles double-prefix repair)
    if s.startswith("TEAM:"):
        s = s[5:]"""

if old in content:
    content = content.replace(old, new, 1)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: normalize_side_key idempotent check fixed")
else:
    print("FAILED: anchor not found")
