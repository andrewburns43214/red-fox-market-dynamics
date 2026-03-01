with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

old = """    # Strip TEAM: prefix if present before normalizing (handles double-prefix repair)
    if s.startswith("TEAM:"):
        s = s[5:]"""

new = """    # Strip TEAM: prefix if present before normalizing (handles double-prefix repair)
    if s.startswith("TEAM:"):
        s = s[5:]
        # Also strip any residual "team" prefix left by prior double-normalization
        import re as _re2
        s = _re2.sub(r"^team(?=[a-z])", "", s).strip()"""

if old in content:
    content = content.replace(old, new, 1)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: normalize_side_key strips teamteam artifact")
else:
    print("FAILED: anchor not found")
