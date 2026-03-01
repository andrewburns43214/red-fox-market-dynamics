with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

old = '''    import re as _re
    m = (market_display or "").strip().upper()
    s = (side_raw or "").strip()
    su = s.upper()
    if m == "TOTAL" or "OVER" in su or "UNDER" in su:
        if "UNDER" in su:
            return "TOTAL_UNDER"
        if "OVER" in su:
            return "TOTAL_OVER"
        return "TOTAL_UNKNOWN"
    team = _re.sub(r"\\s[+-]\\d+(?:\\.\\d+)?\\s*$", "", s).strip()
    team_norm = normalize_team_name(team)
    return f"TEAM:{team_norm}"'''

new = '''    import re as _re
    m = (market_display or "").strip().upper()
    s = (side_raw or "").strip()
    su = s.upper()
    # Idempotent: already canonical -- return as-is
    if s in ("TOTAL_OVER", "TOTAL_UNDER", "TOTAL_UNKNOWN"):
        return s
    if s.startswith("TEAM:"):
        return s
    if m == "TOTAL" or "OVER" in su or "UNDER" in su:
        if "UNDER" in su:
            return "TOTAL_UNDER"
        if "OVER" in su:
            return "TOTAL_OVER"
        return "TOTAL_UNKNOWN"
    team = _re.sub(r"\\s[+-]\\d+(?:\\.\\d+)?\\s*$", "", s).strip()
    team_norm = normalize_team_name(team)
    return f"TEAM:{team_norm}"'''

if old in content:
    content = content.replace(old, new, 1)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: normalize_side_key is now idempotent")
else:
    print("FAILED: anchor not found")
