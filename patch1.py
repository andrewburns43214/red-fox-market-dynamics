import re

with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

anchor = '    # Fallback: keep something deterministic\n    return (m or "MONEYLINE"), s\n'

new_func = '''

def normalize_side_key(sport: str, market_display: str, side_raw: str) -> str:
    """
    Canonical side key for row_state, elig_map joins, and migration.
    Returns a stable string key -- never used for display.
    TOTAL  -> TOTAL_OVER or TOTAL_UNDER
    SPREAD -> TEAM: + normalize_team_name(team_only)
    ML     -> TEAM: + normalize_team_name(team_only)
    No fuzzy matching -- deterministic only.
    """
    import re as _re
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
    return f"TEAM:{team_norm}"

'''

if anchor in content:
    content = content.replace(anchor, anchor + new_func, 1)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: normalize_side_key inserted")
else:
    print("FAILED: anchor not found -- file unchanged")
