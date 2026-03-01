with open("main.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the dead code block and replace the whole function ending
out = []
i = 0
found = False
while i < len(lines):
    line = lines[i]
    # Detect start of dead code (unreachable after first return True)
    if (not found and 
        i > 2800 and
        line.strip() == "sport = str(row.get(\"sport\",\"\")).upper()"):
        # Write the replacement instead
        out.append("        sport = str(row.get(\"sport\",\"\")).upper()\n")
        out.append("        # Sport-specific persistence threshold\n")
        out.append("        min_streak = NCAAB_STRONG_MIN_PERSIST if sport == \"NCAAB\" else 2\n")
        out.append("        if ss < min_streak:\n")
        out.append("            return False\n")
        out.append("        # Sport-specific early/late blocks\n")
        out.append("        if sport == \"NCAAB\":\n")
        out.append("            if NCAAB_EARLY_STRONG_BLOCK and tb == \"EARLY\":\n")
        out.append("                return False\n")
        out.append("            if NCAAB_LATE_STRONG_BLOCK and tb == \"LATE\":\n")
        out.append("                return False\n")
        out.append("        if sport == \"NCAAF\":\n")
        out.append("            if NCAAF_EARLY_INSTANT_STRONG_BLOCK and tb == \"EARLY\":\n")
        out.append("                return False\n")
        out.append("            if NCAAF_LATE_NEW_STRONG_BLOCK and tb == \"LATE\":\n")
        out.append("                return False\n")
        out.append("        # NCAAB multi-market requirement\n")
        out.append("        if sport == \"NCAAB\" and NCAAB_REQUIRE_MULTI_MARKET:\n")
        out.append("            spread_ok = str(row.get(\"SPREAD_favored\",\"\")).strip() != \"\"\n")
        out.append("            ml_ok = str(row.get(\"MONEYLINE_favored\",\"\")).strip() != \"\"\n")
        out.append("            if not (spread_ok and ml_ok):\n")
        out.append("                return False\n")
        out.append("        # Stability: last_score must be within delta of peak\n")
        out.append("        delta = NCAAB_STRONG_STABILITY_DELTA if sport == \"NCAAB\" else NCAAF_STRONG_STABILITY_DELTA if sport == \"NCAAF\" else 3.0\n")
        out.append("        if ls < (ps - delta):\n")
        out.append("            return False\n")
        out.append("        return True\n")
        found = True
        # Skip lines until we hit latest["strong_eligible"]
        i += 1
        while i < len(lines) and "strong_eligible" not in lines[i]:
            i += 1
        # Don't skip the strong_eligible line itself
        continue
    out.append(line)
    i += 1

if found:
    with open("main.py", "w", encoding="utf-8") as f:
        f.writelines(out)
    print("SUCCESS: Patch 4 applied")
else:
    print("FAILED: anchor not found")
