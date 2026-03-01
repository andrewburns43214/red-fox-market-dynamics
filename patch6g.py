with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

old = "    # DEBUG: trace _is_strong_eligible on high-score rows"
new = """    # DEBUG: show latest columns and sample score values
    if True:
        print("[latest debug] columns:", list(latest.columns))
        print("[latest debug] sample game_confidence:", latest.get("game_confidence", latest.get("model_score", "MISSING")).head(5).tolist() if hasattr(latest.get("game_confidence", None), "head") else "NO COL")
        _score_cols = [c for c in latest.columns if "score" in c.lower() or "confidence" in c.lower()]
        print("[latest debug] score-related cols:", _score_cols)
        if _score_cols:
            print("[latest debug] sample values:", latest[_score_cols].head(3).to_dict())
    # DEBUG: trace _is_strong_eligible on high-score rows"""

if old in content:
    content = content.replace(old, new, 1)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS")
else:
    print("FAILED")
