with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

old = """                _combined = _combined.drop_duplicates(
                    subset=["sport","game_id","market_display","side"],
                    keep="first"
                )"""

new = """                # Prefer STRONG_BET over BET when deduplicating (upgrade path)
                _decision_rank = {"NO BET": 0, "LEAN": 1, "BET": 2, "STRONG_BET": 3}
                _combined["_rank"] = _combined["game_decision"].map(_decision_rank).fillna(0)
                _combined = _combined.sort_values("_rank", ascending=False)
                _combined = _combined.drop_duplicates(
                    subset=["sport","game_id","market_display","side"],
                    keep="first"
                )
                _combined = _combined.drop(columns=["_rank"], errors="ignore")"""

if old in content:
    content = content.replace(old, new, 1)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS")
else:
    print("FAILED: anchor not found")
