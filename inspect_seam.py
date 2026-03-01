from pathlib import Path
import re

text = Path("main.py").read_text(encoding="utf-8")
start = text.find('idx_fav = latest.groupby(game_keys)')
print(text[start-800:start+200])
