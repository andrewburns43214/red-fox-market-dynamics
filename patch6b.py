with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()

old = '        _rs = pd.read_csv(ROW_STATE_PATH, dtype=str, keep_default_na=False)'
new = '        _rs = pd.read_csv("data/row_state.csv", dtype=str, keep_default_na=False)'

if old in content:
    content = content.replace(old, new, 1)
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS")
else:
    print("FAILED: anchor not found")
