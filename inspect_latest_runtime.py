import main
x = main.build_dashboard()
print("TYPE:", type(x))
try:
    print("LEN:", len(x))
    print("COLS:", list(x.columns)[:10])
except Exception as e:
    print("FAILED:", e)
