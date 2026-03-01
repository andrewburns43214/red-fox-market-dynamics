import subprocess, shutil, pandas as pd, os, glob, sys, time

def run_test(path):
    shutil.copy(path, "main.py")
    r = subprocess.run(["python","-m","py_compile","main.py"], capture_output=True, text=True)
    if r.returncode != 0:
        return "compile_fail"

    subprocess.run(["python","main.py","report"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        s = pd.read_csv("data/row_state.csv")
        if "last_score_ts" not in s.columns:
            return "no_column"
        if s["last_score_ts"].notna().any():
            return "GOOD"
        return "nan_only"
    except:
        return "read_fail"


files = sorted(glob.glob("bak/main.py.*.ps1safe.py"), key=os.path.getmtime, reverse=True)

print("\nSearching recent backups...\n")

for f in files[:80]:  # last several hours worth
    res = run_test(f)
    print(f"{os.path.basename(f):60} -> {res}")
    if res == "GOOD":
        print("\nFOUND WORKING METRICS VERSION:\n", f)
        sys.exit()

print("\nNo working timestamp version found in recent backups.")
