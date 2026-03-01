import pandas as pd

r = pd.read_csv("data/results_resolved.csv")

print("\n[RESULTS SANITY CHECK]")
print("Total rows:", len(r))
print("Grade status breakdown:")
print(r["grade_status"].value_counts())

print("\nOutcome breakdown:")
print(r["outcome"].value_counts())

print("\nAny duplicate result_id?")
if "result_id" in r.columns:
    print(r["result_id"].duplicated().sum())
