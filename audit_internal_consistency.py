import pandas as pd

d = pd.read_csv("data/dashboard.csv")

# Check internal consistency
print("Check 1: confidence equals max_side_score")
print("Mismatches:",
      (d["game_confidence"] != d["max_side_score"]).sum())

# Check edge consistency again
print("\nCheck 2: edge equals max - min")
print("Mismatches:",
      ((d["max_side_score"] - d["min_side_score"]) != d["net_edge"]).sum())

# Check favored side presence
print("\nCheck 3: favored_side non-null")
print("Null favored_side:", d["favored_side"].isna().sum())
