import pandas as pd

p = "data/snapshots.csv"
df = pd.read_csv(p, keep_default_na=False, dtype=str)

mlb = df[df["sport"].str.upper()=="MLB"].copy()
new_rows = []

for game in mlb["game"].unique():
    teams = mlb[mlb["game"]==game]["side"].tolist()
    if len(teams)!=2:
        continue
    
    fav, dog = teams[0], teams[1]

    base = mlb[mlb["game"]==game].iloc[0].to_dict()

    r1 = base.copy()
    r1["market"] = "spread"
    r1["side"] = fav
    r1["current_line"] = f"{fav} -1.5"
    r1["current_line_val"] = "-1.5"

    r2 = base.copy()
    r2["market"] = "spread"
    r2["side"] = dog
    r2["current_line"] = f"{dog} +1.5"
    r2["current_line_val"] = "+1.5"

    new_rows += [r1,r2]

df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
df.to_csv(p, index=False)

print("Added run line rows:", len(new_rows))
