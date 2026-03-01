import pandas as pd

p="data/snapshots.csv"
df=pd.read_csv(p,keep_default_na=False,dtype=str)

mlb=df[df["sport"].str.upper()=="MLB"]

for gid in mlb["game_id"].unique():

    mask=(df["game_id"]==gid)&(df["market"]=="moneyline")

    df.loc[mask,"open_line"]="-150"
    df.loc[mask,"prev_line_val"]="-155"
    df.loc[mask,"current_line_val"]="-170"
    df.loc[mask,"line_move_open"]="-20"
    df.loc[mask,"line_move_prev"]="-15"

    df.loc[mask,"money_pct"]="70"
    df.loc[mask,"bets_pct"]="40"

df.to_csv(p,index=False)
print("Injected MLB movement")
