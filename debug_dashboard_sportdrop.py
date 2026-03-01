import pandas as pd
import re

SNAP = "data/snapshots.csv"

def vc(df, label):
    if df is None:
        print(f"\n[{label}] df=None")
        return
    if "sport" in df.columns:
        print(f"\n[{label}] rows={len(df)} sports={df['sport'].value_counts().to_dict()}")
    else:
        print(f"\n[{label}] rows={len(df)} (no sport col)")

def main():
    df = pd.read_csv(SNAP, keep_default_na=False, dtype=str)
    vc(df, "READ snapshots")

    # mirror build_dashboard() early normalization
    df = df.rename(columns={"open":"open_line", "current":"current_line", "news":"injury_news"})
    if "timestamp" not in df.columns:
        raise SystemExit("snapshots.csv missing timestamp column")

    df["timestamp"] = df["timestamp"].astype(str).str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, format="mixed")
    vc(df, "AFTER timestamp parse (before dropna)")

    df2 = df.dropna(subset=["timestamp"]).copy()
    vc(df2, "AFTER timestamp dropna")

    # sport normalize
    df2["sport"] = df2["sport"].astype(str).str.lower().str.strip()
    df2["game_id"] = df2["game_id"].fillna("").astype(str)
    df2["side"] = df2["side"].fillna("").astype(str)
    df2["market"] = df2["market"].fillna("unknown").astype(str)

    # junk strip
    df2["market"] = df2["market"].astype(str).str.replace(r"\s+opens in a new tab.*$", "", regex=True)
    df2["game"]   = df2["game"].astype(str).str.replace(r"\s+opens in a new tab.*$", "", regex=True)

    # === replicate your current market_display logic (as seen in your snippet) ===
    _line_txt = df2["current_line"].fillna("").astype(str)
    _side_txt = df2["side"].fillna("").astype(str)

    _line_u = _line_txt.str.upper()
    _side_u = _side_txt.str.upper()

    _is_total = (_side_u.str.contains("UNDER") | _side_u.str.contains("OVER") |
                 _line_u.str.contains("UNDER") | _line_u.str.contains("OVER"))

    # SPREAD detection: signed number before "@ price" token
    _cl = df2.get("current_line", "").fillna("").astype(str).str.upper()
    _is_spread = (~_is_total) & _cl.str.contains(r"\s[+-]\d+(?:\.\d+)?\s*@\s*[+-]?\d+\s*$", regex=True, na=False)
    _is_ml = (~_is_total) & (~_is_spread)

    df2["market_display"] = "SPREAD"
    df2.loc[_is_total, "market_display"] = "TOTAL"
    df2.loc[_is_ml, "market_display"] = "MONEYLINE"

    # side_key logic as currently implemented
    df2["side_key"] = df2["side"].astype(str)
    df2.loc[df2["market_display"]=="SPREAD", "side_key"] = (
        df2.loc[df2["market_display"]=="SPREAD", "side_key"]
           .str.replace(r"\s[+-]\d+(?:\.\d+)?\s*$", "", regex=True)
           .str.strip()
    )
    df2.loc[df2["market_display"]=="TOTAL", "side_key"] = (
        df2.loc[df2["market_display"]=="TOTAL", "side_key"]
           .str.extract(r"^(Over|Under)", expand=False)
           .fillna(df2.loc[df2["market_display"]=="TOTAL", "side_key"])
           .str.strip()
    )

    # show how NBA is being classified
    nba = df2[df2["sport"]=="nba"].copy()
    print("\n[NBA] market_display distribution:", nba["market_display"].value_counts().to_dict())
    print("[NBA] example current_line samples:")
    for x in nba["current_line"].head(8).tolist():
        print("  ", x)

    # group to latest (same key as build_dashboard)
    df2 = df2.sort_values("timestamp")
    latest = (
        df2.groupby(["sport","game_id","market_display","side_key"], as_index=False)
           .tail(1)
           .copy()
           .reset_index(drop=True)
    )
    vc(latest, "LATEST grouped tail(1)")

    # IMPORTANT: check for game_time_ny existence and filter effect
    print("\n[CHECK] columns containing 'time':", [c for c in latest.columns if "time" in c.lower()])

    if "game_time_ny" in latest.columns:
        today_ny = pd.Timestamp.now(tz="America/New_York").normalize()
        before = len(latest)
        latest2 = latest[(latest["game_time_ny"].isna()) | (latest["game_time_ny"] >= today_ny)]
        print(f"\n[FILTER game_time_ny] before={before} after={len(latest2)} sports={latest2['sport'].value_counts().to_dict()}")
    else:
        print("\n[FILTER game_time_ny] skipped (no game_time_ny column)")

    # Also check if _game_time exists and whether it’s blank for nba
    if "_game_time" in latest.columns:
        nba_latest = latest[latest["sport"]=="nba"].copy()
        blank = (nba_latest["_game_time"].astype(str).str.strip() == "").sum()
        print(f"\n[NBA latest] _game_time blanks={blank}/{len(nba_latest)}")
        print("[NBA latest] _game_time sample:", nba_latest["_game_time"].head(5).tolist())

if __name__ == "__main__":
    main()
