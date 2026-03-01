import pandas as pd
import numpy as np
import re
from pathlib import Path

DASH = Path("data/dashboard.csv")
SNAP = Path("data/snapshots.csv")

def clean_cols(df):
    df = df.copy()
    df.columns = [str(c).replace("\ufeff","").strip() for c in df.columns]
    return df

def pick_col(df, candidates):
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # fuzzy contains
    for c in candidates:
        rx = re.compile(re.escape(c), re.I)
        for col in df.columns:
            if rx.search(col):
                return col
    return None

def norm_market(x):
    s = str(x).strip().upper()
    if "MONEY" in s: return "MONEYLINE"
    if "SPREAD" in s: return "SPREAD"
    if "TOTAL" in s or "O/U" in s or "OU" in s: return "TOTAL"
    return s

def extra_flag_cols(df):
    pats = [
        r"timing", r"bucket", r"veto", r"contrad", r"govern", r"dampen",
        r"late_", r"reversion", r"alt", r"esco", r"rsn", r"injur",
        r"mahomes", r"jordan", r"cert", r"strong_block", r"risk", r"cap"
    ]
    rx = re.compile("|".join(pats), re.I)
    return [c for c in df.columns if rx.search(c)]

def main():
    if not DASH.exists() or not SNAP.exists():
        raise SystemExit("Missing data/dashboard.csv or data/snapshots.csv")

    dash = clean_cols(pd.read_csv(DASH))
    snap = clean_cols(pd.read_csv(SNAP))

    # --- dashboard (LONG) columns ---
    d_gid = pick_col(dash, ["game_id","event_id","id"])
    d_sport = pick_col(dash, ["sport","league"])
    d_game = pick_col(dash, ["game","matchup","game_display","game_name"])
    d_mkt = pick_col(dash, ["market_display","market","market_type"])
    d_fav = pick_col(dash, ["favored_side","favored","pick","side"])
    d_gc = pick_col(dash, ["game_confidence","confidence","conf"])
    d_ne = pick_col(dash, ["net_edge_market","net_edge","netedge"])
    d_dec = pick_col(dash, ["decision","bucket"])

    missing = [("game_id",d_gid),("market",d_mkt),("favored",d_fav)]
    miss = [name for name,val in missing if not val]
    if miss:
        raise SystemExit(f"dashboard.csv missing required cols: {miss}. Found cols={list(dash.columns)}")

    dash2 = dash.copy()
    dash2["_gid"] = dash2[d_gid].astype(str)
    dash2["_mkt"] = dash2[d_mkt].apply(norm_market)

    # Keep only ML + SPREAD summary rows
    ms = dash2[dash2["_mkt"].isin(["MONEYLINE","SPREAD"])].copy()

    # If dashboard is truly at (sport, game_id, market_display), there should be 0/1 row per gid per market.
    # We'll take the first row per gid per market.
    ms = ms.sort_values(["_gid","_mkt"]).groupby(["_gid","_mkt"], as_index=False).head(1)

    # Build ML + SPREAD join per game_id
    ml = ms[ms["_mkt"]=="MONEYLINE"].copy()
    sp = ms[ms["_mkt"]=="SPREAD"].copy()

    keep = ["_gid"]
    if d_sport: keep.append(d_sport)
    if d_game: keep.append(d_game)
    for c in [d_fav,d_gc,d_ne,d_dec]:
        if c: keep.append(c)

    ml = ml[keep].rename(columns={
        d_fav:"ML_favored",
        d_gc:"ML_game_confidence" if d_gc else d_gc,
        d_ne:"ML_net_edge" if d_ne else d_ne,
        d_dec:"ML_decision" if d_dec else d_dec,
    })
    sp = sp[keep].rename(columns={
        d_fav:"SPREAD_favored",
        d_gc:"SPREAD_game_confidence" if d_gc else d_gc,
        d_ne:"SPREAD_net_edge" if d_ne else d_ne,
        d_dec:"SPREAD_decision" if d_dec else d_dec,
    })

    # Avoid duplicate column names from sport/game
    for c in [d_sport,d_game]:
        if c and c in sp.columns:
            sp = sp.drop(columns=[c])

    merged = ml.merge(sp, on="_gid", how="inner")

    mism = merged[
        merged["ML_favored"].astype(str).str.strip() != merged["SPREAD_favored"].astype(str).str.strip()
    ].copy()

    print("\n=== SUMMARY (dashboard LONG) ===")
    print(f"games with BOTH ML+SPREAD rows: {len(merged)}")
    print(f"mismatches (ML != SPREAD): {len(mism)}")

    # --- choose 5 games per your spec ---
    # 2 NBA mismatches (tight spreads heuristic comes from snapshots), 2 NCAAB mismatches, 1 NCAAB control (Omaha if possible)
    sport_series = merged[d_sport].astype(str).str.upper() if d_sport else pd.Series([""]*len(merged))

    nba_m = mism[sport_series.loc[mism.index].str.contains("NBA", na=False)].copy() if d_sport else mism.iloc[0:0]
    ncaab_m = mism[sport_series.loc[mism.index].str.contains("NCAAB", na=False)].copy() if d_sport else mism.iloc[0:0]

    # snapshot columns
    s_gid = pick_col(snap, ["game_id","event_id","id"])
    s_mkt = pick_col(snap, ["market","market_display"])
    s_side = pick_col(snap, ["side","normalized_side"])
    s_open = pick_col(snap, ["open_line","open","open_price"])
    s_cur = pick_col(snap, ["current_line","current","price","line"])
    s_bets = pick_col(snap, ["bets_pct","bet_pct","tickets_pct"])
    s_money = pick_col(snap, ["money_pct","handle_pct"])
    s_ts = pick_col(snap, ["timestamp","ts","last_updated","updated","snapshot_time","run_time"])

    if not all([s_gid, s_mkt, s_side, s_cur]):
        raise SystemExit("snapshots.csv missing required cols for audit (need game_id, market, side, current_line at minimum).")

    snap2 = snap.copy()
    snap2["_gid"] = snap2[s_gid].astype(str)
    snap2["_mkt"] = snap2[s_mkt].astype(str).str.upper()
    if s_ts:
        snap2["_ts_dt"] = pd.to_datetime(snap2[s_ts], errors="coerce", utc=True)
    else:
        snap2["_ts_dt"] = pd.NaT

    def spread_abs_from_current(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        s = str(v)
        m = re.search(r'([+-]\d+(?:\.\d+)?)', s)
        if not m: return np.nan
        try:
            return abs(float(m.group(1).replace("+","")))
        except Exception:
            return np.nan

    def pick_tight_nba_ids(df, n=2):
        if df.empty: return []
        gids = df["_gid"].astype(str).tolist()
        sp_rows = snap2[(snap2["_gid"].isin(gids)) & (snap2["_mkt"].str.contains("SPREAD"))].copy()
        if sp_rows.empty:
            return gids[:n]
        if s_ts:
            sp_rows = sp_rows.sort_values(["_gid","_ts_dt"]).groupby(["_gid", s_side], as_index=False).tail(1)
        # take one side per game to estimate tightness (min abs spread among sides)
        sp_rows["_abs"] = sp_rows[s_cur].apply(spread_abs_from_current)
        tight = sp_rows.groupby("_gid")["_abs"].min().sort_values()
        # prefer <=2.5 first
        tight_ids = [gid for gid, val in tight.items() if pd.notna(val) and val <= 2.5]
        if len(tight_ids) >= n:
            return tight_ids[:n]
        # else smallest abs spreads
        return list(tight.index)[:n]

    nba_pick = pick_tight_nba_ids(nba_m, 2)
    ncaab_pick = ncaab_m["_gid"].astype(str).tolist()[:2]

    # Control: NCAAB agreement with Omaha (if present), else first NCAAB agree game
    agree = merged[
        merged["ML_favored"].astype(str).str.strip() == merged["SPREAD_favored"].astype(str).str.strip()
    ].copy()
    ncaab_a = agree[sport_series.loc[agree.index].str.contains("NCAAB", na=False)].copy() if d_sport else agree.iloc[0:0]

    control_id = None
    if not ncaab_a.empty and d_game:
        omaha = ncaab_a[ncaab_a[d_game].astype(str).str.contains("Omaha", case=False, na=False)]
        if not omaha.empty:
            control_id = str(omaha.iloc[0]["_gid"])
        else:
            control_id = str(ncaab_a.iloc[0]["_gid"])
    elif not ncaab_a.empty:
        control_id = str(ncaab_a.iloc[0]["_gid"])

    picks = []
    for x in nba_pick + ncaab_pick + ([control_id] if control_id else []):
        if x and x not in picks:
            picks.append(x)

    print("\n=== SELECTED GAME_IDS ===")
    for gid in picks:
        print(" -", gid)

    flags = extra_flag_cols(dash2)

    def print_dash_rows(gid):
        sub = dash2[dash2["_gid"]==str(gid)].copy()
        if sub.empty:
            print("\n[DASHBOARD] no rows for game_id", gid)
            return
        sub = sub[sub["_mkt"].isin(["MONEYLINE","SPREAD"])].copy()
        if sub.empty:
            print("\n[DASHBOARD] no ML/SPREAD rows for game_id", gid)
            return
        cols = []
        for c in [d_sport,d_game,d_gid,d_mkt,d_fav,d_gc,d_ne,d_dec]:
            if c and c in sub.columns and c not in cols:
                cols.append(c)
        # include flags (only if present)
        for c in flags:
            if c in sub.columns and c not in cols:
                cols.append(c)
        print("\n[DASHBOARD rows (ML + SPREAD)]")
        subp = sub[cols].copy()
        # stable sort: market
        subp = subp.sort_values([d_mkt], kind="stable")
        for _, rr in subp.iterrows():
            parts = [f"{c}={rr.get(c)}" for c in subp.columns]
            print("  - " + " | ".join(parts))

    def print_snapshot_rows(gid):
        sg = snap2[snap2["_gid"]==str(gid)].copy()
        if sg.empty:
            print("\n[SNAPSHOTS] no rows for game_id", gid)
            return
        sg = sg[sg["_mkt"].apply(lambda s: ("MONEY" in s) or ("SPREAD" in s))].copy()
        if sg.empty:
            print("\n[SNAPSHOTS] no ML/SPREAD rows for game_id", gid)
            return
        # keep latest per (market, side)
        if s_ts:
            sg = sg.sort_values(["_mkt","_ts_dt"]).groupby(["_mkt", s_side], as_index=False).tail(1)

        out_cols = []
        for c in [s_gid,s_mkt,s_side,s_open,s_cur,s_bets,s_money,s_ts]:
            if c and c in sg.columns and c not in out_cols:
                out_cols.append(c)

        print("\n[SNAPSHOTS latest per market+side]")
        sgp = sg[out_cols].copy().sort_values([s_mkt,s_side], kind="stable")
        for _, rr in sgp.iterrows():
            parts = [f"{c}={rr.get(c)}" for c in sgp.columns]
            print("  - " + " | ".join(parts))

    for gid in picks:
        print("\n" + "="*110)
        # header
        hdr = [f"game_id={gid}"]
        if d_sport or d_game:
            row_any = dash2[dash2["_gid"]==str(gid)].head(1)
            if not row_any.empty:
                if d_sport: hdr.insert(0, f"sport={row_any.iloc[0].get(d_sport)}")
                if d_game: hdr.insert(1, f"game={row_any.iloc[0].get(d_game)}")
        print(" | ".join(hdr))

        print_dash_rows(gid)
        print_snapshot_rows(gid)

    print("\nDONE")

if __name__ == "__main__":
    main()
