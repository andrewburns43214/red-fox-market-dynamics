import pandas as pd
import numpy as np
import re
from pathlib import Path

DASH = Path("data/dashboard.csv")
SNAP = Path("data/snapshots.csv")

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff","").strip() for c in df.columns]
    return df

def _pick_col(df, candidates):
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # fuzzy contains
    for c in candidates:
        pat = re.compile(re.escape(c), re.I)
        for col in df.columns:
            if pat.search(col):
                return col
    return None

def _to_num(x):
    try:
        if pd.isna(x): return np.nan
        s = str(x).strip()
        if s in ("", "—", "-", "None", "nan"): return np.nan
        # remove plus sign
        s = s.replace("+","")
        return float(s)
    except Exception:
        return np.nan

def _extract_spread_number(current_line):
    """
    Tries to pull the numeric spread from common strings, e.g.:
    '-1.5 @ -110', 'Warriors -2.0', '-2', '+1.5 (-105)'
    Returns float or NaN.
    """
    if current_line is None or (isinstance(current_line, float) and np.isnan(current_line)):
        return np.nan
    s = str(current_line)
    # find first signed number with optional decimal
    m = re.search(r'([+-]\d+(?:\.\d+)?)', s)
    if not m:
        return np.nan
    return _to_num(m.group(1))

def _find_dashboard_market_cols(df, market_prefix):
    # canonical expected
    favored = _pick_col(df, [f"{market_prefix}_favored", f"{market_prefix}_favored_side", f"{market_prefix}_pick", f"{market_prefix}_side"])
    gc = _pick_col(df, [f"{market_prefix}_game_confidence", f"{market_prefix}_confidence", f"{market_prefix}_conf"])
    ne = _pick_col(df, [f"{market_prefix}_net_edge", f"{market_prefix}_netedge"])
    score = _pick_col(df, [f"{market_prefix}_model_score", f"{market_prefix}_score"])
    decision = _pick_col(df, [f"{market_prefix}_decision", f"{market_prefix}_bucket"])
    return {"favored": favored, "game_confidence": gc, "net_edge": ne, "model_score": score, "decision": decision}

def _guess_id_cols(df):
    game_id = _pick_col(df, ["game_id", "event_id", "id"])
    sport = _pick_col(df, ["sport", "league"])
    game = _pick_col(df, ["game", "matchup", "game_display", "game_name"])
    home = _pick_col(df, ["home_team", "home"])
    away = _pick_col(df, ["away_team", "away"])
    return {"game_id": game_id, "sport": sport, "game": game, "home": home, "away": away}

def _extra_flags_cols(df):
    # show anything that smells like timing/veto/contradiction/governor/dampener
    pats = [
        r"timing", r"bucket", r"veto", r"contrad", r"govern", r"dampen", r"late_", r"reversion",
        r"alt", r"esco", r"rsn", r"injur", r"mahomes", r"jordan", r"cert", r"strong_block"
    ]
    rx = re.compile("|".join(pats), re.I)
    cols = [c for c in df.columns if rx.search(c)]
    # keep stable order
    return cols

def _snapshot_cols(df):
    market = _pick_col(df, ["market", "market_display"])
    side = _pick_col(df, ["side", "normalized_side"])
    open_line = _pick_col(df, ["open_line", "open", "open_price"])
    current_line = _pick_col(df, ["current_line", "current", "price", "line"])
    bets = _pick_col(df, ["bets_pct", "bet_pct", "bets", "tickets_pct"])
    money = _pick_col(df, ["money_pct", "handle_pct", "money"])
    ts = _pick_col(df, ["timestamp", "ts", "last_updated", "updated", "snapshot_time", "run_time"])
    return {"market": market, "side": side, "open_line": open_line, "current_line": current_line, "bets_pct": bets, "money_pct": money, "timestamp": ts}

def main():
    if not DASH.exists() or not SNAP.exists():
        raise SystemExit(f"Missing files. Need {DASH} and {SNAP}")

    dash = _clean_cols(pd.read_csv(DASH))
    snap = _clean_cols(pd.read_csv(SNAP))

    did = _guess_id_cols(dash)
    if not did["game_id"]:
        raise SystemExit("dashboard.csv: could not find game_id column (expected 'game_id' or similar).")

    m_ml = _find_dashboard_market_cols(dash, "MONEYLINE")
    m_sp = _find_dashboard_market_cols(dash, "SPREAD")

    if not m_ml["favored"] or not m_sp["favored"]:
        raise SystemExit(
            "dashboard.csv: missing favored columns for MONEYLINE or SPREAD. "
            f"Found ML favored={m_ml['favored']} SPREAD favored={m_sp['favored']}"
        )

    # base view
    base_cols = [c for c in [did["sport"], did["game"], did["home"], did["away"], did["game_id"]] if c]
    ml_cols = [c for c in [m_ml["favored"], m_ml["game_confidence"], m_ml["net_edge"], m_ml["model_score"], m_ml["decision"]] if c]
    sp_cols = [c for c in [m_sp["favored"], m_sp["game_confidence"], m_sp["net_edge"], m_sp["model_score"], m_sp["decision"]] if c]
    flag_cols = _extra_flags_cols(dash)

    # mismatch detection
    d2 = dash.copy()
    d2["_ml_fav"] = d2[m_ml["favored"]].astype(str).str.strip()
    d2["_sp_fav"] = d2[m_sp["favored"]].astype(str).str.strip()

    mism = d2[(d2["_ml_fav"] != "") & (d2["_sp_fav"] != "") & (d2["_ml_fav"] != d2["_sp_fav"])].copy()
    agree = d2[(d2["_ml_fav"] != "") & (d2["_sp_fav"] != "") & (d2["_ml_fav"] == d2["_sp_fav"])].copy()

    sport_col = did["sport"]
    game_col = did["game"]

    print("\n=== SUMMARY ===")
    print(f"dashboard rows: {len(d2)}")
    print(f"mismatches (ML!=SPREAD): {len(mism)}")
    print(f"agreements (ML==SPREAD): {len(agree)}")

    # --- pick sample set ---
    picks = []

    def _pick_n(df, n):
        return list(df.head(n)[did["game_id"]].astype(str).tolist())

    # NBA: try tight spreads (<=2.5) using snapshots SPREAD current_line
    nba_mism = mism[mism[sport_col].astype(str).str.upper().str.contains("NBA")] if sport_col else mism.iloc[0:0]
    nba_pick = []

    if len(nba_mism) > 0:
        # build tightness using snapshots
        s_cols = _snapshot_cols(snap)
        if not all([s_cols["market"], s_cols["current_line"], s_cols["timestamp"]]):
            nba_pick = _pick_n(nba_mism, 2)
        else:
            # for each game_id, find latest SPREAD row (any side) and extract abs spread
            snap2 = snap.copy()
            snap2["_gid"] = snap2[_pick_col(snap2, ["game_id", "event_id", "id"])].astype(str) if _pick_col(snap2, ["game_id","event_id","id"]) else ""
            snap2["_mkt"] = snap2[s_cols["market"]].astype(str).str.upper()
            snap2["_ts"] = snap2[s_cols["timestamp"]]
            # parse ts to datetime if possible
            snap2["_ts_dt"] = pd.to_datetime(snap2["_ts"], errors="coerce", utc=True)
            # filter SPREAD
            sp = snap2[snap2["_mkt"].str.contains("SPREAD")].copy()
            # compute spread number from current_line
            sp["_spread_num"] = sp[s_cols["current_line"]].apply(_extract_spread_number).abs()
            # latest per game
            sp_latest = sp.sort_values(["_gid","_ts_dt"]).groupby("_gid", as_index=False).tail(1)
            tight_gids = set(sp_latest[sp_latest["_spread_num"].notna() & (sp_latest["_spread_num"] <= 2.5)]["_gid"].tolist())
            nba_tight = nba_mism[nba_mism[did["game_id"]].astype(str).isin(tight_gids)].copy()
            if len(nba_tight) >= 2:
                nba_pick = _pick_n(nba_tight, 2)
            else:
                nba_pick = _pick_n(nba_mism, 2)

    # NCAAB mismatches
    ncaab_mism = mism[mism[sport_col].astype(str).str.upper().str.contains("NCAAB")] if sport_col else mism.iloc[0:0]
    ncaab_pick = _pick_n(ncaab_mism, 2) if len(ncaab_mism) else []

    # NCAAB control: Omaha + agree
    ncaab_agree = agree[agree[sport_col].astype(str).str.upper().str.contains("NCAAB")] if sport_col else agree.iloc[0:0]
    omaha_control = []
    if len(ncaab_agree) and game_col:
        omaha = ncaab_agree[ncaab_agree[game_col].astype(str).str.contains("Omaha", case=False, na=False)]
        if len(omaha):
            omaha_control = _pick_n(omaha, 1)
        else:
            omaha_control = _pick_n(ncaab_agree, 1)
    elif len(ncaab_agree):
        omaha_control = _pick_n(ncaab_agree, 1)

    picks = nba_pick + ncaab_pick + omaha_control

    # de-dupe while preserving order
    seen = set()
    picks = [x for x in picks if not (x in seen or seen.add(x))]

    print("\n=== SELECTED GAMES (game_id) ===")
    for gid in picks:
        print(" -", gid)

    # snapshot helpers
    s_id = _pick_col(snap, ["game_id","event_id","id"])
    s_cols = _snapshot_cols(snap)
    if not s_id:
        raise SystemExit("snapshots.csv: could not find game_id column (expected 'game_id' or similar).")

    snap2 = snap.copy()
    snap2["_gid"] = snap2[s_id].astype(str)
    snap2["_mkt"] = snap2[s_cols["market"]].astype(str).str.upper() if s_cols["market"] else ""
    snap2["_ts_raw"] = snap2[s_cols["timestamp"]] if s_cols["timestamp"] else None
    snap2["_ts_dt"] = pd.to_datetime(snap2["_ts_raw"], errors="coerce", utc=True) if s_cols["timestamp"] else pd.NaT

    def print_game(gid):
        row = d2[d2[did["game_id"]].astype(str) == str(gid)]
        if row.empty:
            print(f"\n[WARN] game_id {gid} not found in dashboard.")
            return
        r = row.iloc[0]

        print("\n" + "="*110)
        header = []
        if did["sport"]: header.append(f"sport={r[did['sport']]}")
        if did["game"]: header.append(f"game={r[did['game']]}")
        header.append(f"game_id={gid}")
        print(" | ".join(header))

        show_cols = base_cols + ml_cols + sp_cols
        # include flags but only those that exist and have any non-null in the row (or are informative)
        extra = []
        for c in flag_cols:
            if c in row.columns:
                v = r.get(c)
                if not (pd.isna(v) or str(v).strip() in ("", "—", "-", "None", "nan")):
                    extra.append(c)
        # if no values, still include key structural columns if they exist
        if not extra:
            for c in flag_cols:
                if c in row.columns and any(k in c.lower() for k in ["timing","bucket"]):
                    extra.append(c)

        show_cols2 = show_cols + extra
        show_cols2 = [c for c in show_cols2 if c in row.columns]

        print("\n[DASHBOARD]")
        for c in show_cols2:
            print(f"  {c}: {r.get(c)}")

        # snapshots: latest rows for ML + SPREAD for this game
        sg = snap2[snap2["_gid"] == str(gid)].copy()
        if sg.empty:
            print("\n[SNAPSHOTS] (no rows found for this game_id)")
            return

        wanted_markets = ["MONEYLINE", "SPREAD"]
        sg = sg[sg["_mkt"].apply(lambda x: any(w in x for w in wanted_markets))].copy()
        if sg.empty:
            print("\n[SNAPSHOTS] (no ML/SPREAD rows found for this game_id)")
            return

        # keep latest per (market, side) if timestamp exists; else just show all
        if s_cols["timestamp"]:
            sg = sg.sort_values(["_mkt","_ts_dt"]).groupby(["_mkt", s_cols["side"]], as_index=False).tail(1)

        out_cols = []
        for key in ["market","side","open_line","current_line","bets_pct","money_pct","timestamp"]:
            c = s_cols.get(key)
            if c and c in sg.columns:
                out_cols.append(c)

        # Always print market+side even if mapping failed
        if s_cols["market"] and s_cols["market"] in sg.columns and s_cols["market"] not in out_cols:
            out_cols.insert(0, s_cols["market"])
        if s_cols["side"] and s_cols["side"] in sg.columns and s_cols["side"] not in out_cols:
            out_cols.insert(1, s_cols["side"])

        print("\n[SNAPSHOTS: latest per market+side]")
        sg_print = sg[out_cols].copy() if out_cols else sg.copy()

        # nicer sort: market, side
        if s_cols["market"] and s_cols["side"]:
            sg_print = sg_print.sort_values([s_cols["market"], s_cols["side"]], kind="stable")

        for _, rr in sg_print.iterrows():
            parts = []
            for c in sg_print.columns:
                parts.append(f"{c}={rr.get(c)}")
            print("  - " + " | ".join(parts))

    for gid in picks:
        print_game(gid)

    print("\nDONE.")

if __name__ == "__main__":
    main()
