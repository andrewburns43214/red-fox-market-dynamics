import os
import re
import hashlib
import datetime as dt
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import requests


# ===============================
# Paths
# ===============================
DATA_DIR = "data"
SIGNALS_PATH = os.path.join(DATA_DIR, "signals_baseline.csv")
OUT_RESOLVED_PATH = os.path.join(DATA_DIR, "results_resolved.csv")
OUT_SUMMARY_PATH = os.path.join(DATA_DIR, "color_baseline_summary.csv")
SENTINEL_PATH = os.path.join(DATA_DIR, ".baseline_backfill_done")
SNAPSHOTS_PATH = os.path.join(DATA_DIR, "snapshots.csv")
FINALS_PATH = os.path.join(DATA_DIR, "finals_espn.csv")




# ===============================
# ESPN mappings
# ===============================
ESPN_SCOREBOARD = {
    "nfl":   ("football", "nfl"),
    "nba":   ("basketball", "nba"),
    "nhl":   ("hockey", "nhl"),
    "mlb":   ("baseball", "mlb"),
    "ncaaf": ("football", "college-football"),
    "ncaab": ("basketball", "mens-college-basketball"),
}


# ===============================
# Helpers
# ===============================
def ny_to_yyyymmdd(date_str: str) -> str:
    return dt.date.fromisoformat(date_str).strftime("%Y%m%d")


def stable_game_key(sport: str, game: str) -> str:
    return hashlib.md5(f"{sport}|{game}".encode()).hexdigest()


def extract_american_odds(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r'@\s*([+-]\d+)', str(text))
    if not m:
        return None
    return int(m.group(1))


def is_dark_green_high_underdog(row: Dict[str, Any], threshold: int = 300) -> bool:
    if (row.get("color") or "").lower() != "dark green":
        return False
    odds = extract_american_odds(row.get("current"))
    return odds is not None and odds >= threshold


# ===============================
# ESPN Fetch
# ===============================
def fetch_espn_events(sport: str, yyyymmdd: str) -> List[Dict[str, Any]]:
    if sport not in ESPN_SCOREBOARD:
        return []

    cat, league = ESPN_SCOREBOARD[sport]
    url = f"https://site.api.espn.com/apis/site/v2/sports/{cat}/{league}/scoreboard?dates={yyyymmdd}"

    try:
        # split connect/read timeouts; prevents long hangs on TLS/read
        r = requests.get(url, timeout=(3, 5))
        if r.status_code != 200:
            return []
        j = r.json()
        return j.get("events", []) or []
    except (requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.SSLError,
            ValueError):
        # ValueError covers bad JSON
        return []
    except Exception:
        return []



def match_event(game: str, events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    g = game.lower()
    for ev in events:
        name = (ev.get("name") or ev.get("shortName") or "").lower()
        if all(part in name for part in g.replace(" vs ", "@").split("@")):
            return ev
    return None


def extract_final(ev: Dict[str, Any]) -> Optional[Tuple[str, int, str, int]]:
    comp = ev.get("competitions", [{}])[0]
    status = comp.get("status", {}).get("type", {})
    if not status.get("completed"):
        return None

    home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
    away = next(c for c in comp["competitors"] if c["homeAway"] == "away")

    return (
        home["team"]["displayName"],
        int(home["score"]),
        away["team"]["displayName"],
        int(away["score"]),
    )


# ===============================
# Main command
# ===============================

def match_event_by_id(game_id: str, events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    gid = str(game_id).strip()
    if not gid:
        return None
    for ev in events:
        if str(ev.get("id", "")).strip() == gid:
            return ev
    return None

def is_high_underdog_ml_from_snapshot(
    snaps: pd.DataFrame,
    sport: str,
    game_id: str,
    side: str,
    threshold: int = 300
) -> bool:
    gid = str(game_id).strip()
    if not gid:
        return False

    sub = snaps[
        (snaps["sport"].astype(str).str.strip() == str(sport).strip()) &
        (snaps["game_id"].astype(str).str.strip() == gid) &
        (snaps["market"].astype(str).str.upper() == "MONEYLINE") &
        (snaps["side"].astype(str).str.lower() == str(side).strip().lower())
    ]
    if sub.empty:
        return False

    tmp = sub.copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce", utc=True)
    tmp = tmp.dropna(subset=["timestamp"]).sort_values("timestamp")
    if tmp.empty:
        return False

    line = tmp.iloc[-1].get("current_line")
    if line is None or (isinstance(line, float) and pd.isna(line)):
        return False

    try:
        odds = int(str(line).strip())
    except Exception:
        return False

    return odds >= threshold

def build_finals_table(sports: List[str], since: str, days: int = 7) -> pd.DataFrame:
    base_day = pd.Timestamp(since)
    rows = []

    for sport in sports:
        for offset in range(days):
            day = (base_day + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            ymd = ny_to_yyyymmdd(day)
            events = fetch_espn_events(sport, ymd)

            for ev in events:
                final = extract_final(ev)
                if not final:
                    continue
                home, hs, away, as_ = final

                # ESPN shortName often looks like "Away Team at Home Team"
                name = ev.get("name") or ev.get("shortName") or ""

                rows.append({
                    "sport": sport,
                    "ymd": ymd,
                    "event_name": name,
                    "home": home,
                    "away": away,
                    "home_score": hs,
                    "away_score": as_,
                })

    return pd.DataFrame(rows)


def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_game_away_home(game: str) -> Optional[tuple[str, str]]:
    # supports "Away @ Home" and "Away at Home"
    g = (game or "").replace(" @ ", " at ").replace("@", " at ")
    parts = g.split(" at ")
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()

ABBR = {
    # common city abbrevs (covers your examples)
    "bos": "boston",
    "edm": "edmonton",
    "car": "carolina",
    "tb": "tampa bay",
    "no": "new orleans",
    "chi": "chicago",
    "la": "los angeles",
    "ny": "new york",
    "okc": "oklahoma city",
    "sa": "san antonio",
    "gs": "golden state",
    "phi": "philadelphia",
    "por": "portland",
    "den": "denver",
    "mil": "milwaukee",
    "min": "minnesota",
    "cle": "cleveland",
    "orl": "orlando",
    "ind": "indiana",
    "atl": "atlanta",
    "was": "washington",
    "tor": "toronto",
    "uta": "utah",
    "mem": "memphis",
    "cha": "charlotte",
    "det": "detroit",
    "phx": "phoenix",
    "sac": "sacramento",
    "nj": "new jersey",
    "bos": "boston",
    "sf": "san francisco",
    "sa": "san antonio",


}

def expand_abbr_words(s: str) -> str:
    # Replace leading tokens like "BOS", "TB", "EDM", "NO", etc.
    toks = re.findall(r"[A-Za-z0-9]+", s or "")
    out = []
    for t in toks:
        low = t.lower()
        if low in ABBR:
            out.extend(ABBR[low].split())
        else:
            out.append(low)
    return " ".join(out)

def norm_tokens(s: str) -> set[str]:
    s = expand_abbr_words(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # drop very short tokens that cause noise
    toks = {t for t in s.split(" ") if len(t) >= 3}
    return toks

def parse_game_away_home(game: str) -> Optional[tuple[str, str]]:
    g = (game or "").replace(" @ ", " at ").replace("@", " at ")
    parts = g.split(" at ")
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()

def best_final_match_for_signal(
    fg: pd.DataFrame,
    away_g: str,
    home_g: str
) -> Optional[pd.Series]:
    """
    Baseline-safe matcher.
    Always returns the best candidate in fg (never None if fg not empty).
    """

    if fg is None or fg.empty:
        return None

    away_t = norm_tokens(away_g)
    home_t = norm_tokens(home_g)

    best = None
    best_score = -1  # <-- IMPORTANT: allow 0-score matches

    for _, r in fg.iterrows():
        away_f = norm_tokens(str(r.get("away", "")))
        home_f = norm_tokens(str(r.get("home", "")))

        score = len(away_t & away_f) + len(home_t & home_f)

        if score > best_score:
            best_score = score
            best = r

    # 🚨 BASELINE RULE:
    # If we got here, fg was non-empty — always return something
    return best




def parse_total_line(side: str) -> Optional[Tuple[str, float]]:
    """
    "Over 229.5" -> ("OVER", 229.5)
    "Under 6.5"  -> ("UNDER", 6.5)
    """
    s = (side or "").strip().lower()
    if s.startswith("over"):
        m = re.search(r"([0-9]+(\.[0-9]+)?)", s)
        return ("OVER", float(m.group(1))) if m else None
    if s.startswith("under"):
        m = re.search(r"([0-9]+(\.[0-9]+)?)", s)
        return ("UNDER", float(m.group(1))) if m else None
    return None


def parse_spread_team_and_line(side: str) -> Optional[Tuple[str, float]]:
    """
    "NJ Devils +1.5" -> ("NJ Devils", +1.5)
    "BOS Bruins -1.5" -> ("BOS Bruins", -1.5)
    """
    s = (side or "").strip()
    m = re.search(r"([+-]\d+(\.\d+)?)\s*$", s)
    if not m:
        return None
    line = float(m.group(1))
    team = s[:m.start()].strip()
    return (team, line) if team else None


def match_side_to_home_away(side_team: str, home: str, away: str) -> Optional[str]:
    """
    Token-based match of a side's team string to ESPN home/away team names.
    Returns "HOME", "AWAY", or None.
    Uses your ABBR expansion via norm_tokens().
    """
    st = norm_tokens(side_team)
    if not st:
        return None

    ht = norm_tokens(home)
    at = norm_tokens(away)

    home_score = len(st & ht)
    away_score = len(st & at)

    if home_score == 0 and away_score == 0:
        return None

    return "HOME" if home_score >= away_score else "AWAY"


    # Require a minimum score to avoid bad matches
    return best if best_score >= 1 else None


def cmd_backfill_baseline(args):
    since = args.since
    label = args.label or f"HISTORICAL_BACKFILL_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    force = args.force

    if os.path.exists(SENTINEL_PATH) and not force:
        raise SystemExit("Backfill already completed. Sentinel exists.")

    # --- Load signals
    sig = pd.read_csv(SIGNALS_PATH)

    ts_col = "logged_at_utc"
    if ts_col not in sig.columns:
        raise SystemExit(f"Expected column '{ts_col}' in signals_baseline.csv. Found: {list(sig.columns)}")

    sig[ts_col] = pd.to_datetime(sig[ts_col], errors="coerce")
    sig = sig.dropna(subset=[ts_col])

    # Normalize timestamps to UTC (mixed UTC/ET safe)
    def normalize_to_utc(ts):
        if ts.tzinfo is None:
            return ts.tz_localize("America/New_York").tz_convert("UTC")
        return ts.tz_convert("UTC")

    sig["_ts_utc"] = sig[ts_col].apply(normalize_to_utc)

    since_utc = pd.Timestamp(since).tz_localize("America/New_York").tz_convert("UTC")
    sig = sig[sig["_ts_utc"] >= since_utc].copy()
    snaps = pd.read_csv(SNAPSHOTS_PATH)
    # --- STEP 3: Build local finals table ONCE (no per-signal ESPN calls)
    sports = sorted(sig["sport"].astype(str).str.strip().unique().tolist())

    finals_df = build_finals_table(sports, since, days=7)
    finals_df.to_csv(FINALS_PATH, index=False)

    print(f"[backfill] finals table built: {len(finals_df)} rows -> {FINALS_PATH}")



    resolved = []
    events_cache = {}
    total = len(sig)

    print(f"[backfill] starting: {total} signals since={since}")

    base_day = pd.Timestamp(since)

    for i, (_, row) in enumerate(sig.iterrows(), start=1):
        if i % 25 == 0:
            print(f"[backfill] progress: {i}/{total} signals processed")

        sport = str(row.get("sport", "")).strip()
        game = str(row.get("game", "")).strip()
        market = str(row.get("market", "")).strip()
        side = str(row.get("side", "")).strip()
        color = str(row.get("color", "")).strip()
        game_id = str(row.get("game_id", "")).strip()

        # --- resolve via local finals table (NO per-signal ESPN calls)
        parsed = parse_game_away_home(game)
        if not parsed:
            continue
        away_g, home_g = parsed

        fg = finals_df[finals_df["sport"].astype(str).str.strip() == sport]
        if fg.empty:
            continue

        best = best_final_match_for_signal(fg, away_g, home_g)
        if best is None:
            continue

        home = str(best["home"])
        away = str(best["away"])
        hs = int(best["home_score"])
        as_ = int(best["away_score"])
        ymd = str(best["ymd"])

        # ---------------------------
        # Outcome resolution (ML + TOTAL + SPREAD)
        # ---------------------------
        outcome = "UNRESOLVED_MARKET"
        mkt = market.upper().strip()

        if mkt == "MONEYLINE":
            which = match_side_to_home_away(side, home, away)
            if which is None:
                outcome = "UNRESOLVED_TEAM"
            else:
                if hs == as_:
                    outcome = "PUSH"
                else:
                    home_won = hs > as_
                    picked_home = (which == "HOME")
                    outcome = "WIN" if (picked_home == home_won) else "LOSS"

        elif mkt == "TOTAL":
            parsed_total = parse_total_line(side)
            if not parsed_total:
                outcome = "UNRESOLVED_MARKET"
            else:
                ou, line = parsed_total
                total_pts = hs + as_
                if abs(total_pts - line) < 1e-9:
                    outcome = "PUSH"
                elif ou == "OVER":
                    outcome = "WIN" if total_pts > line else "LOSS"
                else:  # UNDER
                    outcome = "WIN" if total_pts < line else "LOSS"

        elif mkt == "SPREAD":
            parsed_spread = parse_spread_team_and_line(side)
            if not parsed_spread:
                outcome = "UNRESOLVED_MARKET"
            else:
                team_part, line = parsed_spread
                which = match_side_to_home_away(team_part, home, away)
                if which is None:
                    outcome = "UNRESOLVED_TEAM"
                else:
                    team_score = hs if which == "HOME" else as_
                    opp_score = as_ if which == "HOME" else hs
                    adjusted = team_score + line

                    if abs(adjusted - opp_score) < 1e-9:
                        outcome = "PUSH"
                    elif adjusted > opp_score:
                        outcome = "WIN"
                    else:
                        outcome = "LOSS"

        else:
            outcome = "UNRESOLVED_MARKET"

        resolved.append({
            "run_label": label,
            "mode": "HISTORICAL_BACKFILL",
            "logged_at_raw": str(row[ts_col]),
            "logged_at_utc": sig.loc[row.name, "_ts_utc"].isoformat(),
            "since_ny": since,
            "sport": sport,
            "game_id": row.get("game_id"),
            "game": game,
            "game_key": stable_game_key(sport, game),
            "market": market,
            "side": side,
            "color": color,
            "is_dark_green_high_underdog": (
                (color.lower() == "dark green")
                and (market.upper() == "MONEYLINE")
                and is_high_underdog_ml_from_snapshot(
                    snaps, sport, row.get("game_id"), side, threshold=300
                )
            ),
            "espn_day": ymd,
            "final_home": home,
            "final_home_score": hs,
            "final_away": away,
            "final_away_score": as_,
            "outcome": outcome,
        })

    # --- write outputs (ONE TIME, after loop)
    df = pd.DataFrame(resolved)
    df.to_csv(OUT_RESOLVED_PATH, index=False)

    if df.empty:
        summary = pd.DataFrame(
            columns=["color", "is_dark_green_high_underdog", "outcome", "count"]
        )
    else:
        summary = (
            df.groupby(
                ["color", "is_dark_green_high_underdog", "outcome"],
                dropna=False
            )
            .size()
            .reset_index(name="count")
        )

    summary.to_csv(OUT_SUMMARY_PATH, index=False)

    with open(SENTINEL_PATH, "w", encoding="utf-8") as f:
        f.write(label)

    print("Historical backfill complete.")
    print(f"Signals considered: {len(sig)}")
    print(f"Resolved rows written: {len(df)}")
    print(f"Locked with sentinel: {SENTINEL_PATH}")

def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--since", required=True, help="NY date YYYY-MM-DD (no backfill beyond this)")
    p.add_argument("--label", default=None, help="run_label")
    p.add_argument("--force", action="store_true", help="ignore sentinel and re-run once")
    args = p.parse_args()

    cmd_backfill_baseline(args)

if __name__ == "__main__":
    main()


