import argparse
import csv

# --- BASELINE LOG CACHE ---
_BASELINE_SEEN_KEYS = None  # set[(sport, game_id, market, side, bucket)]

import datetime as dt
import os
import re

from pathlib import Path

OPEN_REG_PATH = Path("data") / "open_registry.csv"

def _load_open_registry() -> dict:
    """
    Key: (sport, game_id, market, side) -> open_line string
    Persistent across runs so Open is stable even when lines move.
    """
    reg = {}
    if not OPEN_REG_PATH.exists():
        return reg
    import csv
    with OPEN_REG_PATH.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            k = (row.get("sport",""), row.get("game_id",""), row.get("market",""), row.get("side",""))
            reg[k] = (row.get("open_line","") or "").strip()
    return reg

def _save_open_registry(reg: dict) -> None:
    import csv
    OPEN_REG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OPEN_REG_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sport","game_id","market","side","open_line"])
        for (sport, game_id, market, side), open_line in sorted(reg.items()):
            w.writerow([sport, game_id, market, side, open_line])
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dk_headless import get_splits
from logging_utils import setup_logger
import logging
logger = logging.getLogger("dk")
import re
import pandas as pd
from movement import movement_report
import json
import urllib.request
import csv
import os
from datetime import datetime, timezone, timedelta
from backfill_baseline import cmd_backfill_baseline
import gzip
from urllib.error import HTTPError, URLError

from datetime import datetime, timezone, timedelta
def _parse_iso_dt(s: str):
    """
    Parse an ISO datetime string safely.
    Accepts: '2026-01-02T19:00:00Z' or '2026-01-02T19:00:00+00:00'
    Returns timezone-aware datetime in UTC, or None.
    """
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def compute_minutes_to_kickoff(row: dict):
    """
    Uses existing ESPN kickoff field already on the row.
    IMPORTANT: We are NOT touching snapshot timestamps.
    """
    kickoff_iso = (
        row.get("espn_kickoff_iso")
        or row.get("espn_kickoff")
        or row.get("game_time_iso")
    )
    dt = _parse_iso_dt(kickoff_iso) if kickoff_iso else None
    if not dt:
        return None
    now_utc = datetime.now(timezone.utc)
    return int((dt - now_utc).total_seconds() // 60)

def compute_timing_bucket(sport: str, minutes_to_kickoff):
    """
    v1.1 timing buckets (minute-only helper):
      - > 480 min: EARLY
      -  60..480: MID
      -   0..60:  LATE
      - negative: LIVE
    NOTE: game-day anchoring is handled in the dashboard timing block.
    """
    try:
        if minutes_to_kickoff is None:
            return "UNKNOWN"
        m2k = int(minutes_to_kickoff)
    except Exception:
        return "UNKNOWN"

    if m2k < 0:
        return "LIVE"
    if m2k > 480:
        return "EARLY"
    if m2k > 60:
        return "MID"
    return "LATE"


    sport = (sport or "").lower()
    # Keep these conservative; can tune later without touching timestamps.
    if sport in ("nfl", "ncaaf"):
        early, mid = 24*60, 6*60
    else:
        early



BASELINE_FILE = "data/signals_baseline.csv"


import re

# ---------------- TEAM / GAME NORMALIZATION HELPERS ----------------

import re

# Super forgiving split: supports "Away @ Home", "Away vs Home", "Away v Home"
def _split_game(game) -> tuple[str, str]:
    # Defensive: snapshots.csv may contain NaN/float/None for game
    try:
        import pandas as pd
        if pd.isna(game):
            return "", ""
    except Exception:
        pass

    g = str(game).strip() if game is not None else ""
    if not g or g.lower() == "nan":
        return "", ""

    # normalize separators
    g = re.sub(r"\s+vs\.?\s+|\s+v\.?\s+", " @ ", g, flags=re.IGNORECASE)
    if " @ " in g:
        a, h = g.split(" @ ", 1)
        return a.strip(), h.strip()
    return "", ""

    g = str(game).strip() if game is not None else ""
    if not g or g.lower() == "nan":
        return "", ""

    # normalize separators
    g = re.sub(r"\s+vs\.?\s+|\s+v\.?\s+", " @ ", g, flags=re.IGNORECASE)
    if " @ " in g:
        a, h = g.split(" @ ", 1)
        return a.strip(), h.strip()
    return "", ""

def _norm_team(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s&.-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Main alias map (DK name -> ESPN shortDisplayName style)
# Add more over time without breaking anything.
# Main alias map (DK name -> ESPN shortDisplayName style)
# Single source of truth. All keys MUST be lowercase.

# ---- Step C (metrics instrumentation only) ----
LOGIC_VERSION = "v1.1"   # tag every ledger/state write; official tracking may start at v1.1 later

# ============================================================
# v1.1 STEP 2 — SPORT-SPECIFIC DAMPENERS (INSTRUMENTATION ONLY)
# ============================================================

# NOTE:
# - NO score math changes
# - NO threshold changes
# - Flags only (explanatory / gating)

# --- NCAAB ---
NCAAB_EARLY_STRONG_BLOCK = True          # early window cannot certify STRONG
NCAAB_STRONG_MIN_PERSIST = 3             # ≥3 consecutive snapshots ≥72
NCAAB_STRONG_STABILITY_DELTA = 2          # last ≥ peak − 2
NCAAB_LATE_STRONG_BLOCK = True            # never certify STRONG late
NCAAB_REQUIRE_MULTI_MARKET = True         # single-market dependency blocks STRONG

# --- NCAAF ---
NCAAF_EARLY_INSTANT_STRONG_BLOCK = True   # no instant STRONG early
NCAAF_STRONG_STABILITY_DELTA = 3
NCAAF_LATE_NEW_STRONG_BLOCK = True        # late can hold, not create

# --- GLOBAL ---
PUBLIC_DRIFT_BLOCKS_STRONG = True
FAST_SNAP_BLOCKS_STRONG = True


TEAM_ALIASES = {
    # Miami variations
    "miami fl": "miami",
    "miami (fl)": "miami",
    "miami florida": "miami",

    # State abbreviations
    "arizona state": "arizona st",
    "arizona st.": "arizona st",
    "penn state": "penn st",
    "oklahoma state": "oklahoma st",
    "ohio state": "ohio st",
    "kansas state": "kansas st",
    "iowa state": "iowa st",
    "florida state": "florida st",
    "mississippi state": "mississippi st",
    "louisiana state": "lsu",

    # Directional schools
    "southern miss": "southern miss",
    "central michigan": "central mich",
    "eastern michigan": "eastern mich",
    "western michigan": "western mich",

    # NCAAB problem teams (DK -> ESPN-ish)
    "boston university": "boston u",
    "saint peters": "st peters",
    "siu edwardsville": "siue",
    "iu indianapolis": "iu indy",
    "cal state fullerton": "cs fullerton",
    "cal st fullerton": "cs fullerton",
    "queens nc": "queens",
    "queens charlotte": "queens",
    "east texas am": "tx am commerce",
    "texas am commerce": "tx am commerce",

    # NBA shorthand (DK-style city prefixes)
    "no pelicans": "new orleans pelicans",
    "la clippers": "los angeles clippers",
    "ny knicks": "new york knicks",

    # ESPN shortDisplayName quirks
    "pittsburgh": "pitt",
    "western michigan": "w mich",
    "albany": "ualbany",
    "albany ny": "ualbany",
}



def normalize_team_name(name: str) -> str:
    n = _norm_team(name)
    # apply aliases on normalized keys
    return TEAM_ALIASES.get(n, n)

def lookup_time_from_game(game: str, kickoff_map: dict[str, str]) -> str:
    if not game:
        return ""

    a_raw, h_raw = _split_game(game)
    if not a_raw or not h_raw:
        return ""

    away = normalize_team_name(a_raw)
    home = normalize_team_name(h_raw)

    # kickoff_map keys are expected to look like "away @ home" in normalized form
    key = f"{away} @ {home}"
    return kickoff_map.get(key, "")

# -------------------------------------------------------------------


def espn_cfb_kickoff_map(games: list[str]) -> dict[str, str]:
    """
    Returns { 'Away @ Home': ISO datetime } for CFB games.
    Free ESPN public scoreboard endpoint.
    """
    url = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"

    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}



    out = {}
    for ev in data.get("events", []):
        iso = ev.get("date", "")
        comps = ev.get("competitions", [])
        if not comps:
            continue

        competitors = comps[0].get("competitors", [])
        if len(competitors) != 2:
            continue

        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        home_name = home.get("team", {}).get("shortDisplayName")
        away_name = away.get("team", {}).get("shortDisplayName")
        if not home_name or not away_name:
            continue

        away_norm = normalize_team_name(away_name)
        home_norm = normalize_team_name(home_name)

        away_norm = TEAM_ALIASES.get(away_norm, away_norm)
        home_norm = TEAM_ALIASES.get(home_norm, home_norm)

        key = f"{away_norm} @ {home_norm}"
        out[key] = iso


    # map DK game strings -> ESPN-normalized keys
    result = {}
    for g in games:
        if " @ " not in g:
            result[g] = ""
            continue
        a, h = g.split(" @ ", 1)
        result[g] = out.get(f"{normalize_team_name(a)} @ {normalize_team_name(h)}", "")

    return result
# ESPN scoreboard base URLs by sport
# Adding a new ESPN-supported sport should be ONE LINE here
ESPN_SCOREBOARD_BASE = {
    "nfl":   "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
    "nba":   "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "ncaaf": "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard",
    "nhl":   "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
    "mlb":   "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
    "ncaab": "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard",
    # ufc intentionally omitted (event-based, not scoreboard-based)
}

def _norm_game_key(s: str) -> str:
    """Normalize game strings for fuzzy matching (DK vs ESPN)."""
    if s is None:
        return ""
    s = str(s).lower().strip()
    # standardize separators
    s = s.replace(" vs. ", " @ ").replace(" vs ", " @ ").replace(" v ", " @ ")
    # remove punctuation
    s = re.sub(r"[^a-z0-9@ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # common city/abbr normalizations (NFL-heavy but harmless elsewhere)
    rep = {
        "ny giants": "new york giants",
        "ny jets": "new york jets",
        "la rams": "los angeles rams",
        "la chargers": "los angeles chargers",
        "sf 49ers": "san francisco 49ers",
        "tb buccaneers": "tampa bay buccaneers",
        "kc chiefs": "kansas city chiefs",
        "no saints": "new orleans saints",
        "ne patriots": "new england patriots",
        "gb packers": "green bay packers",
        "lv raiders": "las vegas raiders",
        "dal cowboys": "dallas cowboys",
        "cle browns": "cleveland browns",
        "cin bengals": "cincinnati bengals",
        "den broncos": "denver broncos",
        "atl falcons": "atlanta falcons",
        "no saints": "new orleans saints",

    }
    for a, b in rep.items():
        s = s.replace(a, b)

    return s

def _espn_kickoff_map_date_range(scoreboard_url_base: str, games: list[str], days: int = 5) -> dict[str, str]:
    """
    Returns DK-game-keyed kickoff ISO map by querying ESPN scoreboard across a date range.
    Robust matching across NFL/NBA/NHL/CFB/CBB/MLB.
    """
    import json
    import urllib.request
    from datetime import datetime, timedelta

    def _safe(x):
        return (x or "").strip()

    def _norm_team(x: str) -> str:
        """
        Normalize ESPN team strings for matching:
        - strip
        - expand a few common leading abbreviations
        - apply your _normalize_team_name (which handles DK-style quirks too)
        """
        s = _safe(x)

        s = _normalize_team_name(s)  # applies TEAM_ALIASES (and any DK quirks)
        return s

        try:
            return _normalize_team_name(s)
        except Exception:
            return s.strip()



    def _split_game(g: str):
        g = _safe(g)
        if " @ " in g:
            a, h = g.split(" @ ", 1)
            return _safe(a), _safe(h)
        if " vs " in g:
            h, a = g.split(" vs ", 1)
            return _safe(a), _safe(h)
        return "", ""

    # Build ESPN index: many key variants -> iso
    espn_index: dict[str, str] = {}



    # Include recent past because DK "n7days" often includes already-played games

    start = datetime.now() - timedelta(days=7)
    for i in range(days + 1):
        d = start + timedelta(days=i)
        ymd = d.strftime("%Y%m%d")

        # Sport-specific params (CBB + CFB need groups + bigger limits)
        extra = ""
        if "mens-college-basketball" in scoreboard_url_base:
            extra = "&groups=50&limit=500"
        elif "football/college-football" in scoreboard_url_base:
            extra = "&groups=80&limit=500"
        else:
            # avoid silent truncation on some scoreboards
            extra = "&limit=500"

        url = f"{scoreboard_url_base}?dates={ymd}{extra}"

                # --- ESPN fetch (instrumented; do not swallow errors) ---
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json,text/plain,*/*",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            },
        )

        print(f"[espn] GET {url}")

        try:
            resp = urllib.request.urlopen(req, timeout=20)
            status = getattr(resp, "status", None) or resp.getcode()
            raw = resp.read()
            enc = (resp.headers.get("Content-Encoding") or "").lower()
            ctype = resp.headers.get("Content-Type") or ""

            if "gzip" in enc:
                raw = gzip.decompress(raw)

            print(f"[espn] status={status} content_type={ctype} body_len={len(raw)} enc={enc}")

            text = raw.decode("utf-8", errors="replace")
            data = json.loads(text)

            if isinstance(data, dict):
                print(f"[espn] top_keys={list(data.keys())[:20]}")
            else:
                print(f"[espn] top_type={type(data)}")

        except (HTTPError, URLError, json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[espn] ERROR {type(e).__name__}: {e}")
            raise
        # --- end ESPN fetch ---


        for ev in data.get("events", []):
            iso = _safe(ev.get("date", ""))
            if not iso:
                continue

            comps = ev.get("competitions", [])
            if not comps:
                continue

            competitors = comps[0].get("competitors", [])
            if len(competitors) != 2:
                continue

            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue

            ht = (home.get("team") or {})
            at = (away.get("team") or {})
            home_name = _norm_team(ht.get("shortDisplayName") or ht.get("displayName") or ht.get("name") or "")
            away_name = _norm_team(at.get("shortDisplayName") or at.get("displayName") or at.get("name") or "")

            # Canonical ESPN key
            espn_game = f"{away_name} @ {home_name}".strip()

            # Index multiple variants + normalized version
            if espn_game:
                espn_index[espn_game] = iso
                espn_index[_norm_game_key(espn_game)] = iso



            # multiple ESPN name fields
            pairs = [
                (_safe(at.get("shortDisplayName")), _safe(ht.get("shortDisplayName"))),
                (_safe(at.get("displayName")), _safe(ht.get("displayName"))),
                (_safe(at.get("abbreviation")), _safe(ht.get("abbreviation"))),
            ]

            for a, h in pairs:
                if not a or not h:
                    continue




                # raw
                espn_index.setdefault(f"{a} @ {h}", iso)

                # normalized
                an = _norm_team(a)
                hn = _norm_team(h)
                if an and hn:
                    espn_index.setdefault(f"{an} @ {hn}", iso)
                    espn_index.setdefault(_norm_game_key(f"{an} @ {hn}"), iso)


    # Resolve DK games -> iso
    result: dict[str, str] = {}
    for g in games:
        g = _safe(g)
        iso = ""

        candidates: list[str] = []

        # 1) raw DK
        candidates.append(g)

        # 1b) normalized-away/home DK key (fixes "Albany NY", "Miami FL", etc.)
        away0, home0 = _split_game(g)
        if away0 and home0:
            candidates.append(f"{_normalize_team_name(away0)} @ {_normalize_team_name(home0)}")

        # 2) DK->ESPN (your existing key transform)
        try:
            candidates.append(_dk_game_to_espn_key(g))
        except Exception:
            pass

        # 3) normalized DK->ESPN split using the local normalizer (kept for safety)
        a0, h0 = _split_game(g)
        if a0 and h0:
            candidates.append(f"{_norm_team(a0)} @ {_norm_team(h0)}")


        # Try exact + normalized candidate keys first
        iso = ""
        for c in candidates:
            if not c:
                continue
            iso = espn_index.get(c, "") or espn_index.get(_norm_game_key(c), "")
            if iso:
                break

        # Fuzzy fallback (LAST RESORT): token overlap against ESPN matchup keys.
        if not iso:
            ng = _norm_game_key(g)
            if "@" in ng:
                gtoks = set(t for t in ng.replace("@", " ").split() if len(t) >= 3)
                best_iso = ""
                best_score = 0

                for k, v in espn_index.items():
                    if not v:
                        continue
                    nk = _norm_game_key(k)
                    if "@" not in nk:
                        continue
                    ktoks = set(t for t in nk.replace("@", " ").split() if len(t) >= 3)
                    score = len(gtoks & ktoks)
                    if score > best_score:
                        best_score = score
                        best_iso = v

                if best_score >= 2:
                    iso = best_iso

        result[g] = iso

    return result

import re

def get_espn_kickoff_map(sport: str, games: list[str]) -> dict[str, str]:
    """
    Generic ESPN kickoff resolver.
    Returns DK-game-keyed kickoff ISO map.
    Safe no-op for unsupported sports.
    """
    base = ESPN_SCOREBOARD_BASE.get(sport)
    print(f"[espn debug] sport={sport} base={base!r} games={len(games)}")
    if not base or not games:
        return {}

    try:
        # DK "n7days" spans past week; also want ~2-3 weeks ahead
        km = _espn_kickoff_map_date_range(base, games, days=5)
        return km if isinstance(km, dict) else {}

    except Exception as e:
        print(f"[espn] kickoff fetch failed for {sport}: {type(e).__name__}: {e}")
        raise


def _espn_finals_map_date_range(scoreboard_url_base: str, games: list[str], days: int = 5) -> dict[str, tuple[int, int]]:
    """
    Returns DK-game-keyed finals map: "Away @ Home" -> (away_score, home_score)
    Only includes completed/final games.
    """
    import json
    import urllib.request
    from datetime import datetime, timedelta

    def _safe(x):
        return (x or "").strip()

    def _norm_team(x: str) -> str:
        try:
            return _normalize_team_name(_safe(x))
        except Exception:
            return _safe(x)

    def _split_game(g: str):
        g = _safe(g)
        if " @ " in g:
            a, h = g.split(" @ ", 1)
            return _safe(a), _safe(h)
        if " vs " in g:
            h, a = g.split(" vs ", 1)
            return _safe(a), _safe(h)
        return "", ""

    want_keys = set()
    for g in games or []:
        a, h = _split_game(g)
        if a and h:
            want_keys.add(f"{_norm_team(a)} @ {_norm_team(h)}")

    finals: dict[str, tuple[int, int]] = {}

    start = datetime.now() - timedelta(days=7)  # include recent past
    for i in range(days + 1):
        d = start + timedelta(days=i)
        ymd = d.strftime("%Y%m%d")

        url = f"{scoreboard_url_base}&dates={ymd}"
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            continue

        for ev in (data.get("events") or []):
            comps = ev.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            status = (comp.get("status") or {}).get("type") or {}
            if not status.get("completed"):
                continue

            competitors = comp.get("competitors") or []
            if len(competitors) < 2:
                continue

            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            if not away or not home:
                continue

            away_name = _norm_team(((away.get("team") or {}).get("shortDisplayName") or (away.get("team") or {}).get("displayName") or ""))
            home_name = _norm_team(((home.get("team") or {}).get("shortDisplayName") or (home.get("team") or {}).get("displayName") or ""))

            try:
                away_score = int((away.get("score") or "0").strip())
                home_score = int((home.get("score") or "0").strip())
            except Exception:
                continue

            key = f"{away_name} @ {home_name}"
            if (not want_keys) or (key in want_keys):
                finals[key] = (away_score, home_score)

    return finals


def get_espn_finals_map(sport: str, games: list[str]) -> dict[str, tuple[int, int]]:
    """
    Generic ESPN final-score resolver.
    Returns DK-game-keyed final score map: "Away @ Home" -> (away_score, home_score)
    """
    base = ESPN_SCOREBOARD_BASE.get(sport)
    if not base or not games:
        return {}
    try:
        return _espn_finals_map_date_range(base, games, days=5)
    except Exception as e:
        print(f"[espn] finals fetch failed for {sport}: {e}")
        return {}


def update_snapshots_with_espn_finals():
    """
    Updates data/snapshots.csv with final_score_for / final_score_against
    for TEAM rows only (moneyline/spread rows where side looks like a team name).
    Safe no-op if no finals found.
    """
    import pandas as pd
    import os

    src = "data/snapshots.csv"
    if not os.path.exists(src):
        return

    try:
        df = pd.read_csv(src, keep_default_na=False, dtype=str)
    except Exception as e:
        print(f"[finals] read failed: {e}")
        return

    if df.empty or "sport" not in df.columns or "game" not in df.columns or "side" not in df.columns:
        return

    # ensure columns exist (string-safe; do NOT use pd.NA here)
    if "final_score_for" not in df.columns:
        df["final_score_for"] = ""
    if "final_score_against" not in df.columns:
        df["final_score_against"] = ""

    # Normalize these columns so blanks count as missing (robust vs NA parsing)
    df["final_score_for"] = df["final_score_for"].fillna("").astype(str).str.strip()
    df["final_score_against"] = df["final_score_against"].fillna("").astype(str).str.strip()

    # only try to fill rows missing finals (blank-safe)
    need = df[(df["final_score_for"] == "") | (df["final_score_against"] == "")]

    if need.empty:
        return

    # build finals maps per sport for the games present
    finals_by_sport = {}
    for sport, gdf in need.groupby("sport"):
        games = sorted(set(gdf["game"].dropna().astype(str).tolist()))
        if not games:
            continue
        finals_by_sport[sport] = get_espn_finals_map(sport, games)

    if not finals_by_sport:
        return

    def _split_game(g: str):
        # Defensive: allow NaN/float/None
        try:
            import pandas as pd
            if pd.isna(g):
                return "", ""
        except Exception:
            pass
    
        s = str(g).strip() if g is not None else ""
        if not s or s.lower() == "nan":
            return "", ""
        if " @ " in s:
            a, h = s.split(" @ ", 1)
            return a.strip(), h.strip()
        if " vs " in s:
            h, a = s.split(" vs ", 1)
            return a.strip(), h.strip()
        return "", ""
    
    def _norm(x: str) -> str:
        try:
            return _normalize_team_name((x or "").strip())
        except Exception:
            return (x or "").strip()

    # fill finals for TEAM rows (side matches away/home team)
    updated = 0
    for idx, row in df.iterrows():
        fsf = str(row.get("final_score_for") or "").strip()
        fsa = str(row.get("final_score_against") or "").strip()
        if fsf != "" and fsa != "":
            continue

        sport = row.get("sport")
        game = row.get("game")
        side = row.get("side")

        if not sport or not game or not side:
            continue

        a_raw, h_raw = _split_game(game)
        if not a_raw or not h_raw:
            continue

        away = _norm(a_raw)
        home = _norm(h_raw)
        key = f"{away} @ {home}"

        finals_map = finals_by_sport.get(sport) or {}
        if key not in finals_map:
            continue

        away_score, home_score = finals_map[key]
        side_norm = _norm(side)

        # only set for team-side rows
        if side_norm == away:
            df.at[idx, "final_score_for"] = away_score
            df.at[idx, "final_score_against"] = home_score
            updated += 1
        elif side_norm == home:
            df.at[idx, "final_score_for"] = home_score
            df.at[idx, "final_score_against"] = away_score
            updated += 1

    if updated > 0:
        df.to_csv(src, index=False)
        print(f"[finals] updated {updated} rows in {src}")


# ---- Step C (metrics instrumentation only): row_state + signal_ledger ----
def _metrics_now_iso_utc() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _metrics_blank(x) -> str:
    try:
        if x is None:
            return ""
        s = str(x).strip()
        if s.lower() in ("nan", "none", "null"):
            return ""
        return s
    except Exception:
        return ""


def _metrics_float(x, default=None):
    s = _metrics_blank(x)
    if s == "":
        return default
    try:
        return float(s)
    except Exception:
        return default


def _metrics_key(row) -> tuple[str, str, str, str]:
    # Keyed by (sport, game_id, market, side) as locked
    sport = _metrics_blank(row.get("sport"))
    game_id = _metrics_blank(row.get("game_id"))
    # Prefer report-level market_display (SPREAD/TOTAL/MONEYLINE) if present.
    # Fallback to inferred _market_display (metrics may create it), else raw snapshots market (usually "splits").
    market = _metrics_blank(row.get("market_display"))
    if not market:
        market = _metrics_blank(row.get("_market_display"))
    if not market:
        market = _metrics_blank(row.get("market"))
    side = _metrics_blank(row.get("side"))
    return (sport, game_id, market, side)


def _score_bucket(score: float) -> str:
    # Instrumentation-only bucket (does NOT affect scoring logic)
    # Buckets:
    #   NO_BET < 60
    #   LEAN   60?65
    #   BET    66?71
    #   STRONG_BET >= 72
    try:
        s = float(score)
    except Exception:
        return "NO_BET"
    if s >= 72.0:
        return "STRONG_BET"
    if s >= 66.0:
        return "BET"
    if s >= 60.0:
        return "LEAN"
    return "NO_BET"
def _load_row_state(path: str):
    import pandas as pd
    import os
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "sport", "game_id", "market", "side",
            "logic_version",
            "last_score", "last_ts",
            "peak_score", "peak_ts",
            "last_bucket",
            "last_net_edge", "last_net_edge_ts",
        ])
    try:
        df = pd.read_csv(path, keep_default_na=False, dtype=str)
        # Normalize instrumentation fields (avoid blank strong_streak from older rows)
        try:
            if "strong_streak" in df.columns:
                df["strong_streak"] = df["strong_streak"].astype(str).str.strip()
                df.loc[df["strong_streak"] == "", "strong_streak"] = "0"
        except Exception:
            pass

        if df.empty:
            return pd.DataFrame(columns=[
                "sport", "game_id", "market", "side",
                "logic_version",
                "last_score", "last_ts",
                "peak_score", "peak_ts",
                "last_bucket",
                "last_net_edge", "last_net_edge_ts",
            ])
        # ensure cols exist
        for c in ["sport","game_id","market","side","logic_version","last_score","last_ts","peak_score","peak_ts","last_bucket","last_net_edge","last_net_edge_ts","strong_streak"]:
            if c not in df.columns:
                df[c] = ""
        return df
    except Exception:
        # if state file is corrupted, fail safe (do not break report)
        return pd.DataFrame(columns=[
            "sport", "game_id", "market", "side",
            "logic_version",
            "last_score", "last_ts",
            "peak_score", "peak_ts",
            "last_bucket",
            "last_net_edge", "last_net_edge_ts",
        ])


def _append_signal_ledger(path: str, rows: list[dict]):
    import pandas as pd
    import os

    # Canonical ledger schema (stable across runs)
    cols = [
        "ts","logic_version","event","from_bucket","to_bucket",
        "sport","game_id","market","side","game",
        "current_line","current_odds","bets_pct","money_pct",
        "score","net_edge"
    ]

    # Ensure directory exists
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Ensure file exists AND has required columns (even if it already exists)
    try:
        if os.path.exists(path):
            df0 = pd.read_csv(path, keep_default_na=False, dtype=str)
            if df0.empty:
                pd.DataFrame(columns=cols).to_csv(path, index=False)
            else:
                for c in cols:
                    if c not in df0.columns:
                        df0[c] = ""
                extra_cols = [c for c in df0.columns if c not in cols]
                df0 = df0[cols + extra_cols]
                df0.to_csv(path, index=False)
        else:
            pd.DataFrame(columns=cols).to_csv(path, index=False)
    except Exception:
        # Never break report due to ledger hygiene
        return

    # Nothing to append: schema is now guaranteed
    if not rows:
        return

    # Append rows in canonical schema (extras allowed but canonical columns guaranteed)
    try:
        out = pd.DataFrame(rows)
        for c in cols:
            if c not in out.columns:
                out[c] = ""
        for c in out.columns:
            out[c] = out[c].fillna("").astype(str)
        extra_cols = [c for c in out.columns if c not in cols]
        out = out[cols + extra_cols]
        out.to_csv(path, index=False, mode="a", header=False)
    except Exception:
        return


def update_row_state_and_signal_ledger(latest):
    """
    Instrumentation-only.
    Writes:
      - data/row_state.csv: last_score, peak_score, peak_ts (plus last_ts/last_bucket)
      - data/signal_ledger.csv: bucket crossings w/ context + logic_version
    Key: (sport, game_id, market, side)
    """
    try:
        import os
        import pandas as pd

        if latest is None or len(latest) == 0:
            return

        # --- Metrics persistence key (DO NOT USE market_display for state keys) ---
        # Persisted key schema MUST remain: sport | game_id | market | side
        # We store the key label in latest['_metrics_market'] and always persist it under state field 'market'.
        #
        # Priority:
        #   1) raw 'market' if it already looks like a real market (SPREAD/TOTAL/MONEYLINE)
        #   2) else 'market_display' if present
        #   3) else infer from current_line (when raw market is only 'splits' or blank)
        def _infer_market_from_line(v):
            try:
                d = parse_line_and_odds(str(v))
                mt = (d.get("market_type") or "").lower()
                if mt == "spread":
                    return "SPREAD"
                if mt == "total":
                    return "TOTAL"
                if mt == "moneyline":
                    return "MONEYLINE"
                return "OTHER"
            except Exception:
                return "OTHER"

        if "market" in latest.columns:
            mv = latest.get("market", "").fillna("").astype(str).str.strip()
            mv_u = mv.str.upper()
            looks_real = mv_u.isin(["SPREAD", "TOTAL", "MONEYLINE"])
            if looks_real.any():
                latest["_metrics_market"] = mv_u.where(looks_real, "")
            else:
                latest["_metrics_market"] = ""
        else:
            latest["_metrics_market"] = ""

        if "_metrics_market" not in latest.columns:
            latest["_metrics_market"] = ""

        # If still blank, fall back to market_display
        if "market_display" in latest.columns:
            md = latest.get("market_display", "").fillna("").astype(str).str.strip().str.upper()
            latest["_metrics_market"] = latest["_metrics_market"].mask(latest["_metrics_market"].astype(str).str.strip() == "", md)

        # If still blank OR raw market is only 'splits', infer from current_line
        mm = latest["_metrics_market"].fillna("").astype(str).str.strip()
        if (mm == "").any():
            inferred = latest.get("current_line", "").fillna("").astype(str).apply(_infer_market_from_line)
            latest["_metrics_market"] = latest["_metrics_market"].mask(mm == "", inferred)

        market_col = "_metrics_market"

        # Robust score column selection (some builds use 'score', others 'model_score')
        score_col = "score" if "score" in latest.columns else ("model_score" if "model_score" in latest.columns else "")
        if not score_col:
            return

        if os.environ.get("METRICS_DEBUG","") == "1":
            try:
                tmp_s = latest.get(score_col, "").fillna("").astype(str).str.strip()
                nonblank = int((tmp_s != "").sum())
                print(f"[metrics debug] score_col={score_col} nonblank_scores={nonblank}/{len(latest)} sample={list(tmp_s.head(3))}")
            except Exception as e:
                print(f"[metrics debug] score debug failed: {e}")

        if not market_col:
            return


        for req in ("sport", "game_id", "side"):
            if req not in latest.columns:
                return

        # Require score; if missing, do nothing safely
        score_col = "confidence_score" if "confidence_score" in latest.columns else ("model_score" if "model_score" in latest.columns else "")
        if not score_col:
            return

        os.makedirs("data", exist_ok=True)
        state_path = os.path.join("data", "row_state.csv")
        ledger_path = os.path.join("data", "signal_ledger.csv")
        # Ensure ledger file exists even if there are no crossings this run (prevents FileNotFoundError)
        _append_signal_ledger(ledger_path, [])

        state_df = _load_row_state(state_path)

        # Build state index (PERSISTED row_state schema: sport, game_id, market, side)
        if state_df is not None and not state_df.empty:
            for c in ("sport","game_id","market","side"):
                if c not in state_df.columns:
                    state_df[c] = ""
            idx_cols = ["sport","game_id","market","side"]
            state_df["_k"] = state_df[idx_cols].fillna("").astype(str).agg("|".join, axis=1)
        else:
            state_df = None

        state_map = {}
        if state_df is not None and not state_df.empty:
            for _, r in state_df.iterrows():
                k = "|".join([
                    _metrics_blank(r.get("sport")),
                    _metrics_blank(r.get("game_id")),
                    _metrics_blank(r.get("market")),
                    _metrics_blank(r.get("side")),
                ])
                state_map[k] = r.to_dict()

        now_ts = _metrics_now_iso_utc()
        ledger_rows = []
        processed = 0
        if os.environ.get('METRICS_DEBUG','') == '1':
            print(f"[metrics debug] now_ts={now_ts}")


        # --- precompute net_edge per (sport, game_id, market) from latest rows (instrumentation only)
        # Net Edge = max(score) - min(score) across sides within the same market.
        # IMPORTANT: normalize keys exactly like the per-row loop uses.
        edge_map = {}
        try:
            tmp = latest.copy()
            tmp["_sport_k"] = tmp.get("sport", "").apply(_metrics_blank)
            tmp["_gid_k"]   = tmp.get("game_id", "").apply(_metrics_blank)
            tmp["_mkt_k"]   = tmp.get(market_col, "").apply(_metrics_blank)
            tmp["_score_k"] = tmp[score_col].apply(lambda v: _metrics_float(v, default=None))

            tmp = tmp[tmp["_score_k"].notna()]
            if not tmp.empty:
                g = (
                    tmp.groupby(["_sport_k","_gid_k","_mkt_k"])["_score_k"]
                       .agg(["max","min"])
                       .reset_index()
                )
                g["net_edge"] = (g["max"] - g["min"]).astype(float)

                for _, rr in g.iterrows():
                    k = (str(rr["_sport_k"]), str(rr["_gid_k"]), str(rr["_mkt_k"]))
                    edge_map[k] = float(rr["net_edge"])
        except Exception:
            edge_map = {}

        # Iterate latest rows (instrumentation only)
        for _, r in latest.iterrows():
            sport = _metrics_blank(r.get('sport'))
            game_id = _metrics_blank(r.get('game_id'))
            market = _metrics_blank(r.get('_metrics_market') if '_metrics_market' in getattr(latest,'columns',[]) else r.get(market_col)).upper()
            side = _metrics_blank(r.get('side'))

            if sport == "" or game_id == "" or market == "" or side == "":
                continue

            k = f"{sport}|{game_id}|{market}|{side}"

            score = _metrics_float(r.get(score_col), default=None)
            # net_edge from precomputed market edge_map (fallback 0.0)
            net_edge = _metrics_float(r.get('net_edge'), default=None)
            if net_edge is None:
                net_edge = float(edge_map.get((sport, game_id, market), 0.0) or 0.0)

            if score is None:
                continue
            processed += 1

            bucket = _score_bucket(score)

            # --- v1.1 STRONG streak (instrumentation only) ---
            # strong_streak = consecutive runs where score >= 72
            prev = state_map.get(k, {})
            prev_streak = 0
            try:
                prev_streak = int(str(prev.get("strong_streak","0")).strip() or "0")
            except Exception:
                prev_streak = 0
            try:
                strong_now = (str(bucket).strip().upper() == "STRONG_BET")
            except Exception:
                strong_now = False
            strong_streak = str((prev_streak + 1) if strong_now else 0)
            # --- end v1.1 ---

            prev = state_map.get(k, {})
            prev_bucket = _metrics_blank(prev.get("last_bucket"))

            # peak tracking
            prev_peak = _metrics_float(prev.get("peak_score"), default=None)
            peak_score = score if (prev_peak is None or score > prev_peak) else prev_peak
            peak_ts = now_ts if (prev_peak is None or score > prev_peak) else _metrics_blank(prev.get("peak_ts"))

            # threshold crossing logging
            # Only log upward crossings into LEAN/BET/STRONG_BET (including brand-new rows)
            rank = {"": 0, "NO_BET": 0, "LEAN": 1, "BET": 2, "STRONG_BET": 3}
            pb = prev_bucket if prev_bucket in rank else "NO_BET"
            cb = bucket if bucket in ("NO_BET", "LEAN", "BET", "STRONG_BET") else "NO_BET"
            if rank.get(pb, 0) < rank.get(cb, 0) and cb in ("LEAN", "BET", "STRONG_BET"):
                ledger_rows.append({
                    "ts": now_ts,
                    "logic_version": LOGIC_VERSION,
                    "event": "THRESHOLD_CROSS",
                    "from_bucket": pb,
                    "to_bucket": cb,
                    "sport": sport,
                    "game_id": game_id,
                    "market": market,
                    "side": side,
                    "game": _metrics_blank(r.get("game")),
                    "current_line": _metrics_blank(r.get("current_line")),
                    "current_odds": _metrics_blank(r.get("current_odds")),
                    "bets_pct": _metrics_blank(r.get("bets_pct")),
                    "money_pct": _metrics_blank(r.get("money_pct")),
                    "score": f"{score:.2f}",
                    "net_edge": f"{(net_edge or 0.0):.2f}",
                })

            # upsert state
            state_map[k] = {
                "sport": sport,
                "game_id": game_id,
                "market": market,
                "side": side,
                "logic_version": LOGIC_VERSION,
                "last_score": f"{score:.2f}",
                "last_ts": now_ts,
                "last_net_edge": (f"{_metrics_float(net_edge, default=0.0):.2f}" if str(net_edge).strip() != "" else ""),
                "last_net_edge_ts": (now_ts if str(net_edge).strip() != "" else ""),

                "peak_score": f"{peak_score:.2f}",
                "peak_ts": peak_ts,
                "last_bucket": bucket,
                "strong_streak": str(strong_streak),
            }

        if os.environ.get('METRICS_DEBUG','') == '1':
            print(f"[metrics debug] processed_rows={processed} state_rows={len(state_map)} ledger_rows_this_run={len(ledger_rows)}")
        # Write state (full rewrite for simplicity + safety)
        out_rows = list(state_map.values())
        if out_rows:
            out = pd.DataFrame(out_rows)
            for c in out.columns:
                out[c] = out[c].fillna("").astype(str)
            out = out.sort_values(["sport", "game_id", "market", "side"], kind="mergesort")
            out.to_csv(state_path, index=False)

        _append_signal_ledger(ledger_path, ledger_rows)
    except Exception as e:
        # never break report; but show traceback when debugging
        try:
            import os, traceback
            if os.environ.get('METRICS_DEBUG','') == '1':
                print(f"[metrics debug] EXCEPTION in update_row_state_and_signal_ledger: {repr(e)}")
                print(traceback.format_exc())
        except Exception:
            pass
        return

# ---- end Step C metrics helpers ----

# ---- kickoff wrappers (unchanged behavior) ----
def espn_nfl_kickoff_map(games: list[str]) -> dict[str, str]:
    return get_espn_kickoff_map("nfl", games)

def espn_nba_kickoff_map(games: list[str]) -> dict[str, str]:
    return get_espn_kickoff_map("nba", games)




def _normalize_team_name(s: str) -> str:
    """
    Normalize DK team strings to match ESPN naming.
    Goal: make DK 'Away @ Home' keys line up with ESPN scoreboard keys.
    """
    if not s:
        return ""


    # normalize casing + spacing FIRST
    k = str(s).strip().lower()
    k = re.sub(r"\s+", " ", k)

    # one alias table only (TEAM_ALIASES)
    return TEAM_ALIASES.get(k, k)

def _dk_game_to_espn_key(game: str) -> str:
    """
    Convert:
      'BOS Celtics @ UTA Jazz' -> 'Celtics @ Jazz'
      'DEN Broncos @ LA Chargers' -> 'Broncos @ Chargers'
    """
    if not game or " @ " not in game:
        return game or ""

    away, home = game.split(" @ ", 1)
    away_n = _normalize_team_name(away)
    home_n = _normalize_team_name(home)

    return f"{away_n} @ {home_n}"









SPORT_CONFIG = {
    "nfl": {
        "label": "NFL",
        "url": "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/?tb_eg=88808&tb_edate=n7days&tb_emt=0",
    },
    "nba": {
        "label": "NBA",
        "url": "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/?tb_eg=42648&tb_edate=n7days&tb_emt=0",
    },
    "mlb": {
        "label": "MLB",
        "url": "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/?tb_eg=84240&tb_edate=n7days&tb_emt=0",
    },
    "nhl": {
        "label": "NHL",
        "url": "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/?tb_eg=42133&tb_edate=n7days&tb_emt=0",
    },
    "ncaaf": {
        "label": "CFB",
        "url": "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/?tb_eg=87637&tb_edate=n7days&tb_emt=0",
    },
    "ncaab": {
        "label": "NCAAB",
        "url": "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/?tb_eg=92483&tb_edate=n7days&tb_emt=0",
    },
    "ufc": {
        "label": "UFC",
        "url": "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/?tb_eg=9034&tb_edate=n7days&tb_emt=0",
    },
}




def parse_line_and_odds(s: str):
    """
    Input examples:
      "ATL Falcons @ +280"
      "LA Rams -7 @ -120"
      "Under 48.5 @ -110"
    Returns dict with:
      market_type: 'moneyline' | 'spread' | 'total' | 'other'
      line_val: float|None
      odds: int|None
    """
    if not isinstance(s, str) or not s.strip():
        return {"market_type": "other", "line_val": None, "odds": None}

    txt = s.strip()

    # odds at end: "@ -110" or "@ +280"
    m_odds = re.search(r"@\s*([+-]?\d+)\s*$", txt)
    odds = int(m_odds.group(1)) if m_odds else None

    # totals: Over/Under N
    m_total = re.search(r"^(Over|Under)\s+(\d+(\.\d+)?)", txt, flags=re.I)
    if m_total:
        return {"market_type": "total", "line_val": float(m_total.group(2)), "odds": odds}

    # spreads: team ... +/- number
    m_spread = re.search(r"\s([+-]\d+(\.\d+)?)\b", txt)
    if m_spread:
        return {"market_type": "spread", "line_val": float(m_spread.group(1)), "odds": odds}

    # moneyline: has odds but no spread/total number
    if odds is not None:
        return {"market_type": "moneyline", "line_val": None, "odds": odds}

    return {"market_type": "other", "line_val": None, "odds": None}

# =========================
# Market Read (Observation Mode, additive only)
# =========================

KEY_NUMBERS = {3, 7, 10, 14, 17}

def _crossed_key(prev_line, now_line) -> str:
    try:
        if prev_line is None or now_line is None:
            return ""
        a = float(prev_line)
        b = float(now_line)
        lo, hi = sorted([a, b])
        for k in sorted(KEY_NUMBERS):
            if lo < k < hi or abs(a - k) < 1e-9 or abs(b - k) < 1e-9:
                return str(k)
    except Exception:
        return ""
    return ""

def _toward_side_by_odds(open_odds, cur_odds) -> int:
    """
    Returns +1 if price moved AGAINST bettor on this side (more expensive),
            -1 if moved in favor (cheaper),
             0 if unknown/no move.
    Works for American odds where:
      -120 -> -140 is more expensive (toward side)
      +150 -> +130 is more expensive (toward side)
    """
    try:
        if open_odds is None or cur_odds is None:
            return 0
        o = int(open_odds)
        c = int(cur_odds)
        if o == c:
            return 0
        # both negative: more negative is more expensive
        if o < 0 and c < 0:
            return +1 if c < o else -1
        # both positive: smaller is more expensive
        if o > 0 and c > 0:
            return +1 if c < o else -1
        return 0
    except Exception:
        return 0

def _toward_side_by_spread(open_line, cur_line) -> int:
    """
    For spread rows, direction toward this side means line becomes more extreme:
      -7 -> -8 (toward favorite side)
      +7 -> +8 (toward dog side)
    """
    try:
        if open_line is None or cur_line is None:
            return 0
        o = float(open_line)
        c = float(cur_line)
        if o == c:
            return 0
        if o < 0 and c < 0:
            return +1 if c < o else -1
        if o > 0 and c > 0:
            return +1 if c > o else -1
        return 0
    except Exception:
        return 0

def _toward_side_by_total(open_total, cur_total, side_label: str) -> int:
    """
    Totals:
      Over benefits when total goes DOWN (better number), but market moving UP is "toward Over"
      Under is opposite.
    We define "toward side" as market moving in the direction that makes that side harder to bet:
      Over: total increasing is toward Over
      Under: total decreasing is toward Under
    """
    try:
        if open_total is None or cur_total is None:
            return 0
        o = float(open_total)
        c = float(cur_total)
        if o == c:
            return 0
        s = (side_label or "").lower()
        if "over" in s:
            return +1 if c > o else -1
        if "under" in s:
            return +1 if c < o else -1
        return 0
    except Exception:
        return 0

def _toward_side_by_juice(open_odds, cur_odds) -> int:
    """
    +1 if this side became more expensive (juice moved toward this side),
    -1 if it became cheaper, 0 if unchanged/unknown.
    """
    try:
        if open_odds is None or cur_odds is None:
            return 0
        o = int(open_odds)
        c = int(cur_odds)
        if o == c:
            return 0

        # Negative odds: more negative = more expensive
        if o < 0 and c < 0:
            return +1 if c < o else -1

        # Positive odds: smaller positive = more expensive
        if o > 0 and c > 0:
            return +1 if c < o else -1

        return 0
    except Exception:
        return 0

def _classify_market_read(D, bets_pct, move_dir, meaningful_move, is_high_bet_side):
    """
    Frozen v0.1 labels (tune later; not now).
    """
    # Public Drift: high-bet side AND market moved toward it (never green)
    if is_high_bet_side and move_dir == +1 and meaningful_move:
        return "Public Drift"

    # Stealth Move: strong D+, low bets, and move toward side
    if D >= 12 and (bets_pct is not None and bets_pct <= 40) and move_dir == +1 and meaningful_move:
        return "Stealth Move"

    # Aligned Sharp: D+ and move toward side
    if D >= 8 and move_dir == +1 and meaningful_move:
        return "Aligned Sharp"

        # Reverse Pressure: D+ but market moved away (stronger than freeze)
    if D >= 8 and move_dir == -1 and meaningful_move:
        return "Reverse Pressure"

    # Freeze Pressure: D+ but no meaningful move toward the side
    if D >= 8 and (move_dir == 0 or not meaningful_move):
        return "Freeze Pressure"
    
    # Contradiction: D- but market still moved toward side
    if D <= -8 and move_dir == +1 and meaningful_move:
        return "Contradiction"

    return "Neutral"

def add_market_read_to_latest(latest: pd.DataFrame) -> pd.DataFrame:
    """
    Additive only. Uses columns you already compute in build_dashboard():
      market_display, side, bets_pct, money_pct,
      open_odds/current_odds, open_line_val/current_line_val,
      odds_move_open/line_move_open
    """
    df = latest.copy()

    # D
    def _D(r):
        try:
            if pd.isna(r.get("money_pct")) or pd.isna(r.get("bets_pct")):
                return 0.0
            return float(r["money_pct"]) - float(r["bets_pct"])
        except Exception:
            return 0.0

    df["divergence_D"] = df.apply(_D, axis=1)

    # Pairing: determine which side is "high-bet" within each (sport, game_id, market_display)
    grp_cols = ["sport", "game_id", "market_display"]
    if "espn_day" in df.columns:
        grp_cols.append("espn_day")

    df["_pair_key"] = df[grp_cols].astype(str).agg("|".join, axis=1)

    # high bet by pair (2-sided assumption; safe fallback)
    high_bet_idx = set()
    for k, g in df.groupby("_pair_key"):
        try:
            if len(g) != 2:
                continue
            g2 = g.copy()
            g2["bets_pct_num"] = pd.to_numeric(g2["bets_pct"], errors="coerce")
            if g2["bets_pct_num"].isna().any():
                continue
            winner = g2["bets_pct_num"].idxmax()
            high_bet_idx.add(winner)
        except Exception:
            pass

    # Movement + label + why
    mr = []
    favors = []
    why = []
    move_summary = []

    for idx, r in df.iterrows():
        mkt = str(r.get("market_display", "")).upper()
        side = str(r.get("side", "")).strip()
        bets = pd.to_numeric(r.get("bets_pct"), errors="coerce")
        money = pd.to_numeric(r.get("money_pct"), errors="coerce")
        D = float(r.get("divergence_D", 0.0))

        is_high_bet = idx in high_bet_idx

        move_dir = 0
        key_cross = ""
        meaningful = False
        # Pull deltas if they exist (already computed earlier in build_dashboard)
        od_open = r.get("odds_move_open", None)
        ln_open = r.get("line_move_open", None)

        # Normalize numeric
        try:
            od_open_num = None if pd.isna(od_open) else float(od_open)
        except Exception:
            od_open_num = None

        try:
            ln_open_num = None if pd.isna(ln_open) else float(ln_open)
        except Exception:
            ln_open_num = None


        if mkt == "MONEYLINE":
            move_dir = _toward_side_by_odds(r.get("open_odds"), r.get("current_odds"))
            od = r.get("odds_move_open")
            meaningful = (pd.notna(od) and abs(float(od)) >= 10)

            move_summary.append(f"ML: {r.get('open_odds')}â†’{r.get('current_odds')} (Î”odds={od})")

        elif mkt == "SPREAD":
            move_dir = _toward_side_by_spread(r.get("open_line_val"), r.get("current_line_val"))
            if move_dir == 0 and pd.notna(od) and abs(float(od)) >= 10:
                move_dir = _toward_side_by_juice(r.get("open_odds"), r.get("current_odds"))

            ld = r.get("line_move_open")
            od = r.get("odds_move_open")
            key_cross = _crossed_key(abs(r.get("open_line_val")) if pd.notna(r.get("open_line_val")) else None,
                                     abs(r.get("current_line_val")) if pd.notna(r.get("current_line_val")) else None)
            meaningful = (
                (pd.notna(ld) and abs(float(ld)) >= 0.5) or
                (pd.notna(od) and abs(float(od)) >= 10) or
                (key_cross != "")
            )

            move_summary.append(
                f"SPREAD: {r.get('open_line_val')}â†’{r.get('current_line_val')} (Î”line={ld}), "
                f"odds {r.get('open_odds')}â†’{r.get('current_odds')} (Î”odds={od})"
                + (f", key={key_cross}" if key_cross else "")
            )

        elif mkt == "TOTAL":
            move_dir = _toward_side_by_total(r.get("open_line_val"), r.get("current_line_val"), side)
            ld = r.get("line_move_open")
            od = r.get("odds_move_open")
            meaningful = (pd.notna(ld) and abs(float(ld)) >= 0.5) or (pd.notna(od) and abs(float(od)) >= 10)

            move_summary.append(
                f"TOTAL: {r.get('open_line_val')}â†’{r.get('current_line_val')} (Î”num={ld}), "
                f"odds {r.get('open_odds')}â†’{r.get('current_odds')} (Î”odds={od})"
            )
        else:
            move_summary.append("Unknown market move")

        label = _classify_market_read(D, None if pd.isna(bets) else float(bets), move_dir, meaningful, is_high_bet)

        mr.append(label)
        favors.append(side if side else "this side")

        # explain string (clean, decision-intel only)
        btxt = "n/a" if pd.isna(bets) else f"{float(bets):.1f}%"
        mtxt = "n/a" if pd.isna(money) else f"{float(money):.1f}%"
        why.append(
            f"{label}: bets={btxt}, money={mtxt}, D={D:+.1f}; "
            f"move_dir={move_dir}"
            + (f", key={key_cross}" if key_cross else "")
            + f". {move_summary[-1]}"
        )

    df["move_open_to_current"] = move_summary
    df["market_read"] = mr
    df["market_favors"] = favors
    df["market_why"] = why

    df.drop(columns=["_pair_key"], inplace=True, errors="ignore")
    return df

def add_market_pair_checks(latest: pd.DataFrame) -> pd.DataFrame:
    df = latest.copy()

    pair_cols = ["sport", "game_id", "market_display"]
    if "espn_day" in df.columns:
        pair_cols.append("espn_day")

    sharp_pressure = {"Aligned Sharp", "Stealth Move", "Freeze Pressure", "Reverse Pressure", "Contradiction"}
    # Optional debug mode to validate wiring (does not affect default behavior)
    import os as _os
    _pair_mode = (_os.environ.get('PAIR_CHECK_MODE','') or '').strip().upper()
    if _pair_mode == 'LOOSE':
        sharp_pressure = sharp_pressure.union({'Public Drift'})
    df["market_pair_check"] = ""

    # Debug only when PAIR_CHECK_MODE is set (never noisy by default)
    try:
        import os as _os
        _dbg_pair = (_os.environ.get('PAIR_CHECK_MODE','') or '').strip()
    except Exception:
        _dbg_pair = ''

    for _, g in df.groupby(pair_cols):
        if len(g) != 2:
            continue

        idx = list(g.index)
        a = str(df.loc[idx[0], "market_read"] or "")
        b = str(df.loc[idx[1], "market_read"] or "")

        a_sp = (a in sharp_pressure)
        b_sp = (b in sharp_pressure)
        # Flag rare pairing anomaly: both sides sharp-pressure
        if a_sp and b_sp:
            df.loc[idx[0], "market_pair_check"] = "PAIR_CHECK: both sides sharp-pressure"
            df.loc[idx[1], "market_pair_check"] = "PAIR_CHECK: both sides sharp-pressure"


    # debug summary (only when PAIR_CHECK_MODE is set)
    try:
        if _dbg_pair:
            nflag = (df['market_pair_check'].fillna('').astype(str).str.strip() != '').sum()
            print(f"[pair_check debug] mode={_dbg_pair} flagged_rows={int(nflag)}")
    except Exception:
        pass

    return df


# =========================
# Settings (edit these)

# Toggle noisy dashboard diagnostics
DASH_DEBUG = False
# =========================

# Put the DK Network betting splits URL you want to scrape here.
# Example format might be a DK Network page that shows splits for NFL/NBA.
SPLITS_URL = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"

# Where we store historical snapshots
DATA_DIR = "data"
SNAPSHOT_CSV = os.path.join(DATA_DIR, "snapshots.csv")

# Where we output the dashboard
REPORT_HTML = os.path.join(DATA_DIR, "dashboard.html")

# Use a polite user-agent
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36"
}

TIMEZONE = "local"  # for display only


# =========================
# Thresholds (your â€œrare dark greenâ€ setup)
# =========================

@dataclass(frozen=True)
class Thresholds:
    # Money vs Bets
    light_money_min: int = 65
    light_bets_max: int = 45

    dark_money_min: int = 70
    dark_bets_max: int = 40

    # Public pressure
    yellow_public_bets_min: int = 70
    red_public_bets_min: int = 75
    red_money_max: int = 55

    # Line movement thresholds (optional, if you capture open/current)
    meaningful_move_pts: float = 1.5  # used for â€œstrong line behaviorâ€

    # â€œNo movementâ€ / resistance trigger (optional)
    resistance_public_bets_min: int = 72
    resistance_move_max: float = 0.25

TH = Thresholds()


# =========================
# Data model
# =========================

@dataclass
class SideRow:
    sport: str
    game_id: str
    game: str
    side: str  # team name or side label
    market: str  # spread / total / moneyline (if available)
    bets_pct: Optional[int]
    money_pct: Optional[int]

    # Optional line fields
    open_line: Optional[str] = None
    current_line: Optional[str] = None

    # Optional context
    injury_news: Optional[str] = None  # "yes"/"no"/None
    key_number_note: Optional[str] = None  # any note


# =========================
# Utilities
# =========================

def now_iso() -> str:
    return dt.datetime.now().replace(microsecond=0).isoformat()


def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def fetch_html(url: str) -> str:
    if not url or "PASTE_" in url:
        raise ValueError("You must set SPLITS_URL at the top of main.py")

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def pct_to_int(x: str) -> Optional[int]:
    if x is None:
        return None
    m = re.search(r"(\d{1,3})\s*%", str(x))
    if not m:
        return None
    v = int(m.group(1))
    if 0 <= v <= 100:
        return v
    return None


def safe_text(el) -> str:
    return el.get_text(" ", strip=True) if el else ""


# =========================
# Parsing (the only part we may need to tweak)
# =========================

def parse_splits_generic(html: str, sport: str, debug: bool = False) -> List[SideRow]:
    """
    Generic parser:
    - Finds repeating â€œgame cardsâ€ or table rows
    - Extracts game name, side/team labels, bets% and money% if present
    This may need minor tweaks depending on DK Network page structure.
    """
    soup = BeautifulSoup(html, "lxml")

    # Try tables first
    rows_out: List[SideRow] = []

    # --- Strategy A: parse table rows that contain percentages ---
    table_rows = soup.select("tr")
    if debug:
        print(f"[debug] found {len(table_rows)} <tr> rows")

    for idx, tr in enumerate(table_rows):
        txt = safe_text(tr)
        if "%" not in txt:
            continue

        # Heuristic: look for two percentages in the row
        pcts = re.findall(r"(\d{1,3})\s*%", txt)
        if len(pcts) < 2:
            continue

        # Try to find team/side labels inside row
        tds = tr.select("td")
        if len(tds) < 2:
            continue

        # Very loose assumptions:
        # - One cell may contain game/team info
        # - other cells contain bets% / money%
        # We will attempt to pull two sides if we see two team names.
        if debug and idx < 10:
            print(f"[debug] row {idx}: {txt[:160]}...")

        # Try to locate something that looks like "Team A vs Team B"
        game_guess = txt
        game_guess = re.sub(r"\s+", " ", game_guess).strip()

        # Pull first 2 percentages as bets and money (may be per side; varies by site)
        # This is why we may need to tweak once we see DK Network structure.
        bets_pct = int(pcts[0]) if pcts else None
        money_pct = int(pcts[1]) if len(pcts) > 1 else None

        # Create a single â€œside rowâ€ record as a fallback
        # We'll show it in output even if itâ€™s not perfectly split by team yet.
        rows_out.append(SideRow(
            sport=sport,
            game_id=f"row-{idx}",
            game=game_guess,
            side="(unparsed side)",
            market="unknown",
            bets_pct=bets_pct,
            money_pct=money_pct
        ))

    # --- Strategy B: card-like blocks (common on modern pages) ---
    if not rows_out:
        cards = soup.select("[class*='card'], [class*='split'], [class*='game']")
        if debug:
            print(f"[debug] no table rows parsed; found {len(cards)} card-ish elements")

        for i, c in enumerate(cards):
            txt = safe_text(c)
            if "%" not in txt:
                continue
            pcts = re.findall(r"(\d{1,3})\s*%", txt)
            if len(pcts) < 2:
                continue

            rows_out.append(SideRow(
                sport=sport,
                game_id=f"card-{i}",
                game=txt[:140],
                side="(unparsed side)",
                market="unknown",
                bets_pct=int(pcts[0]),
                money_pct=int(pcts[1]),
            ))

    if debug:
        print(f"[debug] parsed {len(rows_out)} records (generic)")

    return rows_out


# =========================
# Color logic (per SIDE)
# =========================

def classify_side(
    bets_pct: Optional[int],
    money_pct: Optional[int],
    open_line: Optional[str] = None,
    current_line: Optional[str] = None,
    injury_news: Optional[str] = None,
    key_number_note: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Returns: (color, explanation)
    color âˆˆ {"DARK_GREEN","LIGHT_GREEN","GREY","YELLOW","RED"}
    """
    # If we don't have percentages, we can't score well
    if bets_pct is None or money_pct is None:
        return "GREY", "Missing bet%/money% â†’ default Grey"

    # Context modifiers
    has_news = (injury_news or "").strip().lower() in {"yes", "y", "true", "1"}

    # Basic money-vs-bets signals
    light_money_signal = (money_pct >= TH.light_money_min and bets_pct <= TH.light_bets_max)
    dark_money_signal = (money_pct >= TH.dark_money_min and bets_pct <= TH.dark_bets_max)

    # Public-heavy signals
    is_yellow = bets_pct >= TH.yellow_public_bets_min and money_pct < TH.light_money_min
    is_red = bets_pct >= TH.red_public_bets_min and money_pct <= TH.red_money_max

    # Optional: line behavior (if you store open/current)
    # We keep it conservative: without line info, we do NOT hand out Dark Green easily.
    strong_line_signal = False
    if open_line and current_line:
        # Try to parse numeric spread points from lines like "-3.5" or "+6"
        def parse_num(s: str) -> Optional[float]:
            m = re.search(r"([+-]?\d+(\.\d+)?)", s)
            return float(m.group(1)) if m else None

        o = parse_num(open_line)
        c = parse_num(current_line)
        if o is not None and c is not None:
            move = abs(c - o)
            if move >= TH.meaningful_move_pts:
                strong_line_signal = True

        # Key number note can promote â€œstrong line behaviorâ€ (NFL)
        if key_number_note and key_number_note.strip():
            strong_line_signal = True

    # DARK GREEN (rare): requires strong line behavior + strong money signal, AND no obvious news explanation
    if strong_line_signal and dark_money_signal and not has_news:
        return "DARK_GREEN", "Book behavior + strong money-vs-bets imbalance; no obvious news â†’ Market Edge Confirmed"

    # If news explains it, keep in Light Green even if strong
    if strong_line_signal and dark_money_signal and has_news:
        return "LIGHT_GREEN", "Strong signals but major news present â†’ downgrade to Market Edge Developing" 

    # LIGHT GREEN: money-vs-bets imbalance without strong line confirmation
    if light_money_signal:
        return "LIGHT_GREEN", "Money concentration vs bet count â†’ Market Edge Developing (watch for confirmation)"

    # RED: avoid this side
    if is_red:
        return "RED", "Extremely public + weak money support â†’ Wrong Side / Trap (evaluate opposite side)"

    # YELLOW: public-driven
    if is_yellow:
        return "YELLOW", "Public-heavy demand without strong money support â†’ Caution"

    return "GREY", "No strong market signal on this side"


# =========================
# Snapshot storage + report
# =========================

SNAPSHOT_FIELDS = [
    "timestamp", "sport", "game_id", "game", "side", "market",
    "bets_pct", "money_pct", "open_line", "current_line",
    "injury_news", "key_number_note", "dk_start_iso"

]

def append_snapshot(rows, sport: str):
    import csv
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).isoformat()

    with open(SNAPSHOT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SNAPSHOT_FIELDS)

        # ALWAYS write header if file is empty
        if f.tell() == 0:
            writer.writeheader()

        if rows:
            logger.debug("sample row keys=%s", list(rows[0].keys()))
            logger.debug(
                "sample game_id=%s game=%s",
                rows[0].get("game_id"),
                rows[0].get("game"),
            )

        for row in rows:
            writer.writerow({
                "timestamp": ts,
                "sport": sport,
                "game_id": row.get("game_id"),
                "game": row.get("game"),
                "side": row.get("side"),
                "market": row.get("market"),
                "bets_pct": row.get("bets_pct"),
                "money_pct": row.get("money_pct"),
                "open_line": row.get("open") or row.get("open_line"),
                "current_line": row.get("current") or row.get("current_line"),
                "injury_news": row.get("news"),
                "key_number_note": row.get("key_number_note"),
                "dk_start_iso": row.get("start_time_iso") or row.get("start_time") or row.get("startDate") or row.get("dk_start_iso") or row.get("game_time_iso") or row.get("game_time") or ""
            })


def infer_market_type(side_txt: str, line_txt: str) -> str:
    s = (side_txt or "").strip().lower()
    t = (str(line_txt) if line_txt is not None else "").strip()


    # TOTAL
    if s.startswith("over") or s.startswith("under"):
        return "TOTAL"

    # SPREAD (explicit line in side)
    if re.search(r"([+-]\d+(?:\.\d+)?)\b", side_txt or ""):
        return "SPREAD"

    # MONEYLINE (odds only)
    if re.search(r"@\s*[+-]\d{3,4}\b", t):
        return "MONEYLINE"

    return ""

def _color_rank(c):
    c = (c or "").upper()
    if c == "DARK_GREEN": return 3
    if c == "LIGHT_GREEN": return 2
    if c == "YELLOW": return 1
    return 0

def build_dashboard():
    ensure_data_dir()

    if not os.path.exists(SNAPSHOT_CSV):
        print(f"[warn] no snapshots yet: {SNAPSHOT_CSV}")
        return

    df = pd.read_csv(SNAPSHOT_CSV, keep_default_na=False, dtype=str)
    if DASH_DEBUG:
        print(f"[dash debug] rows after read_csv: {len(df)}")
    if "sport" in df.columns:
        if DASH_DEBUG:
            print(f"[dash debug] sports present: {sorted(df['sport'].dropna().astype(str).unique().tolist())}")
    else:
        if DASH_DEBUG:
            print("[dash debug] sports present: NO sport col")

    # Normalize column names for dashboard
    df = df.rename(columns={
        "open": "open_line",
        "current": "current_line",
        "news": "injury_news",
    })

    # Timestamps
    if "timestamp" not in df.columns:
        print("[warn] snapshots.csv missing timestamp column")
        return

    df["timestamp"] = df["timestamp"].astype(str).str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, format="mixed")
    bad = df[df["timestamp"].isna()]
    if DASH_DEBUG:
        print(f"[dash debug] bad timestamp rows: {len(bad)}")

    # If timestamp couldn't parse, drop it
    df = df.dropna(subset=["timestamp"]).copy()
    if DASH_DEBUG:
        print(f"[dash debug] rows after timestamp parse/dropna: {len(df)}")

    # Sport display labels (keep consistent with your existing mapping if present elsewhere)
    df["sport"] = df["sport"].astype(str).str.lower().str.strip()
    df["sport_label"] = df["sport"].str.upper()


    # Grouping keys must not be NaN
    df["game_id"] = df["game_id"].fillna("").astype(str)
    df["side"] = df["side"].fillna("").astype(str)
    df["market"] = df["market"].fillna("unknown").astype(str)

    # Clean DK â€œopens in a new tabâ€¦â€ junk if present
    df["market"] = df["market"].astype(str).str.replace(r"\s+opens in a new tab.*$", "", regex=True)
    df["game"] = df["game"].astype(str).str.replace(r"\s+opens in a new tab.*$", "", regex=True)

    # ---------- Stable keys for OPEN/LATEST/PREV (prevents "open resets" when line changes) ----------
    # market_display is consistent even if df["market"] varies (or contains alt labels).
    df["market_display"] = df.apply(
        lambda rr: infer_market_type(rr.get("side", ""), rr.get("current_line", "")),
        axis=1
    )

    # side_key makes a stable identifier per side within a game/market even when the numeric line moves.
    df["side_key"] = df["side"].astype(str)

    # For spreads, strip trailing spread number so:
    #   "Iowa +5.5" and "Iowa +3" both key to "Iowa"
    df.loc[df["market_display"] == "SPREAD", "side_key"] = (
        df.loc[df["market_display"] == "SPREAD", "side_key"]
          .str.replace(r"\s[+-]\d+(?:\.\d+)?\s*$", "", regex=True)
          .str.strip()
    )

    # For totals, normalize to just Over/Under so alternates don't create new keys
    # (the actual total number is in current_line_val and will be handled by main-line filter)
    df.loc[df["market_display"] == "TOTAL", "side_key"] = (
        df.loc[df["market_display"] == "TOTAL", "side_key"]
          .str.extract(r"^(Over|Under)", expand=False)
          .fillna(df.loc[df["market_display"] == "TOTAL", "side_key"])
          .str.strip()
    )

    # --- OPEN = first seen current_line for that (sport, game_id, market_display, side_key)
    df = df.sort_values("timestamp")
    first_seen = (
        df.groupby(["sport", "game_id", "market_display", "side_key"], as_index=False)
          .first()[["sport", "game_id", "market_display", "side_key", "current_line"]]
          .rename(columns={"current_line": "open_line"})
    )
    df = df.merge(
        first_seen,
        on=["sport", "game_id", "market_display", "side_key"],
        how="left",
        suffixes=("", "_first")
    )

    # FIX: the merge creates open_line_first (because df already has open_line).

    # Prefer persisted open_line (open_registry/snapshots). Only fill blanks from first_seen.

    if "open_line_first" in df.columns:

        ol = df["open_line"].fillna("").astype(str)

        mask = ol.str.len().eq(0)

        if mask.any():

                        # Ensure open_line is string dtype before assigning strings (avoids pandas FutureWarning)
            if "open_line" in df.columns:
                df["open_line"] = df["open_line"].fillna("").astype(str)

            df.loc[mask, "open_line"] = df.loc[mask, "open_line_first"].fillna("").astype(str)


    # --- LATEST row per selection
    latest = (
        df.groupby(["sport", "game_id", "market_display", "side_key"], as_index=False)
          .tail(1)
          .copy()
          .reset_index(drop=True)
    )
    if DASH_DEBUG:
        print(f"[dash debug] rows in latest: {len(latest)}")

    # ---- FILTER: today and future games only (NY time) ----
    today_ny = pd.Timestamp.now(tz="America/New_York").normalize()

    if "game_time_ny" in latest.columns:
        latest = latest[
            (latest["game_time_ny"].isna()) | (latest["game_time_ny"] >= today_ny)
        ]





    # --- PREV = previous snapshot row per selection (2nd to last)
    prev = (
        df.groupby(["sport", "game_id", "market_display", "side_key"], as_index=False)
          .nth(-2)
          .rename(columns={"current_line": "prev_current_line"})
          [["sport", "game_id", "market_display", "side_key", "prev_current_line"]]
    )
    latest = latest.merge(prev, on=["sport", "game_id", "market_display", "side_key"], how="left")
    # --------------------------------------------------------------------


    # Debug sample
    if len(latest) > 0:
        sample = latest.head(5)
        if DASH_DEBUG:
            print("[dash debug] inferred market types:")
        for _, rr in sample.iterrows():
            mt = infer_market_type(rr.get("side", ""), rr.get("current_line", ""))
            print("  ", rr.get("side", ""), "|", rr.get("current_line", ""), "->", mt)

    # Parse OPEN/CURRENT/PREV into numbers safely
    def _safe_parse(series):
        def _one(x):
            try:
                d = parse_line_and_odds("" if pd.isna(x) else str(x))
                if isinstance(d, dict):
                    return d
                return {"market_type": "other", "line_val": None, "odds": None}
            except Exception:
                return {"market_type": "other", "line_val": None, "odds": None}
        return series.apply(_one).apply(pd.Series)

    cur_parsed = _safe_parse(latest["current_line"])
    opn_parsed = _safe_parse(latest["open_line"])
    prv_parsed = _safe_parse(latest["prev_current_line"])

    latest["market_type"] = cur_parsed.get("market_type", "other")
    latest["current_odds"] = cur_parsed.get("odds", None)
    latest["current_line_val"] = cur_parsed.get("line_val", None)

    latest["open_odds"] = opn_parsed.get("odds", None)
    latest["open_line_val"] = opn_parsed.get("line_val", None)

    latest["prev_odds"] = prv_parsed.get("odds", None)
    latest["prev_line_val"] = prv_parsed.get("line_val", None)

    # ========= MAIN LINE FILTER (drop alternates; keep only 1 number per game for SPREAD + TOTAL) =========
    # DK often includes alternate spreads/totals (ex: +3, +5.5, O45.5, O46.5).
    # Deterministic rule based ONLY on scraped rows in *latest*:
    #   TOTAL: pick mode of total number per (sport, game_id); tie-breaker = closest to median, then smallest.
    #   SPREAD: pick mode of ABS(spread) per (sport, game_id); tie-breaker = closest to 0 (smallest abs).
    # Keep both sides (Over+Under / both teams) for the chosen main number.
    # MONEYLINE is kept as-is.

    # Ensure numeric
    latest["current_line_val"] = pd.to_numeric(latest["current_line_val"], errors="coerce")

    # --- TOTAL main number
    totals = latest[latest["market_display"] == "TOTAL"].copy()
    if len(totals) > 0:
        total_counts = (
            totals.groupby(["sport", "game_id", "current_line_val"], as_index=False)
                  .size()
                  .rename(columns={"size": "n"})
        )

        # Tie-breaker: closest to median total for that game (stable + data-driven)
        med = (
            totals.groupby(["sport", "game_id"], as_index=False)["current_line_val"]
                  .median()
                  .rename(columns={"current_line_val": "med_total"})
        )
        total_counts = total_counts.merge(med, on=["sport", "game_id"], how="left")
        total_counts["dist"] = (total_counts["current_line_val"] - total_counts["med_total"]).abs()

        total_main = (
            total_counts.sort_values(
                ["sport", "game_id", "n", "dist", "current_line_val"],
                ascending=[True, True, False, True, True]
            )
            .groupby(["sport", "game_id"], as_index=False)
            .head(1)[["sport", "game_id", "current_line_val"]]
            .rename(columns={"current_line_val": "main_total"})
        )

        totals = totals.merge(total_main, on=["sport", "game_id"], how="left")
        totals = totals[totals["current_line_val"] == totals["main_total"]].drop(columns=["main_total"])

    # --- SPREAD main number (mode of ABS(line))
    spreads = latest[latest["market_display"] == "SPREAD"].copy()
    if len(spreads) > 0:
        spreads["abs_line"] = spreads["current_line_val"].abs()

        spread_counts = (
            spreads.groupby(["sport", "game_id", "abs_line"], as_index=False)
                   .size()
                   .rename(columns={"size": "n"})
        )

        spread_main = (
            spread_counts.sort_values(
                ["sport", "game_id", "n", "abs_line"],
                ascending=[True, True, False, True]  # tie-breaker: closest to 0
            )
            .groupby(["sport", "game_id"], as_index=False)
            .head(1)[["sport", "game_id", "abs_line"]]
            .rename(columns={"abs_line": "main_abs_spread"})
        )

        spreads = spreads.merge(spread_main, on=["sport", "game_id"], how="left")
        spreads = spreads[spreads["abs_line"] == spreads["main_abs_spread"]].drop(columns=["abs_line", "main_abs_spread"])

    # MONEYLINE untouched; reassemble
    money = latest[latest["market_display"] == "MONEYLINE"].copy()
    other = latest[~latest["market_display"].isin(["TOTAL", "SPREAD", "MONEYLINE"])].copy()

    latest = pd.concat([money, totals, spreads, other], ignore_index=True)

    # Debug: how many rows did we keep?
    try:
        if DASH_DEBUG:
            print(f"[dash debug] after main-line filter: rows in latest = {len(latest)}")
        if DASH_DEBUG:
            print("[dash debug] after main-line filter: unique games =", latest["game_id"].nunique())
    except Exception:
        pass

    # ========= END MAIN LINE FILTER =========


    # Deltas
    latest["odds_move_open"] = latest["current_odds"] - latest["open_odds"]
    latest["line_move_open"] = latest["current_line_val"] - latest["open_line_val"]
    latest["odds_move_prev"] = latest["current_odds"] - latest["prev_odds"]
    latest["line_move_prev"] = latest["current_line_val"] - latest["prev_line_val"]

    # Market Read (Observation Mode, additive only)
    latest = add_market_read_to_latest(latest)
    latest = add_market_pair_checks(latest)

    # TEMP DEBUG: distribution + spread-only samples (remove after validation)
    if DASH_DEBUG:
        print(
            "[dash debug] market_read counts:",
            latest["market_read"].value_counts(dropna=False).head(10).to_dict()
        )
    if DASH_DEBUG:
        print(
            "[dash debug] pair_check counts:",
            latest["market_pair_check"].value_counts(dropna=False).head(5).to_dict()
        )

    # Classify each row (this is your existing signal logic)
    colors = []
    explains = []
    scores = []
    ml_green = set()


    # --- v1.1 market presence map (per game) for NCAAB single-market governors ---
    _mktset = {}
    try:
        for _, _r in latest.iterrows():
            sp_u = str(_r.get('sport','')).strip().upper()
            gid  = str(_r.get('game_id','')).strip()
            mk_u = str(_r.get('market','') or '').strip().upper()
            if not sp_u or not gid:
                continue
            if mk_u in ('SPREAD','TOTAL','MONEYLINE'):
                _mktset.setdefault((sp_u, gid), set()).add(mk_u)
        _mktcount = {k: len(v) for k, v in _mktset.items()}
    except Exception:
        _mktcount = {}
    # --- end v1.1 ---

    for _, row in latest.iterrows():
        color, expl = classify_side(
            bets_pct=int(row["bets_pct"]) if pd.notna(row.get("bets_pct")) else None,
            money_pct=int(row["money_pct"]) if pd.notna(row.get("money_pct")) else None,
            open_line=row.get("open_line") if pd.notna(row.get("open_line")) else None,
            current_line=row.get("current_line") if pd.notna(row.get("current_line")) else None,
            injury_news=row.get("injury_news") if pd.notna(row.get("injury_news")) else None,
            key_number_note=row.get("key_number_note") if pd.notna(row.get("key_number_note")) else None,
        )

        game = row.get("game")
        side = row.get("side")
        mkt = row.get("market_display")
        # -----------------------------
        # Numeric confidence score (0â€“100), interpretive only
        # -----------------------------
        score = 50.0

        mr = str(row.get("market_read") or "").strip()
        if mr == "Stealth Move":
            score += 8
        elif mr == "Freeze Pressure":
            score += 10
        elif mr == "Aligned Sharp":
            score += 6
        elif mr == "Contradiction":
            score += 2
        elif mr == "Public Drift":
            score -= 10

        try:
            D = float(row.get("divergence_D")) if pd.notna(row.get("divergence_D")) else 0.0
        except Exception:
            D = 0.0
        score += min(12.0, abs(D) * 0.4)

        try:
            lm = float(row.get("line_move_open")) if pd.notna(row.get("line_move_open")) else 0.0
        except Exception:
            lm = 0.0
        score += min(8.0, abs(lm) * 2.0)

        if str(row.get("key_number_note") or "").strip():
            score += 3

        tb = str(row.get("timing_bucket") or "").lower()
        if tb == "early":
            score += 2
        elif tb == "mid":
            score += 1
        elif tb == "late":
            score -= 1

        # --- v1.1 NCAAB early-window dampener (governor; score adjustment) ---
        # Reduce early NCAAB confidence to prevent premature upgrades.
        try:
            if str(row.get("sport", "")).strip().upper() == "NCAAB" and tb == "early":
                score -= 4
        except Exception:
            pass
        # --- end v1.1 ---

        # --- v1.1 NCAAB single-market dependency penalty (governor; score adjustment) ---
        # Score-based (WIDE dashboard): penalize when only one primary market supports the game.
        # If this row is SPREAD but TOTAL score is missing/weak (<60) -> -3.
        # If this row is TOTAL but SPREAD score is missing/weak (<60) -> -3.
        try:
            if str(row.get('sport','')).strip().upper() == 'NCAAB':
                _side = str(row.get('side','') or '').strip().lower()
                _mk_guess = ''
                if _side.startswith('over') or _side.startswith('under'):
                    _mk_guess = 'TOTAL'
                else:
                    _mk_guess = 'SPREAD'

                def _f(x):
                    try:
                        xs = str(x).strip()
                        if xs == '':
                            return None
                        return float(xs)
                    except Exception:
                        return None

                _sp = _f(row.get('SPREAD_model_score',''))
                _to = _f(row.get('TOTAL_model_score',''))

                if _mk_guess == 'SPREAD':
                    if (_to is None) or (_to < 60.0):
                        score -= 3
                elif _mk_guess == 'TOTAL':
                    if (_sp is None) or (_sp < 60.0):
                        score -= 3
        except Exception:
            pass
        # --- end v1.1 ---



        if color == "DARK_GREEN":
            score += 6
        elif color == "LIGHT_GREEN":
            score += 3
        elif color == "RED":
            score -= 6

        score = max(0.0, min(100.0, score))


        if mkt == "MONEYLINE" and color in ("DARK_GREEN", "LIGHT_GREEN"):
            ml_green.add((game, side))
        # ---- BIG DOG DARK GREEN NOTE (visual only) ----
        # Add a warning when a DARK_GREEN moneyline is a very large underdog
        try:
            if (
                mkt == "MONEYLINE"
                and color == "DARK_GREEN"
                and pd.notna(row.get("current_odds"))
                and int(row["current_odds"]) >= 300
            ):
                expl = f"{expl} | âš ï¸ Big underdog moneyline (+{int(row['current_odds'])})"
        except Exception:
            pass

        if (
            mkt == "SPREAD"
            and (game, side) in ml_green
            and color not in ("DARK_GREEN", "LIGHT_GREEN")
        ):
            expl = f"{expl} | Sharp ML, margin risk â€” ML favored over spread"

        colors.append(color)
        explains.append(expl)
        scores.append(score)


    latest = latest.copy()

    # ---------- GAME TIME (SINGLE SOURCE OF TRUTH) ----------
    # Always initialize so downstream code never breaks
    if "game_time_iso" not in latest.columns:
        latest["game_time_iso"] = ""
    # DK start time is primary source if present
    if "dk_start_iso" in latest.columns:
        dk_iso = latest["dk_start_iso"].fillna("").astype(str)
        # only overwrite if DK actually has values
        if dk_iso.str.len().gt(0).any():
            latest["game_time_iso"] = dk_iso

    # If dk_start_iso is actually a DK display string (not ISO), convert it -> ISO
    s = latest["game_time_iso"].fillna("").astype(str).str.strip()
    m_not_iso = s.ne("") & ~s.str.contains(r"^\d{4}-\d{2}-\d{2}T", regex=True)

    if m_not_iso.any():
        # Parse common DK display formats in NY time
        parsed = pd.to_datetime(
            s.where(m_not_iso),
            errors="coerce",
            format="mixed"
        )

        # If parsed is naive, localize to NY then convert to UTC ISO
        try:
            parsed = parsed.dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT")
        except Exception:
            pass

        # Store as ISO UTC (string)
        latest.loc[m_not_iso, "game_time_iso"] = parsed.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("")


    try:
        # ESPN kickoff enrichment disabled (DK dk_start_iso is source of truth).
        # ESPN FINALS remains enabled via update_snapshots_with_espn_finals().
        pass  # ESPN kickoff enrichment disabled (DK dk_start_iso is source of truth)
        
        latest["game"] = latest["game"].fillna("").astype(str)

        all_kickoffs = {}
        for sp in latest["sport"].dropna().astype(str).unique():
            games = (
                latest.loc[latest["sport"] == sp, "game"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            if not games:
                continue

        if False:
            km = get_espn_kickoff_map(sp, games)
            if DASH_DEBUG:
                print(f"[dash debug] ESPN map sport={sp} type={type(km)} len={len(km) if isinstance(km, dict) else 'NA'}")
            if isinstance(km, dict):
                nonblank = sum(1 for v in km.values() if str(v).strip())
                if DASH_DEBUG:
                    print(f"[dash debug] ESPN map sport={sp} nonblank_values={nonblank} sample={list(km.items())[:2]}")
                all_kickoffs.update({k: v for k, v in km.items() if str(v).strip()})

                
            if DASH_DEBUG:
                print(f"[dash debug] ESPN kickoffs total={len(all_kickoffs)}")

            # Fill ONLY blanks from ESPN (never overwrite DK-provided kickoff)
            espn_iso = latest["game"].map(all_kickoffs).fillna("")
            m_blank = latest["game_time_iso"].fillna("").astype(str).str.strip().eq("")
            latest.loc[m_blank, "game_time_iso"] = espn_iso.loc[m_blank]

            # ---- DEBUG: confirm we actually have kickoff values ----
            _s = latest["game_time_iso"].fillna("").astype(str).str.strip()
            if DASH_DEBUG:
                print(f"[dash debug] game_time_iso nonblank={(_s!='').sum()} / {len(_s)}  sample={_s[_s!=''].head(3).tolist()}")

    except Exception as e:
        import traceback
        if DASH_DEBUG:
            print("[dash debug] ESPN kickoff enrichment failed:")
        print(traceback.format_exc())
        # Do NOT overwrite DK kickoff times on ESPN failure
        latest["game_time_iso"] = latest["game_time_iso"].fillna("").astype(str)

    # ---- PARSE ISO -> NY datetime (used for filtering + display) ----
    latest["game_time_ny"] = (
        pd.to_datetime(latest["game_time_iso"], errors="coerce", utc=True)
          .dt.tz_convert("America/New_York")
    )        

    # =========================
    # TIMING CONTEXT (ADD-ONLY, DOES NOT TOUCH snapshot timestamps)
    # =========================
    now_ny = pd.Timestamp.now(tz="America/New_York")

    # minutes_to_kickoff: positive = time until start; negative = already started
    latest["minutes_to_kickoff"] = (
        (latest["game_time_ny"] - now_ny).dt.total_seconds() / 60.0
    ).round(0)

    # timing_bucket (v1.1): game-day anchored windows
    #   - NOT game day => EARLY (regardless of minutes, unless LIVE)
    #   - game day => EARLY (>480), MID (60..480), LATE (0..60), LIVE (<0)
    # NOTE: This is ADD-ONLY; does not touch snapshot timestamps or kickoff source.
    latest["is_game_day"] = False
    try:
        # compare NY dates (kickoff vs now)
        latest["is_game_day"] = (latest["game_time_ny"].dt.date == now_ny.date())
    except Exception:
        latest["is_game_day"] = False

    def _timing_bucket_v11(m2k, is_game_day):
        try:
            if pd.isna(m2k):
                return "UNKNOWN"
            m2k = int(m2k)
            if m2k < 0:
                return "LIVE"
            if not bool(is_game_day):
                return "EARLY"
            # game day: strict v1.1 windows
            return compute_timing_bucket("", m2k)
        except Exception:
            return "UNKNOWN"

    latest["timing_bucket"] = latest.apply(
        lambda r: _timing_bucket_v11(
            r.get("minutes_to_kickoff"),
            r.get("is_game_day"),
        ),
        axis=1,
    )



    latest["color"] = colors
    latest["why"] = explains
    latest["confidence_score"] = scores


    for _, r in latest.iterrows():
        if os.environ.get("RF_DISABLE_BASELINE_LOG","") != "1":
                log_baseline_signal(r)




    order = {"DARK_GREEN": 0, "LIGHT_GREEN": 1, "GREY": 2, "YELLOW": 3, "RED": 4}
    latest["rank"] = latest["color"].map(order).fillna(99).astype(int)

    # Styling
    def color_style(c: str) -> str:
        return {
            "DARK_GREEN": "background:#0B5A12;color:white;",
            "LIGHT_GREEN": "background:#9AF0A0;color:black;",
            "GREY": "background:#E0E0E0;color:black;",
            "YELLOW": "background:#F6E38A;color:black;",
            "RED": "background:#F08A8A;color:black;",
        }.get(c, "")


    # --------- GAME TIME: build a reliable display column (DK preferred, ESPN/ISO fallback) ---------
    # Always ensure game_time_display exists and is usable; later steps may change game_time_iso.
    if "game_time_display" not in latest.columns:
        latest["game_time_display"] = ""
    latest["game_time_display"] = latest["game_time_display"].fillna("").astype(str)

    # Build display from ISO (ESPN/DK ISO) when available
    if "game_time_iso" in latest.columns:
        ts = pd.to_datetime(latest["game_time_iso"], errors="coerce", utc=True)
        try:
            ts_local = ts.dt.tz_convert("America/New_York")
        except Exception:
            ts_local = ts
        espn_display = ts_local.dt.strftime("%a, %b %d %I:%M %p ET").fillna("")
    else:
        espn_display = pd.Series([""] * len(latest), index=latest.index)

    # 1) Prefer DK's human string if present (fills blanks only)
    if "game_time" in latest.columns:
        dk_disp = latest["game_time"].fillna("").astype(str)
        m_blank = latest["game_time_display"].str.strip().eq("")
        looks_like_time = dk_disp.str.contains(r"(AM|PM|ET|:\d{2})", regex=True, na=False)
        latest.loc[m_blank & looks_like_time, "game_time_display"] = dk_disp.loc[m_blank & looks_like_time]


    # 2) Fill remaining blanks from ISO-derived display
    m_blank = latest["game_time_display"].str.strip().eq("")
    latest.loc[m_blank, "game_time_display"] = espn_display.loc[m_blank]

    # Decide which column to show in table (prefer display)
    time_col = "game_time_display" if "game_time_display" in latest.columns else ("game_time" if "game_time" in latest.columns else None)
    show_game_time = (time_col is not None)
    # ---------------------------------------------------------------------------------------------

    show_news = (
        "injury_news" in latest.columns
        and latest["injury_news"].fillna("").astype(str).str.strip().ne("").any()
    )
    show_key_note = (
        "key_number_note" in latest.columns
        and latest["key_number_note"].fillna("").astype(str).str.strip().ne("").any()
    )

    news_th = "<th>News?</th>" if show_news else ""
    key_th = "<th>Key # Note</th>" if show_key_note else ""

    # Compute colspan from the EXACT columns we will render in the table header.
    # Sort for display: kickoff time first (like the old dashboard), then group by game
    if "game_time_iso" in latest.columns:
        # --- Guardrail: DK Network kickoff is authoritative for filtering/display ---
        # ESPN finals/kickoff enrichment is allowed for mapping/metrics, but must NOT overwrite DK kickoff time.
        if "dk_start_iso" in latest.columns:
            latest["game_time_iso"] = latest["dk_start_iso"].fillna("").astype(str)
        
        latest["_sort_time"] = pd.to_datetime(latest["game_time_iso"], errors="coerce", utc=True)

        # --- Time bucket (prevents TBD from breaking logic) ---
        now_utc = datetime.now(timezone.utc)

        def _time_bucket(kick):
            if pd.isna(kick):
                return "TBD"
            if kick >= now_utc:
                return "FUTURE"
            return "STALE"

        latest["time_bucket"] = latest["_sort_time"].apply(_time_bucket)

    
        # --- HARD FILTER: drop stale games quickly (clean dashboard) ---
        now_utc = datetime.now(timezone.utc)

        # Policy:
        # - Keep upcoming games through horizon_days
        # - Drop games fast after kickoff (postgame_grace_hours)
        # v1.1 dashboard rule: show ONLY upcoming games (no old DK n7days clutter)
        # Keep games from now through horizon_days.
        # If you want a tiny kickoff grace, change to: now_utc - timedelta(minutes=15)
        horizon_days = 7

        window_start = now_utc
        window_end   = now_utc + timedelta(days=horizon_days)

        before = len(latest)
        kick = latest["_sort_time"]
        # --- kickoff window diagnostics (debug only) ---
        try:
            _kick_na = int(kick.isna().sum())
        except Exception:
            _kick_na = -1
        try:
            _kick_min = kick.min()
            _kick_max = kick.max()
        except Exception:
            _kick_min = None
            _kick_max = None
        try:
            _lt = int((kick < window_start).sum())
            _in = int(((kick >= window_start) & (kick <= window_end)).sum())
            _gt = int((kick > window_end).sum())
        except Exception:
            _lt = _in = _gt = -1
        if DASH_DEBUG:
                    print(
            f"[dash debug] kickoff dist: na={_kick_na} lt_start={_lt} in_window={_in} gt_end={_gt} "
            f"kick_min={str(_kick_min)} kick_max={str(_kick_max)} "
            f"window_start={window_start.isoformat()} window_end={window_end.isoformat()}"
        )
        # --- end kickoff window diagnostics ---

        # Keep games with unknown kickoff (avoid silently hiding unresolved rows),
        # OR games whose kickoff is within [window_start, window_end].
        # Count unknown kickoffs (NaT) so we can see whether DK is supplying times
        try:
            unknown_kick = int(kick.isna().sum())
        except Exception:
            unknown_kick = -1

        # Keep unknown kickoff rows (NaT) OR rows within the window
        latest = latest.loc[
            (kick.isna()) | ((kick >= window_start) & (kick <= window_end))
        ].copy()
        after = len(latest)

        # Recompute sort time after filtering (keeps downstream stable)
        latest["_sort_time"] = pd.to_datetime(latest["game_time_iso"], errors="coerce", utc=True)

        if DASH_DEBUG:
            print(

            f"[dash debug] stale-kickoff filter: window_start={window_start.isoformat()} "
            f"window_end={window_end.isoformat()} kept={after}/{before} dropped={before-after}"
        )
    else:
        latest["_sort_time"] = pd.NaT
    # Final display sort time (ET)
    if "_sort_time" in latest.columns:
        try:
            latest["_sort_time"] = latest["_sort_time"].dt.tz_convert("America/New_York")
        except Exception:
            pass
    else:
        latest["_sort_time"] = pd.NaT

    # Group markets together within each game in a consistent order:
    # MONEYLINE -> SPREAD -> TOTAL -> everything else
    market_order = {"MONEYLINE": 0, "SPREAD": 1, "TOTAL": 2}
    latest["_mkt_rank"] = latest["market_display"].map(market_order).fillna(99).astype(int)

    latest = latest.sort_values(
        by=["sport_label", "_sort_time", "game", "_mkt_rank", "market_display", "side"],
        ascending=[True, True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    latest = latest.drop(columns=["_mkt_rank"], errors="ignore")


    # Ensure no duplicate column labels (can happen after merges)
    latest = latest.loc[:, ~latest.columns.duplicated()].copy()
    
        # -----------------------------
    # GAME-LEVEL AGGREGATION (interpretive only; no side flipping)
    # One row per (sport, game_id, market_display)
    # -----------------------------
    if "confidence_score" not in latest.columns:
        latest["confidence_score"] = 50.0

    game_keys = ["sport", "game_id", "market_display"]

    # Ensure numeric
    # --- ensure score_num / score_bucket are defined (lint + runtime safety) ---
    score_raw = ''
    try:
        # r is the row dict/Series in the row-building loop (common pattern in this file)
        score_raw = r.get('model_score','') if 'r' in locals() else (row.get('model_score','') if 'row' in locals() else '')
    except Exception:
        score_raw = ''
    try:
        s = str(score_raw).strip()
        score_num = float(s) if s and s.lower() not in ('nan','none','null') else 0.0
    except Exception:
        score_num = 0.0
    try:
        # simple buckets; adjust thresholds later if needed
        score_bucket = 'DARK_GREEN' if score_num >= 75 else ('GREEN' if score_num >= 65 else ('LEAN' if score_num >= 55 else ''))
    except Exception:
        score_bucket = ''
    # --- end score vars ---
    
    latest["_score_num"] = pd.to_numeric(latest["confidence_score"], errors="coerce").fillna(50.0)

    # Favored side = max score within the game+market
    idx_fav = latest.groupby(game_keys)["_score_num"].idxmax()
    fav_rows = latest.loc[
        idx_fav,
        game_keys + ["game", "sport_label", "side", "_score_num"]
    ].copy()

    fav_rows = fav_rows.rename(columns={
        "side": "favored_side",
        "_score_num": "game_confidence"
    })

    # Min/Max side scores within each game+market
    min_rows = latest.groupby(game_keys)["_score_num"].min().reset_index().rename(columns={"_score_num": "min_side_score"})
    max_rows = latest.groupby(game_keys)["_score_num"].max().reset_index().rename(columns={"_score_num": "max_side_score"})

    game_view = (
        fav_rows
        .merge(min_rows, on=game_keys, how="left")
        .merge(max_rows, on=game_keys, how="left")
    )

    game_view["net_edge"] = (game_view["max_side_score"] - game_view["min_side_score"]).round(1)

    # Game decision: BET requires strong score + meaningful net edge; otherwise LEAN/NO BET
    def _game_decision(score, net_edge):
        try:
            s = float(score)
        except Exception:
            s = 50.0
        try:
            ne = float(net_edge)
        except Exception:
            ne = 0.0

        if s >= 72 and ne >= 10:
            return "BET"
        if s >= 62:
            return "LEAN"
        return "NO BET"

    game_view["game_decision"] = game_view.apply(lambda r: _game_decision(r.get("game_confidence", 50), r.get("net_edge", 0)), axis=1)
    game_view["opp_weak"] = game_view["min_side_score"] <= 35.0
    game_view["opp_weak_mark"] = game_view["opp_weak"].apply(lambda x: "âš‘" if bool(x) else "")

    # Sort for display: sport, game time, market order
    if "_sort_time" in latest.columns:
        time_map = (
            latest.groupby(["sport", "game_id"])["_sort_time"]
            .min()
            .reset_index()
            .rename(columns={"_sort_time": "_game_time"})
        )
        game_view = game_view.merge(time_map, on=["sport", "game_id"], how="left")
    else:
        game_view["_game_time"] = pd.NaT

    market_order = {"MONEYLINE": 0, "SPREAD": 1, "TOTAL": 2}
    game_view["_game_mkt_rank"] = game_view["market_display"].map(market_order).fillna(99).astype(int)

    game_view = game_view.sort_values(
        ["sport_label", "_game_time", "_game_mkt_rank", "game"],
        na_position="last"
    ).reset_index(drop=True)

    # -----------------------------
    # TABLE HEADERS + HYBRID ROW BUILD (UI ONLY)
    # GAME rows visible; SIDE rows hidden + directly under their GAME row
    # -----------------------------

    # --- Game Time toggle (display-only) ---
    show_game_time = True  # always show column; cell renderer will blank if missing

    # --- GAME header columns (this is the table schema) ---
    header_cols = (
        ["Sport", "Game"]
        + (["Game Time"] if show_game_time else [])
        + ["Market", "Decision", "Lean", "Model Score", "Net Edge"]
    )
    colspan = len(header_cols)

    # Build a fast lookup of SIDE rows by parent key so we can render children immediately under each game row
    latest["_parent_gk"] = (
        latest["sport"].astype(str) + "|" +
        latest["game_id"].astype(str) + "|" +
        latest["market_display"].astype(str)
    )
    side_groups = {k: g.copy() for k, g in latest.groupby("_parent_gk", dropna=False)}

        # ---- GAME TIME cell helper (GAME rows use game_view._game_time; SIDE rows use rr game_time_*) ----
    def _time_cell(rr):
        """
        Returns a <td>...</td> for Game Time.
        - For GAME rows (game_view), we use rr['_game_time'] if present.
        - For SIDE rows (latest), we use rr['game_time_display'] then rr['game_time_iso'].
        """
        # 1) GAME VIEW preferred: _game_time is already a tz-aware Timestamp (NY)
        gt = rr.get("_game_time", None)
        try:
            if pd.notna(gt):
                # ensure NY tz
                if getattr(gt, "tzinfo", None) is None:
                    gt = gt.tz_localize("America/New_York")
                else:
                    gt = gt.tz_convert("America/New_York")
                return f"<td>{gt.strftime('%a %m/%d %I:%M %p ET')}</td>"
        except Exception:
            pass

        # 2) SIDE ROW preferred: preformatted display if present
        v = rr.get("game_time_display", "")
        if isinstance(v, str) and v.strip():
            return f"<td>{v}</td>"

        # 3) SIDE ROW fallback: ISO -> ET
        iso = rr.get("game_time_iso", "")
        if isinstance(iso, str) and iso.strip():
            try:
                dt = pd.to_datetime(iso.replace("Z", "+00:00"), utc=True, errors="coerce")
                if pd.notna(dt):
                    dt = dt.tz_convert("America/New_York")
                    return f"<td>{dt.strftime('%a %m/%d %I:%M %p ET')}</td>"
            except Exception:
                pass

        # 4) Last resort: TBD is acceptable if truly missing
        return "<td>TBD</td>"


    # -----------------------------
    # BUILD HTML ROWS (interleaved)
    # -----------------------------
    rows_html = []

    for _, gr in game_view.iterrows():
        gk = f"{gr.get('sport','')}|{gr.get('game_id','')}|{gr.get('market_display','')}"

        # --- GAME SUMMARY ROW (visible) ---
        gt = _time_cell(gr)

        rows_html.append(f"""
<tr class="game-row" data-gamekey="{gk}" onclick="toggleGroup('{gk}')">
  <td>{gr.get("sport_label","")}</td>
  <td><b>{gr.get("game","")}</b></td>
  {gt}
  <td>{gr.get("market_display","")}</td>
  <td><span class="pill decision">{gr.get("game_decision","NO BET")}</span></td>
  <td><span class="pill lean">{gr.get("favored_side","")}</span></td>
  <td><span class="pill score">{round(float(gr.get("game_confidence",0) or 0),1)}</span></td>
  <td><span class="pill edge">{round(float(gr.get("net_edge",0) or 0),1)}</span></td>
</tr>
""")

        # --- SIDE ROWS (hidden by default, packed into GAME schema) ---
        sg = side_groups.get(gk)
        if sg is None or sg.empty:
            continue

        # Stable order: higher confidence first (UI only)
        try:
            sg = sg.copy()
            sg["_score_num"] = pd.to_numeric(sg["confidence_score"], errors="coerce").fillna(50.0)
            sg = sg.sort_values("_score_num", ascending=False, kind="mergesort")
        except Exception:
            pass

        for _, rr in sg.iterrows():
            st = color_style(rr.get("color", "GREY"))

            # display-only side label cleanup
            mkt = rr.get("market_display", "")
            side_disp = rr.get("side", "")
            if mkt == "SPREAD":
                side_disp = re.sub(r"\s[+-]\d+(?:\.\d+)?\s*$", "", str(side_disp)).strip()

            # Market cell
            market_cell = f"{side_disp} â€” {rr.get('market_display','')}".strip(" â€”")

            # Decision: Bets / Money
            bets_cell = "" if pd.isna(rr.get("bets_pct")) else f"{int(rr['bets_pct'])}%"
            money_cell = "" if pd.isna(rr.get("money_pct")) else f"{int(rr['money_pct'])}%"
            decision_cell = f"B {bets_cell} / $ {money_cell}".strip()

            # Open â†’ Current
            o = "" if pd.isna(rr.get("open_line")) else str(rr.get("open_line"))
            c = "" if pd.isna(rr.get("current_line")) else str(rr.get("current_line"))
            oc_cell = (o + " â†’ " + c).strip(" â†’")

            # Side model score
            try:
                sc = "" if pd.isna(rr.get("confidence_score")) else f"{float(rr.get('confidence_score')):.1f}"
            except Exception:
                sc = ""

            # Market read
            mr = str(rr.get("market_read","") or "").strip()

            # Blank game-time cell for side rows
            gt_side = "<td></td>" if show_game_time else ""

            rows_html.append(f"""
<tr class="side-row" style="{st}display:none;"
    data-row="1"
    data-parent="{gk}"
    data-color="{rr.get('color','')}"
    data-sport="{rr.get('sport_label','')}"
    data-market="{rr.get('market_display','')}"
    data-ml-odds="{'' if pd.isna(rr.get('current_odds')) else int(rr.get('current_odds'))}"
    data-search="{str(rr.get('game',''))} {str(side_disp)} {str(rr.get('market_display',''))}">
  <td></td>
  <td>â†³ {rr.get("why","")}</td>
  {gt_side}
  <td>{market_cell}</td>
  <td>{decision_cell}</td>
  <td>{oc_cell}</td>
  <td>{sc}</td>
  <td>{mr}</td>
</tr>
""")

    # -----------------------------
    # HEADERS
    # -----------------------------
    # Wide game-level dashboard (one row per game; grouped markets)

    def _blank(x):
        if x is None:
            return ""
        try:
            if pd.isna(x):
                return ""
        except Exception:
            pass
        return str(x)

    def _fmt_int(x):
        x = _blank(x)
        if x == "":
            return ""
        try:
            return f"{int(float(x))}"
        except Exception:
            return x

    def _fmt_pct(x):
        x = _blank(x)
        if x == "":
            return ""
        try:
            return f"{int(float(x))}%"
        except Exception:
            return x

    def _fmt_num(x):
        x = _blank(x)
        if x == "":
            return ""
        try:
            v = float(x)
            if v.is_integer():
                return str(int(v))
            return f"{v:.1f}"
        except Exception:
            return x

    def _fmt_score(x):
        x = _blank(x)
        if x == "":
            return ""
        try:
            return f"{float(x):.1f}"
        except Exception:
            return x

    # Ensure these exist (no NaN leaks)
    for c in ["confidence_score", "color", "why", "market_read", "game_time_iso"]:
        if c not in latest.columns:
            latest[c] = ""

    # Decide thresholds (UI-only label; does NOT change scoring)
    def _decision(score, net_edge=None):
        try:
            sc = float(score)
        except Exception:
            sc = 50.0

        try:
            ne = float(net_edge) if net_edge is not None else 0.0
        except Exception:
            ne = 0.0

        # STRONG BET = high score + meaningful asymmetry (v1.1 timing enforcement)
        # Enforce timing discipline in the decision pipeline (not just dashboard flags):
        #   - Global: no STRONG BET in LATE
        #   - NCAAB: STRONG blocked in EARLY and LATE
        tb = str(rr.get("timing_bucket") or "").strip().upper()
        sp = str(rr.get("sport") or "").strip().lower()

        strong_block = False
        if tb == "LATE":
            strong_block = True
        if sp == "ncaab" and tb in ("EARLY", "LATE"):
            strong_block = True

        if sc >= 72 and ne >= 10 and not strong_block:
            return "STRONG BET"
        if sc >= 68:
            return "BET"
        if sc >= 60:
            return "LEAN"
        return "NO BET"

    # -----------------------------
    # Build per-market winners + net edge
    # -----------------------------
    # For each (sport, game_id, market_display): pick best side row by confidence_score
    l2 = latest.copy()

    # Side display cleanup for SPREAD (strip trailing number for display)
    l2["side_disp"] = l2["side"].astype(str)
    l2.loc[l2["market_display"] == "SPREAD", "side_disp"] = (
        l2.loc[l2["market_display"] == "SPREAD", "side_disp"]
          .str.replace(r"\s[+-]\d+(?:\.\d+)?\s*$", "", regex=True)
          .str.strip()
    )

    # Safety: numeric score
    l2["_score"] = pd.to_numeric(l2["confidence_score"], errors="coerce").fillna(50.0)

    # Net edge per game+market
    edge = (
        l2.groupby(["sport", "game_id", "market_display"])["_score"]
          .agg(["max", "min"])
          .reset_index()
    )
    edge["net_edge"] = edge["max"] - edge["min"]

    # Winner row per game+market
    winners = (
        l2.sort_values(
            ["sport", "game_id", "market_display", "_score"],
            ascending=[True, True, True, False],
            kind="mergesort"
        )
        .groupby(["sport", "game_id", "market_display"], as_index=False)
        .head(1)
        .merge(
            edge[["sport", "game_id", "market_display", "net_edge"]],
            on=["sport", "game_id", "market_display"],
            how="left"
        )
    )

    # Base game rows: derive kickoff from SIDE-level latest (already passed stale-kickoff filter)
    _tm = latest[["sport", "game_id", "game_time_iso"]].copy() if "game_time_iso" in latest.columns else pd.DataFrame(columns=["sport","game_id","game_time_iso"])
    if not _tm.empty:
        _tm["game_time_iso"] = _tm["game_time_iso"].fillna("").astype(str).str.strip()
        _tm = _tm[_tm["game_time_iso"] != ""]
        _tm = _tm.drop_duplicates(subset=["sport", "game_id"], keep="first")
    
    base = winners.groupby(["sport", "game_id"], as_index=False).first()[["sport", "game_id", "game", "sport_label"]]

    # --- market pair check (game-level flag from side-level latest) ---
    try:
        if 'market_pair_check' in latest.columns:
            _pc = latest[['sport','game_id','market_pair_check']].copy()
            _pc['market_pair_check'] = _pc['market_pair_check'].fillna('').astype(str).str.strip()
            _pc = _pc[_pc['market_pair_check'] != '']
            if not _pc.empty:
                _pc = _pc.drop_duplicates(subset=['sport','game_id'])
                _pc = _pc.rename(columns={'market_pair_check':'dash_pair_check'})
                base = base.merge(_pc[['sport','game_id','dash_pair_check']], on=['sport','game_id'], how='left')
            else:
                base['dash_pair_check'] = ''
        else:
            base['dash_pair_check'] = ''
    except Exception:
        # never break report for this
        base['dash_pair_check'] = ''
    # --- end pair check ---

    if not _tm.empty:
        base = base.merge(_tm, on=["sport", "game_id"], how="left")
    else:
        base["game_time_iso"] = ""
    base["game_time_iso"] = base["game_time_iso"].fillna("").astype(str)
    
    # Spread fields
    sp = winners[winners["market_display"] == "SPREAD"].copy()
    if sp.empty:
        sp = pd.DataFrame(columns=["sport", "game_id"])
    else:
        sp["open_spread"] = sp.get("open_line_val", "")
        sp["current_spread"] = sp.get("current_line_val", "")
        sp["current_spread_price"] = sp.get("current_odds", "")

    # Total fields
    tt = winners[winners["market_display"] == "TOTAL"].copy()
    if tt.empty:
        tt = pd.DataFrame(columns=["sport", "game_id"])
    else:
        tt["open_total"] = tt.get("open_line_val", "")
        tt["current_total"] = tt.get("current_line_val", "")
        tt["current_total_price"] = tt.get("current_odds", "")

    # ML fields (open/current are ODDS)
    ml = winners[winners["market_display"] == "MONEYLINE"].copy()
    if ml.empty:
        ml = pd.DataFrame(columns=["sport", "game_id"])
    else:
        ml["open_ml"] = ml.get("open_odds", "")
        ml["current_ml"] = ml.get("current_odds", "")
        ml["current_ml_price"] = ml.get("current_odds", "")

    # Assemble wide df
    dash = base.copy()

    def _join_market(dash_df, sub_df, prefix):
        if sub_df is None or len(sub_df) == 0:
            return dash_df

        keep = [
            "sport", "game_id", "side_disp", "bets_pct", "money_pct",
            "confidence_score", "net_edge", "color",
            "open_spread", "current_spread", "current_spread_price",
            "open_total", "current_total", "current_total_price",
            "open_ml", "current_ml", "current_ml_price",
        ]
        present = [c for c in keep if c in sub_df.columns]
        sub = sub_df[present].copy()

        ren = {
            "side_disp": f"{prefix}_side",
            "bets_pct": f"{prefix}_bets_pct",
            "money_pct": f"{prefix}_money_pct",
            "confidence_score": f"{prefix}_model_score",
            "net_edge": f"{prefix}_net_edge",
            "color": f"{prefix}_color",

            "open_spread": f"{prefix}_open_line",
            "current_spread": f"{prefix}_current_line",
            "current_spread_price": f"{prefix}_current_price",

            "open_total": f"{prefix}_open_line",
            "current_total": f"{prefix}_current_line",
            "current_total_price": f"{prefix}_current_price",

            "open_ml": f"{prefix}_open_line",
            "current_ml": f"{prefix}_current_line",
            "current_ml_price": f"{prefix}_current_price",
        }

        sub = sub.rename(columns=ren)
        sub[f"{prefix}_decision"] = sub.apply(
            lambda r: _decision(r.get(f"{prefix}_model_score"), r.get(f"{prefix}_net_edge")),
            axis=1
        )

        return dash_df.merge(sub, on=["sport", "game_id"], how="left")

    dash = _join_market(dash, sp, "SPREAD")
    dash = _join_market(dash, tt, "TOTAL")
    dash = _join_market(dash, ml, "MONEYLINE")

    # Fill blanks (NO NaN in output)
    dash = dash.fillna("")

    # Ensure timing audit columns survive into dashboard.csv
    for _c in ("minutes_to_kickoff", "is_game_day", "timing_bucket"):
        if _c not in dash.columns and _c in latest.columns:
            try:
                # map from latest -> dash by sport/game_id
                _map = {(str(r.get("sport","")).strip(), str(r.get("game_id","")).strip()): r.get(_c) for _, r in latest.iterrows()}
                dash[_c] = dash.apply(lambda rr: _map.get((str(rr.get("sport","")).strip(), str(rr.get("game_id","")).strip())), axis=1)
            except Exception:
                dash[_c] = ""

    # ---------- Column order (LOCKED) ----------
    # Markets grouped left→right as: SPREAD → TOTAL → MONEYLINE
    col_order = [
        "sport_label", "game", "game_time_iso",
        # SPREAD
        "SPREAD_decision", "SPREAD_side", "SPREAD_model_score", "SPREAD_net_edge", "SPREAD_bets_pct", "SPREAD_money_pct",
        "SPREAD_open_line", "SPREAD_current_line", "SPREAD_current_price",
        # TOTAL
        "TOTAL_decision", "TOTAL_side", "TOTAL_model_score", "TOTAL_net_edge", "TOTAL_bets_pct", "TOTAL_money_pct",
        "TOTAL_open_line", "TOTAL_current_line", "TOTAL_current_price",
        # MONEYLINE
        "MONEYLINE_decision", "MONEYLINE_side", "MONEYLINE_model_score", "MONEYLINE_net_edge", "MONEYLINE_bets_pct", "MONEYLINE_money_pct",
        "MONEYLINE_open_line", "MONEYLINE_current_line", "MONEYLINE_current_price",
    ]

    present = [c for c in col_order if c in dash.columns]
    extras = [c for c in dash.columns if c not in present]
    dash = dash[present + extras]

    # ---------- Write dashboard.csv (wide schema) ----------
    ensure_data_dir()
    out_csv = os.path.join(DATA_DIR, "dashboard.csv")

    # Default sort: sport order -> kickoff time -> game
    sport_order = {"NFL": 0, "NBA": 1, "NHL": 2, "NCAAF": 3, "NCAAB": 4}
    _sport = dash.get("sport_label", dash.get("sport", "")).fillna("").astype(str).str.upper()
    dash["_sport_rank"] = _sport.map(sport_order).fillna(99).astype(int)
    dash["_kickoff"] = pd.to_datetime(dash.get("game_time_iso", ""), errors="coerce", utc=True)
    dash["_game_sort"] = dash.get("game", "").fillna("").astype(str)

    dash = (
        dash.sort_values(["_sport_rank", "_kickoff", "_game_sort"], kind="stable")
            .drop(columns=["_sport_rank", "_kickoff", "_game_sort"], errors="ignore")
    )

    # --- v1.1 timing buckets (canonical; flags only; no scoring) ---
    # Canonical source:
    #   - minutes_to_kickoff + is_game_day are computed upstream from DK kickoff
    #   - timing_bucket uses strict v1.1 windows on GAME DAY only; otherwise EARLY (unless LIVE)
    dash["timing_bucket"] = ""
    # dash["is_game_day"] already exists upstream; keep default False if missing

    if "minutes_to_kickoff" in dash.columns:
        def _dash_tb(m2k, is_gd):
            try:
                if m2k in ("", None):
                    return ""
                m = int(float(m2k))
                if m < 0:
                    return "LIVE"
                if not bool(is_gd):
                    return "EARLY"
                return compute_timing_bucket("", m)
            except Exception:
                return ""

        try:
            if "is_game_day" not in dash.columns:
                dash["is_game_day"] = False
        except Exception:
            pass

        try:
            dash["timing_bucket"] = dash.apply(lambda r: _dash_tb(r.get("minutes_to_kickoff"), r.get("is_game_day")), axis=1)
        except Exception:
            pass
    # --- end v1.1 timing buckets ---

    # --- v1.1 persistence & stability flags (flags only; no scoring) ---
    try:
        import pandas as _pd

        state_path = os.path.join(DATA_DIR, "row_state.csv")
        if os.path.exists(state_path):
            _state = _pd.read_csv(state_path, keep_default_na=False)

            key_cols = ["sport", "game_id", "market", "side"]
            # NOTE: dashboard is WIDE (no generic 'market'/'side' cols). Per-market flags are computed below.
            if all(c in dash.columns for c in key_cols):
                dash_key = dash[key_cols].astype(str).agg("|".join, axis=1).tolist()
            else:
                dash_key = ["" for _ in range(len(dash))]

            # Normalize state keys to match dashboard lookups (case-sensitive dict keys!)
            _tmpk = _state[key_cols].copy()
            _tmpk["sport"]  = _tmpk["sport"].astype(str).str.strip().str.upper()
            _tmpk["market"] = _tmpk["market"].astype(str).str.strip().str.upper()
            _tmpk["game_id"]= _tmpk["game_id"].astype(str).str.strip()
            _tmpk["side"]   = _tmpk["side"].astype(str).str.strip()
            _state["_k"] = _tmpk.astype(str).agg("|".join, axis=1)
            state_map = _state.set_index("_k").to_dict("index")

            persist_ok = []
            stable_ok = []

            for k in dash_key:
                r = state_map.get(k)
                if not r:
                    persist_ok.append(False)
                    stable_ok.append(False)
                    continue

                # persistence (instrumentation-only):
                # Global STRONG requires >=2 consecutive snapshots >=72
                # NCAAB STRONG requires >=3 consecutive snapshots >=72
                try:
                    ss = int(str(r.get("strong_streak","0")).strip() or "0")
                except Exception:
                    ss = 0
                sp = str(k).split("|", 1)[0].strip().upper()
                need = 3 if sp == "NCAAB" else 2
                persist_ok.append(ss >= need)

                # stability: last_score >= peak_score - 3 (NCAAB tighter: -2)
                try:
                    ls = float(r.get("last_score", ""))
                    ps = float(r.get("peak_score", ""))
                    # dash_key is "sport|game_id|market|side"
                    sp = str(k).split("|", 1)[0].strip().upper()
                    delta = 2 if sp == "NCAAB" else 3
                    stable_ok.append(ls >= (ps - delta))
                except Exception:
                    stable_ok.append(False)

            dash["persist_ok"] = persist_ok
            dash["stable_ok"] = stable_ok
            # --- v1.1 per-market persistence/stability (dashboard-only; no scoring) ---
            # Uses row_state key: sport|game_id|market|side  (market in SPREAD/TOTAL/MONEYLINE)
            for _m in ("SPREAD","TOTAL","MONEYLINE"):
                dash[f"{_m}_persist_ok"] = False
                dash[f"{_m}_stable_ok"] = False

            try:
                # Prefer per-market side columns if present; otherwise fall back to per-market favored columns.
                for _m in ("SPREAD","TOTAL","MONEYLINE"):
                    side_col = f"{_m}_side"
                    if side_col not in dash.columns:
                        side_col = f"{_m}_favored"
            
                    p_ok = []
                    s_ok = []
            
                    for _, rr in dash.iterrows():
                        sp = str(rr.get("sport_label", rr.get("sport",""))).strip().upper()
                        gid = str(rr.get("game_id","")).strip()
                        side = str(rr.get(side_col,"")).strip()
            
                        if not sp or not gid or not side:
                            p_ok.append(False)
                            s_ok.append(False)
                            continue
            
                        kk = f"{sp}|{gid}|{_m}|{side}"
                        r = state_map.get(kk)
                        if not r:
                            p_ok.append(False)
                            s_ok.append(False)
                            continue
            
                        # persistence streak threshold (>=2 global, >=3 NCAAB)
                        try:
                            ss = int(str(r.get("strong_streak","0")).strip() or "0")
                        except Exception:
                            ss = 0
                        need = 3 if sp == "NCAAB" else 2
                        p_ok.append(ss >= need)
            
                        # stability (NCAAB tighter)
                        try:
                            ls = float(str(r.get("last_score","")).strip() or "0")
                            ps = float(str(r.get("peak_score","")).strip() or "0")
                            delta = 2 if sp == "NCAAB" else 3
                            s_ok.append(ls >= (ps - delta))
                        except Exception:
                            s_ok.append(False)
            
                    dash[f"{_m}_persist_ok"] = p_ok
                    dash[f"{_m}_stable_ok"] = s_ok
            except Exception:
                pass
            # --- end v1.1 per-market persistence/stability ---


        else:
            dash["persist_ok"] = False
            dash["stable_ok"] = False
    except Exception:
        dash["persist_ok"] = False
        dash["stable_ok"] = False
    # --- end v1.1 persistence & stability flags ---

    # --- v1.1 STRONG certification flags (dashboard-only; no scoring) ---
    dash.to_csv(out_csv, index=False, encoding="utf-8")
    print("[ok] wrote dashboard csv:", out_csv)

    # ---- Step C metrics input (instrumentation only) ----
    # IMPORTANT: snapshots.csv does NOT contain model scores; dashboard.csv does.
    try:
        import pandas as _pd

        _dash_path = os.path.join("data", "dashboard.csv")
        _dash = _pd.read_csv(_dash_path, keep_default_na=False)

        def _best_side_col(mkt: str) -> str:
            direct = f"{mkt}_side"
            if direct in _dash.columns:
                return direct
            for c in _dash.columns:
                cu = c.upper()
                cl = c.lower()
                if not cu.startswith(mkt + "_"):
                    continue
                if any(k in cl for k in ("favored_side", "favourite_side", "favorite_side", "favored", "side", "pick", "selection")):
                    return c
            return ""

        rows = []

        def _add_market(mkt: str, score_col: str):
            if score_col not in _dash.columns:
                return
            side_col = _best_side_col(mkt)
            if not side_col:
                return

            # net edge column (this is what we need for C2 wiring)
            net_col = f"{mkt}_net_edge"
            has_net = net_col in _dash.columns

            for _, rr in _dash.iterrows():
                sport = str(rr.get("sport", rr.get("sport_label", ""))).strip()
                gid = str(rr.get("game_id", "")).strip()
                game = str(rr.get("game", "")).strip()
                side = str(rr.get(side_col, "")).strip()
                sc = str(rr.get(score_col, "")).strip()

                if not sport or not gid or not side or not sc:
                    continue

                rows.append({
                    "sport": sport,
                    "game_id": gid,
                    "game": game,
                    "market_display": mkt,
                    "side": side,
                    "model_score": sc,
                    "net_edge": str(rr.get(net_col, "")).strip() if has_net else "",
                    "current_line": "",
                    "current_odds": "",
                    "bets_pct": "",
                    "money_pct": "",
                })

        _add_market("SPREAD", "SPREAD_model_score")
        _add_market("TOTAL", "TOTAL_model_score")
        _add_market("MONEYLINE", "MONEYLINE_model_score")

        if rows:
            _metrics_df = _pd.DataFrame(rows)
            update_row_state_and_signal_ledger(_metrics_df)

    except Exception:
        pass
    # ---- end Step C metrics input ----



    # ---------- HTML table (wide) ----------
    header_cols = [
        "Sport",
        "Game",
        "Time (ET)",
        # SPREAD
        "Spread Decision", "Spread Side", "Spread Score", "Spread Net Edge", "Spread Bets%", "Spread Money%",
        "Open Spread", "Current Spread", "Current Spread Price",
        # TOTAL
        "Total Decision", "Total Side", "Total Score", "Total Net Edge", "Total Bets%", "Total Money%",
        "Open Total", "Current Total", "Current Total Price",
        # ML
        "ML Decision", "ML Side", "ML Score", "ML Net Edge", "ML Bets%", "ML Money%",
        "Open ML", "Current ML", "Current ML Price",
    ]

    header_ths = "".join(f"<th>{c}</th>" for c in header_cols)

    rows_html = []

    def _time_et(iso):
        iso = _blank(iso)
        if iso == "":
            return ""
        try:
            ts = pd.to_datetime(iso, utc=True, errors="coerce")
            if pd.isna(ts):
                return ""
            return ts.tz_convert("America/New_York").strftime("%a %m/%d %I:%M%p")
        except Exception:
            return ""

    for _, rr in dash.iterrows():
        sport = _blank(rr.get("sport_label"))
        game = _blank(rr.get("game"))
        t_et = _time_et(rr.get("game_time_iso"))

        # Spread cells
        sp_dec = _blank(rr.get("SPREAD_decision"))
        sp_side = _blank(rr.get("SPREAD_side")) or _blank(rr.get("SPREAD_favored"))
        sp_sc = _fmt_score(rr.get("SPREAD_model_score"))
        sp_edge = _fmt_score(rr.get("SPREAD_net_edge"))
        sp_b = _fmt_pct(rr.get("SPREAD_bets_pct"))
        sp_m = _fmt_pct(rr.get("SPREAD_money_pct"))
        sp_o = _fmt_num(rr.get("SPREAD_open_line"))
        sp_c = _fmt_num(rr.get("SPREAD_current_line"))
        sp_cp = _fmt_int(rr.get("SPREAD_current_price"))

        # Total cells
        t_dec = _blank(rr.get("TOTAL_decision"))
        t_side = _blank(rr.get("TOTAL_side")) or _blank(rr.get("TOTAL_favored"))
        t_sc = _fmt_score(rr.get("TOTAL_model_score"))
        t_edge = _fmt_score(rr.get("TOTAL_net_edge"))
        t_b = _fmt_pct(rr.get("TOTAL_bets_pct"))
        t_m = _fmt_pct(rr.get("TOTAL_money_pct"))
        t_o = _fmt_num(rr.get("TOTAL_open_line"))
        t_c = _fmt_num(rr.get("TOTAL_current_line"))
        t_cp = _fmt_int(rr.get("TOTAL_current_price"))

        # ML cells
        ml_dec = _blank(rr.get("MONEYLINE_decision"))
        ml_side = _blank(rr.get("MONEYLINE_side")) or _blank(rr.get("MONEYLINE_favored"))
        ml_sc = _fmt_score(rr.get("MONEYLINE_model_score"))
        ml_edge = _fmt_score(rr.get("MONEYLINE_net_edge"))
        ml_b = _fmt_pct(rr.get("MONEYLINE_bets_pct"))
        ml_m = _fmt_pct(rr.get("MONEYLINE_money_pct"))
        ml_o = _fmt_int(rr.get("MONEYLINE_open_line"))
        ml_c = _fmt_int(rr.get("MONEYLINE_current_line"))
        ml_cp = _fmt_int(rr.get("MONEYLINE_current_price"))

        rows_html.append(
            f"""
<tr class="game-row">
  <td>{sport}</td>
  <td>{game}</td>
  <td>{t_et}</td>

  <td>{sp_dec}</td><td>{sp_side}</td><td>{sp_sc}</td><td>{sp_edge}</td><td>{sp_b}</td><td>{sp_m}</td>
  <td>{sp_o}</td><td>{sp_c}</td><td>{sp_cp}</td>

  <td>{t_dec}</td><td>{t_side}</td><td>{t_sc}</td><td>{t_edge}</td><td>{t_b}</td><td>{t_m}</td>
  <td>{t_o}</td><td>{t_c}</td><td>{t_cp}</td>

  <td>{ml_dec}</td><td>{ml_side}</td><td>{ml_sc}</td><td>{ml_edge}</td><td>{ml_b}</td><td>{ml_m}</td>
  <td>{ml_o}</td><td>{ml_c}</td><td>{ml_cp}</td>
</tr>
"""
        )

    # Snapshot timestamps (current + previous)
    current_ts_disp = ""
    prev_ts_disp = ""
    try:
        if "timestamp" in df.columns:
            ts_all = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna()
            ts_unique = ts_all.drop_duplicates().sort_values()

            if len(ts_unique) > 0:
                cur = ts_unique.iloc[-1]
                current_ts_disp = cur.tz_convert("America/New_York").strftime("%a %b %d, %Y %I:%M:%S %p ET")

            if len(ts_unique) > 1:
                prv = ts_unique.iloc[-2]
                prev_ts_disp = prv.tz_convert("America/New_York").strftime("%a %b %d, %Y %I:%M:%S %p ET")
    except Exception:
        pass

    snapshot_html = f"""
<div style="margin:10px 0 14px 0; padding:10px 12px; background:#f7f7f7; border:1px solid #ddd; border-radius:8px;">
  <div style="font-size:12px;">
    <b>Current snapshot:</b> {current_ts_disp or "—"}
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <b>Previous snapshot:</b> {prev_ts_disp or "—"}
  </div>
</div>
"""

    legend_html = """
<div style="margin:0 0 14px 0; padding:10px 12px; background:#fff; border:1px solid #ddd; border-radius:8px;">
  <div style="font-weight:bold; margin-bottom:6px;">Legend</div>
  <div style="font-size:12px; line-height:1.6;">
    <div><b>Decision</b>: STRONG BET (≥72 + Net Edge ≥10), BET (≥68), LEAN (≥60), NO BET (&lt;60).</div>
    <div style="margin-top:6px;"><b>Note</b>: Observation Mode output (interpretive only).</div>
  </div>
</div>
"""

    filters_html = ""
    try:
        filters_js
    except NameError:
        filters_js = ""

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Market Intelligence Dashboard</title>
<style>
  body {{ font-family: Arial, sans-serif; padding: 16px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; white-space: nowrap; }}
  th {{ background: #f5f5f5; position: sticky; top: 0; z-index: 2; }}
  tr.game-row:hover {{ background: #f0f0f0; }}
</style>
{filters_js}
</head>
<body>
<h1>Market Intelligence Dashboard</h1>
{snapshot_html}
{legend_html}
{filters_html}

<div style="overflow:auto; border:1px solid #ddd; border-radius:8px;">
<table id="dashboard">
  <thead>
    <tr data-header="1">
      {header_ths}
    </tr>
  </thead>
  <tbody>
    {''.join(rows_html)}
  </tbody>
</table>
</div>

<script>
(function () {{
  try {{
    const table = document.getElementById("dashboard");
    if (!table) return;
    const tbody = table.querySelector("tbody");
    if (!tbody) return;

    function getCellValue(tr, idx) {{
      const td = tr.children[idx];
      return td ? (td.innerText || td.textContent || "").trim() : "";
    }}

    function asNumber(s) {{
      const x = String(s || "").replace(/[%,$]/g, "").replace(/—/g, "").trim();
      const n = parseFloat(x);
      return Number.isNaN(n) ? null : n;
    }}

    function comparer(idx, asc) {{
      return function (a, b) {{
        const va = getCellValue(asc ? a : b, idx);
        const vb = getCellValue(asc ? b : a, idx);
        const na = asNumber(va), nb = asNumber(vb);
        if (na !== null && nb !== null) return na - nb;
        return va.localeCompare(vb);
      }};
    }}

    const headerRow = table.querySelector('thead tr[data-header="1"]');
    if (!headerRow) return;

    Array.from(headerRow.children).forEach(function (cell, idx) {{
      cell.style.cursor = "pointer";
      cell.addEventListener("click", function () {{
        const rows = Array.from(tbody.querySelectorAll("tr"));
        const asc = cell.dataset.asc !== "1";
        cell.dataset.asc = asc ? "1" : "0";
        rows.sort(comparer(idx, asc));
        rows.forEach(r => tbody.appendChild(r));
      }});
    }});
  }} catch (e) {{
    // swallow errors so dashboard still renders
  }}
}})();
</script>

</body>
</html>
"""


    # -----------------------------
    # WRITE DASHBOARD HTML (single source of truth)
    # -----------------------------
    try:
        from pathlib import Path
        ensure_data_dir()
        Path(REPORT_HTML).write_text(html, encoding="utf-8")
        print(f"[ok] wrote dashboard: {REPORT_HTML}")
    except Exception as e:
        print(f"[warn] failed to write dashboard HTML: {e}")

# =========================
# CLI
# =========================


def _strong_flags(row, market, pb_map=None):
    """v1.1 STRONG certification gate."""
    try:
        m = str(market).strip().upper()
        sp = str(row.get('sport','')).strip().upper()
        game_id = str(row.get('game_id','')).strip()

        score_col = f"{m}_model_score"
        try:
            score = float(str(row.get(score_col, '')).strip() or 'nan')
        except Exception:
            score = float('nan')
        if not (score == score) or score < 72.0:
            return False, 'score_lt_72'

        tb = str(row.get('timing_bucket','')).strip().upper()
        if tb == 'LATE':
            return False, 'late_block'

        p_ok = row.get(f"{m}_persist_ok", False)
        s_ok = row.get(f"{m}_stable_ok", False)

        p_ok = bool(p_ok) if isinstance(p_ok,(bool,int)) else str(p_ok).lower() in ('true','1','yes')
        s_ok = bool(s_ok) if isinstance(s_ok,(bool,int)) else str(s_ok).lower() in ('true','1','yes')

        if not p_ok:
            return False, 'no_persistence'
        if not s_ok:
            return False, 'unstable'

        try:
            side = str(row.get(f"{m}_side") or row.get(f"{m}_favored") or '').strip()
            if pb_map and side and sp and game_id:
                prev = pb_map.get(f"{sp}|{game_id}|{m}|{side}", '')
        except Exception:
            pass

        spread_fav = str(row.get('SPREAD_favored','')).strip()
        ml_fav = str(row.get('MONEYLINE_favored','')).strip()
        if m in ('SPREAD','MONEYLINE') and spread_fav and ml_fav and spread_fav != ml_fav:
            return False, 'cross_market_contradiction'

        txt = " ".join(
            str(row.get(c,"")).upper()
            for c in (
                f"{m}_market_read", f"{m}_market_why", f"{m}_why",
                "market_read", "market_why", "why"
            )
        )
        if "PUBLIC DRIFT" in txt or "LINE INFLATION" in txt:
            return False, 'public_drift_block'

        return True, ''
    except Exception:
        return False, 'strong_flags_exception'


def cmd_snapshot(args):
    # No hard skips here.
    # If a sport has no games, dk_headless/get_splits should return 0 records,
    # and we will print "[snapshot] no games available for <sport>" below.

    """v1.1 STRONG certification gate.
    Returns (eligible: bool, reason: str)
    market in {SPREAD,TOTAL,MONEYLINE}
    """
    try:
        m = str(market).strip().upper()
        sp = str(row.get('sport','')).strip().upper()
        game_id = str(row.get('game_id','')).strip()

        # Score must be >= 72 to even be considered
        score_col = f"{m}_model_score"
        try:
            score = float(str(row.get(score_col, '')).strip() or 'nan')
        except Exception:
            score = float('nan')
        if not (score == score) or score < 72.0:
            return False, 'score_lt_72'

        # Timing: no STRONG in LATE (invariant)
        tb = str(row.get('timing_bucket','')).strip().upper()
        if tb == 'LATE':
            return False, 'late_block'

        # Persistence / stability (prefer per-market flags computed from row_state)
        p_ok = row.get(f"{m}_persist_ok", False)
        s_ok = row.get(f"{m}_stable_ok", False)
        try:
            p_ok = bool(p_ok) if isinstance(p_ok, (bool,int)) else str(p_ok).strip().lower() in ('true','1','yes')
        except Exception:
            p_ok = False
        try:
            s_ok = bool(s_ok) if isinstance(s_ok, (bool,int)) else str(s_ok).strip().lower() in ('true','1','yes')
        except Exception:
            s_ok = False
        if not p_ok:
            return False, 'no_persistence'
        if not s_ok:
            return False, 'unstable'

        # LATE 'no NEW STRONG' rule is already enforced by tb==LATE, but keep prior-bucket hook for safety.
        # If you ever loosen tb handling later, this will still prevent 'new' strong in LATE.
        prev_bucket = ''
        try:
            # Prefer per-market side column if present; else favored
            side = str(row.get(f"{m}_side", '')).strip() or str(row.get(f"{m}_favored", '')).strip()
            if pb_map is not None and side and sp and game_id:
                kk = f"{sp}|{game_id}|{m}|{side}"
                prev_bucket = str(pb_map.get(kk,'')).strip().upper()
        except Exception:
            prev_bucket = ''

        # Cross-market non-contradiction (only apply to SPREAD <-> MONEYLINE)
        # If favored sides disagree, block STRONG.
        try:
            spread_fav = str(row.get('SPREAD_favored','')).strip()
            ml_fav = str(row.get('MONEYLINE_favored','')).strip()
            if m in ('SPREAD','MONEYLINE') and spread_fav and ml_fav and spread_fav != ml_fav:
                return False, 'cross_market_contradiction'
        except Exception:
            pass

        # Public Drift / Line Inflation block
        # We don't assume a specific schema: we scan common per-market columns if present.
        def _get_txt(*cols):
            out = []
            for c in cols:
                if c and c in row and row.get(c, '') not in (None, ''):
                    out.append(str(row.get(c, '')).upper())
            return ' | '.join(out)

        txt = _get_txt(f"{m}_market_read", f"{m}_market_why", f"{m}_why", 'market_read', 'market_why', 'why')
        if 'PUBLIC DRIFT' in txt or 'LINE INFLATION' in txt:
            return False, 'public_drift_block'

        return True, ''
    except Exception:
        return False, 'strong_flags_exception'

# --- end v1.1 STRONG CERTIFICATION HELPERS ---

    dash["strong_eligible"] = False
    dash["strong_block_reason"] = ""
# --- v1.1 STRONG CERTIFICATION WIRING (DO NOT EDIT BY HAND) ---
    # Compute per-market STRONG eligibility (v1.1 spec)
    # Writes: {M}_strong_eligible, {M}_strong_block_reason
    for _m in ("SPREAD","TOTAL","MONEYLINE"):
        dash[f"{_m}_strong_eligible"] = False
        dash[f"{_m}_strong_block_reason"] = ""

    try:
        # _pb should already be built above (row_state prior bucket map). If not, use empty.
        _pb_map = _pb if isinstance(_pb, dict) else {}
    except Exception:
        _pb_map = {}

    try:
        _elig_cols = {}
        for _m in ("SPREAD","TOTAL","MONEYLINE"):
            elig = []
            rsn = []
            for _, _r in dash.iterrows():
                ok, why = _strong_flags(_r, _m, _pb_map)
                elig.append(bool(ok))
                rsn.append(str(why or ""))
            dash[f"{_m}_strong_eligible"] = elig
            dash[f"{_m}_strong_block_reason"] = rsn

        # Preserve existing global columns conservatively: use SPREAD market as primary.
        if "SPREAD_strong_eligible" in dash.columns:
            dash["strong_eligible"] = dash["SPREAD_strong_eligible"].astype(bool)
            dash["strong_block_reason"] = dash.get("SPREAD_strong_block_reason", "")
    except Exception:
        pass
    # --- end v1.1 STRONG CERTIFICATION WIRING ---

    # --- v1.1: attach prior bucket from row_state (used for LATE "no NEW STRONG" rule) ---
    dash["prev_bucket"] = ""
    try:
        import pandas as _pd
        _sp = os.path.join(DATA_DIR, 'row_state.csv')
        if os.path.exists(_sp):
            _rs = _pd.read_csv(_sp, keep_default_na=False, dtype=str)
            if all(c in _rs.columns for c in ['sport','game_id','market','side','last_bucket']):
                _rs['sport'] = _rs['sport'].astype(str).str.strip().str.upper()
                _rs['market'] = _rs['market'].astype(str).str.strip().str.upper()
                _rs['game_id'] = _rs['game_id'].astype(str).str.strip()
                _rs['side'] = _rs['side'].astype(str).str.strip()
                _rs['_k'] = _rs[['sport','game_id','market','side']].astype(str).agg('|'.join, axis=1)
                _pb = dict(zip(_rs['_k'], _rs['last_bucket'].astype(str)))
                _p_peak = dict(zip(_rs['_k'], _rs.get('peak_score', '').astype(str)))
                _p_last = dict(zip(_rs['_k'], _rs.get('last_score', '').astype(str)))
                # dashboard is WIDE: each market has its own side column
                _prev = []
                for _, _r in dash.iterrows():
                    sp = str(_r.get('sport','')).strip().upper()
                    gid = str(_r.get('game_id','')).strip()
                    # choose a side (prefer spread side, else total, else moneyline) just to create a stable key
                    side = str(_r.get('SPREAD_side') or _r.get('TOTAL_side') or _r.get('MONEYLINE_side') or '').strip()
                    # market will be set inside strong loop; default here is empty
                    _prev.append('')
                dash['prev_bucket'] = _prev
                # We will fill prev_bucket per-market in the STRONG loop below (more accurate).
    except Exception:
        pass
    # --- end v1.1 ---

    # --- v1.1: map prior bucket from row_state for LATE "no NEW STRONG" rule ---
    _pb = {}
    try:
        import pandas as _pd
        _rsp = os.path.join(DATA_DIR, 'row_state.csv')
        if os.path.exists(_rsp):
            _rs = _pd.read_csv(_rsp, keep_default_na=False, dtype=str)
            need = ['sport','game_id','market','side','last_bucket']
            if all(c in _rs.columns for c in need):
                _rs['sport']  = _rs['sport'].astype(str).str.strip().str.upper()
                _rs['market'] = _rs['market'].astype(str).str.strip().str.upper()
                _rs['game_id']= _rs['game_id'].astype(str).str.strip()
                _rs['side']   = _rs['side'].astype(str).str.strip()
                _rs['_k'] = _rs[['sport','game_id','market','side']].astype(str).agg('|'.join, axis=1)
                _pb = dict(zip(_rs['_k'], _rs['last_bucket'].astype(str)))
    except Exception:
        _pb = {}
        _p_peak = {}
        _p_last = {}
    # --- end v1.1 ---

    if False:  # legacy STRONG eligibility loop disabled (v1.1 uses per-market _strong_flags)
        for mkt in ("SPREAD", "TOTAL", "MONEYLINE"):
            sc_col = f"{mkt}_model_score"
            if sc_col not in dash.columns:
                continue

            elig = []
            reason = []

            for _, rr in dash.iterrows():
                rr2 = rr.copy()
                sp = str(rr2.get("sport","")).strip().upper()
                gid = str(rr2.get("game_id","")).strip()
                side = str(rr2.get(f"{mkt}_side","") or "").strip()
                k = f"{sp}|{gid}|{mkt}|{side}"
                rr2["prev_bucket"] = _pb.get(k, "")
                rr2["_rs_peak"] = _p_peak.get(k, "")
                rr2["_rs_last"] = _p_last.get(k, "")
                elig.append(ok)
                reason.append(why)

            dash[f"{mkt}_strong_eligible"] = elig
            dash[f"{mkt}_strong_block_reason"] = reason
    # --- end v1.1 STRONG certification flags ---


    # --- v1.1: ENFORCE STRONG eligibility into decisions (not just flags) ---
    # If a market is labeled STRONG BET by score/edge, but fails STRONG certification,
    # downgrade decision to BET. This makes NCAAB EARLY/LATE blocks and other STRONG gates real.
    try:
        for _mkt in ("SPREAD", "TOTAL", "MONEYLINE"):
            _dec = f"{_mkt}_decision"
            _elig = f"{_mkt}_strong_eligible"
            if _dec in dash.columns and _elig in dash.columns:
                try:
                    _is_strong = dash[_dec].astype(str).str.upper().eq("STRONG BET")
                    _ok = dash[_elig].astype(bool)
                    _mask = _is_strong & (~_ok)
                    if int(_mask.sum()) > 0:
                        dash.loc[_mask, _dec] = "BET"
                except Exception:
                    pass
    except Exception:
        pass
    # --- end v1.1 STRONG decision enforcement ---



        # --- v1.1 STRONG CERTIFICATION (module-scope _strong_flags) ---
    try:
        _pb_map = _pb if isinstance(_pb, dict) else {}
    except Exception:
        _pb_map = {}

    for _m in ("SPREAD","TOTAL","MONEYLINE"):
        elig = []
        rsn = []
        for _, _r in dash.iterrows():
            ok, why = _strong_flags(_r, _m, _pb_map)
            elig.append(bool(ok))
            rsn.append(str(why or ""))
        dash[f"{_m}_strong_eligible"] = elig
        dash[f"{_m}_strong_block_reason"] = rsn

        # Enforce downgrade
        _dec = f"{_m}_decision"
        if _dec in dash.columns:
            mask = (
                dash[_dec].astype(str).str.upper().eq("STRONG BET")
                & (~dash[f"{_m}_strong_eligible"])
            )
            if int(mask.sum()) > 0:
                dash.loc[mask, _dec] = "BET"

    # Global convenience columns (SPREAD is canonical)
    dash["strong_eligible"] = dash.get("SPREAD_strong_eligible", False)
    dash["strong_block_reason"] = dash.get("SPREAD_strong_block_reason", "")
    # --- end v1.1 STRONG CERTIFICATION ---

# REFRESH present/extras AFTER adding dashboard-only flags


    try:


        present = [c for c in col_order if c in dash.columns]


        extras = [c for c in dash.columns if c not in present]


        dash = dash[present + extras]


    except Exception:


        pass


    # --- v1.1 STRONG CERTIFICATION FINAL OVERWRITE (DO NOT EDIT BY HAND) ---
    # Force canonical per-market STRONG eligibility + reasons right before dashboard.csv write.
    # This prevents any earlier legacy logic from overwriting the final columns.
    try:
        _pb_map = _pb if isinstance(_pb, dict) else {}
    except Exception:
        _pb_map = {}
    
    try:
        for _m in ("SPREAD","TOTAL","MONEYLINE"):
            elig = []
            rsn = []
            for _, _r in dash.iterrows():
                ok, why = _strong_flags(_r, _m, _pb_map)
                elig.append(bool(ok))
                rsn.append(str(why or ""))
            dash[f"{_m}_strong_eligible"] = elig
            dash[f"{_m}_strong_block_reason"] = rsn
    
        # keep global convenience columns aligned to SPREAD
        dash["strong_eligible"] = dash.get("SPREAD_strong_eligible", False)
        dash["strong_block_reason"] = dash.get("SPREAD_strong_block_reason", "")
    except Exception:
        pass
    # --- end v1.1 STRONG CERTIFICATION FINAL OVERWRITE ---



    # All sports use the same DK Network splits page
    url = SPORT_CONFIG[args.sport]["url"]



    result = get_splits(
    url,
    args.sport,
    debug_dump_path=f"data/dk_rendered_{args.sport}.html"

)


    # Use the records extracted by dk_headless (JSON or DOM)
    rows = result.get("records", [])
    for r in rows:
        r["sport"] = args.sport
    if not rows:
        print(f"[snapshot] no games available for {args.sport}")
        return



    if args.debug:
        logger.debug("json_candidates_found=%s", result.get("json_candidates_found"))
        logger.debug("json_records_found=%s", result.get("json_records_found"))
        logger.debug("extracted_records=%d", len(rows))

    # --- Model Open (persistent) ---
    # DK Network Splits often has no true "open". We define open_line as the first observed current_line
    # per (sport, game_id, market, side) and persist it across runs.
    open_reg = _load_open_registry()
    new_opens = 0
    for r in rows:  
        sport = str(r.get("sport","") or "")
        game_id = str(r.get("game_id","") or "")
        market = str(r.get("market","") or "")
        side = str(r.get("side","") or "")
        current_line = str(r.get("current_line","") or r.get("current","") or "").strip()
        k = (sport, game_id, market, side)

        # If registry has it, use it; else seed from current_line
        if k in open_reg and open_reg[k]:
            r["open_line"] = open_reg[k]
        else:
            r["open_line"] = current_line
            if current_line:
                open_reg[k] = current_line
                new_opens += 1

    if new_opens:
        _save_open_registry(open_reg)
        logger.info(f"[open] seeded {new_opens} model-open lines into {OPEN_REG_PATH.as_posix()}")
    # --- /Model Open ---

    append_snapshot(rows, args.sport)
    print(f"[ok] appended {len(rows)} rows to {SNAPSHOT_CSV}")


    build_dashboard()
    resolve_results_for_baseline()



def cmd_report(_args):
    # ESPN finals (results) update is best-effort; never block dashboard build
    try:
        update_snapshots_with_espn_finals()
    except KeyboardInterrupt:
        print("[espn finals] skipped (KeyboardInterrupt)")
    except Exception as e:
        print(f"[espn finals] skipped due to error: {repr(e)}")
    build_dashboard()
    resolve_results_for_baseline()
    build_color_baseline_summary()



def cmd_movement(args):
    movement_report(
        SNAPSHOT_CSV,
        args.sport,
        lookback=args.lookback,
    )

def cmd_baseline_market_read_joined(args):
    print("[baseline_market_read_joined] loading files")

    # Load inputs
    signals = pd.read_csv("data/signals_baseline.csv", parse_dates=["logged_at_utc"])
    snapshots = pd.read_csv("data/snapshots.csv", parse_dates=["timestamp"])
    results = pd.read_csv("data/results_resolved.csv")

    # SPREADS only (signals already normalized)
    signals = signals[signals["market"] == "SPREAD"].copy()

    # Normalize snapshot market (same logic as dashboard)
    snapshots["market_norm"] = snapshots.apply(
        lambda r: infer_market_type(r.get("side", ""), r.get("current_line", "")),
        axis=1
    )

    # SPREADS only (snapshots, normalized)
    snapshots = snapshots[snapshots["market_norm"] == "SPREAD"].copy()

    print(f"[baseline_market_read_joined] signals (SPREAD): {len(signals)}")
    print(f"[baseline_market_read_joined] snapshots (SPREAD): {len(snapshots)}")
    print(f"[baseline_market_read_joined] results rows: {len(results)}")


    # =========================
    # B3: Nearest snapshot â‰¤ signal time
    # =========================

    # Align snapshot timestamp name to match signal time
    snapshots = snapshots.rename(columns={"timestamp": "snapshot_time"})

    # Ensure proper datetime types
    snapshots["snapshot_time"] = pd.to_datetime(snapshots["snapshot_time"], utc=True)
    signals["logged_at_utc"] = pd.to_datetime(signals["logged_at_utc"], utc=True)

    # Sort for asof merge
    snapshots = snapshots.sort_values("snapshot_time")
    signals = signals.sort_values("logged_at_utc")

    # Keep only join-relevant snapshot columns
    snap_cols = [
        "sport",
        "game_id",
        "side",
        "snapshot_time",
        "bets_pct",
        "money_pct",
        "open_line",
        "current_line",
        "key_number_note",
    ]
    snapshots_j = snapshots[snap_cols].copy()

    print("[baseline_market_read_joined] performing asof join")

    # Nearest snapshot at or before signal time
    joined = pd.merge_asof(
        signals,
        snapshots_j,
        left_on="logged_at_utc",
        right_on="snapshot_time",
        by=["sport", "game_id", "side"],
        direction="backward",
        tolerance=pd.Timedelta("48h"),
    )

    matched = joined["snapshot_time"].notna().sum()
    total = len(joined)

    print(f"[baseline_market_read_joined] matched snapshots: {matched}/{total}")

    # =========================
    # B4: Recompute Market Read at signal time
    # =========================

    # Compute divergence D
    joined["divergence_D"] = joined["money_pct"] - joined["bets_pct"]

    market_reads = []
    market_whys = []

    for _, r in joined.iterrows():
        D = r["divergence_D"]
        bets = r["bets_pct"]
        money = r["money_pct"]

        # Parse numeric spread values from line strings
        try:
            open_parsed = parse_line_and_odds(r.get("open_line"))
            cur_parsed = parse_line_and_odds(r.get("current_line"))

            open_val = open_parsed.get("line_val")
            cur_val = cur_parsed.get("line_val")
        except Exception:
            open_val = None
            cur_val = None

        # Movement direction (SPREAD)
        move_dir = _toward_side_by_spread(open_val, cur_val)

        # Key-number awareness
        key_cross = _crossed_key(
            abs(open_val) if open_val is not None else None,
            abs(cur_val) if cur_val is not None else None
        )

        meaningful = (
            open_val is not None
            and cur_val is not None
            and (
                abs(cur_val - open_val) >= 0.5
                or key_cross != ""
            )
        )


        # High-bet side check (simple, safe)
        is_high_bet_side = False
        try:
            is_high_bet_side = bets >= 50
        except Exception:
            pass

        label = _classify_market_read(
            D=D,
            bets_pct=bets,
            move_dir=move_dir,
            meaningful_move=meaningful,
            is_high_bet_side=is_high_bet_side
        )

        market_reads.append(label)

        market_whys.append(
            f"{label}: bets={bets:.1f}%, money={money:.1f}%, "
            f"D={D:+.1f}, move_dir={move_dir}"
            + (f", key={key_cross}" if key_cross else "")
        )

    joined["market_read"] = market_reads
    joined["market_why"] = market_whys

    print(
        "[baseline_market_read_joined] market_read distribution:",
        joined["market_read"].value_counts().to_dict()
    )

    # =========================
    # B5: Join outcomes + summarize by Market Read
    # =========================

        # Deduplicate outcomes: keep latest logged outcome per signal
    results_latest = (
        results
        .sort_values("logged_at_utc")
        .groupby(["sport", "game_id", "market", "side"], as_index=False)
        .tail(1)
    )


    # Join deduplicated outcomes
    joined = joined.merge(
        results_latest,
        on=["sport", "game_id", "market", "side"],
        how="left"
    )


    # Keep only resolved outcomes
    decided = joined[joined["outcome"].isin(["WIN", "LOSS", "PUSH"])].copy()

    summary = (
        decided
        .groupby("market_read")
        .agg(
            n=("outcome", "count"),
            wins=("outcome", lambda x: (x == "WIN").sum()),
            losses=("outcome", lambda x: (x == "LOSS").sum()),
            pushes=("outcome", lambda x: (x == "PUSH").sum()),
        )
        .reset_index()
    )

    summary["win_rate_ex_push"] = (
        summary["wins"] / (summary["wins"] + summary["losses"])
    )

    print("\n[baseline_market_read_joined] performance by Market Read:")
    print(summary.sort_values("n", ascending=False))





def cmd_baseline_market_read(args):
    import pandas as pd

    # Inputs (ledgers)
    sig = pd.read_csv("data/signals_baseline.csv")
    res = pd.read_csv("data/results_resolved.csv")


    # Optional: exclude dark green high ML underdogs (flag column name may differ)
    if args.exclude_high_ml_underdogs:
        for col in ["is_dark_green_high_underdog", "is_dark_green_high_underdog_ml"]:
            if col in sig.columns:
                sig = sig[~sig[col].fillna(False)]
                break

    # Join on your dedupe key (source of truth for uniqueness)
    base_key = ["sport", "game_id", "market", "side", "color"]
    key = base_key + (["espn_day"] if ("espn_day" in sig.columns and "espn_day" in res.columns) else [])

    print("[baseline_market_read] using key:", key)
    print("[baseline_market_read] signals cols has espn_day?", "espn_day" in sig.columns,
      "| results cols has espn_day?", "espn_day" in res.columns)


    s2 = sig.drop_duplicates(subset=key).copy()
    r2 = res.drop_duplicates(subset=key).copy()

    df = s2.merge(
        r2[key + ["outcome"]],
        on=key,
        how="left"
    )

    # Outcome normalization (just in case)
    df["outcome"] = df["outcome"].fillna("UNRESOLVED")

    # Focus option: spreads only
    if args.spreads_only:
        df = df[df["market"].astype(str).str.upper().str.contains("SPREAD")]

    # Summary by Market Read label (if column exists)
    if "market_read" not in df.columns:
        print("[baseline_market_read] market_read not in signals_baseline.csv (expected). Running COLOR/MARKET baseline summary instead.")

    summary = (
        df.groupby(["market", "color"], dropna=False)
          .agg(
              n=("outcome", "size"),
              wins=("outcome", lambda x: (x == "WIN").sum()),
              losses=("outcome", lambda x: (x == "LOSS").sum()),
              pushes=("outcome", lambda x: (x == "PUSH").sum()),
              unresolved=("outcome", lambda x: (x == "UNRESOLVED").sum()),
          )
          .reset_index()
    )
    summary["decided"] = summary["wins"] + summary["losses"] + summary["pushes"]
    summary["win_rate_ex_push"] = summary.apply(
        lambda r: (r["wins"] / (r["wins"] + r["losses"])) if (r["wins"] + r["losses"]) > 0 else None,
        axis=1
    )

    out_path = "data/color_market_baseline_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"[ok] wrote baseline color/market summary: {out_path}")
    return


    summary = (
        df.groupby(["market_read", "market", "color"], dropna=False)
          .agg(
              n=("outcome", "size"),
              wins=("outcome", lambda x: (x == "WIN").sum()),
              losses=("outcome", lambda x: (x == "LOSS").sum()),
              pushes=("outcome", lambda x: (x == "PUSH").sum()),
              unresolved=("outcome", lambda x: (x == "UNRESOLVED").sum()),
          )
          .reset_index()
    )
    summary["decided"] = summary["wins"] + summary["losses"] + summary["pushes"]
    summary["win_rate_ex_push"] = summary.apply(
        lambda r: (r["wins"] / (r["wins"] + r["losses"])) if (r["wins"] + r["losses"]) > 0 else None,
        axis=1
    )

    out_path = "data/market_read_baseline_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"[ok] wrote baseline market_read summary: {out_path}")


def log_baseline_signal(row):

    # Fast-path: build baseline "seen" cache once (avoid O(N^2) scans)
    global _BASELINE_SEEN_KEYS
    if _BASELINE_SEEN_KEYS is None:
        try:
            import pandas as _pd
            rs = _pd.read_csv("data/row_state.csv", keep_default_na=False, dtype=str)
            cols = set(rs.columns)
            sport_c = "sport" if "sport" in cols else None
            game_c  = "game_id" if "game_id" in cols else None
            mkt_c   = "market" if "market" in cols else ("market_display" if "market_display" in cols else ("_market_display" if "_market_display" in cols else None))
            side_c  = "side" if "side" in cols else None
            bucket_c = "last_bucket" if "last_bucket" in cols else ("score_bucket" if "score_bucket" in cols else None)

            def _get(r, c):
                return (str(r.get(c, "")).strip() if c else "")

            seen = set()
            for _, rr in rs.iterrows():
                seen.add((
                    _get(rr, sport_c).lower(),
                    _get(rr, game_c),
                    _get(rr, mkt_c).upper(),
                    _get(rr, side_c),
                    _get(rr, bucket_c),
                ))
            _BASELINE_SEEN_KEYS = seen
        except Exception:
            _BASELINE_SEEN_KEYS = set()

    try:
        _sport = str(row.get("sport","")).strip().lower()
        _gid   = str(row.get("game_id","")).strip()
        _mkt   = str(row.get("_market_display", row.get("market_display", row.get("market","")))).strip().upper()
        _side  = str(row.get("side","")).strip()
        _bucket = str(row.get("score_bucket", row.get("last_bucket",""))).strip()
        _k = (_sport, _gid, _mkt, _side, _bucket)
        if _k in _BASELINE_SEEN_KEYS:
            return
        _BASELINE_SEEN_KEYS.add(_k)
    except Exception:
        pass
    import csv, os
    from datetime import datetime, timezone

    BASELINE_FILE = "data/signals_baseline.csv"
    market_key = row.get("market")

    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if (
                    r.get("logic_version") == "v0.1"
                    and r.get("sport") == row.get("sport")
                    and r.get("game_id") == str(row.get("game_id"))
                    and r.get("market") == market_key
                    and r.get("side") == row.get("side")
                ):
                    return



    if row.get("color") in (None, "", "GREY"):
        return
        # require game time (freeze ~30 min pregame)
    if not row.get("game_time_iso"):
        return

    try:
        game_time = datetime.fromisoformat(row["game_time_iso"].replace("Z", "+00:00"))
    except Exception:
        return

    minutes_to_start = (game_time - datetime.now(timezone.utc)).total_seconds() / 60

    # only log once between 30 and 25 minutes pregame
    if minutes_to_start > 30 or minutes_to_start < 25:
        return



    # --- model score snapshot for result tracking ---
    model_score_val = row.get("score_num")
    if model_score_val is None:
        model_score_val = row.get("_score_num")
    if model_score_val is None:
        model_score_val = row.get("score_num")
    if model_score_val is None:
        model_score_val = row.get("score")
    try:
        model_score = float(model_score_val) if model_score_val not in (None, "") else None
    except Exception:
        model_score = None
    if model_score is None:
        model_score_bucket = ""
    else:
        model_score_bucket = (
            "65+" if model_score >= 65 else
            "60-64" if model_score >= 60 else
            "55-59" if model_score >= 55 else
            "50-54"
        )

    file_exists = os.path.exists(BASELINE_FILE)

    with open(BASELINE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "logged_at_utc",
                "logic_version",
                "sport",
                "game_id",
                "game",
                "market",
                "side",
                "color",
                  "model_score",
                  "model_score_bucket",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "logged_at_utc": datetime.now(timezone.utc).isoformat(),
            "logic_version": "v0.1",
            "sport": row.get("sport"),
            "game_id": row.get("game_id"),
            "game": row.get("game"),
            "market": row.get("market"),
            "side": row.get("side"),
            "color": row.get("color"),
            "model_score": "" if model_score is None else model_score,
            "model_score_bucket": model_score_bucket,
        })

def resolve_results_for_baseline():
    import pandas as pd
    import os

    baseline_file = "data/signals_baseline.csv"
    snapshot_file = "data/snapshots.csv"
    out_file = "data/results_resolved.csv"

    if not os.path.exists(baseline_file) or not os.path.exists(snapshot_file):
        return

    base = pd.read_csv(baseline_file)
    snaps = pd.read_csv(snapshot_file, keep_default_na=False, dtype=str)

    # Only finished games (must have final score)
        # If final scores are not present yet, exit safely
    if "final_score_for" not in snaps.columns or "final_score_against" not in snaps.columns:
        return

    snaps["final_score_for"] = snaps["final_score_for"].fillna("").astype(str).str.strip()
    snaps["final_score_against"] = snaps["final_score_against"].fillna("").astype(str).str.strip()

    snaps = snaps[(snaps["final_score_for"] != "") & (snaps["final_score_against"] != "")]

    if snaps.empty:
        return

    # Build result key
    snaps["result"] = snaps.apply(
        lambda r: "WIN" if r["final_score_for"] > r["final_score_against"]
        else "LOSS" if r["final_score_for"] < r["final_score_against"]
        else "PUSH",
        axis=1
    )

    merged = base.merge(
        snaps[["game_id", "side", "result"]],
        on=["game_id", "side"],
        how="left"
    )

    merged.to_csv(out_file, index=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)

    s1 = sub.add_parser("snapshot", help="Fetch splits + append a snapshot + rebuild dashboard")
    s1.add_argument("--sport", choices=SPORT_CONFIG.keys(), required=True)
    s1.add_argument("--debug", action="store_true", help="Print debug info to help tune parsing")
    s1.set_defaults(func=cmd_snapshot)
    s4 = sub.add_parser("backfill_baseline", help="ONE-TIME historical backfill for baseline signals (since yesterday)")
    s4.add_argument("--since", required=True, help="YYYY-MM-DD (America/New_York). Do not backfill earlier than model start.")
    s4.add_argument("--label", default=None, help="Label stored in results_resolved.csv (defaults to autogenerated)")
    s4.add_argument("--force", action="store_true", help="Override sentinel lock (not recommended)")
    s4.set_defaults(func=cmd_backfill_baseline)


    s2 = sub.add_parser("report", help="Rebuild dashboard from existing snapshots")
    s2.set_defaults(func=cmd_report)
    s3 = sub.add_parser("movement", help="Compare snapshots")
    s3.add_argument("--sport", choices=SPORT_CONFIG.keys(), required=True)
    s3.add_argument("--lookback", type=int, default=1, help="How many snapshots back to compare (1 = most recent previous)")
    s3.set_defaults(func=cmd_movement)
    s5 = sub.add_parser("baseline_market_read", help="Fast baseline summary by Market Read (no dashboard)")
    s5.add_argument("--exclude-high-ml-underdogs", action="store_true",
                    help="Exclude dark green high moneyline underdogs (if flag exists in signals)")
    s5.add_argument("--spreads-only", action="store_true", help="Only include SPREAD markets")
    s5.set_defaults(func=cmd_baseline_market_read)
    s6 = sub.add_parser(
    "baseline_market_read_joined",
    help="Join baseline signals to snapshots at signal time and summarize outcomes by Market Read (analysis-only)"
)
    s6.set_defaults(func=cmd_baseline_market_read_joined)



    args = ap.parse_args()

    logger = setup_logger(getattr(args, "debug", False))
    logger.info("command=%s sport=%s", args.func.__name__, getattr(args, "sport", None))
    args.func(args)

def build_color_baseline_summary():
    import pandas as pd
    import os

    out = "data/color_baseline_summary.csv"
    src = "data/results_resolved.csv"

    # always create the file so it exists
    pd.DataFrame().to_csv(out, index=False)

    if not os.path.exists(src):
        return
    
    try:
        df = pd.read_csv(src, keep_default_na=False, dtype=str)
    except Exception as e:
        print(f"[finals] read failed: {e}")
        return

    if df.empty or "color" not in df.columns or "result" not in df.columns:
        return


    summary = (
        df.dropna(subset=["result"])
          .groupby("color")["result"]
          .value_counts()
          .unstack(fill_value=0)
          .reset_index()
    )

    summary["total"] = summary.sum(axis=1, numeric_only=True)
    if "WIN" in summary.columns:
        summary["win_pct"] = (summary["WIN"] / summary["total"]).round(3)

    summary.to_csv(out, index=False)

if __name__ == "__main__":
    main()
















