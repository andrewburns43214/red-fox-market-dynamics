import argparse
import csv
import math
from collections import defaultdict

# --- BASELINE LOG CACHE ---
_BASELINE_SEEN_KEYS = None  # set[(sport, game_id, market, side, bucket)]

import datetime as dt
import os
import re

from pathlib import Path

# v2.0: import shared team normalization (also defined locally for backward compat)
try:
    from team_aliases import (
        _split_game as _split_game_mod,
        _norm_team as _norm_team_mod,
        normalize_team_name as normalize_team_name_mod,
        TEAM_ALIASES as TEAM_ALIASES_MOD,
    )
except ImportError:
    _split_game_mod = None

# v2.0: import centralized config (backward compat: fall back to local definitions)
try:
    from engine_config_v3 import V3_VERSION as _CFG_LOGIC_VERSION
    from engine_config import (
        STALE_TICK_THRESHOLD,
    )
except ImportError:
    _CFG_LOGIC_VERSION = None

OPEN_REG_PATH = Path("data") / "open_registry.csv"

def _load_open_registry() -> dict:
    """
    Key: (sport, game_id, market, mkt_type, side) -> open_line string
    Persistent across runs so Open is stable even when lines move.
    mkt_type: ml/spread/total — distinguishes markets that share market="splits"
    """
    reg = {}
    if not OPEN_REG_PATH.exists():
        return reg
    import csv
    with OPEN_REG_PATH.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            mkt_type = row.get("mkt_type", "")
            if mkt_type:
                k = (row.get("sport",""), row.get("game_id",""), row.get("market",""), mkt_type, row.get("side",""))
            else:
                # Legacy 4-tuple rows — skip (will be re-seeded with 5-tuple)
                continue
            reg[k] = (row.get("open_line","") or "").strip()
    return reg

def _save_open_registry(reg: dict) -> None:
    import csv
    OPEN_REG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OPEN_REG_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sport","game_id","market","mkt_type","side","open_line"])
        for k, open_line in sorted(reg.items()):
            if len(k) == 5:
                sport, game_id, market, mkt_type, side = k
            else:
                continue  # Skip legacy 4-tuple entries
            w.writerow([sport, game_id, market, mkt_type, side, open_line])
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

def _validate_iso_tz(s: str) -> str:
    """Ensure ISO datetime string has timezone info. Assumes UTC if naive."""
    s = str(s).strip() if s else ""
    if not s or s.lower() == "nan":
        return ""
    # If it looks like an ISO datetime but has no timezone indicator, append Z
    if "T" in s and "Z" not in s and "+" not in s and "-" not in s[11:]:
        s = s + "Z"
    return s

def compute_minutes_to_kickoff(row: dict):
    """
    Uses existing ESPN kickoff field already on the row.
    IMPORTANT: We are NOT touching snapshot timestamps.
    """
    kickoff_iso = (
        row.get("dk_start_iso")
        or row.get("game_time_iso")
        or row.get("espn_kickoff_iso")
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
LOGIC_VERSION = _CFG_LOGIC_VERSION if _CFG_LOGIC_VERSION else "v3.3f"

# ============================================================
# v1.1 STEP 2 ÃƒÂ¢Ã‚Â€Ã‚Â” SPORT-SPECIFIC DAMPENERS (INSTRUMENTATION ONLY)
# ============================================================

# NOTE:
# - NO score math changes
# - NO threshold changes
# - Flags only (explanatory / gating)

# --- NCAAB ---
    # v3.2: STRONG constants moved to engine_config_v3.py


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


def _espn_finals_map_date_range(scoreboard_url_base: str, games: list[str], days: int = 5, dates: list[str] | None = None) -> dict[str, tuple[int, int]]:
    """
    Returns DK-game-keyed finals map:
      DK game string (as provided in `games`) -> (away_score, home_score)

    Strategy (stable, future-proof):
    1) Build DK abbrev keys: first token of each DK team string (upper), e.g. "CHI @ NSH"
    2) Query ESPN scoreboard for explicit YYYYMMDD dates (if provided) else a rolling range
    3) Index ESPN finals by abbrev using competitors[].team.abbreviation + homeAway
    4) Resolve DK games -> ESPN by abbrev key
    5) Fallback: existing normalize-team-name key ("Away @ Home") if needed
    """
    import json
    import urllib.request
    from datetime import datetime, timedelta

    def _safe(x):
        return (x or "").strip()

    def _split_game(g: str):
        g = _safe(g)
        if " @ " in g:
            a, h = g.split(" @ ", 1)
            return _safe(a), _safe(h)
        if " vs " in g:
            h, a = g.split(" vs ", 1)
            return _safe(a), _safe(h)
        return "", ""

    # ---- DK abbrev key: first token of each side, upper ----
    dk_abbrev_by_game: dict[str, str] = {}
    for g in games or []:
        a, h = _split_game(g)
        if not a or not h:
            continue
        a_ab = a.split()[0].upper()
        h_ab = h.split()[0].upper()
        dk_abbrev_by_game[g] = f"{a_ab} @ {h_ab}"

    # ---- Name-normalized fallback key (your existing normalizer) ----
    def _norm_team(x: str) -> str:
        try:
            return _normalize_team_name(_safe(x))
        except Exception:
            return _safe(x)

    dk_namekey_by_game: dict[str, str] = {}
    for g in games or []:
        a, h = _split_game(g)
        if not a or not h:
            continue
        dk_namekey_by_game[g] = f"{_norm_team(a)} @ {_norm_team(h)}"

    # ---- Build date list ----
    ymds: list[str] = []
    if dates:
        ymds = [str(x).strip() for x in dates if str(x).strip()]
    else:
        start = datetime.now() - timedelta(days=days)
        for i in range(days + 1):
            d = start + timedelta(days=i)
            ymds.append(d.strftime("%Y%m%d"))

    # ---- Sport-specific params to avoid truncation ----
    def _extra_params(base: str) -> str:
        if "mens-college-basketball" in base:
            return "&groups=50&limit=500"
        if "football/college-football" in base:
            return "&groups=80&limit=500"
        return "&limit=500"

    espn_by_abbrev: dict[str, tuple[int, int]] = {}
    espn_by_name: dict[str, tuple[int, int]] = {}

    for ymd in ymds:
        sep = "&" if "?" in scoreboard_url_base else "?"
        url = f"{scoreboard_url_base}{sep}dates={ymd}{_extra_params(scoreboard_url_base)}"
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/json,text/plain,*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            continue

        for ev in data.get("events", []) or []:
            comps = ev.get("competitions") or []
            if not comps:
                continue
            comp0 = comps[0]
            status = (comp0.get("status") or {}).get("type") or {}

            # Accept multiple ESPN final indicators
            is_completed = bool(status.get("completed"))
            is_final_name = status.get("name") == "STATUS_FINAL"
            is_final_desc = (status.get("description") or "").strip().lower() == "final"

            if not (is_completed or is_final_name or is_final_desc):
                continue

            competitors = comp0.get("competitors") or []
            if len(competitors) < 2:
                continue

            try:
                away = next(c for c in competitors if c.get("homeAway") == "away")
                home = next(c for c in competitors if c.get("homeAway") == "home")
            except StopIteration:
                continue

            at = away.get("team") or {}
            ht = home.get("team") or {}

            away_ab = _safe(at.get("abbreviation")).upper()
            home_ab = _safe(ht.get("abbreviation")).upper()
            if not away_ab or not home_ab:
                continue

            try:
                away_score = int(float(_safe(away.get("score"))))
                home_score = int(float(_safe(home.get("score"))))
            except Exception:
                continue

            # Abbrev key index
            espn_by_abbrev[f"{away_ab} @ {home_ab}"] = (away_score, home_score)

            # Name key fallback index (normalized)
            away_name = _norm_team(at.get("shortDisplayName") or at.get("displayName") or at.get("name") or "")
            home_name = _norm_team(ht.get("shortDisplayName") or ht.get("displayName") or ht.get("name") or "")
            if away_name and home_name:
                espn_by_name[f"{away_name} @ {home_name}"] = (away_score, home_score)

    # ---- Resolve DK games -> finals (prefer abbrev) ----
    finals: dict[str, tuple[int, int]] = {}
    for g in games or []:
        ak = dk_abbrev_by_game.get(g, "")
        nk = dk_namekey_by_game.get(g, "")
        if ak and ak in espn_by_abbrev:
            finals[g] = espn_by_abbrev[ak]
        elif nk and nk in espn_by_name:
            finals[g] = espn_by_name[nk]

    return finals


def get_espn_finals_map(sport: str, games: list[str], dates: list[str] | None = None) -> dict[str, tuple[int, int]]:
    """
    Generic ESPN final-score resolver.
    Returns DK-game-keyed final score map: "Away @ Home" -> (away_score, home_score)
    """
    base = ESPN_SCOREBOARD_BASE.get(sport)
    if not base or not games:
        return {}
    try:
        return _espn_finals_map_date_range(base, games, days=10, dates=dates)
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
        
        # ---- Derive ESPN query dates from DK kickoff (ET) ----
        dates = []
        try:
            import pandas as pd
            if "dk_start_iso" in gdf.columns:
                dt = pd.to_datetime(gdf["dk_start_iso"], errors="coerce", utc=True)
            else:
                dt = pd.to_datetime(gdf["timestamp"], errors="coerce", utc=True)

            date_et = dt.dt.tz_convert("America/New_York").dt.strftime("%Y%m%d")
            dates = sorted(set(date_et.dropna().astype(str)))
        except Exception:
            dates = []

        finals_map = get_espn_finals_map(sport, games, dates=dates)
        finals_by_sport[sport] = finals_map


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
        finals_map = finals_by_sport.get(sport) or {}

        # Try raw DK game string first (finals_map is keyed by exact snapshot game strings)
        raw_key = str(game).strip()
        match = finals_map.get(raw_key)

        if match is None:
            # Fallback: normalized key
            key = f"{away} @ {home}"
            norm_key = _norm_game_key(key)
            match = finals_map.get(key) or finals_map.get(norm_key)

        if not match:
            continue

        away_score, home_score = match
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

    # Always persist schema (CSV treated as DB table)
    df.to_csv(src, index=False)

    if updated > 0:
        print(f"[finals] updated {updated} rows in {src}")

    update_final_scores_history()


# ---- Step C (metrics instrumentation only): row_state + signal_ledger ----
def _metrics_now_iso_utc() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")



def _canonical_market_side_for_state(market: str, side: str, current_line: str):
    """
    Enforce invariant:
      - market in {SPREAD,TOTAL,MONEYLINE}
      - side normalized by market:
          SPREAD: team name only
          TOTAL: Under/Over only
          MONEYLINE: team name only
    Uses BOTH side and current_line because DK stores ML odds inside current_line (e.g. "DEN Broncos @ -115").
    """
    import re

    m = (market or "").strip().upper()
    s = (side or "").strip()
    cl = (current_line or "").strip()

    su = s.upper()
    clu = cl.upper()

    # TOTAL detection wins (if side or current_line looks like Under/Over)
    if re.search(r"\b(UNDER|OVER)\b", su) or clu.startswith("UNDER") or clu.startswith("OVER"):
        # Prefer explicit Under/Over from side first
        if "UNDER" in su or clu.startswith("UNDER"):
            return "TOTAL", "Under"
        if "OVER" in su or clu.startswith("OVER"):
            return "TOTAL", "Over"
        return "TOTAL", s  # fallback (shouldn't happen)

    # Spread-ish suffix on side => SPREAD team name
    if re.search(r"\s[+-]\d+(?:\.\d+)?\s*$", s):
        team = re.sub(r"\s[+-]\d+(?:\.\d+)?\s*$", "", s).strip()
        return "SPREAD", team

    # Moneyline detection from current_line: contains "@" and ends with odds
    # Examples: "DEN Broncos @ -115", "BUF Bills @ +105"
    if "@" in cl and re.search(r"@\s*[+-]?\d+\s*$", cl):
        # side should be team name; if missing, parse from current_line before "@"
        if not s:
            team = cl.split("@", 1)[0].strip()
            return "MONEYLINE", team
        # sanitize anyway
        team = re.sub(r"\s[+-]\d+(?:\.\d+)?\s*$", "", s).strip()
        return "MONEYLINE", team

    # If market already valid, sanitize side to match market
    if m in ("SPREAD","TOTAL","MONEYLINE"):
        if m == "TOTAL":
            if "UNDER" in su:
                return "TOTAL", "Under"
            if "OVER" in su:
                return "TOTAL", "Over"
            return "TOTAL", s
        if m == "MONEYLINE":
            team = re.sub(r"\s[+-]\d+(?:\.\d+)?\s*$", "", s).strip()
            return "MONEYLINE", team
        if m == "SPREAD":
            team = re.sub(r"\s[+-]\d+(?:\.\d+)?\s*$", "", s).strip()
            return "SPREAD", team

    # Fallback: keep something deterministic
    return (m or "MONEYLINE"), s


def _parse_line_val(current_line: str, market: str = "") -> float:
    """Extract numeric line value from DK current_line string for CLV tracking.
    SPREAD: 'Team -3.5 @ -110' -> -3.5
    TOTAL:  'Over 225.5 @ -110' -> 225.5
    ML:     'Team @ -150' -> -150 (use American odds as line proxy)
    """
    import re as _re
    cl = str(current_line or "").strip()
    if not cl:
        return None
    # Total: Over/Under X
    _m = _re.match(r"(?:Over|Under)\s+([+-]?\d+\.?\d*)", cl, _re.I)
    if _m:
        return float(_m.group(1))
    # Spread: Team +/-X @ odds
    _m = _re.search(r"([+-]\d+\.?\d*)\s*@", cl)
    if _m:
        return float(_m.group(1))
    # ML: Team @ odds (use odds as value)
    _m = _re.search(r"@\s*([+-]?\d+)", cl)
    if _m:
        return float(_m.group(1))
    return None


def normalize_side_key(sport: str, market_display: str, side_raw: str) -> str:
    """
    Canonical side key for row_state, elig_map joins, and migration.
    Returns a stable string key -- never used for display.
    TOTAL  -> TOTAL_OVER or TOTAL_UNDER
    SPREAD -> TEAM: + normalize_team_name(team_only)
    ML     -> TEAM: + normalize_team_name(team_only)
    No fuzzy matching -- deterministic only.
    """
    import re as _re
    m = (market_display or "").strip().upper()
    s = (side_raw or "").strip()
    su = s.upper()
    # Idempotent: already canonical -- return as-is
    # But only if it looks clean (no double-prefix like TEAM:teamteam...)
    if s in ("TOTAL_OVER", "TOTAL_UNDER", "TOTAL_UNKNOWN"):
        return s
    if s.startswith("TEAM:") and not s.startswith("TEAM:team"):
        return s
    # Strip TEAM: prefix if present before normalizing (handles double-prefix repair)
    if s.startswith("TEAM:"):
        s = s[5:]
        # Also strip any residual "team" prefix left by prior double-normalization
        import re as _re2
        while _re2.match(r"^team[a-z]", s):
            s = s[4:]
    if m == "TOTAL" or "OVER" in su or "UNDER" in su:
        if "UNDER" in su:
            return "TOTAL_UNDER"
        if "OVER" in su:
            return "TOTAL_OVER"
        return "TOTAL_UNKNOWN"
    team = _re.sub(r"\s[+-]\d+(?:\.\d+)?\s*$", "", s).strip()
    team_norm = normalize_team_name(team)
    return f"TEAM:{team_norm}"


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
    # Buckets (v1.2 thresholds):
    #   NO_BET < 60
    #   LEAN   60?66
    #   BET    67?69
    #   HIGH >= 70
    try:
        s = float(score)
    except Exception:
        return "NO_BET"
    if s >= 70.0:
        return "HIGH"
    if s >= 67.0:
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
        for c in ["sport","game_id","market","side","logic_version","last_score","last_ts","last_seen_tick","peak_score","peak_ts","last_bucket","last_net_edge","last_net_edge_ts","strong_streak","bet_candidate_streak"]:
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
        "score",
        "net_edge",
        "favored_side",
        "timing_bucket",
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

    # --- METRICS FILE INIT (v1.1) ---
    import os, csv
    os.makedirs("data", exist_ok=True)

    ROW_STATE_PATH = "data/row_state.csv"
    SIGNAL_LEDGER_PATH = "data/signal_ledger.csv"

    if not os.path.exists(ROW_STATE_PATH):
        with open(ROW_STATE_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["sport","game_id","market","side","last_score","peak_score","peak_ts","last_bucket","last_l2_n_books","last_consensus_tier"])

    if not os.path.exists(SIGNAL_LEDGER_PATH):
        with open(SIGNAL_LEDGER_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ts_utc","sport","game_id","market","side","event","value","logic_version"])

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
        score_col = (
            "score" if "score" in latest.columns else
            ("model_score" if "model_score" in latest.columns else
             ("confidence_score" if "confidence_score" in latest.columns else ""))
        )
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
        # score_col already selected above (do not overwrite here)
        if not score_col:
            return

        os.makedirs("data", exist_ok=True)
        state_path = os.path.join("data", "row_state.csv")
        ledger_path = os.path.join("data", "signal_ledger.csv")
        # Ensure ledger file exists even if there are no crossings this run (prevents FileNotFoundError)
        _append_signal_ledger(ledger_path, [])

        state_df = _load_row_state(state_path)

        # --- v1.1 safety: prune any contaminated legacy keys from row_state on load ---
        try:
            if state_df is not None and (not state_df.empty):
                _side = state_df.get("side", "").fillna("").astype(str)
                _mkt  = state_df.get("market", "").fillna("").astype(str).str.upper()

                _bad_total = _side.str.contains(r"\bUNDER\b|\bOVER\b", case=False, na=False)
                _bad_spreadish_ml = (_mkt=="MONEYLINE") & _side.str.contains(r"\s[+-]\d+(?:\.\d+)?\s*$", regex=True, na=False)
                _bad = ((_mkt.isin(["SPREAD","MONEYLINE"])) & _bad_total) | _bad_spreadish_ml

                if bool(_bad.any()):
                    state_df = state_df[~_bad].copy()
        except Exception:
            pass
        # --- end prune ---

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
                _ld_sport = _metrics_blank(r.get("sport"))
                _ld_game_id = _metrics_blank(r.get("game_id"))
                _ld_market = _metrics_blank(r.get("market"))
                _ld_side_raw = _metrics_blank(r.get("side"))
                _ld_side_norm = normalize_side_key(_ld_sport, _ld_market, _ld_side_raw)
                k = "|".join([_ld_sport, _ld_game_id, _ld_market, _ld_side_norm])
                state_map[k] = r.to_dict()

        now_ts = _metrics_now_iso_utc()
        ledger_rows = []
        processed = 0
        _touched_keys = set()  # track keys updated this run for stale cleanup
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
            market = _metrics_blank(r.get(market_col) or r.get("_metrics_market") or r.get("market_display") or "").upper()
            side = _metrics_blank(r.get("side"))
            current_line = _metrics_blank(r.get("current_line"))

            # --- canonicalize market/side using the STATE market (SPREAD/TOTAL/MONEYLINE) ---
            market, side = _canonical_market_side_for_state(market, side, current_line)
            market = _metrics_blank(market).upper()
            side = _metrics_blank(side)


            # --- v1.1 guardrail: prevent TOTAL/SPREAD/ML cross-contamination in row_state ---
            try:
                _mkt_u = str(market).strip().upper()
                _side_u = str(side).strip().upper()

                # Totals must never be written under SPREAD/MONEYLINE
                if _mkt_u in ("SPREAD","MONEYLINE") and (("UNDER" in _side_u) or ("OVER" in _side_u)):
                    continue

                # Spread-like labels must never be written under MONEYLINE (e.g., "Boise State -5.5")
                if _mkt_u == "MONEYLINE":
                    import re as _re
                    if _re.search(r"\s[+-]\d+(?:\.\d+)?\s*$", str(side).strip()):
                        continue
            except Exception:
                pass
            # --- end guardrail ---

            if sport == "" or game_id == "" or market == "" or side == "":
                continue

            # Use normalize_side_key for canonical row_state key
            _side_norm = normalize_side_key(sport, market, side)
            k = f"{sport}|{game_id}|{market}|{_side_norm}"

            score = _metrics_float(r.get(score_col), default=None)
            # net_edge from precomputed market edge_map (fallback 0.0)
            net_edge = _metrics_float(r.get('net_edge'), default=None)
            if net_edge is None:
                net_edge = float(edge_map.get((sport, game_id, market), 0.0) or 0.0)

            if score is None:
                continue
            processed += 1

            bucket = _score_bucket(score)

            # --- v1.1 STRONG precheck (true eligibility gate for streak) ---
            # strong_precheck_now = structural gates only (no persistence/stability -- those are circular)
            # Gates: score>=70, not LATE, not PUBLIC DRIFT, no cross-market contradiction, sport early blocks
            prev = state_map.get(k, {})
            prev_streak = 0
            try:
                prev_streak = int(str(prev.get("strong_streak","0")).strip() or "0")
            except Exception:
                prev_streak = 0
            try:
                _tb = str(r.get("timing_bucket","")).strip().upper()
                _mr = str(r.get("market_read","")).strip()
                _pc = str(r.get("market_pair_check","")).strip()
                _sp = str(r.get("sport","")).strip().upper()
                _score_ok = (score >= 70.0)
                _late_ok = (_tb != "LATE")
                _drift_ok = (_mr != "Public Drift")
                _xmkt_ok = (_pc == "")
                # Sport-specific early blocks
                _early_ok = True
                if _sp == "NCAAB" and _tb == "EARLY":
                    _early_ok = False  # NCAAB_EARLY_STRONG_BLOCK
                if _sp == "NCAAF" and _tb == "EARLY":
                    _early_ok = False  # NCAAF_EARLY_INSTANT_STRONG_BLOCK
                strong_precheck_now = (
                    _score_ok and _late_ok and _drift_ok and _xmkt_ok and _early_ok
                )
            except Exception:
                strong_precheck_now = False
            # Streak uses precheck only (Grace-1: one missed tick allowed without reset)
            cur_tick = _metrics_blank(r.get("ts") or r.get("snapshot_ts") or r.get("timestamp"))
            prev_tick = _metrics_blank(prev.get("last_seen_tick") or prev.get("last_tick") or prev.get("last_ts"))
            is_new_tick = (cur_tick != "" and cur_tick != prev_tick)
            prev_miss = 0
            try:
                prev_miss = int(str(prev.get("strong_miss_streak","0")).strip() or "0")
            except Exception:
                prev_miss = 0
            if not is_new_tick:
                # Same tick -- idempotent, do not advance
                strong_streak = str(prev_streak)
                strong_miss_streak = str(prev_miss)
            else:
                if strong_precheck_now:
                    strong_streak = str(prev_streak + 1)
                    strong_miss_streak = "0"
                else:
                    # Grace-1: one miss allowed before streak resets
                    if prev_miss >= 1:
                        strong_streak = "0"
                        strong_miss_streak = "0"
                    else:
                        strong_streak = str(prev_streak)
                        strong_miss_streak = str(prev_miss + 1)
            # --- end v1.1 STRONG precheck ---

            # --- v3.3m: BET candidate streak (score >= BET_SCORE_MIN for N cycles) ---
            prev_bet_streak = 0
            try:
                prev_bet_streak = int(str(prev.get("bet_candidate_streak", "0")).strip() or "0")
            except Exception:
                prev_bet_streak = 0
            if not is_new_tick:
                bet_candidate_streak = prev_bet_streak
            else:
                _bet_score_ok = (score >= 70.0)
                if _bet_score_ok:
                    bet_candidate_streak = prev_bet_streak + 1
                else:
                    bet_candidate_streak = 0
            # --- end v3.3m BET candidate streak ---

            prev = state_map.get(k, {})
            prev_bucket = _metrics_blank(prev.get("last_bucket"))

            # peak tracking
            prev_peak = _metrics_float(prev.get("peak_score"), default=None)
            peak_score = score if (prev_peak is None or score > prev_peak) else prev_peak
            peak_ts = now_ts if (prev_peak is None or score > prev_peak) else _metrics_blank(prev.get("peak_ts"))

            # threshold crossing logging — log ALL direction changes
            rank = {"": 0, "NO_BET": 0, "LOCKED": 0, "LEAN": 1, "BET": 2, "HIGH": 3}
            pb = prev_bucket if prev_bucket in rank else "NO_BET"
            cb = bucket if bucket in ("NO_BET", "LOCKED", "LEAN", "BET", "HIGH") else "NO_BET"
            if rank.get(pb, 0) != rank.get(cb, 0):
                direction = "UP" if rank.get(cb, 0) > rank.get(pb, 0) else "DOWN"
                _game_str = r.get("game", "")
                ledger_rows.append({
                    "ts": now_ts,
                    "logic_version": LOGIC_VERSION,
                    "event": f"THRESHOLD_{direction}",
                    "from_bucket": pb,
                    "to_bucket": cb,
                    "sport": sport,
                    "game_id": game_id,
                    "game": _game_str,
                    "market": market,
                    "side": side,
                    "score": f"{score:.2f}",
                    "net_edge": f"{float(net_edge):.2f}",
                })

                
            # ── LINE MOVEMENT TRAJECTORY TRACKING ──
            # Track the line tick-by-tick to detect patterns:
            #   MOVE_AND_HOLD: moved early then settled (book found its number)
            #   REVERSE: direction changed (book uncertain or tested)
            #   STEADY_DRIFT: continuous same-direction movement
            #   FLAT: no movement (book confident at this price)
            # Works for all markets: spread (line_val), total (line_val), ML (odds/juice)
            _cur_line_track = None
            _mkt_u_track = market.upper()
            if _mkt_u_track in ("SPREAD", "TOTAL"):
                try:
                    _clv = r.get("current_line_val")
                    if _clv is not None and str(_clv).strip() != "":
                        _cur_line_track = float(_clv)
                except (ValueError, TypeError):
                    pass
            elif _mkt_u_track == "MONEYLINE":
                try:
                    _co = r.get("current_odds")
                    if _co is not None and str(_co).strip() != "":
                        _cur_line_track = float(_co)
                except (ValueError, TypeError):
                    pass

            _prev_line_track = None
            try:
                _plt = prev.get("line_track_val", "")
                if str(_plt).strip() != "":
                    _prev_line_track = float(_plt)
            except (ValueError, TypeError):
                pass

            _prev_settled = 0
            try:
                _prev_settled = int(str(prev.get("line_settled_ticks", "0")).strip() or "0")
            except Exception:
                pass
            _prev_dir_changes = 0
            try:
                _prev_dir_changes = int(str(prev.get("line_dir_changes", "0")).strip() or "0")
            except Exception:
                pass
            _prev_last_dir = 0
            try:
                _prev_last_dir = int(str(prev.get("line_last_dir", "0")).strip() or "0")
            except Exception:
                pass
            _prev_max_move = 0.0
            try:
                _prev_max_move = float(str(prev.get("line_max_move", "0")).strip() or "0")
            except Exception:
                pass

            # Compute movement this tick
            _new_settled = _prev_settled
            _new_dir = _prev_last_dir
            _new_dir_changes = _prev_dir_changes
            _new_max_move = _prev_max_move
            if _cur_line_track is not None and _prev_line_track is not None and is_new_tick:
                _line_delta = _cur_line_track - _prev_line_track
                # Meaningful change threshold: 0.5 pts for spread/total, 5 cents for ML
                _move_thresh = 5.0 if _mkt_u_track == "MONEYLINE" else 0.25
                if abs(_line_delta) < _move_thresh:
                    _new_settled = _prev_settled + 1
                else:
                    _new_settled = 0
                    _tick_dir = 1 if _line_delta > 0 else -1
                    if _prev_last_dir != 0 and _tick_dir != _prev_last_dir:
                        _new_dir_changes = _prev_dir_changes + 1
                    _new_dir = _tick_dir
                    _new_max_move = max(_prev_max_move, abs(_line_delta))
            elif _cur_line_track is not None and _prev_line_track is None:
                _new_settled = 0
                _new_dir = 0
                _new_dir_changes = 0
                _new_max_move = 0.0

            # upsert state
            state_map[k] = {
                "sport": sport,
                "game_id": game_id,
                "market": market,
                "side": _side_norm,
                "logic_version": LOGIC_VERSION,
                "last_score": f"{score:.2f}",
                "last_ts": now_ts,
                "last_net_edge": (f"{_metrics_float(net_edge, default=0.0):.2f}" if str(net_edge).strip() != "" else ""),
                "last_net_edge_ts": (now_ts if str(net_edge).strip() != "" else ""),

                "peak_score": f"{peak_score:.2f}",
                "peak_ts": peak_ts,
                "last_bucket": bucket,
                "timing_bucket": _metrics_blank(r.get("timing_bucket")),
                "strong_streak": str(strong_streak),
                "strong_miss_streak": str(strong_miss_streak),
                "bet_candidate_streak": str(bet_candidate_streak),
                "last_seen_tick": cur_tick,
                "stale_count": "0",
                # Line movement trajectory
                "line_track_val": str(_cur_line_track) if _cur_line_track is not None else "",
                "line_settled_ticks": str(_new_settled),
                "line_dir_changes": str(_new_dir_changes),
                "line_last_dir": str(_new_dir),
                "line_max_move": f"{_new_max_move:.2f}",
                # v3.3f: persist for delta tracking
                "last_l2_n_books": str(int(_metrics_float(r.get("l2_n_books", 0), default=0))),
                "last_consensus_tier": str(int(_metrics_float(r.get("consensus_tier", 0), default=0))),
            }
            _touched_keys.add(k)

        # --- Stale row cleanup: expire rows not seen in N consecutive ticks ---
        try:
            _stale_threshold = STALE_TICK_THRESHOLD
        except NameError:
            _stale_threshold = 3
        _stale_expired = []
        for _sk, _sv in list(state_map.items()):
            if _sk not in _touched_keys:
                try:
                    _sc = int(str(_sv.get("stale_count", "0")).strip() or "0")
                except Exception:
                    _sc = 0
                _sc += 1
                if _sc >= _stale_threshold:
                    _stale_expired.append(_sk)
                else:
                    _sv["stale_count"] = str(_sc)
        for _sk in _stale_expired:
            del state_map[_sk]

        if os.environ.get('METRICS_DEBUG','') == '1':
            print(f"[metrics debug] processed_rows={processed} state_rows={len(state_map)} stale_expired={len(_stale_expired)} ledger_rows_this_run={len(ledger_rows)}")
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


# --- v1.1 metrics adapter: dashboard -> side rows ---
def _adapt_dashboard_for_metrics(df):
    try:
        import pandas as pd
        if df is None or len(df) == 0:
            return df

        required = {"market_display","favored_side","game_confidence"}
        if not required.issubset(set(df.columns)):
            return df  # already side-level

        out_rows = []

        for _, r in df.iterrows():
            side = str(r.get("favored_side","")).strip()
            if side == "":
                continue

            # Normalize side label (remove number like +2.5)
            side_team = side.split(" +")[0].split(" -")[0]

        if not out_rows:
            return df

        return pd.DataFrame(out_rows)
    except Exception:
        return df

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










def _infer_market_from_side_line(side: str, current_line: str) -> str:
    """
    DK 'current_line' generally looks like '<thing> @ -110' for ALL markets.
    So '@' is NOT a moneyline indicator.
    Priority: TOTAL > SPREAD > MONEYLINE.
    """
    import re
    s = (str(side or "") + " " + str(current_line or "")).strip().upper()

    # TOTAL: Over/Under
    if "OVER" in s or "UNDER" in s:
        return "TOTAL"

    # SPREAD: DK spreads have a decimal (.0/.5) in the line (e.g., -1.5, +3.0, +7.5).
    # IMPORTANT: do NOT treat moneyline odds like "@ -115" as a spread.
    # So we only match signed numbers with a decimal.
    if re.search(r"\s[+-]\d+\.\d+\b", s):
        return "SPREAD"

    return "MONEYLINE"


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
    Returns +1 if odds moved to FAVOR this side (shorter/more expensive),
            -1 if moved AGAINST this side (longer/cheaper),
             0 if unknown/no move/sign-crossing.
    Works for American odds where:
      -120 -> -140 is more expensive (book favors, +1)
      +150 -> +130 is more expensive (book favors, +1)
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

    # Stealth Move: strong D+, LOW bets, and move toward side
    # v2.2: tightened from bets<=40 to bets<=25 — at 36% bets it's not "stealth"
    if D >= 12 and (bets_pct is not None and bets_pct <= 25) and move_dir == +1 and meaningful_move:
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

    # -----------------------------
          market_display, side, bets_pct, money_pct,
      open_odds/current_odds, open_line_val/current_line_val,
      odds_move_open/line_move_open
    """
    df = latest.copy()

    # D (PERF): vectorized money_pct - bets_pct (avoid df.apply(axis=1))
    # Treat missing/non-numeric as 0.0 to match prior behavior.
    _money = pd.to_numeric(df.get("money_pct", 0), errors="coerce").fillna(0.0)
    _bets  = pd.to_numeric(df.get("bets_pct", 0), errors="coerce").fillna(0.0)
    df["divergence_D"] = _money - _bets

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
    move_dirs = []        # v1.2: persist for regime classifier
    meaningfuls = []      # v1.2: persist for regime classifier (juice-aware)

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
            # v2.1: lowered from 10 to 5 — DK juice changes are deliberate pricing signals
            meaningful = (pd.notna(od) and abs(float(od)) >= 5)


            # --- MLB v1.1: record ML movement for run line inheritance ---
            try:
                _sp = str(r.get("sport","")).strip().upper()
                if _sp == "MLB":
                    global _ml_move_dir_map
                    try:
                        _ml_move_dir_map
                    except NameError:
                        _ml_move_dir_map = {}
                    _key = (
                        _sp,
                        str(r.get("game_id","")).strip(),
                        str(r.get("side","")).strip()
                    )
                    _ml_move_dir_map[_key] = move_dir
            except Exception:
                pass
            # --- end ML map ---

            move_summary.append(f"ML: {r.get('open_odds')}ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â€Ã‚Â ÃƒÂ¢Ã‚Â€Ã‚Â™{r.get('current_odds')} (ÃƒÂƒÃ‚ÂŽÃƒÂ¢Ã‚Â€Ã‚Âodds={od})")

        elif mkt == "SPREAD":
            move_dir = _toward_side_by_spread(r.get("open_line_val"), r.get("current_line_val"))
            od = r.get("odds_move_open")  # used below; must be set before referencing
            # v2.1: juice threshold lowered from 10 to 5 for spread move_dir detection
            if move_dir == 0 and pd.notna(od) and abs(float(od)) >= 5:
                move_dir = _toward_side_by_juice(r.get("open_odds"), r.get("current_odds"))

            ld = r.get("line_move_open")
            key_cross = _crossed_key(abs(r.get("open_line_val")) if pd.notna(r.get("open_line_val")) else None,
                                     abs(r.get("current_line_val")) if pd.notna(r.get("current_line_val")) else None)
            # v2.1: juice threshold lowered from 10 to 5 — DK pricing changes are intentional
            meaningful = (
                (pd.notna(ld) and abs(float(ld)) >= 0.5) or
                (pd.notna(od) and abs(float(od)) >= 5) or
                (key_cross != "")
            )


            # --- MLB v1.1: RUN LINE cannot create pressure; SPREAD inherits ML intent ---
            try:
                _sp = str(r.get("sport","")).strip().upper()
                if _sp == "MLB":
                    _cur_lv = r.get("current_line_val", None)
                    _is_runline = False
                    try:
                        if _cur_lv is not None and abs(float(_cur_lv)) == 1.5:
                            _is_runline = True
                    except Exception:
                        _is_runline = False

                    if _is_runline:
                        # Run line movement alone is ignored
                        meaningful = False
                        key_cross = ""

                        # Inherit ML movement direction (ML controls truth)
                        _ml_key = (
                            _sp,
                            str(r.get("game_id","")).strip(),
                            str(r.get("side","")).strip()
                        )
                        try:
                            _ml_md = _ml_move_dir_map.get(_ml_key, 0)
                        except Exception:
                            _ml_md = 0

                        move_dir = _ml_md
                        if _ml_md != 0:
                            meaningful = True
            except Exception:
                pass
            # --- end MLB run line override ---

            move_summary.append(
                f"SPREAD: {r.get('open_line_val')}ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â€Ã‚Â ÃƒÂ¢Ã‚Â€Ã‚Â™{r.get('current_line_val')} (ÃƒÂƒÃ‚ÂŽÃƒÂ¢Ã‚Â€Ã‚Âline={ld}), "
                f"odds {r.get('open_odds')}ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â€Ã‚Â ÃƒÂ¢Ã‚Â€Ã‚Â™{r.get('current_odds')} (ÃƒÂƒÃ‚ÂŽÃƒÂ¢Ã‚Â€Ã‚Âodds={od})"
                + (f", key={key_cross}" if key_cross else "")
            )

        elif mkt == "TOTAL":
            move_dir = _toward_side_by_total(r.get("open_line_val"), r.get("current_line_val"), side)
            ld = r.get("line_move_open")
            od = r.get("odds_move_open")
            # v2.1: juice threshold lowered from 10 to 5
            meaningful = (pd.notna(ld) and abs(float(ld)) >= 0.5) or (pd.notna(od) and abs(float(od)) >= 5)

            move_summary.append(
                f"TOTAL: {r.get('open_line_val')}ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â€Ã‚Â ÃƒÂ¢Ã‚Â€Ã‚Â™{r.get('current_line_val')} (ÃƒÂƒÃ‚ÂŽÃƒÂ¢Ã‚Â€Ã‚Ânum={ld}), "
                f"odds {r.get('open_odds')}ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â€Ã‚Â ÃƒÂ¢Ã‚Â€Ã‚Â™{r.get('current_odds')} (ÃƒÂƒÃ‚ÂŽÃƒÂ¢Ã‚Â€Ã‚Âodds={od})"
            )
        else:
            move_summary.append("Unknown market move")

        label = _classify_market_read(D, None if pd.isna(bets) else float(bets), move_dir, meaningful, is_high_bet)

        mr.append(label)
        move_dirs.append(move_dir)          # v1.2
        meaningfuls.append(meaningful)      # v1.2
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
    df["move_dir"] = move_dirs                # v1.2: persisted for regime classifier
    df["meaningful_move"] = meaningfuls       # v1.2: persisted for regime classifier (juice-aware)
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
# Thresholds (your ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â‚Ã‚Â¬ÃƒÂ…Ã‚Â“rare dark greenÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â‚Ã‚Â¬ÃƒÂ‚Ã‚Â setup)
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
    meaningful_move_pts: float = 1.5  # used for ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â‚Ã‚Â¬ÃƒÂ…Ã‚Â“strong line behaviorÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â‚Ã‚Â¬ÃƒÂ‚Ã‚Â

    # ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â‚Ã‚Â¬ÃƒÂ…Ã‚Â“No movementÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â‚Ã‚Â¬ÃƒÂ‚Ã‚Â / resistance trigger (optional)
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
    - Finds repeating ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â‚Ã‚Â¬ÃƒÂ…Ã‚Â“game cardsÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â‚Ã‚Â¬ÃƒÂ‚Ã‚Â or table rows
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

        # Create a single ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â‚Ã‚Â¬ÃƒÂ…Ã‚Â“side rowÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â‚Ã‚Â¬ÃƒÂ‚Ã‚Â record as a fallback
        # We'll show it in output even if itÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â‚Ã‚Â¬ÃƒÂ¢Ã‚Â„Ã‚Â¢s not perfectly split by team yet.
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
    market_display: str = "",                # v1.2: market-aware move threshold
) -> Tuple[str, str]:
    """
    Returns: (color, explanation)
    color ÃƒÂƒÃ‚Â¢ÃƒÂ‹Ã‚Â†ÃƒÂ‹Ã‚Â† {"DARK_GREEN","LIGHT_GREEN","GREY","YELLOW","RED"}
    """
    # If we don't have percentages, we can't score well
    if bets_pct is None or money_pct is None:
        return "GREY", "Missing bet%/money% ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â€Ã‚Â ÃƒÂ¢Ã‚Â€Ã‚Â™ default Grey"

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
            # v1.2: market-aware threshold (bug fix — ML juice moves like +114->+105 = 9pts
            # were triggering DARK_GREEN at the old 1.5 threshold)
            _mkt_u = str(market_display).strip().upper()
            _thresh = {"MONEYLINE": 8.0, "TOTAL": 1.0, "SPREAD": 0.5}.get(
                _mkt_u, TH.meaningful_move_pts)
            if move >= _thresh:
                strong_line_signal = True

        # Key number note can promote ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â‚Ã‚Â¬ÃƒÂ…Ã‚Â“strong line behaviorÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â‚Ã‚Â¬ÃƒÂ‚Ã‚Â (NFL)
        if key_number_note and key_number_note.strip():
            strong_line_signal = True

    # DARK GREEN (rare): requires strong line behavior + strong money signal, AND no obvious news explanation
    if strong_line_signal and dark_money_signal and not has_news:
        return "DARK_GREEN", "Book behavior + strong money-vs-bets imbalance; no obvious news ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â€Ã‚Â ÃƒÂ¢Ã‚Â€Ã‚Â™ Market Edge Confirmed"

    # If news explains it, keep in Light Green even if strong
    if strong_line_signal and dark_money_signal and has_news:
        return "LIGHT_GREEN", "Strong signals but major news present ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â€Ã‚Â ÃƒÂ¢Ã‚Â€Ã‚Â™ downgrade to Market Edge Developing" 

    # LIGHT GREEN: money-vs-bets imbalance without strong line confirmation
    if light_money_signal:
        return "LIGHT_GREEN", "Money concentration vs bet count ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â€Ã‚Â ÃƒÂ¢Ã‚Â€Ã‚Â™ Market Edge Developing (watch for confirmation)"

    # RED: avoid this side
    if is_red:
        return "RED", "Extremely public + weak money support ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â€Ã‚Â ÃƒÂ¢Ã‚Â€Ã‚Â™ Wrong Side / Trap (evaluate opposite side)"

    # YELLOW: public-driven
    if is_yellow:
        return "YELLOW", "Public-heavy demand without strong money support ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â€Ã‚Â ÃƒÂ¢Ã‚Â€Ã‚Â™ Caution"

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
                "dk_start_iso": _validate_iso_tz(row.get("start_time_iso") or row.get("start_time") or row.get("startDate") or row.get("dk_start_iso") or row.get("game_time_iso") or row.get("game_time") or "")
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

def _generate_book_lines_json():
    """Generate per-book line data for the sportsbook selector dropdown.
    Reads L2 consensus + L1 sharp CSVs, groups by canonical_key/market/side/bookmaker,
    finds earliest (open) and latest (current) per group, writes data/book_lines.json.
    Keys use positional identifiers: MARKET|home, MARKET|away, TOTAL|over, TOTAL|under.
    """
    import tempfile

    DROPDOWN_BOOKS = {
        "draftkings", "fanduel", "betmgm", "betrivers", "bovada",
        "pinnacle", "matchbook", "betonlineag", "williamhill", "bet365",
        "betfair_ex_eu",
    }

    BOOK_LABELS = {
        "draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM",
        "betrivers": "BetRivers", "bovada": "Bovada", "pinnacle": "Pinnacle",
        "matchbook": "Matchbook", "betonlineag": "BetOnline",
        "williamhill": "William Hill", "bet365": "Bet365",
        "betfair_ex_eu": "Betfair",
    }

    _pd = __import__("pandas")
    frames = []

    # Load L2 consensus
    l2_path = "data/l2_consensus.csv"
    if os.path.exists(l2_path):
        try:
            l2 = _pd.read_csv(l2_path)
            l2 = l2[l2["bookmaker"].isin(DROPDOWN_BOOKS)]
            frames.append(l2[["timestamp", "canonical_key", "market", "side", "bookmaker", "line", "odds_american"]])
        except Exception:
            pass

    # Load L1 sharp
    l1_path = "data/l1_sharp.csv"
    if os.path.exists(l1_path):
        try:
            l1 = _pd.read_csv(l1_path)
            l1 = l1[l1["bookmaker"].isin(DROPDOWN_BOOKS)]
            frames.append(l1[["timestamp", "canonical_key", "market", "side", "bookmaker", "line", "odds_american"]])
        except Exception:
            pass

    if not frames:
        print("[book_lines] no L1/L2 data found, skipping")
        return

    combined = _pd.concat(frames, ignore_index=True)
    combined["timestamp"] = _pd.to_datetime(combined["timestamp"], errors="coerce")
    combined = combined.dropna(subset=["canonical_key", "market", "side", "bookmaker"])

    # Determine home/away from canonical_key: "away_team @ home_team|sport|date"
    def _side_position(canonical_key, market, side):
        side_lower = str(side).lower().strip()
        market_upper = str(market).upper().strip()
        if market_upper == "TOTAL":
            if side_lower.startswith("over"):
                return "over"
            elif side_lower.startswith("under"):
                return "under"
            return side_lower
        # SPREAD / MONEYLINE: determine home vs away from canonical_key
        try:
            game_part = canonical_key.split("|")[0].strip()  # "away_team @ home_team"
            if " @ " in game_part:
                away_name, home_name = game_part.split(" @ ", 1)
                away_name = away_name.strip().lower()
                home_name = home_name.strip().lower()
                if side_lower == home_name or home_name.startswith(side_lower) or side_lower.startswith(home_name):
                    return "home"
                elif side_lower == away_name or away_name.startswith(side_lower) or side_lower.startswith(away_name):
                    return "away"
                # Fuzzy: check if side is a substring of home or away
                if side_lower in home_name or home_name in side_lower:
                    return "home"
                if side_lower in away_name or away_name in side_lower:
                    return "away"
        except Exception:
            pass
        return side_lower  # fallback to raw side

    # Group and find open/current per (canonical_key, market, side_position, bookmaker)
    combined["side_position"] = combined.apply(
        lambda r: _side_position(r["canonical_key"], r["market"], r["side"]), axis=1
    )

    grouped = combined.groupby(["canonical_key", "market", "side_position", "bookmaker"])

    lines_data = {}  # canonical_key → { "MARKET|position": { bookmaker: {...} } }
    books_seen = set()

    for (ck, mkt, pos, book), grp in grouped:
        grp_sorted = grp.sort_values("timestamp")
        earliest = grp_sorted.iloc[0]
        latest = grp_sorted.iloc[-1]

        if ck not in lines_data:
            lines_data[ck] = {}

        slot_key = f"{mkt.upper()}|{pos}"
        if slot_key not in lines_data[ck]:
            lines_data[ck][slot_key] = {}

        def _fmt(v):
            if _pd.isna(v) or v is None or str(v).strip() == "":
                return ""
            return str(v).strip()

        lines_data[ck][slot_key][book] = {
            "ol": _fmt(earliest.get("line", "")),
            "oo": _fmt(earliest.get("odds_american", "")),
            "cl": _fmt(latest.get("line", "")),
            "co": _fmt(latest.get("odds_american", "")),
        }
        books_seen.add(book)

    # Build output
    output = {
        "books": sorted(books_seen & DROPDOWN_BOOKS),
        "labels": {b: BOOK_LABELS.get(b, b) for b in sorted(books_seen & DROPDOWN_BOOKS)},
        "lines": lines_data,
    }

    # Write atomically
    os.makedirs("data", exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir="data", suffix=".json")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(output, f, separators=(",", ":"))
        os.replace(tmp_path, "data/book_lines.json")
        os.chmod("data/book_lines.json", 0o644)
        print(f"[ok] wrote book_lines.json ({len(lines_data)} games, {len(books_seen)} books)")
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise


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

    # Clean DK ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â‚Ã‚Â¬ÃƒÂ…Ã‚Â“opens in a new tabÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â‚Ã‚Â¬ÃƒÂ‚Ã‚Â¦ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â‚Ã‚Â¬ÃƒÂ‚Ã‚Â junk if present
    df["market"] = df["market"].astype(str).str.replace(r"\s+opens in a new tab.*$", "", regex=True)
    df["game"] = df["game"].astype(str).str.replace(r"\s+opens in a new tab.*$", "", regex=True)

    # ---------- Stable keys for OPEN/LATEST/PREV (prevents "open resets" when line changes) ----------
    # market_display is consistent even if df["market"] varies (or contains alt labels).
    # PERF: vectorize market_display (avoid df.apply(axis=1))
    _cl = df.get("current_line", "").astype(str).str.lower()

    # TOTAL: lines that look like o/u or contain over/under
    _is_total = _cl.str.match(r"^\s*[ou]\s*\d") | _cl.str.contains(r"over|under", regex=True)

    # MONEYLINE: +### or -### with no decimal (common ML) and not total
    _is_ml = (_cl.str.match(r"^\s*[+-]\d{3,}\s*$")) & (~_is_total)

    # market_display must be derived from rendered line text.
    # NOTE: DK snapshots often store ML odds in current_line (e.g., "DEN Broncos @ -115")
    # while side is just the team name. So MONEYLINE detection must look at current_line.
    _line_txt = (
        df["current_line"].fillna("").astype(str)
        if "current_line" in df.columns
        else df["side"].fillna("").astype(str)
    )
    _side_txt = df["side"].fillna("").astype(str)

    _line_u = _line_txt.str.upper()
    _side_u = _side_txt.str.upper()

    _is_total = (_side_u.str.contains("UNDER") | _side_u.str.contains("OVER") |
                 _line_u.str.contains("UNDER") | _line_u.str.contains("OVER"))

    # ML: detect "@ -115" style odds in current_line (preferred), fallback to side if ever present
    _is_ml = _line_u.str.contains(r"@\s*[-+]?\d+")

    # --- Canonical market_display (DK uses "@ price" for ALL markets; '@' is not ML) ---
    # Priority: TOTAL > SPREAD > MONEYLINE (residual)
    df["market_display"] = "SPREAD"
    df.loc[_is_total, "market_display"] = "TOTAL"

    # MONEYLINE = everything that is not TOTAL and not SPREAD
    try:
        # SPREAD: detect a signed line BEFORE the "@ price" token, e.g. "SEA Seahawks -7 @ -110"
        # Works for -7 and -1.5, and avoids treating ML odds like "@ -340" as spread.
        _cl = df.get("current_line", "").fillna("").astype(str).str.upper()
        _is_spread = (~_is_total) & _cl.str.contains(r"\s[+-]\d+(?:\.\d+)?\s*@\s*[+-]?\d+\s*$", regex=True, na=False)
    except Exception:
        _is_spread = (~_is_total)

    _is_ml = (~_is_total) & (~_is_spread)
    df.loc[_is_ml, "market_display"] = "MONEYLINE"
    # --- end canonical market_display ---

    # ---------- Deterministic main-line selection BEFORE open/latest/prev ----------
    # If multiple alternate spreads/totals exist at the same timestamp for one game,
    # choose a single main number first so line history is built from one coherent market.
    def _parse_line_val_for_filter(x):
        try:
            d = parse_line_and_odds("" if pd.isna(x) else str(x))
            if isinstance(d, dict):
                return d.get("line_val")
        except Exception:
            pass
        return None

    df["_filter_line_val"] = pd.to_numeric(
        df.get("current_line", "").apply(_parse_line_val_for_filter),
        errors="coerce",
    )

    totals_all = df[df["market_display"] == "TOTAL"].copy()
    if len(totals_all) > 0:
        total_counts = (
            totals_all.groupby(["sport", "game_id", "timestamp", "_filter_line_val"], as_index=False)
                     .size()
                     .rename(columns={"size": "n"})
        )
        med = (
            totals_all.groupby(["sport", "game_id", "timestamp"], as_index=False)["_filter_line_val"]
                     .median()
                     .rename(columns={"_filter_line_val": "med_total"})
        )
        total_counts = total_counts.merge(med, on=["sport", "game_id", "timestamp"], how="left")
        total_counts["dist"] = (total_counts["_filter_line_val"] - total_counts["med_total"]).abs()
        total_main = (
            total_counts.sort_values(
                ["sport", "game_id", "timestamp", "n", "dist", "_filter_line_val"],
                ascending=[True, True, True, False, True, True],
                kind="mergesort",
            )
            .groupby(["sport", "game_id", "timestamp"], as_index=False)
            .head(1)[["sport", "game_id", "timestamp", "_filter_line_val"]]
            .rename(columns={"_filter_line_val": "_main_total_line"})
        )
        totals_all = totals_all.merge(total_main, on=["sport", "game_id", "timestamp"], how="left")
        totals_all = totals_all[totals_all["_filter_line_val"] == totals_all["_main_total_line"]]
        totals_all = totals_all.drop(columns=["_main_total_line"], errors="ignore")

    spreads_all = df[df["market_display"] == "SPREAD"].copy()
    if len(spreads_all) > 0:
        spreads_all["_filter_abs_line"] = spreads_all["_filter_line_val"].abs()
        spread_counts = (
            spreads_all.groupby(["sport", "game_id", "timestamp", "_filter_abs_line"], as_index=False)
                      .size()
                      .rename(columns={"size": "n"})
        )
        spread_main = (
            spread_counts.sort_values(
                ["sport", "game_id", "timestamp", "n", "_filter_abs_line"],
                ascending=[True, True, True, False, True],
                kind="mergesort",
            )
            .groupby(["sport", "game_id", "timestamp"], as_index=False)
            .head(1)[["sport", "game_id", "timestamp", "_filter_abs_line"]]
            .rename(columns={"_filter_abs_line": "_main_abs_spread"})
        )
        spreads_all = spreads_all.merge(spread_main, on=["sport", "game_id", "timestamp"], how="left")
        spreads_all = spreads_all[spreads_all["_filter_abs_line"] == spreads_all["_main_abs_spread"]]
        spreads_all = spreads_all.drop(columns=["_filter_abs_line", "_main_abs_spread"], errors="ignore")

    money_all = df[df["market_display"] == "MONEYLINE"].copy()
    other_all = df[~df["market_display"].isin(["TOTAL", "SPREAD", "MONEYLINE"])].copy()
    df = pd.concat([money_all, totals_all, spreads_all, other_all], ignore_index=True)
    df = df.drop(columns=["_filter_line_val"], errors="ignore")

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
    latest["open_line_val"] = pd.to_numeric(latest["open_line_val"], errors="coerce")
    latest["prev_line_val"] = pd.to_numeric(latest["prev_line_val"], errors="coerce")

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

    # Require a complete 2-sided market before scoring/aggregation.
    _pair_counts = (
        latest.groupby(["sport", "game_id", "market_display"], as_index=False)
              .size()
              .rename(columns={"size": "_pair_n"})
    )
    latest = latest.merge(_pair_counts, on=["sport", "game_id", "market_display"], how="left")
    latest = latest[latest["_pair_n"] == 2].drop(columns=["_pair_n"])

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

    # v2.1: Effective move magnitude — combines line number AND juice/odds movement.
    # When the line number doesn't move but DK shifts juice (e.g. -115 → -105),
    # that IS a real pricing signal. Convert juice change to line-equivalent magnitude.
    # 10 cents juice ≈ 0.5 point spread move in probability terms.
    def _effective_move_mag(row):
        mkt = str(row.get("market_display", "")).upper()
        lm = abs(row.get("line_move_open", 0) or 0)
        if pd.isna(lm): lm = 0.0
        om = abs(row.get("odds_move_open", 0) or 0)
        if pd.isna(om): om = 0.0
        if mkt == "MONEYLINE":
            # ML has no line number. Use odds change only, capped.
            if om >= 5:
                return min(3.0, om / 15.0)
            return 0.0
        # Spread/Total: line move is primary signal
        if lm >= 0.5:
            return lm
        # Otherwise, convert juice movement to line-equivalent
        # 5 cents ≈ 0.33 pts, 10 cents ≈ 0.67 pts, 15 cents ≈ 1.0 pts, capped at 3.0
        if om >= 5:
            return min(3.0, om / 15.0)
        return 0.0
    latest["effective_move_mag"] = latest.apply(_effective_move_mag, axis=1)

    # Market Read (Observation Mode, additive only)
    latest = add_market_read_to_latest(latest)
    latest = add_market_pair_checks(latest)

    # --- Join row_state into latest for persistence tracking ---
    _rs_cols_wanted = ["sport","game_id","market","side","strong_streak","last_score","peak_score",
                       "last_bucket",
                       "line_settled_ticks","line_dir_changes","line_last_dir","line_max_move",
                       "last_l2_n_books","last_consensus_tier"]
    try:
        _rs = pd.read_csv("data/row_state.csv", dtype=str, keep_default_na=False)
        _rs = _rs[[c for c in _rs_cols_wanted if c in _rs.columns]].copy()
        if "market_display" in _rs.columns:
            _rs = _rs.drop(columns=["market_display"])
        _rs = _rs.rename(columns={"market": "market_display"})
        _rs["_rs_side_norm"] = _rs.apply(
            lambda r: normalize_side_key(str(r.get("sport","")), str(r.get("market_display","")), str(r.get("side",""))), axis=1
        )
        latest["_rs_side_norm"] = latest.apply(
            lambda r: normalize_side_key(str(r.get("sport","")), str(r.get("market_display","")), str(r.get("side",""))), axis=1
        )
        _rs_dedup = _rs.drop_duplicates(subset=["sport","game_id","market_display","_rs_side_norm"], keep="last")
        _rs_merge_cols = [c for c in ["sport","game_id","market_display","_rs_side_norm",
                          "strong_streak","last_score","peak_score","last_bucket",
                          "line_settled_ticks","line_dir_changes","line_last_dir","line_max_move",
                          "last_l2_n_books","last_consensus_tier"]
                          if c in _rs_dedup.columns]
        _drop_existing = [c for c in ["strong_streak","last_score","peak_score","last_bucket",
                          "line_settled_ticks","line_dir_changes","line_last_dir","line_max_move",
                          "last_l2_n_books","last_consensus_tier"]
                          if c in latest.columns]
        if _drop_existing:
            latest = latest.drop(columns=_drop_existing)
        latest = latest.merge(
            _rs_dedup[_rs_merge_cols],
            on=["sport","game_id","market_display","_rs_side_norm"], how="left"
        )
        latest["strong_streak"] = pd.to_numeric(latest["strong_streak"], errors="coerce").fillna(0).astype(int)
        latest["last_score"] = pd.to_numeric(latest["last_score"], errors="coerce").fillna(0.0)
        latest["peak_score"] = pd.to_numeric(latest["peak_score"], errors="coerce").fillna(0.0)
        # Line movement tracking columns
        latest["line_settled_ticks"] = pd.to_numeric(
            latest["line_settled_ticks"] if "line_settled_ticks" in latest.columns else pd.Series(0, index=latest.index),
            errors="coerce"
        ).fillna(0).astype(int)
        latest["line_dir_changes"] = pd.to_numeric(
            latest["line_dir_changes"] if "line_dir_changes" in latest.columns else pd.Series(0, index=latest.index),
            errors="coerce"
        ).fillna(0).astype(int)
        latest["line_last_dir"] = pd.to_numeric(
            latest["line_last_dir"] if "line_last_dir" in latest.columns else pd.Series(0, index=latest.index),
            errors="coerce"
        ).fillna(0).astype(int)
        latest["line_max_move"] = pd.to_numeric(
            latest["line_max_move"] if "line_max_move" in latest.columns else pd.Series(0.0, index=latest.index),
            errors="coerce"
        ).fillna(0.0)
    except Exception as _rse:
        print(f"[strong] row_state join failed: {_rse}")
        latest["strong_streak"] = 0
        latest["last_score"] = 0.0
        latest["peak_score"] = 0.0
        latest["last_bucket"] = ""
        latest["line_settled_ticks"] = 0
        latest["line_dir_changes"] = 0
        latest["line_last_dir"] = 0
        latest["line_max_move"] = 0.0
    if "last_bucket" not in latest.columns:
        latest["last_bucket"] = ""
    latest["last_bucket"] = latest["last_bucket"].fillna("").astype(str)
    # --- end row_state join ---

    # v3.2: STRONG eligibility handled by certification_v3.certify_decision() at game_view level
    # classify_side() color remains as UI display only — no STRONG gating



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

    # ── v1.2 NaN-safe helper ──
    def _safe_float(val, default=0.0):
        try:
            if val is None or pd.isna(val):
                return default
            out = float(val)
            return default if pd.isna(out) else out
        except Exception:
            return default

    # Classify each row (this is your existing signal logic)
    colors = []
    explains = []
    scores = []
    _v3_results = []  # legacy placeholder only; v3 scorer disabled
    # ml_green disabled — color no longer drives decisions
    _v4_results = []


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

    # --- v2.0 ML-only penalty: build spread movement map per (sport, game_id, side_norm) ---
    _spread_move_map = {}
    try:
        for _, _r in latest.iterrows():
            if str(_r.get("market_display", "")).strip().upper() == "SPREAD":
                _sp_key = "{}|{}".format(
                    str(_r.get("game_id", "")).strip(),
                    str(_r.get("side_key", "")).strip().lower(),
                )
                _sp_lm = abs(float(_r.get("line_move_open", 0) or 0))
                _sp_md = int(float(_r.get("move_dir", 0) or 0))
                _sp_mf = bool(_r.get("meaningful_move", False))
                _spread_move_map[_sp_key] = {"lm": _sp_lm, "dir": _sp_md, "meaningful": _sp_mf}
    except Exception:
        pass
    # --- end spread movement map ---

    # ── Live context enrichment only ──
    # Keeps situational/weather/sport-context data without L1/L2 merge work.
    try:
        import importlib
        _merge_layers = importlib.import_module("merge_layers")
        enrich_context = getattr(_merge_layers, "enrich_context", None)
        if enrich_context is None:
            raise AttributeError("merge_layers.enrich_context missing")
        # Enrich per-sport so each sport gets its own situational/context fields
        if "sport" in latest.columns:
            _sport_vals = latest["sport"].dropna().unique()
            if len(_sport_vals) > 1:
                _merged_parts = []
                for _sv in _sport_vals:
                    _part = latest[latest["sport"] == _sv].copy()
                    _part = enrich_context(_part, sport=str(_sv).lower())
                    _merged_parts.append(_part)
                latest = pd.concat(_merged_parts, ignore_index=True)
            elif len(_sport_vals) == 1:
                latest = enrich_context(latest, sport=str(_sport_vals[0]).lower())
            else:
                latest = enrich_context(latest, sport=None)
        else:
            latest = enrich_context(latest, sport=None)
    except Exception as _merge_err:
        print(f"  [WARN] context enrichment failed: {repr(_merge_err)}")

    # Keep inert legacy defaults until downstream cleanup is complete.
    if "layer_mode" not in latest.columns:
        latest["layer_mode"] = "L3_ONLY"
    if "l1_available" not in latest.columns:
        latest["l1_available"] = False
    if "l2_available" not in latest.columns:
        latest["l2_available"] = False

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

    # ── BUILD CROSS-MARKET CONTEXT for ML vs Spread prob check ──
    # Maps side_key -> {ml_odds_<side_key>: odds, spread_odds_<side_key>: odds}
    _cross_mkt_ctx = {}
    for _, _cm_row in latest.iterrows():
        _cm_gid = str(_cm_row.get("game_id", "")).strip()
        _cm_sk = str(_cm_row.get("side_key", "")).strip()
        _cm_mkt = str(_cm_row.get("market_display", "")).upper()
        _cm_odds = _cm_row.get("current_odds")
        if not _cm_gid or not _cm_sk or _cm_odds is None:
            continue
        try:
            _cm_odds_f = float(_cm_odds)
            if pd.isna(_cm_odds_f):
                continue
        except (ValueError, TypeError):
            continue
        ctx_key = (_cm_gid, _cm_sk)
        if ctx_key not in _cross_mkt_ctx:
            _cross_mkt_ctx[ctx_key] = {}
        if _cm_mkt == "MONEYLINE":
            _cross_mkt_ctx[ctx_key][f"ml_odds_{_cm_sk}"] = _cm_odds_f
        elif _cm_mkt == "SPREAD":
            _cross_mkt_ctx[ctx_key][f"spread_odds_{_cm_sk}"] = _cm_odds_f

    # Compute game-level favored sides for cross-market sanity check
    _game_ml_odds = defaultdict(dict)    # game_id → {side_key: odds}
    _game_spread_odds = defaultdict(dict)
    for (_gid, _sk), _ctx in _cross_mkt_ctx.items():
        for _k, _v in _ctx.items():
            if _k.startswith("ml_odds_"):
                _game_ml_odds[_gid][_sk] = _v
            elif _k.startswith("spread_odds_"):
                _game_spread_odds[_gid][_sk] = _v

    _game_favored = {}
    for _gid in set(list(_game_ml_odds.keys()) + list(_game_spread_odds.keys())):
        _fav = {}
        if _gid in _game_ml_odds and len(_game_ml_odds[_gid]) >= 2:
            _odds_vals = list(_game_ml_odds[_gid].values())
            if abs(_odds_vals[0] - _odds_vals[1]) < 2:
                _fav["ml_favored_side"] = None  # pick'em guard
            else:
                _ml_fav_sk = min(_game_ml_odds[_gid], key=lambda s: _game_ml_odds[_gid][s])
                _fav["ml_favored_side"] = _ml_fav_sk
                _fav["ml_fav_odds"] = _game_ml_odds[_gid][_ml_fav_sk]
        if _gid in _game_spread_odds and len(_game_spread_odds[_gid]) >= 2:
            _odds_vals = list(_game_spread_odds[_gid].values())
            if abs(_odds_vals[0] - _odds_vals[1]) < 2:
                _fav["spread_favored_side"] = None  # pick'em guard
            else:
                _fav["spread_favored_side"] = min(_game_spread_odds[_gid], key=lambda s: _game_spread_odds[_gid][s])
        _game_favored[_gid] = _fav

    # Build deterministic same-market context for semantic classification.
    # Shape: {(game_id, market_display): {"market_rows": [row_dict, ...], "pressure_side": side|None}}
    _market_context_lookup = {}

    def _semantic_pressure_score(_row_dict):
        _bets = _safe_float(_row_dict.get("bets_pct"), 0.0)
        _money = _safe_float(_row_dict.get("money_pct"), 0.0)
        _div = _money - _bets
        _score = 0.0
        if _bets >= 60:
            _score = max(_score, _bets)
        if _money >= 65:
            _score = max(_score, _money)
        if _div >= 15:
            _score = max(_score, _div)
        return _score

    try:
        for (_mc_gid, _mc_mkt), _mc_df in latest.groupby(["game_id", "market_display"], dropna=False):
            _rows = []
            _pressure_side = None
            _pressure_score = 0.0
            for _, _mc_row in _mc_df.iterrows():
                _row_dict = _mc_row.to_dict()
                _rows.append(_row_dict)
                _score = _semantic_pressure_score(_row_dict)
                if _score > _pressure_score:
                    _pressure_score = _score
                    _pressure_side = str(_row_dict.get("side", "")).strip() or None
            _market_context_lookup[(str(_mc_gid).strip(), str(_mc_mkt).strip())] = {
                "market_rows": _rows,
                "pressure_side": _pressure_side if _pressure_score > 0 else None,
            }
    except Exception:
        _market_context_lookup = {}

    for _, row in latest.iterrows():
        color, expl = classify_side(
            bets_pct=int(row["bets_pct"]) if pd.notna(row.get("bets_pct")) else None,
            money_pct=int(row["money_pct"]) if pd.notna(row.get("money_pct")) else None,
            open_line=row.get("open_line") if pd.notna(row.get("open_line")) else None,
            current_line=row.get("current_line") if pd.notna(row.get("current_line")) else None,
            injury_news=row.get("injury_news") if pd.notna(row.get("injury_news")) else None,
            key_number_note=row.get("key_number_note") if pd.notna(row.get("key_number_note")) else None,
            market_display=row.get("market_display", ""),  # v1.2: market-aware ML move threshold
        )

        game = row.get("game")
        side = row.get("side")
        mkt = row.get("market_display")
        tb = str(row.get("timing_bucket") or "").lower()

        # v3 scorer disabled — v4 reaction engine is live
        from scoring_reaction import score_reaction, classify_reaction_live
        from path_behavior import classify_path_behavior
        _v3_row_dict = row.to_dict()
        # Inject path behavior from row_state tracking
        _v3_row_dict["l1_path_behavior"] = classify_path_behavior(_v3_row_dict)
        # Inject cross-market alignment
        _cm_gid = str(row.get("game_id", "")).strip()
        _cm_sk = str(row.get("side_key", "")).strip()
        _cm_data = _cross_mkt_ctx.get((_cm_gid, _cm_sk), {})
        _v3_row_dict.update(_cm_data)
        _v3_row_dict.update(_game_favored.get(_cm_gid, {}))
        _v3_row_dict["spread_move_map"] = _spread_move_map
        _v4_result = score_reaction(_v3_row_dict)
        _semantic_ctx = _market_context_lookup.get((_cm_gid, str(mkt).strip()), {})
        _semantic_result = classify_reaction_live(
            _v3_row_dict,
            market_rows=_semantic_ctx.get("market_rows"),
            evaluated_side=str(side).strip(),
            pressure_side=_semantic_ctx.get("pressure_side"),
        )
        _v4_result.update(_semantic_result)
        score = _v4_result["reaction_score"]
        _v3_result = {}

        # ---- BIG DOG DARK GREEN NOTE (visual only) ----
        # Add a warning when a DARK_GREEN moneyline is a very large underdog
        try:
            if (
                mkt == "MONEYLINE"
                and color == "DARK_GREEN"
                and pd.notna(row.get("current_odds"))
                and int(row["current_odds"]) >= 300
            ):
                expl = f"{expl} | ÃƒÂƒÃ‚Â¢ÃƒÂ…Ã‚Â¡ÃƒÂ‚Ã‚Â ÃƒÂƒÃ‚Â¯ÃƒÂ‚Ã‚Â¸ÃƒÂ‚Ã‚Â Big underdog moneyline (+{int(row['current_odds'])})"
        except Exception:
            pass

        # legacy ml_green spread mutation disabled with v3 scorer removal

        colors.append(color)
        explains.append(expl)

        scores.append(score)
        _v4_results.append(_v4_result)


    latest = latest.copy()

    latest["color"] = colors

    latest["why"] = explains
    latest["confidence_score"] = [r.get("reaction_score", 0) for r in _v4_results]
    latest["total_score"] = [r.get("reaction_score", 0) for r in _v4_results]
    latest["sharp_score"] = 0
    latest["consensus_score"] = 0
    latest["retail_score"] = 0
    latest["timing_modifier"] = 0
    latest["cross_market_adj"] = 0
    latest["market_reaction_score"] = 0.0
    latest["market_reaction_detail"] = ""
    latest["l1_path_behavior"] = "UNKNOWN"
    latest["pattern_primary"] = [r.get("reaction_state", "NOISE") for r in _v4_results]
    latest["pattern_secondary"] = ""
    latest["consensus_tier"] = 0

    # v3.3f: delta columns from row_state (previous cycle tracking)
    _l2nb = latest["l2_n_books"] if "l2_n_books" in latest.columns else pd.Series(0, index=latest.index)
    _last_l2nb = latest["last_l2_n_books"] if "last_l2_n_books" in latest.columns else pd.Series(0, index=latest.index)
    latest["l2_book_count_delta"] = (
        pd.to_numeric(_l2nb, errors="coerce").fillna(0) -
        pd.to_numeric(_last_l2nb, errors="coerce").fillna(0)
    ).astype(int)
    _last_ct = latest["last_consensus_tier"] if "last_consensus_tier" in latest.columns else pd.Series(0, index=latest.index)
    latest["consensus_tier_prev"] = pd.to_numeric(_last_ct, errors="coerce").fillna(0).astype(int)

    latest["score_explanation"] = [r.get("reason", "") for r in _v4_results]
    latest["v4_state"] = [r.get("reaction_state", "") for r in _v4_results]
    latest["v4_score"] = [r.get("reaction_score", "") for r in _v4_results]
    latest["v4_decision"] = [r.get("decision", "") for r in _v4_results]
    latest["semantic_reaction_state"] = [r.get("semantic_reaction_state", "") for r in _v4_results]
    latest["semantic_signal_class"] = [r.get("semantic_signal_class", "") for r in _v4_results]
    latest["semantic_owning_side"] = [r.get("semantic_owning_side", "none") for r in _v4_results]
    latest["semantic_decision"] = [r.get("semantic_decision", "") for r in _v4_results]
    latest["semantic_source"] = [r.get("semantic_source", "") for r in _v4_results]
    # Controlled semantic consumption: semantic layer corrects displayed state/ownership
    # when true same-market context exists. Decision stays with scorer/certification.
    try:
        _semantic_source = latest["semantic_source"].fillna("").astype(str).str.lower()
        _semantic_state = latest["semantic_reaction_state"].fillna("").astype(str).str.upper()
        _coarse_state = latest["v4_state"].fillna("").astype(str).str.upper()
        _ff_mask = (
            _semantic_source.eq("market_context") &
            _semantic_state.isin(["FOLLOW", "FADE"]) &
            _coarse_state.isin(["FOLLOW", "FADE"])
        )
        if _ff_mask.any():
            latest.loc[_ff_mask, "pattern_primary"] = latest.loc[_ff_mask, "semantic_reaction_state"]
            latest.loc[_ff_mask, "v4_state"] = latest.loc[_ff_mask, "semantic_reaction_state"]
        _freeze_mask = (
            _semantic_source.eq("market_context") &
            _semantic_state.str.startswith("FREEZE_") &
            _coarse_state.isin(["FREEZE", "STALE", "NOISE"])
        )
        if _freeze_mask.any():
            latest.loc[_freeze_mask, "pattern_primary"] = latest.loc[_freeze_mask, "semantic_reaction_state"]
            latest.loc[_freeze_mask, "v4_state"] = latest.loc[_freeze_mask, "semantic_reaction_state"]
    except Exception:
        pass
    # Temporary debug — ML price credibility (remove after 1-2 runs)
    latest["ml_implied_prob"] = 0.0
    latest["ml_cred_mult"] = 1.0
    latest["sharp_base_pre_cred"] = 0.0
    latest["sharp_base_post_cred"] = 0.0

    # [v3.3b DEBUG] ML Direction Audit — sample ML rows for verification

    # DK vs Market: difference between DK line and consensus line (per side)
    def _compute_dk_vs_mkt(row):
        try:
            cl = float(row.get("l2_consensus_line", 0) or 0)
            if cl == 0:
                return None
            mkt = str(row.get("market_display", row.get("market", ""))).strip().upper()
            if mkt == "MONEYLINE":
                return None
            dk_val = row.get("current_line_val")
            if dk_val is None or (isinstance(dk_val, float) and pd.isna(dk_val)):
                return None
            dk_val = float(dk_val)
            return round(dk_val - cl, 1)
        except Exception:
            return None
    latest["dk_vs_mkt"] = latest.apply(_compute_dk_vs_mkt, axis=1)

    _n_total = len(latest)
    _n_l1 = 0

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

        today_ny = pd.Timestamp.now(tz="America/New_York").normalize()
        window_start = today_ny.tz_convert("UTC")
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
        # v3: score_bucket derived from v3 thresholds (display only, certification is authoritative)
        score_bucket = 'HIGH' if score_num >= 70 else ('BET' if score_num >= 67 else ('LEAN' if score_num >= 60 else ''))
    except Exception:
        score_bucket = ''
    # --- end score vars ---
    
    latest["_score_num"] = pd.to_numeric(latest["confidence_score"], errors="coerce").fillna(50.0)
    # --- v1.1 METRICS TAP (true side-state before aggregation) ---
    try:
        _metrics_df = latest.copy()

        if not _metrics_df.empty:
            _metrics_df = _metrics_df.rename(columns={
                "market_display": "market",
                "confidence_score": "model_score"
            })

            if False: update_row_state_and_signal_ledger(_metrics_df)
    except Exception as _e:
        print("[metrics] tap failed:", repr(_e))


    # Favored side = max score within the game+market
    # ===== METRICS TAP (SIDE LEVEL — PRE AGGREGATION) =====
    try:
        _metrics_side = latest.copy()

        if 'side' not in _metrics_side.columns:
            if 'team' in _metrics_side.columns:
                _metrics_side['side'] = _metrics_side['team'].astype(str)
            elif 'side_key' in _metrics_side.columns:
                _metrics_side['side'] = _metrics_side['side_key'].astype(str)
            else:
                _metrics_side['side'] = ''

        if 'model_score' not in _metrics_side.columns and '_score_num' in _metrics_side.columns:
            _metrics_side['model_score'] = _metrics_side['_score_num']

        if 'market' not in _metrics_side.columns and 'market_display' in _metrics_side.columns:
            _metrics_side['market'] = _metrics_side['market_display']

        if 'net_edge' not in _metrics_side.columns:
            _metrics_side['net_edge'] = ''

        if 'timing_bucket' not in _metrics_side.columns:
            _metrics_side['timing_bucket'] = ''

        update_row_state_and_signal_ledger(_metrics_side)

    except Exception:
        pass
    # ===== END METRICS TAP =====

    idx_fav = latest.groupby(game_keys)["_score_num"].idxmax()

    # v3: include layer + v3 component columns
    _fav_cols = game_keys + ["game", "sport_label", "side", "_score_num", "market_read", "timing_bucket", "canonical_key", "side_key"]
    _v2_fav_cols = [
                    "timing_modifier", "cross_market_adj",
                    "market_reaction_score", "market_reaction_detail",
                    "pattern_primary", "pattern_secondary", "score_explanation",
                    "semantic_reaction_state", "semantic_signal_class", "semantic_owning_side",
                    "semantic_decision", "semantic_source",
                    "b2b_flag", "home_injury_count", "away_injury_count",
                    "wind_mph", "temp_f", "precip_prob", "weather_flag",
                    "sp_name_home", "sp_name_away", "sp_era_home", "sp_era_away",
                    "sp_flag_home", "sp_flag_away", "sp_hand_home", "sp_hand_away",
                    "park_factor", "goalie_home", "goalie_away",
                    "goalie_flag_home", "goalie_flag_away",
                    "sport_context_adj", "sport_context_flag",
                    "v4_state", "v4_score", "v4_decision", "confidence_score",
                    # DK signal data — must carry through to dashboard
                    "bets_pct", "money_pct", "divergence_D",
                    "open_line", "current_line", "open_odds", "current_odds",
                    "effective_move_mag", "line_move_open", "odds_move_open",
                    "move_dir", "meaningful_move",
                    "line_settled_ticks", "line_dir_changes", "line_last_dir", "line_max_move"]
    for _vc in _v2_fav_cols:
        if _vc in latest.columns:
            _fav_cols.append(_vc)

    fav_rows = latest.loc[
        idx_fav,
        [c for c in _fav_cols if c in latest.columns]
    ].copy()

    fav_rows = fav_rows.rename(columns={
        "side": "favored_side",
        "_score_num": "game_confidence",
    })

    # Min/Max side scores within each game+market
    min_rows = latest.groupby(game_keys)["_score_num"].min().reset_index().rename(columns={"_score_num": "min_side_score"})
    max_rows = latest.groupby(game_keys)["_score_num"].max().reset_index().rename(columns={"_score_num": "max_side_score"})

    game_view = (
        fav_rows
        .merge(min_rows, on=game_keys, how="left")
        .merge(max_rows, on=game_keys, how="left")
    )

    # v3.2: net_edge = raw max - min, no scaling. v3 scores are properly bounded.
    game_view["net_edge"] = (game_view["max_side_score"] - game_view["min_side_score"]).round(1)
    # total_score = game_confidence (v3 final_score is the complete score, no edge additive)
    game_view["total_score"] = pd.to_numeric(game_view["game_confidence"], errors="coerce").fillna(0).round(1)

    # v3.2: Cross-market handled inside scoring_v3.compute_cross_market_sanity()

    # -----------------------------

    # ── v3.2 CERTIFICATION + EXECUTION EXPRESSION ──
    from certification_v3 import certify_decision
    from execution_expression import compute_expression

    # Build persistence data from row_state for STRONG gates
    _rs2_map = {}
    try:
        _rs2 = pd.read_csv("data/row_state.csv", dtype=str, keep_default_na=False)
        if "market_display" in _rs2.columns:
            _rs2 = _rs2.drop(columns=["market_display"])
        _rs2 = _rs2.rename(columns={"market": "market_display"})
        _rs2["_rs_side_norm"] = _rs2.apply(
            lambda r: normalize_side_key(str(r.get("sport","")), str(r.get("market_display","")), str(r.get("side",""))), axis=1
        )
        _rs2["strong_streak"] = pd.to_numeric(_rs2["strong_streak"], errors="coerce").fillna(0).astype(int)
        _rs2["last_score"] = pd.to_numeric(_rs2["last_score"], errors="coerce").fillna(0.0)
        _rs2["peak_score"] = pd.to_numeric(_rs2["peak_score"], errors="coerce").fillna(0.0)
        _rs2_dedup = _rs2.drop_duplicates(subset=["sport","game_id","market_display","_rs_side_norm"], keep="last")
        for _, _rr in _rs2_dedup.iterrows():
            _rk = (str(_rr["sport"]), str(_rr["game_id"]), str(_rr["market_display"]), str(_rr["_rs_side_norm"]))
            _rs2_map[_rk] = _rr
    except Exception as _rs_err:
        print(f"[v4] row_state load for persistence: {repr(_rs_err)}")

    def _v3_certify_row(r):
        """Run v3 certification + execution expression on a game_view row."""
        _score = float(r.get("game_confidence", 50) or 50)
        _ne = float(r.get("net_edge", 0) or 0)
        _sp = str(r.get("sport", "")).strip()
        _gid = str(r.get("game_id", "")).strip()
        _mkt = str(r.get("market_display", "")).strip()
        _fav = str(r.get("favored_side", "")).strip()
        _snorm = normalize_side_key(_sp, _mkt, _fav)
        _ekey = (_sp, _gid, _mkt, _snorm)

        # Get persistence data
        _rs_row = _rs2_map.get(_ekey, None)
        _ss, _ls, _ps = 0, 0.0, 0.0
        if _rs_row is not None:
            try: _ss = int(str(_rs_row.get("strong_streak", 0)).strip() or "0")
            except (ValueError, TypeError): _ss = 0
            try: _ls = float(str(_rs_row.get("last_score", 0)).strip() or "0")
            except (ValueError, TypeError): _ls = 0.0
            try: _ps = float(str(_rs_row.get("peak_score", 0)).strip() or "0")
            except (ValueError, TypeError): _ps = 0.0

        row_dict = r.to_dict()
        cert = certify_decision(row_dict, _score, _ne,
                                strong_streak=_ss, peak_score=_ps, last_score=_ls)
        decision = cert["decision"]

        expr = compute_expression(row_dict, decision, _score)

        return pd.Series({
            "game_decision": decision,
            "is_locked": cert["is_locked"],
            "strong_eligible": cert["strong_eligible"],
            "blocked_by": cert.get("blocked_by", ""),
            "expression": expr["expression"],
            "expression_reason": expr["expression_reason"],
        })

    _cert_result = game_view.apply(_v3_certify_row, axis=1)
    _expected_cert_cols = ["game_decision", "is_locked", "strong_eligible", "blocked_by", "expression", "expression_reason"]
    if not isinstance(_cert_result, pd.DataFrame) or _cert_result.empty:
        _cert_result = pd.DataFrame(index=game_view.index)
    for _cc in _expected_cert_cols:
        if _cc not in _cert_result.columns:
            _cert_result[_cc] = ""
    _cert_result["game_decision"] = _cert_result["game_decision"].replace("", "NO_BET")
    _cert_result["is_locked"] = _cert_result["is_locked"].replace("", False)
    _cert_result["strong_eligible"] = _cert_result["strong_eligible"].replace("", False)
    game_view["game_decision"] = _cert_result["game_decision"]
    game_view["is_locked"] = _cert_result["is_locked"]
    game_view["strong_eligible"] = _cert_result["strong_eligible"]
    game_view["blocked_by"] = _cert_result["blocked_by"]
    game_view["expression"] = _cert_result["expression"]
    game_view["expression_reason"] = _cert_result["expression_reason"]
    if "v4_decision" in game_view.columns:
        _dec_rank = {"NO_BET": 0, "LEAN": 1, "BET": 2, "STRONG_BET": 3}
        _rank_dec = {v: k for k, v in _dec_rank.items()}

        def _combine_decisions(cert_dec, v4_dec):
            cert_u = str(cert_dec or "NO_BET").strip().upper() or "NO_BET"
            v4_u = str(v4_dec or "").strip().upper() or cert_u
            cert_r = _dec_rank.get(cert_u, 0)
            v4_r = _dec_rank.get(v4_u, 0)
            return _rank_dec[min(cert_r, v4_r)]

        game_view["game_decision"] = [
            _combine_decisions(_c, _v)
            for _c, _v in zip(game_view["game_decision"], game_view["v4_decision"])
        ]
    # Minimum score gap gate — downgrade BET to LEAN when net_edge < 5
    _gap = pd.to_numeric(game_view["net_edge"], errors="coerce").fillna(0)
    game_view.loc[(game_view["game_decision"] == "BET") & (_gap < 5), "game_decision"] = "LEAN"
    # Practical market-quality guardrails
    _odds = pd.to_numeric(game_view.get("current_odds", 0), errors="coerce").fillna(0)
    _state = game_view.get("pattern_primary", "").fillna("").astype(str).str.upper()
    _mkt = game_view.get("market_display", "").fillna("").astype(str).str.upper()
    _sport = game_view.get("sport", "").fillna("").astype(str).str.lower()
    _money = pd.to_numeric(game_view.get("money_pct", 0), errors="coerce").fillna(0)
    _settled = pd.to_numeric(game_view.get("line_settled_ticks", 0), errors="coerce").fillna(0)
    _dir_changes = pd.to_numeric(game_view.get("line_dir_changes", 0), errors="coerce").fillna(0)
    _stable = (_settled >= 2) & (_dir_changes == 0)
    _top_state = _state.isin(["FOLLOW", "INITIATED"])

    _extreme_fav_ok = _top_state & (_gap >= 8) & _stable
    _extreme_dog_ok = _top_state & (_gap >= 10) & _stable
    _spread_plus_ok = _top_state & (_gap >= 10) & _stable

    _extreme_fav_mask = (
        (game_view["game_decision"] == "BET") &
        (_mkt == "MONEYLINE") &
        (_odds <= -500) &
        (~_extreme_fav_ok)
    )
    _extreme_dog_mask = (
        (game_view["game_decision"] == "BET") &
        (_mkt == "MONEYLINE") &
        (_odds >= 180) &
        (~_extreme_dog_ok)
    )
    _spread_plus_mask = (
        (game_view["game_decision"] == "BET") &
        (_mkt == "SPREAD") &
        (_sport.isin(["mlb", "nhl"])) &
        (_odds >= 150) &
        (~_spread_plus_ok)
    )
    _freeze_bet_ok = (_money >= 75) & (_gap >= 8) & _stable
    _freeze_bet_mask = (
        (game_view["game_decision"] == "BET") &
        (_state == "FREEZE") &
        (~_freeze_bet_ok)
    )
    game_view.loc[
        _extreme_fav_mask | _extreme_dog_mask | _spread_plus_mask | _freeze_bet_mask,
        "game_decision"
    ] = "LEAN"
    print(f"[v4] certified {len(game_view)} rows: "
          f"{(game_view['game_decision'] == 'STRONG_BET').sum()} STRONG, "
          f"{(game_view['game_decision'] == 'BET').sum()} BET, "
          f"{(game_view['game_decision'] == 'LEAN').sum()} LEAN, "
          f"{game_view['is_locked'].sum()} LOCKED")
    game_view["opp_weak"] = game_view["min_side_score"] <= 35.0
    game_view["opp_weak_mark"] = game_view["opp_weak"].apply(lambda x: "!" if bool(x) else "")
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
    # RESHAPE (UI ONLY): one row per game with grouped SPREAD/ML/TOTAL columns
    # -----------------------------
    try:
        base = ["sport", "game_id", "game", "sport_label", "game_time", "_game_time"]

        tmp = game_view.copy()
        tmp["market_display"] = tmp["market_display"].astype(str)

        # v3 columns to carry through reshape (from primary market)
        _v2_carry = [
                      "timing_modifier", "cross_market_adj",
                      "market_reaction_score", "market_reaction_detail",
                      "pattern_primary", "pattern_secondary", "score_explanation",
                      "semantic_reaction_state", "semantic_signal_class", "semantic_owning_side",
                      "semantic_decision", "semantic_source",
                      "expression", "expression_reason",
                      "b2b_flag", "home_injury_count", "away_injury_count",
                      "wind_mph", "temp_f", "precip_prob", "weather_flag",
                      "sp_name_home", "sp_name_away", "sp_era_home", "sp_era_away",
                      "sp_flag_home", "sp_flag_away", "sp_hand_home", "sp_hand_away",
                      "park_factor",
                      "goalie_home", "goalie_away", "goalie_flag_home", "goalie_flag_away",
                      "sport_context_adj", "sport_context_flag",
                      # Temporary debug — ML price credibility (remove after 1-2 runs)
                      "ml_implied_prob", "ml_cred_mult",
                      "sharp_base_pre_cred", "sharp_base_post_cred"]

        # Determine which market carries context (SPREAD preferred, fallback to first available)
        _carry_market = None
        for _cm in ("SPREAD", "MONEYLINE", "TOTAL"):
            if not tmp[tmp["market_display"] == _cm].empty:
                _carry_market = _cm
                break

        parts = []
        for m in ("SPREAD", "MONEYLINE", "TOTAL"):
            sub = tmp[tmp["market_display"] == m].copy()
            if sub.empty:
                continue

            _keep = base + ["game_confidence", "confidence_score", "favored_side", "net_edge", "total_score", "game_decision", "v4_state", "v4_score", "v4_decision"]
            # Carry v2 + context columns from primary market
            if m == _carry_market:
                for _vc in _v2_carry:
                    if _vc in sub.columns:
                        _keep.append(_vc)

            sub = sub[[c for c in _keep if c in sub.columns]].copy()
            sub = sub.rename(columns={
                "game_confidence": f"{m}_model_score",
                "favored_side": f"{m}_favored",
                "net_edge": f"{m}_net_edge",
                "game_decision": f"{m}_decision",
            })
            parts.append(sub)

        if parts:
            game_view_wide = parts[0]
            for pp in parts[1:]:
                game_view_wide = game_view_wide.merge(pp, on=base, how="outer")
        else:
            game_view_wide = tmp[base].drop_duplicates().copy()

        # Display Net Edge (single column): max of available market net edges, capped
        edge_cols = [c for c in game_view_wide.columns if c.endswith("_net_edge")]
        if edge_cols:
            _raw_wide_edge = (
                pd.DataFrame({c: pd.to_numeric(game_view_wide[c], errors="coerce") for c in edge_cols})
                .max(axis=1)
                .fillna(0.0)
                .round(1)
            )
            game_view_wide["net_edge"] = _raw_wide_edge.clip(upper=20.0)
        else:
            game_view_wide["net_edge"] = 0.0

        # Ensure expected columns exist (no NaN/None in UI)
        for m in ("SPREAD", "MONEYLINE", "TOTAL"):
            for c in (f"{m}_decision", f"{m}_model_score", f"{m}_favored"):
                if c not in game_view_wide.columns:
                    game_view_wide[c] = ""

        game_view = game_view_wide.reset_index(drop=True)

        # --- BOOTSTRAP WRITE: ensure dashboard exists before any reader ---
        try:
            os.makedirs("data", exist_ok=True)
            game_view.to_csv("data/dashboard.csv", index=False)
            print("[ok] wrote dashboard csv (bootstrap)")
        except Exception as e:
            print("[dash] failed writing dashboard csv:", repr(e))


        # Re-sort after reshape (wide merge can scramble row order)
        try:
            if "_game_time" in game_view.columns:
                game_view = game_view.sort_values(
                    ["sport_label", "_game_time", "game"],
                    na_position="last"
                ).reset_index(drop=True)
        except Exception:
            pass

    except Exception:
        # If anything goes wrong, keep the existing game_view to avoid breaking report
        pass


    # -----------------------------
    # TABLE HEADERS + HYBRID ROW BUILD (UI ONLY)
    # GAME rows visible; SIDE rows hidden + directly under their GAME row
    # -----------------------------

    # --- Game Time toggle (display-only) ---
    show_game_time = True  # always show column; cell renderer will blank if missing

    # --- GAME header columns (this is the table schema) ---
    # Must match the GAME row <td> schema exactly.
    header_cols = (
        ["Sport", "Game"]
        + (["Game Time"] if show_game_time else [])
        + [
            "SPREAD Decision","SPREAD Score","SPREAD Side",
            "ML Decision","ML Score","ML Side",
            "TOTAL Decision","TOTAL Score","TOTAL Side",
            "Score Gap",
            "Layer", "Pattern", "Edge Type", "Sharp Action",
            "Breakdown"
        ]
    )
    colspan = len(header_cols)

    # Build a fast lookup of SIDE rows by parent key so we can render children immediately under each game row
    latest["_parent_gk"] = (
        latest["sport"].astype(str) + "|" +
        latest["game_id"].astype(str)
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

    # v2.0 tooltip descriptions for patterns
    _pattern_tips = {
        "A": "Sharp vs Public: Sharp books moved, public betting opposite — highest edge",
        "B": "Retail Alignment: All layers agree — real but cap 70 (public already priced in)",
        "C": "Public Only: No sharp move, heavy public action — penalty applied",
        "D": "Stale Price: Sharp + consensus moved but DK line hasn't caught up — value window",
        "E": "Consensus Rejects: Sharp moved but consensus didn't follow — signal weakened",
        "F": "Late Snap: Rapid late move with noisy data — high risk, penalty applied",
        "G": "Reverse Line Movement: Public bets heavy but sharp money pushing line opposite — strong signal",
        "N": "No pattern detected (L3-only or insufficient data)",
    }
    _layer_labels = {
        "L123": "FULL",
        "L13": "PARTIAL",
        "L23": "PARTIAL",
        "L3_ONLY": "LIMITED",
    }
    _layer_tips = {
        "L123": "All data sources available — highest confidence",
        "L13": "Some data missing — score may be less reliable",
        "L23": "Some data missing — score may be less reliable",
        "L3_ONLY": "Minimal data available — lowest confidence",
    }

    def _safe_str(val):
        s = "" if val is None else str(val)
        return "" if s.strip().lower() == "nan" else s.strip()

    def _build_game_intel(gr, sg):
        """Build a plain-English intelligence summary for a game."""
        parts = []
        try:
            if sg is None or sg.empty:
                return ""
            fav = str(gr.get("favored_side", "")).strip()
            mr = str(gr.get("market_read", "")).strip()
            lm = str(gr.get("layer_mode", "")).strip()
            mkt = str(gr.get("market_display", "")).strip()

            fav_row = sg[(sg["side"].astype(str).str.strip() == fav) &
                         (sg["market_display"].astype(str).str.strip() == mkt)]
            if fav_row.empty:
                fav_row = sg[sg["side"].astype(str).str.strip() == fav]
            if fav_row.empty:
                return ""
            fr = fav_row.iloc[0]

            bets = float(fr.get("bets_pct", 0) or 0)
            money = float(fr.get("money_pct", 0) or 0)
            fav_short = fav.split()[-1] if fav else "?"

            # 1. Money story
            if bets > 0:
                if money > bets * 1.5 and bets < 40:
                    parts.append(f"Smart money flowing to {fav_short} - {int(money)}% of dollars but only {int(bets)}% of bets")
                elif bets > 70 and money > 80:
                    parts.append(f"Heavy public action on {fav_short} ({int(bets)}% bets, {int(money)}% money)")
                elif bets < 25 and money > bets * 1.3:
                    parts.append(f"Big bettors quietly backing {fav_short} ({int(money)}% money on just {int(bets)}% of bets)")
                elif bets < 25:
                    parts.append(f"Contrarian play - only {int(bets)}% of bets on {fav_short}")
                elif abs(money - bets) < 10:
                    parts.append(f"Even action on {fav_short} ({int(bets)}% bets, {int(money)}% money)")
                else:
                    parts.append(f"{int(bets)}% of bets, {int(money)}% of money on {fav_short}")

            # 2. Line movement
            import re as _re
            def _extract_line_num(val):
                s = str(val).strip()
                m = _re.search(r'([+-]?\d+(?:\.\d+)?)\s*$', s)
                if m:
                    return m.group(1)
                m = _re.search(r'@\s*([+-]?\d+)', s)
                return m.group(1) if m else s
            o = _extract_line_num(fr.get("open_line", ""))
            c = _extract_line_num(fr.get("current_line", ""))
            md = int(float(fr.get("move_dir", 0) or 0))
            if o and c and o != c:
                if md == 1:
                    parts.append(f"Line moved {o} to {c}, confirming this side")
                elif md == -1:
                    parts.append(f"Line moved {o} to {c}, against this side")
                else:
                    parts.append(f"Line moved {o} to {c}")
            elif o and o == c:
                parts.append("Line has not moved")

            # 3. Sharp/consensus confirmation (v3: use component scores)
            _sharp = float(fr.get("sharp_score", 0) or 0)
            _cons = float(fr.get("consensus_score", 0) or 0)
            if _sharp > 2 and _cons > 2:
                parts.append("Sharp books and consensus both confirm")
            elif _sharp > 2:
                parts.append("Sharp books confirm")
            elif _sharp < -2:
                parts.append("Sharp books disagree")
            if _cons > 2 and _sharp <= 2:
                parts.append("Consensus books confirm")

            # 4. Data quality (only if limited)
            if lm == "L3_ONLY":
                parts.append("Limited data - DraftKings only")
            elif lm == "L13":
                parts.append("No consensus data available")

        except Exception:
            pass
        return " | ".join(parts)

    for _, gr in game_view.iterrows():
        gk = f"{gr.get('sport','')}|{gr.get('game_id','')}"

        # --- GAME SUMMARY ROW (visible) ---
        gt = _time_cell(gr)
        # v2.3: Game intelligence summary
        _intel_sg = side_groups.get(gk)
        _intel_text = _build_game_intel(gr, _intel_sg)
        _intel_html = f'<br><span class="intel-summary">{_intel_text}</span>' if _intel_text else ""

        # v2.0 badge cells
        _lm = _safe_str(gr.get("layer_mode"))
        if _lm:
            _lm_css = f"layer-{_lm}" if _lm in ("L123","L13","L23","L3_ONLY") else "layer-L3_ONLY"
            _lm_lbl = _layer_labels.get(_lm, _lm)
            _lm_tip = _layer_tips.get(_lm, "")
            _lm_td = f'<td><span class="layer-badge {_lm_css}" title="{_lm_tip}">{_lm_lbl}</span></td>'
        else:
            _lm_td = "<td></td>"

        _pat = _safe_str(gr.get("pattern_primary"))
        _pat2 = _safe_str(gr.get("pattern_secondary"))
        if _pat and _pat != "NEUTRAL":
            _pat_td = f'<td><span class="pattern-badge" title="v3 pattern label">{_pat}</span>'
            if _pat2:
                _pat_td += f' <span class="pattern-secondary" title="secondary pattern">({_pat2})</span>'
            _pat_td += '</td>'
        else:
            _pat_td = "<td></td>"

        _expr = _safe_str(gr.get("expression"))
        _expr_td = f'<td><span class="edge-label" title="Execution expression">{_expr}</span></td>' if _expr else "<td></td>"

        # Score decomposition tooltip (v3: 5 components)
        _decomp_parts = []
        try:
            _sh = float(gr.get("sharp_score", 0) or 0)
            _co = float(gr.get("consensus_score", 0) or 0)
            _rt = float(gr.get("retail_score", 0) or 0)
            _tm = float(gr.get("timing_modifier", 0) or 0)
            _xm = float(gr.get("cross_market_adj", 0) or 0)
            if _sh != 0: _decomp_parts.append(f"Sharp:{_sh:+.0f}")
            if _co != 0: _decomp_parts.append(f"Cons:{_co:+.0f}")
            if _rt != 0: _decomp_parts.append(f"Retail:{_rt:+.0f}")
            if _tm != 0: _decomp_parts.append(f"Time:{_tm:+.0f}")
            if _xm != 0: _decomp_parts.append(f"Cross:{_xm:+.0f}")
        except Exception:
            pass
        _decomp_str = " | ".join(_decomp_parts) if _decomp_parts else ""

        # Score breakdown cell (visible, not just tooltip)
        _bkdn_parts = []
        try:
            _sh = float(gr.get("sharp_score", 0) or 0)
            _co = float(gr.get("consensus_score", 0) or 0)
            _rt = float(gr.get("retail_score", 0) or 0)
            if _sh != 0: _bkdn_parts.append(f"Sh:{_sh:+.0f}")
            if _co != 0: _bkdn_parts.append(f"Co:{_co:+.0f}")
            if _rt != 0: _bkdn_parts.append(f"Rt:{_rt:+.0f}")
        except Exception:
            pass
        _bkdn_td = f'<td style="font-size:10px;color:#555;white-space:nowrap;">{" ".join(_bkdn_parts)}</td>' if _bkdn_parts else "<td></td>"

        # Mkt Line cell (consensus line — compare visually to Open/Current)
        _dvm_td = "<td></td>"
        try:
            _cons = float(gr.get("l2_consensus_line", 0) or 0)
            if _cons != 0:
                _dvm_td = f'<td style="font-size:10px;text-align:center;color:#555;" title="Market consensus (31 books)">{_cons:.1f}</td>'
        except Exception:
            pass

        # B2B badge
        _b2b = _safe_str(gr.get("b2b_flag"))
        _b2b_html = ""
        if _b2b and _b2b not in ("", "nan"):
            _b2b_lbl = {"HOME_B2B": "H-B2B", "AWAY_B2B": "A-B2B", "BOTH_B2B": "B2B"}.get(_b2b, _b2b)
            _b2b_html = f' <span class="b2b-badge">{_b2b_lbl}</span>'

        # Injury badges
        _inj_html = ""
        try:
            _hi = int(float(gr.get("home_injury_count", 0) or 0))
            _ai = int(float(gr.get("away_injury_count", 0) or 0))
            if _hi >= 2:
                _inj_html += f' <span class="inj-badge" title="Home {_hi} injuries">H:{_hi}inj</span>'
            if _ai >= 2:
                _inj_html += f' <span class="inj-badge" title="Away {_ai} injuries">A:{_ai}inj</span>'
        except Exception:
            pass

        # Weather badge (outdoor sports)
        _wx_html = ""
        try:
            _wx_flag = _safe_str(gr.get("weather_flag"))
            _wind = int(float(gr.get("wind_mph", 0) or 0))
            _temp = int(float(gr.get("temp_f", 70) or 70))
            _precip = int(float(gr.get("precip_prob", 0) or 0))
            if _wx_flag and _wx_flag not in ("", "nan", "DOME"):
                _wx_icon = ""
                if "HIGH_WIND" in _wx_flag:
                    _wx_icon = "&#x1F32C;"  # wind face
                elif "WINDY" in _wx_flag:
                    _wx_icon = "&#x1F32C;"
                if "RAIN" in _wx_flag:
                    _wx_icon = "&#x1F327;"  # rain cloud
                if "COLD" in _wx_flag:
                    _wx_icon = "&#x2744;"   # snowflake
                _wx_detail = f"{_temp}F {_wind}mph"
                if _precip > 0:
                    _wx_detail += f" {_precip}%rain"
                _wx_html = f' <span class="wx-badge" title="{_wx_detail}">{_wx_icon} {_wx_flag.replace("|","/")}</span>'
            elif _wx_flag == "DOME":
                _wx_html = ' <span class="wx-dome" title="Dome stadium">&#x1F3DF;</span>'
            elif _safe_str(gr.get("sport")).lower() in ("nfl","ncaaf","mlb") and _wind > 0:
                # Outdoor but no weather flags — show conditions anyway
                _wx_detail = f"{_temp}F {_wind}mph"
                if _precip > 0:
                    _wx_detail += f" {_precip}%rain"
                _wx_html = f' <span class="wx-clear" title="{_wx_detail}">&#x2600;</span>'
        except Exception:
            pass

        # Sport context badges (pitcher, goalie, park factor, rankings)
        _ctx_html = ""
        try:
            _sport_lower = _safe_str(gr.get("sport")).lower()
            if _sport_lower == "mlb":
                _sp_home = _safe_str(gr.get("sp_name_home"))
                _sp_away = _safe_str(gr.get("sp_name_away"))
                _sp_era_h = gr.get("sp_era_home")
                _sp_era_a = gr.get("sp_era_away")
                _sp_flag_h = _safe_str(gr.get("sp_flag_home"))
                _sp_flag_a = _safe_str(gr.get("sp_flag_away"))
                _hand_h = _safe_str(gr.get("sp_hand_home"))
                _hand_a = _safe_str(gr.get("sp_hand_away"))
                if _sp_home:
                    _era_str_h = f" {_sp_era_h:.2f}" if _sp_era_h is not None else ""
                    _hand_str_h = f"{_hand_h}HP " if _hand_h else ""
                    _sp_cls_h = "sp-ace" if _sp_flag_h in ("SP_ACE","SP_STRONG") else "sp-weak" if _sp_flag_h in ("SP_WEAK","SP_BAD") else "sp-avg"
                    _ctx_html += f' <span class="sp-badge {_sp_cls_h}" title="Home SP: {_sp_home}">{_hand_str_h}{_sp_home.split()[-1] if _sp_home else "?"}{_era_str_h}</span>'
                if _sp_away:
                    _era_str_a = f" {_sp_era_a:.2f}" if _sp_era_a is not None else ""
                    _hand_str_a = f"{_hand_a}HP " if _hand_a else ""
                    _sp_cls_a = "sp-ace" if _sp_flag_a in ("SP_ACE","SP_STRONG") else "sp-weak" if _sp_flag_a in ("SP_WEAK","SP_BAD") else "sp-avg"
                    _ctx_html += f' <span class="sp-badge {_sp_cls_a}" title="Away SP: {_sp_away}">{_hand_str_a}{_sp_away.split()[-1] if _sp_away else "?"}{_era_str_a}</span>'
                _pf = float(gr.get("park_factor", 1.0) or 1.0)
                if _pf >= 1.05 or _pf <= 0.95:
                    _pf_cls = "park-hit" if _pf >= 1.05 else "park-pitch"
                    _ctx_html += f' <span class="park-badge {_pf_cls}" title="Park factor {_pf:.2f}">&#x26BE; {_pf:.2f}x</span>'
            elif _sport_lower == "nhl":
                _g_home = _safe_str(gr.get("goalie_home"))
                _g_away = _safe_str(gr.get("goalie_away"))
                _gf_home = _safe_str(gr.get("goalie_flag_home"))
                _gf_away = _safe_str(gr.get("goalie_flag_away"))
                if _g_home:
                    _g_cls_h = "g-confirmed" if "CONFIRMED" in _gf_home else "g-probable" if "PROBABLE" in _gf_home else "g-unknown"
                    _g_label_h = "&#x2705;" if "CONFIRMED" in _gf_home else "&#x2753;"
                    _ctx_html += f' <span class="goalie-badge {_g_cls_h}" title="Home G: {_g_home} ({_gf_home})">{_g_label_h} {_g_home.split()[-1]}</span>'
                if _g_away:
                    _g_cls_a = "g-confirmed" if "CONFIRMED" in _gf_away else "g-probable" if "PROBABLE" in _gf_away else "g-unknown"
                    _g_label_a = "&#x2705;" if "CONFIRMED" in _gf_away else "&#x2753;"
                    _ctx_html += f' <span class="goalie-badge {_g_cls_a}" title="Away G: {_g_away} ({_gf_away})">{_g_label_a} {_g_away.split()[-1]}</span>'
            elif _sport_lower == "ncaab":
                _rank_our = gr.get("rank_our")
                _rank_opp = gr.get("rank_opp")
                if _rank_our and int(float(_rank_our)) > 0:
                    _ctx_html += f' <span class="rank-badge" title="Our team rank">#{int(float(_rank_our))}</span>'
                if _rank_opp and int(float(_rank_opp)) > 0:
                    _ctx_html += f' <span class="rank-badge rank-opp" title="Opponent rank">vs #{int(float(_rank_opp))}</span>'
        except Exception:
            pass

        try:
            _sn = int(float(gr.get("l1_n_books", 0) or 0))
        except (ValueError, TypeError):
            _sn = 0
        _cert_tier = _safe_str(gr.get("sharp_cert_tier")).upper()
        _cert_badge = ""
        if _cert_tier == "FULL":
            _cert_badge = ' <span class="sharp-cert sharp-cert-full" title="Sharp Certified: multiple sharp books confirm">SHARP &#x2713;</span>'
        elif _cert_tier == "HALF":
            _cert_badge = ' <span class="sharp-cert sharp-cert-half" title="Sharp Lean: Pinnacle moved meaningfully">SHARP ~</span>'
        if _sn > 0:
            _sn = max(1, min(_sn, 6))
            _spct = round(_sn / 6 * 100)
            _sharp_td = f'<td>{_spct}% <span class="sharp-meter"><span class="sharp-meter-fill sharp-{_sn}"></span></span>{_cert_badge}</td>'
        elif _cert_badge:
            _sharp_td = f'<td>{_cert_badge}</td>'
        else:
            _sharp_td = '<td style="color:#bbb;text-align:center;">&mdash;</td>'

        # Row class based on data quality
        _row_cls = f"game-row data-{_lm.lower()}" if _lm else "game-row data-l3_only"

        rows_html.append(f"""
<tr class="{_row_cls}" data-gamekey="{gk}" onclick="toggleGroup('{gk}')">
  <td>{gr.get("sport_label","")}</td>
  <td><b>{gr.get("game","")}</b>{_b2b_html}{_inj_html}{_wx_html}{_ctx_html}{_intel_html}</td>
  {gt}
  <td>{("" if (gr.get("SPREAD_decision","") is None or str(gr.get("SPREAD_decision","")).lower()=="nan") else str(gr.get("SPREAD_decision","")))}</td>
  <td title="{_decomp_str}">{("" if (gr.get("SPREAD_model_score","") is None or str(gr.get("SPREAD_model_score","")).strip()=="" or str(gr.get("SPREAD_model_score","")).lower()=="nan") else f"{float(gr.get('SPREAD_model_score')):.1f}")}</td>
  <td>{("" if (gr.get("SPREAD_favored","") is None or str(gr.get("SPREAD_favored","")).lower()=="nan") else str(gr.get("SPREAD_favored","")))}</td>

  <td>{("" if (gr.get("MONEYLINE_decision","") is None or str(gr.get("MONEYLINE_decision","")).lower()=="nan") else str(gr.get("MONEYLINE_decision","")))}</td>
  <td>{("" if (gr.get("MONEYLINE_model_score","") is None or str(gr.get("MONEYLINE_model_score","")).strip()=="" or str(gr.get("MONEYLINE_model_score","")).lower()=="nan") else f"{float(gr.get('MONEYLINE_model_score')):.1f}")}</td>
  <td>{("" if (gr.get("MONEYLINE_favored","") is None or str(gr.get("MONEYLINE_favored","")).lower()=="nan") else str(gr.get("MONEYLINE_favored","")))}</td>

  <td>{("" if (gr.get("TOTAL_decision","") is None or str(gr.get("TOTAL_decision","")).lower()=="nan") else str(gr.get("TOTAL_decision","")))}</td>
  <td>{("" if (gr.get("TOTAL_model_score","") is None or str(gr.get("TOTAL_model_score","")).strip()=="" or str(gr.get("TOTAL_model_score","")).lower()=="nan") else f"{float(gr.get('TOTAL_model_score')):.1f}")}</td>
  <td>{("" if (gr.get("TOTAL_favored","") is None or str(gr.get("TOTAL_favored","")).lower()=="nan") else str(gr.get("TOTAL_favored","")))}</td>

  <td><span class="pill edge">{round(float(gr.get("net_edge",0) or 0),1)}</span></td>
  {_lm_td}
  {_pat_td}
  {_expr_td}
  {_sharp_td}
  {_bkdn_td}
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
            market_cell = f"{side_disp} ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â‚Ã‚Â¬ÃƒÂ¢Ã‚Â€Ã‚Â {rr.get('market_display','')}".strip(" ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â‚Ã‚Â¬ÃƒÂ¢Ã‚Â€Ã‚Â")

            # Decision: Bets / Money
            bets_cell = "" if pd.isna(rr.get("bets_pct")) else f"{int(rr['bets_pct'])}%"
            money_cell = "" if pd.isna(rr.get("money_pct")) else f"{int(rr['money_pct'])}%"
            decision_cell = f"B {bets_cell} / $ {money_cell}".strip()

            # Open ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â€Ã‚Â ÃƒÂ¢Ã‚Â€Ã‚Â™ Current
            o = "" if pd.isna(rr.get("open_line")) else str(rr.get("open_line"))
            c = "" if pd.isna(rr.get("current_line")) else str(rr.get("current_line"))
            oc_cell = (o + " ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â€Ã‚Â ÃƒÂ¢Ã‚Â€Ã‚Â™ " + c).strip(" ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â€Ã‚Â ÃƒÂ¢Ã‚Â€Ã‚Â™")

            # Side model score
            try:
                sc = "" if pd.isna(rr.get("confidence_score")) else f"{float(rr.get('confidence_score')):.1f}"
            except Exception:
                sc = ""

            # DK Signal (renamed from market_read)
            mr = str(rr.get("market_read","") or "").strip()
            if mr:
                mr = f"DK: {mr}"

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
  <td>- {rr.get("why","")}</td>
  {gt_side}
  <td colspan="{colspan - 4}">{market_cell} | {decision_cell} | {oc_cell} | {sc} | {mr}</td>
  <td></td>
</tr>
""")



    # -----------------------------
    # WRITE DASHBOARD HTML (WIDE ONE-ROW-PER-GAME)
    # Source of truth: data/dashboard.csv (already normalized; no NaNs)
    # -----------------------------
    try:
        import pandas as _pd

        ensure_data_dir()

        # -----------------------------
        # FREEZE DECISION SNAPSHOT (canonical side grain)
        # -----------------------------
        try:
            _side_cols = ["sport","game_id","market_display","side"]
            if "market_read" in latest.columns:
                _side_cols = _side_cols + ["market_read"]
            if "timing_bucket" in latest.columns:
                _side_cols = _side_cols + ["timing_bucket"]
            # CLV: capture line data at decision time
            for _lc in ("current_line", "current_odds", "dk_start_iso"):
                if _lc in latest.columns:
                    _side_cols = _side_cols + [_lc]
            # v3.2+: KPI analytics columns
            for _lc in ("l1_sharp_agreement", "l1_pinnacle_moved", "l1_support_agreement",
                         "l1_path_behavior", "pattern_primary", "pattern_secondary", "consensus_tier",
                         "consensus_tier_prev", "l2_book_count_delta",
                         "sharp_score", "consensus_score", "retail_score", "layer_mode",
                         "market_reaction_score", "market_reaction_detail",
                         "semantic_reaction_state", "semantic_signal_class", "semantic_owning_side",
                         "semantic_decision", "semantic_source"):
                if _lc in latest.columns:
                    _side_cols = _side_cols + [_lc]
            # v3.3j: additional scoring columns for Results tab
            for _lc in ("timing_modifier", "cross_market_adj", "bets_pct", "money_pct", "open_line"):
                if _lc in latest.columns:
                    _side_cols = _side_cols + [_lc]
            _ds_side = latest[_side_cols].drop_duplicates().copy()
            _ds_game_cols = ["sport","game_id","market_display",
                "favored_side","game_confidence","net_edge","game_decision","is_locked"]
            # Note: pattern_primary, sharp_score, consensus_score, retail_score, layer_mode
            # are already carried via _ds_side from latest — do NOT duplicate here
            # or pandas merge creates _x/_y suffixes and the original names go missing.
            _ds_game = game_view[_ds_game_cols].drop_duplicates().copy()

            for _c in ["sport","game_id","market_display","side"]:
                if _c in _ds_side.columns:
                    _ds_side[_c] = _ds_side[_c].astype(str)

            for _c in ["sport","game_id","market_display"]:
                if _c in _ds_game.columns:
                    _ds_game[_c] = _ds_game[_c].astype(str)

            _ds = _ds_side.merge(
                _ds_game,
                on=["sport","game_id","market_display"],
                how="left"
            )

            _cols = [
                "sport","game_id","market_display","side",
                "favored_side","game_confidence","net_edge","total_score","game_decision",
                "market_read",       # v1.2: persisted for historical evaluation
                "timing_bucket",     # v1.2: persisted for KPI by timing
                "decision_line",     # v2.1: CLV — full line string at decision time
                "decision_line_val", # v2.1: CLV — numeric line value
                "decision_odds",     # v2.1: CLV — American odds at decision time
                "dk_start_iso",      # v2.1: CLV — game start for closing line lookup
                # v3.2+: KPI analytics columns
                "pattern_primary", "pattern_secondary", "consensus_tier",
                "consensus_tier_prev", "l2_book_count_delta", "noisy_signal_flag",
                "l1_sharp_agreement", "l1_pinnacle_moved", "l1_support_agreement",
                "sharp_score", "consensus_score", "retail_score",
                "layer_mode", "l1_path_behavior",
                "market_reaction_score", "market_reaction_detail",
                "semantic_reaction_state", "semantic_signal_class", "semantic_owning_side",
                "semantic_decision", "semantic_source",
                # v3.3j: additional scoring + lock flag
                "timing_modifier", "cross_market_adj",
                "bets_pct", "money_pct", "open_line",
                "is_locked",
                "logic_version",   # v3.3m: track engine version in freeze ledger
            ]
            # CLV: rename current_line/current_odds to decision_line/decision_odds
            if "current_line" in _ds.columns:
                _ds["decision_line"] = _ds["current_line"]
            if "current_odds" in _ds.columns:
                _ds["decision_odds"] = _ds["current_odds"]
            # Parse numeric line value for CLV math
            _ds["decision_line_val"] = _ds.get("decision_line", _ds.get("current_line", "")).apply(
                lambda x: _parse_line_val(x)
            )

            # v3.3f: noisy_signal_flag — True when signal may be cliff-edge-driven
            _tier_col = pd.to_numeric(_ds.get("consensus_tier", 0), errors="coerce").fillna(0)
            _tier_prev_col = pd.to_numeric(_ds.get("consensus_tier_prev", 0), errors="coerce").fillna(0)
            _delta_col = pd.to_numeric(_ds.get("l2_book_count_delta", 0), errors="coerce").fillna(0)
            _ds["noisy_signal_flag"] = ((_delta_col.abs() > 3) | (_tier_col != _tier_prev_col))

            for _c in _cols:
                if _c not in _ds.columns:
                    _ds[_c] = ""

            _ds = _ds[_cols]

            # -----------------------------
            # APPEND TO FREEZE LEDGER (append-only)
            # -----------------------------
            try:
                from pathlib import Path
                import pandas as _pd

                _ledger_path = Path("data/decision_freeze_ledger.csv")

                _ds["logic_version"] = LOGIC_VERSION
                _ds["_frozen_at_utc"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()

                # ── v3.3m: BET stability gate — require bet_candidate_streak >= 3
                #    (or >= 1 with score >= 74 and < 60 min to kickoff)
                #    Only blocks NEW BET entries; existing frozen BETs kept by dedup.
                try:
                    _rs_path = Path("data/row_state.csv")
                    if _rs_path.exists():
                        _rs_df = _pd.read_csv(_rs_path, dtype=str)
                        if "bet_candidate_streak" in _rs_df.columns:
                            _rs_df["_bcs"] = _pd.to_numeric(_rs_df["bet_candidate_streak"], errors="coerce").fillna(0).astype(int)
                            _rs_df["_rs_key"] = (_rs_df["sport"] + "|" + _rs_df["game_id"] + "|" +
                                _rs_df["market"].str.upper() + "|" +
                                _rs_df["side"].str.replace(r"^TEAM:", "", regex=True).str.strip().str.lower())
                            _streak_map = dict(zip(_rs_df["_rs_key"], _rs_df["_bcs"]))

                            def _get_bcs(r):
                                _s = re.sub(r"\s*[+-]?\d+\.?\d*\s*$", "", str(r.get("side",""))).strip().lower()
                                _m = str(r.get("market_display","")).upper()
                                _k = f'{r["sport"]}|{r["game_id"]}|{_m}|{_s}'
                                return _streak_map.get(_k, 0)

                            _ds["_bcs"] = _ds.apply(_get_bcs, axis=1)

                            # Build set of already-frozen keys
                            _already_frozen = set()
                            if _ledger_path.exists():
                                try:
                                    _old_check = _pd.read_csv(_ledger_path, dtype=str)
                                    _bet_old = _old_check[_old_check["game_decision"].isin(["BET","STRONG_BET"])]
                                    for _, _or in _bet_old.iterrows():
                                        _ns = re.sub(r"\s*[+-]?\d+\.?\d*\s*$", "", str(_or["side"])).strip().lower()
                                        _already_frozen.add(f'{_or["sport"]}|{_or["game_id"]}|{_or["market_display"]}|{_ns}')
                                except Exception:
                                    pass

                            _ds = _ds.drop(columns=["_bcs"], errors="ignore")
                except Exception as _stab_e:
                    print(f"[v4] stability check skipped: {repr(_stab_e)}")

                _old = None
                if _ledger_path.exists():
                    _old = _pd.read_csv(_ledger_path, dtype=str)
                    _combined = _pd.concat([_old, _ds], ignore_index=True)
                else:
                    _combined = _ds.copy()

                # Prefer STRONG_BET over BET when deduplicating (upgrade path)
                _decision_rank = {"NO BET": 0, "NO_BET": 0, "LOCKED": 0, "LEAN": 1, "BET": 2, "STRONG_BET": 3}
                _combined["_rank"] = _combined["game_decision"].map(_decision_rank).fillna(0)
                _combined = _combined.sort_values("_rank", ascending=False)
                _combined["_norm_side"] = _combined["side"].apply(
                    lambda s: re.sub(r"\s*[+-]?\d+\.?\d*\s*$", "", str(s)).strip()
                )
                _combined = _combined.drop_duplicates(
                    subset=["sport","game_id","market_display","_norm_side"],
                    keep="first"
                )
                _combined = _combined.drop(columns=["_rank","_norm_side"], errors="ignore")

                _combined.to_csv(_ledger_path, index=False)
                print("[ok] updated decision_freeze_ledger.csv")

                # ── Email notifications for new BET/STRONG_BET ──
                try:
                    from notify import notify_new_bets
                    notify_new_bets(_ds, _old)
                except Exception as _ne:
                    print(f"[notify] skipped: {repr(_ne)}")

            except Exception as _e:
                print(f"[freeze-ledger] write skipped: {repr(_e)}")

            print("[ok] freeze ledger updated")
        except Exception as _e:
            print(f"[freeze] decision snapshot write skipped: {repr(_e)}")

        # v2.3: Add intel_summary column for board.html (plain text, no HTML entities)
        def _intel_plain(gr):
            try:
                gk = f"{gr.get('sport','')}|{gr.get('game_id','')}"
                sg = side_groups.get(gk)
                text = _build_game_intel(gr, sg)
                # Convert HTML entities to plain text
                return text
            except Exception:
                return ""
        game_view["intel_summary"] = game_view.apply(_intel_plain, axis=1)

        # ── Compute side_position for sportsbook selector (home/away/over/under) ──
        def _compute_side_position(row):
            mkt = str(row.get("market_display", "")).upper()
            sk = str(row.get("side_key", "")).strip()
            game = str(row.get("game", ""))
            if mkt == "TOTAL":
                return "over" if sk.lower().startswith("over") else "under"
            # SPREAD / MONEYLINE: determine home vs away from game "AWAY @ HOME"
            if " @ " in game:
                away_part, home_part = game.split(" @ ", 1)
                sk_lower = sk.lower()
                away_lower = away_part.strip().lower()
                home_lower = home_part.strip().lower()
                if sk_lower in home_lower or home_lower in sk_lower:
                    return "home"
                if sk_lower in away_lower or away_lower in sk_lower:
                    return "away"
            return sk.lower()
        if "side_key" in game_view.columns:
            game_view["side_position"] = game_view.apply(_compute_side_position, axis=1)

        game_view.to_csv("data/dashboard.csv", index=False)

        # ── Generate book_lines.json for sportsbook selector ──
        try:
            _generate_book_lines_json()
        except Exception as _bl_e:
            print(f"[book_lines] generation skipped: {repr(_bl_e)}")

        # ── Append score history (one row per game/market/side per engine run) ──
        try:
            _hist_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            _hist_cols = ["sport_label", "game_id", "game", "market_display",
                          "favored_side",
                          "game_confidence", "net_edge", "total_score", "game_decision",
                          "expression", "expression_reason",
                          "sharp_score", "consensus_score", "retail_score",
                          "timing_modifier", "cross_market_adj",
                          "market_reaction_score",
                          "l1_path_behavior", "pattern_primary", "pattern_secondary", "consensus_tier",
                          "semantic_reaction_state", "semantic_signal_class", "semantic_owning_side",
                          "semantic_decision", "semantic_source",
                          "current_line", "open_line", "move_dir",
                          "bets_pct", "money_pct",
                          "market_read", "layer_mode"]
            _hist_present = [c for c in _hist_cols if c in game_view.columns]
            _hist_df = game_view[_hist_present].copy()
            _hist_df.insert(0, "snapshot_ts", _hist_ts)
            _hist_path = "data/score_history.csv"
            _hist_header = not os.path.exists(_hist_path) or os.path.getsize(_hist_path) == 0
            _hist_mode = "a"
            if not _hist_header:
                try:
                    _existing_cols = list(pd.read_csv(_hist_path, nrows=0).columns)
                    _new_cols = list(_hist_df.columns)
                    if _existing_cols != _new_cols:
                        _hist_header = True
                        _hist_mode = "w"
                except Exception:
                    _hist_header = True
                    _hist_mode = "w"
            _hist_df.to_csv(_hist_path, mode=_hist_mode, index=False, header=_hist_header)
            # Prune: keep last 48 hours max (~288 rows per game at 10-min intervals)
            try:
                _full = pd.read_csv(_hist_path, dtype=str)
                _full["_ts"] = pd.to_datetime(_full["snapshot_ts"], errors="coerce")
                _cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
                _full = _full[_full["_ts"] >= _cutoff].drop(columns=["_ts"])
                _full.to_csv(_hist_path, index=False)
            except Exception:
                pass
            print(f"[ok] appended {len(_hist_df)} rows to score_history.csv")
        except Exception as _she:
            print(f"[score_history] write skipped: {repr(_she)}")

        # ── Write freshness.json for UI staleness banner ──
        import json as _json
        _fresh = {"engine_ts": datetime.now(timezone.utc).isoformat()}
        try:
            _snap_df = pd.read_csv("data/snapshots.csv", usecols=["timestamp"])
            _fresh["dk_ts"] = str(pd.to_datetime(_snap_df["timestamp"]).max())
        except Exception:
            _fresh["dk_ts"] = None
        try:
            _l1_df = pd.read_csv("data/l1_sharp.csv", usecols=["timestamp"], nrows=50000)
            _fresh["l1_ts"] = str(pd.to_datetime(_l1_df["timestamp"]).max())
        except Exception:
            _fresh["l1_ts"] = None
        try:
            _l2_df = pd.read_csv("data/l2_consensus_agg.csv", usecols=["timestamp"], nrows=50000)
            _fresh["l2_ts"] = str(pd.to_datetime(_l2_df["timestamp"]).max())
        except Exception:
            _fresh["l2_ts"] = None
        with open("data/freshness.json", "w") as _ff:
            _json.dump(_fresh, _ff)

        # Build interactive header from header_cols
        _ths = "".join(f'<th>{h}</th>' for h in header_cols)
        thead = f"<tr>{_ths}</tr>"
        tbody = "\n".join(rows_html)

        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Red Fox Market Dynamics</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 12px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f7f7f7; z-index: 2; cursor: pointer; user-select: none; }}
    /* Interactive game/side rows */
    .game-row {{ cursor: pointer; }}
    .game-row:hover {{ background: #f0f8ff; }}
    .side-row {{ font-size: 11px; color: #555; }}
    .intel-summary {{ font-size: 10px; color: #666; font-weight: normal; line-height: 1.3; }}
    /* Row tint by data quality — full data = clean, missing = muted */
    .data-l123 {{ background: #f6fef6; }}
    .data-l13 td, .data-l23 td {{ background: #fffdf0; }}
    .data-l3_only td {{ background: #f5f5f5; color: #888; }}
    .pill {{ padding: 2px 8px; border-radius: 10px; font-weight: bold; display: inline-block; }}
    .pill.edge {{ background: #e8e8e8; }}
    /* v2.0 Layer badges */
    .layer-badge {{ display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 10px; font-weight: bold; color: #fff; }}
    .layer-L123 {{ background: #2ea043; }}
    .layer-L13 {{ background: #d4a017; }}
    .layer-L23 {{ background: #d4a017; }}
    .layer-L3_ONLY {{ background: #999; }}
    /* Pattern badges */
    .pattern-badge {{ display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px; font-weight: bold; }}
    .pattern-secondary {{ font-size: 9px; color: #888; font-weight: normal; }}
    .pattern-A {{ background: #2ea043; color: #fff; }}
    .pattern-B {{ background: #f0c040; color: #333; }}
    .pattern-C {{ background: #e67e22; color: #fff; }}
    .pattern-D {{ background: #3498db; color: #fff; }}
    .pattern-E {{ background: #e74c3c; color: #fff; }}
    .pattern-F {{ background: #8b0000; color: #fff; }}
    .pattern-G {{ background: #9b59b6; color: #fff; }}
    .pattern-N {{ background: #ddd; color: #666; }}
    /* Stale price pulse */
    .stale-alert {{ display: inline-block; padding: 2px 8px; border-radius: 4px; background: #3498db; color: #fff; font-size: 10px; font-weight: bold; animation: pulse 1.5s infinite; }}
    @keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.6; }} }}
    /* Dispersion indicator */
    .disp-tight {{ color: #2ea043; font-weight: bold; }}
    .disp-normal {{ color: #666; }}
    .disp-wide {{ color: #e67e22; font-weight: bold; }}
    .disp-vwide {{ color: #e74c3c; font-weight: bold; }}
    /* B2B badge */
    .b2b-badge {{ display: inline-block; padding: 1px 4px; border-radius: 3px; background: #e74c3c; color: #fff; font-size: 9px; font-weight: bold; margin-left: 3px; }}
    /* Injury count */
    .inj-badge {{ display: inline-block; padding: 1px 4px; border-radius: 3px; background: #ff6b6b; color: #fff; font-size: 9px; margin-left: 3px; }}
    .wx-badge {{ display: inline-block; padding: 1px 4px; border-radius: 3px; background: #3498db; color: #fff; font-size: 9px; margin-left: 3px; cursor: help; }}
    .wx-dome {{ display: inline-block; font-size: 11px; margin-left: 3px; cursor: help; opacity: 0.5; }}
    .wx-clear {{ display: inline-block; font-size: 11px; margin-left: 3px; cursor: help; opacity: 0.6; }}
    /* Sport context badges */
    .sp-badge {{ display: inline-block; padding: 1px 4px; border-radius: 3px; color: #fff; font-size: 9px; margin-left: 3px; cursor: help; }}
    .sp-ace {{ background: #2ea043; }}
    .sp-avg {{ background: #888; }}
    .sp-weak {{ background: #e74c3c; }}
    .park-badge {{ display: inline-block; padding: 1px 4px; border-radius: 3px; font-size: 9px; margin-left: 3px; cursor: help; }}
    .park-hit {{ background: #d4a017; color: #fff; }}
    .park-pitch {{ background: #3498db; color: #fff; }}
    .goalie-badge {{ display: inline-block; padding: 1px 4px; border-radius: 3px; font-size: 9px; margin-left: 3px; cursor: help; }}
    .g-confirmed {{ background: #2ea043; color: #fff; }}
    .g-probable {{ background: #d4a017; color: #fff; }}
    .g-unknown {{ background: #888; color: #fff; }}
    .rank-badge {{ display: inline-block; padding: 1px 4px; border-radius: 3px; background: #7c3aed; color: #fff; font-size: 9px; margin-left: 3px; }}
    .rank-opp {{ background: #555; }}
    /* Sharp meter bar */
    .sharp-meter {{ display: inline-block; width: 50px; height: 8px; background: #eee; border-radius: 4px; overflow: hidden; vertical-align: middle; margin-left: 4px; }}
    .sharp-meter-fill {{ height: 100%; border-radius: 4px; }}
    .sharp-1 {{ width: 17%; background: #999; }}
    .sharp-2 {{ width: 33%; background: #d4a017; }}
    .sharp-3 {{ width: 50%; background: #d4a017; }}
    .sharp-4 {{ width: 67%; background: #8fbc3a; }}
    .sharp-5 {{ width: 83%; background: #2ea043; }}
    .sharp-6 {{ width: 100%; background: #2ea043; }}
    /* Sharp Certified badges */
    .sharp-cert {{ display: inline-block; padding: 1px 5px; border-radius: 4px; font-size: 9px; font-weight: bold; margin-left: 4px; white-space: nowrap; }}
    .sharp-cert-full {{ background: #2ea043; color: #fff; animation: sharpPulse 2s ease-in-out infinite; }}
    .sharp-cert-half {{ background: #d4a017; color: #fff; }}
    @keyframes sharpPulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.7; }} }}
    /* Edge type label */
    .edge-label {{ font-size: 9px; color: #666; }}
    /* DK vs Market positive = green (DK has value), negative = red */
    .dvm-pos {{ color: #2ea043; font-weight: bold; }}
    .dvm-neg {{ color: #e74c3c; font-weight: bold; }}
    .dvm-neutral {{ color: #999; }}
    /* Legend */
    .legend {{ font-size: 11px; color: #666; margin-bottom: 10px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 4px; }}
  </style>
  <script>
    // Toggle side-row visibility when game row is clicked
    function toggleGroup(gk) {{
      var rows = document.querySelectorAll('tr[data-parent="' + gk + '"]');
      for (var i = 0; i < rows.length; i++) {{
        rows[i].style.display = rows[i].style.display === 'none' ? '' : 'none';
      }}
    }}
  </script>
</head>
<body>
  <h2 style="margin: 0 0 6px 0;">Red Fox Market Dynamics</h2>
  <div class="legend">
    <span style="color:#999;font-size:10px;">DATA:</span>
    <span class="legend-item"><span class="layer-badge layer-L123">FULL</span></span>
    <span class="legend-item"><span class="layer-badge layer-L13">PARTIAL</span></span>
    <span class="legend-item"><span class="layer-badge layer-L3_ONLY">LIMITED</span></span>
    <span style="color:#ccc;">|</span>
    <span style="color:#999;font-size:10px;">PATTERN:</span>
    <span class="legend-item"><span class="pattern-badge pattern-A">A</span> Sharp vs Public</span>
    <span class="legend-item"><span class="pattern-badge pattern-B">B</span> All Aligned</span>
    <span class="legend-item"><span class="pattern-badge pattern-C">C</span> Public Only</span>
    <span class="legend-item"><span class="pattern-badge pattern-D">D</span> Stale Price</span>
    <span class="legend-item"><span class="pattern-badge pattern-E">E</span> Consensus Rejects</span>
    <span class="legend-item"><span class="pattern-badge pattern-G">G</span> RLM</span>
    <span style="color:#ccc;">|</span>
    <span style="color:#999;font-size:10px;">Click a game row to expand side details</span>
  </div>
  <table>
    <thead>{thead}</thead>
    <tbody>
{tbody}
    </tbody>
  </table>
</body>
</html>"""

        with open(REPORT_HTML, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"[ok] wrote dashboard html: {REPORT_HTML}")
# --- METRICS: post-dashboard FINAL (v1.1) ---
        try:
            import pandas as _pd
            _dash = _pd.read_csv('data/dashboard.csv')
            if not _dash.empty:
                pass  # metrics disabled here; per-market feed is the SSOT
        except Exception as e:
            print('[metrics] post-dashboard failed:', repr(e))


    except Exception as e:
        print(f"[dash] wide html write failed: {repr(e)}")
        # --- METRICS: post-dashboard (v1.1) ---
        try:
            import pandas as _pd
            # Prefer in-memory rows if available; fallback to dashboard.csv
            _latest2 = _pd.DataFrame(_rows) if ('_rows' in locals()) else None
            if _latest2 is not None and not _latest2.empty:
                pass  # metrics disabled here; per-market feed is the SSOT
            else:
                _dash = _pd.read_csv('data/dashboard.csv')
                if not _dash.empty:
                    pass  # metrics disabled here; per-market feed is the SSOT
        except Exception as e:
            print('[metrics] wiring failed:', repr(e))
        try:
            import traceback
            print(traceback.format_exc())
        except Exception:
            pass

    # do NOT return early; let the rest of report logic run
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




def cmd_snapshot(args):
    # No hard skips here.
    # If a sport has no games, dk_headless/get_splits should return 0 records,
    # and we will print "[snapshot] no games available for <sport>" below.


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
    import re as _re_open
    for r in rows:
        sport = str(r.get("sport","") or "")
        game_id = str(r.get("game_id","") or "")
        market = str(r.get("market","") or "")
        side = str(r.get("side","") or "")
        current_line = str(r.get("current_line","") or r.get("current","") or "").strip()

        # Detect market type from side string and normalize for stable key
        # DK uses market="splits" for all types, so we infer from side content
        _is_total = bool(_re_open.match(r"^(Over|Under)\b", side))
        _has_spread = bool(_re_open.search(r"\s[+-]?\d+(?:\.\d+)?\s*$", side)) and not _is_total
        if _is_total:
            side_norm = _re_open.match(r"^(Over|Under)", side).group(1)
            mkt_type = "total"
        elif _has_spread:
            side_norm = _re_open.sub(r"\s[+-]?\d+(?:\.\d+)?\s*$", "", side).strip()
            mkt_type = "spread"
        else:
            side_norm = side
            mkt_type = "ml"

        k = (sport, game_id, market, mkt_type, side_norm)

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


    # build_dashboard() moved to report_live — called once after all snapshots complete
    # resolve_results_for_baseline() moved to report_maintenance



def cmd_report_live(_args):
    build_dashboard()

    # -----------------------------
    # UI/CSV: ensure net_edge exists in data/dashboard.csv (instrumentation-only)
    # NOTE: must run after build_dashboard() writes dashboard.csv
    # -----------------------------
    try:
        import pandas as _pd
        _dpath = "data/dashboard.csv"
        _d = _pd.read_csv(_dpath, keep_default_na=False)

        if "net_edge" not in _d.columns:
            # Prefer any per-market net edge columns already present
            _edge_cols = [c for c in _d.columns if c.endswith("_net_edge")]
            if _edge_cols:
                _tmp = _pd.DataFrame({c: _pd.to_numeric(_d[c], errors="coerce") for c in _edge_cols})
                _d["net_edge"] = _tmp.max(axis=1).fillna(0.0).round(1)
            else:
                _d["net_edge"] = 0.0

            _d.to_csv(_dpath, index=False)
            print("[ok] added net_edge to data/dashboard.csv")
    except Exception as _e:
        print(f"[dash] net_edge post-fix failed: {repr(_e)}")

    print("[report] done")


def cmd_report_maintenance(_args):
    # ESPN finals (results) update is best-effort; never block maintenance
    try:
        update_snapshots_with_espn_finals()
        update_final_scores_history()
    except KeyboardInterrupt:
        print("[espn finals] skipped (KeyboardInterrupt)")
    except Exception as e:
        print(f"[espn finals] skipped due to error: {repr(e)}")
    resolve_results_for_baseline()
    build_color_baseline_summary()


def cmd_report(_args):
    cmd_report_live(_args)
    cmd_report_maintenance(_args)



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
    # B3: Nearest snapshot ÃƒÂƒÃ‚Â¢ÃƒÂ¢Ã‚Â€Ã‚Â°ÃƒÂ‚Ã‚Â¤ signal time
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
                    r.get("logic_version") == LOGIC_VERSION
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
            "logic_version": LOGIC_VERSION,
            "sport": row.get("sport"),
            "game_id": row.get("game_id"),
            "game": row.get("game"),
            "market": row.get("market"),
            "side": row.get("side"),
            "color": row.get("color"),
            "model_score": "" if model_score is None else model_score,
            "model_score_bucket": model_score_bucket,
        })




def update_final_scores_history():
    import pandas as pd
    from pathlib import Path
    import datetime

    data_dir = Path("data")
    snap_path = data_dir / "snapshots.csv"
    hist_path = data_dir / "final_scores_history.csv"
    hist_path = data_dir / "final_scores_history.csv"

    if not snap_path.exists():
        return

    s = pd.read_csv(snap_path, dtype=str)

    required = ["game_id","side","final_score_for","final_score_against"]
    for col in required:
        if col not in s.columns:
            return

    s = s[
        s["final_score_for"].notna() &
        (s["final_score_for"].astype(str).str.strip() != "")
    ].copy()

    def is_total(x):
        return str(x).startswith("Over") or str(x).startswith("Under")

    def is_spread(x):
        return " +" in str(x) or " -" in str(x)

    s = s[
        ~s["side"].apply(is_total) &
        ~s["side"].apply(is_spread)
    ]

    if s.empty:
        return

    s["team_norm"] = s["side"].apply(lambda x: _normalize_team_name(x))
    s["final_score_for"] = s["final_score_for"].astype(float)

    games = (
        s
.sort_values("game_id")
.drop_duplicates(subset=["game_id","team_norm"], keep="last")

    )

    # Build deterministic away/home mapping from game column ("AWAY @ HOME")
    _game_team_map = {}  # game_id -> (away_norm, home_norm)
    if "game" in s.columns:
        for _, _r in s[["game_id", "game"]].drop_duplicates(subset=["game_id"]).iterrows():
            _g = str(_r.get("game", "")).strip()
            if " @ " in _g:
                _parts = _g.split(" @ ", 1)
                _game_team_map[_r["game_id"]] = (
                    _normalize_team_name(_parts[0].strip()),
                    _normalize_team_name(_parts[1].strip()),
                )

    rows = []
    for game_id, g in games.groupby("game_id"):
        if len(g) < 2:
            continue

        # Deterministic ordering: team1=away, team2=home
        team_map = _game_team_map.get(game_id)
        if team_map:
            away_norm, home_norm = team_map
            score_by_team = dict(zip(g["team_norm"], g["final_score_for"]))
            # Find scores by matching normalized team names
            t1_score = score_by_team.get(away_norm)
            t2_score = score_by_team.get(home_norm)
            if t1_score is None or t2_score is None:
                # Fuzzy fallback: match by substring
                for tn, sc in score_by_team.items():
                    if t1_score is None and (tn in away_norm or away_norm in tn):
                        t1_score = sc
                    elif t2_score is None and (tn in home_norm or home_norm in tn):
                        t2_score = sc
            if t1_score is not None and t2_score is not None:
                rows.append({
                    "game_id": game_id,
                    "team1": away_norm,
                    "team1_score": t1_score,
                    "team2": home_norm,
                    "team2_score": t2_score,
                    "resolved_at_utc": datetime.datetime.now(__import__('datetime').timezone.utc).isoformat()
                })
                continue

        # Fallback: use iloc order (legacy behavior)
        rows.append({
            "game_id": game_id,
            "team1": g.iloc[0]["team_norm"],
            "team1_score": g.iloc[0]["final_score_for"],
            "team2": g.iloc[1]["team_norm"],
            "team2_score": g.iloc[1]["final_score_for"],
            "resolved_at_utc": datetime.datetime.now(__import__('datetime').timezone.utc).isoformat()
        })

    if not rows:
        return

    df_new = pd.DataFrame(rows)

    if hist_path.exists():
        df_old = pd.read_csv(hist_path, dtype=str)
        df = pd.concat([df_old, df_new])
        df = df.drop_duplicates(subset=["game_id"], keep="last")
    else:
        df = df_new


    # ---- Enforce freeze epoch alignment ----
    try:
        freeze = pd.read_csv("data/decision_freeze_ledger.csv", dtype=str)
        if "game_id" in freeze.columns:
            df = df[df["game_id"].isin(freeze["game_id"])]
    except Exception:
        pass


    df.to_csv(hist_path, index=False)
    
    if len(df) > 0:
        print(f"[final_history] upserted {len(df)} freeze-aligned games")






def capture_closing_lines():
    """For each frozen decision, find the last DK snapshot before game start.
    Writes data/clv_closing_lines.csv for CLV calculation."""
    import pandas as pd
    from pathlib import Path

    freeze_path = Path("data/decision_freeze_ledger.csv")
    snap_path = Path("data/snapshots.csv")
    out_path = Path("data/clv_closing_lines.csv")

    if not freeze_path.exists() or not snap_path.exists():
        print("[clv] missing freeze ledger or snapshots")
        return

    freeze = pd.read_csv(freeze_path, dtype=str)
    snaps = pd.read_csv(snap_path, dtype=str)

    # Only process rows that have game start time and line data
    if "dk_start_iso" not in freeze.columns:
        print("[clv] freeze ledger missing dk_start_iso — skipping (pre-v2.1 data)")
        return

    snaps["_ts"] = pd.to_datetime(snaps["timestamp"], utc=True, errors="coerce")

    closing_rows = []
    for _, row in freeze.iterrows():
        game_id = str(row.get("game_id", "")).strip()
        side = str(row.get("side", "")).strip()
        start_iso = str(row.get("dk_start_iso", "")).strip()

        if not start_iso or not game_id:
            continue

        start_dt = pd.to_datetime(start_iso, utc=True, errors="coerce")
        if pd.isna(start_dt):
            continue

        # Last snapshot before game start for this game+side
        mask = (
            (snaps["game_id"].astype(str) == game_id) &
            (snaps["side"].astype(str) == side) &
            (snaps["_ts"] < start_dt) &
            (snaps["_ts"].notna())
        )
        candidates = snaps.loc[mask].sort_values("_ts", ascending=False)

        if candidates.empty:
            continue

        last = candidates.iloc[0]
        cl = str(last.get("current_line", ""))
        closing_rows.append({
            "sport": row.get("sport", ""),
            "game_id": game_id,
            "market_display": row.get("market_display", ""),
            "side": side,
            "closing_line": cl,
            "closing_line_val": _parse_line_val(cl),
            "closing_odds": last.get("current_odds", ""),
            "snapshot_ts": str(last.get("timestamp", "")),
        })

    if closing_rows:
        pd.DataFrame(closing_rows).to_csv(out_path, index=False)
        print(f"[clv] wrote {len(closing_rows)} closing lines to clv_closing_lines.csv")
    else:
        print("[clv] no closing lines found (games may not have started yet)")


def resolve_results_for_baseline():
    import pandas as pd
    import re
    from pathlib import Path
    from datetime import datetime

    data_dir = Path("data")
    snap_path = data_dir / "snapshots.csv"
    hist_path = data_dir / "final_scores_history.csv"
    out_path  = data_dir / "results_resolved.csv"

    if not snap_path.exists() or not hist_path.exists():
        print("[outcomes] missing snapshots or final history")
        return

    snaps = pd.read_csv(snap_path, dtype=str)
    history = pd.read_csv(hist_path, dtype=str)

    history["team1_score"] = pd.to_numeric(history["team1_score"], errors="coerce")
    history["team2_score"] = pd.to_numeric(history["team2_score"], errors="coerce")

    snaps["timestamp"] = pd.to_datetime(snaps["timestamp"], errors="coerce")
    snaps = snaps.sort_values("timestamp")
    snaps = snaps.drop_duplicates(subset=["sport","game_id","side"], keep="last")

    # Build game-level team identity from snapshots (canonical source)
    def _split_teams(game_str):
        g = str(game_str).strip()
        if " @ " in g:
            a, h = g.split(" @ ", 1)
            def _n(x):
                return str(x).strip().lower().replace(" state", " st").replace(".", "")
            return _n(a.strip()), _n(h.strip())
        return None, None

    game_teams = snaps[["game_id", "game"]].drop_duplicates(subset=["game_id"]).copy()
    game_teams[["team1", "team2"]] = game_teams["game"].apply(
        lambda g: pd.Series(_split_teams(g))
    )
    game_teams = game_teams[["game_id", "team1", "team2"]]

    # Merge scores from history — must align team1/team2 ordering
    # History has its own team1/team2 order that may differ from _split_teams
    hist_cols = history[["game_id", "team1", "team2", "team1_score", "team2_score"]].copy()
    hist_cols = hist_cols.rename(columns={"team1": "hist_t1", "team2": "hist_t2"})
    merged = snaps.merge(hist_cols, on="game_id", how="left")

    # Merge identity from snapshots (away=team1, home=team2)
    merged = merged.merge(game_teams, on="game_id", how="left")

    # Align scores: match history teams to snapshot teams by name
    def _align_scores(row):
        ht1 = str(row.get("hist_t1", "")).strip().lower()
        ht2 = str(row.get("hist_t2", "")).strip().lower()
        st1 = str(row.get("team1", "")).strip().lower()  # away from _split_teams
        st2 = str(row.get("team2", "")).strip().lower()  # home from _split_teams
        s1 = row.get("team1_score")
        s2 = row.get("team2_score")
        if pd.isna(s1) or pd.isna(s2):
            return s1, s2
        # If history team1 matches snapshot team1 → same order
        if ht1 == st1 or st1 in ht1 or ht1 in st1:
            return s1, s2
        # If history team1 matches snapshot team2 → swap
        if ht1 == st2 or st2 in ht1 or ht1 in st2:
            return s2, s1
        return s1, s2  # fallback: keep as-is

    _aligned = merged.apply(_align_scores, axis=1, result_type="expand")
    merged["team1_score"] = _aligned[0]
    merged["team2_score"] = _aligned[1]

    results = []
    market_types = []


    for _, row in merged.iterrows():
        s1 = row.get("team1_score")
        s2 = row.get("team2_score")
        side_raw = str(row["side"]).strip()

        # classify market
        if re.match(r"^(Over|Under)\s+(\d+(\.\d+)?)$", side_raw, re.IGNORECASE):
            market = "TOTAL"
        elif re.search(r"[+-]\d+(\.\d+)?$", side_raw):
            market = "SPREAD"
        else:
            market = "MONEYLINE"

        market_types.append(market)

        # if no final scores yet
        if pd.isna(s1) or pd.isna(s2):
            results.append(None)
            continue

        # TOTAL grading
        if market == "TOTAL":
            m = re.match(r"^(Over|Under)\s+(\d+(\.\d+)?)$", side_raw, re.IGNORECASE)
            direction = m.group(1).lower()
            line = float(m.group(2))
            total = s1 + s2

            if total == line:
                results.append("PUSH")
            elif direction == "over":
                results.append("WIN" if total > line else "LOSS")
            else:
                results.append("WIN" if total < line else "LOSS")

        # SPREAD grading
        elif market == "SPREAD":
            parts = side_raw.rsplit(" ", 1)
            team_pick = parts[0].strip()
            spread = float(parts[1])

            t1 = row.get("team1")
            t2 = row.get("team2")

            def _norm(x):
                return (
                    str(x)
                    .lower()
                    .replace(" state", " st")
                    .replace(".", "")
                    .strip()
                )

            norm_pick = _norm(team_pick)
            norm_t1   = _norm(t1)
            norm_t2   = _norm(t2)

            if norm_pick == norm_t1:
                adj = s1 + spread
                opp = s2
            elif norm_pick == norm_t2:
                adj = s2 + spread
                opp = s1
            else:
                results.append(None)
                continue

            if adj == opp:
                results.append("PUSH")
            elif adj > opp:
                results.append("WIN")
            else:
                results.append("LOSS")

        # MONEYLINE grading
        else:
            team_pick = side_raw
            t1 = row.get("team1")
            t2 = row.get("team2")

            def _norm(x):
                return (
                    str(x)
                    .lower()
                    .replace(" state", " st")
                    .replace(".", "")
                    .strip()
                )

            norm_pick = _norm(team_pick)
            norm_t1   = _norm(t1)
            norm_t2   = _norm(t2)

            if norm_pick == norm_t1:
                if s1 == s2:
                    results.append("PUSH")
                elif s1 > s2:
                    results.append("WIN")
                else:
                    results.append("LOSS")
            elif norm_pick == norm_t2:
                if s2 == s1:
                    results.append("PUSH")
                elif s2 > s1:
                    results.append("WIN")
                else:
                    results.append("LOSS")
            else:
                results.append(None)
    merged["market_display"] = market_types
    merged["result"] = results
    merged["resolved_at_utc"] = datetime.now(__import__('datetime').timezone.utc).isoformat()

    cols = [
        "sport",
        "game_id",
        "game",
        "market_display",
        "side",
        "current_line",
        "team1",
        "team2",
        "team1_score",
        "team2_score",
        "result",
        "resolved_at_utc",
        "favored_side",
        "game_confidence",
        "net_edge",
        "game_decision",
        "market_read",
        "timing_bucket",
    ]


    # -------------------------------------------------
    # Enrich results_resolved with decision snapshot
    # (post-scoring, post-aggregation, enrichment only)
    # -------------------------------------------------
    try:
        decision_snapshot = pd.read_csv("data/decision_freeze_ledger.csv")

        # normalize key types
        for _df in (decision_snapshot, merged):
            for _c in ("sport","game_id","market_display","side"):
                if _c in _df.columns:
                    _df[_c] = _df[_c].astype(str)

        _freeze_cols = ["sport","game_id","market_display","side",
             "favored_side","game_confidence","net_edge","total_score","game_decision"]
        # v1.2: include market_read and timing_bucket if available in freeze ledger
        for _extra in ("market_read", "timing_bucket"):
            if _extra in decision_snapshot.columns:
                _freeze_cols.append(_extra)
        # v2.1: include CLV decision line data if available
        for _extra in ("decision_line", "decision_line_val", "decision_odds", "dk_start_iso"):
            if _extra in decision_snapshot.columns:
                _freeze_cols.append(_extra)
        # v3.3i: frozen timestamp for trend dating
        for _extra in ("_frozen_at_utc",):
            if _extra in decision_snapshot.columns:
                _freeze_cols.append(_extra)
        # v3.2: KPI analytics columns
        for _extra in ("pattern_primary", "pattern_secondary", "consensus_tier",
                        "consensus_tier_prev", "l2_book_count_delta", "noisy_signal_flag",
                        "l1_sharp_agreement", "l1_pinnacle_moved",
                        "l1_support_agreement", "sharp_score", "consensus_score",
                        "retail_score", "layer_mode", "l1_path_behavior",
                        "market_reaction_score", "market_reaction_detail",
                        "semantic_reaction_state", "semantic_signal_class", "semantic_owning_side",
                        "semantic_decision", "semantic_source"):
            if _extra in decision_snapshot.columns:
                _freeze_cols.append(_extra)
        # v3.3j: additional scoring columns + lock flag
        for _extra in ("timing_modifier", "cross_market_adj", "bets_pct", "money_pct", "open_line", "is_locked"):
            if _extra in decision_snapshot.columns:
                _freeze_cols.append(_extra)
        # v3.3m: logic_version for epoch filtering
        if "logic_version" in decision_snapshot.columns:
            _freeze_cols.append("logic_version")
        decision_snapshot = decision_snapshot[_freeze_cols].drop_duplicates()

        # Prevent pandas _x/_y overwrite of frozen fields
        for _c in ("favored_side","game_confidence","net_edge","total_score","game_decision","market_read","timing_bucket",
                    "_frozen_at_utc",
                    "decision_line","decision_line_val","decision_odds","dk_start_iso",
                    "pattern_primary","pattern_secondary","consensus_tier",
                    "consensus_tier_prev","l2_book_count_delta","noisy_signal_flag",
                    "l1_sharp_agreement","l1_pinnacle_moved",
                    "l1_support_agreement","sharp_score","consensus_score",
                    "retail_score","layer_mode","l1_path_behavior",
                    "market_reaction_score","market_reaction_detail",
                    "semantic_reaction_state","semantic_signal_class","semantic_owning_side",
                    "semantic_decision","semantic_source",
                    "timing_modifier","cross_market_adj","bets_pct","money_pct","open_line","is_locked",
                    "logic_version"):
            if _c in merged.columns:
                merged = merged.drop(columns=[_c])

        merged = merged.merge(
            decision_snapshot,
            on=["sport","game_id","market_display","side"],
            how="left"
        )

    except Exception as e:
        print(f"[outcomes] decision snapshot enrichment skipped: {e}")

    # -------------------------------------------------
    # CLV: capture closing lines and compute CLV
    # -------------------------------------------------
    try:
        capture_closing_lines()
        _clv_path = data_dir / "clv_closing_lines.csv"
        if _clv_path.exists():
            _clv = pd.read_csv(_clv_path, dtype=str)
            for _c in ("sport","game_id","market_display","side"):
                if _c in _clv.columns:
                    _clv[_c] = _clv[_c].astype(str)

            # Drop CLV cols from merged if they already exist (prevent _x/_y)
            for _c in ("closing_line","closing_line_val","closing_odds","clv"):
                if _c in merged.columns:
                    merged = merged.drop(columns=[_c])

            _clv_join = _clv[["sport","game_id","market_display","side",
                              "closing_line","closing_line_val","closing_odds"]].drop_duplicates()
            merged = merged.merge(_clv_join, on=["sport","game_id","market_display","side"], how="left")

            # Compute CLV: positive = got better number than closing
            def _calc_clv(row):
                try:
                    d_val = float(row.get("decision_line_val", ""))
                    c_val = float(row.get("closing_line_val", ""))
                except (ValueError, TypeError):
                    return ""
                mkt = str(row.get("market_display", "")).upper()
                if mkt == "MONEYLINE":
                    # ML: compare implied probabilities
                    def _imp(odds):
                        o = float(odds)
                        return abs(o) / (abs(o) + 100) if o < 0 else 100 / (o + 100)
                    try:
                        d_odds = float(row.get("decision_odds", ""))
                        c_odds = float(row.get("closing_odds", ""))
                        return round(_imp(c_odds) - _imp(d_odds), 4)
                    except (ValueError, TypeError):
                        return ""
                elif mkt == "SPREAD":
                    # SPREAD: positive CLV = closing line more negative (more points)
                    # e.g. decision -3.5, closing -5 → CLV = -5 - (-3.5) = -1.5
                    # Wait — that's negative. But getting -3.5 when it closes at -5 IS good.
                    # Convention: CLV = abs(closing) - abs(decision) for favorites
                    # Simpler: CLV = decision_val - closing_val (for spread, more negative = worse for bettor)
                    # If you bet -3.5 and it closes -5: you got 1.5 pts of value
                    return round(d_val - c_val, 2)
                elif mkt == "TOTAL":
                    # TOTAL: depends on Over vs Under
                    side = str(row.get("side", "")).upper()
                    if "OVER" in side:
                        # Over bettor wants lower number: CLV = closing - decision
                        return round(c_val - d_val, 2)
                    else:
                        # Under bettor wants higher number: CLV = decision - closing
                        return round(d_val - c_val, 2)
                return ""

            merged["clv"] = merged.apply(_calc_clv, axis=1)
            _clv_count = merged["clv"].apply(lambda x: x != "" and pd.notna(x)).sum()
            print(f"[clv] computed CLV for {_clv_count} rows")
    except Exception as e:
        print(f"[clv] CLV computation skipped: {e}")

    # v3.3m: CLV disabled — decision_line and closing_line both from DK (same book).
    # Re-enable when Pinnacle closing lines are wired in (backlog v3.4).
    merged["clv"] = ""
    merged["closing_line"] = ""
    merged["closing_line_val"] = ""

    # Restrict outcomes to rows with freeze coverage (epoch start)
    if "game_decision" in merged.columns:
        merged = merged[merged["game_decision"].notna()].copy()

    # v3.3m: Only keep favored side — both sides exist in snapshots but
    # we only grade the picked/favored side for KPI accuracy
    if "side" in merged.columns and "favored_side" in merged.columns:
        _before = len(merged)
        merged = merged[merged["side"].str.strip() == merged["favored_side"].str.strip()].copy()
        print(f"[outcomes] filtered to favored side: {_before} -> {len(merged)} rows")

    # v4 epoch gate: only keep rows frozen on/after the v4 launch date
    _v4_epoch = pd.Timestamp("2026-03-20T00:00:00Z")
    if "_frozen_at_utc" in merged.columns:
        _frozen_ts = pd.to_datetime(merged["_frozen_at_utc"], errors="coerce", utc=True)
        merged = merged[_frozen_ts.isna() | (_frozen_ts >= _v4_epoch)].copy()

    # v3.3i: frozen timestamp + KPI analytics columns in output
    _extra_cols = ["_frozen_at_utc",
                   "total_score", "sharp_score", "consensus_score", "retail_score",
                   "timing_modifier", "cross_market_adj",
                   "bets_pct", "money_pct", "open_line", "is_locked",
                   "pattern_primary", "pattern_secondary", "layer_mode",
                   "l1_sharp_agreement", "l1_pinnacle_moved", "l1_support_agreement",
                   "l1_path_behavior", "market_reaction_score", "market_reaction_detail",
                   "semantic_reaction_state", "semantic_signal_class", "semantic_owning_side",
                   "semantic_decision", "semantic_source",
                   "consensus_tier", "consensus_tier_prev", "l2_book_count_delta",
                   "noisy_signal_flag",
                   "decision_line", "decision_line_val", "decision_odds",
                   "closing_line", "closing_line_val", "clv",
                   "logic_version"]
    for _c in _extra_cols:
        if _c not in cols:
            cols.append(_c)

    # Ensure all output columns exist (market_read/timing_bucket may be missing for pre-v1.2 data)
    for _c in cols:
        if _c not in merged.columns:
            merged[_c] = ""
    merged = merged[cols]

    merged.to_csv(out_path, index=False)

    print(f"[outcomes] rebuilt results_resolved.csv with {len(merged)} rows")

def cmd_odds_snapshot(args):
    """Fetch L1 (sharp) + L2 (consensus) via The-Odds-API for a sport."""
    sport = args.sport
    print(f"[odds_snapshot] Fetching L1+L2 for {sport}...")
    try:
        from l2_scraper import scrape_l1_and_l2
        result = scrape_l1_and_l2(sport)

        l1 = result.get("l1", {})
        l2 = result.get("l2", {})
        cached = result.get("from_cache", False)
        remaining = result.get("remaining_requests", "?")

        l1_source = l1.get("source", "unknown")
        l1_books = l1.get("books_found", [])
        print(f"[odds_snapshot] L1 ({l1_source}): {l1.get('games_found', 0)} games, {l1.get('rows_written', 0)} rows | books: {', '.join(l1_books) if l1_books else 'none'}")
        if l1.get("error"):
            print(f"[odds_snapshot] L1 note: {l1['error']}")
        print(f"[odds_snapshot] L2: {l2.get('games_found', 0)} games, {l2.get('rows_written', 0)} raw rows, {l2.get('agg_rows', 0)} agg rows")
        print(f"[odds_snapshot] L2 books: {', '.join(l2.get('books_seen', []))}")
        print(f"[odds_snapshot] From cache: {cached} | L2 API remaining: {remaining}")
    except Exception as e:
        print(f"[odds_snapshot] FAILED: {repr(e)}")
        import traceback
        traceback.print_exc()


def cmd_odds_snapshot_all(args):
    """Fetch L1+L2 for ALL active sports via The-Odds-API."""
    from engine_config import get_active_sports, API_BUDGET_RESERVE

    active = get_active_sports()
    print(f"[odds_snapshot_all] Active sports: {', '.join(active)}")

    # Budget guard (The-Odds-API)
    l2_skip = False
    try:
        from odds_api import get_quota
        quota = get_quota()
        if not quota.get("error"):
            remaining = int(quota.get("remaining", 999))
            print(f"[odds_snapshot_all] API budget: {remaining} requests remaining")
            if remaining <= API_BUDGET_RESERVE:
                print(f"[odds_snapshot_all] BUDGET GUARD: only {remaining} left. Skipping ALL pulls this run.")
                l2_skip = True
        else:
            print(f"[odds_snapshot_all] Budget check returned error — skipping to protect budget")
            l2_skip = True
    except Exception as _e:
        print(f"[odds_snapshot_all] Budget check failed ({_e}) — skipping to protect budget")
        l2_skip = True

    if l2_skip:
        print("[odds_snapshot_all] Budget guard active — no API calls this run.")
        return

    # Smart sport filter: skip sports with no upcoming games (saves API budget)
    def _has_upcoming_games(sport_key):
        """Check L1 sharp CSV for games within next 36h for this sport."""
        try:
            l1_path = os.path.join("data", "l1_sharp.csv")
            if not os.path.exists(l1_path):
                return True  # No data yet — pull to seed
            cutoff = datetime.now(timezone.utc) - timedelta(hours=4)
            future = datetime.now(timezone.utc) + timedelta(hours=36)
            with open(l1_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("sport") != sport_key:
                        continue
                    ct = row.get("commence_time", "")
                    if not ct:
                        continue
                    try:
                        game_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                        if cutoff <= game_dt <= future:
                            return True
                    except (ValueError, TypeError):
                        continue
            return False
        except Exception:
            return True  # On error, pull to be safe

    filtered = []
    for sport in active:
        if sport not in SPORT_CONFIG and sport != "ufc":
            print(f"[odds_snapshot_all] Skipping {sport} (not in SPORT_CONFIG)")
            continue
        if not _has_upcoming_games(sport):
            print(f"[odds_snapshot_all] Skipping {sport} (no upcoming games within 36h)")
            continue
        filtered.append(sport)

    if not filtered:
        print("[odds_snapshot_all] No sports with upcoming games — nothing to pull.")
        return

    print(f"[odds_snapshot_all] Pulling: {', '.join(filtered)}")

    total_l1 = 0
    total_l2 = 0
    for sport in filtered:
        print(f"\n[odds_snapshot_all] --- {sport.upper()} ---")
        try:
            from l2_scraper import scrape_l1_and_l2
            result = scrape_l1_and_l2(sport)

            l1 = result.get("l1", {})
            l2 = result.get("l2", {})
            remaining = result.get("remaining_requests", "?")

            l1_books = l1.get("books_found", [])
            print(f"  [L1] {l1.get('games_found', 0)} games, {l1.get('rows_written', 0)} rows | books: {', '.join(l1_books) if l1_books else 'n/a'}")
            if l1.get("error"):
                print(f"  [L1] note: {l1['error']}")
            print(f"  [L2] {l2.get('games_found', 0)} games, {l2.get('agg_rows', 0)} agg rows | API remaining: {remaining}")
            if l2.get("error"):
                print(f"  [L2] note: {l2['error']}")

            total_l1 += l1.get("rows_written", 0)
            total_l2 += l2.get("rows_written", 0)
        except Exception as e:
            print(f"  FAILED: {repr(e)}")

    print(f"\n[odds_snapshot_all] Done. L1: {total_l1} rows | L2: {total_l2} rows across {len(filtered)} sports.")


def cmd_full_snapshot(args):
    """Run odds_snapshot (L1+L2) then regular snapshot (L3/DK) then report."""
    print(f"[full_snapshot] Starting full pipeline for {args.sport}...")
    cmd_odds_snapshot(args)
    print(f"[full_snapshot] Now running DK snapshot...")
    cmd_snapshot(args)


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
    s2a = sub.add_parser("report_live", help="Rebuild live dashboard outputs only")
    s2a.set_defaults(func=cmd_report_live)
    s2b = sub.add_parser("report_maintenance", help="Run slow maintenance rebuilds")
    s2b.set_defaults(func=cmd_report_maintenance)
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

    s7 = sub.add_parser("odds_snapshot", help="Fetch L1+L2 odds from The-Odds-API")
    s7.add_argument("--sport", choices=SPORT_CONFIG.keys(), required=True)
    s7.set_defaults(func=cmd_odds_snapshot)

    s8 = sub.add_parser("full_snapshot", help="Fetch L1+L2 odds, then DK snapshot + report")
    s8.add_argument("--sport", choices=SPORT_CONFIG.keys(), required=True)
    s8.add_argument("--debug", action="store_true")
    s8.set_defaults(func=cmd_full_snapshot)

    s9 = sub.add_parser("odds_snapshot_all", help="Fetch L1+L2 for all active sports (auto-detects season)")
    s9.set_defaults(func=cmd_odds_snapshot_all)

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
