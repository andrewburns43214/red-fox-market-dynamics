import argparse
import csv
import datetime as dt
import os
import re
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

    # CFB name normalizer (DK vs ESPN differences)
    def norm_team(s: str) -> str:
        s = (s or "").strip()
        repl = {
            "Miami FL": "Miami",
            "Arizona State": "Arizona St",
        }
        return repl.get(s, s)

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

        key = f"{norm_team(away_name)} @ {norm_team(home_name)}"
        out[key] = iso

    # map DK game strings -> ESPN-normalized keys
    result = {}
    for g in games:
        if " @ " not in g:
            result[g] = ""
            continue
        a, h = g.split(" @ ", 1)
        result[g] = out.get(f"{norm_team(a)} @ {norm_team(h)}", "")
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

def _espn_kickoff_map_date_range(scoreboard_url_base: str, games: list[str], days: int = 14) -> dict[str, str]:
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
        # normalize team names consistently using your existing helper
        try:
            return _normalize_team_name(_safe(x))
        except Exception:
            return _safe(x).strip()


        # Expand common leading abbreviations used in DK game strings
        # (keep this small + safe; it only rewrites when the string starts with the abbrev + space)
        abbr = {
            "DAL ": "Dallas ",
            "CLE ": "Cleveland ",
            "CIN ": "Cincinnati ",
            "DEN ": "Denver ",
            "NY ":  "New York ",
            "LA ":  "Los Angeles ",
            "NO ":  "New Orleans ",
            "NE ":  "New England ",
            "GB ":  "Green Bay ",
            "KC ":  "Kansas City ",
            "SF ":  "San Francisco ",
            "TB ":  "Tampa Bay ",
            "LV ":  "Las Vegas ",
        }
        for k, v in abbr.items():
            if s.startswith(k):
                s = v + s[len(k):]
                break

        # normalize team names consistently using your existing helper
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

        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            continue

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
            home_name = _norm_team(ht.get("displayName") or ht.get("name") or "")
            away_name = _norm_team(at.get("displayName") or at.get("name") or "")

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

        # Try in order
        for c in candidates:
                # Fuzzy fallback (LAST RESORT): token overlap against ESPN matchup keys.
        # This prevents future breakage when DK/ESPN naming drifts (abbrev/city changes).
            if not iso:
                ng = _norm_game_key(g)
            # must look like a matchup
            if "@" in ng:
                gtoks = set([t for t in ng.replace("@", " ").split() if len(t) >= 3])
                best_iso = ""
                best_score = 0

                for k, v in espn_index.items():
                    if not v:
                        continue
                    nk = _norm_game_key(k)
                    if "@" not in nk:
                        continue

                    ktoks = set([t for t in nk.replace("@", " ").split() if len(t) >= 3])
                    score = len(gtoks & ktoks)

                    if score > best_score:
                        best_score = score
                        best_iso = v

                # Conservative threshold to avoid bad matches
                if best_score >= 4:
                    iso = best_iso

            c = _safe(c)
            if not c:
                continue
            iso = espn_index.get(c, "")
            if not iso:
                iso = espn_index.get(_norm_game_key(c), "")
            if iso:
                break


        result[g] = iso or ""

    return result


import re

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


def get_espn_kickoff_map(sport: str, games: list[str]) -> dict[str, str]:
    """
    Generic ESPN kickoff resolver.
    Returns DK-game-keyed kickoff ISO map.
    Safe no-op for unsupported sports.
    """
    base = ESPN_SCOREBOARD_BASE.get(sport)
    if not base or not games:
        return {}
    try:
                    # DK "n7days" spans past week; also want ~2 weeks ahead
            return _espn_kickoff_map_date_range(base, games, days=21)
    except Exception as e:
        print(f"[espn] kickoff fetch failed for {sport}: {e}")
        return {}

# Keep these wrappers for backwards-compatibility with existing code below
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

    s = str(s).strip()

    # Drop DK leading abbreviations like "BOS", "UTA", "DET", "SJ", "TB", etc.
    parts = s.split()
    if len(parts) >= 2 and parts[0].isupper() and len(parts[0]) <= 4:
        s = " ".join(parts[1:])

    # NBA fixes
    s = s.replace("LA Lakers", "Lakers")
    s = s.replace("LA Clippers", "Clippers")
    s = s.replace("NY Knicks", "Knicks")
    s = s.replace("GS Warriors", "Warriors")

    # NFL fixes
    s = s.replace("LA Rams", "Rams")
    s = s.replace("LA Chargers", "Chargers")
    s = s.replace("NY Giants", "Giants")
    s = s.replace("NY Jets", "Jets")
    s = s.replace("NE Patriots", "Patriots")

    # Strip trailing 2-letter state tags DK sometimes includes (college)
    # Examples: "Miami FL" -> "Miami", "Albany NY" -> "Albany"
    if len(s) >= 3 and s[-3] == " " and s[-2:].isalpha() and s[-2:].isupper():
        s = s[:-3].strip()

    # Minimal alias map for remaining ESPN shortDisplayName quirks
    COLLEGE_ALIASES = {
        # Colleges with "State" abbreviations
        "Arizona State": "Arizona St",
        "Florida State": "Florida St",

        # Common ESPN short names
        "Pittsburgh": "Pitt",
        "Western Michigan": "W Michigan",

        # ESPN often uses mascot / shortened forms for these
        "Coastal Carolina": "Coastal Carolina Chanticleers",
        "Louisiana Tech": "Louisiana Tech Bulldogs",

        # Albany fix (after NY is stripped)
        "Albany": "UAlbany",
        "Albany NY": "UAlbany",

    }
    s = COLLEGE_ALIASES.get(s, s)

    return s.strip()





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
# Settings (edit these)
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
# Thresholds (your “rare dark green” setup)
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
    meaningful_move_pts: float = 1.5  # used for “strong line behavior”

    # “No movement” / resistance trigger (optional)
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
    - Finds repeating “game cards” or table rows
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

        # Create a single “side row” record as a fallback
        # We'll show it in output even if it’s not perfectly split by team yet.
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
    color ∈ {"DARK_GREEN","LIGHT_GREEN","GREY","YELLOW","RED"}
    """
    # If we don't have percentages, we can't score well
    if bets_pct is None or money_pct is None:
        return "GREY", "Missing bet%/money% → default Grey"

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

        # Key number note can promote “strong line behavior” (NFL)
        if key_number_note and key_number_note.strip():
            strong_line_signal = True

    # DARK GREEN (rare): requires strong line behavior + strong money signal, AND no obvious news explanation
    if strong_line_signal and dark_money_signal and not has_news:
        return "DARK_GREEN", "Book behavior + strong money-vs-bets imbalance; no obvious news → Market Edge Confirmed"

    # If news explains it, keep in Light Green even if strong
    if strong_line_signal and dark_money_signal and has_news:
        return "LIGHT_GREEN", "Strong signals but major news present → downgrade to Market Edge Developing" 

    # LIGHT GREEN: money-vs-bets imbalance without strong line confirmation
    if light_money_signal:
        return "LIGHT_GREEN", "Money concentration vs bet count → Market Edge Developing (watch for confirmation)"

    # RED: avoid this side
    if is_red:
        return "RED", "Extremely public + weak money support → Wrong Side / Trap (evaluate opposite side)"

    # YELLOW: public-driven
    if is_yellow:
        return "YELLOW", "Public-heavy demand without strong money support → Caution"

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
                "open_line": row.get("open"),
                "current_line": row.get("current"),
                "injury_news": row.get("news"),
                "key_number_note": row.get("key_number_note"),
                "dk_start_iso": row.get("start_time") or row.get("startDate") or row.get("start_time_iso"),
            })


def infer_market_type(side_txt: str, line_txt: str) -> str:
    s = (side_txt or "").strip().lower()
    t = (str(line_txt) if line_txt is not None else "").strip()

    import re

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

def build_dashboard():
    ensure_data_dir()

    if not os.path.exists(SNAPSHOT_CSV):
        print(f"[warn] no snapshots yet: {SNAPSHOT_CSV}")
        return

    df = pd.read_csv(SNAPSHOT_CSV)
    print(f"[dash debug] rows after read_csv: {len(df)}")
    if "sport" in df.columns:
        print(f"[dash debug] sports present: {sorted(df['sport'].dropna().astype(str).unique().tolist())}")
    else:
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
    print(f"[dash debug] bad timestamp rows: {len(bad)}")

    # Drop rows that are unusable for grouping/rendering
    for col in ["sport", "game", "side", "market", "current_line"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).replace({"nan": "", "None": ""})

    df = df[
        (df.get("sport", "").astype(str).str.strip() != "")
        & (df.get("game", "").astype(str).str.strip() != "")
        & (df.get("side", "").astype(str).str.strip() != "")
        & (df.get("market", "").astype(str).str.strip() != "")
        & (df.get("current_line", "").astype(str).str.strip() != "")
    ].copy()

    # If timestamp couldn't parse, drop it
    df = df.dropna(subset=["timestamp"]).copy()
    print(f"[dash debug] rows after timestamp parse/dropna: {len(df)}")

    # Sport display labels (keep consistent with your existing mapping if present elsewhere)
    df["sport"] = df["sport"].astype(str).str.lower().str.strip()
    df["sport_label"] = df["sport"].str.upper()


    # Grouping keys must not be NaN
    df["game_id"] = df["game_id"].fillna("").astype(str)
    df["side"] = df["side"].fillna("").astype(str)
    df["market"] = df["market"].fillna("unknown").astype(str)

    # Clean DK “opens in a new tab…” junk if present
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
    # We want the computed stable-open to be the real open_line used downstream.
    if "open_line_first" in df.columns:
        df["open_line"] = df["open_line_first"]



    # --- LATEST row per selection
        latest = (
        df.groupby(["sport", "game_id", "market_display", "side_key"], as_index=False)
          .tail(1)
          .copy()
          .reset_index(drop=True)
    )

    print(f"[dash debug] rows in latest: {len(latest)}")


        # DK start time is primary source (stable, keyed by game_id)
    if "dk_start_iso" in latest.columns:
        latest["game_time_iso"] = latest["dk_start_iso"]

        # --- ESPN kickoff enrichment (Game Time)
    # Minimal: add game_time_iso + game_time_display to `latest` (does not affect grouping)
    try:
        latest["game"] = latest["game"].fillna("").astype(str)
        print(f"[dash debug] ESPN: unique games in latest = {latest['game'].nunique()}")


        all_kickoffs = {}
        for sp in latest["sport"].dropna().astype(str).unique().tolist():
            games = latest.loc[latest["sport"] == sp, "game"].dropna().astype(str).unique().tolist()
            if not games:
                continue
            km = get_espn_kickoff_map(sp, games)
            if isinstance(km, dict) and km:
                all_kickoffs.update(km)

                # First: exact match
        latest["game_time_iso"] = latest["game"].map(lambda g: all_kickoffs.get(g, ""))

        # Fallback: normalized match (fixes DK vs ESPN naming differences)
        if (latest["game_time_iso"].fillna("") == "").any() and all_kickoffs:
            norm_to_time = {}
            for k, v in all_kickoffs.items():
                nk = _norm_game_key(k)
                if nk and v and nk not in norm_to_time:
                    norm_to_time[nk] = v

            def _lookup_time(g):
                t = all_kickoffs.get(g, "")
                if t:
                    return t
                ng = _norm_game_key(g)
                return norm_to_time.get(ng, "")

            latest["game_time_iso"] = latest["game"].map(_lookup_time)

        print(f"[dash debug] ESPN: kickoff map size = {len(all_kickoffs)}")
        # show a few example mappings to confirm key format matches latest['game']
        try:
            sample_games = latest["game"].dropna().astype(str).unique().tolist()[:5]
            for sg in sample_games:
                print(f"[dash debug] ESPN sample map: game='{sg}' -> '{all_kickoffs.get(sg, '')}'")
        except Exception as _e:
            pass


        # Display in America/New_York (ET). Keep empty if parse fails.
        ts = pd.to_datetime(latest["game_time_iso"], errors="coerce", utc=True)
        try:
            ts_local = ts.dt.tz_convert("America/New_York")
        except Exception:
            ts_local = ts  # fallback (still UTC)

        # Always set display column (this was broken by indentation)
        latest["game_time_display"] = ts_local.dt.strftime("%a, %b %d %I:%M %p ET").fillna("")
        print(f"[dash debug] latest game_time_display non-empty: {(latest['game_time_display'].fillna('') != '').sum()} / {len(latest)}")



        # --- debug: verify columns are actually populated
        if "open_line" in latest.columns:
            print(f"[dash debug] latest open_line non-empty: {(latest['open_line'].fillna('') != '').sum()} / {len(latest)}")
        if "game_time_display" in latest.columns:
            print(f"[dash debug] latest game_time_display non-empty: {(latest['game_time_display'].fillna('') != '').sum()} / {len(latest)}")
        if "open_line_first" in df.columns:
            print(f"[dash debug] df has open_line_first (merge suffix): yes")

    except Exception as e:
        print(f"[dash debug] ESPN kickoff enrichment failed: {e}")



        # --- PREV = previous snapshot row per selection (2nd to last)
    prev = (
        df.groupby(["sport", "game_id", "market_display", "side_key"], as_index=False)
          .nth(-2)
          .rename(columns={"current_line": "prev_current_line"})
          [["sport", "game_id", "market_display", "side_key", "prev_current_line"]]
    )
    latest = latest.merge(prev, on=["sport", "game_id", "market_display", "side_key"], how="left")


    # Market display
    latest["market_display"] = latest.apply(
        lambda rr: infer_market_type(rr.get("side", ""), rr.get("current_line", "")),
        axis=1
    )

    # Debug sample
    if len(latest) > 0:
        sample = latest.head(5)
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
        print(f"[dash debug] after main-line filter: rows in latest = {len(latest)}")
        print("[dash debug] after main-line filter: unique games =", latest["game_id"].nunique())
    except Exception:
        pass

    # ========= END MAIN LINE FILTER =========



    # Deltas
    latest["odds_move_open"] = latest["current_odds"] - latest["open_odds"]
    latest["line_move_open"] = latest["current_line_val"] - latest["open_line_val"]
    latest["odds_move_prev"] = latest["current_odds"] - latest["prev_odds"]
    latest["line_move_prev"] = latest["current_line_val"] - latest["prev_line_val"]

    # Classify each row (this is your existing signal logic)
    colors = []
    explains = []
    ml_green = set()

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
                expl = f"{expl} | ⚠️ Big underdog moneyline (+{int(row['current_odds'])})"
        except Exception:
            pass

        if (
            mkt == "SPREAD"
            and (game, side) in ml_green
            and color not in ("DARK_GREEN", "LIGHT_GREEN")
        ):
            expl = f"{expl} | Sharp ML, margin risk — ML favored over spread"

        colors.append(color)
        explains.append(expl)

    latest = latest.copy()
    latest["color"] = colors
    latest["why"] = explains

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

    # Ensure we have a displayable game time column if we have ISO times
    if "game_time_display" not in latest.columns and "game_time_iso" in latest.columns:
        ts = pd.to_datetime(latest["game_time_iso"], errors="coerce", utc=True)
        try:
            ts_local = ts.dt.tz_convert("America/New_York")
        except Exception:
            ts_local = ts
            espn_display = ts_local.dt.strftime("%a, %b %d %I:%M %p ET").fillna("")

        # Prefer DK time if present, otherwise fallback to ESPN
        if "game_time" in latest.columns:
            latest["game_time_display"] = latest["game_time"].fillna("").astype(str)
            m = latest["game_time_display"].str.strip().eq("")
            latest.loc[m, "game_time_display"] = espn_display.loc[m]
        else:
            latest["game_time_display"] = espn_display


    # Optional Game Time column if it exists (don’t break if missing)
    time_col = None
    if "game_time_display" in latest.columns:
        time_col = "game_time_display"
    elif "game_time" in latest.columns:
        time_col = "game_time"

    # Always show the column if we have any time field (even if some rows are blank)
    show_game_time = (time_col is not None)



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
    # This prevents HTML column misalignment when optional columns (time/news/key) are toggled.
    header_cols = (
        ["Sport", "Game"]
        + (["Game Time"] if show_game_time else [])
        + ["Side", "Market", "Bets % (on Side)", "Money % (on Side)", "Open", "Current"]
        + (["News?"] if show_news else [])
        + (["Key # Note"] if show_key_note else [])
        + [
            "Price Change (Last)",
            "Line Change (Last)",
            "Price Change (Since Open)",
            "Line Change (Since Open)",
            "Why",
        ]
    )
    colspan = len(header_cols)

    print(f"[dash debug] header column count={colspan} show_game_time={show_game_time} show_news={show_news} show_key_note={show_key_note}")


    # Sort for display: kickoff time first (like the old dashboard), then group by game
    if "game_time_iso" in latest.columns:
        latest["_sort_time"] = pd.to_datetime(latest["game_time_iso"], errors="coerce", utc=True)
        try:
            latest["_sort_time"] = latest["_sort_time"].dt.tz_convert("America/New_York")
        except Exception:
            pass
    elif "game_time_display" in latest.columns:
        latest["_sort_time"] = pd.to_datetime(latest["game_time_display"], errors="coerce")
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




    rows_html = []
    last_label = None

    for _, rr in latest.iterrows():
        if rr["sport_label"] != last_label:
            last_label = rr["sport_label"]
            rows_html.append(f"""
<tr data-header="1">
  <td colspan="{colspan}" style="background:#222;color:#fff;font-weight:bold;font-size:14px;padding:10px;">
    {last_label}
  </td>
</tr>
""")

        st = color_style(rr.get("color", "GREY"))
                # ---- DISPLAY-ONLY SIDE CLEANUP (do NOT affect data) ----
        mkt = rr.get("market_display", "")
        side_disp = rr.get("side", "")

        # For spreads, remove the numeric spread from the side text
        # Example: "Navy -5.5" -> "Navy"
        if mkt == "SPREAD":
            import re
            side_disp = re.sub(r"\s[+-]\d+(?:\.\d+)?\s*$", "", str(side_disp)).strip()


        time_td = ""
        if show_game_time:
            v = rr.get(time_col, "")
            time_td = f"<td>{'' if pd.isna(v) else v}</td>"

        rows_html.append(f"""
<tr style="{st}"
    data-row="1"
    data-color="{rr.get('color','')}"
    data-sport="{rr.get('sport_label','')}"
    data-market="{rr.get('market_display','')}"
    data-ml-odds="{'' if pd.isna(rr.get('current_odds')) else int(rr.get('current_odds'))}"
    data-search="{str(rr.get('game',''))} {str(side_disp)} {str(rr.get('market_display',''))}">
  <td>{rr.get('sport_label','')}</td>
  <td>{rr.get('game','')}</td>
  {time_td}
    <td>{side_disp}</td>
  <td>{rr.get('market_display','')}</td>
    <td>{'' if pd.isna(rr.get('bets_pct')) else f"{int(rr['bets_pct'])}% on {side_disp}"}</td>
    <td>{'' if pd.isna(rr.get('money_pct')) else f"{int(rr['money_pct'])}% on {side_disp}"}</td>
  <td>{'' if pd.isna(rr.get('open_line')) else rr.get('open_line')}</td>
  <td>{'' if pd.isna(rr.get('current_line')) else rr.get('current_line')}</td>
  {f"<td>{'' if pd.isna(rr.get('injury_news')) else rr.get('injury_news')}</td>" if show_news else ""}
  {f"<td>{'' if pd.isna(rr.get('key_number_note')) else rr.get('key_number_note')}</td>" if show_key_note else ""}

  <td>{"—" if pd.isna(rr.get("odds_move_prev")) else f"{int(rr['odds_move_prev']):+d}"}</td>
  <td>{"—" if pd.isna(rr.get("line_move_prev")) else f"{rr['line_move_prev']:+.1f}"}</td>
  <td>{"—" if pd.isna(rr.get("odds_move_open")) else f"{int(rr['odds_move_open']):+d}"}</td>
  <td>{"—" if pd.isna(rr.get("line_move_open")) else f"{rr['line_move_open']:+.1f}"}</td>

  <td>{rr.get('why','')}</td>
</tr>
""")

    # =========================
    # Legend + Snapshot timestamps (current + previous)
    # =========================
    # Use raw snapshot timestamps from df (not latest) so it matches your snapshot history.
    current_ts_disp = ""
    prev_ts_disp = ""

    try:
        if "timestamp" in df.columns:
            ts_all = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna()
            ts_unique = ts_all.drop_duplicates().sort_values()

            if len(ts_unique) > 0:
                cur = ts_unique.iloc[-1]
                try:
                    cur_local = cur.tz_convert("America/New_York")
                except Exception:
                    cur_local = cur
                current_ts_disp = cur_local.strftime("%a %b %d, %Y %I:%M:%S %p ET")

            if len(ts_unique) > 1:
                prv = ts_unique.iloc[-2]
                try:
                    prv_local = prv.tz_convert("America/New_York")
                except Exception:
                    prv_local = prv
                prev_ts_disp = prv_local.strftime("%a %b %d, %Y %I:%M:%S %p ET")
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
    <div>
      <span style="display:inline-block; width:12px; height:12px; background:#0B5A12; border:1px solid #0B5A12; vertical-align:middle; margin-right:6px;"></span>
      <b>Dark Green</b> — Market Edge Confirmed
    </div>
    <div>
      <span style="display:inline-block; width:12px; height:12px; background:#9AF0A0; border:1px solid #9AF0A0; vertical-align:middle; margin-right:6px;"></span>
      <b>Light Green</b> — Market Edge Developing
    </div>
    <div>
      <span style="display:inline-block; width:12px; height:12px; background:#E0E0E0; border:1px solid #E0E0E0; vertical-align:middle; margin-right:6px;"></span>
      <b>Grey</b> — No clear market signal
    </div>
    <div>
      <span style="display:inline-block; width:12px; height:12px; background:#F6E38A; border:1px solid #F6E38A; vertical-align:middle; margin-right:6px;"></span>
      <b>Yellow</b> — Caution / conflicting signals
    </div>
    <div>
      <span style="display:inline-block; width:12px; height:12px; background:#F08A8A; border:1px solid #F08A8A; vertical-align:middle; margin-right:6px;"></span>
      <b>Red</b> — Negative market signal / fade zone
    </div>
  </div>

</div>
"""
    # =========================
    # Filters UI (client-side)
    # =========================
    sports = []
    try:
        sports = [s for s in latest["sport_label"].dropna().unique().tolist() if str(s).strip() != ""]
        sports = sorted(sports)
    except Exception:
        sports = []

    sport_opts = '<option value="ALL">All Sports</option>' + "".join(
        f'<option value="{s}">{s}</option>' for s in sports
    )

    filters_html = f"""
<div style="margin:0 0 14px 0; padding:10px 12px; background:#f7f7f7; border:1px solid #ddd; border-radius:8px;">
  <div style="font-weight:bold; margin-bottom:6px;">Filters</div>
  <div style="display:flex; gap:14px; flex-wrap:wrap; align-items:center; font-size:12px;">
    <label><input type="checkbox" id="fGreens"> Greens (Dark + Light)</label>
    <label><input type="checkbox" id="fYellow"> Yellow</label>
    <label><input type="checkbox" id="fRed"> Red</label>
    <label><input type="checkbox" id="fHideGrey"> Hide Grey</label>
    <label title="Hides Dark Green moneylines that are heavy favorites (e.g. -250, -300, -450)">
  <input type="checkbox" id="fHideHeavyMLDG">
  Hide heavy-favorite ML Dark Greens
</label>


    <label>Sport:
      <select id="fSport" style="margin-left:6px;">
        {sport_opts}
      </select>
    </label>
        <label>Search:
      <input id="fSearch" type="text" placeholder="Team, game, side..." style="margin-left:6px; padding:4px 6px; width:220px;">
    </label>

    <label><input type="checkbox" id="fCompact"> Compact</label>
    <button id="fApply" style="padding:4px 8px; cursor:pointer;">Apply</button>
    <button id="fReset" style="padding:4px 8px; cursor:pointer;">Reset</button>

  </div>
  <div id="fCount" style="margin-top:6px; font-size:12px; color:#333;"></div>
</div>
"""
    filters_js = """
<script>
document.addEventListener("DOMContentLoaded", () => {

  function applyFilters() {
    const showGreens = document.getElementById("fGreens").checked;
    const showYellow = document.getElementById("fYellow").checked;
    const showRed = document.getElementById("fRed").checked;
    const hideGrey = document.getElementById("fHideGrey").checked;
    const hideHeavyMLDG = document.getElementById("fHideHeavyMLDG").checked;
const HEAVY_ML_ODDS_THRESHOLD = -250;
    const sportVal = document.getElementById("fSport").value;
    const q = (document.getElementById("fSearch").value || "").trim().toLowerCase();

    const anyColor = showGreens || showYellow || showRed;

    let visible = 0;

    document.querySelectorAll("tr[data-row='1']").forEach(tr => {
      const color = tr.dataset.color || "";
      const sport = tr.dataset.sport || "";

      let ok = true;
      let forceHide = false;

      // Sport filter
      if (sportVal !== "ALL" && sport !== sportVal) ok = false;

      // Color filter set
      if (anyColor) {
        const isGreen = (color === "DARK_GREEN" || color === "LIGHT_GREEN");
        const isYellow = (color === "YELLOW");
        const isRed = (color === "RED");
        ok = ok && (
          (showGreens && isGreen) ||
          (showYellow && isYellow) ||
          (showRed && isRed)
        );
      }

      // Hide Grey always applies
            if (hideGrey && color === "GREY") ok = false;
            if (hideHeavyMLDG) {
  const market = (tr.dataset.market || "").toUpperCase();

  const colorVal = (tr.dataset.color || "").toUpperCase();
  const colorNorm = colorVal.replace(/\s+/g, "_"); // "DARK GREEN" -> "DARK_GREEN"

  const mlOddsRaw = (tr.dataset.mlOdds || "").replaceAll("−", "-").trim();
  const mlOdds = mlOddsRaw ? parseInt(mlOddsRaw, 10) : NaN;

  if (market === "MONEYLINE" && colorNorm === "DARK_GREEN" && Number.isFinite(mlOdds)) {
    if (mlOdds <= HEAVY_ML_ODDS_THRESHOLD) forceHide = true;
  }
}



      if (q) {
        const hay = (tr.dataset.search || "").toLowerCase();
        if (!hay.includes(q)) ok = false;
      }

      tr.style.display = (!forceHide && ok) ? "" : "none";
      if (ok) visible++;
    });

    // Hide sport header rows if everything under that header is hidden
    document.querySelectorAll("tr[data-header='1']").forEach(h => {
      let anyVisible = false;
      let sib = h.nextElementSibling;
      while (sib && !sib.hasAttribute("data-header")) {
        if (sib.getAttribute("data-row") === "1" && sib.style.display !== "none") {
          anyVisible = true;
          break;
        }
        sib = sib.nextElementSibling;
      }
      h.style.display = anyVisible ? "" : "none";
    });

    const c = document.getElementById("fCount");
    if (c) c.textContent = "Showing " + visible + " rows";
  }

  function resetFilters() {
    document.getElementById("fGreens").checked = false;
    document.getElementById("fYellow").checked = false;
    document.getElementById("fRed").checked = false;
    document.getElementById("fHideGrey").checked = false;
    document.getElementById("fHideHeavyMLDG").checked = false;
    document.getElementById("fSport").value = "ALL";
    document.getElementById("fSearch").value = "";
    document.getElementById("fCompact").checked = false;
    document.body.classList.remove("compact");


    // Show all rows + headers again
    document.querySelectorAll("tr[data-row='1']").forEach(tr => { tr.style.display = ""; });
    document.querySelectorAll("tr[data-header='1']").forEach(tr => { tr.style.display = ""; });

    const c = document.getElementById("fCount");
    if (c) c.textContent = "";
  }

  // Buttons
  const btnApply = document.getElementById("fApply");
  if (btnApply) btnApply.addEventListener("click", e => { e.preventDefault(); applyFilters(); });

  const btnReset = document.getElementById("fReset");
  if (btnReset) btnReset.addEventListener("click", e => { e.preventDefault(); resetFilters(); });
  const cbCompact = document.getElementById("fCompact");
  if (cbCompact) cbCompact.addEventListener("change", () => {
    document.body.classList.toggle("compact", cbCompact.checked);
});


});
</script>
"""

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Market Intelligence Dashboard</title>
<style>
  body {{ font-family: Arial, sans-serif; padding: 16px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; }}
  th {{ background: #f5f5f5; position: sticky; top: 0; z-index: 2; }}
  body.compact th, body.compact td {{ padding: 3px 6px; font-size: 11px; }}
  body.compact h1 {{ margin: 8px 0; }}
  
</style>
{filters_js}
</head>
<body>
<h1>Market Intelligence Dashboard</h1>
{snapshot_html}
{legend_html}
{filters_html}

<table>
  <thead>
    <tr>
      <th>Sport</th>
      <th>Game</th>
      { "<th>Game Time</th>" if show_game_time else "" }
      <th>Side</th>
      <th>Market</th>
      <th>Bets % (on Side)</th>
      <th>Money % (on Side)</th>
      <th>Open</th>
      <th>Current</th>
      {news_th}
      {key_th}
      <th>Price Change (Last)</th>
      <th>Line Change (Last)</th>
      <th>Price Change (Since Open)</th>
      <th>Line Change (Since Open)</th>
      <th>Edge Insight</th>
    </tr>
  </thead>
  <tbody>
    {''.join(rows_html)}
  </tbody>
</table>
</body>
</html>
"""
    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[ok] wrote dashboard: {REPORT_HTML}")


# =========================
# CLI
# =========================

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

    append_snapshot(rows, args.sport)
    print(f"[ok] appended {len(rows)} rows to {SNAPSHOT_CSV}")

    build_dashboard()

def cmd_report(_args):
    build_dashboard()

def cmd_movement(args):
    movement_report(
        SNAPSHOT_CSV,
        args.sport,
        lookback=args.lookback,
    )




def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)

    s1 = sub.add_parser("snapshot", help="Fetch splits + append a snapshot + rebuild dashboard")
    s1.add_argument("--sport", choices=SPORT_CONFIG.keys(), required=True)
    s1.add_argument("--debug", action="store_true", help="Print debug info to help tune parsing")
    s1.set_defaults(func=cmd_snapshot)

    s2 = sub.add_parser("report", help="Rebuild dashboard from existing snapshots")
    s2.set_defaults(func=cmd_report)
    s3 = sub.add_parser("movement", help="Compare snapshots")
    s3.add_argument("--sport", choices=SPORT_CONFIG.keys(), required=True)
    s3.add_argument("--lookback", type=int, default=1, help="How many snapshots back to compare (1 = most recent previous)")
    s3.set_defaults(func=cmd_movement)


    args = ap.parse_args()

    logger = setup_logger(getattr(args, "debug", False))
    logger.info("command=%s sport=%s", args.func.__name__, getattr(args, "sport", None))
    args.func(args)


if __name__ == "__main__":
    main()

