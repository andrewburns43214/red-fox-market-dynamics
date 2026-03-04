"""
OddsPapi API client for Red Fox engine (Layer 1 primary source).

Provides access to 6 sharp books with timestamped line movements and betting limits:
  - Pinnacle, Singbet, SBOBet, BetCRIS, Circa Sports, Bookmaker.eu

API docs: https://oddspapi.io/docs
Endpoints used:
  - /sports          — list available sports
  - /tournaments     — list tournaments for a sport
  - /fixtures        — list fixtures (games) with team names
  - /odds-by-tournaments — odds per tournament (sharp books filtered)
"""
import json
import os
import time
from datetime import datetime, timezone, timedelta

import requests

from engine_config import (
    ODDSPAPI_KEY,
    ODDSPAPI_BASE_URL,
    ODDSPAPI_TOURNAMENT_MAP,
    ODDSPAPI_SHARP_BOOKS,
    ODDSPAPI_CACHE_JSON,
)


# OddsPapi sport IDs (from /v4/sports endpoint)
ODDSPAPI_SPORT_MAP = {
    "nba": 11,      # Basketball
    "ncaab": 11,    # Basketball (same sport, different tournament)
    "nhl": 15,      # Ice Hockey
    "mlb": 13,      # Baseball
    "nfl": 14,      # American Football
    "ncaaf": 14,    # American Football (same sport, different tournament)
    "ufc": 20,      # MMA
}

# OddsPapi market type IDs
MARKET_MAP = {
    1: "MONEYLINE",    # 1X2 / Moneyline
    2: "SPREAD",       # Handicap / Spread
    3: "TOTAL",        # Over/Under / Total
}

# Reverse: our names → OddsPapi IDs
MARKET_ID_MAP = {
    "MONEYLINE": 1,
    "SPREAD": 2,
    "TOTAL": 3,
}


# Fixture cache to avoid redundant API calls within a day
# Persists to disk so cron runs (separate processes) share the cache
_FIXTURE_CACHE_DIR = os.path.join("data", "oddspapi_fixture_cache")
FIXTURE_CACHE_TTL = 43200  # 12 hours — fixtures don't change within a day
ODDS_CACHE_TTL = 7200      # 2 hours — per-fixture odds cache
MAX_FIXTURE_ODDS_PER_PULL = 5  # budget guard: max /odds calls per snapshot

# Bookmaker key aliases — OddsPapi may use abbreviated keys
BOOK_ALIASES = {
    "pin": "pinnacle", "pinnacle": "pinnacle",
    "sin": "singbet", "singbet": "singbet",
    "sbo": "sbobet", "sbobet": "sbobet",
    "betcris": "betcris", "bcr": "betcris",
    "circa": "circasports", "circasports": "circasports", "cir": "circasports",
    "bookmaker": "bookmaker.eu", "bookmakereu": "bookmaker.eu",
    "beu": "bookmaker.eu", "bookmaker.eu": "bookmaker.eu",
}


def _get_api_key() -> str:
    """Get OddsPapi API key from config or environment."""
    return os.environ.get("ODDSPAPI_KEY", "") or ODDSPAPI_KEY


def _api_get(endpoint: str, params: dict = None) -> dict:
    """
    Make authenticated GET request to OddsPapi.

    Returns:
        dict with "data" (response JSON) and "error" (str or None)
    """
    api_key = _get_api_key()
    if not api_key:
        return {"data": None, "error": "ODDSPAPI_KEY not set"}

    url = f"{ODDSPAPI_BASE_URL}{endpoint}"
    if params is None:
        params = {}
    params["apiKey"] = api_key

    try:
        resp = requests.get(url, params=params, timeout=30)

        if resp.status_code == 401:
            return {"data": None, "error": "Invalid OddsPapi API key"}
        if resp.status_code == 429:
            return {"data": None, "error": "OddsPapi rate limit exceeded"}
        if resp.status_code != 200:
            return {"data": None, "error": f"OddsPapi HTTP {resp.status_code}: {resp.text[:200]}"}

        return {"data": resp.json(), "error": None}

    except requests.exceptions.Timeout:
        return {"data": None, "error": "OddsPapi request timed out"}
    except requests.exceptions.RequestException as e:
        return {"data": None, "error": f"OddsPapi request failed: {str(e)[:200]}"}


def discover_tournament_id(sport: str) -> dict:
    """
    Discover the OddsPapi tournament ID for a sport.

    Calls /tournaments with the sport ID and looks for the primary
    active tournament matching the sport.

    Returns:
        dict with "tournament_id" (int or None) and "error" (str or None)
    """
    sport_lower = sport.lower()
    sport_id = ODDSPAPI_SPORT_MAP.get(sport_lower)
    if sport_id is None:
        return {"tournament_id": None, "error": f"Unknown sport: {sport}"}

    # Check if we already have it
    known_id = ODDSPAPI_TOURNAMENT_MAP.get(sport_lower)
    if known_id is not None:
        return {"tournament_id": known_id, "error": None}

    result = _api_get("/tournaments", params={"sportId": sport_id})
    if result["error"]:
        return {"tournament_id": None, "error": result["error"]}

    tournaments = result["data"]
    if not isinstance(tournaments, list):
        return {"tournament_id": None, "error": "Unexpected tournament response format"}

    # Search patterns for each sport
    search_terms = {
        "nba": ["nba"],
        "ncaab": ["ncaa", "college", "ncaab"],
        "nhl": ["nhl"],
        "mlb": ["mlb"],
        "nfl": ["nfl"],
        "ncaaf": ["ncaa", "college", "ncaaf"],
        "ufc": ["ufc", "mma"],
    }

    terms = search_terms.get(sport_lower, [sport_lower])

    for t in tournaments:
        name = (t.get("name") or "").lower()
        tid = t.get("id")
        if tid and any(term in name for term in terms):
            return {"tournament_id": tid, "error": None}

    # Return first tournament as fallback if only one exists for the sport
    if len(tournaments) == 1 and tournaments[0].get("id"):
        return {"tournament_id": tournaments[0]["id"], "error": None}

    return {
        "tournament_id": None,
        "error": f"Could not find tournament for {sport}. Available: {[t.get('name') for t in tournaments[:10]]}",
    }


def fetch_fixtures(sport: str, date_from: str = None, date_to: str = None) -> dict:
    """
    Fetch fixtures (games) for a sport from OddsPapi.

    Uses a 12-hour in-memory cache to avoid redundant API calls —
    fixtures don't change within a day, so we only need 1 call per sport per day.

    Args:
        sport: Our sport key (nba, nfl, etc.)
        date_from: ISO date string (default: today)
        date_to: ISO date string (default: tomorrow)

    Returns:
        dict with:
            "fixtures": list of fixture dicts
            "fixture_map": dict {fixtureId: {home, away, commence_time}}
            "error": str or None
            "from_cache": bool
    """
    sport_lower = sport.lower()

    # Check disk cache first (shared across cron runs — saves API calls)
    cache_file = os.path.join(_FIXTURE_CACHE_DIR, f"{sport_lower}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as cf:
                cached = json.load(cf)
            cache_ts = datetime.fromisoformat(cached["timestamp"])
            age = (datetime.now(timezone.utc) - cache_ts).total_seconds()
            if age < FIXTURE_CACHE_TTL:
                cached["result"]["from_cache"] = True
                return cached["result"]
        except Exception:
            pass

    # Get tournament ID
    tid_result = discover_tournament_id(sport_lower)
    if tid_result["error"]:
        return {"fixtures": [], "fixture_map": {}, "error": tid_result["error"], "from_cache": False}

    tournament_id = tid_result["tournament_id"]

    # Default date range: today and tomorrow (UTC)
    now_utc = datetime.now(timezone.utc)
    if date_from is None:
        date_from = now_utc.strftime("%Y-%m-%dT00:00:00Z")
    if date_to is None:
        date_to = (now_utc + timedelta(days=1)).strftime("%Y-%m-%dT23:59:59Z")

    params = {
        "tournamentId": tournament_id,
        "from": date_from,
        "to": date_to,
    }

    result = _api_get("/fixtures", params=params)
    if result["error"]:
        return {"fixtures": [], "fixture_map": {}, "error": result["error"], "from_cache": False}

    fixtures = result["data"]
    if not isinstance(fixtures, list):
        return {"fixtures": [], "fixture_map": {}, "error": "Unexpected fixtures response", "from_cache": False}

    # Build fixture map for team name lookup
    fixture_map = {}
    for f in fixtures:
        fid = f.get("id") or f.get("fixtureId")
        if not fid:
            continue

        # OddsPapi uses participant1/participant2 (home/away varies by response)
        home = f.get("participant1Name") or f.get("homeTeam") or ""
        away = f.get("participant2Name") or f.get("awayTeam") or ""
        start = f.get("startTime") or f.get("commence_time") or ""

        fixture_map[str(fid)] = {
            "home": home,
            "away": away,
            "commence_time": start,
        }

    fetch_result = {
        "fixtures": fixtures,
        "fixture_map": fixture_map,
        "error": None,
        "from_cache": False,
    }

    # Cache to disk (12 hours, shared across cron runs)
    try:
        os.makedirs(_FIXTURE_CACHE_DIR, exist_ok=True)
        with open(cache_file, "w") as cf:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "result": fetch_result,
            }, cf)
    except Exception:
        pass

    return fetch_result


def _resolve_bookmaker(bk_key: str) -> str:
    """Resolve OddsPapi bookmaker key to our canonical name."""
    key_lower = bk_key.lower().strip()
    if key_lower in {b.lower() for b in ODDSPAPI_SHARP_BOOKS}:
        return key_lower
    return BOOK_ALIASES.get(key_lower, key_lower)


def _classify_market(mkt_id_str: str) -> str:
    """Classify OddsPapi market ID into MONEYLINE / SPREAD / TOTAL."""
    try:
        mkt_id = int(mkt_id_str)
    except (ValueError, TypeError):
        return None
    # Exact match (1, 2, 3)
    if mkt_id in MARKET_MAP:
        return MARKET_MAP[mkt_id]
    # First-digit heuristic for multi-digit IDs (1XX→ML, 2XX→Spread, 3XX→Total)
    if mkt_id >= 100:
        first = mkt_id // 100
    elif mkt_id >= 10:
        first = mkt_id // 10
    else:
        return None
    return MARKET_MAP.get(first)


def _resolve_side(market_name: str, outcome_id: str, home: str, away: str) -> str:
    """Map OddsPapi outcome ID to human-readable side."""
    try:
        out_num = int(outcome_id)
    except (ValueError, TypeError):
        return outcome_id
    last_digit = out_num % 10
    if market_name == "TOTAL":
        return "Over" if last_digit % 2 == 1 else "Under"
    # Moneyline / Spread: odd outcome = participant1 (home), even = participant2 (away)
    return home if last_digit % 2 == 1 else away


def _load_odds_cache(fid: str):
    """Load cached odds for a specific fixture. Returns data or None."""
    safe_fid = str(fid).replace("/", "_").replace("\\", "_")
    cache_file = os.path.join(_FIXTURE_CACHE_DIR, f"odds_{safe_fid}.json")
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, "r") as f:
            cached = json.load(f)
        cache_ts = datetime.fromisoformat(cached["timestamp"])
        age = (datetime.now(timezone.utc) - cache_ts).total_seconds()
        if age < ODDS_CACHE_TTL:
            return cached["data"]
    except Exception:
        pass
    return None


def _save_odds_cache(fid: str, data):
    """Save per-fixture odds data to disk cache."""
    try:
        safe_fid = str(fid).replace("/", "_").replace("\\", "_")
        os.makedirs(_FIXTURE_CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(_FIXTURE_CACHE_DIR, f"odds_{safe_fid}.json")
        with open(cache_file, "w") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": data,
            }, f)
    except Exception:
        pass


def _parse_odds_response(fid: str, raw, fixture_map: dict, sharp_set: set) -> list:
    """
    Parse the nested OddsPapi /odds response for a single fixture.

    Response structure:
        bookmakerOdds → {bookmaker_key} → markets → {market_id}
            → outcomes → {outcome_id} → players → {index} → {price, changedAt, limit}
    """
    parsed = []

    if isinstance(raw, list):
        # If response is a list, find the matching fixture
        for item in raw:
            if isinstance(item, dict) and str(item.get("fixtureId", "")) == str(fid):
                raw = item
                break
        else:
            raw = raw[0] if raw and isinstance(raw[0], dict) else {}

    if not isinstance(raw, dict):
        return parsed

    fx_info = fixture_map.get(str(fid), {})
    home_team = fx_info.get("home", "Home")
    away_team = fx_info.get("away", "Away")

    bookmaker_odds = raw.get("bookmakerOdds", {})
    if not isinstance(bookmaker_odds, dict):
        return parsed

    for bk_key, bk_data in bookmaker_odds.items():
        bk_norm = _resolve_bookmaker(bk_key)
        if bk_norm not in sharp_set:
            continue

        if not isinstance(bk_data, dict):
            continue
        if not bk_data.get("bookmakerIsActive", True):
            continue

        markets = bk_data.get("markets", {})
        if not isinstance(markets, dict):
            continue

        for mkt_id_str, mkt_data in markets.items():
            market_name = _classify_market(mkt_id_str)
            if not market_name:
                continue
            if not isinstance(mkt_data, dict):
                continue

            outcomes = mkt_data.get("outcomes", {})
            if not isinstance(outcomes, dict):
                continue

            for out_id_str, out_data in outcomes.items():
                if not isinstance(out_data, dict):
                    continue

                players = out_data.get("players", {})
                if not isinstance(players, dict):
                    continue

                # Take primary line (index "0"), fall back to first available
                p_data = players.get("0") or players.get(0)
                if not isinstance(p_data, dict):
                    for p_key in sorted(players.keys(), key=str):
                        if isinstance(players[p_key], dict):
                            p_data = players[p_key]
                            break

                if not isinstance(p_data, dict):
                    continue

                price = p_data.get("price")
                if price is None:
                    continue
                if not p_data.get("active", True):
                    continue

                changed_at = p_data.get("changedAt", "")
                limit = p_data.get("limit")
                handicap = (p_data.get("handicap") or p_data.get("line")
                            or p_data.get("point") or p_data.get("specialBetValue"))

                side = _resolve_side(market_name, out_id_str, home_team, away_team)

                try:
                    odds_american = _decimal_to_american(float(price))
                except (ValueError, TypeError):
                    odds_american = 0

                parsed.append({
                    "fixture_id": str(fid),
                    "bookmaker": bk_norm,
                    "market": market_name,
                    "side": side,
                    "line": handicap,
                    "odds_decimal": price,
                    "odds_american": odds_american,
                    "changed_at": changed_at,
                    "limit": limit,
                })

    return parsed


def fetch_odds(sport: str, fixture_ids: list = None) -> dict:
    """
    Fetch sharp book odds from OddsPapi.

    Calls /odds per fixture (fixtureId required by API).
    Uses per-fixture caching to stay within monthly budget.

    Args:
        sport: Our sport key
        fixture_ids: Optional list of specific fixture IDs to fetch

    Returns:
        dict with:
            "odds": list of parsed outcome dicts
            "raw": list of raw API responses (for caching)
            "error": str or None
    """
    sport_lower = sport.lower()

    # Fetch fixtures for IDs and team names
    fix_result = fetch_fixtures(sport_lower)
    if fix_result["error"] and not fixture_ids:
        return {"odds": [], "raw": None, "error": fix_result["error"]}

    fixture_map = fix_result.get("fixture_map", {})

    if fixture_ids is None:
        fixture_ids = list(fixture_map.keys())

    if not fixture_ids:
        return {"odds": [], "raw": None, "error": f"No fixtures found for {sport}"}

    # Budget guard: limit per-fixture calls
    fixture_ids = fixture_ids[:MAX_FIXTURE_ODDS_PER_PULL]

    all_parsed = []
    all_raw = []
    errors = []
    sharp_set = set(b.lower() for b in ODDSPAPI_SHARP_BOOKS)

    for fid in fixture_ids:
        # Check per-fixture odds cache first
        cached_data = _load_odds_cache(fid)
        if cached_data is not None:
            parsed = _parse_odds_response(fid, cached_data, fixture_map, sharp_set)
            all_parsed.extend(parsed)
            all_raw.append(cached_data)
            continue

        result = _api_get("/odds", params={"fixtureId": fid})
        if result["error"]:
            errors.append(f"fixture {fid}: {result['error']}")
            continue

        raw = result["data"]
        if raw is not None:
            _save_odds_cache(fid, raw)
            all_raw.append(raw)
            parsed = _parse_odds_response(fid, raw, fixture_map, sharp_set)
            all_parsed.extend(parsed)

    error = "; ".join(errors) if errors and not all_parsed else None
    return {"odds": all_parsed, "raw": all_raw, "error": error}


def fetch_odds_with_cache(sport: str) -> dict:
    """
    Fetch OddsPapi odds with cache fallback.

    Args:
        sport: Our sport key

    Returns:
        Same as fetch_odds + from_cache flag
    """
    result = fetch_odds(sport)

    if not result["error"]:
        # Success — save to cache
        try:
            cache_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sport": sport,
                "raw": result["raw"],
            }
            os.makedirs(os.path.dirname(ODDSPAPI_CACHE_JSON) or ".", exist_ok=True)
            with open(ODDSPAPI_CACHE_JSON, "w") as f:
                json.dump(cache_data, f)
        except Exception:
            pass
        result["from_cache"] = False
        return result

    # API failed — try cache
    if os.path.exists(ODDSPAPI_CACHE_JSON):
        try:
            with open(ODDSPAPI_CACHE_JSON, "r") as f:
                cache_data = json.load(f)

            from engine_config import CACHE_TTL_SECONDS
            cache_ts = datetime.fromisoformat(cache_data["timestamp"])
            age = (datetime.now(timezone.utc) - cache_ts).total_seconds()

            if age <= CACHE_TTL_SECONDS and cache_data.get("sport") == sport:
                # Re-parse from cached raw data
                return {
                    "odds": _parse_cached_odds(cache_data["raw"], sport),
                    "raw": cache_data["raw"],
                    "error": None,
                    "from_cache": True,
                }
        except Exception:
            pass

    result["from_cache"] = False
    return result


def _parse_cached_odds(raw_data, sport: str = None) -> list:
    """Re-parse odds from cached raw API responses (new per-fixture format)."""
    if not isinstance(raw_data, list):
        return []

    # Need fixture map for team names — fetch from cache
    fixture_map = {}
    if sport:
        fix_result = fetch_fixtures(sport)
        fixture_map = fix_result.get("fixture_map", {})

    sharp_set = set(b.lower() for b in ODDSPAPI_SHARP_BOOKS)
    all_parsed = []

    for item in raw_data:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("fixtureId", ""))
        if fid:
            all_parsed.extend(_parse_odds_response(fid, item, fixture_map, sharp_set))

    return all_parsed


def build_fixture_map(fixtures_result: dict) -> dict:
    """
    Build a fixture ID → team names/time map from fetch_fixtures result.
    Convenience wrapper around the fixture_map already returned by fetch_fixtures.
    """
    return fixtures_result.get("fixture_map", {})


def _decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American format."""
    if decimal_odds is None or decimal_odds <= 1.0:
        return 0
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    else:
        return round(-100 / (decimal_odds - 1))


def get_sports() -> dict:
    """List available sports on OddsPapi."""
    return _api_get("/sports")


def get_tournaments(sport_id: int) -> dict:
    """List tournaments for a given sport ID."""
    return _api_get("/tournaments", params={"sportId": sport_id})
