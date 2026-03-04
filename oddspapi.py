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


# OddsPapi sport IDs (from /sports endpoint)
ODDSPAPI_SPORT_MAP = {
    "nba": 2,       # Basketball
    "ncaab": 2,     # Basketball (same sport, different tournament)
    "nhl": 4,       # Ice Hockey
    "mlb": 3,       # Baseball
    "nfl": 1,       # American Football
    "ncaaf": 1,     # American Football (same sport, different tournament)
    "ufc": 9,       # MMA
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

    headers = {"X-Api-Key": api_key}
    url = f"{ODDSPAPI_BASE_URL}{endpoint}"

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)

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


def fetch_odds(sport: str, fixture_ids: list = None) -> dict:
    """
    Fetch sharp book odds from OddsPapi.

    Gets odds from all 6 sharp books for the given sport/tournament.
    Returns flat list of outcome dicts ready for L1 processing.

    Args:
        sport: Our sport key
        fixture_ids: Optional list of specific fixture IDs to filter

    Returns:
        dict with:
            "odds": list of parsed outcome dicts
            "raw": raw API response (for caching)
            "error": str or None
    """
    sport_lower = sport.lower()

    tid_result = discover_tournament_id(sport_lower)
    if tid_result["error"]:
        return {"odds": [], "raw": None, "error": tid_result["error"]}

    tournament_id = tid_result["tournament_id"]

    # Request odds for sharp books
    params = {
        "tournamentId": tournament_id,
        "oddType": "prematch",
    }

    result = _api_get("/odds-by-tournaments", params=params)
    if result["error"]:
        return {"odds": [], "raw": None, "error": result["error"]}

    raw_data = result["data"]
    if not isinstance(raw_data, list):
        return {"odds": [], "raw": raw_data, "error": "Unexpected odds response format"}

    # Parse odds into flat structure
    parsed = []
    sharp_set = set(b.lower() for b in ODDSPAPI_SHARP_BOOKS)

    for fixture in raw_data:
        fid = str(fixture.get("fixtureId") or fixture.get("id") or "")
        if not fid:
            continue

        # Filter to specific fixtures if requested
        if fixture_ids and fid not in [str(x) for x in fixture_ids]:
            continue

        for odds_entry in fixture.get("odds", []):
            bookmaker = (odds_entry.get("bookmakerName") or
                        odds_entry.get("bookmaker") or "").lower()

            # Filter to sharp books only
            if bookmaker not in sharp_set:
                continue

            market_id = odds_entry.get("marketTypeId") or odds_entry.get("marketId")
            market_name = MARKET_MAP.get(market_id)
            if not market_name:
                continue

            for outcome in odds_entry.get("outcomes", []):
                name = outcome.get("name") or outcome.get("outcomeName") or ""
                line = outcome.get("point") or outcome.get("line") or outcome.get("handicap")
                price = outcome.get("price") or outcome.get("odds")
                changed_at = outcome.get("changedAt") or outcome.get("updatedAt") or ""
                limit = outcome.get("limit") or outcome.get("maxBet")

                if price is None:
                    continue

                # Convert decimal odds to American
                try:
                    price_float = float(price)
                    odds_american = _decimal_to_american(price_float)
                except (ValueError, TypeError):
                    odds_american = 0

                # Determine side label
                if market_name == "TOTAL":
                    side = name  # "Over" / "Under"
                else:
                    side = name  # team name

                parsed.append({
                    "fixture_id": fid,
                    "bookmaker": bookmaker,
                    "market": market_name,
                    "side": side,
                    "line": line,
                    "odds_decimal": price,
                    "odds_american": odds_american,
                    "changed_at": changed_at,
                    "limit": limit,
                })

    return {"odds": parsed, "raw": raw_data, "error": None}


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
                    "odds": _parse_cached_odds(cache_data["raw"]),
                    "raw": cache_data["raw"],
                    "error": None,
                    "from_cache": True,
                }
        except Exception:
            pass

    result["from_cache"] = False
    return result


def _parse_cached_odds(raw_data: list) -> list:
    """Re-parse odds from cached raw API response."""
    parsed = []
    sharp_set = set(b.lower() for b in ODDSPAPI_SHARP_BOOKS)

    if not isinstance(raw_data, list):
        return parsed

    for fixture in raw_data:
        fid = str(fixture.get("fixtureId") or fixture.get("id") or "")
        if not fid:
            continue

        for odds_entry in fixture.get("odds", []):
            bookmaker = (odds_entry.get("bookmakerName") or
                        odds_entry.get("bookmaker") or "").lower()
            if bookmaker not in sharp_set:
                continue

            market_id = odds_entry.get("marketTypeId") or odds_entry.get("marketId")
            market_name = MARKET_MAP.get(market_id)
            if not market_name:
                continue

            for outcome in odds_entry.get("outcomes", []):
                name = outcome.get("name") or outcome.get("outcomeName") or ""
                line = outcome.get("point") or outcome.get("line") or outcome.get("handicap")
                price = outcome.get("price") or outcome.get("odds")
                changed_at = outcome.get("changedAt") or outcome.get("updatedAt") or ""
                limit = outcome.get("limit") or outcome.get("maxBet")

                if price is None:
                    continue

                try:
                    odds_american = _decimal_to_american(float(price))
                except (ValueError, TypeError):
                    odds_american = 0

                side = name

                parsed.append({
                    "fixture_id": fid,
                    "bookmaker": bookmaker,
                    "market": market_name,
                    "side": side,
                    "line": line,
                    "odds_decimal": price,
                    "odds_american": odds_american,
                    "changed_at": changed_at,
                    "limit": limit,
                })

    return parsed


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
