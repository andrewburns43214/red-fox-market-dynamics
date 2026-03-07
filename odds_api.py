"""
The-Odds-API wrapper for Red Fox engine.
Handles authentication, rate limiting, retries, and data normalization.

API docs: https://the-odds-api.com/liveapi/guides/v4/

One API call per sport returns both sharp book (L1) and consensus (L2) data.
"""
import json
import os
import time
from datetime import datetime, timezone

import requests

from engine_config import (
    ODDS_API_KEY,
    ODDS_API_BASE_URL,
    API_SPORT_MAP,
    L1_SHARP_BOOKS,
    L2_CONSENSUS_REGIONS,
    L1_CACHE_JSON,
    L2_CACHE_JSON,
    CACHE_TTL_SECONDS,
)


def _get_api_key() -> str:
    """Get API key from config or environment (env takes priority)."""
    return os.environ.get("ODDS_API_KEY", "") or ODDS_API_KEY


def _american_odds(decimal_odds: float) -> int:
    """Convert decimal odds to American format."""
    if decimal_odds is None or decimal_odds <= 1.0:
        return 0
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    else:
        return round(-100 / (decimal_odds - 1))


def fetch_odds(sport: str, markets: list = None, regions: list = None,
               bookmakers: list = None) -> dict:
    """
    Fetch odds from The-Odds-API for a single sport.

    Args:
        sport: Our sport key (nba, nfl, etc.)
        markets: List of markets to fetch (default: spreads, totals, h2h)
        regions: List of regions (default: us)
        bookmakers: List of specific bookmakers (optional, overrides regions)

    Returns:
        dict with keys:
            "events": list of event dicts from API
            "remaining_requests": int (API quota remaining)
            "used_requests": int (API quota used)
            "error": str or None
            "from_cache": bool
    """
    api_key = _get_api_key()
    if not api_key:
        return {
            "events": [],
            "remaining_requests": None,
            "used_requests": None,
            "error": "ODDS_API_KEY not set",
            "from_cache": False,
        }

    api_sport = API_SPORT_MAP.get(sport.lower())
    if not api_sport:
        return {
            "events": [],
            "remaining_requests": None,
            "used_requests": None,
            "error": f"Unknown sport: {sport}",
            "from_cache": False,
        }

    if markets is None:
        markets = ["spreads", "totals", "h2h"]
    if regions is None:
        regions = L2_CONSENSUS_REGIONS

    params = {
        "apiKey": api_key,
        "markets": ",".join(markets),
        "oddsFormat": "decimal",
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)
    else:
        params["regions"] = ",".join(regions)

    url = f"{ODDS_API_BASE_URL}/sports/{api_sport}/odds/"

    try:
        resp = requests.get(url, params=params, timeout=30)

        remaining = resp.headers.get("x-requests-remaining")
        used = resp.headers.get("x-requests-used")

        if resp.status_code == 401:
            return {
                "events": [],
                "remaining_requests": remaining,
                "used_requests": used,
                "error": "Invalid API key",
                "from_cache": False,
            }

        if resp.status_code == 429:
            return {
                "events": [],
                "remaining_requests": 0,
                "used_requests": used,
                "error": "Rate limit exceeded",
                "from_cache": False,
            }

        if resp.status_code != 200:
            return {
                "events": [],
                "remaining_requests": remaining,
                "used_requests": used,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                "from_cache": False,
            }

        events = resp.json()
        return {
            "events": events,
            "remaining_requests": remaining,
            "used_requests": used,
            "error": None,
            "from_cache": False,
        }

    except requests.exceptions.Timeout:
        return {
            "events": [],
            "remaining_requests": None,
            "used_requests": None,
            "error": "Request timed out",
            "from_cache": False,
        }
    except requests.exceptions.RequestException as e:
        return {
            "events": [],
            "remaining_requests": None,
            "used_requests": None,
            "error": f"Request failed: {str(e)[:200]}",
            "from_cache": False,
        }


def fetch_odds_with_cache(sport: str, cache_path: str = None,
                          markets: list = None) -> dict:
    """
    Fetch odds with cache fallback. If API fails and cache is fresh, use cache.

    Args:
        sport: Our sport key
        cache_path: Path to cache JSON file
        markets: Markets to fetch

    Returns:
        Same dict as fetch_odds, with from_cache=True if using cached data.
    """
    result = fetch_odds(sport, markets=markets)

    if not result["error"]:
        # Success — save to cache
        if cache_path:
            try:
                cache_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "sport": sport,
                    "events": result["events"],
                }
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(cache_data, f)
            except Exception:
                pass
        return result

    # API failed — try cache
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cache_data = json.load(f)

            cache_ts = datetime.fromisoformat(cache_data["timestamp"])
            age_seconds = (datetime.now(timezone.utc) - cache_ts).total_seconds()

            if age_seconds <= CACHE_TTL_SECONDS:
                return {
                    "events": cache_data["events"],
                    "remaining_requests": result["remaining_requests"],
                    "used_requests": result["used_requests"],
                    "error": None,
                    "from_cache": True,
                }
        except Exception:
            pass

    return result


def parse_event_odds(event: dict) -> list:
    """
    Parse a single API event into flat row dicts for L1/L2 processing.

    Args:
        event: Single event dict from API response

    Returns:
        List of dicts, each representing one bookmaker+market+outcome:
        {
            "event_id": str,
            "commence_time": str (ISO UTC),
            "home_team": str,
            "away_team": str,
            "bookmaker": str,
            "market": str (SPREAD/TOTAL/MONEYLINE),
            "side": str,
            "line": float or None,
            "odds_decimal": float,
            "odds_american": int,
        }
    """
    rows = []
    event_id = event.get("id", "")
    commence = event.get("commence_time", "")
    home = event.get("home_team", "")
    away = event.get("away_team", "")

    for bm in event.get("bookmakers", []):
        bm_key = bm.get("key", "")
        bm_last_update = bm.get("last_update", "")

        for mkt in bm.get("markets", []):
            mkt_key = mkt.get("key", "")
            mkt_last_update = mkt.get("last_update", "") or bm_last_update

            # Map API market keys to our format
            if mkt_key == "spreads":
                market_name = "SPREAD"
            elif mkt_key == "totals":
                market_name = "TOTAL"
            elif mkt_key == "h2h":
                market_name = "MONEYLINE"
            else:
                continue

            for outcome in mkt.get("outcomes", []):
                name = outcome.get("name", "")
                point = outcome.get("point")
                price = outcome.get("price")

                if price is None:
                    continue

                # Build side label
                if market_name == "TOTAL":
                    side = name  # "Over" or "Under"
                elif market_name == "MONEYLINE":
                    side = name  # team name
                else:
                    side = name  # team name (spread)

                rows.append({
                    "event_id": event_id,
                    "commence_time": commence,
                    "home_team": home,
                    "away_team": away,
                    "bookmaker": bm_key,
                    "market": market_name,
                    "side": side,
                    "line": point,
                    "odds_decimal": price,
                    "odds_american": _american_odds(price),
                    "last_update": mkt_last_update,
                })

    return rows


def get_quota() -> dict:
    """Check current API quota usage without consuming a request."""
    api_key = _get_api_key()
    if not api_key:
        return {"error": "ODDS_API_KEY not set"}

    try:
        # Use a lightweight endpoint
        resp = requests.get(
            f"{ODDS_API_BASE_URL}/sports/",
            params={"apiKey": api_key},
            timeout=15,
        )
        return {
            "remaining": resp.headers.get("x-requests-remaining"),
            "used": resp.headers.get("x-requests-used"),
            "error": None if resp.status_code == 200 else f"HTTP {resp.status_code}",
        }
    except Exception as e:
        return {"error": str(e)[:200]}
