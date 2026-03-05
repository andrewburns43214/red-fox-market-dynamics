"""
ESPN Situational Data for Red Fox engine.

Pulls free data from ESPN's public scoreboard API:
  - Injuries (key players out/questionable)
  - Rest days (back-to-back detection)
  - Probable pitchers (MLB only)

No API key required. Cached for 30 minutes.
"""
import json
import os
import time
from datetime import datetime, timezone, timedelta

import requests

from engine_config import ESPN_SPORT_PATHS, ESPN_CACHE_TTL_SECONDS
from team_aliases import normalize_team_name

ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports"

# Disk cache directory
_ESPN_CACHE_DIR = os.path.join("data", "espn_cache")


def _espn_get(sport: str, endpoint: str = "scoreboard", params: dict = None) -> dict:
    """
    Fetch from ESPN API with disk caching.

    Returns:
        dict with "data" and "error" keys
    """
    sport_path = ESPN_SPORT_PATHS.get(sport.lower())
    if not sport_path:
        return {"data": None, "error": f"No ESPN path for sport: {sport}"}

    # Build cache key
    cache_key = f"{sport}_{endpoint}"
    if params:
        cache_key += "_" + "_".join(f"{k}={v}" for k, v in sorted(params.items()))
    cache_file = os.path.join(_ESPN_CACHE_DIR, f"{cache_key}.json")

    # Check disk cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)
            age = time.time() - cached.get("_cache_ts", 0)
            if age < ESPN_CACHE_TTL_SECONDS:
                return {"data": cached["data"], "error": None, "from_cache": True}
        except Exception:
            pass

    # Fetch from ESPN
    url = f"{ESPN_BASE_URL}/{sport_path}/{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return {"data": None, "error": f"ESPN HTTP {resp.status_code}"}

        data = resp.json()

        # Save to cache
        try:
            os.makedirs(_ESPN_CACHE_DIR, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump({"data": data, "_cache_ts": time.time()}, f)
        except Exception:
            pass

        return {"data": data, "error": None, "from_cache": False}

    except requests.exceptions.RequestException as e:
        return {"data": None, "error": f"ESPN request failed: {str(e)[:200]}"}


def fetch_injuries(sport: str) -> dict:
    """
    Fetch injury data from ESPN's dedicated injuries endpoint.

    Args:
        sport: Our sport key (nba, nhl, mlb, nfl, etc.)

    Returns:
        dict mapping normalized team name → list of injury dicts:
        {
            "team_norm": [
                {"player": "LeBron James", "status": "Out", "detail": "Left ankle"},
                ...
            ]
        }
    """
    result = _espn_get(sport, "injuries")
    if result["error"]:
        return {"injuries": {}, "error": result["error"]}

    data = result["data"]
    injuries = {}

    for team_data in data.get("injuries", []):
        team_name = team_data.get("displayName", "")
        team_norm = normalize_team_name(team_name)

        if not team_norm:
            continue

        team_injuries = []
        for inj in team_data.get("injuries", []):
            athlete = inj.get("athlete", {})
            player_name = athlete.get("displayName") or athlete.get("fullName", "")
            status = inj.get("status", "")
            detail = inj.get("type", {}).get("description", "")

            if not status:
                status = inj.get("description", "")

            if player_name:
                team_injuries.append({
                    "player": player_name,
                    "status": status,
                    "detail": detail,
                })

        if team_injuries:
            injuries[team_norm] = team_injuries

    return {"injuries": injuries, "error": None}


def fetch_rest_days(sport: str) -> dict:
    """
    Compute rest days per team from ESPN schedule.

    Checks yesterday's scoreboard to see who played.
    Rest = 0 means back-to-back (played yesterday).

    Args:
        sport: Our sport key

    Returns:
        dict mapping normalized team name → int rest days
        Teams not found played >1 day ago (default 1 = normal rest)
    """
    # Fetch yesterday's scoreboard
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y%m%d")
    result = _espn_get(sport, "scoreboard", params={"dates": yesterday})

    if result["error"]:
        return {"rest_days": {}, "error": result["error"]}

    data = result["data"]
    teams_played_yesterday = set()

    for event in data.get("events", []):
        for comp in event.get("competitions", []):
            status = comp.get("status", {}).get("type", {}).get("name", "")
            # Only count completed games
            if status not in ("STATUS_FINAL", "STATUS_FULL_TIME"):
                continue
            for team_info in comp.get("competitors", []):
                team_name = team_info.get("team", {}).get("displayName", "")
                team_norm = normalize_team_name(team_name)
                if team_norm:
                    teams_played_yesterday.add(team_norm)

    # Also check 2 days ago for "well rested" detection
    two_days_ago = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y%m%d")
    result2 = _espn_get(sport, "scoreboard", params={"dates": two_days_ago})

    teams_played_2days = set()
    if not result2.get("error"):
        data2 = result2["data"]
        for event in data2.get("events", []):
            for comp in event.get("competitions", []):
                status = comp.get("status", {}).get("type", {}).get("name", "")
                if status not in ("STATUS_FINAL", "STATUS_FULL_TIME"):
                    continue
                for team_info in comp.get("competitors", []):
                    team_name = team_info.get("team", {}).get("displayName", "")
                    team_norm = normalize_team_name(team_name)
                    if team_norm:
                        teams_played_2days.add(team_norm)

    rest_days = {}
    # Teams that played yesterday = back-to-back (0 days rest)
    for t in teams_played_yesterday:
        rest_days[t] = 0

    # Teams that played 2 days ago but NOT yesterday = 1 day rest (normal)
    for t in teams_played_2days:
        if t not in rest_days:
            rest_days[t] = 1

    # Teams not found in either = 2+ days rest (well-rested)
    # We don't explicitly set these; consumers default to 1 if not found

    return {"rest_days": rest_days, "error": None}


def fetch_probable_pitchers(sport: str = "mlb", date: str = None) -> dict:
    """
    Fetch probable pitchers from ESPN scoreboard (MLB only).

    Args:
        sport: Should be "mlb"
        date: Optional date string YYYYMMDD (default: today)

    Returns:
        dict mapping canonical_key → pitcher matchup info:
        {
            "canonical_key": {
                "away_pitcher": "Chris Sale",
                "home_pitcher": "Gerrit Cole",
                "away_hand": "L",
                "home_hand": "R",
            }
        }
    """
    if sport.lower() != "mlb":
        return {"pitchers": {}, "error": "Pitchers only available for MLB"}

    params = {}
    if date:
        params["dates"] = date

    result = _espn_get(sport, "scoreboard", params=params)
    if result["error"]:
        return {"pitchers": {}, "error": result["error"]}

    data = result["data"]
    pitchers = {}

    for event in data.get("events", []):
        for comp in event.get("competitions", []):
            home_team = ""
            away_team = ""
            home_pitcher = ""
            away_pitcher = ""
            home_hand = ""
            away_hand = ""
            home_era = None
            away_era = None
            home_wins = 0
            away_wins = 0
            home_losses = 0
            away_losses = 0

            for team_info in comp.get("competitors", []):
                team_name = team_info.get("team", {}).get("displayName", "")
                is_home = team_info.get("homeAway") == "home"

                if is_home:
                    home_team = team_name
                else:
                    away_team = team_name

                # Probable pitcher is in the probables list
                for probable in team_info.get("probables", []):
                    athlete = probable.get("athlete", {})
                    pitcher_name = athlete.get("displayName") or athlete.get("fullName", "")
                    # Try to get handedness
                    hand = ""
                    if athlete.get("hand", {}).get("abbreviation"):
                        hand = athlete["hand"]["abbreviation"]
                    elif "Left" in str(athlete.get("hand", {}).get("displayValue", "")):
                        hand = "L"
                    elif "Right" in str(athlete.get("hand", {}).get("displayValue", "")):
                        hand = "R"

                    # Extract ERA, W, L from statistics array
                    era = None
                    wins = 0
                    losses = 0
                    for stat in probable.get("statistics", []):
                        abbr = stat.get("abbreviation", "").upper()
                        val = stat.get("displayValue", "")
                        if abbr == "ERA":
                            try:
                                era = float(val)
                            except (ValueError, TypeError):
                                pass
                        elif abbr == "W":
                            try:
                                wins = int(val)
                            except (ValueError, TypeError):
                                pass
                        elif abbr == "L":
                            try:
                                losses = int(val)
                            except (ValueError, TypeError):
                                pass

                    if is_home:
                        home_pitcher = pitcher_name
                        home_hand = hand
                        home_era = era
                        home_wins = wins
                        home_losses = losses
                    else:
                        away_pitcher = pitcher_name
                        away_hand = hand
                        away_era = era
                        away_wins = wins
                        away_losses = losses

            # Build a matchup key if we have both teams
            if home_team and away_team:
                home_norm = normalize_team_name(home_team)
                away_norm = normalize_team_name(away_team)

                # Use normalized team names as a pseudo-key
                # The merge step will match this to canonical keys
                match_key = f"{away_norm} @ {home_norm}"

                pitchers[match_key] = {
                    "away_pitcher": away_pitcher,
                    "home_pitcher": home_pitcher,
                    "away_hand": away_hand,
                    "home_hand": home_hand,
                    "away_pitcher_era": away_era,
                    "away_pitcher_wins": away_wins,
                    "away_pitcher_losses": away_losses,
                    "home_pitcher_era": home_era,
                    "home_pitcher_wins": home_wins,
                    "home_pitcher_losses": home_losses,
                    "home_team_norm": home_norm,
                    "away_team_norm": away_norm,
                }

    return {"pitchers": pitchers, "error": None}


def get_situational_flags(sport: str, canonical_key: str,
                          home_team_norm: str, away_team_norm: str) -> dict:
    """
    Get all situational context for a specific game.

    Combines injuries, rest days, and pitchers (MLB) into a single dict.

    Args:
        sport: Our sport key
        canonical_key: Game's canonical key
        home_team_norm: Normalized home team name
        away_team_norm: Normalized away team name

    Returns:
        dict with all situational flags for this game
    """
    flags = {
        "home_injuries": [],
        "away_injuries": [],
        "home_rest_days": 1,  # default normal
        "away_rest_days": 1,
        "b2b_flag": "",
        "pitcher_matchup": None,
    }

    # Injuries
    inj_result = fetch_injuries(sport)
    if not inj_result.get("error"):
        injuries = inj_result["injuries"]
        flags["home_injuries"] = injuries.get(home_team_norm, [])
        flags["away_injuries"] = injuries.get(away_team_norm, [])

    # Rest days
    rest_result = fetch_rest_days(sport)
    if not rest_result.get("error"):
        rest = rest_result["rest_days"]
        flags["home_rest_days"] = rest.get(home_team_norm, 1)
        flags["away_rest_days"] = rest.get(away_team_norm, 1)

        # B2B detection
        home_b2b = flags["home_rest_days"] == 0
        away_b2b = flags["away_rest_days"] == 0
        if home_b2b and away_b2b:
            flags["b2b_flag"] = "BOTH_B2B"
        elif home_b2b:
            flags["b2b_flag"] = "HOME_B2B"
        elif away_b2b:
            flags["b2b_flag"] = "AWAY_B2B"

    # Pitchers (MLB only)
    if sport.lower() == "mlb":
        pitch_result = fetch_probable_pitchers()
        if not pitch_result.get("error"):
            # Try to match by team names
            match_key = f"{away_team_norm} @ {home_team_norm}"
            if match_key in pitch_result["pitchers"]:
                flags["pitcher_matchup"] = pitch_result["pitchers"][match_key]
            else:
                # Fuzzy search through all pitcher entries
                for pk, pv in pitch_result["pitchers"].items():
                    if (pv.get("home_team_norm") == home_team_norm and
                            pv.get("away_team_norm") == away_team_norm):
                        flags["pitcher_matchup"] = pv
                        break

    return flags


def fetch_all_situational(sport: str) -> dict:
    """
    Fetch all situational data for a sport in bulk.
    More efficient than calling get_situational_flags per game.

    Returns:
        dict with:
            "injuries": {team_norm: [...]},
            "rest_days": {team_norm: int},
            "pitchers": {match_key: {...}},  # MLB only
            "error": str or None
    """
    result = {
        "injuries": {},
        "rest_days": {},
        "pitchers": {},
        "error": None,
    }

    # Injuries
    inj = fetch_injuries(sport)
    if not inj.get("error"):
        result["injuries"] = inj["injuries"]

    # Rest days
    rest = fetch_rest_days(sport)
    if not rest.get("error"):
        result["rest_days"] = rest["rest_days"]

    # Use today's date for scoreboard-based fetches (ESPN defaults to yesterday)
    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")

    # Pitchers (MLB only)
    if sport.lower() == "mlb":
        pitch = fetch_probable_pitchers(date=today_str)
        if not pitch.get("error"):
            result["pitchers"] = pitch["pitchers"]

    # Goalies (NHL only)
    if sport.lower() == "nhl":
        goalies = fetch_probable_goalies(date=today_str)
        if not goalies.get("error"):
            result["goalies"] = goalies["goalies"]

    return result


def fetch_probable_goalies(sport: str = "nhl", date: str = None) -> dict:
    """Fetch probable starting goalies from ESPN NHL scoreboard."""
    if sport.lower() != "nhl":
        return {"goalies": {}, "error": "Goalies only for NHL"}

    params = {}
    if date:
        params["dates"] = date

    result = _espn_get(sport, "scoreboard", params=params)
    if result["error"]:
        return {"goalies": {}, "error": result["error"]}

    data = result["data"]
    goalies = {}

    for event in data.get("events", []):
        for comp in event.get("competitions", []):
            home_team = ""
            away_team = ""
            home_goalie = ""
            away_goalie = ""
            home_goalie_status = ""
            away_goalie_status = ""
            home_goalie_record = ""
            away_goalie_record = ""

            for team_info in comp.get("competitors", []):
                team_name = team_info.get("team", {}).get("displayName", "")
                is_home = team_info.get("homeAway") == "home"

                if is_home:
                    home_team = team_name
                else:
                    away_team = team_name

                for probable in team_info.get("probables", []):
                    athlete = probable.get("athlete", {})
                    goalie_name = athlete.get("displayName") or athlete.get("fullName", "")
                    status_info = probable.get("status", {})
                    status = status_info.get("name", "") if isinstance(status_info, dict) else ""
                    record = probable.get("record", "") or ""

                    if is_home:
                        home_goalie = goalie_name
                        home_goalie_status = status
                        home_goalie_record = record
                    else:
                        away_goalie = goalie_name
                        away_goalie_status = status
                        away_goalie_record = record

            if home_team and away_team:
                home_norm = normalize_team_name(home_team)
                away_norm = normalize_team_name(away_team)
                match_key = f"{away_norm} @ {home_norm}"

                goalies[match_key] = {
                    "home_goalie": home_goalie,
                    "away_goalie": away_goalie,
                    "home_goalie_status": home_goalie_status,
                    "away_goalie_status": away_goalie_status,
                    "home_goalie_record": home_goalie_record,
                    "away_goalie_record": away_goalie_record,
                    "home_team_norm": home_norm,
                    "away_team_norm": away_norm,
                }

    return {"goalies": goalies, "error": None}


def fetch_ncaab_rankings() -> dict:
    """Fetch NCAAB rankings (AP/NET) from ESPN."""
    result = _espn_get("ncaab", "rankings")
    if result["error"]:
        return {"rankings": {}, "error": result["error"]}

    data = result["data"]
    rankings = {}

    for ranking_group in data.get("rankings", []):
        rank_name = ranking_group.get("name", "")
        # Prefer AP Top 25 or NET Rankings
        if "AP" not in rank_name and "NET" not in rank_name and "Poll" not in rank_name:
            continue

        for rank_entry in ranking_group.get("ranks", []):
            rank_num = rank_entry.get("current", 0)
            team_info = rank_entry.get("team", {})
            team_name = team_info.get("displayName", "") or team_info.get("name", "")
            if team_name:
                team_norm = normalize_team_name(team_name)
                if team_norm and team_norm not in rankings:
                    rankings[team_norm] = rank_num

        if rankings:
            break  # Use first valid ranking set

    return {"rankings": rankings, "error": None}
