"""
Weather data for Red Fox engine — outdoor sport scoring adjustments.

Uses Open-Meteo API (completely free, no API key required).
Fetches weather forecast for game venues based on home team location.

Affects: NFL, NCAAF, MLB (outdoor stadiums only)
Skips: NBA, NHL, UFC (indoor)
"""
import json
import os
import time
from datetime import datetime, timezone

import requests

from team_aliases import normalize_team_name

# 30-minute cache (matches ESPN cache)
WEATHER_CACHE_TTL = 1800
_WEATHER_CACHE_DIR = os.path.join("data", "weather_cache")

# Open-Meteo free forecast API
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# ─── STADIUM DATABASE ───
# Maps normalized home team name → (lat, lon, is_dome)
# Dome stadiums get no weather adjustment (controlled environment)

STADIUMS = {
    # NFL
    "arizona cardinals":     (33.5276, -112.2626, True),   # State Farm Stadium (retractable roof)
    "atlanta falcons":       (33.7554, -84.4010, True),    # Mercedes-Benz Stadium
    "baltimore ravens":      (39.2780, -76.6227, False),   # M&T Bank Stadium
    "buffalo bills":         (42.7738, -78.7870, False),   # Highmark Stadium
    "carolina panthers":     (35.2258, -80.8528, False),   # Bank of America Stadium
    "chicago bears":         (41.8623, -87.6167, False),   # Soldier Field
    "cincinnati bengals":    (39.0954, -84.5160, False),   # Paycor Stadium
    "cleveland browns":      (41.5061, -81.6995, False),   # Cleveland Browns Stadium
    "dallas cowboys":        (32.7473, -97.0945, True),    # AT&T Stadium
    "denver broncos":        (39.7439, -105.0201, False),  # Empower Field
    "detroit lions":         (42.3400, -83.0456, True),    # Ford Field
    "green bay packers":     (44.5013, -88.0622, False),   # Lambeau Field
    "houston texans":        (29.6847, -95.4107, True),    # NRG Stadium (retractable roof)
    "indianapolis colts":    (39.7601, -86.1639, True),    # Lucas Oil Stadium
    "jacksonville jaguars":  (30.3239, -81.6373, False),   # EverBank Stadium
    "kansas city chiefs":    (39.0489, -94.4839, False),   # Arrowhead Stadium
    "las vegas raiders":     (36.0907, -115.1833, True),   # Allegiant Stadium
    "la chargers":           (33.9534, -118.3387, True),   # SoFi Stadium
    "la rams":               (33.9534, -118.3387, True),   # SoFi Stadium
    "miami dolphins":        (25.9580, -80.2389, False),   # Hard Rock Stadium
    "minnesota vikings":     (44.9736, -93.2575, True),    # U.S. Bank Stadium
    "new england patriots":  (42.0909, -71.2643, False),   # Gillette Stadium
    "new orleans saints":    (29.9511, -90.0812, True),    # Caesars Superdome
    "ny giants":             (40.8128, -74.0742, False),   # MetLife Stadium
    "ny jets":               (40.8128, -74.0742, False),   # MetLife Stadium
    "oakland raiders":       (36.0907, -115.1833, True),   # (alias for LV Raiders)
    "philadelphia eagles":   (39.9008, -75.1675, False),   # Lincoln Financial Field
    "pittsburgh steelers":   (40.4468, -80.0158, False),   # Acrisure Stadium
    "san francisco 49ers":   (37.4033, -121.9694, False),  # Levi's Stadium
    "seattle seahawks":      (47.5952, -122.3316, False),  # Lumen Field (open air)
    "tampa bay buccaneers":  (27.9759, -82.5033, False),   # Raymond James Stadium
    "tennessee titans":      (36.1665, -86.7713, False),   # Nissan Stadium
    "washington commanders": (38.9076, -76.8645, False),   # Northwest Stadium

    # MLB
    "arizona diamondbacks":  (33.4453, -112.0667, True),   # Chase Field (retractable roof)
    "atlanta braves":        (33.8908, -84.4678, False),   # Truist Park
    "baltimore orioles":     (39.2839, -76.6216, False),   # Camden Yards
    "boston red sox":         (42.3467, -71.0972, False),   # Fenway Park
    "chi cubs":              (41.9484, -87.6553, False),   # Wrigley Field
    "chi white sox":         (41.8299, -87.6338, False),   # Guaranteed Rate Field
    "cincinnati reds":       (39.0974, -84.5082, False),   # Great American Ball Park
    "cleveland guardians":   (41.4962, -81.6852, False),   # Progressive Field
    "colorado rockies":      (39.7559, -104.9942, False),  # Coors Field
    "detroit tigers":        (42.3390, -83.0485, False),   # Comerica Park
    "houston astros":        (29.7573, -95.3555, True),    # Minute Maid Park (retractable roof)
    "kansas city royals":    (39.0517, -94.4803, False),   # Kauffman Stadium
    "la angels":             (33.8003, -117.8827, False),  # Angel Stadium
    "la dodgers":            (34.0739, -118.2400, False),  # Dodger Stadium
    "miami marlins":         (25.7781, -80.2197, True),    # LoanDepot Park (retractable roof)
    "milwaukee brewers":     (43.0280, -87.9712, True),    # American Family Field (retractable roof)
    "minnesota twins":       (44.9817, -93.2778, False),   # Target Field
    "ny mets":               (40.7571, -73.8458, False),   # Citi Field
    "ny yankees":            (40.8296, -73.9262, False),   # Yankee Stadium
    "oakland athletics":     (37.7516, -122.2005, False),  # Oakland Coliseum
    "philadelphia phillies": (39.9061, -75.1665, False),   # Citizens Bank Park
    "pittsburgh pirates":    (40.4469, -80.0057, False),   # PNC Park
    "san diego padres":      (32.7076, -117.1570, False),  # Petco Park
    "san francisco giants":  (37.7786, -122.3893, False),  # Oracle Park
    "seattle mariners":      (47.5914, -122.3325, True),   # T-Mobile Park (retractable roof)
    "st louis cardinals":    (38.6226, -90.1928, False),   # Busch Stadium
    "tampa bay rays":        (27.7682, -82.6534, True),    # Tropicana Field (dome)
    "texas rangers":         (32.7512, -97.0832, True),    # Globe Life Field (retractable roof)
    "toronto blue jays":     (43.6414, -79.3894, True),    # Rogers Centre (retractable roof)
    "washington nationals":  (38.8730, -77.0075, False),   # Nationals Park
}


def _get_stadium(home_team_norm: str):
    """Look up stadium for a normalized home team name.
    Returns (lat, lon, is_dome) or None."""
    if not home_team_norm:
        return None
    key = home_team_norm.lower().strip()
    if key in STADIUMS:
        return STADIUMS[key]
    # Fuzzy: try partial match (e.g. "patriots" in "new england patriots")
    for name, coords in STADIUMS.items():
        if key in name or name in key:
            return coords
    return None


def fetch_game_weather(lat: float, lon: float, game_time_iso: str) -> dict:
    """Fetch weather forecast for a specific location and time.

    Returns dict with:
        wind_mph, temp_f, precip_prob, precip_mm, weather_code, is_outdoor
    """
    result = {
        "wind_mph": 0, "temp_f": 70, "precip_prob": 0,
        "precip_mm": 0.0, "weather_code": 0, "error": None,
    }

    try:
        game_dt = datetime.fromisoformat(game_time_iso.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        result["error"] = "invalid game time"
        return result

    date_str = game_dt.strftime("%Y-%m-%d")

    # Check cache
    cache_key = f"wx_{lat:.2f}_{lon:.2f}_{date_str}"
    cache_file = os.path.join(_WEATHER_CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)
            age = time.time() - cached.get("_cache_ts", 0)
            if age < WEATHER_CACHE_TTL:
                cached.pop("_cache_ts", None)
                return cached
        except Exception:
            pass

    # Fetch from Open-Meteo
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,wind_speed_10m,precipitation_probability,precipitation,weather_code",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "start_date": date_str,
            "end_date": date_str,
            "timezone": "auto",
        }
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=10)
        if resp.status_code != 200:
            result["error"] = f"Open-Meteo HTTP {resp.status_code}"
            return result

        data = resp.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        # Find the hour closest to game time
        game_hour = game_dt.hour
        best_idx = 0
        for i, t in enumerate(times):
            try:
                h = int(t.split("T")[1].split(":")[0])
                if abs(h - game_hour) < abs(int(times[best_idx].split("T")[1].split(":")[0]) - game_hour):
                    best_idx = i
            except (IndexError, ValueError):
                continue

        result["wind_mph"] = hourly.get("wind_speed_10m", [0])[best_idx]
        result["temp_f"] = hourly.get("temperature_2m", [70])[best_idx]
        result["precip_prob"] = hourly.get("precipitation_probability", [0])[best_idx]
        result["precip_mm"] = hourly.get("precipitation", [0.0])[best_idx]
        result["weather_code"] = hourly.get("weather_code", [0])[best_idx]

        # Save to cache
        try:
            os.makedirs(_WEATHER_CACHE_DIR, exist_ok=True)
            cache_data = {**result, "_cache_ts": time.time()}
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)
        except Exception:
            pass

    except requests.exceptions.RequestException as e:
        result["error"] = f"Weather fetch failed: {str(e)[:200]}"

    return result


def get_weather_for_game(sport: str, home_team_norm: str, game_time_iso: str) -> dict:
    """Get weather context for a game. Returns empty/neutral for indoor sports/venues.

    Returns:
        dict with wind_mph, temp_f, precip_prob, weather_flag, weather_adj
    """
    defaults = {
        "wind_mph": 0, "temp_f": 70, "precip_prob": 0,
        "weather_flag": "", "weather_adj": 0.0, "is_outdoor": False,
    }

    # Indoor sports — no weather impact
    if sport.lower() in ("nba", "nhl", "ufc", "ncaab"):
        return defaults

    # Look up stadium
    stadium = _get_stadium(home_team_norm)
    if not stadium:
        return defaults

    lat, lon, is_dome = stadium
    if is_dome:
        defaults["weather_flag"] = "DOME"
        return defaults

    defaults["is_outdoor"] = True

    # Fetch weather
    wx = fetch_game_weather(lat, lon, game_time_iso)
    if wx.get("error"):
        return defaults

    wind = wx.get("wind_mph", 0) or 0
    temp = wx.get("temp_f", 70) or 70
    precip_prob = wx.get("precip_prob", 0) or 0
    precip_mm = wx.get("precip_mm", 0) or 0

    defaults["wind_mph"] = wind
    defaults["temp_f"] = temp
    defaults["precip_prob"] = precip_prob

    # Weather flags and scoring adjustment
    flags = []
    adj = 0.0

    # HIGH WIND — affects totals (especially MLB), also NFL passing
    if wind >= 20:
        flags.append("HIGH_WIND")
        adj -= 1.5  # Strong signal: unders more likely, games less predictable
    elif wind >= 15:
        flags.append("WINDY")
        adj -= 0.5

    # PRECIPITATION — rain/snow affects footing, ball handling
    if precip_prob >= 70 or precip_mm >= 2.0:
        flags.append("RAIN_LIKELY")
        adj -= 1.0
    elif precip_prob >= 40:
        flags.append("RAIN_POSSIBLE")
        adj -= 0.5

    # EXTREME COLD (NFL/NCAAF) — affects passing game, kicking
    if sport.lower() in ("nfl", "ncaaf") and temp <= 20:
        flags.append("EXTREME_COLD")
        adj -= 1.0
    elif sport.lower() in ("nfl", "ncaaf") and temp <= 32:
        flags.append("COLD")
        adj -= 0.5

    defaults["weather_flag"] = "|".join(flags) if flags else ""
    defaults["weather_adj"] = round(adj, 1)

    return defaults
