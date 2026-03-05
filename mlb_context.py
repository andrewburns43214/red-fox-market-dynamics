"""
MLB-specific context for Red Fox engine.

Provides:
- Starting pitcher quality score (ERA-based with sample gates)
- Park factors (static lookup, affects totals)
- Bullpen fatigue detection
"""

MLB_LEAGUE_AVG_ERA = 4.30  # 2025 MLB league average
MLB_MIN_DECISIONS = 3      # W+L must be >= 3 for ERA to be trusted

# Park factors: >1.0 = hitter friendly (more runs), <1.0 = pitcher friendly
# Source: historical 3-year averages
PARK_FACTORS = {
    "colorado rockies": 1.28,    # Coors Field — extreme altitude
    "boston red sox": 1.10,       # Fenway Park — Green Monster
    "chi cubs": 1.08,            # Wrigley Field — wind variable
    "cincinnati reds": 1.07,     # Great American Ball Park
    "ny yankees": 1.05,          # Yankee Stadium — short porch
    "texas rangers": 1.04,       # Globe Life Field
    "philadelphia phillies": 1.03,
    "atlanta braves": 1.02,
    "toronto blue jays": 1.02,
    "minnesota twins": 1.01,
    "detroit tigers": 1.00,
    "chi white sox": 1.00,
    "kansas city royals": 0.99,
    "baltimore orioles": 0.99,
    "la angels": 0.98,
    "cleveland guardians": 0.98,
    "milwaukee brewers": 0.97,
    "houston astros": 0.97,
    "pittsburgh pirates": 0.96,
    "st louis cardinals": 0.96,
    "washington nationals": 0.96,
    "ny mets": 0.95,
    "tampa bay rays": 0.95,
    "san francisco giants": 0.94,
    "la dodgers": 0.93,
    "miami marlins": 0.92,
    "seattle mariners": 0.92,
    "san diego padres": 0.91,
    "oakland athletics": 0.91,
    "arizona diamondbacks": 0.98,
}


def sp_quality_score(era: float, wins: int, losses: int) -> tuple:
    """Compute starting pitcher quality from ERA with sample gates.

    Returns (score: float, flag: str)
        score: -3.0 to +3.0
        flag: 'SP_ACE', 'SP_STRONG', 'SP_AVG', 'SP_WEAK', 'SP_BAD', 'SP_UNKNOWN'
    """
    decisions = (wins or 0) + (losses or 0)

    # Sample gate: not enough data
    if era is None or decisions < MLB_MIN_DECISIONS:
        return -1.0, "SP_UNKNOWN"

    # ERA-based quality (continuous mapping)
    if era < 2.50:
        return 3.0, "SP_ACE"
    elif era < 3.00:
        score = 2.0 + (3.00 - era) / 0.50  # 2.0 to 3.0
        return round(score, 1), "SP_ACE"
    elif era < 3.50:
        score = 1.0 + (3.50 - era) / 0.50  # 1.0 to 2.0
        return round(score, 1), "SP_STRONG"
    elif era < 3.75:
        score = 0.5 + (3.75 - era) / 0.50  # 0.5 to 1.0
        return round(score, 1), "SP_STRONG"
    elif era < 4.25:
        return 0.0, "SP_AVG"
    elif era < 4.75:
        score = -0.5 - (era - 4.25) / 0.50  # -0.5 to -1.0
        return round(score, 1), "SP_WEAK"
    elif era < 5.50:
        score = -1.5 - (era - 4.75) / 0.75  # -1.5 to -2.5
        return round(score, 1), "SP_WEAK"
    else:
        return -3.0, "SP_BAD"


def get_park_factor(home_team_norm: str) -> float:
    """Get park factor for home team. Returns 1.0 if unknown."""
    return PARK_FACTORS.get(home_team_norm.lower().strip(), 1.0)


def park_adjustment(park_factor: float, market: str) -> float:
    """Score adjustment based on park factor. Only affects TOTAL market confidence."""
    if market.upper() != "TOTAL":
        return 0.0
    if park_factor >= 1.10:
        return 1.5   # Strong hitter park — overs historically more likely
    elif park_factor >= 1.05:
        return 0.5
    elif park_factor <= 0.92:
        return -1.0  # Strong pitcher park — unders more likely
    elif park_factor <= 0.95:
        return -0.5
    return 0.0


def get_mlb_context(row: dict) -> dict:
    """Compute all MLB context for a game row.

    Expects pitcher_matchup dict in row (from espn_situational merge).
    Returns dict with sp_*, park_*, bullpen_* fields.
    """
    result = {
        "sp_era_home": None,
        "sp_era_away": None,
        "sp_quality_home": 0.0,
        "sp_quality_away": 0.0,
        "sp_flag_home": "",
        "sp_flag_away": "",
        "sp_hand_home": "",
        "sp_hand_away": "",
        "sp_name_home": "",
        "sp_name_away": "",
        "sp_quality_diff": 0.0,   # positive = our pitcher better
        "park_factor": 1.0,
        "park_adj": 0.0,
        "mlb_adj": 0.0,
    }

    # Pitcher matchup data
    pm = row.get("pitcher_matchup")
    if pm and isinstance(pm, dict):
        # Home pitcher
        h_era = pm.get("home_pitcher_era")
        h_w = pm.get("home_pitcher_wins", 0) or 0
        h_l = pm.get("home_pitcher_losses", 0) or 0
        h_score, h_flag = sp_quality_score(h_era, h_w, h_l)
        result["sp_era_home"] = h_era
        result["sp_quality_home"] = h_score
        result["sp_flag_home"] = h_flag
        result["sp_hand_home"] = pm.get("home_hand", "")
        result["sp_name_home"] = pm.get("home_pitcher", "")

        # Away pitcher
        a_era = pm.get("away_pitcher_era")
        a_w = pm.get("away_pitcher_wins", 0) or 0
        a_l = pm.get("away_pitcher_losses", 0) or 0
        a_score, a_flag = sp_quality_score(a_era, a_w, a_l)
        result["sp_era_away"] = a_era
        result["sp_quality_away"] = a_score
        result["sp_flag_away"] = a_flag
        result["sp_hand_away"] = pm.get("away_hand", "")
        result["sp_name_away"] = pm.get("away_pitcher", "")

        # Quality differential (from perspective of side we're evaluating)
        result["sp_quality_diff"] = round(h_score - a_score, 1)

    # Park factor
    home_norm = str(row.get("home_team_norm", "")).lower()
    pf = get_park_factor(home_norm)
    result["park_factor"] = pf
    mkt = str(row.get("market_display", ""))
    result["park_adj"] = park_adjustment(pf, mkt)

    # Combined MLB adjustment
    # For ML/SPREAD: use pitcher quality diff relative to side
    # For TOTAL: use park factor adjustment
    side = str(row.get("side", "")).lower()
    home_norm_lower = str(row.get("home_team_norm", "")).lower()
    away_norm_lower = str(row.get("away_team_norm", "")).lower()

    sp_adj = 0.0
    if mkt.upper() in ("MONEYLINE", "SPREAD"):
        # Determine which pitcher is "ours"
        if home_norm_lower and home_norm_lower in side:
            sp_adj = result["sp_quality_home"] - result["sp_quality_away"]
        elif away_norm_lower and away_norm_lower in side:
            sp_adj = result["sp_quality_away"] - result["sp_quality_home"]
        sp_adj = round(max(-3.0, min(3.0, sp_adj * 0.5)), 1)  # Scale down
    elif mkt.upper() == "TOTAL":
        # For totals: both pitchers matter — average quality affects confidence
        avg_q = (result["sp_quality_home"] + result["sp_quality_away"]) / 2
        sp_adj = round(avg_q * 0.3, 1)  # Mild adjustment

    result["mlb_adj"] = round(sp_adj + result["park_adj"], 1)

    return result
