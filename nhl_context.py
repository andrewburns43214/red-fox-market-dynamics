"""
NHL-specific context for Red Fox engine.

Provides:
- Goalie confirmation status (confirmed starter vs backup vs unknown)
- Starter/backup detection from W-L-OT record
"""

# Threshold: if total games (W+L+OT) > this, likely a starter
STARTER_GAMES_THRESHOLD = 20


def _parse_goalie_record(record: str) -> tuple:
    """Parse W-L-OT record string like '25-10-5'. Returns (wins, losses, ot)."""
    try:
        parts = str(record).strip().split("-")
        if len(parts) >= 3:
            return int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]), int(parts[1]), 0
    except (ValueError, TypeError):
        pass
    return 0, 0, 0


def is_likely_starter(record: str) -> bool:
    """Infer if goalie is the team's starter based on W-L-OT record."""
    w, l, ot = _parse_goalie_record(record)
    return (w + l + ot) >= STARTER_GAMES_THRESHOLD


def goalie_scoring(goalie_name: str, status: str, record: str) -> tuple:
    """Compute goalie context score and flag.

    Returns (adj: float, flag: str)
        adj: -2.0 to +1.0
        flag: 'G_STARTER_CONFIRMED', 'G_BACKUP_CONFIRMED', 'G_CONFIRMED',
              'G_PROBABLE', 'G_UNKNOWN'
    """
    if not goalie_name:
        return 0.0, "G_UNKNOWN"

    starter = is_likely_starter(record)
    status_lower = str(status).lower().strip()

    if status_lower == "confirmed":
        if starter:
            return 1.0, "G_STARTER_CONFIRMED"
        else:
            # Backup confirmed — line may not have fully adjusted
            return -2.0, "G_BACKUP_CONFIRMED"
    elif status_lower == "probable":
        if starter:
            return 0.5, "G_PROBABLE"
        else:
            return -1.0, "G_PROBABLE"
    else:
        return 0.0, "G_UNKNOWN"


def get_nhl_context(row: dict) -> dict:
    """Compute NHL context for a game row.

    Expects goalie_matchup dict in row (from espn_situational merge).
    Returns dict with goalie_* fields.
    """
    result = {
        "goalie_home": "",
        "goalie_away": "",
        "goalie_home_status": "",
        "goalie_away_status": "",
        "goalie_home_record": "",
        "goalie_away_record": "",
        "goalie_flag_home": "",
        "goalie_flag_away": "",
        "goalie_adj": 0.0,
        "nhl_adj": 0.0,
    }

    gm = row.get("goalie_matchup")
    if not gm or not isinstance(gm, dict):
        return result

    # Home goalie
    h_name = gm.get("home_goalie", "")
    h_status = gm.get("home_goalie_status", "")
    h_record = gm.get("home_goalie_record", "")
    h_adj, h_flag = goalie_scoring(h_name, h_status, h_record)

    result["goalie_home"] = h_name
    result["goalie_home_status"] = h_status
    result["goalie_home_record"] = h_record
    result["goalie_flag_home"] = h_flag

    # Away goalie
    a_name = gm.get("away_goalie", "")
    a_status = gm.get("away_goalie_status", "")
    a_record = gm.get("away_goalie_record", "")
    a_adj, a_flag = goalie_scoring(a_name, a_status, a_record)

    result["goalie_away"] = a_name
    result["goalie_away_status"] = a_status
    result["goalie_away_record"] = a_record
    result["goalie_flag_away"] = a_flag

    # Determine which goalie is "ours" based on side
    side = str(row.get("side", "")).lower()
    home_norm = str(row.get("home_team_norm", "")).lower()
    away_norm = str(row.get("away_team_norm", "")).lower()

    our_adj = 0.0
    opp_adj = 0.0
    if home_norm and home_norm in side:
        our_adj = h_adj
        opp_adj = a_adj
    elif away_norm and away_norm in side:
        our_adj = a_adj
        opp_adj = h_adj

    # Our goalie's quality helps us; opponent's backup helps us too
    goalie_adj = our_adj
    if opp_adj < 0:
        goalie_adj += abs(opp_adj) * 0.5  # Opponent backup = slight edge for us

    result["goalie_adj"] = round(goalie_adj, 1)
    result["nhl_adj"] = round(goalie_adj, 1)

    return result
