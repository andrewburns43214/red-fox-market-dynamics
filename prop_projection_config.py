"""Configuration for the isolated Red Fox player-prop projection subsystem.

No game-line, Red Fox read, ranking, split, or favorite field is accepted here.
The values below are versioned model assumptions, not sportsbook inputs.
"""

from __future__ import annotations


API_BASE = "https://api.prop-line.com/v1"
LOCAL_DAILY_REQUEST_CAP = 190  # leaves headroom under PropLine's 1,000/day free tier
REQUEST_TIMEOUT_SECONDS = 18
TRADITIONAL_BOOKS = {
    "betmgm", "betrivers", "betonlineag", "bovada", "caesars", "circasports",
    "draftkings", "espnbet", "fanduel", "fanatics", "lowvig", "mybookieag",
    "pinnacle", "williamhill_us",
}

SPORTS = {
    "nfl": {
        "provider_key": "football_nfl",
        "enabled": True,
        "model_version": "prop_projection_nfl_v1",
        "markets": (
            "player_pass_yds", "player_pass_tds", "player_pass_interceptions",
            "player_rush_yds", "player_reception_yds", "player_receptions",
            "player_rush_tds", "player_reception_tds", "player_kicking_points",
            "player_pass_attempts", "player_pass_completions", "player_rush_attempts",
            "player_reception_targets", "player_field_goals_made", "player_extra_points_made",
        ),
    },
    "ncaaf": {
        "provider_key": "football_ncaaf",
        "enabled": True,
        "model_version": "prop_projection_ncaaf_v1",
        "markets": (
            "player_pass_yds", "player_pass_tds", "player_pass_interceptions",
            "player_rush_yds", "player_reception_yds", "player_receptions",
            "player_rush_tds", "player_reception_tds", "player_kicking_points",
            "player_pass_attempts", "player_pass_completions", "player_rush_attempts",
            "player_reception_targets", "player_field_goals_made", "player_extra_points_made",
        ),
    },
    "mlb": {
        "provider_key": "baseball_mlb",
        "enabled": True,
        "model_version": "prop_projection_mlb_v1",
        "markets": (
            "pitcher_outs", "pitcher_hits_allowed", "pitcher_walks", "pitcher_earned_runs",
            "pitcher_strikeouts", "batter_hits", "batter_total_bases", "batter_home_runs",
            "batter_runs", "batter_rbis", "batter_strikeouts", "batter_walks",
            "batter_stolen_bases",
        ),
    },
    # Architecture is present, but these sports stay unavailable until a new
    # provider acceptance run verifies sufficient real-game coverage.
    "nba": {"provider_key": "basketball_nba", "enabled": False, "model_version": "prop_projection_nba_pending", "markets": (
        "player_points", "player_rebounds", "player_assists", "player_threes", "player_field_goals",
        "player_free_throws", "player_turnovers", "player_steals", "player_blocks",
    )},
    "ncaab": {"provider_key": "basketball_ncaab", "enabled": False, "model_version": "prop_projection_ncaab_pending", "markets": (
        "player_points", "player_rebounds", "player_assists", "player_threes", "player_field_goals",
        "player_free_throws", "player_turnovers", "player_steals", "player_blocks",
    )},
    "nhl": {"provider_key": "icehockey_nhl", "enabled": False, "model_version": "prop_projection_nhl_pending", "markets": (
        "player_goalie_saves", "player_shots_on_goal", "player_goals", "player_assists", "player_points",
        "player_power_play_points", "player_blocked_shots", "player_anytime_goal_scorer",
    )},
}

# Standard deviation assumptions translate a de-vigged probability around a
# posted threshold into a mean. They are fixed and versioned with each model.
STAT_SIGMA = {
    "player_pass_yds": 55.0, "player_pass_tds": 0.85, "player_pass_interceptions": 0.65,
    "player_rush_yds": 25.0, "player_reception_yds": 24.0, "player_receptions": 2.0,
    "player_rush_tds": 0.55, "player_reception_tds": 0.55, "player_kicking_points": 2.8,
    "player_pass_attempts": 6.0, "player_pass_completions": 5.0, "player_rush_attempts": 4.5,
    "player_reception_targets": 2.5, "player_field_goals_made": 1.0, "player_extra_points_made": 1.0,
    "pitcher_outs": 3.0, "pitcher_hits_allowed": 2.0, "pitcher_walks": 1.3,
    "pitcher_earned_runs": 1.8, "pitcher_strikeouts": 2.3, "batter_hits": 0.72,
    "batter_total_bases": 1.35, "batter_home_runs": 0.38, "batter_runs": 0.62,
    "batter_rbis": 0.72, "batter_strikeouts": 0.75, "batter_walks": 0.55,
    "batter_stolen_bases": 0.28,
}

FAMILIES = {
    "player_pass_yds": "passing", "player_pass_tds": "passing", "player_pass_interceptions": "turnovers",
    "player_rush_yds": "rushing", "player_rush_tds": "touchdowns",
    "player_reception_yds": "receiving", "player_receptions": "receiving",
    "player_reception_tds": "touchdowns", "player_kicking_points": "kicking",
    "player_pass_attempts": "passing", "player_pass_completions": "passing",
    "player_rush_attempts": "rushing", "player_reception_targets": "receiving",
    "player_field_goals_made": "kicking", "player_extra_points_made": "kicking",
    "pitcher_outs": "pitching", "pitcher_hits_allowed": "pitching", "pitcher_walks": "pitching",
    "pitcher_earned_runs": "pitching", "pitcher_strikeouts": "pitching",
    "batter_hits": "hitting", "batter_total_bases": "hitting", "batter_home_runs": "power",
    "batter_runs": "run_creation", "batter_rbis": "run_creation", "batter_strikeouts": "discipline",
    "batter_walks": "discipline", "batter_stolen_bases": "baserunning",
}

SUPPLEMENTAL = {
    "player_pass_attempts", "player_pass_completions", "player_rush_attempts",
    "player_reception_targets", "player_field_goals_made", "player_extra_points_made",
    "pitcher_strikeouts", "batter_strikeouts", "batter_walks", "batter_stolen_bases",
}
