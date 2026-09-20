"""Versioned assumptions for the private props-only v2 shadow model."""

from __future__ import annotations

from prop_projection_config import STAT_SIGMA


V2_MODEL_VERSION = {
    "nfl": "prop_projection_nfl_v2_shadow_1",
    "ncaaf": "prop_projection_ncaaf_v2_shadow_1",
    "mlb": "prop_projection_mlb_v2_shadow_1",
}

# Every configured market has an explicit distribution class. Disabled sports
# are classified now so enabling them later cannot silently fall back to a
# universal Normal conversion.
CONTINUOUS = "CONTINUOUS"
COUNT = "COUNT"
BINARY_EVENT = "BINARY_EVENT"

DISTRIBUTION_KIND = {
    # Football
    "player_pass_yds": CONTINUOUS,
    "player_rush_yds": CONTINUOUS,
    "player_reception_yds": CONTINUOUS,
    "player_pass_tds": COUNT,
    "player_pass_interceptions": COUNT,
    "player_receptions": COUNT,
    "player_rush_tds": COUNT,
    "player_reception_tds": COUNT,
    "player_kicking_points": COUNT,
    "player_pass_attempts": COUNT,
    "player_pass_completions": COUNT,
    "player_rush_attempts": COUNT,
    "player_reception_targets": COUNT,
    "player_field_goals_made": COUNT,
    "player_extra_points_made": COUNT,
    # Baseball. Total bases is deliberately treated as a count.
    "pitcher_outs": COUNT,
    "pitcher_hits_allowed": COUNT,
    "pitcher_walks": COUNT,
    "pitcher_earned_runs": COUNT,
    "pitcher_strikeouts": COUNT,
    "batter_hits": COUNT,
    "batter_total_bases": COUNT,
    "batter_home_runs": BINARY_EVENT,
    "batter_runs": COUNT,
    "batter_rbis": COUNT,
    "batter_strikeouts": COUNT,
    "batter_walks": COUNT,
    "batter_stolen_bases": COUNT,
    # Basketball (disabled pending provider acceptance).
    "player_points": COUNT,
    "player_rebounds": COUNT,
    "player_assists": COUNT,
    "player_threes": COUNT,
    "player_field_goals": COUNT,
    "player_free_throws": COUNT,
    "player_turnovers": COUNT,
    "player_steals": COUNT,
    "player_blocks": COUNT,
    # Hockey (disabled pending provider acceptance).
    "player_goalie_saves": COUNT,
    "player_shots_on_goal": COUNT,
    "player_goals": COUNT,
    "player_assists": COUNT,
    "player_power_play_points": COUNT,
    "player_blocked_shots": COUNT,
    "player_anytime_goal_scorer": BINARY_EVENT,
}

# Continuous sigma assumptions remain configurable and auditable. They are
# inherited from v1 initially and must be calibrated only from settled games.
CONTINUOUS_SIGMA = {
    market: STAT_SIGMA[market]
    for market in ("player_pass_yds", "player_rush_yds", "player_reception_yds")
}

# Reproduced by tools/research_nfl_residual.py from 2021-2025 regular-season
# nflverse play-by-play: 2,718 team-games. Residual points are non-pass/rush
# touchdowns, safeties, successful two-point tries, and defensive conversions.
NFL_HISTORICAL_BASELINE = {
    "seasons": [2021, 2022, 2023, 2024, 2025],
    "team_games": 2718,
    "source": "nflverse play-by-play releases",
    "source_url": "https://github.com/nflverse/nflverse-data/releases/tag/pbp",
    "residual_points_per_team_game": 1.1111111111,
    "rushing_td_per_team_game": 0.9135393672,
    "kicking_points_per_team_game": 7.1688741722,
}

# MLB candidate assumptions remain separate and are evaluated rather than
# silently selected. The bullpen rate is inherited from v1 pending calibration.
MLB_RBI_TO_RUNS = 1.06
MLB_LINEAR_WEIGHTS = {"hits": 0.17, "total_bases": 0.13, "home_runs": 0.42, "walks": 0.22}
MLB_BULLPEN_RUNS_PER_INNING = 0.465
