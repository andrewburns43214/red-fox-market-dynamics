"""Reproduce the league baselines used by the props-only v2 shadow model.

The script reads nflverse play-by-play releases directly and prints regular-
season scoring frequencies per team-game. It is research-only and is never
invoked by the production collector.
"""

from __future__ import annotations

import argparse

import pandas as pd


URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
COLUMNS = [
    "game_id",
    "season_type",
    "touchdown",
    "pass_touchdown",
    "rush_touchdown",
    "return_touchdown",
    "safety",
    "two_point_conv_result",
    "defensive_two_point_conv",
    "defensive_extra_point_conv",
    "field_goal_result",
    "extra_point_result",
]


def summarize(season: int) -> dict[str, float]:
    frame = pd.read_csv(URL.format(season=season), usecols=COLUMNS, low_memory=False)
    frame = frame[frame["season_type"].eq("REG")]
    team_games = 2 * frame["game_id"].nunique()

    touchdowns = frame["touchdown"].fillna(0).eq(1)
    pass_tds = frame["pass_touchdown"].fillna(0).eq(1)
    rush_tds = frame["rush_touchdown"].fillna(0).eq(1)
    other_tds = touchdowns & ~pass_tds & ~rush_tds
    safeties = frame["safety"].fillna(0).eq(1)
    offensive_twos = frame["two_point_conv_result"].eq("success")
    defensive_twos = (
        frame["defensive_two_point_conv"].fillna(0).eq(1)
        | frame["defensive_extra_point_conv"].fillna(0).eq(1)
    )
    made_field_goals = frame["field_goal_result"].eq("made")
    made_extra_points = frame["extra_point_result"].eq("good")

    residual_points = (
        6 * other_tds.sum()
        + 2 * safeties.sum()
        + 2 * offensive_twos.sum()
        + 2 * defensive_twos.sum()
    )
    return {
        "season": season,
        "team_games": team_games,
        "passing_td_per_team_game": float(pass_tds.sum() / team_games),
        "rushing_td_per_team_game": float(rush_tds.sum() / team_games),
        "other_td_per_team_game": float(other_tds.sum() / team_games),
        "residual_points_per_team_game": float(residual_points / team_games),
        "kicking_points_per_team_game": float((3 * made_field_goals.sum() + made_extra_points.sum()) / team_games),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("seasons", nargs="*", type=int, default=list(range(2021, 2026)))
    args = parser.parse_args()
    rows = [summarize(season) for season in args.seasons]
    for row in rows:
        print(row)
    weights = sum(row["team_games"] for row in rows)
    pooled = {
        key: sum(row[key] * row["team_games"] for row in rows) / weights
        for key in rows[0]
        if key not in {"season", "team_games"}
    }
    print({"seasons": args.seasons, "team_games": weights, **pooled})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
