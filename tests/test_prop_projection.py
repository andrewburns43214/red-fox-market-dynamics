from datetime import datetime, timedelta, timezone

import pytest

from prop_projection import (
    attach_teams,
    canonical_player_lines,
    devig_pair,
    flatten_event,
    project_event,
)


NOW = datetime(2026, 9, 19, 16, 0, tzinfo=timezone.utc)
BOOKS = ("draftkings", "fanduel", "betmgm")


def market(key, player, point, team, timestamp=NOW, suspended=False):
    return {
        "key": key,
        "suspended_at": timestamp.isoformat() if suspended else None,
        "last_update": timestamp.isoformat(),
        "outcomes": [
            {"name": "Over", "description": f"{player} ({team})", "price": -115, "point": point, "last_seen_at": timestamp.isoformat()},
            {"name": "Under", "description": f"{player} ({team})", "price": -105, "point": point, "last_seen_at": timestamp.isoformat()},
        ],
    }


def event_from_markets(markets, sport="football_nfl", away="Away Wolves", home="Home Bears"):
    return {
        "id": "evt-1", "sport_key": sport, "away_team": away, "home_team": home,
        "commence_time": (NOW + timedelta(hours=2)).isoformat(),
        "bookmakers": [{"key": book, "last_update": NOW.isoformat(), "markets": markets} for book in BOOKS],
    }


def football_fixture(sport="nfl"):
    specs = []
    rosters = {}
    for team, token, prefix in (("Away Wolves", "AW", "Away"), ("Home Bears", "HB", "Home")):
        players = [f"{prefix} QB", f"{prefix} RB", f"{prefix} WR1", f"{prefix} WR2", f"{prefix} WR3", f"{prefix} WR4", f"{prefix} K"]
        rosters[team], rosters[f"{team}__tokens"] = players, [token]
        specs.extend([
            ("player_pass_yds", players[0], 245.5), ("player_pass_tds", players[0], 1.5),
            ("player_pass_interceptions", players[0], 0.5), ("player_rush_yds", players[1], 64.5),
            ("player_rush_tds", players[1], 0.5), ("player_kicking_points", players[6], 6.5),
        ])
        for receiver in players[2:6]:
            specs.extend([("player_reception_yds", receiver, 43.5), ("player_receptions", receiver, 3.5)])
    markets = [market(key, player, point, "AW" if player.startswith("Away") else "HB") for key, player, point in specs]
    event = event_from_markets(markets, sport="football_ncaaf" if sport == "ncaaf" else "football_nfl")
    return event, rosters


def mlb_fixture():
    away, home = "Away Aces", "Home Hammers"
    rosters, specs = {}, []
    pitcher_names = {}
    for team, token, prefix in ((away, "AA", "Away"), (home, "HH", "Home")):
        batters = [f"{prefix} Batter{i}" for i in range(1, 10)]
        pitcher = f"{prefix} Pitcher"
        pitcher_names[team] = pitcher
        rosters[team], rosters[f"{team}__tokens"] = batters + [pitcher], [token]
        specs.extend([("pitcher_outs", pitcher, 17.5), ("pitcher_earned_runs", pitcher, 2.5), ("pitcher_hits_allowed", pitcher, 5.5)])
        for batter in batters:
            specs.extend([
                ("batter_hits", batter, 0.5), ("batter_total_bases", batter, 1.5),
                ("batter_home_runs", batter, 0.5), ("batter_runs", batter, 0.5),
                ("batter_rbis", batter, 0.5), ("batter_walks", batter, 0.5),
            ])
    markets = [market(key, player, point, "AA" if player.startswith("Away") else "HH") for key, player, point in specs]
    event = event_from_markets(markets, sport="baseball_mlb", away=away, home=home)
    context = {"lineup_confirmed": True, "away_probable_pitcher": pitcher_names[away], "home_probable_pitcher": pitcher_names[home]}
    return event, rosters, context


def test_devig_removes_two_sided_hold():
    fair = devig_pair(-110, -110)
    assert fair == pytest.approx(0.5)


def test_suspended_stale_and_one_sided_props_are_rejected():
    old = NOW - timedelta(hours=3)
    markets = [
        market("player_pass_yds", "Good QB", 250.5, "AW"),
        market("player_rush_yds", "Suspended RB", 50.5, "AW", suspended=True),
        market("player_receptions", "Old WR", 4.5, "AW", timestamp=old),
        {"key": "player_pass_tds", "outcomes": [{"name": "Over", "description": "Half QB (AW)", "price": -110, "point": 1.5, "last_seen_at": NOW.isoformat()}]},
    ]
    event = event_from_markets(markets)
    rows = flatten_event(event, now=NOW)
    assert {row["player"] for row in rows} == {"Good QB", "Half QB"}
    rosters = {"Away Wolves": ["Good QB", "Half QB"], "Home Bears": [], "Away Wolves__tokens": ["AW"], "Home Bears__tokens": ["HB"]}
    lines = canonical_player_lines(attach_teams(rows, event, rosters))
    assert {line["player"] for line in lines} == {"Good QB"}


def test_wrong_game_player_and_conflicting_team_suffix_are_rejected():
    event = event_from_markets([market("player_pass_yds", "Visitor QB", 220.5, "HB")])
    rosters = {"Away Wolves": ["Visitor QB"], "Home Bears": ["Home QB"], "Away Wolves__tokens": ["AW"], "Home Bears__tokens": ["HB"]}
    assert attach_teams(flatten_event(event, now=NOW), event, rosters) == []


def test_canonical_line_prefers_book_count_then_central_threshold():
    rows = []
    for index, book in enumerate(("draftkings", "fanduel", "betmgm", "caesars", "pinnacle")):
        point = 51.5 if index < 3 else 50.5
        for side, price in (("over", -110), ("under", -110)):
            rows.append({"team": "Away Wolves", "player": "Runner", "player_key": "runner", "player_id": "1", "market": "player_rush_yds", "book": book, "point": point, "side": side, "price": price, "timestamp": NOW})
    line = canonical_player_lines(rows)[0]
    assert line["line"] == 51.5
    assert line["book_count"] == 3
    assert line["selection_audit"]["available_thresholds"] == [50.5, 51.5]


def test_source_age_uses_oldest_contributing_book_leg():
    event, rosters = football_fixture()
    older = NOW - timedelta(minutes=16)
    event["bookmakers"][0]["markets"][0]["outcomes"][0]["last_seen_at"] = older.isoformat()
    result = project_event("nfl", event, rosters, now=NOW)
    assert result["status"] == "AVAILABLE"
    assert result["oldest_observation_age_minutes"] == 16.0


@pytest.mark.parametrize("sport", ["nfl", "ncaaf"])
def test_qualifying_football_projection_is_deterministic(sport):
    event, rosters = football_fixture(sport)
    first = project_event(sport, event, rosters, now=NOW)
    second = project_event(sport, event, rosters, now=NOW)
    assert first == second
    assert first["status"] == "AVAILABLE"
    assert first["confidence"] == "MODERATE"
    assert first["display_status"] == "Props-Only · Moderate"
    assert all(
        side["scoring"]["rushing_td_source"] == "DIRECT_RUSH_TD"
        and side["scoring"]["rushing_td_players"] == 1
        for side in first["coverage"]["teams"].values()
    )
    assert 5 <= len(first["anchors"]) <= 7
    assert 10 <= first["away_score"] <= 45
    assert first["model_version"] == f"prop_projection_{sport}_v1"


def test_zero_coverage_ncaaf_is_unavailable_not_guessed():
    event, _ = football_fixture("ncaaf")
    result = project_event("ncaaf", event, {"Away Wolves": [], "Home Bears": []}, now=NOW)
    assert result["status"] == "UNAVAILABLE"
    assert result["confidence"] == "INSUFFICIENT"
    assert "away_score" not in result


def test_future_event_without_open_prop_rows_has_explicit_not_open_status():
    event, rosters = football_fixture("nfl")
    event["bookmakers"] = []
    result = project_event("nfl", event, rosters, now=NOW)
    assert result["status"] == "NOT_OPEN"
    assert result["display_status"] == "Props not open yet"
    assert result["reason"] == "props_not_open_yet"


def test_live_high_coverage_nfl_stops_at_start():
    event, rosters = football_fixture("nfl")
    event["commence_time"] = NOW.isoformat()
    assert project_event("nfl", event, rosters, now=NOW)["reason"] == "game_started"


def test_nfl_proxy_rushing_tds_lower_confidence_and_stale_source_is_unavailable():
    event, rosters = football_fixture()
    for book in event["bookmakers"]:
        book["markets"] = [item for item in book["markets"] if item["key"] != "player_rush_tds"]
    result = project_event("nfl", event, rosters, now=NOW)
    assert result["status"] == "AVAILABLE"
    assert result["confidence"] == "MODERATE"
    assert all(
        side["scoring"]["rushing_td_source"] == "RUSH_YARDS_PROXY"
        for side in result["coverage"]["teams"].values()
    )
    event["commence_time"] = (NOW + timedelta(hours=8)).isoformat()
    later = project_event("nfl", event, rosters, now=NOW + timedelta(minutes=26))
    assert later["status"] == "UNAVAILABLE"
    assert later["reason"] == "no_fresh_qualified_props"


def test_nfl_scorer_ladders_do_not_enter_v1_score():
    event, rosters = football_fixture()
    first = project_event("nfl", event, rosters, now=NOW)
    for book in event["bookmakers"]:
        book["markets"].extend([
            {"key": "player_anytime_td", "outcomes": [
                {"name": "Away RB", "description": "Away RB (AW)", "price": -200, "point": None,
                 "last_seen_at": NOW.isoformat()},
            ]},
            {"key": "player_2plus_td", "outcomes": [
                {"name": "Away RB", "description": "Away RB (AW)", "price": 300, "point": None,
                 "last_seen_at": NOW.isoformat()},
            ]},
        ])
    second = project_event("nfl", event, rosters, now=NOW)
    assert (first["away_mean"], first["home_mean"]) == (second["away_mean"], second["home_mean"])
    assert second["line_count"] == first["line_count"]


def test_nfl_high_requires_direct_rushing_depth_and_fresh_prices():
    event, rosters = football_fixture()
    for prefix, team, token in (("Away", "Away Wolves", "AW"), ("Home", "Home Bears", "HB")):
        rosters[team].append(f"{prefix} RB2")
        event["bookmakers"][0]["markets"].append(
            market("player_rush_tds", f"{prefix} RB2", 0.5, token)
        )
    fresh = project_event("nfl", event, rosters, now=NOW)
    assert fresh["status"] == "AVAILABLE"
    assert fresh["confidence"] == "HIGH"
    older = project_event("nfl", event, rosters, now=NOW + timedelta(minutes=16))
    assert older["status"] == "AVAILABLE"
    assert older["confidence"] == "MODERATE"


def test_football_score_requires_rushing_and_complete_kicking_component():
    event, rosters = football_fixture()
    for book in event["bookmakers"]:
        book["markets"] = [
            item for item in book["markets"]
            if item["key"] not in {"player_rush_yds", "player_rush_tds"}
        ]
    missing_rush = project_event("nfl", event, rosters, now=NOW)
    assert missing_rush["status"] == "UNAVAILABLE"
    assert missing_rush["confidence"] == "INSUFFICIENT"


def test_receiving_tds_and_fg_pat_do_not_double_count_passing_or_kicker_points():
    event, rosters = football_fixture()
    baseline = project_event("nfl", event, rosters, now=NOW)
    for book in event["bookmakers"]:
        book["markets"].extend([
            market("player_reception_tds", "Away WR1", 0.5, "AW"),
            market("player_reception_tds", "Home WR1", 0.5, "HB"),
            market("player_field_goals_made", "Away K", 1.5, "AW"),
            market("player_extra_points_made", "Away K", 2.5, "AW"),
            market("player_field_goals_made", "Home K", 1.5, "HB"),
            market("player_extra_points_made", "Home K", 2.5, "HB"),
        ])
    enriched = project_event("nfl", event, rosters, now=NOW)
    assert (enriched["away_mean"], enriched["home_mean"]) == (
        baseline["away_mean"], baseline["home_mean"]
    )


def test_mlb_high_requires_probable_pitchers_and_lineup_coverage():
    event, rosters, context = mlb_fixture()
    result = project_event("mlb", event, rosters, context=context, now=NOW)
    assert result["status"] == "AVAILABLE"
    assert result["confidence"] == "HIGH"
    assert result["display_status"] == "Props-Only · High"
    assert 1 <= result["away_score"] <= 10
    assert result["oldest_observation_age_minutes"] == 0
    assert all(team["probable_pitcher_represented"] for team in result["coverage"]["teams"].values())
    missing = project_event("mlb", event, rosters, context={"lineup_confirmed": True}, now=NOW)
    assert missing["status"] == "UNAVAILABLE"


def test_mlb_complete_hitter_surface_with_one_missing_pitcher_is_moderate():
    event, rosters, context = mlb_fixture()
    for book in event["bookmakers"]:
        book["markets"] = [
            item for item in book["markets"]
            if "Home Pitcher" not in str(item.get("outcomes", [{}])[0].get("description", ""))
        ]
    result = project_event("mlb", event, rosters, context=context, now=NOW)
    assert result["status"] == "AVAILABLE"
    assert result["confidence"] == "MODERATE"
    assert result["display_status"] == "Props-Only · Moderate"
    assert result["coverage"]["teams"]["Away Aces"]["batters"] == 9
    assert result["coverage"]["teams"]["Home Hammers"]["batters"] == 9
    assert 5 <= len(result["anchors"]) <= 7


def test_mlb_complete_hitter_surface_allows_unannounced_second_starter_at_moderate():
    event, rosters, context = mlb_fixture()
    context["home_probable_pitcher"] = None
    for book in event["bookmakers"]:
        book["markets"] = [
            item for item in book["markets"]
            if "Home Pitcher" not in str(item.get("outcomes", [{}])[0].get("description", ""))
        ]
    result = project_event("mlb", event, rosters, context=context, now=NOW)
    assert result["status"] == "AVAILABLE"
    assert result["confidence"] == "MODERATE"


def test_no_game_line_or_red_fox_decision_input_exists():
    import inspect
    import prop_projection

    source = inspect.getsource(prop_projection).lower()
    for forbidden in ("moneyline", "game_total", "market_read", "supported_side", "red_fox_favorite", "anomaly_board"):
        assert forbidden not in source

