"""Strict customer-facing props-only consensus projection.

V3 is deliberately a publication gate as well as a model.  It publishes only
when independent scoring methods are both complete and reasonably close.  The
legacy projection remains available to the private ledger for comparison, but
never supplies a missing V3 scoring component.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

from prop_projection import (
    PLAYER_SUFFIX,
    TRADITIONAL_BOOKS,
    _active_timestamp,
    _is_suspended,
    american_probability,
    normalized_name,
    parse_time,
    project_event,
    utc_now,
)
from prop_projection_v2 import convert_lines, poisson_mean_from_over, project_event_v2


FOOTBALL_SCORER_MARKETS = {
    "player_anytime_td": 1,
    "player_2plus_td": 2,
    "player_3plus_td": 3,
}
MAX_FOOTBALL_METHOD_GAP_POINTS = 4.5  # three quarters of one touchdown
MAX_MLB_ESTIMATOR_RANGE = 2.1  # one published team-score standard deviation


def _unavailable(legacy, reason, detail, *, agreement=None):
    result = {
        **legacy,
        "model_version": f"prop_projection_{legacy['sport']}_v3_consensus_1",
        "status": "UNAVAILABLE",
        "display_status": "Projection not qualified",
        "confidence": "INSUFFICIENT",
        "reason": reason,
        "reason_detail": detail,
    }
    for key in ("away_score", "home_score", "away_mean", "home_mean", "score_distribution", "components"):
        result.pop(key, None)
    if agreement is not None:
        result["agreement"] = agreement
    return result


def _player_from_outcome(outcome):
    raw = str(outcome.get("description") or outcome.get("player_name") or "").strip()
    if not raw:
        candidate = str(outcome.get("name") or "").strip()
        if candidate.lower() not in {"yes", "no", "over", "under"}:
            raw = candidate
    match = PLAYER_SUFFIX.search(raw)
    token = match.group(1).strip().upper() if match else ""
    return PLAYER_SUFFIX.sub("", raw).strip(), token


def _max_age_minutes(sport, event, now):
    start = parse_time(event.get("commence_time"))
    lead = (start - now).total_seconds() if start else 0
    if sport == "nfl":
        return 15 if lead <= 3600 else (25 if lead <= 86400 else 75)
    return 30 if lead <= 3600 else (45 if lead <= 86400 else 90)


def _scorer_surface(sport, event, rosters, now):
    """Build conservative player TD means from one-sided scorer ladders.

    A one-sided market cannot be perfectly de-vigged.  Across books, the lowest
    implied probability is the least expensive quote and therefore the most
    conservative observable estimate.  At least two books per included player
    and three books per team are required before it can publish.
    """
    teams = (event["away_team"], event["home_team"])
    indexes = {team: {normalized_name(name) for name in rosters.get(team, [])} for team in teams}
    token_map = {}
    for team in teams:
        for token in rosters.get(f"{team}__tokens", []):
            token_map[str(token).upper()] = team
    max_age = _max_age_minutes(sport, event, now)
    quotes = defaultdict(dict)
    labels = {}
    for book in event.get("bookmakers") or []:
        book_key = str(book.get("key") or "").lower()
        if book_key not in TRADITIONAL_BOOKS:
            continue
        for market in book.get("markets") or []:
            market_key = str(market.get("key") or "")
            rung = FOOTBALL_SCORER_MARKETS.get(market_key)
            if not rung:
                continue
            for outcome in market.get("outcomes") or []:
                if _is_suspended(outcome, market):
                    continue
                timestamp = _active_timestamp(outcome, market, book)
                if not timestamp or timestamp > now or (now - timestamp).total_seconds() > max_age * 60:
                    continue
                player, token = _player_from_outcome(outcome)
                player_key = normalized_name(player)
                matches = [team for team in teams if player_key in indexes[team]]
                if len(matches) != 1:
                    continue
                team = matches[0]
                if token and token_map.get(token) not in {None, team}:
                    continue
                probability = american_probability(outcome.get("price"))
                if probability is None:
                    continue
                quotes[(team, player_key, rung)][book_key] = {
                    "probability": probability,
                    "price": outcome.get("price"),
                    "timestamp": timestamp,
                }
                labels[(team, player_key)] = player

    ladders = defaultdict(dict)
    for (team, player_key, rung), books in quotes.items():
        if len(books) < 2:
            continue
        # Best offered price has the lowest raw implied probability and is the
        # conservative choice when the provider does not expose a paired No.
        chosen_book, chosen = min(books.items(), key=lambda item: item[1]["probability"])
        ladders[(team, player_key)][rung] = {
            **chosen,
            "book": chosen_book,
            "book_count": len(books),
            "books": sorted(books),
        }

    players = defaultdict(list)
    for (team, player_key), rungs in ladders.items():
        if 1 not in rungs:
            continue
        probabilities = {1: min(0.97, max(0.001, rungs[1]["probability"]))}
        if 2 in rungs:
            probabilities[2] = min(probabilities[1], max(0.001, rungs[2]["probability"]))
        if 3 in rungs:
            ceiling = probabilities.get(2, probabilities[1])
            probabilities[3] = min(ceiling, max(0.001, rungs[3]["probability"]))
        means = [-math.log1p(-probabilities[1])]
        if 2 in probabilities:
            means.append(poisson_mean_from_over(1.5, probabilities[2]))
        if 3 in probabilities:
            means.append(poisson_mean_from_over(2.5, probabilities[3]))
        td_mean = statistics.median(means)
        players[team].append({
            "player": labels[(team, player_key)],
            "player_key": player_key,
            "td_mean": round(td_mean, 5),
            "rungs": sorted(probabilities),
            "anytime_price": rungs[1]["price"],
            "book_count": rungs[1]["book_count"],
            "books": rungs[1]["books"],
            "observed_at": rungs[1]["timestamp"].isoformat(),
        })
    return players


def _best(lines, team, market):
    choices = [line for line in lines if line["team"] == team and line["market"] == market]
    return sorted(choices, key=lambda line: (-line["book_count"], line["player_key"]))[0] if choices else None


def _football_team(lines, scorer_players, team):
    passing = _best(lines, team, "player_pass_tds")
    rushing = [line for line in lines if line["team"] == team and line["market"] == "player_rush_tds"]
    kicker = _best(lines, team, "player_kicking_points")
    field_goals = _best(lines, team, "player_field_goals_made")
    extra_points = _best(lines, team, "player_extra_points_made")
    if kicker and kicker["book_count"] >= 2:
        kicking = kicker["v2_mean"]
        kicking_method = "direct_kicker_points"
        kicking_books = kicker["book_count"]
    elif field_goals and extra_points and min(field_goals["book_count"], extra_points["book_count"]) >= 2:
        kicking = 3 * field_goals["v2_mean"] + extra_points["v2_mean"]
        kicking_method = "field_goals_plus_extra_points"
        kicking_books = min(field_goals["book_count"], extra_points["book_count"])
    else:
        kicking = None
        kicking_method = "missing"
        kicking_books = 0
    scorer_books = {book for player in scorer_players for book in player["books"]}
    structural_ready = bool(
        passing and passing["book_count"] >= 2 and len(rushing) >= 2
        and all(line["book_count"] >= 2 for line in rushing) and kicking is not None
    )
    scorer_ready = len(scorer_players) >= 5 and len(scorer_books) >= 3
    structural_td = passing["v2_mean"] + sum(line["v2_mean"] for line in rushing) if structural_ready else None
    scorer_td = sum(player["td_mean"] for player in scorer_players) if scorer_ready else None
    structural_points = 6 * structural_td + kicking if structural_td is not None else None
    scorer_points = 6 * scorer_td + kicking if scorer_td is not None and kicking is not None else None
    return {
        "ready": structural_ready and scorer_ready,
        "passing_td": passing["v2_mean"] if passing else None,
        "rushing_td": sum(line["v2_mean"] for line in rushing) if rushing else None,
        "kicking": kicking,
        "kicking_method": kicking_method,
        "kicking_books": kicking_books,
        "structural_points": structural_points,
        "scorer_points": scorer_points,
        "method_gap_points": abs(structural_points - scorer_points) if structural_points is not None and scorer_points is not None else None,
        "scorer_player_count": len(scorer_players),
        "scorer_book_count": len(scorer_books),
        "scorer_players": sorted(scorer_players, key=lambda item: -item["td_mean"]),
    }


def _football_projection(sport, event, rosters, legacy, lines, now):
    converted = convert_lines(lines)
    scorer = _scorer_surface(sport, event, rosters, now)
    teams = (event["away_team"], event["home_team"])
    components = {team: _football_team(converted, scorer[team], team) for team in teams}
    agreement = {
        "gate": "structural_vs_scorer_surface",
        "maximum_gap_points": MAX_FOOTBALL_METHOD_GAP_POINTS,
        "teams": {team: {
            "structural_points": None if components[team]["structural_points"] is None else round(components[team]["structural_points"], 3),
            "scorer_points": None if components[team]["scorer_points"] is None else round(components[team]["scorer_points"], 3),
            "gap_points": None if components[team]["method_gap_points"] is None else round(components[team]["method_gap_points"], 3),
        } for team in teams},
    }
    if not all(components[team]["ready"] for team in teams):
        return _unavailable(
            legacy, "insufficient_independent_scoring_methods",
            "Requires paired passing TDs, at least two paired rushing-TD players, reliable kicking, and a five-player/three-book anytime-TD surface for each team.",
            agreement=agreement,
        )
    if any(components[team]["method_gap_points"] > MAX_FOOTBALL_METHOD_GAP_POINTS for team in teams):
        return _unavailable(
            legacy, "projection_methods_disagree",
            "The paired scoring props and scorer ladder differ by more than 4.5 points for at least one team.",
            agreement=agreement,
        )
    means = {team: statistics.mean((components[team]["structural_points"], components[team]["scorer_points"])) for team in teams}
    high = all(
        components[team]["scorer_player_count"] >= 8
        and components[team]["scorer_book_count"] >= 4
        and components[team]["method_gap_points"] <= 3.0
        for team in teams
    )
    confidence = "HIGH" if high else "MODERATE"
    anchors = []
    for team in teams:
        passing = _best(converted, team, "player_pass_tds")
        if passing:
            anchors.append({"team": team, "player": passing["player"], "market": "player_pass_tds", "line": passing["line"], "book_count": passing["book_count"], "role": "scoring"})
        for player in components[team]["scorer_players"][:2]:
            anchors.append({"team": team, "player": player["player"], "market": "player_anytime_td", "line": player["anytime_price"], "book_count": player["book_count"], "role": "scoring"})
    result = {
        **legacy,
        "model_version": f"prop_projection_{sport}_v3_consensus_1",
        "status": "AVAILABLE",
        "confidence": confidence,
        "display_status": f"Props-Only · Verified {confidence.title()}",
        "away_mean": round(means[teams[0]], 2),
        "home_mean": round(means[teams[1]], 2),
        "away_score": int(round(means[teams[0]])),
        "home_score": int(round(means[teams[1]])),
        "components": components,
        "agreement": agreement,
        "anchors": anchors[:7],
        "projection_method": "paired_scoring_props_and_scorer_ladder_consensus",
        "score_distribution": {"away_sd": 6.8, "home_sd": 6.8},
    }
    return result


def _mlb_projection(legacy, lines, context, now):
    shadow = project_event_v2("mlb", legacy, lines, context=context, now=now)
    if shadow.get("status") != "SHADOW_AVAILABLE":
        return _unavailable(legacy, "missing_independent_run_estimators", "At least three independent run estimators are required for both teams.")
    teams = (legacy["away_team"], legacy["home_team"])
    components = shadow.get("components", {})
    ranges = {}
    for team in teams:
        estimates = components.get(team, [])
        values = [float(item["value"]) for item in estimates]
        ordered = sorted(values)
        windows = [ordered[index:index + 3] for index in range(max(0, len(ordered) - 2))]
        consensus_window = min(windows, key=lambda window: window[-1] - window[0]) if windows else []
        ranges[team] = {
            "estimator_count": len(values),
            "minimum": round(min(values), 3) if values else None,
            "maximum": round(max(values), 3) if values else None,
            "full_range": round(max(values) - min(values), 3) if values else None,
            "consensus_range": round(consensus_window[-1] - consensus_window[0], 3) if consensus_window else None,
            "estimators": [item["name"] for item in estimates],
        }
    agreement = {"gate": "tightest_three_independent_run_estimators", "maximum_consensus_range": MAX_MLB_ESTIMATOR_RANGE, "teams": ranges}
    if any(ranges[team]["estimator_count"] < 3 for team in teams):
        return _unavailable(legacy, "missing_independent_run_estimators", "At least three independent run estimators are required for both teams.", agreement=agreement)
    if any(ranges[team]["consensus_range"] > MAX_MLB_ESTIMATOR_RANGE for team in teams):
        return _unavailable(legacy, "projection_methods_disagree", "No three independent MLB run estimators agree within one team-score standard deviation.", agreement=agreement)
    consensus = shadow["variants"]["consensus_candidate"]
    high = legacy.get("confidence") == "HIGH" and all(ranges[team]["estimator_count"] >= 4 and ranges[team]["full_range"] <= 1.25 for team in teams)
    confidence = "HIGH" if high else "MODERATE"
    return {
        **legacy,
        "model_version": "prop_projection_mlb_v3_consensus_1",
        "status": "AVAILABLE",
        "confidence": confidence,
        "display_status": f"Props-Only · Verified {confidence.title()}",
        "away_mean": round(consensus["away_mean"], 2),
        "home_mean": round(consensus["home_mean"], 2),
        "away_score": consensus["away_score"],
        "home_score": consensus["home_score"],
        "components": components,
        "agreement": agreement,
        "projection_method": "independent_run_estimator_consensus",
    }


def project_event_v3(sport, event, rosters, context=None, now=None):
    """Run legacy parsing/identity checks, then apply the strict V3 gate."""
    now = now or utc_now()
    legacy = project_event(sport, event, rosters, context=context, now=now)
    legacy_benchmark = {
        key: legacy.get(key) for key in (
            "model_version", "status", "confidence", "away_mean", "home_mean",
            "away_score", "home_score", "reason",
        )
    }
    legacy["model_version"] = f"prop_projection_{sport}_v3_consensus_1"
    if legacy.get("status") != "AVAILABLE":
        legacy["_legacy_benchmark"] = legacy_benchmark
        return legacy
    lines = legacy.get("_private_lines", [])
    if sport in {"nfl", "ncaaf"}:
        result = _football_projection(sport, event, rosters, legacy, lines, now)
    elif sport == "mlb":
        result = _mlb_projection(legacy, lines, context or {}, now)
    else:
        result = _unavailable(legacy, "sport_model_not_validated", "No sport-specific props-only score model has passed validation.")
    result["_legacy_benchmark"] = legacy_benchmark
    return result
