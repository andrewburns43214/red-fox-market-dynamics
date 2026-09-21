"""Private, pre-kickoff NFL A/B/C/D predictions and prospective grading.

Only the collector writes candidate files, while the game is still upcoming.
The grader reads those files and never reprojects from post-kickoff odds.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from collections import defaultdict

from prop_projection import _active_timestamp, _is_suspended, _player_parts, american_probability, devig_pair, normalized_name, parse_time


VERSION = "nfl_prospective_abcd_1"
SCORER_MARGIN_FACTOR = 0.957  # exploratory paired-Pinnacle / DK ratio; private only


def _player_identity(outcome, rosters, teams):
    name, _ = _player_parts(outcome)
    key = normalized_name(name)
    pid = str(outcome.get("player_id") or "")
    matches = []
    for team in teams:
        positions = rosters.get(f"{team}__positions", {})
        if pid and pid in positions:
            matches.append((team, positions.get(f"{pid}__name", key), positions[pid]))
        elif key in positions:
            matches.append((team, key, positions[key]))
        else:
            # Provider sometimes expands Cam to Cameron. Require unique roster
            # match before accepting that abbreviation.
            pieces = name.split()
            if len(pieces) >= 2:
                tail = normalized_name(pieces[-1])
                initial = normalized_name(pieces[0])[:1]
                candidates = [n for n in rosters.get(team, []) if normalized_name(n).endswith(tail) and normalized_name(n).startswith(initial)]
                if len(candidates) == 1:
                    matches.append((team, normalized_name(candidates[0]), positions.get(normalized_name(candidates[0]), "")))
    return matches[0] if len(matches) == 1 else None


def _poisson_two_plus_mean(probability):
    lo, hi = 0.0, 8.0
    for _ in range(45):
        mid = (lo + hi) / 2
        if 1 - math.exp(-mid) * (1 + mid) < probability:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def scorer_surface(event, rosters, lines, now):
    """Infer scorer TD means from fresh paired Pinnacle or adjusted DK prices."""
    teams = (event["away_team"], event["home_team"])
    kickoff = parse_time(event.get("commence_time"))
    lead = (kickoff - now).total_seconds()
    max_age = 15 if lead <= 3600 else (25 if lead <= 86400 else 75)
    quotes = defaultdict(dict)
    for book in event.get("bookmakers") or []:
        book_key = str(book.get("key") or "").lower()
        if book_key not in {"pinnacle", "draftkings"}:
            continue
        for market in book.get("markets") or []:
            market_key = market.get("key")
            if market_key not in {"player_anytime_td", "player_2plus_td"}:
                continue
            for outcome in market.get("outcomes") or []:
                if _is_suspended(outcome, market):
                    continue
                timestamp = _active_timestamp(outcome, market, book)
                if not timestamp or timestamp > now or timestamp >= kickoff or (now - timestamp).total_seconds() > max_age * 60:
                    continue
                identity = _player_identity(outcome, rosters, teams)
                probability = american_probability(outcome.get("price"))
                if not identity or probability is None:
                    continue
                side = str(outcome.get("name") or "").lower()
                if side not in {"over", "under"}:
                    side = "over"
                quotes[(identity[0], identity[1], identity[2], market_key, book_key)][side] = (probability, timestamp, outcome.get("price"))
    by_player = defaultdict(dict)
    for (team, player, position, market, book), sides in quotes.items():
        if market == "player_anytime_td" and book == "pinnacle" and "over" in sides and "under" in sides:
            probability = devig_pair(sides["over"][2], sides["under"][2])
            by_player[(team, player, position)]["paired"] = (probability, min(sides["over"][1], sides["under"][1]))
        elif book == "draftkings" and "over" in sides:
            by_player[(team, player, position)]["any" if market == "player_anytime_td" else "two"] = (sides["over"][0], sides["over"][1])
    team_players = defaultdict(list)
    for (team, player, position), data in by_player.items():
        if "paired" in data:
            p_any, at = data["paired"]
            source = "PINNACLE_PAIRED"
        elif "any" in data:
            p_any, at = data["any"]
            p_any *= SCORER_MARGIN_FACTOR
            source = "DRAFTKINGS_ADJUSTED"
        else:
            continue
        p_any = min(0.98, max(0.001, p_any))
        means = [-math.log1p(-p_any)]
        if "two" in data:
            p_two, two_at = data["two"]
            means.append(_poisson_two_plus_mean(min(p_any, p_two * SCORER_MARGIN_FACTOR)))
            at = min(at, two_at)
        lam = statistics.mean(means)
        rush_line = next((x for x in lines if x["team"] == team and x["player_key"] == player and x["market"] == "player_rush_yds"), None)
        recv_line = next((x for x in lines if x["team"] == team and x["player_key"] == player and x["market"] == "player_reception_yds"), None)
        if position == "QB":
            rush_share = 1.0
        elif position in {"RB", "FB"}:
            rushing = rush_line["mean"] if rush_line else 0.0
            receiving = recv_line["mean"] if recv_line else 0.0
            rush_share = rushing / (rushing + receiving) if rushing + receiving else 0.0
        else:
            rush_share = 0.0
        team_players[team].append({"player": player, "position": position, "td_mean": round(lam, 5),
                                   "rushing_share": round(rush_share, 5), "source": source, "quote_at": at.isoformat(),
                                   "two_plus_used": "two" in data})
    return team_players


def _score(mean):
    return max(6.0, min(45.0, mean))


def _method(away, home, means, confidence, components, source_at, version, coverage):
    return {"status": "AVAILABLE", "model_version": version, "confidence": confidence,
            "away_mean": round(means[away], 4), "home_mean": round(means[home], 4),
            "components": components, "oldest_source_at": source_at,
            "coverage": coverage}


def _scoring_timestamps(lines, teams):
    """Select the canonical lines used by v1's pass/rush/kick score terms."""
    selected = {"A": [], "B": [], "C": []}
    for team in teams:
        by_market = defaultdict(list)
        for line in lines:
            if line.get("team") == team:
                by_market[line.get("market")].append(line)
        passing = by_market["player_pass_tds"]
        rushing = by_market["player_rush_tds"] or by_market["player_rush_yds"]
        kicking = by_market["player_kicking_points"]
        kick_lines = [max(kicking, key=lambda x: x["mean"])] if kicking else []
        if not kick_lines:
            for market in ("player_field_goals_made", "player_extra_points_made"):
                choices = by_market[market]
                if choices:
                    kick_lines.append(max(choices, key=lambda x: x["mean"]))
        pass_lines = [max(passing, key=lambda x: x["mean"])] if passing else []
        selected["A"].extend(pass_lines + rushing + kick_lines)
        selected["B"].extend(pass_lines + kick_lines)
        selected["C"].extend(kick_lines)
    return {name: [parse_time(x.get("oldest_observed_at") or x.get("observed_at")) for x in subset]
            for name, subset in selected.items()}


def build_candidate(event, production, lines, shadow, rosters, now):
    kickoff = parse_time(event.get("commence_time"))
    if not kickoff or now >= kickoff or event.get("live") is True:
        return None
    away, home = event["away_team"], event["home_team"]
    if production.get("status") != "AVAILABLE":
        return {"schema_version": VERSION, "sport": "nfl", "event_id": str(event["id"]),
                "away_team": away, "home_team": home, "commence_time": kickoff.isoformat(),
                "frozen_at": now.isoformat(), "production_audit_hash": production.get("audit_hash"),
                "methods": {name: {"status": "UNAVAILABLE", "reason": production.get("reason", "no_qualified_pregame_projection"),
                                   "model_version": version} for name, version in
                            (("A", production.get("model_version")), ("B", "nfl_scorer_rushing_shadow_1"),
                             ("C", "nfl_full_scorer_shadow_1"), ("D", shadow.get("model_version", "nfl_v2_shadow")))}}
    canonical_times = [parse_time(x.get("oldest_observed_at") or x.get("observed_at")) for x in lines]
    if any(x > now or x >= kickoff for x in canonical_times if x):
        return None
    scoring_times = _scoring_timestamps(lines, (away, home))
    oldest = min((x for x in scoring_times["A"] if x), default=None)
    source_at = oldest.isoformat() if oldest else None
    coverage = production.get("coverage", {}).get("teams", {})
    methods = {"A": _method(away, home, {away: production["away_mean"], home: production["home_mean"]},
                            production["confidence"], production.get("components", {}), source_at,
                            production["model_version"], coverage)}
    scorer = scorer_surface(event, rosters, lines, now)
    scorer_coverage = {team: {"players": len(scorer[team]), "paired_players": sum(p["source"] == "PINNACLE_PAIRED" for p in scorer[team]),
                              "two_plus_players": sum(p["two_plus_used"] for p in scorer[team]),
                              "rush_source": "SCORER_DERIVED" if scorer[team] else "UNAVAILABLE",
                              "production_rush_source": coverage.get(team, {}).get("scoring", {}).get("rushing_td_source")}
                       for team in (away, home)}
    if all(scorer[team] for team in (away, home)):
        b_times = [parse_time(p["quote_at"]) for team in (away, home) for p in scorer[team] if p["rushing_share"] > 0]
        c_times = [parse_time(p["quote_at"]) for team in (away, home) for p in scorer[team]]
        b_oldest = min([x for x in b_times + scoring_times["B"] if x]).isoformat()
        c_oldest = min([x for x in c_times + scoring_times["C"] if x]).isoformat()
        b_components, c_components, b_means, c_means = {}, {}, {}, {}
        for team in (away, home):
            base = production["components"][team]
            rush_td = sum(p["td_mean"] * p["rushing_share"] for p in scorer[team])
            total_td = sum(p["td_mean"] for p in scorer[team])
            b_components[team] = {"passing_td": base["passing_td"], "rushing_td": round(rush_td, 5),
                                  "kicking": base["kicking"], "scorer_players": scorer[team]}
            c_components[team] = {"total_offensive_td": round(total_td, 5), "kicking": base["kicking"],
                                  "scorer_players": scorer[team], "rushing_allocation": "UNRESOLVED"}
            b_means[team] = _score(6 * (base["passing_td"] + rush_td) + base["kicking"])
            c_means[team] = _score(6 * total_td + base["kicking"])
        methods["B"] = _method(away, home, b_means, production["confidence"], b_components,
                               b_oldest, "nfl_scorer_rushing_shadow_1", scorer_coverage)
        methods["C"] = _method(away, home, c_means, production["confidence"], c_components,
                               c_oldest, "nfl_full_scorer_shadow_1", scorer_coverage)
    else:
        for method in ("B", "C"):
            methods[method] = {"status": "UNAVAILABLE", "reason": "incomplete_fresh_scorer_surface",
                               "model_version": f"nfl_scorer_{method.lower()}_shadow_1", "coverage": scorer_coverage}
    baseline = shadow.get("baseline_variant")
    variant = shadow.get("variants", {}).get(baseline, {})
    if shadow.get("status") == "SHADOW_AVAILABLE" and variant:
        methods["D"] = _method(away, home, {away: variant["away_mean"], home: variant["home_mean"]},
                               shadow.get("confidence"), shadow.get("components", {}),
                               min((x for x in canonical_times if x), default=None).isoformat() if any(canonical_times) else None,
                               shadow.get("model_version"), coverage)
        methods["D"]["baseline_variant"] = baseline
    else:
        methods["D"] = {"status": "UNAVAILABLE", "reason": shadow.get("reason", "shadow_error"),
                        "model_version": shadow.get("model_version", "nfl_v2_shadow")}
    return {"schema_version": VERSION, "sport": "nfl", "event_id": str(event["id"]),
            "away_team": away, "home_team": home, "commence_time": kickoff.isoformat(),
            "frozen_at": now.isoformat(), "production_audit_hash": production.get("audit_hash"),
            "methods": methods, "assumptions": {"one_sided_margin_factor": SCORER_MARGIN_FACTOR,
                                           "scorer_count": "Poisson anytime/2plus average",
                                           "first_td_market": "excluded"}}


def freeze_candidate(root, event, production, lines, shadow, rosters, now):
    candidate = build_candidate(event, production, lines, shadow, rosters, now)
    if candidate is None:
        return False
    for method in candidate["methods"].values():
        method["frozen_at"] = now.isoformat()
    path = root / "prospective_frozen" / f"nfl_{candidate['event_id']}.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        old_start = parse_time(old.get("commence_time"))
        old_freeze = parse_time(old.get("frozen_at"))
        if not old_start or now >= old_start or (old_freeze and now <= old_freeze):
            return False
        for name, method in candidate["methods"].items():
            previous = old.get("methods", {}).get(name, {})
            if method.get("status") != "AVAILABLE" and previous.get("status") == "AVAILABLE":
                candidate["methods"][name] = previous
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(candidate, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    os.chmod(temp, 0o600)
    temp.replace(path)
    return True


def frozen_candidates(root, now, graded):
    result = {}
    for path in (root / "prospective_frozen").glob("nfl_*.json"):
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
            key = f"nfl:{candidate['event_id']}"
            kickoff = parse_time(candidate.get("commence_time"))
            frozen_at = parse_time(candidate.get("frozen_at"))
            if key not in graded and kickoff and frozen_at and frozen_at < kickoff < now:
                result[key] = candidate
        except (OSError, ValueError, KeyError):
            continue
    return result


def grade_candidate(candidate, away_actual, home_actual, now):
    kickoff = parse_time(candidate["commence_time"])
    frozen_at = parse_time(candidate["frozen_at"])
    if not kickoff or not frozen_at or frozen_at >= kickoff or now <= kickoff:
        raise ValueError("prediction_not_frozen_before_kickoff")
    methods = {}
    for name, prediction in candidate["methods"].items():
        if prediction.get("status") != "AVAILABLE":
            methods[name] = {"status": "UNAVAILABLE", "reason": prediction.get("reason"), "coverage": prediction.get("coverage")}
            continue
        a, h = prediction["away_mean"], prediction["home_mean"]
        source = parse_time(prediction.get("oldest_source_at"))
        method_freeze = parse_time(prediction.get("frozen_at")) or frozen_at
        if method_freeze >= kickoff:
            raise ValueError("method_not_frozen_before_kickoff")
        methods[name] = {"status": "GRADED", "model_version": prediction["model_version"],
                         "team_score_mae": round((abs(a-away_actual)+abs(h-home_actual))/2, 4),
                         "game_total_error": round(a+h-away_actual-home_actual, 4),
                         "scoring_margin_error": round((h-a)-(home_actual-away_actual), 4),
                         "source_age_minutes_at_freeze": round((method_freeze-source).total_seconds()/60, 3) if source else None,
                         "frozen_at": method_freeze.isoformat(), "oldest_source_at": prediction.get("oldest_source_at"),
                         "confidence": prediction.get("confidence"), "coverage": prediction.get("coverage"),
                         "predicted_away": a, "predicted_home": h}
    return {"event_id": candidate["event_id"], "away_team": candidate["away_team"], "home_team": candidate["home_team"],
            "commence_time": candidate["commence_time"], "frozen_at": candidate["frozen_at"],
            "actual_away": away_actual, "actual_home": home_actual, "graded_at": now.isoformat(), "methods": methods}


def performance_summary(state, now):
    games = list(state.get("games", {}).values())
    metrics = {}
    for method in "ABCD":
        rows = [g["methods"][method] for g in games if g.get("methods", {}).get(method, {}).get("status") == "GRADED"]
        metrics[method] = {"games": len(rows),
                           "team_score_mae": round(statistics.mean(r["team_score_mae"] for r in rows), 4) if rows else None,
                           "game_total_mae": round(statistics.mean(abs(r["game_total_error"]) for r in rows), 4) if rows else None,
                           "scoring_margin_mae": round(statistics.mean(abs(r["scoring_margin_error"]) for r in rows), 4) if rows else None}
    return {"generated_at": now.isoformat(), "prospective_only": True, "metrics": metrics, "game_count": len(games)}
