"""Deterministic player-prop normalization and score projection.

Pipeline: raw odds -> validation -> de-vig -> canonical main line -> player
distribution -> correlation-controlled team production -> score distribution.
The module deliberately has no imports from the Red Fox market engine.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from statistics import NormalDist

from prop_projection_config import FAMILIES, SPORTS, STAT_SIGMA, SUPPLEMENTAL, TRADITIONAL_BOOKS


PLAYER_SUFFIX = re.compile(r"\s*\(([A-Za-z0-9 .&'-]{2,12})\)\s*$")
MILESTONE_MARKERS = ("milestone", "alternate", "alt_", "1st_", "2plus", "3plus", "4plus", "longest")


def utc_now():
    return datetime.now(timezone.utc)


def parse_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def normalized_name(value):
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    text = PLAYER_SUFFIX.sub("", text).lower()
    return re.sub(r"[^a-z0-9]", "", text)


def matchup_key(sport, away, home):
    return f"{str(sport).lower()}|{normalized_name(away)}|{normalized_name(home)}"


def american_probability(price):
    try:
        value = float(price)
    except (TypeError, ValueError):
        return None
    if value == 0:
        return None
    return (-value / (-value + 100.0)) if value < 0 else (100.0 / (value + 100.0))


def devig_pair(over_price, under_price):
    over = american_probability(over_price)
    under = american_probability(under_price)
    if over is None or under is None or over + under <= 0:
        return None
    return over / (over + under)


def observation_hash(event):
    kept = {
        "id": event.get("id"), "sport_key": event.get("sport_key"),
        "home_team": event.get("home_team"), "away_team": event.get("away_team"),
        "commence_time": event.get("commence_time"), "bookmakers": event.get("bookmakers", []),
    }
    return hashlib.sha256(json.dumps(kept, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _active_timestamp(outcome, market, bookmaker):
    for source in (outcome, market, bookmaker):
        for field in ("last_seen_at", "book_updated_at", "last_change_at", "last_update"):
            parsed = parse_time(source.get(field))
            if parsed:
                return parsed
    return None


def _is_suspended(outcome, market):
    return bool(outcome.get("suspended_at") or market.get("suspended_at") or outcome.get("suspended") is True or market.get("suspended") is True)


def _player_parts(outcome):
    raw = str(outcome.get("description") or outcome.get("player_name") or "").strip()
    match = PLAYER_SUFFIX.search(raw)
    return PLAYER_SUFFIX.sub("", raw).strip(), (match.group(1).strip().upper() if match else "")


def flatten_event(event, now=None, max_age_minutes=None):
    """Return validated raw prop legs; stale/suspended/malformed legs disappear."""
    now = now or utc_now()
    if max_age_minutes is None:
        start = parse_time(event.get("commence_time"))
        lead = (start - now).total_seconds() if start else 0
        max_age_minutes = 20 if lead <= 3600 else (40 if lead <= 6 * 3600 else 75)
    rows = []
    for book in event.get("bookmakers") or []:
        book_key = str(book.get("key") or "").lower()
        if book_key not in TRADITIONAL_BOOKS:
            continue
        for market in book.get("markets") or []:
            market_key = str(market.get("key") or "")
            if market_key not in STAT_SIGMA or any(marker in market_key for marker in MILESTONE_MARKERS):
                continue
            for outcome in market.get("outcomes") or []:
                side = str(outcome.get("name") or "").strip().lower()
                if side not in {"over", "under"} or _is_suspended(outcome, market):
                    continue
                player, team_token = _player_parts(outcome)
                point = outcome.get("point")
                try:
                    point = float(point)
                except (TypeError, ValueError):
                    continue
                timestamp = _active_timestamp(outcome, market, book)
                if not timestamp or (now - timestamp).total_seconds() > max_age_minutes * 60:
                    continue
                rows.append({
                    "event_id": str(event.get("id") or ""), "book": book_key, "market": market_key,
                    "side": side, "player": player, "player_key": normalized_name(player),
                    "player_id": str(outcome.get("player_id") or ""), "team_token": team_token,
                    "point": point, "price": outcome.get("price"), "timestamp": timestamp,
                })
    return rows


def _nfl_max_age_minutes(event, now):
    start = parse_time(event.get("commence_time"))
    lead = (start - now).total_seconds() if start else 0
    if lead <= 3600:
        return 15
    if lead <= 24 * 3600:
        return 25
    return 75


def attach_teams(rows, event, rosters):
    """Attach a team only after exact roster validation; ambiguity is rejected."""
    teams = [str(event.get("away_team") or ""), str(event.get("home_team") or "")]
    indexes = {team: {normalized_name(name) for name in rosters.get(team, [])} for team in teams}
    token_map = {}
    for team in teams:
        candidates = rosters.get(f"{team}__tokens", [])
        for token in candidates:
            token_map[str(token).upper()] = team
    accepted = []
    for row in rows:
        matches = [team for team in teams if row["player_key"] in indexes.get(team, set())]
        if len(matches) != 1:
            continue
        team = matches[0]
        token_team = token_map.get(row["team_token"]) if row["team_token"] else None
        if token_team and token_team != team:
            continue
        accepted.append({**row, "team": team})
    return accepted


def canonical_player_lines(rows):
    """Choose one main threshold per player/stat and calculate its fair mean."""
    paired = defaultdict(dict)
    for row in rows:
        key = (row["team"], row["player_key"], row["market"], row["book"], row["point"])
        paired[key][row["side"]] = row
    candidates = defaultdict(list)
    for (team, player_key, market, book, point), sides in paired.items():
        if set(sides) != {"over", "under"}:
            continue
        probability = devig_pair(sides["over"].get("price"), sides["under"].get("price"))
        if probability is None:
            continue
        row = sides["over"]
        candidates[(team, player_key, market, point)].append({
            "book": book, "over_probability": probability,
            "observed_at": max(sides["over"]["timestamp"], sides["under"]["timestamp"]),
            "player": row["player"], "player_id": row["player_id"],
        })
    by_stat = defaultdict(list)
    for (team, player_key, market, point), books in candidates.items():
        by_stat[(team, player_key, market)].append((point, books))
    output = []
    for (team, player_key, market), thresholds in sorted(by_stat.items()):
        all_points = [point for point, _ in thresholds]
        center = statistics.median(all_points)
        point, books = sorted(thresholds, key=lambda item: (-len(item[1]), abs(item[0] - center), item[0]))[0]
        fair_over = statistics.median(item["over_probability"] for item in books)
        fair_over = min(0.98, max(0.02, fair_over))
        mean = max(0.0, point + STAT_SIGMA[market] * NormalDist().inv_cdf(fair_over))
        output.append({
            "team": team, "player": books[0]["player"], "player_key": player_key,
            "player_id": next((b["player_id"] for b in books if b["player_id"]), ""),
            "market": market, "family": FAMILIES[market], "classification": "SUPPLEMENTAL" if market in SUPPLEMENTAL else "CORE",
            "line": point, "mean": round(mean, 4), "fair_over": round(fair_over, 5),
            "books": sorted(b["book"] for b in books), "book_count": len(books),
            "observed_at": max(b["observed_at"] for b in books).isoformat(),
            "selection_audit": {"available_thresholds": sorted(all_points), "rule": "book_count_then_central_threshold"},
        })
    return output


def _values(lines, team, market):
    return [line["mean"] for line in lines if line["team"] == team and line["market"] == market]


def _players(lines, team, prefix=""):
    return {line["player_key"] for line in lines if line["team"] == team and (not prefix or line["market"].startswith(prefix))}


def _football_score(lines, team):
    pass_tds = _values(lines, team, "player_pass_tds")
    rush_tds = _values(lines, team, "player_rush_tds")
    rush_yards = _values(lines, team, "player_rush_yds")
    kicking = _values(lines, team, "player_kicking_points")
    field_goals = _values(lines, team, "player_field_goals_made")
    extra_points = _values(lines, team, "player_extra_points_made")
    # Receiving TDs validate passing TD production and are not added again.
    # One-sided anytime/2+ scorer quotes are collected separately for research;
    # they cannot be de-vigged as paired lines and do not enter this v1 score.
    passing_td = max(pass_tds) if pass_tds else 0.0
    rushing_td = sum(rush_tds) if rush_tds else (sum(rush_yards) / 92.0 if rush_yards else 0.0)
    kick_points = max(kicking) if kicking else ((3.0 * max(field_goals) if field_goals else 0.0) + (max(extra_points) if extra_points else 0.0))
    mean = 6.0 * (passing_td + rushing_td) + kick_points
    return min(45.0, max(6.0, mean)), {"passing_td": passing_td, "rushing_td": rushing_td, "kicking": kick_points}


def _mlb_score(lines, offense, defense):
    sums = lambda market: sum(_values(lines, offense, market))
    estimates = []
    runs, rbis = sums("batter_runs"), sums("batter_rbis")
    if runs: estimates.append(1.12 * runs)
    if rbis: estimates.append(1.06 * rbis)
    hits, total_bases, homers, walks = sums("batter_hits"), sums("batter_total_bases"), sums("batter_home_runs"), sums("batter_walks")
    if hits and total_bases:
        # Correlation controlled: TB and hits enter one linear-weights estimate,
        # not as separate run totals. HR receives only its incremental value.
        estimates.append(0.17 * hits + 0.13 * total_bases + 0.42 * homers + 0.22 * walks)
    earned = _values(lines, defense, "pitcher_earned_runs")
    outs = _values(lines, defense, "pitcher_outs")
    if earned and outs:
        bullpen_innings = max(0.0, 9.0 - max(outs) / 3.0)
        estimates.append(max(earned) + bullpen_innings * 0.465)
    if not estimates:
        return None, {"estimators": []}
    return min(10.0, max(1.0, statistics.median(estimates))), {"estimators": [round(x, 3) for x in estimates]}


def _best_anchor(lines, team, markets, used):
    priorities = {market: index for index, market in enumerate(markets)}
    candidates = [line for line in lines if line["team"] == team and line["market"] in priorities]
    candidates.sort(key=lambda line: (priorities[line["market"]], -line["book_count"], line["player_key"]))
    for line in candidates:
        key = (line["team"], line["player_key"], line["market"])
        if key not in used:
            used.add(key)
            return {
                "team": line["team"], "player": line["player"], "market": line["market"],
                "line": line["line"], "book_count": line["book_count"],
                "role": "scoring" if line["market"] not in {
                    "player_pass_yds", "player_reception_yds", "player_receptions",
                    "batter_hits", "batter_total_bases",
                } else "validation",
            }
    return None


def strongest_anchors(sport, lines, away, home, maximum=7):
    """Return a small, balanced explanation set without changing broad collection."""
    used, anchors = set(), []

    def add(team, markets):
        anchor = _best_anchor(lines, team, markets, used)
        if anchor and len(anchors) < maximum:
            anchors.append(anchor)

    if sport in {"nfl", "ncaaf"}:
        for team in (away, home):
            add(team, ("player_pass_tds", "player_pass_yds"))
            add(team, ("player_rush_tds", "player_rush_yds"))
            add(team, ("player_kicking_points", "player_field_goals_made", "player_extra_points_made"))
    else:
        # Pitcher ER + outs form one opponent-run estimator. Hitter props form
        # separate run-creation estimators that are blended, never summed.
        for team in (away, home):
            add(team, ("pitcher_earned_runs",))
            add(team, ("pitcher_outs",))
        for team in (away, home):
            add(team, ("batter_runs", "batter_rbis"))
        # One final representative contact/power line keeps the explanation
        # concise while the full hitter surface remains in the model.
        best_team = max((away, home), key=lambda team: max(
            (line["book_count"] for line in lines if line["team"] == team and line["market"] in {"batter_total_bases", "batter_hits"}),
            default=-1,
        ))
        add(best_team, ("batter_total_bases", "batter_hits"))
        other_team = home if best_team == away else away
        add(other_team, ("batter_total_bases", "batter_hits"))
    return anchors[:maximum]


def _coverage(sport, lines, event, context, now):
    teams = [event["away_team"], event["home_team"]]
    ages = [parse_time(line["observed_at"]) for line in lines]
    age_minutes = max(((now - age).total_seconds() / 60 for age in ages if age), default=9999)
    books = {book for line in lines for book in line["books"]}
    per_team = {}
    for team in teams:
        team_lines = [line for line in lines if line["team"] == team]
        per_team[team] = {
            "players": len(_players(lines, team)), "books": len({book for line in team_lines for book in line["books"]}),
            "families": sorted({line["family"] for line in team_lines}),
            "markets": sorted({line["market"] for line in team_lines}),
        }
    if sport in {"nfl", "ncaaf"}:
        moderate = all(x["players"] >= 4 and x["books"] >= 2 and len(x["families"]) >= 3 for x in per_team.values())
        score_ready = all(
            "player_pass_tds" in x["markets"]
            and ("player_rush_tds" in x["markets"] or "player_rush_yds" in x["markets"])
            and (
                "player_kicking_points" in x["markets"]
                or {"player_field_goals_made", "player_extra_points_made"}.issubset(x["markets"])
            )
            for x in per_team.values()
        )
        high = moderate and score_ready and all(x["players"] >= 7 and x["books"] >= 3 and len(x["families"]) >= 4 for x in per_team.values())
        for team in teams:
            team_lines = [line for line in lines if line["team"] == team]
            passing = [line for line in team_lines if line["market"] == "player_pass_tds"]
            direct_rushing = [line for line in team_lines if line["market"] == "player_rush_tds"]
            rushing_yards = [line for line in team_lines if line["market"] == "player_rush_yds"]
            kicker = [line for line in team_lines if line["market"] == "player_kicking_points"]
            field_goals = [line for line in team_lines if line["market"] == "player_field_goals_made"]
            extra_points = [line for line in team_lines if line["market"] == "player_extra_points_made"]
            pass_books = max((line["book_count"] for line in passing), default=0)
            kicker_books = max((line["book_count"] for line in kicker), default=0)
            if not kicker_books and field_goals and extra_points:
                kicker_books = min(
                    max(line["book_count"] for line in field_goals),
                    max(line["book_count"] for line in extra_points),
                )
            rush_source = "DIRECT_RUSH_TD" if direct_rushing else ("RUSH_YARDS_PROXY" if rushing_yards else "MISSING")
            per_team[team]["scoring"] = {
                "passing_td_books": pass_books,
                "rushing_td_source": rush_source,
                "rushing_td_players": len(direct_rushing),
                "rushing_td_min_books": min((line["book_count"] for line in direct_rushing), default=0),
                "kicking_books": kicker_books,
            }
            high = high and pass_books >= 3 and kicker_books >= 2 and len(direct_rushing) >= 2 and all(
                line["book_count"] >= 2 for line in direct_rushing
            )
        # A broad receiving/yardage surface cannot compensate for old scoring
        # prices. NFL collection aims for 10-15 minute source age pregame.
        high = high and age_minutes <= (15 if sport == "nfl" else 30)
    else:
        lineup_confirmed = bool(context.get("lineup_confirmed"))
        pitchers_by_team = {team: str(context.get(f"{side}_probable_pitcher") or "")
                            for side, team in (("away", teams[0]), ("home", teams[1]))}
        pitchers = {normalized_name(name) for name in pitchers_by_team.values()} - {""}
        represented_by_team = {team: {line["player_key"] for line in lines if line["team"] == team}
                               for team in teams}
        represented_pitchers = {normalized_name(name) for team, name in pitchers_by_team.items()
                                if normalized_name(name) in represented_by_team[team]}
        pitchers_ready = len(pitchers) == 2 and len(represented_pitchers) == 2
        # The second probable starter may be named but lack props, or may still
        # be unannounced in context. Either case is acceptable only when one
        # identified starter is represented and both nine-hitter surfaces are
        # otherwise complete.
        one_pitcher_missing = len(pitchers) in {1, 2} and len(represented_pitchers) == 1
        batter_counts = {team: len({line["player_key"] for line in lines if line["team"] == team and line["market"].startswith("batter_")}) for team in teams}
        ordinary_ready = pitchers_ready and all(batter_counts[team] >= 7 and x["books"] >= 2 and len(x["families"]) >= 3 for team, x in per_team.items())
        complete_hitter_ready = one_pitcher_missing and all(
            batter_counts[team] >= 9 and x["books"] >= 2 and
            {"hitting", "run_creation"}.issubset(set(x["families"]))
            for team, x in per_team.items()
        )
        moderate = ordinary_ready or complete_hitter_ready
        high = moderate and lineup_confirmed and all(batter_counts[team] >= 8 and x["books"] >= 3 and len(x["families"]) >= 4 for team, x in per_team.items())
        high = high and pitchers_ready
        for team in teams:
            per_team[team]["batters"] = batter_counts[team]
            per_team[team]["probable_pitcher"] = pitchers_by_team[team]
            per_team[team]["probable_pitcher_represented"] = normalized_name(pitchers_by_team[team]) in represented_by_team[team]
        score_ready = moderate
    confidence = "HIGH" if high else ("MODERATE" if moderate and score_ready else "INSUFFICIENT")
    coverage = {"teams": per_team, "books": sorted(books), "age_minutes": round(age_minutes, 1)}
    if sport == "mlb":
        coverage["lineup_confirmed"] = lineup_confirmed
        coverage["one_pitcher_missing_fallback"] = complete_hitter_ready and not pitchers_ready
    return confidence, coverage


def project_event(sport, event, rosters, context=None, now=None):
    """Build one auditable public projection payload from a provider event."""
    now = now or utc_now()
    context = context or {}
    config = SPORTS[sport]
    base = {
        "sport": sport, "event_id": str(event.get("id") or ""), "away_team": event.get("away_team", ""),
        "home_team": event.get("home_team", ""), "commence_time": event.get("commence_time"),
        "matchup_key": matchup_key(sport, event.get("away_team"), event.get("home_team")),
        "model_version": config["model_version"], "provider": "PropLine", "generated_at": now.isoformat(),
    }
    start = parse_time(event.get("commence_time"))
    if start and start <= now:
        return {**base, "status": "UNAVAILABLE", "reason": "game_started", "confidence": "INSUFFICIENT"}
    raw_rows = flatten_event(
        event, now=now,
        max_age_minutes=_nfl_max_age_minutes(event, now) if sport == "nfl" else None,
    )
    if not raw_rows:
        had_main_props = any(
            market.get("key") in STAT_SIGMA
            for book in event.get("bookmakers") or []
            if str(book.get("key") or "").lower() in TRADITIONAL_BOOKS
            for market in book.get("markets") or []
        )
        return {
            **base,
            "status": "UNAVAILABLE" if had_main_props else "NOT_OPEN",
            "display_status": "Insufficient coverage" if had_main_props else "Props not open yet",
            "reason": "no_fresh_qualified_props" if had_main_props else "props_not_open_yet",
            "confidence": "INSUFFICIENT" if had_main_props else "NOT_OPEN",
        }
    rows = attach_teams(raw_rows, event, rosters)
    lines = canonical_player_lines(rows)
    confidence, coverage = _coverage(sport, lines, event, context, now)
    result = {**base, "confidence": confidence, "coverage": coverage, "line_count": len(lines)}
    if confidence == "INSUFFICIENT":
        return {
            **result, "status": "UNAVAILABLE", "display_status": "Insufficient coverage",
            "reason": "insufficient_verified_prop_coverage",
        }
    away, home = event["away_team"], event["home_team"]
    if sport in {"nfl", "ncaaf"}:
        away_mean, away_components = _football_score(lines, away)
        home_mean, home_components = _football_score(lines, home)
    else:
        away_mean, away_components = _mlb_score(lines, away, home)
        home_mean, home_components = _mlb_score(lines, home, away)
        if away_mean is None or home_mean is None:
            return {
                **result, "status": "UNAVAILABLE", "display_status": "Insufficient coverage",
                "reason": "missing_scoring_components", "confidence": "INSUFFICIENT",
            }
    result.update({
        "status": "AVAILABLE", "away_score": int(round(away_mean)), "home_score": int(round(home_mean)),
        "display_status": f"Props-Only · {confidence.title()}",
        "away_mean": round(away_mean, 2), "home_mean": round(home_mean, 2),
        "components": {away: away_components, home: home_components},
        "players": {team: coverage["teams"][team]["players"] for team in (away, home)},
        "families": sorted({line["family"] for line in lines}),
        "oldest_observation_age_minutes": coverage["age_minutes"],
        "score_distribution": {
            "away_sd": 6.8 if sport in {"nfl", "ncaaf"} else 2.1,
            "home_sd": 6.8 if sport in {"nfl", "ncaaf"} else 2.1,
        },
        "anchors": strongest_anchors(sport, lines, away, home),
    })
    # Full line-level audit remains private; public UI receives compact evidence.
    result["audit_hash"] = hashlib.sha256(json.dumps(lines, sort_keys=True, default=str).encode()).hexdigest()
    result["_private_lines"] = lines
    return result


def public_projection(projection):
    hidden = {"_private_lines", "components"}
    return {key: value for key, value in projection.items() if key not in hidden}
