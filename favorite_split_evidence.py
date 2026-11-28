"""Selected-side evidence for the existing private Favorite lifecycle logs."""

import json

import pandas as pd

from red_fox_favorite import (
    CFB_HIGH_MONEY, CFB_HIGH_MAX_TICKETS, CFB_HIGH_MIN_GAP, CFB_PUBLIC_FOLLOW,
    CFB_PUBLIC_MIN_MONEY, CFB_PUBLIC_MIN_TICKETS, CLOSING_ADDITION_MAX_SPLIT, CONFIG,
    MLB_STATIC_PATH, NBA_PUBLIC_PATH, NBA_PUBLIC_MIN_MONEY, NBA_PUBLIC_MIN_TICKETS,
    NOV24_VERSION, _basketball_moneyline_alignment, _corresponding_moneyline_state,
    _line_value, _number, _side_identity, _sides, _truthy, ufc_market_proof, ufc_row_proof,
)

SPLIT_FIELDS = ("tickets", "money")
MARKET_FIELDS = (
    "flagged_side", "open_line", "current_line", "source_latest_at", "reaction",
    "response_direction", "path", "data_badge", "observation_count", "line_move_abs", "price_move_pct",
    "return_toward_open", "active_worsening_reversal", "whipsaw_retention_ratio",
)


def _json_object(value):
    try:
        obj = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return obj if isinstance(obj, dict) else {}


def matchup_rows(board):
    if board.empty or not {"sport", "game_id"}.issubset(board):
        return {}
    return {key: group for key, group in board.groupby(["sport", "game_id"], sort=False)}


def _quote(row, team):
    if row is None or not team:
        return None
    pair = _sides(row)
    if len(pair) != 2 or not all(isinstance(side, dict) for side in pair):
        return None
    matches = [side for side in pair if _side_identity(side.get("flagged_side")) == team]
    if len(matches) != 1:
        return None
    side = matches[0]
    other = next(item for item in pair if item is not side)
    if not _side_identity(other.get("flagged_side")):
        return None
    return {**{key: side[key] for key in MARKET_FIELDS if key in side},
            "bets_pct": _percent(side.get("bets_pct")), "money_pct": _percent(side.get("money_pct")),
            "opponent_bets_pct": _percent(other.get("bets_pct")),
            "opponent_money_pct": _percent(other.get("money_pct")),
            "supported_side": str(row.get("supported_side", "") or ""),
            "cross_market_mismatch": _truthy(row.get("cross_market_mismatch")),
            "cross_market_opener_mismatch": _truthy(row.get("cross_market_opener_mismatch_verified"))}


def _percent(value):
    number = _number(value)
    return number if number is not None and 0 <= number <= 100 else None


def _split_checks(sport, market, pathway, version, quotes, *, closing_add, routed, prior_profile):
    """Describe the lane's existing split thresholds without evaluating eligibility."""
    if pathway == MLB_STATIC_PATH or pathway == "genuine_crossover":
        return [], None
    gap = None
    if pathway == NBA_PUBLIC_PATH:
        bounds = ((NBA_PUBLIC_MIN_TICKETS, None), (NBA_PUBLIC_MIN_MONEY, None))
    elif pathway == CFB_PUBLIC_FOLLOW:
        bounds = ((CFB_PUBLIC_MIN_TICKETS, None), (CFB_PUBLIC_MIN_MONEY, None))
    elif pathway == "public_backed_moneyline":
        from nhl_release_policy import PUBLIC_MIN_MONEY, PUBLIC_MIN_TICKETS
        bounds = ((PUBLIC_MIN_TICKETS, None), (PUBLIC_MIN_MONEY, None))
    elif pathway == CFB_HIGH_MONEY:
        bounds = ((None, CFB_HIGH_MAX_TICKETS), (None, None))
        gap = CFB_HIGH_MIN_GAP
    elif pathway == "low_ticket_high_money_contrarian":
        if version == NOV24_VERSION and sport == "nhl":
            bounds = ((None, CONFIG.low_support_max), (None, None))
        else:
            bounds = ((None, CONFIG.ncaaf_sharp_money_max_bets),
                      (CONFIG.ncaaf_sharp_money_min, CONFIG.ncaaf_sharp_money_max))
        gap = CONFIG.ncaaf_sharp_money_min_gap
    elif pathway == "moderate_support_follow":
        bounds = ((CONFIG.moderate_support_min, CONFIG.moderate_support_max),) * 2
    elif pathway in {"low_support_contrarian", "low_support_freeze"}:
        cap = CONFIG.low_support_max
        # The NBA/NCAAB translated expression inherits the qualifying spread evidence.
        if routed:
            market = "SPREAD"
        if sport == "nfl" and market == "SPREAD" and (
            prior_profile.get("extended_nfl_spread")
        ):
            bounds = ((None, CONFIG.nfl_extended_spread_max_bets),
                      (None, CONFIG.nfl_extended_spread_max_money))
        else:
            bounds = ((None, cap),) * 2
    else:
        return [], None

    checks = [(market, *bounds)]
    if market == "MONEYLINE" and sport in {"nfl", "ncaaf", "cfb"}:
        spread = _line_value((quotes.get("SPREAD") or {}).get("current_line", ""), "SPREAD")
        if pathway == "low_ticket_high_money_contrarian":
            checks.append(("SPREAD", *bounds))
        else:
            short = spread is not None and -3 < spread < 0
            checks.extend([
                ("MONEYLINE", (None, CONFIG.football_favorite_max_bets),
                 (None, CONFIG.football_short_home_max_money if short else CONFIG.football_favorite_max_money)),
                ("SPREAD", (None, None),
                 (None, CONFIG.football_short_home_max_spread_money if short else CONFIG.football_favorite_max_spread_money)),
            ])
    if closing_add:
        checks.append((market, (None, CLOSING_ADDITION_MAX_SPLIT), (None, CLOSING_ADDITION_MAX_SPLIT)))
    return checks, gap


def _breach(quotes, checks):
    failed = set()
    unknown = set()
    for market, *bounds in checks:
        quote = quotes.get(market) or {}
        for field, value, (minimum, maximum) in zip(
            SPLIT_FIELDS, (quote.get("bets_pct"), quote.get("money_pct")), bounds
        ):
            if minimum is None and maximum is None:
                continue
            if value is None:
                unknown.add(field)
            elif (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
                failed.add(field)
    if (unknown - failed):
        kind = "unavailable"
    elif failed == set(SPLIT_FIELDS):
        kind = "both"
    elif failed:
        kind = "ticket_only" if "tickets" in failed else "money_only"
    else:
        kind = "none" if checks else "not_applicable"
    return {"split_breach": kind, "breached_fields": sorted(failed),
            "unavailable_split_fields": sorted(unknown)}


def capture(row, identity, game_rows, *, phase, closing_add=False, market_available=True):
    """Persist the failing quote as observed, without re-running decision gates."""
    prior_evidence = _json_object(identity.get("decision_evidence"))
    pathway = identity.get("favorite_pathway") or prior_evidence.get("pathway", "")
    version = identity.get("favorite_rule_version", "") or prior_evidence.get("rule_version", "")
    name = identity.get("favorite_side") or prior_evidence.get("selected_side", "")
    team = _side_identity(name)
    sport, market = row["sport"], row["market_display"]
    quotes = {market: _quote(row, team) if market_available else None}
    if market_available and game_rows is not None and market in {"SPREAD", "MONEYLINE"}:
        other = "MONEYLINE" if market == "SPREAD" else "SPREAD"
        paired = game_rows.loc[game_rows["market_display"].eq(other)]
        if len(paired) == 1:
            quotes[other] = _quote(paired.iloc[0], team)
    split_profile = prior_evidence.get("split_profile") or {
        "extended_nfl_spread": sport == "nfl" and market == "SPREAD"
        and CONFIG.spread_max < (_line_value((quotes.get(market) or {}).get("current_line", ""), market) or 0)
        <= CONFIG.nfl_positive_spread_max,
        "routed_from_spread": _json_object(identity.get("favorite_supporting_evidence") or identity.get("supporting_evidence")).get("basketball_expression") == "moneyline",
        "closing_addition": closing_add,
    }
    checks, gap = _split_checks(sport, market, pathway, version, quotes,
                                closing_add=split_profile["closing_addition"],
                                routed=split_profile["routed_from_spread"], prior_profile=split_profile)
    selected = quotes[market] or {}
    moves = {"state": row.get("favorite_cross_market_state", ""),
             "mismatch": _truthy(row.get("cross_market_mismatch")),
             "opener_mismatch": _truthy(row.get("cross_market_opener_mismatch_verified"))}
    spread, moneyline = quotes.get("SPREAD"), quotes.get("MONEYLINE")
    if not moves["state"] and spread and moneyline:
        if sport in {"nba", "ncaab", "cbb"}:
            moves["state"] = _basketball_moneyline_alignment(moneyline)
        else:
            moves["state"] = _corresponding_moneyline_state(spread, game_rows)
    gap_values = [q["money_pct"] - q["bets_pct"] for check_market in dict.fromkeys(c[0] for c in checks)
                  if (q := quotes.get(check_market))
                  and q["money_pct"] is not None and q["bets_pct"] is not None]
    outcome = {"archive": "final_scores_history.csv", "game_id": str(row["game_id"])}
    if sport == "ufc":
        proof = (ufc_market_proof(row) or prior_evidence.get("bout", {})
                 or ufc_row_proof(identity, "supporting_evidence") or ufc_row_proof(identity))
        outcome.update(archive="var/ufc-integrity/bouts.json", bout_id=proof.get("bout_id", ""))
    else:
        proof = {}
    result = {**_breach(quotes, checks), "phase": phase, "pathway": pathway, "rule_version": version,
              "selected_side": name, "source_observed_at": selected.get("source_latest_at", ""),
              "quotes": quotes, "cross_market": moves, "split_checks": checks, "split_profile": split_profile,
              "market_available": market_available,
              "minimum_money_minus_tickets": gap,
              "gap_breached": any(value < gap for value in gap_values) if gap is not None and gap_values else None,
              "gate_reason": str(row.get("_favorite_gate_reason", "") or ""),
              "market_decision_reason": str(row.get("favorite_reason", "") or ""),
              "outcome_lookup": outcome}
    if sport in {"nba", "nhl"} and phase == "removal":
        result["split_breach_margins"] = [
            {"market": checked, "field": field, "observed": value, "limit": bound,
             "operator": operator, "percentage_points": direction * (value - bound)}
            for checked, *bounds in checks
            for field, limits in zip(SPLIT_FIELDS, bounds)
            if (value := (quotes.get(checked) or {}).get("bets_pct" if field == "tickets" else "money_pct")) is not None
            for bound, direction, operator in zip(limits, (-1, 1), ("minimum", "maximum"))
            if bound is not None and direction * (value - bound) > 0
        ]
    if not pathway:
        result["split_breach"] = "unavailable"
    if proof:
        result["bout"] = {key: proof[key] for key in ("bout_id", "card_id", "athlete_ids", "pair") if key in proof}
    return result


def annotate_reviews(records, history, games):
    if not records:
        return
    candidate_games = {(r["sport"], r["game_id"]) for r in records if r["review_action"] == "addition"}
    seen = set()
    if candidate_games and not history.empty:
        confirmed = history.loc[history["favorite_state"].eq("qualified"), ["sport", "game_id", "favorite_side"]]
        for row in confirmed.itertuples(index=False):
            if (row.sport, row.game_id) in candidate_games:
                seen.add((row.sport, row.game_id, _side_identity(row.favorite_side)))
    for record in records:
        actual, identity = record.pop("_observation"), record.pop("_candidate")
        name = identity.get("favorite_side") or _json_object(identity.get("decision_evidence")).get("selected_side")
        game_key = (record["sport"], record["game_id"])
        if record["review_action"] == "removal":
            phase = "removal"
        elif (*game_key, _side_identity(name)) in seen:
            phase = "reinstatement"
        else:
            phase = "initial_confirmation"
        closing_add = record["review_action"] == "addition" and record["reason"].startswith("Material closing addition")
        details = capture(actual, identity, games.get(game_key), phase=phase, closing_add=closing_add)
        if record["review_state"] == "cleared":
            details["pending_review_at"] = identity.get("recorded_at", "")
            details["pending_source_at"] = identity.get("source_observed_at", "")
        record["decision_evidence"] = json.dumps(details, separators=(",", ":"), default=str)
