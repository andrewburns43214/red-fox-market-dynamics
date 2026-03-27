"""
Reaction-based scoring engine.

Pure function only.
No imports from the existing project.
No I/O.
No side effects.
"""

from typing import Any, Dict


def classify_reaction_market(
    market_rows: list[Dict[str, Any]],
    evaluated_side: str,
    pressure_side: str | None,
) -> Dict[str, Any]:
    """
    Market-level semantic classifier used by the scoring spec fixtures.

    This classifier is intentionally spec-driven and does not attempt to
    calculate confidence weights. It answers:
      - which semantic pattern applies
      - what signal class it belongs to
      - who owns the signal relative to the pressure side
      - whether the pattern is actionable at the decision layer
    """
    rows = market_rows or []
    evaluated = _find_market_row(rows, evaluated_side)
    if evaluated is None:
        raise ValueError(f"evaluated_side not found in market_rows: {evaluated_side!r}")

    pressure = _find_market_row(rows, pressure_side) if pressure_side else None
    pressure_row = pressure if pressure is not None else evaluated

    if _bool(evaluated.get("market_stale")):
        return {
            "reaction_state": "STALE",
            "signal_class": "directional_price_opportunity",
            "owning_side": "self",
            "decision": "LEAN",
        }

    freeze_subtype = _upper(pressure_row.get("freeze_subtype_candidate"))
    if freeze_subtype:
        return _classify_freeze_subtype(freeze_subtype)

    effective_move_mag = abs(_num(evaluated.get("effective_move_mag")))
    line_dir_changes = int(_num(evaluated.get("line_dir_changes")))
    move_dir = _row_move_dir(evaluated)

    if line_dir_changes >= 2 and effective_move_mag >= 0.5:
        return {
            "reaction_state": "BUYBACK",
            "signal_class": "non_directional_descriptive",
            "owning_side": "none",
            "decision": "NO_BET",
        }

    if pressure_side:
        if _text(evaluated.get("side")) == _text(pressure_side):
            if move_dir == 1 and effective_move_mag >= 0.5:
                return {
                    "reaction_state": "FOLLOW",
                    "signal_class": "directional_conviction",
                    "owning_side": "self",
                    "decision": "LEAN",
                }
        else:
            if move_dir == 1 and effective_move_mag >= 0.5:
                return {
                    "reaction_state": "FADE",
                    "signal_class": "directional_conviction",
                    "owning_side": "opposite",
                    "decision": "LEAN",
                }

    if not pressure_side and move_dir == 1 and effective_move_mag >= 0.5:
        return {
            "reaction_state": "INITIATED",
            "signal_class": "directional_conviction",
            "owning_side": "self",
            "decision": "LEAN",
        }

    return {
        "reaction_state": "NOISE",
        "signal_class": "non_directional_descriptive",
        "owning_side": "none",
        "decision": "NO_BET",
    }


def classify_reaction_live(
    row: Dict[str, Any],
    market_rows: list[Dict[str, Any]] | None = None,
    evaluated_side: str | None = None,
    pressure_side: str | None = None,
) -> Dict[str, Any]:
    """
    Live semantic adapter.

    Uses market-level semantic classification when true same-market context is
    available. Otherwise degrades gracefully to a conservative row-level fallback
    without fabricating semantic certainty.
    """
    rows = _derive_live_market_rows(market_rows or [], pressure_side)
    evaluated = _text(evaluated_side) or _text(row.get("side"))

    if _has_market_context(rows, evaluated):
        semantic = classify_reaction_market(
            market_rows=rows,
            evaluated_side=evaluated,
            pressure_side=pressure_side,
        )
        semantic["semantic_source"] = "market_context"
        return _prefix_semantic_fields(semantic)

    coarse = score_reaction(row)
    semantic = _semantic_from_row_fallback(coarse)
    semantic["semantic_source"] = "row_fallback"
    return _prefix_semantic_fields(semantic)


def score_reaction(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score one normalized side row from observed market behavior only.

    Expected normalized row fields when available:
      bets_pct
      money_pct
      market_display
      side
      open_line_val
      current_line_val
      prev_line_val
      open_odds
      current_odds
      prev_odds
      line_move_open
      line_move_prev
      odds_move_open
      odds_move_prev
      effective_move_mag
      timing_bucket
      line_dir_changes
      line_settled_ticks
      line_last_dir
      line_max_move
      last_score

    Returns:
      {
        "reaction_state": "FREEZE|FOLLOW|FADE|INITIATED|BUYBACK|STALE|NOISE",
        "reaction_score": float,
        "decision": "NO_BET|LEAN|BET",
        "reason": str
      }
    """
    market = _upper(row.get("market_display"))
    sport = _text(row.get("sport")).lower()
    side = _text(row.get("side"))
    timing_bucket = _upper(row.get("timing_bucket"))

    bets_pct = _num(row.get("bets_pct"))
    money_pct = _num(row.get("money_pct"))
    divergence = money_pct - bets_pct

    current_line_val = _num_or_none(row.get("current_line_val"))
    open_line_val = _num_or_none(row.get("open_line_val"))
    prev_line_val = _num_or_none(row.get("prev_line_val"))

    current_odds = _num_or_none(row.get("current_odds"))
    open_odds = _num_or_none(row.get("open_odds"))
    prev_odds = _num_or_none(row.get("prev_odds"))

    line_move_open = _num(row.get("line_move_open"))
    line_move_prev = _num(row.get("line_move_prev"))
    odds_move_open = _num(row.get("odds_move_open"))
    odds_move_prev = _num(row.get("odds_move_prev"))
    effective_move_mag = _fallback_effective_move(
        row.get("effective_move_mag"),
        market,
        line_move_open,
        odds_move_open,
    )

    line_dir_changes = int(_num(row.get("line_dir_changes")))
    line_settled_ticks = int(_num(row.get("line_settled_ticks")))
    line_last_dir = int(_num(row.get("line_last_dir")))
    line_max_move = _num(row.get("line_max_move"))

    public_side = _public_side(bets_pct, money_pct, divergence)
    move_dir = _move_toward_side(
        sport=sport,
        market=market,
        side=side,
        open_line_val=open_line_val,
        current_line_val=current_line_val,
        prev_line_val=prev_line_val,
        open_odds=open_odds,
        current_odds=current_odds,
        prev_odds=prev_odds,
        line_move_open=line_move_open,
        line_move_prev=line_move_prev,
        odds_move_open=odds_move_open,
        odds_move_prev=odds_move_prev,
    )

    persistence = _persistence_bucket(
        effective_move_mag=effective_move_mag,
        line_settled_ticks=line_settled_ticks,
        line_dir_changes=line_dir_changes,
        line_max_move=line_max_move,
    )

    stale_flag = _is_stale(
        effective_move_mag=effective_move_mag,
        public_side=public_side,
        move_dir=move_dir,
        line_settled_ticks=line_settled_ticks,
        line_dir_changes=line_dir_changes,
        timing_bucket=timing_bucket,
    )

    state = _classify_state(
        public_side=public_side,
        move_dir=move_dir,
        effective_move_mag=effective_move_mag,
        bets_pct=bets_pct,
        money_pct=money_pct,
        divergence=divergence,
        persistence=persistence,
        line_dir_changes=line_dir_changes,
        stale_flag=stale_flag,
    )
    state = _validate_state_for_side(
        state=state,
        public_side=public_side,
        move_dir=move_dir,
    )
    freeze_subtype = _upper(
        row.get("freeze_subtype_candidate") or row.get("semantic_reaction_state")
    )
    if freeze_subtype == "FREEZE_RESISTANCE":
        state = "FREEZE_RESISTANCE"

    score = 50.0
    reasons = []

    state_base = {
        "FREEZE": 18.0,
        "FREEZE_RESISTANCE": 10.0,
        "FOLLOW": 10.0,
        # FADE is an anti-signal for the displayed side and should depress confidence.
        "FADE": -12.0,
        "INITIATED": 15.0,
        "BUYBACK": -8.0,
        "STALE": 16.0,
        "NOISE": -10.0,
    }
    score += state_base[state]
    reasons.append(_state_reason(state))

    if state == "FREEZE_RESISTANCE":
        score += 10.0
        reasons.append("meaningful pressure held firm")

        if money_pct >= 75:
            score += 6.0
            reasons.append("heavy money pressure absorbed")

        if bets_pct >= 70:
            score += 4.0
            reasons.append("heavy ticket pressure absorbed")

        if line_settled_ticks >= 3:
            score += 6.0
            reasons.append("held across multiple settled ticks")

        if line_settled_ticks >= 5:
            score += 4.0
            reasons.append("extended hold stability")

        if odds_move_open > 0:
            score += 4.0
            reasons.append("juice moved against public side")

        if line_dir_changes >= 1:
            score -= 8.0
            reasons.append("resistance lost stability")

    if divergence >= 25:
        score += 8.0
        reasons.append("large money/bets divergence")
    elif divergence >= 15:
        score += 5.0
        reasons.append("meaningful money/bets divergence")
    elif divergence <= -15:
        score -= 6.0
        reasons.append("money trails bets")

    if state != "FREEZE_RESISTANCE":
        if effective_move_mag >= 1.5:
            score += 8.0
            reasons.append("strong market move")
        elif effective_move_mag >= 0.5:
            score += 4.0
            reasons.append("meaningful market move")
        elif effective_move_mag == 0 and public_side != 0:
            score -= 2.0
            reasons.append("no meaningful move")

    if persistence == "STABLE":
        score += 6.0
        reasons.append("move held")
    elif persistence == "FORMING":
        score += 2.0
        reasons.append("move forming")
    elif persistence == "REVERSING":
        score -= 10.0
        reasons.append("move reversing")
    else:
        score -= 4.0
        reasons.append("weak persistence")

    timing_adj = {
        "EARLY": -2.0,
        "MID": 3.0,
        "LATE": 1.0,
        "LIVE": -20.0,
        "UNKNOWN": -1.0,
    }
    score += timing_adj.get(timing_bucket, -1.0)
    if timing_bucket:
        reasons.append("timing " + timing_bucket.lower())

    if market == "MONEYLINE" and current_odds is not None:
        if current_odds >= 250:
            score -= 8.0
            reasons.append("long moneyline price")
        elif current_odds >= 180:
            score -= 4.0
            reasons.append("elevated underdog price")
        elif current_odds <= -220:
            score -= 5.0
            reasons.append("expensive favorite price")

    if line_dir_changes >= 2:
        score -= 6.0
        reasons.append("direction unstable")

    if stale_flag:
        reasons.append("book appears stale")

    if score > 72.0:
        if line_settled_ticks < 2 or line_dir_changes > 0:
            score = 72.0
    if score > 75.0 and state not in ("FOLLOW", "INITIATED"):
        score = 75.0

    score = round(_clamp(score, 0.0, 100.0), 1)
    decision = _decision_from_score(score, state, persistence, timing_bucket, effective_move_mag)

    return {
        "reaction_state": state,
        "reaction_score": score,
        "decision": decision,
        "reason": _format_reason(reasons),
    }


def _classify_state(
    public_side: int,
    move_dir: int,
    effective_move_mag: float,
    bets_pct: float,
    money_pct: float,
    divergence: float,
    persistence: str,
    line_dir_changes: int,
    stale_flag: bool,
) -> str:
    if stale_flag and public_side != 0:
        return "STALE"

    if line_dir_changes >= 2 and effective_move_mag >= 0.5:
        return "BUYBACK"

    if public_side != 0:
        if effective_move_mag < 0.5:
            if bets_pct >= 60 or money_pct >= 60 or divergence >= 12:
                return "FREEZE"
            return "NOISE"
        # move_dir is always from the perspective of the displayed side:
        # +1 means the market moved toward this side, -1 away from this side.
        if move_dir == 1:
            return "FOLLOW"
        if move_dir == -1:
            return "FADE"

    if public_side == 0 and effective_move_mag >= 0.5:
        return "INITIATED"

    if persistence == "REVERSING":
        return "BUYBACK"

    return "NOISE"


def _classify_freeze_subtype(freeze_subtype: str) -> Dict[str, Any]:
    mapping = {
        "FREEZE_RESISTANCE": {
            "reaction_state": "FREEZE_RESISTANCE",
            "signal_class": "directional_conviction",
            "owning_side": "opposite",
            "decision": "LEAN",
        },
        "FREEZE_BALANCED": {
            "reaction_state": "FREEZE_BALANCED",
            "signal_class": "non_directional_descriptive",
            "owning_side": "none",
            "decision": "NO_BET",
        },
        "FREEZE_KEY_NUMBER": {
            "reaction_state": "FREEZE_KEY_NUMBER",
            "signal_class": "non_directional_descriptive",
            "owning_side": "none",
            "decision": "NO_BET",
        },
        "FREEZE_STALE": {
            "reaction_state": "FREEZE_STALE",
            "signal_class": "directional_price_opportunity",
            "owning_side": "self",
            "decision": "LEAN",
        },
        "FREEZE_WEAK": {
            "reaction_state": "FREEZE_WEAK",
            "signal_class": "non_directional_descriptive",
            "owning_side": "none",
            "decision": "NO_BET",
        },
    }
    if freeze_subtype not in mapping:
        raise ValueError(f"unknown freeze subtype: {freeze_subtype}")
    return mapping[freeze_subtype]


def _derive_live_market_rows(
    rows: list[Dict[str, Any]],
    pressure_side: str | None,
) -> list[Dict[str, Any]]:
    derived = [dict(r) for r in (rows or [])]
    if not derived or not pressure_side:
        return derived

    pressure_row = _find_market_row(derived, pressure_side)
    if pressure_row is None:
        return derived

    if _upper(pressure_row.get("freeze_subtype_candidate")):
        return derived

    meaningful_pressure = _meaningful_pressure(pressure_row)
    balanced_counteraction = _balanced_counteraction(pressure_row)
    key_number_pinned = _key_number_pinned(pressure_row)
    market_stale = _bool(pressure_row.get("market_stale"))
    sport = _text(pressure_row.get("sport")).lower()
    market = _upper(pressure_row.get("market_display"))
    current_line_val = _num_or_none(pressure_row.get("current_line_val"))
    effective_move_mag = abs(_num(pressure_row.get("effective_move_mag")))
    if effective_move_mag == 0:
        effective_move_mag = _fallback_effective_move(
            pressure_row.get("effective_move_mag"),
            _upper(pressure_row.get("market_display")),
            _num(pressure_row.get("line_move_open")),
            _num(pressure_row.get("odds_move_open")),
        )
    runline_or_puckline = (
        market == "SPREAD"
        and sport in {"mlb", "nhl"}
        and current_line_val is not None
        and abs(current_line_val) == 1.5
    )
    if runline_or_puckline:
        effective_move_mag = abs(_num(pressure_row.get("odds_move_open")))
    move_dir = _row_move_dir(pressure_row)

    freeze_subtype = None
    if move_dir == 0 and effective_move_mag < 0.5:
        if market_stale:
            freeze_subtype = "FREEZE_STALE"
        elif key_number_pinned:
            freeze_subtype = "FREEZE_KEY_NUMBER"
        elif balanced_counteraction:
            freeze_subtype = "FREEZE_BALANCED"
        elif meaningful_pressure and not runline_or_puckline:
            freeze_subtype = "FREEZE_RESISTANCE"
        else:
            freeze_subtype = "FREEZE_WEAK"

    pressure_row["meaningful_pressure"] = meaningful_pressure
    pressure_row["balanced_counteraction"] = balanced_counteraction
    pressure_row["key_number_pinned"] = key_number_pinned
    pressure_row["market_stale"] = market_stale
    pressure_row["freeze_subtype_candidate"] = freeze_subtype
    return derived


def _meaningful_pressure(row: Dict[str, Any]) -> bool:
    bets_pct = _num(row.get("bets_pct"))
    money_pct = _num(row.get("money_pct"))
    divergence = money_pct - bets_pct
    return bets_pct >= 60 or money_pct >= 65 or divergence >= 15


def _balanced_counteraction(row: Dict[str, Any]) -> bool:
    bets_pct = _num(row.get("bets_pct"))
    money_pct = _num(row.get("money_pct"))
    divergence = money_pct - bets_pct
    return bets_pct >= 60 and money_pct <= 55 and divergence <= -8


def _key_number_pinned(row: Dict[str, Any]) -> bool:
    if _upper(row.get("market_display")) != "SPREAD":
        return False
    if _text(row.get("sport")).lower() not in {"nfl", "ncaafb"}:
        return False

    current_line_val = _num_or_none(row.get("current_line_val"))
    open_line_val = _num_or_none(row.get("open_line_val"))
    if current_line_val is None:
        return False

    current_abs = abs(current_line_val)
    open_abs = abs(open_line_val) if open_line_val is not None else current_abs
    for key in (3.0, 7.0, 10.0, 14.0, 17.0):
        if abs(current_abs - key) <= 0.05 and abs(open_abs - key) <= 0.05:
            return True
    return False


def _semantic_from_row_fallback(coarse: Dict[str, Any]) -> Dict[str, Any]:
    state = _upper(coarse.get("reaction_state"))
    decision = _upper(coarse.get("decision")) or "NO_BET"

    # Fallback mode must avoid false semantic certainty.
    if state == "FOLLOW":
        return {
            "reaction_state": state,
            "signal_class": "directional_conviction",
            "owning_side": "self",
            "decision": decision,
        }
    if state == "INITIATED":
        return {
            "reaction_state": state,
            "signal_class": "directional_conviction",
            "owning_side": "self",
            "decision": decision,
        }
    if state == "BUYBACK":
        return {
            "reaction_state": state,
            "signal_class": "non_directional_descriptive",
            "owning_side": "none",
            "decision": decision,
        }
    if state == "NOISE":
        return {
            "reaction_state": state,
            "signal_class": "non_directional_descriptive",
            "owning_side": "none",
            "decision": decision,
        }

    return {
        "reaction_state": state,
        "signal_class": "",
        "owning_side": "none",
        "decision": decision,
    }


def _prefix_semantic_fields(semantic: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "semantic_reaction_state": _text(semantic.get("reaction_state")),
        "semantic_signal_class": _text(semantic.get("signal_class")),
        "semantic_owning_side": _text(semantic.get("owning_side")) or "none",
        "semantic_decision": _text(semantic.get("decision")) or "NO_BET",
        "semantic_source": _text(semantic.get("semantic_source")),
    }


def _validate_state_for_side(
    state: str,
    public_side: int,
    move_dir: int,
) -> str:
    # If the move is not actually toward the displayed side, don't keep
    # positive directional labels on that side.
    if state in ("FOLLOW", "INITIATED") and move_dir != 1:
        return "NOISE"
    # FADE requires a clear move against the public side.
    if state == "FADE" and (public_side == 0 or move_dir != -1):
        return "NOISE"
    return state


def _decision_from_score(
    score: float,
    state: str,
    persistence: str,
    timing_bucket: str,
    effective_move_mag: float,
) -> str:
    if timing_bucket == "LIVE":
        return "NO_BET"

    # Directional anti-signals should never produce a recommendation
    # on the same displayed side.
    if state in ("FADE", "BUYBACK", "NOISE"):
        return "NO_BET"

    if persistence not in ("FORMING", "STABLE"):
        return "NO_BET"

    if effective_move_mag < 0.35 and state not in ("FREEZE", "STALE"):
        return "NO_BET"

    if state == "FADE" and effective_move_mag < 0.5:
        return "NO_BET"

    if score >= 68.0:
        return "BET"
    if score >= 60.0:
        return "LEAN"
    return "NO_BET"


def _public_side(bets_pct: float, money_pct: float, divergence: float) -> int:
    if bets_pct >= 60 or money_pct >= 60:
        return 1
    if bets_pct <= 40 and money_pct <= 40:
        return -1
    if divergence >= 12 and bets_pct <= 55:
        return 1
    if divergence <= -12 and bets_pct >= 45:
        return -1
    return 0


def _move_toward_side(
    sport: str,
    market: str,
    side: str,
    open_line_val: float | None,
    current_line_val: float | None,
    prev_line_val: float | None,
    open_odds: float | None,
    current_odds: float | None,
    prev_odds: float | None,
    line_move_open: float,
    line_move_prev: float,
    odds_move_open: float,
    odds_move_prev: float,
) -> int:
    if market == "MONEYLINE":
        open_prob = _american_implied_prob(open_odds)
        current_prob = _american_implied_prob(current_odds)
        prev_prob = _american_implied_prob(prev_odds)
        if open_prob is not None and current_prob is not None:
            if current_prob > open_prob:
                return 1
            if current_prob < open_prob:
                return -1
        if prev_prob is not None and current_prob is not None:
            if current_prob > prev_prob:
                return 1
            if current_prob < prev_prob:
                return -1
        return 0

    if market == "TOTAL":
        label = side.upper()
        if open_line_val is None or current_line_val is None:
            return 0
        if "OVER" in label:
            if current_line_val > open_line_val:
                return 1
            if current_line_val < open_line_val:
                return -1
        elif "UNDER" in label:
            if current_line_val < open_line_val:
                return 1
            if current_line_val > open_line_val:
                return -1
        return 0

    if market == "SPREAD":
        if open_line_val is None or current_line_val is None:
            if sport == "mlb":
                if odds_move_open < 0:
                    return 1
                if odds_move_open > 0:
                    return -1
                if odds_move_prev < 0:
                    return 1
                if odds_move_prev > 0:
                    return -1
            return 0
        side_line = _extract_side_line(side)
        if side_line is None:
            if sport == "mlb":
                if odds_move_open < 0:
                    return 1
                if odds_move_open > 0:
                    return -1
                if odds_move_prev < 0:
                    return 1
                if odds_move_prev > 0:
                    return -1
            return 0
        if side_line < 0:
            if current_line_val < open_line_val:
                return 1
            if current_line_val > open_line_val:
                return -1
        elif side_line > 0:
            if current_line_val < open_line_val:
                return 1
            if current_line_val > open_line_val:
                return -1
        if sport == "mlb":
            if odds_move_open < 0:
                return 1
            if odds_move_open > 0:
                return -1
            if odds_move_prev < 0:
                return 1
            if odds_move_prev > 0:
                return -1
        return 0

    if odds_move_open < 0:
        return 1
    if odds_move_open > 0:
        return -1
    if odds_move_prev < 0:
        return 1
    if odds_move_prev > 0:
        return -1
    return 0


def _persistence_bucket(
    effective_move_mag: float,
    line_settled_ticks: int,
    line_dir_changes: int,
    line_max_move: float,
) -> str:
    if line_dir_changes >= 2:
        return "REVERSING"
    if effective_move_mag >= 0.5 and line_settled_ticks >= 2:
        return "STABLE"
    if effective_move_mag >= 0.35:
        return "FORMING"
    return "WEAK"


def _is_stale(
    effective_move_mag: float,
    public_side: int,
    move_dir: int,
    line_settled_ticks: int,
    line_dir_changes: int,
    timing_bucket: str,
) -> bool:
    if timing_bucket == "LIVE":
        return False
    if public_side == 0:
        return False
    if effective_move_mag > 0.2:
        return False
    if move_dir != 0:
        return False
    if line_dir_changes > 0:
        return False
    return line_settled_ticks >= 1


def _state_reason(state: str) -> str:
    return {
        "FREEZE": "public pressure without meaningful move",
        "FREEZE_RESISTANCE": "book held firm against meaningful public pressure",
        "FOLLOW": "book moved with pressure",
        "FADE": "book moved against pressure",
        "INITIATED": "book moved before clear public pressure",
        "BUYBACK": "move reversed after prior action",
        "STALE": "pressure visible while book price held stale",
        "NOISE": "mixed or weak market behavior",
    }[state]


def _format_reason(parts: list[str]) -> str:
    cleaned = []
    seen = set()
    for part in parts:
        part = _text(part)
        if part and part not in seen:
            cleaned.append(part)
            seen.add(part)
    return "; ".join(cleaned) if cleaned else "no clear market reaction"


def _extract_side_line(side: str) -> float | None:
    side = side.strip()
    num = ""
    found = False
    for idx, ch in enumerate(side):
        if ch in "+-" and idx + 1 < len(side) and side[idx + 1].isdigit():
            num = ch
            pos = idx + 1
            while pos < len(side) and (side[pos].isdigit() or side[pos] == "."):
                num += side[pos]
                pos += 1
            found = True
    if not found:
        return None
    try:
        return float(num)
    except Exception:
        return None


def _find_market_row(rows: list[Dict[str, Any]], side: str | None) -> Dict[str, Any] | None:
    target = _text(side)
    if not target:
        return None
    for row in rows:
        if _text(row.get("side")) == target:
            return row
    return None


def _has_market_context(rows: list[Dict[str, Any]], evaluated_side: str) -> bool:
    if not rows or len(rows) < 2:
        return False
    return _find_market_row(rows, evaluated_side) is not None


def _row_move_dir(row: Dict[str, Any]) -> int:
    return _move_toward_side(
        sport=_text(row.get("sport")).lower(),
        market=_upper(row.get("market_display")),
        side=_text(row.get("side")),
        open_line_val=_num_or_none(row.get("open_line_val")),
        current_line_val=_num_or_none(row.get("current_line_val")),
        prev_line_val=_num_or_none(row.get("prev_line_val")),
        open_odds=_num_or_none(row.get("open_odds")),
        current_odds=_num_or_none(row.get("current_odds")),
        prev_odds=_num_or_none(row.get("prev_odds")),
        line_move_open=_num(row.get("line_move_open")),
        line_move_prev=_num(row.get("line_move_prev")),
        odds_move_open=_num(row.get("odds_move_open")),
        odds_move_prev=_num(row.get("odds_move_prev")),
    )


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _text(value).lower()
    return text in ("1", "true", "yes", "y", "on")


def _fallback_effective_move(raw: Any, market: str, line_move_open: float, odds_move_open: float) -> float:
    parsed = _num_or_none(raw)
    if parsed is not None:
        return abs(parsed)

    line_mag = abs(line_move_open)
    odds_mag = abs(odds_move_open)

    if market == "MONEYLINE":
        if odds_mag >= 5:
            return min(3.0, odds_mag / 15.0)
        return 0.0

    if line_mag >= 0.5:
        return line_mag

    if odds_mag >= 5:
        return min(3.0, odds_mag / 15.0)

    return 0.0


def _american_implied_prob(odds: float | None) -> float | None:
    if odds is None:
        return None
    try:
        o = float(odds)
    except Exception:
        return None
    if o == 0:
        return None
    if o < 0:
        return abs(o) / (abs(o) + 100.0)
    return 100.0 / (o + 100.0)


def _num(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        text = str(value).strip()
        if text == "" or text.lower() in ("nan", "none", "null"):
            return 0.0
        return float(text)
    except Exception:
        return 0.0


def _num_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.lower() in ("nan", "none", "null"):
            return None
        return float(text)
    except Exception:
        return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _upper(value: Any) -> str:
    return _text(value).upper()


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value
