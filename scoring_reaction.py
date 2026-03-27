"""
Reaction-based scoring engine.

Pure function only.
No imports from the existing project.
No I/O.
No side effects.
"""

from typing import Any, Dict


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

    score = 50.0
    reasons = []

    state_base = {
        "FREEZE": 18.0,
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

    if divergence >= 25:
        score += 8.0
        reasons.append("large money/bets divergence")
    elif divergence >= 15:
        score += 5.0
        reasons.append("meaningful money/bets divergence")
    elif divergence <= -15:
        score -= 6.0
        reasons.append("money trails bets")

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
