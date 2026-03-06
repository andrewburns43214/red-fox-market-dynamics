"""
Red Fox v3.2 — Certification Layer

Separate from scoring. Answers: "Is this actionable?"
Score answers: "How strong is the signal?"
Certification answers: "Can we act on it?"

Decision hierarchy: STRONG_BET > BET > LEAN > NO_BET
Freeze ledger: append-only, never downgrade.
"""

import engine_config_v3 as C


def certify_decision(row: dict, score: float, net_edge: float,
                     strong_streak: int = 0, peak_score: float = None,
                     last_score: float = None) -> dict:
    """v3.2 certification: STRONG_BET / BET / LEAN / NO_BET.

    Args:
        row: Full row dict with all fields
        score: v3.2 final_score (0-100)
        net_edge: max_side_score - min_side_score for this game/market
        strong_streak: consecutive qualifying runs for STRONG persistence
        peak_score: highest score seen for this row (for stability gate)
        last_score: most recent previous score (for stability check)

    Returns:
        dict with:
            decision: str (STRONG_BET / BET / LEAN / NO_BET)
            blocked_by: str or None (reason STRONG was blocked)
            l1_cap_applied: bool
            strong_eligible: bool
    """
    sport = (row.get("sport") or "").lower()
    market = (row.get("market_display") or row.get("market") or "").upper()
    l1_present = _bool(row.get("l1_available", row.get("l1_present", False)))
    timing_bucket = (row.get("timing_bucket") or "MID").upper()
    path = (row.get("l1_path_behavior") or "UNKNOWN").upper()
    pattern = (row.get("pattern_primary") or "").upper()
    cross_adj = _num(row.get("cross_market_adj", 0))

    # Edge threshold depends on market
    if market == "TOTAL":
        edge_min = C.BET_EDGE_MIN_TOTAL
        strong_edge_min = C.STRONG_EDGE_MIN_TOTAL
    else:
        edge_min = C.BET_EDGE_MIN_SIDES
        strong_edge_min = C.STRONG_EDGE_MIN_SIDES

    # ── NO_BET ──
    if score < C.LEAN_SCORE_MIN:
        return _result("NO_BET", None, False, False)

    # ── LEAN ──
    if score < C.BET_SCORE_MIN or net_edge < edge_min:
        return _result("LEAN", None, False, False)

    # ── BET eligible — check STRONG gates ──
    blocked_by = _check_strong_gates(
        score=score,
        net_edge=net_edge,
        strong_edge_min=strong_edge_min,
        l1_present=l1_present,
        timing_bucket=timing_bucket,
        path=path,
        pattern=pattern,
        cross_adj=cross_adj,
        sport=sport,
        strong_streak=strong_streak,
        peak_score=peak_score,
        last_score=last_score,
    )

    if blocked_by is None:
        return _result("STRONG_BET", None, False, True)
    else:
        return _result("BET", blocked_by, False, False)


def _check_strong_gates(*, score, net_edge, strong_edge_min,
                        l1_present, timing_bucket, path, pattern,
                        cross_adj, sport, strong_streak,
                        peak_score, last_score) -> str:
    """Check all STRONG gates. Returns blocking reason or None if all pass."""

    # Gate 1: Score threshold
    if score < C.STRONG_SCORE_MIN:
        return f"score {score} < {C.STRONG_SCORE_MIN}"

    # Gate 2: Net edge
    if net_edge < strong_edge_min:
        return f"edge {net_edge} < {strong_edge_min}"

    # Gate 3: L1 must be present — no exceptions
    if not l1_present:
        return "L1 absent"

    # Gate 4: Timing ≠ LATE
    if timing_bucket == "LATE":
        return "LATE timing"

    # Gate 5: NCAAB/NCAAF early block
    if sport in C.STRONG_EARLY_BLOCK_SPORTS and timing_bucket == "EARLY":
        return f"{sport.upper()} EARLY blocked"

    # Gate 6: Pattern ≠ PUBLIC_DRIFT
    if pattern == "PUBLIC_DRIFT":
        return "PUBLIC_DRIFT pattern"

    # Gate 7: No cross-market contradiction
    if cross_adj < 0:
        return "cross-market contradiction"

    # Gate 8: Path not REVERSED or OSCILLATED
    if path in C.STRONG_BLOCKED_PATHS:
        return f"path {path}"

    # Gate 9: Persistence streak
    streak_min = C.NCAAB_STREAK_MIN if sport == "ncaab" else C.STRONG_STREAK_MIN
    if strong_streak < streak_min:
        return f"streak {strong_streak} < {streak_min}"

    # Gate 10: Stability — score hasn't collapsed from peak
    if peak_score is not None and last_score is not None:
        stability_delta = (C.NCAAB_STABILITY_DELTA if sport == "ncaab"
                           else C.STRONG_STABILITY_DELTA)
        if last_score < peak_score - stability_delta:
            return f"stability: {last_score} < peak {peak_score} - {stability_delta}"

    return None  # All gates passed


def apply_l1_absent_cap(score: float, row: dict) -> tuple:
    """Apply L1-absent scoring caps. Returns (capped_score, was_capped)."""
    l1_present = _bool(row.get("l1_available", row.get("l1_present", False)))
    if l1_present:
        return score, False

    l2_agreement = _num(row.get("l2_consensus_agreement", 0))
    if l2_agreement < C.L1_ABSENT_L2_WEAK_THRESHOLD:
        if score > C.L1_ABSENT_L2_WEAK_CAP:
            return C.L1_ABSENT_L2_WEAK_CAP, True

    return score, False


def _result(decision, blocked_by, l1_cap, strong_eligible):
    return {
        "decision": decision,
        "blocked_by": blocked_by,
        "l1_cap_applied": l1_cap,
        "strong_eligible": strong_eligible,
    }


def _num(val) -> float:
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    try:
        return bool(val)
    except (ValueError, TypeError):
        return False
