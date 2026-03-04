"""
3-Layer Scoring Model for Red Fox engine v2.0.

Score = 50 (base)
  + L1 contribution (0 to +18): sharp book signal strength
  + L2 contribution (-8 to +10): consensus validation
  + L3 contribution (-10 to +10): DK behavior modifier
  + Pattern bonus (A-F): interaction pattern effects
  + Cross-market: spread-total agreement/contradiction
  Clamped to [floor, cap] by layer_mode

Layer modes:
  L123 (full): cap 100, STRONG eligible
  L13 (no consensus): cap 85, no STRONG
  L23 (no sharp): cap 80, no STRONG
  L3_ONLY: cap 75, no STRONG (current v1.2 behavior)
"""
from engine_config import (
    L1_MAX_CONTRIBUTION,
    L2_MAX_POSITIVE,
    L2_MAX_NEGATIVE,
    L3_MAX_POSITIVE,
    L3_MAX_NEGATIVE,
    SCORE_CAP_L123,
    SCORE_CAP_L13,
    SCORE_CAP_L23,
    SCORE_CAP_L3_ONLY,
    PATTERN_EFFECTS,
    FAST_SNAP_EARLY_BONUS,
    FAST_SNAP_LATE_PENALTY,
    SLOW_GRIND_PENALTY,
    RLM_BETS_THRESHOLD,
    RLM_MONEY_GAP_MIN,
    RLM_L2_AGREEMENT_MIN,
)
from dk_rules import compute_l3_contribution


def _safe_float(val, default=0.0) -> float:
    try:
        return float(val) if val not in ("", None) else default
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0) -> int:
    try:
        return int(float(val)) if val not in ("", None) else default
    except (ValueError, TypeError):
        return default


def _safe_bool(val, default=False) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return default


# ─── LAYER 1 CONTRIBUTION ───

def compute_l1_contribution(row: dict) -> dict:
    """
    Compute Layer 1 (sharp book) contribution: 0 to +18.

    Components:
      - Direction × magnitude (base signal)
      - Agreement multiplier (1-6 books)
      - Limit-weighted confidence
      - Move speed bonus/penalty
      - Stability modifier
      - Key number crossing
    """
    if not _safe_bool(row.get("l1_available")):
        return {"l1_contribution": 0.0, "l1_details": {}}

    direction = _safe_int(row.get("l1_move_dir"))
    if direction == 0:
        return {"l1_contribution": 0.0, "l1_details": {"reason": "no_move"}}

    magnitude = _safe_float(row.get("l1_move_magnitude"))  # 0-1 normalized
    agreement_mult = _safe_float(row.get("l1_agreement_mult"), 1.0)
    stability = _safe_float(row.get("l1_stability"), 0.5)
    speed_label = str(row.get("l1_speed_label", ""))
    key_cross = _safe_bool(row.get("l1_key_number_cross"))
    limit_conf = _safe_float(row.get("l1_limit_confidence"))

    # Base signal: magnitude × agreement
    base = magnitude * agreement_mult

    # Stability modifier (0.5 to 1.2)
    stab_mult = 0.5 + (stability * 0.7)

    # Speed modifier
    hours_to_game = _safe_float(row.get("hours_to_game", row.get("time_to_start_hours", 4)))
    speed_bonus = 0.0
    if speed_label == "FAST_SNAP":
        if hours_to_game > 2:
            speed_bonus = FAST_SNAP_EARLY_BONUS  # +3 early/mid
        else:
            speed_bonus = FAST_SNAP_LATE_PENALTY  # -4 late
    elif speed_label == "SLOW_GRIND":
        speed_bonus = SLOW_GRIND_PENALTY  # -2

    # Key number crossing bonus
    key_bonus = 2.0 if key_cross else 0.0

    # Limit confidence multiplier (higher limits = more confident)
    # Normalize: $10K = 1.0, $50K+ = 1.2
    limit_mult = 1.0
    if limit_conf > 0:
        limit_mult = min(1.0 + (limit_conf / 50000.0) * 0.2, 1.2)

    # Compute raw contribution
    raw = (base * stab_mult * limit_mult * L1_MAX_CONTRIBUTION) + speed_bonus + key_bonus

    # Clamp to [0, L1_MAX_CONTRIBUTION]
    contribution = max(0.0, min(L1_MAX_CONTRIBUTION, raw))

    return {
        "l1_contribution": round(contribution, 2),
        "l1_details": {
            "base": round(base, 3),
            "stability_mult": round(stab_mult, 3),
            "limit_mult": round(limit_mult, 3),
            "speed_bonus": speed_bonus,
            "key_bonus": key_bonus,
            "speed_label": speed_label,
            "raw": round(raw, 2),
        },
    }


# ─── LAYER 2 CONTRIBUTION ───

def compute_l2_contribution(row: dict) -> dict:
    """
    Compute Layer 2 (consensus) contribution: -8 to +10.

    Positive when market confirms L1, negative when it rejects.
    """
    if not _safe_bool(row.get("l2_available")):
        return {"l2_contribution": 0.0, "l2_details": {}}

    agreement = _safe_float(row.get("l2_consensus_agreement"))
    disp_mult = _safe_float(row.get("l2_dispersion_mult"), 1.0)
    disp_trend = str(row.get("l2_dispersion_trend", ""))
    validation = _safe_float(row.get("l2_validation_strength"))

    # Trend bonus
    trend_bonus = 0.0
    if disp_trend == "TIGHTENING":
        trend_bonus = 1.5  # Books converging = confirms move
    elif disp_trend == "WIDENING":
        trend_bonus = -1.5  # Books diverging = move may be noise

    # Base: validation strength scaled
    # High agreement + tight dispersion = strong positive
    # Low agreement + wide dispersion = strong negative
    if agreement >= 0.6:
        # Market confirms
        raw = validation * disp_mult * L2_MAX_POSITIVE + trend_bonus
        contribution = max(0.0, min(L2_MAX_POSITIVE, raw))
    elif agreement <= 0.3:
        # Market rejects
        raw = -(1.0 - validation) * L2_MAX_NEGATIVE * -1 + trend_bonus
        contribution = max(L2_MAX_NEGATIVE, min(0.0, -raw))
    else:
        # Ambiguous
        contribution = trend_bonus * 0.5

    return {
        "l2_contribution": round(contribution, 2),
        "l2_details": {
            "agreement": agreement,
            "disp_mult": disp_mult,
            "trend": disp_trend,
            "trend_bonus": trend_bonus,
            "validation": validation,
        },
    }


# ─── REVERSE LINE MOVEMENT ───

def _detect_rlm(row: dict) -> dict:
    """
    Detect Reverse Line Movement from cross-layer data.

    RLM = public bets heavy on one side + money NOT following +
          sharp books moved + consensus confirms.

    Returns {"detected": bool, "strength": float, "bets_money_gap": float}
    """
    bets_pct = _safe_float(row.get("bets_pct", row.get("dk_bets_pct", 50)))
    money_pct = _safe_float(row.get("money_pct", row.get("dk_money_pct", 50)))
    l1_dir = _safe_int(row.get("l1_move_dir"))
    l1_strength = _safe_float(row.get("l1_sharp_strength"))
    l2_agreement = _safe_float(row.get("l2_consensus_agreement"))

    bets_money_gap = bets_pct - money_pct

    detected = (
        bets_pct >= RLM_BETS_THRESHOLD and
        bets_money_gap >= RLM_MONEY_GAP_MIN and
        l1_dir != 0 and
        l2_agreement >= RLM_L2_AGREEMENT_MIN
    )

    if not detected:
        return {"detected": False, "strength": 0.0, "bets_money_gap": bets_money_gap}

    # RLM strength (0 to 1):
    # - Larger bets-money gap = stronger signal
    # - Stronger sharp move = more confirmed
    # - Higher consensus = more validated
    gap_factor = min(bets_money_gap / 30.0, 1.0)
    strength = gap_factor * max(l1_strength, 0.3) * max(l2_agreement, 0.5)
    strength = min(1.0, strength)

    return {"detected": True, "strength": round(strength, 3), "bets_money_gap": round(bets_money_gap, 1)}


# ─── INTERACTION PATTERNS ───

def detect_interaction_pattern(row: dict) -> dict:
    """
    Detect which interaction pattern (A-F) applies.

    Returns dict with pattern letter, label, and explanation.
    """
    l1_available = _safe_bool(row.get("l1_available"))
    l2_available = _safe_bool(row.get("l2_available"))
    l1_dir = _safe_int(row.get("l1_move_dir"))
    l1_strength = _safe_float(row.get("l1_sharp_strength"))
    l2_agreement = _safe_float(row.get("l2_consensus_agreement"))
    stale = _safe_bool(row.get("l2_stale_price_flag"))
    speed_label = str(row.get("l1_speed_label", ""))

    # DK public direction (from bets%)
    dk_bets = _safe_float(row.get("bets_pct", row.get("dk_bets_pct", 50)))
    dk_public_heavy = dk_bets > 65 or dk_bets < 35  # Public has strong opinion

    hours_to_game = _safe_float(row.get("hours_to_game", row.get("time_to_start_hours", 4)))

    # Pattern F: Late snap (check first — timing overrides everything)
    if l1_available and speed_label == "FAST_SNAP" and hours_to_game < 1:
        return {
            "pattern": "F",
            "label": "LATE_SNAP_WARNING",
            "explanation": "Rapid sharp movement in final hour — information or chaos",
        }

    # Pattern G: Reverse Line Movement
    # Public bets heavy on one side but money + line moving the other way
    if l1_available and l1_dir != 0:
        rlm = _detect_rlm(row)
        if rlm["detected"]:
            return {
                "pattern": "G",
                "label": "REVERSE_LINE_MOVE",
                "explanation": f"Public bets heavy but line moved opposite — sharp money confirmed (gap: {rlm['bets_money_gap']}%)",
                "rlm_strength": rlm["strength"],
                "rlm_gap": rlm["bets_money_gap"],
            }

    # Pattern A: Sharp vs Public (best edge)
    if l1_available and l1_dir != 0 and l2_agreement >= 0.5 and dk_public_heavy:
        # Sharp moved + consensus confirms + public on OTHER side
        return {
            "pattern": "A",
            "label": "SHARP_VS_PUBLIC",
            "explanation": "Sharp books moved, consensus confirms, public betting opposite",
        }

    # Pattern D: Stale price (DK lagging)
    if l1_available and l1_dir != 0 and l2_agreement >= 0.5 and stale:
        return {
            "pattern": "D",
            "label": "STALE_PRICE",
            "explanation": "Sharp and consensus moved, DK line hasn't caught up",
        }

    # Pattern B: Sharp + Public aligned (priced in)
    if l1_available and l1_dir != 0 and l2_agreement >= 0.5 and not dk_public_heavy:
        return {
            "pattern": "B",
            "label": "RETAIL_ALIGNMENT",
            "explanation": "Sharp, consensus, and public all agree — likely priced in",
        }

    # Pattern E: Consensus rejects sharp
    if l1_available and l1_dir != 0 and l2_available and l2_agreement < 0.3:
        return {
            "pattern": "E",
            "label": "CONSENSUS_REJECTS",
            "explanation": "Sharp moved but broader market didn't follow",
        }

    # Pattern C: Public only (no sharp move)
    if (not l1_available or l1_dir == 0) and dk_public_heavy:
        return {
            "pattern": "C",
            "label": "RETAIL_ALIGNMENT",
            "explanation": "No sharp movement, heavy public action — pure retail bias",
        }

    # Default: no clear pattern
    return {
        "pattern": "N",
        "label": "NEUTRAL",
        "explanation": "No strong interaction pattern detected",
    }


# ─── CROSS-MARKET CHECK ───

def spread_total_cross_check(row: dict) -> float:
    """
    Check if spread and total are consistent.

    Spread favors Team A + Total Over = consistent (offensive)
    Spread favors Team A + Total Under = mild contradiction

    Returns bonus/penalty (-2 to +1).
    """
    spread_dir = _safe_float(row.get("dk_spread_dir", row.get("spread_move_dir", 0)))
    total_dir = _safe_float(row.get("dk_total_dir", row.get("total_move_dir", 0)))

    if spread_dir == 0 or total_dir == 0:
        return 0.0

    # Spread positive + Total positive = consistent (+1)
    # Spread positive + Total negative = contradiction (-2)
    if (spread_dir > 0) == (total_dir > 0):
        return 1.0  # Consistent
    else:
        return -2.0  # Contradiction


# ─── MOMENTUM DECAY ───

def momentum_decay(row: dict) -> float:
    """
    If score hasn't increased in 4+ ticks, apply decay.
    Prevents stale persistent signals.

    Returns penalty (0 to -3).
    """
    flat_ticks = _safe_int(row.get("flat_ticks", row.get("score_flat_count", 0)))

    if flat_ticks >= 4:
        # -1 per tick over 3, max -3
        return max(-3.0, -(flat_ticks - 3) * 1.0)

    return 0.0


# ─── SCORE FLOOR ───

def score_floor(pattern: str, layer_mode: str) -> float:
    """
    Minimum score based on signal quality.
    Good sharp signals never go below neutral.
    """
    if layer_mode == "L3_ONLY":
        return 0.0

    floors = {
        "A": 50.0,
        "D": 50.0,
        "G": 50.0,
        "B": 45.0,
        "C": 40.0,
        "E": 40.0,
        "F": 40.0,
        "N": 40.0,
    }
    return floors.get(pattern, 0.0)


# ─── MAIN SCORING FUNCTION ───

def compute_3layer_score(row: dict) -> dict:
    """
    Full 3-layer scoring.

    Args:
        row: dict with all L1/L2/L3/situational features

    Returns:
        dict with:
            "score": int (0-100)
            "layer_mode": str
            "pattern": str (A-F or N)
            "pattern_label": str
            "edge_type": str
            "l1_contribution": float
            "l2_contribution": float
            "l3_contribution": float
            "pattern_bonus": float
            "cross_market_adj": float
            "decay": float
            "flags": list of str
            "strong_eligible": bool
            "details": dict
    """
    layer_mode = str(row.get("layer_mode", "L3_ONLY"))

    # Layer contributions
    l1_result = compute_l1_contribution(row)
    l2_result = compute_l2_contribution(row)
    l3_result = compute_l3_contribution(row)

    l1_contrib = l1_result["l1_contribution"]
    l2_contrib = l2_result["l2_contribution"]
    l3_contrib = l3_result["l3_contribution"]

    # Interaction pattern
    pattern_result = detect_interaction_pattern(row)
    pattern = pattern_result["pattern"]
    pattern_label = pattern_result["label"]

    # Pattern effects
    pattern_config = PATTERN_EFFECTS.get(pattern, {})
    pattern_bonus = pattern_config.get("bonus", 0)
    pattern_cap = pattern_config.get("cap")
    strong_eligible = pattern_config.get("strong_eligible", False)
    edge_type = pattern_config.get("label", pattern_label)

    # Cross-market check
    cross_adj = spread_total_cross_check(row)

    # Momentum decay
    decay = momentum_decay(row)

    # B2B situational adjustment
    b2b_adj = 0.0
    b2b_flag = str(row.get("b2b_flag", ""))
    if b2b_flag in ("HOME_B2B", "AWAY_B2B"):
        b2b_adj = -1.0  # Slight penalty for B2B opponent advantage

    # Compute raw score
    raw_score = 50.0 + l1_contrib + l2_contrib + l3_contrib + pattern_bonus + cross_adj + decay + b2b_adj

    # Apply layer mode cap
    layer_caps = {
        "L123": SCORE_CAP_L123,
        "L13": SCORE_CAP_L13,
        "L23": SCORE_CAP_L23,
        "L3_ONLY": SCORE_CAP_L3_ONLY,
    }
    cap = layer_caps.get(layer_mode, SCORE_CAP_L3_ONLY)

    # Apply pattern-specific cap (e.g., Pattern B caps at 70)
    if pattern_cap is not None:
        cap = min(cap, pattern_cap)

    # Apply floor
    floor = score_floor(pattern, layer_mode)

    # Clamp
    final_score = int(max(floor, min(cap, raw_score)))

    # STRONG eligibility
    # Must be: L123 mode + Pattern A or D + score >= 70
    strong_ok = (
        strong_eligible and
        layer_mode == "L123" and
        final_score >= 70 and
        pattern in ("A", "D", "G")
    )

    # Collect flags
    all_flags = list(l3_result.get("l3_flags", []))
    if pattern != "N":
        all_flags.append(f"PATTERN_{pattern}")
    if b2b_adj != 0:
        all_flags.append(b2b_flag)
    if cross_adj > 0:
        all_flags.append("CROSS_MKT_CONFIRM")
    elif cross_adj < 0:
        all_flags.append("CROSS_MKT_CONTRADICT")

    return {
        "score": final_score,
        "layer_mode": layer_mode,
        "pattern": pattern,
        "pattern_label": pattern_label,
        "edge_type": edge_type,
        "l1_contribution": l1_contrib,
        "l2_contribution": l2_contrib,
        "l3_contribution": l3_contrib,
        "pattern_bonus": pattern_bonus,
        "cross_market_adj": cross_adj,
        "decay": decay,
        "b2b_adj": b2b_adj,
        "flags": all_flags,
        "strong_eligible": strong_ok,
        "details": {
            "l1": l1_result.get("l1_details", {}),
            "l2": l2_result.get("l2_details", {}),
            "l3": l3_result.get("l3_details", {}),
            "pattern": pattern_result,
            "rlm": {
                "strength": pattern_result.get("rlm_strength", 0),
                "gap": pattern_result.get("rlm_gap", 0),
            } if pattern == "G" else None,
            "raw_score": round(raw_score, 2),
            "cap": cap,
            "floor": floor,
        },
    }


def score_dk_only(row: dict) -> dict:
    """
    L3-only scoring — identical to current v1.2 behavior.

    This is the fallback for games without L1/L2 data.
    The actual v1.2 scoring logic lives in main.py and is called directly.
    This wrapper just sets the right metadata.
    """
    # When running L3-only, we compute a simplified L3 contribution
    l3_result = compute_l3_contribution(row)
    l3_contrib = l3_result["l3_contribution"]

    pattern_result = detect_interaction_pattern(row)
    pattern = pattern_result["pattern"]  # Will be C or N for L3-only

    # For L3-only, base score comes from the existing v1.2 engine
    # This function just provides the v2.0 metadata wrapper
    return {
        "score": None,  # Caller should use existing v1.2 score
        "layer_mode": "L3_ONLY",
        "pattern": pattern,
        "pattern_label": pattern_result["label"],
        "edge_type": "DK_ONLY_DEGRADED",
        "l1_contribution": 0.0,
        "l2_contribution": 0.0,
        "l3_contribution": l3_contrib,
        "pattern_bonus": 0,
        "cross_market_adj": 0.0,
        "decay": momentum_decay(row),
        "b2b_adj": 0.0,
        "flags": l3_result.get("l3_flags", []) + ["DK_ONLY"],
        "strong_eligible": False,
        "details": {
            "l3": l3_result.get("l3_details", {}),
            "pattern": pattern_result,
        },
    }
