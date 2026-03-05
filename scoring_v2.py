"""
Unified Scoring Model for Red Fox engine.

Score = dk_base (from dk_scoring.py)
  + l1_adjustment (-5 to +10): sharp book signal, BIDIRECTIONAL
  + l2_adjustment (-5 to +7): consensus validation
  + pattern_bonus (-8 to +5): interaction pattern effects
  + cross_adj (-2 to +1): spread/total consistency
  + line_diff_bonus (0 to +8): DK vs consensus/Pinnacle differential
  + momentum_decay (-3 to 0): flat tick penalty (ALL rows)
  Clamped to [floor, cap] by layer_mode

Layer modes:
  L123 (full): cap 100, STRONG eligible
  L13 (no consensus): cap 85, no STRONG
  L23 (no sharp): cap 80, no STRONG
  L3_ONLY: cap 75, no STRONG (dk_base only)
"""
import math

from engine_config import (
    L1_MAX_ADJUSTMENT,
    L1_MIN_ADJUSTMENT,
    L2_MAX_POSITIVE_ADJ,
    L2_MAX_NEGATIVE_ADJ,
    LINE_DIFF_MAX_BONUS,
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


def _bets_money_intensity(bets_pct: float, money_pct: float) -> float:
    """Continuous 0.0-1.0 intensity from bets/money split."""
    if bets_pct <= 0:
        return 0.5
    ratio = money_pct / bets_pct
    intensity = 1.0 / (1.0 + math.exp(-2.0 * (ratio - 1.3)))
    return round(max(0.0, min(1.0, intensity)), 3)


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


# ─── LAYER 1 ADJUSTMENT (BIDIRECTIONAL) ───

def compute_l1_adjustment(row: dict) -> dict:
    """
    Compute Layer 1 (sharp book) adjustment: -5 to +10.

    BIDIRECTIONAL: penalizes when sharp opposes DK favored side.

    Components:
      - Direction × magnitude (base signal, can be negative)
      - Agreement multiplier (1-6 books)
      - Quality gate: n_books / 4
      - Limit-weighted confidence
      - Move speed bonus/penalty
      - Stability modifier
      - Key number crossing
      - Leader book bonus
    """
    if not _safe_bool(row.get("l1_available")):
        return {"l1_adjustment": 0.0, "l1_details": {}}

    direction = _safe_int(row.get("l1_move_dir"))
    if direction == 0:
        return {"l1_adjustment": 0.0, "l1_details": {"reason": "no_move"}}

    magnitude = _safe_float(row.get("l1_move_magnitude"))  # 0-1 normalized
    agreement_mult = _safe_float(row.get("l1_agreement_mult"), 1.0)
    n_books = _safe_int(row.get("l1_n_books"))
    sharp_agreement = _safe_int(row.get("l1_sharp_agreement"))
    stability = _safe_float(row.get("l1_stability"), 0.5)
    speed_label = str(row.get("l1_speed_label", ""))
    key_cross = _safe_bool(row.get("l1_key_number_cross"))
    limit_conf = _safe_float(row.get("l1_limit_confidence"))
    leader_book = str(row.get("l1_leader_book", "")).lower()

    # Base signal: magnitude × agreement
    base = magnitude * agreement_mult

    # Quality gate: more books = more confidence
    quality_gate = min(n_books / 4.0, 1.0) if n_books > 0 else 0.5

    # Agreement multiplier: 4+ books agreeing = x1.2
    if sharp_agreement >= 4:
        agree_boost = 1.2
    elif sharp_agreement >= 2:
        agree_boost = 1.0
    else:
        agree_boost = 0.8

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

    # Limit confidence multiplier
    limit_mult = 1.0
    if limit_conf > 0:
        limit_mult = min(1.0 + (limit_conf / 50000.0) * 0.2, 1.2)

    # Leader book bonus
    leader_bonus = 0.0
    if leader_book == "pinnacle":
        leader_bonus = 1.0
    elif leader_book in ("bookmaker.eu", "bookmaker"):
        leader_bonus = 0.5

    # Compute raw adjustment
    raw = (base * stab_mult * limit_mult * quality_gate * agree_boost
           * L1_MAX_ADJUSTMENT) + speed_bonus + key_bonus + leader_bonus

    # BIDIRECTIONAL: check if sharp direction matches DK favored side
    if direction < 0:
        # Sharp opposes this side — invert to penalty range
        raw = -abs(raw) * 0.5  # Penalty is 50% of absolute value

    # GAP 5: DK money confirmation — cross-check L1 with bets/money context
    money_pct = _safe_float(row.get("money_pct"))
    bets_pct = _safe_float(row.get("bets_pct"))
    if money_pct > 0 and bets_pct > 0:
        bm_ratio = money_pct / bets_pct
        if direction > 0 and bm_ratio >= 1.5:
            raw *= 1.15  # Sharps AND money agree -> boost
        elif direction > 0 and bm_ratio < 0.7:
            raw *= 0.75  # Sharps say yes but money says no -> dampen
        elif direction < 0 and bm_ratio >= 1.5:
            raw *= 0.70  # Sharps oppose but money concentrated -> soften penalty

    # Clamp to [L1_MIN_ADJUSTMENT, L1_MAX_ADJUSTMENT]
    adjustment = max(L1_MIN_ADJUSTMENT, min(L1_MAX_ADJUSTMENT, raw))

    return {
        "l1_adjustment": round(adjustment, 2),
        "l1_details": {
            "base": round(base, 3),
            "stability_mult": round(stab_mult, 3),
            "limit_mult": round(limit_mult, 3),
            "quality_gate": round(quality_gate, 3),
            "agree_boost": agree_boost,
            "speed_bonus": speed_bonus,
            "key_bonus": key_bonus,
            "leader_bonus": leader_bonus,
            "speed_label": speed_label,
            "raw": round(raw, 2),
            "direction": direction,
        },
    }


# ─── LAYER 2 ADJUSTMENT ───

def compute_l2_adjustment(row: dict) -> dict:
    """
    Compute Layer 2 (consensus) adjustment: -5 to +7.

    Rescaled from old -8 to +10 range.
    """
    if not _safe_bool(row.get("l2_available")):
        return {"l2_adjustment": 0.0, "l2_details": {}}

    agreement = _safe_float(row.get("l2_consensus_agreement"))
    disp_mult = _safe_float(row.get("l2_dispersion_mult"), 1.0)
    disp_trend = str(row.get("l2_dispersion_trend", ""))
    validation = _safe_float(row.get("l2_validation_strength"))
    n_books = _safe_int(row.get("l2_n_books"))

    # More books = stronger consensus. Scale by min(n_books/20, 1.0)
    book_scale = min(n_books / 20.0, 1.0) if n_books > 0 else 0.5

    # Trend bonus
    trend_bonus = 0.0
    if disp_trend == "TIGHTENING":
        trend_bonus = 1.5
    elif disp_trend == "WIDENING":
        trend_bonus = -1.5

    # Base: validation strength scaled
    if agreement >= 0.6:
        # Market confirms — positive adjustment
        raw = validation * disp_mult * book_scale * L2_MAX_POSITIVE_ADJ + trend_bonus
        # GAP 5: DK divergence context — strong D confirms L2 consensus
        D = abs(_safe_float(row.get("divergence_D")))
        if D >= 15 and raw > 0:
            raw *= 1.10  # Strong DK divergence + L2 consensus = amplify
        elif D <= 5 and raw > 0:
            raw *= 0.85  # Weak DK signal + L2 consensus = modest dampening
        adjustment = max(0.0, min(L2_MAX_POSITIVE_ADJ, raw))
    elif agreement <= 0.3:
        # Market rejects — negative adjustment
        raw = -(1.0 - validation) * abs(L2_MAX_NEGATIVE_ADJ) + trend_bonus
        adjustment = max(L2_MAX_NEGATIVE_ADJ, min(0.0, raw))
    else:
        # Ambiguous
        adjustment = trend_bonus * 0.5

    return {
        "l2_adjustment": round(adjustment, 2),
        "l2_details": {
            "agreement": agreement,
            "disp_mult": disp_mult,
            "trend": disp_trend,
            "trend_bonus": trend_bonus,
            "validation": validation,
            "book_scale": round(book_scale, 3),
        },
    }


# ─── LINE DIFFERENTIAL BONUS ───

def compute_line_differential_bonus(row: dict) -> float:
    """
    DK line vs consensus/Pinnacle = exploitable edge.

    DK vs Consensus gap: 0 to +4
    Pinnacle vs DK gap: 0 to +4
    Total: capped at +8

    Returns bonus (0 to LINE_DIFF_MAX_BONUS).
    """
    dk_line = _safe_float(row.get("current_line", row.get("dk_line")))
    consensus_line = _safe_float(row.get("l2_consensus_line"))
    pinn_line = _safe_float(row.get("l2_pinn_line"))

    if not consensus_line and not pinn_line:
        return 0.0

    # Market type affects thresholds
    market = str(row.get("market", row.get("market_display", ""))).strip().upper()

    consensus_bonus = 0.0
    if consensus_line and dk_line:
        consensus_gap = abs(dk_line - consensus_line)
        # Dead zone: DK is sharp — small gaps are noise, not edge
        # Spreads: ignore < 1.0 pt, Totals: ignore < 2.0 pt
        dead_zone = 2.0 if market == "TOTAL" else 1.0
        if consensus_gap >= dead_zone:
            effective_gap = consensus_gap - dead_zone
            if market == "TOTAL":
                consensus_bonus = min(effective_gap * 1.5, 4.0)
            else:
                consensus_bonus = min(effective_gap * 2.0, 4.0)

    pinn_bonus = 0.0
    if pinn_line and dk_line:
        pinn_gap = abs(dk_line - pinn_line)
        # Pinnacle is sharpest — same dead zone, but higher multiplier
        dead_zone_p = 1.5 if market == "TOTAL" else 0.5
        if pinn_gap >= dead_zone_p:
            effective_pinn = pinn_gap - dead_zone_p
            pinn_bonus = min(effective_pinn * 2.5, 4.0)

    return min(consensus_bonus + pinn_bonus, LINE_DIFF_MAX_BONUS)


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

    gap_factor = min(bets_money_gap / 30.0, 1.0)
    bm_intensity = _bets_money_intensity(bets_pct, money_pct)
    strength = gap_factor * max(l1_strength, 0.3) * max(l2_agreement, 0.5) * (0.7 + bm_intensity * 0.6)
    strength = min(1.0, strength)

    return {"detected": True, "strength": round(strength, 3), "bets_money_gap": round(bets_money_gap, 1)}


# ─── INTERACTION PATTERNS ───

def detect_interaction_pattern(row: dict) -> dict:
    """
    Detect which interaction pattern (A-G) applies.

    Returns dict with pattern letter, label, and explanation.
    """
    l1_available = _safe_bool(row.get("l1_available"))
    l2_available = _safe_bool(row.get("l2_available"))
    l1_dir = _safe_int(row.get("l1_move_dir"))
    l1_strength = _safe_float(row.get("l1_sharp_strength"))
    l2_agreement = _safe_float(row.get("l2_consensus_agreement"))
    stale = _safe_bool(row.get("l2_stale_price_flag"))
    speed_label = str(row.get("l1_speed_label", ""))

    dk_bets = _safe_float(row.get("bets_pct", row.get("dk_bets_pct", 50)))
    dk_public_heavy = dk_bets > 65 or dk_bets < 35

    hours_to_game = _safe_float(row.get("hours_to_game", row.get("time_to_start_hours", 4)))

    # Pattern F: Late snap
    if l1_available and speed_label == "FAST_SNAP" and hours_to_game < 1:
        return {
            "pattern": "F",
            "label": "LATE_SNAP_WARNING",
            "explanation": "Rapid sharp movement in final hour — information or chaos",
        }

    # Pattern G: Reverse Line Movement
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
        return {
            "pattern": "A",
            "label": "SHARP_VS_PUBLIC",
            "explanation": "Sharp books moved, consensus confirms, public betting opposite",
        }

    # Pattern D: Stale price (DK lagging)
    if l1_available and l1_dir != 0 and l2_agreement >= 0.5 and stale:
        stale_gap = _safe_float(row.get("l2_stale_price_gap"))
        return {
            "pattern": "D",
            "label": "STALE_PRICE",
            "explanation": f"Sharp and consensus moved, DK line hasn't caught up (gap: {stale_gap})",
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

    # Default
    return {
        "pattern": "N",
        "label": "NEUTRAL",
        "explanation": "No strong interaction pattern detected",
    }


# ─── CROSS-MARKET CHECK ───

def spread_total_cross_check(row: dict) -> float:
    """
    Check if spread and total are consistent.
    Returns bonus/penalty (-2 to +1).
    """
    spread_dir = _safe_float(row.get("dk_spread_dir", row.get("spread_move_dir", 0)))
    total_dir = _safe_float(row.get("dk_total_dir", row.get("total_move_dir", 0)))

    if spread_dir == 0 or total_dir == 0:
        return 0.0

    if (spread_dir > 0) == (total_dir > 0):
        return 1.0  # Consistent
    else:
        return -2.0  # Contradiction


# ─── MOMENTUM DECAY ───

def momentum_decay(row: dict) -> float:
    """
    If score hasn't increased in 4+ ticks, apply decay.
    Now applies to ALL rows (including L3_ONLY).
    Returns penalty (0 to -3).
    """
    flat_ticks = _safe_int(row.get("flat_ticks", row.get("score_flat_count", 0)))

    if flat_ticks >= 4:
        return max(-3.0, -(flat_ticks - 3) * 1.0)

    return 0.0


# ─── SCORE FLOOR ───

def score_floor(pattern: str, layer_mode: str) -> float:
    """Minimum score based on signal quality."""
    if layer_mode == "L3_ONLY":
        return 0.0

    floors = {
        "A": 50.0, "D": 50.0, "G": 50.0,
        "B": 45.0, "C": 40.0, "E": 40.0,
        "F": 40.0, "N": 40.0,
    }
    return floors.get(pattern, 0.0)


# ─── MAIN SCORING FUNCTION ───

def compute_unified_score(row: dict) -> dict:
    """
    Unified scoring — dk_base + L1/L2 adjustments + patterns + line_diff.

    Args:
        row: dict with dk_base_score already set (from dk_scoring.compute_dk_base),
             plus all L1/L2/situational features from merge_layers.

    Returns:
        dict with:
            "score": int (0-100)
            "layer_mode": str
            "pattern": str (A-G or N)
            "pattern_label": str
            "edge_type": str
            "l1_adjustment": float
            "l2_adjustment": float
            "pattern_bonus": float
            "cross_market_adj": float
            "line_diff_bonus": float
            "decay": float
            "flags": list of str
            "strong_eligible": bool
            "details": dict
    """
    layer_mode = str(row.get("layer_mode", "L3_ONLY"))
    dk_base = _safe_float(row.get("dk_base_score", 50))

    # Layer adjustments
    l1_result = compute_l1_adjustment(row)
    l2_result = compute_l2_adjustment(row)

    l1_adj = l1_result["l1_adjustment"]
    l2_adj = l2_result["l2_adjustment"]

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

    # Line differential bonus — DISABLED: 3x/day L2 data too stale for
    # precise line-vs-line comparisons. L2 directional validation (agreement,
    # dispersion) still active. Re-enable when real-time feeds available.
    line_diff = 0.0

    # Momentum decay (now for ALL rows)
    decay = momentum_decay(row)

    # B2B situational adjustment
    b2b_adj = 0.0
    b2b_flag = str(row.get("b2b_flag", ""))
    if b2b_flag in ("HOME_B2B", "AWAY_B2B"):
        b2b_adj = -1.0

    # Injury situational adjustment (NBA/NHL/NFL only — college rosters too deep)
    injury_adj = 0.0
    _inj_sport = str(row.get("sport", "")).lower()
    if _inj_sport in ("nba", "nhl", "nfl"):
        _home_inj = int(row.get("home_injury_count", 0) or 0)
        _away_inj = int(row.get("away_injury_count", 0) or 0)
        _side = str(row.get("side", "")).lower()
        _home_norm = str(row.get("home_team_norm", "")).lower()
        _away_norm = str(row.get("away_team_norm", "")).lower()
        _mkt = str(row.get("market_display", "")).upper()

        if _mkt == "TOTAL":
            # Totals: heavy injuries on either side reduce confidence
            _total_inj = _home_inj + _away_inj
            if _total_inj >= 4:
                injury_adj = -1.5
            elif _total_inj >= 2:
                injury_adj = -0.5
        else:
            # ML/SPREAD: determine which team we're on
            _on_home = _home_norm and _home_norm in _side
            _on_away = _away_norm and _away_norm in _side
            if _on_home:
                _our_inj, _opp_inj = _home_inj, _away_inj
            elif _on_away:
                _our_inj, _opp_inj = _away_inj, _home_inj
            else:
                _our_inj, _opp_inj = 0, 0

            # Our team has injuries → dampen confidence
            if _our_inj >= 3:
                injury_adj = -2.0
            elif _our_inj >= 2:
                injury_adj = -1.0
            # Opponent has injuries → slight edge for us
            if _opp_inj >= 3:
                injury_adj += 1.0
            elif _opp_inj >= 2:
                injury_adj += 0.5

    # Compute raw score: dk_base + adjustments
    raw_score = dk_base + l1_adj + l2_adj + pattern_bonus + cross_adj + line_diff + decay + b2b_adj + injury_adj

    # Apply layer mode cap
    layer_caps = {
        "L123": SCORE_CAP_L123,
        "L13": SCORE_CAP_L13,
        "L23": SCORE_CAP_L23,
        "L3_ONLY": SCORE_CAP_L3_ONLY,
    }
    cap = layer_caps.get(layer_mode, SCORE_CAP_L3_ONLY)

    # Apply pattern-specific cap
    if pattern_cap is not None:
        cap = min(cap, pattern_cap)

    # Apply floor
    floor = score_floor(pattern, layer_mode)

    # Clamp
    final_score = int(max(floor, min(cap, raw_score)))

    # STRONG eligibility (basic check — full check in dk_scoring.is_strong_eligible)
    strong_ok = (
        strong_eligible and
        layer_mode == "L123" and
        final_score >= 70 and
        pattern in ("A", "D", "G")
    )

    # Collect flags
    all_flags = []
    if pattern != "N":
        all_flags.append(f"PATTERN_{pattern}")
    if b2b_adj != 0:
        all_flags.append(b2b_flag)
    if injury_adj != 0:
        all_flags.append(f"INJ_ADJ:{injury_adj:+.1f}")
    if cross_adj > 0:
        all_flags.append("CROSS_MKT_CONFIRM")
    elif cross_adj < 0:
        all_flags.append("CROSS_MKT_CONTRADICT")
    if line_diff > 0:
        all_flags.append(f"LINE_DIFF_+{line_diff:.1f}")

    return {
        "score": final_score,
        "dk_base_score": round(dk_base, 2),
        "layer_mode": layer_mode,
        "pattern": pattern,
        "pattern_label": pattern_label,
        "edge_type": edge_type,
        "l1_adjustment": l1_adj,
        "l2_adjustment": l2_adj,
        "pattern_bonus": pattern_bonus,
        "cross_market_adj": cross_adj,
        "line_diff_bonus": round(line_diff, 2),
        "decay": decay,
        "b2b_adj": b2b_adj,
        "injury_adj": injury_adj,
        "flags": all_flags,
        "strong_eligible": strong_ok,
        "details": {
            "l1": l1_result.get("l1_details", {}),
            "l2": l2_result.get("l2_details", {}),
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


# ─── LEGACY COMPATIBILITY ───

def compute_3layer_score(row: dict) -> dict:
    """Legacy wrapper — calls compute_unified_score."""
    return compute_unified_score(row)


def score_dk_only(row: dict) -> dict:
    """Legacy wrapper for L3-only scoring."""
    result = compute_unified_score(row)
    result["edge_type"] = "DK_ONLY"
    return result
