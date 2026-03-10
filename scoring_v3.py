"""
Red Fox v3.3 — Scoring Engine

Score = 50 + Sharp + Consensus + Retail + Timing + CrossMarket + MarketReaction
Clamped to [0, 100]

Six pure functions. No shared state. No side effects.
Each reads from a row dict and returns its contribution.
"""

import engine_config_v3 as C


# ═══════════════════════════════════════════════════════════════
# Component 1 — Sharp Signal  [-15, +20]
# ═══════════════════════════════════════════════════════════════

def compute_sharp_signal(row: dict) -> dict:
    """Primary component. Sharp books = informed money. Bidirectional."""
    sport = (row.get("sport") or "").lower()
    market = (row.get("market_display") or row.get("market") or "").upper()
    l1_present = _bool(row.get("l1_available", row.get("l1_present", False)))

    # L1 absent → Sharp = 0 exactly
    if not l1_present:
        return {"sharp_score": 0.0, "sharp_detail": "L1 absent"}

    move_dir = _num(row.get("l1_move_dir", 0))

    # TOTAL: same line move has opposite meaning for Over vs Under.
    # Line UP (+1) = books favor Over, oppose Under. Negate for Under.
    if market == "TOTAL" and move_dir != 0:
        side = (row.get("side") or row.get("favored_side") or "").upper()
        if not side:
            import logging
            logging.warning("[v3.3c] TOTAL row missing side field — cannot determine Over/Under polarity")
        elif "UNDER" in side:
            move_dir = -move_dir

    # Step 1 — Direction gate
    if move_dir == 0:
        return {"sharp_score": 0.0, "sharp_detail": "No L1 movement"}

    # Step 2 — Magnitude × timing bucket
    magnitude_raw = abs(_num(row.get("l1_move_magnitude_raw", 0)))
    timing_bucket = (row.get("timing_bucket") or row.get("l1_timing_bucket") or "MID").upper()
    timing_mult = C.SHARP_TIMING_MULT.get(timing_bucket, 0.90)

    # Debug vars (populated for ML only)
    ml_implied_prob = 0.0
    ml_cred_mult = 1.0
    sharp_base_pre_cred = 0.0
    sharp_base_post_cred = 0.0

    if market == "SPREAD":
        base = min(C.SHARP_MAG_SPREAD_CAP, magnitude_raw * C.SHARP_MAG_SPREAD_MULT)
    elif market == "TOTAL":
        base = min(C.SHARP_MAG_TOTAL_CAP, magnitude_raw * C.SHARP_MAG_TOTAL_MULT)
    else:  # ML
        base = min(C.SHARP_MAG_ML_CAP, magnitude_raw * C.SHARP_MAG_ML_MULT)
        # ML price credibility — extreme dogs get dampened
        odds = _num(row.get("current_odds", 0))
        sharp_base_pre_cred = base
        if odds > 0:
            ml_implied_prob = 100.0 / (odds + 100.0) * 100  # +800 → 11.1%
            ml_cred_mult = C.ML_PRICE_CREDIBILITY[-1][1]    # default: lowest tier
            for threshold, mult in C.ML_PRICE_CREDIBILITY:
                if ml_implied_prob >= threshold:
                    ml_cred_mult = mult
                    break
            base *= ml_cred_mult
        sharp_base_post_cred = base

    base *= timing_mult

    # NHL puck line dampening
    if sport == "nhl" and market == "SPREAD":
        base *= C.NHL_PUCK_LINE_SHARP_MULT

    # Step 3 — Graduated agreement multiplier (Pinnacle-weighted, direction-aware)
    pinnacle_moved = _bool(row.get("l1_pinnacle_moved", False))
    sharp_agree = _num(row.get("l1_sharp_agreement", 0))
    support_agree = _num(row.get("l1_support_agreement", 0))

    if pinnacle_moved and sharp_agree >= 2:
        base *= C.SHARP_AGREEMENT_BOOST       # Pinnacle + cluster = strongest
    elif pinnacle_moved:
        base *= C.SHARP_AGREEMENT_MODERATE    # Pinnacle alone = still meaningful (early steam)
    elif sharp_agree >= 2:
        base *= C.SHARP_AGREEMENT_MODERATE    # Cluster without Pinnacle = moderate
    elif sharp_agree >= 1 and support_agree >= 1:
        base *= C.SHARP_AGREEMENT_SLIGHT      # Cross-tier = slight boost
    else:
        base *= C.SHARP_AGREEMENT_DAMPEN      # Weak/noise

    # Step 4 — Path behavior adjustment
    path = (row.get("l1_path_behavior") or "UNKNOWN").upper()
    path_adj = C.SHARP_PATH_ADJ.get(path, 0.0)
    result = base + path_adj

    # Step 5 — Key number bonus (NFL/NCAAF only)
    if sport in C.KEY_NUMBER_SPORTS:
        key_cross = _num(row.get("l1_key_number_cross", 0))
        if key_cross:
            result += C.SHARP_KEY_NUMBER_BONUS

    # Step 6 — Direction flip (sharps oppose side)
    detail = f"dir={move_dir} mag={magnitude_raw:.2f} path={path}"
    if move_dir == -1:
        result = result * C.SHARP_DIRECTION_FLIP_MULT
        result = max(result, C.SHARP_MIN)
        detail += " [FLIPPED]"

    # Step 7 — Hard cap
    result = max(C.SHARP_MIN, min(C.SHARP_MAX, result))

    return {
        "sharp_score": round(result, 2),
        "sharp_detail": detail,
        "ml_implied_prob": round(ml_implied_prob, 1),
        "ml_cred_mult": ml_cred_mult,
        "sharp_base_pre_cred": round(sharp_base_pre_cred, 2),
        "sharp_base_post_cred": round(sharp_base_post_cred, 2),
    }


# ═══════════════════════════════════════════════════════════════
# Component 2 — Consensus Validation  [-10, +18]
# ═══════════════════════════════════════════════════════════════

def compute_consensus_validation(row: dict) -> dict:
    """31-book consensus validates directional read.
    Uses time-series agreement tiers when direction history exists,
    falls back to cross-sectional signals (pinn vs market) when it doesn't.
    """
    l2_available = _bool(row.get("l2_available", False))
    if not l2_available:
        return {"consensus_score": 0.0, "consensus_detail": "L2 absent", "consensus_tier": 0}

    n_books = _num(row.get("l2_n_books", 0))
    disp_label = (row.get("l2_dispersion_label") or "NORMAL").upper()
    agreement = _num(row.get("l2_consensus_agreement", 0))

    # Determine if we have time-series direction data
    # Fallback to l1_move_dir: consensus_agreement is already computed relative
    # to l1_move_dir in l2_features.py, so this is semantically correct.
    consensus_dir = row.get("l2_consensus_direction") or row.get("consensus_direction")
    if consensus_dir is None or str(consensus_dir).lower() in ("", "nan", "none"):
        _l1_dir = _num(row.get("l1_move_dir", 0))
        consensus_dir = _l1_dir if _l1_dir != 0 else None
    has_direction = (consensus_dir is not None
                     and str(consensus_dir).lower() not in ("", "nan", "none", "0", "0.0")
                     and agreement > 0)

    if has_direction:
        # ── TIME-SERIES PATH (existing logic, unchanged) ──
        base = C.CONSENSUS_TIERS[-1][1]  # default: rejects
        consensus_tier = 0  # default: rejects (below all tiers)
        for i, (threshold, value) in enumerate(C.CONSENSUS_TIERS):
            if agreement >= threshold:
                base = value
                consensus_tier = i + 1  # 1=strong, 2=moderate, 3=weak
                break

        disp_mult = C.CONSENSUS_DISPERSION_MULT.get(disp_label, 1.0)
        result = base * disp_mult

        trend = (row.get("l2_dispersion_trend") or "STABLE").upper()
        trend_adj = C.CONSENSUS_TREND_ADJ.get(trend, 0.0)
        result += trend_adj

        detail = f"agree={agreement:.2f} disp={disp_label} trend={trend} books={int(n_books)}"
    else:
        consensus_tier = 0  # cross-sectional path
        # ── CROSS-SECTIONAL PATH (single-snapshot fallback) ──
        # Compute pinn gap from raw fields if pre-computed field unavailable
        pinn_gap = abs(_num(row.get("pinn_vs_consensus",
                                     row.get("l2_pinn_vs_consensus", 0))))
        if pinn_gap == 0:
            pinn_line = _num(row.get("l2_pinn_line", 0))
            cons_line = _num(row.get("l2_consensus_line", 0))
            if pinn_line != 0 and cons_line != 0:
                pinn_gap = abs(pinn_line - cons_line)

        # Pinnacle vs market gap — core cross-sectional signal (4-tier)
        result = 0.0
        for min_gap, min_books, tier_score in C.CROSS_PINN_TIERS:
            if pinn_gap >= min_gap and n_books >= min_books:
                result = tier_score
                break

        # Dispersion-based fallback: when pinn gap is 0 (e.g. moneyline,
        # or first-cycle with no line data) but books exist, use dispersion
        # tightness as a consensus signal — tight books = agreement on price.
        if result == 0.0 and pinn_gap == 0 and n_books >= 10:
            if disp_label == "TIGHT":
                result = 5.0
                consensus_tier = 2  # moderate
            elif disp_label == "NORMAL":
                result = 2.0
                consensus_tier = 3  # weak
            # WIDE / VERY_WIDE → no signal (result stays 0)

        # Dispersion guard — tight books with small gap = less signal
        if disp_label == "TIGHT" and pinn_gap < 1.5 and pinn_gap > 0:
            result *= C.CROSS_TIGHT_DAMPENING
        elif disp_label == "VERY_WIDE":
            result *= C.CROSS_VERY_WIDE_DAMPENING

        detail = f"cross-section pinn_gap={pinn_gap:.2f} disp={disp_label} books={int(n_books)}"

    # Stale price bonus (applies to both paths)
    stale_gap = _num(row.get("l2_stale_price_gap", 0))
    if stale_gap >= C.CONSENSUS_STALE_LARGE_THRESHOLD:
        result += C.CONSENSUS_STALE_LARGE
    elif stale_gap >= C.CONSENSUS_STALE_SMALL_THRESHOLD:
        result += C.CONSENSUS_STALE_SMALL

    # Book count guard (applies to both paths)
    book_mult = C.CONSENSUS_BOOK_GUARD[-1][1]
    for threshold, mult in C.CONSENSUS_BOOK_GUARD:
        if n_books >= threshold:
            book_mult = mult
            break
    result *= book_mult

    # Hard cap
    result = max(C.CONSENSUS_MIN, min(C.CONSENSUS_MAX, result))

    return {"consensus_score": round(result, 2), "consensus_detail": detail,
            "consensus_tier": consensus_tier}


# ═══════════════════════════════════════════════════════════════
# Component 3 — Retail Context  [-8, +8]
# ═══════════════════════════════════════════════════════════════

def compute_retail_context(row: dict) -> dict:
    """DK retail data. ±8 cap. Cannot manufacture BET/STRONG alone."""
    sport = (row.get("sport") or "").lower()
    market = (row.get("market_display") or row.get("market") or "").upper()
    bets_pct = _num(row.get("bets_pct", 0))
    money_pct = _num(row.get("money_pct", 0))

    # Step 1 — Divergence signal (D = money - bets, NOT raw money)
    D = money_pct - bets_pct

    if market == "SPREAD":
        raw_div = min(C.RETAIL_DIV_SPREAD_CAP, abs(D) * C.RETAIL_DIV_SPREAD_MULT)
    elif market == "TOTAL":
        raw_div = min(C.RETAIL_DIV_TOTAL_CAP, abs(D) * C.RETAIL_DIV_TOTAL_MULT)
    else:  # ML
        raw_div = min(C.RETAIL_DIV_ML_CAP, abs(D) * C.RETAIL_DIV_ML_MULT)

    # Negative D = money follows bets, no smart-money gap
    if D < 0:
        result = raw_div * C.RETAIL_NEG_DIV_MULT
    else:
        result = raw_div

    # Step 2 — Public crowding penalty
    if (bets_pct >= C.RETAIL_CROWDING_BETS_THRESHOLD and
            money_pct >= C.RETAIL_CROWDING_MONEY_THRESHOLD):
        result += C.RETAIL_CROWDING_PENALTY

    # Step 3 — Parlay distortion penalty (ML only)
    if market in ("MONEYLINE", "ML"):
        current_odds = _num(row.get("current_odds", 0))
        if (money_pct > C.RETAIL_PARLAY_MONEY_THRESHOLD and
                current_odds < C.RETAIL_PARLAY_ODDS_THRESHOLD):
            result += C.RETAIL_PARLAY_PENALTY

    # Step 4 — DK line confirmation
    l1_present = _bool(row.get("l1_available", row.get("l1_present", False)))
    if l1_present:
        l1_dir = _num(row.get("l1_move_dir", 0))
        dk_dir = _num(row.get("move_dir", row.get("dk_line_move_dir", 0)))
        dk_mag = abs(_num(row.get("effective_move_mag", 0)))
        if l1_dir != 0 and dk_mag >= C.RETAIL_DK_CONFIRM_MIN_MOVE:
            if dk_dir == l1_dir:
                result += C.RETAIL_DK_CONFIRM_BONUS
            elif dk_dir == -l1_dir:
                result += C.RETAIL_DK_OPPOSE_PENALTY

    # Step 5 — Sample credibility
    sample = bets_pct  # bets_pct itself is the credibility proxy
    sample_mult = C.RETAIL_SAMPLE_CREDIBILITY[-1][1]
    for threshold, mult in C.RETAIL_SAMPLE_CREDIBILITY:
        if sample >= threshold:
            sample_mult = mult
            break
    result *= sample_mult

    # Step 6 — ML instrument multiplier (permanent, all sports)
    if market in ("MONEYLINE", "ML"):
        result *= C.RETAIL_ML_MULTIPLIER

    # NHL retail dampening
    if sport == "nhl":
        result *= C.NHL_RETAIL_SAMPLE_MULT

    # Step 7 — Hard cap
    result = max(C.RETAIL_MIN, min(C.RETAIL_MAX, result))

    detail = f"D={D:.0f} bets={bets_pct:.0f}% money={money_pct:.0f}%"
    return {"retail_score": round(result, 2), "retail_detail": detail}


# ═══════════════════════════════════════════════════════════════
# Component 4 — Timing Modifier  [-5, +1]
# ═══════════════════════════════════════════════════════════════

def compute_timing_modifier(row: dict) -> dict:
    """Credibility filter. Max positive is +1. Suppression is the point."""
    sport = (row.get("sport") or "").lower()
    timing_bucket = (row.get("timing_bucket") or "MID").upper()
    l1_present = _bool(row.get("l1_available", row.get("l1_present", False)))
    path = (row.get("l1_path_behavior") or "UNKNOWN").upper()

    if timing_bucket == "EARLY":
        result = C.TIMING_EARLY
    elif timing_bucket == "MID":
        # +1 when path HELD or EXTENDED (path tracked independently of L1)
        if path in ("HELD", "EXTENDED"):
            result = C.TIMING_MID_BOOST
        else:
            result = C.TIMING_MID_BASE
    elif timing_bucket == "LATE":
        if path in ("REVERSED", "OSCILLATED"):
            result = C.TIMING_LATE_REVERSED  # -5
        elif path in ("HELD", "EXTENDED") and l1_present:
            result = C.TIMING_LATE_CONFIRMED  # 0 — confirmed late signal
        else:
            result = C.TIMING_LATE_BASE  # -3

        # NBA exception: LATE never worse than -3
        if sport == "nba":
            result = max(result, C.TIMING_NBA_LATE_CAP)
    else:
        result = 0  # LIVE, UNKNOWN, etc.

    result = max(C.TIMING_MIN, min(C.TIMING_MAX, result))

    detail = f"bucket={timing_bucket} path={path}"
    return {"timing_score": round(result, 2), "timing_detail": detail}


# ═══════════════════════════════════════════════════════════════
# Component 5 — Cross-Market Sanity  [-4, +4]
# ═══════════════════════════════════════════════════════════════

def compute_cross_market_sanity(row: dict) -> dict:
    """Do spread and ML agree on which side to favor?"""
    sport = (row.get("sport") or "").lower()
    market = (row.get("market_display") or row.get("market") or "").upper()

    # UFC: ML only, no spread to compare
    if sport in C.CROSS_MARKET_EXEMPT_SPORTS:
        return {"cross_market_score": 0, "cross_market_detail": "Exempt (UFC)"}

    # MLB run line exempt
    if sport == "mlb" and market == "SPREAD":
        return {"cross_market_score": 0, "cross_market_detail": "Exempt (MLB run line)"}

    spread_fav = (row.get("spread_favored_side") or "").strip()
    ml_fav = (row.get("ml_favored_side") or "").strip()

    # Only one market present
    if not spread_fav or not ml_fav:
        return {"cross_market_score": 0, "cross_market_detail": "Single market only"}

    # Compute ML favorite implied probability gap from 0.5 (pick'em)
    ml_fav_odds = _num(row.get("ml_fav_odds", 0))
    if ml_fav_odds != 0:
        if ml_fav_odds < 0:
            impl_prob = abs(ml_fav_odds) / (abs(ml_fav_odds) + 100)
        else:
            impl_prob = 100 / (ml_fav_odds + 100)
        prob_gap = abs(impl_prob - 0.5)
    else:
        prob_gap = 0.0

    # Near pick'em — ML favorite call is unreliable
    if prob_gap < C.CROSS_MARKET_PICKEM_THRESHOLD:
        return {"cross_market_score": 0, "cross_market_detail": f"Pick'em (gap={prob_gap:.3f})"}

    if spread_fav == ml_fav:
        if prob_gap >= C.CROSS_MARKET_STRONG_THRESHOLD:
            result = C.CROSS_MARKET_AGREE_STRONG
        else:
            result = C.CROSS_MARKET_AGREE_MARGINAL
        detail = f"Aligned: {spread_fav} (gap={prob_gap:.3f})"
    else:
        if prob_gap >= C.CROSS_MARKET_STRONG_THRESHOLD:
            result = C.CROSS_MARKET_CONTRADICT_STRONG
        else:
            result = C.CROSS_MARKET_CONTRADICT_MARGINAL
        detail = f"Contradiction: spread={spread_fav} ml={ml_fav} (gap={prob_gap:.3f})"

    return {"cross_market_score": result, "cross_market_detail": detail}


# ═══════════════════════════════════════════════════════════════
# Component 6 — Market Reaction  [-4, +12]
# ═══════════════════════════════════════════════════════════════

def compute_market_reaction(row: dict) -> dict:
    """Book-initiated movement without public pressure."""
    bets_pct = _num(row.get("bets_pct", 0))
    money_pct = _num(row.get("money_pct", 0))
    result = 0.0
    detail_parts = []

    eff_move = abs(_num(row.get("effective_move_mag", 0)))
    line_move = abs(_num(row.get("line_move_open", 0)))
    move_mag = max(eff_move, line_move)

    if (move_mag >= C.MR_BOOK_MOVE_THRESHOLD
            and bets_pct < C.MR_BOOK_BETS_CEILING
            and money_pct < C.MR_BOOK_MONEY_CEILING):
        result += C.MR_BOOK_INITIATED_BONUS
        detail_parts.append(f"book-init(+{C.MR_BOOK_INITIATED_BONUS})")

    result = max(C.MARKET_REACTION_MIN, min(C.MARKET_REACTION_MAX, result))
    detail = " | ".join(detail_parts) if detail_parts else "no reaction"
    return {"market_reaction_score": round(result, 2), "market_reaction_detail": detail}


# ═══════════════════════════════════════════════════════════════
# Orchestrator — Score = 50 + Sharp + Consensus + Retail + Timing + Cross + MarketReaction
# ═══════════════════════════════════════════════════════════════

def compute_v3_score(row: dict) -> dict:
    """Compute the full v3.3 score. Returns all components + metadata."""
    sharp = compute_sharp_signal(row)
    consensus = compute_consensus_validation(row)
    retail = compute_retail_context(row)
    timing = compute_timing_modifier(row)
    cross = compute_cross_market_sanity(row)
    market_rx = compute_market_reaction(row)

    # Soft retail dampening when L1 absent + L2 weak
    l1_present = _bool(row.get("l1_available", row.get("l1_present", False)))
    l2_agreement = _num(row.get("l2_consensus_agreement", 0))
    if not l1_present and l2_agreement < C.L1_ABSENT_L2_WEAK_THRESHOLD:
        retail["retail_score"] = round(
            retail["retail_score"] * C.RETAIL_L1_ABSENT_MULT, 2)

    raw = (C.BASE
           + sharp["sharp_score"]
           + consensus["consensus_score"]
           + retail["retail_score"]
           + timing["timing_score"]
           + cross["cross_market_score"]
           + market_rx["market_reaction_score"])

    # L1-absent + L2-weak hard cap
    l1_cap_applied = False
    if not l1_present and l2_agreement < C.L1_ABSENT_L2_WEAK_THRESHOLD:
        if raw > C.L1_ABSENT_L2_WEAK_CAP:
            raw = C.L1_ABSENT_L2_WEAK_CAP
            l1_cap_applied = True

    final_score = max(0, min(100, round(raw, 1)))

    # Pattern detection (output label only — never affects score)
    patterns = _detect_pattern(row, sharp, consensus, retail)
    explanation = _build_explanation(row, sharp, consensus, retail, timing, cross, final_score, market_rx)

    return {
        "final_score": final_score,
        "sharp_score": sharp["sharp_score"],
        "consensus_score": consensus["consensus_score"],
        "consensus_tier": consensus["consensus_tier"],
        "retail_score": retail["retail_score"],
        "timing_modifier": timing["timing_score"],
        "cross_market_adj": cross["cross_market_score"],
        "market_reaction_score": market_rx["market_reaction_score"],
        "sharp_detail": sharp["sharp_detail"],
        "consensus_detail": consensus["consensus_detail"],
        "retail_detail": retail["retail_detail"],
        "timing_detail": timing["timing_detail"],
        "cross_market_detail": cross["cross_market_detail"],
        "market_reaction_detail": market_rx["market_reaction_detail"],
        "pattern_primary": patterns["pattern_primary"],
        "pattern_secondary": patterns["pattern_secondary"],
        "score_explanation": explanation,
        "l1_present": l1_present,
        "l1_cap_applied": l1_cap_applied,
        "ml_implied_prob": sharp.get("ml_implied_prob", 0.0),
        "ml_cred_mult": sharp.get("ml_cred_mult", 1.0),
        "sharp_base_pre_cred": sharp.get("sharp_base_pre_cred", 0.0),
        "sharp_base_post_cred": sharp.get("sharp_base_post_cred", 0.0),
    }


# ═══════════════════════════════════════════════════════════════
# Pattern Detection (labels only, never score inputs)
# ═══════════════════════════════════════════════════════════════

def _detect_pattern(row, sharp, consensus, retail) -> dict:
    """Detect all matching pattern labels in priority order.
    Returns dict with pattern_primary (highest) and pattern_secondary (second, or None).
    Output only — never changes score."""
    l1_present = _bool(row.get("l1_available", row.get("l1_present", False)))
    path = (row.get("l1_path_behavior") or "UNKNOWN").upper()
    bets_pct = _num(row.get("bets_pct", 0))
    money_pct = _num(row.get("money_pct", 0))
    l2_agreement = _num(row.get("l2_consensus_agreement", 0))
    stale = _bool(row.get("l2_stale_price_flag", False))
    move_dir = _num(row.get("l1_move_dir", 0))
    dk_dir = _num(row.get("move_dir", 0))
    sharp_score = sharp["sharp_score"]
    pinnacle_moved = _bool(row.get("l1_pinnacle_moved", False))

    matched = []

    # SHARP_REVERSAL: L1 moved + public heavy opposite + path HELD/EXTENDED
    if (l1_present and move_dir != 0 and path in ("HELD", "EXTENDED")
            and bets_pct >= 60 and sharp_score > 0):
        D = money_pct - bets_pct
        if D < -5:
            matched.append("SHARP_REVERSAL")

    # STALE_PRICE: DK lags consensus + L1 confirmed
    if stale and l1_present and sharp_score > 0:
        matched.append("STALE_PRICE")

    # FREEZE_PRESSURE: L1 moved + L2 strongly aligned + no DK response
    if (l1_present and move_dir != 0 and l2_agreement >= 0.75
            and dk_dir == 0):
        matched.append("FREEZE_PRESSURE")

    # SHARP_CONFIRMED: Strong sharp conviction — multiple books moved in agreement
    sharp_agree = _num(row.get("l1_sharp_agreement", 0))
    if sharp_score > 6 and pinnacle_moved and sharp_agree >= 1:
        matched.append("SHARP_CONFIRMED")

    # BOOK_RESISTANCE: Heavy public but book hasn't moved toward public (line or juice)
    line_move = abs(_num(row.get("line_move_open", 0)))
    eff_move = _num(row.get("effective_move_mag", 0))
    if (bets_pct >= 65 or money_pct >= 65) and line_move < 0.5 and eff_move < 0.3:
        if dk_dir <= 0:
            if not (l1_present and move_dir != 0 and l2_agreement >= 0.75):
                matched.append("BOOK_RESISTANCE")

    # BOOK_INITIATED / SHARP_BOOK_CONFLICT: Line moved without public pressure
    if dk_dir != 0 and bets_pct < 40 and money_pct < 40:
        if dk_dir < 0 and sharp_score > 3:
            matched.append("SHARP_BOOK_CONFLICT")
        elif dk_dir > 0 and (sharp_score > 2 or pinnacle_moved):
            matched.append("BOOK_INITIATED_FOR")
        elif dk_dir < 0:
            matched.append("BOOK_INITIATED_AGAINST")
        # dk_dir > 0 but no sharp confirmation → juice noise, skip

    # SHARP_PUBLIC_SPLIT: Sharp books favor this side, public is on the other side
    if l1_present and sharp_score > 3 and bets_pct <= 35:
        matched.append("SHARP_PUBLIC_SPLIT")

    # PUBLIC_DRIFT: Heavy public + line toward public + NO sharp support (v3.3f)
    if bets_pct >= 70 and money_pct >= 70 and dk_dir != 0 and sharp_score <= 0:
        matched.append("PUBLIC_DRIFT")

    # CONSENSUS_HOLD: L2 strongly aligned without clear L1
    if not l1_present and l2_agreement >= 0.75:
        matched.append("CONSENSUS_HOLD")

    # RETAIL_CROWD: Extreme public, no sharp support
    if bets_pct >= 75 and money_pct >= 75 and sharp_score <= 0:
        matched.append("RETAIL_CROWD")

    primary = matched[0] if matched else "NEUTRAL"
    secondary = matched[1] if len(matched) > 1 else None
    return {"pattern_primary": primary, "pattern_secondary": secondary}


def _build_explanation(row, sharp, consensus, retail, timing, cross, score, market_rx=None) -> str:
    """Build one plain-English sentence explaining the score."""
    parts = []
    sport = (row.get("sport") or "").upper()
    side = row.get("favored_side") or "this side"

    if market_rx and market_rx.get("market_reaction_score", 0) > 0:
        parts.append("book-initiated movement detected")

    if sharp["sharp_score"] > 5:
        parts.append(f"sharp books strongly favor {side}")
    elif sharp["sharp_score"] > 0:
        parts.append(f"sharp books lean toward {side}")
    elif sharp["sharp_score"] < -5:
        parts.append(f"sharp books oppose {side}")
    elif sharp["sharp_score"] < 0:
        parts.append(f"sharp books lean against {side}")

    if consensus["consensus_score"] > 5:
        parts.append("consensus confirms")
    elif consensus["consensus_score"] < -3:
        parts.append("consensus disagrees")

    if retail["retail_score"] > 3:
        parts.append("smart money divergence detected")
    elif retail["retail_score"] < -3:
        parts.append("retail crowding concern")

    if not parts:
        return f"Score {score}: no strong signals in any component"

    return f"Score {score}: " + ", ".join(parts)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

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
