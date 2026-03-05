"""
DK Scoring Module — Extracted from v1.2 main.py.

All 16 per-row DraftKings retail scoring components in a single function.
This is the foundation score for every row; L1/L2 adjustments are applied
on top by scoring_v2.py.

Components (in order):
  1. Dynamic base (44/50/52)
  2. Market read bonuses (halved: max ±5)
  3. Reverse Line Movement (RLM)
  4. Regime classifier (A/B/C/D/N)
  5. 5-factor divergence scoring
  6. Line movement (SPREAD x2.0, others x2.0)
  7. Key number crossing
  8. Timing bucket adjustment
  9. NCAAF early dampener
  10. NCAAB single-market penalty
  11. NHL puck line governor
  12. Color classification (enhanced with L1/L2)
  13. ML risk governor
  14. Sport-relative longshot penalty
  15. ML-only penalty
  16. Retail alignment penalty

Changes from v1.2 raw extraction:
  - Market read bonuses halved (was ±10, now ±5) — overlap with color classification
  - RLM reduced 50% when Pattern G fires — Pattern G is superior version
  - SPREAD line movement x2.0 (was x3.0) — root cause fix for SPREAD dampening
  - Late timing: -1 (was +0) — late splits are liability balancing, not signal
  - SPREAD dampening REMOVED (no longer needed with x2.0 line movement)
  - Color classification enhanced with L1/L2 direction confirmation
"""
import math
import pandas as pd

from engine_config import DK_ML_INST_MULT


# ─── Sport-specific STRONG eligibility config ───
# Moved from main.py top-level constants
NCAAB_EARLY_STRONG_BLOCK = True
NCAAB_STRONG_MIN_PERSIST = 3
NCAAB_STRONG_STABILITY_DELTA = 2
NCAAB_LATE_STRONG_BLOCK = True
NCAAB_REQUIRE_MULTI_MARKET = True

NCAAF_EARLY_INSTANT_STRONG_BLOCK = True
NCAAF_STRONG_STABILITY_DELTA = 3
NCAAF_LATE_NEW_STRONG_BLOCK = True

# Sport-relative longshot baselines (implied probability)
SPORT_LONGSHOT_BASELINE = {
    "NBA": 0.35, "NCAAB": 0.30, "NHL": 0.28,
    "NFL": 0.30, "NCAAF": 0.28, "MLB": 0.32, "UFC": 0.25,
}


def _safe_float(val, default=0.0):
    """Convert value to float safely, handling None/NaN/strings."""
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        out = float(val)
        return default if math.isnan(out) else out
    except Exception:
        return default


def compute_dk_base(row: dict, context: dict = None) -> dict:
    """
    Full DraftKings retail scoring — all 16 v1.2 components.

    Args:
        row: dict with DK row data (bets_pct, money_pct, move_dir, etc.)
        context: dict with cross-row data:
            - spread_move_map: {game_id|side_key -> {lm, dir, meaningful}}
            - mkt_count: {(sport, game_id) -> count of distinct markets}
            - pattern: current v2 pattern (for RLM dedup with Pattern G)

    Returns:
        dict with:
            - dk_base_score: float (0-100)
            - regime: str (A/B/C/D/N)
            - color: str (from classify_side, if provided)
            - dk_flags: list of applied component labels
            - dk_details: dict of component-level scores
    """
    context = context or {}
    spread_move_map = context.get("spread_move_map", {})
    mkt_count = context.get("mkt_count", {})
    pattern = context.get("pattern", "")

    flags = []
    details = {}

    mkt = str(row.get("market_display", row.get("market", ""))).strip()
    mkt_upper = mkt.upper()
    sport_upper = str(row.get("sport", "")).strip().upper()
    tb = str(row.get("timing_bucket", "")).strip().lower()

    # ── 1. DYNAMIC BASE SCORE ──
    # Based on data quality, not just timing. Early games with real signals
    # shouldn't start 8 points behind a mid game with weak splits.
    base_bets = _safe_float(row.get("bets_pct"))
    base_lm = abs(_safe_float(row.get("line_move_open")))
    has_real_data = base_bets >= 30 or base_lm >= 0.5
    if tb == "early" and not has_real_data:
        score = 46.0  # Genuinely low-info early (was 44)
        flags.append("base:low_info_early")
    elif tb == "late" or (tb == "mid" and base_lm >= 1.0):
        score = 52.0
        flags.append("base:movement_boost")
    elif tb == "early" and has_real_data:
        score = 50.0  # Early but has signal — same as mid baseline
        flags.append("base:early_with_data")
    else:
        score = 50.0
    details["dynamic_base"] = score

    # ── 2. MARKET READ BONUSES (halved from v1.2: max ±5) ──
    mr = str(row.get("market_read", "")).strip()
    mr_bonus = 0
    if mr == "Stealth Move":
        mr_bonus = 4
    elif mr == "Freeze Pressure":
        mr_bonus = 5
    elif mr == "Aligned Sharp":
        mr_bonus = 3
    elif mr == "Reverse Pressure":
        mr_bonus = 4
    elif mr == "Contradiction":
        mr_bonus = -2
    elif mr == "Neutral":
        mr_bonus = -1
    elif mr == "Public Drift":
        mr_bonus = -5
    score += mr_bonus
    details["market_read"] = mr_bonus
    if mr_bonus != 0:
        flags.append(f"market_read:{mr}({mr_bonus:+d})")

    # ── 3. REVERSE LINE MOVEMENT (RLM) ──
    rlm_score = 0.0
    try:
        rlm_bets = _safe_float(row.get("bets_pct"))
        rlm_money = _safe_float(row.get("money_pct"))
        rlm_move_dir = int(_safe_float(row.get("move_dir")))
        rlm_mv = row.get("meaningful_move", False)
        if isinstance(rlm_mv, str):
            rlm_meaningful = rlm_mv.strip().lower() in {"1", "true", "yes", "y"}
        else:
            rlm_meaningful = bool(rlm_mv)

        is_public_majority = rlm_bets >= 55
        line_moved_against = rlm_move_dir == -1 and rlm_meaningful

        if is_public_majority and line_moved_against:
            if rlm_bets >= 75:
                rlm_score = 8.0
            elif rlm_bets >= 65:
                rlm_score = 6.0
            else:
                rlm_score = 4.0
            if rlm_money >= 60 and rlm_bets >= 65:
                rlm_score += 2.0

            # Reduce by 50% when Pattern G fires (Pattern G is the cross-layer RLM)
            if pattern == "G":
                rlm_score *= 0.5
                flags.append("rlm:dedup_pattern_g")

            score += rlm_score
            flags.append(f"rlm:{rlm_score:+.1f}")
    except Exception:
        pass
    details["rlm"] = rlm_score

    # ── 4. REGIME CLASSIFIER ──
    D = _safe_float(row.get("divergence_D"))
    bets = _safe_float(row.get("bets_pct"))
    money = _safe_float(row.get("money_pct"))

    move_dir = int(_safe_float(row.get("move_dir")))
    mv = row.get("meaningful_move", False)
    if isinstance(mv, str):
        meaningful = mv.strip().lower() in {"1", "true", "yes", "y"}
    else:
        meaningful = bool(mv)

    if bets < 30 and money >= 55 and move_dir == -1 and meaningful:
        regime = "A"; div_mult = 0.40
    elif bets >= 55 and money >= 55 and move_dir == -1 and meaningful:
        regime = "B"; div_mult = 0.0
    elif bets < 25 and money < 40 and move_dir == +1 and meaningful:
        regime = "C"; div_mult = 1.20
    elif D >= 8 and move_dir == +1:
        regime = "D"; div_mult = 1.00
    else:
        regime = "N"; div_mult = 0.85
    details["regime"] = regime

    # ── 5. DIVERGENCE SCORING — combined multiplier ──
    contradiction = (D > 8 and move_dir == -1 and meaningful) or \
                    (D < -8 and move_dir == +1 and meaningful)
    if contradiction:
        div_mult = min(div_mult, 0.60)

    timing_mult = {"early": 0.85, "mid": 0.92, "late": 1.00}.get(tb, 0.88)
    ml_mult = DK_ML_INST_MULT
    inst_mult = {"SPREAD": 1.00, "MONEYLINE": ml_mult, "TOTAL": 0.90}.get(mkt_upper, 0.85)

    # Puck/run line dampener (±1.5 lines are uniform)
    side_str = str(row.get("side", "")).lower()
    if mkt_upper == "SPREAD" and any(t in side_str for t in ["+1.5", "-1.5", "+1", "-1"]):
        if sport_upper == "NHL":
            inst_mult = 0.85
        elif sport_upper == "MLB":
            inst_mult = 0.90

    bets_num = _safe_float(row.get("bets_pct"))
    sample_mult = 0.65 if bets_num < 10 else (0.80 if bets_num < 20 else 1.00)

    price_mult = 1.00
    if mkt_upper == "MONEYLINE":
        try:
            odds = _safe_float(row.get("current_odds"))
            if odds > 0:
                impl_prob = 100 / (odds + 100)
            elif odds < 0:
                impl_prob = abs(odds) / (abs(odds) + 100)
            else:
                impl_prob = 0.5
            if impl_prob < 0.15:
                price_mult = 0.50
            elif impl_prob < 0.20:
                price_mult = 0.60
            elif impl_prob < 0.25:
                price_mult = 0.70
            elif impl_prob < 0.33:
                price_mult = 0.85
        except Exception:
            pass

    if mkt_upper == "TOTAL":
        div_raw_base = min(10.0, abs(D) * 0.3)
    else:
        div_raw_base = min(12.0, abs(D) * 0.4)

    combined_mult = div_mult * timing_mult * price_mult * inst_mult * sample_mult
    div_contrib = div_raw_base * combined_mult
    score += div_contrib
    details["divergence"] = round(div_contrib, 2)
    details["div_mult_breakdown"] = {
        "div_mult": div_mult, "timing_mult": timing_mult,
        "price_mult": price_mult, "inst_mult": inst_mult,
        "sample_mult": sample_mult,
    }

    # ── 6. LINE MOVEMENT (SPREAD x2.0, others x2.0) ──
    try:
        lm = float(row.get("line_move_open")) if row.get("line_move_open") is not None else 0.0
        if isinstance(lm, float) and math.isnan(lm):
            lm = 0.0
    except Exception:
        lm = 0.0
    # SPREAD: x2.0 (was x3.0 in v1.2 — root cause fix for SPREAD inflation)
    if mkt_upper == "SPREAD":
        lm_bonus = min(7.0, abs(lm) * 2.0)
    else:
        lm_bonus = min(8.0, abs(lm) * 2.0)
    score += lm_bonus
    details["line_movement"] = round(lm_bonus, 2)

    # ── 7. KEY NUMBER CROSSING ──
    # Key numbers (3, 7, 10, 14) only matter for football (NFL/NCAAF).
    # NBA/NCAAB spreads are continuous — crossing 7 means nothing.
    # NHL/MLB puck/run lines are fixed ±1.5 — no key number concept.
    kn_bonus = 0
    kn_note = str(row.get("key_number_note", "")).strip()
    if kn_note:
        if sport_upper in ("NFL", "NCAAF") and mkt_upper == "SPREAD":
            kn_bonus = 6  # Football key numbers are massive (3, 7)
            flags.append(f"key_number:+{kn_bonus} ({kn_note})")
        elif sport_upper in ("NFL", "NCAAF") and mkt_upper == "TOTAL":
            kn_bonus = 3  # Totals key numbers matter less
            flags.append(f"key_number:+{kn_bonus}")
        # NBA/NCAAB/NHL/MLB/UFC: no key number bonus
    score += kn_bonus
    details["key_number"] = kn_bonus

    # ── 8. TIMING BUCKET ──
    # No additive timing penalty — timing already handled via dynamic base
    # (46 vs 50 vs 52) and divergence multiplier (0.85/0.92/1.00).
    # Double-dipping suppressed scores unnecessarily.
    timing_adj = 0
    if tb == "mid":
        timing_adj = 1
    score += timing_adj
    details["timing"] = timing_adj

    # ── 9. NCAAF EARLY DAMPENER ──
    ncaaf_early = 0
    if sport_upper == "NCAAF" and tb == "early":
        ncaaf_early = -2
        score += ncaaf_early
        flags.append("ncaaf_early:-2")
    details["ncaaf_early"] = ncaaf_early

    # ── 10. NCAAB SINGLE-MARKET PENALTY ──
    ncaab_single = 0
    if sport_upper == "NCAAB":
        gid = str(row.get("game_id", "")).strip()
        if gid and mkt_count.get(("NCAAB", gid), 0) <= 1:
            ncaab_single = -3
            score += ncaab_single
            flags.append("ncaab_single_market:-3")
    details["ncaab_single_market"] = ncaab_single

    # ── 11. NHL PUCK LINE GOVERNOR ──
    nhl_puck = 0
    if sport_upper == "NHL" and mkt_upper == "SPREAD":
        nhl_puck = -3
        score += nhl_puck
        flags.append("nhl_puck_line:-3")
    details["nhl_puck_line"] = nhl_puck

    # ── 12. COLOR CLASSIFICATION (enhanced with L1/L2) ──
    color = str(row.get("color", "")).strip()
    color_bonus = 0
    if color == "DARK_GREEN":
        color_bonus = 6
        # Enhancement: L1 sharps confirm DK color direction
        l1_dir = int(_safe_float(row.get("l1_move_dir")))
        if l1_dir == 1 and row.get("l1_available"):
            color_bonus = 8  # L1 confirms: stronger
            flags.append("color:l1_confirms_dark_green")
        elif l1_dir == -1 and row.get("l1_available"):
            color_bonus = 3  # L1 opposes: weaker
            flags.append("color:l1_opposes_dark_green")
    elif color == "LIGHT_GREEN":
        color_bonus = 3
    elif color == "RED":
        color_bonus = -6
    score += color_bonus
    details["color_classification"] = color_bonus

    score = max(0.0, min(100.0, score))

    # ── 13. ML RISK GOVERNOR ──
    ml_risk = 0
    if mkt_upper == "MONEYLINE":
        try:
            odds = row.get("current_odds")
            if odds is not None:
                o = int(float(odds))
                if -109 <= o <= 109:
                    ml_risk = 0
                elif (-180 <= o <= -110) or (100 <= o <= 180):
                    ml_risk = 0
                elif (-250 <= o <= -181) or (181 <= o <= 250):
                    ml_risk = -2
                elif (-400 <= o <= -251) or (251 <= o <= 400):
                    ml_risk = -4
                else:
                    ml_risk = -6
                score += ml_risk
                if ml_risk:
                    flags.append(f"ml_risk:{ml_risk}")
        except Exception:
            pass
    details["ml_risk_governor"] = ml_risk

    # ── 14. SPORT-RELATIVE LONGSHOT PENALTY ──
    longshot = 0
    if mkt_upper == "MONEYLINE":
        try:
            ls_odds = _safe_float(row.get("current_odds"))
            baseline = SPORT_LONGSHOT_BASELINE.get(sport_upper, 0.30)
            if ls_odds > 0:
                impl_prob = 100 / (ls_odds + 100)
            elif ls_odds < 0:
                impl_prob = abs(ls_odds) / (abs(ls_odds) + 100)
            else:
                impl_prob = 0.5
            prob_gap = max(0.0, baseline - impl_prob)
            if prob_gap > 0.15:
                longshot = -10
            elif prob_gap > 0.10:
                longshot = -7
            elif prob_gap > 0.05:
                longshot = -4
            score += longshot
            if longshot:
                flags.append(f"longshot:{longshot}")
        except Exception:
            pass
    details["longshot_penalty"] = longshot

    # ── 15. ML-ONLY PENALTY ──
    ml_only = 0
    if mkt_upper == "MONEYLINE":
        try:
            ml_side_key = "{}|{}".format(
                str(row.get("game_id", "")).strip(),
                str(row.get("side_key", "")).strip().lower(),
            )
            sp_info = spread_move_map.get(ml_side_key)
            ml_lm = abs(float(row.get("line_move_open", 0) or 0))
            ml_mv = row.get("meaningful_move", False)
            ml_has_move = ml_lm > 0 and bool(ml_mv)
            if ml_has_move and sp_info and not sp_info.get("meaningful"):
                ml_only = -3
                score += ml_only
                flags.append("ml_only:-3")
        except Exception:
            pass
    details["ml_only_penalty"] = ml_only

    # ── 16. RETAIL ALIGNMENT PENALTY ──
    retail_align = 0
    if mkt_upper == "MONEYLINE":
        try:
            ra_bets = float(row.get("bets_pct", 0) or 0)
            ra_money = float(row.get("money_pct", 0) or 0)
            if ra_bets > 70 and ra_money > 70:
                retail_align = -5
                score += retail_align
                flags.append(f"retail_align:-5 (bets {ra_bets:.0f}% money {ra_money:.0f}%)")
        except Exception:
            pass
    details["retail_alignment"] = retail_align

    # Final clamp
    score = max(0.0, min(100.0, score))

    return {
        "dk_base_score": round(score, 3),
        "regime": regime,
        "color": color,
        "dk_flags": flags,
        "dk_details": details,
    }


def is_strong_eligible(row: dict) -> bool:
    """
    Unified STRONG_BET eligibility check.

    Requirements:
      1. score >= 70
      2. layer_mode == "L123" (all 3 data sources)
      3. pattern in ("A", "D", "G") — strong signal patterns
      4. strong_streak >= min_streak (sport-specific)
      5. last_score within delta of peak_score (stability)
      6. Not LATE timing bucket
      7. Sport-specific blocks (NCAAB/NCAAF early, NCAAB multi-market)

    Args:
        row: dict with scoring output + row_state data

    Returns:
        True if row qualifies for STRONG_BET.
    """
    try:
        score = float(row.get("confidence_score",
                               row.get("game_confidence",
                                        row.get("dk_base_score", 0))))
    except Exception:
        score = 0.0

    if score < 70:
        return False

    # Timing gate
    tb = str(row.get("timing_bucket", "")).strip().upper()
    if tb == "LATE":
        return False

    # Layer mode gate — require L123 (all 3 data sources)
    layer_mode = str(row.get("layer_mode", "")).strip()
    if layer_mode != "L123":
        return False

    # Pattern gate — only strong signal patterns
    pattern = str(row.get("v2_pattern", row.get("pattern", ""))).strip()
    if pattern not in ("A", "D", "G"):
        return False

    # MLB price-band override for cross-market contradiction
    sport = str(row.get("sport", "")).strip().upper()
    mkt = str(row.get("market", row.get("market_display", ""))).strip().upper()
    ml_price_band_override = False
    if sport == "MLB" and mkt == "MONEYLINE":
        try:
            od = float(row.get("current_odds", 0))
            if -210 <= od <= -135:
                ml_price_band_override = True
        except Exception:
            pass

    # Persistence: require consecutive STRONG-eligible snapshots
    try:
        ss = int(str(row.get("strong_streak", "0")).strip() or "0")
    except Exception:
        ss = 0

    min_streak = NCAAB_STRONG_MIN_PERSIST if sport == "NCAAB" else 2
    if ss < min_streak:
        return False

    # Stability: last_score close to peak_score
    try:
        ls = float(row.get("last_score", "0"))
        ps = float(row.get("peak_score", "0"))
    except Exception:
        return False

    # Sport-specific blocks
    if sport == "NCAAB":
        if NCAAB_EARLY_STRONG_BLOCK and tb == "EARLY":
            return False
        if NCAAB_LATE_STRONG_BLOCK and tb == "LATE":
            return False
    if sport == "NCAAF":
        if NCAAF_EARLY_INSTANT_STRONG_BLOCK and tb == "EARLY":
            return False
        if NCAAF_LATE_NEW_STRONG_BLOCK and tb == "LATE":
            return False

    # NCAAB multi-market requirement
    if sport == "NCAAB" and NCAAB_REQUIRE_MULTI_MARKET:
        spread_ok = str(row.get("SPREAD_favored", "")).strip() != ""
        ml_ok = str(row.get("MONEYLINE_favored", "")).strip() != ""
        if not (spread_ok and ml_ok):
            return False

    # Stability delta
    delta = (NCAAB_STRONG_STABILITY_DELTA if sport == "NCAAB"
             else NCAAF_STRONG_STABILITY_DELTA if sport == "NCAAF"
             else 3.0)
    if ls < (ps - delta):
        return False

    return True
