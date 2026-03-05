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
  - Market read scaled by D magnitude — D=25 gets bigger bonus than D=9
  - RLM reduced 50% when Pattern G fires — Pattern G is superior version
  - SPREAD line movement x2.0 (was x3.0) — root cause fix for SPREAD dampening
  - SPREAD dampening REMOVED (no longer needed with x2.0 line movement)
  - Color classification adaptive to D intensity + L1/L2 confirmation
  - Divergence scaled by bets/money intensity (continuous signal)
  - Smooth sample confidence curve (sigmoid replaces 3-bucket cliff)
  - ML vs Spread implied probability cross-check (+4/-3)
"""
import math
import pandas as pd

from engine_config import DK_ML_INST_MULT


def _bets_money_intensity(bets_pct: float, money_pct: float) -> float:
    """Continuous 0.0-1.0 intensity from bets/money split.
    Higher = stronger smart-money signal (money concentrated relative to bets).
    Ratio 1.0->0.5, 1.5->0.73, 2.0->0.88, 2.5->0.95"""
    if bets_pct <= 0:
        return 0.5  # No data — neutral
    ratio = money_pct / bets_pct
    intensity = 1.0 / (1.0 + math.exp(-2.0 * (ratio - 1.3)))
    return round(max(0.0, min(1.0, intensity)), 3)


def _implied_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    elif odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 0.5


def classify_line_movement(row: dict) -> dict:
    """
    Classify DK's line movement trajectory into a pattern.

    Uses tick-by-tick tracking from row_state to detect HOW the book
    moved, not just WHERE it ended up.

    Patterns:
      MOVE_AND_HOLD: Book moved early, then settled (found its number)
      FLAT: No movement at all (book confident at opening price)
      STEADY_DRIFT: Continuous same-direction movement (book adjusting)
      REVERSE: Direction changed once (book uncertain or tested)
      VOLATILE: Multiple direction changes (high uncertainty)

    Returns:
      {"pattern": str, "bonus": float, "explanation": str}
    """
    settled = int(_safe_float(row.get("line_settled_ticks"), 0))
    dir_changes = int(_safe_float(row.get("line_dir_changes"), 0))
    # v2.1: Use effective_move_mag (combines line number + juice movement)
    lm = abs(_safe_float(row.get("effective_move_mag", row.get("line_move_open"))))
    move_dir = int(_safe_float(row.get("move_dir")))
    tb = str(row.get("timing_bucket", "")).lower()
    D = _safe_float(row.get("divergence_D"))

    # Not enough ticks yet to classify
    if settled == 0 and dir_changes == 0 and tb == "early":
        return {"pattern": "EARLY", "bonus": 0.0, "explanation": "Too early to classify"}

    if dir_changes >= 2:
        # Multiple direction changes — book is uncertain
        return {
            "pattern": "VOLATILE",
            "bonus": -2.0,
            "explanation": f"Line reversed {dir_changes}x — book uncertain at this price",
        }

    if dir_changes == 1:
        # Changed direction once
        if lm < 0.3:
            # Reversed back near open — book tested and returned
            return {
                "pattern": "SNAP_BACK",
                "bonus": 1.0,
                "explanation": "Line reversed back to open — book tested market, confident at original",
            }
        else:
            return {
                "pattern": "REVERSE",
                "bonus": -1.0,
                "explanation": "Line changed direction — follow final position but lower confidence",
            }

    # No direction changes from here down
    if lm < 0.3 and settled >= 3:
        # No meaningful movement, stable for 3+ ticks
        # v2.1: boosted FLAT bonus (was 0.5/1.5, now 1.0/2.5)
        # DK sitting on an unchanged price = confidence in that number
        bonus = 2.5 if tb in ("mid", "late") else 1.0
        return {
            "pattern": "FLAT",
            "bonus": bonus,
            "explanation": f"Line hasn't moved ({settled} ticks) — DK confident at this price",
        }

    if lm >= 0.3 and settled >= 3:
        # Moved, then held for 3+ ticks — book found its number
        # v2.1: boosted bonuses (was 1.0-2.5, now 2.0-5.0)
        # This is the strongest DK trajectory signal: moved deliberately, then held
        if move_dir == 1 and D > 5:
            bonus = 5.0  # Book moved with concentrated money AND holds — strongest confirm
        elif move_dir == 1:
            bonus = 3.0  # Book moved with action and holds
        elif move_dir == -1:
            bonus = 2.0  # Book moved against and holds (fade already scored)
        else:
            bonus = 2.0
        return {
            "pattern": "MOVE_AND_HOLD",
            "bonus": bonus,
            "explanation": f"Line moved then held ({settled} ticks) — DK found its number",
        }

    if lm >= 0.3 and settled < 3:
        # Still actively moving — give partial credit since DK IS moving
        # v2.1: was 0.0, now 1.0 — active movement is itself a signal
        return {
            "pattern": "ACTIVE",
            "bonus": 1.0,
            "explanation": "Line actively moving — DK repricing",
        }

    return {"pattern": "UNKNOWN", "bonus": 0.0, "explanation": "Insufficient data"}


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
    # v2.1: widened range for better differentiation. Base stays at 50 for neutral,
    # rewards movement with higher base, penalizes genuinely empty early games.
    base_bets = _safe_float(row.get("bets_pct"))
    base_lm = abs(_safe_float(row.get("effective_move_mag", row.get("line_move_open"))))
    has_real_data = base_bets >= 30 or base_lm >= 0.3
    if tb == "early" and not has_real_data:
        score = 44.0  # Genuinely low-info early
        flags.append("base:low_info_early")
    elif tb == "late" and base_lm >= 0.5:
        score = 55.0  # Late with confirmed movement — highest base
        flags.append("base:late_with_movement")
    elif tb == "late" or (tb == "mid" and base_lm >= 0.5):
        score = 52.0
        flags.append("base:movement_boost")
    elif tb == "early" and has_real_data:
        score = 50.0  # Early but has signal
        flags.append("base:early_with_data")
    else:
        score = 50.0  # Mid baseline — neutral starting point
    details["dynamic_base"] = score

    # ── 2. MARKET READ BONUSES — "read the book, not the bettors" ──
    # Positive bonuses ONLY when the book's line movement CONFIRMS the money.
    # When book holds or moves against money → book disagrees → fade signal.
    # DK is a retail book — money there is recreational, not sharp.
    mr = str(row.get("market_read", "")).strip()
    # v2.1: boosted Stealth Move (4→6) and Aligned Sharp (2→4) — confirmed signals deserve more weight
    _MR_MAP = {
        "Stealth Move": 6,      # Concentrated money + book confirms → genuine edge
        "Aligned Sharp": 4,     # D≥8 + book confirms → decent but might be priced in
        "Freeze Pressure": -2,  # Money in, book HOLDS → book comfortable on other side
        "Reverse Pressure": -5, # Money in, book moves AGAINST → book actively fading
        "Contradiction": -3,    # Conflicting signals
        "Neutral": -1,
        "Public Drift": -6,     # Public momentum, likely wrong
    }
    mr_bonus_base = _MR_MAP.get(mr, 0)
    D_abs = abs(_safe_float(row.get("divergence_D")))
    if mr_bonus_base > 0 and D_abs > 0:
        # Scale positive bonuses by D intensity: D=8->x0.8, D=15->x1.0, D=25->x1.25
        d_scale = min(0.5 + (D_abs / 20.0), 1.25)
        mr_bonus = round(mr_bonus_base * d_scale, 1)
    elif mr_bonus_base < 0:
        # Negative signals get STRONGER with higher D
        d_scale = min(0.7 + (D_abs / 25.0), 1.3)
        mr_bonus = round(mr_bonus_base * d_scale, 1)
    else:
        mr_bonus = mr_bonus_base
    score += mr_bonus
    details["market_read"] = mr_bonus
    if mr_bonus != 0:
        flags.append(f"market_read:{mr}({mr_bonus:+.1f})")

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

    # Smooth sample confidence: sigmoid from 0.60 (0 bets) to 1.00 (40+ bets)
    bets_num = _safe_float(row.get("bets_pct"))
    if bets_num <= 0:
        sample_mult = 0.60
    else:
        sample_mult = 0.60 + 0.40 / (1.0 + math.exp(-0.15 * (bets_num - 15)))
        sample_mult = min(sample_mult, 1.00)

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

    # v2.1: raised caps (was 10/12) — divergence is the primary signal, shouldn't be capped so tight
    if mkt_upper == "TOTAL":
        div_raw_base = min(13.0, abs(D) * 0.35)
    else:
        div_raw_base = min(16.0, abs(D) * 0.5)

    combined_mult = div_mult * timing_mult * price_mult * inst_mult * sample_mult

    # GAP 1: Scale divergence by bets/money intensity (continuous signal)
    base_money = _safe_float(row.get("money_pct"))
    bm_intensity = _bets_money_intensity(base_bets, base_money)
    # Intensity scales the MAGNITUDE of the signal (both positive and negative)
    intensity_scale = 0.4 + (bm_intensity * 0.9)
    div_contrib = div_raw_base * combined_mult * intensity_scale

    # BOOK RESPONSE: The book's line movement relative to money is the key signal.
    # DK is retail — money there is recreational whales, not sharps.
    # What the BOOK does with that money tells you whether it's real edge or noise.
    # - Book confirms money (moves with) → positive divergence (trust the money)
    # - Book holds (no meaningful move) → mild negative (book absorbs, disagrees)
    # - Book fades money (moves against) → negative divergence (book says money is wrong)
    if D > 5 and move_dir == -1 and meaningful:
        # Book actively moves AGAINST money side: strong fade signal
        # Intensity AMPLIFIES the negative — more concentrated money being faded = stronger
        div_contrib = max(-5.0, -abs(div_contrib) * 0.5)
        flags.append(f"book_fades:{div_contrib:.1f}")
    elif D > 5 and (move_dir == 0 or not meaningful):
        # Book absorbs money without moving: mild fade signal
        div_contrib = max(-2.0, -abs(div_contrib) * 0.2)
        flags.append(f"book_holds:{div_contrib:.1f}")
    # else: book confirms (move_dir == +1) → keep positive div_contrib

    score += div_contrib
    details["divergence"] = round(div_contrib, 2)
    details["bm_intensity"] = bm_intensity
    details["div_mult_breakdown"] = {
        "div_mult": div_mult, "timing_mult": timing_mult,
        "price_mult": price_mult, "inst_mult": inst_mult,
        "sample_mult": round(sample_mult, 3),
        "intensity_scale": round(intensity_scale, 3),
    }

    # ── 6. LINE MOVEMENT (line number + juice-equivalent) ──
    # v2.1: Use effective_move_mag which combines line number AND juice/odds movement.
    # When DK shifts juice without moving the number, that's still a real pricing signal.
    try:
        eff_mag = float(row.get("effective_move_mag") or 0)
        if math.isnan(eff_mag): eff_mag = 0.0
    except Exception:
        eff_mag = 0.0
    # Fallback: if effective_move_mag not computed, use raw line_move_open
    if eff_mag == 0:
        try:
            lm = float(row.get("line_move_open") or 0)
            if math.isnan(lm): lm = 0.0
            eff_mag = abs(lm)
        except Exception:
            pass
    # v2.1: DK line movement is the most reliable signal we have.
    # Large moves should score proportionally higher — a $50 ML move is way
    # more meaningful than a $10 move, so use a 2-tier multiplier.
    if mkt_upper == "SPREAD":
        if eff_mag >= 1.5:
            lm_bonus = min(14.0, 4.0 + (eff_mag - 1.5) * 3.0)  # steeper for big moves
        else:
            lm_bonus = min(14.0, eff_mag * 2.5)
    else:
        if eff_mag >= 2.0:
            lm_bonus = min(16.0, 5.0 + (eff_mag - 2.0) * 2.5)  # ML/total big moves
        else:
            lm_bonus = min(16.0, eff_mag * 2.5)
    score += lm_bonus
    details["line_movement"] = round(lm_bonus, 2)
    details["effective_move_mag"] = round(eff_mag, 2)

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

    # ── 12. COLOR CLASSIFICATION — gated by book response ──
    # Color = concentrated money pattern (DARK_GREEN = high $, low bets).
    # But concentrated DK money is NOT inherently positive — it's retail whales.
    # The bonus is GATED by the book's line response:
    #   Book confirms (moves with) → full bonus
    #   Book holds → reduced bonus (book not convinced)
    #   Book fades (moves against) → inverted to penalty (book says money is wrong)
    color = str(row.get("color", "")).strip()
    color_bonus = 0
    l1_dir = int(_safe_float(row.get("l1_move_dir")))
    l1_avail = row.get("l1_available")

    # Book response gate for color bonus
    book_confirms = (move_dir == 1 and meaningful)
    book_holds = (move_dir == 0 or not meaningful)
    book_fades = (move_dir == -1 and meaningful)

    if color == "DARK_GREEN":
        if book_confirms:
            # v2.1: Book agrees with concentrated money → strong bonus, scale by D
            # Raised cap 8→12 — this is the strongest DK signal (money + book alignment)
            color_bonus = min(6.0 + (D_abs / 12.0) * 4.0, 12.0)
            if l1_dir == 1 and l1_avail:
                color_bonus = min(color_bonus + 2.0, 14.0)
                flags.append("color:l1_confirms")
            elif l1_dir == -1 and l1_avail:
                color_bonus = max(color_bonus * 0.5, 2.0)
                flags.append("color:l1_opposes")
        elif book_holds:
            # Book absorbs money without moving → small bonus at best
            color_bonus = 1.5
            flags.append("color:dark_green_book_holds")
        elif book_fades:
            # Book moves AGAINST concentrated money → this is a negative signal
            color_bonus = -3.0
            flags.append("color:dark_green_book_fades")
    elif color == "LIGHT_GREEN":
        if book_confirms:
            # v2.1: raised cap 3→5
            color_bonus = min(3.0 + (D_abs / 15.0) * 2.0, 5.0)
        elif book_holds:
            color_bonus = 0.5
        else:
            color_bonus = -1.0
    elif color == "RED":
        # Public pile-on — always negative, worse if book confirms the fade
        color_bonus = max(-4.0 - (D_abs / 15.0) * 2.0, -7.0)
    score += color_bonus
    details["color_classification"] = round(color_bonus, 1)

    # ── 12b. ML vs SPREAD IMPLIED PROBABILITY CROSS-CHECK ──
    prob_check = 0.0
    if sport_upper != "UFC":
        side_key = str(row.get("side_key", row.get("side", ""))).strip()
        ml_odds_val = _safe_float(context.get(f"ml_odds_{side_key}"))
        spread_odds_val = _safe_float(context.get(f"spread_odds_{side_key}"))
        if ml_odds_val != 0 and spread_odds_val != 0:
            ml_prob = _implied_prob(ml_odds_val)
            sp_prob = _implied_prob(spread_odds_val)
            prob_gap = ml_prob - sp_prob
            if abs(prob_gap) >= 0.05:  # Only act on meaningful gaps
                if mkt_upper == "MONEYLINE":
                    prob_check = max(-3.0, min(4.0, prob_gap * 40.0))
                elif mkt_upper == "SPREAD":
                    prob_check = max(-3.0, min(4.0, -prob_gap * 40.0))
                score += prob_check
                if prob_check != 0:
                    flags.append(f"prob_check:{prob_check:+.1f}")
    details["prob_check"] = round(prob_check, 2)

    # ── 12c. LINE MOVEMENT TRAJECTORY ──
    # DK runs a sophisticated pricing machine. HOW the line moved matters:
    # - Moved early then held = book found its number, high confidence
    # - Flat all day = book confident at opening price
    # - Reversed = book uncertain, lower confidence
    # - Multiple reversals = volatile, significantly lower confidence
    lm_pattern = classify_line_movement(row)
    lm_pattern_bonus = lm_pattern["bonus"]
    score += lm_pattern_bonus
    details["line_pattern"] = lm_pattern["pattern"]
    details["line_pattern_bonus"] = lm_pattern_bonus
    if lm_pattern_bonus != 0:
        flags.append(f"line_pattern:{lm_pattern['pattern']}({lm_pattern_bonus:+.1f})")

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
