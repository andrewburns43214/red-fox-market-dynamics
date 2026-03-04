"""
DraftKings Interpretation Rules for Red Fox engine v2.0.

DK data is RETAIL — it tells us where the public is, not where the market is going.
These rules transform DK signals into L3 contributions (+/-10 max).

Categories:
  1. Market Hierarchy — which DK markets to trust
  2. Retail Distortion — detect parlay/favorite bias
  3. Line Movement — DK confirms, never discovers
  4. Timing — early splits unreliable, late moves need confirmation
  5. Cross-Book — DK vs sharp/consensus alignment
"""
from engine_config import (
    DK_ML_INST_MULT,
    DK_DIVERGENCE_THRESHOLD,
    DK_ML_DIVERGENCE_THRESHOLD,
    DK_RETAIL_ALIGN_BETS_MIN,
    DK_RETAIL_ALIGN_MONEY_MIN,
    DK_RETAIL_ALIGN_PENALTY,
    DK_ML_ONLY_PENALTY,
)


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


# ─── 1. MARKET HIERARCHY ───

def market_weight(market: str) -> float:
    """
    DK market credibility weights.
    Spread is primary, ML is heavily discounted.
    """
    weights = {
        "SPREAD": 1.00,
        "TOTAL": 0.90,
        "MONEYLINE": DK_ML_INST_MULT,  # 0.60 in v2.0
    }
    return weights.get(market, 0.50)


def ml_only_penalty(row: dict) -> tuple:
    """
    Rule: If DK ML moved but spread didn't, it's likely pricing noise.
    Returns (penalty: float, flag: str).
    """
    ml_move = _safe_float(row.get("ml_move_dir", row.get("dk_ml_move", 0)))
    spread_move = _safe_float(row.get("spread_move_dir", row.get("dk_spread_move", 0)))
    market = row.get("market", "")

    if market == "MONEYLINE" and abs(ml_move) > 0 and abs(spread_move) == 0:
        return (DK_ML_ONLY_PENALTY, "ML_ONLY_NO_CONFIRM")

    return (0.0, "")


# ─── 2. RETAIL DISTORTION ───

def retail_alignment_penalty(row: dict) -> tuple:
    """
    Rule: High bets% + high money% on same side = parlay/retail alignment.
    Zero L3 contribution when this triggers.
    Returns (penalty: float, flag: str).
    """
    bets_pct = _safe_float(row.get("bets_pct", row.get("dk_bets_pct", 0)))
    money_pct = _safe_float(row.get("money_pct", row.get("dk_money_pct", 0)))

    if bets_pct >= DK_RETAIL_ALIGN_BETS_MIN and money_pct >= DK_RETAIL_ALIGN_MONEY_MIN:
        return (DK_RETAIL_ALIGN_PENALTY, "RETAIL_ALIGNMENT")

    return (0.0, "")


def parlay_distortion_penalty(row: dict) -> tuple:
    """
    Rule: DK money% > 80% on a favorite ML (odds < -150) is likely parlay-inflated.
    Returns (penalty: float, flag: str).
    """
    market = row.get("market", "")
    money_pct = _safe_float(row.get("money_pct", row.get("dk_money_pct", 0)))
    odds = _safe_int(row.get("dk_odds", row.get("odds_american", 0)))

    if market == "MONEYLINE" and money_pct > 80 and odds < -150:
        return (-4.0, "PARLAY_DISTORTION")

    return (0.0, "")


# ─── 3. LINE MOVEMENT ───

def dk_divergence_score(row: dict) -> float:
    """
    Score DK divergence (bets% vs money% gap).
    Only meaningful above threshold. DK exaggerates gaps.
    """
    bets_pct = _safe_float(row.get("bets_pct", 0))
    money_pct = _safe_float(row.get("money_pct", 0))
    market = row.get("market", "")

    divergence = abs(money_pct - bets_pct)

    # Market-specific thresholds
    threshold = DK_ML_DIVERGENCE_THRESHOLD if market == "MONEYLINE" else DK_DIVERGENCE_THRESHOLD

    if divergence < threshold:
        return 0.0

    # Sharp indicator: low bets% + high money% (few bettors, big money)
    if bets_pct < 40 and money_pct > 60:
        # Normalized excess divergence (0 to ~5)
        return min((divergence - threshold) / 10.0 * 5.0, 5.0)
    else:
        # Retail divergence (less meaningful)
        return min((divergence - threshold) / 10.0 * 2.0, 2.0)


def dk_line_move_score(row: dict) -> float:
    """
    Score DK line movement. DK confirms, never discovers.
    Requires >= 1.0 point move on spreads to generate signal.
    """
    market = row.get("market", "")
    move = _safe_float(row.get("dk_line_move", row.get("move_magnitude", 0)))

    if market == "SPREAD" and abs(move) < 1.0:
        return 0.0  # Half-point DK moves on spreads are noise

    # Scale: 1 point = 1.0, 2 points = 2.0, cap at 3.0
    return min(abs(move), 3.0)


def cross_market_confirmation(row: dict) -> tuple:
    """
    Rule: Do DK spread + ML + total agree?
    Returns (bonus: float, flag: str).
    """
    spread_dir = _safe_float(row.get("dk_spread_dir", row.get("spread_move_dir", 0)))
    ml_dir = _safe_float(row.get("dk_ml_dir", row.get("ml_move_dir", 0)))
    total_dir = _safe_float(row.get("dk_total_dir", row.get("total_move_dir", 0)))

    # All three agree
    if spread_dir != 0 and ml_dir != 0:
        if (spread_dir > 0) == (ml_dir > 0):
            return (2.0, "DK_CROSS_CONFIRM")
        else:
            return (-2.0, "DK_CROSS_CONTRADICT")

    return (0.0, "")


# ─── 4. TIMING ───

def timing_credibility(row: dict) -> float:
    """
    DK split credibility based on time to game.

    Early splits (>8 hours) are least reliable.
    Late splits (<1 hour) need multi-market confirmation.
    """
    hours_to_game = _safe_float(row.get("hours_to_game", row.get("time_to_start_hours", 4)))

    if hours_to_game > 8:
        return 0.60  # Early — DK splits barely informational
    elif hours_to_game > 4:
        return 0.80  # Mid — some signal
    elif hours_to_game > 1:
        return 1.00  # Prime — best DK data
    else:
        return 0.70  # Late — could be info or could be public balancing


# ─── 5. CROSS-BOOK RULES ───

def dk_vs_sharp_alignment(row: dict) -> tuple:
    """
    Rule: Compare DK line to sharp book (L1) line.
    Returns (bonus: float, flag: str).
    """
    l1_available = row.get("l1_available", False)
    if not l1_available:
        return (0.0, "")

    l1_dir = _safe_int(row.get("l1_move_dir", 0))
    dk_dir = _safe_float(row.get("dk_move_dir", row.get("move_dir", 0)))

    # DK matching sharp = high confidence
    if l1_dir != 0 and dk_dir != 0:
        if (l1_dir > 0) == (dk_dir > 0):
            return (3.0, "DK_MATCHES_SHARP")
        else:
            # DK moving opposite to sharp = DK managing internal liability
            return (-4.0, "DK_CONFLICTS_SHARP")

    return (0.0, "")


def dk_stale_vs_consensus(row: dict) -> tuple:
    """
    Rule: DK lagging consensus by >1 point = stale price opportunity (Pattern D).
    Returns (bonus: float, flag: str).
    """
    stale_flag = row.get("l2_stale_price_flag", False)
    stale_gap = _safe_float(row.get("l2_stale_price_gap", 0))

    if stale_flag and stale_gap >= 1.0:
        # Bigger gap = more value
        bonus = min(stale_gap * 2.0, 5.0)
        return (bonus, "STALE_PRICE")

    return (0.0, "")


# ─── COMPOSITE L3 CONTRIBUTION ───

def compute_l3_contribution(row: dict) -> dict:
    """
    Compute the DraftKings Layer 3 contribution.

    Returns dict with:
        "l3_contribution": float (-10 to +10)
        "l3_flags": list of str (active rule flags)
        "l3_details": dict (individual rule outputs)
    """
    market = row.get("market", "")
    mkt_weight = market_weight(market)
    flags = []
    details = {}

    # Base DK signal from divergence
    div_score = dk_divergence_score(row)
    details["divergence"] = div_score

    # DK line movement
    move_score = dk_line_move_score(row)
    details["line_move"] = move_score

    # Timing credibility
    timing = timing_credibility(row)
    details["timing_credibility"] = timing

    # Start with weighted base signal
    contribution = (div_score + move_score) * mkt_weight * timing

    # Apply rules (penalties and bonuses)

    # ML-only penalty
    ml_pen, ml_flag = ml_only_penalty(row)
    if ml_pen != 0:
        contribution += ml_pen
        flags.append(ml_flag)
    details["ml_only_penalty"] = ml_pen

    # Retail alignment
    retail_pen, retail_flag = retail_alignment_penalty(row)
    if retail_pen != 0:
        contribution += retail_pen
        flags.append(retail_flag)
    details["retail_alignment"] = retail_pen

    # Parlay distortion
    parlay_pen, parlay_flag = parlay_distortion_penalty(row)
    if parlay_pen != 0:
        contribution += parlay_pen
        flags.append(parlay_flag)
    details["parlay_distortion"] = parlay_pen

    # Cross-market confirmation
    cross_bonus, cross_flag = cross_market_confirmation(row)
    if cross_bonus != 0:
        contribution += cross_bonus
        flags.append(cross_flag)
    details["cross_market"] = cross_bonus

    # Cross-book alignment (L1 vs DK)
    sharp_bonus, sharp_flag = dk_vs_sharp_alignment(row)
    if sharp_bonus != 0:
        contribution += sharp_bonus
        flags.append(sharp_flag)
    details["sharp_alignment"] = sharp_bonus

    # Stale price
    stale_bonus, stale_flag = dk_stale_vs_consensus(row)
    if stale_bonus != 0:
        contribution += stale_bonus
        flags.append(stale_flag)
    details["stale_price"] = stale_bonus

    # Zero out L3 if retail alignment is too strong
    if "RETAIL_ALIGNMENT" in flags and retail_pen <= -5:
        contribution = 0.0
        flags.append("L3_ZEROED_RETAIL")

    # Clamp to [-10, +10]
    contribution = max(-10.0, min(10.0, contribution))

    return {
        "l3_contribution": round(contribution, 2),
        "l3_flags": flags,
        "l3_details": details,
    }
