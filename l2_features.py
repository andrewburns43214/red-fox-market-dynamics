"""
Layer 2 Feature Extraction for Red Fox engine.

Computes consensus features from l2_consensus_agg.csv:
  - Consensus agreement (% of books moving same direction as L1)
  - Dispersion and dispersion multiplier
  - Dispersion trend (tightening vs widening)
  - Pinnacle vs consensus gap
  - Stale price detection (DK lagging consensus)
  - Validation strength composite
"""
import csv
import os
import statistics
from collections import defaultdict

from engine_config import (
    L2_CONSENSUS_AGG_CSV,
    L2_CONSENSUS_CSV,
    DISPERSION_TIGHT_SPREAD,
    DISPERSION_TIGHT_TOTAL,
    DISPERSION_WIDE_SPREAD,
    DISPERSION_WIDE_TOTAL,
    DISPERSION_VERY_WIDE_SPREAD,
    DISPERSION_VERY_WIDE_TOTAL,
    DISPERSION_TIGHT_MULT,
    DISPERSION_NORMAL_MULT,
    DISPERSION_WIDE_MULT,
    DISPERSION_VERY_WIDE_MULT,
)


def _parse_float(val, default=0.0) -> float:
    try:
        return float(val) if val not in ("", None) else default
    except (ValueError, TypeError):
        return default


def _parse_int(val, default=0) -> int:
    try:
        return int(float(val)) if val not in ("", None) else default
    except (ValueError, TypeError):
        return default


def _load_consensus_agg() -> dict:
    """
    Load latest L2 consensus aggregation.

    Returns dict keyed by (canonical_key, market, side) → agg row dict.
    """
    if not os.path.exists(L2_CONSENSUS_AGG_CSV):
        return {}

    result = {}
    try:
        with open(L2_CONSENSUS_AGG_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = (
                    row.get("canonical_key", ""),
                    row.get("market", ""),
                    row.get("side", ""),
                )
                result[k] = row
    except Exception as e:
        print(f"[WARN] l2 consensus agg load: {repr(e)}")
    return result


def _load_raw_consensus_latest() -> dict:
    """
    Load raw L2 consensus data, keeping only the latest snapshot per bookmaker.

    Returns dict keyed by (canonical_key, market, side) → list of book entries.
    """
    if not os.path.exists(L2_CONSENSUS_CSV):
        return {}

    # Keep latest per (canon, market, side, bookmaker)
    latest = {}
    try:
        with open(L2_CONSENSUS_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = (
                    row.get("canonical_key", ""),
                    row.get("market", ""),
                    row.get("side", ""),
                    row.get("bookmaker", ""),
                )
                ts = row.get("timestamp", "")
                if k not in latest or ts > latest[k]["timestamp"]:
                    latest[k] = row
    except Exception as e:
        print(f"[WARN] l2 raw consensus load: {repr(e)}")
        return {}

    # Group by (canonical_key, market, side)
    grouped = defaultdict(list)
    for (canon, market, side, bm), row in latest.items():
        grouped[(canon, market, side)].append(row)

    return dict(grouped)


def _compute_dispersion_mult(line_std: float, market: str) -> tuple:
    """
    Compute dispersion category and multiplier.

    Returns (category_label, multiplier).
    """
    if market == "TOTAL":
        tight = DISPERSION_TIGHT_TOTAL
        wide = DISPERSION_WIDE_TOTAL
        very_wide = DISPERSION_VERY_WIDE_TOTAL
    else:
        tight = DISPERSION_TIGHT_SPREAD
        wide = DISPERSION_WIDE_SPREAD
        very_wide = DISPERSION_VERY_WIDE_SPREAD

    if line_std <= tight:
        return ("TIGHT", DISPERSION_TIGHT_MULT)
    elif line_std <= wide:
        return ("NORMAL", DISPERSION_NORMAL_MULT)
    elif line_std <= very_wide:
        return ("WIDE", DISPERSION_WIDE_MULT)
    else:
        return ("VERY_WIDE", DISPERSION_VERY_WIDE_MULT)


def _compute_dispersion_trend(current_std: float, prev_std: float) -> str:
    """
    Determine if dispersion is tightening, stable, or widening.

    Tightening = books converging (confirms move).
    Widening = books diverging (move may be noise).
    """
    if prev_std <= 0:
        return "UNKNOWN"

    change_ratio = current_std / prev_std if prev_std > 0 else 1.0

    if change_ratio < 0.85:
        return "TIGHTENING"
    elif change_ratio > 1.15:
        return "WIDENING"
    else:
        return "STABLE"


def compute_l2_features(sport: str = None, l1_features: dict = None) -> dict:
    """
    Compute Layer 2 features for all games (or filtered by sport).

    Args:
        sport: Optional sport filter
        l1_features: Optional dict from l1_features.compute_l1_features()
                     Used to compute agreement (% of consensus books matching L1 direction)

    Returns:
        dict keyed by (canonical_key, market, side) → feature dict:
        {
            "l2_n_books": int,
            "l2_consensus_line": float,
            "l2_consensus_odds": int,
            "l2_dispersion": float,        # line_std
            "l2_dispersion_label": str,    # TIGHT/NORMAL/WIDE/VERY_WIDE
            "l2_dispersion_mult": float,   # 0.40 - 1.20
            "l2_dispersion_trend": str,    # TIGHTENING/STABLE/WIDENING
            "l2_pinn_vs_consensus": float, # Pinnacle - consensus
            "l2_pinn_line": float,
            "l2_consensus_agreement": float,  # 0-1 (% of books matching L1 dir)
            "l2_stale_price_flag": bool,   # DK lagging consensus
            "l2_stale_price_gap": float,   # how many points DK is stale by
            "l2_validation_strength": float, # composite 0-1
            "l2_available": True,
        }
    """
    agg_data = _load_consensus_agg()

    features = {}

    for (canon, market, side), agg_row in agg_data.items():
        # Filter by sport if specified
        if sport:
            row_sport = agg_row.get("sport", "")
            if row_sport != sport.lower():
                continue

        n_books = _parse_int(agg_row.get("n_books"))
        consensus_line = _parse_float(agg_row.get("consensus_line"))
        consensus_odds = _parse_int(agg_row.get("consensus_odds"))
        line_std = _parse_float(agg_row.get("line_std"))
        line_std_prev = _parse_float(agg_row.get("line_std_prev"))
        pinn_line = _parse_float(agg_row.get("pinn_line"))
        pinn_odds = _parse_int(agg_row.get("pinn_odds"))
        pinn_vs = _parse_float(agg_row.get("pinn_vs_consensus"))

        # Dispersion
        disp_label, disp_mult = _compute_dispersion_mult(line_std, market)
        disp_trend = _compute_dispersion_trend(line_std, line_std_prev)

        # Consensus agreement with L1
        consensus_agreement = 0.0
        if l1_features:
            l1_key = (canon, market, side)
            l1_feat = l1_features.get(l1_key)
            if l1_feat and l1_feat.get("l1_move_dir", 0) != 0:
                consensus_agreement = _compute_consensus_agreement(
                    canon, market, side, l1_feat["l1_move_dir"]
                )

        # Stale price detection (placeholder — computed during merge with DK data)
        stale_flag = False
        stale_gap = 0.0

        # Composite validation strength
        validation = _compute_validation_strength(
            n_books, disp_mult, disp_trend, consensus_agreement, abs(pinn_vs),
        )

        features[(canon, market, side)] = {
            "l2_n_books": n_books,
            "l2_consensus_line": consensus_line,
            "l2_consensus_odds": consensus_odds,
            "l2_dispersion": round(line_std, 3),
            "l2_dispersion_label": disp_label,
            "l2_dispersion_mult": disp_mult,
            "l2_dispersion_trend": disp_trend,
            "l2_pinn_vs_consensus": round(pinn_vs, 3),
            "l2_pinn_line": pinn_line,
            "l2_consensus_agreement": round(consensus_agreement, 3),
            "l2_stale_price_flag": stale_flag,
            "l2_stale_price_gap": stale_gap,
            "l2_validation_strength": round(validation, 3),
            "l2_available": True,
        }

    return features


def _compute_consensus_agreement(canon: str, market: str, side: str,
                                  l1_direction: int) -> float:
    """
    Compute what % of consensus books are moving in the same direction as L1.

    Loads raw L2 data and compares each book's line vs consensus to determine
    if they're leaning with or against L1's direction.

    Returns 0.0 - 1.0
    """
    raw_data = _load_raw_consensus_latest()
    key = (canon, market, side)
    book_entries = raw_data.get(key, [])

    if not book_entries or l1_direction == 0:
        return 0.0

    # Get median line as reference
    lines = []
    for entry in book_entries:
        line = _parse_float(entry.get("line"))
        if line != 0.0 or entry.get("line") == "0":
            lines.append(line)

    if not lines:
        return 0.0

    median_line = statistics.median(lines)

    # Count books whose line is above/below median in the L1 direction
    agreeing = 0
    total = 0
    for entry in book_entries:
        line = _parse_float(entry.get("line"))
        if line == 0.0 and entry.get("line") != "0":
            continue
        total += 1

        deviation = line - median_line
        if l1_direction > 0 and deviation > 0:
            agreeing += 1
        elif l1_direction < 0 and deviation < 0:
            agreeing += 1
        elif abs(deviation) < 0.01:
            agreeing += 0.5  # neutral, half credit

    return agreeing / total if total > 0 else 0.0


def _compute_validation_strength(n_books: int, disp_mult: float,
                                  disp_trend: str, agreement: float,
                                  pinn_gap: float) -> float:
    """
    Compute composite validation strength (0 to 1).

    Weights:
      - Book count:    0.15 (more books = stronger consensus)
      - Dispersion:    0.30 (tighter = stronger)
      - Trend:         0.15 (tightening = confirming)
      - Agreement:     0.30 (books agreeing with L1 = strongest signal)
      - Pinn proximity:0.10 (small Pinn gap = books aligned with sharp)
    """
    # Book count (15 books = 0.5, 30+ = 1.0)
    book_score = min(n_books / 30.0, 1.0)
    book_component = book_score * 0.15

    # Dispersion (mult 0.40-1.20 → normalize to 0-1)
    disp_score = min((disp_mult - 0.40) / 0.80, 1.0)
    disp_component = disp_score * 0.30

    # Trend
    trend_scores = {"TIGHTENING": 1.0, "STABLE": 0.5, "WIDENING": 0.1, "UNKNOWN": 0.3}
    trend_component = trend_scores.get(disp_trend, 0.3) * 0.15

    # Agreement (already 0-1)
    agree_component = agreement * 0.30

    # Pinnacle proximity (small gap = good, >2 pts = bad)
    pinn_score = max(0.0, 1.0 - (pinn_gap / 2.0))
    pinn_component = pinn_score * 0.10

    total = book_component + disp_component + trend_component + agree_component + pinn_component
    return min(max(total, 0.0), 1.0)
