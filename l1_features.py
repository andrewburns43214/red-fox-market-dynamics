"""
Layer 1 Feature Extraction for Red Fox engine.

Computes sharp book features from l1_sharp.csv and l1_open_registry.csv:
  - Move direction and magnitude
  - Sharp book agreement (how many of 6 books agree)
  - Limit-weighted direction (higher limits = more confident)
  - Leader detection (which book moved first via changedAt)
  - Move speed (from changedAt timestamps — no multi-snapshot needed)
  - Stability (did sharps hold their position?)
  - Key number crossing detection
"""
import csv
import os
import statistics
from datetime import datetime, timezone
from collections import defaultdict

from engine_config import (
    L1_SHARP_CSV,
    L1_OPEN_REGISTRY_CSV,
    KEY_NUMBERS,
    FAST_SNAP_SPEED,
    SLOW_GRIND_SPEED,
)


# Magnitude normalization by sport/market (1 point of spread movement ≠ 1 point of total)
MAGNITUDE_NORMS = {
    # (sport, market) → typical meaningful move size
    ("nba", "SPREAD"): 2.0,
    ("nba", "TOTAL"): 3.0,
    ("nba", "MONEYLINE"): 30.0,  # cents
    ("nhl", "SPREAD"): 0.5,
    ("nhl", "TOTAL"): 0.5,
    ("nhl", "MONEYLINE"): 20.0,
    ("mlb", "SPREAD"): 0.5,
    ("mlb", "TOTAL"): 1.0,
    ("mlb", "MONEYLINE"): 20.0,
    ("nfl", "SPREAD"): 2.0,
    ("nfl", "TOTAL"): 2.0,
    ("nfl", "MONEYLINE"): 30.0,
    ("ncaab", "SPREAD"): 2.0,
    ("ncaab", "TOTAL"): 3.0,
    ("ncaab", "MONEYLINE"): 30.0,
    ("ncaaf", "SPREAD"): 2.5,
    ("ncaaf", "TOTAL"): 3.0,
    ("ncaaf", "MONEYLINE"): 30.0,
}

DEFAULT_MAGNITUDE_NORM = 2.0

# Sharp agreement multipliers
AGREEMENT_MULTIPLIERS = {
    1: 1.00,
    2: 1.15,
    3: 1.30,
    4: 1.40,
    5: 1.50,
    6: 1.50,  # cap at 1.5x even with all 6 books
}


def _load_latest_l1() -> dict:
    """
    Load the latest L1 sharp data per (canonical_key, market, side, bookmaker).

    Returns dict keyed by (canonical_key, market, side) → list of book entries.
    """
    if not os.path.exists(L1_SHARP_CSV):
        return {}

    # Read all rows, keep only the latest timestamp per (canon, market, side, bookmaker)
    latest = {}
    try:
        with open(L1_SHARP_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, restval="", restkey="_extra")
            for row in reader:
                try:
                    if "_extra" in row:
                        continue  # skip malformed rows with extra fields
                    k = (
                        row.get("canonical_key", ""),
                        row.get("market", ""),
                        row.get("side", ""),
                        row.get("bookmaker", ""),
                    )
                    ts = row.get("timestamp", "")
                    if k not in latest or ts > latest[k]["timestamp"]:
                        latest[k] = row
                except Exception:
                    continue  # skip individual bad rows
    except Exception:
        return {}

    # Group by (canonical_key, market, side) → list of bookmaker entries
    grouped = defaultdict(list)
    for (canon, market, side, bm), row in latest.items():
        grouped[(canon, market, side)].append(row)

    return dict(grouped)


def _load_open_registry() -> dict:
    """Load open registry for computing moves from open."""
    if not os.path.exists(L1_OPEN_REGISTRY_CSV):
        return {}

    reg = {}
    try:
        with open(L1_OPEN_REGISTRY_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = (
                    row.get("canonical_key", ""),
                    row.get("bookmaker", ""),
                    row.get("market", ""),
                    row.get("side", ""),
                )
                reg[k] = row
    except Exception:
        pass
    return reg


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


def _compute_move_speed(changed_at: str, commence_time: str, magnitude: float) -> tuple:
    """
    Compute move speed from changedAt timestamp.

    Returns (speed_pts_per_hour, speed_label).
    """
    if not changed_at or not commence_time:
        return (0.0, "UNKNOWN")

    try:
        changed = datetime.fromisoformat(changed_at.replace("Z", "+00:00"))
        commence = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)

        # Hours from change to game time
        hours_to_game = max((commence - changed).total_seconds() / 3600, 0.01)
        # Hours from change to now
        hours_since_change = max((now - changed).total_seconds() / 3600, 0.01)

        # Speed = magnitude / hours since change (rate of recent movement)
        speed = abs(magnitude) / hours_since_change if hours_since_change > 0 else 0.0

        if speed >= FAST_SNAP_SPEED:
            label = "FAST_SNAP"
        elif speed <= SLOW_GRIND_SPEED:
            label = "SLOW_GRIND"
        else:
            label = "NORMAL"

        return (round(speed, 3), label)

    except Exception:
        return (0.0, "UNKNOWN")


def _crossed_key_number(open_line: float, current_line: float) -> bool:
    """Check if line movement crossed a key number (3, 7, 10, 14, 17)."""
    if open_line == current_line:
        return False

    low = min(abs(open_line), abs(current_line))
    high = max(abs(open_line), abs(current_line))

    for kn in KEY_NUMBERS:
        if low < kn <= high:
            return True
    return False


def compute_l1_features(sport: str = None) -> dict:
    """
    Compute Layer 1 features for all games (or filtered by sport).

    Returns:
        dict keyed by (canonical_key, market, side) → feature dict:
        {
            "l1_move_dir": int,           # +1/-1/0
            "l1_move_magnitude": float,   # normalized 0-1
            "l1_move_magnitude_raw": float,  # raw points
            "l1_sharp_agreement": int,    # number of books agreeing on direction
            "l1_agreement_mult": float,   # agreement multiplier (1.0-1.5)
            "l1_limit_weighted_dir": float,  # limit-weighted direction (-1 to +1)
            "l1_leader_book": str,        # which book moved first
            "l1_move_speed": float,       # pts/hour
            "l1_speed_label": str,        # FAST_SNAP / NORMAL / SLOW_GRIND
            "l1_stability": float,        # 0-1 (how consistent across books)
            "l1_key_number_cross": bool,  # crossed a key number
            "l1_sharp_strength": float,   # composite 0-1
            "l1_limit_confidence": float, # max limit across books
            "l1_n_books": int,           # number of sharp books with data
            "l1_books": list,            # list of bookmaker names
            "l1_available": True,
        }
    """
    latest = _load_latest_l1()
    open_reg = _load_open_registry()

    features = {}

    for (canon, market, side), book_entries in latest.items():
        # Filter by sport if specified
        if sport:
            entry_sport = book_entries[0].get("sport", "")
            if entry_sport != sport.lower():
                continue

        entry_sport = book_entries[0].get("sport", "")
        commence = book_entries[0].get("commence_time", "")

        # Compute per-book moves
        book_moves = []
        for entry in book_entries:
            bm = entry.get("bookmaker", "")
            current_line = _parse_float(entry.get("line"))
            current_odds = _parse_int(entry.get("odds_american"))
            changed_at = entry.get("changed_at", "")
            limit = _parse_float(entry.get("limit"))

            # Get open line for this book
            reg_key = (canon, bm, market, side)
            open_entry = open_reg.get(reg_key, {})
            open_line = _parse_float(open_entry.get("open_line"))
            open_odds = _parse_int(open_entry.get("open_odds"))

            # Compute move
            if market == "MONEYLINE":
                # For ML, compute direction from odds change
                move = current_odds - open_odds if open_odds != 0 else 0
            else:
                move = current_line - open_line

            direction = 1 if move > 0 else (-1 if move < 0 else 0)

            book_moves.append({
                "bookmaker": bm,
                "direction": direction,
                "move": move,
                "magnitude": abs(move),
                "current_line": current_line,
                "open_line": open_line,
                "current_odds": current_odds,
                "limit": limit,
                "changed_at": changed_at,
            })

        if not book_moves:
            continue

        # Aggregate features across books

        # 1. Majority direction
        dir_counts = defaultdict(int)
        for bm in book_moves:
            dir_counts[bm["direction"]] += 1
        majority_dir = max(dir_counts, key=dir_counts.get) if dir_counts else 0

        # 2. Agreement: how many books agree with majority direction
        agreement = dir_counts.get(majority_dir, 0)
        agreement_mult = AGREEMENT_MULTIPLIERS.get(agreement, 1.0)

        # 3. Average magnitude
        magnitudes = [bm["magnitude"] for bm in book_moves if bm["magnitude"] > 0]
        avg_magnitude = statistics.mean(magnitudes) if magnitudes else 0.0

        # Normalize magnitude
        norm_key = (entry_sport, market)
        norm_val = MAGNITUDE_NORMS.get(norm_key, DEFAULT_MAGNITUDE_NORM)
        normalized_mag = min(avg_magnitude / norm_val, 1.0) if norm_val > 0 else 0.0

        # 4. Limit-weighted direction
        total_limit = sum(bm["limit"] for bm in book_moves if bm["limit"] > 0)
        if total_limit > 0:
            limit_weighted = sum(
                bm["direction"] * bm["limit"]
                for bm in book_moves if bm["limit"] > 0
            ) / total_limit
        else:
            limit_weighted = float(majority_dir)

        # 5. Leader detection (earliest changedAt)
        books_with_time = [
            bm for bm in book_moves
            if bm["changed_at"] and bm["direction"] == majority_dir
        ]
        leader_book = ""
        if books_with_time:
            books_with_time.sort(key=lambda x: x["changed_at"])
            leader_book = books_with_time[0]["bookmaker"]

        # 6. Move speed (from leader's changedAt)
        move_speed = 0.0
        speed_label = "UNKNOWN"
        if books_with_time:
            leader = books_with_time[0]
            move_speed, speed_label = _compute_move_speed(
                leader["changed_at"], commence, leader["magnitude"]
            )

        # 7. Stability (how consistent are books? low variance = stable)
        if len(book_moves) >= 2:
            lines = [bm["current_line"] for bm in book_moves]
            line_var = statistics.variance(lines) if len(lines) >= 2 else 0.0
            # Normalize: 0 variance = 1.0 stability, high variance = 0.0
            stability = max(0.0, 1.0 - min(line_var / (norm_val ** 2), 1.0))
        else:
            stability = 0.5  # single book, uncertain

        # 8. Key number crossing
        key_cross = False
        for bm in book_moves:
            if market != "MONEYLINE" and _crossed_key_number(bm["open_line"], bm["current_line"]):
                key_cross = True
                break

        # 9. Max limit (confidence proxy)
        max_limit = max((bm["limit"] for bm in book_moves), default=0.0)

        # 10. Composite sharp strength (0-1)
        sharp_strength = _compute_sharp_strength(
            normalized_mag, agreement_mult, stability,
            speed_label, key_cross, max_limit,
        )

        features[(canon, market, side)] = {
            "l1_move_dir": majority_dir,
            "l1_move_magnitude": round(normalized_mag, 3),
            "l1_move_magnitude_raw": round(avg_magnitude, 3),
            "l1_sharp_agreement": agreement,
            "l1_agreement_mult": agreement_mult,
            "l1_limit_weighted_dir": round(limit_weighted, 3),
            "l1_leader_book": leader_book,
            "l1_move_speed": move_speed,
            "l1_speed_label": speed_label,
            "l1_stability": round(stability, 3),
            "l1_key_number_cross": key_cross,
            "l1_sharp_strength": round(sharp_strength, 3),
            "l1_limit_confidence": max_limit,
            "l1_n_books": len(book_moves),
            "l1_books": sorted(set(bm["bookmaker"] for bm in book_moves)),
            "l1_available": True,
        }

    return features


def _compute_sharp_strength(magnitude: float, agreement_mult: float,
                            stability: float, speed_label: str,
                            key_cross: bool, max_limit: float) -> float:
    """
    Compute composite sharp strength score (0 to 1).

    Weights:
      - magnitude:  0.30 (how much did sharps move?)
      - agreement:  0.25 (how many books agree?)
      - stability:  0.20 (are they holding position?)
      - speed:      0.15 (fast snap = stronger signal)
      - key cross:  0.05 (crossed key number = extra conviction)
      - limit:      0.05 (higher limits = more confident)
    """
    # Magnitude component (already 0-1)
    mag_component = magnitude * 0.30

    # Agreement component (mult 1.0-1.5, normalize to 0-1)
    agree_component = min((agreement_mult - 1.0) / 0.5, 1.0) * 0.25

    # Stability component (already 0-1)
    stab_component = stability * 0.20

    # Speed component
    speed_scores = {"FAST_SNAP": 1.0, "NORMAL": 0.5, "SLOW_GRIND": 0.2, "UNKNOWN": 0.3}
    speed_component = speed_scores.get(speed_label, 0.3) * 0.15

    # Key number cross
    key_component = (1.0 if key_cross else 0.0) * 0.05

    # Limit confidence (normalize assuming $50K is max meaningful limit)
    limit_norm = min(max_limit / 50000.0, 1.0) if max_limit > 0 else 0.3
    limit_component = limit_norm * 0.05

    total = mag_component + agree_component + stab_component + speed_component + key_component + limit_component
    return min(max(total, 0.0), 1.0)
