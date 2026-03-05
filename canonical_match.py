"""
Canonical matching system for cross-source game identification.

Matches games across DraftKings, The-Odds-API (Pinnacle/consensus),
and other future sources using normalized team names + game date.

Canonical key format:
    "{away_norm} @ {home_norm}|{sport}|{game_date_utc}"
    e.g. "atlanta hawks @ washington wizards|nba|2026-03-02"

UFC uses alphabetical sort (no home/away concept):
    "{fighter_a} vs {fighter_b}|ufc|{date}"
"""
import csv
import os
from datetime import datetime, timedelta

from team_aliases import normalize_team_name, _split_game, _norm_team


def build_canonical_key(away_name: str, home_name: str, sport: str, commence_time_iso: str) -> str:
    """
    Build a canonical key from raw fields.

    Args:
        away_name: Away team name (any source format)
        home_name: Home team name (any source format)
        sport: Lowercase sport key (nba, nfl, etc.)
        commence_time_iso: ISO 8601 UTC game start time

    Returns:
        Canonical key string, or "" if inputs are invalid.
    """
    sport = (sport or "").strip().lower()
    away_norm = normalize_team_name(away_name, sport=sport)
    home_norm = normalize_team_name(home_name, sport=sport)

    if not away_norm or not home_norm or not sport:
        return ""

    # Extract date portion from ISO time
    date_str = ""
    try:
        if commence_time_iso and str(commence_time_iso).strip():
            date_str = str(commence_time_iso).strip()[:10]  # "2026-03-02"
    except Exception:
        pass

    if not date_str:
        return ""

    # UFC: alphabetical sort (no home/away concept in MMA)
    if sport == "ufc":
        fighters = sorted([away_norm, home_norm])
        return f"{fighters[0]} vs {fighters[1]}|{sport}|{date_str}"

    return f"{away_norm} @ {home_norm}|{sport}|{date_str}"


def build_canonical_key_from_dk(game: str, sport: str, dk_start_iso: str) -> str:
    """
    Build canonical key from DraftKings game string.

    Args:
        game: DK game string like "SA Spurs @ TOR Raptors"
        sport: Lowercase sport key
        dk_start_iso: DK start time in ISO format

    Returns:
        Canonical key string, or "" if parsing fails.
    """
    away, home = _split_game(game)
    if not away or not home:
        return ""
    return build_canonical_key(away, home, sport, dk_start_iso)


def fuzzy_match_key(key1: str, key2: str, date_tolerance_days: int = 1) -> float:
    """
    Compute match score between two canonical keys.

    Handles both standard "away @ home" and UFC "a vs b" formats.

    Returns:
        Score 0.0-1.0. >= 0.8 is a confident match.
    """
    if not key1 or not key2:
        return 0.0

    try:
        parts1 = key1.split("|")
        parts2 = key2.split("|")
        if len(parts1) != 3 or len(parts2) != 3:
            return 0.0

        teams1, sport1, date1 = parts1
        teams2, sport2, date2 = parts2

        # Sport must match exactly
        if sport1 != sport2:
            return 0.0

        score = 0.0

        # Date check (exact = 0.3, +-1 day = 0.2)
        try:
            d1 = datetime.strptime(date1, "%Y-%m-%d")
            d2 = datetime.strptime(date2, "%Y-%m-%d")
            diff = abs((d1 - d2).days)
            if diff == 0:
                score += 0.3
            elif diff <= date_tolerance_days:
                score += 0.2
            else:
                return 0.0  # too far apart
        except Exception:
            return 0.0

        # Detect separator (" @ " or " vs ")
        for sep in (" @ ", " vs "):
            if sep in teams1:
                t1_parts = teams1.split(sep, 1)
                break
        else:
            t1_parts = [teams1, ""]

        for sep in (" @ ", " vs "):
            if sep in teams2:
                t2_parts = teams2.split(sep, 1)
                break
        else:
            t2_parts = [teams2, ""]

        name1_a, name1_b = t1_parts[0], t1_parts[1] if len(t1_parts) > 1 else ""
        name2_a, name2_b = t2_parts[0], t2_parts[1] if len(t2_parts) > 1 else ""

        # Exact match on both teams
        if name1_a == name2_a and name1_b == name2_b:
            score += 0.7
        elif name1_a == name2_b and name1_b == name2_a:
            # Reversed order (can happen with UFC or data quirks)
            score += 0.65
        else:
            # Token overlap matching
            tokens1_a = set(name1_a.split())
            tokens2_a = set(name2_a.split())
            tokens1_b = set(name1_b.split())
            tokens2_b = set(name2_b.split())

            # Try both orderings
            overlap_normal = (
                len(tokens1_a & tokens2_a) / max(len(tokens1_a | tokens2_a), 1)
                + len(tokens1_b & tokens2_b) / max(len(tokens1_b | tokens2_b), 1)
            )
            overlap_reversed = (
                len(tokens1_a & tokens2_b) / max(len(tokens1_a | tokens2_b), 1)
                + len(tokens1_b & tokens2_a) / max(len(tokens1_b | tokens2_a), 1)
            )

            score += max(overlap_normal, overlap_reversed) * 0.35

        return min(1.0, score)
    except Exception:
        return 0.0


def normalize_side_for_match(market: str, side: str, sport: str = "") -> str:
    """
    Normalize a side label for cross-source matching.

    For SPREAD/MONEYLINE: normalize team name (with sport for mascot stripping)
    For TOTAL: "over" or "under"
    """
    import re

    market = (market or "").strip().upper()
    side = (side or "").strip()

    if market == "TOTAL":
        s = side.lower()
        if "over" in s:
            return "over"
        if "under" in s:
            return "under"
        return s

    # For SPREAD/ML: normalize team name, strip spread numbers
    cleaned = re.sub(r"[+-]?\d+\.?\d*\s*$", "", side).strip()
    return normalize_team_name(cleaned, sport=sport)


# --- Match failure logging ---

MATCH_FAILURES_PATH = os.path.join("data", "match_failures.csv")

def log_match_failure(sport: str, dk_game: str, dk_game_id: str, dk_start: str, reason: str = ""):
    """Log a game that couldn't be matched across sources."""
    try:
        write_header = not os.path.exists(MATCH_FAILURES_PATH)
        with open(MATCH_FAILURES_PATH, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp", "sport", "dk_game", "dk_game_id", "dk_start", "reason"])
            w.writerow([
                datetime.utcnow().isoformat(),
                sport, dk_game, dk_game_id, dk_start, reason,
            ])
    except Exception:
        pass


# --- Manual override support ---

OVERRIDES_PATH = os.path.join("data", "canonical_overrides.csv")

def load_overrides() -> dict:
    """
    Load manual canonical key overrides.
    Returns dict mapping dk_game_id -> canonical_key.
    """
    overrides = {}
    try:
        if os.path.exists(OVERRIDES_PATH):
            with open(OVERRIDES_PATH, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dk_id = str(row.get("dk_game_id", "")).strip()
                    canon = str(row.get("canonical_key", "")).strip()
                    if dk_id and canon:
                        overrides[dk_id] = canon
    except Exception:
        pass
    return overrides
