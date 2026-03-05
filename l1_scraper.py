"""
Layer 1 Scraper: Sharp book data collection.

Primary source: OddsPapi (6 sharp books with timestamps + limits)
Fallback: The-Odds-API (Pinnacle only, no timestamps/limits)

Writes to:
  - data/l1_sharp.csv       (append-only, all snapshots)
  - data/l1_open_registry.csv (first-seen lines per game/market/side)
  - data/match_failures.csv  (games that couldn't be matched)
"""
import csv
import os
from datetime import datetime, timezone

from engine_config import (
    L1_SHARP_CSV,
    L1_OPEN_REGISTRY_CSV,
    L1_CACHE_JSON,
    MATCH_FAILURES_CSV,
    API_SPORT_MAP,
    L1_SHARP_BOOKS,
)
from canonical_match import build_canonical_key
from team_aliases import normalize_team_name


# Updated columns with OddsPapi fields
L1_SHARP_COLUMNS = [
    "timestamp", "sport", "canonical_key", "bookmaker",
    "home_team_norm", "away_team_norm", "commence_time",
    "market", "side", "line", "odds_american",
    "changed_at", "limit", "source",
]

L1_OPEN_REG_COLUMNS = [
    "sport", "canonical_key", "bookmaker", "market", "side",
    "open_line", "open_odds", "first_seen",
    "changed_at", "limit",
]


def _load_l1_open_registry() -> dict:
    """Load existing open registry. Key: (sport, canonical_key, bookmaker, market, side)."""
    reg = {}
    if not os.path.exists(L1_OPEN_REGISTRY_CSV):
        return reg
    try:
        with open(L1_OPEN_REGISTRY_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = (
                    row.get("sport", ""),
                    row.get("canonical_key", ""),
                    row.get("bookmaker", ""),
                    row.get("market", ""),
                    row.get("side", ""),
                )
                reg[k] = {
                    "open_line": row.get("open_line", ""),
                    "open_odds": row.get("open_odds", ""),
                    "first_seen": row.get("first_seen", ""),
                    "changed_at": row.get("changed_at", ""),
                    "limit": row.get("limit", ""),
                }
    except Exception as e:
        print(f"[WARN] l1 open registry load: {repr(e)}")
    return reg


def _save_l1_open_registry(reg: dict) -> None:
    """Write full open registry."""
    os.makedirs(os.path.dirname(L1_OPEN_REGISTRY_CSV) or ".", exist_ok=True)
    with open(L1_OPEN_REGISTRY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(L1_OPEN_REG_COLUMNS)
        for (sport, canon, bm, market, side), vals in sorted(reg.items()):
            w.writerow([
                sport, canon, bm, market, side,
                vals["open_line"], vals["open_odds"], vals["first_seen"],
                vals.get("changed_at", ""), vals.get("limit", ""),
            ])


def _log_match_failure(sport: str, source: str, home: str, away: str, reason: str) -> None:
    """Log games that couldn't be canonically matched."""
    try:
        write_header = not os.path.exists(MATCH_FAILURES_CSV)
        os.makedirs(os.path.dirname(MATCH_FAILURES_CSV) or ".", exist_ok=True)
        with open(MATCH_FAILURES_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp", "sport", "source", "home", "away", "reason"])
            w.writerow([
                datetime.now(timezone.utc).isoformat(),
                sport, source, home, away, reason,
            ])
    except Exception as e:
        print(f"[WARN] l1 match failure log: {repr(e)}")


def scrape_l1_oddspapi(sport: str) -> dict:
    """
    Scrape Layer 1 data from OddsPapi (primary source).

    Returns 6 sharp books with changedAt timestamps and betting limits.

    Args:
        sport: Our sport key (nba, nfl, etc.)

    Returns:
        dict with:
            "rows_written": int
            "games_found": int
            "books_found": list
            "error": str or None
            "from_cache": bool
            "source": "oddspapi"
    """
    try:
        from oddspapi import fetch_fixtures, fetch_odds_with_cache as oddspapi_fetch
    except ImportError as e:
        return {"rows_written": 0, "games_found": 0, "books_found": [],
                "error": f"OddsPapi module not available: {e}",
                "from_cache": False, "source": "oddspapi"}

    sport_lower = sport.lower()

    # Step 1: Fetch fixtures to get team names
    fix_result = fetch_fixtures(sport_lower)
    if fix_result["error"]:
        return {"rows_written": 0, "games_found": 0, "books_found": [],
                "error": f"Fixtures: {fix_result['error']}",
                "from_cache": False, "source": "oddspapi"}

    fixture_map = fix_result["fixture_map"]
    if not fixture_map:
        return {"rows_written": 0, "games_found": 0, "books_found": [],
                "error": "No fixtures found for today",
                "from_cache": False, "source": "oddspapi"}

    # Step 2: Fetch odds from sharp books
    odds_result = oddspapi_fetch(sport_lower)
    if odds_result["error"]:
        return {"rows_written": 0, "games_found": 0, "books_found": [],
                "error": f"Odds: {odds_result['error']}",
                "from_cache": odds_result.get("from_cache", False),
                "source": "oddspapi"}

    odds_data = odds_result["odds"]
    if not odds_data:
        return {"rows_written": 0, "games_found": 0, "books_found": [],
                "error": "No sharp book odds returned",
                "from_cache": odds_result.get("from_cache", False),
                "source": "oddspapi"}

    now_ts = datetime.now(timezone.utc).isoformat()
    l1_rows = []
    games_seen = set()
    books_seen = set()

    for od in odds_data:
        fid = str(od.get("fixture_id", ""))

        # Look up team names from fixture map
        fix_info = fixture_map.get(fid)
        if not fix_info:
            continue

        home_raw = fix_info["home"]
        away_raw = fix_info["away"]
        commence = fix_info["commence_time"]

        if not home_raw or not away_raw:
            _log_match_failure(sport_lower, "oddspapi", home_raw, away_raw, "Missing team name")
            continue

        home_norm = normalize_team_name(home_raw, sport=sport_lower)
        away_norm = normalize_team_name(away_raw, sport=sport_lower)
        canon_key = build_canonical_key(away_raw, home_raw, sport_lower, commence)

        if not canon_key:
            _log_match_failure(sport_lower, "oddspapi", home_raw, away_raw, "No canonical key")
            continue

        games_seen.add(canon_key)
        books_seen.add(od["bookmaker"])

        # Normalize side
        side = od["side"]
        if od["market"] == "TOTAL":
            side = side.lower()
        else:
            side = normalize_team_name(side, sport=sport_lower)

        line_val = od["line"] if od["line"] is not None else ""
        limit_val = od.get("limit")
        limit_str = str(limit_val) if limit_val is not None else ""

        l1_rows.append({
            "timestamp": now_ts,
            "sport": sport_lower,
            "canonical_key": canon_key,
            "bookmaker": od["bookmaker"],
            "home_team_norm": home_norm,
            "away_team_norm": away_norm,
            "commence_time": commence,
            "market": od["market"],
            "side": side,
            "line": line_val,
            "odds_american": od["odds_american"],
            "changed_at": od.get("changed_at", ""),
            "limit": limit_str,
            "source": "oddspapi",
        })

    if not l1_rows:
        return {"rows_written": 0, "games_found": len(games_seen),
                "books_found": sorted(books_seen),
                "error": "No sharp book data matched to fixtures",
                "from_cache": odds_result.get("from_cache", False),
                "source": "oddspapi"}

    # Update open registry
    open_reg = _load_l1_open_registry()
    for row in l1_rows:
        reg_key = (
            row["sport"], row["canonical_key"], row["bookmaker"],
            row["market"], row["side"],
        )
        if reg_key not in open_reg:
            open_reg[reg_key] = {
                "open_line": str(row["line"]),
                "open_odds": str(row["odds_american"]),
                "first_seen": now_ts,
                "changed_at": row.get("changed_at", ""),
                "limit": row.get("limit", ""),
            }
    _save_l1_open_registry(open_reg)

    # Append to L1 sharp CSV
    write_header = not os.path.exists(L1_SHARP_CSV)
    os.makedirs(os.path.dirname(L1_SHARP_CSV) or ".", exist_ok=True)
    with open(L1_SHARP_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=L1_SHARP_COLUMNS,
                           extrasaction="ignore", quoting=csv.QUOTE_ALL)
        if write_header:
            w.writeheader()
        for row in l1_rows:
            w.writerow(row)

    return {
        "rows_written": len(l1_rows),
        "games_found": len(games_seen),
        "books_found": sorted(books_seen),
        "error": None,
        "from_cache": odds_result.get("from_cache", False),
        "source": "oddspapi",
    }


def scrape_l1(sport: str) -> dict:
    """
    Scrape Layer 1 data via The-Odds-API (fallback source).

    Only returns Pinnacle data. No timestamps or limits.
    Used when OddsPapi is unavailable or lacks coverage.

    Args:
        sport: Our sport key (nba, nfl, etc.)

    Returns:
        dict with:
            "rows_written": int
            "games_found": int
            "error": str or None
            "from_cache": bool
            "remaining_requests": str or None
            "source": "oddsapi"
    """
    from odds_api import fetch_odds_with_cache, parse_event_odds

    if sport.lower() not in API_SPORT_MAP:
        return {"rows_written": 0, "games_found": 0,
                "error": f"Unknown sport: {sport}", "from_cache": False,
                "remaining_requests": None, "source": "oddsapi"}

    result = fetch_odds_with_cache(
        sport=sport,
        cache_path=L1_CACHE_JSON,
        markets=["spreads", "totals", "h2h"],
    )

    if result["error"]:
        return {"rows_written": 0, "games_found": 0,
                "error": result["error"], "from_cache": result["from_cache"],
                "remaining_requests": result.get("remaining_requests"),
                "source": "oddsapi"}

    now_ts = datetime.now(timezone.utc).isoformat()
    sport_lower = sport.lower()
    sharp_books_set = set(b.lower() for b in L1_SHARP_BOOKS)
    l1_rows = []
    games_seen = set()

    for event in result["events"]:
        parsed = parse_event_odds(event)
        for row in parsed:
            if row["bookmaker"].lower() not in sharp_books_set:
                continue

            home_norm = normalize_team_name(row["home_team"], sport=sport_lower)
            away_norm = normalize_team_name(row["away_team"], sport=sport_lower)
            canon_key = build_canonical_key(
                row["away_team"], row["home_team"],
                sport_lower, row["commence_time"],
            )

            if not canon_key:
                continue

            games_seen.add(canon_key)

            side = row["side"]
            if row["market"] == "TOTAL":
                side = side.lower()
            else:
                side = normalize_team_name(side, sport=sport_lower)

            l1_rows.append({
                "timestamp": now_ts,
                "sport": sport_lower,
                "canonical_key": canon_key,
                "bookmaker": row["bookmaker"].lower(),
                "home_team_norm": home_norm,
                "away_team_norm": away_norm,
                "commence_time": row["commence_time"],
                "market": row["market"],
                "side": side,
                "line": row["line"] if row["line"] is not None else "",
                "odds_american": row["odds_american"],
                "changed_at": "",    # Not available from The-Odds-API
                "limit": "",         # Not available from The-Odds-API
                "source": "oddsapi",
            })

    if not l1_rows:
        return {"rows_written": 0, "games_found": len(games_seen),
                "error": "No sharp book data found (Pinnacle may not cover this sport)",
                "from_cache": result["from_cache"],
                "remaining_requests": result.get("remaining_requests"),
                "source": "oddsapi"}

    # Update open registry
    open_reg = _load_l1_open_registry()
    for row in l1_rows:
        reg_key = (
            row["sport"], row["canonical_key"], row["bookmaker"],
            row["market"], row["side"],
        )
        if reg_key not in open_reg:
            open_reg[reg_key] = {
                "open_line": str(row["line"]),
                "open_odds": str(row["odds_american"]),
                "first_seen": now_ts,
                "changed_at": "",
                "limit": "",
            }
    _save_l1_open_registry(open_reg)

    # Append to L1 sharp CSV
    write_header = not os.path.exists(L1_SHARP_CSV)
    os.makedirs(os.path.dirname(L1_SHARP_CSV) or ".", exist_ok=True)
    with open(L1_SHARP_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=L1_SHARP_COLUMNS,
                           extrasaction="ignore", quoting=csv.QUOTE_ALL)
        if write_header:
            w.writeheader()
        for row in l1_rows:
            w.writerow(row)

    return {
        "rows_written": len(l1_rows),
        "games_found": len(games_seen),
        "error": None,
        "from_cache": result["from_cache"],
        "remaining_requests": result.get("remaining_requests"),
        "source": "oddsapi",
    }


def scrape_l1_auto(sport: str) -> dict:
    """
    Auto-select best L1 source: OddsPapi first, The-Odds-API fallback.

    Returns:
        Combined result dict with source indicator.
    """
    # Try OddsPapi first (6 sharp books, timestamps, limits)
    result = scrape_l1_oddspapi(sport)

    if not result["error"]:
        return result

    # OddsPapi failed — fall back to The-Odds-API (Pinnacle only)
    print(f"  [L1] OddsPapi failed ({result['error']}), falling back to The-Odds-API...")
    fallback = scrape_l1(sport)
    fallback["oddspapi_error"] = result["error"]
    return fallback
