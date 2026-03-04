"""
Layer 1 Scraper: Sharp book data collection.

Fetches odds from sharp books (Pinnacle, etc.) via The-Odds-API.
Writes to:
  - data/l1_sharp.csv       (append-only, all snapshots)
  - data/l1_open_registry.csv (first-seen lines per game/market/side)

Schema supports multiple bookmakers from day one.
"""
import csv
import os
from datetime import datetime, timezone

from engine_config import (
    L1_SHARP_CSV,
    L1_OPEN_REGISTRY_CSV,
    L1_CACHE_JSON,
    API_SPORT_MAP,
    L1_SHARP_BOOKS,
)
from odds_api import fetch_odds_with_cache, parse_event_odds
from canonical_match import build_canonical_key
from team_aliases import normalize_team_name


L1_SHARP_COLUMNS = [
    "timestamp", "sport", "canonical_key", "bookmaker",
    "home_team_norm", "away_team_norm", "commence_time",
    "market", "side", "line", "odds_american",
]

L1_OPEN_REG_COLUMNS = [
    "sport", "canonical_key", "bookmaker", "market", "side",
    "open_line", "open_odds", "first_seen",
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
                }
    except Exception:
        pass
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
            ])


def scrape_l1(sport: str) -> dict:
    """
    Scrape Layer 1 (sharp book) data for a sport.

    Args:
        sport: Our sport key (nba, nfl, etc.)

    Returns:
        dict with:
            "rows_written": int
            "games_found": int
            "error": str or None
            "from_cache": bool
            "remaining_requests": str or None
    """
    if sport.lower() not in API_SPORT_MAP:
        return {"rows_written": 0, "games_found": 0,
                "error": f"Unknown sport: {sport}", "from_cache": False,
                "remaining_requests": None}

    # Fetch from API (sharp books only)
    result = fetch_odds_with_cache(
        sport=sport,
        cache_path=L1_CACHE_JSON,
        markets=["spreads", "totals", "h2h"],
    )

    if result["error"]:
        return {"rows_written": 0, "games_found": 0,
                "error": result["error"], "from_cache": result["from_cache"],
                "remaining_requests": result.get("remaining_requests")}

    now_ts = datetime.now(timezone.utc).isoformat()
    sport_lower = sport.lower()

    # Parse all events, filter to sharp books only
    sharp_books_set = set(b.lower() for b in L1_SHARP_BOOKS)
    l1_rows = []
    games_seen = set()

    for event in result["events"]:
        parsed = parse_event_odds(event)
        for row in parsed:
            if row["bookmaker"].lower() not in sharp_books_set:
                continue

            home_norm = normalize_team_name(row["home_team"])
            away_norm = normalize_team_name(row["away_team"])
            canon_key = build_canonical_key(
                row["away_team"], row["home_team"],
                sport_lower, row["commence_time"],
            )

            if not canon_key:
                continue

            games_seen.add(canon_key)

            # Normalize side for storage
            side = row["side"]
            if row["market"] == "TOTAL":
                side = side.lower()  # "over" / "under"
            else:
                side = normalize_team_name(side)

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
            })

    if not l1_rows:
        return {"rows_written": 0, "games_found": len(games_seen),
                "error": "No sharp book data found (Pinnacle may not cover this sport)",
                "from_cache": result["from_cache"],
                "remaining_requests": result.get("remaining_requests")}

    # Update open registry (first-seen lines)
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
            }
    _save_l1_open_registry(open_reg)

    # Append to L1 sharp CSV
    write_header = not os.path.exists(L1_SHARP_CSV)
    os.makedirs(os.path.dirname(L1_SHARP_CSV) or ".", exist_ok=True)
    with open(L1_SHARP_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=L1_SHARP_COLUMNS)
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
    }
