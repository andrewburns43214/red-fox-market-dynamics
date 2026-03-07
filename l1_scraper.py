"""
Layer 1 Scraper: Sharp book data collection.

Source: The-Odds-API (tiered sharp cluster — Pinnacle/Matchbook + Betfair/Bet365)

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
    L1_SUPPORTING_BOOKS,
)
from canonical_match import build_canonical_key
from team_aliases import normalize_team_name


# L1 sharp CSV columns
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


def _ensure_l1_header():
    """One-time check: if L1 CSV header doesn't match current schema, fix it."""
    if not os.path.exists(L1_SHARP_CSV):
        return
    with open(L1_SHARP_CSV, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
    existing_cols = [c.strip().strip('"').replace('\ufeff', '') for c in header_line.split(',')]
    if existing_cols == list(L1_SHARP_COLUMNS):
        return
    print(f"  [L1] Header mismatch ({len(existing_cols)} cols -> {len(L1_SHARP_COLUMNS)}) — rewriting header")
    with open(L1_SHARP_CSV, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines[0] = ",".join(f'"{c}"' for c in L1_SHARP_COLUMNS) + "\n"
    with open(L1_SHARP_CSV, "w", encoding="utf-8", newline="") as f:
        f.writelines(lines)


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



def scrape_l1(sport: str) -> dict:
    """
    Scrape Layer 1 data via The-Odds-API.

    Captures tiered sharp cluster:
      - L1_SHARP_BOOKS (pinnacle, matchbook) — tier "sharp"
      - L1_SUPPORTING_BOOKS (betfair_ex_eu, bet365) — tier "supporting"

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
    sharp_set = set(b.lower() for b in L1_SHARP_BOOKS)
    support_set = set(b.lower() for b in L1_SUPPORTING_BOOKS)
    all_l1_books = sharp_set | support_set
    l1_rows = []
    games_seen = set()

    for event in result["events"]:
        parsed = parse_event_odds(event)
        for row in parsed:
            bm_lower = row["bookmaker"].lower()
            if bm_lower not in all_l1_books:
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
                "bookmaker": bm_lower,
                "home_team_norm": home_norm,
                "away_team_norm": away_norm,
                "commence_time": row["commence_time"],
                "market": row["market"],
                "side": side,
                "line": row["line"] if row["line"] is not None else "",
                "odds_american": row["odds_american"],
                "changed_at": row.get("last_update", ""),
                "limit": "",
                "source": "oddsapi",
            })

    if not l1_rows:
        return {"rows_written": 0, "games_found": len(games_seen),
                "error": "No sharp book data found",
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
    """Scrape L1 sharp data. Single source: The-Odds-API."""
    _ensure_l1_header()
    return scrape_l1(sport)
