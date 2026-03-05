"""
Layer 2 Scraper: Consensus data collection.

Fetches odds from all US-region books via The-Odds-API.
Writes to:
  - data/l2_consensus.csv      (append-only, raw per-book data)
  - data/l2_consensus_agg.csv  (overwritten each run, aggregated metrics)

Aggregated metrics per game/market/side:
  - n_books: how many books are offering this line
  - consensus_line: median line across books
  - consensus_odds: median American odds
  - line_std: standard deviation of lines (dispersion)
  - pinn_line / pinn_odds: Pinnacle values (if present)
  - pinn_vs_consensus: Pinnacle line minus consensus line
  - consensus_direction: majority side direction
"""
import csv
import os
import statistics
from datetime import datetime, timezone

from engine_config import (
    L2_CONSENSUS_CSV,
    L2_CONSENSUS_AGG_CSV,
    L2_CACHE_JSON,
    API_SPORT_MAP,
    L1_SHARP_BOOKS,
)
from odds_api import fetch_odds_with_cache, parse_event_odds
from canonical_match import build_canonical_key
from team_aliases import normalize_team_name


L2_RAW_COLUMNS = [
    "timestamp", "sport", "canonical_key", "commence_time",
    "market", "side", "bookmaker", "line", "odds_american",
]

L2_AGG_COLUMNS = [
    "timestamp", "sport", "canonical_key", "market", "side",
    "n_books", "consensus_line", "consensus_odds",
    "line_std", "line_std_prev",
    "pinn_vs_consensus", "consensus_direction",
    "pinn_line", "pinn_odds",
]


def _load_prev_agg() -> dict:
    """Load previous aggregation for line_std_prev (dispersion trend)."""
    prev = {}
    if not os.path.exists(L2_CONSENSUS_AGG_CSV):
        return prev
    try:
        with open(L2_CONSENSUS_AGG_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                k = (
                    row.get("sport", ""),
                    row.get("canonical_key", ""),
                    row.get("market", ""),
                    row.get("side", ""),
                )
                try:
                    prev[k] = float(row.get("line_std", "0") or "0")
                except (ValueError, TypeError):
                    prev[k] = 0.0
    except Exception:
        pass
    return prev


def scrape_l2(sport: str) -> dict:
    """
    Scrape Layer 2 (consensus) data for a sport.

    Uses the SAME API call as L1 — all US books are returned.
    Sharp books are included in the consensus (they're real market participants).

    Args:
        sport: Our sport key (nba, nfl, etc.)

    Returns:
        dict with:
            "rows_written": int (raw rows)
            "agg_rows": int (aggregated rows)
            "games_found": int
            "books_seen": list of bookmaker keys
            "error": str or None
            "from_cache": bool
            "remaining_requests": str or None
    """
    if sport.lower() not in API_SPORT_MAP:
        return {"rows_written": 0, "agg_rows": 0, "games_found": 0,
                "books_seen": [], "error": f"Unknown sport: {sport}",
                "from_cache": False, "remaining_requests": None}

    # Fetch from API (all US books)
    result = fetch_odds_with_cache(
        sport=sport,
        cache_path=L2_CACHE_JSON,
        markets=["spreads", "totals", "h2h"],
    )

    if result["error"]:
        return {"rows_written": 0, "agg_rows": 0, "games_found": 0,
                "books_seen": [], "error": result["error"],
                "from_cache": result["from_cache"],
                "remaining_requests": result.get("remaining_requests")}

    now_ts = datetime.now(timezone.utc).isoformat()
    sport_lower = sport.lower()
    sharp_books_set = set(b.lower() for b in L1_SHARP_BOOKS)

    # Parse all events — ALL books go into L2
    l2_rows = []
    games_seen = set()
    books_seen = set()

    for event in result["events"]:
        parsed = parse_event_odds(event)
        for row in parsed:
            canon_key = build_canonical_key(
                row["away_team"], row["home_team"],
                sport_lower, row["commence_time"],
            )
            if not canon_key:
                continue

            games_seen.add(canon_key)
            books_seen.add(row["bookmaker"].lower())

            # Normalize side
            side = row["side"]
            if row["market"] == "TOTAL":
                side = side.lower()
            else:
                side = normalize_team_name(side)

            l2_rows.append({
                "timestamp": now_ts,
                "sport": sport_lower,
                "canonical_key": canon_key,
                "commence_time": row["commence_time"],
                "market": row["market"],
                "side": side,
                "bookmaker": row["bookmaker"].lower(),
                "line": row["line"] if row["line"] is not None else "",
                "odds_american": row["odds_american"],
            })

    if not l2_rows:
        return {"rows_written": 0, "agg_rows": 0, "games_found": 0,
                "books_seen": list(books_seen),
                "error": "No consensus data found",
                "from_cache": result["from_cache"],
                "remaining_requests": result.get("remaining_requests")}

    # Write raw L2 CSV (append)
    write_header = not os.path.exists(L2_CONSENSUS_CSV)
    os.makedirs(os.path.dirname(L2_CONSENSUS_CSV) or ".", exist_ok=True)
    with open(L2_CONSENSUS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=L2_RAW_COLUMNS)
        if write_header:
            w.writeheader()
        for row in l2_rows:
            w.writerow(row)

    # Aggregate: group by (sport, canonical_key, market, side)
    prev_agg = _load_prev_agg()
    groups = {}
    for row in l2_rows:
        gk = (row["sport"], row["canonical_key"], row["market"], row["side"])
        if gk not in groups:
            groups[gk] = {"lines": [], "odds": [], "pinn_line": None, "pinn_odds": None}
        try:
            line_val = float(row["line"]) if row["line"] != "" else None
        except (ValueError, TypeError):
            line_val = None
        try:
            odds_val = int(row["odds_american"])
        except (ValueError, TypeError):
            odds_val = None

        if line_val is not None:
            groups[gk]["lines"].append(line_val)
        if odds_val is not None:
            groups[gk]["odds"].append(odds_val)

        # Track Pinnacle specifically
        if row["bookmaker"] in sharp_books_set:
            if line_val is not None:
                groups[gk]["pinn_line"] = line_val
            if odds_val is not None:
                groups[gk]["pinn_odds"] = odds_val

    # Build aggregated rows
    agg_rows = []
    for (sport_k, canon, market, side), data in sorted(groups.items()):
        n_books = len(data["lines"]) if data["lines"] else len(data["odds"])
        if n_books == 0:
            continue

        consensus_line = ""
        line_std = ""
        if data["lines"]:
            consensus_line = f"{statistics.median(data['lines']):.1f}"
            if len(data["lines"]) >= 2:
                line_std = f"{statistics.stdev(data['lines']):.3f}"
            else:
                line_std = "0.000"

        consensus_odds = ""
        if data["odds"]:
            consensus_odds = str(int(statistics.median(data["odds"])))

        # Pinnacle vs consensus
        pinn_vs = ""
        pinn_line_str = ""
        pinn_odds_str = ""
        if data["pinn_line"] is not None:
            pinn_line_str = f"{data['pinn_line']:.1f}"
            if consensus_line:
                pinn_vs = f"{data['pinn_line'] - float(consensus_line):.2f}"
        if data["pinn_odds"] is not None:
            pinn_odds_str = str(data["pinn_odds"])

        # Consensus direction (for spreads/totals: which way is the median leaning)
        consensus_dir = ""
        if data["lines"] and market in ("SPREAD", "TOTAL"):
            med = statistics.median(data["lines"])
            if med > 0:
                consensus_dir = "positive"
            elif med < 0:
                consensus_dir = "negative"
            else:
                consensus_dir = "neutral"

        # Previous line_std for dispersion trend
        prev_key = (sport_k, canon, market, side)
        line_std_prev = ""
        if prev_key in prev_agg:
            line_std_prev = f"{prev_agg[prev_key]:.3f}"

        agg_rows.append({
            "timestamp": now_ts,
            "sport": sport_k,
            "canonical_key": canon,
            "market": market,
            "side": side,
            "n_books": n_books,
            "consensus_line": consensus_line,
            "consensus_odds": consensus_odds,
            "line_std": line_std,
            "line_std_prev": line_std_prev,
            "pinn_vs_consensus": pinn_vs,
            "consensus_direction": consensus_dir,
            "pinn_line": pinn_line_str,
            "pinn_odds": pinn_odds_str,
        })

    # Write aggregated CSV — merge with existing (keep other sports' data)
    if agg_rows:
        # Read existing agg rows from OTHER sports
        existing_other = []
        if os.path.exists(L2_CONSENSUS_AGG_CSV):
            try:
                with open(L2_CONSENSUS_AGG_CSV, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("sport", "") != sport_lower:
                            existing_other.append(row)
            except Exception:
                pass

        # Write all: existing other sports + new current sport
        with open(L2_CONSENSUS_AGG_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=L2_AGG_COLUMNS)
            w.writeheader()
            for row in existing_other:
                w.writerow(row)
            for row in agg_rows:
                w.writerow(row)

    return {
        "rows_written": len(l2_rows),
        "agg_rows": len(agg_rows),
        "games_found": len(games_seen),
        "books_seen": sorted(books_seen),
        "error": None,
        "from_cache": result["from_cache"],
        "remaining_requests": result.get("remaining_requests"),
    }


def scrape_l1_and_l2(sport: str) -> dict:
    """
    Scrape both L1 (sharp) and L2 (consensus) data for a sport.

    L1: OddsPapi (6 sharp books, timestamps, limits) → fallback to The-Odds-API Pinnacle
    L2: The-Odds-API (31 US+EU books for consensus)

    These are now SEPARATE API calls to different services with independent budgets.

    Returns combined result dict.
    """
    from l1_scraper import scrape_l1_auto

    # L1: OddsPapi first, The-Odds-API fallback
    l1_result = scrape_l1_auto(sport)

    # L2: Always The-Odds-API (irreplaceable for 31-book consensus)
    l2_result = scrape_l2(sport)

    return {
        "l1": {
            "rows_written": l1_result.get("rows_written", 0),
            "games_found": l1_result.get("games_found", 0),
            "books_found": l1_result.get("books_found", []),
            "error": l1_result.get("error"),
            "source": l1_result.get("source", "unknown"),
        },
        "l2": {
            "rows_written": l2_result.get("rows_written", 0),
            "agg_rows": l2_result.get("agg_rows", 0),
            "games_found": l2_result.get("games_found", 0),
            "books_seen": l2_result.get("books_seen", []),
            "error": l2_result.get("error"),
        },
        "from_cache": l1_result.get("from_cache", False) or l2_result.get("from_cache", False),
        "remaining_requests": l2_result.get("remaining_requests"),
    }
