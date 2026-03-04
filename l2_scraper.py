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

    # Write aggregated CSV (overwrite — latest snapshot only)
    if agg_rows:
        with open(L2_CONSENSUS_AGG_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=L2_AGG_COLUMNS)
            w.writeheader()
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
    Single API call that populates BOTH L1 and L2 data.
    More efficient than calling l1_scraper + l2_scraper separately
    (saves an API request).

    Returns combined result dict.
    """
    if sport.lower() not in API_SPORT_MAP:
        return {"error": f"Unknown sport: {sport}"}

    # One API call gets everything
    result = fetch_odds_with_cache(
        sport=sport,
        cache_path=L2_CACHE_JSON,
        markets=["spreads", "totals", "h2h"],
    )

    if result["error"]:
        return {
            "l1": {"rows_written": 0, "games_found": 0, "error": result["error"]},
            "l2": {"rows_written": 0, "agg_rows": 0, "games_found": 0, "error": result["error"]},
            "from_cache": result["from_cache"],
            "remaining_requests": result.get("remaining_requests"),
        }

    now_ts = datetime.now(timezone.utc).isoformat()
    sport_lower = sport.lower()
    sharp_books_set = set(b.lower() for b in L1_SHARP_BOOKS)

    l1_rows = []
    l2_rows = []
    l1_games = set()
    l2_games = set()
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

            bm = row["bookmaker"].lower()
            books_seen.add(bm)

            home_norm = normalize_team_name(row["home_team"])
            away_norm = normalize_team_name(row["away_team"])

            side = row["side"]
            if row["market"] == "TOTAL":
                side = side.lower()
            else:
                side = normalize_team_name(side)

            line_val = row["line"] if row["line"] is not None else ""

            # L2: ALL books
            l2_games.add(canon_key)
            l2_rows.append({
                "timestamp": now_ts,
                "sport": sport_lower,
                "canonical_key": canon_key,
                "commence_time": row["commence_time"],
                "market": row["market"],
                "side": side,
                "bookmaker": bm,
                "line": line_val,
                "odds_american": row["odds_american"],
            })

            # L1: sharp books only
            if bm in sharp_books_set:
                l1_games.add(canon_key)
                l1_rows.append({
                    "timestamp": now_ts,
                    "sport": sport_lower,
                    "canonical_key": canon_key,
                    "bookmaker": bm,
                    "home_team_norm": home_norm,
                    "away_team_norm": away_norm,
                    "commence_time": row["commence_time"],
                    "market": row["market"],
                    "side": side,
                    "line": line_val,
                    "odds_american": row["odds_american"],
                })

    # Write L1
    l1_written = 0
    if l1_rows:
        from l1_scraper import L1_SHARP_COLUMNS, _load_l1_open_registry, _save_l1_open_registry
        write_header = not os.path.exists(L2_CONSENSUS_CSV.replace("l2_consensus", "l1_sharp"))
        from engine_config import L1_SHARP_CSV
        write_header = not os.path.exists(L1_SHARP_CSV)
        os.makedirs(os.path.dirname(L1_SHARP_CSV) or ".", exist_ok=True)
        with open(L1_SHARP_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=L1_SHARP_COLUMNS)
            if write_header:
                w.writeheader()
            for row in l1_rows:
                w.writerow(row)
        l1_written = len(l1_rows)

        # Update L1 open registry
        open_reg = _load_l1_open_registry()
        for row in l1_rows:
            reg_key = (row["sport"], row["canonical_key"], row["bookmaker"],
                       row["market"], row["side"])
            if reg_key not in open_reg:
                open_reg[reg_key] = {
                    "open_line": str(row["line"]),
                    "open_odds": str(row["odds_american"]),
                    "first_seen": now_ts,
                }
        _save_l1_open_registry(open_reg)

    # Write L2 raw
    l2_written = 0
    if l2_rows:
        write_header = not os.path.exists(L2_CONSENSUS_CSV)
        os.makedirs(os.path.dirname(L2_CONSENSUS_CSV) or ".", exist_ok=True)
        with open(L2_CONSENSUS_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=L2_RAW_COLUMNS)
            if write_header:
                w.writeheader()
            for row in l2_rows:
                w.writerow(row)
        l2_written = len(l2_rows)

    # Build L2 aggregation
    prev_agg = _load_prev_agg()
    groups = {}
    for row in l2_rows:
        gk = (row["sport"], row["canonical_key"], row["market"], row["side"])
        if gk not in groups:
            groups[gk] = {"lines": [], "odds": [], "pinn_line": None, "pinn_odds": None}
        try:
            lv = float(row["line"]) if row["line"] != "" else None
        except (ValueError, TypeError):
            lv = None
        try:
            ov = int(row["odds_american"])
        except (ValueError, TypeError):
            ov = None
        if lv is not None:
            groups[gk]["lines"].append(lv)
        if ov is not None:
            groups[gk]["odds"].append(ov)
        if row["bookmaker"] in sharp_books_set:
            if lv is not None:
                groups[gk]["pinn_line"] = lv
            if ov is not None:
                groups[gk]["pinn_odds"] = ov

    agg_rows = []
    for (sport_k, canon, market, side), data in sorted(groups.items()):
        n_books = len(data["lines"]) if data["lines"] else len(data["odds"])
        if n_books == 0:
            continue
        consensus_line = ""
        line_std = ""
        if data["lines"]:
            consensus_line = f"{statistics.median(data['lines']):.1f}"
            line_std = f"{statistics.stdev(data['lines']):.3f}" if len(data["lines"]) >= 2 else "0.000"
        consensus_odds = ""
        if data["odds"]:
            consensus_odds = str(int(statistics.median(data["odds"])))
        pinn_vs = ""
        pinn_line_str = ""
        pinn_odds_str = ""
        if data["pinn_line"] is not None:
            pinn_line_str = f"{data['pinn_line']:.1f}"
            if consensus_line:
                pinn_vs = f"{data['pinn_line'] - float(consensus_line):.2f}"
        if data["pinn_odds"] is not None:
            pinn_odds_str = str(data["pinn_odds"])
        consensus_dir = ""
        if data["lines"] and market in ("SPREAD", "TOTAL"):
            med = statistics.median(data["lines"])
            consensus_dir = "positive" if med > 0 else ("negative" if med < 0 else "neutral")
        prev_key = (sport_k, canon, market, side)
        line_std_prev = f"{prev_agg[prev_key]:.3f}" if prev_key in prev_agg else ""

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

    if agg_rows:
        with open(L2_CONSENSUS_AGG_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=L2_AGG_COLUMNS)
            w.writeheader()
            for row in agg_rows:
                w.writerow(row)

    return {
        "l1": {
            "rows_written": l1_written,
            "games_found": len(l1_games),
            "error": None if l1_rows else "No sharp book data (Pinnacle may not cover this sport)",
        },
        "l2": {
            "rows_written": l2_written,
            "agg_rows": len(agg_rows),
            "games_found": len(l2_games),
            "books_seen": sorted(books_seen),
            "error": None,
        },
        "from_cache": result["from_cache"],
        "remaining_requests": result.get("remaining_requests"),
    }
