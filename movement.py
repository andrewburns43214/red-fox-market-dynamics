import csv
from datetime import datetime

import re

def parse_line_and_odds(text: str):
    """
    Returns:
      line_val: float | None   (spread / total points)
      odds_val: int | None    (moneyline or juice)
    """
    if not text:
        return None, None

    # Odds always appear after '@'
    m_odds = re.search(r'@\s*([+-]?\d+)', text)
    odds = int(m_odds.group(1)) if m_odds else None

    # Line value should be a small number (spread/total), not a moneyline
    m_line = re.search(r'([+-]?\d+(\.\d+)?)', text)
    line = None

    if m_line:
        val = float(m_line.group(1))
        # Spreads/totals are small; moneylines are large
        if abs(val) <= 50:
            line = val

    return line, odds
def moneyline_direction(prev_odds: int, now_odds: int) -> str:
    # Favorite (negative odds)
    if prev_odds < 0 and now_odds < 0:
        if abs(now_odds) > abs(prev_odds):
            return "FAVORITE more expensive"
        else:
            return "FAVORITE cheaper"

    # Underdog (positive odds)
    if prev_odds > 0 and now_odds > 0:
        if now_odds > prev_odds:
            return "UNDERDOG more expensive"
        else:
            return "UNDERDOG cheaper"

    return "DIRECTION unclear"

def spread_direction(prev_line: float, now_line: float) -> str:
    """
    Determines direction of spread movement.
    """
    # Favorite line moved (more negative)
    if now_line < prev_line:
        return "TOWARD FAVORITE"

    # Underdog line moved (more positive)
    if now_line > prev_line:
        return "TOWARD UNDERDOG"

    return "NO DIRECTION"




def _parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", ""))
    if dt.tzinfo is not None:
        dt = dt.astimezone(None).replace(tzinfo=None)
    return dt


def movement_report(csv_path: str, sport: str, lookback: int = 1) -> None:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("sport") == sport]

    if not rows:
        print("[movement] no rows found")
        return

    stamps = sorted({r["timestamp"] for r in rows}, key=_parse_ts)
    if len(stamps) < 2:
        print("[movement] need at least two snapshots")
        return

    # Choose NOW as the most recent snapshot, PREV as N snapshots back.
    now_ts = stamps[-1]

    # Normalize + clamp lookback so it never crashes and behaves predictably.
    if lookback is None:
        lookback = 1
    try:
        lookback = int(lookback)
    except Exception:
        lookback = 1

    if lookback < 1:
        lookback = 1

    if lookback >= len(stamps):
        print(
            f"[movement] not enough snapshots for lookback={lookback} "
            f"(have {len(stamps)} snapshots for sport={sport})"
        )
        # fallback to comparing the most recent two snapshots
        lookback = 1

    prev_ts = stamps[-1 - lookback]

    print("[movement] Comparing snapshots:")
    print(" PREV:", prev_ts)
    print(" NOW :", now_ts)
    print()

    prev = {}
    now = {}
    open_price = {}

    for r in rows:
        key = (r["game_id"], r["side"], r.get("market"))
        ts = r["timestamp"]

        if key not in open_price:
            open_price[key] = r["current_line"]

        if ts == prev_ts:
            prev[key] = r["current_line"]
        elif ts == now_ts:
            now[key] = r["current_line"]

    moved = False

    for k, now_txt in now.items():
        prev_txt = prev.get(k)
        if not prev_txt or prev_txt == now_txt:
            continue

        opn_txt = open_price.get(k)

        prev_line, prev_odds = parse_line_and_odds(prev_txt)
        now_line, now_odds = parse_line_and_odds(now_txt)

        print(k[1])

        # SPREAD MOVE
        if prev_line is not None and now_line is not None and prev_line != now_line:
            print("  SPREAD MOVE")
            print(f"    direction: {spread_direction(prev_line, now_line)}")
            if opn_txt and opn_txt != prev_txt:
                print(f"    open: {opn_txt}")
            print(f"    was : {prev_txt}")
            print(f"    now : {now_txt}")

        # MONEYLINE MOVE
        elif prev_line is None and now_line is None and prev_odds != now_odds:
            print("  MONEYLINE MOVE")
            print(f"    direction: {moneyline_direction(prev_odds, now_odds)}")
            if opn_txt and opn_txt != prev_txt:
                print(f"    open: {opn_txt}")
            print(f"    was : {prev_txt}")
            print(f"    now : {now_txt}")

        # JUICE MOVE
        elif prev_odds != now_odds:
            print("  JUICE MOVE")
            if opn_txt and opn_txt != prev_txt:
                print(f"    open: {opn_txt}")
            print(f"    was : {prev_txt}")
            print(f"    now : {now_txt}")
