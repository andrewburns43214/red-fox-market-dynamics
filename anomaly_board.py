import json
import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd


KEY_NUMBERS_BY_SPORT = {
    "nfl": [3, 7, 10, 14, 17],
    "ncaaf": [3, 7, 10, 14, 17, 21],
}

MEANINGFUL_MOVE_BY_MARKET = {
    "SPREAD": 0.5,
    "TOTAL": 1.0,
    "MONEYLINE": 15.0,
}

HOLD_MOVE_BY_MARKET = {
    "SPREAD": 0.25,
    "TOTAL": 0.5,
    "MONEYLINE": 1.0,
}

MEANINGFUL_PRICE_MOVE_PCT = 2.5
HOLD_PRICE_MOVE_PCT = 1.0
HEAVY_FAVORITE_ODDS = -300
PARLAY_RISK_ODDS = -200


def build_anomaly_outputs(latest_side_df, history_df, l2_df=None, as_of=None):
    latest_side_df = (latest_side_df if latest_side_df is not None else pd.DataFrame()).copy()
    history_df = (history_df if history_df is not None else pd.DataFrame()).copy()
    l2_df = (l2_df if l2_df is not None else pd.DataFrame()).copy()

    if latest_side_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    as_of = _coerce_ts(as_of) or datetime.now(timezone.utc)

    latest_side_df["market_display"] = latest_side_df.get("market_display", "").fillna("").astype(str).str.upper()
    latest_side_df["sport"] = latest_side_df.get("sport", "").fillna("").astype(str).str.lower()
    latest_side_df["game_id"] = latest_side_df.get("game_id", "").fillna("").astype(str)
    latest_side_df["side_key"] = latest_side_df.get("side_key", latest_side_df.get("side", "")).fillna("").astype(str)

    if not history_df.empty:
        history_df["timestamp"] = pd.to_datetime(history_df.get("timestamp"), errors="coerce", utc=True, format="mixed")
        history_df = history_df.dropna(subset=["timestamp"]).copy()
        history_df["market_display"] = history_df.get("market_display", "").fillna("").astype(str).str.upper()
        history_df["sport"] = history_df.get("sport", "").fillna("").astype(str).str.lower()
        history_df["game_id"] = history_df.get("game_id", "").fillna("").astype(str)
        history_df["side_key"] = history_df.get("side_key", history_df.get("side", "")).fillna("").astype(str)

        # Timeline history is only needed for markets represented in the current board.
        # Filtering before the group-by prevents old, unrelated snapshots from delaying
        # the live report without shortening any displayed market's timeline.
        history_key_cols = ["sport", "game_id", "market_display", "side_key"]
        candidate_keys = latest_side_df[history_key_cols].drop_duplicates()
        history_df = history_df.merge(candidate_keys, on=history_key_cols, how="inner")

    if not l2_df.empty:
        l2_df["timestamp"] = pd.to_datetime(l2_df.get("timestamp"), errors="coerce", utc=True, format="mixed")
        l2_df = l2_df.dropna(subset=["timestamp"]).copy()
        l2_df["sport"] = l2_df.get("sport", "").fillna("").astype(str).str.lower()
        l2_df["market"] = l2_df.get("market", "").fillna("").astype(str).str.upper()
        l2_df["side_norm"] = l2_df.get("side", "").fillna("").astype(str).map(_normalize_side_label)
        active_keys = set(latest_side_df.get("canonical_key", pd.Series(dtype=str)).fillna("").astype(str))
        l2_df = l2_df[l2_df.get("canonical_key", "").fillna("").astype(str).isin(active_keys)].copy()

    history_groups = {}
    if not history_df.empty:
        for group_key, group in history_df.groupby(["sport", "game_id", "market_display", "side_key"], dropna=False):
            history_groups[group_key] = group.sort_values("timestamp", kind="mergesort").copy()

    board_rows = []
    event_rows = []
    pair_cols = ["sport", "game_id", "market_display"]

    for pair_key, pair_df in latest_side_df.groupby(pair_cols, dropna=False):
        pair_rows = []
        for _, side_row in pair_df.iterrows():
            latest_key = (
                str(side_row.get("sport", "")).lower(),
                str(side_row.get("game_id", "")),
                str(side_row.get("market_display", "")).upper(),
                str(side_row.get("side_key", "")),
            )
            hist = history_groups.get(latest_key, pd.DataFrame()).copy()
            evaluation = _evaluate_side(side_row, hist, pair_df, l2_df, as_of)
            if evaluation is None:
                continue
            pair_rows.append(evaluation)
            event_rows.extend(evaluation.pop("_event_rows"))
        board_rows.extend(pair_rows)

    board_df = pd.DataFrame(board_rows)
    events_df = pd.DataFrame(event_rows)

    if board_df.empty:
        return board_df, events_df

    board_df = board_df.sort_values(
        ["anomaly_sort", "maturity_sort", "severity_sort", "kickoff_sort", "game", "market_display"],
        ascending=[True, True, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    board_df["board_rank"] = range(1, len(board_df) + 1)
    board_df = board_df.drop(columns=["severity_sort", "maturity_sort", "kickoff_sort"], errors="ignore")

    if not events_df.empty:
        events_df = events_df.sort_values(
            ["sport", "game_id", "market_display", "flagged_side", "timestamp"],
            ascending=[True, True, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)

    return board_df, events_df


def _evaluate_side(latest_row, history_rows, pair_df, l2_df, as_of):
    market = str(latest_row.get("market_display", "")).upper()
    sport = str(latest_row.get("sport", "")).lower()
    if market not in MEANINGFUL_MOVE_BY_MARKET:
        return None

    points = _build_history_points(history_rows, market)
    observation_count = len(points)
    if observation_count < 2:
        return None

    line_move_threshold = MEANINGFUL_MOVE_BY_MARKET[market]
    line_hold_threshold = HOLD_MOVE_BY_MARKET[market]
    move_threshold = MEANINGFUL_PRICE_MOVE_PCT if market == "MONEYLINE" else line_move_threshold
    hold_threshold = HOLD_PRICE_MOVE_PCT if market == "MONEYLINE" else line_hold_threshold
    open_value = points[0]["value"]
    current_value = points[-1]["value"]
    line_move_abs = abs(current_value - open_value)
    price_move_pct = _price_move_abs(points)
    move_abs = _movement_abs(points, market)

    dir_changes = _count_direction_changes(points, market)
    max_excursion = _max_excursion(points, market)
    path_min = min(_motion_value(point, market) for point in points)
    path_max = max(_motion_value(point, market) for point in points)
    line_moved = line_move_abs >= line_move_threshold
    juice_moved = price_move_pct >= MEANINGFUL_PRICE_MOVE_PCT
    meaningful_move = move_abs >= move_threshold if market == "MONEYLINE" else line_moved or juice_moved
    late_move = _is_late_move(points, market, move_threshold)
    whipsaw = dir_changes >= 1 and max_excursion >= move_threshold
    held = move_abs <= hold_threshold if market == "MONEYLINE" else (
        line_move_abs <= line_hold_threshold and price_move_pct <= HOLD_PRICE_MOVE_PCT
    )
    one_way = (not whipsaw) and meaningful_move
    path_label = ""
    if whipsaw:
        path_label = "Whipsaw"
    elif juice_moved and not line_moved:
        path_label = "Juice Move"
    elif held:
        path_label = "Held"
    elif late_move:
        path_label = "Late"
    elif one_way:
        path_label = "One-Way"

    bets_pct = _num(latest_row.get("bets_pct"))
    money_pct = _num(latest_row.get("money_pct"))
    split_capped = _is_split_capped(bets_pct, money_pct)
    current_odds = points[-1].get("odds")
    heavy_favorite = market == "MONEYLINE" and current_odds is not None and current_odds <= HEAVY_FAVORITE_ODDS
    parlay_risk = market == "MONEYLINE" and current_odds is not None and current_odds <= PARLAY_RISK_ODDS and money_pct >= 80
    split_alert_eligible = not split_capped and not parlay_risk
    low_support = bets_pct <= 40 and money_pct <= 45
    very_low_support = bets_pct <= 35 and money_pct <= 40
    ticket_heavy = bets_pct >= 70
    public_support = ticket_heavy and money_pct >= 55
    ticket_led = split_alert_eligible and ticket_heavy and money_pct < 55
    extreme_public = split_alert_eligible and public_support and bets_pct >= 80
    low_bets_high_money = split_alert_eligible and bets_pct <= 35 and money_pct >= 60

    move_toward_side = _move_toward_side(points, market, latest_row)
    key_number = _key_number_chip(sport, market, points)
    broader = _broader_market_context(latest_row, l2_df, market, move_threshold, hold_threshold)
    stale_dk = broader["stale_dk"]

    reaction = ""
    if split_alert_eligible and low_support and move_toward_side and meaningful_move:
        reaction = "Contrarian"
    elif split_alert_eligible and public_support and held:
        reaction = "Freeze"
    elif split_alert_eligible and public_support and move_toward_side and meaningful_move:
        reaction = "Follow"

    # Every active market has a primary state; secondary path/context signals add detail.
    if not reaction:
        reaction = "Watch"

    chips = [chip for chip in [reaction, path_label] if chip]
    context_chips = []
    if split_capped:
        context_chips.append("Split Cap")
    if heavy_favorite:
        context_chips.append("Heavy Favorite")
    if key_number:
        context_chips.append(key_number)
    if stale_dk:
        context_chips.append("Stale DK")
    if low_bets_high_money:
        context_chips.append("Low Bets / High $")
    if ticket_led:
        context_chips.append("Ticket-led")

    data_badge = _data_badge(points, latest_row, split_capped)
    path_summary = _path_summary(points)
    first_seen = _first_anomaly_seen(
        points, reaction, path_label, stale_dk, market, latest_row,
        move_threshold, hold_threshold, split_alert_eligible,
    )
    return_to_open = _return_toward_open(points, market)
    reason = _reason_line(
        reaction=reaction,
        path_label=path_label,
        stale_dk=stale_dk,
        low_support=low_support,
        public_support=public_support,
        ticket_led=ticket_led,
        low_bets_high_money=low_bets_high_money,
        move_abs=move_abs,
        move_threshold=move_threshold,
        held=held,
        broader_summary=broader["summary"],
        split_capped=split_capped,
        parlay_risk=parlay_risk,
        price_move_pct=price_move_pct,
    )

    flagged_side = str(latest_row.get("side", "")).strip() or str(latest_row.get("side_key", "")).strip()
    kickoff_ts = _coerce_ts(latest_row.get("_sort_time")) or _coerce_ts(latest_row.get("_game_time")) or _coerce_ts(latest_row.get("dk_start_iso"))
    kickoff_label = _format_kickoff(kickoff_ts)
    hours_to_kickoff = (kickoff_ts - as_of).total_seconds() / 3600 if kickoff_ts else None
    maturity_sort = 1 if hours_to_kickoff is not None and hours_to_kickoff > 48 else 0
    rank_reason = _rank_reason(
        reaction, path_label, stale_dk, split_capped, parlay_risk, hours_to_kickoff,
    )
    severity = _severity_score(
        reaction=reaction,
        whipsaw=whipsaw,
        extreme_public=extreme_public,
        move_abs=_movement_severity(market, line_move_abs, price_move_pct),
        max_excursion=_movement_severity(market, _line_max_excursion(points), _price_max_excursion(points)),
        stale_dk=stale_dk,
        low_bets_high_money=low_bets_high_money,
        very_low_support=very_low_support and split_alert_eligible,
    )

    event_rows = []
    for index, point in enumerate(points):
        event_rows.append({
            "sport": sport,
            "game_id": str(latest_row.get("game_id", "")),
            "canonical_key": str(latest_row.get("canonical_key", "")),
            "game": str(latest_row.get("game", "")),
            "market_display": market,
            "flagged_side": flagged_side,
            "timestamp": point["timestamp"].isoformat(),
            "step_index": index + 1,
            "observation_count": observation_count,
            "line_value": point["value"],
            "line_display": point["display"],
            "price_odds": point.get("odds", ""),
            "implied_pct": point.get("implied_pct", ""),
            "bets_pct": point["bets_pct"],
            "money_pct": point["money_pct"],
            "is_open": index == 0,
            "is_current": index == (observation_count - 1),
            "reaction": reaction,
            "path": path_label,
            "first_anomaly_seen": first_seen,
            "max_excursion": round(max_excursion, 3),
            "return_toward_open": return_to_open,
            "broader_market_comparison": broader["summary"],
            "key_number_note": key_number,
        })

    return {
        "sport": sport,
        "game_id": str(latest_row.get("game_id", "")),
        "canonical_key": str(latest_row.get("canonical_key", "")),
        "kickoff_time": kickoff_label,
        "kickoff_sort": kickoff_ts.isoformat() if kickoff_ts else "",
        "kickoff_iso": kickoff_ts.isoformat() if kickoff_ts else "",
        "game": str(latest_row.get("game", "")),
        "market_display": market,
        "flagged_side": flagged_side,
        "reaction": reaction,
        "path": path_label,
        "context_chips": " | ".join(context_chips),
        "anomaly_chips": " | ".join(chips + context_chips),
        "bets_pct": round(bets_pct, 1),
        "money_pct": round(money_pct, 1),
        "open_line": points[0]["display"],
        "current_line": points[-1]["display"],
        "path_summary": path_summary,
        "reason": reason,
        "data_badge": data_badge,
        "observation_count": observation_count,
        "first_anomaly_seen": first_seen,
        "max_excursion": round(max_excursion, 3),
        "return_toward_open": return_to_open,
        "broader_market_comparison": broader["summary"],
        "key_number_note": key_number,
        "open_line_value": round(open_value, 3),
        "current_line_value": round(current_value, 3),
        "move_abs": round(move_abs, 3),
        "line_move_abs": round(line_move_abs, 3),
        "price_move_pct": round(price_move_pct, 3),
        "movement_unit": "implied probability points" if market == "MONEYLINE" else "line points",
        "line_dir_changes": dir_changes,
        "path_min": round(path_min, 3),
        "path_max": round(path_max, 3),
        "observed_path": json.dumps([point["display"] for point in points]),
        "rank_reason": rank_reason,
        "anomaly_sort": _sort_rank(reaction, whipsaw, extreme_public, stale_dk, split_capped, parlay_risk),
        "maturity_sort": maturity_sort,
        "severity_sort": severity,
        "_event_rows": event_rows,
    }


def _build_history_points(history_rows, market):
    if history_rows is None or history_rows.empty:
        return []

    points = []
    for _, row in history_rows.iterrows():
        raw_line = row.get("current_line", "")
        parsed = _parse_snapshot_value(raw_line, market)
        if parsed["value"] is None:
            continue
        points.append({
            "timestamp": _coerce_ts(row.get("timestamp")),
            "value": parsed["value"],
            "display": parsed["display"],
            "base_display": parsed.get("base_display", parsed["display"]),
            "odds": parsed.get("odds"),
            "implied_pct": parsed.get("implied_pct"),
            "bets_pct": _num(row.get("bets_pct")),
            "money_pct": _num(row.get("money_pct")),
        })

    points = [point for point in points if point["timestamp"] is not None]
    points.sort(key=lambda point: point["timestamp"])
    if not points:
        return []

    return points


def _parse_snapshot_value(raw_line, market):
    text = str(raw_line or "").strip()
    if not text:
        return {"value": None, "display": ""}

    odds = _extract_last_number(text)
    implied_pct = _american_to_implied_pct(odds)

    if market == "MONEYLINE":
        odds = _extract_last_number(text)
        if odds is None:
            return {"value": None, "display": ""}
        return {
            "value": float(odds),
            "display": _format_odds(odds),
            "base_display": _format_odds(odds),
            "odds": odds,
            "implied_pct": implied_pct,
        }

    if market == "TOTAL":
        line_val = _extract_total_value(text)
        if line_val is None:
            return {"value": None, "display": ""}
        return {
            "value": float(line_val),
            "display": _format_line_with_odds(_format_total_display(text, line_val), odds),
            "base_display": _format_total_display(text, line_val),
            "odds": odds,
            "implied_pct": implied_pct,
        }

    line_val = _extract_spread_value(text)
    if line_val is None:
        return {"value": None, "display": ""}
    return {
        "value": float(line_val),
        "display": _format_line_with_odds(_format_spread_display(line_val), odds),
        "base_display": _format_spread_display(line_val),
        "odds": odds,
        "implied_pct": implied_pct,
    }


def _extract_last_number(text):
    import re

    match = re.search(r"@\s*([+-]?\d+)\s*$", text)
    if match:
        return int(match.group(1))
    match = re.search(r"([+-]?\d+)\s*$", text)
    if match:
        return int(match.group(1))
    return None


def _extract_total_value(text):
    import re

    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return None


def _extract_spread_value(text):
    import re

    match = re.search(r"([+-]\d+(?:\.\d+)?)\s*@\s*[+-]?\d+\s*$", text)
    if match:
        return float(match.group(1))
    match = re.search(r"([+-]\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return None


def _motion_value(point, market):
    if market == "MONEYLINE" and point.get("implied_pct") is not None:
        return point["implied_pct"]
    return point["value"]


def _movement_abs(points, market):
    return abs(_motion_value(points[-1], market) - _motion_value(points[0], market))


def _line_max_excursion(points):
    if not points:
        return 0.0
    open_value = points[0]["value"]
    return max(abs(point["value"] - open_value) for point in points)


def _price_move_abs(points):
    if not points or points[0].get("implied_pct") is None or points[-1].get("implied_pct") is None:
        return 0.0
    return abs(points[-1]["implied_pct"] - points[0]["implied_pct"])


def _price_max_excursion(points):
    if not points or points[0].get("implied_pct") is None:
        return 0.0
    open_price = points[0]["implied_pct"]
    values = [abs(point["implied_pct"] - open_price) for point in points if point.get("implied_pct") is not None]
    return max(values, default=0.0)


def _movement_severity(market, line_move_abs, price_move_pct):
    if market == "MONEYLINE":
        return price_move_pct
    return max(
        line_move_abs / MEANINGFUL_MOVE_BY_MARKET[market],
        price_move_pct / MEANINGFUL_PRICE_MOVE_PCT,
    )


def _count_direction_changes(points, market):
    if len(points) < 3:
        return 0
    deltas = []
    for left, right in zip(points, points[1:]):
        delta = _motion_value(right, market) - _motion_value(left, market)
        if math.isclose(delta, 0.0, abs_tol=1e-9):
            continue
        deltas.append(1 if delta > 0 else -1)
    changes = 0
    for left, right in zip(deltas, deltas[1:]):
        if left != right:
            changes += 1
    return changes


def _max_excursion(points, market):
    if not points:
        return 0.0
    open_value = _motion_value(points[0], market)
    return max(abs(_motion_value(point, market) - open_value) for point in points)


def _move_toward_side(points, market, latest_row):
    if len(points) < 2:
        return False
    open_value = points[0]["value"]
    current_value = points[-1]["value"]

    if market == "MONEYLINE":
        return _motion_value(points[-1], market) > _motion_value(points[0], market)

    if math.isclose(current_value, open_value, abs_tol=1e-9):
        open_price = points[0].get("implied_pct")
        current_price = points[-1].get("implied_pct")
        if open_price is not None and current_price is not None:
            return current_price > open_price

    if market == "TOTAL":
        side_text = str(latest_row.get("side", "")).lower()
        if "over" in side_text:
            return current_value > open_value
        if "under" in side_text:
            return current_value < open_value
        return False

    return current_value < open_value


def _is_late_move(points, market, move_threshold):
    if len(points) < 3:
        return False
    open_value = _motion_value(points[0], market)
    target_index = None
    for index, point in enumerate(points[1:], start=1):
        if abs(_motion_value(point, market) - open_value) >= move_threshold:
            target_index = index
            break
    if target_index is None:
        return False
    return (target_index / max(1, len(points) - 1)) >= 0.67


def _key_number_chip(sport, market, points):
    if market != "SPREAD":
        return ""
    keys = KEY_NUMBERS_BY_SPORT.get(sport, [])
    if not keys:
        return ""
    values = [abs(point["value"]) for point in points]
    for key in keys:
        for left, right in zip(values, values[1:]):
            lo, hi = sorted([left, right])
            if math.isclose(left, key, abs_tol=1e-9) or math.isclose(right, key, abs_tol=1e-9):
                return f"K{int(key)}"
            if lo < key < hi:
                return f"K{int(key)}"
    return ""


def _broader_market_context(latest_row, l2_df, market, move_threshold, hold_threshold):
    if l2_df is None or l2_df.empty:
        return {"stale_dk": False, "summary": ""}

    canonical_key = str(latest_row.get("canonical_key", "")).strip()
    side_norm = _normalize_side_label(latest_row.get("side_key", latest_row.get("side", "")))
    if not canonical_key or not side_norm:
        return {"stale_dk": False, "summary": ""}

    subset = l2_df[
        (l2_df["canonical_key"].fillna("").astype(str) == canonical_key) &
        (l2_df["market"].fillna("").astype(str).str.upper() == market) &
        (l2_df["side_norm"] == side_norm)
    ].copy()

    if subset.empty:
        return {"stale_dk": False, "summary": ""}

    metric_col = "odds_american" if market == "MONEYLINE" else "line"
    subset[metric_col] = pd.to_numeric(subset[metric_col], errors="coerce")
    if market == "MONEYLINE":
        subset["movement_value"] = subset[metric_col].map(_american_to_implied_pct)
    else:
        subset["movement_value"] = subset[metric_col]
    subset = subset.dropna(subset=["movement_value"])
    if subset.empty:
        return {"stale_dk": False, "summary": ""}

    values = subset["movement_value"].tolist()
    market_range = max(values) - min(values)
    market_start = float(subset.sort_values("timestamp").iloc[0]["movement_value"])
    market_end = float(subset.sort_values("timestamp").iloc[-1]["movement_value"])

    dk_open_parsed = _parse_snapshot_value(latest_row.get("open_line", ""), market)
    dk_current_parsed = _parse_snapshot_value(latest_row.get("current_line", ""), market)
    dk_open = dk_open_parsed["implied_pct"] if market == "MONEYLINE" else dk_open_parsed["value"]
    dk_current = dk_current_parsed["implied_pct"] if market == "MONEYLINE" else dk_current_parsed["value"]
    if dk_open is None or dk_current is None:
        return {"stale_dk": False, "summary": ""}

    dk_move = abs(dk_current - dk_open)
    stale = dk_move <= hold_threshold and abs(market_end - market_start) >= move_threshold and market_range >= move_threshold

    if market == "MONEYLINE":
        summary = f"Market price moved {market_start:.1f}% to {market_end:.1f}% while DK held near {dk_current_parsed['display']}"
    else:
        summary = f"Market moved {market_start:g} to {market_end:g} while DK held near {dk_current:g}"

    return {"stale_dk": stale, "summary": summary if stale else ""}


def _path_summary(points):
    prices_changed = len({point.get("odds") for point in points if point.get("odds") is not None}) > 1
    displays = [point["display"] if prices_changed else point.get("base_display", point["display"]) for point in points]
    if len(displays) <= 4:
        chosen = displays
    else:
        chosen = [displays[0], displays[1], displays[-2], displays[-1]]
    compact = []
    for display in chosen:
        if not compact or compact[-1] != display:
            compact.append(display)
    return " -> ".join(compact[:4])


def _first_anomaly_seen(points, reaction, path_label, stale_dk, market, latest_row, move_threshold, hold_threshold, split_alert_eligible):
    open_value = points[0]["value"]
    open_price = points[0].get("implied_pct")
    path_changes = 0
    last_dir = 0
    for index, point in enumerate(points[1:], start=1):
        delta = _motion_value(point, market) - _motion_value(points[index - 1], market)
        if not math.isclose(delta, 0.0, abs_tol=1e-9):
            cur_dir = 1 if delta > 0 else -1
            if last_dir and cur_dir != last_dir:
                path_changes += 1
            last_dir = cur_dir
        line_move_abs = abs(point["value"] - open_value)
        price_move_abs = abs(point.get("implied_pct", open_price) - open_price) if open_price is not None and point.get("implied_pct") is not None else 0.0
        move_abs = abs(_motion_value(point, market) - _motion_value(points[0], market))
        meaningful_move = move_abs >= move_threshold if market == "MONEYLINE" else (
            line_move_abs >= MEANINGFUL_MOVE_BY_MARKET[market] or price_move_abs >= MEANINGFUL_PRICE_MOVE_PCT
        )
        held = move_abs <= hold_threshold if market == "MONEYLINE" else (
            line_move_abs <= HOLD_MOVE_BY_MARKET[market] and price_move_abs <= HOLD_PRICE_MOVE_PCT
        )
        toward_side = _move_toward_side(points[: index + 1], market, latest_row)
        bets_pct = points[index]["bets_pct"]
        money_pct = points[index]["money_pct"]
        capped = _is_split_capped(bets_pct, money_pct)
        if not split_alert_eligible or capped:
            continue
        if reaction == "Contrarian" and bets_pct <= 40 and money_pct <= 45 and toward_side and meaningful_move:
            return point["timestamp"].isoformat()
        if reaction == "Freeze" and bets_pct >= 70 and money_pct >= 55 and held:
            return point["timestamp"].isoformat()
        if reaction == "Follow" and bets_pct >= 70 and money_pct >= 55 and toward_side and meaningful_move:
            return point["timestamp"].isoformat()
        if path_label == "Whipsaw" and path_changes >= 1 and move_abs >= move_threshold:
            return point["timestamp"].isoformat()
        if path_label == "Juice Move" and price_move_abs >= MEANINGFUL_PRICE_MOVE_PCT:
            return point["timestamp"].isoformat()
        if stale_dk and move_abs <= hold_threshold:
            return point["timestamp"].isoformat()
    return points[-1]["timestamp"].isoformat()


def _return_toward_open(points, market):
    if len(points) < 3:
        return False
    open_value = _motion_value(points[0], market)
    current_value = _motion_value(points[-1], market)
    best_excursion = max(abs(_motion_value(point, market) - open_value) for point in points[:-1])
    return abs(current_value - open_value) < best_excursion


def _reason_line(reaction, path_label, stale_dk, low_support, public_support, ticket_led, low_bets_high_money, move_abs, move_threshold, held, broader_summary, split_capped, parlay_risk, price_move_pct):
    if split_capped:
        if price_move_pct >= MEANINGFUL_PRICE_MOVE_PCT:
            return f"Price moved {price_move_pct:.1f} implied points, but a capped 0%/100% split is excluded from alert ranking"
        return "A capped 0%/100% split is shown for context but excluded from alert ranking"
    if parlay_risk:
        return "Heavy favorite price makes public split alignment prone to parlay bias"
    if reaction == "Contrarian":
        if path_label == "Whipsaw":
            return "Line improved for the weak side, then gave some back"
        if low_bets_high_money:
            return "Weak ticket support still drew a meaningful move with sharp-money shape"
        return "Low-support side still drew a meaningful move toward it"
    if reaction == "Freeze":
        if stale_dk and broader_summary:
            return broader_summary
        if held:
            return "Heavy tickets and money drew little or no move from open"
        return "Strong ticket and money support met a mostly held number"
    if reaction == "Follow":
        if path_label == "Late":
            return "Public side finally got a late move toward it"
        if path_label == "Juice Move":
            return "Strong ticket and money support matched a meaningful price move"
        return "Strong ticket and money support matched the move direction"
    if stale_dk and broader_summary:
        return broader_summary
    if path_label == "Whipsaw":
        return "Line reversed direction after a meaningful excursion"
    if path_label == "Juice Move":
        return f"Point line held while price moved {price_move_pct:.1f} implied points"
    if ticket_led and held:
        return "High ticket share had weak money support and the line held"
    if ticket_led:
        return "High ticket share had weak money support"
    if public_support:
        return "Strong ticket and money support had a mixed path"
    if low_support and move_abs >= move_threshold:
        return "Low-support side moved more than expected"
    return "Path shape stood out versus the split support"


def _data_badge(points, latest_row, split_capped=False):
    if len(points) < 2:
        return "Feed Risk"
    open_ok = _parse_snapshot_value(latest_row.get("open_line", ""), str(latest_row.get("market_display", "")).upper())["value"] is not None
    current_ok = _parse_snapshot_value(latest_row.get("current_line", ""), str(latest_row.get("market_display", "")).upper())["value"] is not None
    if not open_ok or not current_ok:
        return "Feed Risk"
    if split_capped:
        return "Split Risk"
    if len(points) >= 3:
        return "Clean"
    return "Thin"


def _severity_score(reaction, whipsaw, extreme_public, move_abs, max_excursion, stale_dk, low_bets_high_money, very_low_support):
    score = move_abs + max_excursion
    if reaction == "Contrarian":
        score += 5
    if reaction == "Freeze":
        score += 4
    if whipsaw:
        score += 4
    if stale_dk:
        score += 3
    if extreme_public:
        score += 2
    if low_bets_high_money:
        score += 2
    if very_low_support:
        score += 1
    return round(score, 3)


def _sort_rank(reaction, whipsaw, extreme_public, stale_dk, split_capped=False, parlay_risk=False):
    if split_capped and not stale_dk:
        return 8
    if parlay_risk and not stale_dk:
        return 8
    if reaction == "Contrarian" and whipsaw:
        return 0
    if reaction == "Freeze" and extreme_public:
        return 1
    if reaction == "Contrarian":
        return 2
    if reaction == "Freeze":
        return 3
    if stale_dk:
        return 4
    if whipsaw:
        return 5
    if reaction == "Follow":
        return 6
    return 7


def _normalize_side_label(text):
    value = str(text or "").strip().lower()
    for token in ["team:", "over ", "under "]:
        if value.startswith(token):
            value = value[len(token):]
    value = value.replace("st.", "st")
    value = value.replace("@", " ")
    value = " ".join(value.split())
    return value


def _format_spread_display(value):
    return f"{value:+g}"


def _format_total_display(text, value):
    upper = str(text or "").strip().lower()
    if upper.startswith("over"):
        return f"O {value:g}"
    if upper.startswith("under"):
        return f"U {value:g}"
    return f"{value:g}"


def _format_line_with_odds(line_display, odds):
    if odds is None:
        return line_display
    return f"{line_display} ({_format_odds(odds)})"


def _format_odds(value):
    number = int(round(float(value)))
    return f"{number:+d}"


def _american_to_implied_pct(odds):
    if odds is None:
        return None
    odds = float(odds)
    if math.isclose(odds, 0.0, abs_tol=1e-9):
        return None
    if odds < 0:
        return round((-odds / (-odds + 100)) * 100, 4)
    return round((100 / (odds + 100)) * 100, 4)


def _is_split_capped(bets_pct, money_pct):
    return bets_pct <= 0 or bets_pct >= 100 or money_pct <= 0 or money_pct >= 100


def _rank_reason(reaction, path_label, stale_dk, split_capped, parlay_risk, hours_to_kickoff):
    if split_capped:
        return "Split cap: a 0% or 100% source value cannot earn an alert rank."
    if parlay_risk:
        return "Heavy favorite: public split alignment is downranked for parlay risk."
    if stale_dk:
        return "Stale DK: broader market moved while the displayed DK price held."
    if reaction == "Contrarian":
        base = "Contrarian: low support paired with a move toward the side."
    elif reaction == "Freeze":
        base = "Freeze: strong tickets and money with no meaningful line or price move."
    elif reaction == "Follow":
        base = "Follow: strong tickets and money moved with the side."
    elif path_label == "Whipsaw":
        base = "Whipsaw: the observed price path materially reversed."
    elif path_label == "Juice Move":
        base = "Juice move: the point line held while the price changed materially."
    else:
        base = "Watch: active market without a qualifying alert."
    if hours_to_kickoff is not None and hours_to_kickoff > 48:
        return f"{base} Early market: ranked after closer games within the same signal class."
    return base


def _format_kickoff(ts):
    if ts is None:
        return ""
    return ts.astimezone(ZoneInfo("America/New_York")).strftime("%b %d %I:%M %p")


def _coerce_ts(value):
    if value is None or value == "":
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def _num(value):
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0
