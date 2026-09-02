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
MARKET_MOVE_PRICE_PCT = 3.0
MARKET_MOVE_LINE_BY_SPORT = {
    "nfl": {"SPREAD": 1.0, "TOTAL": 1.5},
    "ncaaf": {"SPREAD": 1.5, "TOTAL": 1.5},
    "cfb": {"SPREAD": 1.5, "TOTAL": 1.5},
    "nba": {"SPREAD": 1.5, "TOTAL": 2.0},
    "ncaab": {"SPREAD": 1.5, "TOTAL": 2.0},
    "cbb": {"SPREAD": 1.5, "TOTAL": 2.0},
    "mlb": {"SPREAD": 1.0, "TOTAL": 1.0},
    "nhl": {"SPREAD": 0.5, "TOTAL": 0.5},
    "ufc": {"SPREAD": 1.5, "TOTAL": 1.5},
}
HEAVY_FAVORITE_ODDS = -300
EXTREME_UNDERDOG_ODDS = 300
PARLAY_RISK_ODDS = -200
EXPENSIVE_POINT_PRICE_ODDS = -125
HIGH_SPREAD_BY_SPORT = {
    "nfl": 10.0,
    "ncaaf": 21.0,
    "nba": 12.0,
    "ncaab": 15.0,
}
LATE_WINDOW_HOURS_BY_SPORT = {
    "nfl": 6.0,
    "ncaaf": 6.0,
}
MIN_CANDIDATE_OBSERVATIONS = 3
MIN_FREEZE_FADE_OBSERVATIONS = 4


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
    # The one-side-per-market publisher uses maturity and kickoff to preserve
    # the same ordering rationale in the final public board.

    if not events_df.empty:
        events_df = events_df.sort_values(
            ["sport", "game_id", "market_display", "flagged_side", "timestamp"],
            ascending=[True, True, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)

    return board_df, events_df


def select_market_leaders(board_df):
    """Publish one evidence-leading side for each game and market.

    The board is an at-a-glance market list; the timeline remains the place to
    compare both sides.  A previously recorded signal takes precedence so a
    valid signal is not hidden by a later, less notable live update.
    """
    if board_df is None or board_df.empty:
        return pd.DataFrame() if board_df is None else board_df.copy()

    keys = ["sport", "game_id", "market_display"]
    work = board_df.copy()
    for column in keys:
        if column not in work.columns:
            work[column] = ""
    recorded_reaction = work.get("recorded_reaction", pd.Series("", index=work.index))
    work["_has_recorded_signal"] = recorded_reaction.fillna("").astype(str).str.strip().ne("")
    current_reaction = work.get("reaction", pd.Series("Watch", index=work.index))
    effective_reaction = recorded_reaction.where(work["_has_recorded_signal"], current_reaction).fillna("Watch")
    context = work.get("context_chips", pd.Series("", index=work.index)).fillna("").astype(str)
    market_move = context.str.contains(r"(?:^|\|)\s*Market Move\s*(?:\||$)", regex=True)
    price_risk = context.str.contains(r"(?:^|\|)\s*Price Risk\s*(?:\||$)", regex=True)
    unreliable_move = context.str.contains(r"Split Cap|Heavy Favorite", regex=True)
    # Current independently material movement ranks above a held public split.
    # Capped split and heavy-favorite markets remain downranked as context only.
    work["_signal_rank"] = effective_reaction.map({"Contrarian": 0, "Freeze": 3, "Follow": 4, "Watch": 5}).fillna(6).astype(float)
    work.loc[market_move & ~unreliable_move & (effective_reaction != "Contrarian"), "_signal_rank"] = 2
    # An extreme price is useful movement context, but a clean market move is
    # more comparable across the board. Keep confirmed Contrarian evidence in
    # its own class rather than erasing it because of the attached price risk.
    work.loc[market_move & price_risk & ~unreliable_move & (effective_reaction == "Watch"), "_signal_rank"] = 2.5
    anomaly_sort = work.get("anomaly_sort", pd.Series(99, index=work.index))
    severity_sort = work.get("severity_sort", pd.Series(0, index=work.index))
    maturity_sort = work.get("maturity_sort", pd.Series(0, index=work.index))
    kickoff_sort = work.get("kickoff_sort", pd.Series("", index=work.index))
    work["_anomaly_sort"] = pd.to_numeric(anomaly_sort, errors="coerce").fillna(99)
    work["_severity_sort"] = pd.to_numeric(severity_sort, errors="coerce").fillna(0)
    work["_maturity_sort"] = pd.to_numeric(maturity_sort, errors="coerce").fillna(0)
    work["_kickoff_sort"] = kickoff_sort.fillna("").astype(str)

    work = work.sort_values(
        ["_signal_rank", "_anomaly_sort", "_maturity_sort", "_severity_sort", "_kickoff_sort", "_has_recorded_signal", "flagged_side"],
        ascending=[True, True, True, False, True, False, True],
        kind="mergesort",
    )
    leaders = work.drop_duplicates(keys, keep="first").copy()
    leaders = leaders.sort_values(
        ["_signal_rank", "_anomaly_sort", "_maturity_sort", "_severity_sort", "_kickoff_sort", "_has_recorded_signal", "game", "market_display"],
        ascending=[True, True, True, False, True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    leaders["board_rank"] = range(1, len(leaders) + 1)
    return leaders.drop(columns=["_has_recorded_signal", "_signal_rank", "_anomaly_sort", "_maturity_sort", "_severity_sort", "_kickoff_sort"], errors="ignore")


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
    high_spread = market == "SPREAD" and abs(current_value) >= HIGH_SPREAD_BY_SPORT.get(sport, float("inf"))
    if high_spread:
        # A half-point around a 25-plus-point college spread is common noise;
        # require a full point before it can qualify through line movement.
        line_move_threshold = max(line_move_threshold, 1.0)
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
    developing_move = (
        move_abs >= move_threshold / 2
        if market == "MONEYLINE"
        else line_move_abs >= line_move_threshold / 2 or price_move_pct >= MEANINGFUL_PRICE_MOVE_PCT / 2
    )
    kickoff_ts = _coerce_ts(latest_row.get("_sort_time")) or _coerce_ts(latest_row.get("_game_time")) or _coerce_ts(latest_row.get("dk_start_iso"))
    hours_to_kickoff = (kickoff_ts - as_of).total_seconds() / 3600 if kickoff_ts else None
    late_move = _is_late_move(points, market, move_threshold, hours_to_kickoff, sport)
    whipsaw = dir_changes >= 1 and max_excursion >= move_threshold
    held = move_abs <= hold_threshold if market == "MONEYLINE" else (
        line_move_abs <= line_hold_threshold and price_move_pct <= HOLD_PRICE_MOVE_PCT
    )
    one_way = (not whipsaw) and meaningful_move
    # Market Move is deliberately stricter than a split-backed signal. It only
    # describes a material change still present from the opening observation.
    market_move = _is_current_market_move(
        sport, market, line_move_abs, price_move_pct,
    )
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
    # Ticket concentration on a short favorite is commonly parlay-driven. It is
    # useful board context, but should not promote a split-only alert by itself.
    parlay_risk = market == "MONEYLINE" and current_odds is not None and current_odds <= PARLAY_RISK_ODDS and bets_pct >= 80
    favorite_risk = heavy_favorite or parlay_risk
    split_alert_eligible = not split_capped and not favorite_risk
    # A side can still be meaningfully low-support when tickets are in the low
    # forties but money is materially lower. Avoid erasing a valid move on a
    # one-point ticket-share change.
    low_support = bets_pct <= 45 and money_pct <= 45
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
    developing_read = (
        reaction == "Watch"
        and split_alert_eligible
        and (low_support or public_support)
        and move_toward_side
        and developing_move
    )

    focus_basis = _focus_basis(
        reaction, low_bets_high_money, ticket_led, split_capped, favorite_risk,
    )
    chips = [chip for chip in [reaction, path_label] if chip]
    context_chips = []
    if split_capped:
        context_chips.append("Split Cap")
    if favorite_risk:
        context_chips.append("Heavy Favorite")
    if key_number:
        context_chips.append(key_number)
    if stale_dk:
        context_chips.append("Market Lag")
    if reaction == "Watch" and public_support and not split_capped and not favorite_risk:
        context_chips.append("Public Pressure")
    if developing_read:
        context_chips.append("Developing Read")
    if market_move:
        context_chips.append("Market Move")
    price_risk_note = _price_risk_note(reaction, market, sport, current_value, current_odds)
    if price_risk_note:
        context_chips.append("Price Risk")
    if low_bets_high_money:
        context_chips.append("Low Bets / High $")
    if ticket_led:
        context_chips.append("Ticket-led")

    data_badge = _data_badge(points, latest_row, split_capped)
    path_summary = _path_summary(points)
    first_seen = _first_anomaly_seen(
        points, reaction, path_label, stale_dk, market, latest_row,
        move_threshold, line_move_threshold, hold_threshold, split_alert_eligible,
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
        favorite_risk=favorite_risk,
        price_move_pct=price_move_pct,
        developing_read=developing_read,
        market_move=market_move,
        bets_pct=bets_pct,
        money_pct=money_pct,
    )
    market_move_note = _market_move_note(sport, market, line_move_abs, price_move_pct) if market_move else ""
    if market_move_note:
        reason = market_move_note if reaction == "Watch" else _append_sentence(reason, market_move_note)
    if price_risk_note:
        reason = _append_sentence(reason, price_risk_note)
    reason = _finish_sentence(reason)

    flagged_side = str(latest_row.get("side", "")).strip() or str(latest_row.get("side_key", "")).strip()
    action = _action_fields(
        reaction=reaction,
        flagged_side=flagged_side,
        latest_row=latest_row,
        pair_df=pair_df,
        observation_count=observation_count,
        key_number=key_number,
        stale_dk=stale_dk,
        split_capped=split_capped,
        favorite_risk=favorite_risk,
        price_risk_note=price_risk_note,
        bets_pct=bets_pct,
        money_pct=money_pct,
    )
    kickoff_label = _format_kickoff(kickoff_ts)
    maturity_sort = 1 if hours_to_kickoff is not None and hours_to_kickoff > 48 else 0
    rank_reason = _rank_reason(
        reaction, path_label, stale_dk, split_capped, favorite_risk, hours_to_kickoff, market_move,
        price_risk=bool(price_risk_note),
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
            "focus_basis": focus_basis,
            **action,
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
        "focus_basis": focus_basis,
        **action,
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
        "anomaly_sort": _sort_rank(reaction, whipsaw, extreme_public, stale_dk, split_capped, favorite_risk, developing_read, market_move, bool(price_risk_note)),
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

    # A scraper retry can append two values with the same source timestamp.
    # Keep the final capture for that instant so the timeline remains one
    # coherent state sequence rather than plotting contradictory duplicates.
    latest_by_timestamp = {}
    for point in points:
        latest_by_timestamp[point["timestamp"]] = point
    return [latest_by_timestamp[timestamp] for timestamp in sorted(latest_by_timestamp)]


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


def _market_move_line_threshold(sport, market):
    return MARKET_MOVE_LINE_BY_SPORT.get(sport, {}).get(
        market, MEANINGFUL_MOVE_BY_MARKET[market]
    )


def _is_current_market_move(sport, market, line_move_abs, price_move_pct):
    if market == "MONEYLINE":
        return price_move_pct >= MARKET_MOVE_PRICE_PCT
    return (
        line_move_abs >= _market_move_line_threshold(sport, market)
        or price_move_pct >= MARKET_MOVE_PRICE_PCT
    )


def _market_move_note(sport, market, line_move_abs, price_move_pct):
    if market == "MONEYLINE":
        return f"Market Move: moneyline price changed {price_move_pct:.1f} implied points from open to current"
    if line_move_abs >= _market_move_line_threshold(sport, market):
        return f"Market Move: line changed {line_move_abs:g} points from open to current"
    return f"Market Move: attached price changed {price_move_pct:.1f} implied points while the line stayed below its movement threshold"


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


def _is_late_move(points, market, move_threshold, hours_to_kickoff, sport):
    """Label a move late only when it occurs in the actual closing window."""
    if hours_to_kickoff is None or hours_to_kickoff > LATE_WINDOW_HOURS_BY_SPORT.get(sport, 3.0):
        return False
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
        summary = f"Broader market price moved {market_start:.1f}% to {market_end:.1f}% while the observed price held near {dk_current_parsed['display']}"
    else:
        summary = f"Broader market moved {market_start:g} to {market_end:g} while the observed price held near {dk_current:g}"

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


def _first_anomaly_seen(points, reaction, path_label, stale_dk, market, latest_row, move_threshold, line_move_threshold, hold_threshold, split_alert_eligible):
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
            line_move_abs >= line_move_threshold or price_move_abs >= MEANINGFUL_PRICE_MOVE_PCT
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
        if reaction == "Contrarian" and bets_pct <= 45 and money_pct <= 45 and toward_side and meaningful_move:
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


def _reason_line(reaction, path_label, stale_dk, low_support, public_support, ticket_led, low_bets_high_money, move_abs, move_threshold, held, broader_summary, split_capped, favorite_risk, price_move_pct, developing_read=False, market_move=False, bets_pct=0.0, money_pct=0.0):
    if split_capped:
        if price_move_pct >= MEANINGFUL_PRICE_MOVE_PCT:
            return f"Price moved {price_move_pct:.1f} implied points, but a capped 0%/100% split is excluded from alert ranking"
        return "A capped 0%/100% split is shown for context but excluded from alert ranking"
    if favorite_risk:
        return "Short moneyline favorite is shown as context; ticket concentration may be parlay-biased"
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
            return f"{bets_pct:.0f}% bets and {money_pct:.0f}% money stayed on the observed side while its number held near open"
        return f"{bets_pct:.0f}% bets and {money_pct:.0f}% money stayed on the observed side with a mostly held number"
    if reaction == "Follow":
        if path_label == "Late":
            return "Public side finally got a late move toward it"
        if path_label == "Juice Move":
            return "Strong ticket and money support matched a meaningful price move"
        return "Strong ticket and money support matched the move direction"
    if stale_dk and broader_summary:
        return broader_summary
    if developing_read:
        return "Split support and direction agree, but the move is below the confirmed signal threshold"
    if path_label == "Whipsaw":
        return "Line reversed direction after a meaningful excursion"
    if path_label == "Juice Move":
        return f"Point line held while price moved {price_move_pct:.1f} implied points"
    if low_bets_high_money and held:
        return f"{bets_pct:.0f}% bets versus {money_pct:.0f}% money held near open; the split is notable but has no confirming move"
    if ticket_led and held:
        return "High ticket share had weak money support and the line held"
    if ticket_led:
        return "High ticket share had weak money support"
    if public_support:
        return "High tickets and money are present, but the market was neither held nor meaningfully aligned"
    if low_support and move_abs >= move_threshold:
        return "Low-support side moved more than expected"
    return "Path shape stood out versus the split support"


def _append_sentence(base, detail):
    """Combine independent rationale notes without run-on punctuation."""
    base = str(base or "").strip().rstrip(".")
    detail = str(detail or "").strip()
    if not detail:
        return base
    return f"{base}. {detail}"


def _finish_sentence(value):
    """Keep board rationale copy readable even when there is only one clause."""
    text = str(value or "").strip()
    if not text or text.endswith((".", "!", "?")):
        return text
    return f"{text}."


def _focus_basis(reaction, low_bets_high_money, ticket_led, split_capped, favorite_risk):
    if split_capped:
        return "Split unavailable; price tracking only"
    if favorite_risk:
        return "Short favorite; context only"
    if reaction == "Freeze":
        return "High-split side; market held"
    if reaction == "Contrarian":
        return "Low-support side; price moved toward it"
    if reaction == "Follow":
        return "High-split side; price moved with it"
    if low_bets_high_money:
        return "Low tickets, high money"
    if ticket_led:
        return "Ticket-led side"
    return "Observed side"


def _action_fields(
    reaction,
    flagged_side,
    latest_row,
    pair_df,
    observation_count,
    key_number,
    stale_dk,
    split_capped,
    favorite_risk,
    price_risk_note,
    bets_pct,
    money_pct,
):
    """Keep observed evidence distinct from KPI-eligible action candidates."""
    observed_line = str(latest_row.get("current_line", "")).strip()
    base = {
        "action_side": "",
        "action_line": "",
        "action_type": "OBSERVE ONLY",
        "action_basis": "Evidence only; no reportable action candidate.",
        "kpi_eligible": False,
    }
    if split_capped or favorite_risk:
        return base

    if reaction == "Contrarian" and observation_count >= MIN_CANDIDATE_OBSERVATIONS:
        return {
            "action_side": flagged_side,
            "action_line": observed_line,
            "action_type": "CONTRARIAN CANDIDATE",
            "action_basis": "Low-support side received a sustained move toward it.",
            "kpi_eligible": True,
        }

    if reaction != "Freeze":
        return base

    if observation_count < MIN_FREEZE_FADE_OBSERVATIONS:
        base["action_basis"] = "Freeze needs four timestamped observations before it can be tracked as a fade candidate."
        return base
    if key_number:
        base["action_basis"] = "Freeze is pinned at a key number; treat it as evidence only."
        return base
    if stale_dk:
        base["action_basis"] = "DraftKings appears stale versus the broader market; do not grade this as a public fade."
        return base
    if bets_pct < 80 or money_pct < 60:
        base["action_basis"] = "Freeze pressure is not strong enough for a reportable public-fade candidate."
        return base

    counterpart = _counterpart_row(pair_df, latest_row)
    if counterpart is None:
        base["action_basis"] = "No verified opposing market side is available to grade."
        return base
    action_side = str(counterpart.get("side", "")).strip() or str(counterpart.get("side_key", "")).strip()
    action_line = str(counterpart.get("current_line", "")).strip()
    if not action_side or not action_line:
        base["action_basis"] = "The opposing market side is incomplete, so this Freeze remains evidence only."
        return base
    basis = "Sustained high public pressure held away from a key number; grade the opposing side as a public-fade candidate."
    if price_risk_note:
        basis = f"{basis} {price_risk_note}"
    return {
        "action_side": action_side,
        "action_line": action_line,
        "action_type": "FADE CANDIDATE",
        "action_basis": basis,
        "kpi_eligible": True,
    }


def _counterpart_row(pair_df, latest_row):
    if pair_df is None or pair_df.empty:
        return None
    side_key = str(latest_row.get("side_key", "")).strip()
    candidates = pair_df[pair_df.get("side_key", pd.Series(dtype=str)).fillna("").astype(str).str.strip() != side_key]
    if len(candidates) != 1:
        return None
    return candidates.iloc[0]


def _price_risk_note(reaction, market, sport, current_value, current_odds):
    """Add pricing or large-spread context without manufacturing an alert."""
    if market == "MONEYLINE" and current_odds is not None and (
        current_odds <= HEAVY_FAVORITE_ODDS or current_odds >= EXTREME_UNDERDOG_ODDS
    ):
        return f"Price Risk: the current {int(current_odds):+d} moneyline is an extreme price; any movement is context, not a standalone signal."
    if market == "SPREAD" and abs(current_value) >= HIGH_SPREAD_BY_SPORT.get(sport, float("inf")):
        return f"Price Risk: at this {abs(current_value):g}-point spread, small line changes are less reliable; require at least a full-point move to confirm a split-based directional read."
    if reaction != "Freeze":
        return ""
    if market in {"SPREAD", "TOTAL"} and current_odds is not None and current_odds <= EXPENSIVE_POINT_PRICE_ODDS:
        return f"Price Risk: the attached {int(current_odds):+d} juice is expensive for a held high-split side."
    return ""


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


def _sort_rank(reaction, whipsaw, extreme_public, stale_dk, split_capped=False, favorite_risk=False, developing_read=False, market_move=False, price_risk=False):
    if split_capped and not stale_dk:
        return 8
    if favorite_risk and not stale_dk:
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
    if developing_read:
        return 6.5
    if market_move:
        return 6.9 if price_risk else 6.75
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


def _rank_reason(reaction, path_label, stale_dk, split_capped, favorite_risk, hours_to_kickoff, market_move=False, price_risk=False):
    if split_capped:
        return "Split cap: a 0% or 100% source value cannot earn an alert rank."
    if favorite_risk:
        return "Heavy favorite: short-price ticket concentration is downranked for parlay risk."
    if stale_dk:
        return "Market lag: the broader market moved while the observed price held."
    if reaction == "Contrarian":
        base = "Contrarian: low support paired with a move toward the side."
    elif reaction == "Freeze":
        base = "Freeze: the high-split side had no meaningful line or price move."
    elif reaction == "Follow":
        base = "Follow: strong tickets and money moved with the side."
    elif path_label == "Whipsaw":
        base = "Whipsaw: the observed price path materially reversed."
    elif path_label == "Juice Move":
        base = "Juice move: the point line held while the price changed materially."
    elif market_move:
        base = "Market move: the number changed meaningfully, independent of the current split."
    else:
        base = "Watch: surfaced for a notable split, path, or price change, but it is not a directional candidate."
    if price_risk and market_move:
        base = f"{base} Price risk: the caution condition reduces priority."
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
