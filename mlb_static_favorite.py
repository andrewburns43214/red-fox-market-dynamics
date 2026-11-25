"""Paired DraftKings stability and the standalone MLB Static Favorite rule."""
import csv
import json
import re

import pandas as pd

import red_fox_favorite as rf


RETRACE_LOG = "mlb_static_retrace_exclusions.csv"
RETRACE_REASON = "Movement retraced toward the opener."


def _odds(quote, side):
    value = str(quote).strip()
    match = re.search(r"([+-]\d{3,})\)?$", value)
    if not match or abs(int(match[1])) < 100:
        return None
    label = value[:match.start()].rstrip(" @(\t")
    if label and rf._side_identity(label) != side:
        return None
    return int(match[1])


def _probability(price):
    return -price / (100 - price) if price < 0 else 100 / (100 + price)


def _pair(rows):
    if len(rows) != 2:
        return None
    result = {}
    for row in rows:
        side = rf._side_identity(row["side"])
        current = _odds(row["current_line"], side)
        opening = _odds(row["open_line"], side)
        if not side or side in result or current is None:
            return None
        result[side] = (current, opening, rf._number(row["bets_pct"]), rf._number(row["money_pct"]))
    return result


def _fair(pair, index):
    probabilities = {side: _probability(values[index]) for side, values in pair.items() if values[index] is not None}
    if len(probabilities) != 2:
        return {}
    total = sum(probabilities.values())
    return {side: value / total for side, value in probabilities.items()}


def static_profile(quotes, teams, captured):
    """Return the exact prequalified side from distinct, paired raw captures."""
    current = quotes.get(captured)
    if not current or len(quotes) < 3 or set(current) != teams:
        return None
    fair, opening = _fair(current, 0), _fair(current, 1)
    if not opening or not fair:
        return None
    side = max(fair, key=fair.get)
    price, _, _, _ = current[side]
    movement = 100 * (fair[side] - opening[side])
    splits = [values[2:] for values in current.values()]
    if (fair[side] <= .5 or not -165 <= price <= -100 or abs(movement) > .5
            or any(v is None or v <= 0 or v >= 100 for p in splits for v in p)
            or any(abs(sum(values) - 100) > 2 for values in zip(*splits))):
        return None
    lo = hi = fair[side]
    earliest = captured
    for stamp in sorted((t for t in quotes if t < captured), reverse=True):
        if earliest - stamp > pd.Timedelta(minutes=15) or set(quotes[stamp]) != teams:
            break
        value = _fair(quotes[stamp], 0)[side]
        if max(hi, value) - min(lo, value) > .005 + 1e-10:
            break
        earliest, lo, hi = stamp, min(lo, value), max(hi, value)
        if captured - earliest >= pd.Timedelta(minutes=30):
            return side, current[side], movement, earliest
    return None


def _captures(history, now):
    fields = ["game_id", "timestamp", "side", "current_line", "open_line", "bets_pct", "money_pct"]
    if history is None or history.empty or not set(fields + ["sport", "market_display"]).issubset(history):
        return {}
    source = history.loc[history["sport"].eq("mlb") & history["market_display"].eq("MONEYLINE"), fields].copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True, errors="coerce", format="mixed")
    source = source.loc[source["timestamp"].le(now)].drop_duplicates()
    games = {}
    for (event, captured), rows in source.groupby(["game_id", "timestamp"], sort=False):
        if pair := _pair(rows.to_dict("records")):
            games.setdefault(str(event), {})[captured] = pair
    return games


def apply_static_moneyline(board, tracking, history, at, enabled, data_dir, *, diagnostics=False, promotion_games=None):
    now = pd.Timestamp(at)
    mlb = board.loc[board["sport"].eq("mlb")]
    if mlb.empty:
        return
    eligible = _captures(history, now)
    prior_by_game = {}
    if not tracking.empty:
        prior = tracking.loc[tracking["sport"].eq("mlb") & tracking["game_id"].isin(mlb["game_id"])]
        clocks = pd.to_datetime(prior["recorded_at"], utc=True, errors="coerce", format="mixed")
        prior_by_game = dict(tuple(prior.loc[clocks.le(now)].sort_values("recorded_at", kind="mergesort").groupby("game_id")))
    exclusions = []
    for event, game in mlb.groupby("game_id", sort=False):
        moneyline = game.loc[game["market_display"].eq("MONEYLINE")]
        if len(moneyline) != 1 or game["red_fox_favorite"].map(rf._truthy).any():
            continue
        index = moneyline.index[0]
        row = moneyline.iloc[0]
        previous = prior_by_game.get(event)
        past = previous.loc[previous["favorite_state"].eq("qualified")] if previous is not None else None
        if past is not None and not past.empty and (~past["favorite_rule_version"].eq(rf.MLB_STATIC_VERSION)).any():
            continue
        active = previous is not None and previous.iloc[-1]["favorite_state"] == "qualified"
        kickoff = pd.to_datetime(row.get("kickoff_iso"), errors="coerce", utc=True)
        if pd.isna(kickoff) or not pd.Timedelta(0) < kickoff - now <= pd.Timedelta(hours=3):
            continue
        if not active and kickoff - now <= pd.Timedelta(minutes=40):
            continue
        observed = pd.to_datetime(row.get("state_as_of_utc") or row.get("source_latest_at"), errors="coerce", utc=True)
        if pd.isna(observed):
            continue
        quotes = eligible.get(str(event), {})
        # Board serialization can truncate a raw Timestamp to microseconds.
        # Accept that representation only when it proves one unique capture.
        if observed not in quotes:
            matches = [stamp for stamp in quotes if stamp.floor("us") == observed]
            if len(matches) != 1:
                continue
            observed = matches[0]
        sides = {rf._side_identity(item["flagged_side"]): item for item in rf._sides(row)}
        profile = static_profile(quotes, set(sides), observed)
        if profile is None:
            continue
        identity, (price, opener, bets, money), movement, earliest = profile
        side = sides[identity]
        team = side["flagged_side"]
        if past is not None and not past.empty and rf._side_identity(past.iloc[0]["favorite_side"]) != identity:
            continue
        board_stamp = pd.to_datetime(side.get("source_latest_at"), errors="coerce", utc=True)
        if (board_stamp not in {observed, observed.floor("us")}
                or _odds(side.get("current_line"), identity) != price or _odds(side.get("open_line"), identity) != opener
                or rf._number(side.get("bets_pct")) != bets or rf._number(side.get("money_pct")) != money):
            continue
        if (any(game.get(field, pd.Series(dtype=object)).map(rf._truthy).any()
                for field in ("cross_market_mismatch", "cross_market_opener_mismatch_verified"))
                or rf._side_identity(row.get("supported_side")) not in {"", identity}):
            continue
        quality = rf._quality_failure(side)
        probe = row.copy()
        probe["favorite_side"] = team
        freshness = rf._freshness_failure(probe, None, now)
        if quality or freshness:
            if diagnostics and quality == RETRACE_REASON and not freshness:
                exclusions.append(dict(game_id=event, game=row["game"], source_observed_at=observed.isoformat(),
                                       side=team, entry_price=price, open_price=opener, no_vig_movement=movement,
                                       stable_since=earliest.isoformat(), reason=quality))
            continue
        if not active and (not enabled or (promotion_games is not None and str(event) not in promotion_games)):
            continue
        values = dict(red_fox_favorite="true", favorite_side=team, favorite_state="qualified",
                      favorite_rule_version=rf.MLB_STATIC_VERSION, favorite_pathway=rf.MLB_STATIC_PATH,
                      favorite_originally_qualified="true", favorite_first_qualified_at=at, favorite_final_qualified_at=at,
                      favorite_snapshot_id=rf._snapshot_id(row, at), favorite_reason="MLB Static Moneyline qualified.",
                      favorite_final_market_read=side.get("reaction", ""), favorite_final_market_rank=str(row.get("board_rank", "")),
                      favorite_supporting_evidence=json.dumps({"current_line": side["current_line"], "open_line": side["open_line"],
                          "source_latest_at": observed.isoformat(), "bets_pct": bets, "money_pct": money,
                          "stable_since": earliest.isoformat(), "no_vig_movement": movement}, separators=(",", ":")))
        for column, value in values.items():
            board.at[index, column] = value
    if exclusions:
        _log_retrace(data_dir / RETRACE_LOG, exclusions)


def _log_retrace(path, exclusions):
    try:
        known = set()
        if path.exists():
            with path.open(encoding="utf-8", newline="") as stream:
                known = {(row["game_id"], row["side"], row["source_observed_at"]) for row in csv.DictReader(stream)}
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(exclusions[0]))
            if stream.tell() == 0:
                writer.writeheader()
            for row in exclusions:
                if (str(row["game_id"]), row["side"], row["source_observed_at"]) not in known:
                    writer.writerow(row)
    except (OSError, csv.Error, KeyError) as exc:
        print(f"[mlb-static] retrace logging failed: {type(exc).__name__}", flush=True)
