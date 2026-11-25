"""Private, prospective tracking of confirmed NBA legacy recoveries blocked by suspension."""

import json
from pathlib import Path

import pandas as pd

from build_live_recent import favorite_handoff_states
from performance_ledger import (
    _american_profit, _atomic_csv, _atomic_json, _attach_results, _cohort_row, _line_and_price, _read_csv,
    _side_at_decision_line,
)
from red_fox_favorite import (
    CONFIG, NBA_PUBLIC_VERSION, NOV24_VERSION, _apply_nba_release, _confirmed_prior_favorite_recovery, _read_tracking,
    _side_identity, _sides, _timestamp, _truthy, _visibility_lock_minutes, favorite_side_quote,
)

ROUTES = ("same_market", "spread_to_moneyline")
MARKETS = ("SPREAD", "MONEYLINE")
LEGACY_VERSIONS = {CONFIG.basketball_version, NOV24_VERSION}
WITHHELD = "New legacy NBA Favorite promotion withheld."


def _stamp(value):
    return pd.to_datetime(value, errors="coerce", utc=True)


def _history(path, games):
    rows = _read_tracking(path)
    rows = rows.loc[rows["sport"].eq("nba") & rows["game_id"].isin(games)].copy()
    rows["_at"] = pd.to_datetime(rows["recorded_at"], format="mixed", errors="coerce", utc=True)
    return rows.sort_values("_at", kind="stable")


def _team_quote(row, team):
    matches = [side for side in _sides(row) if _side_identity(side.get("flagged_side")) == team]
    return matches[0] if len(matches) == 1 else {}


def _removed_legacy(history, team, market, quote_at):
    """Find the latest confirmed official legacy selection actually withdrawn pregame."""
    matches = []
    for old_market, rows in history.groupby("market_display", sort=False):
        if old_market not in MARKETS or (old_market != market and (old_market, market) != MARKETS):
            continue
        if len(rows) < 2:
            continue
        removed = rows.iloc[-1]
        prior = rows.iloc[-2]
        observed = _stamp(removed["source_observed_at"])
        reason = removed["transition_reason"]
        if (
            removed["favorite_state"] == "not_qualified" and prior["favorite_state"] == "qualified"
            and prior["favorite_rule_version"] in LEGACY_VERSIONS
            and _side_identity(prior["favorite_side"]) == team
            and removed["disappearance_reason"] == "current qualification gates no longer satisfied"
            and reason and not any(token in reason.lower() for token in ("unavailable", "temporarily", "stale", "feed gap"))
            and pd.notna(observed) and _stamp(prior["source_observed_at"]) <= observed < quote_at
            and prior["_at"] < removed["_at"]
        ):
            matches.append((prior, removed))
    return max(matches, key=lambda pair: pair[1]["_at"], default=(None, None))


def _discover(raw, legacy, official, history, official_history, data_dir, at, config):
    already_official = set(official.loc[official["sport"].eq("nba") & (
        official["red_fox_favorite"].map(_truthy)
        | official.get("manual_revision", pd.Series("", index=official.index)).fillna("").ne("")
    ), "game_id"]) if not official.empty else set()
    prior_official = official_history.loc[official_history["_at"].lt(at)]
    withheld = _apply_nba_release(raw.copy(), prior_official, _timestamp(at), *config)
    qualified = legacy.loc[legacy["red_fox_favorite"].map(_truthy) & legacy["market_display"].isin(MARKETS)]
    entries = []
    archives = None
    for _, row in qualified.iterrows():
        event, market, name = row["game_id"], row["market_display"], row["favorite_side"]
        if event in already_official or row["favorite_rule_version"] not in LEGACY_VERSIONS:
            continue
        team = _side_identity(name)
        games = history.loc[history["game_id"].eq(event) & history["market_display"].eq(market)]
        transition = games.loc[games["_at"].eq(at) & games["favorite_state"].eq("qualified")]
        preceding = games.loc[games["_at"].lt(at)]
        if len(transition) != 1 or (not preceding.empty and preceding.iloc[-1]["favorite_state"] == "qualified"):
            continue
        matching_raw = raw.loc[raw["game_id"].eq(event) & raw["market_display"].eq(market)]
        if len(matching_raw) != 1:
            continue
        index = matching_raw.index[0]
        current, kickoff = matching_raw.iloc[0], _stamp(row["kickoff_iso"])
        quote = _team_quote(current, team)
        observed = _stamp(quote.get("source_latest_at"))
        display = quote.get("current_line")
        line, price = _line_and_price(market, name, display)
        if (
            not team or team != _side_identity(current.get("supported_side"))
            or team != _side_identity(transition.iloc[0]["favorite_side"]) or _american_profit(price) is None
            or (market == "SPREAD" and not line) or display != favorite_side_quote(row)
            or pd.isna(observed) or observed != _stamp(transition.iloc[0]["source_observed_at"])
            or pd.isna(kickoff) or not observed <= at < kickoff - pd.Timedelta(minutes=_visibility_lock_minutes("nba"))
        ):
            continue
        official_game = prior_official.loc[prior_official["game_id"].eq(event)]
        original, removal = _removed_legacy(official_game, team, market, observed)
        first_published = official_game.loc[official_game["favorite_state"].eq("qualified")]
        if (original is None or removal["_at"] >= at
                or first_published.iloc[0]["favorite_rule_version"] == NBA_PUBLIC_VERSION):
            continue
        if withheld.at[index, "favorite_reason"] != WITHHELD:
            # The hold also explicitly disables the existing same-market Whipsaw
            # restoration branch, which can qualify without a raw Favorite flag.
            if _truthy(current["red_fox_favorite"]) or market != original["market_display"]:
                continue
            if archives is None:
                archives = _read_csv(data_dir / "red_fox_favorite_freeze_candidates.csv")
            if archives.empty:
                continue
            archived = archives.loc[archives["sport"].eq("nba") & archives["game_id"].eq(event)
                                    & archives["market_display"].eq(market)]
            if archived.empty or not _confirmed_prior_favorite_recovery(current, archived.iloc[-1])[0]:
                continue
        companion_market = next(candidate for candidate in MARKETS if candidate != market)
        companion = raw.loc[raw["game_id"].eq(event) & raw["market_display"].eq(companion_market)]
        entries.append({
            "event_id": event, "sport": "nba", "game": row["game"], "scheduled_start": kickoff.isoformat(),
            "market": market, "side": _side_at_decision_line(name, market, line), "team": team,
            "recovery_type": ROUTES[0] if market == original["market_display"] else ROUTES[1],
            "original_market": original["market_display"], "original_side": original["favorite_side"],
            "original_qualified_at": original["first_qualified_at"], "original_entry_quote": original["first_qualified_line"],
            "removed_at": removal["recorded_at"], "removal_source_observed_at": removal["source_observed_at"],
            "removal_reason": removal["transition_reason"], "reinstated_at": at.isoformat(),
            "entry_source_at": observed.isoformat(), "entry_quote": display, "decision_line": line, "decision_price": price,
            "entry_market_evidence": json.dumps(quote, separators=(",", ":")),
            "companion_market_evidence": json.dumps(_team_quote(companion.iloc[0], team) if len(companion) == 1 else {},
                                                   separators=(",", ":")),
            "cross_market_state": row.get("favorite_cross_market_state", ""),
            "legacy_qualification_evidence": row.get("favorite_supporting_evidence", ""),
            "subsequent_removals": "[]", "final_eligibility": "pending", "final_pregame_at": "",
            "count_for_performance": "no", "grade": "",
        })
    return entries


def _finish(entries, history, candidates, raw_candidates, at):
    for event, group in entries.groupby("event_id", sort=False):
        start = _stamp(group.iloc[-1]["scheduled_start"])
        rows = history.loc[history["game_id"].eq(event) & history["_at"].le(start)]
        for index, entry in group.iterrows():
            removals = rows.loc[rows["_at"].gt(_stamp(entry["reinstated_at"]))
                                & rows["market_display"].eq(entry["market"])
                                & rows["favorite_side"].map(_side_identity).eq(entry["team"])
                                & rows["favorite_state"].eq("not_qualified"),
                                ["recorded_at", "source_observed_at", "transition_reason", "disappearance_reason"]]
            entries.at[index, "subsequent_removals"] = removals.to_json(orient="records")
        if at < start or not group["final_eligibility"].eq("pending").any():
            continue
        entries.loc[group.index, "final_eligibility"] = "ineligible"
        if candidates.empty:
            continue
        archive = candidates.loc[candidates["game_id"].eq(event)]
        raw = raw_candidates.loc[raw_candidates["game_id"].eq(event)] if not raw_candidates.empty else raw_candidates
        frozen = favorite_handoff_states(archive, at, rows, raw)
        if frozen.empty:
            continue
        frozen = frozen.loc[frozen["red_fox_favorite"].map(_truthy)]
        active = frozen.loc[frozen["freeze_method"].ne("favorite_confirmed_by_final_paired_market")]
        if not active.empty:
            frozen = active
        if len(frozen) != 1:
            if len(frozen) > 1:
                entries.loc[group.index, "final_eligibility"] = "ambiguous"
            continue
        final = frozen.iloc[0]
        match = group.loc[group["market"].eq(final["market_display"])
                          & group["team"].eq(_side_identity(final["favorite_side"]))]
        if match.empty:
            continue
        selected = match.sort_values("reinstated_at", kind="stable").index[-1]
        entries.loc[match.index, "final_eligibility"] = "superseded"
        entries.loc[selected, ["final_eligibility", "final_pregame_at", "count_for_performance"]] = [
            "eligible", final["final_pregame_state_at_utc"], "yes",
        ]


def _performance(rows):
    values = _cohort_row(rows, "cohort", "nba_recovered_legacy", "accepted")
    return {"wins": values["wins"], "losses": values["losses"], "pushes": values["pushes"],
            "sample_size": values["graded"], "pending": values["candidates"] - values["graded"],
            "stake_units": values["stake_units"] or 0, "net_profit_units": values["net_profit_units"] or 0,
            "roi_pct": round(values["roi"] * 100, 4) if values["roi"] != "" else None}


def record_recoveries(raw, legacy, official, data_dir: Path, as_of, config, *, now=None):
    """Track current confirmations, then freeze and settle them through the existing private feed."""
    if not config[1] or legacy is None:
        return
    at = _stamp(as_of)
    now = _stamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    where = data_dir / "nba_legacy_hypothetical"
    output = where / "recovered_favorites.csv"
    report = where / "recovered_favorite_performance.json"
    started = json.loads(report.read_text(encoding="utf8"))["started_at"] if report.exists() else now.isoformat()
    entries = _read_csv(output)
    events = set(raw.get("game_id", ())) | set(entries.get("event_id", ()))
    history = _history(where / "red_fox_favorite_tracking.csv", events)
    through_capture = history.loc[history["_at"].le(at)]
    prior_state = through_capture.groupby(["game_id", "market_display"], sort=False)["favorite_state"].shift()
    new_confirmations = through_capture.loc[through_capture["_at"].eq(at)
                                            & through_capture["favorite_state"].eq("qualified")
                                            & prior_state.ne("qualified")]
    new_games = set(new_confirmations["game_id"]) & set(raw.get("game_id", ()))
    if new_games and not legacy.empty and at >= _stamp(started):
        official_history = _history(data_dir / "red_fox_favorite_tracking.csv", new_games)
        additions = pd.DataFrame(_discover(raw.loc[raw["game_id"].isin(new_games)],
                                           legacy.loc[legacy["game_id"].isin(new_games)], official,
                                           history, official_history, data_dir, at, config))
        if not additions.empty:
            entries = pd.concat([entries, additions], ignore_index=True).fillna("").drop_duplicates(
                ["event_id", "market", "team", "reinstated_at"], keep="first"
            )
    if not entries.empty:
        _finish(entries, history, _read_csv(where / "red_fox_favorite_freeze_candidates.csv"),
                _read_csv(data_dir / "live_recent_market_candidates.csv"), at)
        scores = _read_csv(data_dir / "final_scores_history.csv")
        # Outcomes of non-survivors are still useful evidence. Only the unique
        # survivor per game goes into the performance cohort.
        if not scores.empty:
            finished = entries.loc[entries["final_eligibility"].ne("pending")]
            if not finished.empty:
                graded = _attach_results(finished, scores.loc[scores["game_id"].isin(finished["event_id"])]).fillna("")
                entries = entries.reindex(columns=entries.columns.union(graded.columns, sort=False), fill_value="")
                entries.loc[graded.index, graded.columns] = graded
        _atomic_csv(entries, output)
    counted = entries.loc[entries["count_for_performance"].eq("yes")] if not entries.empty else pd.DataFrame(
        columns=["grade", "recovery_type"]
    )
    _atomic_json({"started_at": started, "last_observed_at": at.isoformat(), **_performance(counted),
                  "by_recovery_type": {route: _performance(counted.loc[counted["recovery_type"].eq(route)])
                                       for route in ROUTES}}, report)
