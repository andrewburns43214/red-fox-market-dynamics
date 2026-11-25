"""Project the verified NHL lifecycle into the existing Favorite record."""

import json

import pandas as pd

from favorite_nhl_goalies import _atomic, paper_dir, read_league_cache
from nhl_paper_release import index_schedule, record_paper, stamp, team
from nhl_release_config import (OFFICIAL_FLAGS, OFFICIAL_HANDOFF, OFFICIAL_VERSIONS,
                                enabled, official_nhl_favorite, official_start, switches)
from nhl_release_policy import handoff_issue
from performance_ledger import _read_csv
from red_fox_favorite import _late_invalidate_favorite, _sides, _snapshot_id, _truthy

STATE_FILE = "replacement_official_state.json"
LABELS = {
    "public_backed_moneyline": "Public-Backed Moneyline",
    "genuine_crossover": "Genuine Underdog-to-Favorite Moneyline Crossover",
}


def _state_path(data_dir):
    return data_dir.parent / "var" / "nhl-qualification" / STATE_FILE


def read_official_games(data_dir):
    try:
        return json.loads(_state_path(data_dir).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _occupied_games(archive, schedule):
    if archive.empty:
        return set()
    occupied = set()
    for row in archive.loc[archive["sport"].eq("nhl")].to_dict("records"):
        if not _truthy(row.get("red_fox_favorite")) or official_nhl_favorite(row):
            continue
        matchup = str(row.get("game", "")).split(" @ ")
        kickoff = stamp(row.get("kickoff_iso"))
        if len(matchup) != 2 or pd.isna(kickoff):
            continue
        away, home = map(team, matchup)
        occupied.update(key for key, game in schedule.items() if (game["away"], game["home"]) == (away, home)
                        and abs(kickoff.timestamp() - game["start"]) <= 4 * 3600)
    return occupied


def capture_official(board, history, goalies, data_dir, *, observed=None):
    settings = switches()
    active = {lane for lane, flag in OFFICIAL_FLAGS.items() if enabled(flag, settings)}
    if not active and not _state_path(data_dir).exists():
        return {}
    since = stamp(official_start())
    if pd.isna(since):
        return {}
    if observed is None:
        observed = pd.Timestamp.now(tz="UTC")
    if observed < since:
        return {}
    games = read_official_games(data_dir)
    schedule = index_schedule(read_league_cache(data_dir), observed)
    archive = _read_csv(data_dir / "red_fox_favorite_freeze_candidates.csv")
    occupied = _occupied_games(archive, schedule)
    # A disabled lane can finish a published selection. Its pending first
    # observations cannot be used to confirm a selection after it is reenabled.
    for game in games.values():
        for state in game["lanes"].values():
            if state["pathway"] not in active and not state.get("entry"):
                state.pop("pending", None)
    record_paper(board, history, goalies, schedule, games, active, observed,
                 official_since=since, blocked_games=occupied)
    _atomic(paper_dir(data_dir) / STATE_FILE, games)
    print(f"[nhl-official] lanes={len(active)} schedule={len(schedule)} lifecycle_games={len(games)}", flush=True)
    return {game["vendor_id"]: (key, game) for key, game in games.items()
            if game.get("selected_lane") and key not in occupied}


def _selected(game):
    return game["lanes"].get(game.get("selected_lane"), {})


def publish_official(board, selections, at):
    """Set ordinary Favorite fields after the legacy visibility pass."""
    if not selections or board.empty:
        return board
    nhl = board.loc[board["sport"].eq("nhl")]
    occupied = set(nhl.loc[nhl["red_fox_favorite"].map(_truthy), "game_id"].astype(str))
    for index, row in nhl.loc[nhl["market_display"].eq("MONEYLINE")].iterrows():
        vendor = str(row["game_id"])
        if vendor not in selections or vendor in occupied:
            continue
        key, game = selections[vendor]
        state = _selected(game)
        if state.get("removed"):
            _late_invalidate_favorite(board, index, state["removed"], at)
            continue
        if game.get("finalized"):
            continue
        entry, quote = state["entry"], state["quote"]
        matching = [side for side in _sides(row) if team(side["flagged_side"]) == entry["side"]]
        if len(matching) != 1 or stamp(matching[0].get("source_latest_at")) != stamp(quote["source_at"]):
            continue
        side = matching[0]
        pathway = state["pathway"]
        evidence = dict(provider_event_id=key, opening_fair=game["open"].get(entry["side"]),
                        open_line=side.get("open_line", ""), current_line=quote["line"],
                        entry_line=entry["line"], entry_source_at=entry["source_at"],
                        source_state_at=quote["source_at"], goalie_status=quote["goalie_status"],
                        bets_pct=quote["bets"], money_pct=quote["money"], reaction=side.get("reaction", ""),
                        path=side.get("path", ""))
        values = dict(red_fox_favorite="true", favorite_side=side["flagged_side"], favorite_pathway=pathway,
                      favorite_rule_version=OFFICIAL_VERSIONS[pathway], favorite_first_qualified_at=entry["source_at"],
                      favorite_final_qualified_at=quote["source_at"], favorite_state="qualified",
                      favorite_originally_qualified="true", favorite_snapshot_id=_snapshot_id(row, quote["source_at"]),
                      favorite_supporting_evidence=json.dumps(evidence, separators=(",", ":")),
                      favorite_final_market_read=side.get("reaction", ""), favorite_final_market_rank=row.get("board_rank", ""),
                      favorite_reason=f"NHL {LABELS[pathway]} qualified.", favorite_late_invalidated="false",
                      favorite_late_invalidated_at="", favorite_late_invalidated_reason="",
                      state_as_of_utc=quote["source_at"])
        for column, value in values.items():
            board.at[index, column] = value
    return board


def reconcile_official_handoffs(live, candidates, games, now):
    """Keep only the bound Favorite when its pathway passes at the safe cutoff."""
    if not {"sport", "game_id", "market_display"}.issubset(live.columns):
        return live
    result = live.copy()
    replacements = pd.Series(False, index=result.index)
    nhl = result["sport"].eq("nhl") & result["market_display"].eq("MONEYLINE")
    replacements.loc[nhl] = [official_nhl_favorite(row) for _, row in result.loc[nhl].iterrows()]
    already_frozen = result.get("freeze_method", pd.Series("", index=result.index)).eq(OFFICIAL_HANDOFF)
    provisional = replacements & ~already_frozen
    result.loc[provisional, "red_fox_favorite"] = "false"
    result.loc[provisional, "favorite_state"] = "not_qualified"
    if not games or candidates.empty:
        return result
    archive = {str(row["game_id"]): row for row in candidates.loc[candidates["sport"].eq("nhl")].to_dict("records")
               if official_nhl_favorite(row)}
    additions = []
    for key, game in games.items():
        state = _selected(game)
        if not state.get("entry") or now.timestamp() < game["cutoff"]:
            continue
        identity = (result["sport"].eq("nhl") & result["game_id"].astype(str).eq(game["vendor_id"])
                    & result["market_display"].eq("MONEYLINE"))
        if (identity & already_frozen).any():
            continue
        archived = archive.get(game["vendor_id"])
        if archived is None:
            continue
        row = {column: value for column, value in archived.items() if not column.startswith("_")}
        entry, quote = state["entry"], state["quote"]
        try:
            evidence = json.loads(row.get("favorite_supporting_evidence") or "{}")
        except (TypeError, ValueError):
            continue
        # Require both a board-published archive row and the exact last
        # authorized selected-side quote. Never resurrect a raw fallback.
        if (str(evidence.get("provider_event_id")) != key or evidence.get("entry_source_at") != entry["source_at"]
                or evidence.get("entry_line") != entry["line"] or team(row["favorite_side"]) != entry["side"]):
            continue
        reason = state.get("removed") or handoff_issue(state, game["cutoff"])
        if not reason and (evidence.get("source_state_at") != quote["source_at"]
                           or _truthy(row.get("favorite_late_invalidated"))):
            reason = "No published proof of the final verified NHL quote."
        # Do not let a raw or provisional market for this confirmed Favorite
        # claim the same ledger identity while the kickoff record is selected.
        replacements_or_raw = identity & (replacements | ~result["red_fox_favorite"].map(_truthy))
        result = result.loc[~replacements_or_raw].copy()
        replacements = replacements.reindex(result.index)
        already_frozen = already_frozen.reindex(result.index)
        if reason:
            row.update(red_fox_favorite="false", favorite_state="not_qualified", favorite_reason=reason)
        else:
            row.update(freeze_method=OFFICIAL_HANDOFF, favorite_reason="NHL Favorite passed its T−30 rule and kickoff handoff.")
        row.update(frozen_at_utc=now.isoformat(), final_pregame_state_at_utc=evidence["source_state_at"])
        additions.append(row)
    if additions:
        return pd.concat([result, pd.DataFrame(additions)], ignore_index=True, sort=False)
    return result
