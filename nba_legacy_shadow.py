"""Prospective NBA legacy comparison using separate lifecycle and grading files."""

from pathlib import Path

import pandas as pd

from build_live_recent import favorite_handoff_states
from performance_ledger import (_atomic_csv, _atomic_json, _attach_results, _cohort_row, _frozen_records,
                                _read_csv, summary)
from red_fox_favorite import NBA_PUBLIC_VERSION, _read_tracking, update_favorite_tracking


def record_legacy(board: pd.DataFrame, data_dir: Path, at, *, deferred_markets=frozenset(),
                  unavailable_sports=frozenset()) -> pd.DataFrame | None:
    nba = board.loc[board["sport"].eq("nba")] if "sport" in board else board.iloc[0:0]
    where = data_dir / "nba_legacy_hypothetical"
    where.mkdir(parents=True, exist_ok=True)
    marker = where / "started_at"
    start = pd.Timestamp(at)
    if not marker.exists():
        _seed(where, data_dir, nba, start)
    start = pd.Timestamp(marker.read_text().strip())
    if start > pd.Timestamp(at):
        return

    # Feed the unmodified legacy decisions through the same confirmation and
    # closing rules. The release switches apply only to the official state.
    legacy = update_favorite_tracking(nba, where, as_of=at, nba_config=(False, False),
                                     deferred_markets=deferred_markets, unavailable_sports=unavailable_sports)
    history = _read_tracking(where / "red_fox_favorite_tracking.csv")
    candidates = _read_csv(where / "red_fox_favorite_freeze_candidates.csv")
    ledger_path = where / "hypothetical_results.csv"
    ledger = _read_csv(ledger_path)
    if not candidates.empty:
        starts = pd.to_datetime(candidates["kickoff_iso"], utc=True, format="mixed", errors="coerce")
        first = pd.to_datetime(candidates["favorite_first_qualified_at"], utc=True, format="mixed", errors="coerce")
        scoped = candidates.loc[starts.le(pd.Timestamp(at)) & first.ge(start)].copy().astype(object)
        if not scoped.empty:
            raw = _read_csv(data_dir / "live_recent_market_candidates.csv")
            if not raw.empty:
                raw = raw.loc[raw["sport"].eq("nba")]
            handoff = favorite_handoff_states(scoped, pd.Timestamp(at), history, raw)
            records = [row for row in _frozen_records(handoff, history, "nba legacy hypothetical")
                       if row["favorite_qualified"] == "yes"]
            known = set(ledger.get("ledger_id", pd.Series(dtype=str)))
            additions = [row for row in records if row["ledger_id"] not in known]
            if additions:
                ledger = pd.concat([ledger, pd.DataFrame(additions)], ignore_index=True)
    if not ledger.empty:
        scores = _read_csv(data_dir / "final_scores_history.csv")
        if not scores.empty:
            ledger = _attach_results(ledger, scores.loc[scores["game_id"].isin(ledger["event_id"])])
        _atomic_csv(ledger, ledger_path)
    control = _cohort_row(ledger, "cohort", "nba_legacy_hypothetical", "accepted") if not ledger.empty else {}
    _atomic_json({**summary(ledger), "started_at": start.isoformat(), "last_observed_at": pd.Timestamp(at).isoformat(),
                  **{key: control.get(key) for key in ("stake_units", "net_profit_units", "roi")}},
                 where / "hypothetical_performance.json")
    return legacy


def _seed(where: Path, data_dir: Path, board: pd.DataFrame, start: pd.Timestamp) -> None:
    keys = set(board.get("game_id", pd.Series(dtype=str)))
    archive_name = "red_fox_favorite_freeze_candidates.csv"
    archive = _read_csv(data_dir / archive_name)
    if not archive.empty:
        day = start.tz_convert("America/New_York").normalize()
        tip = pd.to_datetime(archive["kickoff_iso"], utc=True, format="mixed", errors="coerce")
        archive = archive.loc[archive["sport"].eq("nba") & tip.ge(day) & archive["favorite_rule_version"].ne(NBA_PUBLIC_VERSION)]
        keys.update(archive["game_id"])
        _atomic_csv(archive, where / archive_name)
    for name in ("red_fox_favorite_tracking.csv", "red_fox_favorite_review.csv"):
        rows = _read_csv(data_dir / name)
        if not rows.empty:
            _atomic_csv(rows.loc[rows["sport"].eq("nba") & rows["game_id"].isin(keys) &
                                  rows["favorite_rule_version"].ne(NBA_PUBLIC_VERSION)], where / name)
    temporary = where / ".started_at.tmp"
    temporary.write_text(start.isoformat(), encoding="utf8")
    temporary.replace(where / "started_at")
