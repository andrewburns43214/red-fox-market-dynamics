"""List possible missed Favorite kickoff handoffs without changing any record."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() and path.stat().st_size else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def audit(data_dir: Path) -> pd.DataFrame:
    tracking = read(data_dir / "red_fox_favorite_tracking.csv")
    snapshots = read(data_dir / "snapshots.csv")
    favorites = read(data_dir / "performance_favorites.csv")
    live = read(data_dir / "live_recent.csv")
    if tracking.empty:
        return pd.DataFrame()
    tracking["_at"] = pd.to_datetime(tracking.get("recorded_at", ""), errors="coerce", utc=True)
    snapshots["_seen"] = pd.to_datetime(snapshots.get("timestamp", ""), errors="coerce", utc=True)
    snapshots["_kickoff"] = pd.to_datetime(snapshots.get("dk_start_iso", ""), errors="coerce", utc=True)
    official = set(zip(
        favorites.get("sport", pd.Series(dtype=str)).astype(str),
        favorites.get("event_id", pd.Series(dtype=str)).astype(str),
        favorites.get("market", pd.Series(dtype=str)).astype(str),
    ))
    live_favorites = live[
        live.get("red_fox_favorite", pd.Series("false", index=live.index))
        .astype(str).str.lower().isin({"1", "true", "yes"})
    ] if not live.empty else pd.DataFrame()
    live_keys = set(zip(
        live_favorites.get("sport", pd.Series(dtype=str)).astype(str),
        live_favorites.get("game_id", pd.Series(dtype=str)).astype(str),
        live_favorites.get("market_display", pd.Series(dtype=str)).astype(str),
    ))
    records = []
    keys = ["sport", "game_id", "market_display"]
    for key, group in tracking.sort_values("_at", kind="mergesort").groupby(keys, sort=False):
        group = group.reset_index(drop=True)
        for index, row in group.iterrows():
            if row.get("favorite_state") != "not_qualified" or row.get("disappearance_reason") != "no longer published or eligible":
                continue
            prior = group.iloc[:index]
            prior = prior[prior.favorite_state.eq("qualified")]
            if prior.empty:
                continue
            qualified = prior.iloc[-1]
            disappeared_at = row["_at"]
            game_snaps = snapshots[
                snapshots.get("sport", "").astype(str).eq(str(key[0]))
                & snapshots.get("game_id", "").astype(str).eq(str(key[1]))
                & snapshots["_seen"].notna()
                & (snapshots["_seen"] <= disappeared_at)
                & snapshots["_kickoff"].notna()
            ]
            kickoff = game_snaps.sort_values("_seen").iloc[-1]["_kickoff"] if not game_snaps.empty else pd.NaT
            minutes_to_kickoff = (kickoff - disappeared_at).total_seconds() / 60 if pd.notna(kickoff) and pd.notna(disappeared_at) else None
            normalized_key = (str(key[0]), str(key[1]), str(key[2]))
            status = "outside_close_review"
            if normalized_key in official or normalized_key in live_keys:
                status = "already_handed_off"
            elif minutes_to_kickoff is not None and -15 <= minutes_to_kickoff <= 30:
                status = "possible_missed_handoff" if minutes_to_kickoff >= 0 else "kickoff_shift_timing_review"
            records.append({
                "sport": key[0], "game_id": key[1], "game": qualified.get("game", ""),
                "market": key[2], "favorite_side": qualified.get("favorite_side", ""),
                "last_qualified_at": qualified.get("recorded_at", ""),
                "disappeared_at": row.get("recorded_at", ""),
                "kickoff_as_known_at_disappearance": kickoff.isoformat() if pd.notna(kickoff) else "",
                "minutes_to_kickoff": round(minutes_to_kickoff, 3) if minutes_to_kickoff is not None else "",
                "review_status": status,
            })
    result = pd.DataFrame(records)
    return result.sort_values(["disappeared_at", "sport", "game_id", "market"], kind="mergesort") if not result.empty else result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.data_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    review = result[result.review_status.isin(["possible_missed_handoff", "kickoff_shift_timing_review"])] if not result.empty else result
    print(review.to_string(index=False) if not review.empty else "No possible missed Favorite handoffs found.")


if __name__ == "__main__":
    main()
