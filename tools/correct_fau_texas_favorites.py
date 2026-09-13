"""Apply the approved audited FAU/Texas V2 Favorite correction once."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import pandas as pd

from performance_ledger import update_performance_ledger


FAU_ID = "34603696"
TEXAS_ID = "34226011"
FAU_MARKET = "SPREAD"
FAU_SIDE = "Florida Atlantic +4"
CORRECTION_TAG = "audited_fau_texas_v2_correction_20260912"


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() and path.stat().st_size else pd.DataFrame()


def write(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name("." + path.name + ".correction.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def backup(data_dir: Path, audit_dir: Path, names: list[str]) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = data_dir / name
        if source.exists() and not (audit_dir / name).exists():
            shutil.copy2(source, audit_dir / name)


def fau_row(data_dir: Path, live_columns: list[str]) -> dict:
    tracking = read(data_dir / "red_fox_favorite_tracking.csv")
    qualified = tracking[
        tracking.game_id.eq(FAU_ID)
        & tracking.market_display.eq(FAU_MARKET)
        & tracking.favorite_state.eq("qualified")
    ].copy()
    if qualified.empty:
        raise RuntimeError("FAU qualified tracking record is missing")
    qualified["_at"] = pd.to_datetime(qualified.recorded_at, errors="coerce", utc=True)
    tracked = qualified.sort_values("_at").iloc[-1]
    final_qualified = pd.to_datetime(tracked.final_qualified_at, errors="coerce", utc=True)

    snapshots = read(data_dir / "snapshots.csv")
    snapshots["_seen"] = pd.to_datetime(snapshots.timestamp, errors="coerce", utc=True)
    game = snapshots[snapshots.game_id.eq(FAU_ID)].copy()
    pregame = game[game._seen.notna() & (game._seen <= final_qualified)].copy()
    if pregame.empty:
        raise RuntimeError("FAU pregame snapshots are missing")
    state_at = pregame._seen.max()
    state = pregame[pregame._seen.eq(state_at)]
    spread = state[state.side.str.contains(r"(?:^| )[-+]\d", regex=True)].copy()
    if len(spread) != 2 or FAU_SIDE not in set(spread.side):
        raise RuntimeError("FAU final qualified spread pair is incomplete")
    latest_kickoff = pd.to_datetime(game.dk_start_iso, errors="coerce", utc=True).dropna().max()
    if pd.isna(latest_kickoff) or state_at > latest_kickoff:
        raise RuntimeError("FAU source state is not provably pregame")

    sides = []
    for _, source in spread.iterrows():
        supported = source.side == FAU_SIDE
        sides.append({
            "flagged_side": source.side,
            "bets_pct": float(source.bets_pct),
            "money_pct": float(source.money_pct),
            "open_line": source.open_line,
            "current_line": source.current_line,
            "reaction": "Contrarian" if supported else "Freeze",
            "path": "One-Way",
            "response_direction": "TOWARD" if supported else "AGAINST",
            "data_badge": "Clean",
            "kpi_eligible": supported,
            "action_type": "CONTRARIAN CANDIDATE" if supported else "FADE CANDIDATE",
            "action_side": FAU_SIDE if supported else "",
            "anomaly_chips": "Contrarian | One-Way | Market Move" if supported else "Freeze | One-Way | Public Pressure | Market Move",
        })
    evidence = tracked.supporting_evidence or json.dumps({
        "reaction": "Contrarian", "path": "One-Way", "bets_pct": 20.0,
        "money_pct": 19.0, "open_line": "+6.5 (-105)",
        "current_line": "+4 (-108)", "cross_market": "neutral",
    }, separators=(",", ":"))
    row = {column: "" for column in live_columns}
    row.update({
        "sport": "ncaaf", "game_id": FAU_ID, "canonical_key": f"ncaaf|{FAU_ID}",
        "kickoff_iso": latest_kickoff.isoformat(), "kickoff_sort": latest_kickoff.isoformat(),
        "game": "Navy @ Florida Atlantic", "market_display": FAU_MARKET,
        "flagged_side": FAU_SIDE, "focus_basis": "Low-support side; line moved toward it",
        "action_side": FAU_SIDE, "action_line": "Florida Atlantic +4 @ -108",
        "action_type": "CONTRARIAN CANDIDATE", "action_basis": "Low-support side received a sustained move toward it.",
        "kpi_eligible": "True", "reaction": "Contrarian", "path": "One-Way",
        "anomaly_chips": "Contrarian | One-Way | Market Move", "bets_pct": "20", "money_pct": "19",
        "open_line": "+6.5 (-105)", "current_line": "+4 (-108)",
        "reason": "Lower-supported side received a meaningful, intact move toward it.",
        "data_badge": "Clean", "market_sides": json.dumps(sides, separators=(",", ":")),
        "read_anchor_side": FAU_SIDE, "supported_side": FAU_SIDE,
        "directional_lean_side": FAU_SIDE,
        "market_rationale": "With only 20% bets / 19% money, Florida Atlantic moved +6.5 (-105) to +4 (-108).",
        "red_fox_favorite": "true", "favorite_side": FAU_SIDE,
        "favorite_pathway": tracked.favorite_pathway,
        "favorite_rule_version": "red_fox_favorite_v2",
        "favorite_first_qualified_at": tracked.first_qualified_at,
        "favorite_final_qualified_at": tracked.final_qualified_at,
        "favorite_state": "qualified",
        "favorite_final_market_read": tracked.final_pregame_market_read,
        "favorite_final_market_rank": tracked.final_pregame_rank,
        "favorite_supporting_evidence": evidence,
        "favorite_whipsaw_state": tracked.whipsaw_state,
        "favorite_cross_market_state": tracked.cross_market_state,
        "favorite_snapshot_id": tracked.snapshot_id,
        "favorite_reason": "Audited recovery of the last valid published V2 Favorite state.",
        "state_as_of_utc": state_at.isoformat(),
        "final_pregame_state_at_utc": state_at.isoformat(),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_method": "manual_audited_recovery_last_qualified_pregame_state",
        "score_away": "-", "score_home": "-", "score_state": "unknown",
        "score_status": "Score pending", "score_match_state": "unmatched",
        "candidate_recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return row


def apply(data_dir: Path) -> dict:
    audit_dir = Path("audit") / "favorite_correction_20260912"
    names = [
        "live_recent.csv", "performance_ledger.csv", "performance_favorites.csv",
        "favorite_performance.json", "red_fox_favorite_tracking.csv",
        "red_fox_favorite_freeze_candidates.csv",
    ]
    backup(data_dir, audit_dir, names)
    live_path = data_dir / "live_recent.csv"
    live = read(live_path)
    if live.empty:
        raise RuntimeError("live_recent.csv is missing")
    texas = live.game_id.eq(TEXAS_ID)
    if texas.any():
        live.loc[texas, "red_fox_favorite"] = "false"
        live.loc[texas, "favorite_state"] = "not_qualified"
        live.loc[texas, "favorite_reason"] = "Reviewed negative-money football Favorite gate not satisfied."
    fau = fau_row(data_dir, list(live.columns))
    fau_mask = live.game_id.eq(FAU_ID) & live.market_display.eq(FAU_MARKET)
    live = live.loc[~fau_mask].copy()
    live = pd.concat([live, pd.DataFrame([fau])], ignore_index=True, sort=False)
    write(live, live_path)

    candidates_path = data_dir / "red_fox_favorite_freeze_candidates.csv"
    candidates = read(candidates_path)
    if candidates.empty:
        candidates = pd.DataFrame(columns=live.columns)
    fau_candidate = {column: fau.get(column, "") for column in set(candidates.columns) | set(fau)}
    if "candidate_recorded_at_utc" not in candidates:
        candidates["candidate_recorded_at_utc"] = ""
    candidate_mask = candidates.get("game_id", pd.Series("", index=candidates.index)).eq(FAU_ID) & candidates.get("market_display", pd.Series("", index=candidates.index)).eq(FAU_MARKET)
    candidates = candidates.loc[~candidate_mask].copy()
    candidates = pd.concat([candidates, pd.DataFrame([fau_candidate])], ignore_index=True, sort=False)
    write(candidates, candidates_path)

    ledger_path = data_dir / "performance_ledger.csv"
    ledger = read(ledger_path)
    texas_ledger = ledger.event_id.eq(TEXAS_ID) & ledger.market.eq("MONEYLINE")
    if texas_ledger.any():
        ledger.loc[texas_ledger, "favorite_qualified"] = "no"
        ledger.loc[texas_ledger, "freeze_method"] = ledger.loc[texas_ledger, "freeze_method"].map(
            lambda value: f"{value}|{CORRECTION_TAG}" if CORRECTION_TAG not in value else value
        )
    write(ledger, ledger_path)

    tracking_path = data_dir / "red_fox_favorite_tracking.csv"
    tracking = read(tracking_path)
    already = tracking.game_id.eq(TEXAS_ID) & tracking.disappearance_reason.str.contains(CORRECTION_TAG, regex=False)
    if not already.any():
        prior = tracking[tracking.game_id.eq(TEXAS_ID) & tracking.market_display.eq("MONEYLINE")].iloc[-1].to_dict()
        prior.update({
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "favorite_state": "not_qualified",
            "disappearance_reason": f"{CORRECTION_TAG}: reviewed football favorite gate not satisfied",
        })
        tracking = pd.concat([tracking, pd.DataFrame([prior])], ignore_index=True, sort=False)
        write(tracking, tracking_path)

    result = update_performance_ledger(data_dir)
    manifest = {
        "correction": CORRECTION_TAG, "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "fau": {"game_id": FAU_ID, "favorite": FAU_SIDE, "restored": True},
        "texas": {"game_id": TEXAS_ID, "favorite_removed": True, "audit_history_retained": True},
        "performance": result,
    }
    (audit_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(apply(Path("data")), indent=2))
