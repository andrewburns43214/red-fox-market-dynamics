"""Audit the published board's directional-side contract without mutating data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from anomaly_board import _market_read_semantics
from cross_market_split import apply_cross_market_split


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _identity(value: object, market: object) -> str:
    text = _text(value)
    if _text(market).upper() == "TOTAL":
        return "under" if re.match(r"^(?:under|u\b)", text, re.I) else "over" if text else ""
    team = re.sub(r"\s[+-]\d+(?:\.\d+)?(?:\s.*)?$", "", text)
    return re.sub(r"[^a-z0-9]+", "", team.lower())


def audit(board: pd.DataFrame) -> dict[str, object]:
    before = board.copy()
    after = board.copy()
    resolved = after.apply(_market_read_semantics, axis=1, result_type="expand")
    after[["read_anchor_side", "supported_side"]] = resolved
    after["directional_lean_side"] = after["supported_side"]
    after = apply_cross_market_split(after)

    old_side = before.apply(lambda row: _identity(row.get("read_anchor_side", ""), row.get("market_display", "")), axis=1)
    new_side = after.apply(lambda row: _identity(row.get("supported_side", ""), row.get("market_display", "")), axis=1)
    changed = old_side.ne(new_side)
    directional_to_neutral = old_side.ne("") & new_side.eq("")
    team_to_team = old_side.ne("") & new_side.ne("") & old_side.ne(new_side)

    anomalies: list[dict[str, str]] = []
    for _, row in after.iterrows():
        market = row.get("market_display", "")
        supported = _text(row.get("supported_side", ""))
        try:
            sides = json.loads(_text(row.get("market_sides", "")))
        except (TypeError, ValueError, json.JSONDecodeError):
            sides = []
        identities = [_identity(side.get("flagged_side", ""), market) for side in sides if isinstance(side, dict)]
        if supported and identities.count(_identity(supported, market)) != 1:
            anomalies.append({"game": _text(row.get("game")), "market": _text(market), "issue": "supported side does not uniquely match market_sides"})
        if not supported and _text(row.get("cross_market_split", "")).lower() == "true":
            anomalies.append({"game": _text(row.get("game")), "market": _text(market), "issue": "neutral market contributes to Cross-Market Split"})

    remaining = []
    split_rows = after.loc[after["cross_market_split"].astype(str).str.lower().eq("true")]
    for (_, game_id), rows in split_rows.groupby(["sport", "game_id"], sort=False):
        spread = rows.loc[rows.market_display.astype(str).str.upper().eq("SPREAD")]
        moneyline = rows.loc[rows.market_display.astype(str).str.upper().eq("MONEYLINE")]
        if len(spread) == 1 and len(moneyline) == 1:
            remaining.append({
                "game": _text(rows.iloc[0].get("game")),
                "spread_supported_side": _text(spread.iloc[0].get("supported_side")),
                "moneyline_supported_side": _text(moneyline.iloc[0].get("supported_side")),
            })

    before_pairs = before.loc[before.get("cross_market_split", pd.Series(index=before.index, dtype=str)).astype(str).str.lower().eq("true"), ["sport", "game_id"]].drop_duplicates()
    changed_rows = after.loc[changed, [column for column in ["game", "market_display", "read_anchor_side", "supported_side"] if column in after.columns]]
    return {
        "published_markets": int(len(after)),
        "confirmed_directional_markets": int(new_side.ne("").sum()),
        "neutral_markets": int(new_side.eq("").sum()),
        "visible_supported_side_changed": int(changed.sum()),
        "directional_to_neutral": int(directional_to_neutral.sum()),
        "team_to_team": int(team_to_team.sum()),
        "cross_market_splits_before": int(len(before_pairs)),
        "cross_market_splits_after": int(len(remaining)),
        "remaining_cross_market_splits": remaining,
        "anomalies": anomalies,
        "changed_rows": changed_rows.fillna("").to_dict("records"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("board", type=Path)
    parser.add_argument("--changes", action="store_true", help="include every changed market row")
    args = parser.parse_args()
    result = audit(pd.read_csv(args.board, keep_default_na=False))
    if not args.changes:
        result.pop("changed_rows", None)
    print(json.dumps(result, indent=2))
    return 1 if result["anomalies"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
