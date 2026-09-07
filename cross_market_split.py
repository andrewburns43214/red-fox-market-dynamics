"""Annotate trustworthy Spread/Moneyline support disagreement as context only.

Cross-Market Split is deliberately separate from the stricter pricing-integrity
evaluation. It consumes the same resolved read anchor that drives the board's
green supported-side treatment and never changes reads, ranks, scoring, or
Favorite state.
"""

from __future__ import annotations

import json
import re

import pandas as pd


CROSS_MARKET_SPLIT_COLUMNS = [
    "cross_market_split",
    "cross_market_split_state",
    "cross_market_split_explanation",
    "cross_market_split_spread_team",
    "cross_market_split_moneyline_team",
]


def apply_cross_market_split(board: pd.DataFrame) -> pd.DataFrame:
    """Persist valid supported-side disagreement on Spread/Moneyline rows.

    ``read_anchor_side`` is the resolved state used for the board's green row.
    It may be valid even when the primary directional classification is Watch.
    Both anchors must map unambiguously to clean, reliable market sides.
    Confirmed Cross-Market Mismatch always suppresses Split.
    """
    if board is None:
        return pd.DataFrame(columns=CROSS_MARKET_SPLIT_COLUMNS)
    result = board.copy()
    for column in CROSS_MARKET_SPLIT_COLUMNS:
        result[column] = ""
    result["cross_market_split"] = "false"
    if result.empty or not {"sport", "game_id", "market_display"}.issubset(result.columns):
        return result

    for _, indexes in result.groupby(["sport", "game_id"], dropna=False, sort=False).groups.items():
        game_rows = result.loc[indexes]
        market_names = game_rows["market_display"].astype(str).str.upper()
        spread_rows = game_rows.loc[market_names.eq("SPREAD")]
        moneyline_rows = game_rows.loc[market_names.eq("MONEYLINE")]
        if len(spread_rows) != 1 or len(moneyline_rows) != 1:
            continue

        pair = pd.concat([spread_rows, moneyline_rows])
        if pair.get("cross_market_mismatch", pd.Series("false", index=pair.index)).map(_is_true).any():
            continue

        spread_side = _resolved_supported_side(spread_rows.iloc[0])
        moneyline_side = _resolved_supported_side(moneyline_rows.iloc[0])
        spread_identity = _team_identity(spread_side)
        moneyline_identity = _team_identity(moneyline_side)
        if not spread_identity or not moneyline_identity or spread_identity == moneyline_identity:
            continue

        spread_team = _team_label(spread_side)
        moneyline_team = _team_label(moneyline_side)
        pair_mask = result.index.isin(pair.index)
        values = {
            "cross_market_split": "true",
            "cross_market_split_state": "confirmed_reads",
            "cross_market_split_explanation": (
                f"The Spread Market Read supports {spread_team}, while the Moneyline Market Read supports {moneyline_team}."
            ),
            "cross_market_split_spread_team": spread_team,
            "cross_market_split_moneyline_team": moneyline_team,
        }
        for column, value in values.items():
            result.loc[pair_mask, column] = value
    return result


def _resolved_supported_side(row: pd.Series) -> str:
    """Return the trustworthy board-highlight anchor, or an empty string.

    Requiring a unique match against both clean market sides prevents Split
    from inferring support from public percentages, chip presence, or a stale
    or ambiguous label.
    """
    anchor = _text(row.get("read_anchor_side", ""))
    anchor_identity = _team_identity(anchor)
    if not anchor_identity:
        return ""

    raw_sides = row.get("market_sides", "")
    try:
        sides = json.loads(raw_sides) if isinstance(raw_sides, str) else raw_sides
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    if not isinstance(sides, list) or len(sides) != 2:
        return ""

    identities: list[str] = []
    labels: dict[str, str] = {}
    unreliable_context = {"market lag", "feed risk", "split risk", "split cap"}
    for side in sides:
        if not isinstance(side, dict) or _text(side.get("data_badge", "")).lower() != "clean":
            return ""
        context = side.get("context_chips", "")
        chips = context if isinstance(context, list) else _text(context).split("|")
        if any(_text(chip).lower() in unreliable_context for chip in chips):
            return ""
        label = _team_label(side.get("flagged_side", ""))
        identity = _team_identity(label)
        if not identity:
            return ""
        identities.append(identity)
        labels[identity] = label

    if len(set(identities)) != 2 or identities.count(anchor_identity) != 1:
        return ""
    return labels[anchor_identity]


def _is_true(value: object) -> bool:
    return _text(value).lower() == "true"


def _team_identity(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", _team_label(value).lower())


def _team_label(value: object) -> str:
    return re.sub(r"\s[+-]\d+(?:\.\d+)?(?:\s.*)?$", "", _text(value))


def _text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()
