"""Annotate confirmed Spread/Moneyline read disagreement as context only.

Cross-Market Split is deliberately separate from the stricter pricing-integrity
evaluation. It consumes the resolved directional lean already produced by the
Market Read engine and never changes reads, ranks, scoring, or Favorite state.
"""

from __future__ import annotations

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
    """Persist confirmed read disagreement on Spread and Moneyline rows.

    ``directional_lean_side`` is the engine's resolved directional output. A
    read anchor without a directional lean (including Watch or descriptive
    resistance context) is intentionally not enough to create this state.
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

        spread_side = _text(spread_rows.iloc[0].get("directional_lean_side", ""))
        moneyline_side = _text(moneyline_rows.iloc[0].get("directional_lean_side", ""))
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


def _is_true(value: object) -> bool:
    return _text(value).lower() == "true"


def _team_identity(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", _team_label(value).lower())


def _team_label(value: object) -> str:
    return re.sub(r"\s[+-]\d+(?:\.\d+)?(?:\s.*)?$", "", _text(value))


def _text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()
