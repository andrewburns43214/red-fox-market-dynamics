"""
Layer Merge for Red Fox engine.

Joins L1 (sharp), L2 (consensus), and situational data onto the L3 (DK) DataFrame.
Sets layer_mode (L123/L13/L23/L3_ONLY) per row and determines available features.
"""
import pandas as pd

from canonical_match import build_canonical_key_from_dk
from team_aliases import normalize_team_name
from l1_features import compute_l1_features
from l2_features import compute_l2_features
from espn_situational import fetch_all_situational


# Default feature values when a layer is unavailable
L1_DEFAULTS = {
    "l1_move_dir": 0,
    "l1_move_magnitude": 0.0,
    "l1_move_magnitude_raw": 0.0,
    "l1_sharp_agreement": 0,
    "l1_agreement_mult": 1.0,
    "l1_limit_weighted_dir": 0.0,
    "l1_leader_book": "",
    "l1_move_speed": 0.0,
    "l1_speed_label": "",
    "l1_stability": 0.0,
    "l1_key_number_cross": False,
    "l1_sharp_strength": 0.0,
    "l1_limit_confidence": 0.0,
    "l1_n_books": 0,
    "l1_books": [],
    "l1_available": False,
}

L2_DEFAULTS = {
    "l2_n_books": 0,
    "l2_consensus_line": 0.0,
    "l2_consensus_odds": 0,
    "l2_dispersion": 0.0,
    "l2_dispersion_label": "",
    "l2_dispersion_mult": 1.0,
    "l2_dispersion_trend": "",
    "l2_pinn_vs_consensus": 0.0,
    "l2_pinn_line": 0.0,
    "l2_consensus_agreement": 0.0,
    "l2_stale_price_flag": False,
    "l2_stale_price_gap": 0.0,
    "l2_validation_strength": 0.0,
    "l2_available": False,
}

SITUATIONAL_DEFAULTS = {
    "home_injuries": [],
    "away_injuries": [],
    "home_injury_count": 0,
    "away_injury_count": 0,
    "home_rest_days": 1,
    "away_rest_days": 1,
    "b2b_flag": "",
    "pitcher_matchup": None,
}


def _determine_layer_mode(l1_available: bool, l2_available: bool) -> str:
    """Determine layer mode based on data availability."""
    if l1_available and l2_available:
        return "L123"
    elif l1_available:
        return "L13"
    elif l2_available:
        return "L23"
    else:
        return "L3_ONLY"


def _detect_stale_price(dk_line, consensus_line: float, market: str) -> tuple:
    """
    Detect if DK line is stale compared to consensus.

    Returns (is_stale: bool, gap: float).
    A gap > 1 point for spreads or > 1.5 for totals indicates stale pricing.
    """
    try:
        dk_val = float(dk_line) if dk_line not in ("", None) else None
    except (ValueError, TypeError):
        dk_val = None

    if dk_val is None or consensus_line == 0.0:
        return (False, 0.0)

    gap = abs(dk_val - consensus_line)

    if market == "TOTAL":
        threshold = 1.5
    else:
        threshold = 1.0

    return (gap >= threshold, round(gap, 2))


def merge_all_layers(dk_df: pd.DataFrame, sport: str = None) -> pd.DataFrame:
    """
    Merge L1, L2, and situational data onto the DK DataFrame.

    This is the main entry point called by build_dashboard().

    Args:
        dk_df: DataFrame from DK scraper with at least:
            - 'game' or 'canonical_key' column
            - 'market_key' or 'market' column
            - 'side' column
            - 'home_team_norm', 'away_team_norm' columns
        sport: Sport key for filtering

    Returns:
        dk_df with added columns:
            - All l1_* features
            - All l2_* features
            - layer_mode
            - Situational flags
    """
    if dk_df is None or dk_df.empty:
        return dk_df

    # Compute L1 features
    l1_features = compute_l1_features(sport=sport)

    # Compute L2 features (needs L1 for agreement calculation)
    l2_features = compute_l2_features(sport=sport, l1_features=l1_features)

    # Fetch situational data
    situational = None
    if sport:
        try:
            situational = fetch_all_situational(sport)
        except Exception:
            situational = None

    # Ensure canonical_key column exists
    if "canonical_key" not in dk_df.columns:
        # Try to build from game column
        if "game" in dk_df.columns:
            dk_df["canonical_key"] = dk_df.apply(
                lambda r: build_canonical_key_from_dk(
                    r["game"],
                    r.get("sport", sport or ""),
                    r.get("dk_start_iso", ""),
                ) if pd.notna(r.get("game")) else "",
                axis=1,
            )
        else:
            dk_df["canonical_key"] = ""

    # Ensure market column exists
    if "market" not in dk_df.columns and "market_key" in dk_df.columns:
        dk_df["market"] = dk_df["market_key"]

    # Normalize side for matching
    if "side" in dk_df.columns:
        dk_df["_side_norm"] = dk_df.apply(
            lambda r: r["side"].lower() if str(r.get("market", "")) == "TOTAL"
            else normalize_team_name(str(r.get("side", ""))),
            axis=1,
        )
    else:
        dk_df["_side_norm"] = ""

    # Join L1 features
    for col, default in L1_DEFAULTS.items():
        if isinstance(default, (list, dict)):
            dk_df[col] = [default.copy() if isinstance(default, (list, dict)) else default for _ in range(len(dk_df))]
        else:
            dk_df[col] = default

    for idx, row in dk_df.iterrows():
        canon = row.get("canonical_key", "")
        market = row.get("market", "")
        side = row.get("_side_norm", "")

        key = (canon, market, side)
        l1_feat = l1_features.get(key)

        if l1_feat:
            for col, val in l1_feat.items():
                dk_df.at[idx, col] = val

    # Join L2 features
    for col, default in L2_DEFAULTS.items():
        if isinstance(default, (list, dict)):
            dk_df[col] = [default.copy() for _ in range(len(dk_df))]
        else:
            dk_df[col] = default

    for idx, row in dk_df.iterrows():
        canon = row.get("canonical_key", "")
        market = row.get("market", "")
        side = row.get("_side_norm", "")

        key = (canon, market, side)
        l2_feat = l2_features.get(key)

        if l2_feat:
            for col, val in l2_feat.items():
                dk_df.at[idx, col] = val

            # Stale price detection (now that we have DK line + consensus)
            dk_line = row.get("line", row.get("dk_line", ""))
            consensus_line = l2_feat.get("l2_consensus_line", 0.0)
            is_stale, stale_gap = _detect_stale_price(dk_line, consensus_line, market)
            dk_df.at[idx, "l2_stale_price_flag"] = is_stale
            dk_df.at[idx, "l2_stale_price_gap"] = stale_gap

    # Set layer mode
    dk_df["layer_mode"] = dk_df.apply(
        lambda r: _determine_layer_mode(r.get("l1_available", False), r.get("l2_available", False)),
        axis=1,
    )

    # Join situational data
    for col, default in SITUATIONAL_DEFAULTS.items():
        dk_df[col] = default if not isinstance(default, list) else [default.copy() for _ in range(len(dk_df))]

    if situational:
        injuries = situational.get("injuries", {})
        rest_days = situational.get("rest_days", {})
        pitchers = situational.get("pitchers", {})

        for idx, row in dk_df.iterrows():
            home = row.get("home_team_norm", "")
            away = row.get("away_team_norm", "")

            # Injuries
            home_inj = injuries.get(home, [])
            away_inj = injuries.get(away, [])
            dk_df.at[idx, "home_injuries"] = home_inj
            dk_df.at[idx, "away_injuries"] = away_inj
            dk_df.at[idx, "home_injury_count"] = len(home_inj)
            dk_df.at[idx, "away_injury_count"] = len(away_inj)

            # Rest days
            dk_df.at[idx, "home_rest_days"] = rest_days.get(home, 1)
            dk_df.at[idx, "away_rest_days"] = rest_days.get(away, 1)

            # B2B flag
            home_b2b = rest_days.get(home, 1) == 0
            away_b2b = rest_days.get(away, 1) == 0
            if home_b2b and away_b2b:
                dk_df.at[idx, "b2b_flag"] = "BOTH_B2B"
            elif home_b2b:
                dk_df.at[idx, "b2b_flag"] = "HOME_B2B"
            elif away_b2b:
                dk_df.at[idx, "b2b_flag"] = "AWAY_B2B"

            # Pitchers (MLB)
            if pitchers:
                match_key = f"{away} @ {home}"
                pitcher_info = pitchers.get(match_key)
                if not pitcher_info:
                    # Try fuzzy match
                    for pk, pv in pitchers.items():
                        if pv.get("home_team_norm") == home and pv.get("away_team_norm") == away:
                            pitcher_info = pv
                            break
                if pitcher_info:
                    dk_df.at[idx, "pitcher_matchup"] = pitcher_info

    # Clean up temp column
    if "_side_norm" in dk_df.columns:
        dk_df.drop(columns=["_side_norm"], inplace=True)

    return dk_df
