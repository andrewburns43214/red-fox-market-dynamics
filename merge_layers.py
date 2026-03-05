"""
Layer Merge for Red Fox engine.

Joins L1 (sharp), L2 (consensus), and situational data onto the L3 (DK) DataFrame.
Sets layer_mode (L123/L13/L23/L3_ONLY) per row and determines available features.

Includes fuzzy fallback matching when exact canonical keys don't match,
and match rate monitoring for visibility into data pipeline health.
"""
import re
import pandas as pd

from canonical_match import build_canonical_key_from_dk, fuzzy_match_key
from team_aliases import normalize_team_name
from l1_features import compute_l1_features
from l2_features import compute_l2_features
from espn_situational import fetch_all_situational, fetch_ncaab_rankings
from weather import get_weather_for_game
from mlb_context import get_mlb_context
from nhl_context import get_nhl_context


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
    "sharp_cert_tier": "NONE",
    "sharp_cert_strength": 0.0,
    "sharp_cert_detail": "",
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
    "home_injury_count": 0,
    "away_injury_count": 0,
    "home_rest_days": 1,
    "away_rest_days": 1,
    "b2b_flag": "",
    "wind_mph": 0,
    "temp_f": 70,
    "precip_prob": 0,
    "weather_flag": "",
    "weather_adj": 0.0,
    # Sport-specific context
    "sport_context_adj": 0.0,
    "sport_context_flag": "",
    # MLB
    "sp_era_home": None,
    "sp_era_away": None,
    "sp_name_home": "",
    "sp_name_away": "",
    "sp_hand_home": "",
    "sp_hand_away": "",
    "sp_flag_home": "",
    "sp_flag_away": "",
    "park_factor": 1.0,
    # NHL
    "goalie_home": "",
    "goalie_away": "",
    "goalie_flag_home": "",
    "goalie_flag_away": "",
    # NCAAB
    "rank_home": 0,
    "rank_away": 0,
}


def _infer_market_type(row) -> str:
    """
    Infer market type from DK side column.
    DK stores all markets as 'splits' with the type embedded in the side:
      "Over 221.5" / "Under 221.5" -> TOTAL
      "OKC Thunder -4.5"           -> SPREAD
      "NY Knicks"                  -> MONEYLINE
    """
    side = str(row.get("side", "")).strip()
    existing = str(row.get("market", "")).strip().upper()
    # If already properly tagged, keep it
    if existing in ("SPREAD", "MONEYLINE", "TOTAL"):
        return existing
    if not side:
        return existing
    side_lower = side.lower()
    if side_lower.startswith("over ") or side_lower.startswith("under "):
        return "TOTAL"
    # Check for line number (e.g., "-4.5", "+7.5")
    if re.search(r'[+-]\d+\.?\d*$', side):
        return "SPREAD"
    return "MONEYLINE"


def _clean_side_for_matching(side: str, market: str, sport: str = "") -> str:
    """
    Clean DK side for matching against L1/L2 keys.
    Strips line numbers and normalizes team names (with sport for mascot stripping).
      "OKC Thunder -4.5" -> "oklahoma city thunder"
      "Over 221.5"       -> "over"
      "NY Knicks"        -> "new york knicks"
    """
    side = side.strip()
    if not side:
        return ""
    if market == "TOTAL":
        # "Over 221.5" -> "over", "Under 142.5" -> "under"
        return side.split()[0].lower() if side else ""
    # Strip trailing line number: "OKC Thunder -4.5" -> "OKC Thunder"
    cleaned = re.sub(r'\s*[+-]\d+\.?\d*$', '', side).strip()
    return normalize_team_name(cleaned, sport=sport)


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


def _fuzzy_find_feature(canon_key: str, market: str, side: str,
                         feature_dict: dict, sport: str) -> dict | None:
    """
    Fuzzy fallback: when exact key match fails, try fuzzy matching
    against all keys in the feature dict.

    Returns the best-matching feature dict, or None.
    """
    if not canon_key or not feature_dict:
        return None

    best_score = 0.0
    best_feat = None

    for (feat_canon, feat_market, feat_side), feat_data in feature_dict.items():
        # Market and side must match exactly
        if feat_market != market or feat_side != side:
            continue

        # Fuzzy match on canonical key
        score = fuzzy_match_key(canon_key, feat_canon)
        if score > best_score and score >= 0.8:
            best_score = score
            best_feat = feat_data

    return best_feat


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

    sport_lower = (sport or "").lower()

    # Compute L1 features
    l1_features = compute_l1_features(sport=sport)

    # Compute L2 features (needs L1 for agreement calculation)
    l2_features = compute_l2_features(sport=sport, l1_features=l1_features)

    # Fetch situational data
    situational = None
    if sport:
        try:
            situational = fetch_all_situational(sport)
        except Exception as e:
            print(f"[WARN] situational fetch for {sport}: {repr(e)}")
            situational = None

    # Ensure canonical_key column exists and is populated
    # If canonical_key is missing or empty, try to build from game column
    def _build_canon(r):
        if pd.notna(r.get("game")) and str(r.get("game", "")).strip():
            game_str = str(r["game"])
            row_sport = r.get("sport", sport or "")
            # Use dk_start_iso, or fall back to snapshot timestamp for date
            raw_start = r.get("dk_start_iso", "")
            start_iso = str(raw_start).strip() if pd.notna(raw_start) and raw_start else ""
            if not start_iso:
                # Try snapshot timestamp as date fallback
                raw_ts = r.get("timestamp", "")
                start_iso = str(raw_ts).strip() if pd.notna(raw_ts) and raw_ts else ""
            if not start_iso:
                # Last resort: use today's date
                from datetime import datetime, timezone
                start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            return build_canonical_key_from_dk(game_str, row_sport, start_iso)
        return ""

    if "canonical_key" not in dk_df.columns:
        if "game" in dk_df.columns:
            dk_df["canonical_key"] = dk_df.apply(_build_canon, axis=1)
        else:
            dk_df["canonical_key"] = ""
    else:
        # Fill in any empty canonical keys
        empty_mask = dk_df["canonical_key"].isna() | (dk_df["canonical_key"] == "")
        if empty_mask.any() and "game" in dk_df.columns:
            dk_df.loc[empty_mask, "canonical_key"] = dk_df.loc[empty_mask].apply(_build_canon, axis=1)

    # Ensure market column has standard values (SPREAD/MONEYLINE/TOTAL)
    # DK snapshots store market as "splits" with market type embedded in side column
    if "market" not in dk_df.columns and "market_key" in dk_df.columns:
        dk_df["market"] = dk_df["market_key"]
    if "side" in dk_df.columns:
        dk_df["market"] = dk_df.apply(lambda r: _infer_market_type(r), axis=1)

    # Normalize side for matching: strip line numbers, normalize team name
    if "side" in dk_df.columns:
        dk_df["_side_norm"] = dk_df.apply(
            lambda r: _clean_side_for_matching(
                str(r.get("side", "")),
                str(r.get("market", "")),
                sport=sport_lower,
            ),
            axis=1,
        )
    else:
        dk_df["_side_norm"] = ""

    # ─── Match monitoring counters ───
    l1_exact = 0
    l1_fuzzy = 0
    l1_miss = 0
    l2_exact = 0
    l2_fuzzy = 0
    l2_miss = 0
    total_rows = len(dk_df)

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
            l1_exact += 1
        elif l1_features and canon:
            # Fuzzy fallback
            l1_feat = _fuzzy_find_feature(canon, market, side, l1_features, sport_lower)
            if l1_feat:
                l1_fuzzy += 1

        if not l1_feat:
            l1_miss += 1

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
            l2_exact += 1
        elif l2_features and canon:
            # Fuzzy fallback
            l2_feat = _fuzzy_find_feature(canon, market, side, l2_features, sport_lower)
            if l2_feat:
                l2_fuzzy += 1

        if not l2_feat:
            l2_miss += 1

        if l2_feat:
            for col, val in l2_feat.items():
                dk_df.at[idx, col] = val

            # Stale price detection (now that we have DK line + consensus)
            dk_line = row.get("line", row.get("dk_line", ""))
            consensus_line = l2_feat.get("l2_consensus_line", 0.0)
            is_stale, stale_gap = _detect_stale_price(dk_line, consensus_line, market)
            dk_df.at[idx, "l2_stale_price_flag"] = is_stale
            dk_df.at[idx, "l2_stale_price_gap"] = stale_gap

    # ─── Log match rates ───
    l1_total_available = l1_exact + l1_fuzzy
    l2_total_available = l2_exact + l2_fuzzy
    if l1_features:
        print(f"  L1 merge: {l1_exact} exact + {l1_fuzzy} fuzzy = {l1_total_available} matched, {l1_miss} unmatched ({total_rows} DK rows)")
    if l2_features:
        print(f"  L2 merge: {l2_exact} exact + {l2_fuzzy} fuzzy = {l2_total_available} matched, {l2_miss} unmatched ({total_rows} DK rows)")

    # Set layer mode
    dk_df["layer_mode"] = dk_df.apply(
        lambda r: _determine_layer_mode(r.get("l1_available", False), r.get("l2_available", False)),
        axis=1,
    )

    # ─── Extract home/away team names from "game" column ───
    if "home_team_norm" not in dk_df.columns and "game" in dk_df.columns:
        def _parse_teams(game_str):
            """Parse 'AWAY @ HOME' format, return (home_norm, away_norm)."""
            g = str(game_str) if pd.notna(game_str) else ""
            if " @ " in g:
                parts = g.split(" @ ", 1)
                return normalize_team_name(parts[1].strip()), normalize_team_name(parts[0].strip())
            return "", ""
        _teams = dk_df["game"].apply(_parse_teams)
        dk_df["home_team_norm"] = _teams.apply(lambda t: t[0])
        dk_df["away_team_norm"] = _teams.apply(lambda t: t[1])

    # Join situational data
    for col, default in SITUATIONAL_DEFAULTS.items():
        dk_df[col] = default if not isinstance(default, list) else [default.copy() for _ in range(len(dk_df))]

    # Lookup dicts for dict-valued data (can't store dicts in DataFrame cells)
    _pitcher_lookup = {}   # idx -> pitcher_info dict
    _goalie_lookup = {}    # idx -> goalie_info dict

    if situational:
        injuries = situational.get("injuries", {})
        rest_days = situational.get("rest_days", {})
        pitchers = situational.get("pitchers", {})

        for idx, row in dk_df.iterrows():
            home = row.get("home_team_norm", "")
            away = row.get("away_team_norm", "")

            # Injuries
            dk_df.at[idx, "home_injury_count"] = len(injuries.get(home, []))
            dk_df.at[idx, "away_injury_count"] = len(injuries.get(away, []))

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

            # Pitchers (MLB) — store in lookup dict, not DataFrame
            if pitchers:
                match_key = f"{away} @ {home}"
                pitcher_info = pitchers.get(match_key)
                if not pitcher_info:
                    for pk, pv in pitchers.items():
                        if pv.get("home_team_norm") == home and pv.get("away_team_norm") == away:
                            pitcher_info = pv
                            break
                if pitcher_info:
                    _pitcher_lookup[idx] = pitcher_info

    # ─── Weather data (outdoor sports only) ───
    _wx_cache = {}  # cache per (home_team, game_time) to avoid duplicate API calls
    for idx, row in dk_df.iterrows():
        _sport = str(row.get("sport", "")).lower()
        if _sport not in ("nfl", "ncaaf", "mlb"):
            continue
        home = row.get("home_team_norm", "")
        game_time = str(row.get("dk_start_iso", ""))
        if not home or not game_time:
            continue
        _wx_key = (home, game_time)
        if _wx_key not in _wx_cache:
            try:
                _wx_cache[_wx_key] = get_weather_for_game(_sport, home, game_time)
            except Exception as e:
                print(f"[WARN] weather fetch for {_wx_key}: {repr(e)}")
                _wx_cache[_wx_key] = {}
        wx = _wx_cache[_wx_key]
        for col in ("wind_mph", "temp_f", "precip_prob", "weather_flag", "weather_adj"):
            if col in wx:
                dk_df.at[idx, col] = wx[col]
    if _wx_cache:
        print(f"  Weather: fetched for {len(_wx_cache)} unique game/venue combos")

    # ─── Sport-specific context (MLB pitching, NHL goalie, NCAAB rankings) ───
    _sport_str = str(sport).lower() if sport else ""

    if _sport_str == "mlb":
        for idx, row in dk_df.iterrows():
            try:
                row_dict = row.to_dict()
                if idx in _pitcher_lookup:
                    row_dict["pitcher_matchup"] = _pitcher_lookup[idx]
                ctx = get_mlb_context(row_dict)
                for col, val in ctx.items():
                    dk_df.at[idx, col] = val
                dk_df.at[idx, "sport_context_adj"] = ctx.get("mlb_adj", 0.0)
                flags = []
                if ctx.get("sp_flag_home"):
                    flags.append(f"H:{ctx['sp_flag_home']}")
                if ctx.get("sp_flag_away"):
                    flags.append(f"A:{ctx['sp_flag_away']}")
                if ctx.get("park_factor", 1.0) >= 1.05 or ctx.get("park_factor", 1.0) <= 0.95:
                    flags.append(f"PF:{ctx['park_factor']:.2f}")
                dk_df.at[idx, "sport_context_flag"] = "|".join(flags)
            except Exception as e:
                print(f"[WARN] MLB context for row {idx}: {repr(e)}")
        print(f"  MLB context: pitcher stats + park factors applied")

    elif _sport_str == "nhl":
        # Join goalie data from situational fetch
        if situational and "goalies" in situational:
            _goalies = situational["goalies"]
            for idx, row in dk_df.iterrows():
                home = row.get("home_team_norm", "")
                away = row.get("away_team_norm", "")
                match_key = f"{away} @ {home}"
                goalie_info = _goalies.get(match_key)
                if not goalie_info:
                    for gk, gv in _goalies.items():
                        if gv.get("home_team_norm") == home and gv.get("away_team_norm") == away:
                            goalie_info = gv
                            break
                try:
                    # Build row dict with goalie_matchup for nhl_context
                    row_dict = row.to_dict()
                    if goalie_info:
                        row_dict["goalie_matchup"] = goalie_info
                    ctx = get_nhl_context(row_dict)
                    for col, val in ctx.items():
                        dk_df.at[idx, col] = val
                    dk_df.at[idx, "sport_context_adj"] = ctx.get("nhl_adj", 0.0)
                    flags = []
                    if ctx.get("goalie_flag_home"):
                        flags.append(f"H:{ctx['goalie_flag_home']}")
                    if ctx.get("goalie_flag_away"):
                        flags.append(f"A:{ctx['goalie_flag_away']}")
                    dk_df.at[idx, "sport_context_flag"] = "|".join(flags)
                except Exception as e:
                    print(f"[WARN] NHL goalie context for row {idx}: {repr(e)}")
            print(f"  NHL context: goalie status applied")

    elif _sport_str == "ncaab":
        try:
            _rankings = fetch_ncaab_rankings()
            if not _rankings.get("error"):
                _rank_map = _rankings.get("rankings", {})
                for idx, row in dk_df.iterrows():
                    home = row.get("home_team_norm", "")
                    away = row.get("away_team_norm", "")
                    h_rank = _rank_map.get(home, 0)
                    a_rank = _rank_map.get(away, 0)
                    dk_df.at[idx, "rank_home"] = h_rank
                    dk_df.at[idx, "rank_away"] = a_rank

                    # Rank differential scoring
                    rank_adj = 0.0
                    if h_rank > 0 and a_rank > 0:
                        diff = a_rank - h_rank  # positive = home ranked higher
                        if abs(diff) >= 15:
                            rank_adj = 1.0 if diff > 0 else -1.0
                        elif abs(diff) >= 8:
                            rank_adj = 0.5 if diff > 0 else -0.5
                    elif h_rank > 0 and a_rank == 0:
                        rank_adj = 0.5   # Ranked vs unranked home advantage
                    elif a_rank > 0 and h_rank == 0:
                        rank_adj = -0.5  # Unranked home vs ranked away

                    # Apply relative to side
                    side = str(row.get("side", "")).lower()
                    home_lower = str(home).lower()
                    if home_lower and home_lower in side:
                        dk_df.at[idx, "sport_context_adj"] = rank_adj
                    else:
                        dk_df.at[idx, "sport_context_adj"] = -rank_adj

                    if h_rank > 0 or a_rank > 0:
                        dk_df.at[idx, "sport_context_flag"] = f"H:#{h_rank or'NR'}|A:#{a_rank or'NR'}"
                print(f"  NCAAB context: {len(_rank_map)} ranked teams applied")
        except Exception as e:
            print(f"  NCAAB rankings skipped: {e}")

    # UFC: filter out non-MONEYLINE markets (UFC only has 2-way moneyline)
    if _sport_str == "ufc" and "market" in dk_df.columns:
        _pre_ufc = len(dk_df)
        dk_df = dk_df[~dk_df["market"].astype(str).str.upper().isin(["SPREAD", "TOTAL"])].copy()
        _post_ufc = len(dk_df)
        if _pre_ufc != _post_ufc:
            print(f"  UFC: filtered {_pre_ufc - _post_ufc} non-MONEYLINE rows")

    # Clean up temp column
    if "_side_norm" in dk_df.columns:
        dk_df.drop(columns=["_side_norm"], inplace=True)

    # ─── Final layer mode summary ───
    if "layer_mode" in dk_df.columns:
        mode_counts = dk_df["layer_mode"].value_counts().to_dict()
        print(f"  Layer modes: {mode_counts}")

    return dk_df
