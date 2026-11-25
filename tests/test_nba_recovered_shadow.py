import json

import pandas as pd
import pytest

from nba_legacy_shadow import record_legacy
from nba_recovered_shadow import record_recoveries
from red_fox_favorite import apply_red_fox_favorites, update_favorite_tracking
from tests.test_red_fox_favorite import market, side

TIP = "2026-11-25T01:00:00Z"
CONFIG = (True, True)
MATCHUP = "Away @ Home"


def board(at, *, spread=1.5, price="+116", money=30, source=None, sport="nba"):
    observation = source or at
    label = f"Away {spread:+g}"
    rows = [
        market(sport, "SPREAD", [
            side(label, f"{spread:+g} (-104)", open_line="+3.5 (-110)", line_move=3.5 - spread,
                 money=money, source_latest_at=observation),
            side(f"Home {-spread:+g}", f"{-spread:+g} (-116)", reaction="Watch", direction="AGAINST",
                 bets=65, money=100 - money, kpi=False, source_latest_at=observation),
        ], game_id="fixture", game=MATCHUP, supported_side=label if money <= 45 else ""),
        market(sport, "MONEYLINE", [
            side("Away", price, open_line="+136", price_move=5.0, source_latest_at=observation),
            side("Home", "-115", reaction="Watch", direction="AGAINST", bets=65, money=70,
                 kpi=False, source_latest_at=observation),
        ], game_id="fixture", game=MATCHUP, supported_side="Away"),
    ]
    for row in rows:
        row.update(kickoff_iso=TIP, state_as_of_utc=observation)
    return apply_red_fox_favorites(pd.DataFrame(rows), as_of=at, nov24_goalies={})


def publish(tmp_path, at, *, released=True, now=None, manual=False, **changes):
    raw = board(at, **changes)
    hypothetical = record_legacy(raw, tmp_path, at) if released else None
    live = update_favorite_tracking(raw, tmp_path, as_of=at, nba_config=CONFIG if released else (False, False))
    if manual:
        live["manual_revision"] = "42"
    if released:
        record_recoveries(raw[raw.sport.eq("nba")], hypothetical, live, tmp_path, at, CONFIG, now=now or at)
    return live


def recoveries(tmp_path):
    path = tmp_path / "nba_legacy_hypothetical" / "recovered_favorites.csv"
    return pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame()


def test_recovery_routes_store_own_entry_and_settle_only_once(tmp_path):
    publish(tmp_path, "2026-11-24T22:00:00Z", released=False)
    live = publish(tmp_path, "2026-11-24T22:10:00Z", released=False)
    assert live[live.red_fox_favorite.eq("true")].market_display.tolist() == ["SPREAD"]
    for minute in (20, 30):
        publish(tmp_path, f"2026-11-24T22:{minute}:00Z", money=47)
    history_path = tmp_path / "red_fox_favorite_tracking.csv"
    baseline = history_path.read_bytes()
    publish(tmp_path, "2026-11-24T22:40:00Z")
    assert recoveries(tmp_path).empty
    for _ in range(2):
        live = publish(tmp_path, "2026-11-24T22:50:00Z")
    assert not live.red_fox_favorite.eq("true").any()
    rows = recoveries(tmp_path)
    assert rows.recovery_type.tolist() == ["same_market"]
    assert rows.decision_price.tolist() == ["-104"]
    assert rows.removal_reason.str.len().min() > 0
    assert rows.original_side.tolist() == ["Away +1.5"]

    for minute in (0, 10):
        live = publish(tmp_path, f"2026-11-24T23:{minute:02d}:00Z", spread=-1.5, price="-109")
    assert not live.red_fox_favorite.eq("true").any()
    rows = recoveries(tmp_path)
    assert rows.recovery_type.tolist() == ["same_market", "spread_to_moneyline"]
    assert rows.decision_price.tolist() == ["-104", "-109"]
    assert rows.decision_line.tolist() == ["+1.5", ""]
    assert json.loads(rows.iloc[1].entry_market_evidence)["open_line"] == "+136"
    assert json.loads(rows.iloc[1].companion_market_evidence)["current_line"] == "-1.5 (-104)"

    for at in ("2026-11-25T00:20:00Z", "2026-11-25T00:41:00Z", "2026-11-25T00:59:00Z"):
        publish(tmp_path, at, spread=-1.5, price="-114")
    pd.DataFrame([dict(game_id="fixture", team1="Away", team2="Home", team1_score="109", team2_score="108")]).to_csv(
        tmp_path / "final_scores_history.csv", index=False
    )
    for _ in range(2):
        at = "2026-11-25T01:10:00Z"
        raw = board(at).iloc[:0]
        hypothetical = record_legacy(raw, tmp_path, at)
        record_recoveries(raw, hypothetical, raw, tmp_path, at, CONFIG, now=at)
    rows = recoveries(tmp_path)
    assert rows.loc[rows.count_for_performance.eq("yes"), "recovery_type"].tolist() == ["spread_to_moneyline"]
    assert json.loads(rows.iloc[0].subsequent_removals)
    assert rows.iloc[1].grade == "W"
    assert float(rows.iloc[1].net_profit_units) == pytest.approx(100 / 109, rel=1e-6)
    report = json.loads((tmp_path / "nba_legacy_hypothetical" / "recovered_favorite_performance.json").read_text())
    assert report["sample_size"] == report["wins"] == 1
    assert report["roi_pct"] == pytest.approx(10000 / 109, abs=0.0001)
    assert report["by_recovery_type"]["same_market"]["sample_size"] == 0
    assert history_path.read_bytes() == baseline
    assert not (tmp_path / "favorite_performance.json").exists()


@pytest.mark.parametrize("case", ["never_published", "stale", "late", "manual", "pre_activation"])
def test_recovery_excludes_unverified_or_independently_blocked_cases(tmp_path, case):
    if case != "never_published":
        for minute in (0, 10):
            publish(tmp_path, f"2026-11-24T22:{minute:02d}:00Z", released=False)
    for minute in (20, 30):
        publish(tmp_path, f"2026-11-24T22:{minute}:00Z", money=47,
                now="2026-11-24T23:00:00Z" if case == "pre_activation" else None)
    if case == "late":
        times = ("2026-11-25T00:29:00Z", "2026-11-25T00:40:00Z")
    else:
        times = ("2026-11-24T22:40:00Z", "2026-11-24T22:50:00Z")
    for time in times:
        publish(tmp_path, time, source=times[0] if case == "stale" else time, manual=case == "manual")
    assert recoveries(tmp_path).empty
