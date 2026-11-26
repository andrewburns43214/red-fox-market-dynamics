import json

import pandas as pd
import pytest

from favorite_split_evidence import capture
from red_fox_favorite import NBA_PUBLIC_PATH, update_favorite_tracking
from tests.test_nba_recovered_shadow import board as nba_board
from tests.test_nhl_release_policy import AT, START, row as nhl_row, side as nhl_side
from tests.test_red_fox_favorite import market, side


def tables(path, filename):
    return pd.read_csv(path / filename, dtype=str, keep_default_na=False)


def evidence(row):
    return json.loads(row["decision_evidence"])


@pytest.mark.parametrize("bets,money,kind", [(46, 45, "ticket_only"), (45, 46, "money_only"), (46, 46, "both")])
@pytest.mark.parametrize("sport", ["nfl", "ncaaf", "nba", "ncaab", "nhl", "mlb", "ufc"])
def test_selected_quote_labels_independent_breaches(sport, bets, money, kind):
    at = "2026-11-24T22:20:00Z"
    market_type = "SPREAD" if sport in {"nfl", "ncaaf"} else "MONEYLINE"
    away = "Away +2.5" if market_type == "SPREAD" else "Away"
    home = "Home -2.5" if market_type == "SPREAD" else "Home"
    quote = "+2.5 (-120)" if market_type == "SPREAD" else "+120"
    observed = market(sport, market_type, [
        side(home, "-140", bets=100-bets, money=100-money, source_latest_at=at),
        side(away, quote, bets=bets, money=money, source_latest_at=at),
    ])
    e = capture(observed, {"favorite_side": away, "favorite_pathway": "low_support_contrarian"},
                None, phase="removal")
    assert e["split_breach"] == kind
    assert e["source_observed_at"] == at
    assert e["quotes"][market_type]["current_line"] == quote
    assert e["outcome_lookup"]["game_id"] == "1"


def test_ordinary_ncaab_review_evidence_keeps_identity_and_actual_failures(tmp_path):
    def publish(at, *, bets=35, money=30):
        raw = nba_board(at, money=money, sport="ncaab")
        # Preserve real route qualification for the first capture. The raw
        # current board side has no confirmed directional read on a breach.
        sides = json.loads(raw.at[0, "market_sides"])
        sides[0]["bets_pct"] = bets
        sides[1]["bets_pct"] = 100 - bets
        raw.at[0, "market_sides"] = json.dumps(sides)
        if bets > 45:
            raw.loc[0, ["red_fox_favorite", "favorite_side", "favorite_pathway", "supported_side"]] = ["false", "", "", ""]
        return update_favorite_tracking(raw, tmp_path, at, nba_config=(False, False))

    times = pd.date_range("2026-11-24T21:00:00Z", periods=10, freq="10min").map(lambda t: t.isoformat())
    assert not publish(times[0]).red_fox_favorite.eq("true").any()
    assert not publish(times[1], bets=50).red_fox_favorite.eq("true").any()
    publish(times[2])
    assert publish(times[3]).red_fox_favorite.eq("true").any()
    assert publish(times[4], money=47).red_fox_favorite.eq("true").any()
    assert not publish(times[5], money=48).red_fox_favorite.eq("true").any()
    publish(times[6])
    assert not publish(times[7], bets=46).red_fox_favorite.eq("true").any()

    reviews = tables(tmp_path, "red_fox_favorite_review.csv")
    clears = [evidence(r) for _, r in reviews.loc[reviews.review_state.eq("cleared")].iterrows()]
    assert [(r["phase"], r["split_breach"]) for r in clears] == [
        ("initial_confirmation", "ticket_only"), ("reinstatement", "ticket_only"),
    ]
    assert all(r["selected_side"] == "Away +1.5" for r in clears)
    assert clears[0]["quotes"]["SPREAD"]["bets_pct"] == 50
    assert clears[1]["cross_market"]["state"] == "confirmation"
    assert clears[1]["quotes"]["MONEYLINE"]["current_line"] == "+116"

    tracking = tables(tmp_path, "red_fox_favorite_tracking.csv")
    removed = tracking.loc[tracking.favorite_state.eq("not_qualified")].iloc[-1]
    assert float(removed["money_pct"]) == 30
    assert evidence(removed)["quotes"]["SPREAD"]["money_pct"] == 48
    terminal = reviews.loc[reviews.review_action.eq("removal") & reviews.review_state.eq("applied")].iloc[-1]
    assert evidence(terminal)["split_breach"] == "money_only"


@pytest.mark.parametrize("bets,money,kind", [(69, 55, "ticket_only"), (70, 54, "money_only"), (69, 54, "both")])
def test_public_moneyline_records_actual_minimum_shortfalls(bets, money, kind):
    observed = market("nba", "MONEYLINE", [side("Away", "-122", bets=bets, money=money),
                                          side("Home", "+110", bets=100-bets, money=100-money)])
    identity = {"favorite_side": "Away", "favorite_pathway": NBA_PUBLIC_PATH}
    assert capture(observed, identity, None, phase="initial_confirmation")["split_breach"] == kind


def test_nhl_official_attempt_and_removal_link_to_durable_outcome():
    from nhl_paper_release import record_paper
    from nhl_release_policy import PUBLIC
    kickoff = pd.Timestamp(START)
    since = kickoff - pd.Timedelta(hours=4)
    schedule = {"league-id": {"away": "away", "home": "home", "start": int(kickoff.timestamp()), "state": "FUT"}}
    history = pd.DataFrame([
        dict(sport="nhl", game_id="x", market_display="MONEYLINE", timestamp=AT, dk_start_iso=START,
             side=name, open_line=opener, current_line=current)
        for name, opener, current in (("Away", "+110", "-120"), ("Home", "-130", "+105"))
    ])
    saved = {}
    for minutes, bets, money in [(150, 82, 60), (134, 79, 54), (110, 82, 60), (94, 82, 60), (25, 79, 55)]:
        at = kickoff - pd.Timedelta(minutes=minutes)
        frame = pd.DataFrame([nhl_row([nhl_side("Home", "+105", 100-bets, 100-money, at.isoformat()),
                                      nhl_side("Away", "-120", bets, money, at.isoformat())])])
        record_paper(frame, history, {}, schedule, saved, {PUBLIC}, at, official_since=since)
        # Same quote retry may not create another failure or change the lane.
        record_paper(frame, history, {}, schedule, saved, {PUBLIC}, at, official_since=since)
    game = saved["league-id"]
    terminal = [e for e in game["events"] if e["stage"] in {"initial_confirmation_failed", "removed"}]
    assert [(e["stage"], e["decision_evidence"]["split_breach"]) for e in terminal] == [
        ("initial_confirmation_failed", "both"), ("removed", "ticket_only"),
    ]
    assert terminal[-1]["decision_evidence"]["published_lane_at_capture"]
    assert terminal[-1]["decision_evidence"]["quotes"]["MONEYLINE"]["current_line"] == "-120"
    assert terminal[-1]["decision_evidence"]["outcome_lookup"]["league_event_id"] == "league-id"
    schedule["league-id"].update(state="OFF", away_score=3, home_score=2)
    record_paper(pd.DataFrame(), history, {}, schedule, saved, {PUBLIC}, kickoff + pd.Timedelta(hours=3),
                 official_since=since)
    assert game["result"]["winning_team"] == "away"
