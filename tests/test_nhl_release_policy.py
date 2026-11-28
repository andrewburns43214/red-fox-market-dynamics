"""Isolation and hockey-specific release regressions."""
import json

import pandas as pd
import pytest

from favorite_nhl_goalies import _track_goalie_changes, goalie_evidence, paper_dir
from nhl_release_policy import CROSSOVER, PUBLIC, assess, fair, profile
from red_fox_favorite import (
    CONFIG, NHL_HIGH_HANDLE_PAUSE, NHL_ORDINARY_PAUSE, NOV24_VERSION,
    apply_red_fox_favorites, update_favorite_tracking,
)

AT = "2026-11-24T22:00:00Z"
START = "2026-11-25T01:00:00Z"


def side(name, odds, bets, money, at=AT):
    return dict(flagged_side=name,current_line=odds,open_line="+105",bets_pct=bets,
                money_pct=money,source_latest_at=at,data_badge="Clean",
                observation_count=5,price_move_pct=3,response_direction="TOWARD",
                reaction="Watch",kpi_eligible=False)


def row(sides, sport="nhl", support=""):
    return dict(sport=sport,game_id="x",game="Away @ Home",market_display="MONEYLINE",
                kickoff_iso=START,market_sides=json.dumps(sides),supported_side=support,
                current_line=sides[0]["current_line"],state_as_of_utc=sides[0]["source_latest_at"])


def test_public_does_not_require_movement_or_supported_side():
    chosen=side("Away","-120",80,55)
    chosen.update(price_move_pct=0,response_direction="LIMITED")
    rival=side("Home","+105",20,45)
    frame=row([chosen,rival])
    assert assess(frame,"Away",PUBLIC,pd.Timestamp(AT),.55)==("",False)
    assert assess(frame,"Away",PUBLIC,pd.Timestamp(AT),.55,{"status":"verified_change"})[1]
    frame["supported_side"]="Home"
    assert assess(frame,"Away",PUBLIC,pd.Timestamp(AT),.55)[1]
    assert not frame["supported_side"]=="Away"


def test_crossover_uses_paired_fair_probability():
    a,b=side("Away","-105",50,50),side("Home","-115",50,50)
    assert profile(CROSSOVER,a,b,fair(105,-125),fair(-105,-115))
    a["current_line"]="-120"
    assert not profile(CROSSOVER,a,b,fair(105,-125),fair(-120,-105))
    assert profile(CROSSOVER,a,b,.5,fair(-120,-105))


def capture(at, high=False):
    a=side("Away","-115",35,65 if high else 30,at)
    b=side("Home","+105",65,35 if high else 70,at)
    if not high:
        a.update(reaction="Contrarian",kpi_eligible=True)
    data=row([a,b],support="" if high else "Away")
    cache={"20261124":{"observed_at":at,"games":{"away @ home":["Goalie A","Goalie B"]}}}
    return apply_red_fox_favorites(pd.DataFrame([data]),as_of=at,nov24_goalies=cache)


@pytest.mark.parametrize("high,flag,version",[
    (False,NHL_ORDINARY_PAUSE,CONFIG.nhl_version),(True,NHL_HIGH_HANDLE_PAUSE,NOV24_VERSION)])
def test_nhl_pause_blocks_new_promotion_but_keeps_active_state(tmp_path, monkeypatch, high, flag, version):
    second="2026-11-24T22:16:00Z"
    third="2026-11-24T22:26:00Z"
    active=tmp_path/"active"
    active.mkdir()
    for at in (AT,second):
        before=update_favorite_tracking(capture(at,high),active,as_of=at)
    assert before.iloc[0].red_fox_favorite=="true"
    monkeypatch.setenv(flag,"1")
    preserved=update_favorite_tracking(capture(third,high),active,as_of=third)
    assert preserved.iloc[0].red_fox_favorite=="true"
    for at in (AT,second):
        held=update_favorite_tracking(capture(at,high),tmp_path,as_of=at)
        assert held.iloc[0].red_fox_favorite=="false"
        assert held.iloc[0].supported_side==("" if high else "Away")
    shadow=pd.read_csv(tmp_path/"red_fox_favorite_shadow.csv")
    paused=shadow[shadow.shadow_cohort.str.endswith("_promotion_paused",na=False)]
    assert len(paused)==2 and paused.favorite_rule_version.eq(version).all()


def test_goalie_pair_change_detected_across_unconfirmed_capture(tmp_path):
    def make(pair, at=AT):
        return {"20261124":{"observed_at":at,"provider_available":True,
            "evidence":{"away @ home":dict(event_id="espn-event",observed_at=at,scheduled_at=START,
                confirmed_pair=pair,status="both_confirmed" if pair else "unconfirmed",
                starters=pair or [],provider="espn")}}}

    data_dir=tmp_path/"data"
    a,b,c=make(["1","2"]),make(None),make(["1","3"])
    _track_goalie_changes(a,{},data_dir)
    _track_goalie_changes(b,a,data_dir)
    _track_goalie_changes(c,b,data_dir)
    assert goalie_evidence(b,"Away @ Home",START,AT)["status"]=="unconfirmed"
    assert goalie_evidence(c,"Away @ Home",START,AT)["status"]=="both_confirmed"
    assert goalie_evidence(c,"Away @ Home",START,AT,previous_pair=("1","2"))["status"]=="verified_change"
    assert goalie_evidence(c,"Away @ Home",START,"2026-11-24T22:13:00Z")["status"]=="unavailable"
    events=[json.loads(line) for line in (paper_dir(data_dir)/"favorite_nhl_goalie_observations.jsonl").read_text().splitlines()]
    assert events[-1]["status"]=="verified_change"
    assert events[-1]["previous_confirmed_pair"]==["1","2"]


def test_paper_combines_once_at_original_price_with_own_t30_and_no_official_write(tmp_path):
    from nhl_paper_release import paper_totals, record_paper

    kickoff=pd.Timestamp(START)
    cutoff=int(kickoff.timestamp())
    data_dir=tmp_path/"data"
    data_dir.mkdir()
    marker=data_dir/"performance_favorites.csv"
    marker.write_text("published\n",encoding="utf-8")
    schedule={"2026020001":{"away":"away","home":"home","start":cutoff,"state":"FUT"}}
    history=pd.DataFrame([
        dict(sport="nhl",game_id="x",market_display="MONEYLINE",timestamp=AT,dk_start_iso=START,
             side=name,open_line=opener,current_line=current)
        for name,opener,current in (("Away","+110","-120"),("Home","-130","+102"))
    ])
    saved={}
    for minutes,odds in ((65,"-120"),(49,"-120"),(25,"-125"),(6,"-125")):
        now=kickoff-pd.Timedelta(minutes=minutes)
        a=side("Away",odds,80,55,now.isoformat())
        b=side("Home","+105",20,45,now.isoformat())
        record_paper(pd.DataFrame([row([a,b])]),history,{},schedule,saved,{PUBLIC,CROSSOVER},now)
    schedule["2026020001"].update(state="OFF",away_score=3,home_score=2)
    record_paper(pd.DataFrame(),history,{},schedule,saved,{PUBLIC,CROSSOVER},kickoff+pd.Timedelta(hours=3))
    result=paper_totals(saved)
    assert len(result["entries"])==2
    assert result["totals"]["combined"]["entries"]==1
    assert result["totals"][PUBLIC]["wins"]==result["totals"][CROSSOVER]["wins"]==1
    assert result["totals"]["combined"]["net_units"]==pytest.approx(100/120)
    assert all(entry["goalie_status"]=="unavailable" for entry in result["entries"])
    assert marker.read_text(encoding="utf-8")=="published\n"


@pytest.mark.parametrize("active, winner, closing_pass", [
    ({PUBLIC, CROSSOVER}, CROSSOVER, True),
    ({PUBLIC}, PUBLIC, True),
    ({CROSSOVER}, CROSSOVER, True),
    ({PUBLIC}, PUBLIC, False),
])
def test_official_nhl_board_kickoff_and_single_settlement(tmp_path, monkeypatch, active, winner, closing_pass):
    import nhl_official_release as release
    from build_live_recent import favorite_handoff_states, final_pregame_states
    from nhl_release_config import OFFICIAL_FLAGS, OFFICIAL_HANDOFF, OFFICIAL_VERSIONS
    from performance_ledger import _frozen_records, update_performance_ledger

    kickoff = pd.Timestamp(START)
    cutoff = int(kickoff.timestamp())
    since = kickoff - pd.Timedelta(minutes=70)
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(release, "official_start", lambda: since.isoformat())
    monkeypatch.setattr(release, "switches", lambda: {OFFICIAL_FLAGS[lane]: True for lane in active})
    monkeypatch.setattr(release, "index_schedule", lambda *args: {
        "2026020001": {"away": "away", "home": "home", "start": cutoff, "state": "FUT"},
    })
    monkeypatch.setenv(NHL_ORDINARY_PAUSE, "1")
    monkeypatch.setenv(NHL_HIGH_HANDLE_PAUSE, "1")
    history = pd.DataFrame([
        dict(sport="nhl", game_id="x", market_display="MONEYLINE", timestamp=AT, dk_start_iso=START,
             side=name, open_line=opener, current_line=current)
        for name, opener, current in (("Away", "+110", "-120"), ("Home", "-130", "+102"))
    ])
    candidate_path = data / "red_fox_favorite_freeze_candidates.csv"
    for minutes, selected_price in ((80, "-120"), (65, "-120"), (49, "-120"), (25, "-125"), (6, "-125")):
        clock = kickoff - pd.Timedelta(minutes=minutes)
        source = clock.isoformat()
        bets = 75 if not closing_pass and minutes == 25 else 80
        chosen = side("Away", f"Away @ {selected_price}", bets, 55, source)
        rival = side("Home", "Home @ +105", 100-bets, 45, source)
        # The displayed market leader can be the opposite team. The official
        # quote, category and grade must all use the selected side.
        market = row([rival, chosen])
        market["flagged_side"] = "Home"
        classified = pd.DataFrame([market])
        selections = release.capture_official(classified, history, {}, data, observed=clock)
        raw = apply_red_fox_favorites(classified, as_of=clock)
        published = update_favorite_tracking(raw, data, as_of=clock, nhl_selections=selections,
            deferred_markets={("nba", "retained", "MONEYLINE")}, unavailable_sports={"nba"})
        picked = published.iloc[0]
        if minutes in (80, 65) or (not closing_pass and minutes <= 25):
            assert picked.red_fox_favorite == "false"
            if not closing_pass and minutes <= 25:
                assert picked.favorite_review_reason == "Moneyline split failed at 75% tickets, below 80%."
        else:
            assert picked.red_fox_favorite == "true"
            assert picked.favorite_side == "Away"
            assert picked.favorite_pathway == winner
            assert picked.supported_side == ""
    archive = pd.read_csv(candidate_path, dtype=str, keep_default_na=False)
    tracking = pd.read_csv(data / "red_fox_favorite_tracking.csv", dtype=str, keep_default_na=False)
    games = release.read_official_games(data)
    assert not favorite_handoff_states(archive, kickoff, tracking, published).shape[0]
    fallback = final_pregame_states(archive, kickoff)
    assert not _frozen_records(fallback, tracking, "generic fallback")
    started = release.reconcile_official_handoffs(fallback, archive, games, kickoff)
    assert len(started) == 1
    if not closing_pass:
        assert started.iloc[0].red_fox_favorite == "false"
        assert not _frozen_records(started, tracking, "failed closing")
        return
    assert started.iloc[0].freeze_method == OFFICIAL_HANDOFF
    assert started.iloc[0].red_fox_favorite == "true"
    assert started.iloc[0].favorite_rule_version == OFFICIAL_VERSIONS[winner]
    scores = data / "final_scores_history.csv"
    pd.DataFrame([dict(game_id="x", sport="nhl", team1="Away", team2="Home", team1_score=4, team2_score=2,
                       score_provider="espn_scoreboard", score_source="ESPN", score_provider_event_id="espn-x")]).to_csv(scores,index=False)
    started.to_csv(data / "live_recent.csv", index=False)
    alternate = data / "duplicate_handoff.csv"
    started.to_csv(alternate, index=False)
    first = update_performance_ledger(data, [data / "live_recent.csv", alternate])
    second = update_performance_ledger(data)
    official = pd.read_csv(data / "performance_favorites.csv", dtype=str, keep_default_na=False)
    assert len(official) == 1 and official.ledger_id.is_unique
    graded = official.iloc[0]
    assert graded.side == "Away" and graded.grade == "W"
    assert graded.decision_price == "-120" and graded.closing_price == "-125"
    assert float(graded.net_profit_units) == pytest.approx(100/120, abs=1e-6)
    assert first["new_rows"] == 1 and second["new_rows"] == 0
    report = json.loads((data / "favorite_performance.json").read_text())
    assert report["wins"] == 1
    assert next(s for s in report["sports"] if s["sport"] == "nhl")["wins"] == 1
    assert report["nhl_replacement_pathways"][winner]["wins"] == 1


def test_official_overlap_vetoes_opposing_simultaneous_confirmations():
    from nhl_release_policy import first_confirmed_lane

    states = {
        PUBLIC: {"pathway": PUBLIC, "entry": {"at": 1000, "side": "away"}},
        CROSSOVER: {"pathway": CROSSOVER, "entry": {"at": 1000, "side": "home"}},
    }
    assert first_confirmed_lane(states) == ""
    states[CROSSOVER]["entry"]["at"] += 1
    assert first_confirmed_lane(states) == PUBLIC


def test_official_release_occupancy_uses_provider_event_despite_vendor_id_change():
    from nhl_official_release import _occupied_games

    kickoff = pd.Timestamp(START)
    market = row([side("Away", "-120", 80, 55), side("Home", "+105", 20, 45)])
    market.update(red_fox_favorite="true", game_id="old-vendor-id", market_display="SPREAD",
                  favorite_rule_version=CONFIG.nhl_version)
    schedule = {"2026020001": {"away": "away", "home": "home", "start": kickoff.timestamp()}}
    assert _occupied_games(pd.DataFrame([market]), schedule) == {"2026020001"}
