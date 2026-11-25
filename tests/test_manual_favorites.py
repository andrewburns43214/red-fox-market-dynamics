from contextlib import closing
from datetime import datetime, timedelta, timezone
import importlib
import json
from pathlib import Path
import sqlite3
from threading import Thread

import pandas as pd
import pytest
import requests

import manual_favorites as manual
from performance_ledger import update_performance_ledger

NOW = datetime(2026, 11, 25, 22, 50, tzinfo=timezone.utc)
START = NOW + timedelta(minutes=10)
OWNER = {"id": "verified-owner-id", "email": "andrewburns43214@gmail.com"}
EVENT = ("nfl", "101")


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.invalid")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "test-public-key")
    module = importlib.import_module("manual_favorite_server")
    monkeypatch.setattr(module, "now_utc", lambda: NOW)
    monkeypatch.setattr(manual, "now_utc", lambda: NOW)
    return module


def seed(data, *, favorite=True, sport="nfl", price="-105", kickoff=START):
    rows, raw = [], []
    for market, names in {
        "SPREAD": ("Away +3", "Home -3"),
        "MONEYLINE": ("Away", "Home"),
        "TOTAL": ("Over 44", "Under 44"),
    }.items():
        sides = [{"flagged_side": name, "open_line": f'{name} @ {price}', "current_line": f'{name} @ {price}'} for name in names]
        rows.append(dict(sport=sport,game_id="101",game="Away @ Home",kickoff_iso=kickoff.isoformat(),
                         market_display=market, market_sides=json.dumps(sides), data_badge="Clean",
                         red_fox_favorite="true" if favorite and market=="SPREAD" else "false",
                         favorite_side="Away +3" if favorite and market=="SPREAD" else "",
                         favorite_state="qualified" if favorite and market=="SPREAD" else "not_qualified",
                         favorite_rule_version="red_fox_favorite_v5", supported_side="Away +3" if market=="SPREAD" else "",
                         state_as_of_utc=(NOW-timedelta(minutes=1)).isoformat()))
        raw.extend(dict(sport=sport,game_id="101",game="Away @ Home",dk_start_iso=kickoff.isoformat(),
                        timestamp=(NOW-timedelta(minutes=1)).isoformat(),side=name,
                        bets_pct="50",money_pct="50",current_line=f'{name} @ {price}') for name in names)
    board = pd.DataFrame(rows)
    board.to_csv(data/"anomaly_board.csv",index=False)
    pd.DataFrame(raw).to_csv(data/"snapshots.csv",index=False)
    return board


def fixture():
    return {"id":"501", "source":"espn", "observed":NOW.isoformat(), "start":START.isoformat(),
            "provider_start": START.isoformat()}


def commit(server, data, action, revision, token="", event=EVENT):
    return server.persist(data,event,OWNER,dict(action=action,revision=revision,quote=token,reason="private reason"))


def selection(data, market, side):
    rows = manual.event_rows(data, EVENT)
    return next(q for q, _ in manual.latest_quotes(data, rows, NOW).values() if (q["market"],q["side"])==(market,side))


def active(frame):
    return frame.loc[frame["red_fox_favorite"].astype(str).eq("true")]


def test_override_change_remove_restore_and_stale_revision(tmp_path, server, monkeypatch):
    board=seed(tmp_path)
    monkeypatch.setattr(server,"check_pregame",lambda *_:fixture())
    before=manual.project_board(board,tmp_path,NOW)
    pd.testing.assert_frame_equal(before,board)

    initial=selection(tmp_path,"TOTAL","Over")
    first,result=commit(server,tmp_path,"change",0,initial["token"])
    assert first==200
    live=manual.project_board(board,tmp_path,NOW)
    assert list(active(live)["favorite_side"])==["Over 44"]
    assert list(active(live)["manual_entry_price"])==["-105"]
    assert live["supported_side"].equals(board["supported_side"])
    assert commit(server,tmp_path,"remove",0)[0]==409

    # Next request and all workers use the persisted decision, without
    # depending on a projection from the previous request or process.
    revised=selection(tmp_path,"MONEYLINE","Home")
    second,result=commit(server,tmp_path,"change",1,revised["token"])
    assert second==200
    assert list(active(manual.project_board(board,tmp_path,NOW))["favorite_side"])==["Home"]
    assert commit(server,tmp_path,"remove",2)[0]==200
    assert active(manual.project_board(board,tmp_path,NOW+timedelta(minutes=3))).empty
    assert active(manual.project_board(board.assign(kickoff_iso=(START+timedelta(minutes=2)).isoformat()),tmp_path,NOW)).empty
    assert commit(server,tmp_path,"restore",3)[0]==200
    pd.testing.assert_frame_equal(manual.project_board(board,tmp_path,NOW),board)
    with closing(sqlite3.connect(manual.db_path(tmp_path))) as connection:
        rows=connection.execute("SELECT owner_id,owner_email,action,reason,previous FROM decisions ORDER BY id").fetchall()
    assert [row[2] for row in rows]==["change","change","remove","restore"]
    assert all(row[:2]==("verified-owner-id",OWNER["email"]) and row[3]=="private reason" for row in rows)
    assert json.loads(rows[0][-1])["choice"]["side"]=="Away +3"
    assert json.loads(rows[-1][-1])["override"]=="remove"


def test_save_refuses_changed_stale_or_replaced_quote(tmp_path, server, monkeypatch):
    seed(tmp_path,favorite=False)
    monkeypatch.setattr(server,"check_pregame",lambda *_:fixture())
    old=selection(tmp_path,"SPREAD","Away")
    seed(tmp_path,favorite=False,price="-120")
    status,response=commit(server,tmp_path,"add",0,old["token"])
    assert status==409 and response["refresh"]
    fresh=selection(tmp_path,"SPREAD","Away")

    def different_capture(*_):
        with (tmp_path/"snapshots.csv").open("a") as output:
            output.write("\n")
        return fixture()

    monkeypatch.setattr(server,"check_pregame",different_capture)
    assert commit(server,tmp_path,"add",0,fresh["token"])[0]==409
    monkeypatch.setattr(server,"check_pregame",lambda *_:fixture())
    assert commit(server,tmp_path,"add",0,fresh["token"])[0]==200
    assert manual.decisions(tmp_path)[EVENT]["quote"]["price"]=="-120"


@pytest.mark.parametrize("market,side,scores,grade",[
    ("SPREAD","Away",(28,31),"Push"),
    ("MONEYLINE","Home",(28,31),"W"),
    ("TOTAL","Over",(22,22),"Push"),
    ("TOTAL","Under",(22,19),"W"),
    ("TOTAL","Over",(22,19),"L"),
    ("TOTAL","Under",None,"Void"),
])
def test_final_selection_settles_once_from_pinned_fixture(tmp_path,server,monkeypatch,market,side,scores,grade):
    board=seed(tmp_path)
    monkeypatch.setattr(server,"check_pregame",lambda *_:fixture())
    initial=selection(tmp_path,"MONEYLINE","Away")
    assert commit(server,tmp_path,"change",0,initial["token"])[0]==200
    final=selection(tmp_path,market,side)
    assert commit(server,tmp_path,"change",1,final["token"])[0]==200

    stamped=START+timedelta(minutes=30)
    board["final_pregame_state_at_utc"]=NOW.isoformat()
    board["frozen_at_utc"]=stamped.isoformat()
    board["freeze_method"]="published_state_latest_at_or_before_start"
    frozen=manual.freeze_overrides(board,pd.DataFrame(),tmp_path,{},{},stamped)
    official=active(frozen).iloc[0]
    assert official["favorite_side"]==final["label"]
    assert official["manual_revision"]=="2"
    assert official["manual_entry_price"]==final["price"]
    assert len(active(frozen))==1

    # An unrelated archived score must never grade the manual selection.
    wrong=pd.DataFrame([dict(game_id="101",team1="Away",team1_score="50",team2="Home",team2_score="2",
                             score_provider_event_id="wrong-fixture")])
    wrong.to_csv(tmp_path/"final_scores_history.csv",index=False)
    frozen.to_csv(tmp_path/"live_recent.csv",index=False)
    update_performance_ledger(tmp_path)
    pending=pd.read_csv(tmp_path/"performance_favorites.csv",dtype=str,keep_default_na=False)
    assert len(pending)==1 and pending.iloc[0]["grade"]==""

    for key,value in dict(score_state="post",score_provider="espn",score_provider_event_id="501",
                          score_away=str(scores[0]) if scores else "-",score_home=str(scores[1]) if scores else "-",
                          score_status_code="" if scores else "STATUS_CANCELED").items():
        frozen.at[official.name,key]=value
    frozen.to_csv(tmp_path/"live_recent.csv",index=False)
    for _ in range(2):
        update_performance_ledger(tmp_path)
    settled=pd.read_csv(tmp_path/"performance_favorites.csv",dtype=str,keep_default_na=False)
    assert len(settled)==1
    row=settled.iloc[0]
    assert (row["grade"],row["side"],row["decision_price"])==(grade,final["label"],final["price"])
    assert row["decision_line_source"]=="owner_selected_quote"
    assert row["clv_available"]=="no"
    dashboard=json.loads((tmp_path/"favorite_performance.json").read_text())
    assert (dashboard["wins"],dashboard["losses"],dashboard["pushes"])==tuple(int(grade==value) for value in ("W","L","Push"))


def test_removal_suppresses_old_handoff_and_unknown_or_started_is_rejected(tmp_path,server,monkeypatch):
    board=seed(tmp_path)
    monkeypatch.setattr(server,"check_pregame",lambda *_:fixture())
    assert commit(server,tmp_path,"remove",0)[0]==200
    assert active(manual.freeze_overrides(board,pd.DataFrame(),tmp_path,{},{},START)).empty
    monkeypatch.setattr(manual,"now_utc",lambda:START)
    assert commit(server,tmp_path,"restore",1)[0]==409
    monkeypatch.setattr(server,"check_pregame",lambda *_:None)
    assert commit(server,tmp_path,"restore",1)[0]==409
    assert manual.decisions(tmp_path)[EVENT]["action"]=="remove"


def test_exact_provider_fixture_rejects_live_cancelled_ambiguous_and_started(server,tmp_path,monkeypatch):
    row=seed(tmp_path).iloc[0]
    candidate={"score_provider_event_id":"501","event_time":START.isoformat(),"score_state":"pre",
               "score_status_code":"STATUS_SCHEDULED"}
    key=importlib.import_module("build_live_recent").game_key(row["game"],row["sport"])
    monkeypatch.setattr(server,"fetch_scoreboard",lambda *_a,**_k:({key:[candidate]},"available"))
    assert server.check_pregame(tmp_path,row)["id"]=="501"
    for phase,code in (("in","STATUS_IN_PROGRESS"),("pre","STATUS_POSTPONED"),("post","STATUS_CANCELED")):
        candidate.update(score_state=phase,score_status_code=code)
        assert server.check_pregame(tmp_path,row) is None


def test_http_owner_only_authorization_and_csrf(tmp_path,server,monkeypatch):
    marker=tmp_path/"enabled"
    marker.touch()
    monkeypatch.setattr(server,"ENABLE",marker)
    monkeypatch.setattr(server,"DATA",tmp_path)
    approved={**OWNER,"email_confirmed_at":"2025-01-01"}
    accounts={"owner":approved,"guest":{"id":"guest","email":OWNER["email"].replace("@","+redfoxguests@"),
                                      "email_confirmed_at":"2025-01-01","role":"admin"},
              "unverified":OWNER,"other":{"id":"other-admin","email":"another@example.test",
                                            "email_confirmed_at":"2025-01-01","role":"admin"}}
    monkeypatch.setattr(server,"verified_user",lambda token:accounts.get(token,{}))
    with server.ThreadingHTTPServer(("127.0.0.1",0),server.Handler) as http:
        thread=Thread(target=http.serve_forever,daemon=True)
        thread.start()
        url=f"http://127.0.0.1:{http.server_port}{server.ROUTE}"
        try:
            assert requests.get(url,timeout=2).status_code==401
            for identity in ("guest","other","unverified","invalid"):
                headers={"Authorization":"Bearer "+identity,"Origin":"https://redfoxmi.com",
                         "X-Owner-Email":OWNER["email"],"X-Forwarded-User":OWNER["email"]}
                assert requests.get(url,headers=headers,timeout=2).status_code==403
                assert requests.post(url,headers=headers,json={"email":OWNER["email"]},timeout=2).status_code==403
            owner={"Authorization":"Bearer owner"}
            assert requests.get(url,headers=owner,timeout=2).json()=={"enabled":True}
            assert requests.post(url,headers={**owner,"Origin":"https://evil.example"},json={},timeout=2).status_code==403
            assert requests.post(url,headers=owner,json={},timeout=2).status_code==403
            assert requests.post(url,headers={**owner,"Origin":"https://redfoxmi.com",
                                "Sec-Fetch-Site":"same-site"},json={},timeout=2).status_code==403
            assert requests.post(url,headers={**owner,"Origin":"https://redfoxmi.com"},
                                 data="{}",timeout=2).status_code==400
            marker.unlink()
            assert requests.get(url,headers=owner,timeout=2).status_code==404
        finally:
            http.shutdown()
            thread.join(timeout=2)


def test_dashboard_buttons_are_only_inside_existing_popups():
    page=Path("site/board.html").read_text(encoding="utf-8")
    script=Path("site/manual-favorites.js").read_text(encoding="utf-8-sig")
    assert page.count("manualFavoriteButton(")==2
    assert "if(!state.frozenPregame) html+=manualFavoriteButton(r);" in page
    assert "if(!manualFavoriteEnabled||!row?.game_id) return '';" in script
    assert "response?.ok&&(await response.json()).enabled===true" in script


def test_concurrent_owner_edits_have_one_winner(tmp_path,server,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    seed(tmp_path,favorite=False)
    monkeypatch.setattr(server,"check_pregame",lambda *_:fixture())
    offers=[selection(tmp_path,"MONEYLINE",name)["token"] for name in ("Away","Home")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses=list(pool.map(lambda token:commit(server,tmp_path,"add",0,token)[0],offers))
    assert sorted(responses)==[200,409]
    assert manual.decisions(tmp_path)[EVENT]["revision"]==1


def test_ufc_handoff_requires_actual_individual_start_and_pinned_identity():
    proof=dict(bout_id="99",card_id="88",pair=["away","home"],athlete_ids=["1","2"],phase="pre",risks=[])
    decision={"fixture":{"proof":proof,"pre_observed":NOW.isoformat()},
              "quote":{"observed":(NOW-timedelta(minutes=1)).isoformat()},
              "saved_at":NOW.isoformat(),"board_start":(NOW-timedelta(hours=2)).isoformat()}
    bout=dict(id="99",card_id="88",pair=["away","home"],athlete_ids=["1","2"],phase="live",
              start=START.isoformat(),observed=START.isoformat(),risk_names=[])
    assert manual.ufc_handoff(decision,None,{"99":bout},{},START)[0]==START
    assert manual.ufc_handoff(decision,None,{"99":dict(bout,phase="pre")},{},START) is None
    assert manual.ufc_handoff(decision,None,{"99":dict(bout,athlete_ids=["1","3"])},{},START) is None
    assert manual.ufc_handoff(decision,None,{"99":dict(bout,start=NOW.isoformat())},{},START) is None
    assert manual.ufc_handoff(decision,None,{"99":dict(bout,start="")},{},START) is None
    assert manual.ufc_handoff(decision,None,{"99":dict(bout,phase="void",kind="void")},{},START)[1]["score_state"]=="post"


def test_ufc_saves_from_private_provider_proof_without_exposing_it_on_board(tmp_path,server,monkeypatch):
    from collect_ufc_bouts import __name__ as collector_name
    collector=importlib.import_module(collector_name)
    proof=dict(bout_id="99",card_id="88",pair=["away","home"],athlete_ids=["1","2"],phase="pre",risks=[])
    bout=dict(id="99",card_id="88",pair=["away","home"],athlete_ids=["1","2"],phase="pre",
              card_at=START.isoformat(),observed=NOW.isoformat(),risks=[])
    row=dict(sport="ufc",game_id="101",game="Away vs Home",market_display="MONEYLINE",kickoff_iso=START.isoformat())
    pd.DataFrame([dict(row,_ufc_market_proof=json.dumps(proof))]).to_csv(tmp_path/"live_recent_market_candidates.csv",index=False)
    monkeypatch.setattr(server.ufc,"settings",lambda:(True,None))
    monkeypatch.setattr(server.ufc,"load_context",lambda *_:({"99":bout},{}))
    monkeypatch.setattr(server.ufc,"match_bout",lambda *_:bout)
    monkeypatch.setattr(collector,"get",lambda *a,**k:type("Response",(),{"json":lambda self:{"events":[]}})())
    fresh=dict(bout)
    monkeypatch.setattr(collector,"build_bouts",lambda *_:{"99":fresh})
    assert server.check_pregame(tmp_path,row)["id"]=="99"
    fresh["phase"]="live"
    assert server.check_pregame(tmp_path,row) is None
