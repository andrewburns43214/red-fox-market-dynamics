from datetime import datetime, timedelta, timezone
import gzip
import json

from history_evidence import EvidenceStore, Ledger
from history_prospective import Study, proof_roi
from history_public import History
from history_store import connect, store_outcome, utc
import pytest


def test_complete_ledger_preserves_count_price_result_and_frozen_links(tmp_path):
    now = datetime(2026, 11, 27, 20, tzinfo=timezone.utc)
    output = tmp_path / "public"
    output.mkdir()
    stamp = utc(now)
    quote = {"side":"club a","line":-3.5,"open_line":-1.5,"price":-115,"open_price":-110,
             "tickets":48,"money":41,"line_toward":2,"game_context":{"MONEYLINE":{"price":-162}}}
    opponent = quote | {"side":"club b","line":3.5,"price":-105}
    previous = []
    for i in range(72):
        matchup = f"Club A @ Club B {i}"
        row = {"sport":"nfl","event_id":str(i),"matchup":matchup,"entry_at":stamp,"scheduled_at":stamp,
               "sides":[quote,opponent]}
        previous.append((.1,str(i),row,quote,i%2==0,[("Move size",0,1)]))
    def evidence(sport, event, before):
        return [("test","verified","game-"+event,stamp,{
            "team_a":"Club A","team_b":"Club B", "score_a":24,"score_b":20,"matchup":"Club A @ Club B"})]
    with connect(tmp_path / "db") as db:
        history = History(db)
        history.evidence = evidence
        ledger = Ledger()
        reports = {}
        for group in ("closest","broader"):
            rows = [x for x in previous if group=="broader" or x[4]]
            reports[group] = history.report(rows,"SPREAD",stamp,ledger=ledger,group=group)
        store = EvidenceStore(output,now)
        name = ledger.publish(store,quote,"nfl","test","SPREAD",{"observed":stamp},reports)
        path = output / (name+".json.gz")
        body = json.loads(gzip.decompress(path.read_bytes()))
        assert reports["broader"]["graded"] == reports["broader"]["win"] == len(body["groups"]["broader"]) == 72
        assert reports["closest"]["graded"] == len(body["groups"]["closest"]) == 36
        assert len(body["rows"]) == 72
        assert sum(100/abs(body["rows"][i]["side"]["price"]) for i in body["groups"]["broader"]) == pytest.approx(
            reports["broader"]["units"], abs=.005)
        assert body["rows"][-1]["peers"]["MONEYLINE"]["price"] == -162
        frozen = {"markets":{"SPREAD":{"sides":[{"comparison":{"ledger":name}}]}}}
        store.pin(frozen)
        stale = EvidenceStore(output,now+timedelta(hours=2))
        old = path.read_bytes()
        stale.retire()
        assert not path.exists()
        stale.retain(frozen)
        assert old == path.read_bytes()


def test_prospective_receipt_keeps_original_quote_and_requires_verified_grade(tmp_path):
    now = datetime(2026, 11, 27, 20, tzinfo=timezone.utc)
    starts = utc(now+timedelta(minutes=55))
    capture = utc(now-timedelta(minutes=10))
    base = {"sport":"nfl","event":"trial","game":"Club A @ Club B","markets":{"SPREAD":{
        "scheduled": starts,"observed":capture,"verified_by":utc(now),
        "sides":[{"side":"club a","price":-110,"open_price":-110,"line":-3.5,"comparison":{}},
                 {"side":"club b","price":-110,"open_price":-110,"line":3.5,"comparison":{}}]}}}
    study = Study(tmp_path/"study.sqlite3", tmp_path)
    try:
        study.observe(base,now-timedelta(minutes=20))
        assert study.db.execute("SELECT count(*) FROM study_v1").fetchone()[0] == 0
        study.observe(base,now)
        study.observe(base,now+timedelta(minutes=1))
        assert study.db.execute("SELECT count(*) FROM study_v1").fetchone()[0] == 1
        with connect(tmp_path/"db") as db:
            history = History(db)
            then = now+timedelta(hours=4)
            study.reconcile(history,then)
            assert study.db.execute("SELECT grade FROM study_v1").fetchone()[0] is None
            store_outcome(db,"nfl","trial","test","verified",{
                "provider_event_id":"real","matchup":"Club A @ Club B","score_away":24,"score_home":17,
                "observed_at":utc(then+timedelta(minutes=5))})
            study.reconcile(History(db),then+timedelta(hours=3))
            assert study.db.execute("SELECT grade FROM study_v1").fetchone()[0] in ("win","loss")
            summary = study.summary()["nfl SPREAD"]
            assert summary["closest"]["decisive"] == summary["no_vig"]["decisive"] == 1
            assert summary["closest"]["paired_brier_delta"] == 0
    finally:
        study.close()


def test_historical_roi_sign_uses_fixed_receipts_and_paired_market_baseline(tmp_path):
    old_proof = gzip.compress(json.dumps({
        "rows":[{"grade":"loss","side":{"price":-110}}]*15,
        "groups":{"closest":list(range(15)),"broader":list(range(15))},
    }).encode())
    assert proof_roi(old_proof)["closest"] == {"graded":15,"roi":-100}
    study = Study(tmp_path/"study.sqlite3", tmp_path)
    try:
        methods = json.dumps({"no_vig":.5,"closest":.6,"broader":.6})
        with study.db:
            study.db.execute("INSERT INTO study_v1_proof VALUES (?,?)", ("old",old_proof))
            for fixture, entry, proof, grade in (
                ("positive", {"price":-110,"historical_roi":{"closest":{"graded":15,"roi":8.1}}}, None, "win"),
                ("legacy", {"price":-110}, "old", "loss"),
                ("push", {"price":-110,"historical_roi":{"closest":{"graded":15,"roi":8.1}}}, None, "push"),
                ("sparse", {"price":-110,"historical_roi":{"closest":{"graded":14,"roi":12}}}, None, None),
            ):
                study.db.execute("""INSERT INTO study_v1(sport,fixture,market,entry,methods,ledger,grade)
                                 VALUES ('nba',?,'SPREAD',?,?,?,?)""", (fixture,json.dumps(entry),methods,proof,grade))
        result = study.historical_roi_review()["nba SPREAD"]["closest"]
        assert result["negative"]["losses"] == 1
        assert result["insufficient_history"]["captured"] == 1
        assert result["insufficient_history"]["observed_roi"] is None
        positive = result["positive"]
        assert (positive["wins"],positive["pushes"]) == (1,1)
        assert positive["expected_wins_no_vig"] == .5
        assert positive["observed_roi"] == pytest.approx((100/110)/2)
        assert positive["roi_minus_no_vig"] == pytest.approx((1+100/110)/4)
        assert positive["mean_brier_difference"] == pytest.approx(.16-.25)
    finally:
        study.close()
