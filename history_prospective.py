"""Capture and settle a fixed prospective comparison study off the live feed."""

from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

from history_comparisons import grade, implied, side_key
from history_evidence import fair_probability, lead_minutes
from history_store import packed, start_evidence, utc
from team_aliases import _split_game


SCHEMA = """
CREATE TABLE IF NOT EXISTS study_v1 (
    sport TEXT, fixture TEXT, market TEXT, event TEXT, scheduled TEXT, observed TEXT,
    published_at TEXT, next_check TEXT, entry TEXT, methods TEXT, ledger TEXT,
    grade TEXT, verified_at TEXT, result TEXT, PRIMARY KEY(sport,fixture,market)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS study_v1_review ON study_v1(next_check) WHERE grade IS NULL;
CREATE TABLE IF NOT EXISTS study_v1_proof (
    name TEXT PRIMARY KEY, payload BLOB NOT NULL
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS study_v1_provider (
    sport TEXT, market TEXT, source TEXT, provider_id TEXT,
    PRIMARY KEY(sport,market,source,provider_id)
) WITHOUT ROWID;
"""


class Study:
    def __init__(self, path, output):
        self.db = sqlite3.connect(path, timeout=2)
        Path(path).chmod(0o600)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.output = Path(output)

    def close(self):
        self.db.close()

    def save_summary(self, now):
        from history_store import atomic_text

        output = Path(self.db.execute("PRAGMA database_list").fetchone()[2]).with_suffix(".json")
        if output.exists() and now.timestamp()-output.stat().st_mtime < 3600:
            return
        with atomic_text(output, packed({"as_of":utc(now),"version":1,"results":self.summary(),
                                        "by_historical_roi":self.historical_roi_review()})):
            pass
        output.chmod(0o600)

    def observe(self, payload, now=None):
        if payload.get("frozen"):
            return
        made = utc(now if now is not None else datetime.now(timezone.utc))
        sport, event = payload["sport"], payload["event"]
        for market, setup in payload["markets"].items():
            scheduled, observed = setup["scheduled"], setup["observed"]
            lead = lead_minutes(made, scheduled)
            if not 40 <= lead <= 60 or not 60 <= lead_minutes(observed, scheduled) <= 80:
                continue
            game = payload["game"]
            fixture = event if sport == "mlb" else packed((
                sorted(side_key(team, sport, "MONEYLINE") for team in _split_game(game)), scheduled[:10],
            ))
            key = sport, fixture, market
            if self.db.execute("SELECT 1 FROM study_v1 WHERE (sport,fixture,market)=(?,?,?)", key).fetchone():
                continue
            lanes = setup["sides"]
            if market == "TOTAL":
                index = next(i for i, side in enumerate(lanes) if side["side"] == "over")
            else:
                # The stable choice does not depend on outcomes, match count
                # or the direction of the reference quotes.
                index = hashlib.sha256((sport+fixture).encode()).digest()[0] % 2
            side, opposite = lanes[index], lanes[1-index]
            fair = fair_probability(side, opposite)
            line, opposing_line = side.get("line"), opposite.get("line")
            if fair is None:
                continue
            if market != "MONEYLINE" and (line is None or opposing_line is None or
                    line != (-opposing_line if market == "SPREAD" else opposing_line)):
                continue
            ref = side["comparison"].get("ledger")
            document = None
            if ref:
                document = (self.output / (ref+".json.gz")).read_bytes()
            forecasts, counts = forecast(fair, document, market, side, opposite)
            receipt = {key: value for key, value in side.items() if key != "comparison"}
            receipt.update(game=game, participant_sides=[team["side"] for team in lanes],
                           paired_price=opposite["price"], verification_cutoff=setup.get("verified_by"),
                           neighbor_counts=counts, historical_roi={
                               group: {"graded":report["graded"],"roi":report["roi"]}
                               for group in ("closest","broader")
                               if (report := side["comparison"].get(group)) and report["graded"]
                           })
            with self.db:
                if document:
                    self.db.execute("INSERT OR IGNORE INTO study_v1_proof VALUES (?,?)", (ref, document))
                self.db.execute("""INSERT OR IGNORE INTO study_v1(
                    sport,fixture,market,event,scheduled,observed,published_at,next_check,entry,methods,ledger
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (*key, event, scheduled, observed, made, scheduled, packed(receipt), packed(forecasts), ref))

    def reconcile(self, history, now, limit=30):
        from history_public import outcome_signature

        stamp = utc(now)
        due = self.db.execute("SELECT * FROM study_v1 WHERE grade IS NULL AND next_check < ? ORDER BY next_check LIMIT ?",
                              (stamp,limit)).fetchall()
        for row in due:
            sport, market, event = (row[key] for key in ("sport","market","event"))
            key = sport, row["fixture"], market
            entry = json.loads(row["entry"])
            sources = [(source, status, provider, at, data)
                       for source, status, provider, at, data in history.evidence(sport,event,stamp) if provider]
            if sport == "ufc":
                sources = [item for item in sources if set(entry["participant_sides"]) == {
                    side_key(team, sport, "MONEYLINE") for team in _split_game(item[4].get("matchup"))}]
            grades = {grade(sport,market,entry,status,data) for _,status,_,_,data in sources}
            signatures = {outcome_signature(sport,status,data) for _,status,_,_,data in sources}
            actual = start_evidence(history.db,sport,event)
            result = None
            if actual and actual <= row["published_at"]:
                result = "after_verified_start"
            elif len(grades)==len(signatures)==1 and None not in grades|signatures:
                result = grades.pop()
            with self.db:
                if result:
                    ids = [(sport, market, "nhl-league" if source=="verified_research_nhl" else sport, provider)
                           for source,_,provider,_,_ in sources]
                    if result in ("win","loss","push"):
                        old = any(self.db.execute("SELECT 1 FROM study_v1_provider WHERE (sport,market,source,provider_id)=(?,?,?,?)",
                                                 item).fetchone() for item in ids)
                        if old:
                            result = "duplicate"
                        else:
                            self.db.executemany("INSERT OR IGNORE INTO study_v1_provider VALUES (?,?,?,?)", ids)
                    self.db.execute("""UPDATE study_v1 SET grade=?,verified_at=?,result=?
                        WHERE (sport,fixture,market)=(?,?,?)""",
                        (result,stamp,packed({"verified_sources":[[s,p,t] for s,_,p,t,_ in sources],
                                              "signatures":list(signatures),"actual_start":actual}),*key))
                else:
                    self.db.execute("UPDATE study_v1 SET next_check=? WHERE (sport,fixture,market)=(?,?,?)",
                                    (utc(now+timedelta(hours=2)),*key))

    def summary(self):
        results = {}
        for row in self.db.execute("SELECT sport,market,entry,methods,grade FROM study_v1"):
            cohort = results.setdefault(row["sport"]+" "+row["market"], {})
            forecasts=json.loads(row["methods"])
            entry=json.loads(row["entry"])
            price=implied(entry["price"])
            for recipe, forecast in forecasts.items():
                item = cohort.setdefault(recipe, {"queries":0,"adjusted":0,"decisive":0,"wins":0,"losses":0,"pushes":0,
                                                   "brier_sum":0,"paired_brier_delta":0,"quote_units":0})
                item["queries"] += 1
                item["adjusted"] += entry.get("neighbor_counts",{}).get(recipe,0)>=15
                grade = row["grade"]
                if grade in ("win","loss"):
                    item["decisive"] += 1
                    item["wins" if grade=="win" else "losses"] += 1
                    win = grade=="win"
                    item["quote_units"] += 1/price-1 if win else -1
                    brier = (forecast-win)**2
                    item["brier_sum"] += brier
                    item["paired_brier_delta"] += brier - (forecasts["no_vig"]-win)**2
                elif grade=="push":
                    item["pushes"] += 1
        for cohort in results.values():
            for item in cohort.values():
                n = item["decisive"]
                graded = n + item["pushes"]
                item["brier"] = item["brier_sum"]/n if n else None
                item["mean_brier_difference"] = item["paired_brier_delta"]/n if n else None
                item["quote_roi"] = item["quote_units"]/graded if graded else None
        return results

    def historical_roi_review(self):
        """Review the sign of the originally published cohort without changing its membership."""
        results, legacy = {}, {}
        for row in self.db.execute("SELECT sport,market,entry,methods,ledger,grade FROM study_v1"):
            entry, methods = json.loads(row["entry"]), json.loads(row["methods"])
            history_roi = entry.get("historical_roi")
            if history_roi is None:
                name = row["ledger"]
                if name not in legacy:
                    saved = self.db.execute("SELECT payload FROM study_v1_proof WHERE name=?", (name,)).fetchone()
                    legacy[name] = proof_roi(saved[0]) if saved else {}
                history_roi = legacy[name]

            for group in ("closest","broader"):
                sample = history_roi.get(group)
                signed = sample.get("roi") if sample else None
                bucket = ("insufficient_history" if signed is None or sample["graded"] < 15 else
                          "positive" if signed > 0 else "negative" if signed < 0 else "flat")
                key = row["sport"]+" "+row["market"]
                result = results.setdefault(key, {}).setdefault(group, {}).setdefault(bucket, {
                    "captured":0,"wins":0,"losses":0,"pushes":0,"graded":0,
                    "units":0.0,"expected_units_no_vig":0.0,"expected_wins_no_vig":0.0,"brier_delta_sum":0.0,
                })
                result["captured"] += 1
                outcome = row["grade"]
                if outcome not in ("win","loss","push"):
                    continue
                result["graded"] += 1
                result[{"win":"wins","loss":"losses","push":"pushes"}[outcome]] += 1
                if outcome == "push":
                    continue
                fair = methods["no_vig"]
                observed = outcome=="win"
                payout = 1/implied(entry["price"])
                result["units"] += payout*observed-1
                result["expected_units_no_vig"] += payout*fair-1
                result["expected_wins_no_vig"] += fair
                result["brier_delta_sum"] += (methods[group]-observed)**2-(fair-observed)**2

        for sport_market in results.values():
            for group in sport_market.values():
                for item in group.values():
                    n = item["graded"]
                    decisive = item["wins"]+item["losses"]
                    item["observed_roi"] = item["units"]/n if n else None
                    item["expected_roi_no_vig"] = item["expected_units_no_vig"]/n if n else None
                    item["roi_minus_no_vig"] = (item["units"]-item["expected_units_no_vig"])/n if n else None
                    item["mean_brier_difference"] = item["brier_delta_sum"]/decisive if decisive else None
        return results


def proof_roi(document):
    """Use immutable publication evidence for receipts captured before the ROI field existed."""
    ledger = json.loads(gzip.decompress(document))
    result = {}
    for group in ("closest","broader"):
        rows = [ledger["rows"][i] for i in ledger["groups"][group]]
        if rows:
            units = sum(1/implied(row["side"]["price"])-1 if row["grade"]=="win" else
                        -1 if row["grade"]=="loss" else 0 for row in rows)
            result[group] = {"graded":len(rows),"roi":round(100*units/len(rows),2)}
    return result


def signature(market, side, opposite, peers):
    def direction(key, quote):
        value = quote.get("price_toward_implied" if key == "MONEYLINE" else "line_toward")
        threshold = .005 if key == "MONEYLINE" else .25
        return None if value is None else 0 if abs(value)<threshold else 1 if value>0 else -1

    def role(value):
        return None if value is None else (value>0)-(value<0)

    if market == "MONEYLINE":
        def strength(field):
            a, b = implied(side.get(field)), implied(opposite.get(field))
            return None if a is None or b is None else role(a-b)
        roles = strength("open_price"),strength("price")
    else:
        reference = peers.get("SPREAD",{}) if market == "TOTAL" else side
        roles = role(reference.get("open_line")),role(reference.get("line"))
    spread = side if market=="SPREAD" else peers.get("SPREAD",{})
    moneyline = side if market=="MONEYLINE" else peers.get("MONEYLINE",{})
    total = side if market=="TOTAL" else peers.get("TOTAL",{})
    return ((*roles,direction(market,side)),
            (direction("SPREAD",spread),direction("MONEYLINE",moneyline),direction("TOTAL",total)))


def forecast(fair, document, market, side, opposite):
    methods = {"no_vig": fair}
    counts = {}
    ledger = json.loads(gzip.decompress(document)) if document else {"rows":[], "groups":{}}
    primary, context = signature(market,side,opposite,side.get("game_context",{}))
    for recipe, cohort in (("closest","closest"),("broader","broader"),
                           ("same_role_direction","broader"),("same_joint_context","broader")):
        residuals = []
        for index in ledger["groups"].get(cohort, []):
            row = ledger["rows"][index]
            if recipe.startswith("same_"):
                prior_primary, prior_context = signature(market,row["side"],row["opposite"],row["peers"])
                if None in primary or prior_primary != primary:
                    continue
                if recipe=="same_joint_context" and (None in context or prior_context!=context):
                    continue
            prior = fair_probability(row["side"],row["opposite"])
            if prior is not None and row["grade"] in ("win","loss"):
                residuals.append((row["grade"]=="win") - prior)
            if len(residuals)==100:
                break
        methods[recipe] = min(.99,max(.01,fair+sum(residuals)/(len(residuals)+20))) if len(residuals)>=15 else fair
        counts[recipe] = len(residuals)
    return methods, counts
