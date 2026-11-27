"""Export verified fixtures from the sealed Red Fox research datasets."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3


def read_rows(path, query):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        yield from db.execute(query)


def start(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat() if epoch else None


def record(sport, event_id, provenance, **evidence):
    return dict(sport=sport, event_id=str(event_id), source="verified_research_" + sport,
                status="verified", research_provenance=provenance, **evidence)


def all_results(project):
    worktrees = project / ".worktrees"
    stage = project / ".redfox-research-20261124" / "stage4"
    for sport, file in (
        ("nba", "nba_authoritative_v1.sqlite3"), ("ncaaf", "cfb_authoritative_v2.sqlite3"),
        ("ncaab", "ncaab_authoritative_v3.sqlite3"),
    ):
        for r in read_rows(stage / file, "SELECT * FROM event WHERE score_a IS NOT NULL AND score_b IS NOT NULL"):
            row = dict(r)
            provider = re.search(r"/gameId/(\d+)", row["score_url"] or "")
            yield record(sport, row["id"], row["score_provenance"], matchup=row["matchup"],
                         team_a=row["a"], team_b=row["b"], score_a=row["score_a"], score_b=row["score_b"],
                         provider_event_id=provider[1] if provider else None, url=row["score_url"],
                         fixture_scheduled_at=start(row["kickoff"]),
                         provider_start=start(row.get("provider_schedule")),
                         settlement_review=bool(row.get("exclude_wagers")))
    nhl = worktrees / "nhl-historical-qualification" / "nhl-assessment" / "nhl_authoritative_v2.sqlite3"
    for r in read_rows(nhl, "SELECT * FROM games WHERE score_away IS NOT NULL AND score_home IS NOT NULL"):
        yield record("nhl", r["event"], r["local_final"], matchup=r["game"],
                     score_away=r["score_away"], score_home=r["score_home"],
                     provider_event_id=str(r["league_id"]), url=r["final_source"],
                     provider_start=start(r["league_start"]), final_period=r["final_period"],
                     fixture_scheduled_at=start(r["kickoff"]))
    mlb = worktrees / "mlb-historical-qualification" / "mlb_research" / "deliverables" / "mlb_verified_v3.sqlite3"
    for r in read_rows(mlb, "SELECT * FROM event WHERE game_pk IS NOT NULL AND score_away IS NOT NULL AND score_home IS NOT NULL"):
        if not (r["ml_gradeable"] or r["sp_gradeable"]):
            continue
        yield record("mlb", r["event"], r["score_provenance"], matchup=r["game"],
                     away_team_id=r["away"], home_team_id=r["home"], score_away=r["score_away"],
                     score_home=r["score_home"], provider_event_id=str(r["game_pk"]), url=r["url"],
                     fixture_scheduled_at=start(r["kickoff"]), provider_start=start(r["official_start"]),
                     actual_start=start(r["official_first_pitch"]), game_number=r["game_number"],
                     doubleheader=r["double_header"], innings=r["innings"], moneyline_gradeable=bool(r["ml_gradeable"]),
                     spread_gradeable=bool(r["sp_gradeable"]), status_code=r["official_state"])
    ufc = worktrees / "ufc-final-qualification" / "ufc-research" / "package" / "bouts.csv"
    with ufc.open(encoding="utf-8-sig", newline="") as source:
        for r in csv.DictReader(source):
            if not r["vendor_id"] or r["authoritative"].lower() != "true":
                continue
            status = r["result_status"].lower()
            state = {"final": "verified", "draw": "push", "push": "push", "void": "void"}.get(status)
            if not state or state == "verified" and not r["winner"]:
                continue
            row = record("ufc", r["vendor_id"], r["outcome_evidence"], matchup=r["market_game"],
                         winner=r["winner"], athlete_ids=[r["athlete_a"], r["athlete_b"]],
                         names=[r["display_a"], r["display_b"]], card_id=r["card_id"],
                         provider_event_id=r["bout_id"], actual_start=r["actual_start"], url=r["official_url"],
                         start_source=r["actual_clock_source"], void_reason=r["void_reason"],
                         fixture_scheduled_at=r["dk_scheduled_start"])
            yield row | {"status": state}
    nfl = json.loads((stage / "nfl_nonfavorites_audit.json").read_text(encoding="utf-8-sig"))
    for r in nfl.get("supplemental_contrarian_checks", ()):
        scores = r.get("score")
        provider = re.search(r"/gameId/(\d+)", r.get("source") or "")
        if scores and len(scores) == 2 and provider:
            yield record("nfl", r["event"], "independent ESPN score reconciliation", matchup=r["game"],
                         team_a=r["team"], team_b=r["opponent"], score_a=scores[0], score_b=scores[1],
                         provider_event_id=provider[1], url=r["source"], fixture_scheduled_at=start(r["kickoff"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("w", newline="\n", encoding="utf-8") as file:
        for row in all_results(args.project):
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
