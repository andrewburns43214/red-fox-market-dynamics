"""Prospective NHL paper entries and grading, separate from official Favorites."""
import json
import re

import pandas as pd

from build_live_recent import favorite_handoff_states
from favorite_nhl_goalies import _atomic, goalie_evidence, paper_dir, read_league_cache
from favorite_split_evidence import capture as capture_review_evidence, matchup_rows
from nhl_release_config import enabled, switches
from nhl_release_policy import (PUBLIC, CROSSOVER, advance, assess, fair, first_confirmed_by_pathway,
                                first_confirmed_lane, handoff_issue)
from performance_ledger import _atomic_csv, _attach_results, _frozen_records, _read_csv
from red_fox_favorite import _line_value, _sides, _read_tracking, update_favorite_tracking
from team_aliases import normalize_team_name

PAPER_FLAGS = {PUBLIC:"RED_FOX_NHL_PAPER_PUBLIC", CROSSOVER:"RED_FOX_NHL_PAPER_CROSSOVER"}
LEGACY_FLAG = "RED_FOX_NHL_LEGACY_SHADOW"
STATE = "replacement_paper_state.json"


def stamp(value):
    return pd.to_datetime(value, utc=True, errors="coerce")


def team(name):
    return normalize_team_name(re.sub(r"(?i)^st\. louis", "St Louis", name), "nhl")


def league_team(row):
    full = row.get("name", {}).get("default")
    place = row.get("placeName", {}).get("default", "")
    common = row.get("commonName", {}).get("default", "")
    return team(f"{place} {common}".strip() if place and common else f"{row.get('abbrev','')} {full or ''}".strip())


def index_schedule(cache, now):
    games = {}
    for block in cache.values():
        observed = stamp(block["observed_at"])
        if pd.isna(observed) or not pd.Timedelta(0) <= now-observed <= pd.Timedelta(minutes=12):
            continue
        for record in block["games"]:
            if record.get("gameType") != 2:
                continue
            away,home = (league_team(record[pos+"Team"]) for pos in ("away","home"))
            start=stamp(record.get("startTimeUTC"))
            if not away or not home or away==home or pd.isna(start):
                continue
            games[str(record["id"])] = dict(away=away,home=home,start=int(start.timestamp()),
                state=record.get("gameState"),away_score=record["awayTeam"].get("score"),
                home_score=record["homeTeam"].get("score"))
    return games


def paired_opener(history, away, home):
    valid=history.assign(_clock=stamp_series(history["timestamp"])).dropna(subset=["_clock"])
    for at, group in valid.sort_values("_clock",kind="stable").groupby("_clock"):
        pair={}
        conflicts=False
        for raw in group.to_dict("records"):
            name=team(raw["side"])
            if name not in (away,home):
                continue
            quote=(_line_value(raw["open_line"],"MONEYLINE"),_line_value(raw["current_line"],"MONEYLINE"))
            if name in pair and pair[name]!=quote:
                conflicts=True
            pair[name]=quote
        if (not conflicts and set(pair)=={away,home} and fair(pair[away][1],pair[home][1]) is not None
                and .985 <= sum(1 - 1/(1+abs(p)/100) if p<0 else 1/(1+p/100)
                                for p in (pair[away][1],pair[home][1])) <= 1.20):
            return {name:fair(pair[name][0],pair[other][0]) for name,other in ((away,home),(home,away))}
    return {}


def stamp_series(values):
    return pd.to_datetime(values, utc=True, format="mixed", errors="coerce")


def unique_pairs(row, away, home):
    sides=_sides(row)
    names=[team(str(x.get("flagged_side",""))) for x in sides]
    if len(names)!=2 or set(names)!={away,home}:
        return {}
    seconds=[stamp(x.get("source_latest_at")) for x in sides]
    if any(pd.isna(t) for t in seconds) or seconds[0]!=seconds[1]:
        return {}
    return dict(zip(names,sides))


def record_paper(board, history, cache, schedule, saved, active, observed, *,
                 official_since=None, blocked_games=frozenset()):
    second=int(observed.timestamp())
    games_by_vendor = matchup_rows(board) if official_since is not None else {}
    if not board.empty:
        current=board[board.sport.eq("nhl")&board.market_display.eq("MONEYLINE")]
        for row in current.to_dict("records"):
            match=str(row["game"]).split(" @ ")
            if len(match)!=2:
                continue
            away,home=map(team,match)
            kickoff=stamp(row.get("kickoff_iso"))
            if pd.isna(kickoff):
                continue
            vendor=int(kickoff.timestamp())
            matches=[(key, g) for key,g in schedule.items()
                     if (g["away"],g["home"])==(away,home) and abs(vendor-g["start"])<=4*3600]
            if len(matches)!=1:
                continue
            key,g=matches[0]
            if key in blocked_games:
                continue
            sides=unique_pairs(row,away,home)
            if not sides:
                continue
            at=stamp(sides[away]["source_latest_at"])
            if (not pd.Timedelta(0)<=observed-at<=pd.Timedelta(minutes=12)
                    or (official_since is not None and at < official_since)):
                continue
            clock=int(at.timestamp())
            rows=history[(history.sport=="nhl")&(history.game_id==str(row["game_id"]))&
                         (history.market_display=="MONEYLINE")]
            earliest=stamp_series(rows["dk_start_iso"]).min()
            cutoff=min(g["start"],vendor,int(earliest.timestamp()) if pd.notna(earliest) else vendor)
            if clock>=cutoff or second>=cutoff:
                continue
            game=saved.setdefault(key,{"vendor_id":str(row["game_id"]), "game":row["game"], "cutoff":cutoff,
                                       "lanes":{}, "events":[]})
            if game["vendor_id"]!=str(row["game_id"]) or game.get("finalized"):
                continue
            game["cutoff"]=min(game["cutoff"],cutoff)
            if clock>=game["cutoff"] or second>=game["cutoff"]:
                continue
            remaining=(vendor-clock)/60
            potential=40<remaining<=1440
            if not potential and not game["lanes"]:
                continue
            if not game.get("open"):
                game["open"]=paired_opener(rows,away,home)
            lanes = active
            if official_since is not None:
                bound = game["lanes"].get(game.get("selected_lane"), {})
                if bound.get("entry"):
                    lanes = active | {bound["pathway"]}
            for lane in sorted(lanes):
                for selected in (away,home):
                    side=sides[selected]
                    price=_line_value(side.get("current_line"),"MONEYLINE")
                    if price is None or abs(price)<100:
                        continue
                    lane_key=f"{lane}|{selected}"
                    if lane not in active and lane_key != game.get("selected_lane"):
                        continue
                    state=game["lanes"].setdefault(lane_key,{"pathway":lane,"team":selected})
                    status=goalie_evidence(cache,row["game"],kickoff,observed,previous_pair=state.get("known_goalies"))
                    if status.get("status")=="both_confirmed":
                        state["known_goalies"]=status["confirmed_pair"]
                    quote=dict(at=clock,price=price,side=selected,goalie_status=status["status"],
                               bets=side.get("bets_pct"),money=side.get("money_pct"))
                    if official_since is not None:
                        quote.update(source_at=at.isoformat(), line=str(side["current_line"]),
                                     team_name=side["flagged_side"])
                    goalie_change = status["status"] == "verified_change" and state.get("entry") and not state.get("removed")
                    if goalie_change:
                        state["removed"] = issue = "Verified confirmed goalie pair changed"
                        stage = "removed"
                    else:
                        issue, hard = assess(row, side["flagged_side"], lane, at, game["open"].get(selected), status)
                        pending_start = (official_since is not None and not state.get("entry")
                                         and 40 < remaining <= 1440 and clock > state.get("last", 0)
                                         and any(0 < clock - previous <= 1800 for previous in state.get("pending", ())))
                        stage = advance(state, clock, remaining, quote, issue, hard)
                        if pending_start and issue and not state.get("pending") and not stage:
                            stage = "initial_confirmation_failed"
                    if stage:
                        event = dict(lane=lane, stage=stage, quote=quote, reason=state.get("removed", "") or issue)
                        if goalie_change:
                            event["evaluated_at"] = second
                        if official_since is not None:
                            identity = {"favorite_side": side["flagged_side"], "favorite_pathway": lane}
                            phase = "removal" if stage == "removed" else "initial_confirmation"
                            evidence = capture_review_evidence(
                                row, identity, games_by_vendor.get(("nhl", row["game_id"])), phase=phase,
                            )
                            evidence["outcome_lookup"].update(
                                archive="var/nhl-qualification/replacement_official_state.json", league_event_id=key,
                            )
                            evidence["published_lane_at_capture"] = game.get("selected_lane") == lane_key
                            event["decision_evidence"] = evidence
                        game["events"].append(event)
            if "selected_lane" not in game and (winner:=first_confirmed_lane(game["lanes"])) is not None:
                game["selected_lane"]=winner
    for key,game in saved.items():
        if not game.get("finalized") and second>=game["cutoff"]:
            for state in game["lanes"].values():
                if state.get("entry") and not state.get("removed"):
                    continuing = official_since is not None and state is game["lanes"].get(game.get("selected_lane"))
                    reason=(handoff_issue(state,game["cutoff"]) if state["pathway"] in active or continuing
                            else "Pathway disabled before handoff")
                    if reason:
                        state["removed"]=reason
                    else:
                        state["handoff"]=True
            game["finalized"]=True
        score=schedule.get(key,{})
        a,b=score.get("away_score"),score.get("home_score")
        if not game.get("result") and game.get("finalized") and score.get("state") in ("OFF","FINAL") and isinstance(a,int) and isinstance(b,int) and a!=b:
            game["result"]={"winning_team":score["away"] if a>b else score["home"],"away":a,"home":b}
    return saved


def paper_totals(saved):
    totals={}
    entries=[]
    for key,game in saved.items():
        # Match replay attribution: the first confirmed side owns its lane
        # even if it later fails handoff. A later side cannot replace it.
        lane_winners = first_confirmed_by_pathway(game["lanes"])
        for lane,state in game["lanes"].items():
            if not state.get("handoff") or lane_winners.get(state["pathway"]) != lane:
                continue
            entry=state["entry"]
            selected=lane==game.get("selected_lane")
            result=game.get("result")
            units=None
            if result is not None:
                odds=entry["price"]
                units=(odds/100 if odds>0 else -100/odds) if result["winning_team"]==entry["side"] else -1
            entries.append(dict(provider_event_id=key,vendor_id=game["vendor_id"],game=game["game"],
                                pathway=state["pathway"],combined_selected=selected,net_units=units,**entry))
    for lane in (*PAPER_FLAGS,"combined"):
        rows=[r for r in entries if r["pathway"]==lane or (lane=="combined" and r["combined_selected"])]
        graded=[r["net_units"] for r in rows if r["net_units"] is not None]
        totals[lane]=dict(entries=len(rows),wins=sum(p>0 for p in graded),losses=sum(p<0 for p in graded),
                          net_units=sum(graded),roi=sum(graded)/len(graded) if graded else None)
    return dict(paper_only=True, totals=totals, entries=entries)


def capture_legacy(raw, data_dir, at, where, *, deferred_markets=frozenset(), unavailable_sports=frozenset()):
    where=where/"legacy_hypothetical"
    where.mkdir(exist_ok=True,mode=0o700)
    update_favorite_tracking(raw[raw.sport.eq("nhl")], where, as_of=at, nhl_hypothetical=True,
                             deferred_markets=deferred_markets, unavailable_sports=unavailable_sports)
    tracking=_read_tracking(where/"red_fox_favorite_tracking.csv")
    candidates=_read_csv(where/"red_fox_favorite_freeze_candidates.csv")
    ledger_path=where/"hypothetical_results.csv"
    ledger=_read_csv(ledger_path)
    if not candidates.empty:
        starts = stamp_series(candidates["kickoff_iso"])
        scoped = candidates.loc[starts.le(at)].copy().astype(object)
        markets = _read_csv(data_dir/"live_recent_market_candidates.csv")
        if not markets.empty:
            markets = markets.loc[markets["sport"].eq("nhl")]
        handoff=favorite_handoff_states(scoped,at,tracking,markets)
        known=set(ledger.get("ledger_id",[]))
        additions=[row for row in _frozen_records(handoff,tracking,"nhl legacy hypothetical")
                   if row["favorite_qualified"]=="yes" and row["ledger_id"] not in known]
        if additions:
            ledger=pd.concat([ledger,pd.DataFrame(additions)],ignore_index=True)
    if not ledger.empty:
        scores=_read_csv(data_dir/"final_scores_history.csv")
        if not scores.empty:
            ledger=_attach_results(ledger,scores[scores.game_id.isin(ledger.event_id)])
        _atomic_csv(ledger,ledger_path)
    totals={}
    groups=ledger.groupby("favorite_rule_version") if "favorite_rule_version" in ledger else []
    for version, cohort in groups:
        units=pd.to_numeric(cohort["net_profit_units"],errors="coerce").dropna()
        totals[version]=dict(entries=len(cohort),graded=len(units),wins=int(units.gt(0).sum()),
                             losses=int(units.lt(0).sum()),net_units=float(units.sum()),
                             roi=float(units.mean()) if len(units) else None)
    _atomic(where/"hypothetical_performance.json",totals)


def capture(board, raw, history, goalies, data_dir, at, *, deferred_markets=frozenset(),
            unavailable_sports=frozenset()):
    settings=switches()
    active={lane for lane,flag in PAPER_FLAGS.items() if enabled(flag,settings)}
    legacy=enabled(LEGACY_FLAG,settings)
    if not active and not legacy:
        return
    where=paper_dir(data_dir)
    if active:
        path=where/STATE
        saved=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        # Wall time governs provider freshness, while paired source time
        # governs qualification and confirmation.
        observed=pd.Timestamp.now(tz="UTC")
        schedule=index_schedule(read_league_cache(data_dir),observed)
        record_paper(board,history,goalies,schedule,saved,active,observed)
        _atomic(path,saved)
        output=paper_totals(saved)
        _atomic(where/"replacement_paper_performance.json",output)
        print(f"[nhl-paper] lanes={len(active)} schedule={len(schedule)} lifecycle_games={len(saved)}",flush=True)
    if legacy:
        capture_legacy(raw, data_dir, stamp(at), where, deferred_markets=deferred_markets,
                       unavailable_sports=unavailable_sports)
