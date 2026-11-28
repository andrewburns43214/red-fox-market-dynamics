"""Build cached Team Intelligence from read-only Red Fox evidence."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
import gzip
import io
import json
from pathlib import Path
import re
import shutil
import sqlite3
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from history_blocks import decode
from history_comparisons import WINDOWS, grade, side_key
from history_public import outcome_signature
from history_store import FREE_DISK_RESERVE, atomic_text, latest_provider_outcomes, packed, utc
from team_aliases import _split_game
from team_independent_sources import independent_context, updated_credentials
from team_sport_context import baseball_profile_context, cfb_profile_context, nfl_profile_context, ufc_profile_context

SPORTS = ("nfl", "ncaaf", "nba", "ncaab", "mlb", "nhl", "ufc")
MARKETS = ("SPREAD", "MONEYLINE", "TOTAL")
NFL_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
ET = ZoneInfo("America/New_York")
TOKEN = re.compile(r"[A-Za-z0-9_.%+-]{1,90}\Z")
NFL_NAMES = dict(ARI="Arizona Cardinals", ATL="Atlanta Falcons", BAL="Baltimore Ravens",
    BUF="Buffalo Bills", CAR="Carolina Panthers", CHI="Chicago Bears", CIN="Cincinnati Bengals",
    CLE="Cleveland Browns", DAL="Dallas Cowboys", DEN="Denver Broncos", DET="Detroit Lions",
    GB="Green Bay Packers", HOU="Houston Texans", IND="Indianapolis Colts", JAX="Jacksonville Jaguars",
    KC="Kansas City Chiefs", LAC="LA Chargers", LA="LA Rams", LAR="LA Rams", LV="Las Vegas Raiders",
    MIA="Miami Dolphins", MIN="Minnesota Vikings", NE="New England Patriots", NO="New Orleans Saints",
    NYG="NY Giants", NYJ="NY Jets", PHI="Philadelphia Eagles", PIT="Pittsburgh Steelers",
    SEA="Seattle Seahawks", SF="San Francisco 49ers", TB="Tampa Bay Buccaneers",
    TEN="Tennessee Titans", WAS="Washington Commanders")


def clock():
    return datetime.now(timezone.utc)


def atomic(path, value):
    value = packed(value)
    if path.is_file() and path.read_text(encoding="utf-8") == value:
        return
    with atomic_text(path, value):
        pass
    path.chmod(0o644)


def running(lock):
    import fcntl

    try:
        with lock.open("rb") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
    except FileNotFoundError:
        return False
    return False


def slot(lock):
    now = clock()
    return not (now.minute % 5 == 4 and now.second >= 42) and not running(lock)


def identities(game, sport):
    names = _split_game(game)
    keys = [side_key(n, sport, "MONEYLINE") for n in names]
    return (names, keys) if all(keys) and len(keys) == 2 and keys[0] != keys[1] else (None, None)


def season(sport, start):
    date = datetime.fromisoformat(start).astimezone(ET)
    return date.year - (sport in {"nfl", "ncaaf"} and date.month <= 2 or
                        sport in {"nba", "nhl", "ncaab"} and date.month < 8)


def number(value):
    try:
        n = float(value)
        return int(n) if n == int(n) and 0 <= n <= 400 else None
    except (TypeError, ValueError, OverflowError):
        return None


def nfl_schedule(cache, seconds, permit_network):
    path = cache / "nflverse.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current_schema = not stored.get("games") or "codes" in stored["games"][0]
    if not permit_network or (current_schema and (clock().timestamp() - stored.get("checked", 0)) < 10800):
        return stored
    try:
        headers = {"User-Agent": "RedFox-Team-Intelligence/1", "Accept-Encoding": "gzip"}
        if current_schema and stored.get("etag"):
            headers["If-None-Match"] = stored["etag"]
        with urlopen(Request(NFL_URL, headers=headers), timeout=seconds) as response:
            body = response.read(6_000_001)
            if len(body) > 6_000_000:
                raise ValueError("nflverse response exceeds size limit")
            if response.headers.get("Content-Encoding") == "gzip" or body.startswith(b"\x1f\x8b"):
                with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream:
                    body = stream.read(9_000_001)
            if len(body) > 9_000_000:
                raise ValueError("nflverse decoded response exceeds size limit")
            rows = csv.DictReader(io.StringIO(body.decode("utf-8-sig")))
            if not {"game_id", "gameday", "gametime", "away_score", "home_score", "espn"}.issubset(rows.fieldnames or ()):
                raise ValueError("nflverse schedule columns changed")
            games = [x for row in rows if (x := nfl_game(row))]
            if len(games) < 300:
                raise ValueError("nflverse two-season result coverage is unexpectedly small")
            stored = dict(games=games, etag=response.headers.get("ETag"), source_at=utc(clock()),
                          checked=clock().timestamp(), error="")
    except (OSError, ValueError, EOFError) as error:
        if isinstance(error, HTTPError) and error.code == 304:
            stored["source_at"] = utc(clock())
            stored["error"] = ""
        else:
            stored["error"] = type(error).__name__
        stored["checked"] = clock().timestamp() - 7200 if stored["error"] else clock().timestamp()
    atomic(path, stored)
    return stored


def nfl_game(row):
    try:
        year = int(row["season"])
    except (ValueError, KeyError):
        return None
    if not clock().year - 1 <= year <= clock().year:
        return None
    names = [NFL_NAMES.get(row.get(field)) for field in ("away_team", "home_team")]
    if not all(names):
        return None
    try:
        tip = utc(datetime.fromisoformat(row["gameday"] + "T" + row["gametime"]).replace(tzinfo=ET))
    except ValueError:
        return None
    score = [number(row.get(field)) for field in ("away_score", "home_score")]
    game_type = row.get("game_type")
    if game_type not in {"REG", "WC", "DIV", "CON", "SB"}:
        return None
    return dict(uid=row["game_id"], provider=row["espn"], start=tip, names=names,
                codes=[row.get("away_team"), row.get("home_team")], week=number(row.get("week")),
                keys=[side_key(n, "nfl", "MONEYLINE") for n in names],
                scores=score if None not in score else None, season=year, stage=game_type,
                venue="neutral" if row.get("location") == "Neutral" else "home" if row.get("location") == "Home" else "",
                source="nflverse")


def published_rows(root, sport):
    by_event = {}
    for name in ("live_recent.csv", "anomaly_board.csv"):
        with (root / "data" / name).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                event = row.get("game_id", "")
                if row.get("sport") != sport or not TOKEN.fullmatch(event):
                    continue
                start = utc(row["kickoff_iso"])
                frozen = name == "live_recent.csv"
                previous = by_event.setdefault(event, {"game": row["game"], "start": start, "markets": {}, "frozen": frozen})
                if previous["game"] != row["game"]:
                    previous["conflict"] = True
                elif previous["start"] != start:
                    if frozen or not previous["frozen"] or not start or not previous["start"]:
                        previous["conflict"] = True
                    elif abs((datetime.fromisoformat(start) - datetime.fromisoformat(previous["start"])).total_seconds()) <= 900:
                        # Two published surfaces may report different start times for the
                        # same handoff. Use the frozen cutoff, and accept either button.
                        alternate = previous.setdefault("published_starts", [previous["start"]])
                        if start not in alternate:
                            alternate.append(start)
                    else:
                        previous["conflict"] = True
                if previous["frozen"] and not frozen:
                    continue
                market = row.get("market_display", "")
                try:
                    sides = json.loads(row.get("market_sides") or "[]")
                except ValueError:
                    continue
                if market in MARKETS and isinstance(sides, list) and len(sides) == 2 and all(isinstance(s, dict) for s in sides):
                    stamp = row.get("final_pregame_state_at_utc") or row.get("state_as_of_utc") or row.get("source_latest_at")
                    previous["markets"][market] = [{
                        field: side.get(field, "") for field in ("flagged_side", "open_line", "current_line",
                                                                "bets_pct", "money_pct", "response_direction")
                    } | {"source_latest_at": side.get("source_latest_at") or stamp,
                         "quality": row.get("data_badge", ""), "frozen": bool(row.get("freeze_method"))} for side in sides]
    return by_event


def outcome_identity(sport, event, fixture, item):
    source, status, provider, _, evidence = item
    if status != "verified":
        return None
    if provider:
        return str(provider)
    # School box scores can be verified by an official event URL without a
    # scoreboard provider ID. The stored verified receipt must bind it to this fixture.
    if (sport == "ncaaf" and source == "verified_research_ncaaf"
            and evidence.get("research_provenance") == "school_athletics_official_final"
            and evidence.get("event_id") == event and utc(evidence.get("fixture_scheduled_at")) == fixture["start"]):
        url = urlsplit(evidence.get("url") or "")
        if url.scheme == "https" and url.hostname and "/boxscore/" in url.path:
            return url.hostname + url.path.rstrip("/")
    return None


def valid_outcomes(db, sport, catalog, now):
    groups = defaultdict(list)
    for r in db.execute("""SELECT * FROM outcomes WHERE sport=? AND observed_at<=?
                            ORDER BY event_id,observed_at DESC,digest""", (sport,now)):
        if r["event_id"] in catalog:
            groups[r["event_id"]].append((r["source"],r["status"],r["provider_event_id"],r["observed_at"],json.loads(r["evidence"])))
    result, conflicts, providers = {}, set(), {}
    for event, rows in groups.items():
        fx = catalog[event]
        verified = [x for x in latest_provider_outcomes(rows, now) if outcome_identity(sport, event, fx, x)]
        if not verified:
            continue
        signatures = {outcome_signature(sport, x[1], x[4]) for x in verified}
        if len(signatures) != 1 or None in signatures:
            conflicts.add(event)
            continue
        chosen = max(verified, key=lambda x: (x[0].startswith("espn_exact"), x[3]))
        source, status, _, stamp, raw = chosen
        final = next(iter(signatures))
        if sport == "ufc":
            _, confirmed = identities(raw.get("matchup"), sport)
            if not confirmed or set(confirmed) != set(fx["keys"]) or final not in fx["keys"] or not raw.get("actual_start"):
                conflicts.add(event)
                continue
            outcome = dict(winner=final, actual=utc(raw["actual_start"]))
        else:
            scores = dict(final)
            if set(scores) != set(fx["keys"]) or None in [number(scores[k]) for k in fx["keys"]]:
                conflicts.add(event)
                continue
            outcome = dict(scores=[int(scores[k]) for k in fx["keys"]])
        if not fx["start"] or fx["start"] >= now:
            continue
        outcome.update(uid=event, vendor=event, provider=outcome_identity(sport, event, fx, chosen),
                       start=fx["start"], keys=fx["keys"], names=fx["names"], source=source, source_at=stamp,
                       season=fx["season"], stage="", venue="", _receipt=(status, raw))
        result[event] = outcome
        # nflverse supplies ESPN identifiers, so never join via another namespace.
        for item in verified:
            if item[2] and (item[0].startswith("espn") or (sport == "nfl" and espn_nfl_receipt(item))):
                providers.setdefault(item[2], []).append(event)
    by_provider = {}
    for event, outcome in result.items():
        by_provider.setdefault(outcome["provider"], []).append(event)
    clean = {}
    for events in by_provider.values():
        values = [result[e] for e in events]
        if len({(tuple(v["keys"]), tuple(v.get("scores") or ()), v.get("winner")) for v in values}) != 1:
            conflicts.update(events)
            continue
        representative = min(values, key=lambda v: (v["start"], v["vendor"]))
        clean[representative["uid"]] = representative
    return clean, conflicts, providers


def espn_nfl_receipt(item):
    link = urlsplit(item[4].get("url") or "")
    return (
        link.scheme == "https" and link.hostname == "www.espn.com"
        and link.path.rstrip("/") == "/nfl/game/_/gameId/" + item[2]
    )


def nfl_merge(archive, external, mappings, now):
    nfl = [g.copy() for g in external.get("games", []) if g["scores"] and g["start"] < now]
    merged = dict(archive)
    invalid = set()
    same_matchup = defaultdict(list)
    for match in archive.values():
        same_matchup[tuple(match["keys"])].append(match)
    for row in nfl:
        matching = [archive[e] for e in mappings.get(row["provider"], []) if e in archive]
        for candidate in same_matchup[tuple(row["keys"])]:
            delay = abs((datetime.fromisoformat(candidate["start"]) - datetime.fromisoformat(row["start"])).total_seconds())
            if delay <= 10800:
                merged.pop(candidate["uid"], None)
        if matching:
            old = matching[0]
            age = abs((datetime.fromisoformat(old["start"]) - datetime.fromisoformat(row["start"])).total_seconds())
            if any(x["keys"] != row["keys"] or x["scores"] != row["scores"] for x in matching) or age > 10800:
                invalid.update(x["vendor"] for x in matching)
                continue
            row["vendor"] = old["vendor"]
            row["market_start"] = old["start"]
            for old in matching:
                merged.pop(old["uid"], None)
        row["source_at"] = external.get("source_at")
        merged[row["uid"]] = row
    return merged, invalid


def graded_quotes(db, sport, match, market, receipt):
    event = match.get("vendor")
    if not event or not receipt:
        return {}
    market_start = match.get("market_start", match["start"])
    limit = min(match.get("actual") or market_start, market_start)
    for stamp, checkpoint, scheduled, raw in db.execute(
        f"""SELECT entry_at, decision_at, scheduled_at, evidence FROM comparison_windows
            WHERE sport=? AND market=? AND minutes IN ({",".join("?" for _ in WINDOWS)})
              AND event_id=? AND entry_at<? AND decision_at<?
            ORDER BY entry_at DESC, decision_at DESC LIMIT 4""",
        (sport, market, *WINDOWS, event, limit, limit),
    ):
        if scheduled != market_start or stamp > checkpoint:
            continue
        evidence = decode(raw)
        verified_start = evidence.get("verified_start")
        if verified_start and (stamp >= verified_start or checkpoint >= verified_start):
            continue
        sides = evidence.get("sides", [])
        expected = {"over", "under"} if market == "TOTAL" else set(match["keys"])
        if len(sides) != 2 or {q.get("side") for q in sides} != expected:
            continue
        prices = {}
        for q in sides:
            price = q.get("price")
            if price is None or abs(price) < 100:
                continue
            result = grade(sport, market, q, *receipt)
            if result not in {"win", "loss", "push"}:
                continue
            net = (100 / abs(price) if price < 0 else price / 100) if result == "win" else -1 if result == "loss" else 0
            prices[q["side"]] = dict(at=stamp, checkpoint=checkpoint, grade=result, units=round(net, 6),
                                    **{f: q.get(f) for f in ("line", "price", "open_line", "open_price", "tickets",
                                                             "money", "line_toward", "price_toward_implied",
                                                             "path_retrace", "path_direction_changes")})
        if prices:
            return prices
    return {}


def favorites(root, sport):
    result = defaultdict(list)
    with (root / "data/performance_favorites.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["sport"] == sport:
                official = {k:row.get(k,"") for k in
                    ("market", "side", "decision_line", "decision_price", "decision_quote_observed_at",
                     "decision_line_source", "closing_line", "closing_price", "closing_price_source",
                     "closing_price_observed_at", "grade")}
                official["side_key"] = side_key(official["side"], sport, official["market"])
                result[row["event_id"]].append(official)
    return result


def build(root, db, sport, external, now, guard, source_state, permit_network):
    rows = published_rows(root, sport)
    catalog, invalid = {}, set()
    for row in db.execute("SELECT event_id,matchup,scheduled_at FROM fixtures WHERE sport=?", (sport,)):
        event, game, start = row
        names, keys = identities(game,sport)
        if not keys or not start or not TOKEN.fullmatch(event):
            continue
        catalog[event] = dict(names=names, keys=keys, start=start, season=season(sport,start), game=game)
    for event, r in rows.items():
        names, keys = identities(r["game"],sport)
        if not r["start"] or not keys or r.get("conflict"):
            invalid.add(event)
            continue
        old = catalog.get(event)
        if old and old["keys"] != keys:
            invalid.add(event)
            continue
        catalog[event] = dict(names=names,keys=keys,start=r["start"],season=season(sport,r["start"]),
                              game=r["game"], markets=r["markets"], published_starts=r.get("published_starts") or [r["start"]])
    known, conflicts, mappings = valid_outcomes(db,sport,catalog,now)
    invalid.update(conflicts)
    games = {k:v for k,v in known.items() if k not in invalid}
    if sport == "nfl":
        games, mismatch = nfl_merge(games,external,mappings,now)
        invalid.update(mismatch)
    catalog = {k:v for k,v in catalog.items() if k not in invalid}
    teams = defaultdict(list)
    for event, match in sorted(games.items(), key=lambda kv:kv[1]["start"], reverse=True):
        if match.get("vendor") not in invalid:
            for k in match["keys"]:
                teams[k].append(event)
    # Grade only the initial recent evidence set. Browser expansion beyond it
    # can still show completed scores without warming the entire quote archive.
    recent_ids = {event for team in teams.values() for event in team[:12]}
    official = favorites(root,sport)
    for match in games.values():
        if match.get("vendor") in official:
            match["favorites"] = official[match["vendor"]]
    for i, event in enumerate(recent_ids):
        if i % 25 == 0 and not guard():
            raise TimeoutError("publisher priority")
        match = games[event]
        r = known.get(match.get("vendor"), {}).get("_receipt")
        match["quotes"] = {}
        for m in (("MONEYLINE",) if sport == "ufc" else MARKETS):
            available = graded_quotes(db,sport,match,m,r)
            if available:
                match["quotes"][m] = available
    sporting = {}
    if sport == "nfl":
        sporting = nfl_profile_context(source_state, catalog, games, external, now, guard, permit_network)
    elif sport == "ncaaf":
        sporting = cfb_profile_context(source_state, catalog, now, guard, permit_network)
    elif sport == "mlb":
        sporting = baseball_profile_context(source_state, now, guard, permit_network)
    elif sport == "ufc":
        receipts = [outcome["_receipt"][1] for outcome in known.values() if "_receipt" in outcome]
        sporting = ufc_profile_context(source_state, receipts, catalog, guard, permit_network)
    if guard():
        sporting.update(independent_context(source_state, catalog, sport, now, guard, permit_network))
    for match in games.values():
        match.pop("provider", None)
        match.pop("_receipt", None)
    regular = defaultdict(list)
    for match in external.get("games", []):
        if match["stage"] == "REG":
            regular[match["season"]].append(match)
    past = utc(datetime.fromisoformat(now) - timedelta(hours=24))
    complete = sorted(year for year, schedule in regular.items()
                      if len({match["uid"] for match in schedule}) == 272
                      and all(match["scores"] or match["start"] > past for match in schedule))
    return dict(version=1,sport=sport,generated=now,fixtures=catalog,teams=teams,games=games,sporting=sporting,
                sources={"nflverse":{"url":"https://github.com/nflverse/nflverse-data/releases/tag/schedules",
                                     "at":external.get("source_at"),"error":external.get("error",""),
                                     "complete_regular_schedules":complete if not mismatch else []}
                         if sport=="nfl" else {}},
                diagnostics=dict(fixtures=len(catalog), active=len(rows), active_unmapped=len(invalid.intersection(rows)), unmapped=len(invalid),
                                 completed=len(games),market_games=sum(bool(m.get("quotes")) for m in games.values())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/opt/red-fox-market-dynamics"))
    parser.add_argument("--archive", type=Path, default=Path("/var/lib/redfox-history/private/history.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path("/var/lib/redfox-team-intelligence/public"))
    parser.add_argument("--state", type=Path, default=Path("/var/lib/redfox-team-intelligence/private"))
    parser.add_argument("--publisher-lock", type=Path, default=Path("/var/lock/redfox_run.lock"))
    parser.add_argument("--sport", choices=SPORTS)
    parser.add_argument("--bootstrap", action="store_true", help="Build on a separate host or isolated backup")
    parser.add_argument("--no-network", action="store_true")
    args = parser.parse_args()
    guard = (lambda: True) if args.bootstrap else (lambda: slot(args.publisher_lock))
    if not guard():
        print('{"deferred":"publisher priority"}', flush=True)
        return
    args.output.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < FREE_DISK_RESERVE:
        print('{"deferred":"protected disk reserve"}', flush=True)
        return
    args.state.mkdir(mode=0o700, parents=True,exist_ok=True)
    args.output.mkdir(mode=0o755, parents=True,exist_ok=True)
    status_path = args.output / "status.json"
    old = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    status = old.get("sports",{})
    new_credentials = set() if args.no_network else updated_credentials(status)
    order = [args.sport] if args.sport else sorted(SPORTS,key=lambda s: (s not in new_credentials, status.get(s,{}).get("checked","")))
    targets = order if args.bootstrap else order[:1]
    if (not args.bootstrap and not args.sport and targets[0] not in new_credentials
            and status.get(targets[0],{}).get("checked","") > utc(clock()-timedelta(hours=2))):
        print('{"deferred":"team profiles current"}', flush=True)
        return
    now = utc(clock())
    with sqlite3.connect(args.archive.resolve().as_uri()+"?mode=ro",uri=True,timeout=1) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        for sport in targets:
            if not guard():
                break
            started = time.monotonic()
            try:
                ext = nfl_schedule(args.state, 5, not args.no_network) if sport == "nfl" else {}
                if not guard():
                    break
                profile = build(args.root, db, sport, ext, now, guard, args.state, not args.no_network)
                if not guard():
                    break
                if profile["diagnostics"]["fixtures"] == 0:
                    raise ValueError("empty fixture inventory")
                atomic(args.output / (sport+".json"),profile)
                status[sport] = profile["diagnostics"] | {
                    "checked": now, "source_error": ext.get("error", ""), "source_at": ext.get("source_at"),
                    "bytes": (args.output / (sport+".json")).stat().st_size,
                    "seconds": round(time.monotonic() - started, 2),
                    "sporting": {name: {field:value[field] for field in
                                        ("at","checked","error","coverage","partitions","expected_partitions","fixture_coverage")
                                        if field in value}
                                 for name,value in profile["sporting"].items() if name != "fighters"},
                }
                atomic(status_path,{"version":1,"sports":status})
                print(packed({"sport":sport,**status[sport]}),flush=True)
            except TimeoutError:
                print('{"deferred":"publisher priority"}',flush=True)
                break


if __name__ == "__main__":
    main()
