"""Permitted independent sources for Team Intelligence. No market-engine writes."""

from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import io
import json
import math
import re
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

from history_comparisons import side_key
from history_store import atomic_text, packed, utc

NFL_RELEASE = "https://github.com/nflverse/nflverse-data/releases/download/"
RETRO = "https://www.retrosheet.org/gamelogs/"
WIKI = "https://query.wikidata.org/sparql"
CFB_ELO_URL = "https://collegefootballdigest.com/api/v1/elo/"
AGENT = "RedFox-Team-Intelligence/2 (+https://redfoxmi.com)"
MLB_CODES = dict(zip(
    "ANA ARI ATH ATL BAL BOS CHA CHN CIN CLE COL DET HOU KCA LAN MIA MIL MIN NYA NYN PHI PIT SDN SEA SFN SLN TBA TEX TOR WAS".split(),
    ("LA Angels", "ARI Diamondbacks", "Athletics", "ATL Braves", "BAL Orioles", "BOS Red Sox", "CHI White Sox",
     "CHI Cubs", "CIN Reds", "CLE Guardians", "COL Rockies", "DET Tigers", "HOU Astros", "KC Royals", "LA Dodgers",
     "MIA Marlins", "MIL Brewers", "MIN Twins", "NY Yankees", "NY Mets", "PHI Phillies", "PIT Pirates",
     "SD Padres", "SEA Mariners", "SF Giants", "STL Cardinals", "TB Rays", "TEX Rangers", "TOR Blue Jays", "WAS Nationals"),
))


def saved_source(path, url, decode, guard, network, *, hours, limit=900_000, headers=None):
    now = datetime.now(timezone.utc)
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = {}
    if not network or not guard() or now.timestamp() < stored.get("next", 0):
        return stored
    request_headers = {"User-Agent": AGENT, **(headers or {})}
    if stored.get("etag"):
        request_headers["If-None-Match"] = stored["etag"]
    try:
        with urlopen(Request(url, headers=request_headers), timeout=5) as response:
            raw = response.read(limit + 1)
            if len(raw) > limit:
                raise ValueError("source response exceeds limit")
            modified = response.headers.get("Last-Modified")
            source_at = utc(parsedate_to_datetime(modified)) if modified else utc(now)
            stored = {
                "data": decode(raw), "source_at": source_at, "etag": response.headers.get("ETag"),
                "checked": utc(now), "next": now.timestamp() + hours * 3600,
            }
    except (OSError, ValueError, KeyError, BadZipFile) as error:
        if isinstance(error, HTTPError) and error.code == 304:
            stored.pop("error", None)
            delay = hours
        else:
            stored["error"] = str(error.code) if isinstance(error, HTTPError) else type(error).__name__
            delay = hours if isinstance(error, HTTPError) and error.code == 404 else min(hours, 1)
        stored["checked"], stored["next"] = utc(now), now.timestamp() + delay * 3600
    with atomic_text(path, packed(stored)):
        pass
    return stored


def csv_records(raw, columns):
    rows = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if not set(columns).issubset(rows.fieldnames or ()):
        raise ValueError("source schema changed")
    return rows


def nfl_efficiency(raw):
    fields = ("game_id", "team", "opponent_team", "season_type", "attempts", "sacks_suffered", "carries",
              "passing_epa", "rushing_epa")
    stats = defaultdict(dict)
    invalid = set()
    for row in csv_records(raw, fields):
        if row["season_type"] != "REG" or not row["game_id"]:
            continue
        try:
            attempts, sacks, rushes = (int(float(row[k])) for k in ("attempts", "sacks_suffered", "carries"))
            pass_epa, rush_epa = (float(row[k]) for k in ("passing_epa", "rushing_epa"))
        except (ValueError, OverflowError):
            continue
        plays = attempts + sacks + rushes
        epa = pass_epa + rush_epa
        target = stats[row["game_id"]]
        if row["team"] in target:
            invalid.add(row["game_id"])
        if 0 < plays < 160 and min(attempts, sacks, rushes) >= 0 and math.isfinite(epa) and abs(epa) <= 200:
            target[row["team"]] = {"opponent": row["opponent_team"], "plays": plays, "epa": round(epa, 4),
                                   "dropbacks": attempts+sacks, "pass_epa": round(pass_epa, 4),
                                   "rushes": rushes, "rush_epa": round(rush_epa, 4), "sacks": sacks}
    return {key: value for key, value in stats.items() if len(value) == 2 and key not in invalid}


def nfl_availability(raw):
    fields = ("season", "season_type", "week", "team", "gsis_id", "full_name", "position",
              "report_primary_injury", "report_status", "practice_primary_injury", "practice_status")
    groups = defaultdict(dict)
    conflict = set()
    for row in csv_records(raw, fields):
        if not row["gsis_id"] or not row["full_name"] or row["season_type"] != "REG":
            continue
        status = row["report_status"].strip().title()
        injury = row["report_primary_injury"]
        if not status and row["practice_status"].strip().casefold() == "did not participate in practice":
            status, injury = "Did not practice", row["practice_primary_injury"]
        if status not in {"Out", "Doubtful", "Questionable", "Did not practice"}:
            continue
        key = row["season"] + ":" + row["week"] + ":" + row["team"]
        identity = row["gsis_id"]
        item = {"name": row["full_name"], "position": row["position"], "status": status,
                "injury": injury}
        group = groups[key]
        if identity in group and group[identity] != item:
            conflict.add((key, identity))
        group[identity] = item
    order = {"Out": 0, "Doubtful": 1, "Questionable": 2, "Did not practice": 3}
    return {
        key: sorted((item for identity, item in items.items() if (key, identity) not in conflict),
                    key=lambda item: (order[item["status"]], item["name"]))
        for key, items in groups.items()
    }


def nfl_profile_context(state, fixtures, games, schedule, now, guard, network):
    date = datetime.fromisoformat(now)
    season = date.year - (date.month <= 2)
    stats_url = NFL_RELEASE + f"stats_team/stats_team_week_{season}.csv"
    metrics = saved_source(state / f"nfl-team-stats-v2-{season}.json",
                           stats_url, nfl_efficiency, guard, network, hours=6)
    if not metrics.get("data"):
        prior = saved_source(state / f"nfl-team-stats-{season}.json", stats_url,
                             nfl_efficiency, guard, False, hours=6)
        metrics = {**metrics, **{k:prior[k] for k in ("data","source_at") if k in prior}}
    injuries = saved_source(state / ("nfl-availability-" + str(season) + ".json"),
                            NFL_RELEASE + f"injuries/injuries_{season}.csv",
                            nfl_availability, guard, network, hours=2)
    efficiency = metrics.get("data", {})
    seen = 0
    lookup = defaultdict(list)
    for game in schedule.get("games", ()):
        if game.get("stage") == "REG":
            lookup[tuple(game["keys"])].append(game)
    for game in games.values():
        codes = game.pop("codes", None)
        measures = efficiency.get(game["uid"], {})
        if (game["source"] == "nflverse" and codes and set(codes) == set(measures)
                and all(measures[code].get("opponent") == codes[1-i] for i,code in enumerate(codes))):
            game["sporting"] = {"efficiency": {key: measures[code] for key,code in zip(game["keys"],codes)}}
            seen += 1
    injury_stamp = injuries.get("source_at")
    if injury_stamp and injury_stamp <= now:
        for fixture in fixtures.values():
            if fixture["start"] <= now:
                continue
            when = datetime.fromisoformat(fixture["start"])
            matches = [row for row in lookup[tuple(fixture["keys"])] if row.get("codes") and row.get("week")
                       and abs((datetime.fromisoformat(row["start"]) - when).total_seconds()) <= 10800]
            if len(matches) != 1:
                continue
            match = matches[0]
            if not timedelta(0) <= when - datetime.fromisoformat(injury_stamp) < timedelta(days=7):
                continue
            reports = {key: injuries.get("data", {}).get(f'{match["season"]}:{match["week"]}:{code}', [])
                       for key, code in zip(match["keys"], match["codes"])}
            if any(reports.values()):
                fixture["sporting"] = {"availability": reports, "availability_at": injury_stamp}
    return {"nfl_team_stats": source_info(metrics, seen),
            "nfl_availability": source_info(injuries, sum("sporting" in f for f in fixtures.values()))}


def source_info(snapshot, count):
    return {"at": snapshot.get("source_at"), "checked": snapshot.get("checked"), "error": snapshot.get("error", ""),
            "coverage": count}


def cfb_elo(raw, season):
    envelope = json.loads(raw)
    if not isinstance(envelope, dict):
        raise ValueError("unexpected CollegeFootballDigest response")
    data, meta, attribution = (envelope.get(field) for field in ("data", "meta", "attribution"))
    license_url = "https://creativecommons.org/licenses/by/4.0/"
    if (not isinstance(data, dict) or not isinstance(meta, dict) or not isinstance(attribution, list)
            or data.get("season") != season or data.get("week", False) is not None
            or meta.get("truncated") is not False or not isinstance(meta.get("as_of"), str)
            or not isinstance(data.get("teams"), list)
            or not any(isinstance(row, dict) and row.get("license_url") == license_url for row in attribution)):
        raise ValueError("unexpected CollegeFootballDigest ratings envelope")
    as_of = datetime.fromisoformat(meta["as_of"].replace("Z", "+00:00"))
    if as_of.tzinfo is None:
        raise ValueError("ratings timestamp requires a time zone")
    teams, ids = {}, set()
    for row in data["teams"]:
        if not isinstance(row, dict):
            raise ValueError("invalid CollegeFootballDigest row")
        if row["classification"] != "fbs":
            continue
        team = row.get("team")
        if not isinstance(team, dict) or not isinstance(team.get("school"), str) or type(team.get("id")) is not int:
            raise ValueError("invalid CollegeFootballDigest team")
        key = side_key("ULM" if team.get("slug") == "ul-monroe" else team["school"], "ncaaf", "MONEYLINE")
        provider_id = team["id"]
        rank, rating = row["fbs_rank"], row["rating"]
        if (not key or key in teams or provider_id in ids or type(rank) is not int or rank < 1
                or type(rating) not in (int, float) or not 0 < rating < 10000):
            raise ValueError("invalid or ambiguous CollegeFootballDigest team")
        teams[key] = {"rank": rank, "rating": rating}
        ids.add(provider_id)
    if not teams or max(item["rank"] for item in teams.values()) > len(teams):
        raise ValueError("inconsistent CollegeFootballDigest ranks")
    return {"as_of": utc(as_of), "teams": teams}


def cfb_profile_context(state, fixtures, now, guard, network):
    timestamp = datetime.fromisoformat(now)
    season = timestamp.year - (timestamp.month <= 2)
    cache = saved_source(state / f"cfb-digest-elo-{season}.json", CFB_ELO_URL + str(season),
                         lambda raw: cfb_elo(raw, season), guard, network, hours=24)
    data = cache.get("data") or {}
    as_of = data.get("as_of")
    teams = data.get("teams", {})
    covered = 0
    if as_of and timedelta(0) <= timestamp - datetime.fromisoformat(as_of) <= timedelta(hours=48):
        for fixture in fixtures.values():
            if (fixture.get("season") == season and fixture.get("markets") and fixture["start"] > now
                    and all(key in teams for key in fixture["keys"])):
                fixture["sporting"] = {"cfb_elo": {"at": as_of, "field": len(teams),
                                                    "teams": {key: teams[key] for key in fixture["keys"]}}}
                covered += 1
    return {"cfb_elo": {**source_info(cache, covered), "at": as_of}}


def retrosheet_year(raw, year, *, include_history=False):
    with ZipFile(io.BytesIO(raw)) as archive:
        member = next((name for name in archive.namelist() if name.lower() == f"gl{year}.txt"), "")
        if not member or archive.getinfo(member).file_size > 6_000_000:
            raise ValueError("unexpected Retrosheet archive")
        body = archive.read(member)
    games = list(csv.reader(io.StringIO(body.decode("latin-1"))))
    teams = defaultdict(lambda: Counter())
    seen, history = set(), []
    for row in games:
        if len(row) < 12 or row[3] not in MLB_CODES or row[6] not in MLB_CODES or not row[0].startswith(str(year)):
            raise ValueError("unexpected Retrosheet row")
        identity = row[0],row[1],row[3],row[6]
        scores = int(row[9]),int(row[10])
        if identity in seen or min(scores) < 0:
            raise ValueError("duplicate or invalid Retrosheet result")
        seen.add(identity)
        if include_history:
            stamp = datetime.strptime(row[0], "%Y%m%d").date().isoformat()
            names = [MLB_CODES[row[3]], MLB_CODES[row[6]]]
            history.append({
                "uid": "retrosheet-" + "-".join(identity), "names": names,
                "keys": [side_key(name, "mlb", "MONEYLINE") for name in names],
                "date": stamp, "scores": list(scores), "venue": "home",
            })
        for i, code in enumerate((row[3], row[6])):
            tally = teams[code]
            win = scores[i] > scores[1-i]
            loss = scores[i] < scores[1-i]
            tally.update({"wins": int(win), "losses": int(loss), "tie": int(not win and not loss),
                          "scored": scores[i], "allowed": scores[1-i], "games": 1,
                          ("home" if i else "away") + "_wins": int(win),
                          ("home" if i else "away") + "_losses": int(loss)})
    if len(seen) != 2430 or len(teams) != 30 or any(team["games"] != 162 for team in teams.values()):
        raise ValueError("Retrosheet regular season is incomplete")
    totals = {side_key(MLB_CODES[code],"mlb","MONEYLINE"): dict(tally) for code,tally in teams.items()}
    return {"teams": totals, "history": history} if include_history else totals


def baseball_profile_context(state, now, guard, network, *, market_history=None):
    year = datetime.fromisoformat(now).year
    candidates = [year,year-1] if datetime.fromisoformat(now).month >= 10 else [year-1]
    for value in candidates:
        snapshot = saved_source(state / f"retrosheet-games-{value}.json", RETRO + f"gl{value}.zip",
                                lambda raw: retrosheet_year(raw, value, include_history=True), guard, network, hours=168)
        if snapshot.get("data"):
            data = snapshot["data"]
            if market_history is not None:
                market_history.extend(data["history"])
            return {"baseball": {"season": value, "teams": data["teams"], **source_info(snapshot, len(data["teams"]))}}
        # Keep the already published season totals while the richer cache loads.
        previous = state / f"retrosheet-{value}.json"
        try:
            cached = json.loads(previous.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = {}
        if cached.get("data"):
            return {"baseball": {"season": value, "teams": cached["data"], **source_info(cached, len(cached["data"]))}}
    return {"baseball": {"error": "no_complete_season"}}


def fighter_identities(receipts):
    ids_by_key = defaultdict(set)
    keys_by_id = defaultdict(set)
    for receipt in receipts:
        if receipt.get("source") != "verified_research_ufc":
            continue
        names = receipt.get("names", ())
        ids = receipt.get("athlete_ids", ())
        if len(names) != 2 or len(ids) != 2:
            continue
        for name, identity in zip(names,ids):
            identity = str(identity)
            key = side_key(name,"ufc","MONEYLINE")
            if key and re.fullmatch(r"\d{5,10}", identity):
                ids_by_key[key].add(identity)
                keys_by_id[identity].add(key)
    return {key: next(iter(ids)) for key, ids in ids_by_key.items()
            if len(ids) == 1 and len(keys_by_id[next(iter(ids))]) == 1}


def wikidata_fighters(raw):
    bindings = json.loads(raw)["results"]["bindings"]
    by_id = defaultdict(set)
    for row in bindings:
        get = lambda key: row.get(key, {}).get("value", "")
        espn, item, unit = get("espn"),get("item").rsplit("/",1)[-1],get("unit").rsplit("/",1)[-1]
        if not re.fullmatch(r"Q\d+", item) or not espn.isdigit():
            continue
        try:
            height = float(get("height")) * {"Q11573":100,"Q174728":1,"Q218593":2.54}[unit]
        except (ValueError, KeyError):
            continue
        if 140 <= height <= 230:
            by_id[espn].add((item,round(height,1)))
    return {key: {"item": next(iter(vals))[0], "height_cm": next(iter(vals))[1]}
            for key,vals in by_id.items() if len(vals) == 1}


def ufc_profile_context(state, receipts, fixtures, guard, network):
    joined = fighter_identities(receipts)
    keys = {key for fixture in fixtures.values() if "markets" in fixture for key in fixture["keys"]}
    # Use one modest batch, favoring active fighters while retaining recent verified fighters.
    ordered = sorted(joined, key=lambda key: (key not in keys,key))[:180]
    if not ordered:
        return {}
    values = " ".join(json.dumps(joined[key]) for key in ordered)
    query = (
        "SELECT ?item ?espn ?height ?unit WHERE { VALUES ?espn {" + values + "} ?item wdt:P10073 ?espn ; "
        "p:P2048/psv:P2048 ?v . ?v wikibase:quantityAmount ?height ; wikibase:quantityUnit ?unit . }"
    )
    snapshot = saved_source(state / "wikidata-fighter-profiles.json", WIKI + "?" + urlencode({"query":query,"format":"json"}),
                            wikidata_fighters, guard, network, hours=48, headers={"Accept":"application/sparql-results+json"})
    data = snapshot.get("data", {})
    return {"fighters": {key: data[joined[key]] for key in ordered if joined[key] in data},
            "wikidata": source_info(snapshot,len(data))}
