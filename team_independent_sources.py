"""Independent sporting data for display in Team Intelligence only."""

from calendar import monthrange
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import re
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo

from history_comparisons import side_key
from history_store import atomic_text, packed, utc


KEY_DIRECTORY = Path("/etc/redfox-team-intelligence/credentials")
ET = ZoneInfo("America/New_York")
PROVIDERS = {
    "nba": ("balldontlie", "https://api.balldontlie.io/nba/v1/games"),
    "mlb": ("balldontlie", "https://api.balldontlie.io/mlb/v1/games"),
    "ncaaf": ("rad-sports", "https://api.collegefootballdata.com/games"),
    "ncaab": ("rad-sports", "https://api.collegebasketballdata.com/games"),
    "nhl": ("puckapi", "https://mcp.puckapi.com/v1/get_games"),
    "ufc": ("ufcalendar", "https://api.ufcalendar.com/v1"),
}
DAILY = {"balldontlie": 150, "rad-sports": 60, "puckapi": 70, "ufcalendar": 12}
MONTHLY = {"rad-sports": 850, "puckapi": 440, "ufcalendar": 95}
LABELS = {"balldontlie": "BALLDONTLIE", "rad-sports": "College sports data",
          "puckapi": "PuckAPI", "ufcalendar": "UFCalendar"}
IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")
NHL_NAMES = dict(zip(
    "ANA BOS BUF CAR CBJ CGY CHI COL DAL DET EDM FLA LAK MIN MTL NJD NSH NYI NYR OTT PHI PIT SEA SJS STL TBL TOR UTA VAN VGK WPG WSH".split(),
    ("Anaheim Ducks", "Boston Bruins", "Buffalo Sabres", "Carolina Hurricanes", "Columbus Blue Jackets",
     "Calgary Flames", "Chicago Blackhawks", "Colorado Avalanche", "Dallas Stars", "Detroit Red Wings",
     "Edmonton Oilers", "Florida Panthers", "Los Angeles Kings", "Minnesota Wild", "Montreal Canadiens",
     "New Jersey Devils", "Nashville Predators", "New York Islanders", "New York Rangers", "Ottawa Senators",
     "Philadelphia Flyers", "Pittsburgh Penguins", "Seattle Kraken", "San Jose Sharks", "St. Louis Blues",
     "Tampa Bay Lightning", "Toronto Maple Leafs", "Utah Mammoth", "Vancouver Canucks", "Vegas Golden Knights",
     "Winnipeg Jets", "Washington Capitals"),
))


def load(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def save(path, data):
    with atomic_text(path, packed(data)):
        pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class FeedUnavailable(Exception):
    pass


class Feed:
    def __init__(self, state, provider, now, guard, network, key_directory):
        self.provider = provider
        self.now = now
        self.guard = guard
        self.network = network
        self.key = ""
        try:
            key = (key_directory / provider).read_text(encoding="utf-8").strip()
            if re.fullmatch(r"[A-Za-z0-9_.~+/=-]{8,512}", key):
                self.key = key
        except OSError:
            pass
        self.path = state / f"requests-{provider}.json"
        self.usage = load(self.path)
        self.remaining = 4

    def request(self, url, params, *, cost=1, fighter=False):
        if not self.key:
            raise FeedUnavailable("not configured")
        if not self.network or not self.guard() or not self.remaining:
            raise FeedUnavailable("deferred")
        if self.usage.get("paused_until", 0) > self.now.timestamp():
            raise FeedUnavailable("provider backoff")
        minute = self.now.strftime("%Y-%m-%dT%H:%M")
        day = minute[:10]
        month = day[:7]
        daily = self.usage.get("daily", 0) if self.usage.get("day") == day else 0
        monthly = self.usage.get("monthly", 0) if self.usage.get("month") == month else 0
        minute_count = self.usage.get("minute_count", 0) if self.usage.get("minute") == minute else 0
        ceiling = 73 if fighter else MONTHLY.get(self.provider, 10_000)
        if daily + cost > DAILY[self.provider] or monthly + cost > ceiling or minute_count >= 4:
            raise FeedUnavailable("budget deferred")
        self.remaining -= 1
        self.usage.update(day=day, daily=daily+cost, month=month, monthly=monthly+cost,
                          minute=minute, minute_count=minute_count+1)
        save(self.path, self.usage)
        headers = {"User-Agent": "RedFox-Team-Intelligence/3 (+https://redfoxmi.com)",
                   "Accept": "application/json"}
        body = None
        if self.provider == "puckapi":
            headers.update({"x-api-key": self.key, "Content-Type": "application/json"})
            body = packed(params).encode()
        else:
            headers["Authorization"] = self.key if self.provider == "balldontlie" else f"Bearer {self.key}"
            if params:
                url += "?" + urlencode(params, doseq=True)
        try:
            with build_opener(NoRedirect()).open(Request(url, data=body, headers=headers), timeout=4) as response:
                data = response.read(4_000_001)
                if len(data) > 4_000_000:
                    raise ValueError("oversized response")
                return json.loads(data)
        except HTTPError as error:
            if error.code in (401, 403, 429) or error.code >= 500:
                delay = 2 * 3600 if error.code in (401, 403) else 900
                retry = error.headers.get("Retry-After", "")
                if retry.isdecimal():
                    delay = max(delay, int(retry))
                self.usage["paused_until"] = self.now.timestamp() + delay
                save(self.path, self.usage)
            raise FeedUnavailable(f"HTTP {error.code}") from None
        except (OSError, ValueError):
            raise FeedUnavailable("transport or response error") from None


def windows(sport, year):
    if sport == "ncaaf":
        return [(date(year, 8, 1), date(year+1, 2, 28))]
    first, last = {"nba": ((year, 10), (year+1, 6)), "nhl": ((year, 9), (year+1, 6)),
                   "ncaab": ((year, 11), (year+1, 4)), "mlb": ((year, 3), (year, 11))}[sport]
    result = []
    while first <= last:
        y, m = first
        end = monthrange(y, m)[1]
        if sport == "ncaab":
            result.extend([(date(y, m, 1), date(y, m, 15)), (date(y, m, 16), date(y, m, end))])
        else:
            result.append((date(y, m, 1), date(y, m, end)))
        first = (y+1, 1) if m == 12 else (y, m+1)
    return result


def canonical(name, sport):
    if not isinstance(name, str) or not name.strip() or len(name) > 100:
        raise ValueError("missing participant")
    key = side_key(name, sport, "MONEYLINE")
    if not key:
        raise ValueError("invalid participant")
    return key


def utc_start(value, *, day_only=False):
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        if not day_only:
            raise ValueError("unverified start")
        stamp = stamp.replace(tzinfo=ET)
    return utc(stamp)


def game_row(sport, year, row):
    ident = str(row["id"])
    if not IDENTIFIER.fullmatch(ident):
        raise ValueError("invalid game ID")
    venue = elo = None
    if sport == "nba":
        if row["season"] != year:
            raise ValueError("wrong NBA season")
        if row.get("postponed") or row.get("postseason") or row.get("season_type") in {"preseason", "playin", "playoffs"} or row.get("ist_stage") == "Championship":
            return None
        names = [row["visitor_team"]["full_name"], row["home_team"]["full_name"]]
        ids = [provider_id(row[k]["id"]) for k in ("visitor_team", "home_team")]
        start = utc_start(row.get("datetime") or row["date"], day_only=True)
        status = row["status"]
        scores = [row["visitor_team_score"], row["home_team_score"]] if status == "Final" else None
        scheduled = isinstance(status, str) and bool(re.search(r"\bET$|^Scheduled$", status, re.IGNORECASE))
    elif sport == "mlb":
        if row["season"] != year:
            raise ValueError("wrong MLB season")
        if row.get("season_type") != "regular":
            return None
        names = [row["away_team_name"], row["home_team_name"]]
        ids = [provider_id(row[k]["id"]) for k in ("away_team", "home_team")]
        start = utc_start(row["date"])
        scores = [row[k]["runs"] for k in ("away_team_data", "home_team_data")] if row["status_state"] == "final" else None
        scheduled = row["status_state"] == "scheduled"
    elif sport == "nhl":
        if str(row["season"]) != f"{year}{year+1}":
            raise ValueError("wrong NHL season")
        if row["gameType"] != "regular":
            return None
        names = [row.get(k+"TeamName") or NHL_NAMES[row[k+"Team"]] for k in ("away", "home")]
        ids = [provider_id(row[k+"Team"]) for k in ("away", "home")]
        # The documented local clock has no zone. Date supports form and joins without inventing a kickoff.
        start = utc_start(row["gameDate"], day_only=True)
        scores = [row[k] for k in ("awayScore", "homeScore")] if row["gameState"] in {"FINAL", "OFF"} else None
        scheduled = row["gameState"] in {"FUT", "PRE"}
    else:
        if row["season"] != year + (sport == "ncaab"):
            raise ValueError("wrong college season")
        if row.get("seasonType") not in {"regular", "postseason"}:
            return None
        names = [row["awayTeam"], row["homeTeam"]]
        id_field = "Id" if sport == "ncaaf" else "TeamId"
        ids = [provider_id(row[k+id_field]) for k in ("away", "home")]
        start = utc_start(row["startDate"])
        final = row["completed"] is True if sport == "ncaaf" else row["status"].casefold() in {"final", "completed"}
        scheduled = row["completed"] is False if sport == "ncaaf" else row["status"].casefold() == "scheduled"
        scores = [row[k+"Points"] for k in ("away", "home")] if final else None
        site = row.get("neutralSite")
        venue = "neutral" if site is True else "home" if site is False else None
        fields = ("awayPregameElo", "homePregameElo") if sport == "ncaaf" else ("awayTeamEloStart", "homeTeamEloStart")
        elo = [row.get(k) for k in fields]
    keys = [canonical(name, sport) for name in names]
    if (keys[0] == keys[1] or ids[0] == ids[1]
            or scores is not None and any(type(x) is not int or not 0 <= x <= 400 for x in scores)):
        raise ValueError("invalid game")
    return {"id": ident, "names": names, "keys": keys, "ids": ids, "start": start, "venue": venue,
            "scores": scores, "elo": elo, "scheduled": scheduled}


def schedule_page(feed, sport, year, start, end, cursor):
    url = PROVIDERS[sport][1]
    if sport in {"nba", "mlb"}:
        query = {"seasons[]": year, "per_page": 100}
        if sport == "mlb":
            query["dates[]"] = [str(start+timedelta(days=n)) for n in range((end-start).days+1)]
            query["season_type"] = "regular"
        else:
            query.update(start_date=str(start), end_date=str(end))
        if cursor is not None:
            query["cursor"] = cursor
        payload = feed.request(url, query)
        return payload["data"], payload["meta"].get("next_cursor"), 1500
    if sport == "nhl":
        payload = feed.request(url, {"league": "nhl", "season": f"{year}{year+1}", "game_type": "regular",
                                     "date_from": str(start), "date_to": str(end), "limit": 500}, cost=5)
        return payload["games"], None, 499
    if sport == "ncaaf":
        return feed.request(url, {"year": year, "seasonType": "both", "classification": "fbs"}), None, 2500
    return feed.request(url, {"season": year+1, "startDateRange": f"{start}T00:00:00Z",
                              "endDateRange": f"{end}T23:59:59.999Z"}), None, 2999


def schedule(feed, state, sport, year):
    path = state / f"games-{sport}-{year}.json"
    shards = load(path)
    today = feed.now.astimezone(ET).date()
    now = feed.now.timestamp()
    choices = []
    for start, end in windows(sport, year):
        token = str(start)
        shard = shards.setdefault(token, {})
        if shard.get("next", 0) <= now:
            distance = max((start-today).days, (today-end).days, 0)
            choices.append((bool(shard.get("as_of")), distance, start, end))
    error = ""
    if feed.key and feed.network and choices and feed.guard():
        _, _, start, end = min(choices)
        shard = shards[str(start)]
        working = shard.get("working", {})
        position = shard.get("cursor")
        try:
            while True:
                rows, next_cursor, limit = schedule_page(feed, sport, year, start, end, position)
                if not isinstance(rows, list) or len(rows) + len(working) > limit:
                    raise ValueError("unexpected schedule size")
                for source_row in rows:
                    row = game_row(sport, year, source_row)
                    if row is None:
                        continue
                    if row["id"] in working:
                        raise ValueError("duplicate provider game")
                    working[row["id"]] = row
                if next_cursor is None:
                    if any(row["scores"] and key not in working for key,row in shard.get("rows", {}).items()):
                        raise ValueError("existing finals missing from response")
                    if start <= today + timedelta(days=14) and end >= today - timedelta(days=4):
                        delay = 8 * 3600
                    else:
                        delay = (21 if end < today else 4) * 86400
                    shard.clear()
                    shard.update(rows=working, as_of=utc(feed.now), next=now+delay)
                    break
                if type(next_cursor) is not int or next_cursor <= (position or 0) or not rows:
                    raise ValueError("invalid pagination cursor")
                position = next_cursor
                shard.update(working=working, cursor=position)
                save(path, shards)
                if not feed.remaining:
                    break
        except FeedUnavailable as issue:
            error = str(issue)
            if error not in {"deferred", "budget deferred", "provider backoff"}:
                shard.update(error=error, next=now+3600)
        except (TypeError, ValueError, KeyError, AttributeError):
            error = "schema or coverage validation"
            shard.pop("working", None)
            shard.pop("cursor", None)
            shard.update(error=error, next=now+3600)
        save(path, shards)
    merged, dates = {}, []
    for shard in sorted(shards.values(), key=lambda item: item.get("as_of", "")):
        if shard.get("as_of"):
            dates.append(shard["as_of"])
            for ident, row in shard.get("rows", {}).items():
                if ident not in merged or row["scores"] or not merged[ident]["scores"]:
                    merged[ident] = dict(row, as_of=shard["as_of"])
    failures = [s["error"] for s in shards.values() if s.get("error")]
    return list(merged.values()), {"at": max(dates, default=None), "coverage": len(merged),
                                  "partitions": len(dates), "expected_partitions": len(windows(sport, year)),
                                  "error": error or (failures[0] if failures else "") or ("" if feed.key else "not configured")}


def record(rows, key):
    counts = [0, 0, 0]
    for row in rows:
        index = row["keys"].index(key)
        score, allowed = row["scores"][index], row["scores"][1-index]
        counts[0 if score > allowed else 1 if score < allowed else 2] += 1
    return counts


def summarize(rows, fixtures, now, status, provider):
    as_of = status["at"]
    if not as_of:
        return 0
    identities, aliases = defaultdict(set), defaultdict(set)
    for row in rows:
        for key, ident in zip(row["keys"], row["ids"]):
            identities[key].add(ident)
            aliases[ident].add(key)
    grouped = defaultdict(list)
    pairs = defaultdict(list)
    finals = defaultdict(list)
    for row in rows:
        if any(len(identities[k]) != 1 or len(aliases[i]) != 1 for k,i in zip(row["keys"],row["ids"])):
            continue
        for key in row["keys"]:
            grouped[key].append(row)
        pairs[tuple(sorted(row["keys"]))].append(row)
        if row["scores"] is not None and row["start"] < now:
            finals[datetime.fromisoformat(row["start"]).astimezone(ET).date()].append(row)
    # Treat every result on a calendar day as unavailable to other games on that day.
    # This keeps date-only and full timestamps comparable and avoids using later scores.
    outcomes, prior = defaultdict(Counter), {}
    for day in sorted(finals):
        for row in finals[day]:
            rates = []
            for key in row["keys"]:
                counter = outcomes[key]
                rates.append((counter["points"]/counter["games"]) if counter["games"] else None)
            prior[row["id"]] = rates
        for row in finals[day]:
            for index, key in enumerate(row["keys"]):
                scored, allowed = row["scores"][index], row["scores"][1-index]
                outcomes[key].update({"points": 1 if scored > allowed else 0.5 if scored == allowed else 0, "games": 1})
    cutoff = utc(datetime.fromisoformat(now) - timedelta(days=7))
    total = 0
    for fixture in fixtures:
        keys = fixture["keys"]
        start = datetime.fromisoformat(fixture["start"]).astimezone(ET).date()
        same = [r for r in pairs[tuple(sorted(keys))]
                if datetime.fromisoformat(r["start"]).astimezone(ET).date() == start]
        # Same-day doubleheaders are ambiguous absent exact kickoff reconciliation.
        if len(same) != 1 or not same[0]["scheduled"] or same[0]["as_of"] < cutoff:
            continue
        teams = {}
        for key in keys:
            games = sorted((r for r in grouped[key] if r["scores"] is not None and r["start"] < now
                            and datetime.fromisoformat(r["start"]).astimezone(ET).date() < start),
                           key=lambda r: r["start"], reverse=True)
            if not games:
                continue
            sums = []
            strengths, win_rates = [], []
            recent = []
            for i, row in enumerate(games):
                side = row["keys"].index(key)
                scores = [row["scores"][side], row["scores"][1-side]]
                sums.append(scores)
                opponent = row["keys"][1-side]
                rating = row["elo"][1-side] if row["elo"] else None
                if type(rating) in (int, float) and 0 < rating < 10000:
                    strengths.append(rating)
                rate = prior[row["id"]][1-side]
                if rate is not None:
                    win_rates.append(rate)
                if i < 5:
                    game_date = datetime.fromisoformat(row["start"]).astimezone(ET).date()
                    recent.append({"opponent": row["names"][1-side], "at": str(game_date), "scores": scores})
            sites = {}
            for label, side in (("home", 1), ("away", 0), ("neutral", None)):
                at_site = [r for r in games if (r["venue"] == "neutral" if side is None else
                           r["venue"] != "neutral" and r["keys"][side] == key)]
                if at_site:
                    sites[label] = record(at_site, key)
            teams[key] = {"games": len(games), "record": record(games, key),
                          "scored": round(sum(s[0] for s in sums)/len(games), 2),
                          "allowed": round(sum(s[1] for s in sums)/len(games), 2),
                          "recent": recent, "sites": sites,
                          "scheduled": sum(r["scheduled"] and r["start"] >= now for r in grouped[key]),
                          "opponent_elo": round(sum(strengths)/len(strengths), 1) if strengths else None,
                          "rated_games": len(strengths),
                          "opponent_win_pct": round(sum(win_rates)/len(win_rates), 4) if win_rates else None,
                          "opponent_win_games": len(win_rates),
                          "venue_verified": all(r["venue"] is not None for r in games)}
        if len(teams) == 2:
            fixture.setdefault("sporting", {})["independent"] = {
                "provider": LABELS[provider], "at": same[0]["as_of"], "teams": teams,
                "partitions": status["partitions"], "expected_partitions": status["expected_partitions"],
            }
            total += 1
    return total


def cached(feed, state, name, url, params, decode, hours, *, cost=1, fighter=False):
    path = state / (name + ".json")
    stored = load(path)
    now = feed.now.timestamp()
    if not feed.key or stored.get("next", 0) > now:
        return stored
    try:
        response = feed.request(url, params, cost=cost, fighter=fighter)
        stored = {"data": decode(response), "at": utc(feed.now), "next": now + hours * 3600}
    except FeedUnavailable as issue:
        if str(issue) in {"deferred", "budget deferred", "provider backoff"}:
            return stored
        stored.update(error=str(issue), next=now+3600)
    except (TypeError, ValueError, KeyError, AttributeError, IndexError):
        stored.update(error="schema or coverage validation", next=now+3600)
    save(path, stored)
    return stored


def nhl_standings(payload):
    table = {}
    for row in payload["standings"]:
        key = canonical(row.get("teamName") or NHL_NAMES[row["teamAbbrev"]], "nhl")
        fields = ("rank", "wins", "losses", "otLosses", "points", "gamesPlayed")
        if key in table or any(type(row[f]) is not int or row[f] < 0 for f in fields):
            raise ValueError("invalid standings")
        if row["wins"] + row["losses"] + row["otLosses"] != row["gamesPlayed"]:
            raise ValueError("inconsistent standings")
        table[key] = {f: row[f] for f in fields}
        for field in ("corsiPct", "fenwickPct", "expectedGoalsFor", "expectedGoalsAgainst"):
            value = row.get(field)
            if type(value) in (int, float) and 0 <= value <= 10000:
                table[key][field] = value
    if len(table) != 32:
        raise ValueError("incomplete league standings")
    return table


def event_cards(payload):
    rows = payload["data"]
    if not isinstance(rows, list):
        raise ValueError("invalid calendar")
    return [{"id": provider_id(r["id"]), "start": utc_start(r["starts_at"])}
            for r in rows if r.get("org") == "ufc" and r.get("sport") == "mma"
            and r.get("status") in {"announced", "scheduled"}][:4]


def provider_id(value):
    if type(value) not in (int, str) or not value:
        raise ValueError("invalid provider ID")
    identity = str(value)
    if not IDENTIFIER.fullmatch(identity):
        raise ValueError("invalid provider ID")
    return identity


def event_pairs(payload, identity):
    event = payload["data"]
    if provider_id(event["id"]) != identity or event.get("org") != "ufc":
        raise ValueError("bad event mapping")
    rows = []
    for bout in event["card"]:
        if bout.get("status") not in {"announced", "confirmed"} or bout.get("sport") != "mma":
            continue
        sides = [bout.get(k) for k in ("fighter_a", "fighter_b")]
        if any(not s or not s.get("id") or not s.get("name") for s in sides):
            continue
        keys = [canonical(s["name"], "ufc") for s in sides]
        ids = [provider_id(s["id"]) for s in sides]
        if len(set(ids)) != 2 or len(set(keys)) != 2:
            raise ValueError("ambiguous fighter identity")
        rows.append(dict(zip(keys,ids)))
    return {"start": utc_start(event["starts_at"]), "bouts": rows,
            "active": event.get("status") in {"announced", "scheduled"}}


def career_profile(payload, identity):
    row = payload["data"]
    if provider_id(row["id"]) != identity:
        raise ValueError("fighter ID mismatch")
    records = {}
    for scope in ("pro_mma", "ufc"):
        item = (row.get("records") or {}).get(scope) or {}
        counts = [item.get(k) for k in ("wins", "losses", "draws")]
        if all(type(x) is int and 0 <= x <= 500 for x in counts):
            records[scope] = counts
            if type(item.get("no_contests")) is int and item["no_contests"] > 0:
                records[scope + "_nc"] = item["no_contests"]
    result = {"key": canonical(row["name"], "ufc"), "records": records}
    for field in ("height_inches", "reach_inches"):
        value = row.get(field)
        if type(value) in (int, float) and 36 <= value <= 105:
            result[field] = value
    if row.get("stance") and isinstance(row["stance"], str) and len(row["stance"]) < 30:
        result["stance"] = row["stance"]
    return result


def career_recent(payload):
    rows = payload["data"]
    if not isinstance(rows, list):
        raise ValueError("invalid fight history")
    result, seen = [], set()
    for row in rows:
        if (row.get("competition_level") not in {"pro", "professional"} or row.get("sport") != "mma"
                or row.get("status") not in {"win", "loss", "draw", "no_contest"}):
            continue
        when = date.fromisoformat(row["date"][:10])
        opponent = (row.get("opponent") or {}).get("name") or row.get("opponent_name")
        if not opponent:
            continue
        key = str(when), canonical(opponent, "ufc")
        if key not in seen:
            seen.add(key)
            result.append({"date": str(when), "opponent": opponent, "result": row["status"]})
    return sorted(result, key=lambda item: item["date"], reverse=True)[:8]


def ufc_context(feed, state, fixtures):
    url = PROVIDERS["ufc"][1]
    calendar = cached(feed, state, "ufc-calendar", url+"/events", {"org":"ufc","status":"upcoming","limit":12},
                      event_cards, 48)
    pairs = defaultdict(list)
    for item in calendar.get("data", []):
        if item["start"] < utc(feed.now) or datetime.fromisoformat(item["start"]) > feed.now + timedelta(days=55):
            continue
        ident = item["id"]
        detail = cached(feed, state, f"ufc-card-{ident}", url+f"/events/{ident}", {},
                        lambda payload: event_pairs(payload, ident), 72)
        data = detail.get("data") or {}
        if data.get("active") and detail["at"] >= utc(feed.now - timedelta(days=7)):
            for bout in data["bouts"]:
                pairs[tuple(sorted(bout))].append((data["start"], bout))
    targets = {}
    for fixture in sorted(fixtures, key=lambda row: row["start"]):
        kickoff = datetime.fromisoformat(fixture["start"])
        if kickoff > feed.now + timedelta(days=35):
            continue
        matches = [bout for start,bout in pairs[tuple(sorted(fixture["keys"]))]
                   if abs((datetime.fromisoformat(start)-kickoff).total_seconds()) <= 20*3600]
        if len(matches) == 1:
            targets[fixture["start"], tuple(fixture["keys"])] = fixture, matches[0]
    covered = 0
    matched_ids = dict.fromkeys(ident for _, matchup in targets.values() for ident in matchup.values())
    histories = {ident: cached(feed, state, f"ufc-history-{ident}", url+f"/fighters/{ident}/history", {},
                               career_recent, 24*30, fighter=True) for ident in matched_ids}
    profiles = {}
    for ident, history in histories.items():
        profile = load(state / f"ufc-profile-{ident}.json")
        if history.get("at") and not history.get("error") and feed.remaining:
            profile = cached(feed, state, f"ufc-profile-{ident}", url+f"/fighters/{ident}", {},
                             lambda payload: career_profile(payload, ident), 24*30, fighter=True)
        profiles[ident] = profile
    for fixture, matchup in targets.values():
        fighters = {}
        for key, ident in matchup.items():
            history = histories[ident]
            profile = profiles[ident]
            current = profile.get("data", {}) if profile.get("at", "") >= utc(feed.now - timedelta(days=60)) else {}
            recent = [row for row in history.get("data", []) if row["date"] < fixture["start"][:10]]
            if current and current.get("key") != key:
                continue
            if current or recent:
                fighters[key] = {**current, "recent": recent[:5]}
        if fighters:
            fixture.setdefault("sporting", {})["ufc_independent"] = fighters
            covered += 1
    return {"at": calendar.get("at"), "coverage": covered,
            "error": calendar.get("error", "" if feed.key else "not configured")}


def independent_context(state, fixtures, sport, now, guard, network, key_directory=KEY_DIRECTORY):
    if sport not in PROVIDERS:
        return {}
    timestamp = datetime.fromisoformat(now)
    upcoming = [r for r in fixtures.values() if r["start"] > now and r.get("markets")]
    provider = PROVIDERS[sport][0]
    feed = Feed(state, provider, timestamp, guard, network, key_directory)
    if not upcoming:
        return {provider: {"coverage": 0, "error": "no upcoming market matchups"}}
    if not feed.key:
        return {provider: {"coverage": 0, "error": "not configured"}}
    if sport == "ufc":
        return {provider: ufc_context(feed, state, upcoming)}
    upcoming.sort(key=lambda fixture: fixture["start"])
    year = upcoming[0]["season"]
    upcoming = [r for r in upcoming if r["season"] == year]
    rows, status = schedule(feed, state, sport, year)
    covered = summarize(rows, upcoming, now, status, provider)
    if sport == "nhl":
        latest = cached(feed, state, "nhl-standings", "https://mcp.puckapi.com/v1/get_standings",
                        {"league": "nhl"}, nhl_standings, 16, cost=2)
        table = latest.get("data", {})
        if latest.get("at", "") >= utc(timestamp-timedelta(days=3)):
            for fixture in upcoming:
                if all(k in table for k in fixture["keys"]) and fixture.get("sporting", {}).get("independent"):
                    independent = fixture["sporting"]["independent"]
                    independent["standings"] = {k:table[k] for k in fixture["keys"]}
                    independent["standings_at"] = latest["at"]
    return {provider: {**status, "fixture_coverage": covered}}
