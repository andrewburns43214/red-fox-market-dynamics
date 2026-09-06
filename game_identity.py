"""Deterministic cross-source identity only; never changes scoring/ledger keys."""
import re
import unicodedata


def clean_name(value):
    value = unicodedata.normalize("NFKD", str(value or "")).casefold()
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = re.sub(r"['’‘ʻʼ]", "", value)
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


# Explicit school identities. Never collapse directional schools, Miami campuses,
# Nevada/UNLV, USC/South Carolina, or Saint/State using token overlap.
COLLEGE_ALIASES = {
    "unlv": "unlv", "nevada las vegas": "unlv", "university of nevada las vegas": "unlv",
    "unlv rebels": "unlv", "nevada las vegas rebels": "unlv",
    "hawaii": "hawaii", "hawaii rainbow warriors": "hawaii",
    "hawaii rainbow wahines": "hawaii", "haw": "hawaii", "hawaii manoa": "hawaii",
    "university of hawaii at manoa": "hawaii",
    "miami fl": "miami fl", "miami florida": "miami fl", "miami hurricanes": "miami fl",
    "miami oh": "miami oh", "miami ohio": "miami oh", "miami oh redhawks": "miami oh",
    "miami redhawks": "miami oh", "miami university": "miami oh",
    "lsu": "lsu", "louisiana state": "lsu", "louisiana state tigers": "lsu",
    "lsu tigers": "lsu", "uconn": "connecticut", "connecticut huskies": "connecticut",
    "ole miss": "mississippi", "ole miss rebels": "mississippi", "mississippi rebels": "mississippi",
    "nc state": "north carolina state", "nc st": "north carolina state",
    "pitt": "pittsburgh", "pittsburgh panthers": "pittsburgh",
    "umass": "massachusetts", "massachusetts minutemen": "massachusetts",
    "ucf": "central florida", "ucf knights": "central florida", "central florida knights": "central florida",
    "usf": "south florida", "south florida bulls": "south florida",
    "byu": "brigham young", "byu cougars": "brigham young",
    "tcu": "texas christian", "tcu horned frogs": "texas christian",
    "smu": "southern methodist", "smu mustangs": "southern methodist",
    "utsa": "texas san antonio", "utsa roadrunners": "texas san antonio",
    "utep": "texas el paso", "utep miners": "texas el paso",
    "ulm": "louisiana monroe", "ul monroe": "louisiana monroe",
    "app state": "appalachian state", "appalachian st": "appalachian state",
    "w mich": "western michigan", "e mich": "eastern michigan", "c mich": "central michigan",
    "n illinois": "northern illinois", "niu": "northern illinois",
    "s illinois": "southern illinois", "se missouri state": "southeast missouri state",
    "st johns": "saint johns", "st marys": "saint marys", "st peters": "saint peters",
    "n carolina": "north carolina", "s carolina": "south carolina",
    "n dakota": "north dakota", "s dakota": "south dakota",
    "n dakota state": "north dakota state", "s dakota state": "south dakota state",
    "e kentucky": "eastern kentucky", "w kentucky": "western kentucky",
    "n arizona": "northern arizona", "e washington": "eastern washington",
    "fiu": "florida international", "fau": "florida atlantic",
    "bgsu": "bowling green", "bowling green state": "bowling green",
    "jmu": "james madison", "odu": "old dominion", "ecu": "east carolina",
}
AMBIGUOUS_COLLEGE = {"miami", "usc", "uh", "state", "saint marys", "saint johns", "rainbow warriors"}


def team_identity(value, sport):
    name = clean_name(value)
    if not name:
        return "", "INVALID_TEAM_IDENTITY"
    if sport in {"ncaaf", "ncaab"}:
        # Context-free abbreviations/mascot-only strings must not invent a school.
        if name in AMBIGUOUS_COLLEGE:
            return name, "AMBIGUOUS_TEAM_IDENTITY"
        name = COLLEGE_ALIASES.get(name, name)
        if name.endswith(" st"):
            name = name[:-3] + " state"
        name = COLLEGE_ALIASES.get(name, name)
        if name in AMBIGUOUS_COLLEGE:
            return name, "AMBIGUOUS_TEAM_IDENTITY"
    else:
        from team_aliases import TEAM_ALIASES
        aliases = {clean_name(k): clean_name(v) for k, v in TEAM_ALIASES.items()}
        name = aliases.get(name, name)
        if sport == "nfl":
            name = {"ny jets": "new york jets", "ny giants": "new york giants",
                    "la rams": "los angeles rams", "la chargers": "los angeles chargers"}.get(name, name)
    return name, "IDENTIFIED"


def game_identity(game, sport):
    parts = re.split(r"\s*(?:@|\bvs\.?\b|\bv\.\b)\s*", str(game or ""), flags=re.I)
    if len(parts) != 2:
        return None, "INVALID_GAME_IDENTITY"
    a, a_state = team_identity(parts[0], sport)
    h, h_state = team_identity(parts[1], sport)
    if a_state != "IDENTIFIED" or h_state != "IDENTIFIED":
        return None, "AMBIGUOUS_TEAM_IDENTITY" if "AMBIGUOUS" in a_state + h_state else "INVALID_GAME_IDENTITY"
    if a == h:
        return None, "INVALID_GAME_IDENTITY"
    return (a, h), "IDENTIFIED"


class KickoffMatches(dict):
    def __init__(self):
        super().__init__()
        self.states = {}


def match_games(games, events, sport):
    """Use exact pairs of explicit ESPN name variants, keeping collisions visible."""
    index = {}
    for event in events:
        comps = event.get("competitions") or []
        if not comps or not event.get("date"):
            continue
        teams = comps[0].get("competitors") or []
        variants = {}
        for c in teams:
            team = c.get("team") or {}
            variants[c.get("homeAway")] = {
                team_identity(team.get(field), sport)[0] for field in
                ["displayName", "shortDisplayName", "location", "abbreviation"]
                if team.get(field) and team_identity(team[field], sport)[1] == "IDENTIFIED"}
        for away in variants.get("away", set()):
            for home in variants.get("home", set()):
                index.setdefault((away, home), set()).add((str(event.get("id", event["date"])), event["date"]))
    result = KickoffMatches()
    for game in games:
        key, state = game_identity(game, sport)
        matches = index.get(key, set()) if key else set()
        result[game] = next(iter(matches))[1] if len(matches) == 1 else ""
        result.states[game] = state if not key else "ESPN_MATCHED" if len(matches) == 1 else "ESPN_AMBIGUOUS" if matches else "ESPN_UNMATCHED"
    return result
