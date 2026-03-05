"""
Team name normalization and alias resolution.
Extracted from main.py for reuse across the 3-layer engine.

Single source of truth for:
- _norm_team(): basic string cleanup
- TEAM_ALIASES: DK name -> canonical form
- API_TEAM_ALIASES: Odds-API name -> canonical form
- NCAAB_MASCOTS: college mascots to strip from Odds-API names
- normalize_team_name(): full normalization pipeline
- _split_game(): "Away @ Home" -> (away, home) tuple
"""
import re


def _split_game(game) -> tuple:
    """Split 'Away @ Home' or 'Away vs Home' into (away, home) tuple."""
    try:
        import pandas as pd
        if pd.isna(game):
            return "", ""
    except Exception:
        pass

    g = str(game).strip() if game is not None else ""
    if not g or g.lower() == "nan":
        return "", ""

    g = re.sub(r"\s+vs\.?\s+|\s+v\.?\s+", " @ ", g, flags=re.IGNORECASE)
    if " @ " in g:
        a, h = g.split(" @ ", 1)
        return a.strip(), h.strip()
    return "", ""


def _norm_team(s: str) -> str:
    """Basic cleanup: lowercase, strip non-alnum except &.-, collapse whitespace."""
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s&.-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ─── COLLEGE MASCOTS (for stripping from Odds-API names) ───
# Odds-API returns "school mascot" format for NCAAB/NCAAF.
# DK returns school name only. We strip the mascot to match.
# Sorted by length descending at runtime for greedy matching.
NCAAB_MASCOTS = {
    # Multi-word mascots (MUST be checked first)
    "fighting irish", "fighting illini", "fighting hawks", "fighting camels",
    "fighting leathernecks", "fighting scots",
    "yellow jackets", "golden eagles", "golden gophers",
    "golden bears", "golden flashes", "golden griffins",
    "golden knights", "golden panthers", "golden hurricane",
    "blue devils", "blue demons", "blue raiders", "blue jays",
    "blue hose", "blue hens", "blazing speed",
    "red storm", "red raiders", "red foxes", "red flash",
    "red wolves",
    "green wave", "mean green",
    "nittany lions", "crimson tide", "tar heels",
    "horned frogs", "runnin bulldogs",
    "sun devils", "rain bows", "rainbow warriors",
    "beach riders", "scarlet knights",
    "mountain hawks", "war eagles",
    "lumberjacks", "thundering herd",
    "river hawks", "beach boys",
    "black bears", "black knights",
    "great danes", "hilltoppers",
    "jack rabbits", "jackrabbits",
    "trail blazers",
    "running rebels",
    "flying dutchmen",
    "mad ants",
    "sea wolves",
    "purple aces", "purple eagles",
    "orange men", "orangemen",
    "holy cross crusaders",
    "red hawks", "redhawks",
    "tribe", "pride",

    # Single-word mascots (alphabetical)
    "49ers",
    "aggies", "anteaters", "aztecs",
    "badgers", "banana slugs", "bandits", "battlers",
    "beacons", "bearkats",
    "bears", "beavers", "bearcats", "bengals", "bighornss",
    "billikens", "bison", "blazers", "blue jays",
    "bobcats", "boilermakers", "bonnies", "boxers",
    "brahmans", "braves", "broncos", "broncs",
    "bruins", "buckeyes", "buccaneers", "buffaloes", "buffalo",
    "bulldogs", "bulls",
    "cadets", "camels", "canaries", "cardinals", "catamounts",
    "cavaliers", "chanticleers", "chippewas",
    "colonels", "colonials", "comets", "commodores",
    "cornhuskers", "cougars", "cowboys", "crimson",
    "crusaders", "cyclones",
    "deacons", "demon deacons", "dons", "dolphins",
    "dragons", "ducks", "dukes",
    "eagles", "explorers",
    "falcons", "flames", "flyers",
    "friars", "frogs",
    "gaels", "gamecocks", "gators",
    "generals", "governors", "greyhounds", "griffins", "grizzlies",
    "governors",
    "hatters", "hawkeyes", "hawks", "herons", "highlanders",
    "hokies", "hoosiers", "hornets", "huskies", "hurricanes",
    "ichabods",
    "jaguars", "jaspers", "javelinas", "jayhawks",
    "kangaroos", "keydets", "kingsmen", "knights", "koalas",
    "lancers", "leopards", "lions", "lobos",
    "longhorns",
    "mavericks", "mariners", "marlins", "mastodons", "matadors",
    "midshipmen", "miners", "minutemen", "mocassins", "monarchs",
    "mountaineers", "musketeers", "mustangs",
    "nighthawks",
    "ospreys", "otters", "owls",
    "paladins", "panthers", "patriots", "peacocks",
    "pelicans", "penguins", "phoenix", "pilots", "pioneers",
    "pirates", "privateers", "purple",
    "quakers",
    "racers", "railsplitters", "rams", "rattlers",
    "razorbacks", "rebels", "retrievers",
    "revolutionaries", "roadrunners", "rockets", "roos",
    "sagehens", "saints", "salukis", "samurai",
    "scarlet", "seahawks", "seawolves",
    "seminoles", "senators", "sharks",
    "shockers", "skyhawks", "sooners", "spartans",
    "spiders", "stags", "stallions", "statesmen", "sycamores",
    "stormy petrels",
    "terrapins", "terriers", "texans", "thunderbirds",
    "tigers", "titans", "tommies", "toads",
    "toppers", "toreros", "trojans",
    "tritons", "tribe",
    "utes",
    "vandals", "vikings", "volunteers", "vulcans",
    "warriors", "wasps", "waves", "westerners",
    "wildcats", "wolfpack", "wolverines", "wolves",
    "wombats", "wren",
    "yellowjackets",
    "zags", "zips",

    # NCAAF-specific multi-word
    "soaring eagles", "bronco busters",
}

# Pre-sorted by length (longest first) for greedy matching
_MASCOTS_SORTED = sorted(NCAAB_MASCOTS, key=len, reverse=True)


def _strip_mascot(name: str) -> str:
    """
    Strip trailing mascot from Odds-API style college names.
    "butler bulldogs" -> "butler"
    "florida st seminoles" -> "florida st"
    "csu northridge matadors" -> "csu northridge"
    """
    for mascot in _MASCOTS_SORTED:
        if name.endswith(" " + mascot):
            stripped = name[: -(len(mascot) + 1)].strip()
            if stripped:  # Don't return empty string
                return stripped
    return name


# ─── MAIN ALIAS MAP ───
# All keys MUST be lowercase (output of _norm_team).
# DK name -> canonical form.
TEAM_ALIASES = {
    # ─── Miami variations ───
    "miami fl": "miami",
    "miami (fl)": "miami",
    "miami florida": "miami",
    "miami oh": "miami oh",

    # ─── State abbreviations (NCAAB/NCAAF) ───
    "arizona state": "arizona st",
    "arizona st.": "arizona st",
    "penn state": "penn st",
    "oklahoma state": "oklahoma st",
    "ohio state": "ohio st",
    "kansas state": "kansas st",
    "iowa state": "iowa st",
    "florida state": "florida st",
    "mississippi state": "mississippi st",
    "louisiana state": "lsu",
    "michigan state": "michigan st",
    "colorado state": "colorado st",
    "washington state": "washington st",
    "oregon state": "oregon st",
    "boise state": "boise st",
    "fresno state": "fresno st",
    "san diego state": "san diego st",
    "san jose state": "san jose st",
    "ball state": "ball st",
    "kent state": "kent st",
    "wright state": "wright st",
    "wichita state": "wichita st",
    "weber state": "weber st",
    "murray state": "murray st",
    "norfolk state": "norfolk st",
    "portland state": "portland st",
    "baylor state": "baylor",
    "north carolina state": "nc state",
    "north carolina st": "nc state",
    "nc state": "nc state",
    "illinois state": "illinois st",
    "indiana state": "indiana st",
    "idaho state": "idaho st",
    "utah state": "utah st",
    "appalachian state": "appalachian st",
    "georgia state": "georgia st",
    "sacramento state": "sacramento st",
    "chicago state": "chicago st",
    "alcorn state": "alcorn st",
    "coppin state": "coppin st",
    "delaware state": "delaware st",
    "jackson state": "jackson st",
    "morgan state": "morgan st",
    "south carolina state": "sc state",
    "virginia state": "virginia st",
    "tarleton state": "tarleton st",

    # ─── Directional schools ───
    "southern miss": "southern miss",
    "central michigan": "central mich",
    "eastern michigan": "eastern mich",
    "western michigan": "western mich",
    "western michigan": "w michigan",
    "northern illinois": "n illinois",
    "southern illinois": "southern illinois",
    "eastern illinois": "e illinois",
    "western illinois": "w illinois",
    "eastern kentucky": "eastern kentucky",
    "western kentucky": "western kentucky",
    "northern kentucky": "n kentucky",
    "east carolina": "east carolina",
    "west virginia": "west virginia",

    # ─── NCAAB problem teams (DK -> canonical) ───
    "boston university": "boston u",
    "saint peters": "st peters",
    "saint peter's": "st peters",
    "siu edwardsville": "siue",
    "siu-edwardsville": "siue",
    "iu indianapolis": "iu indy",
    "cal state fullerton": "cs fullerton",
    "cal st fullerton": "cs fullerton",
    "queens nc": "queens",
    "queens charlotte": "queens",
    "east texas am": "tx am commerce",
    "texas am commerce": "tx am commerce",
    "texas a&m commerce": "tx am commerce",
    "texas a&m": "texas am",
    "texas am": "texas am",
    "connecticut": "uconn",
    "southern california": "usc",
    "southern methodist": "smu",
    "central florida": "ucf",
    "nevada-las vegas": "unlv",
    "texas-el paso": "utep",
    "virginia commonwealth": "vcu",
    "massachusetts": "umass",
    "george washington": "george washington",
    "george mason": "george mason",
    "saint louis": "saint louis",
    "saint marys": "saint marys",
    "st. marys": "saint marys",
    "saint josephs": "saint josephs",
    "st. josephs": "saint josephs",
    "saint johns": "st johns",
    "st. johns": "st johns",
    "saint bonaventure": "st bonaventure",
    "st. bonaventure": "st bonaventure",
    "saint thomas mn": "st thomas mn",
    "st. thomas mn": "st thomas mn",
    "south carolina upstate": "usc upstate",
    "florida international": "fiu",
    "florida intl": "fiu",
    "loyola chicago": "loyola chicago",
    "loyola chi": "loyola chicago",
    "loyola marymount": "loyola marymount",
    "long island": "liu",
    "long island university": "liu",
    "csun": "csu northridge",
    "csu northridge": "csu northridge",
    "cal st northridge": "csu northridge",
    "csu bakersfield": "csu bakersfield",
    "cal st bakersfield": "csu bakersfield",
    "uc davis": "uc davis",
    "uc irvine": "uc irvine",
    "uc riverside": "uc riverside",
    "uc santa barbara": "uc santa barbara",
    "uc san diego": "uc san diego",
    "bowling green": "bowling green",
    "north carolina": "north carolina",
    "south carolina": "south carolina",
    "north texas": "north texas",
    "middle tennessee": "middle tenn",
    "middle tenn": "middle tenn",
    "long beach state": "long beach st",
    "long beach st": "long beach st",
    "cal state fullerton": "cs fullerton",
    "cal st fullerton": "cs fullerton",
    "california baptist": "cal baptist",
    "abilene christian": "abilene christian",
    "jacksonville state": "jacksonville st",
    "jacksonville st": "jacksonville st",
    "new mexico state": "new mexico st",
    "new mexico st": "new mexico st",
    "kennesaw state": "kennesaw st",
    "kennesaw st": "kennesaw st",
    "south dakota state": "south dakota st",
    "south dakota st": "south dakota st",
    "north dakota state": "north dakota st",
    "north dakota st": "north dakota st",
    "ut arlington": "ut arlington",
    "texas-arlington": "ut arlington",
    "missouri state": "missouri st",
    "missouri st": "missouri st",
    "sam houston": "sam houston",
    "sam houston state": "sam houston",

    # ─── NBA shorthand (DK -> Odds-API full names) ───
    "atl hawks": "atlanta hawks",
    "bkn nets": "brooklyn nets",
    "bos celtics": "boston celtics",
    "cha hornets": "charlotte hornets",
    "chi bulls": "chicago bulls",
    "cle cavaliers": "cleveland cavaliers",
    "dal mavericks": "dallas mavericks",
    "den nuggets": "denver nuggets",
    "det pistons": "detroit pistons",
    "gs warriors": "golden state warriors",
    "hou rockets": "houston rockets",
    "ind pacers": "indiana pacers",
    "la clippers": "los angeles clippers",
    "la lakers": "los angeles lakers",
    "mem grizzlies": "memphis grizzlies",
    "mia heat": "miami heat",
    "mil bucks": "milwaukee bucks",
    "min timberwolves": "minnesota timberwolves",
    "no pelicans": "new orleans pelicans",
    "ny knicks": "new york knicks",
    "okc thunder": "oklahoma city thunder",
    "orl magic": "orlando magic",
    "phi 76ers": "philadelphia 76ers",
    "pho suns": "phoenix suns",
    "por trail blazers": "portland trail blazers",
    "sa spurs": "san antonio spurs",
    "sac kings": "sacramento kings",
    "tor raptors": "toronto raptors",
    "uta jazz": "utah jazz",
    "was wizards": "washington wizards",

    # ─── NHL shorthand (DK -> Odds-API full names) ───
    "ana ducks": "anaheim ducks",
    "bos bruins": "boston bruins",
    "buf sabres": "buffalo sabres",
    "car hurricanes": "carolina hurricanes",
    "cbj blue jackets": "columbus blue jackets",
    "cgy flames": "calgary flames",
    "chi blackhawks": "chicago blackhawks",
    "col avalanche": "colorado avalanche",
    "dal stars": "dallas stars",
    "det red wings": "detroit red wings",
    "edm oilers": "edmonton oilers",
    "fla panthers": "florida panthers",
    "la kings": "los angeles kings",
    "min wild": "minnesota wild",
    "mtl canadiens": "montreal canadiens",
    "nj devils": "new jersey devils",
    "nsh predators": "nashville predators",
    "ny islanders": "new york islanders",
    "ny rangers": "new york rangers",
    "ott senators": "ottawa senators",
    "phi flyers": "philadelphia flyers",
    "pit penguins": "pittsburgh penguins",
    "sea kraken": "seattle kraken",
    "sj sharks": "san jose sharks",
    "stl blues": "st louis blues",
    "tb lightning": "tampa bay lightning",
    "tor maple leafs": "toronto maple leafs",
    "van canucks": "vancouver canucks",
    "vgk golden knights": "vegas golden knights",
    "was capitals": "washington capitals",
    "wpg jets": "winnipeg jets",
    # NHL expansion — Utah Mammoth (2024-25 rebrand)
    "uta mammoth": "utah mammoth",
    "utah hockey club": "utah mammoth",

    # ─── NFL shorthand (DK -> full names) ───
    "ari cardinals": "arizona cardinals",
    "atl falcons": "atlanta falcons",
    "bal ravens": "baltimore ravens",
    "buf bills": "buffalo bills",
    "car panthers": "carolina panthers",
    "chi bears": "chicago bears",
    "cin bengals": "cincinnati bengals",
    "cle browns": "cleveland browns",
    "dal cowboys": "dallas cowboys",
    "den broncos": "denver broncos",
    "det lions": "detroit lions",
    "gb packers": "green bay packers",
    "hou texans": "houston texans",
    "ind colts": "indianapolis colts",
    "jax jaguars": "jacksonville jaguars",
    "kc chiefs": "kansas city chiefs",
    "lv raiders": "las vegas raiders",
    "lac chargers": "los angeles chargers",
    "lar rams": "los angeles rams",
    "mia dolphins": "miami dolphins",
    "min vikings": "minnesota vikings",
    "ne patriots": "new england patriots",
    "no saints": "new orleans saints",
    "nyg giants": "new york giants",
    "nyj jets": "new york jets",
    "phi eagles": "philadelphia eagles",
    "pit steelers": "pittsburgh steelers",
    "sf 49ers": "san francisco 49ers",
    "sea seahawks": "seattle seahawks",
    "tb buccaneers": "tampa bay buccaneers",
    "ten titans": "tennessee titans",
    "was commanders": "washington commanders",

    # ─── MLB shorthand (DK -> full names) ───
    "ari diamondbacks": "arizona diamondbacks",
    "atl braves": "atlanta braves",
    "bal orioles": "baltimore orioles",
    "bos red sox": "boston red sox",
    "chc cubs": "chicago cubs",
    "chw white sox": "chicago white sox",
    "cin reds": "cincinnati reds",
    "cle guardians": "cleveland guardians",
    "col rockies": "colorado rockies",
    "det tigers": "detroit tigers",
    "hou astros": "houston astros",
    "kc royals": "kansas city royals",
    "laa angels": "los angeles angels",
    "lad dodgers": "los angeles dodgers",
    "mia marlins": "miami marlins",
    "mil brewers": "milwaukee brewers",
    "min twins": "minnesota twins",
    "nym mets": "new york mets",
    "nyy yankees": "new york yankees",
    "oak athletics": "oakland athletics",
    "phi phillies": "philadelphia phillies",
    "pit pirates": "pittsburgh pirates",
    "sd padres": "san diego padres",
    "sf giants": "san francisco giants",
    "sea mariners": "seattle mariners",
    "stl cardinals": "st louis cardinals",
    "tb rays": "tampa bay rays",
    "tex rangers": "texas rangers",
    "tor blue jays": "toronto blue jays",
    "was nationals": "washington nationals",

    # ─── ESPN shortDisplayName quirks ───
    "pittsburgh": "pitt",
    "albany": "ualbany",
    "albany ny": "ualbany",
}

# ─── API-SOURCE ALIASES ───
# The-Odds-API team names -> canonical form.
# Handles typos, accent issues, and format differences from the API.
API_TEAM_ALIASES = {
    # NHL — Odds-API typos and accent issues
    "montral canadiens": "montreal canadiens",  # Missing 'e' in Montreal

    # NCAAB — Odds-API abbreviations that differ from DK after mascot stripping
    "gw": "george washington",
    "siu-edwardsville": "siue",
    "siu edwardsville": "siue",
    "loyola chi": "loyola chicago",
    "florida intl": "fiu",
    "south carolina upstate": "usc upstate",
    "cal st northridge": "csu northridge",
    "cal st bakersfield": "csu bakersfield",
    "st. thomas mn": "st thomas mn",
    "san jos st": "san jose st",  # Odds-API typo: missing 'e' in Jose

    # NCAAB — schools that map to common abbreviations
    "uconn": "uconn",
    "connecticut": "uconn",
    "southern cal": "usc",
    "southern methodist": "smu",
    "central florida": "ucf",

    # UFC — fighter name variations from Odds-API
    "jeongyoung lee": "jeong yeong lee",
    "jesus santos aguilar": "jesus aguilar",
    "su mudaerji": "sumudaerji",
    "sumudaerji sumudaerji": "sumudaerji",
    "raul rosas jr.": "raul rosas jr",
    "raul rosas jr": "raul rosas jr",
}


def normalize_team_name(name: str, sport: str = "") -> str:
    """
    Full normalization: cleanup + mascot strip (college) + alias lookup.

    Args:
        name: Raw team/fighter name from any source
        sport: Sport key (ncaab, ncaaf, ufc, nba, etc.) — used for
               college mascot stripping. Optional for backward compat.
    """
    n = _norm_team(name)

    # Strip mascots for college sports (Odds-API uses "school mascot" format)
    if sport in ("ncaab", "ncaaf"):
        n = _strip_mascot(n)

    # Check DK aliases first, then API aliases
    if n in TEAM_ALIASES:
        return TEAM_ALIASES[n]
    if n in API_TEAM_ALIASES:
        return API_TEAM_ALIASES[n]
    return n
