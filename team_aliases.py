"""
Team name normalization and alias resolution.
Extracted from main.py for reuse across the 3-layer engine.

Single source of truth for:
- _norm_team(): basic string cleanup
- TEAM_ALIASES: DK name -> canonical form
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


# Main alias map. All keys MUST be lowercase (output of _norm_team).
# DK name -> canonical form (ESPN shortDisplayName style).
TEAM_ALIASES = {
    # Miami variations
    "miami fl": "miami",
    "miami (fl)": "miami",
    "miami florida": "miami",

    # State abbreviations
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

    # Directional schools
    "southern miss": "southern miss",
    "central michigan": "central mich",
    "eastern michigan": "eastern mich",
    "western michigan": "western mich",

    # NCAAB problem teams (DK -> ESPN-ish)
    "boston university": "boston u",
    "saint peters": "st peters",
    "siu edwardsville": "siue",
    "iu indianapolis": "iu indy",
    "cal state fullerton": "cs fullerton",
    "cal st fullerton": "cs fullerton",
    "queens nc": "queens",
    "queens charlotte": "queens",
    "east texas am": "tx am commerce",
    "texas am commerce": "tx am commerce",

    # NBA shorthand (DK-style city prefixes → Odds-API full names)
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

    # NHL shorthand (DK-style city prefixes → Odds-API full names)
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

    # ESPN shortDisplayName quirks
    "pittsburgh": "pitt",
    "western michigan": "w mich",
    "albany": "ualbany",
    "albany ny": "ualbany",
}

# API-source aliases (The-Odds-API team names -> canonical form).
# Populated in Phase 1 by running API scrape and comparing to DK names.
API_TEAM_ALIASES = {}


def normalize_team_name(name: str) -> str:
    """Full normalization: cleanup + alias lookup (DK + API sources)."""
    n = _norm_team(name)
    # Check DK aliases first, then API aliases
    if n in TEAM_ALIASES:
        return TEAM_ALIASES[n]
    if n in API_TEAM_ALIASES:
        return API_TEAM_ALIASES[n]
    return n
