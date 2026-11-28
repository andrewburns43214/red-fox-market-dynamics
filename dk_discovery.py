"""Raw DK market-header discovery, independent of successful split parsing."""
import re
import pandas as pd
from bs4 import BeautifulSoup
from dk_headless import parse_dk_start_text_to_utc_iso

HEADER_MARKETS = {"RUN LINE": "SPREAD", "PUCK LINE": "SPREAD"}
SUPPORTED_MARKETS = {"SPREAD", "MONEYLINE", "TOTAL"}
SPORT_LABELS = {
    "mlb": "mlb", "nfl": "nfl", "nfl preseason": "nfl",
    "national football league": "nfl", "nba": "nba", "nhl": "nhl",
    "ufc": "ufc", "ncaa football": "ncaaf", "college football": "ncaaf",
    "ncaaf": "ncaaf", "ncaa basketball": "ncaab",
    "college basketball": "ncaab", "ncaab": "ncaab",
}


def selected_sport_key(soup):
    """Read the selected league across DK's numeric-value and legacy forms."""
    selected = soup.select_one('select[name="tb_eg"] option[selected]')
    hidden = soup.select_one('input[name="itm_content"]')
    candidates = []
    if selected:
        candidates.extend((selected.get_text(" ", strip=True), selected.get("value", "")))
    if hidden:
        candidates.append(hidden.get("value", ""))
    for candidate in candidates:
        sport = SPORT_LABELS.get(str(candidate or "").strip().casefold())
        if sport:
            return sport
    return ""


def utc(value):
    value = pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")

def inventory_html(html, sport, observed_at, source=""):
    """Discover headers independently of progress bars and odds parsing.

    Sport is verified against the selected form, not inferred from the requested
    URL. Headerless event blocks get UNKNOWN market records for investigation.
    Yearless historical DK labels are anchored to capture time, not today's year.
    """
    soup = BeautifulSoup(html, "html.parser")
    selected = soup.select_one('select[name="tb_eg"] option[selected]')
    label = selected.get_text(" ", strip=True) if selected else ""
    league = selected_sport_key(soup)
    identified = league == sport
    now = utc(observed_at).tz_convert("America/New_York")
    out = []
    for index, section in enumerate(soup.select("div.tb-se")):
        anchor = section.select_one('a[href*="/event/"], a[href*="eventId="]')
        href = anchor.get("href", "") if anchor else ""
        match = re.search(r"/event/(\d+)|eventId=(\d+)", href)
        gid = next((x for x in match.groups() if x), "") if match else ""
        game = anchor.get_text(" ", strip=True) if anchor else ""
        kickoff = next((parsed for span in section.select(".tb-se-title span")
                        if (parsed := parse_dk_start_text_to_utc_iso(span.get_text(" ", strip=True), now))), "")
        kickoff = pd.Timestamp(kickoff).isoformat() if kickoff else ""
        # Current pages use a shared Selection/Odds/Handle/Bets header and
        # give each actual market its own heading below it.
        headers = section.select(".tb-market-wrap > div > div:first-child")
        if not headers:
            headers = [header.find("div") or header for header in section.select(".tb-se-head")]
        names = []
        for header in headers:
            name = header.get_text(" ", strip=True).upper()
            market = HEADER_MARKETS.get(name, name)
            names.append(market if market in SUPPORTED_MARKETS else "UNKNOWN")
        for market in dict.fromkeys(names or ["UNKNOWN"]):
            out.append(dict(sport=sport, game_id=gid or f"UNKNOWN:{source}:{index}", game=game,
                            market_display=market, dk_start_iso=kickoff, source=source,
                            discovered_at=str(observed_at), source_league=label,
                            league_identified=identified, identity_verified=bool(gid),
                            discovery_basis="RAW_DK_HEADER"))
    return pd.DataFrame(out)
