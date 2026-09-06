"""Raw DK market-header discovery, independent of successful split parsing."""
import re
import pandas as pd
from bs4 import BeautifulSoup
HEADER_MARKETS = {"RUN LINE": "SPREAD", "PUCK LINE": "SPREAD"}
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
    label = selected.get("value", "") if selected else ""
    league = {"NCAA Football": "ncaaf", "College Football": "ncaaf", "NCAAF": "ncaaf",
              "NCAA Basketball": "ncaab", "College Basketball": "ncaab", "NCAAB": "ncaab",
              "NFL Preseason": "nfl", "National Football League": "nfl"}.get(label, label.lower())
    identified = league == sport
    now = utc(observed_at).tz_convert("America/New_York")
    out = []
    for index, section in enumerate(soup.select("div.tb-se")):
        anchor = section.select_one('a[href*="/event/"], a[href*="eventId="]')
        href = anchor.get("href", "") if anchor else ""
        match = re.search(r"/event/(\d+)|eventId=(\d+)", href)
        gid = next((x for x in match.groups() if x), "") if match else ""
        game = anchor.get_text(" ", strip=True) if anchor else ""
        time_text = " ".join(x.get_text(" ", strip=True) for x in section.select(".tb-se-title span"))
        m = re.search(r"(\d{1,2})/(\d{1,2}),\s*(\d{1,2}):(\d{2})\s*(AM|PM)", time_text, re.I)
        kickoff = ""
        if m:
            month, day, hour, minute = map(int, m.groups()[:4])
            hour = hour % 12 + (12 if m.group(5).upper() == "PM" else 0)
            year = now.year + int(now.month == 12 and month == 1)
            try:
                kickoff = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute,
                                       tz="America/New_York").tz_convert("UTC").isoformat()
            except ValueError:
                pass
        headers = section.select(".tb-se-head")
        names = []
        for header in headers:
            first = header.find("div")
            name = (first or header).get_text(" ", strip=True).upper()
            names.append(HEADER_MARKETS.get(name, name))
        for market in dict.fromkeys(names or ["UNKNOWN"]):
            out.append(dict(sport=sport, game_id=gid or f"UNKNOWN:{source}:{index}", game=game,
                            market_display=market, dk_start_iso=kickoff, source=source,
                            discovered_at=str(observed_at), source_league=label,
                            league_identified=identified, identity_verified=bool(gid),
                            discovery_basis="RAW_DK_HEADER"))
    return pd.DataFrame(out)
