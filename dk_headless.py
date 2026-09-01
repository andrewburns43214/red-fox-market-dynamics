import json
import re
import shutil
from typing import Any, Dict, List, Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC
import time

import os
import tempfile
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
import logging

# Pylance/static: ensure this constant exists even if patterns are built later
_JSON_SCRIPT_PATTERNS = []
logger = logging.getLogger("dk")

SPORT_FILTER_ALIASES = {
    "nfl": ["NFL", "NFL Preseason", "National Football League"],
    "nba": ["NBA", "National Basketball Association"],
    "mlb": ["MLB", "Major League Baseball"],
    "nhl": ["NHL", "National Hockey League"],
    "ncaaf": ["NCAA Football", "College Football", "NCAAF"],
    "ncaab": ["NCAA Basketball", "College Basketball", "NCAAB"],
    "ufc": ["UFC"],
}


def _select_sport_option(select_el, requested_labels: list[str]) -> str:
    """
    Select the best visible sport option on the live DK form.
    Supports cases where DK splits preseason/postseason into different labels.
    Returns the visible text that was selected.
    """
    sel = Select(select_el)
    options = [o.text.strip() for o in sel.options if (o.text or "").strip()]
    option_map = {opt.lower(): opt for opt in options}

    for label in requested_labels:
        match = option_map.get(label.strip().lower())
        if match:
            sel.select_by_visible_text(match)
            return match

    for label in requested_labels:
        needle = label.strip().lower()
        for opt in options:
            opt_l = opt.lower()
            if needle in opt_l or opt_l in needle:
                sel.select_by_visible_text(opt)
                return opt

    raise ValueError(f"sport filter not found for labels={requested_labels!r}; options={options!r}")


def _with_cache_buster(url: str) -> str:
    epoch_ms = int(time.time() * 1000)
    p = urlparse(url)
    q = dict(parse_qsl(p.query, keep_blank_values=True))

    # IMPORTANT:
    # Do NOT overwrite tb_emt. The site uses tb_emt as a filter (0 / Moneyline / Spread / Total).
    # We only add a cache-buster.
    if "tb_emt" not in q:
        q["tb_emt"] = "0"

    q["_cb"] = str(epoch_ms)

    new_query = urlencode(q, doseq=True)
    return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))


def _set_tb_page(url: str, page: int) -> str:
    p = urlparse(url)
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    q["tb_page"] = str(page)
    new_query = urlencode(q, doseq=True)
    return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))


def _load_filtered_splits_page(driver, url: str, sport_filter_labels: list[str]) -> None:
    """
    Drive the live DK form instead of trusting raw query params.
    This is more reliable on the server, where direct deep links can still
    render a mixed all-sports slate.
    """
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    target_page = int(query.get("tb_page") or "1")
    target_date = query.get("tb_edate") or "n30days"
    target_market = query.get("tb_emt") or "0"
    base_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, "", parsed.fragment))

    driver.get(base_url)

    wait = WebDriverWait(driver, 30)
    sport_select = wait.until(EC.presence_of_element_located((By.NAME, "tb_eg")))
    selected_label = _select_sport_option(sport_select, sport_filter_labels)
    logger.info("[dk] selected sport filter: %s", selected_label)

    try:
        date_select = driver.find_element(By.NAME, "tb_edate")
        Select(date_select).select_by_value(target_date)
    except Exception:
        pass

    form = driver.find_element(By.CSS_SELECTOR, "form.tb-tform")
    form.submit()
    wait.until(lambda d: "tb_eg=" in (d.current_url or ""))
    time.sleep(2.0)

    filtered_url = driver.current_url
    if target_market and target_market != "0":
        market_url = filtered_url
        market_url = urlunparse(urlparse(market_url)._replace(query=urlencode({**dict(parse_qsl(urlparse(market_url).query, keep_blank_values=True)), "tb_emt": target_market}, doseq=True)))
        driver.get(market_url)
        time.sleep(1.0)

    if target_page > 1:
        driver.get(_set_tb_page(driver.current_url, target_page))
        time.sleep(1.0)


def fetch_rendered_html(url: str, timeout: int = 180, sport_filter_labels: Optional[list[str]] = None) -> str:
    """Render a URL using Selenium.
    - Never hang indefinitely
    - Never return None (returns "" on failure)
    - Never hardcode Linux chrome paths on Windows
    """
    driver = None
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except Exception as e:
        try:
            print(f"[dk] selenium import failed: {repr(e)}")
        except Exception:
            pass
        return ""

    try:
        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-extensions")
        opts.add_argument("--disable-background-networking")
        opts.add_argument("--disable-default-apps")
        opts.add_argument("--js-flags=--max-old-space-size=256")
        opts.add_argument("--window-size=1400,900")
        opts.add_argument("--blink-settings=imagesEnabled=false")
        opts.page_load_strategy = "eager"

        # Block fonts, stylesheets, images via DevTools
        prefs = {"profile.managed_default_content_settings.images": 2}
        opts.add_experimental_option("prefs", prefs)

        driver = webdriver.Chrome(options=opts)
        try:
            driver.set_page_load_timeout(timeout)
        except Exception:
            pass

        if sport_filter_labels:
            _load_filtered_splits_page(driver, url, sport_filter_labels)
        else:
            driver.get(url)
        html = driver.page_source or ""
        return html
    except Exception as e:
        try:
            print(f"[dk] fetch_rendered_html ERROR: {repr(e)} url={url}")
        except Exception:
            pass
        return ""
    finally:
        try:
            if driver is not None:
                driver.quit()
        except Exception:
            # swallow quit errors (driver may already be dead)
            pass

def _try_load_json(text: str) -> Optional[Any]:
    text = text.strip()
    if not text:
        return None
    # unescape common HTML entities if present
    text = text.replace("&quot;", '"').replace("&#x27;", "'").replace("&amp;", "&")
    try:
        return json.loads(text)
    except Exception:
        return None

def extract_json_candidates(html: str) -> List[Any]:
    candidates: List[Any] = []

    for pat in _JSON_SCRIPT_PATTERNS:
        for m in re.finditer(pat, html, flags=re.DOTALL | re.IGNORECASE):
            blob = m.group(1)
            data = _try_load_json(blob)
            if data is not None:
                candidates.append(data)

    # Also try to find large-ish JSON-ish blobs (fallback)
    # This is intentionally conservative to avoid tons of false positives.
    for m in re.finditer(r'(\{".{200,}?\})', html, flags=re.DOTALL):
        data = _try_load_json(m.group(1))
        if data is not None:
            candidates.append(data)

    return candidates


def find_splits_records_in_json(obj: Any) -> List[Dict[str, Any]]:
    """
    Heuristic search: looks for dicts that resemble betting split records.
    We keep it flexible because sites vary: "bets", "betPct", "handle", "moneyPct", etc.
    """
    results: List[Dict[str, Any]] = []

    def walk(x: Any):
        if isinstance(x, dict):
            # Common keys that hint "splits"
            keys = {k.lower() for k in x.keys()}
            if any(k in keys for k in ["betpct", "bet_pct", "bets_pct", "betspercentage", "tickets_pct", "ticketspercentage"]) \
               and any(k in keys for k in ["moneypct", "money_pct", "handle_pct", "handlepercentage", "moneypercentage", "handle"]):
                results.append(x)

            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return results


# -------------------------
# Option 2: DOM scrape (fallback)
# -------------------------

from bs4 import BeautifulSoup
import re

from bs4 import BeautifulSoup
import re

def parse_dk_start_text_to_utc_iso(start_text: str) -> str:
    """Convert DK splits start text like '1/10, 04:30PM' (ET) to UTC ISO 'YYYY-MM-DDTHH:MM:SSZ'."""
    try:
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo
        import re
        s = (start_text or '').strip()
        if not s:
            return ''
        m = re.search(r'(\d{1,2})/(\d{1,2}),\s*(\d{1,2}):(\d{2})(AM|PM)', s, re.I)
        if not m:
            return ''
        mon = int(m.group(1)); day = int(m.group(2))
        hh = int(m.group(3)); mm = int(m.group(4))
        ap = m.group(5).upper()
        if ap == 'PM' and hh != 12: hh += 12
        if ap == 'AM' and hh == 12: hh = 0
        now_et = datetime.now(ZoneInfo('America/New_York'))
        year = now_et.year
        if now_et.month == 12 and mon == 1:
            year = now_et.year + 1
        dt_local = datetime(year, mon, day, hh, mm, tzinfo=ZoneInfo('America/New_York'))
        dt_utc = dt_local.astimezone(timezone.utc)
        return dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception:
        return ''


def pct_from_style(el):
    if not el:
        return None
    style = el.get("style", "")
    m = re.search(r"width:\s*(\d+)%", style)
    return int(m.group(1)) if m else None


def dom_scrape_splits(html, sport):
    from bs4 import BeautifulSoup
    import re

    def pct_from_style(el):
        if not el:
            return None
        style = (el.get("style", "") or "")
        m = re.search(r"width:\s*(\d+)%", style)
        return int(m.group(1)) if m else None

    def pct_from_progress(progress_div):
        # width is usually on an inner div
        inner = (
            progress_div.select_one("div[style*='width']")
            or progress_div.find("div", attrs={"style": True})
        )
        # A real 0% is valid data. Do not fall through to a parent 100% width.
        value = pct_from_style(inner)
        return value if value is not None else pct_from_style(progress_div)

    soup = BeautifulSoup(html, "html.parser")
    rows = []

    # sanity counts

    # 1) Find "row containers" by climbing up from each progress bar
    # 1) Find "row containers" by climbing up from each progress bar
    containers = []
    seen = set()

    for p in soup.select("div.tb-progress"):
        cur = p
        for _ in range(10):  # climb up a few levels (a bit more room)
            cur = getattr(cur, "parent", None)
            if not cur or not hasattr(cur, "select"):
                break

            progs = cur.select("div.tb-progress")
            # We want the *actual* bet row: typically EXACTLY 2 progress bars
            if len(progs) == 2:
                # Must look like a real row (has the selection line or odds link)
                if cur.select_one("div.tb-slipline") or cur.select_one("a.tb-odd-s"):
                    key = id(cur)
                    if key not in seen:
                        seen.add(key)
                        containers.append(cur)
                    break



    # 2) Parse each container
    for row in containers:
        progs = row.select("div.tb-progress")
        if len(progs) < 2:
            continue

        handle_pct = pct_from_progress(progs[0])
        bets_pct = pct_from_progress(progs[1])
        if handle_pct is None or bets_pct is None:
            continue

        # Side label (try a few known DK classes)
        side_el = (
            row.select_one("div.tb-sb-title")
            or row.select_one("div.tb-team")
            or row.select_one("div.tb-title")
        )
        side = side_el.get_text(" ", strip=True) if side_el else None

        # Market label (walk up to tb-se block)
                # Market label (walk up to tb-se block)
        sec = row.find_parent("div", class_="tb-se")
        market_el = None
        if sec:
            market_el = sec.select_one("div.tb-se-title, div.tb-se-head")

        market = market_el.get_text(" ", strip=True) if market_el else "splits"
        if market and ("@" in market or "opens in a new tab" in market):
            market = "splits"

        # Game label + game_id (robust): walk backward to nearest event anchor
        game_name = None
        game_id = None

        for a in row.find_all_previous("a", limit=60):
            href = a.get("href") or ""
            txt = a.get_text(" ", strip=True) or ""

            txt = txt.replace("opens in a new tab", "").strip()

            if href and ("/event/" in href or "eventId=" in href):
                if txt and (("@" in txt) or ("vs" in txt.lower())):
                    game_name = txt
                    m = re.search(r"/event/(\d+)", href) or re.search(r"eventId=(\d+)", href)
                    if m:
                        game_id = m.group(1)
                    break

        # Selection label (best source for side + spread/total text)
        sl = row.select_one("div.tb-slipline")
        sl_text = sl.get_text(" ", strip=True) if sl else None
        side = sl_text or (side or "unknown")

        # Odds text (ML/spread/total odds)
        odd_a = row.select_one("a.tb-odd-s")
        odd_text = odd_a.get_text(" ", strip=True) if odd_a else None
        if odd_text:
            odd_text = odd_text.replace("opens in a new tab", "").strip()
            odd_text = odd_text.replace("\u2212", "-").replace("âˆ’", "-")


        # Current = what we can track historically
        current = None
        if sl_text and odd_text:
            current = f"{sl_text} @ {odd_text}"


        # --- DK kickoff time (dk_start_text / dk_start_iso) ---
        dk_start_text = ''
        dk_start_iso = ''
        try:
            # sec is the tb-se event container for this row (set above)
            if sec is not None:
                spans = sec.select('div.tb-se-title span')
                # Pick the first span that looks like a real kickoff time, not screen-reader junk
                _rx = re.compile(r'\b\d{1,2}/\d{1,2},\s*\d{1,2}:\d{2}\s*(AM|PM)\b')
                for spn in spans:
                    t = spn.get_text(' ', strip=True) if spn else ''
                    if not t:
                        continue
                    if 'opens in a new tab' in t.lower():
                        continue
                    if _rx.search(t):
                        dk_start_text = t
                        dk_start_iso = parse_dk_start_text_to_utc_iso(dk_start_text)
                        break
        except Exception:
            dk_start_text = ''
            dk_start_iso = ''
        # --- end kickoff ---

        rows.append({
            "sport": sport,
            "game_id": game_id,
            "game": game_name,
            "dk_start_text": dk_start_text,
            "dk_start_iso": dk_start_iso,
            "side": side,
            "market": market,
            "bets_pct": bets_pct,
            "money_pct": handle_pct,
            "open": None,
            "current": current,
            "news": None,
            "key_number_note": None,
        })


    logger.debug(
        "counts tb-progress=%d tb-sb-title=%d tb-team=%d tb-splitline=%d tb-se=%d row_containers=%d dom_rows=%d",
        len(soup.select("div.tb-progress")),
        len(soup.select("div.tb-sb-title")),
        len(soup.select("div.tb-team")),
        len(soup.select("div.tb-splitline")),
        len(soup.select("div.tb-se")),
        len(containers),
        len(rows),
    )
    return rows

def get_splits(url: str, sport: str, debug_dump_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch DraftKings betting splits for a given sport.
    Walks pagination (tb_page=1,2,3,...) until exhausted.
    """

    MAX_PAGES = 20  # safety cap
    all_records = []
    seen_keys = set()
    sport_filter_labels = SPORT_FILTER_ALIASES.get((sport or "").strip().lower())

    def rec_key(r):
        return (r.get("sport"), r.get("game_id"), r.get("market"), r.get("side"))

    last_page_had_new = True

    for page in range(1, MAX_PAGES + 1):
        if page == 1:
            page_url = url
        else:
            page_url = _set_tb_page(url, page)

        html = None
        last_err = None
        for attempt in (1, 2):
            try:
                html = fetch_rendered_html(page_url, sport_filter_labels=sport_filter_labels)
                if attempt == 2:
                    logger.info("[dk] fetch_rendered_html succeeded on retry (attempt 2/2) page_url=%s", page_url)
                last_err = None
                break
            except Exception as e:
                last_err = e
                logger.warning("[dk] fetch_rendered_html failed (attempt %s/2) page_url=%s err=%r", attempt, page_url, e)
                try:
                    time.sleep(1.0)
                except Exception:
                    pass

        if not html:
            logger.error("[dk] giving up on page=%s sport=%s url=%s last_err=%r", page, sport, page_url, last_err)
            break

        if debug_dump_path and page == 1:
            with open(debug_dump_path, "w", encoding="utf-8") as f:
                f.write(html)

        page_records = dom_scrape_splits(html, sport)

        if not page_records:
            logger.info("[dk] page %d empty, stopping paging", page)
            break

        new_count = 0
        for r in page_records:
            k = rec_key(r)
            if k in seen_keys:
                continue
            seen_keys.add(k)
            all_records.append(r)
            new_count += 1

        logger.info("[dk] page %d rows=%d new=%d", page, len(page_records), new_count)

        if new_count == 0:
            logger.info("[dk] page %d produced no new rows, stopping paging", page)
            break

    if (not all_records) and html and ("No events match your current selections" in html):
        logger.info("[dk] no events found across all pages")
        return {
            "html": html,
            "json_candidates_found": 0,
            "json_records_found": 0,
            "records": [],
        }

    return {
        "html": html,
        "json_candidates_found": 0,
        "json_records_found": 0,
        "records": all_records,
    }



