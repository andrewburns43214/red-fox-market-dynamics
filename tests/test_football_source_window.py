from urllib.parse import parse_qs, urlparse

import dk_headless
from main import SPORT_CONFIG


def test_nfl_uses_live_seven_day_source_window():
    assert parse_qs(urlparse(SPORT_CONFIG["nfl"]["url"]).query)["tb_edate"] == ["n7days"]


def test_empty_football_source_retries_other_date_window(monkeypatch):
    requested = []
    empty = "No events match your current selections"
    populated = '<select name="tb_eg"><option value="NFL" selected>NFL</option></select>AVAILABLE'

    def fetch(url):
        requested.append(url)
        query = parse_qs(urlparse(url).query)
        if query.get("tb_edate") == ["n7days"] and "tb_page" not in query:
            return populated
        return empty

    monkeypatch.setattr(dk_headless, "fetch_server_rendered_html", fetch)
    monkeypatch.setattr(dk_headless, "fetch_rendered_html", lambda *args, **kwargs: empty)
    monkeypatch.setattr(dk_headless, "dom_scrape_splits", lambda html, sport: [
        {"sport": sport, "game_id": "giants-rams", "market": "splits", "side": "NY Giants"}
    ] if "AVAILABLE" in html else [])

    result = dk_headless.get_splits("https://example.test/splits?tb_eg=NFL&tb_edate=n30days&tb_emt=0", "nfl")
    assert len(result["records"]) == 1
    assert result["records"][0]["_source_league_verified"] is True
    assert [parse_qs(urlparse(url).query)["tb_edate"][0] for url in requested] == [
        "n30days", "n30days", "n30days", "n7days", "n7days"
    ]


def test_empty_cfb_source_does_not_publish_other_league(monkeypatch):
    empty = "No events match your current selections"
    requested = []
    monkeypatch.setattr(dk_headless, "fetch_server_rendered_html", lambda url: requested.append(url) or empty)
    monkeypatch.setattr(dk_headless, "fetch_rendered_html", lambda *args, **kwargs: empty)
    result = dk_headless.get_splits("https://example.test/splits?tb_eg=NCAA+Football&tb_edate=n30days", "ncaaf")
    assert result["records"] == []
    assert [parse_qs(urlparse(url).query)["tb_edate"][0] for url in requested] == ["n30days"] * 3 + ["n7days"]


def test_intermittent_empty_mlb_source_recovers_without_changing_sport(monkeypatch):
    requested = []
    empty = "No events match your current selections"
    populated = '<select name="tb_eg"><option value="MLB" selected>MLB</option></select>AVAILABLE'

    def fetch(url):
        requested.append(url)
        return populated if len(requested) == 2 else empty

    monkeypatch.setattr(dk_headless, "fetch_server_rendered_html", fetch)
    monkeypatch.setattr(dk_headless, "fetch_rendered_html", lambda *args, **kwargs: empty)
    monkeypatch.setattr(dk_headless, "dom_scrape_splits", lambda html, sport: [
        {"sport": sport, "game_id": "nationals-tigers", "market": "splits", "side": "Washington Nationals"}
    ] if "AVAILABLE" in html else [])

    result = dk_headless.get_splits("https://example.test/splits?tb_eg=MLB&tb_edate=n30days", "mlb")
    assert len(result["records"]) == 1
    assert result["records"][0]["_source_league_verified"] is True
    assert parse_qs(urlparse(requested[1]).query).get("_cb")
