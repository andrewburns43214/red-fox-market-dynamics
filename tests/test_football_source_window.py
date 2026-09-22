from urllib.parse import parse_qs, urlparse

import dk_headless
from main import SPORT_CONFIG


def test_nfl_uses_live_seven_day_source_window():
    assert parse_qs(urlparse(SPORT_CONFIG["nfl"]["url"]).query)["tb_edate"] == ["n7days"]


def test_cfb_uses_live_seven_day_source_window():
    assert parse_qs(urlparse(SPORT_CONFIG["ncaaf"]["url"]).query)["tb_edate"] == ["n7days"]


def test_advertised_second_page_recovers_on_fresh_request_without_browser(monkeypatch):
    requested = []
    form = '<select name="tb_eg"><option value="NCAA Football" selected>NCAA Football</option></select>'
    first = form + '<a href="?tb_page=2">2</a>PAGE_ONE'
    second = form + 'PAGE_TWO'
    empty = form + 'No events match your current selections'

    def fetch(url):
        requested.append(url)
        query = parse_qs(urlparse(url).query)
        if query.get("tb_page") == ["2"]:
            return second if "_cb" in query else empty
        if query.get("tb_page") == ["3"]:
            return second
        return first

    monkeypatch.setattr(dk_headless, "fetch_server_rendered_html", fetch)
    monkeypatch.setattr(dk_headless, "fetch_rendered_html", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("browser fallback")))
    monkeypatch.setattr(dk_headless, "dom_scrape_splits", lambda html, sport: [
        {"sport": sport, "game_id": marker, "market": "TOTAL", "side": "Over"}
        for marker in ("PAGE_ONE", "PAGE_TWO") if marker in html
    ])

    result = dk_headless.get_splits("https://example.test/splits?tb_eg=NCAA+Football&tb_edate=n7days", "ncaaf")
    assert {row["game_id"] for row in result["records"]} == {"PAGE_ONE", "PAGE_TWO"}
    assert all(row["_source_league_verified"] for row in result["records"])
    assert len(requested) == 4
    assert parse_qs(urlparse(requested[2]).query).get("_cb")


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


def test_numeric_provider_league_value_is_verified_from_selected_text(monkeypatch):
    form = '''<select name="tb_eg">
        <option value="84240" selected="selected">MLB</option>
    </select><input type="hidden" name="itm_content" value="MLB">'''
    populated = form + "AVAILABLE"
    monkeypatch.setattr(dk_headless, "fetch_server_rendered_html", lambda url: populated)
    monkeypatch.setattr(dk_headless, "dom_scrape_splits", lambda html, sport: [
        {"sport": sport, "game_id": "numeric-league", "market": "TOTAL", "side": "Over"}
    ])

    result = dk_headless.get_splits("https://example.test/splits?tb_eg=84240", "mlb")

    assert result["records"][0]["_source_league_verified"] is True


def test_mlb_terminal_empty_page_is_complete_not_empty_source(monkeypatch):
    form = '<select name="tb_eg"><option value="MLB" selected>MLB</option></select>'
    first = form + '<a href="?tb_page=2">2</a>FIRST'
    second = form + 'SECOND'
    empty = form + 'No events match your current selections'
    requested = []

    def fetch(url):
        page = parse_qs(urlparse(url).query).get("tb_page", ["1"])[0]
        requested.append(page)
        return {"1": first, "2": second}.get(page, empty)

    class Coverage:
        state = None
        def page(self, *_): pass
        def finish(self, state): self.state = state

    monkeypatch.setattr(dk_headless, "fetch_server_rendered_html", fetch)
    monkeypatch.setattr(dk_headless, "fetch_rendered_html", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("browser fallback")))
    monkeypatch.setattr(dk_headless, "dom_scrape_splits", lambda html, sport: [
        {"sport": sport, "game_id": marker, "market": "TOTAL", "side": "Over"}
        for marker in ("FIRST", "SECOND") if marker in html
    ])
    coverage = Coverage()
    result = dk_headless.get_splits("https://example.test/splits?tb_eg=MLB&tb_edate=n30days", "mlb", coverage=coverage)
    assert len(result["records"]) == 2
    assert requested == ["1", "2", "3"]
    assert coverage.state == "COMPLETE"


def test_mlb_empty_page_inside_advertised_range_is_incomplete(monkeypatch):
    form = '<select name="tb_eg"><option value="MLB" selected>MLB</option></select>'
    populated = form + '<a href="?tb_page=3">3</a>FIRST'
    empty = form + 'No events match your current selections'

    class Coverage:
        state = None
        def page(self, *_): pass
        def finish(self, state): self.state = state

    monkeypatch.setattr(dk_headless, "fetch_server_rendered_html", lambda url: populated if "tb_page" not in parse_qs(urlparse(url).query) else empty)
    monkeypatch.setattr(dk_headless, "fetch_rendered_html", lambda *args, **kwargs: empty)
    monkeypatch.setattr(dk_headless, "dom_scrape_splits", lambda html, sport: [
        {"sport": sport, "game_id": "one", "market": "TOTAL", "side": "Over"}
    ] if "FIRST" in html else [])
    coverage = Coverage()
    result = dk_headless.get_splits("https://example.test/splits?tb_eg=MLB&tb_edate=n30days", "mlb", coverage=coverage)
    assert len(result["records"]) == 1
    assert coverage.state == "PAGINATION_INCOMPLETE"


def test_genuinely_empty_mlb_first_page_remains_empty_source(monkeypatch):
    empty = '<select name="tb_eg"><option value="MLB" selected>MLB</option></select>No events match your current selections'

    class Coverage:
        state = None
        def page(self, *_): pass
        def finish(self, state): self.state = state

    monkeypatch.setattr(dk_headless, "fetch_server_rendered_html", lambda url: empty)
    monkeypatch.setattr(dk_headless, "fetch_rendered_html", lambda *args, **kwargs: empty)
    coverage = Coverage()
    result = dk_headless.get_splits("https://example.test/splits?tb_eg=MLB&tb_edate=n30days", "mlb", coverage=coverage)
    assert result["records"] == []
    assert coverage.state == "EMPTY_COMPLETE"
