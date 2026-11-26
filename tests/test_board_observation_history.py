from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


BOARD = Path(__file__).resolve().parents[1] / "site" / "board.html"


@pytest.mark.parametrize("width", [1440, 375])
def test_market_popup_shows_latest_observations_first(width):
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as browser_api:
        for channel in ("chromium", "msedge"):
            try:
                browser = browser_api.chromium.launch(channel=channel, headless=True)
                break
            except playwright.Error:
                continue
        else:
            pytest.skip("Chromium or Edge is required for the board browser test")

        with browser:
            page = browser.new_page(
                viewport={"width": width, "height": 900}, timezone_id="America/New_York"
            )
            row = {
                "sport": "ncaab",
                "game_id": "test-game",
                "game": "Baylor @ Gonzaga",
                "market_display": "SPREAD",
                "flagged_side": "Baylor +3.5",
                "open_line": "+4.5 (-110)",
                "current_line": "+3.5 (-108)",
            }
            start = datetime(2026, 11, 25, 23, 0, tzinfo=timezone.utc)
            events = [
                {
                    **row,
                    "timestamp": (start + timedelta(minutes=index)).isoformat(),
                    "line_display": row["open_line"] if index == 0 else row["current_line"],
                    "line_value": 4.5 if index == 0 else 3.5,
                    "price_odds": -110 if index == 0 else -108,
                    "bets_pct": 34,
                    "money_pct": index,
                }
                for index in range(26)
            ]
            page.route("**/*", lambda route: route.fulfill(status=404))
            page.route("**/board.html", lambda route: route.fulfill(path=BOARD))
            page.route(
                "**/manual-favorites.js*",
                lambda route: route.fulfill(path=BOARD.with_name("manual-favorites.js")),
            )
            page.route(
                lambda url: "/data/anomaly_event_details/" in url,
                lambda route: route.fulfill(json=events[::-1]),
            )
            page.add_init_script("""
                window.SUPABASE_URL = 'https://supabase.test';
                window.SUPABASE_ANON_KEY = 'test';
                window.supabase = {createClient: () => ({auth: {
                    getSession: async () => ({data: {session: null}}),
                    onAuthStateChange: () => {}
                }})};
            """)
            page.goto("https://redfox.test/board.html")
            page.evaluate("row => openGameDetail(row)", row)

            popup = page.locator("#drill-overlay")
            table = popup.locator(".detail-observation-table")
            rows = table.locator("tbody tr")
            playwright.expect(table.locator("th").first).to_have_attribute("aria-sort", "descending")
            playwright.expect(rows.locator('td[data-label="Money"]')).to_have_text(
                [f"{index}%" for index in reversed(range(26))]
            )
            playwright.expect(rows.locator('td[data-label="Time"]').first).to_contain_text("06:25 PM")
            playwright.expect(rows.locator('td[data-label="Time"]').last).to_contain_text("06:00 PM")
            if width == 375:
                popup.get_by_role("button", name="View 26 observations").click()
            playwright.expect(table.locator("tbody tr:visible")).to_have_count(24)
            playwright.expect(popup.locator(".prop-score")).to_have_count(0)
            playwright.expect(popup.locator(".market-story")).to_contain_text("+4.5 (-110) \u2192 +3.5 (-108)")
            chart = popup.locator('[data-movement-view="market"]')
            playwright.expect(chart.locator(".movement-fact b")).to_have_text(
                ["+4.5", "+3.5", "+4.5", "+3.5"]
            )

            popup.get_by_role("button", name="Show 2 more observations").click()
            playwright.expect(table.locator("tbody tr:visible")).to_have_count(26)
            playwright.expect(popup.locator(".detail-observations-more")).to_be_hidden()
