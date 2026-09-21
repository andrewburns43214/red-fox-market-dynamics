"""The browser's NFL slate must agree with the Tuesday-to-Monday publisher."""

import re
from pathlib import Path

import pytest


BOARD = (Path(__file__).resolve().parents[1] / "site" / "board.html").read_text(encoding="utf-8")


def test_nfl_browser_week_keeps_monday_and_waits_until_tuesday():
    playwright = pytest.importorskip("playwright.sync_api")
    source = re.search(r"function nflPublicationDay\(value\)\{.*?\n\}\nfunction nflCurrentWeekRows\(rows,now=Date\.now\(\)\)\{.*?\n\}", BOARD, re.S)
    assert source, "NFL browser publication filter is missing"
    rows = [
        {"game": "Giants @ Rams", "kickoff_iso": "2026-09-22T00:15:00Z", "observation_count": "1850"},
        {"game": "next Thursday", "kickoff_iso": "2026-09-25T00:15:00Z"},
        {"game": "next Monday", "kickoff_iso": "2026-09-29T00:15:00Z"},
    ]
    with playwright.sync_playwright() as browser_api:
        try:
            browser = browser_api.chromium.launch(headless=True)
        except Exception:
            installed = list((Path.home() / "AppData/Local/ms-playwright").glob("chromium-*/chrome-win64/chrome.exe"))
            if not installed:
                pytest.skip("Chromium is not installed")
            browser = browser_api.chromium.launch(headless=True, executable_path=str(installed[-1]))
        try:
            page = browser.new_page()
            page.add_script_tag(content=source.group(0))
            monday = page.evaluate("([rows, now]) => nflCurrentWeekRows(rows, Date.parse(now))", [rows, "2026-09-21T16:00:00Z"])
            tuesday = page.evaluate("([rows, now]) => nflCurrentWeekRows(rows, Date.parse(now))", [rows, "2026-09-22T16:00:00Z"])
            assert [row["game"] for row in monday] == ["Giants @ Rams"]
            assert monday[0]["observation_count"] == "1850"
            assert [row["game"] for row in tuesday] == ["next Thursday", "next Monday"]
        finally:
            browser.close()
