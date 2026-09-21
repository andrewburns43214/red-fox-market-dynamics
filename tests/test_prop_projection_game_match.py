"""A prop score must belong to the board game on the same date."""

import re
from pathlib import Path

import pytest


BOARD = (Path(__file__).resolve().parents[1] / "site" / "board.html").read_text(encoding="utf-8")


def test_mlb_prop_score_matches_the_game_time():
    playwright = pytest.importorskip("playwright.sync_api")
    source = re.search(
        r"function propTeamKey\(value\)\{.*?\n\}\nfunction propTeamMatches\(boardTeam,providerTeam\)\{.*?\n\}\nfunction gamePropProjection\(row\)\{.*?\n\}",
        BOARD,
        re.S,
    )
    assert source, "Prop projection game matcher is missing"

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
            page.add_script_tag(content="const normSportKey = value => value; let propProjections = [];" + source.group(0))
            row = {"sport": "mlb", "game": "Washington Nationals @ Detroit Tigers", "kickoff_iso": "2026-09-21T22:40:00Z"}
            other_day = {"sport": "mlb", "away_team": "Washington Nationals", "home_team": "Detroit Tigers", "commence_time": "2026-09-22T22:40:00Z", "away_mean": 3.3}
            same_day = {**other_day, "commence_time": "2026-09-21T22:40:00Z", "away_mean": 4.1}

            assert page.evaluate("([row, items]) => { propProjections = items; return gamePropProjection(row); }", [row, [other_day]]) is None
            assert page.evaluate("([row, items]) => { propProjections = items; return gamePropProjection(row); }", [row, [other_day, same_day]])["away_mean"] == 4.1
        finally:
            browser.close()
