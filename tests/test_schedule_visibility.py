"""Schedule context is shown only where verified market rows are absent."""

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
import pandas as pd

from publication_coverage import PublicationCoverage
from schedule_visibility import uncovered_schedule


def test_scheduled_nhl_games_are_distinct_from_verified_mlb_markets():
    requested = []

    def fetch(url):
        requested.append(url)
        if parse_qs(urlparse(url).query)["dates"] != ["20260921"]:
            return {"events": []}
        return {"events": [{
            "id": "nhl-1", "date": "2026-09-21T23:00Z",
            "competitions": [{"competitors": [
                {"homeAway": "away", "team": {"displayName": "Buffalo Sabres"}},
                {"homeAway": "home", "team": {"displayName": "Pittsburgh Penguins"}},
            ]}],
        }]}

    games, errors = uncovered_schedule(
        [{"sport": "mlb", "game": "Washington Nationals @ Detroit Tigers"}],
        {"expected_active_sports": ["mlb", "nhl"]},
        now=datetime(2026, 9, 21, 16, tzinfo=timezone.utc),
        fetch=fetch,
    )

    assert errors == {}
    assert games == [{
        "sport": "nhl", "game_id": "nhl-1", "game": "Buffalo Sabres @ Pittsburgh Penguins",
        "kickoff_iso": "2026-09-21T23:00:00+00:00", "status": "AWAITING_VERIFIED_MARKETS",
    }]
    assert len(requested) == 4  # The NHL's 72-hour publication horizon.


def test_board_shows_schedule_only_game_without_a_market_read():
    playwright = pytest.importorskip("playwright.sync_api")
    board_path = Path(__file__).resolve().parents[1] / "site" / "board.html"
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
            page.goto(board_path.as_uri(), wait_until="domcontentloaded")
            result = page.evaluate("""() => {
              allDash=[]; sport='NHL'; market='ALL'; filter='all';
              publicationCoverage={scheduled_without_markets:[{sport:'nhl',game:'Buffalo Sabres @ Pittsburgh Penguins',kickoff_iso:new Date(Date.now()+3600000).toISOString()}]};
              renderBoard();
              return {cards:document.querySelectorAll('.schedule-coverage-game').length,
                      text:document.getElementById('schedule-coverage').textContent,
                      markets:window._boardRows.length};
            }""")
            assert result["cards"] == 1
            assert "Buffalo Sabres @ Pittsburgh Penguins" in result["text"]
            assert "awaiting verified markets" in result["text"].lower()
            assert result["markets"] == 0
        finally:
            browser.close()


def test_publication_keeps_schedule_context_until_refresh_and_removes_replaced_sport(tmp_path):
    path = tmp_path / "anomaly_board.csv"
    path.write_text("sport,game_id,market_display\n", encoding="utf-8")
    clock = pd.Timestamp("2026-09-21T16:00:00Z")
    PublicationCoverage(tmp_path, clock).publish(pd.DataFrame(), path, lambda frame, now: frame)
    coverage_path = tmp_path / "publication_coverage.json"
    payload = json.loads(coverage_path.read_text())
    payload["scheduled_without_markets"] = [{"sport": "nhl", "kickoff_iso": "2026-09-21T23:00:00Z", "game": "Buffalo @ Pittsburgh"}]
    coverage_path.write_text(json.dumps(payload), encoding="utf-8")

    PublicationCoverage(tmp_path, clock + pd.Timedelta(minutes=1)).publish(pd.DataFrame(), path, lambda frame, now: frame)
    assert len(json.loads(coverage_path.read_text())["scheduled_without_markets"]) == 1

    board = pd.DataFrame([{"sport": "nhl", "game_id": "game", "market_display": "MONEYLINE"}])
    board.to_csv(path, index=False)
    PublicationCoverage(tmp_path, clock + pd.Timedelta(minutes=2)).publish(board, path, lambda frame, now: frame)
    assert json.loads(coverage_path.read_text())["scheduled_without_markets"] == []
