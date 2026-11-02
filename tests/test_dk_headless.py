from datetime import datetime
from zoneinfo import ZoneInfo

from dk_headless import dom_scrape_splits, parse_dk_start_text_to_utc_iso


def test_start_parser_accepts_legacy_and_current_dk_labels_across_dst():
    now_et = datetime(2026, 11, 2, 15, 30, tzinfo=ZoneInfo("America/New_York"))

    assert parse_dk_start_text_to_utc_iso("11/2, 08:20PM", now_et) == "2026-11-03T01:20:00Z"
    assert parse_dk_start_text_to_utc_iso("Mon 11/2 · 8:20 PM ET", now_et) == "2026-11-03T01:20:00Z"


def test_dom_scrape_extracts_current_dk_weekday_kickoff_label():
    html = """
    <div class="tb-se">
      <div class="tb-se-title"><span>Mon 11/2 · 8:20 PM ET</span></div>
      <a href="/event/123">Away @ Home</a>
      <div class="bet-row">
        <div class="tb-slipline">Home -3.5</div>
        <a class="tb-odd-s">-110</a>
        <div class="tb-progress"><div style="width:55%"></div></div>
        <div class="tb-progress"><div style="width:45%"></div></div>
      </div>
    </div>
    """

    rows = dom_scrape_splits(html, "nfl")

    assert len(rows) == 1
    assert rows[0]["dk_start_text"] == "Mon 11/2 · 8:20 PM ET"
    assert rows[0]["dk_start_iso"].endswith("T01:20:00Z")


def test_dom_scrape_preserves_zero_percent_progress_values():
    html = """
    <div class="tb-se">
      <div class="tb-se-title"><span>9/05, 12:00PM</span></div>
      <a href="/event/123">Away @ Home</a>
      <div class="bet-row">
        <div class="tb-slipline">Home -3.5</div>
        <a class="tb-odd-s">-110</a>
        <div class="tb-progress" style="width:100%"><div style="width:0%"></div></div>
        <div class="tb-progress" style="width:100%"><div style="width:0%"></div></div>
      </div>
    </div>
    """

    rows = dom_scrape_splits(html, "nfl")

    assert len(rows) == 1
    assert rows[0]["money_pct"] == 0
    assert rows[0]["bets_pct"] == 0
