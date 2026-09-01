from dk_headless import dom_scrape_splits


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
