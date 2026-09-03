from pathlib import Path


def test_drilldown_uses_existing_side_payload_and_side_specific_event_rows():
    source = Path("site/board.html").read_text(encoding="utf-8")

    assert "function detailSideRows(events,row,side)" in source
    # A stable identity keeps a selected total side intact when its displayed
    # number changes (for example, Over 8 to Over 7.5).
    assert "marketSideIdentity(event,row)===identity" in source
    assert "function setDetailSide(index)" in source
    assert "window._gameDetailState.sideIndex=index" in source
    assert "const evRows=detailSideRows(state.events,r,side);" in source
    assert "const sides=marketSides(r);" in source


def test_drilldown_journey_is_deterministic_and_keeps_proof_visible():
    source = Path("site/board.html").read_text(encoding="utf-8")

    assert "function detailJourneyMilestones(rows,side)" in source
    assert "add('Open',first);" in source
    assert "add('Current',last" in source
    assert "add('Max move',max);" in source
    assert "add('Reversal',reversal,'Whipsaw');" in source
    assert "slice(0,7)" in source
    assert "function toggleMarketJourney(button)" in source
    assert '<section class="detail-proof">' in source
    assert '<details class="detail-proof">' not in source
    assert "Proof / Observation History" in source
    assert 'aria-label="Select market side"' in source


def test_drilldown_keeps_existing_chart_controls_and_market_tabs():
    source = Path("site/board.html").read_text(encoding="utf-8")

    for label in ("'Line'", "'Price'", "'Bets %'", "'Money %'"):
        assert label in source
    assert "detail-market-switcher" in source
    assert "market-journey" in source
    assert "market-story" in source


def test_drilldown_journey_selection_and_proof_metrics_are_data_focused():
    source = Path("site/board.html").read_text(encoding="utf-8")

    assert "function selectJourneyEvent(timestamp)" in source
    assert "selectedJourneyTimestamp" in source
    assert "Show this observation on the trajectory chart" in source
    assert "Latest Development" in source
    assert "<span title=\"Counts every raw line-direction change" in source
    assert "Raw Reversals" in source
    assert "detailPointValue(selected.max_excursion)" in source
    assert "detail-quality-clean" in source
    assert "const chips=item.signal" not in source
    assert "function detailCoverageRange(rows)" in source


def test_drilldown_keeps_cooccurring_journey_signals_without_duplicate_nodes():
    source = Path("site/board.html").read_text(encoding="utf-8")

    assert "const existing=items.find(item=>String(item.row.timestamp||'')===timestamp);" in source
    assert "existing.signals=[...new Set([...(existing.signals||[]),signal])]" in source
    assert "if(signalName&&firstSignalAt)" in source
    assert "const chips=(item.signals||[]).map(anomalyBadge).join('');" in source
    assert "timestamped observations for ${escHtml(selectedSideText)}." in source
    assert "for this exact signal side" not in source
