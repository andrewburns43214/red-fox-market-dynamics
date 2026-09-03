import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOARD = (ROOT / "site" / "board.html").read_text(encoding="utf-8")
STAGING = ROOT / "data" / "two_side_staging"


def _timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_frozen_detail_uses_one_event_series_for_every_surface():
    """The live card and frozen detail must share the final pregame event."""
    assert "function frozenLiveRecord(row,events)" in BOARD
    assert "function frozenSeriesForSide(events,row,side)" in BOARD
    assert "frozenPregameRows(detailSideRows(events,row,side),row.frozen_at_utc||'')" in BOARD
    assert "_frozen_last_snapshot_at:last.timestamp" in BOARD
    assert "frozenAt:r._frozen_last_snapshot_at||''" in BOARD
    assert "const detailRows=state.frozenPregame?frozenPregameRows(evRows,state.frozenCutoff):evRows;" in BOARD

    # This is a frozen historical control.  It can rightly disappear from the
    # current publishable board after kickoff, so it must not be sourced from
    # the rolling board artifact.
    cutoff = _timestamp("2026-09-03T17:10:04.463575+00:00")
    events = json.loads((STAGING / "anomaly_event_details" / "mlb--34608985.json").read_text())
    series = [
        row for row in events
        if row["market_display"] == "MONEYLINE"
        and row["flagged_side"] == "CLE Guardians"
        and _timestamp(row["timestamp"]) <= cutoff
    ]
    assert series
    latest = series[-1]

    # This is the canonical frozen control point used by the card, current
    # header, chart current, Journey current, proof row, and coverage end.
    assert latest["line_display"] == "-110"
    assert latest["timestamp"] == "2026-09-03T01:01:00.582925+00:00"
    assert latest["bets_pct"] == 19.0
    assert latest["money_pct"] == 0.0  # A real source zero, not a fallback.
    assert _timestamp(series[0]["timestamp"]) <= _timestamp(latest["first_anomaly_seen"]) <= _timestamp(latest["timestamp"])


def test_moneyline_excursion_uses_implied_probability_points_not_line_points():
    assert "unit.includes('implied probability')" in BOARD
    assert "`${value.toFixed(1)} pp`" in BOARD


def test_missing_historical_splits_render_unavailable_not_zero():
    assert "function detailNumber(value)" in BOARD
    assert "return number===null?'--':number.toFixed(0)+'%';" in BOARD
    assert "bets===null&&money===null" not in BOARD
    assert "bets!==null&&money!==null" in BOARD


def test_live_cards_group_trusted_frozen_sides_by_market_and_open_the_clicked_side():
    """A frozen compact card is one paired section per market, not two summaries."""
    assert "function liveMarketSections(rows)" in BOARD
    assert "function frozenMarketSides(row,events)" in BOARD
    assert "The frozen event" in BOARD
    assert "const order=['MONEYLINE','TOTAL','SPREAD'];" in BOARD
    assert "marketSides(row).forEach(function(side)" in BOARD
    assert 'class="live-market-section"' in BOARD
    assert 'class="live-market-side-row"' in BOARD
    assert "openGameDetail(allLiveRecent[${item.index}],${sideArg})" in BOARD
    assert "async function openGameDetail(r,selectedSideLabel)" in BOARD


def test_frozen_current_journey_primary_read_matches_the_selected_side_read():
    """The final Journey node cannot fall back to an older event-level Watch."""
    assert "reaction:published.reaction||event.reaction||''" in BOARD
    assert "add('Current',last,String(side.reaction||last.reaction||'').trim());" in BOARD
