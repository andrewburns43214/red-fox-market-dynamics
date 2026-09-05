from pathlib import Path
import json

import pandas as pd


BOARD = (Path(__file__).resolve().parents[1] / "site" / "board.html").read_text(encoding="utf-8")
SANDBOX_BOARD = Path(__file__).resolve().parents[1] / "data" / "two_side_staging" / "anomaly_board.csv"
SERVER = (Path(__file__).resolve().parents[1] / "serve.py").read_text(encoding="utf-8")


def test_tracked_production_board_contains_the_approved_application_shell():
    """The public board must not rely on the local sandbox response rewriter."""
    required = (
        'class="sandbox-rail"',
        'Market Guide',
        'Saved Games',
        'External Research',
        'MARKET EXPLANATION',
        'rail-foot',
        'sandbox-static-controls',
        'sandbox-action-widgets',
        'redfox-saved-markets-v1',
    )
    for text in required:
        assert text in BOARD
    assert 'sandbox-data/' not in BOARD
    assert 'board-sandbox.html' not in BOARD


def test_sandbox_server_adapts_data_only_not_the_sidebar_shell():
    """Sandbox and production must share the tracked visual component."""
    assert "replace(\"'/data/\", \"'/sandbox-data/\")" in SERVER
    assert "<aside" not in SERVER
    assert "<style" not in SERVER


def test_tracked_board_owns_the_approved_toolbar_and_hides_legacy_surfaces():
    assert '.app-board-controls{display:none!important}' in BOARD
    assert '.pane-hdr{display:none!important}' in BOARD
    assert '.board-tools{display:none!important}' in BOARD
    assert '.info-btn,.td-rank br{display:none!important}' in BOARD
    assert 'background:#050708;color:#fff' in BOARD


def _sandbox_side(game, market, name):
    board = pd.read_csv(SANDBOX_BOARD, dtype=str, keep_default_na=False)
    row = board[(board.game == game) & (board.market_display == market)].iloc[0]
    side = next(side for side in json.loads(row.market_sides) if side["flagged_side"] == name)
    return row, side


def test_spread_movement_is_paired_and_side_relative():
    """A numeric spread move must classify exactly one paired side toward."""
    # The browser evaluates this per side. A spread becoming more negative is
    # toward that side; its paired line necessarily becomes less negative.
    direction = lambda open_value, current_value: "TOWARD" if current_value - open_value < 0 else "AGAINST"
    pairs = [
        ((+23.5, +22.5), (-23.5, -22.5)),
        ((+41.5, +40.5), (-41.5, -40.5)),
        ((-3.0, -4.0), (+3.0, +4.0)),
    ]
    for first, second in pairs:
        directions = {direction(*first), direction(*second)}
        assert directions == {"TOWARD", "AGAINST"}

    # Keep the browser implementation bound to the same paired-side rule.
    assert "market==='SPREAD'?primary<0" in BOARD
    assert "direction=isToward?'TOWARD':'AGAINST'" in BOARD
    assert "direction='AWAY'" not in BOARD


def test_same_side_signal_chips_deduplicate_by_displayed_label():
    # Distinct raw key crossings map to the same visible Key Number chip.
    raw_to_label = {"K10": "Key Number", "K14": "Key Number", "Market Move": "Market Move"}
    seen = set()
    rendered = [
        raw for raw in ("K10", "K14", "Market Move")
        if not (raw_to_label[raw].lower() in seen or seen.add(raw_to_label[raw].lower()))
    ]
    assert rendered == ["K10", "Market Move"]
    assert "const label=String(signalMetaFor(value).label||value).trim().toLowerCase();" in BOARD


def test_current_movement_copy_uses_row_relative_descriptors_without_arrows():
    assert "A Watch, Held," in BOARD
    assert "if(!['One-Way','Whipsaw','Juice Move'].includes(path)) return '';" not in BOARD
    assert "descriptor=points+' '+(Math.abs(primary)===1?'pt':'pts')" in BOARD
    assert "if(market==='MONEYLINE') descriptor='PRICE'" in BOARD
    assert "else descriptor='JUICE'" in BOARD
    assert "if(direction==='HELD') return '';" in BOARD
    assert "if(direction==='MIXED')" in BOARD
    assert "const arrow=delta>0?'↑':delta<0?'↓':'•';" not in BOARD


def test_drilldown_reuses_the_dashboard_movement_helper_for_the_selected_side():
    assert "const currentMovement=sideMovementHtml(selected,r);" in BOARD
    assert "detail-current-movement" in BOARD


def test_dashboard_and_drilldown_movement_text_share_the_current_tooltip_wording():
    assert "const MOVEMENT_TOOLTIP='Movement is relative to each side:" in BOARD
    assert "title=\"'+escHtml(MOVEMENT_TOOLTIP)+'\"" in BOARD


def test_drilldown_control_payloads_keep_the_board_movement_context():
    controls = [
        ("West Georgia @ Kennesaw State", "SPREAD", "West Georgia +22.5", "+23.5 (-105)", "+22.5 (-108)", "One-Way"),
        ("West Georgia @ Kennesaw State", "SPREAD", "Kennesaw State -22.5", "-23.5 (-115)", "-22.5 (-112)", "One-Way"),
        ("UTEP @ Oklahoma", "SPREAD", "UTEP +40.5", "+41.5 (-115)", "+40.5 (-110)", "One-Way"),
        ("UTEP @ Oklahoma", "SPREAD", "Oklahoma -40.5", "-41.5 (-105)", "-40.5 (-110)", "One-Way"),
        ("San Jose State @ Eastern Michigan", "MONEYLINE", "San Jose State", "+130", "+114", "One-Way"),
        ("San Jose State @ Eastern Michigan", "MONEYLINE", "Eastern Michigan", "-155", "-135", "One-Way"),
    ]
    for game, market, name, opening, current, path in controls:
        row, side = _sandbox_side(game, market, name)
        assert row.market_display == market
        assert side["open_line"] == opening
        assert side["current_line"] == current
        assert side["path"] == path


def test_returned_to_open_whipsaw_payload_remains_blank_under_current():
    for name, opening in [("Ball State +50.5", "+50.5 (-112)"), ("Ohio State -50.5", "-50.5 (-108)")]:
        _, side = _sandbox_side("Ball State @ Ohio State", "SPREAD", name)
        assert side["path"] == "Whipsaw"
        assert side["open_line"] == opening == side["current_line"]
    # The shared helper returns no text when the current state is held even
    # when the retained journey path was a Whipsaw.
    assert "if(direction==='HELD') return '';" in BOARD


def test_detail_side_and_market_switches_rebuild_shared_movement_state():
    assert "window._gameDetailState.sideIndex=index;" in BOARD
    assert "renderGameDetail();" in BOARD
    assert "window._gameDetailRows=markets;" in BOARD
    assert "const currentMovement=sideMovementHtml(selected,r);" in BOARD
