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


def test_strongest_side_uses_confirmed_backend_support_not_anchor_or_chip_count():
    assert "const supported=String(r.supported_side||'').trim().toLowerCase();" in BOARD
    assert "r.read_anchor_side" not in BOARD[BOARD.index("function strongestSideIndex"):BOARD.index("function twoSideRationale")]
    assert "return -1;" in BOARD[BOARD.index("function strongestSideIndex"):BOARD.index("function twoSideRationale")]
    assert "signalCount=sideSignals(side).length" not in BOARD
    assert "side.evidence_role" in BOARD
    assert "Public Pressure must never" not in BOARD  # implementation, not user-facing prose
    assert "never a positive vote" in BOARD


def test_evidence_role_chips_use_neutral_context_color_without_changing_movement_colors():
    assert "'Pressure Side':{cls:'mr-neutral'" in BOARD
    assert "'Resistance Side':{cls:'mr-neutral'" in BOARD
    assert ".signal-pressure-side,.signal-resistance-side" in BOARD
    assert ".market-move.toward { color:#15866a; }" in BOARD
    assert ".market-move.against { color:#c74138; }" in BOARD


def test_mobile_market_read_chips_are_compact_without_changing_global_or_desktop_chips():
    assert "@media (max-width:850px)" in BOARD
    assert ".two-side-read .two-side-row{display:grid!important;grid-template-columns:84px minmax(0,1fr)!important;column-gap:5px!important;row-gap:0!important" in BOARD
    assert ".two-side-read-chips{grid-column:2;align-items:flex-start;align-content:flex-start;gap:2px 3px!important;overflow:hidden}" in BOARD
    assert ".two-side-read .mr{min-height:28px!important;padding:2px 5px!important;font-size:8px!important}" in BOARD
    assert ".two-side-read .two-side-row.is-strongest{background:rgba(21,134,106,.08)!important;box-shadow:none!important}" in BOARD
    assert '<span class="two-side-read-chips">' in BOARD
    assert ".mr { border-radius: 999px; padding: 4px 8px; font-size: 9px; letter-spacing: .02em; }" in BOARD
    assert ".signal-chip { cursor:pointer; border:1px solid transparent; font-family:var(--sans); }" in BOARD


def test_market_guide_matches_role_and_display_semantics():
    assert "Contrarian and Follow are directional." in BOARD
    assert "Otherwise, Freeze is neutral. Watch is also neutral." in BOARD
    assert "Freeze identifies concentrated pressure that fails to produce the expected favorable market response" in BOARD
    assert "Red Fox may establish the opposing side as the supported fade" in BOARD
    assert "Red Fox may support the opposing action side—not necessarily the side carrying the Freeze chip" in BOARD
    assert "The side receiving qualifying concentrated betting pressure" in BOARD
    assert "The opposing side supported by the market’s resistance to concentrated pressure" in BOARD
    assert "The spread or total number, or the moneyline price" in BOARD
    assert "late in the observed path while the market was inside its pregame closing window" in BOARD
    assert "['Risk & Data Quality',['Price Risk','Capped Split','Thin','Feed Risk','Split Risk']]" in BOARD
    assert "incomplete price or line data cannot support a reliable market read" in BOARD
    assert "Market Rank measures the relative significance of the selected market across the board." in BOARD
    assert "It does not establish direction or imply a recommendation." in BOARD
    assert "meaningful Open-to-Current movement" in BOARD
    assert "no qualifying Open-to-Current directional movement" not in BOARD


def test_red_fox_favorite_guide_section_and_tooltip_are_isolated_copy_additions():
    intro = BOARD.index("<h3>What Red Fox Is Reading</h3>")
    favorite = BOARD.index("redFoxFavoriteGuideSection()+", intro)
    directional = BOARD.index("marketGuideSection(MARKET_GUIDE_SECTIONS[0]", favorite)
    assert intro < favorite < directional
    assert '<h3>Red Fox Favorite</h3>' in BOARD
    assert '<p><strong>Red Fox Favorite</strong></p>' not in BOARD
    assert "A selective designation for a confirmed Supported Side that also matches Red Fox’s preferred wager profile." in BOARD
    assert "based on current DraftKings positioning" not in BOARD
    assert "Lower-support Freeze / Fade" in BOARD
    assert "reasonably actionable range" in BOARD
    assert "Not every directional read or wager type is Favorite-eligible." in BOARD
    assert "Whipsaw does not automatically disqualify a Favorite." in BOARD
    assert "Red Fox Favorite is <strong>binary</strong>" in BOARD
    assert "Matches Red Fox’s preferred wager criteria based on market support, movement, price/number, and data quality." in BOARD
    assert 'aria-describedby="favorite-badge-tooltip"' in BOARD
    assert 'onmouseenter="showFavoriteTooltip(this)"' in BOARD
    assert "onclick=\"event.stopPropagation();hideFavoriteTooltip();openSignalGuide(\\'Red Fox Favorite\\')\"" in BOARD


def test_favorite_badge_hover_and_exact_guide_navigation_share_authoritative_copy():
    assert 'onmouseenter="showFavoriteTooltip(this)"' in BOARD
    assert "if(requested==='Red Fox Favorite')" in BOARD
    assert "title.textContent='Red Fox Favorite';" in BOARD
    assert "body.innerHTML=redFoxFavoriteGuideSection(false);" in BOARD
    assert "hideFavoriteTooltip();openSignalGuide('Red Fox Favorite');" in BOARD
    assert ".red-fox-favorite-badge{display:block;flex:0 0 auto;width:108px;height:32px" in BOARD
    assert ".red-fox-favorite-badge{width:108px;height:36px;max-width:calc(100% - 42px)}" in BOARD


def test_market_read_tooltips_reuse_favorite_visuals_without_changing_chip_actions():
    shared_style = ".red-fox-tooltip{position:fixed;z-index:1300;width:min(270px,calc(100vw - 24px));padding:8px 10px;border:1px solid #d8dede;border-radius:8px;background:#11181c;color:#f8fafa;box-shadow:0 8px 24px rgba(5,7,8,.22);font:500 10px/1.4 var(--sans);pointer-events:none}"
    assert shared_style in BOARD
    assert 'class="red-fox-tooltip favorite-badge-tooltip"' in BOARD
    assert 'class="red-fox-tooltip signal-chip-tooltip"' in BOARD
    assert 'data-signal-tooltip="true" data-tooltip-title="${escHtml(meta.label||t)}" data-tooltip-body="${escHtml(meta.tip)}"' in BOARD
    assert 'data-signal-tooltip="true" data-tooltip-title="${escHtml(label)}" data-tooltip-body="${escHtml(meta.tip)}"' in BOARD
    assert "button.dataset.tooltipBody=meta.tip;" in BOARD
    assert "const DESKTOP_TOOLTIP_MEDIA='(hover: hover) and (pointer: fine)';" in BOARD
    assert "document.addEventListener('mouseover'" in BOARD
    assert "document.addEventListener('focusin'" in BOARD
    assert "onclick=\"event.stopPropagation();openSignalGuide(${clickArg})\"" in BOARD
    assert "function openSignalGuide(selected){\n  hideSignalTooltip();" in BOARD


def test_cross_market_integrity_badge_tooltip_and_guide_are_market_level():
    favorite = BOARD.index("redFoxFavoriteGuideSection()+")
    integrity = BOARD.index("marketIntegrityGuideSection()+", favorite)
    risk = BOARD.index("marketGuideSection(MARKET_GUIDE_SECTIONS[3]", integrity)
    assert favorite < integrity < risk
    assert '<h3>Market Integrity</h3>' in BOARD
    assert "A stricter market-integrity flag indicating that synchronized Spread and Moneyline pricing disagree in a sustained and reliable way" in BOARD
    assert "does not change the underlying Market Read or Market Rank" in BOARD
    assert "confirmed Cross-Market Mismatch" in BOARD
    assert 'id="integrity-badge-tooltip" role="tooltip" hidden' in BOARD
    assert "The synchronized Spread and Moneyline imply materially different market positions." in BOARD
    assert "function hasCrossMarketMismatch(row)" in BOARD
    assert "(market==='SPREAD'||market==='MONEYLINE')" in BOARD
    assert "crossMarketMismatchBadge()" in BOARD
    assert "Cross-Market Mismatch</button>" in BOARD
    assert "if(requested==='Cross-Market Mismatch')" in BOARD
    assert "body.innerHTML=marketIntegrityGuideSection(false,'Cross-Market Mismatch');" in BOARD
    assert "hideIntegrityTooltip();openSignalGuide('Cross-Market Mismatch');" in BOARD
    assert ".cross-market-badge{display:inline-flex" in BOARD
    assert ".cross-market-mismatch-badge{border-color:#aeb7bb" in BOARD
    assert "const mismatch=hasCrossMarketMismatch(r), split=hasCrossMarketSplit(r), crossMarket=mismatch||split, hasHeader=favorite||crossMarket;" in BOARD
    assert "favorite?favoriteBadge():''" in BOARD
    assert "mismatch?crossMarketMismatchBadge():split?crossMarketSplitBadge():''" in BOARD
    assert "market_sides.cross_market_mismatch" not in BOARD


def test_cross_market_split_badge_is_pair_context_with_mismatch_priority():
    assert "function hasCrossMarketSplit(row)" in BOARD
    assert "(market==='SPREAD'||market==='MONEYLINE')&&!hasCrossMarketMismatch(row)" in BOARD
    assert "mismatch?crossMarketMismatchBadge():split?crossMarketSplitBadge():''" in BOARD
    assert "isRedFoxFavorite(row)||hasCrossMarketMismatch(row)||hasCrossMarketSplit(row)" in BOARD
    assert 'id="cross-market-split-tooltip" role="tooltip" hidden' in BOARD
    assert "Spread and Moneyline each have a confirmed supported side, and those sides are different teams." in BOARD
    assert 'class="cross-market-badge cross-market-split-badge"' in BOARD
    assert 'class="cross-market-badge cross-market-mismatch-badge"' in BOARD
    assert "if(requested==='Cross-Market Split')" in BOARD
    assert "body.innerHTML=marketIntegrityGuideSection(false,'Cross-Market Split');" in BOARD
    assert "hideSplitTooltip();openSignalGuide('Cross-Market Split');" in BOARD
    assert "['Cross-Market Split','Cross-Market Mismatch']" in BOARD
    assert "is the more serious label when both descriptions could otherwise apply." in BOARD
    assert "A neutral or unresolved market cannot create a Cross-Market Split." in BOARD


def test_market_guide_explains_authoritative_supported_side_semantics():
    assert "['Supported Side', 'A green-highlighted row means Red Fox has confirmed a supported side for that market." in BOARD
    assert "If neither side is highlighted, no confirmed directional side has been established." in BOARD
    assert "rank is independent of directional support." in BOARD
    assert "'Contrarian':{cls:'mr-aligned',tip:'Directional." in BOARD
    assert "'Follow':{cls:'mr-aligned',tip:'Directional." in BOARD
    assert "'Freeze':{cls:'mr-freeze',tip:'Conditional." in BOARD
    assert "eligible Fade Candidate" not in BOARD
    assert "Otherwise, Freeze remains non-directional." in BOARD
    assert "'Watch':{cls:'mr-neutral',tip:'Non-directional." in BOARD
    assert "Market-behavior and context chips do not independently create a Supported Side." in BOARD
    assert "No green means the market is neutral" in BOARD
    assert "A highly ranked market can still be neutral and have no green Supported Side." in BOARD


def test_mobile_selected_market_rank_sits_beneath_unchanged_saved_control():
    assert '<span class="mobile-selected-rank">MKT #${boardRank(r)}</span>' in BOARD
    assert ".mobile-selected-rank{display:none}" in BOARD
    assert ".td-rank .mobile-selected-rank{display:block;position:absolute;top:34px;left:50%;width:44px" in BOARD
    assert "color:#7b858a;font:400 7px/1.2 var(--sans)" in BOARD
    assert ".td-rank .production-save{right:7px!important;left:auto!important;font-size:24px!important;color:#dd4a40!important}" in BOARD
    assert "boardGameSelections.set(String(gameId),String(marketName).toUpperCase());\n  renderBoard(false);" in BOARD
    assert '<span class="rank-number">${boardRank(r)}</span><span class="rank-scope">MKT</span>' in BOARD
    assert ".lpane>.tscroll>table>tbody>tr,.lpane>.tscroll>table>tbody>tr.row-hi,.lpane>.tscroll>table>tbody>tr.row-mid{position:relative;display:grid;grid-template-columns:1fr 1fr" in BOARD


def test_market_guide_terminology_order_and_deep_link_contract():
    ordered_markers = (
        "<h3>What Red Fox Is Reading</h3>",
        "<h3>Market Read</h3>",
        "<h3>Supported Side</h3>",
        "<h3>Market Rank</h3>",
        "redFoxFavoriteGuideSection()+",
        "marketGuideSection(MARKET_GUIDE_SECTIONS[0]",
        "marketGuideSection(MARKET_GUIDE_SECTIONS[1]",
        "marketGuideSection(MARKET_GUIDE_SECTIONS[2]",
        "<h3>Market Movement</h3>",
        "marketIntegrityGuideSection()+",
        "marketGuideSection(MARKET_GUIDE_SECTIONS[3]",
        "<h3>Board Fields</h3>",
        "<h3>Drilldown</h3>",
        "<h3>How to Read Red Fox</h3>",
    )
    positions = [BOARD.index(marker) for marker in ordered_markers]
    assert positions == sorted(positions)

    directional = BOARD.index("['Directional Reads'")
    behavior = BOARD.index("['Market Behavior'", directional)
    positioning = BOARD.index("['Positioning & Context'", behavior)
    risk = BOARD.index("['Risk & Data Quality'", positioning)
    assert directional < behavior < positioning < risk
    assert "Red Fox studies how the book responds to betting pressure, not just where the bets are" in BOARD
    assert "Tap or click any guide-backed chip or badge on the board to jump directly to its definition." in BOARD
    assert "These labels describe market behavior and evidence—not picks or wager recommendations." not in BOARD
    assert "Red Fox’s interpretation of how the book is behaving in the selected market" in BOARD
    assert "The plain-language explanation of how the book is responding" in BOARD
    assert "Market Reads are intentionally broader than Red Fox Favorites" in BOARD
    assert "based on current DraftKings positioning" not in BOARD

    for label in (
        "Contrarian", "Freeze", "Follow", "Watch", "Whipsaw", "Whipsaw Recovered", "Held", "Late",
        "One-Way", "Market Move", "Market Lag", "Juice Move", "Key Number", "Pressure Side",
        "Resistance Side", "Ticket-led", "Public Pressure", "Developing Read", "Low Bets / High $",
        "Heavy Favorite", "Price Risk", "Capped Split", "Thin", "Feed Risk", "Split Risk",
    ):
        assert f"'{label}'" in BOARD
    assert "onclick=\"event.stopPropagation();openSignalGuide(${clickArg})\"" in BOARD
    assert "onclick=\"openSignalGuide('Split Cap')\"" in BOARD
    assert "['Market Rank','Market Rank (MKT)','MKT'].includes(requested)" in BOARD
    assert "['Market Read','Supported Side'].includes(requested)" in BOARD


def test_market_guide_board_fields_excludes_duplicate_core_concepts():
    assert "const MARKET_GUIDE_BOARD_FIELDS=['Bets / Money','Open / Current','Data Quality','Market Explanation'];" in BOARD
    assert "MARKET_GUIDE_BOARD_FIELDS.map(boardFieldGuideItem).join('')" in BOARD
    assert "const MARKET_GUIDE_DRILLDOWN_FIELDS=['Market Trajectory','Market Journey'];" in BOARD


def test_market_guide_generalizes_implementation_sensitive_details():
    guide_start = BOARD.index("const SIGNAL_META={")
    guide_end = BOARD.index("function closeSignalGuide()", guide_start)
    guide = BOARD[guide_start:guide_end]
    for revealing_copy in (
        "1 to 8 points",
        "-165 to +125",
        "45% or less",
        "46% to 60%",
        "Only two valid observations",
        "at least three valid observations",
        "capped 0% or 100%",
        "persistence and reliability thresholds",
        "hold threshold",
        "confirmed signal threshold",
    ):
        assert revealing_copy not in guide
    for conceptual_copy in (
        "meaningful movement",
        "concentrated pressure",
        "lower-supported side",
        "reasonably actionable range",
        "sufficient confirmation",
        "reliable market data",
    ):
        assert conceptual_copy in guide.lower()


def test_capped_split_and_split_risk_keep_distinct_context_and_quality_roles():
    assert "This describes the observed source condition; directional split signals are disabled." in BOARD
    assert "creates a data-quality limitation, so the split cannot support a directional read." in BOARD
    assert "if(raw==='Capped Split') return SIGNAL_META['Split Cap'];" in BOARD
    assert "t==='SPLIT RISK'?'SPLIT RISK':'FEED RISK'" in BOARD


def test_visible_rank_is_selected_market_rank():
    assert 'Selected-market rank: ${boardRank(r)}' in BOARD
    assert '<span class="rank-number">${boardRank(r)}</span><span class="rank-scope">MKT</span>' in BOARD
    assert '<span class="rank-number">${i+1}</span>' not in BOARD


def test_market_chip_presentation_order_is_fixed_without_changing_selection_or_board_sorting():
    assert "const order={SPREAD:0,MONEYLINE:1,TOTAL:2};" in BOARD
    assert "marketsInPresentationOrder(group).map(row=>" in BOARD
    assert "markets.sort((a,b)=>marketPresentationRank(a.market_display)-marketPresentationRank(b.market_display));" in BOARD
    assert "return marketPresentationRank(a[0])-marketPresentationRank(b[0]);" in BOARD
    assert "const market=String(row.market_display||'').toUpperCase(), active=market===selectedMarket;" in BOARD
    assert "const order={MONEYLINE:0,SPREAD:1,TOTAL:2};" in BOARD


def test_live_recent_responsive_polish_reuses_board_controls_without_logic_changes():
    assert 'class="live-recent-inner"' in BOARD
    assert ".live-recent-inner { width:100%; max-width:1440px; margin:0; }" in BOARD
    assert ".live-sort-control select { width:auto; min-height:30px; max-width:220px; border:1px solid var(--border2);" in BOARD
    assert ".live-control-group .schip{min-height:25px;padding:5px 9px" in BOARD
    assert ".live-control-group .schip{min-height:26px!important;padding:3px 7px!important" in BOARD
    assert ".live-empty{width:min(100%,620px);margin:7px 0 0;padding:12px 14px" in BOARD
    assert '>Market board <span class="tbadge" id="b-all">' in BOARD
    assert '>Today <span class="tbadge" id="b-today">' in BOARD
    assert '>Live &amp; recent <span class="tbadge" id="b-live">' in BOARD
    assert '<span class="rail-label">Today\'s games</span>' in BOARD


def test_mobile_board_refresh_uses_safe_refresh_with_double_trigger_guard():
    assert 'class="mobile-board-refresh"' in BOARD
    assert 'aria-label="Refresh board"' in BOARD
    assert 'onclick="refreshMobileBoard(this)"' in BOARD
    assert '.mobile-board-refresh { display:none; }' in BOARD
    assert 'flex:0 0 44px;width:44px;height:44px' in BOARD
    assert ".mobile-board-refresh::before{content:'';position:absolute;inset:3px" in BOARD
    assert '.mobile-board-refresh svg{position:relative;z-index:1;width:20px;height:20px' in BOARD
    assert 'if(mobileBoardRefreshPending)return;' in BOARD
    assert 'await refreshBoardNow();' in BOARD
    assert "button.setAttribute('aria-busy','true')" in BOARD
    assert "button.removeAttribute('aria-busy')" in BOARD
    assert 'overscroll-behavior:none' not in BOARD.replace(' ', '').lower()
