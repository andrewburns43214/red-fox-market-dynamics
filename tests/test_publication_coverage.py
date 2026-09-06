"""Audit-only regression evidence; production gates are deliberately unchanged."""
import pandas as pd
import pytest

from audit.publication_coverage import (
    assert_accounted, captured_inventory, inventory_html, reconcile, summarize,
)
from main import validate_snapshot_rows
from refresh_anomaly_board import filter_publication_eligible_markets

NOW = "2026-09-05T18:00:00Z"
KICK = "2026-09-06T02:00:00Z"  # Saturday 10 PM Eastern / 4 PM Hawaii


def captures(gid="g", times=("2026-09-05T17:55:00Z", "2026-09-05T17:59:00Z")):
    return pd.DataFrame([
        dict(sport="ncaaf", game_id=gid, game="UNLV @ Hawaii", market="splits",
             timestamp=t, side=f"{side} 50.5", current_line=f"{side} 50.5 @ -110",
             open_line=f"{side} 50.5 @ -110", bets_pct="50", money_pct="50", dk_start_iso=KICK)
        for t in times for side in ["Over", "Under"]
    ])


def run(snapshots, board=None, now=NOW):
    inv = captured_inventory(snapshots)
    return reconcile(inv, snapshots, board, now)


def test_every_in_window_market_is_published_or_has_a_specific_exclusion():
    good = captures("good")
    stale = captures("stale", ("2026-09-05T17:40:00Z", "2026-09-05T17:49:59Z"))
    new = captures("new", ("2026-09-05T17:59:00Z",))
    partial = captures("partial").iloc[::2]
    missing = captures("missing"); missing.loc[missing.side.str.startswith("Under"), "money_pct"] = ""
    future = captures("future"); future["dk_start_iso"] = "2026-09-19T19:30:00Z"
    snapshots = pd.concat([good, stale, new, partial, missing, future], ignore_index=True)
    published = pd.DataFrame([dict(sport="ncaaf", game_id="good", market_display="TOTAL")])
    result = run(snapshots, published)
    statuses = dict(zip(result.game_id, result.status))
    assert statuses == dict(good="PUBLISHED", stale="STALE_CAPTURE",
                           new="INSUFFICIENT_PARSEABLE_PAIRED_HISTORY", partial="NO_SYNCHRONIZED_SIDE_PAIR",
                           missing="INCOMPLETE_MARKET_FIELDS_OR_SIDE_COUNT", future="OUTSIDE_PUBLICATION_WINDOW")
    assert_accounted(result)
    summary = summarize(result, True)
    assert summary["eligible_in_window_scope"] == 5
    assert summary["publication_gate_eligible"] == summary["published_of_gate_eligible"] == 1
    assert summary["outside_publication_window"] == 1


def test_unexplained_eligible_loss_fails_certification():
    result = run(captures(), pd.DataFrame())
    assert result.iloc[0].status == "UNEXPLAINED_PUBLICATION_GAP"
    with pytest.raises(AssertionError):
        assert_accounted(result)


def test_no_access_is_not_zero_publication():
    result = run(captures())
    assert result.iloc[0].status == "PUBLICATION_EVIDENCE_UNAVAILABLE"
    assert summarize(result, False)["published_of_gate_eligible"] is None
    inv = captured_inventory(captures())
    unavailable = reconcile(inv, None, None, NOW)
    assert unavailable.iloc[0].status == "CAPTURE_EVIDENCE_UNAVAILABLE"
    with pytest.raises(AssertionError):
        assert_accounted(unavailable)


@pytest.mark.parametrize("last,status", [("17:50:00", "PUBLICATION_EVIDENCE_UNAVAILABLE"), ("17:49:59", "STALE_CAPTURE")])
def test_ten_minute_boundary_uses_capture_time_even_when_values_never_change(last, status):
    data = captures(times=("2026-09-05T17:40:00Z", f"2026-09-05T{last}Z"))
    assert run(data).iloc[0].status == status


@pytest.mark.parametrize("now,status", [("2026-09-06T02:04:59Z", "PUBLICATION_EVIDENCE_UNAVAILABLE"),
                                       ("2026-09-06T02:05:00Z", "KICKOFF_EXPIRED")])
def test_hawaii_kickoff_uses_eastern_day_and_exact_five_minute_boundary(now, status):
    data = captures(times=("2026-09-06T02:00:00Z", "2026-09-06T02:03:00Z"))
    assert run(data, now=now).iloc[0].status == status


@pytest.mark.parametrize("sport,now,kickoff,inside", [
    ("ncaaf", NOW, "2026-09-05T04:00:00Z", True),
    ("ncaaf", NOW, "2026-09-13T03:59:59Z", True),
    ("ncaaf", NOW, "2026-09-13T04:00:00Z", False),
    ("nfl", NOW, "2026-09-09T03:59:59Z", False),
    ("nfl", NOW, "2026-09-09T04:00:00Z", True),
    ("nfl", NOW, "2026-09-15T03:59:59Z", True),
    ("nfl", NOW, "2026-09-15T04:00:00Z", False),
    ("nfl", "2026-09-22T03:59:59Z", "2026-09-25T00:15:00Z", False),
    ("nfl", "2026-09-22T04:00:00Z", "2026-09-25T00:15:00Z", True),
])
def test_existing_football_horizons_are_not_expanded(sport, now, kickoff, inside):
    probe = pd.DataFrame([dict(sport=sport, dk_start_iso=kickoff)])
    assert bool(len(filter_publication_eligible_markets(probe, now=now))) == inside


def test_inventory_retains_market_with_no_parseable_split_bars():
    html = '''<select name="tb_eg"><option value="NCAA Football" selected>CFB</option></select>
    <div class="tb-se"><div class="tb-se-title"><a href="/event/123">UNLV @ Hawaii</a>
    <span>9/5, 10:00PM</span></div><div class="tb-se-head"><div>Total</div><div>Odds</div></div></div>'''
    inventory = inventory_html(html, "ncaaf", NOW)
    assert len(inventory) == 1
    assert inventory.iloc[0].market_display == "TOTAL"
    assert inventory.iloc[0].dk_start_iso == "2026-09-06T02:00:00+00:00"
    assert bool(inventory.iloc[0].league_identified)
    result = reconcile(inventory, pd.DataFrame(), pd.DataFrame(), NOW)
    assert result.iloc[0].status == "NOT_CAPTURED_OR_MARKET_NOT_NORMALIZED"
    with pytest.raises(AssertionError):
        assert_accounted(result)


def test_source_sport_is_not_verified_when_form_disagrees():
    html = '<select name="tb_eg"><option selected value="MLB">MLB</option></select><div class="tb-se"><a href="/event/1">A @ B</a></div>'
    result = inventory_html(html, "ncaaf", NOW)
    assert not bool(result.iloc[0].league_identified)
    assert result.iloc[0].market_display == "UNKNOWN"


def test_espn_90_percent_match_retains_valid_unmatched_dk_game(monkeypatch):
    rows = [dict(game=f"Away{i} @ Home{i}", game_id=str(i), side=f"{side} 50.5",
                 current=f"{side} 50.5 @ -110", dk_start_iso=KICK, _source_league_verified=True)
            for i in range(10) for side in ["Over", "Under"]]
    monkeypatch.setattr("main.get_espn_kickoff_map", lambda sport, games: {r["game"]: KICK for r in rows[:18]})
    accepted, note = validate_snapshot_rows(rows, "ncaaf")
    assert len(accepted) == 20
    assert rows[-1]["_validation_state"] == "ESPN_UNMATCHED"
    assert not rows[-1]["_capture_exclusion_reason"]


def test_newer_partial_capture_does_not_refresh_old_pair():
    old = captures(times=("2026-09-05T17:40:00Z", "2026-09-05T17:49:00Z"))
    new = captures(times=("2026-09-05T17:59:00Z",)).iloc[:1]
    result = run(pd.concat([old, new]))
    assert result.iloc[0].status == "STALE_CAPTURE"
    assert "17:49:00" in result.iloc[0].last_paired_capture


def test_audit_does_not_mutate_inputs():
    snapshots = captures(); original = snapshots.copy(deep=True)
    inv = captured_inventory(snapshots); before = inv.copy(deep=True)
    reconcile(inv, snapshots, pd.DataFrame(), NOW)
    pd.testing.assert_frame_equal(snapshots, original)
    pd.testing.assert_frame_equal(inv, before)


def test_missing_capture_does_not_become_zero_gate_eligibility():
    result = reconcile(captured_inventory(captures()), None, None, NOW)
    summary = summarize(result, False)
    assert summary["publication_gate_eligible"] is None
    assert summary["unresolved_evidence_records"] == 1
    assert not summary["coverage_certified"]


def test_existing_complete_gate_accepts_mismatched_total_audit_flags_it():
    data = captures()
    data.loc[data.side.str.startswith("Under"), "current_line"] = "Under 51.5 @ -110"
    result = run(data)
    assert result.iloc[0].publication_eligible  # characterize; do not change production
    assert "TOTAL_SIDE_LINE_DISAGREEMENT" in result.iloc[0].pair_integrity_warnings


def test_dst_fall_back_cfb_keeps_last_calendar_hour():
    # Eight local calendar dates remain eligible across fall-back.
    rows = pd.DataFrame([dict(sport="ncaaf", dk_start_iso="2026-11-08T23:30:00-05:00")])
    assert not filter_publication_eligible_markets(rows, now="2026-11-01T12:00:00-05:00").empty


def test_empty_inventory_is_explicitly_not_certified():
    rows = reconcile(pd.DataFrame(), None, None, NOW)
    assert summarize(rows, False)["discovered_market_records"] == 0
    assert not summarize(rows, False)["coverage_certified"]


def test_wide_moneyline_odds_are_accounted_without_weakening_pair_requirement():
    data = captures()
    over = data.side.str.startswith("Over")
    data.loc[over, "side"] = "Utah"
    data.loc[~over, "side"] = "Idaho"
    data.loc[over, "current_line"] = "Utah @ -100000"
    data.loc[~over, "current_line"] = "Idaho @ +5000"
    inventory = captured_inventory(data)
    assert len(inventory) == 1 and inventory.iloc[0].market_display == "MONEYLINE"
    board = inventory[["sport", "game_id", "market_display"]]
    result = reconcile(inventory, data, board, NOW)
    assert result.iloc[0].status == "PUBLISHED"
    assert result.iloc[0].publication_eligible
    assert_accounted(result)


def test_nhl_offseason_source_is_intentional_not_a_coverage_gap():
    data = captures(); data["sport"] = "nhl"
    result = run(data)
    assert result.iloc[0].status == "SPORT_DISABLED_BY_SEASON"
    assert not result.iloc[0].in_window_target
