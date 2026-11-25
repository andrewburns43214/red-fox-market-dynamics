import json

import pandas as pd
import pytest

from audit_directional_integrity import audit
from build_live_recent import favorite_handoff_states
from nba_legacy_shadow import record_legacy
from performance_ledger import update_performance_ledger
from red_fox_favorite import (CONFIG, NBA_PUBLIC_PATH, NBA_PUBLIC_VERSION, apply_red_fox_favorites,
                              favorite_side_quote, update_favorite_tracking)
from tests.test_red_fox_favorite import is_favorite, market, side


def fixture(at, *, quote_at=None, price=-122, tickets=70, money=55, spread=-4,
            supported="", mismatch=False, sport="nba"):
    quote_at = quote_at or at
    source = dict(reaction="Watch", direction="LIMITED", path="Held", source_latest_at=quote_at)
    sp = market(sport, "SPREAD", [side(f"Away {-spread:+g}", f"{-spread:+g} (-116)", **source),
                side(f"Home {spread:+g}", f"{spread:+g} (-104)", **source)], supported_side="", game_id="public")
    ml = market(sport, "MONEYLINE", [side("Away", "+110", **source), side("Home", f"{price:+d}",
                open_line="-120", bets=tickets, money=money, **source)], supported_side=supported, game_id="public")
    for row in (sp, ml):
        row.update(kickoff_iso="2026-11-25T01:00:00Z", state_as_of_utc=quote_at, cross_market_mismatch=mismatch)
    return apply_red_fox_favorites(pd.DataFrame([sp, ml]), as_of=at, nov24_goalies={})


def test_nba_public_release_confirms_selected_side_and_grades_once(tmp_path):
    first, second = "2026-11-24T22:00:00Z", "2026-11-24T22:10:00Z"

    def capture(at, config=(True, True), **kwargs):
        return update_favorite_tracking(fixture(at, **kwargs), tmp_path, as_of=at, nba_config=config)

    assert capture(first).iloc[1].favorite_review_state == "pending_addition"
    assert not is_favorite(capture("2026-11-24T22:05:00Z", quote_at=first), 1)
    picked = capture(second)
    assert is_favorite(picked, 1) and not is_favorite(picked, 0)
    assert picked.iloc[1].favorite_pathway == NBA_PUBLIC_PATH
    assert picked.iloc[1].supported_side == ""
    assert favorite_side_quote(picked.iloc[1]) == "-122"
    assert audit(picked)["favorites_without_supported_side"] == 0
    lock = capture("2026-11-25T00:45:00Z", config=(False, True), price=-125)
    assert is_favorite(lock, 1)
    read = lambda name: pd.read_csv(tmp_path/name, dtype=str, keep_default_na=False)
    frozen = favorite_handoff_states(read("red_fox_favorite_freeze_candidates.csv"),
                                    pd.Timestamp("2026-11-25T01:04:00Z"), read("red_fox_favorite_tracking.csv"))
    frozen.to_csv(tmp_path/"live_recent.csv", index=False)
    pd.DataFrame([dict(game_id="public", team1="Home", team2="Away", team1_score="109", team2_score="106")]).to_csv(
        tmp_path/"final_scores_history.csv", index=False)
    for _ in range(2):
        update_performance_ledger(tmp_path)
    records = read("performance_ledger.csv")
    assert len(records) == 1 and records.iloc[0].favorite_rule_version == NBA_PUBLIC_VERSION
    assert float(records.iloc[0].decision_price) == -122 and records.iloc[0].grade == "W"
    payload = json.loads((tmp_path/"favorite_performance.json").read_text())
    assert payload["wins"] == payload["nba_public_neutral_moneyline"]["wins"] == 1
    assert next(s for s in payload["sports"] if s["sport"] == "nba")["wins"] == 1


@pytest.mark.parametrize("changes", [dict(tickets=69), dict(money=54), dict(price=-166), dict(price=100),
    dict(price=-145), dict(spread=-4.5), dict(spread=1), dict(supported="Away"), dict(mismatch=True), dict(sport="ncaab")])
def test_nba_public_release_rejects_disqualified_inputs(tmp_path, changes):
    at = "2026-11-24T22:20:00Z"
    result = update_favorite_tracking(fixture(at, **changes), tmp_path, as_of=at, nba_config=(True, True))
    assert result.iloc[1].favorite_review_state != "pending_addition"


def legacy(at):
    board = fixture(at)
    board.loc[0, ["red_fox_favorite", "favorite_side", "favorite_state", "favorite_rule_version"]] = [
        "true", "Away +4", "qualified", CONFIG.basketball_version]
    board.at[0,"supported_side"] = "Away +4"
    board.at[0,"favorite_pathway"] = "low_support_contrarian"
    board.at[0,"favorite_snapshot_id"] = at
    board.at[0,"favorite_first_qualified_at"] = at
    board.at[0,"favorite_final_qualified_at"] = at
    return board


def test_nba_public_release_preserves_raw_legacy_exclusion_and_isolates_legacy(tmp_path):
    for at in ("2026-11-24T22:30:00Z", "2026-11-24T22:40:00Z"):
        board = legacy(at)
        record_legacy(board, tmp_path, at)
        actual = update_favorite_tracking(board, tmp_path, as_of=at, nba_config=(True, True))
        assert not actual.red_fox_favorite.eq("true").any()
        assert not actual.favorite_review_state.eq("pending_addition").any()
    sandbox = tmp_path/"nba_legacy_hypothetical"
    shadow = pd.read_csv(sandbox/"red_fox_favorite_tracking.csv", dtype=str)
    assert shadow.iloc[-1].favorite_state == "qualified"
    pd.DataFrame([dict(game_id="public", team1="Away", team2="Home", team1_score="109", team2_score="106")]).to_csv(
        tmp_path/"final_scores_history.csv", index=False)
    record_legacy(legacy(at).iloc[:0], tmp_path, "2026-11-25T01:04:00Z")
    ledger = pd.read_csv(sandbox/"hypothetical_results.csv", dtype=str)
    assert len(ledger) == 1 and ledger.iloc[0].grade == "W"
    assert not (tmp_path/"performance_ledger.csv").exists()


def test_nba_public_release_preserves_published_legacy(tmp_path):
    for at, settings in [("2026-11-24T22:30:00Z", (False, False)), ("2026-11-24T22:40:00Z", (False, False)),
                         ("2026-11-24T22:50:00Z", (True, True))]:
        result = update_favorite_tracking(legacy(at), tmp_path, as_of=at, nba_config=settings)
    assert is_favorite(result, 0) and result.iloc[0].favorite_side == "Away +4"
    assert not is_favorite(result, 1)


def test_nba_public_release_t40_blocks_late_first_confirmation(tmp_path):
    for at in ("2026-11-25T00:10:00Z", "2026-11-25T00:20:00Z"):
        board = update_favorite_tracking(fixture(at), tmp_path, as_of=at, nba_config=(True, True))
    assert not is_favorite(board, 1)
    assert board.iloc[1].favorite_review_state != "pending_addition"


def test_nba_public_release_hard_contradiction_blocks_handoff(tmp_path):
    for at in ("2026-11-24T22:30:00Z", "2026-11-24T22:40:00Z", "2026-11-25T00:45:00Z"):
        raw = fixture(at)
        if at.endswith("00:45:00Z"):
            raw.at[0, "cross_market_mismatch"] = True
        result = update_favorite_tracking(raw, tmp_path, as_of=at, nba_config=(True, True))
    assert not is_favorite(result, 1) and result.iloc[1].favorite_late_invalidated == "true"
    read = lambda name: pd.read_csv(tmp_path/name, dtype=str, keep_default_na=False)
    assert favorite_handoff_states(read("red_fox_favorite_freeze_candidates.csv"), pd.Timestamp("2026-11-25T01:04:00Z"),
                                    read("red_fox_favorite_tracking.csv"), raw).empty
