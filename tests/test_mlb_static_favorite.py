import json

import pandas as pd
import pytest

from build_live_recent import favorite_handoff_states
from mlb_fixtures import CONTEXT_DIR, SCORE_SOURCE
from performance_ledger import update_performance_ledger
from red_fox_favorite import CONFIG, MLB_STATIC_VERSION, apply_red_fox_favorites, update_favorite_tracking
from tests.test_red_fox_favorite import market, side


FIRST = pd.Timestamp('2026-09-25T22:30:00Z')
KICKOFF = pd.Timestamp('2026-09-26T01:00:00Z')


def capture(clock, *, observed=None, retracing=False, opposite=False, mismatch=False, sport='mlb', price=-120):
    observed = clock if observed is None else observed
    sides = [side('Away', '+110', open_line='+110', bets=40, money=45, reaction='Watch',
                  direction='LIMITED', path='Held', source_latest_at=observed.isoformat()),
             side('Home', str(price), open_line='-120', bets=60, money=55, reaction='Watch',
                  direction='LIMITED', path='Held', source_latest_at=observed.isoformat(), return_toward_open=retracing)]
    row = market(sport, 'MONEYLINE', sides, supported_side='Away' if opposite else '', game_id='static')
    row.update(kickoff_iso=KICKOFF.isoformat(), state_as_of_utc=observed.isoformat(), cross_market_mismatch=mismatch)
    return apply_red_fox_favorites(pd.DataFrame([row]), as_of=clock)


def history(times, *, price=-120):
    return pd.DataFrame([dict(sport='mlb', game_id='static', timestamp=stamp, market_display='MONEYLINE',
                              side=team, current_line=odds, open_line=opening, bets_pct=bets, money_pct=money)
                         for stamp in times for team, odds, opening, bets, money in
                         [('Away','+110','+110',40,45), ('Home',str(price),'-120',60,55)]])


def test_static_confirms_fresh_selected_quote_and_counts_once(tmp_path):
    snapshots = [FIRST-pd.Timedelta(minutes=n) for n in (30,15,0)]
    def check(at, *, observed=None, enabled=True):
        return update_favorite_tracking(capture(at, observed=observed),tmp_path,as_of=at,
                                        market_history=history(snapshots),mlb_enabled=enabled).iloc[0]

    assert check(FIRST).favorite_review_state == 'pending_addition'
    assert check(FIRST+pd.Timedelta(minutes=5), observed=FIRST).red_fox_favorite == 'false'
    second = FIRST+pd.Timedelta(minutes=10)
    snapshots.append(second)
    accepted = check(second)
    assert accepted.red_fox_favorite == 'true' and accepted.favorite_side == 'Home' and not accepted.supported_side
    assert accepted.favorite_rule_version == MLB_STATIC_VERSION
    close = KICKOFF-pd.Timedelta(minutes=10)
    snapshots.append(close)
    assert check(close,enabled=False).red_fox_favorite == 'true'
    read = lambda filename: pd.read_csv(tmp_path/filename,dtype=str,keep_default_na=False)
    frozen = favorite_handoff_states(read('red_fox_favorite_freeze_candidates.csv'),KICKOFF+pd.Timedelta(minutes=1),
                                     read('red_fox_favorite_tracking.csv'))
    frozen.to_csv(tmp_path/'live_recent.csv',index=False)
    cutover = tmp_path/CONTEXT_DIR
    cutover.mkdir()
    (cutover/'active_from').write_text('2026-09-25T00:00:00Z')
    pd.DataFrame([dict(game_id='static',team1='Home',team2='Away',team1_score='6',team2_score='2',
                       score_source=SCORE_SOURCE,score_provider_event_id='8250')]).to_csv(tmp_path/'final_scores_history.csv',index=False)
    for _ in range(2):
        update_performance_ledger(tmp_path)
    ledger = read('performance_ledger.csv')
    assert len(ledger) == 1 and ledger.iloc[0].grade == 'W' and float(ledger.iloc[0].decision_price) == -120
    performance = json.loads((tmp_path/'favorite_performance.json').read_text())
    assert performance['wins'] == performance['mlb_static_moneyline']['wins'] == 1
    assert next(row for row in performance['sports'] if row['sport']=='mlb')['wins'] == 1


@pytest.mark.parametrize('changes,minutes', [(dict(opposite=True),(30,15,0)),(dict(mismatch=True),(30,15,0)),
    (dict(sport='nba'),(30,15,0)),(dict(),(31,10,0)),(dict(),(29,14,0)),(dict(price=-166),(30,15,0))])
def test_static_does_not_promote_unsafe_or_outside_rule(tmp_path, changes, minutes):
    row = update_favorite_tracking(capture(FIRST,**changes),tmp_path,as_of=FIRST,mlb_enabled=True,
                                   market_history=history([FIRST-pd.Timedelta(minutes=m) for m in minutes])).iloc[0]
    assert row.favorite_review_state != 'pending_addition'


def test_static_exclusion_log_is_independent_and_legacy_row_wins(tmp_path, monkeypatch):
    monkeypatch.setenv('RED_FOX_MLB_STATIC_RETRACE_LOG','1')
    snapshots = history([FIRST-pd.Timedelta(minutes=n) for n in (30,15,0)])
    for _ in range(2):
        result = update_favorite_tracking(capture(FIRST,retracing=True),tmp_path,as_of=FIRST,
                                          market_history=snapshots,mlb_enabled=False)
        assert not result.red_fox_favorite.eq('true').any()
    assert len(pd.read_csv(tmp_path/'mlb_static_retrace_exclusions.csv')) == 1
    actual = capture(FIRST)
    actual.loc[0,['red_fox_favorite','favorite_side','favorite_rule_version','favorite_state']] = ['true','Away',CONFIG.version,'qualified']
    final = update_favorite_tracking(actual,tmp_path,as_of=FIRST,market_history=snapshots,mlb_enabled=True)
    assert final.iloc[0].favorite_rule_version != MLB_STATIC_VERSION


def test_legacy_takes_precedence_with_its_own_confirmation(tmp_path):
    stamps = [FIRST-pd.Timedelta(minutes=n) for n in (30,15,0)]
    for clock in (FIRST,FIRST+pd.Timedelta(minutes=10)):
        stamps.append(clock)
        qualified = update_favorite_tracking(capture(clock),tmp_path,as_of=clock,mlb_enabled=True,market_history=history(stamps))
    assert qualified.iloc[0].favorite_rule_version == MLB_STATIC_VERSION
    for clock in (FIRST+pd.Timedelta(minutes=20),FIRST+pd.Timedelta(minutes=30)):
        legacy = capture(clock)
        legacy.loc[0,['red_fox_favorite','favorite_side','favorite_rule_version','favorite_state','supported_side']] = ['true','Away',CONFIG.version,'qualified','Away']
        actual = update_favorite_tracking(legacy,tmp_path,as_of=clock,mlb_enabled=True,market_history=history(stamps))
        if clock == FIRST+pd.Timedelta(minutes=20):
            assert actual.iloc[0].red_fox_favorite == 'false'
    assert actual.iloc[0].red_fox_favorite == 'true' and actual.iloc[0].favorite_side == 'Away'
