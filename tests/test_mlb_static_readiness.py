from datetime import timedelta
import json

import pandas as pd

from mlb_fixtures import CONTEXT_DIR, SCORE_SOURCE, write_json
from mlb_static_readiness import STATE_FILE, ready_games
from red_fox_favorite import MLB_STATIC_VERSION, update_favorite_tracking
from tests.test_mlb_static_favorite import FIRST, KICKOFF, capture, history


def setup(directory):
    context = directory / CONTEXT_DIR
    context.mkdir()
    (context / 'active_from').write_text('2026-09-23T00:00:00Z')
    final_started = KICKOFF - timedelta(days=1)
    final = fixture('10', final_started, state='post')
    upcoming = fixture('11', KICKOFF, state='pre')
    pd.DataFrame([dict(game_id='closed', team1='Away', team2='Home', team1_score='2', team2_score='4',
                       score_source=SCORE_SOURCE, score_provider_event_id='10',
                       resolved_at_utc=(final_started + timedelta(hours=3)).isoformat())]).to_csv(
                           directory / 'final_scores_history.csv', index=False)
    return context, {'10': final, '11': upcoming}


def fixture(pk, start, *, state):
    return dict(game_pk=pk, official_date='2026-09-25', scheduled_start=start.isoformat(),
                time_tbd=False, doubleheader=False, game_number=1, first_pitch=None, state=state,
                away=dict(team_id=1, aliases=['away'], score=2), home=dict(team_id=2, aliases=['home'], score=4))


def publish(context, at, games, streak=2):
    write_json(context / 'latest.json', dict(observed_at=at.isoformat(), games=games, source='MLB Stats API',
                                            schedule_status='fixtures_available' if games else 'no_games_scheduled',
                                            populated_streak=streak))


def readiness(directory):
    return json.loads((directory / CONTEXT_DIR / STATE_FILE).read_text())


def run(directory, at, *, observed=None, auto=True, board=None, retained=()):
    observed = at if observed is None else observed
    board = capture(observed) if board is None else board
    return ready_games(directory, board, history([observed]), at, retained, auto=auto)


def test_empty_feed_is_idle_and_duplicate_samples_cannot_activate(tmp_path):
    context, games = setup(tmp_path)
    publish(context, FIRST, {}, streak=0)
    assert not run(tmp_path, FIRST)
    assert readiness(tmp_path)['reason'] == 'no_current_mlb_fixtures'
    publish(context, FIRST, games, streak=1)
    assert not run(tmp_path, FIRST)
    assert readiness(tmp_path)['reason'] == 'awaiting_consecutive_fixture_collections'
    publish(context, FIRST, games)
    assert not run(tmp_path, FIRST)
    assert not run(tmp_path, FIRST + timedelta(minutes=1), observed=FIRST)
    assert readiness(tmp_path)['ready'] == 1
    second = FIRST + timedelta(minutes=10)
    publish(context, second, games)
    assert not run(tmp_path, second)  # Activation cannot retroactively publish its input capture.
    assert readiness(tmp_path)['status'] == 'active'
    assert (tmp_path / 'mlb_static_activated_at').read_text().strip() == second.isoformat()
    assert not run(tmp_path, second + timedelta(minutes=1), observed=second)
    assert run(tmp_path, second + timedelta(minutes=10)) == {'static'}


def test_failure_recovery_manual_override_and_restart_preserve_first_activation(tmp_path):
    context, games = setup(tmp_path)
    minute = lambda n: FIRST + timedelta(minutes=n)
    publish(context, minute(0), games)
    run(tmp_path, minute(0))
    publish(context, minute(10), games)
    run(tmp_path, minute(10))
    marker = (tmp_path / 'mlb_static_activated_at').read_bytes()

    write_json(context / 'source_failure.json', dict(at=minute(11).isoformat(), http_status=406))
    assert not run(tmp_path, minute(12))
    assert (readiness(tmp_path)['status'], readiness(tmp_path)['reason']) == ('active', 'fixture_source_failure:406')
    assert not run(tmp_path, minute(12))
    assert readiness(tmp_path)['failed'] == 1
    assert not run(tmp_path, minute(13))
    assert readiness(tmp_path)['status'] == 'suspended'

    publish(context, minute(14), games, streak=1)
    assert not run(tmp_path, minute(14))
    publish(context, minute(15), games)
    assert not run(tmp_path, minute(15))
    assert readiness(tmp_path)['status'] == 'suspended'
    publish(context, minute(25), games)
    assert not run(tmp_path, minute(25))
    assert readiness(tmp_path)['status'] == 'active'
    assert run(tmp_path, minute(26)) == {'static'}
    assert (tmp_path / 'mlb_static_activated_at').read_bytes() == marker

    (context / 'disable_static').touch()
    assert not run(tmp_path, minute(27))
    assert readiness(tmp_path)['status'] == 'disabled'
    (context / 'disable_static').unlink()
    assert not run(tmp_path, minute(28), auto=False)  # Environment rollback overrides ready evidence.
    assert readiness(tmp_path)['reason'] == 'manual_disabled'
    assert (tmp_path / 'mlb_static_activated_at').read_bytes() == marker
    transitions = readiness(tmp_path)['transitions']
    assert any(item['status'] == 'suspended' for item in transitions)


def test_identity_selected_quotes_settlement_and_first_pitch_are_required(tmp_path):
    context, games = setup(tmp_path)
    row = capture(FIRST)
    row.loc[0, 'game'] = 'Home @ Away'
    publish(context, FIRST, games)
    assert not run(tmp_path, FIRST, board=row)
    assert readiness(tmp_path)['reason'] == 'no_verified_current_paired_prices'
    assert not run(tmp_path, FIRST, board=capture(FIRST, price=-123))
    assert not run(tmp_path, FIRST, retained={('mlb', 'static', 'MONEYLINE')})
    assert readiness(tmp_path)['ready'] == 0

    games['10'].update(doubleheader=True, game_number=2)
    publish(context, FIRST, games)
    assert not run(tmp_path, FIRST)
    assert readiness(tmp_path)['reason'] == 'settlement_first_pitch_unverified'
    games['10']['first_pitch'] = games['10']['scheduled_start']
    games['10']['home']['score'] = 9
    publish(context, FIRST, games)
    assert not run(tmp_path, FIRST)
    assert readiness(tmp_path)['reason'] == 'settlement_score_or_team_mismatch'
    games['10']['home']['score'] = 4
    publish(context, FIRST, games)
    assert not run(tmp_path, FIRST)
    assert readiness(tmp_path)['reason'] == 'awaiting_consecutive_market_collections'


def test_auto_only_admits_prospective_favorites_then_leaves_them_in_lifecycle(tmp_path):
    context, games = setup(tmp_path)
    prior_captures = [FIRST - timedelta(minutes=n) for n in (30, 15)]
    for n in (0, 10, 20, 30, 40):
        now = FIRST + timedelta(minutes=n)
        board = capture(now)
        prior_captures.append(now)
        raw_history = history(prior_captures)
        publish(context, now, games)
        allowed = ready_games(tmp_path, board, raw_history, now, (), auto=n < 40)
        result = update_favorite_tracking(board, tmp_path, as_of=now, market_history=raw_history,
                                          mlb_enabled=bool(allowed), mlb_promotion_games=allowed).iloc[0]
        if n <= 10:
            assert result.favorite_review_state != 'pending_addition'
        elif n == 20:
            assert result.favorite_review_state == 'pending_addition'
        else:
            assert result.red_fox_favorite == 'true' and result.favorite_rule_version == MLB_STATIC_VERSION
            assert result.favorite_first_qualified_at == (FIRST + timedelta(minutes=30)).isoformat()
