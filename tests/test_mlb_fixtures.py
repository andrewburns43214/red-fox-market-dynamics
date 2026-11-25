from datetime import date, datetime, timedelta, timezone
import json

import pandas as pd
import pytest
import requests

import build_live_recent
import collect_mlb_fixtures as collector
import main
from mlb_fixtures import CONTEXT_DIR, SCORE_SOURCE, match_fixture, read_context, timestamp
from performance_ledger import _attach_results


MATCHUP = 'BAL Orioles @ NY Yankees'


def fixture(pk, start, *, number=1, double=False, tbd=False, detail='Final', runs=(3, 6)):
    away, home = [dict(team=dict(id=id_, name=name, abbreviation=abbr, clubName=name.split()[-1]), score=score)
                  for id_, name, abbr, score in [(110, 'Baltimore Orioles', 'BAL', runs[0]), (147, 'New York Yankees', 'NYY', runs[1])]]
    return dict(gamePk=pk, gameType='R', gameDate=start, officialDate=start[:10], doubleHeader='Y' if double else 'N',
                gameNumber=number, teams=dict(away=away, home=home), linescore=dict(currentInning=9),
                status=dict(detailedState=detail, abstractGameState='Final', startTimeTBD=tbd))


def normalized(*items):
    return {str(item['gamePk']): collector.league_fixture(item) for item in items}


def http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f'HTTP {status}', response=response)


def test_verified_empty_schedule_is_distinct_from_unavailable_fixture_feed(tmp_path, monkeypatch):
    now = datetime(2026, 11, 25, 16, tzinfo=timezone.utc)
    directory = tmp_path / CONTEXT_DIR
    directory.mkdir()
    (directory / 'disable_pitchers').touch()
    requested = []

    def fetch(_, **params):
        requested.append(params)
        if 'season' in params:
            return {'totalGames': 1, 'dates': [{'games': [{'officialDate': '2026-09-25', 'gameType': 'R'}]}]}
        raise http_error(406)

    monkeypatch.setattr(collector, 'fetch', fetch)
    collector.collect(tmp_path, now)
    output = directory / 'latest.json'
    previous = output.read_bytes()
    assert json.loads(previous)['schedule_status'] == 'no_games_scheduled'
    assert json.loads(previous)['games'] == {}
    assert [request.get('hydrate') for request in requested] == ['team,linescore', None, None]
    assert requested[-1]['season'] == 2026

    def populated(_, **params):
        if 'hydrate' in params:
            raise http_error(406)
        return {'totalGames': 1, 'dates': [{'games': [{'officialDate': '2026-11-25', 'gameType': 'R'}]}]}

    monkeypatch.setattr(collector, 'fetch', populated)
    with pytest.raises(requests.HTTPError, match='406'):
        collector.collect(tmp_path, now)
    assert output.read_bytes() == previous

    monkeypatch.setattr(collector, 'fetch', lambda *_args, **_params: {'totalGames': 0, 'dates': []})
    assert collector.schedule_window(date(2026, 11, 10), date(2026, 11, 26)) == {'totalGames': 0, 'dates': []}


def test_optional_pitcher_access_failure_does_not_block_fixture_cache(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    raw = fixture(123, (now + timedelta(minutes=5)).isoformat(), detail='Scheduled')
    raw['officialDate'] = now.astimezone(collector.NEW_YORK).date().isoformat()
    raw['status']['abstractGameState'] = 'Preview'
    requests_seen = []

    def fetch(_, **params):
        requests_seen.append(params['hydrate'])
        if params['hydrate'] == 'probablePitcher':
            raise http_error(406)
        return {'dates': [{'games': [raw]}]}

    monkeypatch.setattr(collector, 'fetch', fetch)
    collector.collect(tmp_path, now)
    game = read_context(tmp_path)['123']
    assert game['state'] == 'pre'
    assert game['away']['pitcher_status'] == 'missing'
    assert not (tmp_path / CONTEXT_DIR / 'pitcher_changes.jsonl').exists()
    assert requests_seen == ['team,linescore', 'probablePitcher']


def test_series_matching_cannot_borrow_previous_or_next_result():
    games = normalized(fixture(1, '2026-09-24T23:05:00Z'), fixture(2, '2026-09-26T23:05:00Z', runs=(7, 1)))
    assert build_live_recent.match_live_score(games, 'mlb', MATCHUP, pd.Timestamp('2026-09-24T23:05Z'))['score_away'] == '3'
    assert build_live_recent.match_live_score(games, 'mlb', MATCHUP, pd.Timestamp('2026-09-26T23:05Z'))['score_away'] == '7'
    assert match_fixture(games, MATCHUP, '2026-09-25T23:05Z') is None
    assert match_fixture(games, 'NY Yankees @ BAL Orioles', '2026-09-26T23:05Z') is None
    assert match_fixture(games, MATCHUP, '2026-09-26T23:05Z', game_pk='1') is None


def test_doubleheader_uses_confirmed_pitch_clock_for_placeholder():
    games = normalized(fixture(823491, '2026-09-25T20:05:00Z', double=True, runs=(10, 2)),
                       fixture(823489, '2026-09-25T20:10:00Z', number=2, double=True, tbd=True))
    games['823491']['first_pitch'] = '2026-09-25T20:08:42.434Z'
    assert match_fixture(games, MATCHUP, '2026-09-25T23:05Z') is None
    games['823489']['first_pitch'] = '2026-09-25T23:31:27.969Z'
    assert match_fixture(games, MATCHUP, '2026-09-25T23:05Z')['game_number'] == 2
    assert match_fixture(games, MATCHUP, '2026-09-25T20:05Z')['game_number'] == 1
    assert match_fixture(games, MATCHUP, '2026-09-25T21:30Z') is None
    games['823489']['first_pitch'] = '2026-09-25T20:31:27.969Z'
    assert match_fixture(games, MATCHUP, '2026-09-25T20:05Z') is None
    assert match_fixture(games, MATCHUP, '2026-09-25T20:05Z', game_pk='823491')['game_number'] == 1


def test_postponed_and_rescheduled_fixture_never_inherits_old_score():
    cancelled = fixture(1, '2026-09-24T23:05:00Z', detail='Postponed')
    replacement = fixture(2, '2026-09-26T23:05:00Z')
    replacement['rescheduledFromDate'] = '2026-09-24'
    games = normalized(cancelled, replacement)
    assert match_fixture(games, MATCHUP, '2026-09-24T23:05Z') is None
    assert match_fixture(games, MATCHUP, '2026-09-26T22:45Z') is None
    assert match_fixture(games, MATCHUP, '2026-09-26T23:05Z')['game_pk'] == '2'
    replacement['resumedFromDate'] = '2026-09-24'
    assert match_fixture(normalized(replacement), MATCHUP, '2026-09-26T23:05Z') is None


def test_new_final_history_uses_fixture_evidence_and_freezes_existing_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / 'data'
    directory = data / CONTEXT_DIR
    directory.mkdir(parents=True)
    (directory / 'active_from').write_text('2026-09-25T00:00:00Z')
    verified = fixture(2, '2026-09-26T23:05:00Z')
    (directory / 'latest.json').write_text(json.dumps(dict(observed_at=datetime.now(timezone.utc).isoformat(), games=normalized(verified))))
    pd.DataFrame([dict(game_id='archived', team1='a', team2='b', team1_score='5', team2_score='2',
                       score_source='old', score_provider_event_id='9', score_evidence_hash='keep')]).to_csv(data/'final_scores_history.csv', index=False)
    rows = [dict(sport='mlb', game_id=game_id, game=MATCHUP, dk_start_iso=f'2026-09-{day}T23:05:00Z',
                 side=side, final_score_for='99', final_score_against='98')
            for game_id, day in [('new', 26), ('nearby_only', 25), ('reused', 25), ('reused', 26)]
            for side in MATCHUP.split(' @ ')]
    pd.DataFrame(rows).to_csv(data/'snapshots.csv', index=False)
    for _ in range(2):
        main.update_final_scores_history()
    history = pd.read_csv(data/'final_scores_history.csv', dtype=str).set_index('game_id')
    assert set(history.index) == {'archived', 'new'}
    assert history.loc['archived', 'score_evidence_hash'] == 'keep'
    assert history.loc['new', ['team1_score', 'team2_score', 'score_source', 'score_provider_event_id']].tolist() == ['3','6',SCORE_SOURCE,'2']


def test_grading_preserves_old_result_and_requires_verified_prospective_score():
    scores = pd.DataFrame([dict(game_id=id_, team1='BAL Orioles', team1_score='3', team2='NY Yankees', team2_score='6',
                                score_provider_event_id='23', score_source=source)
                           for id_, source in [('existing',SCORE_SOURCE), ('pending','snapshots.csv'), ('verified', SCORE_SOURCE)]])
    ledger = pd.DataFrame([dict(event_id=id_, ledger_id=id_, favorite_qualified='yes', sport='mlb', market='MONEYLINE',
                                side='BAL Orioles', scheduled_start='2026-09-26T23:05:00Z', decision_price='+105',
                                grade='W' if id_=='existing' else '', final_score='historical proof' if id_=='existing' else '')
                           for id_ in ['existing','pending','verified']])
    result = _attach_results(ledger, scores, mlb_cutover=timestamp('2026-09-25T00:00Z')).set_index('event_id')
    assert result.loc['existing', ['grade','final_score']].tolist() == ['W','historical proof']
    assert result.loc['pending', 'audit_status'] == 'awaiting_verified_mlb_fixture'
    assert result.loc['verified', 'grade'] == 'L'


def test_pitcher_observations_record_changes_without_inventing_confirmation(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    start = (now + timedelta(hours=1)).isoformat()
    raw = fixture(123, start, detail='Scheduled')
    raw['officialDate'] = now.astimezone(collector.NEW_YORK).date().isoformat()
    raw['status']['abstractGameState'] = 'Preview'
    monkeypatch.setattr(collector, 'fetch', lambda *args, **kwargs: {'dates': [{'games': [raw]}]})
    raw['teams']['away']['probablePitcher'] = {'id': 1, 'fullName': 'First Pitcher'}
    collector.collect(tmp_path, now)
    raw['teams']['away']['probablePitcher'] = {'id': 2, 'fullName': 'Replacement'}
    collector.collect(tmp_path, now)
    collector.collect(tmp_path, now)
    games = read_context(tmp_path)
    assert games['123']['away']['pitcher_status'] == 'probable'
    assert games['123']['home']['pitcher_status'] == 'missing'
    changes = [json.loads(line) for line in (tmp_path/CONTEXT_DIR/'pitcher_changes.jsonl').read_text().splitlines()]
    away = [change for change in changes if change['leg']=='away']
    assert len(away)==2 and away[1]['previous']['pitcher_id']==1 and away[1]['current']['pitcher_id']==2
    assert read_context(tmp_path, now + timedelta(minutes=20)) is None
    (tmp_path/CONTEXT_DIR/'disable_pitchers').touch()
    collector.collect(tmp_path, now)
    assert read_context(tmp_path)['123']['away']['pitcher_status'] == 'missing'
    assert len((tmp_path/CONTEXT_DIR/'pitcher_changes.jsonl').read_text().splitlines()) == len(changes)


def test_first_pitch_evidence_confirms_only_the_pitching_half_and_matching_fixture(monkeypatch):
    game = normalized(fixture(823489, '2026-09-25T23:05:00Z'))['823489']
    pitch = '2026-09-25T23:31:27.969Z'
    payload = {
        'gameData': dict(game={'pk': 823489}, teams={'away': {'id': 110}, 'home': {'id': 147}},
                         players={'ID12': {'fullName': 'Actual Starter'}}),
        'liveData': {
            'plays': {'allPlays': [{'about': {'inning': 1, 'isTopInning': True}, 'matchup': {'pitcher': {'id': 99}},
                                  'playEvents': [{'isPitch': True, 'startTime': pitch}]}]},
            'boxscore': {'teams': {'home': {'pitchers': [12, 99]}}},
        },
    }
    monkeypatch.setattr(collector, 'fetch', lambda *_: payload)
    assert collector.first_pitches(game) == {'home': {'pitch': pitch, 'pitcher_id': 12, 'pitcher_name': 'Actual Starter'}}
    payload['gameData']['game']['pk'] = 823491
    assert collector.first_pitches(game) == {}
