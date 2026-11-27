"""Independent MLB schedule, score and starting-pitcher collector."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import argparse
import json
from pathlib import Path

import requests

from mlb_fixtures import CONTEXT_DIR, CONTEXT_MAX_AGE, NEW_YORK, last_source_failure, team_key, timestamp, write_json


BASE = 'https://statsapi.mlb.com/api'
GAME_TYPES = {'R', 'F', 'D', 'L', 'W', 'C'}
SCHEDULE_FIELDS = ('dates,games,gamePk,gameType,gameDate,officialDate,status,abstractGameState,detailedState,'
                   'startTimeTBD,doubleHeader,gameNumber,scheduledInnings,resumeDate,resumedFrom,resumedFromDate,'
                   'teams,away,home,team,id,name,abbreviation,clubName,score,probablePitcher,fullName,linescore,currentInning')
PITCHER_FIELDS = 'dates,games,gamePk,teams,away,home,team,id,probablePitcher,fullName'


def fetch(url, **params):
    response = requests.get(f'{BASE}/{url}', params=params, headers={'Accept': 'application/json'}, timeout=5)
    response.raise_for_status()
    return response.json()


def schedule_window(start, end):
    params = dict(sportId=1, startDate=start, endDate=end)
    try:
        return fetch('v1/schedule', **params, hydrate='team,linescore', fields=SCHEDULE_FIELDS)
    except requests.HTTPError as error:
        if error.response is None or error.response.status_code != 406:
            raise
        # An unprojected calendar can verify a quiet window. It lacks the
        # fixture detail needed to settle populated windows.
        try:
            calendars = [(None, fetch('v1/schedule', **params))]
        except requests.HTTPError as calendar_error:
            if calendar_error.response is None or calendar_error.response.status_code != 406:
                raise
            calendars = [(year, fetch('v1/schedule', sportId=1, season=year))
                         for year in range(start.year, end.year + 1)]
        first, last = start.isoformat(), end.isoformat()
        for year, calendar in calendars:
            games = [game for day in calendar['dates'] for game in day['games']]
            if calendar.get('totalGames') != len(games) or (year is not None and not games):
                raise error
            for game in games:
                day, kind = game.get('officialDate'), game.get('gameType')
                if not day or not kind or (year is not None and not day.startswith(f'{year}-')):
                    raise error
                in_window = first <= day <= last
                if (year is None and not in_window) or (in_window and kind in GAME_TYPES):
                    raise error
        print('[mlb-context] projected schedule returned 406; official calendar confirms no qualifying fixtures')
        return {'dates': []}


def probable_pitchers(start, end):
    try:
        raw = fetch('v1/schedule', sportId=1, startDate=start, endDate=end,
                    hydrate='probablePitcher', fields=PITCHER_FIELDS)
        pitchers = {}
        for day in raw['dates']:
            for game in day['games']:
                sides = {}
                for leg in ('away', 'home'):
                    side = game['teams'].get(leg, {})
                    team = side.get('team') or {}
                    pitcher = side.get('probablePitcher') or {}
                    if team.get('id') and pitcher.get('id'):
                        sides[leg] = (team['id'], pitcher['id'], pitcher.get('fullName'))
                pitchers[str(game['gamePk'])] = sides
        return pitchers
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError) as error:
        print(f'[mlb-context] optional probable pitchers unavailable: {error}')
        return None


def league_fixture(raw):
    if (not raw.get('gamePk') or raw.get('gameType') not in GAME_TYPES or
            not timestamp(raw.get('gameDate')) or not raw.get('officialDate')):
        return None
    status = raw.get('status', {})
    detail = status.get('detailedState', '')
    state = {'Live': 'in', 'Final': 'post', 'Preview': 'pre'}.get(status.get('abstractGameState'), 'unresolved')
    if 'postpon' in detail.lower() or 'cancel' in detail.lower():
        state = 'postponed'
    elif 'suspend' in detail.lower() or raw.get('resumeDate') or raw.get('resumedFrom') or raw.get('resumedFromDate'):
        state = 'unresolved'
    elif state == 'post':
        # Only send ordinary full-game finals into the common grading paths.
        # Shortened or otherwise unusual settlements need separate review.
        linescore = raw.get('linescore', {})
        inning = linescore.get('currentInning') or 0
        scheduled = raw.get('scheduledInnings') or linescore.get('scheduledInnings') or 9
        if detail != 'Final' or inning < scheduled:
            state = 'unresolved'
    game = dict(game_pk=str(raw['gamePk']), official_date=raw['officialDate'],
                scheduled_start=raw['gameDate'], time_tbd=bool(status.get('startTimeTBD')),
                doubleheader=raw.get('doubleHeader') in {'Y', 'S'}, game_number=raw.get('gameNumber'),
                status=detail, state=state, first_pitch=None)
    if state == 'post':
        # Preserve the same official inning evidence used to validate the
        # final so the separate historical Total grader can use it too.
        game.update(innings=inning, scheduled_innings=scheduled)
    for leg in ('away', 'home'):
        side = raw.get('teams', {}).get(leg, {})
        team = side.get('team', {})
        if not team.get('id') or not team.get('name'):
            return None
        names = [team.get(key) for key in ('name', 'abbreviation', 'clubName')]
        if names[1] and names[2]:
            names.append(f'{names[1]} {names[2]}')
        game[leg] = dict(team_id=team['id'], aliases=sorted({team_key(name) for name in names if name}),
                         score=side.get('score', '-'), pitcher_id=None, pitcher_name=None, pitcher_status='missing')
    scores = [game[leg]['score'] for leg in ('away', 'home')]
    if game['state'] == 'post' and (not all(isinstance(score, int) for score in scores) or scores[0] == scores[1]):
        game['state'] = 'unresolved'
    return game


def first_pitches(game):
    try:
        feed = fetch(f'v1.1/game/{game["game_pk"]}/feed/live')
        meta = feed['gameData']
        if str(meta['game']['pk']) != game['game_pk'] or any(meta['teams'][leg]['id'] != game[leg]['team_id'] for leg in ('away', 'home')):
            return {}
        people = meta.get('players', {})
        evidence = {}
        for play in feed['liveData']['plays']['allPlays']:
            about = play['about']
            if about.get('inning') != 1:
                break
            leg = 'home' if about.get('isTopInning') else 'away'
            if leg in evidence:
                continue
            pitch = next((event['startTime'] for event in play['playEvents']
                          if event.get('isPitch') and timestamp(event.get('startTime'))), None)
            pitcher = play.get('matchup', {}).get('pitcher', {})
            if pitch:
                # The at-bat matchup can change mid-at-bat. Confirm the first
                # pitcher from the official boxscore's starter order below.
                ordered = feed['liveData'].get('boxscore', {}).get('teams', {}).get(leg, {}).get('pitchers', [])
                evidence[leg] = (pitch, ordered[0] if ordered and pitcher.get('id') else None)
        return {leg: dict(pitch=pitch, pitcher_id=pid, pitcher_name=people.get(f'ID{pid}', {}).get('fullName'))
                for leg, (pitch, pid) in evidence.items()}
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError) as error:
        print(f'[mlb-context] fixture {game["game_pk"]}: {type(error).__name__}')
        return {}


def collect(data_dir, now):
    current_day = now.astimezone(NEW_YORK).date()
    directory = data_dir / CONTEXT_DIR
    pitchers_enabled = not (directory / 'disable_pitchers').exists()
    start, end = current_day - timedelta(days=15), current_day + timedelta(days=1)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        raw = schedule_window(start, end)
        records = [row for day in raw['dates'] for row in day['games'] if row.get('gameType') in GAME_TYPES]
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError) as error:
        # Settlement may still use fresh, previously verified evidence. New
        # Static entries must see the latest collection failure immediately.
        write_json(directory / 'source_failure.json', dict(
            at=datetime.now(timezone.utc).isoformat(),
            reason=type(error).__name__,
            http_status=error.response.status_code if isinstance(error, requests.HTTPError) and error.response is not None else None,
        ))
        raise
    try:
        prior_sample = json.loads((directory / 'latest.json').read_text(encoding='utf-8'))
        previous = prior_sample['games']
        failure_at = timestamp(last_source_failure(directory).get('at'))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        prior_sample, previous, failure_at = {}, {}, None
    games = {}
    conflicts = set()
    invalid = 0
    for row in records:
        try:
            game = league_fixture(row)
        except (ValueError, KeyError, TypeError, AttributeError):
            game = None
        if game is None:
            invalid += 1
            continue
        pk = game['game_pk']
        if pk in games and games[pk] != game:
            conflicts.add(pk)
        games[pk] = game
    for pk in conflicts:
        del games[pk]
    for pk, game in games.items():
        old = previous.get(pk, {})
        if (old.get('official_date') != game['official_date'] or old.get('scheduled_start') != game['scheduled_start']
                or game['state'] not in {'in', 'post'}
                or any(old.get(leg, {}).get('team_id') != game[leg]['team_id'] for leg in ('away', 'home'))):
            continue
        game['first_pitch'] = old.get('first_pitch')
        for leg in ('away', 'home'):
            prior = old.get(leg, {})
            if pitchers_enabled and prior.get('team_id') == game[leg]['team_id'] and prior.get('pitcher_status') == 'confirmed':
                game[leg].update({k: prior[k] for k in ('pitcher_status', 'pitcher_id', 'pitcher_name', 'confirmed_at')})
    recent_started = sorted((game for game in games.values() if game['state'] in {'in', 'post'}
                             and now - timestamp(game['scheduled_start']) <= timedelta(hours=9)),
                            key=lambda game: timestamp(game['scheduled_start']), reverse=True)
    essential = [game for game in recent_started if not game['first_pitch'] and (game['doubleheader'] or game['time_tbd'])]
    collect_first_pitches(essential[:8], pitchers_enabled)

    observed = datetime.now(timezone.utc)
    observed_at = observed.isoformat()
    prior_at = timestamp(prior_sample.get('observed_at'))
    recent_population = any(game['official_date'] >= current_day.isoformat() and game['state'] in {'pre', 'in', 'post'}
                            for game in games.values())
    continuing = (prior_at and timedelta(0) < observed - prior_at <= CONTEXT_MAX_AGE
                  and (failure_at is None or failure_at < prior_at))
    populated_streak = (min(2, prior_sample.get('populated_streak', 0) + 1) if continuing else 1)
    if not recent_population or conflicts or invalid:
        populated_streak = 0
    schedule_status = 'invalid_fixture_details' if conflicts or invalid else ('fixtures_available' if games else 'no_games_scheduled')
    payload = dict(observed_at=observed_at, source='MLB Stats API', schedule_status=schedule_status,
                   populated_streak=populated_streak, pitchers_enabled=pitchers_enabled, games=games)
    path = directory / 'latest.json'
    # Publish essential identity/final scores before trying the optional feed.
    write_json(path, payload)
    print(f'[mlb-context] cached {len(games)} fixtures, populated streak {populated_streak}', flush=True)
    if not pitchers_enabled or not games:
        return

    pitcher_start = current_day - timedelta(days=1)
    recent = {pk for pk, game in games.items() if pitcher_start.isoformat() <= game['official_date'] <= end.isoformat()}
    pitchers = probable_pitchers(pitcher_start, end) if recent else None
    for pk in recent.intersection(pitchers or {}):
        for leg, (team_id, pitcher_id, pitcher_name) in pitchers[pk].items():
            side = games[pk][leg]
            if team_id == side['team_id'] and side['pitcher_status'] != 'confirmed':
                side.update(pitcher_id=pitcher_id, pitcher_name=pitcher_name, pitcher_status='probable')
    optional = [game for game in recent_started if game not in essential
                and any(game[leg]['pitcher_status'] != 'confirmed' for leg in ('away', 'home'))]
    collect_first_pitches(optional[:8], True)
    changes = []
    for pk in recent:
        game = games[pk]
        # Retrospective finals provide score evidence, not pregame pitcher records.
        if not (current_day - timedelta(days=1) <= datetime.fromisoformat(game['official_date']).date() <= current_day + timedelta(days=1)):
            continue
        for leg in ('away', 'home'):
            fields = ('pitcher_status', 'pitcher_id', 'pitcher_name')
            current = {k: game[leg].get(k) for k in fields}
            if pitchers is None and current['pitcher_status'] == 'missing':
                continue
            prior = {k: previous.get(pk, {}).get(leg, {}).get(k) for k in fields}
            if current != prior:
                changes.append(dict(game_pk=pk, official_date=game['official_date'], scheduled_start=game['scheduled_start'],
                                    team_id=game[leg]['team_id'], leg=leg, observed_at=observed_at,
                                    confirmed_at=game[leg].get('confirmed_at'), previous=prior, current=current))
    if changes:
        with (directory / 'pitcher_changes.jsonl').open('a', encoding='utf-8') as stream:
            for change in changes:
                stream.write(json.dumps(change, ensure_ascii=False) + '\n')
    write_json(path, payload)
    print(f'[mlb-context] wrote {len(changes)} pitcher observations')


def collect_first_pitches(games, pitchers_enabled):
    if not games:
        return
    with ThreadPoolExecutor(max_workers=4) as pool:
        for game, proof in zip(games, pool.map(first_pitches, games)):
            if 'home' in proof:
                game['first_pitch'] = proof['home']['pitch']
            for leg, result in proof.items():
                if pitchers_enabled and result['pitcher_id']:
                    game[leg].update(pitcher_status='confirmed', pitcher_id=result['pitcher_id'],
                                     pitcher_name=result['pitcher_name'], confirmed_at=result['pitch'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=Path('data'), type=Path)
    args = parser.parse_args()
    collect(args.data_dir, datetime.now(timezone.utc))
