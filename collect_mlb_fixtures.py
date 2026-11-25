"""Independent MLB schedule, score and starting-pitcher collector."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests

from mlb_fixtures import CONTEXT_DIR, NEW_YORK, team_key, timestamp


BASE = 'https://statsapi.mlb.com/api'
GAME_TYPES = {'R', 'F', 'D', 'L', 'W', 'C'}
SCHEDULE_FIELDS = ('dates,games,gamePk,gameType,gameDate,officialDate,status,abstractGameState,detailedState,'
                   'startTimeTBD,doubleHeader,gameNumber,scheduledInnings,resumeDate,resumedFrom,resumedFromDate,'
                   'teams,away,home,team,id,name,abbreviation,clubName,score,probablePitcher,fullName,linescore,currentInning')


def fetch(url, **params):
    response = requests.get(f'{BASE}/{url}', params=params, timeout=5)
    response.raise_for_status()
    return response.json()


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
    for leg in ('away', 'home'):
        side = raw.get('teams', {}).get(leg, {})
        team = side.get('team', {})
        if not team.get('id') or not team.get('name'):
            return None
        names = [team.get(key) for key in ('name', 'abbreviation', 'clubName')]
        if names[1] and names[2]:
            names.append(f'{names[1]} {names[2]}')
        probable = side.get('probablePitcher') or {}
        game[leg] = dict(team_id=team['id'], aliases=sorted({team_key(name) for name in names if name}),
                         score=side.get('score', '-'), pitcher_id=probable.get('id'), pitcher_name=probable.get('fullName'),
                         pitcher_status='probable' if probable.get('id') else 'missing')
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
    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        print(f'[mlb-context] fixture {game["game_pk"]}: {type(error).__name__}')
        return {}


def write_context(directory, payload):
    with NamedTemporaryFile('w', encoding='utf-8', dir=directory, delete=False) as stream:
        json.dump(payload, stream, ensure_ascii=False)
        temporary = Path(stream.name)
    try:
        temporary.replace(directory / 'latest.json')
    finally:
        temporary.unlink(missing_ok=True)


def collect(data_dir, now):
    current_day = now.astimezone(NEW_YORK).date()
    directory = data_dir / CONTEXT_DIR
    pitchers_enabled = not (directory / 'disable_pitchers').exists()
    raw = fetch('v1/schedule', sportId=1, startDate=current_day - timedelta(days=15),
                endDate=current_day + timedelta(days=1),
                hydrate='team,linescore,probablePitcher' if pitchers_enabled else 'team,linescore', fields=SCHEDULE_FIELDS)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        previous = json.loads((directory / 'latest.json').read_text(encoding='utf-8'))['games']
    except (OSError, ValueError, KeyError):
        previous = {}
    games = {}
    conflicts = set()
    for day in raw['dates']:
        for row in day['games']:
            game = league_fixture(row)
            if game is None:
                continue
            if not pitchers_enabled:
                for leg in ('away', 'home'):
                    game[leg].update(pitcher_id=None, pitcher_name=None, pitcher_status='missing')
            pk = game['game_pk']
            if pk in games and games[pk] != game:
                conflicts.add(pk)
            games[pk] = game
    for pk in conflicts:
        del games[pk]
    for pk, game in games.items():
        old = previous.get(pk, {})
        if old.get('official_date') != game['official_date'] or game['state'] not in {'in', 'post'}:
            continue
        game['first_pitch'] = old.get('first_pitch')
        for leg in ('away', 'home'):
            prior = old.get(leg, {})
            if pitchers_enabled and prior.get('team_id') == game[leg]['team_id'] and prior.get('pitcher_status') == 'confirmed':
                game[leg].update({k: prior[k] for k in ('pitcher_status', 'pitcher_id', 'pitcher_name', 'confirmed_at')})
    needed = sorted((game for game in games.values() if game['state'] in {'in', 'post'}
                     and now - timestamp(game['scheduled_start']) <= timedelta(hours=9)
                     and ((not game['first_pitch'] and (pitchers_enabled or game['doubleheader'] or game['time_tbd']))
                          or (pitchers_enabled and any(game[leg]['pitcher_status'] != 'confirmed' for leg in ('away','home'))))),
                    key=lambda game: timestamp(game['scheduled_start']), reverse=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for game, proof in zip(needed[:16], pool.map(first_pitches, needed[:16])):
            if 'home' in proof:
                game['first_pitch'] = proof['home']['pitch']
            for leg, result in proof.items():
                if pitchers_enabled and result['pitcher_id']:
                    game[leg].update(pitcher_status='confirmed', pitcher_id=result['pitcher_id'],
                                     pitcher_name=result['pitcher_name'], confirmed_at=result['pitch'])
    observed_at = datetime.now(timezone.utc).isoformat()
    changes = []
    for pk, game in (games.items() if pitchers_enabled else ()):
        # Retrospective finals provide score evidence, not pregame pitcher records.
        if not (current_day - timedelta(days=1) <= datetime.fromisoformat(game['official_date']).date() <= current_day + timedelta(days=1)):
            continue
        for leg in ('away', 'home'):
            fields = ('pitcher_status', 'pitcher_id', 'pitcher_name')
            current = {k: game[leg].get(k) for k in fields}
            prior = {k: previous.get(pk, {}).get(leg, {}).get(k) for k in fields}
            if current != prior:
                changes.append(dict(game_pk=pk, official_date=game['official_date'], scheduled_start=game['scheduled_start'],
                                    team_id=game[leg]['team_id'], leg=leg, observed_at=observed_at,
                                    confirmed_at=game[leg].get('confirmed_at'), previous=prior, current=current))
    if changes:
        with (directory / 'pitcher_changes.jsonl').open('a', encoding='utf-8') as stream:
            for change in changes:
                stream.write(json.dumps(change, ensure_ascii=False) + '\n')
    write_context(directory, dict(observed_at=observed_at, source='MLB Stats API', pitchers_enabled=pitchers_enabled, games=games))
    print(f'[mlb-context] cached {len(games)} fixtures, wrote {len(changes)} pitcher observations')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=Path('data'), type=Path)
    args = parser.parse_args()
    collect(args.data_dir, datetime.now(timezone.utc))
