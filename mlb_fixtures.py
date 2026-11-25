"""Read the official MLB fixture cache and match games conservatively."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from zoneinfo import ZoneInfo

from team_aliases import normalize_team_name


CONTEXT_DIR = 'mlb_fixture_context'
SCORE_SOURCE = 'mlb_statsapi_fixture'
NEW_YORK = ZoneInfo('America/New_York')
CONTEXT_MAX_AGE = timedelta(minutes=15)


def write_json(path, payload):
    temporary = None
    try:
        with NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def timestamp(value):
    try:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return result.astimezone(timezone.utc) if result.tzinfo else None
    except (ValueError, OverflowError):
        return None


def team_key(name):
    key = normalize_team_name(name, 'mlb')
    if key in {'ath', 'athletics', 'ath athletics', 'oakland athletics', 'sacramento athletics'}:
        return 'athletics'
    return key


def fixture_sample(data_dir, now):
    path = Path(data_dir) / CONTEXT_DIR / 'latest.json'
    try:
        context = json.loads(path.read_text(encoding='utf-8'))
        captured = timestamp(context.get('observed_at'))
        if captured and timedelta(0) <= now - captured <= CONTEXT_MAX_AGE and isinstance(context.get('games'), dict):
            return context
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return None


def read_context(data_dir, now=None):
    context = fixture_sample(data_dir, now or datetime.now(timezone.utc))
    if context is None:
        return None
    return {pk: dict(game, observed_at=timestamp(context['observed_at']).isoformat()) for pk, game in context['games'].items()}


def last_source_failure(directory):
    try:
        return json.loads((directory / 'source_failure.json').read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}


def cutover_at(data_dir):
    try:
        return timestamp((Path(data_dir) / CONTEXT_DIR / 'active_from').read_text(encoding='utf-8').strip())
    except OSError:
        return None


def match_fixture(games, matchup, kickoff, game_pk=''):
    kickoff = timestamp(kickoff)
    teams = [team_key(s.strip()) for s in str(matchup).split(' @ ')]
    if kickoff is None or len(teams) != 2 or not all(teams) or teams[0] == teams[1]:
        return None
    date = kickoff.astimezone(NEW_YORK).date().isoformat()
    paired = [g for g in (games or {}).values() if g.get('official_date') == date
              and all(team in g.get(leg, {}).get('aliases', ()) for leg, team in zip(('away', 'home'), teams))]
    # For a same-day doubleheader require first-pitch proof unless an exact,
    # non-placeholder published start uniquely identifies the game. A known
    # gamePk still needs compatible fixture evidence to prevent stale joins.
    exact = [g for g in paired if not g.get('time_tbd') and timestamp(g.get('scheduled_start')) == kickoff
             and not (g.get('doubleheader') and g.get('game_number') == 2)]
    near_pitch = [g for g in paired if (g.get('doubleheader') or g.get('time_tbd')) and (pitch := timestamp(g.get('first_pitch')))
                  and abs(pitch - kickoff) <= timedelta(hours=1)]
    if game_pk:
        candidates = [g for g in exact + near_pitch if str(g.get('game_pk')) == str(game_pk)]
    elif len(exact) == 1 and all(g['game_pk'] == exact[0]['game_pk'] for g in near_pitch):
        candidates = exact
    elif not exact:
        candidates = near_pitch
        # An unstarted second game with an unproved time can overlap the
        # observed first pitch. Withhold unless every nearby fixture is proved.
        if any(g.get('doubleheader') and not g.get('first_pitch')
               and (g.get('time_tbd') or ((start := timestamp(g.get('scheduled_start'))) and abs(start - kickoff) <= timedelta(hours=1)))
               for g in paired):
            return None
    else:
        return None
    ids = {str(g['game_pk']) for g in candidates}
    if len(ids) != 1:
        return None
    game = candidates[0]
    return game if game.get('state') not in {'postponed', 'unresolved'} else None


def score_payload(game):
    if game is None or game.get('state') not in {'in', 'post'}:
        return None
    return dict(score_away=str(game['away']['score']), score_home=str(game['home']['score']),
                score_status=game['status'], score_state=game['state'], score_provider='mlb',
                score_provider_event_id=str(game['game_pk']), event_time=game['scheduled_start'],
                score_updated_at_utc=game.get('observed_at', ''))
