"""Production readiness for new MLB Static Moneyline promotions."""
import csv
from datetime import timedelta
import json

import pandas as pd

import red_fox_favorite as rf
from mlb_fixtures import (
    CONTEXT_DIR, NEW_YORK, SCORE_SOURCE, cutover_at, fixture_sample, last_source_failure,
    match_fixture, team_key, timestamp, write_json,
)
from mlb_static_favorite import _captures, _odds


STATE_FILE = 'static_readiness.json'
REQUIRED_COLLECTIONS = 2
SETTLEMENT_LOOKBACK = timedelta(days=3)
MARKET_MAX_AGE = timedelta(minutes=rf.CONFIG.normal_freshness_minutes)


def settlement_readiness(data_dir, fixtures, now):
    """Require recent outcomes from the actual sportsbook-to-MLB score writer."""
    verified = False
    try:
        with (data_dir / 'final_scores_history.csv').open(encoding='utf-8-sig', newline='') as source:
            for row in csv.DictReader(source):
                if row.get('score_source') != SCORE_SOURCE:
                    continue
                resolved_at = timestamp(row.get('resolved_at_utc'))
                if not resolved_at or not timedelta(0) <= now - resolved_at <= SETTLEMENT_LOOKBACK:
                    continue
                game = fixtures.get(row.get('score_provider_event_id'))
                if not game or game['state'] != 'post':
                    return 'settlement_fixture_unverified'
                if (game['time_tbd'] or (game['doubleheader'] and game['game_number'] == 2)) and not timestamp(game['first_pitch']):
                    return 'settlement_first_pitch_unverified'
                expected = {leg: team_key(row[column]) for leg, column in (('away', 'team1'), ('home', 'team2'))}
                if any(expected[leg] not in game[leg]['aliases']
                       or str(game[leg]['score']) != row[f'team{number}_score']
                       for number, leg in enumerate(('away', 'home'), 1)):
                    return 'settlement_score_or_team_mismatch'
                scheduled = timestamp(game['scheduled_start'])
                if scheduled and timedelta(0) < now - scheduled <= SETTLEMENT_LOOKBACK and scheduled < resolved_at:
                    verified = True
    except FileNotFoundError:
        pass
    return '' if verified else 'recent_settlement_proof_missing'


def verified_markets(board, history, fixtures, now, retained_keys):
    """Match current paired prices to one official MLB fixture per event."""
    if not {'sport', 'market_display', 'game_id'}.issubset(board):
        return {}
    moneylines = board.loc[board['sport'].eq('mlb') & board['market_display'].eq('MONEYLINE')]
    if moneylines.empty:
        return {}
    latest = pd.Timestamp(now)
    captures = _captures(history, latest)
    verified = {}
    fixture_ids = {}
    for event, rows in moneylines.groupby('game_id', sort=False):
        event = str(event)
        if len(rows) != 1 or ('mlb', event, 'MONEYLINE') in retained_keys:
            continue
        row = rows.iloc[0]
        kickoff = timestamp(row.get('kickoff_iso'))
        if not kickoff or kickoff <= now:
            continue
        game = match_fixture(fixtures, row.get('game'), kickoff)
        if game is None or game['state'] != 'pre':
            continue
        observed = timestamp(row.get('state_as_of_utc') or row.get('source_latest_at'))
        if not observed or not timedelta(0) <= now - observed <= MARKET_MAX_AGE:
            continue
        quotes = captures.get(event, {})
        observed = pd.Timestamp(observed)
        if observed not in quotes:
            same_clock = [stamp for stamp in quotes if stamp.floor('us') == observed]
            if len(same_clock) != 1:
                continue
            observed = same_clock[0]
        paired = quotes[observed]
        sides = rf._sides(row)
        team_ids = {rf._side_identity(team) for team in row['game'].split(' @ ')}
        if len(sides) != 2 or set(paired) != team_ids or {rf._side_identity(side.get('flagged_side')) for side in sides} != team_ids:
            continue
        if any(
            pd.to_datetime(side.get('source_latest_at'), errors='coerce', utc=True) not in (observed, observed.floor('us'))
            or _odds(side.get('current_line'), identity) != paired[identity][0]
            or _odds(side.get('open_line'), identity) is None
            or _odds(side.get('open_line'), identity) != paired[identity][1]
            for side in sides if (identity := rf._side_identity(side.get('flagged_side')))
        ):
            continue
        verified[event] = observed
        fixture_ids.setdefault(game['game_pk'], []).append(event)
    for events in fixture_ids.values():
        if len(events) > 1:
            for event in events:
                verified.pop(event)
    return verified


def evaluate_inputs(data_dir, board, history, now, retained_keys, auto):
    directory = data_dir / CONTEXT_DIR
    if not auto or (directory / 'disable_static').exists():
        return 'manual_disabled', {}
    sample = fixture_sample(data_dir, now)
    failure = last_source_failure(directory)
    failed_at = timestamp(failure.get('at'))
    if failed_at and (not sample or failed_at >= timestamp(sample['observed_at'])):
        reason = failure.get('http_status') or failure.get('reason') or 'unknown'
        return f'fixture_source_failure:{reason}', {}
    if not sample or sample.get('source') != 'MLB Stats API':
        return 'fixture_cache_stale_or_missing', {}
    if sample.get('schedule_status') == 'invalid_fixture_details':
        return 'fixture_details_invalid_or_conflicting', {}
    fixtures = sample['games']
    today = now.astimezone(NEW_YORK).date().isoformat()
    if not any(game['official_date'] >= today and game['state'] in {'pre', 'in', 'post'} for game in fixtures.values()):
        return 'no_current_mlb_fixtures', {}
    if sample.get('populated_streak', 0) < REQUIRED_COLLECTIONS:
        return 'awaiting_consecutive_fixture_collections', {}
    quotes = verified_markets(board, history, fixtures, now, retained_keys)
    if not quotes:
        if not any(game['state'] == 'pre' and (start := timestamp(game['scheduled_start']))
                   and timedelta(0) < start - now <= timedelta(hours=3) for game in fixtures.values()):
            return 'no_near_term_mlb_markets_expected', {}
        return 'no_verified_current_paired_prices', {}
    if cutover_at(data_dir) is None:
        return 'verified_mlb_settlement_disabled', {}
    reason = settlement_readiness(data_dir, fixtures, now)
    return (reason, {}) if reason else ('', quotes)


def ready_games(data_dir, board, history, now, retained_keys, *, auto):
    """Record the readiness transition and return events eligible for new promotions."""
    directory = data_dir / CONTEXT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / STATE_FILE
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        state = {}
    reason, quotes = evaluate_inputs(data_dir, board, history, now, retained_keys, auto)
    previous, prior_reason = state.get('status'), state.get('reason')
    status = previous or 'waiting'
    if reason == 'no_near_term_mlb_markets_expected':
        state['failed'] = 0
        if status != 'active':
            state['ready'] = 0
    elif reason:
        if state.get('last_check_at') != now.isoformat():
            state['failed'] = state.get('failed', 0) + 1
        state['ready'] = 0
        if reason == 'manual_disabled':
            status = 'disabled'
        elif reason == 'no_current_mlb_fixtures':
            status = 'waiting'
        elif status == 'active' and state['failed'] >= REQUIRED_COLLECTIONS:
            status = 'suspended'
        elif status != 'active' and status != 'suspended':
            status = 'waiting'
    else:
        state['failed'] = 0
        newest = max(quotes.values())
        prior_capture = pd.to_datetime(state.get('last_market_at'), errors='coerce', utc=True)
        if pd.isna(prior_capture) or newest > prior_capture:
            if not pd.isna(prior_capture) and newest - prior_capture > MARKET_MAX_AGE:
                state['ready'] = 0
            state['last_market_at'] = newest.isoformat()
            state['ready'] = min(REQUIRED_COLLECTIONS, state.get('ready', 0) + 1)
        if status != 'active' and state.get('ready', 0) >= REQUIRED_COLLECTIONS:
            status = 'active'
            state.setdefault('first_active_at', now.isoformat())
        # Never use a capture preceding an activation or a recovery for a new
        # promotion. In-flight Favorites continue their original lifecycle.
        if status == 'active' and (previous != 'active' or prior_reason not in ('ready', 'no_near_term_mlb_markets_expected')):
            state['promotion_since'] = now.isoformat()
        reason = 'ready' if status == 'active' else 'awaiting_consecutive_market_collections'
        if status == 'disabled':
            status = 'waiting'
    state.update(status=status, reason=reason, last_check_at=now.isoformat())
    if (status, reason) != (previous, prior_reason):
        state.setdefault('transitions', []).append(dict(at=now.isoformat(), previous=previous, status=status, reason=reason))
        print(f'[mlb-static] {previous or "uninitialized"} -> {status}: {reason}', flush=True)
    write_json(path, state)
    if status != 'active' or not quotes:
        return frozenset()
    marker = data_dir / 'mlb_static_activated_at'
    try:
        with marker.open('x', encoding='utf-8') as stream:
            stream.write(state['first_active_at'] + '\n')
    except FileExistsError:
        pass
    since = pd.Timestamp(state['promotion_since'])
    return frozenset(event for event, observed in quotes.items() if observed > since)
