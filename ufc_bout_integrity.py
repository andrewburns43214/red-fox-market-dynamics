"""UFC-only identity, phase and immutable handoff evidence."""

from datetime import timedelta
from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd
from uuid import uuid4

SWITCH = Path('/etc/redfox-ufc-integrity.json')
HANDOFF = 'ufc_verified_bout_handoff'
CONTEXT = 'ufc_verified_bout_context'
SCORE_SOURCE = 'verified_ufc_bout_context'
PROOF = 'ufc_bout_proof'
MAX_PHASE_AGE = pd.Timedelta(seconds=150)
RISK_NAMES = {'Opponent Change', 'Bout Suspended', 'Weight Miss', 'Weigh-In Risk'}

ALIASES = {
    'longxiao': 'xiaolong', 'congwang': 'wangcong', 'josedelgado': 'josemigueldelgado',
    'seanking': 'seankingiii', 'allenfrye': 'allenfryejr', 'lupitagodinez': 'loopygodinez',
    'timothycuamba': 'timmycuamba', 'ramazonbektemirov': 'ramazantemirov', 'josephkropschot': 'joekropschot',
    'josehenrique': 'josesouza', 'beatrizmesquita': 'biamesquita', 'charlieradtke': 'charlesradtke',
    'christianleroyduncan': 'chrisduncan', 'heilialateng': 'alatengheili', 'valescamachado': 'tinablack',
    'marcotuliosilva': 'marcotulio', 'raulrosas': 'raulrosasjr', 'robertfont': 'robfont',
}


def stamp(value):
    return pd.to_datetime(value, utc=True, errors='coerce', format='mixed')


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def settings():
    data = read_json(SWITCH)
    return data.get('enabled') is True, stamp(data.get('since'))


def cache_dir(data_dir):
    return Path(data_dir).parent / 'var' / 'ufc-integrity'


def retain_favorite_quotes(frame, data_dir):
    """Keep exact pre-bout candidates in case the provider reports the bell late."""
    if frame.empty or "sport" not in frame:
        return
    for _, row in frame.loc[frame["sport"].eq("ufc")].iterrows():
        proof = row_proof(row)
        ident = proof.get("bout_id", "")
        if not ident.isdigit() or str(row.get("red_fox_favorite")).lower() != "true":
            continue
        folder = cache_dir(data_dir) / "favorites"
        path = folder / f"{ident}.json"
        saved = read_json(path).get("captures", [])
        candidate = row.fillna("").astype(str).to_dict()
        if saved and row_proof(saved[-1]).get("source_at") == proof.get("source_at"):
            continue
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = folder / f".{uuid4().hex}.tmp"
        try:
            temporary.write_text(json.dumps({"captures": [*saved[-63:], candidate]}, separators=(",", ":")), encoding="utf-8")
            temporary.chmod(0o600)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def favorite_quotes(data_dir, bouts):
    captures = []
    for ident, bout in bouts.items():
        if bout["phase"] in {"live", "final", "void"}:
            path = cache_dir(data_dir) / "favorites" / f"{ident}.json"
            captures.extend(read_json(path).get("captures", []))
    return pd.DataFrame(captures)


def load_context(data_dir):
    root = cache_dir(data_dir)
    phases = read_json(root / 'bouts.json')
    notices = read_json(root / 'notices.json')
    return phases.get('bouts', {}), notices.get('bouts', {})


@lru_cache(maxsize=4096)
def fighter_key(value):
    normalized = str(value or '').lower().translate(str.maketrans('łøđ', 'lod'))
    token = re.sub('[^a-z0-9]', '', unicodedata.normalize('NFKD', normalized).encode('ascii', 'ignore').decode())
    return ALIASES.get(token, token)


@lru_cache(maxsize=4096)
def matchup(value):
    names = re.split(r'\s+(?:vs\.?|@)\s+', str(value or '').strip(), flags=re.I)
    return tuple(sorted(fighter_key(n) for n in names)) if len(names) == 2 else ()


def match_bout(game, schedule, bouts):
    pair, estimate = matchup(game), stamp(schedule)
    if len(pair) != 2 or not pair[0] or pair[0] == pair[1] or pd.isna(estimate):
        return None
    candidates = []
    for bout in bouts.values():
        if tuple(bout.get('pair', ())) != pair:
            continue
        announced = stamp(bout.get('card_at'))
        if pd.notna(announced) and abs(estimate-announced) <= pd.Timedelta(hours=30):
            candidates.append(bout)
    return candidates[0] if len(candidates) == 1 else None


def row_proof(row, field='favorite_supporting_evidence'):
    try:
        evidence = json.loads(str(row.get(field) or '{}'))
        value = evidence.get(PROOF, {}) if isinstance(evidence, dict) else {}
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def risks_for(context, notices, now):
    direct = set(context.get('risks', ())) & RISK_NAMES
    published_risks = notices.get(context['id'], {})
    announced = stamp(published_risks.get('announced_at'))
    if pd.notna(announced) and announced <= now:
        direct.update(set(published_risks.get('risks', ())) & RISK_NAMES)
    return direct


def filter_markets(frame, bouts, notices, now):
    """Return rows for display and exact proof for trusted UFC pairs."""
    proofs = {}
    drop = []
    ufc = frame.loc[frame['sport'].eq('ufc')]
    for (vendor, market), group in ufc.groupby(['game_id', 'market_display']):
        if market != 'MONEYLINE' or len(group) != 2:
            drop.extend(group.index)
            continue
        row = group.iloc[0]
        status = match_bout(row.get('game'), row.get('dk_start_iso'), bouts)
        if not status:
            drop.extend(group.index)
            continue
        if status['phase'] in {'live', 'final', 'void'}:
            drop.extend(group.index)
            continue
        seen = stamp(status.get('observed'))
        at = stamp(row.get('timestamp'))
        exact = (all(group['game'].map(matchup).map(lambda pair: pair == tuple(status['pair'])))
                 and set(group['side'].map(fighter_key)) == set(status['pair'])
                 and group['timestamp'].nunique() == 1)
        if (not exact or status['phase'] not in {'pre','suspended'} or pd.isna(seen) or pd.isna(at)
                or not pd.Timedelta(0) <= now-seen <= MAX_PHASE_AGE or at > seen):
            drop.extend(group.index)
            continue
        proofs[str(vendor)] = dict(bout_id=status['id'], card_id=status['card_id'], pair=status['pair'],
                                  athlete_ids=status['athlete_ids'], source_at=at.isoformat(), verified_pre_at=seen.isoformat(),
                                  version_at=now.isoformat(), phase=status['phase'], risks=sorted(risks_for(status, notices, now)),
                                  weigh_in=notices.get(status['id'], {}).get('weigh_in', 'unavailable'))
    return frame.drop(index=drop), proofs


def is_pregame(row, bouts, notices, now):
    bout = match_bout(row.get("game"), row.get("kickoff_iso"), bouts)
    if not bout or bout["phase"] != "pre":
        return False
    proof = row_proof(row)
    if proof and (proof.get("bout_id") != bout["id"] or proof.get("athlete_ids") != bout["athlete_ids"]
                  or risks_for(bout, notices, now)):
        return False
    observed = stamp(bout.get("observed"))
    return bool(pd.notna(observed) and pd.Timedelta(0) <= now - observed <= MAX_PHASE_AGE)


def scoped_history(history, proofs, bouts):
    """Enforce full opponent and card identity before v1 constructs a price path."""
    if not proofs or history.empty:
        return history
    keep = pd.Series(True, index=history.index)
    for vendor, proof in proofs.items():
        scope = history.loc[history.sport.eq('ufc') & history.game_id.astype(str).eq(vendor)]
        if scope.empty:
            continue
        pair = tuple(proof['pair'])
        timestamps = stamp(scope['timestamp'])
        same_pair = scope['game'].map(matchup).map(lambda observed: observed == pair)
        # If the same vendor key was used for another opponent, discard even
        # earlier quotes for the current pair before the most recent change.
        replaced = timestamps[~same_pair].max()
        card_time = stamp(bouts[proof['bout_id']]['card_at'])
        same_card = (stamp(scope['dk_start_iso']) - card_time).abs().le(pd.Timedelta(hours=30))
        after_replacement = timestamps.gt(replaced) if pd.notna(replaced) else True
        keep.loc[scope.index] = (same_pair & scope['side'].map(fighter_key).isin(pair) & same_card
                                 & timestamps.le(stamp(proof['source_at'])) & after_replacement)
    return history.loc[keep].copy()


def pre_row(proofs, row):
    return proofs.get(str(row.get('game_id'))) if row.get('sport') == 'ufc' else None


def allowed_pre(proof):
    return bool(proof and proof.get('phase') == 'pre' and not proof['risks'])


def annotate_board(board, proofs):
    if board.empty or not board.sport.eq('ufc').any():
        return board
    from red_fox_favorite import _sides

    board = board.copy()
    for index,row in board.loc[board.sport.eq('ufc')].iterrows():
        proof = pre_row(proofs,row) or {}
        board.at[index,'_ufc_integrity'] = 'verified' if allowed_pre(proof) else 'withheld'
        board.at[index,'_ufc_market_proof'] = json.dumps(proof,separators=(',',':'))
        if proof.get('risks'):
            sides = _sides(row)
            for side in sides:
                old = str(side.get('context_chips',''))
                side['context_chips'] = ' | '.join(filter(None,[old,*proof['risks']]))
            board.at[index,'market_sides'] = json.dumps(sides,separators=(',',':'))
    return board


def protected(row, since):
    if row.get('sport') != 'ufc':
        return False
    kickoff = stamp(row.get('kickoff_iso'))
    return bool(row_proof(row) or read_market_proof(row)) or row.get('freeze_method') in {HANDOFF, CONTEXT} or (
        pd.notna(since) and (pd.isna(kickoff) or kickoff >= since))


def unverified_rows(frame, since):
    if frame.empty or 'sport' not in frame:
        return frame
    ufc = frame.loc[frame['sport'].eq('ufc')]
    rejected = [index for index, row in ufc.iterrows() if protected(row, since)]
    return frame.drop(index=rejected)


def cutoff_bounds(bout):
    actual = stamp(bout.get('start'))
    if pd.notna(actual):
        return actual, actual
    return stamp(bout.get('last_pre')), stamp(bout.get('first_started'))


def score_payload(bout):
    phase = bout['phase']
    final = phase in {'final','void'}
    status = bout.get('result') or ('Bout in progress' if phase == 'live' else 'Result awaiting verification')
    return dict(score_state='post' if final else 'in', score_status=status,
                score_provider='espn_ufc_bout', score_provider_event_id=bout['id'],
                score_away='-', score_home='-', score_match_state='matched',
                score_updated_at_utc=bout.get('observed', ''))


def handoff_rows(candidates, tracking, bouts, notices, now, *, favorite=True):
    if candidates.empty or 'sport' not in candidates:
        return pd.DataFrame()
    rows = []
    for row in candidates.loc[candidates['sport'].eq('ufc')].to_dict('records'):
        proof = row_proof(row) if favorite else read_market_proof(row)
        bout = bouts.get(proof.get('bout_id'))
        if not bout or bout['phase'] not in {'live','final','void'} or proof.get('pair') != bout['pair']:
            continue
        if (bout['card_id'] != proof.get('card_id') or proof.get('athlete_ids') != bout['athlete_ids']
                or matchup(row.get('game')) != tuple(bout['pair'])):
            continue
        at, pre = stamp(proof.get('source_at')), stamp(proof.get('verified_pre_at'))
        lo, hi = cutoff_bounds(bout)
        if pd.isna(at) or pd.isna(pre) or at > pre:
            continue
        void = bout['phase']=='void'
        if not void and (pd.isna(lo) or pd.isna(hi) or at >= lo or (pre >= lo if lo == hi else pre > lo)):
            continue
        if favorite:
            if (not void and lo != hi) or not favorite_handoff_valid(row, proof, bout, tracking, notices, at, lo, hi, void):
                continue
        actual = stamp(bout.get('start'))
        kickoff = actual if pd.notna(actual) else hi if pd.notna(hi) else stamp(row.get('kickoff_iso'))
        if pd.isna(kickoff) or (not void and kickoff > now):
            continue
        row.update(score_payload(bout))
        if void and kickoff <= at:
            kickoff = pre + pd.Timedelta(microseconds=1)
        row.update(kickoff_iso=kickoff.isoformat(), state_as_of_utc=at.isoformat(),
                   final_pregame_state_at_utc=at.isoformat(), frozen_at_utc=now.isoformat(),
                   freeze_method=HANDOFF if favorite else CONTEXT)
        rows.append({k:v for k,v in row.items() if not k.startswith('_ufc_')})
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows).sort_values("final_pregame_state_at_utc", kind="stable")
            .drop_duplicates(["sport", "game_id", "market_display"], keep="last"))


def read_market_proof(row):
    try:
        value = json.loads(str(row.get('_ufc_market_proof') or '{}'))
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


def result_rows(ledger, bouts):
    rows = []
    for row in ledger.loc[ledger["freeze_method"].eq(HANDOFF)].to_dict("records"):
        bout = bouts.get(row["score_provider_event_id"])
        if (not bout or bout["phase"] not in {"final", "void"}
                or bout.get("kind") not in {"winner", "push", "void"}
                or matchup(row["game"]) != tuple(bout["pair"])
                or fighter_key(row["side"]) not in bout["pair"]):
            continue
        rows.append(dict(game_id=row["event_id"], score_provider_event_id=bout["id"],
                         score_provider="espn_ufc_bout", score_source=SCORE_SOURCE,
                         resolved_at_utc=bout.get("result_observed", bout["observed"]),
                         winner=bout["winner"], loser=bout["loser"], result=bout["result"], ufc_kind=bout["kind"]))
    return pd.DataFrame(rows)


def favorite_handoff_valid(row, proof, bout, tracking, notices, at, lo, hi, void):
    if str(row.get('red_fox_favorite')).lower() != 'true' or tracking.empty:
        return False
    if not void:
        if (str(row.get('favorite_late_invalidated')).lower() == 'true'
                or risks_for(bout, notices, lo) or pd.isna(stamp(proof.get('version_at')))
                or stamp(proof.get('version_at')) >= lo):
            return False
        freshness = 20 if row.get('favorite_whipsaw_state') in {'recovered','partial_retrace_intact'} else 30
        if hi-at > timedelta(minutes=freshness):
            return False
    records = tracking.loc[tracking['sport'].eq('ufc') & tracking['game_id'].astype(str).eq(str(row['game_id']))
                           & tracking['market_display'].eq('MONEYLINE')].sort_values('recorded_at',kind='stable')
    matches = records.loc[records.apply(lambda r: row_proof(r, 'supporting_evidence').get('bout_id') == proof['bout_id'],axis=1)] if not records.empty else records
    if matches.empty:
        return False
    if not void:
        proved_at = matches["supporting_evidence"].map(
            lambda evidence: stamp(row_proof({"evidence": evidence}, "evidence").get("version_at"))
        )
        matches = matches.loc[stamp(matches['recorded_at']).lt(lo) & stamp(proved_at).lt(lo)]
    if matches.empty:
        return False
    latest = matches.iloc[-1]
    entry = matches.loc[matches['favorite_state'].eq('qualified')]
    if entry.empty or (latest.get('favorite_state') != 'qualified' and not void):
        return False
    first = entry.iloc[0]
    first_at = stamp(first.get('entry_observed_at'))
    first_proof = row_proof(first, 'supporting_evidence')
    try:
        first_quote = json.loads(first['supporting_evidence']).get('current_line')
    except (ValueError, TypeError):
        return False
    if first_quote != first.get('first_qualified_line') or first_at != stamp(first_proof.get('source_at')):
        return False
    first_verified = stamp(first_proof.get('version_at'))
    return (fighter_key(first.get('favorite_side')) == fighter_key(row.get('favorite_side'))
            and bool(first.get('first_qualified_line')) and pd.notna(first_at) and first_at <= at
            and pd.notna(first_verified) and (void or first_verified < lo))
