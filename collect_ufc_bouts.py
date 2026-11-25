"""Collect independent UFC status and weigh-in evidence outside board jobs."""

from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import os
import re
from urllib.parse import urlsplit
from uuid import uuid4
from xml.etree import ElementTree

from bs4 import BeautifulSoup
import pandas as pd
import requests

from ufc_bout_integrity import cache_dir, fighter_key, matchup, read_json, stamp

ESPN = 'https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard'
PLAYS = 'https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/events'
SITEMAP = 'https://www.ufc.com/sitemap.xml'


def get(url, **kwargs):
    response = requests.get(url, timeout=(3, 6), **kwargs)
    response.raise_for_status()
    return response


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(',',':')),encoding='utf-8')
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def outcome(status, competition, names):
    explicit = ' '.join(str(status.get(key, '')) for key in ('name','description','detail','shortDetail'))
    explicit += ' ' + ' '.join(str(p.get('type', {}).get('text', '')) for p in competition.get('details', []))
    explicit = explicit.lower().replace('_',' ')
    if any(value in explicit for value in ('no contest', 'no decision','cancelled','canceled')):
        return 'void', '', '', 'Bout void'
    if 'draw' in explicit:
        return 'push', '', '', 'Draw'
    participants = competition['competitors']
    winners = [name for name, participant in zip(names, participants) if participant.get('winner') is True]
    if len(winners)==1:
        winner = winners[0]
        loser = next(n for n in names if n != winner)
        return 'winner', winner, loser, f'{winner} defeated {loser}'
    return '', '', '', ''


def build_bouts(events, old, observed):
    valid = dict(old)
    current_by_card = {}
    ts = observed.isoformat()
    for event in events:
        if not event.get('id') or re.search(r'contender series|road to ufc',str(event.get('name')),re.I):
            continue
        card = str(event['id'])
        if pd.isna(stamp(event.get('date'))):
            continue
        seen = set()
        card_status = str(event.get('status', {}).get('type', {}).get('name', '')).upper()
        for competition in event.get('competitions',[]):
            participants = competition.get('competitors',[])
            ident = str(competition.get('id') or '')
            if len(participants) != 2 or not ident.isdigit():
                continue
            ids = [str(p.get('id') or '') for p in participants]
            names = [str(p.get('athlete',{}).get('displayName') or '') for p in participants]
            pair = sorted(map(fighter_key,names))
            if not all(ids) or len(set(ids))!=2 or not all(pair) or len(set(pair))!=2:
                continue
            status = competition.get('status',{}).get('type',{})
            label = str(status.get('name','')).upper()
            phase = {'pre':'pre','in':'live','post':'final'}.get(status.get('state'),'unknown')
            risks = []
            kind=winner=loser=result=''
            if 'CANCEL' in label or ('CANCEL' in card_status and phase == 'pre'):
                phase = 'void'
                risks.append('Bout Suspended')
                kind, result='void','Bout cancelled'
            elif (any(value in label for value in ('SUSPEND', 'POSTPON'))
                  or (phase == 'pre' and any(value in card_status for value in ('SUSPEND', 'POSTPON')))):
                phase = 'suspended'
                risks.append('Bout Suspended')
            elif phase=='final':
                kind,winner,loser,result=outcome(status,competition,names)
            previous = old.get(ident,{})
            if previous and (previous.get('card_id')!=card or previous.get('pair')!=pair or previous.get('athlete_ids')!=sorted(ids)):
                # Reused provider identity is an integrity exception. Keep old
                # attribution immutable and make the active record unavailable.
                valid[ident] = {**previous,'phase':'unknown','risks':['Opponent Change'],'observed':ts}
                continue
            if ((previous.get('phase') == 'final' and phase != 'final')
                    or (previous.get('phase') in {'live', 'void'} and phase in {'pre', 'unknown', 'suspended'})):
                continue
            if previous.get('phase') == 'void' and 'Opponent Change' in previous.get('risks', ()):
                continue
            if previous.get('phase') == 'final' and phase == 'final' and previous.get('kind'):
                kind, winner, loser, result = (previous[k] for k in ('kind', 'winner', 'loser', 'result'))
            seen.add(ident)
            valid[ident] = dict(previous,id=ident,card_id=card,card_at=event['date'],card=event.get('name',''),
                                segment_at=competition.get('date', ''), pair=pair, names=names,
                                athlete_ids=sorted(ids), phase=phase, observed=ts,
                                status=label,risks=risks,kind=kind,winner=winner,loser=loser,result=result)
            item = valid[ident]
            if phase=='pre':
                item['last_pre']=ts
            elif phase in {'live','final'}:
                item.setdefault('first_started',ts)
            if kind:
                item.setdefault('result_observed', ts)
        current_by_card[card] = seen
    for ident, previous in old.items():
        card = previous.get('card_id')
        if previous.get('phase') not in {'pre','suspended','unknown'} or card not in current_by_card or ident in current_by_card[card]:
            continue
        roster = [valid[b] for b in current_by_card[card] if valid[b]['phase'] in {'pre','live','final'}]
        if any(set(previous['athlete_ids']) & set(b['athlete_ids']) and previous['pair']!=b['pair'] for b in roster):
            valid[ident] = {**previous,'phase':'void','risks':['Opponent Change'],'observed':ts,
                            'kind':'void','result':'Original opponent replaced'}
        else:
            valid[ident] = {**previous,'phase':'unknown'}
    return valid


def with_start(bout):
    url = f"{PLAYS}/{bout['card_id']}/competitions/{bout['id']}/plays"
    try:
        plays = get(url,params={'limit':100}).json()['items']
        clocks = {stamp(p.get('wallclock')) for p in plays
                  if p.get('type',{}).get('text')=='Round Start' and p.get('period',{}).get('number')==1}
        clocks = {t for t in clocks if pd.notna(t)}
        if len(clocks)==1:
            start = clocks.pop()
            if abs(start-stamp(bout['card_at'])) <= pd.Timedelta(hours=18) and start <= stamp(bout['observed']):
                return dict(bout,start=start.isoformat(),start_source=url)
    except (requests.RequestException, KeyError, ValueError, TypeError):
        pass
    return bout


def official_card_results(bouts):
    card = bouts[0]
    numbered = re.match(r'UFC (\d+)\b', card['card'])
    date = stamp(card['card_at']).tz_convert(ZoneInfo('America/New_York'))
    suffix = f'ufc-{numbered[1]}' if numbered else f'ufc-fight-night-{date:%B-%d-%Y}'.lower()
    url = f'https://www.ufc.com/event/{suffix}'
    try:
        soup = BeautifulSoup(get(url).content, 'html.parser')
    except requests.RequestException:
        return {}
    pairs = {}
    for node in soup.select('.c-listing-fight'):
        names = [node.select_one(f'.c-listing-fight__corner-name--{corner}') for corner in ('red', 'blue')]
        if not all(names):
            continue
        pair = tuple(sorted(fighter_key(name.get_text(' ', strip=True)) for name in names))
        outcomes = tuple(item.get_text(' ', strip=True).casefold()
                         for item in node.select('.c-listing-fight__outcome-wrapper')[:2])
        pairs.setdefault(pair, []).append(outcomes)
    results = {}
    for bout in bouts:
        matched = pairs.get(tuple(bout['pair']), [])
        if matched == [('draw', 'draw')]:
            results[bout['id']] = dict(kind='push', result='Draw', result_source=url)
        elif len(matched) == 1 and len(matched[0]) == 2 and set(matched[0]) <= {'nc', 'no contest', 'no decision'}:
            results[bout['id']] = dict(kind='void', result='No contest', result_source=url)
    return results


def sitemap_items(url):
    raw = ElementTree.fromstring(get(url).content)
    return [(item.findtext('{*}loc'), item.findtext('{*}lastmod')) for item in raw if item.findtext('{*}loc')]


def notices_from_article(url, body, bouts, observed):
    soup = BeautifulSoup(body,'html.parser')
    time = soup.select_one('meta[property="article:published_time"]')
    announced = stamp(time.get('content') if time else None)
    if pd.isna(announced) or announced > observed:
        return {}
    sections = soup.select('main p, main h4, main li') or soup.select('article p, article h4, article li')
    text = [' '.join(segment.stripped_strings) for segment in sections]
    weights = {}
    for paragraph in text:
        m = re.search(r'([^:]+?)\s*\((\d+(?:\.\d+)?)\)[*\s]*\s+vs\s+([^()]+?)\s*\((\d+(?:\.\d+)?)\)',paragraph,re.I)
        if m:
            weights[matchup(f'{m[1].split(" - ")[-1]} vs {m[3]}')] = paragraph
    cards = {}
    for bout in bouts.values():
        if abs(stamp(bout['card_at'])-announced) <= pd.Timedelta(days=3):
            cards.setdefault(bout['card_id'],[]).append(bout)
    matching = [entries for entries in cards.values() if sum(tuple(b['pair']) in weights for b in entries)>=2]
    if len(matching)!=1:
        return {}
    output = {}
    all_names = [name for b in matching[0] for name in b['names']]
    for bout in matching[0]:
        pair = tuple(bout['pair'])
        if pair not in weights:
            continue
        risks = set()
        for paragraph in text:
            case = paragraph.casefold()
            if not re.search(r'missed weight|weighed in above|failed to make weight|forfeits?.*purse|changed to a catchweight',case):
                continue
            words = set(re.findall(r'\b[^\W_]+\b',case))
            for name in bout['names']:
                surname = name.split()[-1].casefold()
                if name.casefold() in case or (surname in words and sum(n.split()[-1].casefold()==surname for n in all_names)==1):
                    risks.add('Weigh-In Risk' if 'changed to a catchweight' in case else 'Weight Miss')
        if re.search(r'\bcatchweight\b',weights[pair],re.I) and '*' in weights[pair]:
            risks.add('Weigh-In Risk')
        output[bout['id']] = dict(weigh_in='confirmed',risks=sorted(risks),source=url,
                                 announced_at=announced.isoformat(),first_observed=observed.isoformat())
    return output


def notice_cycle(root, bouts, observed):
    old = read_json(root/'notices.json')
    previous = stamp(old.get('polled_at'))
    if pd.notna(previous) and observed-previous < pd.Timedelta(minutes=20):
        return
    if not any(abs(stamp(b['card_at'])-observed) <= pd.Timedelta(days=3) for b in bouts.values()):
        return
    pages = sitemap_items(SITEMAP)
    checked = old.get('articles',{})
    latest = dict(old.get('bouts',{}))
    with ThreadPoolExecutor(max_workers=3) as pool:
        for items in pool.map(sitemap_items,[url for url,_ in pages[-3:]]):
            for url, version in items:
                target = urlsplit(url)
                if (target.scheme != 'https' or target.hostname != 'www.ufc.com'
                        or not re.search(r'^/news/[^/]*(weigh-in|weigh-results|weigh-ins)',target.path,re.I)):
                    continue
                changed = stamp(version)
                if pd.isna(changed) or abs(observed-changed)>pd.Timedelta(days=4) or checked.get(url)==version:
                    continue
                body = get(url).text
                for ident, result in notices_from_article(url,body,bouts,observed).items():
                    if ident in latest:
                        result['first_observed'] = latest[ident]['first_observed']
                        result['risks'] = sorted(set(result['risks']) | set(latest[ident]['risks']))
                    latest[ident] = result
                checked[url] = version
    atomic(root/'notices.json',dict(polled_at=observed.isoformat(),bouts=latest,articles=checked))


def run(data_dir=Path('data'), observed=None):
    now = observed or datetime.now(timezone.utc)
    now = pd.Timestamp(now)
    root = cache_dir(data_dir)
    old = read_json(root/'bouts.json').get('bouts',{})
    dates = f"{now-timedelta(days=2):%Y%m%d}-{now+timedelta(days=45):%Y%m%d}"
    source = get(ESPN,params={'dates':dates,'limit':100}).json()['events']
    now = pd.Timestamp.now(tz='UTC') if observed is None else now
    bouts = build_bouts(source,old,now)
    starts = [b for b in bouts.values() if b['phase'] in {'live','final'} and not b.get('start')
              and now-stamp(b['card_at']) < pd.Timedelta(days=3)]
    starts.sort(key=lambda b: (b['phase'] != 'live', b.get('last_start_lookup', '')))
    atomic(root/'bouts.json',dict(observed=now.isoformat(),bouts=bouts))
    if starts:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for bout in pool.map(with_start,starts[:12]):
                bouts[bout['id']] = dict(bout, last_start_lookup=now.isoformat())
        atomic(root/'bouts.json',dict(observed=now.isoformat(),bouts=bouts))
    ambiguous = {}
    for bout in bouts.values():
        if bout['phase'] == 'final' and not bout['kind'] and now - stamp(bout['card_at']) < pd.Timedelta(days=3):
            last = stamp(bout.get('last_result_lookup'))
            if pd.isna(last) or now - last >= pd.Timedelta(minutes=10):
                ambiguous.setdefault(bout['card_id'], []).append(bout)
    if ambiguous:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for matches in pool.map(official_card_results, ambiguous.values()):
                for ident, result in matches.items():
                    bouts[ident].update(result, result_observed=now.isoformat())
        for group in ambiguous.values():
            for bout in group:
                bout['last_result_lookup'] = now.isoformat()
        atomic(root/'bouts.json',dict(observed=now.isoformat(),bouts=bouts))
    try:
        notice_cycle(root,bouts,now)
    except (requests.RequestException, ElementTree.ParseError, ValueError, TypeError) as error:
        print(f'[ufc-integrity] official weigh-in source unavailable: {type(error).__name__}',flush=True)
    print(f'[ufc-integrity] {len(bouts)} matched provider bouts, {len(source)} cards',flush=True)


if __name__=='__main__':
    run()
