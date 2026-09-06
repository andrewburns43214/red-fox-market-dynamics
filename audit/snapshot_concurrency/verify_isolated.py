"""Reproduce the exact baseline function and verify storage on an immutable copy.

Run only against a copied --source and a new --output directory. No live writes,
ESPN requests, scoring changes, or publication operations occur in this script.
"""
import argparse
import ast
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import pandas as pd
import main
from snapshot_store import SCORE_FIELDS, append_snapshot_rows, merge_final_scores


def rows(game_id='original'):
    return [dict(timestamp='2026-09-01T00:00:00+00:00', sport='ncaaf', game_id=game_id,
                 game='UNLV @ Hawaii', side=side, market='splits', bets_pct='50', money_pct='50',
                 open_line=line, current_line=line, injury_news='', key_number_note='',
                 dk_start_iso=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat())
            for side, line in [('UNLV','UNLV @ -110'),('Hawaii','Hawaii @ -110')]]


def reproduce(root, baseline_source, fixed):
    root.mkdir()
    (root/'data').mkdir()
    path = root/'data/snapshots.csv'
    pd.DataFrame(rows()).to_csv(path,index=False)
    original_dir = Path.cwd()
    namespace = dict(vars(main))
    if fixed:
        source = Path(main.__file__).read_text(encoding='utf-8-sig')
    else:
        source = baseline_source
    tree = ast.parse(source.lstrip('\ufeff'))
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in ['append_snapshot','update_snapshots_with_espn_finals']:
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<isolated-writer>','exec'),namespace)
    namespace['SNAPSHOT_CSV'] = str(path)
    namespace['update_final_scores_history'] = lambda: None
    after_append = []
    def fake_espn(*args,**kwargs):
        namespace['append_snapshot'](rows('concurrent'),'ncaaf')
        after_append.append(len(pd.read_csv(path)))
        return {'UNLV @ Hawaii':(21,17)}
    namespace['get_espn_finals_map'] = fake_espn
    try:
        os.chdir(root)
        namespace['update_snapshots_with_espn_finals']()
    finally:
        os.chdir(original_dir)
    final = pd.read_csv(path,dtype=str,keep_default_na=False)
    return dict(rows_after_append=after_append[-1],rows_after_maintenance=len(final),game_ids=sorted(set(final.game_id)))


def semantic_digest(path, columns, limit=None):
    digest = hashlib.sha256()
    count = 0
    with path.open(newline='',encoding='utf-8') as file:
        for row in csv.DictReader(file):
            if limit is not None and count == limit:
                break
            # Length-framed JSON preserves field boundaries, Unicode and row order.
            encoded=json.dumps([row.get(c) or '' for c in columns],ensure_ascii=False).encode()
            digest.update(len(encoded).to_bytes(8,'big')); digest.update(encoded)
            count+=1
    return count,digest.hexdigest()


def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    source_hash = file_hash(args.source)
    result={'baseline_reproduction':reproduce(args.output/'before',args.baseline.read_text(encoding='utf-8-sig'),False),
            'fixed_reproduction':reproduce(args.output/'after',args.baseline.read_text(encoding='utf-8-sig'),True)}
    assert result['baseline_reproduction']['rows_after_maintenance']==2
    assert result['fixed_reproduction']['rows_after_maintenance']==4
    path=args.output/'scale-snapshots.csv'
    shutil.copy2(args.source,path)
    with path.open(newline='',encoding='utf-8') as file:
        columns=next(csv.reader(file))
    immutable=[c for c in columns if c not in SCORE_FIELDS]
    original_count,original_digest=semantic_digest(path,immutable)
    evaluated=pd.read_csv(path,nrows=2,dtype=str,keep_default_na=False)
    for col in SCORE_FIELDS:
        if col not in evaluated:
            evaluated[col]=''
    observed=evaluated[SCORE_FIELDS].copy()
    evaluated[SCORE_FIELDS]=['999','998']  # Temporary storage sentinels, never production scores.
    start=time.monotonic()
    append_snapshot_rows(path,rows('concurrent-scale'),main.SNAPSHOT_FIELDS)
    append_seconds=time.monotonic()-start
    # Compare every historical field, including prior finals, after append.
    assert semantic_digest(path,columns,original_count)==semantic_digest(args.source,columns)
    start=time.monotonic()
    applied=merge_final_scores(path,evaluated,observed)
    merge_seconds=time.monotonic()-start
    assert semantic_digest(path,immutable,original_count)==(original_count,original_digest)
    # Ensure finals for all other history rows stayed byte-value equivalent.
    before_reader=csv.DictReader(args.source.open(newline='',encoding='utf-8'))
    after_reader=csv.DictReader(path.open(newline='',encoding='utf-8'))
    with_keys={tuple(row[c] for c in immutable) for row in evaluated.to_dict('records')}
    checked=0
    for old,new in zip(before_reader,after_reader):
        if tuple(old.get(c) or '' for c in immutable) not in with_keys:
            assert all((old.get(c) or '')==(new.get(c) or '') for c in SCORE_FIELDS)
        checked+=1
    assert checked==original_count
    final_count,_=semantic_digest(path,immutable)
    assert final_count==original_count+2
    assert file_hash(args.source)==source_hash
    result['scale']=dict(source_rows=original_count,final_rows=final_count,finals_rows_applied=applied,
                         historical_non_score_digest=original_digest,source_sha256=source_hash,
                         original_source_unchanged=True,all_historical_rows_preserved=True,
                         unrelated_historical_finals_preserved=True,
                         append_seconds=append_seconds,merge_commit_seconds=merge_seconds)
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
