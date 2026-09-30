"""Reconstruct the frozen Verified outcome matrix without running any agents.

Resolved-ID lists are official outcomes over the full 500-task denominator;
IDs absent from those lists are unresolved, including recorded execution failures.
Embedded per-instance records must cover all 500 tasks. Sources must reconcile
with the frozen public overall score before they enter the pool.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import urllib.request

import pandas as pd

from benchmark_progress.benchmarks import ROOT


def source_bytes(record, cache):
    if record.get('bundled'):
        data = gzip.decompress((ROOT / record['bundled']).read_bytes())
    else:
        path = cache / record['sha256']
        if not path.exists():
            cache.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(record['url'], timeout=120) as response:
                path.write_bytes(response.read())
        data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError(f"Source checksum mismatch: {record['url']}")
    return data


def resolved_ids(row, document, universe):
    if document is not None:
        if not isinstance(document.get('resolved'), list):
            raise ValueError('Expected an official resolved-ID list')
        ids = document['resolved']
        if len(ids) != len(set(ids)) or not set(ids) <= universe:
            raise ValueError('Duplicate or unknown resolved task IDs')
        return set(ids)
    details = row.get('per_instance_details')
    if not details or set(details) != universe:
        return None
    if any(type(v.get('resolved')) is not bool for v in details.values()):
        raise ValueError('Expected binary resolved fields')
    return {iid for iid, value in details.items() if value['resolved']}


def repository_groups(raw):
    """Outcome-independent grouping, fixed before examining attack outcomes."""
    counts = raw.groupby('repo').size().to_dict()
    return {repo: 'Other small repositories' if n < 10 else repo
            for repo, n in counts.items()}


def build(cache, output):
    if output.exists():
        raise FileExistsError(output)
    lock = json.loads((ROOT / 'data/extension_sources/swe_verified_source_lock.json').read_text())
    from io import BytesIO
    raw = pd.read_parquet(BytesIO(source_bytes(lock['dataset'], cache)))
    assert len(raw) == 500 and raw.instance_id.nunique() == 500
    html = source_bytes(lock['leaderboard'], cache).decode()
    match = re.search(r'<script[^>]+id="leaderboard-data"[^>]*>\s*', html)
    if not match:
        raise ValueError('Cannot locate the frozen official leaderboard data')
    boards, _ = json.JSONDecoder().raw_decode(html[match.end():])
    board = next(b for b in boards if b['name'] == 'Verified')
    universe = set(raw.instance_id)
    groups = repository_groups(raw)
    questions = pd.DataFrame([
        dict(question_id=r.instance_id,
             content_hash=hashlib.sha256(' '.join(r.problem_statement.split()).encode()).hexdigest(),
             category=groups[r.repo], task=groups[r.repo], source_repository=r.repo)
        for r in raw.itertuples()
    ]).sort_values('question_id').reset_index(drop=True)
    scores, audit = [], []
    for row in board['results']:
        rec = lock['results'].get(row['folder'])
        document = json.loads(source_bytes(rec, cache)) if rec else None
        ids = resolved_ids(row, document, universe)
        reconstructed = None if ids is None else len(ids) / 5
        included = ids is not None and abs(reconstructed - row['resolved']) <= .011
        model = row['name'] + ' / ' + row['folder']
        audit.append(dict(model=model, folder=row['folder'], included=included,
                          source='resolved_id_list' if rec else 'embedded_details',
                          published_percent=row['resolved'], reconstructed_percent=reconstructed,
                          reason=None if included else 'Missing/incomplete source' if ids is None
                          else 'Released outcomes do not match the published overall score'))
        if included:
            scores.extend(dict(question_id=iid, model=model, score=float(iid in ids))
                          for iid in sorted(universe))
    scores = pd.DataFrame(scores).sort_values(['question_id', 'model']).reset_index(drop=True)
    models = sorted(scores.model.unique())
    assert len(models) == 169 and len(scores) == 500 * 169
    assert not scores[['question_id', 'model']].duplicated().any()
    config = dict(
        benchmark='swe_verified',
        snapshot=dict(project='SWE-bench Verified', release='2026-09-21 frozen snapshot',
                      experiments_revision=lock['experiments_revision'],
                      dataset_revision=lock['dataset']['revision'],
                      release_url='https://www.swebench.com/'),
        categories=sorted(questions.category.unique()), models=models,
        trials=250, budgets=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        split_seed_base=202609210000, routing_seed='swe_verified-bank0-20260921',
        tie_seed='swe_verified-ties0-20260921', holdout_fraction=2/3, repeats=1,
        feedback_grouping=dict(rule='Pool repositories with fewer than 10 tasks',
                               repository_to_coordinate=groups))
    (output / 'data').mkdir(parents=True)
    (output / 'configs').mkdir()
    questions.to_parquet(output / 'data/swe_verified_questions.parquet', index=False)
    scores.to_parquet(output / 'data/swe_verified_scores.parquet', index=False)
    (output / 'data/swe_verified_sources.json').write_text(json.dumps(audit, indent=2) + '\n')
    (output / 'configs/swe_verified.json').write_text(json.dumps(config, indent=2) + '\n')
    print(f'Verified: {len(models)} models, {len(questions)} tasks, {len(config["categories"])} coordinates')
    print(questions.groupby('category').size().to_dict())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.cache, args.output)
