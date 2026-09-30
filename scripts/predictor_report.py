"""Fit threshold predictors and evaluate them on the benchmark snapshots."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_progress.benchmarks import BUDGETS, ROOT, load, partition

RELEASE = ROOT/'results/predictor_v2'


def fit(curves):
    rows = []
    for (n, d), group in curves.groupby(['n', 'd']):
        baseline = float(group.baseline_pp.iloc[0])
        row = dict(n=n, d=d, baseline_pp=baseline)
        for outcome, metric, target in [('score_gap_plus2', 'score_gap_mean_pp', baseline+2),
                                        ('false90', 'false_selection', .9)]:
            hits = group.loc[group[metric] >= target, 'k']
            k = int(hits.min()) if len(hits) else math.inf
            row.update({outcome+'_k': k, outcome+'_status':
                        'static' if k == 1 else 'crossed' if math.isfinite(k) else 'right_censored'})
        rows.append(row)
    thresholds = pd.DataFrame(rows)
    models, points = {}, []
    for outcome in ['score_gap_plus2', 'false90']:
        group = thresholds[thresholds[outcome+'_status'] == 'crossed']
        scales = group.n if outcome == 'score_gap_plus2' else 1
        coefficient = float(np.exp(np.log(group.d*(group[outcome+'_k']-1)/scales).mean()))
        models[outcome] = dict(coefficient=coefficient, training_count=len(group),
                               formula='k = 1 + c*n/d' if outcome == 'score_gap_plus2' else 'k = 1 + c/d')
        for row in group.to_dict('records'):
            points.append(dict(panel=outcome, source='LiveBench fit cell', n=row['n'], d=row['d'],
                observed_k=row[outcome+'_k'], predicted_k=1+coefficient*(row['n'] if outcome == 'score_gap_plus2' else 1)/row['d'], status='crossed'))
    return thresholds, models, points


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    training = pd.read_csv(RELEASE/'livebench_curves.csv')
    training.to_csv(output/'livebench_curves.csv', index=False)
    thresholds, models, points = fit(training)
    thresholds.to_csv(output/'livebench_thresholds.csv', index=False)
    (output/'models.json').write_text(json.dumps(models, indent=2)+'\n')
    curves = pd.read_csv(ROOT/'results/release_v2/budget_curves.csv')
    names = {'helm_capabilities':'HELM Capabilities', 'helm_lite':'HELM Lite',
             'openllm_v2':'Open LLM', 'swe_verified':'SWE-bench Verified'}
    records = []
    for benchmark, name in names.items():
        cfg, questions, _ = load(benchmark)
        split = partition
        if benchmark in {'openllm_v2', 'swe_verified'}:
            from benchmark_progress.extensions import partition as split
        reused, _ = split(cfg, questions, 0)
        n, d = len(reused), len(cfg['categories']) + (benchmark == 'openllm_v2')
        group = curves[(curves.benchmark == benchmark)&(curves.policy == 'top5')]
        baseline = float(group.preload_mean_pp.iloc[0])
        for outcome, metric, target in [('score_gap_plus2', 'score_gap_mean_pp', baseline+2),
                                        ('false90', 'first_s_not_t', .9)]:
            coefficient = models[outcome]['coefficient']
            predicted = 1+coefficient*(n if outcome == 'score_gap_plus2' else 1)/d
            hits = group.loc[group[metric] >= target, 'new_submissions']
            observed = int(hits.min()) if len(hits) else math.inf
            status = 'static' if observed == 1 else 'crossed' if math.isfinite(observed) else 'right_censored'
            ratio = predicted/observed if math.isfinite(observed) else math.nan
            records.append(dict(benchmark=benchmark, benchmark_name=name, n=n, d=d,
                outcome=outcome, genuine_score_gap_baseline_pp=baseline, target=target,
                coefficient=coefficient, predicted_k=predicted, observed_k=observed, status=status,
                max_tested_k=max(BUDGETS), predicted_over_observed=ratio,
                absolute_log2_error=abs(math.log2(ratio)) if math.isfinite(ratio) else math.nan,
                fold_discrepancy=max(ratio, 1/ratio) if math.isfinite(ratio) else math.nan))
            points.append(dict(panel=outcome, source=name, n=n, d=d,
                               observed_k=observed, predicted_k=predicted, status=status))
    pd.DataFrame(records).to_csv(output/'external_validation.csv', index=False)
    pd.DataFrame(points).to_csv(output/'figure_points.csv', index=False)
    checksums = {'training_source':'livebench_curves.csv', 'evaluation_source':'results/release_v2',
        'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())}}
    (output/'checksums.json').write_text(json.dumps(checksums, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
