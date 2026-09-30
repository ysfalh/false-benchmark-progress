"""Check saved predictor summaries against their curves and benchmark results."""

import csv
import hashlib
import json
import math
from statistics import mean

from benchmark_progress.benchmarks import ROOT

RELEASE = ROOT/'results/predictor_v2'


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def test_predictor_summaries_and_external_thresholds():
    manifest = json.loads((RELEASE/'checksums.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256((RELEASE/name).read_bytes()).hexdigest() == expected
    models = json.loads((RELEASE/'models.json').read_text())
    training = rows(RELEASE/'livebench_thresholds.csv')
    curves = rows(RELEASE/'livebench_curves.csv')
    for outcome, metric in [('score_gap_plus2', 'score_gap_mean_pp'), ('false90', 'false_selection')]:
        values = []
        for row in training:
            matching = sorted((r for r in curves if r['n'] == row['n'] and r['d'] == row['d']),
                              key=lambda r: int(r['k']))
            target = float(row['baseline_pp']) + 2 if outcome == 'score_gap_plus2' else .9
            crossed = [int(r['k']) for r in matching if float(r[metric]) >= target]
            assert float(row[outcome+'_k']) == (min(crossed) if crossed else math.inf)
            if row[outcome+'_status'] == 'crossed':
                value = int(row['d']) * (float(row[outcome+'_k']) - 1)
                values.append(value/int(row['n']) if outcome == 'score_gap_plus2' else value)
        assert len(values) == models[outcome]['training_count']
        assert math.isclose(2**mean(math.log2(v) for v in values), models[outcome]['coefficient'], rel_tol=1e-12)

    canonical = rows(ROOT/'results/release_v2/budget_curves.csv')
    predictions = rows(RELEASE/'external_validation.csv')
    for row in predictions:
        score_gap = row['outcome'] == 'score_gap_plus2'
        coefficient = models[row['outcome']]['coefficient']
        predicted = 1 + coefficient * (int(row['n']) if score_gap else 1) / int(row['d'])
        assert math.isclose(predicted, float(row['predicted_k']), rel_tol=1e-12)
        metric = 'score_gap_mean_pp' if score_gap else 'first_s_not_t'
        matching = [r for r in canonical if r['benchmark'] == row['benchmark'] and r['policy'] == 'top5']
        observed = min((int(r['new_submissions']) for r in matching if float(r[metric]) >= float(row['target'])), default=math.inf)
        assert observed == float(row['observed_k'])
        if not math.isfinite(observed):
            assert row['status'] == 'right_censored'
            assert row['fold_discrepancy'] == ''
            assert int(row['max_tested_k']) == 2048
            continue
        ratio = predicted/observed
        for key, expected in [('predicted_over_observed', ratio), ('fold_discrepancy', max(ratio, 1/ratio)),
                              ('absolute_log2_error', abs(math.log2(ratio)))]:
            assert math.isclose(float(row[key]), expected, rel_tol=1e-12)
    for point in rows(RELEASE/'figure_points.csv'):
        model = models[point['panel']]
        scale = int(point['n']) if point['panel'] == 'score_gap_plus2' else 1
        assert math.isclose(float(point['predicted_k']), 1 + model['coefficient']*scale/int(point['d']), rel_tol=1e-12)
