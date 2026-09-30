"""Check paired average differences, tail direction, and displayed crossings."""

import hashlib
import json
import numpy as np
import pandas as pd
import pytest

from benchmark_progress.benchmarks import BENCHMARKS, BUDGETS, ROOT
from scripts.score_comparison import ALPHA, COMPARISONS, basic_lower, compare_scores
from benchmark_progress.reporting import bootstrap_stat


def test_score_summary_averages_all_trials():
    # A rare large gap must contribute to the average rather than disappear.
    values = np.zeros(250)
    values[0] = 100
    indices = np.arange(250)[None, :]
    point, low, high, draws = bootstrap_stat(values, indices, 'mean', .025)
    assert point == low == high == .4
    np.testing.assert_array_equal(draws, [.4])


def test_basic_bound_uses_upper_tail():
    draws = np.array([1., 2., 3., 4.])
    assert basic_lower(3., draws, .25) == 2.
    assert basic_lower(3., draws, .5) == 3.
    assert basic_lower(3., draws, ALPHA/COMPARISONS) <= basic_lower(3., draws, ALPHA)


def test_pairing_recovers_a_constant_shift():
    records = [{'benchmark': b, 'policy': 'top5', 'trial': t, 'new_submissions': k,
                'genuine_preload_score_gap_pp': t/4,
                'score_gap_pp': t/4 + (2 if k >= 4 else 0)}
               for b in BENCHMARKS for k in BUDGETS for t in range(250)]
    raw = pd.DataFrame(records).sample(frac=1, random_state=3)
    result = compare_scores(raw, resamples=100)
    for row in result.itertuples():
        shift = 2. if row.new_submissions >= 4 else 0.
        assert row.increase_mean_pp == row.increase_ci_low == row.increase_ci_high == shift
        assert row.significant_increase == (row.new_submissions >= 4)
        if row.new_submissions > 1:
            assert row.adjusted_lower_pp == shift
    raw.loc[raw.index[0], 'genuine_preload_score_gap_pp'] += 1
    with pytest.raises(AssertionError):
        compare_scores(raw, resamples=100)


def test_published_comparison_matches_release_points():
    comparison = pd.read_csv(ROOT/'site/data/score_comparison.csv')
    curves = pd.read_csv(ROOT/'results/release_v2/budget_curves.csv')
    expected = curves[curves.policy == 'top5'].set_index(['benchmark','new_submissions'])
    actual = comparison.set_index(['benchmark','new_submissions'])
    np.testing.assert_allclose(actual.increase_mean_pp,
        expected.loc[actual.index, 'score_gap_above_preload_pp'], rtol=0, atol=1e-10)
    assert len(actual) == len(BENCHMARKS)*len(BUDGETS)
    ledger = json.loads((ROOT/'site/data/provenance.json').read_text())
    method = json.loads((ROOT/'site/data/score_comparison_method.json').read_text())
    assert method['input_sha256'] == hashlib.sha256((ROOT/'results/release_v2/trials.parquet').read_bytes()).hexdigest()
    assert method['analysis_sha256'] == hashlib.sha256((ROOT/'scripts/score_comparison.py').read_bytes()).hexdigest()
    assert method['comparisons'] == COMPARISONS == 55
    assert method['family_alpha'] == ALPHA == .05
    for benchmark in BENCHMARKS:
        rows = comparison[comparison.benchmark == benchmark]
        assert np.array_equal(rows.significant_increase,
                              (rows.new_submissions > 1)&(rows.adjusted_lower_pp > 1e-10))
        hits = rows[rows.significant_increase].new_submissions
        first = int(hits.min()) if len(hits) else None
        assert ledger['facts']['significant_budget_'+benchmark]['value'] == first
