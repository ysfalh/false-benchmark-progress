"""Paired score-gap comparisons derived from release_v2 trial records."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_progress.benchmarks import BENCHMARKS, BUDGETS, ROOT

RESAMPLES = 100000
SEED = 2026081900
ALPHA = .05
COMPARISONS = len(BENCHMARKS) * (len(BUDGETS)-1)


def basic_lower(point, resampled, alpha):
    """One-sided reverse-percentile bootstrap bound, in percentage points."""
    return 2*point - float(np.quantile(resampled, 1-alpha, method='higher'))


def compare_scores(raw, resamples=RESAMPLES):
    rows = []
    for benchmark in BENCHMARKS:
        frame = raw[(raw.benchmark == benchmark) & (raw.policy == 'top5')]
        attack = frame.pivot(index='new_submissions', columns='trial', values='score_gap_pp')
        baseline = frame.pivot(index='new_submissions', columns='trial', values='genuine_preload_score_gap_pp')
        assert attack.index.tolist() == list(BUDGETS)
        assert attack.columns.tolist() == list(range(250))
        assert not attack.isna().any().any()
        np.testing.assert_array_equal(baseline.to_numpy(), np.broadcast_to(baseline.iloc[0], baseline.shape))
        a, b = attack.to_numpy(), baseline.iloc[0].to_numpy()
        differences = a - b
        point = differences.mean(axis=1)
        boot = np.empty((len(BUDGETS), resamples))
        rng = np.random.default_rng(SEED)
        # Keep trial pairs together and use common resamples for every budget.
        for start in range(0, resamples, 2000):
            stop = min(start+2000, resamples)
            indices = rng.integers(0, len(b), (stop-start, len(b)))
            for j in range(len(BUDGETS)):
                boot[j, start:stop] = differences[j, indices].mean(axis=1)
        for j, budget in enumerate(BUDGETS):
            low = basic_lower(point[j], boot[j], ALPHA/2)
            high = 2*point[j]-float(np.quantile(boot[j], ALPHA/2, method='lower'))
            adjusted = basic_lower(point[j], boot[j], ALPHA/COMPARISONS) if budget > 1 else np.nan
            rows.append({'benchmark': benchmark, 'policy': 'top5', 'new_submissions': budget,
                'increase_mean_pp': point[j], 'increase_ci_low': low, 'increase_ci_high': high,
                'adjusted_lower_pp': adjusted,
                'significant_increase': bool(budget > 1 and adjusted > 1e-10)})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare_scores(pd.read_parquet(ROOT/'results/release_v2/trials.parquet'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result[result.significant_increase].groupby('benchmark').first()[['new_submissions','increase_mean_pp','adjusted_lower_pp']].to_string())
