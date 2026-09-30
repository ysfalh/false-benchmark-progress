"""Descriptive held-out severity of the released false-winner events."""

import numpy as np
import pandas as pd

THRESHOLDS = ('any', '1', '2', '5')
RESAMPLES = 2000
SEED = 2026081900


def summarize(raw, *, trials=250, resamples=RESAMPLES, seed=SEED):
    """Joint event rates over all trials; deficit means conditional on false wins.

    Preserve the rank-based event, including HELM Lite's distinct ranking metric.
    In particular, 'any' is not equivalent to a nonnegative score deficit.
    """
    indices = np.random.default_rng(seed).integers(0, trials, (resamples, trials))
    rates, means = [], []
    for (benchmark, budget), frame in raw[raw.policy == 'top5'].groupby(['benchmark', 'new_submissions']):
        frame = frame.sort_values('trial')
        if frame.trial.tolist() != list(range(trials)):
            raise ValueError(f'{benchmark}, budget {budget}: expected one row per trial')
        false = frame.first_s_not_t.to_numpy(bool)
        deficit = frame.heldout_regret_pp.to_numpy(float)
        if not np.isfinite(deficit).all():
            raise ValueError('Held-out deficits must be finite')
        common = dict(benchmark=benchmark, policy='top5', new_submissions=int(budget), trials=trials)
        for threshold in THRESHOLDS:
            event = false if threshold == 'any' else false & (deficit >= float(threshold))
            low, high = np.quantile(event[indices].mean(axis=1), [.025, .975])
            rates.append(dict(common, threshold_pp=threshold, count=int(event.sum()),
                              rate=float(event.mean()), ci_low=float(low), ci_high=float(high)))
        counts = false[indices].sum(axis=1)
        valid = counts > 0
        boot = (deficit[indices] * false[indices]).sum(axis=1)[valid] / counts[valid]
        low, high = np.quantile(boot, [.025, .975]) if len(boot) else (np.nan, np.nan)
        means.append(dict(common, false_winners=int(false.sum()),
                          mean_deficit_pp=float(deficit[false].mean()) if false.any() else np.nan,
                          ci_low=float(low), ci_high=float(high), bootstrap_valid_resamples=int(valid.sum())))
    return pd.DataFrame(rates), pd.DataFrame(means)
