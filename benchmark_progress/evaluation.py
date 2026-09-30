"""Best-score generalization gap and final-model selection, evaluated after fitting."""

import numpy as np
from .attacks import aggregate
from .benchmarks import BUDGETS


def compact_trace(ps, pt, fs, ft, genuine_count):
    """Keep sufficient score histories and profiles to audit every reported metric."""
    return {'best_loss_s': np.minimum.accumulate((1-ps).mean(axis=1)),
            'best_loss_t': np.minimum.accumulate((1-pt).mean(axis=1)),
            'genuine_s': ps[:genuine_count], 'genuine_t': pt[:genuine_count],
            'final_s': fs, 'final_t': ft}


def outcomes(benchmark, policy, trial, trace, budgets=BUDGETS):
    s, t = trace['best_loss_s'], trace['best_loss_t']
    n = len(trace['genuine_s'])
    rows = []
    for j, k in enumerate(budgets):
        end = n + 2 + k - 1
        fs, ft = trace['final_s'][j], trace['final_t'][j]
        pre = float(np.max(np.abs(s[:end]-t[:end])))
        error = max(pre, abs(min(s[end-1], (1-fs).mean()) - min(t[end-1], (1-ft).mean())))
        ss = aggregate(np.vstack((trace['genuine_s'], fs)), benchmark)
        ts = aggregate(np.vstack((trace['genuine_t'], ft)), benchmark)
        rs = 1 + int(np.sum(np.round(1-ss[:-1], 10) < np.round(1-ss[-1], 10)))
        rt = 1 + int(np.sum(np.round(1-ts[:-1], 10) < np.round(1-ts[-1], 10)))
        rows.append({'benchmark': benchmark, 'policy': policy, 'trial': trial,
            'method': 'task_router' if k == 1 else 'random_task_linear',
            'new_submissions': k, 'original_count_equivalent': k+2,
            'score_gap_pp': 100*error,
            'genuine_preload_score_gap_pp': 100*float(np.max(np.abs(s[:n]-t[:n]))),
            'first_s': rs == 1, 'first_t': rt == 1, 'first_s_not_t': rs == 1 and rt > 1,
            'rank_s': rs, 'rank_t': rt,
            'heldout_regret_pp': 100*(trace['genuine_t'].mean(axis=1).max()-ft.mean()),
            'ranking_regret_pp': 100*(ts[:-1].max()-ts[-1])})
    return rows
