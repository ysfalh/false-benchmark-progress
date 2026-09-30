"""Paired model-selection loss from archived evaluation records; no new trials."""

import hashlib
import numpy as np
import pandas as pd
from benchmark_progress.attacks import aggregate
from benchmark_progress.benchmarks import ROOT, BENCHMARKS
from scripts.no_attack import choose_public_winner

NAMES = tuple(BENCHMARKS)
BUDGETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
RELEASE = ROOT / 'results/release_v2'

def scalar_ranking(profiles, benchmark):
    """Independent scalar implementation of the original ranking rule."""
    n, d = profiles.shape
    if benchmark != 'helm_lite':
        return [sum(map(float, row)) / d for row in profiles]
    return [sum(sum(float(profiles[i, c] > profiles[j, c]) +
                    .5 * float(profiles[i, c] == profiles[j, c])
                    for j in range(n) if j != i) / (n - 1)
                for c in range(d)) / d for i in range(n)]


def source_paths():
    return [RELEASE / b / f for b in NAMES for f in
            ['evaluation.npz', 'public_profiles.npz']] + [
                RELEASE / 'budget_curves.csv', RELEASE / 'trials.parquet',
                ROOT / 'results/no_attack_v1/trials.csv']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calculate(policy='top5', benchmarks=NAMES, budgets=tuple(BUDGETS[1:])):
    reference = pd.read_csv(ROOT / 'results/no_attack_v1/trials.csv').set_index(['benchmark', 'trial'])
    historical = pd.read_parquet(RELEASE / 'trials.parquet')
    historical = historical[historical.policy == policy].set_index(['benchmark', 'trial', 'new_submissions'])
    rows = []
    for benchmark in benchmarks:
        with np.load(RELEASE / benchmark / 'evaluation.npz') as archive, np.load(RELEASE / benchmark / 'public_profiles.npz') as public_archive:
            # NpzFile does not cache reads: decompress each needed array once,
            # rather than once per trial or submission budget.
            z = {key: archive[key] for key in
                 ('policies', 'trials', 'genuine_s', 'genuine_t', 'final_s', 'final_t')}
            p = {key: public_archive[key] for key in ('models', 'trials', 'overall')}
            models = list(map(str, p['models']))
            assert models == sorted(set(models))
            for pos in np.flatnonzero(z['policies'] == policy):
                trial = int(z['trials'][pos])
                public_pos = int(np.flatnonzero(p['trials'] == trial)[0])
                gs, gt = z['genuine_s'][pos], z['genuine_t'][pos]
                original = choose_public_winner(models, p['overall'][public_pos])
                public_loss = np.round(1 - aggregate(gs, benchmark), 10)
                assert original == int(np.argmin(public_loss))
                old = reference.loc[(benchmark, trial)]
                assert original == int(old.selected_index)
                best_t = float(gt.mean(axis=1).max())
                before_t = float(gt[original].mean())
                before = 100 * (best_t - before_t)
                assert abs(before - old.heldout_regret_pp) < 1e-10
                rows.append(dict(benchmark=benchmark, trial=trial, new_submissions=0,
                    no_attack_model=models[original], selected_model=models[original],
                    selected_router=False, router_tied_public_first=False,
                    reused_score_pp=100 * float(gs[original].mean()),
                    heldout_score_pp=100 * before_t, no_attack_shortfall_pp=before,
                    selected_shortfall_pp=before, model_selection_loss_pp=0., rank_drop=0,
                    router_false_winner=False))
                for j, budget in enumerate(BUDGETS):
                    if budget not in budgets:  # Budget 1, when requested, is the static control.
                        continue
                    ss = np.vstack([gs, z['final_s'][pos, j]])
                    tt = np.vstack([gt, z['final_t'][pos, j]])
                    ranks = aggregate(ss, benchmark)
                    rounded = np.round(1 - ranks, 10)
                    # Genuine models precede the final router: ties favor existing models,
                    # then the same lexical order used before attack.
                    chosen = int(np.argmin(rounded))
                    check_scores = scalar_ranking(ss, benchmark)
                    check = min(range(len(check_scores)), key=lambda i: (round(1-check_scores[i], 10), i))
                    assert chosen == check
                    heldout = float(tt[chosen].mean())
                    shortfall = 100 * (best_t - heldout)
                    added = 100 * (before_t - heldout)
                    assert abs(added - (shortfall - before)) < 1e-10
                    independent = 100 * (sum(map(float, gt[original])) / gt.shape[1] -
                                         sum(map(float, tt[check])) / tt.shape[1])
                    assert abs(added - independent) < 1e-10
                    t_ranks = np.round(1 - aggregate(tt, benchmark), 10)
                    rank_before = 1 + int(np.sum(t_ranks < t_ranks[original]))
                    rank_after = 1 + int(np.sum(t_ranks < t_ranks[chosen]))
                    heldout_scalar = [round(1-v, 10) for v in scalar_ranking(tt, benchmark)]
                    assert rank_after == 1 + sum(v < heldout_scalar[chosen] for v in heldout_scalar)
                    assert rank_before == 1 + sum(v < heldout_scalar[original] for v in heldout_scalar)
                    false = bool(rounded[-1] == rounded.min() and t_ranks[-1] > t_ranks.min())
                    archived = historical.loc[(benchmark, trial, budget)]
                    assert false == bool(archived.first_s_not_t)
                    assert abs(100 * (best_t - tt[-1].mean()) - archived.heldout_regret_pp) < 1e-10
                    rows.append(dict(benchmark=benchmark, trial=trial, new_submissions=budget,
                        no_attack_model=models[original],
                        selected_model='Final router' if chosen == len(models) else models[chosen],
                        selected_router=chosen == len(models),
                        router_tied_public_first=bool(rounded[-1] == rounded.min() and chosen != len(models)),
                        reused_score_pp=100 * float(ss[chosen].mean()),
                        heldout_score_pp=100 * heldout, no_attack_shortfall_pp=before,
                        selected_shortfall_pp=shortfall, model_selection_loss_pp=added,
                        rank_drop=rank_after-rank_before,
                        router_false_winner=false))
    return pd.DataFrame(rows)


def summarize(raw):
    indices = np.random.default_rng(2026081900).integers(0, 250, (2000, 250))
    rows = []
    metrics = ['no_attack_shortfall_pp', 'selected_shortfall_pp', 'model_selection_loss_pp',
               'reused_score_pp', 'heldout_score_pp', 'selected_router', 'router_false_winner', 'rank_drop']
    for (benchmark, budget), frame in raw.groupby(['benchmark', 'new_submissions']):
        frame = frame.sort_values('trial')
        assert frame.trial.tolist() == list(range(250))
        eligible = frame.eligible.to_numpy(bool) if 'eligible' in frame else np.ones(250, bool)
        row = dict(benchmark=benchmark, new_submissions=int(budget), trials=int(eligible.sum()))
        for metric in metrics:
            values = frame[metric].to_numpy(float)
            values[~eligible] = np.nan
            average = np.mean if eligible.all() else np.nanmean
            low, high = np.quantile(average(values[indices], axis=1), [.025, .975])
            row.update({metric: float(average(values)), metric + '_low': float(low), metric + '_high': float(high)})
        assert abs(row['model_selection_loss_pp'] - (row['selected_shortfall_pp'] - row['no_attack_shortfall_pp'])) < 1e-10
        rows.append(row)
    return pd.DataFrame(rows)
