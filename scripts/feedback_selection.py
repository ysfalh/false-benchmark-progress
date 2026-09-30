"""Selection loss from the saved controlled-feedback scores."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scripts.numerics import sequential_sum


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def derive(root, out):
    release = root/'results/feedback_dimension_v1'
    checksums = json.loads((release/'checksums.json').read_text())
    used = ['evaluation.npz', 'protocol.json', 'groupings.json', 'dimension_budget_curves.csv']
    for name in used:
        assert sha(release/name) == checksums[name], name
    protocol = json.loads((release/'protocol.json').read_text())
    groups = json.loads((release/'groupings.json').read_text())
    conditions, budgets = protocol['conditions'], protocol['budgets']
    models = protocol['config']['models']
    assert models == sorted(set(models)) and len(models) == 33
    with np.load(release/'evaluation.npz') as saved:
        gs, gt, fs, ft = [saved[k].copy() for k in ['genuine_s', 'genuine_t', 'final_s', 'final_t']]
    assert gs.shape == gt.shape == (250, 33, 6)
    assert fs.shape == ft.shape == (250, 14, 12, 6)
    public = np.round(1-gs.mean(axis=2), 10)
    original = public.argmin(axis=1)
    before = gt[np.arange(250), original].mean(axis=1)
    router_s, router_t = fs.mean(axis=3), ft.mean(axis=3)
    # Genuine models precede the router, so exact public ties preserve the old choice.
    selected = np.round(1-router_s, 10) < public.min(axis=1)[:, None, None]
    loss = 100*(before[:, None, None]-np.where(selected, router_t, before[:, None, None]))
    false_winner = ((np.round(1-router_s, 10) <= public.min(axis=1)[:, None, None]) &
                    (np.round(1-router_t, 10) > np.round(1-gt.mean(axis=2), 10).min(axis=1)[:, None, None]))
    baseline = {int(r['trial']): r for r in read(root/'results/no_attack_v1/trials.csv') if r['benchmark'] == 'livebench'}
    original_trials = {(int(r['trial']), int(r['new_submissions'])): r
                       for r in read(root/'site/data/selection_loss_trials.csv') if r['benchmark'] == 'livebench'}
    trials = []
    for trial in range(250):
        assert original[trial] == int(baseline[trial]['selected_index'])
        assert abs(100*before[trial]-float(baseline[trial]['heldout_score_pp'])) < 1e-10
        # A scalar implementation independently checks every public selection.
        genuine = [sequential_sum(row)/6 for row in gs[trial]]
        for ci, condition in enumerate(conditions):
            for j, budget in enumerate(budgets):
                public_scores = genuine+[sequential_sum(fs[trial, ci, j])/6]
                chosen = min(range(34), key=lambda i: (round(1-public_scores[i], 10), i))
                heldout = sequential_sum(ft[trial, ci, j] if chosen == 33 else gt[trial, chosen])/6
                assert (chosen == 33) == selected[trial, ci, j]
                assert abs(100*(before[trial]-heldout)-loss[trial, ci, j]) < 1e-10
                if condition == 'd18_p0' and budget >= 2:
                    old = original_trials[trial, budget]
                    assert abs(loss[trial, ci, j]-float(old['model_selection_loss_pp'])) < 1e-10
                    assert bool(selected[trial, ci, j]) == (old['selected_router'] == 'True')
                trials.append(dict(condition=condition, trial=trial, new_submissions=budget,
                    no_attack_model=models[original[trial]], selected_router=bool(selected[trial, ci, j]),
                    no_attack_t_pp=100*before[trial], selected_t_pp=100*heldout,
                    model_selection_loss_pp=loss[trial, ci, j]))
    write(out/'data/feedback_selection_loss_trials.csv', trials)
    indices = np.random.default_rng(2026081900).integers(0, 250, (2000, 250))
    original_curves = {(int(r['dimension']), int(r['new_submissions'])): r for r in read(release/'dimension_budget_curves.csv')}
    curves, partitions = [], []
    for d in [1, 2, 3, 6, 9, 18]:
        logical = [g for g in groups if g['dimension'] == d]
        assert [g['partition'] for g in logical] == [0, 1, 2]
        ix = [conditions.index(g['computed_condition']) for g in logical]
        for j, budget in enumerate(budgets):
            values = loss[:, ix, j]
            partition_means = values.mean(axis=0)
            # Pair trial resamples across all three logical groupings, including aliases.
            lo, hi = np.quantile(values.mean(axis=1)[indices].mean(axis=1), [.025, .975])
            false = float(false_winner[:, ix, j].mean())
            assert abs(false-float(original_curves[d, budget]['first_s_not_t'])) < 1e-12
            curves.append(dict(dimension=d, new_submissions=budget, released_score_coordinates=d*(budget-1),
                trials=250, predeclared_partitions=3, model_selection_loss_pp=float(values.mean()),
                model_selection_loss_pp_ci_low=float(lo), model_selection_loss_pp_ci_high=float(hi),
                model_selection_loss_pp_partition_min=float(partition_means.min()),
                model_selection_loss_pp_partition_max=float(partition_means.max()),
                no_attack_t_pp=100*float(before.mean()), selected_t_pp=100*float(before.mean())-float(values.mean()),
                selected_router_rate=float(selected[:, ix, j].mean()), first_s_not_t=false))
            for g, mean in zip(logical, partition_means):
                partitions.append(dict(dimension=d, partition=g['partition'], computed_condition=g['computed_condition'],
                    new_submissions=budget, model_selection_loss_pp=float(mean)))
    write(out/'data/feedback_selection_loss.csv', curves)
    write(out/'data/feedback_selection_loss_partitions.csv', partitions)
    inputs = [release/name for name in used]+[root/'results/no_attack_v1/trials.csv', root/'site/data/selection_loss_trials.csv']
    method = dict(status='PASS', analysis='Post-hoc metric extension from frozen scoring records; no attack rerun.',
        definition='Held-out score of the original public-selected genuine model minus the held-out score of the public-selected model after adding the final router, in pp.',
        evaluation='Equal mean across the six original LiveBench categories, unchanged across feedback dimensions.',
        selection='Minimize round(1 - public mean, 10); genuine models in lexical order precede the router.',
        aggregation='Mean of three grouping-specific means over the same 250 paired trials. Aliases are not independent trials.',
        intervals='Bars show min/max of grouping-specific means. CSV also retains descriptive paired percentile 95% intervals from 2000 resamples, seed 2026081900.',
        source_sha256={str(p.relative_to(root)):sha(p) for p in inputs},
        unique_outcomes_checked=len(trials), dimension_18_trial_losses_matched=2750,
        checks=['All 250 no-attack choices and held-out scores match the frozen reference.',
                'All 42000 public choices independently checked with scalar arithmetic.',
                'All d=18 losses and selected-router flags match Figure 1 trial records.',
                'All 72 false-winner means reproduce the original controlled-study table.'])
    (out/'data/feedback_selection_method.json').write_text(json.dumps(method, indent=2)+'\n')
    return curves

