"""Apply the website's selection metric to saved controls and policy trials."""

import json
import numpy as np
import pandas as pd

from benchmark_progress.benchmarks import ROOT
from benchmark_progress.feedback_policy import overall_scores
from scripts.no_attack import choose_public_winner
from scripts.selection_loss import calculate, summarize, source_paths

POLICY = ROOT / 'results/feedback_policy_v1'


def policy_trials():
    protocol = json.loads((POLICY / 'protocol.json').read_text())
    models, budgets = protocol['config']['models'], protocol['budgets']
    saved = pd.read_parquet(POLICY / 'trials.parquet').set_index(
        ['trial', 'policy', 'new_submissions'])
    rows = []
    for trial in range(protocol['trials']):
        audits = json.loads((POLICY / 'run' / f'feedback_{trial:03d}.json').read_text())
        with np.load(POLICY / 'run' / f'evaluation_{trial:03d}.npz') as trace:
            for audit in audits:
                mode = audit['policy']
                public = overall_scores(trace[mode + '_public_genuine'], audit['categories'])
                final = overall_scores(trace[mode + '_public_final'], audit['categories'])
                original = choose_public_winner(models, public)
                gs = trace[mode + '_genuine_s'].mean(axis=1)
                gt = trace[mode + '_genuine_t'].mean(axis=1)
                fs = trace[mode + '_final_s'].mean(axis=1)
                ft = trace[mode + '_final_t'].mean(axis=1)
                for j, budget in enumerate(budgets):
                    # The released public values decide selection, including ties.
                    s_loss = np.round(1 - np.append(public, final[j]), 10)
                    chosen = int(np.argmin(s_loss))
                    heldout = np.append(gt, ft[j])
                    t_loss = np.round(1 - heldout, 10)
                    false = bool(s_loss[-1] == s_loss.min() and t_loss[-1] > t_loss.min())
                    old = saved.loc[(trial, mode, budget)]
                    assert false == bool(old.false_winner)
                    assert abs(100 * ft[j] - old.heldout_score_pp) < 1e-10
                    assert abs(100 * (gt.max() - ft[j]) - old.heldout_regret_pp) < 1e-10
                    before = 100 * (gt.max() - gt[original])
                    after = 100 * (gt.max() - heldout[chosen])
                    rows.append(dict(study='practical', policy=mode, benchmark='livebench',
                        trial=trial, new_submissions=budget, no_attack_model=models[original],
                        selected_model='Final router' if chosen == len(models) else models[chosen],
                        selected_router=chosen == len(models),
                        reused_score_pp=100 * np.append(gs, fs[j])[chosen],
                        heldout_score_pp=100 * heldout[chosen],
                        no_attack_shortfall_pp=before, selected_shortfall_pp=after,
                        model_selection_loss_pp=100 * (gt[original] - heldout[chosen]),
                        rank_drop=int((t_loss < t_loss[chosen]).sum()) -
                                  int((t_loss < t_loss[original]).sum()),
                        router_false_winner=false))
    return pd.DataFrame(rows)


def tables():
    original = calculate(budgets=(1, 128, 2048)).assign(study='original', policy='top5')
    frontier = calculate('full_frontier', ('helm_capabilities',), (128, 2048)).assign(
        study='original', policy='full_frontier')
    raw = pd.concat([original, frontier, policy_trials()], ignore_index=True)
    summary = pd.concat([summarize(group).assign(study=study, policy=policy)
        for (study, policy), group in raw.groupby(['study', 'policy'])], ignore_index=True)
    return raw, summary


def sources():
    return (source_paths() + [
        ROOT / 'scripts/selection_loss.py', ROOT / 'scripts/secondary_selection.py',
        POLICY / 'protocol.json', POLICY / 'trials.parquet', POLICY / 'checksums.json']
        + sorted((POLICY / 'run').glob('feedback_*.json')) + sorted((POLICY / 'run').glob('evaluation_*.npz')))
