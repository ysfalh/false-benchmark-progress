"""Eligibility and paired selection in a repeated-outcome bank."""
import json
import numpy as np
import pandas as pd
from benchmark_progress.benchmarks import ROOT
from scripts.outcome_bank import Surface, severity, selection_rows
from scripts.selection_loss import summarize


def test_repeats_are_averaged_inside_tasks_before_public_feedback():
    surface = Surface(['a','a','b'], ['a','b'], 3)
    profiles, overall = surface.release(np.array([[3,0,2],[0,3,2]]))
    np.testing.assert_array_equal(profiles,[[.5,2/3],[.5,2/3]])
    np.testing.assert_array_equal(overall,[5/9,5/9])
    assert surface.scale == 18


def test_saved_bank_keeps_ineligible_trials_and_paired_references():
    base = ROOT/'results/outcome_banks/terminal_science'
    rows = pd.read_csv(base/'trials.csv')
    config = json.loads((ROOT/'configs/terminal_science.json').read_text())
    assert len(rows) == config['trials']*(len(config['budgets'])+1)
    attacked = rows[rows.new_submissions>0]
    assert attacked.loc[~attacked.eligible,'model_selection_loss_pp'].isna().all()
    assert attacked.groupby('new_submissions').eligible.sum().nunique() == 1
    paired = attacked[attacked.eligible]
    np.testing.assert_allclose(paired.model_selection_loss_pp,paired.no_attack_t_pp-paired.selected_t_pp,atol=1e-12)
    assert (paired.loc[paired.selected_router.eq(False),'model_selection_loss_pp']==0).all()


def test_shared_summaries_match_the_frozen_bank():
    base = ROOT/'results/outcome_banks/terminal_science'
    saved = pd.read_csv(base/'summary.csv').set_index('new_submissions')
    shared = summarize(selection_rows(base)).set_index('new_submissions')
    for metric in ['model_selection_loss_pp','rank_drop','selected_router']:
        for suffix in ['', '_low', '_high']:
            np.testing.assert_allclose(shared[metric+suffix],saved[metric+suffix],atol=1e-10,rtol=0)
    rates, _ = severity(base)
    rates = rates[rates.threshold_pp=='any'].set_index('new_submissions')
    for actual, expected in [('rate','false_winner'),('ci_low','false_winner_low'),('ci_high','false_winner_high')]:
        np.testing.assert_allclose(rates[actual],saved.loc[rates.index,expected],atol=1e-10,rtol=0)
