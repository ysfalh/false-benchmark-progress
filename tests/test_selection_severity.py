"""Check the scientific event and denominator, independently of the display."""
import numpy as np
import pandas as pd
import pytest
from scripts.selection_severity import summarize


def sample():
    # Negative score deficit can coexist with a rank-based false win (HELM Lite).
    return pd.DataFrame(dict(benchmark=['example']*4, policy=['top5']*4,
        new_submissions=[2]*4, trial=[0,1,2,3], first_s_not_t=[True,True,False,True],
        heldout_regret_pp=[-0.5,2.0,9.0,5.0]))


def test_joint_event_uses_all_trials_and_preserves_rank_based_false_wins():
    rates, means = summarize(sample(), trials=4, resamples=100)
    rates = rates.set_index('threshold_pp')
    assert rates['count'].to_dict() == {'any':3, '1':2, '2':2, '5':1}
    assert rates['rate'].to_dict() == {'any':.75, '1':.5, '2':.5, '5':.25}
    assert means.iloc[0].mean_deficit_pp == pytest.approx(6.5/3)
    assert ((rates.ci_low >= 0) & (rates.ci_high <= 1)).all()


def test_no_false_winners_has_zero_rates_and_undefined_conditional_mean():
    frame = sample().assign(first_s_not_t=False)
    rates, means = summarize(frame, trials=4, resamples=100)
    assert (rates['rate'] == 0).all()
    assert np.isnan(means.iloc[0].mean_deficit_pp)
    assert means.iloc[0].bootstrap_valid_resamples == 0


def test_incomplete_or_duplicate_trials_are_rejected():
    frame = sample()
    frame.loc[3,'trial'] = 2
    with pytest.raises(ValueError, match='one row per trial'):
        summarize(frame, trials=4, resamples=100)
