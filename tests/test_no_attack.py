"""No-attack selection must precede held-out evaluation and keep metrics distinct."""
import json
import numpy as np
import pandas as pd
import pytest
from benchmark_progress.benchmarks import ROOT
from scripts.no_attack import choose_public_winner,selected_outcome


def test_public_selection_and_lexical_ties():
    assert choose_public_winner(['z','a'],[.8,.8])==1
    x=selected_outcome(['a','b'],[.8,.7],np.array([[.8],[.7]]),np.array([[.7],[.8]]),'custom')
    assert x['selected_model']=='a' and x['heldout_rank']==2 and x['selection_failure']
    assert x['heldout_regret_pp']==pytest.approx(10)
    assert x['selected_gap_pp']==pytest.approx(10)


def test_original_preload_gap_is_not_selected_model_gap():
    x=selected_outcome(['a','b'],[.1,.9],np.array([[.1],[.9]]),np.array([[.7],[.8]]),'custom')
    assert x['selected_model']=='b'
    assert x['selected_gap_pp']==pytest.approx(10)
    assert x['preload_gap_pp']==pytest.approx(60)


def test_helm_lite_ranking_and_score_regret_are_distinct():
    s=np.array([[.9,.9,0],[.8,.8,1]])
    t=np.array([[.8,.8,0],[.7,.7,1]])
    x=selected_outcome(['a','b'],[2/3,1/3],s,t,'helm_lite')
    assert x['selected_model']=='a' and x['heldout_rank']==1
    assert x['heldout_regret_pp']==pytest.approx(100*(2.4/3-1.6/3))


def test_references_cover_each_trial_once_and_match_old_gap():
    raw=pd.read_csv(ROOT/'results/no_attack_v1/trials.csv')
    assert len(raw)==1250 and not raw.duplicated(['benchmark','trial']).any()
    summary=pd.read_csv(ROOT/'results/no_attack_v1/summary.csv').set_index('benchmark')
    old=pd.read_csv(ROOT/'results/release_v2/headline.csv').query("policy=='top5'").set_index('benchmark')
    np.testing.assert_allclose(summary.loc[old.index,'preload_gap_mean_pp'],old.score_gap_baseline_pp,rtol=0,atol=1e-10)
    assert json.loads((ROOT/'results/no_attack_v1/verification.json').read_text())['status']=='PASS'
