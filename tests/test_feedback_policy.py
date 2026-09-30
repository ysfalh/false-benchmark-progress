"""Independently rescore real items via the callback adapter, not vectorized evaluate."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from benchmark_progress.adapter import PublicAccess, PublicProfiles, run_attack
from benchmark_progress.benchmarks import ROOT, load, partition, genuine_feedback
from benchmark_progress.feedback_policy import FeedbackPolicy, select_policy_endpoints, overall_scores


@pytest.mark.parametrize('trial',[0,17,249])
@pytest.mark.parametrize('mode',['task','category','aggregate'])
def test_adapter_matches_saved_run_and_direct_item_scoring(trial,mode):
    config,q,matrix=load('livebench')
    reused,_=partition(config,q,trial);mask=q.question_id.isin(reused).to_numpy()
    native,_,coords,cats=genuine_feedback(matrix[:,mask],q.loc[mask],config)
    models=tuple(config['models']);tasks=(q.category+'/'+q.task).to_numpy();hashes=q.content_hash.to_numpy()
    def values(submission,pick):
        indices=np.flatnonzero(pick)
        chosen=np.array([models.index(submission(hashes[i],tasks[i])) for i in indices])
        return matrix[chosen,indices]
    def profile(v,pick):
        # Independent explicit task means, preserving native task publication.
        return np.array([round(float(v[tasks[pick]==c].mean()*100),3)/100 for c in coords])
    native_public=PublicAccess(lambda:PublicProfiles(models,tuple(coords),native,100000,tuple(cats)),
                              lambda submission:profile(values(submission,mask),mask))
    policy=FeedbackPolicy(mode,tuple(coords),tuple(cats))
    result=run_attack(policy.restrict(native_public),128,routing_seed=config['routing_seed'],
                      tie_seed=config['tie_seed'],endpoint_selector=select_policy_endpoints)
    audit=next(a for a in json.loads((ROOT/f'results/feedback_policy_v1/run/feedback_{trial:03d}.json').read_text()) if a['policy']==mode)
    assert result.choice==audit['choice']
    np.testing.assert_array_equal(result.public_feedback[:-1],audit['candidate_feedback'])
    np.testing.assert_array_equal(result.public_feedback[-1],audit['final_feedback'][-1])
    assert [list(x) for x in result.final.rule.weights]==audit['rules'][-1]['weights']
    final=policy.for_evaluator(result).final
    actual=pd.read_parquet(ROOT/'results/feedback_policy_v1/trials.parquet')
    row=actual.query('trial == @trial and policy == @mode and new_submissions == 128').iloc[0]
    for pick,name in [(mask,'reused'),(~mask,'heldout')]:
        v=values(final,pick)
        categories=[]
        for cat in dict.fromkeys(cats):
            means=[v[tasks[pick]==c].mean() for c,cc in zip(coords,cats) if cc==cat]
            categories.append(np.mean(means))
        assert row[name+'_score_pp']==pytest.approx(100*np.mean(categories),abs=1e-10)
    # Verify the static baseline via an independent adapter branch as well.
    static=run_attack(policy.restrict(native_public),1,routing_seed=config['routing_seed'],
                      tie_seed=config['tie_seed'],endpoint_selector=select_policy_endpoints)
    np.testing.assert_array_equal(static.public_feedback[0],audit['final_feedback'][0])
