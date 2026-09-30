"""Regression checks for the new benchmark-specific aggregation boundaries."""
import numpy as np
import pandas as pd
from benchmark_progress.extensions import category_score, overall, partition, scale
from benchmark_progress.benchmarks import partition as original_partition
from benchmark_progress.evaluation import compact_trace, outcomes


def test_item_weighted_ranking_not_equal_domain_average():
    q = pd.DataFrame({'category':['small','large','large','large']})
    cfg = {'benchmark':'swe_verified','categories':['small','large']}
    profiles = np.array([[1,0],[0,1]])
    np.testing.assert_array_equal(overall(profiles,q,cfg), [.25,.75])
    trace = compact_trace(np.array([[.25],[.75],[.8]]),np.array([[.25],[.75],[.2]]),
                          np.array([[.8]]),np.array([[.2]]),2)
    # A standalone final and the usual A/B prefix shape.
    trace['best_loss_s'] = np.array([.75,.25,.25,.25])
    trace['best_loss_t'] = np.array([.75,.25,.25,.25])
    row = outcomes('swe_verified','top5',0,trace,budgets=(1,))[0]
    assert row['first_s_not_t'] and row['rank_t']==3
    assert np.isclose(row['heldout_regret_pp'],55)


def test_ifeval_instruction_denominator_and_single_prompt_routing():
    q=pd.DataFrame({'instruction_count':[1,3]})
    cfg={'benchmark':'openllm_v2'}
    # Prompt-level mean .5; instruction-level accuracy 3/4, not 1/2.
    result=category_score(np.array([[0,1]]),np.array([[0,1]]),q,cfg,'IFEval')
    np.testing.assert_array_equal(result,[.625])


def test_normalize_and_clip_each_subtask_before_averaging():
    q=pd.DataFrame({'task':['a','a','b','b','b']})
    cfg={'benchmark':'openllm_v2','chance_baselines':{'a':.5,'b':.2}}
    values=np.array([[0,0,1,1,1]])
    np.testing.assert_array_equal(category_score(values,None,q,cfg,'BBH'),[.5])
    # GPQA uses all subset occurrences and a single normalization.
    np.testing.assert_allclose(category_score(values,None,q,cfg,'GPQA'),[(.6-.25)/.75])


def test_bitset_split_is_the_original_exact_group_split():
    rows=[]
    for task in ['a','b']:
        for g,size in enumerate([1,2,1,3,2,1,1,1]):
            for i in range(size):
                rows.append(dict(question_id=f'{task}/{g}/{i}',content_hash=f'{task}/{g}',task=task,category=task))
    q=pd.DataFrame(rows);cfg={'benchmark':'new','split_seed_base':99,'holdout_fraction':2/3}
    for trial in range(30):
        assert partition(cfg,q,trial)==original_partition(cfg,q,trial)
