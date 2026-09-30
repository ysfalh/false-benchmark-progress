"""Small checks for the scientific boundary and numerical conventions."""

import inspect
import numpy as np
import pandas as pd
import pytest

from benchmark_progress.attacks import fit_rule, mean_win_rate, select_pair
from benchmark_progress.benchmarks import load, partition
from benchmark_progress.routers import build_gate_bank
from benchmark_progress.splits import select_by_task
from types import SimpleNamespace


def test_measurement_recovers_opposite_endpoint_advantages():
    # A succeeds only on x; B only on y. The random candidate sends x to A,
    # y to B, and receives a perfect public score. Its combination must agree.
    rule = fit_rule([.5], [.5], [[1.]], 'livebench')
    np.testing.assert_array_equal(rule.route(np.array([[1, -1]]), np.ones(2), 0), [1, -1])
    negative = fit_rule([.5], [.5], [[0.]], 'livebench')
    np.testing.assert_array_equal(negative.route(np.array([[1, -1]]), np.ones(2), 0), [-1, 1])


def test_rounding_and_exact_ties():
    rule = fit_rule([.40001], [.40003], [[.40002]], 'livebench')
    np.testing.assert_array_equal(rule.route(np.ones((1, 2)), np.array([-1, 1]), 0), [-1, 1])
    static = fit_rule([.4], [.4], np.empty((0, 1)), 'openllm')
    np.testing.assert_array_equal(static.route(np.empty((0, 2)), np.array([1, -1]), 0), [1, -1])


def test_full_precision_json_remains_distinguishable():
    a = .5
    b = np.nextafter(a, 1.)
    static = fit_rule([a], [b], np.empty((0, 1)), 'helm_lite')
    assert static.fallback == (-1,)


def test_win_rate_half_credit_and_final_population():
    public = np.array([[.9, .1], [.8, .9], [.8, .1]])
    np.testing.assert_array_equal(mean_win_rate(public), [.625, .625, .25])


def test_small_frontier_and_lexical_ties():
    public = np.array([[1., 0.], [0., 1.], [.5, .5], [0., 0.]])
    choice = select_pair(['a', 'b', 'c', 'd'], public, 'helm_capabilities', 'top5')
    assert choice['top_five'] == ['a', 'b', 'c']
    assert choice['canonical_pair'] == ['a', 'b']
    with pytest.raises(ValueError, match='Fewer than two'):
        select_pair(['a', 'b'], np.array([[1., 1.], [.5, .5]]), 'helm_capabilities', 'top5')


def test_openllm_uses_separately_published_average():
    scores = np.eye(6)
    models = list('abcdef')
    choice = select_pair(models, scores, 'openllm', 'top5', np.array([.1,.1,.1,.1,.1,.2]))
    assert choice['top_five'] == ['f','a','b','c','d']
    with pytest.raises(ValueError, match='Average'):
        select_pair(models, scores, 'openllm', 'top5')


def test_routing_ignores_occurrence_and_metadata_order():
    first = build_gate_bank([SimpleNamespace(name='task',content_hashes=('x','x','y'))], 5,'seed','ties')
    second = build_gate_bank([SimpleNamespace(name='task',content_hashes=('y','x'))], 5,'seed','ties')
    np.testing.assert_array_equal(first.candidate_gates['task'][:,0],first.candidate_gates['task'][:,1])
    np.testing.assert_array_equal(first.candidate_gates['task'][:,0],second.candidate_gates['task'][:,1])
    assert first.tie_gates['task'][0] == second.tie_gates['task'][1]


def test_split_keeps_duplicate_prompts_together():
    q = pd.DataFrame({'question_id':['a','b','c','d'],'content_hash':['x','x','y','z'],'task':['t']*4})
    selected = select_by_task(q, {'t':2}, 7)
    assert ('a' in selected) == ('b' in selected)
    assert len(selected) == 2


def test_attack_signatures_contain_public_values_only():
    assert list(inspect.signature(fit_rule).parameters) == ['a','b','candidates','benchmark']
    assert list(inspect.signature(select_pair).parameters) == ['models','public','benchmark','policy','public_overall','coordinate_categories']


@pytest.mark.parametrize('benchmark,s_count,t_count',[('livebench',204,408),('helm_capabilities',1064,2126),('helm_lite',2721,5438),('openllm',1278,2562)])
def test_original_partition_sizes(benchmark,s_count,t_count):
    config, questions, _ = load(benchmark)
    s,t = partition(config,questions,0)
    assert (len(s),len(t)) == (s_count,t_count)
