"""Check that the public adapter preserves routing and keeps T out of fitting."""

import json
from types import SimpleNamespace
import numpy as np
import pytest

from benchmark_progress.adapter import run_attack
from benchmark_progress.attacks import select_pair, select_endpoints
from benchmark_progress.benchmarks import ROOT, BENCHMARKS
from benchmark_progress.routers import build_gate_bank
from examples.custom_benchmark.run import SyntheticBenchmark


@pytest.mark.parametrize('budget', [1, 2, 8])
def test_budget_and_original_hash_routing(budget):
    benchmark = SyntheticBenchmark()
    public = benchmark.public_access()
    assert set(vars(public)) == {'genuine_public_profiles', 'public_feedback'}
    result = run_attack(public, budget, routing_seed='example', tie_seed='example-ties')
    assert benchmark.public_calls == budget
    bank = build_gate_bank([SimpleNamespace(name='task', content_hashes=benchmark._hashes)],
                           budget-1, 'example', 'example-ties')
    for i, router in enumerate(result.candidates):
        expected = np.where(bank.candidate_gates['task'][i] == 1,
                            result.choice['base_a'], result.choice['base_b'])
        assert [router(h, 'task_a') for h in benchmark._hashes] == list(expected)
    for c, task in enumerate(benchmark.coordinates):
        signs = result.final.rule.route(bank.candidate_gates['task'], bank.tie_gates['task'], c)
        expected = np.where(signs == 1, result.choice['base_a'], result.choice['base_b'])
        assert [result.final(h, task) for h in benchmark._hashes] == list(expected)


def test_no_heldout_scoring_during_attack():
    first, second = SyntheticBenchmark(), SyntheticBenchmark()
    # Poison T: any accidental evaluation during fitting would be observable.
    second._scores[:, ~second._s] = np.nan
    a = run_attack(first.public_access(), 8)
    b = run_attack(second.public_access(), 8)
    assert a.choice == b.choice
    assert a.final.rule == b.final.rule
    np.testing.assert_array_equal(a.public_feedback, b.public_feedback)
    assert np.isfinite(b.public_feedback).all()


def test_feedback_shape_is_checked():
    benchmark = SyntheticBenchmark()
    public = SimpleNamespace(genuine_public_profiles=benchmark._public_profiles,
                             public_feedback=lambda submission: np.array([.5]))
    with pytest.raises(ValueError, match='one finite public score'):
        run_attack(public, 2)


@pytest.mark.parametrize('benchmark', BENCHMARKS)
def test_extracted_selector_matches_released_trials(benchmark):
    folder = ROOT/'results/release_v2'/benchmark
    public = np.load(folder/'public_profiles.npz')
    for line in (folder/'endpoints.jsonl').read_text().splitlines():
        expected = json.loads(line)
        t = expected['trial']
        if benchmark in {'openllm_v2', 'swe_verified'}:
            actual = select_endpoints(list(public['models']), public['profiles'][t], expected['policy'],
                scale=expected['public_scale'], overall=public['overall'][t])
        else:
            actual = select_pair(list(public['models']), public['profiles'][t], benchmark,
                                 expected['policy'], public['overall'][t], list(public['categories']))
        for key in ['base_a', 'base_b', 'canonical_pair', 'frontier', 'top_five']:
            assert actual[key] == expected[key]


def test_example_metrics():
    benchmark = SyntheticBenchmark()
    result = run_attack(benchmark.public_access(), 8, routing_seed='example', tie_seed='example-ties')
    metrics = benchmark.evaluate(result)
    # Independently compute the small example's history and competition ranks.
    histories = []
    for mask in [benchmark._s, ~benchmark._s]:
        profiles = np.vstack((benchmark._genuine(mask),
            [benchmark._score(r, mask) for r in (*result.candidates, result.final)]))
        losses = 1-profiles.mean(axis=1)
        histories.append(np.minimum.accumulate(losses))
        rank = 1+sum(np.round(losses[:3], 10) < round(float(losses[-1]), 10))
        assert metrics['rank_s' if mask is benchmark._s else 'rank_t'] == rank
    assert metrics['score_gap_pp'] == pytest.approx(100*max(abs(histories[0]-histories[1])))
