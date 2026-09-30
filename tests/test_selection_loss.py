"""Check paired outcomes, common-pool ranks, and the reconstructed illustration."""

import json
from collections import Counter
from contextlib import contextmanager
import numpy as np
import pandas as pd
import pytest
from benchmark_progress.benchmarks import ROOT
from scripts.selection_loss import NAMES, calculate, scalar_ranking


@pytest.mark.parametrize('benchmark', NAMES)
def test_calculate_matches_saved_trials_without_repeated_decompression(monkeypatch, benchmark):
    reads = Counter()
    original_load = np.load

    @contextmanager
    def tracked_load(path, *args, **kwargs):
        with original_load(path, *args, **kwargs) as archive:
            class TrackedArchive:
                def __getitem__(self, key):
                    reads[(str(path), key)] += 1
                    return archive[key]
            yield TrackedArchive()

    monkeypatch.setattr(np, 'load', tracked_load)
    actual = calculate(benchmarks=(benchmark,), budgets=(1, 128, 2048))
    expected = pd.read_csv(ROOT/'site/data/selection_loss_trials.csv')
    expected = expected[(expected.benchmark == benchmark) &
                        expected.new_submissions.isin([0, 1, 128, 2048])]
    # The CSV also contains ineligible rows from another snapshot, so its
    # nullable boolean/integer columns have different dtypes after reading.
    pd.testing.assert_frame_equal(actual, expected[actual.columns].reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=0, atol=1e-12)
    assert reads and max(reads.values()) == 1, reads


def test_saved_loss_and_rank_summary():
    raw = pd.read_csv(ROOT/'site/data/selection_loss_trials.csv')
    summary = pd.read_csv(ROOT/'site/data/selection_loss.csv')
    assert len(raw) == 19500 and not raw.duplicated(['benchmark', 'trial', 'new_submissions']).any()
    assert raw.loc[~raw.eligible, 'model_selection_loss_pp'].isna().all()
    assert (raw.loc[raw.no_attack_model == raw.selected_model, 'model_selection_loss_pp'].abs() < 1e-10).all()
    assert (raw.loc[raw.no_attack_model == raw.selected_model, 'rank_drop'] == 0).all()
    baseline = raw[raw.new_submissions == 0].set_index(['benchmark', 'trial'])
    for row in raw[raw.eligible].itertuples():
        assert row.model_selection_loss_pp == pytest.approx(
            baseline.loc[(row.benchmark, row.trial)].heldout_score_pp-row.heldout_score_pp, abs=1e-10)
    for row in summary.itertuples():
        group = raw[(raw.benchmark == row.benchmark) & (raw.new_submissions == row.new_submissions) & raw.eligible]
        assert len(group) == row.trials
        assert row.model_selection_loss_pp == pytest.approx(group.model_selection_loss_pp.mean(), abs=1e-10)
        assert row.rank_drop == pytest.approx(group.rank_drop.mean(), abs=1e-10)


def test_helm_lite_uses_win_rate_in_common_pool():
    # A wins two categories despite B's larger average score. Equal profiles tie.
    profiles = np.array([[.9, .9, 0], [.8, .8, 1], [.9, .9, 0]])
    assert scalar_ranking(profiles, 'helm_lite') == pytest.approx([7/12, 1/3, 7/12])


def test_worked_example_matches_archived_trial():
    demo = json.loads((ROOT/'site/data/walkthrough.json').read_text())
    assert (demo['benchmark'], demo['trial'], demo['budget']) == ('helm_lite', 165, 32)
    assert demo['model_selection_loss_pp'] == pytest.approx(3.9545104791331482, abs=1e-10)
    assert demo['model_selection_loss_pp'] == pytest.approx(demo['eligible_median_loss_pp'], abs=1e-10)
    assert demo['eligible_trials'] == 69
    assert [r['rank_t'] for r in demo['scores'][2:]] == [1, 5]
    p = demo['prompt']
    assert p['integer_vote'] == sum(w*g for w, g in zip(p['integer_weights'], p['gates']))
    assert demo['metric']['score_gap_pp'] > demo['no_attack']['preload_gap_pp']
    assert len(demo['tasks']) == 10 and all(len(t['candidates']) == 31 for t in demo['tasks'])


def test_secondary_metrics_keep_the_original_and_new_comparisons_distinct():
    raw = pd.read_csv(ROOT/'site/data/secondary_selection_trials.csv')
    summary = pd.read_csv(ROOT/'site/data/secondary_selection.csv')
    assert not raw.duplicated(['study', 'policy', 'benchmark', 'trial', 'new_submissions']).any()
    np.testing.assert_allclose(raw.model_selection_loss_pp,
        raw.selected_shortfall_pp-raw.no_attack_shortfall_pp, atol=1e-10)
    unchanged = raw.no_attack_model == raw.selected_model
    assert (raw.loc[unchanged, 'model_selection_loss_pp'] == 0).all()
    assert (raw.loc[unchanged, 'rank_drop'] == 0).all()
    # Scalar feedback's static route cannot strictly beat the best public model.
    scalar_static = raw.query('study == "practical" and policy == "aggregate" and new_submissions == 1')
    assert len(scalar_static) == 250 and not scalar_static.selected_router.any()
    # The same original trials must agree exactly with the main figure's summary.
    main = pd.read_csv(ROOT/'site/data/selection_loss.csv').query('new_submissions == 128').set_index('benchmark')
    controls = summary.query('study == "original" and policy == "top5" and new_submissions == 128').set_index('benchmark')
    pd.testing.assert_series_equal(main.loc[controls.index].model_selection_loss_pp, controls.model_selection_loss_pp)
    # At 128, all policy trials select the router. Its archived deficit must equal
    # the no-attack deficit plus the new loss, not the loss alone.
    policies = raw.query('study == "practical" and new_submissions == 128').set_index(['policy', 'trial']).sort_index()
    archived = pd.read_parquet(ROOT/'results/feedback_policy_v1/trials.parquet').query('new_submissions == 128').set_index(['policy', 'trial']).sort_index()
    assert policies.selected_router.all()
    np.testing.assert_allclose(policies.model_selection_loss_pp+policies.no_attack_shortfall_pp,
                               archived.heldout_regret_pp, atol=1e-10)
    assert np.array_equal(policies.router_false_winner, archived.false_winner)
    for row in summary.itertuples():
        group = raw[(raw.study == row.study) & (raw.policy == row.policy) &
                    (raw.benchmark == row.benchmark) & (raw.new_submissions == row.new_submissions)]
        assert len(group) == 250
        assert row.model_selection_loss_pp == pytest.approx(group.model_selection_loss_pp.mean(), abs=1e-10)
