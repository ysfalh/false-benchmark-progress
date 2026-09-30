"""Check release integrity, public endpoint choices, metrics, and tables."""

import argparse
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

from benchmark_progress.benchmarks import BENCHMARKS, BUDGETS, POLICIES, ROOT
from benchmark_progress.reporting import summarize


def ranking_scores(scores, benchmark):
    if benchmark != 'helm_lite':
        return scores.mean(axis=1)
    return np.array([mean([sum(float(row[c] > other[c]) + .5*float(row[c] == other[c])
        for j, other in enumerate(scores) if j != i)/(len(scores)-1)
        for c in range(scores.shape[1])]) for i, row in enumerate(scores)])


def verify_choice(choice, models, scores, overall, benchmark, categories):
    if benchmark in {'openllm_v2', 'swe_verified'}:
        from scripts.verify_extensions import check_choice
        return check_choice(choice, models, scores, overall)
    # Separate loops and exact arithmetic, independent of the attack selector.
    frontier = [i for i, row in enumerate(scores) if not any(
        all(other >= row) and any(other > row) for j, other in enumerate(scores) if i != j)]
    assert choice['frontier'] == [models[i] for i in frontier]
    def exact(v):
        if benchmark in {'livebench', 'openllm'}:
            scale = 100000 if benchmark == 'livebench' else 10000
            return Fraction(int(round(float(v)*scale)), scale)
        return Fraction(float(v))
    if benchmark == 'helm_lite':
        totals = ranking_scores(scores, benchmark)
        np.testing.assert_array_equal(totals, overall)
    elif benchmark == 'openllm':
        totals = [exact(v) for v in overall]
    elif benchmark == 'livebench':
        cats = list(dict.fromkeys(categories))
        totals = [sum(sum(exact(x) for x, c in zip(row, categories) if c == cat)/categories.count(cat)
                      for cat in cats)/len(cats) for row in scores]
    else:
        totals = [sum(exact(v) for v in row)/len(row) for row in scores]
    pool = sorted(frontier, key=lambda i: (-totals[i], models[i]))[:min(5, len(frontier))]
    assert choice['top_five'] == [models[i] for i in pool]
    candidates = pool if choice['policy'] == 'top5' else frontier
    best = min(combinations(sorted(candidates), 2), key=lambda pair: (
        -sum(abs(exact(x)-exact(y)) for x, y in zip(scores[pair[0]], scores[pair[1]])),
        tuple(models[i] for i in pair)))
    assert choice['canonical_pair'] == [models[i] for i in best]
    def loss(i):
        if benchmark == 'livebench':
            return np.mean([1-np.mean(scores[i, np.array(categories) == c]*100)/100 for c in cats])
        return 1-scores[i].mean()
    a, b = sorted(best, key=lambda i: (-round(float(loss(i)), 10), models[i]))
    assert (choice['base_a'], choice['base_b']) == (models[a], models[b])


def verify_outcomes(raw, trace, benchmark):
    s, t = trace['best_loss_s'], trace['best_loss_t']
    n = len(trace['genuine_s'])
    assert np.all(np.diff(s) <= 0) and np.all(np.diff(t) <= 0)
    for part in ['s', 't']:
        expected = np.minimum.accumulate((1-trace['genuine_'+part]).mean(axis=1))
        np.testing.assert_allclose(trace['best_loss_'+part][:n], expected, rtol=0, atol=1e-12)
    assert raw.new_submissions.tolist() == list(BUDGETS)
    for j, k in enumerate(BUDGETS):
        row = raw.iloc[j]
        end = n+2+k-1
        fs, ft = trace['final_s'][j], trace['final_t'][j]
        hs = np.r_[s[:end], min(s[end-1], float(np.mean(1-fs)))]
        ht = np.r_[t[:end], min(t[end-1], float(np.mean(1-ft)))]
        assert abs(row.score_gap_pp - 100*max(abs(hs-ht))) < 1e-10
        assert abs(row.genuine_preload_score_gap_pp - 100*max(abs(s[:n]-t[:n]))) < 1e-10
        for part, final in [('s', fs), ('t', ft)]:
            scores = ranking_scores(np.vstack((trace['genuine_'+part], final)), benchmark)
            rank = 1+sum(round(float(1-v), 10) < round(float(1-scores[-1]), 10) for v in scores[:-1])
            assert row['rank_'+part] == rank and row['first_'+part] == (rank == 1)
        assert row.first_s_not_t == (row.rank_s == 1 and row.rank_t > 1)
        assert abs(row.heldout_regret_pp-100*(trace['genuine_t'].mean(axis=1).max()-ft.mean())) < 1e-10
        scores = ranking_scores(np.vstack((trace['genuine_t'], ft)), benchmark)
        assert abs(row.ranking_regret_pp-100*(max(scores[:-1])-scores[-1])) < 1e-10


def verify_public_inputs(benchmark, saved, trials):
    """Independent feedback aggregation; reuse the documented input/split reader."""
    if benchmark in {'openllm_v2', 'swe_verified'}:
        from benchmark_progress.extensions import load, partition
        from scripts.verify_extensions import independent_feedback
        config, questions, matrix, auxiliary = load(benchmark)
        np.testing.assert_array_equal(saved['models'], config['models'])
        np.testing.assert_array_equal(saved['coordinates'], config['categories'])
        for trial in trials:
            reused, heldout = partition(config, questions, trial)
            assert reused.isdisjoint(heldout) and len(reused)+len(heldout) == len(questions)
            mask = questions.question_id.isin(reused).to_numpy()
            public, overall = independent_feedback(config, questions.loc[mask], matrix[:,mask], auxiliary[:,mask])
            np.testing.assert_allclose(public, saved['profiles'][trial], rtol=0, atol=1e-12)
            np.testing.assert_allclose(overall, saved['overall'][trial], rtol=0, atol=1e-12)
        print(f'{benchmark}: genuine public feedback reconstructed from item inputs.', flush=True)
        return
    from benchmark_progress.benchmarks import load, partition
    config, questions, matrix = load(benchmark)
    np.testing.assert_array_equal(saved['models'], config['models'])
    for trial in trials:
        reused, _ = partition(config, questions, trial)
        mask = questions.question_id.isin(reused).to_numpy()
        q, values = questions.loc[mask], matrix[:, mask]
        parts, coordinates, categories = [], [], []
        for category in config['categories']:
            member = (q.category == category).to_numpy()
            tasks = q.loc[member, 'task'].to_numpy()
            scores = values[:, member]
            if benchmark == 'livebench':
                # The historical publisher rounds pandas task means in percent.
                means = pd.DataFrame(scores.T*100, index=tasks).groupby(level=0).mean().round(3)
                parts.append(np.rint(means.to_numpy().T*1000)/100000)
                coordinates.extend(f'{category}/{t}' for t in means.index)
                categories.extend([category]*len(means))
            else:
                task_scores = [np.ascontiguousarray(scores[:, tasks == t]).mean(axis=1)
                               for t in sorted(set(tasks))]
                parts.append(np.mean(task_scores, axis=0)[:, None])
                coordinates.append(category)
                categories.append(category)
        unrounded = np.concatenate(parts, axis=1)
        public = np.round(unrounded*100, 2)/100 if benchmark == 'openllm' else unrounded
        overall = (np.round(unrounded.mean(axis=1)*100, 2)/100 if benchmark == 'openllm'
                   else np.mean([public[:, np.array(categories) == c].mean(axis=1)
                                 for c in config['categories']], axis=0) if benchmark == 'livebench'
                   else ranking_scores(public, benchmark))
        np.testing.assert_array_equal(public, saved['profiles'][trial])
        np.testing.assert_array_equal(overall, saved['overall'][trial])
        np.testing.assert_array_equal(coordinates, saved['coordinates'])
        np.testing.assert_array_equal(categories, saved['categories'])
    print(f'{benchmark}: genuine public feedback reconstructed from item inputs.', flush=True)


def compare_run(output, release):
    actual = pd.read_parquet(output/'trials.parquet')
    expected = pd.read_parquet(release/'trials.parquet')
    keys = ['benchmark', 'policy', 'trial', 'new_submissions']
    actual = actual.set_index(keys).sort_index()
    expected = expected.set_index(keys).loc[actual.index]
    assert not actual.index.duplicated().any()
    for column in actual.columns:
        if actual[column].dtype.kind == 'f':
            np.testing.assert_allclose(actual[column], expected[column], rtol=0, atol=1e-10, err_msg=column)
        else:
            np.testing.assert_array_equal(actual[column], expected[column], err_msg=column)
    choices = [json.loads(l) for l in (output/'endpoints.jsonl').read_text().splitlines()]
    b = actual.index[0][0]
    ref = {(c['trial'], c['policy']): c for c in map(json.loads, (release/b/'endpoints.jsonl').read_text().splitlines())}
    for c in choices:
        for key in ['base_a', 'base_b', 'canonical_pair', 'frontier', 'top_five']:
            assert c[key] == ref[c['trial'], c['policy']][key]
    actual_profiles = np.load(output/'public_profiles.npz')
    expected_profiles = np.load(release/b/'public_profiles.npz')
    for j, trial in enumerate(actual_profiles['trials']):
        np.testing.assert_array_equal(actual_profiles['profiles'][j], expected_profiles['profiles'][int(trial)])
        np.testing.assert_array_equal(actual_profiles['overall'][j], expected_profiles['overall'][int(trial)])
    print(f'{output.name}: {len(actual):,} outcomes, endpoint choices, and public profiles match {release.name}.')
    return len(actual)


def verify_checksums(release, allow_code_changes=False):
    checksums = json.loads((release/'checksums.json').read_text())
    changed_code = []
    for name, expected in checksums.items():
        path = (release/Path(*Path(name).parts[2:])
                if name.startswith(('results/release_v1/', 'results/release_v2/')) else ROOT/name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            if allow_code_changes and name.startswith(('benchmark_progress/', 'scripts/')) and name.endswith('.py'):
                changed_code.append(name)
            else:
                raise AssertionError(f'Changed file: {name}')
    if changed_code:
        print('Code differs from the release: '+', '.join(changed_code)+
              '. Checking saved results with current code; all input and result hashes remain required.', flush=True)
    return changed_code


def verify(release, smoke=False, inputs=False, allow_code_changes=False):
    if not __debug__:
        raise RuntimeError("Run verification without Python optimization (-O)")
    verify_checksums(release, allow_code_changes)
    raw = pd.read_parquet(release/'trials.parquet')
    benchmarks = json.loads((release/'manifest.json').read_text())['benchmarks']
    assert set(raw.benchmark) == set(benchmarks)
    assert len(raw) == len(benchmarks)*2*250*len(BUDGETS)
    assert not raw.duplicated(['benchmark', 'policy', 'trial', 'new_submissions']).any()
    for b in benchmarks:
        folder = release/b
        p, z = np.load(folder/'public_profiles.npz'), np.load(folder/'evaluation.npz')
        choices = list(map(json.loads, (folder/'endpoints.jsonl').read_text().splitlines()))
        assert len(choices) == 500
        assert [(c['trial'], c['policy']) for c in choices] == [(t, p) for t in range(250) for p in POLICIES]
        checked_choices = choices[:2] if smoke else choices
        for i, c in enumerate(checked_choices):
            trial, policy = c['trial'], c['policy']
            verify_choice(c, list(p['models']), p['profiles'][trial], p['overall'][trial], b, list(p['categories']))
            assert (z['trials'][i], z['policies'][i]) == (trial, policy)
            trace = {k: z[k][i] for k in z.files if k not in ['trials', 'policies']}
            rows = raw[(raw.benchmark == b)&(raw.policy == policy)&(raw.trial == trial)].sort_values('new_submissions')
            verify_outcomes(rows, trace, b)
        print(f'{b}: {len(checked_choices)} endpoint selections and {len(checked_choices)*len(BUDGETS):,} outcomes verified.', flush=True)
        if inputs:
            verify_public_inputs(b, p, [0] if smoke else range(250))
    expected = pd.read_csv(release/'budget_curves.csv').sort_values(['benchmark','policy','new_submissions']).reset_index(drop=True)
    actual = expected if smoke else summarize(raw).sort_values(['benchmark','policy','new_submissions']).reset_index(drop=True)
    if not smoke:
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, atol=1e-10, rtol=0)
    # Headline and thresholds are checked separately against the full curves.
    for row in pd.read_csv(release/'headline.csv').itertuples():
        group = actual[(actual.benchmark == row.benchmark)&(actual.policy == row.policy)].set_index('new_submissions')
        pairs = {'static_false_selection': group.loc[1,'first_s_not_t'],
                 'queried_false_selection': group.loc[2048,'first_s_not_t'],
                 'score_gap_baseline_pp': group.loc[2048,'preload_mean_pp'],
                 'static_score_gap_pp': group.loc[1,'score_gap_mean_pp'],
                 'queried_score_gap_pp': group.loc[2048,'score_gap_mean_pp'],
                 'queried_score_gap_above_preload_pp': group.loc[2048,'score_gap_above_preload_pp'],
                 'queried_median_t_rank': group.loc[2048,'median_rank_t'],
                 'queried_mean_t_regret_pp': group.loc[2048,'mean_heldout_regret_pp'],
                 'queried_mean_ranking_regret_pp': group.loc[2048,'mean_ranking_regret_pp']}
        for key, value in pairs.items(): assert abs(getattr(row,key)-value) < 1e-10, key
    for row in pd.read_csv(release/'threshold_crossings.csv').itertuples():
        group = actual[(actual.benchmark == row.benchmark)&(actual.policy == row.policy)]
        hits = group[(group.new_submissions > 1)&(group[row.metric] >= row.threshold)]
        if hits.empty:
            assert pd.isna(row.smallest_tested_random_budget)
        else:
            assert row.smallest_tested_random_budget == hits.new_submissions.min()
        assert row.static_crosses == (group.loc[group.new_submissions == 1,row.metric].iloc[0] >= row.threshold)
    diagnostic = pd.read_parquet(release/'non_top_trials.parquet')
    summary = json.loads((release/'non_top.json').read_text())
    assert (diagnostic.a_public_rank_s > 1).all() and (diagnostic.b_public_rank_s > 1).all()
    assert np.array_equal(diagnostic.false_promotion, (diagnostic.rank_s == 1)&(diagnostic.rank_t > 1))
    selected = diagnostic[(diagnostic.method == 'random_task_linear')&(diagnostic.new_submissions == 2048)]
    assert len(selected) == summary['eligible_trials'] == summary['trials']
    assert selected.false_promotion.mean() == summary['random_false_selection']
    assert np.median(np.minimum(selected.a_rank_s, selected.b_rank_s)) == summary['better_endpoint_median_rank_s']
    print('PASS: trial 0 per benchmark, checksums and summary consistency.' if smoke else
          f'PASS: checksums, {len(raw):,} outcomes, endpoint rules, tables, and intervals.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path, nargs='?', default=ROOT/'results/release_v2')
    parser.add_argument('--run', type=Path, help='Also compare a reproduced run with the release')
    parser.add_argument('--smoke', action='store_true', help='Check trial 0 per benchmark; skip bootstrap regeneration')
    parser.add_argument('--inputs', action='store_true', help='Reconstruct genuine public feedback from bundled item scores')
    parser.add_argument('--allow-code-changes', action='store_true',
        help='Check saved results with current code; still require every input and result hash')
    args = parser.parse_args()
    verify(args.release, smoke=args.smoke, inputs=args.inputs, allow_code_changes=args.allow_code_changes)
    if args.run: compare_run(args.run, args.release)
