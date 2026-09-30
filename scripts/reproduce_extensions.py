"""Reproduce Open LLM v2 and SWE-bench Verified with the shared attack and metrics."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from benchmark_progress.attacks import fit_task_rule, select_endpoints
from benchmark_progress.benchmarks import BUDGETS, POLICIES, ROOT, make_bank
from benchmark_progress.extensions import EXTENSIONS, evaluate, genuine_feedback, load, scale, partition
from benchmark_progress.evaluation import compact_trace, outcomes


def run(benchmark, output, trials=None):
    if output.exists():
        raise FileExistsError(f'{output} exists; use a new --output directory')
    config, q, matrix, auxiliary = load(benchmark)
    q['_category_position'] = q.groupby('category').cumcount()
    trial_ids = range(config['trials']) if trials is None else trials
    bank = make_bank(config, q)
    rows, choices, profiles, overalls, traces = [], [], [], [], []
    output.mkdir(parents=True)
    protocol = {'benchmark': benchmark, 'config': config, 'budgets': BUDGETS,
        'policies': POLICIES, 'trial_ids': list(trial_ids), 'seed_manifest_sha256': bank.seed_manifest_sha256,
        'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'data').glob(f'{benchmark}_*.parquet'))},
        'code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ['benchmark_progress', 'scripts'] for p in sorted((ROOT/folder).glob('*.py'))}}
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    started = time.monotonic()
    with (output/'router_records.jsonl').open('w') as records:
        for trial in trial_ids:
            reused, holdout = partition(config, q, trial)
            smask = q.question_id.isin(reused).to_numpy()
            qs, ms = q.loc[smask], matrix[:, smask]
            auxs = auxiliary[:, smask]
            public, overall = genuine_feedback(ms, auxs, qs, config)
            coords = cats = config['categories']
            precision = scale(config, qs)
            profiles.append(public); overalls.append(overall)
            fitted = []
            for policy in POLICIES:
                choice = select_endpoints(config['models'], public, policy, scale=precision, overall=overall)
                choice.update(trial=trial, benchmark=benchmark, public_scale=precision)
                choices.append(choice)
                a, b = (config['models'].index(choice[key]) for key in ['base_a', 'base_b'])
                ps, _, released = evaluate(ms, auxs, qs, config, bank, a, b, public=True)
                np.testing.assert_array_equal(released[:len(config['models'])], public)
                n = len(config['models'])
                rules = [fit_task_rule(released[n], released[n+1], released[n+2:n+k+1], scale=precision) for k in BUDGETS]
                records.write(json.dumps({'trial': trial, 'policy': policy,
                    'base_a': choice['base_a'], 'base_b': choice['base_b'],
                    'rules': [rule.record() for rule in rules]})+'\n')
                fitted.append((policy, a, b, ps, rules))
            records.flush()
            # Every policy and budget has been fitted and recorded before T is evaluated.
            for policy, a, b, ps, rules in fitted:
                _, fs, _ = evaluate(ms, auxs, qs, config, bank, a, b, rules)
                pt, ft, _ = evaluate(matrix[:, ~smask], auxiliary[:, ~smask], q.loc[~smask], config, bank, a, b, rules)
                trace = compact_trace(ps, pt, fs, ft, len(config['models']))
                traces.append(trace)
                rows.extend(outcomes(benchmark, policy, trial, trace))
            if (trial+1) % 25 == 0 or len(trial_ids) < 25:
                print(f'{benchmark}: trial {trial+1} ({time.monotonic()-started:.1f}s)', flush=True)
    pd.DataFrame(rows).to_parquet(output/'trials.parquet', index=False)
    (output/'endpoints.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in choices))
    np.savez_compressed(output/'public_profiles.npz', profiles=np.stack(profiles), overall=np.stack(overalls),
        coordinates=np.array(coords), categories=np.array(cats), models=np.array(config['models']), trials=np.array(list(trial_ids)))
    np.savez_compressed(output/'evaluation.npz', **{k: np.stack([t[k] for t in traces]) for k in traces[0]},
        trials=np.array([c['trial'] for c in choices]), policies=np.array([c['policy'] for c in choices]))
    print(f'Saved {len(rows):,} outcomes to {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=EXTENSIONS, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.benchmark, args.output or ROOT/'runs'/'extensions'/args.benchmark)
