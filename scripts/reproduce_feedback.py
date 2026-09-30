"""Reproduce the controlled LiveBench feedback-dimension sweep with current code."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from benchmark_progress.attacks import fit_rule, select_pair
from benchmark_progress.benchmarks import ROOT, BUDGETS, load, partition, make_bank, genuine_feedback, evaluate
from benchmark_progress.evaluation import compact_trace, outcomes
from benchmark_progress.feedback_dimension import RELEASE, definitions, verify_groupings, grouped_feedback, expand_rule

STATE = None


def initialize(output, dimensions):
    global STATE
    config, q, matrix = load('livebench')
    q['_category_position'] = q.groupby('category').cumcount()
    bank = make_bank(config, q)
    protocol = json.loads((RELEASE/'protocol.json').read_text())
    assert config == protocol['config'] and bank.seed_manifest_sha256 == protocol['bank_sha256']
    groups = definitions(); verify_groupings(groups)
    unique = [g for g in groups if g['computed_condition'] == g['condition'] and g['dimension'] in dimensions]
    audits = {(r['trial'], r['condition']): r for r in json.loads((RELEASE/'routing_records.json').read_text())}
    choices = {r['trial']: r for r in map(json.loads, (ROOT/'results/release_v1/livebench/endpoints.jsonl').read_text().splitlines()) if r['policy'] == 'top5'}
    STATE = (output, config, q, matrix, bank, unique, audits, choices)


def trial_run(trial):
    output, config, q, matrix, bank, groups, audits, choices = STATE
    reused, _ = partition(config, q, trial)
    smask = q.question_id.isin(reused).to_numpy()
    qs, ms = q.loc[smask], matrix[:, smask]
    public, overall, coords, cats = genuine_feedback(ms, qs, config)
    choice = select_pair(config['models'], public, 'livebench', 'top5', overall, cats)
    for key in ['base_a','base_b','canonical_pair','frontier','top_five']:
        assert choice[key] == choices[trial][key]
    a, b = (config['models'].index(choice[k]) for k in ['base_a','base_b'])
    ps, _, task = evaluate(ms, qs, config, bank, a, b, public=True)
    rules, records = [], []
    for definition in groups:
        c = definition['condition']; ref = audits[trial,c]
        assert (a,b) == (ref['a'],ref['b'])
        table, mapping = grouped_feedback(task, coords, definition)
        assert hashlib.sha256(table.tobytes()).hexdigest() == ref['public_sha256']
        fitted = [expand_rule(fit_rule(table[33],table[34],table[35:35+k-1],'livebench'), mapping) for k in BUDGETS]
        assert [r.record() for r in fitted] == [{**r, 'fallback':tuple(r['fallback'])} for r in ref['rules']]
        # Check every final item decision, before looking at held-out outcomes.
        signs = np.empty((len(BUDGETS),len(q)), dtype=np.int8)
        for j, name in enumerate(coords):
            category, task_name = name.split('/',1)
            pick = ((q.category == category) & (q.task == task_name)).to_numpy()
            positions = q.loc[pick,'_category_position'].to_numpy(int)
            for ri, rule in enumerate(fitted):
                signs[ri,pick] = rule.route(bank.candidate_gates[category][:,positions],bank.tie_gates[category][positions],j)
        assert hashlib.sha256(signs.tobytes()).hexdigest() == ref['signs_sha256']
        records.append({'condition':c,'rules':[r.record() for r in fitted]});rules.extend(fitted)
    (output/f'rules_{trial:03d}.json').write_text(json.dumps(records)+'\n')
    # All conditions are frozen above before any T evaluation.
    _, fs, _ = evaluate(ms, qs, config, bank, a, b, rules)
    pt, ft, _ = evaluate(matrix[:,~smask],q.loc[~smask],config,bank,a,b,rules)
    fs = fs.reshape(len(groups),len(BUDGETS),6);ft = ft.reshape(fs.shape)
    rows = []
    for ci,g in enumerate(groups):
        trace = compact_trace(ps,pt,fs[ci],ft[ci],33)
        result = outcomes('livebench',g['condition'],trial,trace)
        for r in result:
            r.update(dimension=g['dimension'],partition=g['partition'],
                     released_score_coordinates=g['dimension']*(r['new_submissions']-1),
                     method='static_grouped' if r['new_submissions']==1 else 'random_group_linear')
        rows.extend(result)
    pd.DataFrame(rows).to_parquet(output/f'trial_{trial:03d}.parquet',index=False)
    return trial


def main(output, dimensions, workers):
    output.mkdir(parents=True,exist_ok=False)
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(output,dimensions)) as pool:
        for i,_ in enumerate(pool.map(trial_run, range(250))):
            if (i+1)%25 == 0: print(f'{i+1}/250 paired trials ({time.monotonic()-start:.1f}s)',flush=True)
    actual = pd.concat([pd.read_parquet(output/f'trial_{t:03d}.parquet') for t in range(250)],ignore_index=True)
    actual.to_parquet(output/'trials.parquet',index=False)
    verify_run(output, dimensions)


def verify_run(output, dimensions):
    actual = pd.read_parquet(output/'trials.parquet')
    expected = pd.read_parquet(RELEASE/'trials.parquet')
    keys = ['trial','policy','new_submissions']
    actual = actual.set_index(keys).sort_index();expected = expected.set_index(keys).loc[actual.index]
    pd.testing.assert_frame_equal(actual,expected[actual.columns],check_exact=False,rtol=0,atol=1e-10)
    d18 = actual.reset_index().query('dimension == 18')
    if len(d18):
        old = pd.read_parquet(ROOT/'results/release_v1/trials.parquet').query("benchmark == 'livebench' and policy == 'top5'").sort_values(['trial','new_submissions'])
        for col in ['score_gap_pp','genuine_preload_score_gap_pp','first_s','first_t','first_s_not_t','rank_s','rank_t','heldout_regret_pp','ranking_regret_pp']:
            if d18[col].dtype.kind == 'f':
                np.testing.assert_allclose(d18[col],old[col],rtol=0,atol=1e-10)
            else:
                np.testing.assert_array_equal(d18[col].to_numpy(),old[col].to_numpy())
    (output/'verification.json').write_text(json.dumps({'status':'PASS','trials':250,'outcomes':len(actual),'dimensions':dimensions,'all_public_tables_and_final_routing_decisions':'byte-exact against source hashes','d18_release_v1_metrics':'exact discrete outcomes; floats within original 1e-10 pp tolerance' if len(d18) else 'not requested','other_metric_tolerance_pp':1e-10},indent=2)+'\n')
    print(f'PASS: {len(actual):,} reproduced outcomes; all public tables and final routing decisions match.',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'runs/feedback_dimension')
    p.add_argument('--dimensions',type=int,nargs='+',choices=[1,2,3,6,9,18],default=[1,2,3,6,9,18])
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--verify-run',type=Path,help='Check an already reproduced run without rerunning it')
    args=p.parse_args()
    if args.verify_run: verify_run(args.verify_run,args.dimensions)
    else: main(args.output,args.dimensions,args.workers)
