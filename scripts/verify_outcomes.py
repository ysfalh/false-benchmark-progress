"""Check outcome-bank routing, selected scores, summaries, and optional reruns."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from benchmark_progress.benchmarks import ROOT
from benchmark_progress.routers import build_gate_bank
from scripts.outcome_bank import load, replay, summarize, BUDGETS


def verify(benchmark, rerun=False):
    base = ROOT/'results/outcome_banks'/benchmark
    for name, digest in json.loads((base/'checksums.json').read_text()).items():
        assert hashlib.sha256((base/name).read_bytes()).hexdigest() == digest, name
    config, q, counts = load(benchmark)
    bank = build_gate_bank([SimpleNamespace(name='all',content_hashes=tuple(q.content_hash))],
                           max(BUDGETS)-1,config['routing_seed'],config['tie_seed'])
    gates, ties = bank.candidate_gates['all'],bank.tie_gates['all']
    with gzip.open(base/'routers.jsonl.gz','rt') as f:
        records = [json.loads(line) for line in f]
    trials = pd.read_csv(base/'trials.csv')
    with np.load(base/'evaluation.npz') as saved:
        for record in records:
            i = record['trial']
            mask = saved[f'{i}_s_mask']
            means = {}
            for side, keep in [('s',mask),('t',~mask)]:
                means[side] = counts[:,keep].sum(axis=1)/(keep.sum()*config['repeats'])
                np.testing.assert_allclose(means[side],saved[f'{i}_genuine_{side}'],atol=1e-14,rtol=0)
            if not record['eligible']:
                assert not trials[trials.trial == i].eligible.any()
                continue
            a,b = saved[f'{i}_endpoints']
            for k,rule in zip(BUDGETS,record['rules']):
                signs = np.empty(len(q),dtype=np.int8)
                for c,coordinate in enumerate(record['coordinates']):
                    pick = q.category.to_numpy() == coordinate
                    weights = np.asarray(rule['weights'][c],dtype=np.int64)
                    vote = weights @ gates[:len(weights),pick]
                    default = np.full(pick.sum(),rule['fallback'][c]) if not len(weights) and rule['fallback'][c] else ties[pick]
                    signs[pick] = np.where(vote>0,1,np.where(vote<0,-1,default))
                values = np.where(signs == 1,counts[a],counts[b])
                fs,ft = [values[m].sum()/(m.sum()*config['repeats']) for m in [mask,~mask]]
                j = BUDGETS.index(k)
                np.testing.assert_allclose([fs,ft],[saved[f'{i}_final_s'][j],saved[f'{i}_final_t'][j]],atol=1e-14,rtol=0)
                # Select using released public values, resolving ties in favor of genuine models.
                public = [*record['genuine_overall'],record['final_overall'][j]]
                chosen = min(range(len(public)),key=lambda x:(round(1-public[x],10),x))
                before = min(range(len(public)-1),key=lambda x:(round(1-public[x],10),x))
                heldout = [*means['t'],ft]
                row = trials[(trials.trial==i)&(trials.new_submissions==k)].iloc[0]
                assert abs(row.model_selection_loss_pp-100*(heldout[before]-heldout[chosen])) < 1e-10
                assert bool(row.selected_router) == (chosen==len(public)-1)
        if rerun:
            for record, ev, rows in replay(benchmark):
                i = record['trial']
                assert json.loads(json.dumps(record)) == records[i], i
                for key,value in ev.items():
                    np.testing.assert_allclose(value,saved[f'{i}_{key}'],atol=1e-14,rtol=0)
                pd.testing.assert_frame_equal(pd.DataFrame(rows).reset_index(drop=True),
                    trials[trials.trial==i].reset_index(drop=True),check_dtype=False,atol=1e-10,rtol=0)
    expected = pd.read_csv(base/'summary.csv')
    pd.testing.assert_frame_equal(summarize(trials),expected,check_dtype=False,atol=1e-10,rtol=0)
    return dict(status='PASS',benchmark=benchmark,trials=len(records),eligible=sum(r['eligible'] for r in records),
                routing_outcomes=sum(r['eligible'] for r in records)*len(BUDGETS),rerun=rerun)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',required=True)
    parser.add_argument('--rerun',action='store_true')
    args=parser.parse_args()
    print(json.dumps(verify(args.benchmark,args.rerun)))
