"""Reproduce the published Terminal-Bench counterexample from bundled inputs."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pandas as pd
from benchmark_progress.attacks import select_endpoints, fit_task_rule
from benchmark_progress.benchmarks import ROOT
from benchmark_progress.extensions import partition, genuine_feedback, scale, evaluate
from benchmark_progress.evaluation import compact_trace, outcomes
from benchmark_progress.routers import build_gate_bank

BASE = ROOT / 'results/terminal_counterexample'
BUDGETS = [2,4,8,16,32,64,128,256,512,1024,2048]
DATASETS = ['terminal_bench_4']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def run(output):
    protocol=json.loads((BASE/'protocol.json').read_text());lock=json.loads((BASE/'protocol_lock.json').read_text())
    assert digest(BASE/'protocol.json')==lock['protocol_sha256']
    for name,expected in protocol['input_sha256'].items():assert digest(BASE/name)==expected
    for name,expected in protocol['code_sha256'].items():assert digest(ROOT/name)==expected
    output.mkdir(parents=True, exist_ok=False);all_rows=[]
    for b in DATASETS:
        d=json.loads((BASE/f'{b}_inputs.json').read_text());c=protocol['configs'][b]
        q=pd.DataFrame(dict(question_id=d['question_ids'],content_hash=d['prompt_hashes'],category=d['item_coordinates'],task=d['item_coordinates']))
        q['_category_position']=q.groupby('category').cumcount()
        data=np.array(d['scores'],float);aux=np.zeros_like(data);n=len(d['models'])
        metadata=[SimpleNamespace(name=cat,content_hashes=tuple(q.loc[q.category==cat,'content_hash'])) for cat in c['categories']]
        bank=build_gate_bank(metadata,2047,c['routing_seed'],c['tie_seed'])
        dest=output/b;dest.mkdir();traces=[];choices=[];publics=[];overalls=[];metrics=[]
        start=time.monotonic()
        with (dest/'router_records.jsonl').open('w') as records:
            for trial in range(250):
                s,t=partition(c,q,trial);mask=q.question_id.isin(s).to_numpy();qs=q.loc[mask]
                assert len(s)+len(t)==len(q) and not s.intersection(t)
                profiles,overall=genuine_feedback(data[:,mask],aux[:,mask],qs,c)
                precision=scale(c,qs)
                choice=select_endpoints(c['models'],profiles,'top5',scale=precision,overall=overall)
                a,bidx=[c['models'].index(choice[k]) for k in ['base_a','base_b']]
                ps,_,released=evaluate(data[:,mask],aux[:,mask],qs,c,bank,a,bidx,public=True)
                np.testing.assert_array_equal(released[:n],profiles)
                rules=[fit_task_rule(released[n],released[n+1],released[n+2:n+k+1],scale=precision) for k in BUDGETS]
                before=int(np.argmin(np.round(1-overall,10)))
                records.write(json.dumps(dict(trial=trial,choice=choice,rules=[r.record() for r in rules],no_attack_model=c['models'][before]))+'\n');records.flush()
                # All choices and all rules have been fixed before consulting held-out outcomes.
                _,fs,_=evaluate(data[:,mask],aux[:,mask],qs,c,bank,a,bidx,rules)
                pt,ft,_=evaluate(data[:,~mask],aux[:,~mask],q.loc[~mask],c,bank,a,bidx,rules)
                trace=compact_trace(ps,pt,fs,ft,n);traces.append(trace)
                originals=outcomes(c['benchmark'],'top5',trial,trace,budgets=BUDGETS)
                old_t=float(pt[before,0]);best_t=float(pt[:n,0].max())
                # Validate each coordinate aggregate against direct task means.
                np.testing.assert_allclose(ps[:n,0],data[:,mask].mean(axis=1),atol=1e-12,rtol=0)
                np.testing.assert_allclose(pt[:n,0],data[:,~mask].mean(axis=1),atol=1e-12,rtol=0)
                baseline=dict(benchmark=c['benchmark'],trial=trial,new_submissions=0,
                    no_attack_model=c['models'][before],selected_model=c['models'][before],selected_router=False,
                    router_tied_public_first=False,reused_score_pp=100*float(ps[before,0]),heldout_score_pp=100*old_t,
                    no_attack_heldout_score_pp=100*old_t,model_selection_loss_pp=0.,
                    selected_shortfall_pp=100*(best_t-old_t),no_attack_shortfall_pp=100*(best_t-old_t),added_shortfall_pp=0.,
                    router_false_winner=False,score_gap_pp=originals[0]['genuine_preload_score_gap_pp'],
                    genuine_preload_score_gap_pp=originals[0]['genuine_preload_score_gap_pp'],reused_items=int(mask.sum()),heldout_items=int((~mask).sum()))
                all_rows.append(baseline)
                for j,k in enumerate(BUDGETS):
                    ss=np.r_[ps[:n,0],fs[j,0]];tt=np.r_[pt[:n,0],ft[j,0]]
                    chosen=int(np.argmin(np.round(1-ss,10)))
                    check=min(range(n+1),key=lambda i:(round(1-float(ss[i]),10),i));assert chosen==check
                    # Independent task-level replay of the frozen rule for all tasks.
                    values=np.empty(len(q))
                    for ci,cat in enumerate(c['categories']):
                        local=(q.category==cat).to_numpy()
                        weights=np.array(rules[j].weights[ci],dtype=np.int64)
                        assert sum(map(abs,rules[j].weights[ci])) < np.iinfo(np.int64).max
                        votes=(weights[:,None]*bank.candidate_gates[cat][:k-1]).sum(axis=0)
                        signs=np.where(votes==0,bank.tie_gates[cat],np.where(votes>0,1,-1))
                        values[local]=np.where(signs==1,data[a,local],data[bidx,local])
                    assert abs(values[mask].mean()-fs[j,0])<1e-12
                    assert abs(values[~mask].mean()-ft[j,0])<1e-12
                    delta=100*(old_t-float(tt[chosen]))
                    shortfall=100*(best_t-float(tt[chosen]))
                    assert abs(delta-(shortfall-baseline['no_attack_shortfall_pp']))<1e-10
                    r=dict(baseline,new_submissions=k,selected_model='Final router' if chosen==n else c['models'][chosen],
                        selected_router=chosen==n,router_tied_public_first=bool(round(1-ss[-1],10)==round(1-ss.max(),10) and chosen!=n),
                        reused_score_pp=100*float(ss[chosen]),heldout_score_pp=100*float(tt[chosen]),
                        model_selection_loss_pp=delta,added_shortfall_pp=delta,selected_shortfall_pp=shortfall,
                        router_false_winner=originals[j]['first_s_not_t'],score_gap_pp=originals[j]['score_gap_pp'])
                    all_rows.append(r)
                choices.append(dict(choice,trial=trial));publics.append(profiles);overalls.append(overall);metrics.extend(originals)
                if (trial+1)%25==0:print(f'{c["benchmark"]}: {trial+1}/250 ({time.monotonic()-start:.1f}s)',flush=True)
        pd.DataFrame(metrics).to_parquet(dest/'original_metrics.parquet',index=False)
        np.savez_compressed(dest/'evaluation.npz',**{key:np.stack([t[key] for t in traces]) for key in traces[0]},trials=np.arange(250),policies=np.array(['top5']*250))
        np.savez_compressed(dest/'public_profiles.npz',profiles=np.stack(publics),overall=np.stack(overalls),models=np.array(c['models']),trials=np.arange(250),coordinates=np.array(c['categories']))
        write_json(dest/'endpoints.json',choices)
    raw=pd.DataFrame(all_rows);raw.to_csv(output/'trials.csv',index=False)
    summary=summarize(raw);summary.to_csv(output/'summary.csv',index=False)
    write_json(output/'verification.json',dict(status='PASS',records=len(raw),
        protocol_sha256=lock['protocol_sha256'],checks=['Input and attack-source hashes match packaged protocol',
            'Task repeats never cross partitions; coordinate-stratified split uses public hashes only',
            'All routing fits and selections precede held-out evaluation',
            'Every final router independently replayed from integer weights on task scores',
            'Genuine aggregate scores match direct task means',
            'All model choices and paired loss arithmetic independently checked']))
    print(summary[summary.new_submissions.isin([0,16,128,2048])].to_string(index=False),flush=True)


def summarize(raw):
    indices=np.random.default_rng(2026081900).integers(0,250,(2000,250));summaries=[]
    for (b,k),f in raw.groupby(['benchmark','new_submissions']):
        f=f.sort_values('trial');assert f.trial.tolist()==list(range(250))
        row=dict(benchmark=b,new_submissions=int(k),trials=250)
        for name in ['model_selection_loss_pp','no_attack_heldout_score_pp','heldout_score_pp','reused_score_pp','router_false_winner']:
            v=f[name].to_numpy(float);lo,hi=np.quantile(v[indices].mean(axis=1),[.025,.975])
            row.update({name:float(v.mean()),name+'_low':float(lo),name+'_high':float(hi)})
        for field,target in [('score_gap_pp','score_gap_mean_pp'),('genuine_preload_score_gap_pp','preload_mean_pp')]:
            v=f[field].to_numpy(float);boot=v[indices].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
            row.update({target:float(v.mean()),target+'_ci_low':float(lo),target+'_ci_high':float(hi)})
        summaries.append(row)
    return pd.DataFrame(summaries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/terminal_counterexample')
    args = parser.parse_args()
    run(args.output)
    for name in ['trials.csv', 'summary.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(args.output/name), pd.read_csv(BASE/'run'/name),
                                      check_exact=False, rtol=0, atol=1e-10)
    print('PASS: all 250 trials and summaries match the published counterexample.', flush=True)
