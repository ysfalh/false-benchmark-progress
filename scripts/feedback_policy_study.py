"""Freeze, run, and independently audit the paired practical policy comparison."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from benchmark_progress.adapter import PublicProfiles
from benchmark_progress.attacks import fit_task_rule
from benchmark_progress.benchmarks import ROOT, load, partition, genuine_feedback, evaluate
from benchmark_progress.feedback_dimension import expand_rule
from benchmark_progress.feedback_policy import FeedbackPolicy, MODES, overall_scores, select_policy_endpoints
from benchmark_progress.routers import build_gate_bank
from benchmark_progress.reporting import bootstrap_stat

BUDGETS = (1,2,4,8,16,32,64,128)
FILES = ['configs/livebench.json', 'data/livebench_questions.parquet', 'data/livebench_scores.parquet',
         'benchmark_progress/feedback_policy.py', 'benchmark_progress/adapter.py',
         'benchmark_progress/attacks.py', 'benchmark_progress/benchmarks.py',
         'benchmark_progress/routers.py', 'benchmark_progress/splits.py',
         'benchmark_progress/feedback_dimension.py', 'benchmark_progress/reporting.py',
         'scripts/feedback_policy_study.py']
STATE = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(folder):
    folder.mkdir(parents=True, exist_ok=False)
    config, q, _ = load('livebench')
    protocol = dict(name='practical_feedback_policy_v1', frozen_at=datetime.now(timezone.utc).isoformat(),
        benchmark='livebench',
        config=config, trials=250, budgets=list(BUDGETS), policies=list(MODES), precision_pp=.001,
        endpoints='Top five of all genuine models by allowed equal-category overall; maximum equal-category-weighted L1 pair; lexical ties; A is weaker overall. Same rule for every policy; no Pareto filtering.',
        feedback='Native published task ticks; exact within-category averages for category policy; exact equal-category mean for aggregate policy; round once to .001 pp, ties to even. No extra overall column.',
        submission_budget='k-1 random candidates and one final; k=1 is a separately evaluated static branch; genuine profiles free; candidate gates and splits paired across policies and nested across budgets.',
        rank='Reused rank uses ONLY allowed released feedback aggregated with equal-category weights; held-out rank uses unrounded equal-category score. Compare genuine models plus final, strict better with 1e-10 loss tolerance; ties share first.',
        regret='Best genuine held-out score minus final held-out score, signed pp, unconditional mean over all 250 trials.',
        static_gain='Final minus same-policy k=1 static router, separately for unrounded reused and held-out score in pp; also false-winner-rate difference in pp.',
        utility='Among all genuine-model pairs with absolute held-out overall gap >=1 pp, fraction whose allowed public overall strictly orders them correctly (ties count as failures); trial-level macro mean; denominator saved. No T used for endpoint choice or decoding.',
        comparisons='All policies and budgets; paired differences category-task, aggregate-task, aggregate-category.',
        uncertainty='2000 shared trial bootstrap resamples, seed 2026092201; descriptive pointwise 95% percentile intervals; overlapping splits are not independent datasets.',
        execution='Freeze all public decisions for every policy and budget in a trial before opening T. Save rules and feedback before private evaluation. No outcome-based exclusions or stopping.',
        limitations='One historical snapshot; one fixed router attack; endpoint and decoding information change together. Utility measures overall ordering of large gaps, not task diagnostics. No universally safe policy claim.',
        hashes={f:sha(ROOT/f) for f in FILES})
    (folder/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (folder/'protocol_lock.json').write_text(json.dumps({'protocol_sha256':sha(folder/'protocol.json'),
        'frozen_at':protocol['frozen_at'], 'note':'Local pre-run lock, not an external preregistration.'},indent=2)+'\n')
    print(f'Frozen protocol: {folder}; SHA256 {sha(folder / "protocol.json")}', flush=True)


def check_lock(folder):
    p = json.loads((folder/'protocol.json').read_text())
    lock = json.loads((folder/'protocol_lock.json').read_text())
    if sha(folder/'protocol.json') != lock['protocol_sha256']:
        raise ValueError('Protocol hash mismatch')
    for name, expected in p['hashes'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Frozen input/code changed: {name}')
    return p


def initialize(output):
    global STATE
    config,q,matrix=load('livebench')
    q['_category_position']=q.groupby('category').cumcount()
    metadata=[SimpleNamespace(name=c, content_hashes=tuple(q.loc[q.category==c,'content_hash'])) for c in config['categories']]
    bank=build_gate_bank(metadata, max(BUDGETS)-1, config['routing_seed'], config['tie_seed'])
    STATE=output,config,q,matrix,bank


def final_task_profiles(matrix, questions, config, bank, a, b, rules, coords):
    """Private evaluator implementation; returns native public S feedback only."""
    values=np.empty((len(rules),len(questions)))
    for ci,coordinate in enumerate(coords):
        category,task=coordinate.split('/',1)
        pick=((questions.category==category)&(questions.task==task)).to_numpy()
        pos=questions.loc[pick,'_category_position'].to_numpy(int)
        for ri,rule in enumerate(rules):
            signs=rule.route(bank.candidate_gates[category][:,pos],bank.tie_gates[category][pos],ci)
            values[ri,pick]=np.where(signs==1,matrix[a,pick],matrix[b,pick])
    return genuine_feedback(values,questions,config)[0]


def trial_run(trial):
    output,config,q,matrix,bank=STATE
    reused,heldout=partition(config,q,trial)
    smask=q.question_id.isin(reused).to_numpy()
    qs,ms=q.loc[smask],matrix[:,smask]
    native,_,coords,cats=genuine_feedback(ms,qs,config)
    n=len(config['models'])
    frozen=[]
    for mode in MODES:
        policy=FeedbackPolicy(mode,tuple(coords),tuple(cats))
        public=policy.project(native)
        choice=select_policy_endpoints(config['models'],public,scale=policy.scale,categories=policy.released_categories)
        a,b=[config['models'].index(choice[k]) for k in ['base_a','base_b']]
        ps,_,task=evaluate(ms,qs,config,bank,a,b,public=True)
        released=policy.project(task[n+2:])
        fitted=[fit_task_rule(public[a],public[b],released[:k-1],policy.scale) for k in BUDGETS]
        mapping=tuple(policy.coordinates.index(policy.mapping[t]) for t in coords)
        rules=[expand_rule(rule,mapping) for rule in fitted]
        _,fs,_=evaluate(ms,qs,config,bank,a,b,rules)
        final_public=policy.project(final_task_profiles(ms,qs,config,bank,a,b,rules,coords))
        audit=dict(trial=trial, policy=mode, choice=choice, coordinates=policy.coordinates,
            categories=policy.released_categories, public_profiles=public.tolist(), candidate_feedback=released.tolist(),
            final_feedback=final_public.tolist(), rules=[dict(weights=r.weights,fallback=r.fallback,candidate_count=r.candidate_count) for r in fitted])
        frozen.append((mode,policy,a,b,ps,fs,rules,audit))
    # Persist EVERY condition's feedback and decisions before any T scoring.
    feedback_path=output/f'feedback_{trial:03d}.json'
    feedback_path.write_text(json.dumps([x[-1] for x in frozen],separators=(',',':'))+'\n')
    traces={}; rows=[]
    for mode,policy,a,b,ps,fs,rules,audit in frozen:
        pt,ft,_=evaluate(matrix[:,~smask],q.loc[~smask],config,bank,a,b,rules)
        trace=dict(genuine_s=ps[:n], genuine_t=pt[:n], final_s=fs, final_t=ft,
                   public_genuine=np.array(audit['public_profiles']),public_final=np.array(audit['final_feedback']))
        traces.update({mode+'_'+key:value for key,value in trace.items()})
        rows.extend(metrics(trial,mode,trace,policy.released_categories))
    traces['coverage']=q.assign(split=np.where(smask,'S','T')).groupby(['category','task','split']).size().to_numpy()
    np.savez_compressed(output/f'evaluation_{trial:03d}.npz',**traces)
    return rows


def metrics(trial,mode,trace,categories):
    gs=trace['genuine_s'].mean(axis=1); gt=trace['genuine_t'].mean(axis=1)
    fs=trace['final_s'].mean(axis=1); ft=trace['final_t'].mean(axis=1)
    pg=overall_scores(trace['public_genuine'],categories)
    pf=overall_scores(trace['public_final'],categories)
    left,right=np.triu_indices(len(gt),1)
    eligible=np.abs(gt[left]-gt[right]) >= .01-1e-12
    correct=((pg[left]-pg[right])*(gt[left]-gt[right])>1e-15)&eligible
    utility=float(correct.sum()/eligible.sum()) if eligible.any() else float('nan')
    rows=[]
    for j,k in enumerate(BUDGETS):
        rs=1+int((np.round(1-pg,10)<round(1-float(pf[j]),10)).sum())
        rt=1+int((np.round(1-gt,10)<round(1-float(ft[j]),10)).sum())
        rows.append(dict(trial=trial,policy=mode,new_submissions=k,rank_s=rs,rank_t=rt,
            false_winner=rs==1 and rt>1,heldout_regret_pp=100*(gt.max()-ft[j]),
            reused_score_pp=100*fs[j],heldout_score_pp=100*ft[j],
            reused_gain_static_pp=100*(fs[j]-fs[0]),heldout_gain_static_pp=100*(ft[j]-ft[0]),
            distinguishability=utility,eligible_pairs=int(eligible.sum()),correct_pairs=int(correct.sum())))
    return rows


def summarize(raw):
    indices=np.random.default_rng(2026092201).integers(0,250,(2000,250))
    columns=['false_winner','heldout_regret_pp','reused_gain_static_pp','heldout_gain_static_pp','distinguishability',
             'reused_score_pp','heldout_score_pp']
    rows=[]; pairs=[]
    for (mode,k),f in raw.groupby(['policy','new_submissions']):
        f=f.sort_values('trial'); assert f.trial.tolist()==list(range(250))
        row=dict(policy=mode,new_submissions=k,trials=len(f))
        for col in columns:
            point,lo,hi,_=bootstrap_stat(f[col].to_numpy(float),indices,'mean',.025)
            row.update({col:point,col+'_low':lo,col+'_high':hi})
        static=raw.query('policy == @mode and new_submissions == 1').sort_values('trial')
        delta=(f.false_winner.to_numpy(float)-static.false_winner.to_numpy(float))*100
        point,lo,hi,_=bootstrap_stat(delta,indices,'mean',.025)
        row.update(false_winner_gain_static_pp=point,false_winner_gain_static_pp_low=lo,false_winner_gain_static_pp_high=hi)
        rows.append(row)
    for k in BUDGETS:
        for a,b in [('category','task'),('aggregate','task'),('aggregate','category')]:
            fa=raw.query('policy == @a and new_submissions == @k').sort_values('trial')
            fb=raw.query('policy == @b and new_submissions == @k').sort_values('trial')
            for col in columns:
                delta=fa[col].to_numpy(float)-fb[col].to_numpy(float)
                if col in ['false_winner','distinguishability']: delta*=100
                point,lo,hi,_=bootstrap_stat(delta,indices,'mean',.025)
                pairs.append(dict(comparison=a+' minus '+b,new_submissions=k,metric=col,
                                  difference=point,ci_low=lo,ci_high=hi))
    return pd.DataFrame(rows),pd.DataFrame(pairs)


def run(folder,workers):
    p=check_lock(folder)
    output=folder/'run'; output.mkdir(exist_ok=False)
    (output/'run_start.json').write_text(json.dumps({'started_at':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':sha(folder/'protocol.json'), 'workers':workers},indent=2)+'\n')
    rows=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(output,)) as pool:
        for i,trial_rows in enumerate(pool.map(trial_run,range(p['trials']))):
            rows.extend(trial_rows)
            if (i+1)%25==0: print(f'{i+1}/250 paired trials',flush=True)
    raw=pd.DataFrame(rows); raw.to_parquet(folder/'trials.parquet',index=False)
    summary,paired=summarize(raw)
    summary.to_csv(folder/'summary.csv',index=False); paired.to_csv(folder/'paired_differences.csv',index=False)
    verify(folder)


def verify(folder):
    p=check_lock(folder); rows=[]
    for trial in range(250):
        with np.load(folder/'run'/f'evaluation_{trial:03d}.npz') as trace:
            audits=json.loads((folder/'run'/f'feedback_{trial:03d}.json').read_text())
            for audit in audits:
                mode=audit['policy']; prefix=mode+'_'
                data={key[len(prefix):]:trace[key] for key in trace if key.startswith(prefix)}
                # Independently check ranks, score/regret arithmetic and allowed endpoint selection.
                choice=select_policy_endpoints(p['config']['models'],np.array(audit['public_profiles']),
                    scale=100000,categories=audit['categories'])
                assert choice==audit['choice']
                a,b=[p['config']['models'].index(choice[k]) for k in ['base_a','base_b']]
                for j,k in enumerate(BUDGETS):
                    rule=fit_task_rule(data['public_genuine'][a],data['public_genuine'][b],
                                      np.array(audit['candidate_feedback'])[:k-1],100000)
                    assert json.loads(json.dumps(dict(weights=rule.weights,fallback=rule.fallback,
                                                      candidate_count=rule.candidate_count)))==audit['rules'][j]
                np.testing.assert_array_equal(data['public_final'],audit['final_feedback'])
                rows.extend(metrics(trial,mode,data,audit['categories']))
    expected=pd.DataFrame(rows)
    actual=pd.read_parquet(folder/'trials.parquet')
    pd.testing.assert_frame_equal(expected,actual,check_exact=False,atol=1e-12,rtol=0)
    summary,paired=summarize(actual)
    for name,table in [('summary.csv',summary),('paired_differences.csv',paired)]:
        pd.testing.assert_frame_equal(pd.read_csv(folder/name),table,check_exact=False,atol=1e-10,rtol=0)
    result=dict(status='PASS',trials=250,outcomes=len(actual),protocol_sha256=sha(folder/'protocol.json'),
        checks=['Frozen code and input hashes','Allowed endpoint choices and all decoder weights',
                'Ranks and metrics reconstructed from compact evaluator records','All summaries and paired intervals'],
        limits='Trace verification is not independent item rescoring; tests additionally compare adapter and vectorized execution and independently rescore selected trials.')
    (folder/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    paths=[x for x in folder.rglob('*') if x.is_file() and x.name!='checksums.json']
    (folder/'checksums.json').write_text(json.dumps({str(x.relative_to(folder)):sha(x) for x in sorted(paths)},indent=2)+'\n')
    print(f'PASS: {len(actual)} outcomes and paired summaries',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','run','verify'])
    parser.add_argument('--output',type=Path,default=ROOT/'results/feedback_policy_v1')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.action=='freeze': freeze(args.output)
    elif args.action=='run': run(args.output,args.workers)
    else: verify(args.output)
