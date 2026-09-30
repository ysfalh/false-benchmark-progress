"""Scores of the genuine public winner, with no router or new submissions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark_progress.attacks import aggregate
from benchmark_progress.benchmarks import ROOT, BENCHMARKS
from benchmark_progress.reporting import bootstrap_stat

RELEASE=ROOT/'results/release_v2'
OUTPUT=ROOT/'results/no_attack_v1'
METHOD={
    'selection':'Select the highest archived public overall among genuine models only; ties at 1e-10 score precision are broken lexically. Freeze the model before reading held-out scores. Zero new submissions.',
    'scores':'Report the same selected model on S and T, using the archived evaluator score. HELM Lite score is equal-category mean but selection and rank use mean win rate. Open LLM v2 uses normalized official overall; SWE Verified uses instance-weighted resolution rate.',
    'selection_failure':'The public-selected genuine model is not tied for first among genuine models on T under the original ranking rule. This is pre-existing selection error, not attack-induced false promotion.',
    'regret':'Best genuine held-out score minus the selected model score, in pp. For HELM Lite this is score regret, distinct from mean-win-rate ranking regret.',
    'selected_gap':'Signed S minus T score of the SAME public-selected model, in pp.',
    'preload_gap':'Original no-attack running-best score gap: maximum absolute difference between separate S/T running-best histories over the lexically ordered genuine-model preload. The two best models can differ; this is NOT the selected-model gap.',
    'summary':'250 trials once per benchmark; duplicated endpoint-policy traces are checked equal and counted only once. Means with 2,000 paired trial bootstrap resamples, seed 2026081900. Intervals describe overlapping splits of one snapshot.',
    'purpose':'Descriptive reference for existing plots; no new attack protocol or outcome-selected trials.'}


def choose_public_winner(models, public_overall):
    scores=np.asarray(public_overall,float)
    if scores.shape!=(len(models),) or not np.isfinite(scores).all():
        raise ValueError('Expected finite public overall scores for all models')
    if len(set(models))!=len(models) or not len(models):
        raise ValueError('Model names must be unique and nonempty')
    return min(range(len(models)),key=lambda i:(-round(float(scores[i]),10),models[i]))


def selected_outcome(models, public_overall, genuine_s, genuine_t, benchmark):
    index=choose_public_winner(models,public_overall)
    ss=np.asarray(genuine_s).mean(axis=1);ts=np.asarray(genuine_t).mean(axis=1)
    rank_scores=aggregate(genuine_t,benchmark)
    rank=1+int(np.sum(np.round(1-rank_scores,10)<round(1-float(rank_scores[index]),10)))
    gap=100*np.max(np.abs(np.maximum.accumulate(ss)-np.maximum.accumulate(ts)))
    return dict(selected_model=str(models[index]),selected_index=index,
        public_selection_statistic=100*float(public_overall[index]),
        reused_score_pp=100*float(ss[index]),heldout_score_pp=100*float(ts[index]),
        selected_gap_pp=100*float(ss[index]-ts[index]),heldout_rank=rank,
        selection_failure=rank>1,heldout_regret_pp=100*float(ts.max()-ts[index]),
        best_genuine_heldout_score_pp=100*float(ts.max()),preload_gap_pp=float(gap),
        public_ties=int(np.sum(np.round(public_overall,10)==round(float(public_overall[index]),10))))


def records():
    rows=[]
    for benchmark in BENCHMARKS:
        with np.load(RELEASE/benchmark/'public_profiles.npz') as p, np.load(RELEASE/benchmark/'evaluation.npz') as z:
            for trial in range(250):
                loc=np.flatnonzero(z['trials']==trial)
                assert len(loc)==2
                for part in ['genuine_s','genuine_t']:
                    np.testing.assert_array_equal(z[part][loc[0]],z[part][loc[1]])
                row=selected_outcome(p['models'],p['overall'][trial],z['genuine_s'][loc[0]],z['genuine_t'][loc[0]],benchmark)
                rows.append(dict(benchmark=benchmark,trial=trial,**row))
    return pd.DataFrame(rows)


def summaries(raw):
    indices=np.random.default_rng(2026081900).integers(0,250,(2000,250))
    rows=[]
    fields={name:(name,'mean') for name in ['reused_score_pp','heldout_score_pp','selected_gap_pp',
            'heldout_regret_pp','selection_failure','best_genuine_heldout_score_pp']}
    fields['preload_gap_mean_pp']=('preload_gap_pp','mean')
    for benchmark in BENCHMARKS:
        f=raw[raw.benchmark==benchmark].sort_values('trial');assert f.trial.tolist()==list(range(250))
        row=dict(benchmark=benchmark,trials=250)
        for name,(column,statistic) in fields.items():
            point,low,high,_=bootstrap_stat(f[column].to_numpy(float),indices,statistic,.025)
            row.update({name:point,name+'_ci_low':low,name+'_ci_high':high})
        rows.append(row)
    return pd.DataFrame(rows)


def verify(raw, summary):
    # A second implementation uses scalar loops and manually counted win rates.
    for benchmark in BENCHMARKS:
        with np.load(RELEASE/benchmark/'public_profiles.npz') as p, np.load(RELEASE/benchmark/'evaluation.npz') as z:
            f=raw[raw.benchmark==benchmark].set_index('trial')
            for trial in range(250):
                row=f.loc[trial];i=int(row.selected_index);n=len(p['models'])
                order=sorted(zip(p['models'],p['overall'][trial]),key=lambda x:(-round(float(x[1]),10),x[0]))
                assert row.selected_model==order[0][0]
                pos=int(np.flatnonzero(z['trials']==trial)[0]);s=z['genuine_s'][pos];t=z['genuine_t'][pos]
                ts=[sum(map(float,r))/len(r) for r in t]
                ranking=ts
                if benchmark=='helm_lite':
                    ranking=[sum(sum(float(r[c]>other[c])+.5*float(r[c]==other[c]) for j,other in enumerate(t) if j!=m)/(n-1) for c in range(t.shape[1]))/t.shape[1] for m,r in enumerate(t)]
                rank=1+sum(round(1-v,10)<round(1-ranking[i],10) for v in ranking)
                assert rank==row.heldout_rank and row.selection_failure==(rank>1)
                assert abs(row.heldout_score_pp-100*ts[i])<1e-10
                assert abs(row.reused_score_pp-100*sum(map(float,s[i]))/len(s[i]))<1e-10
                assert abs(row.heldout_regret_pp-100*(max(ts)-ts[i]))<1e-10
    headline=pd.read_csv(RELEASE/'headline.csv').query("policy == 'top5'").set_index('benchmark')
    for r in summary.itertuples():
        assert abs(r.preload_gap_mean_pp-headline.loc[r.benchmark].score_gap_baseline_pp)<1e-10
    pd.testing.assert_frame_equal(summary,summaries(raw),check_exact=False,atol=1e-10,rtol=0)


def build(output=OUTPUT):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    raw=records();summary=summaries(raw);verify(raw,summary)
    raw.to_csv(output/'trials.csv',index=False);summary.to_csv(output/'summary.csv',index=False)
    inputs=[RELEASE/b/name for b in BENCHMARKS for name in ['public_profiles.npz','evaluation.npz']]+[RELEASE/'headline.csv']
    method=dict(METHOD,sources_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'method.json').write_text(json.dumps(method,indent=2)+'\n')
    (output/'verification.json').write_text(json.dumps(dict(status='PASS',trials=len(raw),
        checks=['Independent scalar reconstruction of public selection, held-out scores, ranks and regret',
                'Both archived endpoint policies have identical genuine profiles',
                'All five average preload gaps match published no-attack baselines',
                'Shared-bootstrap summaries reconstructed']),indent=2)+'\n')
    print(summary[['benchmark','reused_score_pp','heldout_score_pp','selection_failure','heldout_regret_pp','preload_gap_mean_pp']].to_string(index=False),flush=True)
    return raw,summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    a=p.parse_args();build(a.output)
