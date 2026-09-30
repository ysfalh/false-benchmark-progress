"""Reconstruct an explicitly outcome-selected HELM Lite illustration from archived trials."""
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
from benchmark_progress.attacks import fit_rule,select_pair,aggregate
from benchmark_progress.benchmarks import ROOT,load,partition,genuine_feedback,evaluate
from benchmark_progress.evaluation import compact_trace,outcomes
from benchmark_progress.routers import build_gate_bank
from scripts.no_attack import selected_outcome


def reconstruct():
    benchmark='helm_lite';k=32
    reference=pd.read_csv(ROOT/'results/no_attack_v1/trials.csv').query('benchmark == @benchmark').set_index('trial')
    attacks=pd.read_parquet(ROOT/'results/release_v2/trials.parquet').query("benchmark == @benchmark and policy == 'top5' and new_submissions == @k").set_index('trial')
    # Select the illustration by paired score loss, not by the final router's regret.
    # Derive it from canonical profiles so this reconstruction does not depend on the preview.
    selection_losses=[]
    with np.load(ROOT/'results/release_v2'/benchmark/'evaluation.npz') as archive:
        # Materialize the profiles once; archive lookups decompress whole arrays.
        z={key:archive[key] for key in
           ('policies','trials','genuine_s','genuine_t','final_s','final_t')}
        j=[1,2,4,8,16,32,64,128,256,512,1024,2048].index(k)
        for pos in np.flatnonzero(z['policies']=='top5'):
            trial_id=int(z['trials'][pos]);before=int(reference.loc[trial_id].selected_index)
            ss=np.vstack([z['genuine_s'][pos],z['final_s'][pos,j]])
            tt=np.vstack([z['genuine_t'][pos],z['final_t'][pos,j]])
            after=int(np.argmin(np.round(1-aggregate(ss,benchmark),10)))
            selection_losses.append(dict(trial=trial_id,added_shortfall_pp=100*float(tt[before].mean()-tt[after].mean())))
    choices=pd.DataFrame(selection_losses).set_index('trial')
    candidates=reference.join(attacks[['first_s_not_t','score_gap_pp']]).join(choices[['added_shortfall_pp']])
    eligible=candidates.query('heldout_rank == 1 and first_s_not_t and score_gap_pp > preload_gap_pp + 1e-10').copy()
    median=float(eligible.added_shortfall_pp.median())
    eligible['distance']=(eligible.added_shortfall_pp-median).abs()
    trial=int(eligible.sort_index().sort_values('distance',kind='stable').index[0])
    config,q,matrix=load(benchmark);q['_category_position']=q.groupby('category').cumcount()
    reused,heldout=partition(config,q,trial);mask=q.question_id.isin(reused).to_numpy()
    public,overall,coords,cats=genuine_feedback(matrix[:,mask],q.loc[mask],config)
    choice=select_pair(config['models'],public,benchmark,'top5',overall,cats)
    a,b=[config['models'].index(choice[key]) for key in ['base_a','base_b']]
    metadata=[SimpleNamespace(name=c,content_hashes=tuple(q.loc[q.category==c,'content_hash'])) for c in config['categories']]
    bank=build_gate_bank(metadata,k-1,config['routing_seed'],config['tie_seed'])
    ps,_,feedback=evaluate(matrix[:,mask],q.loc[mask],config,bank,a,b,public=True)
    n=len(config['models']);released=feedback[n+2:]
    rule=fit_rule(public[a],public[b],released,benchmark)
    _,fs,_=evaluate(matrix[:,mask],q.loc[mask],config,bank,a,b,[rule])
    # Fit above; only now score the held-out set.
    pt,ft,_=evaluate(matrix[:,~mask],q.loc[~mask],config,bank,a,b,[rule])
    metric=outcomes(benchmark,'top5',trial,compact_trace(ps,pt,fs,ft,n),budgets=[k])[0]
    old=pd.read_parquet(ROOT/'results/release_v2/trials.parquet').query("benchmark == @benchmark and policy == 'top5' and trial == @trial and new_submissions == @k").iloc[0]
    for key in ['rank_s','rank_t','first_s_not_t','heldout_regret_pp','score_gap_pp']:
        assert np.isclose(metric[key],old[key],atol=1e-10,rtol=0),key
    no_attack=selected_outcome(config['models'],overall,ps[:n],pt[:n],benchmark)
    leader=no_attack['selected_index']
    assert int(np.argmin(np.round(1-aggregate(np.vstack([ps[:n],fs]),benchmark),10)))==n
    prompt=q.loc[mask].sort_values('content_hash').iloc[0];ci=coords.index(prompt.category)
    pos=int(prompt['_category_position']);gates=bank.candidate_gates[prompt.category][:,pos]
    integers=rule.weights[ci];vote=sum(int(w)*int(g) for w,g in zip(integers,gates))
    weights=100*(2*released[:,ci]-public[a,ci]-public[b,ci])
    final=rule.route(gates[:,None],bank.tie_gates[prompt.category][pos:pos+1],ci)[0]
    rank_s=np.round(1-aggregate(np.vstack([ps[:n],fs]),benchmark),10)
    rank_t=np.round(1-aggregate(np.vstack([pt[:n],ft]),benchmark),10)
    ranks_s=[1+int(np.sum(rank_s < x)) for x in rank_s]
    ranks_t=[1+int(np.sum(rank_t < x)) for x in rank_t]
    names={'deepseek-ai/deepseek-v3':'DeepSeek V3','google/gemma-2-27b-it':'Gemma 2 27B IT','openai/gpt-4o-2024-08-06':'GPT-4o (Aug 2024)'}
    endpoint_names=[names.get(choice[key],choice[key]) for key in ['base_a','base_b']]
    scores=[]
    for label,index in [('Endpoint A',a),('Endpoint B',b),('No attack',leader)]:
        scores.append(dict(label=label,model=config['models'][index],reused=100*float(ps[index].mean()),heldout=100*float(pt[index].mean()),rank_s=ranks_s[index],rank_t=ranks_t[index]))
    scores.append(dict(label='Final router',model='router',reused=100*float(fs[0].mean()),heldout=100*float(ft[0].mean()),rank_s=ranks_s[-1],rank_t=ranks_t[-1]))
    assert abs(scores[2]['heldout']-scores[3]['heldout']-choices.loc[trial].added_shortfall_pp)<1e-10
    tasks=[dict(task=name,a=100*float(public[a,j]),b=100*float(public[b,j]),
        candidates=(100*released[:,j]).tolist(),weights=rule.weights[j]) for j,name in enumerate(coords)]
    return dict(benchmark=benchmark,trial=trial,budget=k,
        selection_rule=f'We fixed HELM Lite and a budget of 32 submissions. In {len(eligible)} trials, the original public choice stays first held out, the router becomes a false winner, and the best-score gap grows. We chose the trial nearest the median selection loss among those cases, breaking ties by trial number. That gives trial {trial}, with a loss of {median:.2f} pp. We selected this example to illustrate the failure; the main figures include all 250 trials.',
        eligible_trials=len(eligible),eligible_median_loss_pp=median,endpoint_names=endpoint_names,
        endpoints=[choice['base_a'],choice['base_b']],scores=scores,tasks=tasks,metric=metric,no_attack=no_attack,
        coverage=dict(models=n,reused=len(reused),heldout=len(heldout),coordinates=len(coords)),
        prompt=dict(hash=prompt.content_hash,task=prompt.category,gates=gates.tolist(),weights_pp=weights.tolist(),
            integer_weights=integers,integer_vote=vote,vote_pp=float(weights@gates),choice='A' if final==1 else 'B'),
        routing_seed=config['routing_seed'],tie_seed=config['tie_seed'],split_seed=config['split_seed_base']+trial,
        model_selection_loss_pp=scores[2]['heldout']-scores[3]['heldout'],
        verification='Reconstructed from item scores; archived rank, regret and score-gap metrics matched within 1e-10 pp. No attack selects the genuine public winner before held-out evaluation.')
