"""Independently check extension inputs, endpoint choices, metrics and tables."""
from fractions import Fraction
from itertools import combinations
import numpy as np


def independent_feedback(cfg,q,m,aux):
    profiles=[]
    for c in cfg['categories']:
        take=q.category.to_numpy()==c
        x=m[:,take];sub=q.loc[take]
        if cfg['benchmark']!='openllm_v2':
            repeats=cfg['repeats']
            v=np.rint(x*repeats).astype(np.int64).sum(axis=1)/(x.shape[1]*repeats)
        elif c=='IFEval':
            counts=sub.instruction_count.to_numpy()
            v=(x.mean(axis=1)+(aux[:,take]*counts).sum(axis=1)/sum(counts))/2
        elif c in ('BBH','MuSR'):
            normalized=[]
            for task in sorted(sub.task.unique()):
                raw=x[:,sub.task.to_numpy()==task].mean(axis=1);chance=cfg['chance_baselines'][task]
                normalized.append(np.clip((raw-chance)/(1-chance),0,None))
            v=np.mean(normalized,axis=0)
        else:
            chance={'GPQA':.25,'MMLU-Pro':.1,'MATH Lvl 5':0}[c]
            v=np.clip((x.mean(axis=1)-chance)/(1-chance),0,None)
        profiles.append(v)
    raw=np.array(profiles).T
    if cfg['benchmark']=='openllm_v2':return np.round(raw*100,2)/100,np.round(raw.mean(axis=1)*100,2)/100
    return raw,m.mean(axis=1)


def check_choice(record,models,scores,overall):
    precision=record['public_scale']
    exact=lambda x:Fraction(round(float(x)*precision),precision)
    frontier=[i for i,row in enumerate(scores) if not any(
        all(other>=row) and any(other>row) for other in scores)]
    assert record['frontier']==[models[i] for i in frontier]
    pool=sorted(frontier,key=lambda i:(-exact(overall[i]),models[i]))[:5]
    assert record['top_five']==[models[i] for i in pool]
    options=pool if record['policy']=='top5' else frontier
    pair=min(combinations(sorted(options),2),key=lambda pair:(
        -sum(abs(exact(a)-exact(b)) for a,b in zip(scores[pair[0]],scores[pair[1]])),
        tuple(models[i] for i in pair)))
    assert record['canonical_pair']==[models[i] for i in pair]
    a,b=sorted(pair,key=lambda i:(-round(float(1-scores[i].mean()),10),models[i]))
    assert (record['base_a'],record['base_b'])==(models[a],models[b])
