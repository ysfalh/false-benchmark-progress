"""Scoring adapters for Open LLM v2 and SWE-bench Verified.

Reuse the archived splitter, gate bank, endpoint selector, linear decoder, and
metrics. Only benchmark aggregation and its released precision differ.
"""
from functools import reduce
from math import gcd
import numpy as np
import pandas as pd
from .benchmarks import load as load_original

EXTENSIONS = ('openllm_v2', 'swe_verified')

def load(benchmark):
    config, q, matrix = load_original(benchmark)
    auxiliary = np.zeros_like(matrix)
    if benchmark != 'openllm_v2':
        counts = matrix*config['repeats']
        if not np.allclose(counts, np.rint(counts), rtol=0, atol=1e-12):
            raise ValueError('Coding scores must be integer successes over recorded repeats')
    if benchmark == 'openllm_v2':
        from .benchmarks import ROOT
        scores = pd.read_parquet(ROOT/'data'/f'{benchmark}_scores.parquet')
        auxiliary = scores.pivot(index='question_id', columns='model', values='instruction_score').reindex(
            index=q.question_id, columns=config['models']).to_numpy().T
        if not np.isfinite(auxiliary).all() or not ((auxiliary >= 0) & (auxiliary <= 1)).all():
            raise ValueError('Invalid instruction-level scores')
    return config, q, matrix, auxiliary

def scale(config, q):
    if config['benchmark'] == 'openllm_v2':
        return 10000
    # Exact count-derived feedback. A common integer denominator avoids binary
    # floating-point artifacts when the unchanged decoder forms rational weights.
    counts = [len(q), *q.groupby('category').size().tolist()]
    denominators = [int(n)*config['repeats'] for n in counts]
    return reduce(lambda a,b: a*b//gcd(a,b), denominators, 1)

def publish(values, config):
    return np.round(values*100, 2)/100 if config['benchmark'] == 'openllm_v2' else values.copy()

def category_score(values, auxiliary, q, config, category):
    if config['benchmark'] != 'openllm_v2':
        repeats = config['repeats']
        # Count successes exactly: summing decimals such as .2 in different
        # orders must not split genuine ties on the public Pareto frontier.
        return np.rint(values*repeats).sum(axis=-1)/(values.shape[-1]*repeats)
    if category == 'IFEval':
        weights = q.instruction_count.to_numpy(float)
        return .5*(values.mean(axis=-1)+(auxiliary*weights).sum(axis=-1)/weights.sum())
    if category in ('BBH', 'MuSR'):
        normalized = []
        for task in sorted(q.task.unique()):
            mask = (q.task == task).to_numpy()
            baseline = config['chance_baselines'][task]
            normalized.append(np.maximum(values[...,mask].mean(axis=-1)-baseline, 0)/(1-baseline))
        return np.mean(normalized, axis=0)
    raw = values.mean(axis=-1)  # MATH and GPQA official groups are item-weighted.
    baseline = {'GPQA': .25, 'MMLU-Pro': .1, 'MATH Lvl 5': 0}[category]
    return np.maximum(raw-baseline, 0)/(1-baseline)

def overall(profiles, q, config):
    if config['benchmark'] == 'openllm_v2':
        return profiles.mean(axis=-1)
    weights = np.array([(q.category == c).sum() for c in config['categories']], float)
    return profiles @ (weights/weights.sum())

def genuine_feedback(matrix, auxiliary, q, config):
    profiles = []
    for c in config['categories']:
        mask = (q.category == c).to_numpy()
        profiles.append(category_score(matrix[:,mask], auxiliary[:,mask], q.loc[mask], config, c))
    raw = np.stack(profiles, axis=1)
    return publish(raw, config), publish(overall(raw, q, config), config)

def evaluate(matrix, auxiliary, q, config, bank, a, b, rules=(), public=False):
    prefix, finals = [], []
    for coordinate, c in enumerate(config['categories']):
        mask = (q.category == c).to_numpy()
        data, aux, part = matrix[:,mask], auxiliary[:,mask], q.loc[mask]
        pos = part['_category_position'].to_numpy(int)
        gates = bank.candidate_gates[c][:,pos]
        ties = bank.tie_gates[c][pos]
        submitted = np.concatenate((data, data[[a,b]], np.where(gates == 1,data[a],data[b])))
        if c == 'IFEval':
            submitted_aux = np.concatenate((aux,aux[[a,b]],np.where(gates == 1,aux[a],aux[b])))
        else:
            submitted_aux = None
        prefix.append(category_score(submitted, submitted_aux, part, config, c))
        if rules:
            signs = np.stack([r.route(gates,ties,coordinate) for r in rules])
            routed = np.where(signs == 1,data[a],data[b])
            routed_aux = np.where(signs == 1,aux[a],aux[b]) if c == 'IFEval' else None
            finals.append(category_score(routed,routed_aux,part,config,c))
    profiles = np.stack(prefix, axis=1)
    final_profiles = np.stack(finals,axis=1) if rules else None
    # Metrics consume one official overall score, retaining their exact old
    # semantics rather than accidentally averaging unequal-sized domains.
    return (overall(profiles,q,config)[:,None],
            overall(final_profiles,q,config)[:,None] if rules else None,
            publish(profiles,config) if public else None)

def partition(config, q, trial):
    """The existing exact task-stratified split, using bitsets for large tasks.

    Same hash ordering and greedy feasibility decisions as select_by_task.
    Bitsets replace Python sets in its subset-sum DP; this matters for 12k MMLU
    rows. Neither scores nor endpoint choices enter the split.
    """
    from .splits import hash_rank
    grouped = {}
    for row in q.itertuples(index=False):
        grouped.setdefault(row.task, {}).setdefault(row.content_hash, []).append(row.question_id)
    heldout = set()
    seed = config['split_seed_base']+trial
    for task, members in sorted(grouped.items()):
        groups = [(key, tuple(sorted(ids))) for key,ids in members.items()]
        groups.sort(key=lambda x: (hash_rank(seed, f'{task}|{x[0]}'),x[0],x[1]))
        target = int(round(config['holdout_fraction']*sum(len(ids) for _,ids in groups)))
        if all(len(ids) == 1 for _,ids in groups):
            heldout.update(ids[0] for _,ids in groups[:target])
            continue
        suffix = [0]*(len(groups)+1)
        suffix[-1] = 1
        cap = (1 << (target+1))-1
        for i in range(len(groups)-1,-1,-1):
            suffix[i] = (suffix[i+1] | (suffix[i+1] << len(groups[i][1]))) & cap
        if not (suffix[0] >> target) & 1:
            raise RuntimeError(f'Cannot keep duplicates intact at target {target}')
        remaining = target
        for i,(_,ids) in enumerate(groups):
            size = len(ids)
            if size <= remaining and (suffix[i+1] >> (remaining-size)) & 1:
                heldout.update(ids)
                remaining -= size
        assert remaining == 0
    reused = set(q.question_id)-heldout
    if set(q.loc[q.question_id.isin(reused),'content_hash']) & set(q.loc[q.question_id.isin(heldout),'content_hash']):
        raise ValueError('A duplicate prompt crosses the split')
    return reused, heldout
