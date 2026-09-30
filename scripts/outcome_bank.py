"""Replay and summarize task-weighted banks of repeated binary outcomes."""
from functools import reduce
from math import gcd
from types import SimpleNamespace
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark_progress.attacks import select_endpoints, fit_task_rule
from benchmark_progress.routers import build_gate_bank
from benchmark_progress.extensions import partition
from benchmark_progress.evaluation import compact_trace, outcomes
from benchmark_progress.benchmarks import ROOT

BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
MEANS = ('model_selection_loss_pp', 'rank_drop', 'no_attack_s_pp', 'no_attack_t_pp',
         'selected_s_pp', 'selected_t_pp', 'router_s_pp', 'router_t_pp',
         'false_winner', 'selected_router', 'loss_vs_static_pp')


def load(benchmark):
    base = ROOT/'results/outcome_banks'/benchmark
    config = json.loads((ROOT/'configs'/f'{benchmark}.json').read_text())
    data = json.loads((base/'inputs.json').read_text())
    q = pd.DataFrame(dict(question_id=data['question_ids'], content_hash=data['prompt_hashes'],
                          category=data['item_coordinates'], task=data['item_coordinates']))
    counts = np.rint(np.asarray(data['scores'])*data['repeats']).astype(np.int64)
    np.testing.assert_allclose(counts/data['repeats'], data['scores'], rtol=0, atol=1e-14)
    return config, q, counts


class Surface:
    def __init__(self, domains, coordinates, repeats):
        self.policy = 'historical_domain'
        self.domains = np.asarray(domains)
        self.coordinates = tuple(coordinates)
        self.groups = [np.flatnonzero(self.domains == c) for c in coordinates]
        self.counts = [len(g) for g in self.groups]
        self.repeats = repeats
        self.scale = reduce(lambda a,b: a*b//gcd(a,b),
                            [repeats*len(domains), *[repeats*n for n in self.counts]], 1)

    def release(self, counts):
        raw = np.stack([np.asarray(counts)[...,g].sum(axis=-1) for g in self.groups], axis=-1)
        return raw/(np.asarray(self.counts)*self.repeats), np.asarray(counts).sum(axis=-1)/(len(self.domains)*self.repeats)


def public_decisions(models, surface, genuine, candidate_callback, final_callback):
    """Attack-side function: only allowed aggregates and public metadata enter."""
    profile, overall = genuine
    before = int(np.argmin(np.round(1-overall, 10)))
    record = dict(policy=surface.policy, coordinates=list(surface.coordinates),
                  weights=[n/sum(surface.counts) for n in surface.counts],
                  coordinate_task_counts=surface.counts, scale=surface.scale,
                  genuine_profiles=profile.tolist(), genuine_overall=overall.tolist(),
                  no_attack_index=before, no_attack_model=models[before])
    try:
        choice = select_endpoints(models, profile, 'top5', scale=surface.scale, overall=overall)
    except ValueError as e:
        if surface.policy != 'historical_domain' or 'Fewer than two' not in str(e):
            raise
        dominates = np.all(profile[:,None] >= profile[None,:],axis=2) & np.any(profile[:,None] > profile[None,:],axis=2)
        record.update(eligible=False, reason=str(e), frontier=[models[i] for i in np.flatnonzero(~dominates.any(axis=0))])
        return record, None
    a,b = [models.index(choice[k]) for k in ('base_a','base_b')]
    candidates, candidate_overall = candidate_callback(a,b)
    rules = [fit_task_rule(profile[a],profile[b],candidates[:k-1],surface.scale) for k in BUDGETS]
    finals, final_overall = final_callback(a,b,rules)
    selected = [int(np.argmin(np.round(1-np.r_[overall,value],10))) for value in final_overall]
    record.update(eligible=True, choice=choice, candidate_feedback=candidates.tolist(),
                  candidate_overall=candidate_overall.tolist(), final_feedback=finals.tolist(),
                  final_overall=final_overall.tolist(), selected_indices=selected,
                  rules=[dict(weights=r.weights, fallback=r.fallback, candidate_count=r.candidate_count,
                              record=r.record()) for r in rules])
    return record, (a,b,rules)


def route(rules, gates, ties, domains, coordinates):
    signs = np.empty((len(rules), len(domains)), dtype=np.int8)
    for j,c in enumerate(coordinates):
        pick = np.ones(len(domains),bool) if c == 'overall' else np.asarray(domains)==c
        for i,r in enumerate(rules):
            signs[i,pick] = r.route(gates[:,pick],ties[pick],j)
    return signs


def make_rows(dataset, trial, record, ev):
    policy = record['policy']; n = len(record['genuine_overall']); before = record['no_attack_index']
    pg = np.array(record['genuine_overall']); gt = ev['genuine_t']; gs = ev['genuine_s']
    preload = float(np.max(np.abs(np.minimum.accumulate(1-gs)-np.minimum.accumulate(1-gt))))*100
    base = dict(benchmark=dataset, policy=policy, trial=trial, eligible=record['eligible'],
                new_submissions=0, no_attack_model=record['no_attack_model'],
                selected_model=record['no_attack_model'], selected_router=False,
                no_attack_s_pp=100*pg[before], no_attack_t_pp=100*gt[before],
                selected_s_pp=100*pg[before], selected_t_pp=100*gt[before],
                router_s_pp=np.nan, router_t_pp=np.nan, model_selection_loss_pp=0.,
                rank_drop=0., false_winner=False, score_gap_pp=preload,
                genuine_gap_pp=preload, loss_vs_static_pp=0.)
    if not record['eligible']:
        return [base] + [dict(base,new_submissions=k,model_selection_loss_pp=np.nan,rank_drop=np.nan,
                              false_winner=np.nan,selected_router=np.nan,selected_s_pp=np.nan,
                              selected_t_pp=np.nan,score_gap_pp=np.nan,loss_vs_static_pp=np.nan,
                              selected_model='INELIGIBLE') for k in BUDGETS]
    ps = np.r_[gs,gs[ev['endpoints']],ev['candidate_s']][:,None]
    pt = np.r_[gt,gt[ev['endpoints']],ev['candidate_t']][:,None]
    trace = compact_trace(ps,pt,ev['final_s'][:,None],ev['final_t'][:,None],n)
    original = outcomes(dataset,policy,trial,trace,budgets=BUDGETS)
    rows = [base]
    for j,k in enumerate(BUDGETS):
        chosen = record['selected_indices'][j]
        ss = np.r_[pg,record['final_overall'][j]]; tt = np.r_[gt,ev['final_t'][j]]
        losses = np.round(1-tt,10)
        ranks = 1+(losses[:,None]>losses[None,:]).sum(axis=1)
        rs = 1+int((np.round(1-pg,10)<round(1-float(ss[-1]),10)).sum())
        rt = int(ranks[-1])
        rows.append(dict(base,new_submissions=k,selected_model='Final router' if chosen==n else record['no_attack_model'],
                         selected_router=chosen==n, selected_s_pp=100*ss[chosen],selected_t_pp=100*tt[chosen],
                         router_s_pp=100*ss[-1],router_t_pp=100*tt[-1],
                         model_selection_loss_pp=100*(gt[before]-tt[chosen]),rank_drop=float(ranks[chosen]-ranks[before]),
                         false_winner=rs==1 and rt>1,score_gap_pp=original[j]['score_gap_pp']))
    for row in rows[1:]:
        row['loss_vs_static_pp'] = row['model_selection_loss_pp']-rows[1]['model_selection_loss_pp']
    return rows


def summarize(frame):
    indices = np.random.default_rng(2026081900).integers(0,250,(2000,250))
    rows = []
    for (dataset,policy,k),f in frame.groupby(['benchmark','policy','new_submissions'],sort=True):
        f = f.sort_values('trial'); assert f.trial.tolist() == list(range(250))
        eligible = f.eligible.to_numpy(bool) if k else np.ones(250,bool)
        r = dict(benchmark=dataset,policy=policy,new_submissions=int(k),eligible_trials=int(eligible.sum()),attempted_trials=250)
        for key in [*MEANS,'score_gap_pp','genuine_gap_pp']:
            v=f[key].to_numpy(float);v[~eligible]=np.nan
            if not np.isfinite(v).any():
                point=lo=hi=np.nan
            else:
                point=np.nanmean(v);lo,hi=np.quantile(np.nanmean(v[indices],axis=1),[.025,.975])
            target={'score_gap_pp':'score_gap_mean_pp','genuine_gap_pp':'genuine_gap_mean_pp'}.get(key,key)
            r.update({target:point,target+'_low':lo,target+'_high':hi})
        rows.append(r)
    return pd.DataFrame(rows)

def replay(benchmark, trial_ids=None):
    config, q, counts = load(benchmark)
    models = config['models']
    bank = build_gate_bank([SimpleNamespace(name='all',content_hashes=tuple(q.content_hash))],
                           max(BUDGETS)-1, config['routing_seed'], config['tie_seed'])
    gates, ties = bank.candidate_gates['all'], bank.tie_gates['all']
    for trial in range(config['trials']) if trial_ids is None else trial_ids:
        reused, _ = partition(config, q, trial)
        mask = q.question_id.isin(reused).to_numpy()
        ms = counts[:,mask]
        domains = q.loc[mask,'category'].to_numpy()
        surface = Surface(domains, config['categories'], config['repeats'])
        def candidates(a,b):
            return surface.release(np.where(gates[:,mask] == 1, ms[a], ms[b]))
        def finals(a,b,rules):
            signs = route(rules,gates[:,mask],ties[mask],domains,surface.coordinates)
            return surface.release(np.where(signs == 1, ms[a], ms[b]))
        public_metadata = SimpleNamespace(policy=surface.policy,coordinates=surface.coordinates,
                                          counts=surface.counts,scale=surface.scale)
        record, decision = public_decisions(models,public_metadata,surface.release(ms),candidates,finals)
        record.update(benchmark=benchmark,trial=trial)
        # Fitting and public selection finish before held-out outcomes are read.
        mt = counts[:,~mask]
        ev = dict(s_mask=mask,genuine_s=ms.sum(axis=1)/(ms.shape[1]*config['repeats']),
                  genuine_t=mt.sum(axis=1)/(mt.shape[1]*config['repeats']))
        if decision is not None:
            a,b,rules = decision
            signs = route(rules,gates,ties,q.category.to_numpy(),surface.coordinates)
            finalcounts = np.where(signs == 1,counts[a],counts[b])
            candidatecounts = np.where(gates == 1,counts[a],counts[b])
            ev['endpoints'] = np.array([a,b])
            for name, values in [('candidate',candidatecounts),('final',finalcounts)]:
                ev[name+'_s'] = values[:,mask].sum(axis=1)/(mask.sum()*config['repeats'])
                ev[name+'_t'] = values[:,~mask].sum(axis=1)/((~mask).sum()*config['repeats'])
        yield record, ev, make_rows(benchmark,trial,record,ev)


def run(benchmark, output, trials=None):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    rows, arrays = [], {}
    with gzip.open(output/'routers.jsonl.gz','wt') as f:
        for record, ev, outcomes_ in replay(benchmark,trials):
            f.write(json.dumps(record)+'\n')
            arrays.update({f"{record['trial']}_{key}": value for key,value in ev.items()})
            rows.extend(outcomes_)
    np.savez_compressed(output/'evaluation.npz',**arrays)
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'trials.csv',index=False)
    if trials is None:
        summarize(frame).to_csv(output/'summary.csv',index=False)
    print(f'Saved {len(rows)} outcomes to {output}')


def severity(base):
    """False-winner rates and score gaps, conditional on eligible partitions."""
    frame = pd.read_csv(base/'trials.csv')
    with np.load(base/'evaluation.npz') as saved:
        best = {i: saved[f'{i}_genuine_t'].max()*100 for i in range(250)}
    indices = np.random.default_rng(2026081900).integers(0,250,(2000,250))
    rates, means = [], []
    for budget, f in frame[frame.new_submissions>0].groupby('new_submissions'):
        f = f.sort_values('trial')
        eligible = f.eligible.to_numpy(bool)
        false = f.false_winner.eq(True).to_numpy(bool)
        gap = np.array([best[i] for i in f.trial])-f.router_t_pp.to_numpy()
        common = dict(benchmark=f.benchmark.iloc[0],policy='top5',new_submissions=int(budget),trials=int(eligible.sum()))
        for threshold in ['any','1','2','5']:
            event = false if threshold=='any' else false & (gap>=float(threshold))
            values = np.where(eligible,event.astype(float),np.nan)
            lo,hi = np.quantile(np.nanmean(values[indices],axis=1),[.025,.975])
            rates.append(dict(common,threshold_pp=threshold,count=int(event.sum()),rate=float(np.nanmean(values)),ci_low=lo,ci_high=hi))
        values = np.where(false & eligible,gap,np.nan)
        boot = np.nanmean(values[indices],axis=1)
        boot = boot[np.isfinite(boot)]
        lo,hi = np.quantile(boot,[.025,.975])
        means.append(dict(common,false_winners=int(false.sum()),mean_deficit_pp=float(np.nanmean(values)),ci_low=lo,ci_high=hi,bootstrap_valid_resamples=len(boot)))
    return pd.DataFrame(rates),pd.DataFrame(means)


def selection_rows(base):
    """Expose bank outcomes in the shared selection-loss table format."""
    frame = pd.read_csv(base/'trials.csv')
    with np.load(base/'evaluation.npz') as saved:
        best = {i: saved[f'{i}_genuine_t'].max()*100 for i in frame.trial.unique()}
    frame = frame.rename(columns={'selected_s_pp':'reused_score_pp', 'selected_t_pp':'heldout_score_pp',
                                  'false_winner':'router_false_winner'})
    frame['eligible'] |= frame.new_submissions.eq(0)
    frame['no_attack_shortfall_pp'] = frame.trial.map(best)-frame.no_attack_t_pp
    frame['selected_shortfall_pp'] = frame.trial.map(best)-frame.heldout_score_pp
    frame['router_tied_public_first'] = ((1-frame.router_s_pp/100).round(10) ==
                                         (1-frame.no_attack_s_pp/100).round(10))
    columns = ['benchmark','trial','new_submissions','no_attack_model','selected_model','selected_router',
               'router_tied_public_first','reused_score_pp','heldout_score_pp','no_attack_shortfall_pp',
               'selected_shortfall_pp','model_selection_loss_pp','rank_drop','router_false_winner','eligible']
    return frame[columns]
