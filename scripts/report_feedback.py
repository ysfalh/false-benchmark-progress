"""Regenerate controlled-study tables from canonical trial records (no experiments)."""
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark_progress.benchmarks import BUDGETS, ROOT
from benchmark_progress.reporting import METRICS, summarize, bootstrap_stat
from benchmark_progress.feedback_dimension import RELEASE, DIMENSIONS, definitions


def tables(raw):
    groups = definitions()
    unique = summarize(raw); logical = []
    for g in groups:
        frame = unique[unique.policy == g['computed_condition']].copy()
        frame['policy'] = g['condition'];frame['partition'] = g['partition'];frame['dimension'] = g['dimension']
        frame['released_score_coordinates'] = frame.dimension*(frame.new_submissions-1)
        logical.append(frame)
    partitions = pd.concat(logical,ignore_index=True)
    indices = np.random.default_rng(2026081900).integers(0,250,(2000,250))
    rows, cached = [], {}
    for d in DIMENSIONS:
        selected = [g for g in groups if g['dimension']==d]
        for k in BUDGETS:
            row = {'dimension':d,'new_submissions':k,'released_score_coordinates':d*(k-1),'trials':250,'predeclared_partitions':3}
            for metric,(column,statistic) in METRICS.items():
                stats = [bootstrap_stat(raw[(raw.policy==g['computed_condition']) & (raw.new_submissions==k)].sort_values('trial')[column].to_numpy(float),indices,statistic,.025) for g in selected]
                points = [r[0] for r in stats];samples = np.mean([r[3] for r in stats],axis=0)
                row.update({metric:np.mean(points),metric+'_ci_low':np.quantile(samples,.025),metric+'_ci_high':np.quantile(samples,.975),metric+'_partition_min':min(points),metric+'_partition_max':max(points)})
                cached[d,k,metric] = (np.mean(points),samples)
            row['score_gap_above_preload_pp'] = row['score_gap_mean_pp']-row['preload_mean_pp'];rows.append(row)
    curves = pd.DataFrame(rows);baseline = curves.preload_mean_pp.iloc[0];crossings = []
    for label,table,keys in [('partition',partitions,['dimension','partition']),('dimension_mean',curves,['dimension'])]:
        for key,g in table.groupby(keys):
            key = key if isinstance(key,tuple) else (key,)
            row = {'summary':label,'dimension':int(key[0]),'partition':key[1] if len(key)>1 else -1}
            for name,metric,target in [('false90','first_s_not_t',.9),('score_gap_plus1','score_gap_mean_pp',baseline+1),('score_gap_plus2','score_gap_mean_pp',baseline+2)]:
                for candidates_only in [False,True]:
                    suffix = '_random' if candidates_only else ''
                    hits = g[(g[metric]>=target) & ((g.new_submissions>=2) if candidates_only else True)].sort_values('new_submissions')
                    row[name+'_k'+suffix] = float(hits.new_submissions.iloc[0]) if len(hits) else np.nan
                    row[name+'_I'+suffix] = float(hits.released_score_coordinates.iloc[0]) if len(hits) else np.nan
            crossings.append(row)
    matches = [];points = {}
    for d in DIMENSIONS:
        for k in BUDGETS[1:]: points.setdefault(d*(k-1),[]).append((d,k))
    for count,pairs in sorted(points.items()):
        for (d1,k1),(d2,k2) in combinations(pairs,2):
            for metric,column,scale in [('first_s_not_t','first_s_not_t',100),('score_gap_mean_pp','score_gap_pp',1)]:
                p1,b1=cached[d1,k1,metric];p2,b2=cached[d2,k2,metric]
                lo,hi = np.quantile(b2*scale-b1*scale,[.025,.975])
                matches.append({'released_score_coordinates':count,'lower_dimension':d1,'higher_dimension':d2,'lower_d_submissions':k1,'higher_d_submissions':k2,'metric':column,'higher_minus_lower_pp':p2*scale-p1*scale,'ci_low':lo,'ci_high':hi})
    return {'dimension_budget_curves':curves,'partition_budget_curves':partitions,'thresholds':pd.DataFrame(crossings),'static_controls':curves[curves.new_submissions==1],'exact_coordinate_matches':pd.DataFrame(matches)}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'runs/feedback_tables');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    for name,table in tables(pd.read_parquet(RELEASE/'trials.parquet')).items():
        table.to_csv(args.output/(name+'.csv'),index=False)
    print(f'Saved regenerated tables to {args.output}')
