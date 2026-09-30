"""Tables and intervals from the complete paired trial records."""

import numpy as np
import pandas as pd

def bootstrap_stat(
    values: np.ndarray,
    indices: np.ndarray,
    statistic: str,
    alpha: float,
) -> tuple[float, float, float, np.ndarray]:
    if statistic == "mean":
        point = float(values.mean())
        boot = values[indices].mean(axis=1)
    elif statistic == "median":
        point = float(np.median(values))
        boot = np.median(values[indices], axis=1)
    else:
        raise ValueError(statistic)
    return point, float(np.quantile(boot, alpha)), float(np.quantile(boot, 1.0 - alpha)), boot

METRICS = {"score_gap_mean_pp": ("score_gap_pp", "mean"),
           "preload_mean_pp": ("genuine_preload_score_gap_pp", "mean"),
           "first_s": ("first_s", "mean"), "first_t": ("first_t", "mean"),
           "first_s_not_t": ("first_s_not_t", "mean"),
           "median_rank_s": ("rank_s", "median"), "median_rank_t": ("rank_t", "median"),
           "mean_heldout_regret_pp": ("heldout_regret_pp", "mean"),
           "mean_ranking_regret_pp": ("ranking_regret_pp", "mean")}

def summarize(raw):
    rows = []
    # Fixed, common trial resamples; no outcomes determine any choice.
    indices = np.random.default_rng(2026081900).integers(0,250,(2000,250))
    for (benchmark,policy,k), frame in raw.groupby(["benchmark","policy","new_submissions"]):
        frame = frame.sort_values("trial")
        assert frame.trial.tolist() == list(range(250))
        row = {"benchmark": benchmark, "policy": policy, "new_submissions": k,
               "original_count_equivalent": k+2, "trials": len(frame)}
        for name,(column,statistic) in METRICS.items():
            point,low,high,_ = bootstrap_stat(frame[column].to_numpy(float),indices,statistic,.025)
            row.update({name:point,name+"_ci_low":low,name+"_ci_high":high})
        row["score_gap_above_preload_pp"] = row["score_gap_mean_pp"]-row["preload_mean_pp"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_tables(raw, output):
    output.mkdir(parents=True, exist_ok=False)
    summary = summarize(raw)
    summary.to_csv(output/'budget_curves.csv', index=False)
    headline, crossings = [], []
    for (benchmark, policy), frame in summary.groupby(['benchmark', 'policy'], sort=False):
        group = frame.set_index('new_submissions')
        static, end = group.loc[1], group.loc[2048]
        headline.append({'benchmark': benchmark, 'policy': policy,
            'static_false_selection': static.first_s_not_t,
            'queried_false_selection': end.first_s_not_t,
            'score_gap_baseline_pp': end.preload_mean_pp,
            'static_score_gap_pp': static.score_gap_mean_pp,
            'queried_score_gap_pp': end.score_gap_mean_pp,
            'queried_score_gap_above_preload_pp': end.score_gap_above_preload_pp,
            'queried_median_t_rank': end.median_rank_t,
            'queried_mean_t_regret_pp': end.mean_heldout_regret_pp,
            'queried_mean_ranking_regret_pp': end.mean_ranking_regret_pp})
        for metric, threshold in [('score_gap_above_preload_pp', 2.), ('first_s_not_t', .9)]:
            random = group.drop(index=1)
            reached = random.index[random[metric] >= threshold].tolist()
            crossings.append({'benchmark': benchmark, 'policy': policy, 'metric': metric,
                'threshold': threshold, 'smallest_tested_random_budget': min(reached) if reached else None,
                'static_crosses': bool(static[metric] >= threshold)})
    pd.DataFrame(headline).to_csv(output/'headline.csv', index=False)
    crossing_table = pd.DataFrame(crossings)
    crossing_table['smallest_tested_random_budget'] = crossing_table['smallest_tested_random_budget'].astype('Int64')
    crossing_table.to_csv(output/'threshold_crossings.csv', index=False)


def write_maintainer_report(report, output):
    """Render evaluator-approved aggregates; never accepts item scores or a scorer."""
    import html
    import json
    from pathlib import Path
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    esc = lambda value: html.escape(str(value))
    title = report['name'] + ' — feedback stress test'
    blocks = []; markdown = ['# ' + title, '']
    def paragraph(text):
        blocks.append('<p>'+esc(text)+'</p>'); markdown.extend([text,''])
    def heading(text):
        blocks.append('<h2>'+esc(text)+'</h2>'); markdown.extend(['## '+text,''])
    def table(headers, rows):
        blocks.append('<div class="table-scroll" tabindex="0"><table><thead><tr>'+''.join('<th scope="col">'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>')
        markdown.extend(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |'])
        markdown.extend('| '+' | '.join(str(x).replace('|','\\|') for x in row)+' |' for row in rows); markdown.append('')
    paragraph(report['scope'])
    heading('Feedback and coverage')
    paragraph(report['feedback'])
    table(['Coverage','Value'],list(report['coverage'].items()))
    heading('Static baseline and attack')
    paragraph('A: '+report['endpoints'][0]+'. B: '+report['endpoints'][1]+'. '+report['endpoint_rule'])
    paragraph('The static router selects the stronger endpoint for each allowed coordinate. The attack combines random prompt-hash routes using only returned scores; its final response cannot change the router.')
    headers=['Outcome','Static (1 submission)',f'Attack ({report["budget"]} submissions)']
    fields=[('Reused score (%)','reused_score_pp'),('Held-out score (%)','heldout_score_pp'),
            ('Reused rank','rank_s'),('Held-out rank','rank_t'),('False winner','false_winner'),
            ('Held-out regret (pp)','heldout_regret_pp')]
    def display(value):
        return ('Yes' if value else 'No') if isinstance(value,bool) else f'{value:.3f}' if isinstance(value,float) else str(value)
    table(headers,[[name,display(report['static'][key]),display(report['attack'][key])] for name,key in fields])
    paragraph(f"Attack minus static: reused {report['attack']['reused_score_pp']-report['static']['reused_score_pp']:+.3f} pp; held-out {report['attack']['heldout_score_pp']-report['static']['heldout_score_pp']:+.3f} pp.")
    paragraph(report['utility'])
    heading('Reading the outcomes')
    paragraph('False winner means tied for first or better than every genuine model on the allowed public score, but strictly below a genuine model on held-out data. Ranks compare genuine models plus the final router. Held-out regret is the best genuine score minus the router score; a negative value means the router exceeds every genuine model. Scores and gains use unrounded evaluator aggregates.')
    for note in report['notes']: paragraph(note)
    heading('Reproduce this run')
    paragraph(report['reproduction']['command'])
    table(['Setting','Value'],[(k,v) for k,v in report['reproduction'].items() if k!='command'])
    paragraph('Private item scoring stays inside the evaluator. This report contains only the aggregates the evaluator explicitly returns. The two-callback adapter is an auditable boundary for trusted local Python, not a sandbox for hostile code.')
    css='''body{margin:0;background:#f6f3ec;color:#202c30;font:17px/1.6 system-ui,sans-serif}main{max-width:960px;margin:0 auto;padding:52px 28px 80px}h1{font:600 clamp(30px,5vw,44px)/1.15 Georgia,serif;max-width:850px}h2{margin-top:38px;font:600 26px/1.3 Georgia,serif}p{max-width:82ch}a{color:#15616b}table{border-collapse:collapse;width:100%;font-size:15px}th,td{text-align:left;border-bottom:1px solid #d2d5cf;padding:12px 15px;vertical-align:top}th{background:#e7ede7}td{overflow-wrap:anywhere}.table-scroll{overflow-x:auto}code{overflow-wrap:anywhere}nav{font-size:14px;letter-spacing:.04em} @media(max-width:600px){main{padding:28px 16px}th,td{padding:10px;min-width:100px}}'''
    document='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+esc(title)+'</title><style>'+css+'</style><main><nav>BENCHMARK REUSE / MAINTAINER REPORT</nav><h1>'+esc(title)+'</h1>'+''.join(blocks)+'</main></html>'
    (output/'report.html').write_text(document)
    (output/'report.md').write_text('\n'.join(markdown)+'\n')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
