"""Selection, score-gap and worked-example figures from verified local records."""

import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from scripts.helm_walkthrough import reconstruct
from scripts.selection_loss import calculate, summarize, source_paths, sha
from scripts.outcome_bank import BUDGETS, selection_rows

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
NAMES = {'livebench': 'LiveBench', 'helm_capabilities': 'HELM Capabilities',
         'helm_lite': 'HELM Lite', 'openllm_v2': 'Open LLM',
         'swe_verified': 'SWE-bench Verified'}
TASKS = {'narrative_qa': 'NarrativeQA', 'natural_qa_openbook_longans': 'NQ (open)',
         'natural_qa_closedbook': 'NQ (closed)', 'openbookqa': 'OpenBookQA',
         'mmlu': 'MMLU', 'math_chain_of_thought': 'MATH', 'gsm': 'GSM8K',
         'legalbench': 'LegalBench', 'med_qa': 'MedQA', 'wmt_14': 'WMT14'}
BLUE, GRAY, GOLD = '#9abaff', '#b4b4b4', '#d8b47a'
EXTRA = ROOT / 'results/terminal_counterexample'


def table(headers, rows, caption):
    def cell(value, tag):
        scope = ' scope="row"' if tag == 'th' else ''
        return f'<{tag}{scope}>{html.escape(str(value))}</{tag}>'
    return ('<div class="table-scroll" tabindex="0" role="region" aria-label="'+html.escape(caption)+'">'
            '<table class="result-details"><caption>{{supplementary_table_label}} '+caption+'</caption><thead><tr>'
            + ''.join('<th scope="col">'+html.escape(h)+'</th>' for h in headers)
            + '</tr></thead><tbody>'
            + ''.join('<tr>'+''.join(cell(v, 'th' if i == 0 else 'td') for i, v in enumerate(row))+'</tr>' for row in rows)
            + '</tbody></table></div>')


def save(fig, name, mobile):
    path = SITE/'figures'/(name+('-mobile' if mobile else '')+'.svg')
    fig.savefig(path, metadata={'Date': None, 'Creator': 'site/selection.py'})
    plt.close(fig)



def example_figure(demo):
    labels = ['A · Gemma 2 27B IT', 'B · DeepSeek V3', 'GPT-4o\n(no attack)', 'Final router']
    for mobile in (False, True):
        fig, ax = plt.subplots(figsize=(4.3, 4.4) if mobile else (8.3, 3.5))
        for y, row in enumerate(demo['scores']):
            ax.plot([row['heldout'], row['reused']], [y, y], color='#535353', lw=1.3)
            for key, color, dy in [('reused', BLUE, -.17), ('heldout', GOLD, .23)]:
                ax.plot(row[key], y, 'o', color=color, markersize=5)
                ax.text(row[key], y+dy, f'{row[key]:.2f}', color=color, ha='center', va='center', fontsize=9)
        ax.set_xlim(65, 77)
        ax.set_ylim(3.55, -.55)
        ax.set_yticks(range(4), labels)
        ax.set_xticks([65, 68, 71, 74, 77])
        ax.set_xlabel('Mean category score (%)', fontsize=10.5)
        ax.tick_params(length=0, pad=7, colors='#c0c4c5', labelsize=9.5)
        ax.spines[['top', 'left', 'right']].set_visible(False)
        ax.spines['bottom'].set_color('#383838')
        handles = [Line2D([], [], color=c, marker='o', ls='', label=l) for c, l in [(BLUE, 'Reused'), (GOLD, 'Held out')]]
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1.2), ncol=2, frameon=False, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, .94])
        save(fig, 'walkthrough', mobile)


def secondary_tables(read_csv, read_json, fact):
    from scripts.secondary_selection import tables, sources
    raw, summary = tables()
    raw.to_csv(SITE/'data/secondary_selection_trials.csv', index=False)
    summary.to_csv(SITE/'data/secondary_selection.csv', index=False)
    method = dict(metric='Same signed model-selection loss and common-pool rank drop as Figure 1.',
        scope='Post-hoc summaries of saved trials: static controls, full-frontier HELM Capabilities, and practical LiveBench policies. No new attack trials.',
        selection='Use only public ranking values; ties favor genuine models, then lexical order. Practical policies use their own released overall.',
        intervals='2,000 paired bootstrap resamples, seed 2026081900; descriptive pointwise 95% intervals.',
        non_leading='The separate non-leading-endpoint archive has shared-first ranks but not scores to resolve public ties. Its 58% false-winner rate is retained, not relabeled as selection loss.',
        sources_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources()})
    (SITE/'data/secondary_selection_method.json').write_text(json.dumps(method, indent=2)+'\n')
    read_json(SITE/'data/secondary_selection_method.json')
    read_csv(SITE/'data/secondary_selection.csv')
    curves = pd.DataFrame(read_csv(ROOT/'results/release_v2/budget_curves.csv'))
    policies = pd.DataFrame(read_csv(ROOT/'results/feedback_policy_v1/summary.csv'))
    def loss(study, policy, benchmark, budget):
        r = summary[(summary.study == study) & (summary.policy == policy) &
                    (summary.benchmark == benchmark) & (summary.new_submissions == budget)].iloc[0]
        value = float(r.model_selection_loss_pp)
        fact(f'secondary_{study}_{policy}_{benchmark}_{budget}', value,
             f'secondary_selection.csv: {study}, {policy}, {benchmark}, {budget}', f'{value:+.2f}')
        return f'{value:+.2f}'
    controls = [[name, loss('original', 'top5', b, 1), loss('original', 'top5', b, 128)] for b, name in NAMES.items()]
    bank = pd.DataFrame(read_csv(ROOT/'results/outcome_banks/terminal_science/summary.csv')).set_index('new_submissions')
    controls.append(['Terminal-Bench-Science',f"{float(bank.loc['1','model_selection_loss_pp']):+.2f}",f"{float(bank.loc['128','model_selection_loss_pp']):+.2f}"])
    frontier = []
    for policy, label in [('top5', 'Top five on the frontier'), ('full_frontier', 'Entire frontier')]:
        r = curves[(curves.benchmark == 'helm_capabilities') & (curves.policy == policy) &
                   (curves.new_submissions.astype(int) == 2048)].iloc[0]
        frontier.append([label, f'{float(r.score_gap_mean_pp):.2f}', loss('original', policy, 'helm_capabilities', 2048)])
    practical = []
    for policy, label in [('task', '18 task scores'), ('category', '6 category scores'), ('aggregate', '1 overall score')]:
        r = policies[(policies.policy == policy) & (policies.new_submissions.astype(int) == 16)].iloc[0]
        practical.append([label, f'{100*float(r.false_winner):.1f}%', loss('practical', policy, 'livebench', 128)])
    return dict(
        static_loss_table=table(['Benchmark', 'Static (1 submission)', 'Attack (128 submissions)'], controls,
            'Mean model-selection loss (pp), each relative to the same no-attack choice. Negative values mean better selection.'),
        frontier_loss_table=table(['Models considered', 'Score gap (pp)', 'Selection loss (pp)'], frontier,
            'HELM Capabilities at 2,048 submissions. Both metrics average all 250 trials.'),
        practical_loss_table=table(['Feedback', 'False winners at 16', 'Selection loss at 128 (pp)'], practical,
            'Paired LiveBench policy comparison. Selection loss averages all 250 trials.'))


def build(read_csv, read_json, fact):
    bank_path = ROOT/'results/outcome_banks/terminal_science'
    raw = pd.concat([calculate(budgets=BUDGETS).assign(eligible=True), selection_rows(bank_path)], ignore_index=True)
    summary = summarize(raw)
    raw.to_csv(SITE/'data/selection_loss_trials.csv', index=False)
    summary.to_csv(SITE/'data/selection_loss.csv', index=False)
    curves = pd.DataFrame(read_csv(ROOT/'results/release_v2/budget_curves.csv'))
    curves = curves[curves.policy == 'top5'].copy()
    for column in curves.columns.difference(['benchmark', 'policy']):
        curves[column] = pd.to_numeric(curves[column])

    sources = source_paths()+[ROOT/'scripts/selection_loss.py', ROOT/'scripts/no_attack.py',
                             ROOT/'scripts/helm_walkthrough.py', ROOT/'scripts/outcome_bank.py',
                             bank_path/'trials.csv', bank_path/'evaluation.npz', Path(__file__)]
    method = dict(metric='Held-out score of the no-attack public choice minus the post-attack public choice; signed mean over eligible paired trials (250 per snapshot, 212 for Terminal-Bench-Science). Budget zero describes all 250 original choices; budget one is static routing.',
        selection='Public winner among genuine models before attack; genuine models plus final router after attack. Ties at ten-decimal loss precision favor genuine models, then lexical order.',
        ranks='Post-attack choice rank minus no-attack choice rank, both held out against the same genuine models plus final router. Competition ties share rank. Pool sizes differ across benchmarks.',
        intervals='2,000 paired bootstrap resamples, seed 2026081900; pointwise descriptive 95% percentile intervals.',
        score_gap='Maximum absolute gap between separate running-best histories, averaged across eligible trials.',
        sources_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources})
    (SITE/'data/selection_loss_method.json').write_text(json.dumps(method, indent=2)+'\n')
    read_json(SITE/'data/selection_loss_method.json')
    read_csv(SITE/'data/selection_loss.csv')
    before = summary[summary.new_submissions == 0].set_index('benchmark')
    after = summary[summary.new_submissions == 128].set_index('benchmark')
    rows = []
    for b, name in NAMES.items():
        r = after.loc[b]
        rows.append([name, f'{before.loc[b].heldout_score_pp:.2f}', f'{r.heldout_score_pp:.2f}',
                     f'{r.model_selection_loss_pp:.2f} [{r.model_selection_loss_pp_low:.2f}, {r.model_selection_loss_pp_high:.2f}]',
                     f'{r.rank_drop:+.2f} [{r.rank_drop_low:+.2f}, {r.rank_drop_high:+.2f}]'])
        fact('selection_loss_'+b, float(r.model_selection_loss_pp), 'selection_loss.csv: '+b+', 128', f'{r.model_selection_loss_pp:.2f}')
    bank = pd.DataFrame(read_csv(ROOT/'results/outcome_banks/terminal_science/summary.csv'))
    for column in bank.columns.difference(['benchmark', 'policy']):
        bank[column] = pd.to_numeric(bank[column])
    h = bank[bank.new_submissions == 128].iloc[0]
    rows.append(['Terminal-Bench-Science', f'{h.no_attack_t_pp:.2f}', f'{h.selected_t_pp:.2f}',
                 f'{h.model_selection_loss_pp:.2f} [{h.model_selection_loss_pp_low:.2f}, {h.model_selection_loss_pp_high:.2f}]',
                 f'{h.rank_drop:+.2f} [{h.rank_drop_low:+.2f}, {h.rank_drop_high:+.2f}]'])
    fact('selection_loss_terminal_science',float(h.model_selection_loss_pp), 'outcome_summary.csv: terminal_science, 128',f'{h.model_selection_loss_pp:.2f}')
    reference = pd.DataFrame(read_csv(ROOT/'results/no_attack_v1/summary.csv'))
    no_attack_rows = [[NAMES[r.benchmark], f'{float(r.reused_score_pp):.2f}', f'{float(r.heldout_score_pp):.2f}'] for r in reference.itertuples()]
    base = bank[bank.new_submissions == 0].iloc[0]
    no_attack_rows.append(['Terminal-Bench-Science',f'{base.no_attack_s_pp:.2f}',f'{base.no_attack_t_pp:.2f}'])
    plotted = []
    for benchmark in NAMES:
        losses = summary[summary.benchmark == benchmark].set_index('new_submissions')
        for row in curves[curves.benchmark == benchmark].itertuples():
            if row.new_submissions < 2: continue
            loss = losses.loc[row.new_submissions]
            plotted.append(dict(benchmark=benchmark,new_submissions=row.new_submissions,trials=250,
                model_selection_loss_pp=loss.model_selection_loss_pp,model_selection_loss_pp_low=loss.model_selection_loss_pp_low,
                model_selection_loss_pp_high=loss.model_selection_loss_pp_high,score_gap_mean_pp=row.score_gap_mean_pp,
                score_gap_mean_pp_low=row.score_gap_mean_pp_ci_low,score_gap_mean_pp_high=row.score_gap_mean_pp_ci_high,
                baseline_gap_pp=row.preload_mean_pp))
    for row in bank[bank.new_submissions >= 2].itertuples():
        plotted.append(dict(benchmark=row.benchmark,new_submissions=row.new_submissions,trials=row.eligible_trials,
            model_selection_loss_pp=row.model_selection_loss_pp,model_selection_loss_pp_low=row.model_selection_loss_pp_low,
            model_selection_loss_pp_high=row.model_selection_loss_pp_high,score_gap_mean_pp=row.score_gap_mean_pp,
            score_gap_mean_pp_low=row.score_gap_mean_pp_low,score_gap_mean_pp_high=row.score_gap_mean_pp_high,
            baseline_gap_pp=row.genuine_gap_mean_pp))
    pd.DataFrame(plotted).to_csv(SITE/'data/benchmark_curves.csv',index=False)
    demo = reconstruct()
    (SITE/'data/walkthrough.json').write_text(json.dumps(demo, indent=2)+'\n')
    read_json(SITE/'data/walkthrough.json')
    example_figure(demo)
    tasks = {t['task']: t for t in demo['tasks']}
    returned = [['A']+[f"{t['a']:.2f}" for t in demo['tasks']], ['B']+[f"{t['b']:.2f}" for t in demo['tasks']]]
    returned += [[f'Candidate {i+1}']+[f"{t['candidates'][i]:.2f}" for t in demo['tasks']] for i in range(demo['budget']-1)]
    small = []
    for label, key, i in [('A', 'a', None), ('B', 'b', None), ('Candidate 1', 'candidates', 0), ('Candidate 2', 'candidates', 1), ('Candidate 3', 'candidates', 2)]:
        values = [tasks[c][key] if i is None else tasks[c][key][i] for c in ['legalbench', 'mmlu']]
        small.append([label]+[f'{v:.2f}' for v in values])
    task = tasks[demo['prompt']['task']]
    alpha = (task['candidates'][0]-(task['a']+task['b'])/2)/100
    def example_fact(name, value, display):
        return fact('example_'+name, value, 'walkthrough.json: trial '+str(demo['trial']), display)
    extra = pd.DataFrame(read_csv(EXTRA/'run/summary.csv'))
    for column in extra.columns.difference(['benchmark']):
        extra[column] = pd.to_numeric(extra[column])
    assert read_json(EXTRA/'run/verification.json')['status'] == 'PASS'
    terminal = extra[(extra.benchmark == 'terminal_bench_4') & (extra.new_submissions == 128)].iloc[0]
    return dict(
        **secondary_tables(read_csv, read_json, fact),
        selection_loss_table=table(['Benchmark', 'No attack (%)', 'With attack (%)', 'Loss (pp) [95% interval]', 'Rank drop [95% interval]'], rows, 'Held-out performance of the public-selected model at 128 submissions. Terminal-Bench-Science uses 212 eligible pairs, including its no-attack reference.'),
        no_attack_table=table(['Benchmark', 'Reused (%)', 'Held out (%)'], no_attack_rows, 'Genuine public winners before any new submissions; means over 250 trials.'),
        example_feedback=table(['Submission', 'LegalBench (%)', 'MMLU (%)'], small, 'Two of the ten returned category scores; first three of 31 random routers.').replace('<table class="result-details">', '<table class="result-details router-scores">').replace('{{supplementary_table_label}}', '{{table_label}}'),
        example_all_feedback=table(['Submission']+[TASKS[t['task']] for t in demo['tasks']], returned, 'All returned category scores in the worked example (%).'),
        example_alpha=example_fact('weight', alpha, f'{alpha:+.5f}'),
        example_vote=example_fact('vote', demo['prompt']['vote_pp']/200, f"{demo['prompt']['vote_pp']/200:+.5f}"),
        example_loss=example_fact('loss', demo['model_selection_loss_pp'], f"{demo['model_selection_loss_pp']:.2f}"),
        example_scores=' to '.join(f"{r['heldout']:.2f}%" for r in demo['scores'][2:]),
        example_rule=html.escape(demo['selection_rule']),
        terminal_gain=fact('terminal_gain', -float(terminal.model_selection_loss_pp), 'agent follow-up summary: terminal_bench_4, 128', f'{-terminal.model_selection_loss_pp:.2f}'),
        terminal_false=fact('terminal_false', float(terminal.router_false_winner), 'agent follow-up summary: terminal_bench_4, 128', f'{100*terminal.router_false_winner:.1f}%'),
    )
