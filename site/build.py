"""Build the local research page from saved reports. No experiments are run."""

import csv
import hashlib
import html
import json
import os
import re
import shutil
import sys
import tempfile
from itertools import count
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "benchmark-reuse-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.score_comparison import ALPHA, COMPARISONS, RESAMPLES, SEED, compare_scores
from scripts.selection_severity import summarize as summarize_severity
from scripts.numerics import sequential_sum
RESULTS = ROOT / "results/release_v2"
NAMES = {
    "livebench": "LiveBench",
    "helm_capabilities": "HELM Capabilities",
    "helm_lite": "HELM Lite",
    "openllm_v2": "Open LLM",
    "swe_verified": "SWE-bench Verified",
}
ACCENT = "#9abaff"
SELECTION_PLOT_LIMIT = 128
SOURCES = {}
FACTS = {}


def read_csv(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def fact(key, value, source, display=None):
    """Record the exact value and its source alongside the displayed formatting."""
    display = str(value) if display is None else display
    FACTS[key] = {"value": value, "display": display, "source": source}
    return f'<span data-fact="{key}">{html.escape(display)}</span>'


def percent(value):
    return f"{100 * float(value):g}%"


def main():
    started = perf_counter()

    def progress(message):
        print(f"[{perf_counter() - started:5.1f}s] {message}", flush=True)

    progress("Loading saved benchmark results...")
    for folder in ("data", "figures"):
        (SITE / folder).mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
        "svg.fonttype": "none", "svg.hashsalt": "benchmark-reuse",
        "figure.facecolor": "#000000", "axes.facecolor": "#000000",
        "savefig.facecolor": "#000000", "text.color": "#e8e6e1",
        "axes.labelcolor": "#e8e6e1",
    })
    curves = read_csv(RESULTS / "budget_curves.csv")
    headlines = read_csv(RESULTS / "headline.csv")
    thresholds = read_csv(RESULTS / "threshold_crossings.csv")
    protocol = read_json(RESULTS / "manifest.json")
    verification = read_json(RESULTS / "verification.json")
    assert verification["status"] == "PASS"
    assert len(curves) == len(NAMES) * 2 * len(protocol["budgets"])
    assert {int(row["trials"]) for row in curves} == {250}
    budget = protocol["headline_budget"]
    groups = {}
    for benchmark in NAMES:
        for policy in ("top5", "full_frontier"):
            rows = sorted((row for row in curves if row["benchmark"] == benchmark and row["policy"] == policy),
                          key=lambda row: int(row["new_submissions"]))
            assert [int(row["new_submissions"]) for row in rows] == protocol["budgets"]
            groups[benchmark, policy] = rows

    raw_path = RESULTS / 'trials.parquet'
    SOURCES[str(raw_path.relative_to(ROOT))] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    raw = pd.read_parquet(raw_path)
    progress("Summarizing false-winner rates...")
    severity, severity_means = summarize_severity(raw)
    from scripts.outcome_bank import severity as bank_severity
    rates, means = bank_severity(ROOT/'results/outcome_banks/terminal_science')
    severity = pd.concat([severity,rates],ignore_index=True)
    severity_means = pd.concat([severity_means,means],ignore_index=True)
    severity.to_csv(SITE/'data/selection_severity.csv', index=False)
    severity_means.to_csv(SITE/'data/selection_deficits.csv', index=False)
    for benchmark in NAMES:
        actual = severity[(severity.benchmark == benchmark) & (severity.threshold_pp == 'any')].set_index('new_submissions')
        for row in groups[benchmark, 'top5']:
            matched = actual.loc[int(row['new_submissions'])]
            for new, saved in [('rate', 'first_s_not_t'), ('ci_low', 'first_s_not_t_ci_low'), ('ci_high', 'first_s_not_t_ci_high')]:
                assert abs(float(matched[new])-float(row[saved])) < 1e-12
        for limit in (SELECTION_PLOT_LIMIT, budget):
            mean = severity_means[(severity_means.benchmark == benchmark) & (severity_means.new_submissions == limit)].iloc[0]
            fact(f'selection_deficit_{benchmark}_{limit}', float(mean.mean_deficit_pp),
                 f'trials.parquet: {benchmark}, top5, {limit}; heldout_regret_pp mean conditional on first_s_not_t',
                 f'{mean.mean_deficit_pp:.2f}')
    analysis_path = ROOT/'scripts/selection_severity.py'
    SOURCES[str(analysis_path.relative_to(ROOT))] = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    severity_method = {
        'policy': 'top5', 'trials_per_benchmark': {**{b:protocol['trials'] for b in NAMES}, 'terminal_science':212},
        'event': 'first_s_not_t AND heldout_regret_pp >= threshold; any uses first_s_not_t alone',
        'rate_denominator': 'All eligible trials, including trials where the router is not a false winner. Ineligible cells are omitted inside each common 250-index resample.',
        'deficit': 'best genuine held-out score minus final router held-out score, in percentage points',
        'mean_deficit': 'conditional on first_s_not_t; ratio recomputed in each bootstrap resample',
        'thresholds_pp': [1, 2, 5], 'main_plot_max_submissions': SELECTION_PLOT_LIMIT,
        'full_plot_max_submissions': budget, 'bootstrap': protocol['bootstrap'],
        'intervals': 'pointwise percentile intervals; resamples with zero false winners omitted for conditional means',
        'scope': 'descriptive thresholds; fixed benchmark snapshots; no fresh-data inference',
        'helm_lite': 'false-winner ranks use mean win rate; score deficit uses mean category score',
        'swe_verified': 'score deficit uses instance-weighted resolution rate',
        'input_sha256': SOURCES[str(raw_path.relative_to(ROOT))],
        'analysis_sha256': SOURCES[str(analysis_path.relative_to(ROOT))],
    }
    (SITE/'data/selection_severity_method.json').write_text(json.dumps(severity_method, indent=2)+'\n')
    progress("Checking score-generalization differences...")
    comparisons = compare_scores(raw)
    comparisons.to_csv(SITE/'data/score_comparison.csv', index=False)
    comparison_method = {'metric': 'mean score generalization gap minus genuine-baseline mean, in percentage points',
        'method': 'paired basic bootstrap; one-sided Bonferroni lower bounds',
        'family': 'all queried budgets across five benchmarks, top5 policy',
        'comparisons': COMPARISONS, 'family_alpha': ALPHA, 'resamples': RESAMPLES, 'seed': SEED,
        'quantile': 'higher', 'positive_tolerance_pp': 1e-10,
        'input_sha256': SOURCES[str(raw_path.relative_to(ROOT))],
        'analysis_sha256': hashlib.sha256((ROOT/'scripts/score_comparison.py').read_bytes()).hexdigest()}
    (SITE/'data/score_comparison_method.json').write_text(json.dumps(comparison_method, indent=2)+'\n')
    threshold_text = []
    for benchmark, name in NAMES.items():
        rows = groups[benchmark, "top5"]
        control, end = rows[0], rows[-1]
        headline = next(row for row in headlines if row["benchmark"] == benchmark and row["policy"] == "top5")
        threshold = next(row for row in thresholds if row["benchmark"] == benchmark and row["policy"] == "top5" and row["metric"] == "first_s_not_t")
        # Cross-file checks only: use the saved summaries, never rescore trials.
        for column, headline_column in (("score_gap_mean_pp", "queried_score_gap_pp"), ("first_s_not_t", "queried_false_selection"), ("preload_mean_pp", "score_gap_baseline_pp")):
            assert float(end[column]) == float(headline[headline_column])
        first = int(threshold["smallest_tested_random_budget"])
        assert first == next(int(row["new_submissions"]) for row in rows[1:] if float(row["first_s_not_t"]) >= .9)
        value = fact(f"threshold_{benchmark}", first, f"threshold_crossings.csv: {benchmark}, top5, first_s_not_t")
        threshold_text.append(f'{name}, {value}')
        first_score_plus1 = next(int(row['new_submissions']) for row in rows[1:]
                                if float(row['score_gap_mean_pp'])-float(row['preload_mean_pp']) >= 1)
        plus1 = fact(f'score_threshold_plus1_{benchmark}', first_score_plus1,
                    f'budget_curves.csv: {benchmark}, top5; first candidate-submission budget with score generalization gap mean at least 1 pp above baseline')
        score_threshold = next(row for row in thresholds if row['benchmark'] == benchmark and row['policy'] == 'top5' and row['metric'] == 'score_gap_above_preload_pp')
        first_score = int(score_threshold['smallest_tested_random_budget']) if score_threshold['smallest_tested_random_budget'] else None
        assert first_score == next((int(row['new_submissions']) for row in rows[1:] if float(row['score_gap_mean_pp'])-float(row['preload_mean_pp']) >= 2), None)
        plus2 = fact(f'score_threshold_{benchmark}', first_score, f'threshold_crossings.csv: {benchmark}, top5, score_gap_above_preload_pp',
                    str(first_score) if first_score is not None else 'Not reached')
        source = f"headline.csv: {benchmark}, top5"
        baseline = fact(f"baseline_{benchmark}", float(headline["score_gap_baseline_pp"]), source, f'{float(headline["score_gap_baseline_pp"]):.2f}')
        queried = fact(f"queried_{benchmark}", float(headline["queried_score_gap_pp"]), source, f'{float(headline["queried_score_gap_pp"]):.2f}')
        comparison = comparisons[comparisons.benchmark == benchmark].set_index('new_submissions')
        supported = comparison.index[comparison.significant_increase].tolist()
        first_supported = min(supported) if supported else None
        significance = fact(f'significant_budget_{benchmark}', first_supported,
            f'trials.parquet: {benchmark}, top5; paired mean difference; 95% Bonferroni lower bound over {COMPARISONS} comparisons',
            str(first_supported) if first_supported is not None else 'Not reached')
        static = fact(f"static_{benchmark}", float(control["first_s_not_t"]), f"budget_curves.csv: {benchmark}, top5, 1", percent(control["first_s_not_t"]))
        final = fact(f"final_{benchmark}", float(end["first_s_not_t"]), source, percent(end["first_s_not_t"]))

    crossing = rates[(rates.threshold_pp=='any') & (rates.new_submissions>1) & (rates.rate>=.9)].new_submissions.min()
    threshold_text.append('Terminal-Bench-Science, '+fact('threshold_terminal_science',int(crossing),
        'selection_severity.csv: terminal_science, first 90% crossing',str(int(crossing))))
    diagnostic = read_json(RESULTS / "non_top.json")
    promotion = fact("promotion", diagnostic["random_false_selection"], "non_top.json: random_false_selection", percent(diagnostic["random_false_selection"]))
    eligible = fact("eligible", diagnostic["eligible_trials"], "non_top.json: eligible_trials")
    trials = fact("trials", protocol["trials"], "manifest.json: trials")

    average_budget = sum(FACTS[f'threshold_{b}']['value'] for b in NAMES)/len(NAMES)
    average_attack = sequential_sum(FACTS[f'queried_{b}']['value'] for b in NAMES)/len(NAMES)
    average_baseline = sequential_sum(FACTS[f'baseline_{b}']['value'] for b in NAMES)/len(NAMES)
    aggregate_values = [fact('mean_selection_budget', average_budget, 'threshold_crossings.csv: top5, first_s_not_t; mean across snapshots', f'{average_budget:g}'),
        fact('mean_attack_gap', average_attack, 'headline.csv: top5, queried_score_gap_pp; mean across snapshots', f'{average_attack:.2f}'),
        fact('mean_baseline_gap', average_baseline, 'headline.csv: top5, score_gap_baseline_pp; mean across snapshots', f'{average_baseline:.2f}')]
    fastest = min(FACTS[f'threshold_{b}']['value'] for b in NAMES)
    fastest_text = fact('abstract_fastest_selection', fastest,
        'threshold_crossings.csv: minimum top5 first_s_not_t crossing', f'{fastest:g}')
    abstract_result = f'In as few as {fastest_text} submissions, our attack can produce a benchmark winner that falls behind on held-out data.'

    significant_budgets = [FACTS[f'significant_budget_{b}']['value'] for b in NAMES]
    average_significant = (sum(significant_budgets)/len(NAMES)
                           if all(value is not None for value in significant_budgets) else None)
    significant_mean = fact('mean_significant_budget', average_significant,
        'trials.parquet: top5; mean first significant budget across snapshots',
        f'{average_significant:g}' if average_significant is not None else 'Not reached')
    threshold_means = []
    for prefix, increase in [('score_threshold_plus1', 1), ('score_threshold', 2)]:
        values = [FACTS[f'{prefix}_{b}']['value'] for b in NAMES]
        value = sum(values)/len(values) if all(v is not None for v in values) else None
        threshold_means.append(fact(f'mean_{prefix}', value,
            f'budget_curves.csv: top5; mean first candidate-submission budget with score generalization gap at least {increase} pp above baseline', f'{value:g}' if value is not None else 'Not reached'))
    average_static = sequential_sum(FACTS[f'static_{b}']['value'] for b in NAMES)/len(NAMES)
    average_final = sequential_sum(FACTS[f'final_{b}']['value'] for b in NAMES)/len(NAMES)
    static_mean = fact('mean_static_selection', average_static,
        'budget_curves.csv: top5, 1, first_s_not_t; mean across snapshots', percent(average_static))
    final_mean = fact('mean_final_selection', average_final,
        f'budget_curves.csv: top5, {budget}, first_s_not_t; mean across snapshots', percent(average_final))

    entries = [entry.strip() for entry in re.split(r'(?m)(?=^@)', (ROOT/'docs/citations.bib').read_text()) if entry.strip()]
    assert len(entries) == 2
    citations = []
    for (key, label), entry in zip([('website', 'Website and software'), ('paper', 'Theory paper')], entries):
        citations.append(f'<div class="citation" id="citation-{key}"><div class="citation-header"><h3>{label}</h3>'
            f'<button type="button" data-copy="bibtex-{key}" aria-label="Copy {key} BibTeX" hidden>Copy BibTeX</button>'
            f'<span class="copy-status" role="status"></span></div>'
            f'<pre id="bibtex-{key}" tabindex="0"><code>{html.escape(entry)}</code></pre></div>')
    from feedback import build as build_feedback
    progress("Building feedback charts and leaderboard examples...")
    feedback = build_feedback(read_csv, read_json, fact)
    from leaderboards import build as build_leaderboards
    leaderboard_examples = build_leaderboards(read_json, fact)
    from predictor import build as build_predictor
    predictor = build_predictor(read_csv, read_json, fact)
    from selection import build as build_selection
    progress("Calculating model-selection losses, controls, and the worked example...")
    selection = build_selection(read_csv, read_json, fact)
    selection_method = read_json(SITE/'data/selection_loss_method.json')
    SOURCES.update(selection_method['sources_sha256'])
    SOURCES.update(read_json(SITE/'data/secondary_selection_method.json')['sources_sha256'])
    replacements = {**leaderboard_examples, **predictor, **feedback, **selection,
        "threshold_text": "; ".join(threshold_text),
        "abstract_result": abstract_result,
        "selection_plot_limit": str(SELECTION_PLOT_LIMIT),
        "static_summary": static_mean, "final_summary": final_mean,

        "citations": "".join(citations),

        "promotion": promotion, "eligible": eligible,
        "trials": trials,
        "budget": fact("budget", budget, "manifest.json: headline_budget", f"{budget:,}"),
    }
    for name in ("budget_curves.csv", "headline.csv", "threshold_crossings.csv", "verification.json"):
        shutil.copyfile(RESULTS / name, SITE / "data" / name)
    shutil.copyfile(ROOT / "docs/benchmark_interfaces.md", SITE / "data/benchmark_interfaces.md")
    shutil.copyfile(ROOT / "docs/citations.bib", SITE / "data/citations.bib")
    for name in ["dimension_budget_curves.csv", "partition_budget_curves.csv", "thresholds.csv", "static_controls.csv", "exact_coordinate_matches.csv", "groupings.json", "protocol.json"]:
        shutil.copyfile(ROOT / "results/feedback_dimension_v1" / name, SITE / "data" / ("feedback_" + name))
    for name in ["feedback_dimension.md"]:
        shutil.copyfile(ROOT / "docs" / name, SITE / "data" / name)
    for name in ['summary.csv', 'protocol.json', 'verification.json']:
        source = ROOT/'results/feedback_policy_v1'/name
        SOURCES[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copyfile(source, SITE/'data'/('policy_'+name))
    from scripts.terminal_counterexample import summarize as summarize_followup
    followup_path = ROOT/'results/terminal_counterexample/run/trials.csv'
    read_csv(followup_path)
    followup = summarize_followup(pd.read_csv(followup_path)).to_dict('records')
    with (SITE/'data/agent_followup_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(followup[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(row for row in followup if row['benchmark']=='terminal_bench_4')
    for name in ['summary.csv','trials.csv','provenance.json']:
        source = ROOT/'results/outcome_banks/terminal_science'/name
        SOURCES[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copyfile(source, SITE/'data'/('outcome_'+name))
    from scripts.feedback_selection import derive
    progress("Summarizing the feedback-dimension experiment...")
    feedback_losses = derive(ROOT, SITE)
    selected = {(row['dimension'], row['new_submissions']):row for row in feedback_losses}
    for dimension,label in [(1,'one'),(18,'eighteen')]:
        value = selected[dimension,8]['model_selection_loss_pp']
        replacements['feedback_loss_'+label] = fact('feedback_loss_'+label,value,'feedback_selection_loss.csv: '+str(dimension)+', 8',f'{value:.2f}')
    values = [row['model_selection_loss_pp'] for row in feedback_losses if row['new_submissions']==128]
    replacements['feedback_loss_range'] = f'{min(values):.2f}–{max(values):.2f}'
    SOURCES.update(read_json(SITE/'data/feedback_selection_method.json')['source_sha256'])
    from charts import build as build_charts
    progress("Rendering charts and the page...")
    replacements.update(build_charts(SITE))
    template = (SITE / "index.template.html").read_text()
    for key, value in replacements.items():
        template = template.replace("{{" + key + "}}", value)
    for key, prefix in [('table_label', ''), ('supplementary_table_label', 'S')]:
        numbers = count(1)
        template = re.sub(r'\{\{' + key + r'\}\}',
            lambda _: f'<strong class="label">Table {prefix}{next(numbers)}.</strong>', template)
    from latex import render
    template = re.sub(r"\{\{(math|display):(.+?)\}\}(?!\})", lambda m: render(m.group(2), m.group(1) == "display"), template)
    assert not re.search(r"\{\{[^}]+\}\}", template), "Unfilled page field"
    # Cache-bust local presentation assets after rebuilding.
    template = re.sub(r'(src|srcset|href)="(style\.css|native-charts\.css|native-charts\.js|interactions\.js|analytics\.js|figures/[^"?]+\.svg)"',
        lambda m: f'{m[1]}="{m[2]}?v={hashlib.sha256((SITE/m[2]).read_bytes()).hexdigest()[:10]}"', template)
    (SITE / "index.html").write_text(template)
    (SITE / "data/results.json").write_text(json.dumps({"release": "release_v2", "curves": curves, "headlines": headlines, "thresholds": thresholds, "facts": FACTS, "score_comparisons": comparisons.astype(object).where(pd.notna(comparisons), None).to_dict(orient="records"), "score_comparison_method": comparison_method}, indent=2, allow_nan=False) + "\n")
    (SITE / "data/provenance.json").write_text(json.dumps({"release": "release_v2", "sources_sha256": SOURCES, "facts": FACTS}, indent=2) + "\n")
    print(f"Built the page and figures from canonical releases; {len(FACTS)} values checked in {perf_counter() - started:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
