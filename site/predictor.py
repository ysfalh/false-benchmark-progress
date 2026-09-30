"""Website presentation of saved predictor results; no fitting or new analysis."""
from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
RELEASE = ROOT/'results/predictor_v2'
BENCHMARKS = ['helm_capabilities', 'helm_lite', 'openllm_v2', 'swe_verified']
NAMES = ['HELM Capabilities', 'HELM Lite', 'Open LLM', 'SWE-bench Verified']
COLORS = ['#69c7aa', '#78b9e8', '#ed869b', '#d8bb78']
MARKERS = ['s', 'D', 'X', '^']


def figure(points):
    """Draw only the saved score generalization gap points from the external-validation figure."""
    points = [p for p in points if p['panel'] == 'score_gap_plus2']
    fig, ax = plt.subplots(figsize=(5.1, 3.7))
    fig.subplots_adjust(left=.14, right=.975, bottom=.19, top=.74)
    training = [p for p in points if p['source'] == 'LiveBench fit cell']
    ax.scatter([float(p['observed_k']) for p in training],
               [float(p['predicted_k']) for p in training], s=17,
               facecolors='none', edgecolors='#65707e', linewidths=.7, zorder=2)
    ax.plot([1, 2048], [1, 2048], color='#a3abb6', linestyle=(0, (3, 3)), linewidth=.8, zorder=1)
    for name, color, marker in zip(NAMES, COLORS, MARKERS):
        point, = [p for p in points if p['source'] == name]
        observed = float(point['observed_k'])
        ax.scatter([min(observed, 2048)], [float(point['predicted_k'])],
                   color=color, marker=marker, s=39, edgecolors='#000000', linewidths=.4, zorder=3)
        if not math.isfinite(observed):
            ax.annotate('>2,048', (2048, float(point['predicted_k'])), xytext=(-6, 10),
                        textcoords='offset points', ha='right', color=color, fontsize=9)
    ax.set_xscale('log', base=2); ax.set_yscale('log', base=2)
    ax.set_xlim(.85, 2600); ax.set_ylim(.85, 2600)
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(FixedLocator([1, 4, 16, 64, 256, 2048]))
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    ax.set_xlabel('Observed submissions', fontsize=10, labelpad=7)
    ax.set_ylabel('Predicted submissions', fontsize=10, labelpad=7)
    ax.tick_params(length=0, pad=5, labelsize=9, colors='#c8cbc9')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#383838')
    ax.grid(axis='y', color='#262626', linewidth=.6)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], linestyle='none', marker='o', markerfacecolor='none',
                     markeredgecolor='#717b87', markersize=4, label='LiveBench')]
    handles += [Line2D([], [], linestyle='none', marker=m, color=c, markersize=4.5, label=n)
                for n, c, m in zip(NAMES, COLORS, MARKERS)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55, .99),
               ncol=2, frameon=False, fontsize=9, columnspacing=1.4, handletextpad=.5)
    path = SITE/'figures/predictor-score-gap.svg'
    fig.savefig(path, metadata={'Date': None, 'Creator': 'site/predictor.py'})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def build(read_csv, read_json, fact):
    points = read_csv(RELEASE/'figure_points.csv')
    model = read_json(RELEASE/'models.json')['score_gap_plus2']
    fact('predictor_coefficient', model['coefficient'],
         'results/predictor_v2/models.json: score_gap_plus2.coefficient', f"{model['coefficient']:.3f}")
    figure(points)
    return {}
