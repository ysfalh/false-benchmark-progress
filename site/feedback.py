"""Presentation derived only from the controlled study's canonical tables."""
from pathlib import Path
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from matplotlib.lines import Line2D

SITE = Path(__file__).resolve().parent
RELEASE = SITE.parent/'results/feedback_dimension_v1'
COLORS = ['#b6babd','#e4ad66','#69c7aa','#78b9e8','#bd97d8','#ed869b']
MARKERS = ['o','^','s','D','v','X']
PLOT_LIMIT = 256


def draw_panel(ax, curves, dimensions, axis, score_gap, limit):
    metric='score_gap_mean_pp' if score_gap else 'first_s_not_t';scale=1 if score_gap else 100
    for d,color,marker in zip(dimensions,COLORS,MARKERS):
        g=[r for r in curves if int(r['dimension'])==d and 1<int(r['new_submissions'])<=limit]
        x=[int(r[axis]) for r in g]
        y=[float(r[metric])*scale for r in g]
        ax.fill_between(x,[float(r[metric+'_partition_min'])*scale for r in g],
                        [float(r[metric+'_partition_max'])*scale for r in g],color=color,alpha=.10,linewidth=0)
        ax.plot(x,y,color=color,marker=marker,markersize=3.4,linewidth=1.25,markeredgewidth=.65)
    baseline=float(curves[0]['preload_mean_pp'])
    ax.axhline(baseline+2 if score_gap else 90,color='#89929d',linestyle=(0,(2,3)),linewidth=.7,zorder=0)
    if score_gap:
        ax.axhline(baseline,color='#89929d',linestyle='--',linewidth=.7)
        ax.margins(y=.12)
    else:
        ax.set_ylim(-3,103);ax.set_yticks([0,50,100],['0','50','100%'])
    ax.set_xscale('log',base=2)
    if limit==PLOT_LIMIT:
        ticks=[2,8,32,128,256] if axis=='new_submissions' else [1,4,16,64,256,1024,4096]
    else:
        ticks=[2,8,32,128,512,2048] if axis=='new_submissions' else [1,8,64,512,4096,32768]
    ax.xaxis.set_major_locator(FixedLocator(ticks));ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:,.0f}'))
    upper=limit if axis=='new_submissions' else max(dimensions)*(limit-1)
    ax.set_xlim(1.65 if axis=='new_submissions' else .8,upper*1.18)
    ax.set_title('Score generalization gap (pp)' if score_gap else 'False-winner rate (%)',fontsize=11.5,loc='left',pad=10)
    ax.set_xlabel('Submissions' if axis=='new_submissions' else 'Released score coordinates',fontsize=10,labelpad=8)
    ax.tick_params(length=0,pad=5,labelsize=9,colors='#c8cbc9')
    ax.get_xticklabels()[-1].set_horizontalalignment('right')
    ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color('#383838')
    ax.grid(axis='y',color='#262626',linewidth=.6)


def figure(curves, dimensions, mobile=False, score_gap=False):
    metrics=[True] if score_gap else [False,True]
    fig,axes=plt.subplots(len(metrics)*(2 if mobile else 1),1 if mobile else 2,
                          figsize=(4.3,12.7) if mobile else (8.6,3.95 if score_gap else 6.7))
    fig.subplots_adjust(left=.14 if mobile else .075,right=.975,bottom=.055 if mobile else .11,
                        top=.88 if mobile else (.66 if score_gap else .84),hspace=.65,wspace=.22)
    for ax,(is_score_gap,axis) in zip(axes.flat,[(v,a) for v in metrics for a in ['new_submissions','released_score_coordinates']]):
        draw_panel(ax,curves,dimensions,axis,is_score_gap,2048 if score_gap else PLOT_LIMIT)
    fig.suptitle('LiveBench: varying feedback per submission',fontsize=12.5,y=.985)
    handles=[Line2D([],[],color=c,marker=m,markersize=4,linewidth=1.2,label=f'{d} score' + ('' if d == 1 else 's')) for d,c,m in zip(dimensions,COLORS,MARKERS)]
    if mobile: handles=[handles[i] for i in [0,3,1,4,2,5]]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.955 if mobile else .94),
               ncol=3,frameon=False,fontsize=9,columnspacing=1.3,handlelength=1.5)
    name='feedback-score-gap' if score_gap else 'feedback-richness'
    if mobile:name+='-mobile'
    path=SITE/'figures'/f'{name}.svg'
    fig.savefig(path,metadata={'Date':None,'Creator':'site/feedback.py'})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def build(read_csv, read_json, fact):
    curves=read_csv(RELEASE/'dimension_budget_curves.csv')
    crossings=read_csv(RELEASE/'thresholds.csv')
    matches=read_csv(RELEASE/'exact_coordinate_matches.csv')
    protocol=read_json(RELEASE/'protocol.json')
    groups=read_json(RELEASE/'groupings.json')
    assert len({g['partition'] for g in groups})==3
    assert protocol['config']['trials']==250 and len(protocol['config']['models'])==33
    assert len(groups[0]['ordered_tasks'])==18 and protocol['quantum_score']*100==.001
    means=sorted((r for r in crossings if r['summary']=='dimension_mean'),key=lambda r:int(r['dimension']))
    dimensions=[int(r['dimension']) for r in means]
    values={};rows=[]
    for r in means:
        d=int(r['dimension']);g=sorted([v for v in curves if int(v['dimension'])==d and int(v['new_submissions'])>=2],key=lambda v:int(v['new_submissions']))
        cells=[]
        baseline=float(g[0]['preload_mean_pp'])
        for name,metric,target in [('false90','first_s_not_t',.9),('score_gap_plus1','score_gap_mean_pp',baseline+1),('score_gap_plus2','score_gap_mean_pp',baseline+2)]:
            first=next(v for v in g if float(v[metric])>=target)
            for axis,column in [('k','new_submissions'),('I','released_score_coordinates')]:
                value=int(first[column]);values[d,name,axis]=value
                if name+'_'+axis+'_random' in r:
                    assert value==int(float(r[name+'_'+axis+'_random']))
                cells.append(fact(f'feedback_{d}_{name}_{axis}',value,f'feedback_dimension_v1/dimension_budget_curves.csv: d={d}; first candidate-submission crossing of {name}, {column}'))
        rows.append('<tr><th scope="row">'+str(d)+'</th>'+''.join('<td>'+v+'</td>' for v in cells)+'</tr>')
    kspread=max(values[d,'false90','k'] for d in dimensions)/min(values[d,'false90','k'] for d in dimensions)
    ispread=max(values[d,'false90','I'] for d in dimensions)/min(values[d,'false90','I'] for d in dimensions)
    sk=fact('feedback_submission_spread',kspread,'feedback_dimension_v1/thresholds.csv: max/min first-tested false90_k',f'{kspread:g}×')
    si=fact('feedback_coordinate_spread',ispread,'feedback_dimension_v1/thresholds.csv: max/min first-tested false90_I',f'{ispread:g}×')
    lo,hi=dimensions[0],dimensions[-1]
    opening=f'On LiveBench, returning {hi} scores per submission instead of {lo} reduces the first tested count reaching a 90% false-winner rate from {values[lo,"false90","k"]} to {values[hi,"false90","k"]} submissions. The benchmark, model pairs, random routing rules, and evaluation stay fixed.'
    # All repeated headline numbers are derived from the same verified cells above.
    opening_html=opening.replace(f'from {values[lo,"false90","k"]} to {values[hi,"false90","k"]} submissions',f'from {fact("feedback_low_budget",values[lo,"false90","k"],"feedback_dimension_v1/thresholds.csv: d=1 false90_k_random")} to {fact("feedback_high_budget",values[hi,"false90","k"],"feedback_dimension_v1/thresholds.csv: d=18 false90_k_random")} submissions')
    score_matches = [r for r in matches if r['metric'] == 'score_gap_pp']
    maximum = max(abs(float(r['higher_minus_lower_pp'])) for r in score_matches)
    excluded = sum(not float(r['ci_low']) <= 0 <= float(r['ci_high']) for r in score_matches)
    comparison = f'Differences are at most {maximum:.2f} pp; {excluded} of {len(score_matches)} paired 95% intervals exclude zero.'
    for mobile in (False,True):figure(curves,dimensions,mobile)
    figure(curves,dimensions,score_gap=True)
    return {'feedback_equal_coordinates':comparison,'feedback_opening':opening_html,'feedback_plot_limit':str(PLOT_LIMIT),'feedback_contraction':f'The submission counts needed for a 90% false-winner rate show a {sk} spread. The corresponding totals of released scores vary by only {si}.','feedback_rows':''.join(rows)}
