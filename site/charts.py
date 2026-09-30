"""Native chart specifications and SVG downloads from saved result tables."""
import csv
import json

BLUE, GOLD, GRAY = '#9abaff', '#d8b47a', '#a6adb7'
MARKERS = ['circle','triangle','square','diamond','triangle-down','cross']
MPL_MARKERS = dict(zip(MARKERS,['o','^','s','D','v','x']))
NAMES = {'livebench':'LiveBench','helm_capabilities':'HELM Capabilities','helm_lite':'HELM Lite',
         'openllm_v2':'Open LLM','swe_verified':'SWE-bench Verified','terminal_science':'Terminal-Bench-Science'}


def save_figure(group, out, mobile):
    """Generate SVG downloads from the same values as the native charts."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    count=len(group['panels'])
    columns=1 if mobile else (3 if count==6 else 2)
    fig, axes=plt.subplots(count if mobile else (count+columns-1)//columns, columns,
        figsize=(4.3, count*3.1+.7) if mobile else (8.3, ((count+columns-1)//columns)*2.8+.65), squeeze=False)
    for ax,p in zip(axes.flat,group['panels']):
        for s in p['series']:
            points=s['points']
            # Tiny floating-point deviations in grouping means can invert a zero-width bar.
            lower=[max(0,v['y']-v.get('low',v['y'])) for v in points]
            upper=[max(0,v.get('high',v['y'])-v['y']) for v in points]
            ax.errorbar([v['x'] for v in points],[v['y'] for v in points],yerr=[lower,upper],
                        color=s['color'],marker=MPL_MARKERS[s['marker']],markersize=3,lw=1,elinewidth=.65,capsize=2,
                        linestyle=(0,tuple(map(float,s['dash'].split()))) if s.get('dash') else '-')
        for r in p['references']:
            ax.axhline(r['value'],color=r['color'],ls=':' if r.get('dash')=='2 3' else '--',lw=.8)
        ax.set_xscale('log',base=2)
        maximum=max(v['x'] for series in p['series'] for v in series['points'])
        ticks=[1,4,16,64,256,1024,4096] if p['xLabel']=='Released score coordinates' else ([2,4,8,16,32,64,128] if maximum<=128 else [2,8,32,128,maximum])
        ax.xaxis.set_major_locator(FixedLocator(ticks));ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:,.0f}'))
        ax.set_title(p['title'],fontsize=10.5)
        ax.set_xlabel(p['xLabel']+' (log scale)',fontsize=9)
        ax.tick_params(length=0,labelsize=8.5,colors='#e8e6e1')
        ax.spines[['top','right','left']].set_visible(False)
        ax.spines['bottom'].set_color('#454545')
        ax.set_axisbelow(True)
        ax.grid(axis='y',color='#292929',lw=.6)
        ax.margins(x=.04,y=.1)
        if 'bounds' in p: ax.set_ylim(*p['bounds'])
    for ax in list(axes.flat)[count:]: ax.set_visible(False)
    first = group['panels'][0]
    handles = [Line2D([],[],color=s['color'],marker=MPL_MARKERS[s['marker']],markersize=4,label=s['name'],
        linestyle=(0,tuple(map(float,s['dash'].split()))) if s.get('dash') else '-') for s in first['series']]
    if not (group['id'].startswith('feedback') or group['id']=='false-winners'):
        handles += [Line2D([],[],color=r['color'],ls='--',label=r['name']) for r in first['references']]
    if len(handles)>1:
        fig.legend(handles=handles,
                   loc='upper center',ncol=3 if mobile else 6,frameon=False,fontsize=9)
    fig.tight_layout(rect=[0,.015,1,(.94 if mobile else .89) if len(handles)>1 else 1],h_pad=2,w_pad=2)
    name=group['id']+('-mobile' if mobile else '')+'.svg'
    fig.savefig(out/'figures'/name,metadata={'Date':None,'Creator':'site/charts.py'})
    path = out/'figures'/name
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def build(out):
    groups=[]
    sources={}
    def read(file):
        if file not in sources:
            with (out/file).open(newline='') as f:sources[file]=list(csv.DictReader(f))
        return sources[file]
    def points(file, predicate, key, lo=None, hi=None, factor=1, x='new_submissions'):
        result=[]
        for i,r in enumerate(read(file)):
            if not predicate(r):continue
            p=dict(x=float(r[x]),y=factor*float(r[key]),budget=int(r['new_submissions']),
                   source=dict(file=file,row=i,key=key,low=lo,high=hi,factor=factor,x=x))
            if lo:p.update(low=factor*float(r[lo]),high=factor*float(r[hi]))
            result.append(p)
        return sorted(result,key=lambda p:p['x'])
    def line(name,points,color=BLUE,band='95% interval',dash=None,marker='circle'):
        return dict(name=name,points=points,color=color,band=band,dash=dash,marker=marker)
    def panel(title,unit,series,refs=None,**extra):
        return dict(title=title,unit=unit,series=series,references=refs or [],xLabel='New submissions',**extra)
    def group(id,caption,panels,download):
        groups.append(dict(id=id,caption=caption,panels=panels,download=download))
    for id,key,limit,caption in [('selection-loss','model_selection_loss_pp',128,'selection-loss-caption'),('generalization','score_gap_mean_pp',512,'generalization-caption')]:
        panels=[]
        for b,name in NAMES.items():
            p=points('data/benchmark_curves.csv',lambda r:r['benchmark']==b and int(r['new_submissions'])<=limit,key,key+'_low',key+'_high')
            refs=[] if id=='selection-loss' else [dict(name='No attack',value=float(next(r for r in read('data/benchmark_curves.csv') if r['benchmark']==b)['baseline_gap_pp']),color=GRAY)]
            panels.append(panel(name,'pp',[line('With attack',p)],
                                refs,
                                note=f"{next(r for r in read('data/benchmark_curves.csv') if r['benchmark']==b)['trials']} splits",metric='Mean selection loss' if id=='selection-loss' else 'Best-score gap'))
        group(id,caption,panels,'data/benchmark_curves.csv')
    panels=[]
    for b,name in NAMES.items():
        series=[]
        for threshold,label,color,dash,marker in [('any','Any false winner',BLUE,None,'circle'),('1','Score gap ≥ 1 pp',GRAY,'5 3','square'),('2','Score gap ≥ 2 pp',GOLD,None,'triangle'),('5','Score gap ≥ 5 pp','#98c4b3','2 3','diamond')]:
            pp=points('data/selection_severity.csv',lambda r:r['benchmark']==b and r['policy']=='top5' and r['threshold_pp']==threshold and 2<=int(r['new_submissions'])<=128,'rate','ci_low' if threshold=='any' else None,'ci_high' if threshold=='any' else None,100)
            series.append(line(label,pp,color,dash=dash,marker=marker))
        deficit=next(r for r in read('data/selection_deficits.csv') if r['benchmark']==b and r['policy']=='top5' and r['new_submissions']=='128')
        panels.append(panel(name,'%',series,[dict(name='90% reference',value=90,color='#707984')],
            note=f"{deficit['trials']} splits",metric='False-winner frequency',bounds=[0,100],footer=f"Mean score gap at 128: {float(deficit['mean_deficit_pp']):.2f} pp"))
    group('false-winners','selection-caption',panels,'data/selection_severity.csv')
    dims=sorted({int(r['dimension']) for r in read('data/feedback_dimension_budget_curves.csv')})
    colors=['#b6babd','#e4ad66','#69c7aa','#78b9e8','#bd97d8','#ed869b']
    panels=[]
    for key,unit,title in [('model_selection_loss_pp','pp','Selection loss'),('score_gap_mean_pp','pp','Best-score gap')]:
        for x in ['new_submissions','released_score_coordinates']:
            file='data/feedback_selection_loss.csv' if key=='model_selection_loss_pp' else 'data/feedback_dimension_budget_curves.csv'
            series=[line(f'{d} score'+('' if d==1 else 's'),points(file,
                lambda r:int(r['dimension'])==d and 2<=int(r['new_submissions'])<=256,key,key+'_partition_min',key+'_partition_max',100 if unit=='%' else 1,x),color,band='Range across groupings',marker=marker) for d,color,marker in zip(dims,colors,MARKERS)]
            baseline=float(read('data/feedback_dimension_budget_curves.csv')[0]['preload_mean_pp'])
            refs=[dict(name='No attack',value=0,color=GRAY)] if key=='model_selection_loss_pp' else [dict(name='No attack',value=baseline,color=GRAY),dict(name='Baseline + 2 pp',value=baseline+2,color='#707984')]
            p=panel(title,unit,series,refs,note='Three balanced groupings',metric='LiveBench',**({'bounds':[0,100]} if unit=='%' else {}))
            p['xLabel']='Submissions' if x=='new_submissions' else 'Released score coordinates'
            panels.append(p)
    group('feedback','feedback-caption',panels,'data/feedback_selection_loss.csv')
    panels=[]
    for x in ['new_submissions','released_score_coordinates']:
        series=[line(f'{d} score'+('' if d==1 else 's'),points('data/feedback_dimension_budget_curves.csv',
            lambda r:int(r['dimension'])==d and 2<=int(r['new_submissions'])<=256,'first_s_not_t',
            'first_s_not_t_partition_min','first_s_not_t_partition_max',100,x),color,band='Range across groupings',marker=marker) for d,color,marker in zip(dims,colors,MARKERS)]
        p=panel('False-winner frequency','%',series,[dict(name='90% reference',value=90,color=GRAY)],
                note='Three balanced groupings',metric='LiveBench',bounds=[0,100])
        p['xLabel']='Submissions' if x=='new_submissions' else 'Released score coordinates'
        panels.append(p)
    group('feedback-frequency','feedback-frequency-caption',panels,'data/feedback_dimension_budget_curves.csv')
    walkthrough=json.loads((out/'data/walkthrough.json').read_text())
    group('walkthrough','walkthrough-caption',[dict(type='dumbbell',title='Public and held-out performance',unit='%',
        rows=[dict(r,label=label) for r,label in zip(walkthrough['scores'],['Gemma 2 27B IT','DeepSeek V3','GPT-4o · no attack','Final router'])],
        note='HELM Lite · trial 165',metric='Mean category score')],'data/walkthrough.json')
    fragments = {}
    for group in groups:
        for p in group['panels']:
            for reference in p.get('references', []):
                if reference['name'] in ['90% reference', 'Baseline + 2 pp']:
                    reference['dash'] = '2 3'
        if group['id'] != 'walkthrough':
            for mobile in (False, True):
                save_figure(group, out, mobile)
        name = group['id']
        fragments[name.replace('-', '_')+'_chart'] = (
            f'<div class="native-chart-group" data-chart-group="{name}"></div>'
            f'<noscript><picture><source media="(max-width: 600px)" srcset="figures/{name}-mobile.svg">'
            f'<img class="selection-figure" src="figures/{name}.svg" alt="{group["panels"][0]["metric"]}"></picture></noscript>')
    fragments['native_chart_data'] = json.dumps(groups, separators=(',', ':')).replace('<', '\\u003c')
    (out/'data/native_charts.json').write_text(json.dumps(groups,indent=2)+'\n')
    return fragments
