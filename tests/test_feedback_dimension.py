"""Controlled-interface invariants and canonical presentation checks."""
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import numpy as np
import pandas as pd
from benchmark_progress.feedback_dimension import RELEASE, definitions, verify_groupings, grouped_feedback
from benchmark_progress.benchmarks import ROOT


def test_fixed_score_blind_groups_and_aliases():
    groups=definitions();verify_groupings(groups)
    assert len({g['computed_condition'] for g in groups})==14
    with np.load(ROOT/'results/release_v1/livebench/public_profiles.npz') as p:
        names=list(p['coordinates']);table=p['profiles'][0]
    for g in groups:
        published,mapping=grouped_feedback(table,names,g)
        assert published.shape==(33,g['dimension']) and len(mapping)==18
        if g['dimension']==18:
            np.testing.assert_array_equal(published[:,mapping],table)
    assert hashlib.sha256((RELEASE/'groupings.json').read_bytes()).hexdigest()==json.loads((RELEASE/'protocol_lock.json').read_text())['sha256']['groupings.json']


def test_rational_quantization_ties_to_even():
    # Means of 0.5, 1.5, 2.5, and 3.5 integer ticks round to 0,2,2,4.
    task=np.array([[0,1],[1,2],[2,3],[3,4]],dtype=float)/100000
    g={'groups':[['a','b']],'task_to_group':{'a':0,'b':0}}
    published,mapping=grouped_feedback(task,['a','b'],g)
    np.testing.assert_array_equal(np.rint(published[:,0]*100000),[0,2,2,4])
    assert mapping==(0,0)


def test_feedback_displayed_numbers_and_docs():
    curves=pd.read_csv(RELEASE/'dimension_budget_curves.csv')
    thresholds=pd.read_csv(RELEASE/'thresholds.csv').query("summary == 'dimension_mean'").set_index('dimension')
    facts=json.loads((ROOT/'site/data/provenance.json').read_text())['facts']
    ks=[];counts=[]
    for d,g in curves.groupby('dimension'):
        assert (g.released_score_coordinates==d*(g.new_submissions-1)).all()
        crossings={}
        for name,metric,target in [('false90','first_s_not_t',.9),('score_gap_plus1','score_gap_mean_pp',g.preload_mean_pp.iloc[0]+1),('score_gap_plus2','score_gap_mean_pp',g.preload_mean_pp.iloc[0]+2)]:
            first=g[(g.new_submissions>1)&(g[metric]>=target)].iloc[0]
            for axis,col in [('k','new_submissions'),('I','released_score_coordinates')]:
                value=int(first[col])
                crossings[name,axis]=value
                assert facts[f'feedback_{d}_{name}_{axis}']['value']==value
                assert value==thresholds.loc[d,name+'_'+axis+'_random']
        ks.append(int(thresholds.loc[d,'false90_k_random']));counts.append(int(thresholds.loc[d,'false90_I_random']))
        line=f'| {d} | {ks[-1]} | {counts[-1]} | {crossings["score_gap_plus1","k"]} | {crossings["score_gap_plus1","I"]} | {thresholds.loc[d,"score_gap_plus2_k_random"]:g} | {thresholds.loc[d,"score_gap_plus2_I_random"]:g} |'
        assert line in (ROOT/'docs/feedback_dimension.md').read_text()
    assert facts['feedback_submission_spread']['value']==max(ks)/min(ks)
    assert facts['feedback_coordinate_spread']['value']==max(counts)/min(counts)
    expected=f'On LiveBench, returning 18 scores per submission instead of 1 reduces the first tested count reaching a 90% false-winner rate from {ks[0]} to {ks[-1]} submissions. These submission counts differ by a factor of {max(ks)/min(ks):g}; the corresponding totals of released scores differ by only {max(counts)/min(counts):g}.'
    assert facts['feedback_submission_spread']['value']==16
    losses=pd.read_csv(ROOT/'site/data/feedback_selection_loss.csv').query('new_submissions == 8').set_index('dimension')
    for d in [1,18]:
        assert f'{losses.loc[d,"model_selection_loss_pp"]:.2f} pp' in (ROOT/'README.md').read_text()
    exact=pd.read_csv(RELEASE/'exact_coordinate_matches.csv')
    assert len(exact)==12
    frequency=exact[exact.metric=='first_s_not_t']
    assert ((frequency.ci_low<=0)&(frequency.ci_high>=0)).all()
    gaps=exact[exact.metric=='score_gap_pp']
    assert ((gaps.ci_low>0)|(gaps.ci_high<0)).sum()==3
    for name in ['feedback_dimension.md']:
        assert (ROOT/'docs'/name).read_bytes()==(ROOT/'site/data'/name).read_bytes()


def test_feedback_trial_export_reproduces_published_precision(tmp_path):
    # In Python 3.12, built-in sum() changes the final digit of some selected
    # scores (e.g. trial 0, d01_p0, budget 64), breaking the CI rebuild check.
    from scripts.feedback_selection import derive

    (tmp_path/'data').mkdir()
    derive(ROOT, tmp_path)
    for name in ['feedback_selection_loss_trials.csv', 'feedback_selection_loss.csv',
                 'feedback_selection_loss_partitions.csv']:
        assert (tmp_path/'data'/name).read_bytes() == (ROOT/'site/data'/name).read_bytes(), name


def test_figures_use_same_outcomes_and_exclude_static_controls(monkeypatch,tmp_path):
    spec=importlib.util.spec_from_file_location('feedback_figure',ROOT/'site/feedback.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    import matplotlib.axes
    captured=[];original=matplotlib.axes.Axes.plot
    def record(self,x,y,*args,**kwargs):
        captured.append((np.array(x),np.array(y)));return original(self,x,y,*args,**kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes,'plot',record)
    monkeypatch.setattr(mod,'SITE',tmp_path);(tmp_path/'figures').mkdir()
    curves=pd.read_csv(RELEASE/'dimension_budget_curves.csv').to_dict('records')
    mod.figure(curves,[1,2,3,6,9,18])
    assert len(captured)==24
    for offset,metric,scale in [(0,'first_s_not_t',100),(12,'score_gap_mean_pp',1)]:
        for i,d in enumerate([1,2,3,6,9,18]):
            k,y=captured[offset+i];count,same_y=captured[offset+i+6]
            np.testing.assert_array_equal(count,d*(k-1));np.testing.assert_array_equal(y,same_y)
            expected=[r for r in curves if int(r['dimension'])==d and 2<=int(r['new_submissions'])<=256]
            np.testing.assert_array_equal(k,[int(r['new_submissions']) for r in expected])
            np.testing.assert_allclose(y,[r[metric]*scale for r in expected])
            assert min(k)==2 and min(count)>0


def test_reader_facing_terminology():
    # Archived source field names in import-compatibility code are intentionally outside this scope.
    published_docs = subprocess.check_output(['git','ls-files','docs/*.md'],cwd=ROOT,text=True).splitlines()
    files=[ROOT/'README.md',*[ROOT/p for p in published_docs],
           *sorted((ROOT/'site').glob('*.html')),*sorted((ROOT/'site/figures').glob('*.svg'))]
    banned=re.compile(r'\bprob(?:e|es|ing)\b',re.I)
    for p in files:assert not banned.search(p.read_text()),p
    for name in ['reproduce_feedback','verify_feedback','report_feedback']:
        help_text=subprocess.check_output([sys.executable,'-m','scripts.'+name,'--help'],cwd=ROOT,text=True)
        assert not banned.search(help_text)
