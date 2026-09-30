"""Verify canonical feedback-dimension records, summaries, and source provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark_progress.benchmarks import ROOT, BUDGETS
from benchmark_progress.feedback_dimension import RELEASE, definitions, verify_groupings
from scripts.verify import verify_outcomes
from scripts.report_feedback import tables


def verify():
    if not __debug__: raise RuntimeError('Run verification without Python optimization (-O)')
    groups=definitions();verify_groupings(groups)
    sums=RELEASE/'checksums.json'
    for name,expected in json.loads(sums.read_text()).items():
        assert hashlib.sha256((RELEASE/name).read_bytes()).hexdigest()==expected,name
    ledger=json.loads((RELEASE/'provenance.json').read_text())
    for name,entry in ledger['files'].items():
        assert hashlib.sha256((RELEASE/name).read_bytes()).hexdigest()==entry['sha256'],name
    raw=pd.read_parquet(RELEASE/'trials.parquet')
    assert len(raw)==42000 and not raw.duplicated(['policy','trial','new_submissions']).any()
    assert (raw.released_score_coordinates==raw.dimension*(raw.new_submissions-1)).all()
    z=np.load(RELEASE/'evaluation.npz');conditions=json.loads((RELEASE/'protocol.json').read_text())['conditions']
    for t in range(250):
        for ci,c in enumerate(conditions):
            trace={key:z[key][t][ci] if key.startswith('final_') else z[key][t] for key in ['best_loss_s','best_loss_t','genuine_s','genuine_t','final_s','final_t']}
            verify_outcomes(raw[(raw.trial==t)&(raw.policy==c)].sort_values('new_submissions'),trace,'livebench')
    original=pd.read_parquet(ROOT/'results/release_v1/trials.parquet').query("benchmark == 'livebench' and policy == 'top5'").sort_values(['trial','new_submissions'])
    current=raw[raw.dimension==18].sort_values(['trial','new_submissions'])
    for key in ['first_s','first_t','first_s_not_t','rank_s','rank_t','score_gap_pp','genuine_preload_score_gap_pp','heldout_regret_pp','ranking_regret_pp']:
        if current[key].dtype.kind == 'f':
            np.testing.assert_allclose(current[key],original[key],rtol=0,atol=1e-10)
        else:
            np.testing.assert_array_equal(current[key].to_numpy(),original[key].to_numpy())
    for name,table in tables(raw).items():
        expected=pd.read_csv(RELEASE/(name+'.csv'))
        pd.testing.assert_frame_equal(table.reset_index(drop=True),expected,check_dtype=False,check_exact=False,rtol=0,atol=1e-10)
    print('PASS: groupings, 42,000 outcomes, d=18 release_v1 metrics at original tolerance, all primary/partition curves, thresholds, paired comparisons.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.parse_args();verify()
