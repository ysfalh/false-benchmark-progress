"""Data-boundary checks for the Verified replay, not its attack implementation."""
import pandas as pd
import pytest
from scripts.import_swe_verified import repository_groups, resolved_ids


def test_resolved_lists_must_use_unique_known_ids():
    assert resolved_ids({}, {'resolved':['a']}, {'a','b'})=={'a'}
    for ids in [['a','a'],['unknown']]:
        with pytest.raises(ValueError):
            resolved_ids({}, {'resolved':ids}, {'a','b'})


def test_partial_detail_records_are_not_imputed_as_failures():
    row={'per_instance_details':{'a':{'resolved':True}}}
    assert resolved_ids(row,None,{'a','b'}) is None
    row['per_instance_details']['b']={'resolved':False}
    assert resolved_ids(row,None,{'a','b'})=={'a'}


def test_small_repository_pool_is_metadata_only_and_preserves_items():
    raw=pd.DataFrame({'repo':['flask']+['seaborn']*2+['requests']*8+['pylint']*10})
    groups=repository_groups(raw)
    assert groups['pylint']=='pylint'
    assert {groups[k] for k in ['flask','seaborn','requests']}=={'Other small repositories'}
    assert raw.repo.map(groups).value_counts().to_dict()=={'Other small repositories':11,'pylint':10}
