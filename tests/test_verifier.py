"""Integrity checks must apply to the release directory supplied by the caller."""

import hashlib
import json
import pytest
from benchmark_progress.benchmarks import ROOT
from scripts.verify import verify, verify_checksums


def test_changed_external_release_fails(tmp_path):
    name = 'results/release_v1/headline.csv'
    checksum = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    (tmp_path/'checksums.json').write_text(json.dumps({name: checksum}))
    (tmp_path/'headline.csv').write_text('altered result\n')
    with pytest.raises(AssertionError, match='Changed file'):
        verify(tmp_path, smoke=True)


def test_allow_code_changes_still_checks_inputs_and_results(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.verify.ROOT', tmp_path)
    files = ['benchmark_progress/adapter.py', 'data/records.py', 'configs/trial.json',
             'results/release_v2/headline.csv']
    release = tmp_path/'results/release_v2'
    for name in files:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original\n')
    checksums = {name: hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in files}
    (release/'checksums.json').write_text(json.dumps(checksums))
    (tmp_path/files[0]).write_text('extended adapter\n')
    with pytest.raises(AssertionError, match='adapter.py'):
        verify_checksums(release)
    assert verify_checksums(release, allow_code_changes=True) == [files[0]]
    for name in files[1:]:
        (tmp_path/name).write_text('changed\n')
        with pytest.raises(AssertionError, match='Changed file'):
            verify_checksums(release, allow_code_changes=True)
        (tmp_path/name).write_text('original\n')
