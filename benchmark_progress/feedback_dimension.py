"""Balanced public task groups; the existing linear decoder remains unchanged."""
import hashlib
import json
import numpy as np
from .attacks import LinearRule
from .benchmarks import ROOT

RELEASE = ROOT / 'results/feedback_dimension_v1'
DIMENSIONS = (1, 2, 3, 6, 9, 18)


def definitions():
    return json.loads((RELEASE / 'groupings.json').read_text())


def verify_groupings(groups):
    names = sorted(groups[0]['ordered_tasks'])
    assert len(names) == len(set(names)) == 18
    assert [(g['partition'], g['dimension']) for g in groups] == [
        (p, d) for p in range(3) for d in DIMENSIONS]
    for g in groups:
        p, d = g['partition'], g['dimension']
        # This frozen domain string is part of the original grouping protocol.
        ordered = sorted(names, key=lambda n: (
            hashlib.sha256(f'livebench-bandwidth-v1-partition-{p}|{n}'.encode()).hexdigest(), n))
        expected = [ordered[i:i+18//d] for i in range(0, 18, 18//d)]
        assert g['ordered_tasks'] == ordered and g['groups'] == expected
        assert g['task_to_group'] == {t: j for j, group in enumerate(expected) for t in group}
        assert g['condition'] == f'd{d:02d}_p{p}'
        assert g['computed_condition'] == f'd{d:02d}_p{0 if d in (1,18) else p}'


def grouped_feedback(task_profiles, coordinates, definition):
    """Average released integer ticks; round the rational mean, ties to even."""
    ticks = np.rint(task_profiles * 100000).astype(np.int64)
    columns = []
    for group in definition['groups']:
        total = ticks[:, [coordinates.index(t) for t in group]].sum(axis=1)
        quotient, remainder = np.divmod(total, len(group))
        up = (2*remainder > len(group)) | ((2*remainder == len(group)) & (quotient % 2 == 1))
        columns.append(quotient + up)
    mapping = tuple(definition['task_to_group'][t] for t in coordinates)
    return np.column_stack(columns) / 100000, mapping


def expand_rule(rule, mapping):
    return LinearRule(tuple(rule.weights[j] for j in mapping),
                      tuple(rule.fallback[j] for j in mapping), rule.candidate_count)
