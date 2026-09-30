"""Cross-check the displayed values against the canonical result files."""

import csv
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from statistics import mean
from urllib.parse import urlsplit
from benchmark_progress.benchmarks import ROOT


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets = []
        self.facts = {}
        self.current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ['src', 'href', 'srcset']:
            if key in attrs:
                self.assets.append(attrs[key])
        self.current = attrs.get('data-fact') if tag == 'span' else self.current

    def handle_data(self, data):
        if self.current:
            self.facts[self.current] = data

    def handle_endtag(self, tag):
        if tag == 'span':
            self.current = None


def test_site_statistics_and_assets():
    site = ROOT/'site'
    page = Page()
    page.feed((site/'index.html').read_text())
    ledger = json.loads((site/'data/provenance.json').read_text())
    for path, sha in ledger['sources_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == sha
    for key, display in page.facts.items():
        assert ledger['facts'][key]['display'] == display
    for asset in page.assets:
        if not asset.startswith(('https:', 'http:', 'data:', '#')):
            assert (site/urlsplit(asset).path).is_file(), asset
    with (ROOT/'results/release_v2/headline.csv').open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['policy'] == 'top5']
    expected = {'mean_attack_gap': mean(float(r['queried_score_gap_pp']) for r in rows),
                'mean_baseline_gap': mean(float(r['score_gap_baseline_pp']) for r in rows)}
    with (ROOT/'results/release_v2/threshold_crossings.csv').open() as stream:
        thresholds = [r for r in csv.DictReader(stream) if r['policy'] == 'top5']
    expected['mean_selection_budget'] = mean(int(r['smallest_tested_random_budget'])
                                            for r in thresholds if r['metric'] == 'first_s_not_t')
    for row in thresholds:
        prefix = 'threshold_' if row['metric'] == 'first_s_not_t' else 'score_threshold_'
        expected[prefix+row['benchmark']] = int(row['smallest_tested_random_budget']) if row['smallest_tested_random_budget'] else None
    for key, value in expected.items():
        if value is None:
            assert ledger['facts'][key]['value'] is None
        else:
            assert math.isclose(ledger['facts'][key]['value'], value, rel_tol=0, abs_tol=1e-12)
    readme = (ROOT/'README.md').read_text()
    assert 'site/figures/selection-loss.svg' in readme
    assert '0.93–4.80' in readme
    assert not re.search(r'\{\{[^}]+\}\}', (site/'index.html').read_text())


def test_native_chart_points_match_their_downloads():
    specs = json.loads((ROOT/'site/data/native_charts.json').read_text())
    assert [len(g['panels']) for g in specs[:2]] == [6,6]
    tables = {}
    for group in specs:
        for panel in group['panels']:
            for series in panel.get('series',[]):
                for point in series['points']:
                    src = point['source']
                    if src['file'] not in tables:
                        with (ROOT/'site'/src['file']).open() as f:
                            tables[src['file']] = list(csv.DictReader(f))
                    row = tables[src['file']][src['row']]
                    assert math.isclose(point['y'],float(row[src['key']])*src['factor'],abs_tol=1e-12)
                    for label,key in [('low',src['low']),('high',src['high'])]:
                        if key:
                            assert math.isclose(point[label],float(row[key])*src['factor'],abs_tol=1e-12)
