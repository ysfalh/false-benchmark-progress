"""Readable excerpts from pinned public leaderboard tables and outcomes."""

import hashlib
import html
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent


def build(read_json, fact):
    sources = read_json(ROOT / 'docs/interface_sources.json')
    paths = {
        'helm_capabilities_core_scenarios.json': ROOT / 'data/interfaces/helm_capabilities_v1.15.0.json',
    }
    for name, path in paths.items():
        source = next(s for s in sources if s.get('file', s.get('name')) == name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256'], name

    helm = read_json(paths['helm_capabilities_core_scenarios.json'])[0]
    assert [h['value'] for h in helm['header']] == [
        'Model', 'Mean score', 'MMLU-Pro - COT correct', 'GPQA - COT correct',
        'IFEval - IFEval Strict Acc', 'WildBench - WB Score', 'Omni-MATH - Acc',
    ]
    helm_rows = []
    for key, name, label in [
        ('claude', 'Claude 3.5 Sonnet (20241022)', 'Claude 3.5 Sonnet'),
        ('gpt4o', 'GPT-4o (2024-11-20)', 'GPT-4o'),
    ]:
        row = next(r for r in helm['rows'] if r[0]['value'] == name)
        cells = []
        for i, cell in enumerate(row[1:], 1):
            value = float(cell['value']) * 100
            displayed = fact(f'example_helm_{key}_{i}', value,
                f'data/interfaces/helm_capabilities_v1.15.0.json: {name}, {helm["header"][i]["value"]}; percentage scale',
                f'{value:.1f}')
            style = ' class="aggregate-score"' if i == 1 else ''
            cells.append(f'<td{style}>{displayed}</td>')
        helm_rows.append(f'<tr><th scope="row">{label}</th>{"".join(cells)}</tr>')

    science_path = ROOT / 'results/outcome_banks/terminal_science/inputs.json'
    science = read_json(science_path)
    provenance = read_json(science_path.parent / 'provenance.json')
    assert hashlib.sha256(science_path.read_bytes()).hexdigest() == provenance['input_sha256']
    assert science['coordinates'] == ['earth', 'engineering', 'life', 'mathematical', 'physical']
    assert science['repeats'] == 3
    science_rows = []
    for key, name in [
        ('gpt6_astra', 'GPT-6 Astra / Codex / max / 76867ebc-8f35-43c5-a8b0-0cee9db46c96'),
        ('opus55', 'Opus 5.5 / Claude Code / max / cb24f3a2-82ab-4f97-be88-92a28118bf1c'),
    ]:
        scores = science['scores'][science['models'].index(name)]
        assert len(scores) == len(science['item_coordinates']) == 70
        # Inputs already average the three runs within each task. Weight tasks
        # equally overall; domain sizes differ, so do not average domain means.
        values = {'aggregate': mean(scores)}
        values.update({domain: mean(score for score, coordinate in zip(scores, science['item_coordinates'])
                                    if coordinate == domain) for domain in science['coordinates']})
        cells = []
        for column, score in values.items():
            value = 100 * score
            displayed = fact(f'example_terminal_science_{key}_{column}', value,
                f'results/outcome_banks/terminal_science/inputs.json: {name}; {column}; equal-task mean of three-run task averages; percentage scale',
                f'{value:.1f}')
            style = ' class="aggregate-score"' if column == 'aggregate' else ''
            cells.append(f'<td{style}>{displayed}</td>')
        model, agent, effort, _ = name.split(' / ')
        label = f'{html.escape(model)}<small>{html.escape(agent)} · {html.escape(effort)} effort</small>'
        science_rows.append(f'<tr><th scope="row">{label}</th>{"".join(cells)}</tr>')
    return {'helm_example_rows': ''.join(helm_rows),
            'terminal_science_example_rows': ''.join(science_rows)}
