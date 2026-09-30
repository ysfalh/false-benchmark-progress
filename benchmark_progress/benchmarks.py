"""Private benchmark data, public feedback, and the recorded split procedures."""

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd

from .attacks import aggregate, publish
from .routers import build_gate_bank
from .splits import select_by_task, stratified_group_selection

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ('livebench', 'helm_capabilities', 'helm_lite', 'openllm_v2', 'swe_verified')
POLICIES = ('top5', 'full_frontier')
BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def load(benchmark, root=ROOT):
    config = json.loads((root / 'configs' / f'{benchmark}.json').read_text())
    questions = pd.read_parquet(root / 'data' / f'{benchmark}_questions.parquet')
    scores = pd.read_parquet(root / 'data' / f'{benchmark}_scores.parquet')
    matrix = scores.pivot(index='question_id', columns='model', values='score').reindex(
        index=questions.question_id, columns=config['models']).to_numpy().T
    if not np.isfinite(matrix).all() or matrix.min() < 0 or matrix.max() > np.nextafter(1., 2.):
        raise ValueError('Incomplete or invalid input scores')
    return config, questions, matrix


def partition(config, questions, trial):
    seed = config['split_seed_base'] + trial
    if config['benchmark'] == 'livebench':
        targets = {c: config['holdout_per_category'] for c in config['categories']}
        holdout, _ = stratified_group_selection(questions, targets, seed)
    else:
        counts = questions.groupby('task').question_id.nunique().to_dict()
        targets = {task: int(round(config['holdout_fraction'] * n)) for task, n in counts.items()}
        holdout = select_by_task(questions, targets, seed)
    reused = set(questions.question_id) - holdout
    if reused & holdout or reused | holdout != set(questions.question_id):
        raise ValueError('S/T must be a disjoint cover')
    if set(questions.loc[questions.question_id.isin(reused), 'content_hash']) & set(
            questions.loc[questions.question_id.isin(holdout), 'content_hash']):
        raise ValueError('A duplicate prompt crosses the split')
    return reused, holdout


def make_bank(config, questions):
    metadata = [SimpleNamespace(name=c, content_hashes=tuple(
        questions.loc[questions.category == c, 'content_hash'])) for c in config['categories']]
    return build_gate_bank(metadata, max(BUDGETS)-1, config['routing_seed'], config['tie_seed'])


def category_mean(values, labels):
    return np.mean([np.ascontiguousarray(values[..., labels == task]).mean(axis=-1)
                    for task in sorted(set(labels))], axis=0)


def genuine_feedback(matrix, questions, config):
    """Evaluator-side publication; the caller passes only S item arrays."""
    parts, coordinates, categories = [], [], []
    for c in config['categories']:
        mask = (questions.category == c).to_numpy()
        data, tasks = matrix[:, mask], questions.loc[mask, 'task'].to_numpy()
        if config['benchmark'] == 'livebench':
            table = pd.DataFrame(data.T * 100, index=tasks).groupby(level=0).mean().round(3)
            parts.append(np.rint(table.to_numpy().T * 1000) / 100000)
            coordinates.extend(f'{c}/{t}' for t in table.index)
            categories.extend([c] * len(table))
        else:
            parts.append(category_mean(data, tasks)[:, None])
            coordinates.append(c)
            categories.append(c)
    values = np.concatenate(parts, axis=1)
    public = publish(values, config['benchmark'])
    overall = (publish(values.mean(axis=1), 'openllm') if config['benchmark'] == 'openllm'
               else np.mean([public[:, np.array(categories) == c].mean(axis=1)
                             for c in config['categories']], axis=0) if config['benchmark'] == 'livebench'
               else aggregate(public, config['benchmark']))
    return public, overall, coordinates, categories


def evaluate(matrix, questions, config, bank, a, b, rules=(), public=False):
    """Score one partition. Fitting receives only the public profile table.

    Gates depend on prompt hashes, never on the partition passed here. Private
    aggregation uses constituent tasks; only LiveBench releases those tasks.
    """
    prefix, finals, feedback = [], [], []
    coordinate = 0
    for category in config['categories']:
        mask = (questions.category == category).to_numpy()
        data = matrix[:, mask]
        labels = questions.loc[mask, 'task'].to_numpy()
        # make_bank's input and this partition share the original row ordering.
        positions = questions.loc[mask, '_category_position'].to_numpy(int)
        gates = bank.candidate_gates[category][:, positions]
        ties = bank.tie_gates[category][positions]
        candidates = np.where(gates == 1, data[a], data[b])
        submitted = np.concatenate((data, data[[a, b]], candidates))
        prefix.append(category_mean(submitted, labels))
        if config['benchmark'] == 'livebench':
            tasks = sorted(set(labels))
            if public:
                table = pd.DataFrame(submitted.T * 100, index=labels).groupby(level=0).mean().round(3)
                feedback.append(np.rint(table.to_numpy().T * 1000) / 100000)
            if rules:
                routed = np.empty((len(rules), len(labels)))
                for c, task in enumerate(tasks):
                    select = labels == task
                    for r, rule in enumerate(rules):
                        signs = rule.route(gates[:, select], ties[select], coordinate+c)
                        routed[r, select] = np.where(signs == 1, data[a, select], data[b, select])
                finals.append(category_mean(routed, labels))
            coordinate += len(tasks)
        else:
            if public:
                feedback.append(publish(prefix[-1], config['benchmark'])[:, None])
            if rules:
                routed = np.stack([np.where(rule.route(gates, ties, coordinate) == 1, data[a], data[b])
                                   for rule in rules])
                finals.append(category_mean(routed, labels))
            coordinate += 1
    return (np.stack(prefix, axis=1), np.stack(finals, axis=1) if rules else None,
            np.concatenate(feedback, axis=1) if public else None)
