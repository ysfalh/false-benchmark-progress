"""Run the task-weighted attack on a small synthetic benchmark."""

import hashlib
import json
import numpy as np

from benchmark_progress.adapter import PublicAccess, PublicProfiles, run_attack
from benchmark_progress.evaluation import compact_trace, outcomes


class SyntheticBenchmark:
    models = ('model_a', 'model_b', 'model_c')
    coordinates = ('task_a', 'task_b')

    def __init__(self):
        rng = np.random.default_rng(7)
        self._tasks = np.repeat(self.coordinates, 40)
        self._hashes = [hashlib.sha256(f'question-{i}'.encode()).hexdigest() for i in range(80)]
        abilities = np.array([[.8, .3], [.3, .8], [.6, .6]])
        self._scores = (rng.random((3, 80)) < np.repeat(abilities, 40, axis=1)).astype(float)
        self._s = np.tile(np.arange(40) < 20, 2)
        self.public_calls = 0

    def _score(self, submission, mask):
        # Item scores, S/T membership, and this scorer remain evaluator-side.
        indices = np.flatnonzero(mask)
        chosen = np.array([self.models.index(submission(self._hashes[i], self._tasks[i])) for i in indices])
        values = self._scores[chosen, indices]
        return np.array([values[self._tasks[indices] == c].mean() for c in self.coordinates])

    def _genuine(self, mask):
        return np.stack([self._score(lambda h, t, m=m: m, mask) for m in self.models])

    def _public_profiles(self):
        return PublicProfiles(self.models, self.coordinates,
                              np.round(self._genuine(self._s)*100, 3)/100, scale=100000)

    def _public_feedback(self, submission):
        self.public_calls += 1
        return np.round(self._score(submission, self._s)*100, 3)/100

    def public_access(self):
        return PublicAccess(self._public_profiles, self._public_feedback)

    def evaluate(self, result):
        """Called only after run_attack returns a completed final router."""
        a, b = [self.models.index(result.choice[k]) for k in ['base_a', 'base_b']]
        profiles = []
        for mask in [self._s, ~self._s]:
            genuine = self._genuine(mask)
            candidates = np.array([self._score(r, mask) for r in result.candidates]).reshape(-1, 2)
            prefix = np.concatenate((genuine, genuine[[a, b]], candidates))
            final = self._score(result.final, mask)[None]
            profiles.append((prefix, final))
        (ps, fs), (pt, ft) = profiles
        trace = compact_trace(ps, pt, fs, ft, len(self.models))
        k = len(result.candidates)+1
        row = outcomes('custom', 'top5', 0, trace, budgets=[k])[0]
        row.update(reused_score_pp=float(fs.mean()*100), heldout_score_pp=float(ft.mean()*100))
        return row

    def evaluate_genuine(self, model_name):
        """Evaluate the public-selected genuine model privately, after selection."""
        i=self.models.index(model_name)
        gs=self._genuine(self._s).mean(axis=1)
        gt=self._genuine(~self._s).mean(axis=1)
        return dict(reused_score_pp=100*float(gs[i]),heldout_score_pp=100*float(gt[i]),
                    heldout_regret_pp=100*float(gt.max()-gt[i]),
                    rank_t=1+int((np.round(1-gt,10)<round(1-float(gt[i]),10)).sum()))

    def describe(self):
        return dict(name='Synthetic two-task example',
            scope='One reproducible synthetic trial demonstrating the adapter and report. It is not evidence about a real benchmark or an estimate of a false-winner rate.',
            coverage={'Genuine models':3,'Reused items':40,'Held-out items':40,
                      'Tasks / categories':2,'Per task': '20 reused + 20 held out',
                      'Missing scores':0,'Duplicate prompts crossing split':0},
            notes=['The evaluator averages the two tasks equally. Random binary outcomes come from fixed synthetic abilities and seed 7.',
                   'One split and one attack cannot establish a safe feedback policy. Task and category feedback coincide here because each category has one task.'],
            reproduction={'data':'Generated locally; no downloads or model calls',
                          'data_seed':7,'split':'First 20 of 40 examples per task reused; remainder held out'})


def main():
    benchmark = SyntheticBenchmark()
    result = run_attack(benchmark.public_access(), budget=8, routing_seed='example', tie_seed='example-ties')
    assert benchmark.public_calls == 8
    metrics = benchmark.evaluate(result)
    keys = ['new_submissions', 'score_gap_pp', 'genuine_preload_score_gap_pp',
            'rank_s', 'rank_t', 'first_s_not_t', 'heldout_regret_pp']
    displayed = {k: round(metrics[k], 6) if isinstance(metrics[k], float) else metrics[k] for k in keys}
    print(json.dumps({'endpoints': result.choice['canonical_pair'], **displayed}, indent=2))
    return metrics


if __name__ == '__main__':
    main()
