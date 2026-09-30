"""Restricted feedback surfaces shared by the paired study and public adapter."""
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations
import numpy as np
from .adapter import PublicAccess, PublicProfiles
from .attacks import rational_scores

MODES = ('task', 'category', 'aggregate')


def overall_scores(scores, categories=None):
    scores = np.asarray(scores)
    if categories is None:
        return scores.mean(axis=-1)
    return np.mean([scores[..., np.array(categories) == c].mean(axis=-1)
                    for c in dict.fromkeys(categories)], axis=0)


def select_policy_endpoints(models, public, policy='top5', *, scale=None,
                            categories=None, overall=None):
    """Top five by allowed overall; farthest weighted-L1 pair, lexical ties.

    This common rule remains defined for a scalar interface. No frontier or
    hidden task profiles enter it. Historical select_endpoints is unchanged.
    """
    if list(models) != sorted(set(models)) or len(models) < 2:
        raise ValueError('Need at least two unique, lexically ordered model names')
    scores = np.asarray(public, float)
    if scores.ndim != 2 or scores.shape[0] != len(models) or not scores.shape[1]:
        raise ValueError('Expected a nonempty public profile per model')
    exact = rational_scores(scores, scale)
    if categories is None:
        weights = [Fraction(1, scores.shape[1])] * scores.shape[1]
    else:
        if len(categories) != scores.shape[1]:
            raise ValueError('Categories must align with released coordinates')
        weights = [Fraction(1, len(set(categories)) * list(categories).count(c)) for c in categories]
    totals = [sum(v*w for v, w in zip(row, weights)) for row in exact]
    pool = sorted(range(len(models)), key=lambda i: (-totals[i], models[i]))[:5]
    distances = {pair: sum(abs(a-b)*w for a,b,w in zip(exact[pair[0]], exact[pair[1]], weights))
                 for pair in combinations(sorted(pool), 2)}
    pair = min(distances, key=lambda p: (-distances[p], tuple(models[i] for i in p)))
    a,b = sorted(pair, key=lambda i: (totals[i], models[i]))
    return dict(base_a=models[a], base_b=models[b], canonical_pair=[models[i] for i in pair],
                top_five=[models[i] for i in pool], policy='public_top5',
                task_mad_pp=float(distances[pair])*100)


@dataclass(frozen=True)
class MappedSubmission:
    submission: object
    mapping: dict

    def __call__(self, prompt_hash, coordinate):
        return self.submission(prompt_hash, self.mapping[coordinate])


@dataclass(frozen=True)
class FeedbackPolicy:
    mode: str
    tasks: tuple
    categories: tuple
    scale: int = 100000

    def __post_init__(self):
        if self.mode not in MODES or not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError('Invalid policy or task names')
        if len(self.categories) != len(self.tasks) or self.scale != 100000:
            raise ValueError('Aligned categories and native 0.001 pp precision required')

    @property
    def coordinates(self):
        return self.tasks if self.mode == 'task' else tuple(dict.fromkeys(self.categories)) if self.mode == 'category' else ('overall',)

    @property
    def released_categories(self):
        return self.categories if self.mode == 'task' else None

    @property
    def mapping(self):
        groups = self.tasks if self.mode == 'task' else self.categories if self.mode == 'category' else ('overall',)*len(self.tasks)
        return dict(zip(self.tasks, groups))

    def project(self, task_scores):
        """Start from native released task ticks; rational averaging, ties to even.

        Aggregate is equal-category, not equal-task. Neither a separate overall
        column nor an unrounded score is added to the restricted interface.
        """
        values = np.asarray(task_scores, float)
        if values.shape[-1] != len(self.tasks) or not np.isfinite(values).all():
            raise ValueError('Finite native task feedback must align with policy')
        ticks = np.rint(values*self.scale).astype(np.int64)
        if self.mode == 'task':
            return ticks/self.scale
        groups = [np.flatnonzero(np.array(self.categories) == c) for c in dict.fromkeys(self.categories)]
        flattened = ticks.reshape(-1, len(self.tasks))
        rows = []
        for row in flattened:
            means = [Fraction(int(row[g].sum()), len(g)) for g in groups]
            if self.mode == 'aggregate':
                means = [sum(means)/len(means)]
            rows.append([round(v) for v in means])
        return np.array(rows, float).reshape(*values.shape[:-1], len(self.coordinates))/self.scale

    def profiles(self, native):
        if tuple(native.coordinates) != self.tasks:
            raise ValueError('Public task order changed')
        return PublicProfiles(native.models, self.coordinates, self.project(native.scores),
                              self.scale, self.released_categories)

    def restrict(self, public):
        """Evaluator-side wrapper: attack receives only these two callbacks."""
        return PublicAccess(lambda: self.profiles(public.genuine_public_profiles()),
            lambda submission: self.project(public.public_feedback(MappedSubmission(submission, self.mapping))))

    def for_evaluator(self, result):
        return replace(result, final=MappedSubmission(result.final, self.mapping))
