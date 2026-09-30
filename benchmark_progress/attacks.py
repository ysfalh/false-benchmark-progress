"""Endpoint selection and linear decoding from public profiles only."""

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
import hashlib
from itertools import combinations
from math import gcd
from statistics import mean

import numpy as np

def rounded_loss(values):
    return np.round(np.asarray(values, dtype=float), 10)


def publish(scores, benchmark):
    """HELM public JSON fractions; Open LLM displayed percentage hundredths."""
    values = np.asarray(scores, dtype=np.float64)
    return np.round(values * 100, 2) / 100 if benchmark == "openllm" else values.copy()


def mean_win_rate(scores):
    """Historical HELM: beat an independently chosen other row; half credit for ties."""
    scores = np.asarray(scores)
    wins = (scores[:, None] > scores[None, :]).sum(axis=1)
    ties = (scores[:, None] == scores[None, :]).sum(axis=1) - 1
    rates = (wins + ties / 2) / (len(scores) - 1)
    return np.array([mean(map(float, row)) for row in rates])


def aggregate(scores, benchmark):
    return mean_win_rate(scores) if benchmark == "helm_lite" else np.asarray(scores).mean(axis=1)


def score_scale(benchmark):
    return {"livebench": 100000, "openllm": 10000}.get(benchmark)


def rational_scores(public, scale=None):
    """Represent only the precision that the public interface releases."""
    values = np.asarray(public, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Public feedback must be finite")
    if scale is not None:
        if not isinstance(scale, int) or scale <= 0:
            raise ValueError("The public score scale must be a positive integer")
        return [[Fraction(int(round(float(v) * scale)), scale) for v in row] for row in values]
    return [[Fraction(float(v)) for v in row] for row in values]


def select_endpoints(models, public, policy="top5", *, scale=None,
                     categories=None, overall=None):
    """Select from public scores only; optional overall values must also be public.

    Scores are fractions, higher is better. Categories group public coordinates
    for equal-category averaging. Without them, all coordinates have equal weight.
    """
    if policy not in {"top5", "full_frontier"}:
        raise ValueError(policy)
    scores = np.asarray(public)
    if scores.ndim != 2 or len(models) != len(scores) or scores.shape[1] == 0:
        raise ValueError("Expected one nonempty profile per model")
    if list(models) != sorted(set(models)):
        raise ValueError("Model names must be unique and lexically ordered")
    exact = rational_scores(scores, scale)
    if categories is not None:
        categories = list(categories)
        if len(categories) != scores.shape[1]:
            raise ValueError("Each public coordinate needs a category")
        cats = tuple(dict.fromkeys(categories))
        totals = [sum(sum(v for v, cat in zip(row, categories) if cat == c) /
                      categories.count(c) for c in cats) / len(cats) for row in exact]
    else:
        totals = [sum(row)/len(row) for row in exact]
    if overall is not None:
        if np.asarray(overall).shape != (len(models),):
            raise ValueError("Expected one public overall score per model")
        totals = rational_scores([overall], scale)[0]
    dominates = np.all(scores[:, None] >= scores[None, :], axis=2) & np.any(
        scores[:, None] > scores[None, :], axis=2)
    frontier = np.flatnonzero(~dominates.any(axis=0)).tolist()
    if len(frontier) < 2:
        raise ValueError("Fewer than two public frontier models; no fallback is defined")
    pool = sorted(frontier, key=lambda i: (-totals[i], models[i]))[:min(5, len(frontier))]
    candidates = pool if policy == "top5" else frontier
    pairs = list(combinations(sorted(candidates), 2))
    distances = {pair: sum(abs(a-b) for a, b in zip(exact[pair[0]], exact[pair[1]])) for pair in pairs}
    pair = min(pairs, key=lambda pair: (-distances[pair], tuple(models[i] for i in pair)))
    def loss(i):
        if categories is not None:
            return np.mean([1 - np.mean(scores[i, np.array(categories) == c] * 100) / 100 for c in cats])
        return 1-scores[i].mean()
    a, b = sorted(pair, key=lambda i: (-float(rounded_loss(loss(i))), models[i]))
    return {"base_a": models[a], "base_b": models[b],
            "canonical_pair": [models[i] for i in pair],
            "frontier": [models[i] for i in frontier], "frontier_size": len(frontier),
            "top_five": [models[i] for i in pool], "policy": policy,
            "task_mad_pp": float(distances[pair]) * 100 / scores.shape[1]}


def select_pair(models, public, benchmark, policy, public_overall=None, coordinate_categories=None):
    """Historical benchmark conventions, using only their released columns."""
    if benchmark == "openllm" and public_overall is None:
        raise ValueError("Open LLM also publishes Average; provide that rounded public column")
    overall = (mean_win_rate(public) if benchmark == "helm_lite" else
               public_overall if benchmark == "openllm" else None)
    return select_endpoints(models, public, policy, scale=score_scale(benchmark),
        categories=coordinate_categories if benchmark == "livebench" else None, overall=overall)


@dataclass(frozen=True)
class LinearRule:
    """Integer scaling preserves exact ties in the released scores."""
    weights: tuple
    fallback: tuple
    candidate_count: int

    def route(self, gates, ties, coordinate):
        if not self.candidate_count:
            sign = self.fallback[coordinate]
            return np.full_like(ties, sign) if sign else ties.copy()
        weights = self.weights[coordinate]
        # Each coordinate can be scaled by any positive constant. Use int64
        # whenever its worst-case sum fits; Python integers otherwise.
        dtype = np.int64 if sum(abs(x) for x in weights) <= np.iinfo(np.int64).max else object
        estimate = np.array(weights, dtype=dtype) @ gates[:self.candidate_count].astype(dtype)
        return np.where(estimate == 0, ties, np.sign(estimate)).astype(np.int8)

    def record(self):
        payload = repr((self.weights, self.fallback)).encode()
        return {"candidate_count": self.candidate_count, "new_submissions": self.candidate_count+1,
                "weights_sha256": hashlib.sha256(payload).hexdigest(),
                "fallback": self.fallback}


def fit_task_rule(a, b, candidates, scale=None):
    """Only public profiles: w[j,c] = 2*p[j,c] - a[c] - b[c]."""
    a, b, *candidates = rational_scores(np.vstack((a, b, candidates)), scale)
    weights = []
    for c in range(len(a)):
        values = [2*p[c] - a[c] - b[c] for p in candidates]
        # HELM denominators are powers of two. Decimal feedback uses divisors
        # of 100,000 or 10,000; lcm is required (e.g. 8 and 25).
        denominator = reduce(lambda x, y: x*y // gcd(x, y), (v.denominator for v in values), 1)
        integers = [v.numerator * (denominator // v.denominator) for v in values]
        divisor = reduce(gcd, integers, 0) or 1
        weights.append(tuple(v // divisor for v in integers))
    return LinearRule(tuple(weights), tuple((x>y)-(x<y) for x,y in zip(a,b)), len(candidates))


def fit_rule(a, b, candidates, benchmark):
    """Apply the same combination at a historical interface's precision."""
    return fit_task_rule(a, b, candidates, score_scale(benchmark))
