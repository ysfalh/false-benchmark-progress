"""A small public interface for running the same attack on another benchmark."""

from dataclasses import dataclass
from typing import Callable, Optional, Protocol
import numpy as np

from .attacks import LinearRule, fit_task_rule, select_endpoints
from .routers import _candidate_key, _prf_sign, _tie_key


@dataclass(frozen=True)
class PublicProfiles:
    models: tuple[str, ...]                 # Unique names in lexical order.
    coordinates: tuple[str, ...]            # Public task/scenario identities.
    scores: np.ndarray                     # Model by coordinate; fractions.
    scale: Optional[int] = None            # E.g. 100000 for 0.001 percentage points.
    categories: Optional[tuple[str, ...]] = None
    overall: Optional[np.ndarray] = None   # Only if separately public.


class Submission(Protocol):
    def __call__(self, prompt_hash: str, coordinate: str) -> str:
        """Choose a model using public prompt metadata only."""
        ...


class BenchmarkAdapter(Protocol):
    def genuine_public_profiles(self) -> PublicProfiles: ...
    def public_feedback(self, submission: Submission) -> np.ndarray: ...


@dataclass(frozen=True)
class PublicAccess:
    """Expose two callbacks; do not pass a private evaluator to the attack."""
    genuine_public_profiles: Callable[[], PublicProfiles]
    public_feedback: Callable[[Submission], np.ndarray]


@dataclass(frozen=True)
class RandomRouter:
    base_a: str
    base_b: str
    key: bytes

    def __call__(self, prompt_hash, coordinate):
        return self.base_a if _prf_sign(self.key, prompt_hash) == 1 else self.base_b


@dataclass(frozen=True)
class FinalRouter:
    base_a: str
    base_b: str
    coordinates: tuple[str, ...]
    keys: tuple[bytes, ...]
    tie_key: bytes
    rule: LinearRule

    def __call__(self, prompt_hash, coordinate):
        c = self.coordinates.index(coordinate)
        gates = np.array([_prf_sign(key, prompt_hash) for key in self.keys], dtype=np.int8)[:, None]
        ties = np.array([_prf_sign(self.tie_key, prompt_hash)], dtype=np.int8)
        return self.base_a if self.rule.route(gates, ties, c)[0] == 1 else self.base_b


@dataclass(frozen=True)
class AttackResult:
    choice: dict
    candidates: tuple[RandomRouter, ...]
    final: FinalRouter
    public_feedback: np.ndarray


def run_attack(public: BenchmarkAdapter, budget=32, *, routing_seed="bank0",
               tie_seed="ties0", policy="top5", endpoint_selector=select_endpoints) -> AttackResult:
    """Use k-1 random submissions and one final submission; never evaluate T."""
    if not isinstance(budget, int) or budget < 1:
        raise ValueError("The budget must be a positive integer")
    profiles = public.genuine_public_profiles()
    scores = np.array(profiles.scores, dtype=float, copy=True)
    coordinates = tuple(profiles.coordinates)
    if len(set(coordinates)) != len(coordinates) or scores.shape != (len(profiles.models), len(coordinates)):
        raise ValueError("Public profiles and coordinate names must align")
    choice = endpoint_selector(profiles.models, scores, policy, scale=profiles.scale,
                              categories=profiles.categories, overall=profiles.overall)
    a, b = choice['base_a'], choice['base_b']
    keys = tuple(_candidate_key(routing_seed, i) for i in range(budget-1))
    candidates = tuple(RandomRouter(a, b, key) for key in keys)
    def feedback(submission):
        row = np.array(public.public_feedback(submission), dtype=float, copy=True)
        if row.shape != (len(coordinates),) or not np.isfinite(row).all():
            raise ValueError("Feedback must be one finite public score per coordinate")
        return row
    released = np.stack([feedback(r) for r in candidates]) if candidates else np.empty((0, len(coordinates)))
    rule = fit_task_rule(scores[profiles.models.index(a)], scores[profiles.models.index(b)],
                         released, profiles.scale)
    final = FinalRouter(a, b, coordinates, keys, _tie_key(tie_seed), rule)
    # The final response is recorded but cannot alter the completed router.
    return AttackResult(choice, candidates, final, np.vstack((released, feedback(final))))
