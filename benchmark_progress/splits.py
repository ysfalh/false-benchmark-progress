"""Task-stratified splits that keep duplicate prompts together."""

from __future__ import annotations
import hashlib
from collections.abc import Mapping
import numpy as np
import pandas as pd

def hash_rank(seed: int, value: str) -> int:
    payload = f"{seed}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")

def largest_remainder_allocation(
    counts: Mapping[str, int], target: int
) -> dict[str, int]:
    """Allocate an exact total proportionally, with stable lexical tie-breaking."""

    total = int(sum(counts.values()))
    if target < 0 or target > total or total <= 0:
        raise ValueError("invalid allocation target")
    exact = {key: target * int(value) / total for key, value in counts.items()}
    allocation = {key: int(np.floor(value)) for key, value in exact.items()}
    remainder = target - sum(allocation.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - allocation[key]), key))
    for key in order[:remainder]:
        allocation[key] += 1
    if sum(allocation.values()) != target:
        raise AssertionError("largest-remainder allocation did not preserve its target")
    return allocation

def _select_exact_groups(
    groups: list[tuple[str, tuple[str, ...]]], target: int
) -> set[str]:
    """Choose the hash-earliest groups subject to an exact row-count target."""

    sizes = [len(question_ids) for _, question_ids in groups]
    suffix: list[set[int]] = [set() for _ in range(len(groups) + 1)]
    suffix[-1] = {0}
    for index in range(len(groups) - 1, -1, -1):
        size = sizes[index]
        suffix[index] = suffix[index + 1] | {
            value + size for value in suffix[index + 1] if value + size <= target
        }
    if target not in suffix[0]:
        raise RuntimeError(f"cannot keep duplicate groups intact at exact target {target}")

    selected: set[str] = set()
    remaining = target
    for index, (_, question_ids) in enumerate(groups):
        size = len(question_ids)
        if size <= remaining and remaining - size in suffix[index + 1]:
            selected.update(question_ids)
            remaining -= size
    if remaining != 0 or len(selected) != target:
        raise AssertionError("group selection failed to hit its exact target")
    return selected

def stratified_group_selection(
    metadata: pd.DataFrame,
    targets_by_category: Mapping[str, int],
    seed: int,
) -> tuple[set[str], dict[str, dict[str, int]]]:
    """Select exact category totals while approximately preserving each task mix.

    Rows sharing a normalized-prompt content hash are indivisible. Every retained
    duplicate group must therefore have one category and one LiveBench task.
    """

    required = {"question_id", "content_hash", "category", "task"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    unique = metadata[list(required)].drop_duplicates("question_id").copy()
    if len(unique) != metadata["question_id"].nunique():
        raise ValueError("question IDs have inconsistent metadata")
    group_assignments = unique.groupby("content_hash").agg(
        categories=("category", "nunique"), tasks=("task", "nunique")
    )
    if (group_assignments[["categories", "tasks"]] != 1).any().any():
        raise ValueError("an exact duplicate-prompt group crosses categories or tasks")

    selected: set[str] = set()
    realized: dict[str, dict[str, int]] = {}
    for category, target_value in targets_by_category.items():
        category_frame = unique[unique["category"] == category]
        target = int(target_value)
        task_counts = {
            str(key): int(value)
            for key, value in category_frame.groupby("task")["question_id"].nunique().items()
        }
        quotas = largest_remainder_allocation(task_counts, target)
        realized[category] = {}
        for task, quota in quotas.items():
            task_frame = category_frame[category_frame["task"] == task]
            groups = []
            for content_hash, group in task_frame.groupby("content_hash", sort=False):
                question_ids = tuple(sorted(group["question_id"].astype(str)))
                groups.append((str(content_hash), question_ids))
            groups.sort(
                key=lambda item: (
                    hash_rank(seed, f"{category}|{task}|{item[0]}"),
                    item[0],
                    item[1],
                )
            )
            task_selected = _select_exact_groups(groups, quota)
            selected.update(task_selected)
            realized[category][task] = len(task_selected)
        if sum(realized[category].values()) != target:
            raise AssertionError(f"wrong selected total for {category}")

    if len(selected) != sum(int(value) for value in targets_by_category.values()):
        raise AssertionError("selected question IDs are not globally unique")
    return selected, realized

def select_by_task(
    metadata: pd.DataFrame,
    targets: Mapping[str, int],
    seed: int,
) -> set[str]:
    selected: set[str] = set()
    for task, target_value in sorted(targets.items()):
        frame = metadata[metadata["task"] == task]
        target = int(target_value)
        groups = []
        for content_hash, group in frame.groupby("content_hash", sort=False):
            ids = tuple(sorted(group["question_id"].astype(str)))
            groups.append((str(content_hash), ids))
        groups.sort(key=lambda item: (hash_rank(seed, f"{task}|{item[0]}"), item[0], item[1]))
        selected.update(_select_exact_groups(groups, target))
    if len(selected) != sum(int(value) for value in targets.values()):
        raise AssertionError("task allocation did not preserve the exact total")
    return selected
