"""Paired strict-success statistics for two trajectory runs."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping

from shopping_grpo.evaluation.summary import is_strict_success

PAIRED_STRICT_SCHEMA_VERSION = "shopping-paired-strict-comparison-v1"


def _index_complete_run(
    label: str,
    trajectories: Iterable[Mapping],
    expected_ids: set[int],
) -> dict[int, Mapping]:
    indexed = {}
    for trajectory in trajectories:
        task_id = int(trajectory["task_id"])
        if task_id not in expected_ids:
            raise ValueError(f"{label} contains unexpected task_id {task_id}")
        if task_id in indexed:
            raise ValueError(f"{label} contains duplicate task_id {task_id}")
        indexed[task_id] = trajectory

    missing = sorted(expected_ids - set(indexed))
    if missing:
        raise ValueError(f"{label} is missing {len(missing)} task(s): {missing[:10]}")
    return indexed


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sample")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _paired_bootstrap_interval(
    deltas: list[int],
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not deltas:
        raise ValueError("at least one paired task is required")

    rng = random.Random(seed)
    task_count = len(deltas)
    estimates = [
        sum(deltas[rng.randrange(task_count)] for _ in range(task_count))
        / task_count
        for _ in range(resamples)
    ]
    estimates.sort()
    alpha = 1.0 - confidence
    return (
        _percentile(estimates, alpha / 2.0),
        _percentile(estimates, 1.0 - alpha / 2.0),
    )


def _exact_mcnemar_p_value(losses: int, gains: int) -> float:
    discordant = losses + gains
    if discordant == 0:
        return 1.0
    smaller = min(losses, gains)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
    return min(1.0, 2.0 * lower_tail / (2**discordant))


def compare_strict_success(
    *,
    expected_task_ids: Iterable[int],
    source_trajectories: Iterable[Mapping],
    target_trajectories: Iterable[Mapping],
    source_label: str,
    target_label: str,
    confidence: float = 0.95,
    bootstrap_resamples: int = 20_000,
    bootstrap_seed: int = 20260809,
) -> dict:
    """Compare strict Gold outcomes on exactly the same complete task set."""

    expected = [int(task_id) for task_id in expected_task_ids]
    if not expected:
        raise ValueError("expected_task_ids must not be empty")
    if len(set(expected)) != len(expected):
        raise ValueError("expected_task_ids contains duplicates")
    expected_set = set(expected)
    source = _index_complete_run(source_label, source_trajectories, expected_set)
    target = _index_complete_run(target_label, target_trajectories, expected_set)

    transitions = {
        "failure_to_failure": 0,
        "failure_to_success": 0,
        "success_to_failure": 0,
        "success_to_success": 0,
    }
    gains = []
    losses = []
    deltas = []
    source_successes = 0
    target_successes = 0
    for task_id in expected:
        source_success = is_strict_success(source[task_id])
        target_success = is_strict_success(target[task_id])
        source_successes += int(source_success)
        target_successes += int(target_success)
        transitions[
            f"{'success' if source_success else 'failure'}_to_"
            f"{'success' if target_success else 'failure'}"
        ] += 1
        delta = int(target_success) - int(source_success)
        deltas.append(delta)
        if delta > 0:
            gains.append(task_id)
        elif delta < 0:
            losses.append(task_id)

    ci_lower, ci_upper = _paired_bootstrap_interval(
        deltas,
        confidence=confidence,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    task_count = len(expected)
    return {
        "schema_version": PAIRED_STRICT_SCHEMA_VERSION,
        "source": {
            "label": source_label,
            "strict_successes": source_successes,
            "strict_success_rate": source_successes / task_count,
        },
        "target": {
            "label": target_label,
            "strict_successes": target_successes,
            "strict_success_rate": target_successes / task_count,
        },
        "paired_tasks": task_count,
        "strict_success_transitions": transitions,
        "gains": len(gains),
        "losses": len(losses),
        "gain_task_ids": gains,
        "loss_task_ids": losses,
        "strict_success_rate_delta_target_minus_source": (
            target_successes - source_successes
        )
        / task_count,
        "paired_confidence_interval": {
            "method": "paired-percentile-bootstrap",
            "confidence": confidence,
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "lower": ci_lower,
            "upper": ci_upper,
        },
        "mcnemar": {
            "method": "exact-two-sided-binomial",
            "discordant_pairs": len(gains) + len(losses),
            "p_value": _exact_mcnemar_p_value(len(losses), len(gains)),
        },
    }
