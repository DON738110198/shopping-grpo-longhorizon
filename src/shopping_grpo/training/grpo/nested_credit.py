"""Fail-closed statistics for nested PSA decision credit.

This module deliberately stops at the contract/statistics boundary.  It does
not sample continuations, touch an optimizer, or decide whether a terminal
outcome is "mixed".  A caller supplies proposal multiplicities and exactly
``L`` continuation slots for each distinct first decision; the estimator then
shrinks noisy decision means toward their state mean and emits bounded local
advantages.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path

from shopping_grpo.environment.manifest import validate_manifest
from shopping_grpo.training.grpo.active_branch import (
    build_active_branch_plan,
    validate_active_branch_plan,
)
from shopping_grpo.training.grpo.active_suffix import (
    sampling_backend_contract_sha256,
    sha256_actor_checkpoint,
    validate_sampling_backend_contract,
)
from shopping_grpo.training.grpo.adapter.runtime import (
    REWARD_V3_TYPES,
    reward_breakdown,
    validate_policy_reward_config,
)
from shopping_grpo.training.grpo.nested_artifacts import (
    ACTOR_RUN_ATTESTATION_VERSION,
    build_nested_journal_contract,
    build_nested_stage1_source_binding,
    nested_journal_header_sha256,
    sha256_file,
    validate_nested_stage1_source_binding,
    verify_completed_collection_manifest,
)
from shopping_grpo.training.grpo.nested_continuation import (
    FORMAL_CONTINUATIONS_PER_DECISION as COLLECTOR_FORMAL_CONTINUATIONS,
)
from shopping_grpo.training.grpo.nested_continuation import (
    GATE_CONTINUATION_INDICES as COLLECTOR_GATE_INDICES,
)
from shopping_grpo.training.grpo.nested_continuation import (
    NESTED_COLLECTION_VERSION,
    NESTED_CONTINUATION_VERSION,
    NESTED_DECISION_CONTENT_VERSION,
    NESTED_DECISION_IDENTITY_VERSION,
    NESTED_DECISION_VERSION,
    NESTED_EXCLUSION_AUDIT_VERSION,
    NESTED_FOLD_CONTRACT_VERSION,
    NESTED_FORMAL_PLAN_VERSION,
    NESTED_HARNESS_CONTRACT_VERSION,
    NESTED_ROLLOUT_CONTENT_VERSION,
    build_nested_formal_plan,
)
from shopping_grpo.training.grpo.nested_continuation import (
    NESTED_SEED_SCHEDULE_VERSION as COLLECTOR_SEED_SCHEDULE_VERSION,
)
from shopping_grpo.training.grpo.nested_continuation import (
    TRAIN_CONTINUATION_INDICES as COLLECTOR_TRAIN_INDICES,
)
from shopping_grpo.training.grpo.nested_structure import (
    classify_nested_stage1_structure,
)
from shopping_grpo.training.grpo.pivotal_states import (
    canonical_replay_action,
    replay_action_sha256,
    token_ids_sha256,
)
from shopping_grpo.training.grpo.stage1_proposal import (
    verify_first_decision_artifacts,
)

NESTED_SAMPLES_VERSION = "shopping-psa-nested-samples-v4"
NESTED_CREDIT_VERSION = "shopping-psa-nested-decision-credit-v4"
NESTED_ESTIMATOR_VERSION = "shopping-psa-nested-crn-shrinkage-v4"
NESTED_HELDOUT_VERSION = "shopping-psa-train4-gate4-v3"
NESTED_TRAINING_GATE_VERSION = "shopping-psa-nested-training-gate-v3"
NESTED_SIGNAL_GATE_VERSION = "shopping-psa-nested-signal-gate-v3"
NESTED_RESAMPLING_VERSION = "shopping-psa-task-cluster-resampling-v3"
NESTED_SOURCE_ATTESTATION_VERSION = "shopping-psa-nested-source-attestation-v1"

NESTED_SEED_SCHEDULE_VERSION = COLLECTOR_SEED_SCHEDULE_VERSION
TRAIN_FOLD_INDICES = COLLECTOR_TRAIN_INDICES
GATE_FOLD_INDICES = COLLECTOR_GATE_INDICES
FORMAL_CONTINUATIONS_PER_DECISION = COLLECTOR_FORMAL_CONTINUATIONS

MIN_PROPOSALS_PER_STATE = 4
MIN_DISTINCT_DECISIONS = 2
MIN_CONTINUATIONS_PER_DECISION = 4
MAX_SAMPLING_INVALID_RATE = 0.05
MIN_TRAINING_STATES = 32
MIN_STATE_ESS = 32.0
MIN_TRAINING_TASKS = 32
MIN_TASK_ESS = 32.0
ADVANTAGE_CLIP = 0.5
SHRINKAGE_EPSILON = 1e-8
CRN_COVARIANCE_DIAGONAL_SHRINKAGE = 0.5
MIN_HIGH_KAPPA_STATE_RATE = 0.30
HIGH_KAPPA_THRESHOLD = 0.10
MIN_HELDOUT_MEAN_DELTA = 0.10
MIN_SPLIT_HALF_CONSISTENCY = 0.60
MIN_IDENTIFIABLE_TASK_ESS = 32.0
BOOTSTRAP_SAMPLES = 20_000
PERMUTATION_SAMPLES = 20_000
DIAGNOSTIC_SEED = 20_260_811

_INPUT_FIELDS = {
    "schema_version",
    "experiment_uid",
    "reward_contract_sha256",
    "collected_continuations_per_decision",
    "train_fold_indices",
    "gate_fold_indices",
    "source_attestation",
    "states",
}
_STATE_FIELDS = {"state_index", "state_uid", "task_id", "proposals", "decisions"}
_PROPOSAL_FIELDS = {"proposal_index", "proposal_uid", "decision_uid"}
_DECISION_FIELDS = {"decision_index", "decision_uid", "continuations"}
_CONTINUATION_FIELDS = {
    "continuation_index",
    "continuation_uid",
    "continuation_seed_uid",
    "downstream_request_seeds",
    "rollout_content_sha256",
    "generation_mode",
    "reward",
    "valid_for_learning",
    "infrastructure_invalid",
    "reward_invalid",
    "reward_unverifiable",
    "sampling_invalid",
    "model_failure",
    "invalid_reason",
}
_SOURCE_ATTESTATION_FIELDS = {
    "schema_version",
    "attested",
    "contract_sha256",
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _required_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
    return value


def _required_index(value: object, expected: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        raise ValueError(f"{name} must equal {expected}")
    return value


def _finite_reward(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    reward = float(value)
    if not -1.0 <= reward <= 1.0:
        raise ValueError(f"{name} must be in [-1, 1]")
    return reward


def proposal_uid(state_uid: str, proposal_index: int) -> str:
    """Return the contract identity for one first-decision proposal slot."""
    _required_sha256(state_uid, "state_uid")
    if not isinstance(proposal_index, int) or isinstance(proposal_index, bool):
        raise TypeError("proposal_index must be an integer")
    if proposal_index < 0:
        raise ValueError("proposal_index must be non-negative")
    return _sha256_json({"state_uid": state_uid, "proposal_index": proposal_index})


def continuation_seed_uid(state_uid: str, continuation_index: int) -> str:
    """Bind one common-random-number slot before any decision is evaluated."""
    _required_sha256(state_uid, "state_uid")
    if not isinstance(continuation_index, int) or isinstance(continuation_index, bool):
        raise TypeError("continuation_index must be an integer")
    if continuation_index < 0:
        raise ValueError("continuation_index must be non-negative")
    return _sha256_json(
        {
            "version": NESTED_SEED_SCHEDULE_VERSION,
            "state_uid": state_uid,
            "continuation_index": continuation_index,
        }
    )


def continuation_turn_seed(
    state_uid: str,
    continuation_index: int,
    turn_index: int,
) -> int:
    """Derive the exact hierarchical seed used for one downstream actor turn."""
    continuation_seed_uid(state_uid, continuation_index)
    if not isinstance(turn_index, int) or isinstance(turn_index, bool):
        raise TypeError("turn_index must be an integer")
    if turn_index < 0:
        raise ValueError("turn_index must be non-negative")
    digest = hashlib.sha256(
        _canonical_json(
            {
                "version": NESTED_SEED_SCHEDULE_VERSION,
                "state_uid": state_uid,
                "continuation_index": continuation_index,
                "turn_index": turn_index,
            }
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def continuation_uid(state_uid: str, decision_uid: str, continuation_index: int) -> str:
    """Bind a continuation to its decision, CRN slot, and seed schedule."""
    _required_sha256(decision_uid, "decision_uid")
    seed_uid = continuation_seed_uid(state_uid, continuation_index)
    return _sha256_json(
        {
            "state_uid": state_uid,
            "decision_uid": decision_uid,
            "continuation_index": continuation_index,
            "seed_schedule_version": NESTED_SEED_SCHEDULE_VERSION,
            "continuation_seed_uid": seed_uid,
        }
    )


def _require_exact_fields(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{name} fields mismatch: missing={missing}, extra={extra}")


def _require_sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an array")
    return value


def validate_nested_samples(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize the complete nested-sampling identity contract."""
    if not isinstance(payload, Mapping):
        raise TypeError("nested samples must be an object")
    _require_exact_fields(payload, _INPUT_FIELDS, "nested samples")
    if payload.get("schema_version") != NESTED_SAMPLES_VERSION:
        raise ValueError("nested samples schema_version mismatch")
    experiment_uid = _required_sha256(payload.get("experiment_uid"), "experiment_uid")
    reward_sha = _required_sha256(
        payload.get("reward_contract_sha256"), "reward_contract_sha256"
    )
    continuation_count = payload.get("collected_continuations_per_decision")
    if (
        not isinstance(continuation_count, int)
        or isinstance(continuation_count, bool)
        or continuation_count < 1
    ):
        raise ValueError("collected_continuations_per_decision must be a positive integer")
    if continuation_count > FORMAL_CONTINUATIONS_PER_DECISION:
        raise ValueError("nested samples exceed the fixed train4/gate4 fold contract")
    train_indices = payload.get("train_fold_indices")
    gate_indices = payload.get("gate_fold_indices")
    if train_indices != list(TRAIN_FOLD_INDICES) or gate_indices != list(
        GATE_FOLD_INDICES
    ):
        raise ValueError("nested samples must use the fixed train4/gate4 fold contract")
    raw_attestation = payload.get("source_attestation")
    if not isinstance(raw_attestation, Mapping):
        raise TypeError("source_attestation must be an object")
    _require_exact_fields(
        raw_attestation,
        _SOURCE_ATTESTATION_FIELDS,
        "source_attestation",
    )
    if raw_attestation.get("schema_version") != NESTED_SOURCE_ATTESTATION_VERSION:
        raise ValueError("source_attestation schema_version mismatch")
    source_attested = raw_attestation.get("attested")
    if not isinstance(source_attested, bool):
        raise TypeError("source_attestation attested must be boolean")
    source_contract_sha = raw_attestation.get("contract_sha256")
    if source_attested:
        source_contract_sha = _required_sha256(
            source_contract_sha, "source_attestation contract_sha256"
        )
    elif source_contract_sha is not None:
        raise ValueError("unattested nested samples cannot carry a source contract hash")

    raw_states = _require_sequence(payload.get("states"), "states")
    normalized_states: list[dict[str, object]] = []
    seen_state_uids: set[str] = set()
    seen_proposal_uids: set[str] = set()
    seen_continuation_uids: set[str] = set()

    for state_index, raw_state in enumerate(raw_states):
        if not isinstance(raw_state, Mapping):
            raise TypeError(f"state {state_index} must be an object")
        _require_exact_fields(raw_state, _STATE_FIELDS, f"state {state_index}")
        _required_index(raw_state.get("state_index"), state_index, "state_index")
        state_uid = _required_sha256(raw_state.get("state_uid"), "state_uid")
        if state_uid in seen_state_uids:
            raise ValueError("nested samples repeat state_uid")
        seen_state_uids.add(state_uid)
        task_id = raw_state.get("task_id")
        if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id < 0:
            raise ValueError(f"state {state_index} task_id must be a non-negative integer")

        raw_proposals = _require_sequence(raw_state.get("proposals"), "proposals")
        if not raw_proposals:
            raise ValueError(f"state {state_index} must contain at least one proposal")
        normalized_proposals: list[dict[str, object]] = []
        proposal_decision_uids: list[str] = []
        for proposal_index, raw_proposal in enumerate(raw_proposals):
            if not isinstance(raw_proposal, Mapping):
                raise TypeError(f"proposal {state_index}:{proposal_index} must be an object")
            _require_exact_fields(
                raw_proposal,
                _PROPOSAL_FIELDS,
                f"proposal {state_index}:{proposal_index}",
            )
            _required_index(
                raw_proposal.get("proposal_index"),
                proposal_index,
                "proposal_index",
            )
            expected_proposal_uid = proposal_uid(state_uid, proposal_index)
            if raw_proposal.get("proposal_uid") != expected_proposal_uid:
                raise ValueError(f"proposal {state_index}:{proposal_index} proposal_uid mismatch")
            if expected_proposal_uid in seen_proposal_uids:
                raise ValueError("nested samples repeat proposal_uid")
            seen_proposal_uids.add(expected_proposal_uid)
            decision_uid = _required_sha256(
                raw_proposal.get("decision_uid"), "proposal decision_uid"
            )
            proposal_decision_uids.append(decision_uid)
            normalized_proposals.append(
                {
                    "proposal_index": proposal_index,
                    "proposal_uid": expected_proposal_uid,
                    "decision_uid": decision_uid,
                }
            )

        raw_decisions = _require_sequence(raw_state.get("decisions"), "decisions")
        if not raw_decisions:
            raise ValueError(f"state {state_index} must contain at least one decision")
        normalized_decisions: list[dict[str, object]] = []
        seen_decision_uids: set[str] = set()
        state_slot_first_seeds: dict[int, int] = {}
        for decision_index, raw_decision in enumerate(raw_decisions):
            if not isinstance(raw_decision, Mapping):
                raise TypeError(f"decision {state_index}:{decision_index} must be an object")
            _require_exact_fields(
                raw_decision,
                _DECISION_FIELDS,
                f"decision {state_index}:{decision_index}",
            )
            _required_index(
                raw_decision.get("decision_index"),
                decision_index,
                "decision_index",
            )
            decision_uid = _required_sha256(raw_decision.get("decision_uid"), "decision_uid")
            if decision_uid in seen_decision_uids:
                raise ValueError(f"state {state_index} repeats decision_uid")
            seen_decision_uids.add(decision_uid)

            raw_continuations = _require_sequence(
                raw_decision.get("continuations"), "continuations"
            )
            if len(raw_continuations) != continuation_count:
                raise ValueError(
                    f"decision {state_index}:{decision_index} must contain exactly "
                    f"{continuation_count} continuation slots"
                )
            normalized_continuations: list[dict[str, object]] = []
            for continuation_index, raw_continuation in enumerate(raw_continuations):
                if not isinstance(raw_continuation, Mapping):
                    raise TypeError(
                        f"continuation {state_index}:{decision_index}:{continuation_index} "
                        "must be an object"
                    )
                _require_exact_fields(
                    raw_continuation,
                    _CONTINUATION_FIELDS,
                    f"continuation {state_index}:{decision_index}:{continuation_index}",
                )
                _required_index(
                    raw_continuation.get("continuation_index"),
                    continuation_index,
                    "continuation_index",
                )
                expected_continuation_uid = continuation_uid(
                    state_uid, decision_uid, continuation_index
                )
                if raw_continuation.get("continuation_uid") != expected_continuation_uid:
                    raise ValueError(
                        f"continuation {state_index}:{decision_index}:{continuation_index} "
                        "continuation_uid mismatch"
                    )
                if expected_continuation_uid in seen_continuation_uids:
                    raise ValueError("nested samples repeat continuation_uid")
                seen_continuation_uids.add(expected_continuation_uid)
                expected_seed_uid = continuation_seed_uid(state_uid, continuation_index)
                if raw_continuation.get("continuation_seed_uid") != expected_seed_uid:
                    raise ValueError(
                        f"continuation {state_index}:{decision_index}:{continuation_index} "
                        "continuation_seed_uid mismatch"
                    )
                raw_request_seeds = _require_sequence(
                    raw_continuation.get("downstream_request_seeds"),
                    "downstream_request_seeds",
                )
                request_seeds: list[int] = []
                for turn_index, raw_seed in enumerate(raw_request_seeds):
                    if (
                        not isinstance(raw_seed, int)
                        or isinstance(raw_seed, bool)
                        or raw_seed != continuation_turn_seed(
                            state_uid, continuation_index, turn_index
                        )
                    ):
                        raise ValueError("downstream request seed schedule mismatch")
                    request_seeds.append(raw_seed)
                if request_seeds:
                    previous = state_slot_first_seeds.setdefault(
                        continuation_index, request_seeds[0]
                    )
                    if previous != request_seeds[0]:
                        raise AssertionError("CRN first seed changed across decisions")
                rollout_content_sha = _required_sha256(
                    raw_continuation.get("rollout_content_sha256"),
                    "rollout_content_sha256",
                )

                valid = raw_continuation.get("valid_for_learning")
                infrastructure_invalid = raw_continuation.get("infrastructure_invalid")
                reward_invalid = raw_continuation.get("reward_invalid")
                reward_unverifiable = raw_continuation.get("reward_unverifiable")
                sampling_invalid = raw_continuation.get("sampling_invalid")
                model_failure = raw_continuation.get("model_failure")
                generation_mode = raw_continuation.get("generation_mode")
                if generation_mode not in {
                    "sampled_continuation",
                    "deterministic_terminal",
                    "no_generation_terminal",
                    "infrastructure_invalid",
                }:
                    raise ValueError("continuation generation_mode is invalid")
                if not all(
                    isinstance(value, bool)
                    for value in (
                        valid,
                        infrastructure_invalid,
                        reward_invalid,
                        reward_unverifiable,
                        sampling_invalid,
                        model_failure,
                    )
                ):
                    raise TypeError("continuation validity fields must be boolean")
                invalid_reason = raw_continuation.get("invalid_reason")
                if sampling_invalid is not (not valid):
                    raise ValueError("sampling_invalid must equal not valid_for_learning")
                if reward_invalid and (infrastructure_invalid or model_failure):
                    raise ValueError("reward-invalid continuation category is inconsistent")
                if model_failure and (not valid or sampling_invalid):
                    raise ValueError(
                        "model-attributable failures must remain learning-valid"
                    )
                if reward_unverifiable is not (
                    reward_invalid and invalid_reason == "reward_unverifiable"
                ):
                    raise ValueError("reward_unverifiable category is inconsistent")
                if valid:
                    if generation_mode == "sampled_continuation" and not request_seeds:
                        raise ValueError(
                            "learning-valid continuation needs downstream request seeds"
                        )
                    if generation_mode in {
                        "deterministic_terminal",
                        "no_generation_terminal",
                    } and request_seeds:
                        raise ValueError(
                            "no-generation continuation cannot sample downstream"
                        )
                    if generation_mode == "infrastructure_invalid":
                        raise ValueError("learning-valid continuation has invalid mode")
                    reward = _finite_reward(raw_continuation.get("reward"), "reward")
                    if (
                        infrastructure_invalid
                        or reward_invalid
                        or reward_unverifiable
                        or sampling_invalid
                        or invalid_reason is not None
                    ):
                        raise ValueError(
                            "learning-valid continuation cannot be sampling-invalid"
                        )
                else:
                    reward = None
                    if raw_continuation.get("reward") is not None:
                        raise ValueError("learning-invalid continuation cannot carry a reward")
                    if infrastructure_invalid:
                        if reward_invalid or generation_mode != "infrastructure_invalid":
                            raise ValueError("infrastructure-invalid continuation mode mismatch")
                    elif reward_invalid:
                        if generation_mode == "infrastructure_invalid":
                            raise ValueError("reward-invalid continuation mode mismatch")
                    else:
                        raise ValueError(
                            "learning-invalid continuation lacks an auditable invalid category"
                        )
                    if (
                        not isinstance(invalid_reason, str)
                        or not invalid_reason
                        or len(invalid_reason) > 128
                    ):
                        raise ValueError(
                            "sampling-invalid continuation needs a bounded invalid_reason"
                        )
                normalized_continuations.append(
                    {
                        "continuation_index": continuation_index,
                        "continuation_uid": expected_continuation_uid,
                        "continuation_seed_uid": expected_seed_uid,
                        "downstream_request_seeds": request_seeds,
                        "rollout_content_sha256": rollout_content_sha,
                        "generation_mode": generation_mode,
                        "reward": reward,
                        "valid_for_learning": valid,
                        "infrastructure_invalid": infrastructure_invalid,
                        "reward_invalid": reward_invalid,
                        "reward_unverifiable": reward_unverifiable,
                        "sampling_invalid": sampling_invalid,
                        "model_failure": model_failure,
                        "invalid_reason": invalid_reason,
                    }
                )
            normalized_decisions.append(
                {
                    "decision_index": decision_index,
                    "decision_uid": decision_uid,
                    "continuations": normalized_continuations,
                }
            )

        first_seeds = list(state_slot_first_seeds.values())
        if len(first_seeds) != len(set(first_seeds)):
            raise ValueError("continuation slots repeat a hierarchical first-turn seed")

        if set(proposal_decision_uids) != seen_decision_uids:
            raise ValueError(
                f"state {state_index} proposals and decisions do not define the same identities"
            )
        normalized_states.append(
            {
                "state_index": state_index,
                "state_uid": state_uid,
                "task_id": task_id,
                "proposals": normalized_proposals,
                "decisions": normalized_decisions,
            }
        )

    return {
        "schema_version": NESTED_SAMPLES_VERSION,
        "experiment_uid": experiment_uid,
        "reward_contract_sha256": reward_sha,
        "collected_continuations_per_decision": continuation_count,
        "train_fold_indices": list(TRAIN_FOLD_INDICES),
        "gate_fold_indices": list(GATE_FOLD_INDICES),
        "source_attestation": {
            "schema_version": NESTED_SOURCE_ATTESTATION_VERSION,
            "attested": source_attested,
            "contract_sha256": source_contract_sha,
        },
        "states": normalized_states,
    }


def _sample_mean_and_variance_of_mean(rewards: Sequence[float]) -> tuple[float, float]:
    mean = math.fsum(rewards) / len(rewards)
    sample_variance = math.fsum((reward - mean) ** 2 for reward in rewards) / (
        len(rewards) - 1
    )
    return mean, sample_variance / len(rewards)


def _paired_crn_covariance_of_means(
    reward_rows: Sequence[Sequence[float]],
) -> tuple[list[float], list[list[float]], list[list[float]]]:
    """Estimate a PSD covariance of decision means from aligned CRN slots.

    With four train slots, the raw sample covariance is rank limited.  A fixed,
    preregistered 50% shrink toward its diagonal remains PSD without estimating
    a reward-dependent shrinkage coefficient from this tiny sample.
    """
    if len(reward_rows) < 2:
        raise ValueError("paired CRN covariance needs at least two decisions")
    slot_count = len(reward_rows[0])
    if slot_count < 2 or any(len(row) != slot_count for row in reward_rows):
        raise ValueError("paired CRN reward rows must have equal slot cardinality")
    means = [math.fsum(row) / slot_count for row in reward_rows]
    centered = [
        [float(value) - mean for value in row]
        for row, mean in zip(reward_rows, means, strict=True)
    ]
    scale = 1.0 / ((slot_count - 1) * slot_count)
    raw = [
        [
            math.fsum(
                centered[left][slot] * centered[right][slot]
                for slot in range(slot_count)
            )
            * scale
            for right in range(len(centered))
        ]
        for left in range(len(centered))
    ]
    shrinkage = CRN_COVARIANCE_DIAGONAL_SHRINKAGE
    shrunk = [
        [
            raw[left][right]
            if left == right
            else (1.0 - shrinkage) * raw[left][right]
            for right in range(len(raw))
        ]
        for left in range(len(raw))
    ]
    if any(
        not math.isfinite(value)
        for matrix in (raw, shrunk)
        for row in matrix
        for value in row
    ):
        raise ValueError("paired CRN covariance contains a non-finite value")
    return means, raw, shrunk


def _heldout_diagnostic(
    decision_rewards: Sequence[tuple[str, Sequence[float], Sequence[float]]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for decision_uid, train_rewards, gate_rewards in decision_rewards:
        train_mean = math.fsum(train_rewards) / len(train_rewards)
        gate_mean = math.fsum(gate_rewards) / len(gate_rewards)
        rows.append(
            {
                "decision_uid": decision_uid,
                "train_mean": train_mean,
                "heldout_mean": gate_mean,
            }
        )
    ordered = sorted(rows, key=lambda row: (float(row["train_mean"]), row["decision_uid"]))
    bottom = ordered[0]
    top = ordered[-1]
    train_delta = float(top["train_mean"]) - float(bottom["train_mean"])
    heldout_delta = float(top["heldout_mean"]) - float(bottom["heldout_mean"])
    identifiable = train_delta > 0.0
    return {
        "version": NESTED_HELDOUT_VERSION,
        "train_indices": list(TRAIN_FOLD_INDICES),
        "heldout_indices": list(GATE_FOLD_INDICES),
        "top_decision_uid": top["decision_uid"],
        "bottom_decision_uid": bottom["decision_uid"],
        "train_top_mean": top["train_mean"],
        "train_bottom_mean": bottom["train_mean"],
        "train_delta": train_delta,
        "heldout_top_mean": top["heldout_mean"],
        "heldout_bottom_mean": bottom["heldout_mean"],
        "heldout_delta": heldout_delta,
        "train_ranking_identifiable": identifiable,
        "split_half_ranking_consistent": identifiable and heldout_delta > 0.0,
        "used_by_structural_gate": False,
        "used_by_signal_gate": True,
    }


def _effective_sample_size(weights: Sequence[float]) -> float:
    if not weights:
        return 0.0
    total = math.fsum(weights)
    squared_total = math.fsum(weight * weight for weight in weights)
    return total * total / squared_total if squared_total else 0.0


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated percentile from an already sorted sample."""
    if not sorted_values:
        raise ValueError("percentile sample cannot be empty")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower]) * (1.0 - fraction) + float(
        sorted_values[upper]
    ) * fraction


def _signal_gate(
    *,
    eligible_states: int,
    high_kappa_states: int,
    heldout_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    identifiable = [row for row in heldout_rows if bool(row["train_ranking_identifiable"])]
    task_clusters: dict[int, dict[str, object]] = {}
    for row in identifiable:
        task_id = int(row["task_id"])
        cluster = task_clusters.setdefault(
            task_id,
            {"deltas": [], "consistency": [], "proposal_weight": 0.0},
        )
        cluster["deltas"].append(float(row["heldout_delta"]))
        cluster["consistency"].append(
            1.0 if bool(row["split_half_ranking_consistent"]) else 0.0
        )
        cluster["proposal_weight"] += float(row["proposal_count"])
    task_rows = [
        {
            "task_id": task_id,
            "mean_delta": math.fsum(cluster["deltas"]) / len(cluster["deltas"]),
            "consistency": math.fsum(cluster["consistency"])
            / len(cluster["consistency"]),
            "proposal_weight": float(cluster["proposal_weight"]),
        }
        for task_id, cluster in sorted(task_clusters.items())
    ]
    deltas = [float(row["mean_delta"]) for row in task_rows]
    consistency = [float(row["consistency"]) for row in task_rows]
    identifiable_task_ess = _effective_sample_size(
        [float(row["proposal_weight"]) for row in task_rows]
    )
    high_kappa_rate = high_kappa_states / eligible_states if eligible_states else 0.0
    mean_delta = math.fsum(deltas) / len(deltas) if deltas else None
    consistency_rate = math.fsum(consistency) / len(consistency) if consistency else None
    failed_checks: list[str] = []

    not_run_bootstrap = {
        "version": NESTED_RESAMPLING_VERSION,
        "status": "not_run_insufficient_eligible_states",
        "seed": DIAGNOSTIC_SEED,
        "samples": BOOTSTRAP_SAMPLES,
        "resampling_unit": "train-identifiable task cluster",
        "mean_delta_ci95": None,
        "consistency_ci95": None,
    }
    not_run_permutation = {
        "version": NESTED_RESAMPLING_VERSION,
        "status": "not_run_insufficient_eligible_states",
        "seed": DIAGNOSTIC_SEED,
        "samples": PERMUTATION_SAMPLES,
        "null": "independent top/bottom label swap per task cluster",
        "observed_mean_delta": mean_delta,
        "null_95th_percentile": None,
        "one_sided_p_value": None,
    }
    if eligible_states < MIN_TRAINING_STATES:
        failed_checks.append("insufficient_eligible_states_for_signal_diagnostics")
        return {
            "version": NESTED_SIGNAL_GATE_VERSION,
            "signal_ready": False,
            "status": "not_ready",
            "failed_checks": failed_checks,
            "thresholds": {
                "minimum_high_kappa_state_rate": MIN_HIGH_KAPPA_STATE_RATE,
                "high_kappa_threshold": HIGH_KAPPA_THRESHOLD,
                "minimum_heldout_mean_delta": MIN_HELDOUT_MEAN_DELTA,
                "minimum_split_half_consistency": MIN_SPLIT_HALF_CONSISTENCY,
                "minimum_identifiable_task_effective_sample_size": (
                    MIN_IDENTIFIABLE_TASK_ESS
                ),
                "bootstrap_mean_delta_ci95_lower_strictly_above": 0.0,
                "bootstrap_consistency_ci95_lower_strictly_above": 0.5,
                "permutation_observed_strictly_above_null_percentile": 0.95,
            },
            "observed": {
                "eligible_states": eligible_states,
                "high_kappa_states": high_kappa_states,
                "high_kappa_state_rate": high_kappa_rate,
                "train_ranking_identifiable_states": len(identifiable),
                "train_ranking_identifiable_tasks": len(task_rows),
                "identifiable_task_effective_sample_size": identifiable_task_ess,
                "heldout_mean_delta": mean_delta,
                "split_half_consistency": consistency_rate,
            },
            "bootstrap": not_run_bootstrap,
            "permutation": not_run_permutation,
            "mixed_strict_used": False,
        }

    if high_kappa_rate < MIN_HIGH_KAPPA_STATE_RATE:
        failed_checks.append("insufficient_high_kappa_state_rate")
    if not task_rows:
        failed_checks.append("no_train_ranking_identifiable_tasks")
        not_run_bootstrap["status"] = "not_run_no_train_ranking_identifiable_tasks"
        not_run_permutation["status"] = "not_run_no_train_ranking_identifiable_tasks"
        return {
            "version": NESTED_SIGNAL_GATE_VERSION,
            "signal_ready": False,
            "status": "not_ready",
            "failed_checks": failed_checks,
            "thresholds": {
                "minimum_high_kappa_state_rate": MIN_HIGH_KAPPA_STATE_RATE,
                "high_kappa_threshold": HIGH_KAPPA_THRESHOLD,
                "minimum_heldout_mean_delta": MIN_HELDOUT_MEAN_DELTA,
                "minimum_split_half_consistency": MIN_SPLIT_HALF_CONSISTENCY,
                "minimum_identifiable_task_effective_sample_size": (
                    MIN_IDENTIFIABLE_TASK_ESS
                ),
                "bootstrap_mean_delta_ci95_lower_strictly_above": 0.0,
                "bootstrap_consistency_ci95_lower_strictly_above": 0.5,
                "permutation_observed_strictly_above_null_percentile": 0.95,
            },
            "observed": {
                "eligible_states": eligible_states,
                "high_kappa_states": high_kappa_states,
                "high_kappa_state_rate": high_kappa_rate,
                "train_ranking_identifiable_states": 0,
                "train_ranking_identifiable_tasks": 0,
                "identifiable_task_effective_sample_size": 0.0,
                "heldout_mean_delta": None,
                "split_half_consistency": None,
            },
            "bootstrap": not_run_bootstrap,
            "permutation": not_run_permutation,
            "mixed_strict_used": False,
        }

    if identifiable_task_ess < MIN_IDENTIFIABLE_TASK_ESS:
        failed_checks.append("insufficient_identifiable_task_effective_sample_size")

    if mean_delta is None or mean_delta < MIN_HELDOUT_MEAN_DELTA:
        failed_checks.append("heldout_mean_delta_below_threshold")
    if consistency_rate is None or consistency_rate < MIN_SPLIT_HALF_CONSISTENCY:
        failed_checks.append("split_half_consistency_below_threshold")

    bootstrap_rng = random.Random(DIAGNOSTIC_SEED)
    bootstrap_means: list[float] = []
    bootstrap_consistency: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled_indices = [bootstrap_rng.randrange(len(task_rows)) for _ in task_rows]
        bootstrap_means.append(
            math.fsum(deltas[index] for index in sampled_indices) / len(sampled_indices)
        )
        bootstrap_consistency.append(
            math.fsum(consistency[index] for index in sampled_indices) / len(sampled_indices)
        )
    bootstrap_means.sort()
    bootstrap_consistency.sort()
    mean_ci = [_percentile(bootstrap_means, 0.025), _percentile(bootstrap_means, 0.975)]
    consistency_ci = [
        _percentile(bootstrap_consistency, 0.025),
        _percentile(bootstrap_consistency, 0.975),
    ]
    if mean_ci[0] <= 0.0:
        failed_checks.append("bootstrap_mean_delta_lower_not_positive")
    if consistency_ci[0] <= 0.5:
        failed_checks.append("bootstrap_consistency_lower_not_above_half")

    permutation_rng = random.Random(DIAGNOSTIC_SEED)
    permutation_means: list[float] = []
    for _ in range(PERMUTATION_SAMPLES):
        permutation_means.append(
            math.fsum(
                delta if permutation_rng.getrandbits(1) else -delta for delta in deltas
            )
            / len(deltas)
        )
    permutation_means.sort()
    null_95th = _percentile(permutation_means, 0.95)
    observed_mean = float(mean_delta)
    if observed_mean <= null_95th:
        failed_checks.append("permutation_mean_delta_not_above_95th_percentile")
    exceedances = sum(value >= observed_mean for value in permutation_means)

    return {
        "version": NESTED_SIGNAL_GATE_VERSION,
        "signal_ready": not failed_checks,
        "status": "ready" if not failed_checks else "not_ready",
        "failed_checks": failed_checks,
        "thresholds": {
            "minimum_high_kappa_state_rate": MIN_HIGH_KAPPA_STATE_RATE,
            "high_kappa_threshold": HIGH_KAPPA_THRESHOLD,
            "minimum_heldout_mean_delta": MIN_HELDOUT_MEAN_DELTA,
            "minimum_split_half_consistency": MIN_SPLIT_HALF_CONSISTENCY,
            "minimum_identifiable_task_effective_sample_size": MIN_IDENTIFIABLE_TASK_ESS,
            "bootstrap_mean_delta_ci95_lower_strictly_above": 0.0,
            "bootstrap_consistency_ci95_lower_strictly_above": 0.5,
            "permutation_observed_strictly_above_null_percentile": 0.95,
        },
        "observed": {
            "eligible_states": eligible_states,
            "high_kappa_states": high_kappa_states,
            "high_kappa_state_rate": high_kappa_rate,
            "train_ranking_identifiable_states": len(identifiable),
            "train_ranking_identifiable_tasks": len(task_rows),
            "identifiable_task_effective_sample_size": identifiable_task_ess,
            "heldout_mean_delta": observed_mean,
            "split_half_consistency": consistency_rate,
        },
        "bootstrap": {
            "version": NESTED_RESAMPLING_VERSION,
            "status": "computed",
            "seed": DIAGNOSTIC_SEED,
            "samples": BOOTSTRAP_SAMPLES,
            "resampling_unit": "train-identifiable task cluster",
            "mean_delta_ci95": mean_ci,
            "consistency_ci95": consistency_ci,
        },
        "permutation": {
            "version": NESTED_RESAMPLING_VERSION,
            "status": "computed",
            "seed": DIAGNOSTIC_SEED,
            "samples": PERMUTATION_SAMPLES,
            "null": "independent top/bottom label swap per task cluster",
            "observed_mean_delta": observed_mean,
            "null_95th_percentile": null_95th,
            "one_sided_p_value": (exceedances + 1) / (PERMUTATION_SAMPLES + 1),
        },
        "mixed_strict_used": False,
    }


def _estimate_nested_decision_credit(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Estimate nested decision values and outcome-blind training eligibility.

    For decision ``d`` with ``L`` paired continuation rewards, ``q_d`` is their
    mean.  A fixed diagonal shrinkage is applied to the PSD covariance of the
    CRN-aligned decision means before de-noising between-decision variance.
    Gate-fold rewards never enter these values or their advantages.
    """
    samples = validate_nested_samples(payload)
    continuation_count = int(samples["collected_continuations_per_decision"])
    estimated_states: list[dict[str, object]] = []
    eligible_proposal_counts: list[float] = []
    eligible_task_weights: dict[int, float] = {}
    high_kappa_states = 0
    infrastructure_invalid = 0
    reward_invalid = 0
    sampling_invalid = 0
    continuation_slots = 0
    heldout_rows: list[dict[str, object]] = []

    for state in samples["states"]:
        proposals = state["proposals"]
        decisions = state["decisions"]
        proposal_count = len(proposals)
        multiplicities = {
            decision["decision_uid"]: sum(
                proposal["decision_uid"] == decision["decision_uid"]
                for proposal in proposals
            )
            for decision in decisions
        }
        valid_train_counts: dict[str, int] = {}
        valid_gate_counts: dict[str, int] = {}
        state_infrastructure_invalid = 0
        state_reward_invalid = 0
        state_sampling_invalid = 0
        for decision in decisions:
            continuations = decision["continuations"]
            valid_train_counts[decision["decision_uid"]] = sum(
                bool(continuation["valid_for_learning"])
                for continuation in continuations
                if continuation["continuation_index"] in TRAIN_FOLD_INDICES
            )
            valid_gate_counts[decision["decision_uid"]] = sum(
                bool(continuation["valid_for_learning"])
                for continuation in continuations
                if continuation["continuation_index"] in GATE_FOLD_INDICES
            )
            state_infrastructure_invalid += sum(
                bool(continuation["infrastructure_invalid"])
                for continuation in continuations
            )
            state_reward_invalid += sum(
                bool(continuation["reward_invalid"])
                for continuation in continuations
            )
            state_sampling_invalid += sum(
                bool(continuation["sampling_invalid"])
                for continuation in continuations
            )
        state_slots = len(decisions) * continuation_count
        continuation_slots += state_slots
        infrastructure_invalid += state_infrastructure_invalid
        reward_invalid += state_reward_invalid
        sampling_invalid += state_sampling_invalid
        state_infrastructure_rate = state_infrastructure_invalid / state_slots
        state_sampling_invalid_rate = state_sampling_invalid / state_slots

        failed_checks: list[str] = []
        if proposal_count < MIN_PROPOSALS_PER_STATE:
            failed_checks.append("insufficient_proposals")
        if len(decisions) < MIN_DISTINCT_DECISIONS:
            failed_checks.append("insufficient_distinct_decisions")
        if continuation_count != FORMAL_CONTINUATIONS_PER_DECISION:
            failed_checks.append("formal_train_gate_cardinality_incomplete")
        if any(
            valid_count != len(TRAIN_FOLD_INDICES)
            for valid_count in valid_train_counts.values()
        ):
            failed_checks.append("incomplete_valid_train_fold")
        if any(
            valid_count != len(GATE_FOLD_INDICES)
            for valid_count in valid_gate_counts.values()
        ):
            failed_checks.append("incomplete_valid_gate_fold")
        if state_sampling_invalid_rate > MAX_SAMPLING_INVALID_RATE:
            failed_checks.append("state_sampling_invalid_rate_exceeded")
        eligible = not failed_checks

        decision_values: list[dict[str, object]] = []
        heldout = None
        mu_s = None
        weighted_observed_variance = None
        weighted_uncertainty = None
        between_variance = None
        paired_crn_covariance = None
        v_s = None
        if eligible:
            eligible_proposal_counts.append(float(proposal_count))
            task_id = int(state["task_id"])
            eligible_task_weights[task_id] = (
                eligible_task_weights.get(task_id, 0.0) + proposal_count
            )
            value_rows: list[dict[str, object]] = []
            decision_reward_rows: list[
                tuple[str, Sequence[float], Sequence[float]]
            ] = []
            train_reward_rows: list[list[float]] = []
            for decision in decisions:
                decision_uid = decision["decision_uid"]
                train_rewards = [
                    float(continuation["reward"])
                    for continuation in decision["continuations"]
                    if continuation["continuation_index"] in TRAIN_FOLD_INDICES
                ]
                gate_rewards = [
                    float(continuation["reward"])
                    for continuation in decision["continuations"]
                    if continuation["continuation_index"] in GATE_FOLD_INDICES
                ]
                train_reward_rows.append(train_rewards)
                q_d, u_d = _sample_mean_and_variance_of_mean(train_rewards)
                c_d = multiplicities[decision_uid]
                value_rows.append(
                    {
                        "decision_index": decision["decision_index"],
                        "decision_uid": decision_uid,
                        "proposal_multiplicity": c_d,
                        "valid_train_continuations": len(train_rewards),
                        "valid_gate_continuations": len(gate_rewards),
                        "q_d": q_d,
                        "u_d": u_d,
                        "loss_weight": c_d,
                    }
                )
                decision_reward_rows.append((decision_uid, train_rewards, gate_rewards))

            crn_means, raw_covariance, shrunk_covariance = (
                _paired_crn_covariance_of_means(train_reward_rows)
            )
            if any(
                abs(float(row["q_d"]) - crn_mean) > 1e-12
                for row, crn_mean in zip(value_rows, crn_means, strict=True)
            ):
                raise AssertionError("paired CRN means differ from decision values")
            for index, row in enumerate(value_rows):
                if abs(float(row["u_d"]) - shrunk_covariance[index][index]) > 1e-12:
                    raise AssertionError("paired CRN covariance diagonal mismatch")

            mu_s = math.fsum(
                row["proposal_multiplicity"] * row["q_d"] for row in value_rows
            ) / proposal_count
            weighted_observed_variance = math.fsum(
                row["proposal_multiplicity"] * (row["q_d"] - mu_s) ** 2
                for row in value_rows
            ) / proposal_count
            proposal_weights = [
                float(row["proposal_multiplicity"]) / proposal_count
                for row in value_rows
            ]
            covariance_times_weight = [
                math.fsum(
                    shrunk_covariance[left][right] * proposal_weights[right]
                    for right in range(len(value_rows))
                )
                for left in range(len(value_rows))
            ]
            weighted_mean_noise = math.fsum(
                proposal_weights[index] * covariance_times_weight[index]
                for index in range(len(value_rows))
            )
            weighted_uncertainty = math.fsum(
                proposal_weights[index] * shrunk_covariance[index][index]
                for index in range(len(value_rows))
            ) - weighted_mean_noise
            if weighted_uncertainty < -1e-12:
                raise AssertionError("paired CRN weighted uncertainty is not PSD")
            weighted_uncertainty = max(weighted_uncertainty, 0.0)
            paired_crn_covariance = {
                "slot_indices": list(TRAIN_FOLD_INDICES),
                "slot_alignment_verified": True,
                "full_decision_slot_cartesian_product_verified": True,
                "seed_schedule_version": NESTED_SEED_SCHEDULE_VERSION,
                "decision_uids": [row["decision_uid"] for row in value_rows],
                "raw_covariance_of_means": raw_covariance,
                "diagonal_shrinkage": CRN_COVARIANCE_DIAGONAL_SHRINKAGE,
                "shrunk_covariance_of_means": shrunk_covariance,
                "psd_by_convex_construction": True,
                "weighted_mean_noise_variance": weighted_mean_noise,
            }
            between_variance = max(weighted_observed_variance - weighted_uncertainty, 0.0)
            for index, row in enumerate(value_rows):
                contrast_noise_variance = (
                    shrunk_covariance[index][index]
                    + weighted_mean_noise
                    - 2.0 * covariance_times_weight[index]
                )
                if contrast_noise_variance < -1e-12:
                    raise AssertionError("paired CRN contrast variance is not PSD")
                contrast_noise_variance = max(contrast_noise_variance, 0.0)
                kappa_d = between_variance / (
                    between_variance
                    + contrast_noise_variance
                    + SHRINKAGE_EPSILON
                )
                row["contrast_noise_variance"] = contrast_noise_variance
                row["kappa_d"] = kappa_d
                row["q_tilde_d"] = mu_s + kappa_d * (row["q_d"] - mu_s)
            if max(float(row["kappa_d"]) for row in value_rows) > HIGH_KAPPA_THRESHOLD:
                high_kappa_states += 1
            v_s = math.fsum(
                row["proposal_multiplicity"] * row["q_tilde_d"] for row in value_rows
            ) / proposal_count
            raw_advantages = [float(row["q_tilde_d"] - v_s) for row in value_rows]
            max_absolute_advantage = max(abs(value) for value in raw_advantages)
            advantage_scale = (
                min(1.0, ADVANTAGE_CLIP / max_absolute_advantage)
                if max_absolute_advantage
                else 1.0
            )
            for row, raw_advantage in zip(value_rows, raw_advantages, strict=True):
                row["advantage"] = raw_advantage * advantage_scale
            weighted_advantage_sum = math.fsum(
                row["proposal_multiplicity"] * row["advantage"]
                for row in value_rows
            )
            if abs(weighted_advantage_sum) > 1e-10:
                raise AssertionError("proposal-weighted nested advantages are not centered")
            decision_values = value_rows
            heldout = _heldout_diagnostic(decision_reward_rows)
            heldout["task_id"] = task_id
            heldout["proposal_count"] = proposal_count
            heldout_rows.append(heldout)

        estimated_states.append(
            {
                "state_index": state["state_index"],
                "state_uid": state["state_uid"],
                "task_id": state["task_id"],
                "proposal_count": proposal_count,
                "distinct_decision_count": len(decisions),
                "continuation_slots": state_slots,
                "valid_train_continuations": sum(valid_train_counts.values()),
                "valid_gate_continuations": sum(valid_gate_counts.values()),
                "infrastructure_invalid_continuations": state_infrastructure_invalid,
                "infrastructure_invalid_rate": state_infrastructure_rate,
                "reward_invalid_continuations": state_reward_invalid,
                "sampling_invalid_continuations": state_sampling_invalid,
                "sampling_invalid_rate": state_sampling_invalid_rate,
                "structural_gate": {
                    "eligible_for_credit": eligible,
                    "failed_checks": failed_checks,
                    "reward_dependent_checks": [],
                },
                "mu_s": mu_s,
                "weighted_observed_variance": weighted_observed_variance,
                "weighted_uncertainty": weighted_uncertainty,
                "between_variance": between_variance,
                "paired_crn_covariance": paired_crn_covariance,
                "v_s": v_s,
                "advantage_scale": advantage_scale if eligible else None,
                "decision_values": decision_values,
                "heldout_diagnostic": heldout,
            }
        )

    eligible_states = len(eligible_proposal_counts)
    state_ess = _effective_sample_size(eligible_proposal_counts)
    eligible_unique_tasks = len(eligible_task_weights)
    task_ess = _effective_sample_size(list(eligible_task_weights.values()))
    global_infrastructure_rate = (
        infrastructure_invalid / continuation_slots if continuation_slots else 0.0
    )
    global_sampling_invalid_rate = (
        sampling_invalid / continuation_slots if continuation_slots else 0.0
    )
    training_failures: list[str] = []
    if eligible_states < MIN_TRAINING_STATES:
        training_failures.append("insufficient_eligible_states")
    if state_ess < MIN_STATE_ESS:
        training_failures.append("insufficient_state_effective_sample_size")
    if eligible_unique_tasks < MIN_TRAINING_TASKS:
        training_failures.append("insufficient_unique_tasks")
    if task_ess < MIN_TASK_ESS:
        training_failures.append("insufficient_task_cluster_effective_sample_size")
    if global_sampling_invalid_rate > MAX_SAMPLING_INVALID_RATE:
        training_failures.append("global_sampling_invalid_rate_exceeded")

    identifiable = [
        row for row in heldout_rows if bool(row["train_ranking_identifiable"])
    ]
    consistent = sum(bool(row["split_half_ranking_consistent"]) for row in identifiable)
    heldout_mean_delta = (
        math.fsum(float(row["heldout_delta"]) for row in identifiable) / len(identifiable)
        if identifiable
        else None
    )
    signal_gate = _signal_gate(
        eligible_states=eligible_states,
        high_kappa_states=high_kappa_states,
        heldout_rows=heldout_rows,
    )
    structural_ready = not training_failures
    signal_ready = bool(signal_gate["signal_ready"])
    statistical_candidate_ready = structural_ready and signal_ready
    input_sha = _sha256_json(samples)
    return {
        "schema_version": NESTED_CREDIT_VERSION,
        "input_schema_version": NESTED_SAMPLES_VERSION,
        "input_sha256": input_sha,
        "experiment_uid": samples["experiment_uid"],
        "reward_contract_sha256": samples["reward_contract_sha256"],
        "source_attestation": samples["source_attestation"],
        "fold_contract": {
            "train_indices": list(TRAIN_FOLD_INDICES),
            "gate_indices": list(GATE_FOLD_INDICES),
            "gate_only_not_refit": True,
            "collected_continuations_per_decision": continuation_count,
        },
        "estimator_contract": {
            "version": NESTED_ESTIMATOR_VERSION,
            "q_d": "mean(valid continuation rewards)",
            "u_d": "unbiased sample variance / L",
            "mu_s_weight": "proposal multiplicity",
            "q_and_u_source": "train fold only (indices 0..3)",
            "gate_reward_role": "signal gate only; never refit into q or advantage",
            "crn_slot_contract": "complete decision x train-slot Cartesian product",
            "crn_covariance": "sample covariance of paired decision means",
            "covariance_shrinkage": (
                "fixed 0.5 convex shrink toward diagonal; PSD by construction"
            ),
            "between_variance": (
                "max(weighted_var(q_d)-trace(diag(p)Sigma)+p^T Sigma p,0)"
            ),
            "kappa_d": "between/(between+Var(q_d-weighted_mean)+epsilon)",
            "q_tilde_d": "mu_s+kappa_d*(q_d-mu_s)",
            "v_s_weight": "proposal multiplicity",
            "advantage": "common_scale(q_tilde_d-v_s,max_abs=0.5)",
            "loss_weight": "proposal multiplicity c_d",
            "proposal_weighted_advantage_sum": 0.0,
            "standard_deviation_normalization": False,
            "epsilon": SHRINKAGE_EPSILON,
            "advantage_clip": [-ADVANTAGE_CLIP, ADVANTAGE_CLIP],
        },
        "states": estimated_states,
        "heldout_diagnostics": {
            "version": NESTED_HELDOUT_VERSION,
            "role": "reward_dependent_signal_gate_evidence",
            "eligible_states": eligible_states,
            "train_ranking_identifiable_states": len(identifiable),
            "split_half_consistent_states": consistent,
            "split_half_consistency_rate": consistent / len(identifiable) if identifiable else None,
            "mean_heldout_delta": heldout_mean_delta,
            "used_by_structural_gate": False,
            "used_by_signal_gate": True,
        },
        "signal_gate": signal_gate,
        "training_gate": {
            "version": NESTED_TRAINING_GATE_VERSION,
            "statistical_ready": statistical_candidate_ready,
            "statistical_candidate_ready": statistical_candidate_ready,
            "attested_collection": False,
            "training_ready": False,
            "optimizer_unlock_allowed": False,
            "structural_ready": structural_ready,
            "signal_ready": signal_ready,
            "failed_checks": training_failures
            + [f"signal:{reason}" for reason in signal_gate["failed_checks"]]
            + ["collection_artifacts_not_attested"],
            "thresholds": {
                "minimum_proposals_per_state": MIN_PROPOSALS_PER_STATE,
                "minimum_distinct_decisions": MIN_DISTINCT_DECISIONS,
                "minimum_valid_continuations_per_decision": MIN_CONTINUATIONS_PER_DECISION,
                "required_train_fold_indices": list(TRAIN_FOLD_INDICES),
                "required_gate_fold_indices": list(GATE_FOLD_INDICES),
                "required_formal_continuations_per_decision": (
                    FORMAL_CONTINUATIONS_PER_DECISION
                ),
                "maximum_sampling_invalid_rate": MAX_SAMPLING_INVALID_RATE,
                "minimum_eligible_states": MIN_TRAINING_STATES,
                "minimum_state_effective_sample_size": MIN_STATE_ESS,
                "minimum_unique_tasks": MIN_TRAINING_TASKS,
                "minimum_task_cluster_effective_sample_size": MIN_TASK_ESS,
            },
            "observed": {
                "input_states": len(estimated_states),
                "eligible_states": eligible_states,
                "state_proposal_weight_effective_sample_size": state_ess,
                "eligible_unique_tasks": eligible_unique_tasks,
                "task_cluster_proposal_weight_effective_sample_size": task_ess,
                "continuation_slots": continuation_slots,
                "infrastructure_invalid_continuations": infrastructure_invalid,
                "infrastructure_invalid_rate": global_infrastructure_rate,
                "reward_invalid_continuations": reward_invalid,
                "sampling_invalid_continuations": sampling_invalid,
                "sampling_invalid_rate": global_sampling_invalid_rate,
            },
            "outcome_blind_structural_failed_checks": training_failures,
            "reward_dependent_signal_failed_checks": signal_gate["failed_checks"],
            "mixed_strict_used": False,
        },
    }


def estimate_nested_decision_credit(payload: Mapping[str, object]) -> dict[str, object]:
    """Compute diagnostics from hand-built samples without unlocking an optimizer."""
    return _estimate_nested_decision_credit(payload)


def _bounded_forbidden_key_paths(value: object, prefix: str = "$") -> list[str]:
    forbidden = {
        "goal",
        "gold",
        "gold_asin",
        "hidden_goal",
        "reward_goal",
        "target",
        "target_asin",
    }
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}"
            if str(key).lower() in forbidden:
                found.append(path)
            found.extend(_bounded_forbidden_key_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_bounded_forbidden_key_paths(item, f"{prefix}[{index}]"))
    return found


def _artifact_rollout_content_sha256(record: Mapping[str, object]) -> str:
    payload = dict(record)
    payload.pop("rollout_content_sha256", None)
    return _sha256_json(
        {"version": NESTED_ROLLOUT_CONTENT_VERSION, "record": payload}
    )


def _read_bound_collection_artifacts(
    artifact_files: Mapping[str, object],
    collection: Mapping[str, object],
) -> dict[str, str]:
    """Read each source artifact once and bind its exact bytes to parsed input."""
    if not isinstance(artifact_files, Mapping):
        raise TypeError("artifact_files must be an object")
    _require_exact_fields(
        artifact_files,
        {"decisions", "continuations", "summary"},
        "artifact_files",
    )
    parsed: dict[str, object] = {}
    hashes: dict[str, str] = {}
    resolved_paths: set[Path] = set()
    for name in ("decisions", "continuations", "summary"):
        raw_path = artifact_files.get(name)
        if not isinstance(raw_path, (str, Path)):
            raise TypeError(f"{name} artifact path must be path-like")
        requested_path = Path(raw_path).expanduser().absolute()
        if requested_path.is_symlink():
            raise ValueError("nested artifact paths must not be symbolic links")
        path = requested_path.resolve()
        if not path.is_file() or path in resolved_paths:
            raise ValueError("nested artifact paths must be distinct existing files")
        resolved_paths.add(path)
        raw_bytes = path.read_bytes()
        hashes[name] = hashlib.sha256(raw_bytes).hexdigest()
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{name} artifact is not UTF-8") from exc
        if name == "summary":
            value = json.loads(text)
            if not isinstance(value, Mapping):
                raise TypeError("nested summary artifact must contain an object")
            parsed[name] = dict(value)
            continue
        rows = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} artifact line {line_number} must be an object")
            rows.append(dict(value))
        parsed[name] = rows
    if any(parsed[name] != collection[name] for name in parsed):
        raise ValueError("nested in-memory collection differs from bound artifact bytes")
    return hashes


def _semantic_rollout_content_sha256(record: Mapping[str, object]) -> str:
    """Hash sampled content without slot identity so copied rollouts are visible."""
    fields = (
        "response_ids",
        "response_mask",
        "old_logprobs",
        "assistant_spans",
        "downstream_prompt_sha256",
        "strict",
        "policy_reward",
        "terminal_utility",
        "model_failure",
        "termination_reason",
    )
    return _sha256_json({name: record.get(name) for name in fields})


def _nonnegative_record_count(record: Mapping[str, object], name: str) -> int:
    value = record.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"nested continuation {name} must be a non-negative integer")
    return value


def _recompute_continuation_policy_reward(
    record: Mapping[str, object],
    policy_config: Mapping[str, object],
) -> float | None:
    """Rebuild policy-v1 reward from the bound public rollout diagnostics."""
    valid = record.get("valid_for_learning")
    infrastructure_invalid = record.get("infrastructure_invalid")
    reward_invalid = record.get("reward_invalid")
    reward_unverifiable = record.get("reward_unverifiable")
    sampling_invalid = record.get("sampling_invalid")
    strict = record.get("strict")
    model_failure = record.get("model_failure")
    if not all(
        isinstance(value, bool)
        for value in (
            valid,
            infrastructure_invalid,
            reward_invalid,
            reward_unverifiable,
            sampling_invalid,
            strict,
            model_failure,
        )
    ):
        raise TypeError("nested continuation reward flags must be boolean")
    if sampling_invalid is not (not valid):
        raise ValueError("nested continuation sampling-invalid flag mismatch")
    if reward_invalid and (infrastructure_invalid or model_failure):
        raise ValueError("nested continuation reward-invalid category mismatch")
    if model_failure and (not valid or sampling_invalid):
        raise ValueError("model-attributable failures must remain learning-valid")
    policy_reward = _finite_reward(record.get("policy_reward"), "policy_reward")
    terminal_utility = _finite_reward(
        record.get("terminal_utility"), "terminal_utility"
    )
    termination_reason = record.get("termination_reason")
    if not isinstance(termination_reason, str) or not termination_reason:
        raise ValueError("nested continuation termination_reason must be non-empty")

    if not valid:
        error_code = record.get("infrastructure_error_code")
        error_class = record.get("infrastructure_error_class")
        invalid_reason = record.get("invalid_reason")
        if not isinstance(invalid_reason, str) or not invalid_reason:
            raise ValueError("sampling-invalid continuation lacks a bounded reason")
        if infrastructure_invalid:
            if (
                reward_invalid
                or reward_unverifiable
                or record.get("generation_mode") != "infrastructure_invalid"
                or strict
                or model_failure
                or policy_reward != 0.0
                or terminal_utility != 0.0
                or termination_reason != "nested_continuation_infrastructure_invalid"
                or not isinstance(error_code, str)
                or not error_code
                or len(error_code) > 128
                or not isinstance(error_class, str)
                or not error_class
                or len(error_class) > 128
                or invalid_reason != "nested_continuation_infrastructure_error"
            ):
                raise ValueError(
                    "infrastructure-invalid continuation reward contract mismatch"
                )
            breakdown = reward_breakdown(
                {
                    "infrastructure_invalid": True,
                    "termination_reason": termination_reason,
                    "error": invalid_reason,
                },
                policy_config,
            )
            if (
                breakdown["valid_for_learning"] is not False
                or breakdown["infrastructure_invalid"] is not True
                or float(breakdown["total"]) != policy_reward
            ):
                raise ValueError("infrastructure-invalid reward recomputation mismatch")
            return None

        if error_code is not None or error_class is not None:
            raise ValueError("non-infrastructure invalid continuation carries infra errors")
        if record.get("generation_mode") == "infrastructure_invalid" or strict:
            raise ValueError("non-infrastructure invalid continuation contract mismatch")
        if reward_invalid:
            if model_failure or policy_reward != 0.0:
                raise ValueError("reward-invalid continuation reward contract mismatch")
            if reward_unverifiable:
                if (
                    invalid_reason != "reward_unverifiable"
                    or record.get("reward_type") != "reward_unverifiable"
                    or termination_reason != "reward_unverifiable"
                ):
                    raise ValueError("reward-unverifiable continuation identity mismatch")
                state = {
                    "done": True,
                    "terminal_result": {"done": True, "over": True},
                    "infrastructure_invalid": False,
                    "final_reward": terminal_utility,
                    "reward_version": "shopsimulator-reward-v3",
                    "reward_valid": False,
                    "reward_type": "reward_unverifiable",
                    "termination_reason": termination_reason,
                }
            else:
                if record.get("reward_type") is not None:
                    raise ValueError("nonterminal reward-invalid continuation has reward type")
                state = {
                    "done": False,
                    "terminal_result": None,
                    "infrastructure_invalid": False,
                    "termination_reason": termination_reason,
                }
            breakdown = reward_breakdown(state, policy_config)
            if (
                breakdown["valid_for_learning"] is not False
                or breakdown["infrastructure_invalid"] is not False
                or breakdown["reward_unverifiable"] is not reward_unverifiable
                or breakdown["model_failure"] is not False
                or breakdown["invalid_reason"] != invalid_reason
                or float(breakdown["total"]) != policy_reward
            ):
                raise ValueError("reward-invalid reward recomputation mismatch")
            return None

        raise ValueError("sampling-invalid continuation category mismatch")

    if (
        infrastructure_invalid
        or reward_invalid
        or reward_unverifiable
        or sampling_invalid
        or record.get("invalid_reason") is not None
    ):
        raise ValueError("learning-valid continuation carries invalid diagnostics")
    if (
        record.get("infrastructure_error_code") is not None
        or record.get("infrastructure_error_class") is not None
    ):
        raise ValueError("learning-valid continuation carries infrastructure errors")
    guard_rejections = _nonnegative_record_count(record, "guard_rejections")
    repeat_actions = _nonnegative_record_count(record, "repeat_actions")
    steps = _nonnegative_record_count(record, "steps")
    reward_type = record.get("reward_type")
    if model_failure:
        if reward_type is not None or terminal_utility != 0.0:
            raise ValueError("model-failure continuation has terminal Reward-v3 diagnostics")
        normal_terminal = False
    else:
        if (
            not isinstance(reward_type, str)
            or reward_type not in REWARD_V3_TYPES
            or termination_reason != reward_type
        ):
            raise ValueError("normal continuation Reward-v3 identity mismatch")
        normal_terminal = True

    state = {
        "done": normal_terminal,
        "terminal_result": (
            {"done": True, "over": True} if normal_terminal else None
        ),
        "infrastructure_invalid": False,
        "final_reward": terminal_utility,
        "reward_version": (
            "shopsimulator-reward-v3" if normal_terminal else None
        ),
        "reward_valid": True,
        "reward_type": reward_type,
        "reward_detail": {},
        "termination_reason": termination_reason,
        "guard_rejection_count": guard_rejections,
        "repeat_action_count": repeat_actions,
        "action_attempt_count": max(steps, 1),
    }
    breakdown = reward_breakdown(state, policy_config)
    if (
        breakdown["valid_for_learning"] is not True
        or breakdown["infrastructure_invalid"] is not False
        or bool(breakdown["model_failure"]) is not model_failure
        or bool(breakdown["strict"]) is not strict
        or float(breakdown["terminal_utility"]) != terminal_utility
        or float(breakdown["total"]) != policy_reward
    ):
        raise ValueError("nested continuation policy reward recomputation mismatch")
    return policy_reward


def _validate_source_proposal(
    source: Mapping[str, object],
    *,
    state_uid: str,
    active_group_uid: str,
    decision: Mapping[str, object],
    group: Mapping[str, object],
    stage1_by_source: Mapping[tuple[str, int], Mapping[str, object]],
) -> dict[str, object]:
    expected_fields = {
        "proposal_index",
        "proposal_uid",
        "stage1_proposal_uid",
        "source_uid",
        "stage1_suffix_uid",
        "stage1_suffix_index",
        "stage1_record_sha256",
        "stage1_first_request_seed",
        "stage1_first_request_prompt_sha256",
        "first_assistant_old_logprobs",
        "first_assistant_old_logprobs_sha256",
    }
    _require_exact_fields(source, expected_fields, "nested source proposal")
    proposal_index = source.get("proposal_index")
    suffix_index = source.get("stage1_suffix_index")
    if (
        not isinstance(proposal_index, int)
        or isinstance(proposal_index, bool)
        or proposal_index < 0
        or suffix_index != proposal_index
    ):
        raise ValueError("nested source proposal indices are invalid")
    expected_proposal_uid = proposal_uid(state_uid, proposal_index)
    if source.get("proposal_uid") != expected_proposal_uid:
        raise ValueError("nested source proposal_uid mismatch")
    suffixes = group.get("suffixes")
    if not isinstance(suffixes, list) or proposal_index >= len(suffixes):
        raise ValueError("nested source proposal is outside the active plan")
    suffix = suffixes[proposal_index]
    if source.get("stage1_suffix_uid") != suffix.get("suffix_uid"):
        raise ValueError("nested source proposal suffix_uid mismatch")
    expected_stage1_proposal_uid = proposal_uid(active_group_uid, proposal_index)
    if source.get("stage1_proposal_uid") != expected_stage1_proposal_uid:
        raise ValueError("nested stage-one proposal_uid mismatch")
    stage1 = stage1_by_source.get((active_group_uid, proposal_index))
    if stage1 is None:
        raise ValueError("nested source proposal has no stage-one record")
    stage1_record_sha = _sha256_json(dict(stage1))
    if source.get("stage1_record_sha256") != stage1_record_sha:
        raise ValueError("nested source proposal stage-one record SHA256 mismatch")
    expected_source_uid = _sha256_json(
        {
            "proposal_uid": expected_proposal_uid,
            "stage1_suffix_uid": suffix["suffix_uid"],
            "stage1_record_sha256": stage1_record_sha,
        }
    )
    if source.get("source_uid") != expected_source_uid:
        raise ValueError("nested source_uid mismatch")
    prompt_hash = group["actor_prompt_tokens"]["sha256"]
    if (
        source.get("stage1_first_request_seed") != suffix["seed"]
        or source.get("stage1_first_request_prompt_sha256") != prompt_hash
    ):
        raise ValueError("nested source first request provenance mismatch")
    first_logprobs = source.get("first_assistant_old_logprobs")
    if not isinstance(first_logprobs, list) or first_logprobs != decision.get(
        "first_assistant_old_logprobs"
    ):
        raise ValueError("nested source first-decision old logprobs mismatch")
    if source.get("first_assistant_old_logprobs_sha256") != _sha256_json(
        first_logprobs
    ):
        raise ValueError("nested source old-logprob hash mismatch")
    if (
        stage1.get("active_group_uid") != active_group_uid
        or stage1.get("suffix_index") != proposal_index
        or stage1.get("suffix_uid") != suffix["suffix_uid"]
        or stage1.get("prompt_token_sha256") != prompt_hash
        or stage1.get("first_action_sha256") != decision.get("first_action_sha256")
    ):
        raise ValueError("stage-one record does not reproduce the nested decision source")
    span = stage1.get("first_action_span")
    first_tokens = decision.get("first_assistant_token_ids")
    if (
        not isinstance(span, list)
        or span != [0, len(first_tokens)]
        or stage1.get("response_ids", [])[: len(first_tokens)] != first_tokens
        or stage1.get("old_logprobs", [])[: len(first_tokens)] != first_logprobs
    ):
        raise ValueError("stage-one decision span, tokens, or old logprobs changed")
    return {
        "proposal_index": proposal_index,
        "proposal_uid": expected_proposal_uid,
        "decision_uid": decision["decision_uid"],
    }


def estimate_nested_collection_credit(
    collection: Mapping[str, object],
    *,
    actor_checkpoint: str | Path,
    environment_manifest: Mapping[str, object],
    environment_manifest_path: str | Path,
    environment_manifest_sha256: str,
    active_branch_plan: Mapping[str, object],
    resolved_selections: Sequence[Mapping[str, object]],
    stage1_records: Sequence[Mapping[str, object]],
    stage1_source_sha256: str,
    stage1_records_path: str | Path,
    stage1_manifest: str | Path,
    sampling_backend_contract: Mapping[str, object],
    artifact_files: Mapping[str, object],
    collection_manifest: str | Path,
    policy_reward: object = None,
) -> dict[str, object]:
    """Attest collector artifacts, then estimate credit without unlocking early.

    Every source object is independently supplied by the caller.  The adapter
    recomputes actor, environment, plan, resolved-selection, stage-one, backend,
    decision, boundary, seed, lease, tensor, and aggregate identities before it
    marks the normalized samples as source-attested.
    """
    if not isinstance(collection, Mapping):
        raise TypeError("nested collection must be an object")
    _require_exact_fields(
        collection, {"decisions", "continuations", "summary"}, "nested collection"
    )
    hidden_paths = _bounded_forbidden_key_paths(collection)
    if hidden_paths:
        raise ValueError(f"nested collection contains forbidden fields: {hidden_paths[:3]}")
    decisions = _require_sequence(collection.get("decisions"), "nested decisions")
    continuations = _require_sequence(
        collection.get("continuations"), "nested continuations"
    )
    summary = collection.get("summary")
    if not isinstance(summary, Mapping):
        raise TypeError("nested collection summary must be an object")
    completed_snapshot = verify_completed_collection_manifest(
        collection_manifest,
        artifact_files=artifact_files,
        include_file_attestation=True,
    )
    completed_manifest = completed_snapshot["manifest"]
    completed_manifest_file_sha256 = str(
        completed_snapshot["manifest_file_sha256"]
    )
    normalized_file_hashes = _read_bound_collection_artifacts(
        artifact_files, collection
    )
    manifest_artifacts = completed_manifest.get("artifacts")
    if (
        not isinstance(manifest_artifacts, Mapping)
        or any(
            normalized_file_hashes[name]
            != (manifest_artifacts.get(name) or {}).get("sha256")
            for name in normalized_file_hashes
        )
    ):
        raise ValueError(
            "nested artifact snapshot changed after manifest verification"
        )
    if summary.get("schema_version") != NESTED_COLLECTION_VERSION:
        raise ValueError("nested collection summary version mismatch")
    exclusion_audit = summary.get("exclusion_audit")
    if not isinstance(exclusion_audit, Mapping):
        raise TypeError("nested collection exclusion audit must be an object")
    _require_exact_fields(
        exclusion_audit,
        {
            "schema_version",
            "pre_registered_states",
            "pre_registered_proposals",
            "eligible_states",
            "eligible_proposals",
            "excluded_states",
            "excluded_proposals",
            "reason_counts",
            "excluded_state_records",
            "proposal_partition_sha256",
            "accounting_verified",
            "no_backfill",
        },
        "nested exclusion audit",
    )
    if (
        exclusion_audit.get("schema_version") != NESTED_EXCLUSION_AUDIT_VERSION
        or exclusion_audit.get("accounting_verified") is not True
        or exclusion_audit.get("no_backfill") is not True
    ):
        raise ValueError("nested exclusion audit safety contract mismatch")

    normalized_plan = validate_active_branch_plan(active_branch_plan)
    plan_sha = _sha256_json(normalized_plan)
    if summary.get("plan_sha256") != plan_sha:
        raise ValueError("nested collection plan SHA256 mismatch")
    actor_sha = sha256_actor_checkpoint(actor_checkpoint)
    if (
        actor_sha != normalized_plan["actor_checkpoint_sha256"]
        or actor_sha != (summary.get("provenance") or {}).get("actor_checkpoint_sha256")
    ):
        raise ValueError("nested collection actor tree SHA256 mismatch")
    rebuilt_plan = build_active_branch_plan(
        list(resolved_selections),
        actor_checkpoint_sha256=actor_sha,
        decoding_config=normalized_plan["decoding_config"],
        seed=normalized_plan["seed"],
        suffixes_per_state=normalized_plan["suffixes_per_state"],
    )
    if rebuilt_plan != normalized_plan:
        raise ValueError("active branch plan does not match the resolved selection")
    normalized_backend = validate_sampling_backend_contract(sampling_backend_contract)
    backend_sha = sampling_backend_contract_sha256(normalized_backend)
    provenance = summary.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError("nested collection provenance must be an object")
    if provenance.get("sampling_backend_contract_sha256") != backend_sha:
        raise ValueError("nested collection sampling backend SHA256 mismatch")
    if normalized_backend.get("actor_checkpoint_sha256") != actor_sha:
        raise ValueError("sampling backend actor identity differs from disk")
    resolved_sha = _sha256_json(list(resolved_selections))
    if summary.get("resolved_selections_sha256") != resolved_sha:
        raise ValueError("nested collection resolved-selection SHA256 mismatch")
    stage1_source_sha = _required_sha256(stage1_source_sha256, "stage1_source_sha256")
    if summary.get("stage1_source_sha256") != stage1_source_sha:
        raise ValueError("nested collection stage-one source SHA256 mismatch")
    stage1_records_file = Path(stage1_records_path).expanduser().resolve()
    stage1_manifest_file = Path(stage1_manifest).expanduser().resolve()
    verified_stage1 = verify_first_decision_artifacts(
        plan=normalized_plan,
        records_path=stage1_records_file,
        manifest_path=stage1_manifest_file,
    )
    if _canonical_json(list(stage1_records)) != _canonical_json(
        verified_stage1["records"]
    ):
        raise ValueError("Stage-1 records differ from their completion manifest")
    if (
        hashlib.sha256(stage1_records_file.read_bytes()).hexdigest()
        != stage1_source_sha
    ):
        raise ValueError("Stage-1 source bytes differ from the supplied digest")
    stage1_binding = build_nested_stage1_source_binding(
        verified_stage1["manifest"],
        final_manifest_sha256=str(verified_stage1["manifest_file_sha256"]),
    )
    summary_stage1_binding = validate_nested_stage1_source_binding(
        provenance.get("stage1_source_binding"),
        expected_records_sha256=stage1_source_sha,
    )
    if stage1_binding != summary_stage1_binding:
        raise ValueError("nested collection Stage-1 completion binding mismatch")

    manifest_file = Path(environment_manifest_path).expanduser().resolve()
    try:
        manifest_from_file = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("environment manifest file is invalid") from exc
    if (
        not isinstance(manifest_from_file, Mapping)
        or _canonical_json(dict(environment_manifest))
        != _canonical_json(dict(manifest_from_file))
    ):
        raise ValueError("environment manifest object differs from its source file")
    manifest = validate_manifest(deepcopy(dict(manifest_from_file)))
    manifest_sha = _required_sha256(
        environment_manifest_sha256, "environment_manifest_sha256"
    )
    if sha256_file(manifest_file) != manifest_sha:
        raise ValueError("environment manifest file SHA256 mismatch")
    environment_version = str(
        manifest.get("environment_version", "shopsimulator-environment-v2.1")
    )
    if any(
        group["environment_manifest_sha256"] != manifest_sha
        for group in normalized_plan["groups"]
    ) or any(
        (resolved.get("trajectory") or {}).get("environment_version")
        != environment_version
        or (resolved.get("trajectory") or {}).get(
            "environment_manifest_sha256"
        )
        != manifest_sha
        for resolved in resolved_selections
    ):
        raise ValueError("resolved environment identity differs from the bound manifest")
    if (
        provenance.get("required_environment_version") != environment_version
        or provenance.get("environment_manifest_sha256s") != [manifest_sha]
    ):
        raise ValueError("nested collection environment manifest binding mismatch")
    normalized_policy_reward = validate_policy_reward_config(policy_reward)
    policy_reward_sha = _sha256_json(normalized_policy_reward)
    if provenance.get("policy_reward_sha256") != policy_reward_sha:
        raise ValueError("nested collection policy reward SHA256 mismatch")
    decoding_sha = _sha256_json(normalized_plan["decoding_config"])
    if (
        normalized_plan.get("decoding_config_sha256") != decoding_sha
        or provenance.get("decoding_config_sha256") != decoding_sha
    ):
        raise ValueError("nested collection decoding config SHA256 mismatch")
    harness = summary.get("harness_contract")
    if (
        not isinstance(harness, Mapping)
        or harness.get("schema_version") != NESTED_HARNESS_CONTRACT_VERSION
        or harness.get("policy_reward_sha256") != policy_reward_sha
    ):
        raise ValueError("nested collection harness contract mismatch")
    harness_sha = _sha256_json(dict(harness))
    if summary.get("harness_contract_sha256") != harness_sha:
        raise ValueError("nested collection harness SHA256 mismatch")
    if (
        stage1_binding["active_plan_sha256"] != plan_sha
        or stage1_binding["policy_reward_sha256"] != policy_reward_sha
        or stage1_binding["harness_contract_sha256"] != harness_sha
        or stage1_binding["required_environment_version"]
        != environment_version
        or stage1_binding["environment_manifest_sha256s"] != [manifest_sha]
    ):
        raise ValueError("finalized Stage-1 contract differs from nested runtime")
    formal_plan = build_nested_formal_plan(
        normalized_plan,
        list(resolved_selections),
        required_environment_version=environment_version,
        policy_reward_sha256=policy_reward_sha,
        harness_contract_sha256=harness_sha,
        sampling_backend_contract_sha256=backend_sha,
    )
    formal_plan_sha = _sha256_json(formal_plan)
    if (
        formal_plan.get("schema_version") != NESTED_FORMAL_PLAN_VERSION
        or summary.get("formal_plan") != formal_plan
        or summary.get("formal_plan_sha256") != formal_plan_sha
    ):
        raise ValueError("nested formal plan v2 identity mismatch")
    expected_experiment_uid = _sha256_json(
        {
            "formal_plan_sha256": formal_plan_sha,
            "stage1_source_sha256": stage1_source_sha,
            "stage1_manifest_sha256": stage1_binding[
                "final_manifest_sha256"
            ],
            "harness_contract_sha256": harness_sha,
            "continuations_per_decision": (summary.get("aggregate") or {}).get(
                "continuations_per_decision"
            ),
            "fold_contract_version": NESTED_FOLD_CONTRACT_VERSION,
        }
    )
    if summary.get("experiment_uid") != expected_experiment_uid:
        raise ValueError("nested formal experiment UID mismatch")

    fold = summary.get("fold_contract")
    if not isinstance(fold, Mapping) or fold != {
        "version": NESTED_FOLD_CONTRACT_VERSION,
        "formal_continuations_per_decision": FORMAL_CONTINUATIONS_PER_DECISION,
        "train_indices": list(TRAIN_FOLD_INDICES),
        "gate_indices": list(GATE_FOLD_INDICES),
        "gate_only_not_refit": True,
    }:
        raise ValueError("nested collection fold contract mismatch")
    aggregate = summary.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise TypeError("nested collection aggregate must be an object")
    collected_count = aggregate.get("continuations_per_decision")
    if collected_count not in {len(TRAIN_FOLD_INDICES), FORMAL_CONTINUATIONS_PER_DECISION}:
        raise ValueError("nested collection is neither a mechanical nor formal fold sample")
    safety = summary.get("safety")
    if (
        not isinstance(safety, Mapping)
        or safety.get("optimizer_enabled") is not False
        or safety.get("training_ready") is not False
        or safety.get("uses_hidden_goal") is not False
        or safety.get("all_eight_refit_allowed") is not False
        or safety.get("outcome_conditioned_resampling") is not False
        or safety.get("fresh_lease_per_continuation") is not True
        or safety.get("common_random_numbers_by_state_slot") is not True
        or safety.get("streaming_journal_and_resume") is not True
        or safety.get("manifest_commit_required") is not True
        or safety.get("actor_start_end_full_hash") is not True
        or safety.get("actor_runtime_stat_binding") is not True
        or safety.get("actor_prehashed_backend_binding") is not True
        or safety.get("actor_stat_checked_before_every_completion") is not True
        or safety.get("finalized_stage1_source") is not True
        or safety.get("scale_collection_ready") is not True
        or safety.get("formal_scale_blockers") != []
    ):
        raise ValueError("nested collector safety lock is invalid")

    stage1_by_source: dict[tuple[str, int], Mapping[str, object]] = {}
    for record in stage1_records:
        if not isinstance(record, Mapping):
            raise TypeError("stage-one record must be an object")
        state_uid = _required_sha256(record.get("active_group_uid"), "stage1 state_uid")
        suffix_index = record.get("suffix_index")
        if not isinstance(suffix_index, int) or isinstance(suffix_index, bool):
            raise TypeError("stage-one suffix_index must be an integer")
        key = (state_uid, suffix_index)
        if key in stage1_by_source:
            raise ValueError("stage-one source repeats one proposal slot")
        stage1_by_source[key] = record

    active_groups = {
        group["active_group_uid"]: group for group in normalized_plan["groups"]
    }
    groups = {
        formal_group["formal_group_uid"]: {
            "formal": formal_group,
            "active": active_groups[formal_group["source_active_group_uid"]],
        }
        for formal_group in formal_plan["groups"]
    }
    recomputed_stage1_structure = classify_nested_stage1_structure(
        normalized_plan, stage1_records
    )
    if recomputed_stage1_structure != verified_stage1["structure"]:
        raise ValueError("Stage-1 structural partition differs from its final manifest")
    if dict(exclusion_audit) != recomputed_stage1_structure["exclusion_audit"]:
        raise ValueError(
            "nested exclusion audit differs from outcome-blind stage-one recomputation"
        )
    expected_stage1_slots = {
        (state_uid, proposal_index)
        for state_uid, group in active_groups.items()
        for proposal_index in range(len(group["suffixes"]))
    }
    if set(stage1_by_source) != expected_stage1_slots:
        raise ValueError("stage-one records do not cover the pre-registered proposal bank")
    continuations_by_decision: dict[str, list[Mapping[str, object]]] = {}
    for record in continuations:
        if not isinstance(record, Mapping):
            raise TypeError("nested continuation must be an object")
        decision_uid = _required_sha256(record.get("decision_uid"), "decision_uid")
        continuations_by_decision.setdefault(decision_uid, []).append(record)

    normalized_states: dict[str, dict[str, object]] = {}
    seen_decision_uids: set[str] = set()
    seen_continuation_uids: set[str] = set()
    seen_lease_sequences: set[int] = set()
    eligible_stage1_proposal_uids: set[str] = set()
    eligible_active_group_uids: set[str] = set()
    computed_valid_decisions = 0
    computed_credit_eligible_decisions = 0
    duplicate_semantic_rollout_count = 0
    for raw_decision in decisions:
        if not isinstance(raw_decision, Mapping):
            raise TypeError("nested decision must be an object")
        if raw_decision.get("schema_version") != NESTED_DECISION_VERSION:
            raise ValueError("nested decision version mismatch")
        if (
            raw_decision.get("optimizer_enabled") is not False
            or raw_decision.get("training_ready") is not False
            or raw_decision.get("uses_hidden_goal") is not False
        ):
            raise ValueError("nested decision safety lock is invalid")
        state_uid = _required_sha256(raw_decision.get("state_uid"), "state_uid")
        group_bundle = groups.get(state_uid)
        if group_bundle is None:
            raise ValueError("nested decision state is not in formal plan v2")
        formal_group = group_bundle["formal"]
        group = group_bundle["active"]
        active_group_uid = _required_sha256(
            raw_decision.get("active_group_uid"), "active_group_uid"
        )
        if active_group_uid != formal_group["source_active_group_uid"]:
            raise ValueError("nested decision active/formal group binding differs")
        task_id = raw_decision.get("task_id")
        if (
            not isinstance(task_id, int)
            or isinstance(task_id, bool)
            or task_id != group.get("task_id")
        ):
            raise ValueError("nested decision task_id mismatch")
        prompt = raw_decision.get("pre_action_prompt_token_ids")
        first_tokens = raw_decision.get("first_assistant_token_ids")
        first_logprobs = raw_decision.get("first_assistant_old_logprobs")
        loss_mask = raw_decision.get("first_assistant_loss_mask")
        if (
            not isinstance(prompt, list)
            or not prompt
            or any(not isinstance(item, int) or isinstance(item, bool) for item in prompt)
            or prompt != group["actor_prompt_tokens"]["tokens"]
            or raw_decision.get("pre_action_prompt_sha256") != token_ids_sha256(prompt)
            or not isinstance(first_tokens, list)
            or not first_tokens
            or any(
                not isinstance(item, int) or isinstance(item, bool) for item in first_tokens
            )
            or raw_decision.get("first_assistant_token_sha256")
            != token_ids_sha256(first_tokens)
            or not isinstance(first_logprobs, list)
            or len(first_logprobs) != len(first_tokens)
            or any(
                not isinstance(item, (int, float))
                or isinstance(item, bool)
                or not math.isfinite(float(item))
                for item in first_logprobs
            )
            or loss_mask != [1] * len(first_tokens)
        ):
            raise ValueError("nested decision prompt, span, tokens, or old logprobs are invalid")
        action = raw_decision.get("first_action")
        if not isinstance(action, Mapping):
            raise TypeError("nested first action must be an object")
        canonical_action = canonical_replay_action(
            action.get("tool"), action.get("parameters")
        )
        action_sha = replay_action_sha256(
            canonical_action["tool"], canonical_action["parameters"]
        )
        if canonical_action != action or raw_decision.get("first_action_sha256") != action_sha:
            raise ValueError("nested first action identity mismatch")
        expected_decision_content_sha = _sha256_json(
            {
                "version": NESTED_DECISION_CONTENT_VERSION,
                "state_uid": state_uid,
                "decision_uid": raw_decision.get("decision_uid"),
                "pre_action_prompt_token_ids": prompt,
                "first_assistant_token_ids": first_tokens,
                "first_assistant_old_logprobs": first_logprobs,
                "first_assistant_loss_mask": loss_mask,
                "first_action": canonical_action,
                "first_action_sha256": action_sha,
            }
        )
        if raw_decision.get("decision_content_sha256") != expected_decision_content_sha:
            raise ValueError("nested decision content SHA256 mismatch")
        decision_identity = {
            "version": NESTED_DECISION_IDENTITY_VERSION,
            "state_uid": state_uid,
            "pre_action_prompt_sha256": token_ids_sha256(prompt),
            "first_assistant_token_ids": first_tokens,
            "first_action_sha256": action_sha,
        }
        decision_uid = _sha256_json(decision_identity)
        if (
            raw_decision.get("decision_uid") != decision_uid
            or decision_uid in seen_decision_uids
        ):
            raise ValueError("nested decision_uid mismatch or duplicate")
        seen_decision_uids.add(decision_uid)
        identity = raw_decision.get("identity")
        expected_identity = {
            "decision_identity": decision_identity,
            "formal_plan_version": NESTED_FORMAL_PLAN_VERSION,
            "formal_plan_sha256": formal_plan_sha,
            "active_group_uid": active_group_uid,
            "parent_branch_uid": group["parent_branch_uid"],
            "replay_state_id": group["replay_state_id"],
            "task_id": task_id,
            "environment_version": environment_version,
            "environment_manifest_sha256": group["environment_manifest_sha256"],
            "policy_reward_sha256": policy_reward_sha,
            "harness_contract_sha256": harness_sha,
            "actor_checkpoint_sha256": actor_sha,
            "decoding_config_sha256": decoding_sha,
            "sampling_backend_contract_sha256": backend_sha,
            "source_hashes": {
                "active_branch_plan_sha256": plan_sha,
                "resolved_selections_sha256": resolved_sha,
                "stage1_source_sha256": stage1_source_sha,
                "stage1_manifest_sha256": stage1_binding[
                    "final_manifest_sha256"
                ],
            },
        }
        if identity != expected_identity or group["environment_manifest_sha256"] != manifest_sha:
            raise ValueError("nested decision source identity mismatch")

        raw_sources = _require_sequence(
            raw_decision.get("source_proposals"), "nested source proposals"
        )
        normalized_proposals = [
            _validate_source_proposal(
                source,
                state_uid=state_uid,
                active_group_uid=active_group_uid,
                decision=raw_decision,
                group=group,
                stage1_by_source=stage1_by_source,
            )
            for source in raw_sources
            if isinstance(source, Mapping)
        ]
        if len(normalized_proposals) != len(raw_sources) or not normalized_proposals:
            raise TypeError("nested source proposals must be non-empty objects")
        eligible_stage1_proposal_uids.update(
            str(source["stage1_proposal_uid"]) for source in raw_sources
        )
        eligible_active_group_uids.add(active_group_uid)
        normalized_proposals.sort(key=lambda item: item["proposal_index"])
        proposal_uids = [item["proposal_uid"] for item in normalized_proposals]
        if (
            raw_decision.get("proposal_multiplicity") != len(normalized_proposals)
            or raw_decision.get("proposal_uids") != proposal_uids
            or len(proposal_uids) != len(set(proposal_uids))
            or raw_decision.get("replay_source_proposal_uid") not in proposal_uids
        ):
            raise ValueError("nested decision proposal multiplicity mismatch")

        boundary = raw_decision.get("post_action_boundary")
        boundary_prompt = None
        boundary_kind = None
        if boundary is None:
            if (
                raw_decision.get("structurally_valid") is not False
                or raw_decision.get("credit_eligible") is not False
            ):
                raise ValueError(
                    "nested decision without a boundary must be structurally invalid"
                )
        elif isinstance(boundary, Mapping):
            boundary_prompt = boundary.get("post_action_prompt_token_ids")
            snapshot = boundary.get("harness_snapshot")
            if (
                not isinstance(boundary_prompt, list)
                or not boundary_prompt
                or boundary.get("post_action_prompt_sha256")
                != token_ids_sha256(boundary_prompt)
                or boundary.get("post_action_prompt_token_count") != len(boundary_prompt)
                or not isinstance(snapshot, Mapping)
                or boundary.get("harness_snapshot_sha256")
                != _sha256_json(dict(snapshot))
                or snapshot.get("first_action_sha256") != action_sha
                or snapshot.get("first_action_span") != [0, len(first_tokens)]
                or snapshot.get("task_id") != task_id
                or snapshot.get("post_action_prompt_sha256")
                != boundary["post_action_prompt_sha256"]
                or snapshot.get("post_action_prompt_token_count") != len(boundary_prompt)
            ):
                raise ValueError("nested post-action prompt or harness snapshot mismatch")
            boundary_kind = snapshot.get("boundary_kind")
            boundary_done = snapshot.get("done")
            boundary_terminate = snapshot.get("terminate")
            termination_reason = snapshot.get("termination_reason")
            if boundary_kind == "continuation_required":
                if (
                    boundary_done is not False
                    or boundary_terminate is not False
                    or termination_reason is not None
                ):
                    raise ValueError(
                        "continuation-required boundary has terminal evidence"
                    )
            elif boundary_kind == "deterministic_terminal":
                if (
                    not isinstance(boundary_done, bool)
                    or not isinstance(boundary_terminate, bool)
                    or not (boundary_done or boundary_terminate)
                    or (boundary_done and not boundary_terminate)
                    or not isinstance(termination_reason, str)
                    or not termination_reason
                ):
                    raise ValueError(
                        "deterministic terminal boundary evidence is invalid"
                    )
            else:
                raise ValueError("nested boundary kind is invalid")
        else:
            raise TypeError("nested post-action boundary must be an object or null")

        raw_decision_continuations = continuations_by_decision.pop(decision_uid, [])
        raw_decision_continuations.sort(key=lambda item: item.get("continuation_index", -1))
        if len(raw_decision_continuations) != collected_count:
            raise ValueError("nested decision continuation cardinality mismatch")
        normalized_continuations = []
        semantic_rollout_hash_counts: dict[str, int] = {}
        decision_infrastructure_invalid = 0
        decision_sampling_invalid = 0
        for continuation_index, record in enumerate(raw_decision_continuations):
            if record.get("schema_version") != NESTED_CONTINUATION_VERSION:
                raise ValueError("nested continuation version mismatch")
            if record.get("continuation_index") != continuation_index:
                raise ValueError("nested continuation indices are not contiguous")
            expected_seed_uid = continuation_seed_uid(state_uid, continuation_index)
            expected_uid = continuation_uid(state_uid, decision_uid, continuation_index)
            expected_fold = (
                "train" if continuation_index in TRAIN_FOLD_INDICES else "gate"
            )
            if (
                record.get("continuation_uid") != expected_uid
                or expected_uid in seen_continuation_uids
                or record.get("continuation_seed_uid") != expected_seed_uid
                or record.get("seed_schedule_version") != NESTED_SEED_SCHEDULE_VERSION
                or record.get("fold_contract_version") != NESTED_FOLD_CONTRACT_VERSION
                or record.get("fold") != expected_fold
                or record.get("generation_mode")
                not in {
                    "sampled_continuation",
                    "deterministic_terminal",
                    "no_generation_terminal",
                    "infrastructure_invalid",
                }
                or record.get("state_uid") != state_uid
                or record.get("decision_uid") != decision_uid
                or record.get("task_id") != task_id
                or record.get("source_proposal_uids") != proposal_uids
                or record.get("proposal_multiplicity") != len(proposal_uids)
                or record.get("optimizer_enabled") is not False
                or record.get("uses_hidden_goal") is not False
            ):
                raise ValueError("nested continuation identity mismatch")
            seen_continuation_uids.add(expected_uid)
            if record.get("rollout_content_sha256") != _artifact_rollout_content_sha256(
                record
            ):
                raise ValueError("nested rollout content SHA256 mismatch")
            seeds = record.get("downstream_request_seeds")
            if not isinstance(seeds, list) or any(
                seed != continuation_turn_seed(state_uid, continuation_index, turn_index)
                for turn_index, seed in enumerate(seeds)
            ):
                raise ValueError("nested continuation turn seed schedule mismatch")
            expected_first_seed = seeds[0] if seeds else None
            if record.get("first_downstream_seed") != expected_first_seed:
                raise ValueError("nested continuation first downstream seed mismatch")
            downstream_prompts = record.get("downstream_prompt_sha256")
            valid = record.get("valid_for_learning")
            infrastructure_invalid = record.get("infrastructure_invalid")
            reward_invalid = record.get("reward_invalid")
            reward_unverifiable = record.get("reward_unverifiable")
            sampling_invalid = record.get("sampling_invalid")
            model_failure = record.get("model_failure")
            generation_mode = record.get("generation_mode")
            if not all(
                isinstance(value, bool)
                for value in (
                    valid,
                    infrastructure_invalid,
                    reward_invalid,
                    reward_unverifiable,
                    sampling_invalid,
                    model_failure,
                )
            ):
                raise TypeError("nested continuation validity fields must be boolean")
            if not infrastructure_invalid and (
                not isinstance(downstream_prompts, list)
                or len(downstream_prompts) != len(seeds)
                or any(
                    not isinstance(prompt_sha, str)
                    or len(prompt_sha) != 64
                    or any(character not in "0123456789abcdef" for character in prompt_sha)
                    for prompt_sha in downstream_prompts
                )
            ):
                raise ValueError("nested continuation downstream prompt schedule mismatch")
            reward = _recompute_continuation_policy_reward(
                record, normalized_policy_reward
            )
            lease_sequence = record.get("lease_sequence")
            if lease_sequence is not None:
                if (
                    not isinstance(lease_sequence, int)
                    or isinstance(lease_sequence, bool)
                    or lease_sequence < 1
                    or lease_sequence in seen_lease_sequences
                ):
                    raise ValueError("nested continuation lease is not independent")
                seen_lease_sequences.add(lease_sequence)
            if not infrastructure_invalid:
                if (
                    not isinstance(boundary, Mapping)
                    or generation_mode == "infrastructure_invalid"
                    or (
                        generation_mode == "sampled_continuation"
                        and (not seeds or boundary_kind != "continuation_required")
                    )
                    or (
                        generation_mode == "deterministic_terminal"
                        and (seeds or boundary_kind != "deterministic_terminal")
                    )
                    or (
                        generation_mode == "no_generation_terminal"
                        and (
                            seeds
                            or boundary_kind != "continuation_required"
                            or record.get("model_failure") is not True
                            or not isinstance(record.get("termination_reason"), str)
                            or not record.get("termination_reason")
                        )
                    )
                    or record.get("fresh_environment_lease") is not True
                    or record.get("release_verified") is not True
                    or lease_sequence is None
                    or record.get("post_action_prompt_sha256")
                    != boundary["post_action_prompt_sha256"]
                    or record.get("harness_snapshot_sha256")
                    != boundary["harness_snapshot_sha256"]
                ):
                    raise ValueError("non-infrastructure continuation lacks replay/release evidence")
                response_ids = record.get("response_ids")
                response_mask = record.get("response_mask")
                old_logprobs = record.get("old_logprobs")
                if (
                    not isinstance(response_ids, list)
                    or not response_ids
                    or response_ids[: len(first_tokens)] != first_tokens
                    or not isinstance(response_mask, list)
                    or len(response_mask) != len(response_ids)
                    or not isinstance(old_logprobs, list)
                    or len(old_logprobs) != len(response_ids)
                    or old_logprobs[: len(first_logprobs)] != first_logprobs
                ):
                    raise ValueError("nested continuation tensor alignment mismatch")
                observed_boundary = record.get("observed_boundary_summary")
                expected_observed_boundary = {
                    "post_action_prompt_sha256": boundary[
                        "post_action_prompt_sha256"
                    ],
                    "post_action_prompt_token_count": len(boundary_prompt),
                    "harness_snapshot_sha256": boundary["harness_snapshot_sha256"],
                    "boundary_kind": boundary_kind,
                    "first_action_sha256": action_sha,
                }
                replay = record.get("replay")
                if observed_boundary != expected_observed_boundary:
                    raise ValueError("nested continuation boundary observation mismatch")
                if (
                    not isinstance(replay, Mapping)
                    or set(replay)
                    != {
                        "verified",
                        "verified_transitions",
                        "raw_observation_sha256",
                        "projected_observation_sha256",
                    }
                    or replay.get("verified") is not True
                    or replay.get("verified_transitions") != group["prefix_action_count"]
                    or _required_sha256(
                        replay.get("raw_observation_sha256"),
                        "nested replay raw observation SHA256",
                    )
                    != replay.get("raw_observation_sha256")
                    or _required_sha256(
                        replay.get("projected_observation_sha256"),
                        "nested replay projected observation SHA256",
                    )
                    != replay.get("projected_observation_sha256")
                ):
                    raise ValueError("nested continuation replay evidence mismatch")
                semantic_hash = _semantic_rollout_content_sha256(record)
                semantic_rollout_hash_counts[semantic_hash] = (
                    semantic_rollout_hash_counts.get(semantic_hash, 0) + 1
                )
                if semantic_rollout_hash_counts[semantic_hash] > 1:
                    duplicate_semantic_rollout_count += 1
                invalid_reason = None if valid else str(record.get("invalid_reason") or "")
                if sampling_invalid:
                    decision_sampling_invalid += 1
            else:
                if generation_mode != "infrastructure_invalid":
                    raise ValueError("infrastructure-invalid continuation mode mismatch")
                invalid_reason = str(record.get("invalid_reason") or "")
                if not invalid_reason:
                    raise ValueError("infrastructure-invalid continuation lacks a reason")
                if (
                    record.get("fresh_environment_lease") is not (lease_sequence is not None)
                    or record.get("release_verified") is not False
                ):
                    raise ValueError(
                        "infrastructure-invalid continuation lease/release contract mismatch"
                    )
                semantic_hash = record["rollout_content_sha256"]
                decision_infrastructure_invalid += 1
                decision_sampling_invalid += 1
            normalized_continuations.append(
                {
                    "continuation_index": continuation_index,
                    "continuation_uid": expected_uid,
                    "continuation_seed_uid": expected_seed_uid,
                    "downstream_request_seeds": list(seeds),
                    "rollout_content_sha256": record["rollout_content_sha256"],
                    "generation_mode": generation_mode,
                    "reward": reward,
                    "valid_for_learning": valid,
                    "infrastructure_invalid": infrastructure_invalid,
                    "reward_invalid": reward_invalid,
                    "reward_unverifiable": reward_unverifiable,
                    "sampling_invalid": sampling_invalid,
                    "model_failure": model_failure,
                    "invalid_reason": invalid_reason,
                }
            )
        continuation_uids = [item["continuation_uid"] for item in normalized_continuations]
        computed_structurally_valid = (
            decision_infrastructure_invalid == 0
            and len(normalized_continuations) == collected_count
        )
        computed_credit_eligible = (
            decision_sampling_invalid == 0
            and len(normalized_continuations) == collected_count
        )
        if (
            raw_decision.get("continuation_uids") != continuation_uids
            or raw_decision.get("continuations_expected") != collected_count
            or raw_decision.get("structurally_valid") is not computed_structurally_valid
            or raw_decision.get("credit_eligible") is not computed_credit_eligible
        ):
            raise ValueError("nested decision continuation summary mismatch")
        computed_valid_decisions += int(computed_structurally_valid)
        computed_credit_eligible_decisions += int(computed_credit_eligible)

        state = normalized_states.setdefault(
            state_uid,
            {
                "task_id": task_id,
                "group_index": raw_decision.get("group_index"),
                "proposals": [],
                "decisions": [],
            },
        )
        if state["task_id"] != task_id:
            raise ValueError("one nested state contains multiple task ids")
        state["proposals"].extend(normalized_proposals)
        state["decisions"].append(
            {
                "decision_index": raw_decision.get("decision_index"),
                "decision_uid": decision_uid,
                "continuations": normalized_continuations,
            }
        )

    if continuations_by_decision:
        raise ValueError("nested collection contains orphan continuations")

    normalized_state_rows = []
    all_proposal_uids: list[str] = []
    for state_uid, state in sorted(
        normalized_states.items(), key=lambda item: (item[1]["group_index"], item[0])
    ):
        proposals = sorted(state["proposals"], key=lambda item: item["proposal_index"])
        decisions_for_state = sorted(
            state["decisions"], key=lambda item: item["decision_index"]
        )
        if [item["proposal_index"] for item in proposals] != list(range(len(proposals))):
            raise ValueError("nested state proposal indices are not a complete partition")
        if [item["decision_index"] for item in decisions_for_state] != list(
            range(len(decisions_for_state))
        ):
            raise ValueError("nested state decision indices are not contiguous")
        all_proposal_uids.extend(item["proposal_uid"] for item in proposals)
        normalized_state_rows.append(
            {
                "state_index": len(normalized_state_rows),
                "state_uid": state_uid,
                "task_id": state["task_id"],
                "proposals": proposals,
                "decisions": decisions_for_state,
            }
        )
    if len(all_proposal_uids) != len(set(all_proposal_uids)):
        raise ValueError("nested collection repeats proposal identities")

    pre_registered_by_state = {
        active_group_uid: [
            proposal_uid(active_group_uid, index)
            for index in range(len(group["suffixes"]))
        ]
        for active_group_uid, group in active_groups.items()
    }
    pre_registered_uids = {
        uid for proposal_uids in pre_registered_by_state.values() for uid in proposal_uids
    }
    eligible_uids = eligible_stage1_proposal_uids
    eligible_state_uids = eligible_active_group_uids
    excluded_records = _require_sequence(
        exclusion_audit.get("excluded_state_records"),
        "nested excluded state records",
    )
    excluded_state_uids: set[str] = set()
    excluded_uids: set[str] = set()
    recomputed_reason_counts: dict[str, int] = {}
    for excluded_record in excluded_records:
        if not isinstance(excluded_record, Mapping):
            raise TypeError("nested excluded state record must be an object")
        _require_exact_fields(
            excluded_record,
            {
                "state_uid",
                "task_id",
                "proposal_uids",
                "source_record_sha256s",
                "reasons",
                "no_backfill",
            },
            "nested excluded state record",
        )
        state_uid = _required_sha256(
            excluded_record.get("state_uid"), "excluded state_uid"
        )
        group = active_groups.get(state_uid)
        if group is None or state_uid in excluded_state_uids:
            raise ValueError("nested exclusion audit has an unknown or repeated state")
        excluded_state_uids.add(state_uid)
        proposal_uids = excluded_record.get("proposal_uids")
        expected_proposal_uids = pre_registered_by_state[state_uid]
        expected_source_hashes = [
            _sha256_json(dict(stage1_by_source[(state_uid, index)]))
            for index in range(len(group["suffixes"]))
        ]
        if (
            excluded_record.get("task_id") != group["task_id"]
            or proposal_uids != expected_proposal_uids
            or excluded_record.get("source_record_sha256s") != expected_source_hashes
            or excluded_record.get("no_backfill") is not True
        ):
            raise ValueError("nested excluded state source accounting mismatch")
        excluded_uids.update(expected_proposal_uids)
        reasons = _require_sequence(
            excluded_record.get("reasons"), "nested state exclusion reasons"
        )
        if not reasons:
            raise ValueError("nested excluded state has no bounded reason")
        for reason in reasons:
            if not isinstance(reason, Mapping):
                raise TypeError("nested state exclusion reason must be an object")
            _require_exact_fields(
                reason,
                {"code", "affected_proposal_uids", "decision_uid"},
                "nested state exclusion reason",
            )
            code = reason.get("code")
            affected = reason.get("affected_proposal_uids")
            decision_uid_value = reason.get("decision_uid")
            if (
                not isinstance(code, str)
                or not code
                or len(code) > 128
                or not isinstance(affected, list)
                or not affected
                or affected != sorted(set(affected))
                or not set(affected).issubset(expected_proposal_uids)
                or (
                    decision_uid_value is not None
                    and _required_sha256(
                        decision_uid_value, "excluded decision_uid"
                    )
                    != decision_uid_value
                )
            ):
                raise ValueError("nested state exclusion reason is invalid")
            recomputed_reason_counts[code] = recomputed_reason_counts.get(code, 0) + 1

    expected_partition_sha = _sha256_json(
        {
            "pre_registered": sorted(pre_registered_uids),
            "eligible": sorted(eligible_uids),
            "excluded": sorted(excluded_uids),
        }
    )
    if (
        eligible_state_uids.intersection(excluded_state_uids)
            or eligible_state_uids.union(excluded_state_uids) != set(active_groups)
        or eligible_uids.intersection(excluded_uids)
        or eligible_uids.union(excluded_uids) != pre_registered_uids
            or exclusion_audit.get("pre_registered_states") != len(active_groups)
        or exclusion_audit.get("pre_registered_proposals") != len(pre_registered_uids)
        or exclusion_audit.get("eligible_states") != len(eligible_state_uids)
        or exclusion_audit.get("eligible_proposals") != len(eligible_uids)
        or exclusion_audit.get("excluded_states") != len(excluded_state_uids)
        or exclusion_audit.get("excluded_proposals") != len(excluded_uids)
        or exclusion_audit.get("reason_counts")
        != dict(sorted(recomputed_reason_counts.items()))
        or exclusion_audit.get("proposal_partition_sha256") != expected_partition_sha
    ):
        raise ValueError("nested exclusion audit partition or counts mismatch")

    expected_aggregate = {
        "proposals": len(all_proposal_uids),
        "pre_registered_proposals": len(pre_registered_uids),
        "eligible_proposals": len(eligible_uids),
        "excluded_proposals": len(excluded_uids),
        "pre_registered_states": len(groups),
        "eligible_states": len(eligible_state_uids),
        "excluded_states": len(excluded_state_uids),
        "decisions": len(decisions),
        "distinct_decisions": len(decisions),
        "continuations_per_decision": collected_count,
        "continuations": len(continuations),
        "valid_decisions": computed_valid_decisions,
        "credit_eligible_decisions": computed_credit_eligible_decisions,
        "valid_for_learning_continuations": sum(
            int(record.get("valid_for_learning") is True) for record in continuations
        ),
        "infrastructure_invalid_continuations": sum(
            int(record.get("infrastructure_invalid") is True) for record in continuations
        ),
        "reward_invalid_continuations": sum(
            int(record.get("reward_invalid") is True) for record in continuations
        ),
        "reward_unverifiable_continuations": sum(
            int(record.get("reward_unverifiable") is True) for record in continuations
        ),
        "sampling_invalid_continuations": sum(
            int(record.get("sampling_invalid") is True) for record in continuations
        ),
        "strict_success_continuations": sum(
            int(record.get("strict") is True) for record in continuations
        ),
        "model_failure_continuations": sum(
            int(record.get("model_failure") is True) for record in continuations
        ),
        "deterministic_terminal_continuations": sum(
            int(record.get("generation_mode") == "deterministic_terminal")
            for record in continuations
        ),
        "fresh_lease_continuations": sum(
            int(record.get("fresh_environment_lease") is True) for record in continuations
        ),
        "release_verified_continuations": sum(
            int(record.get("release_verified") is True) for record in continuations
        ),
    }
    for name, expected in expected_aggregate.items():
        if aggregate.get(name) != expected:
            raise ValueError(f"nested collection aggregate {name} mismatch")
    expected_sampling_invalid_rate = (
        expected_aggregate["sampling_invalid_continuations"] / len(continuations)
        if continuations
        else 0.0
    )
    expected_reward_invalid_reasons = dict(
        sorted(
            Counter(
                str(record.get("invalid_reason"))
                for record in continuations
                if record.get("reward_invalid") is True
            ).items()
        )
    )
    cardinality_complete = len(continuations) == len(decisions) * collected_count
    if (
        aggregate.get("cardinality_complete") is not cardinality_complete
        or cardinality_complete is not True
        or aggregate.get("post_action_parity_complete")
        is not (computed_valid_decisions == len(decisions))
        or aggregate.get("sampling_invalid_rate") != expected_sampling_invalid_rate
        or aggregate.get("reward_invalid_reason_counts")
        != expected_reward_invalid_reasons
        or aggregate.get("mechanical_collection_passed")
        is not (
            bool(decisions)
            and cardinality_complete
            and expected_sampling_invalid_rate <= MAX_SAMPLING_INVALID_RATE
        )
        or aggregate.get("formal_fold_collection_passed")
        is not (
            collected_count == FORMAL_CONTINUATIONS_PER_DECISION
            and bool(decisions)
            and cardinality_complete
            and expected_sampling_invalid_rate <= MAX_SAMPLING_INVALID_RATE
        )
    ):
        raise ValueError("nested collection aggregate gate mismatch")

    actor_run_attestation = completed_manifest["actor_run_attestation"]
    actor_runtime_binding = {
        "schema_version": ACTOR_RUN_ATTESTATION_VERSION,
        "actor_checkpoint_sha256": actor_sha,
        "stat_snapshot_sha256": actor_run_attestation["stat_snapshot_sha256"],
        "stat_entry_count": actor_run_attestation["stat_entry_count"],
        "read_only_run_binding": True,
    }
    expected_journal_contract = build_nested_journal_contract(
        experiment_uid=str(summary["experiment_uid"]),
        active_plan_sha256=plan_sha,
        formal_plan_sha256=formal_plan_sha,
        resolved_selections_sha256=resolved_sha,
        stage1_source_sha256=stage1_source_sha,
        stage1_manifest_sha256=stage1_binding["final_manifest_sha256"],
        stage1_source_finalized=True,
        harness_contract_sha256=harness_sha,
        actor_runtime_binding=actor_runtime_binding,
        continuations_per_decision=int(collected_count),
        expected_continuation_uids=[
            str(continuation_uid_value)
            for decision in decisions
            for continuation_uid_value in decision["continuation_uids"]
        ],
    )
    journal_completion = completed_manifest["journal_completion"]
    if journal_completion.get("header_sha256") != nested_journal_header_sha256(
        expected_journal_contract
    ):
        raise ValueError("nested journal header is not bound to the formal collection")

    source_contract = {
        "collection_manifest_sha256": completed_manifest_file_sha256,
        "artifact_set_uid": completed_manifest["artifact_set_uid"],
        "journal_header_sha256": journal_completion["header_sha256"],
        "artifact_file_sha256s": normalized_file_hashes,
        "canonical_artifact_sha256s": {
            "decisions": _sha256_json(list(decisions)),
            "continuations": _sha256_json(list(continuations)),
            "summary": _sha256_json(dict(summary)),
        },
        "actor_checkpoint_sha256": actor_sha,
        "environment_manifest_sha256": manifest_sha,
        "environment_version": environment_version,
        "active_branch_plan_sha256": plan_sha,
        "nested_formal_plan_version": NESTED_FORMAL_PLAN_VERSION,
        "nested_formal_plan_sha256": formal_plan_sha,
        "resolved_selections_sha256": resolved_sha,
        "stage1_source_sha256": stage1_source_sha,
        "stage1_manifest_sha256": stage1_binding["final_manifest_sha256"],
        "stage1_source_binding_sha256": _sha256_json(stage1_binding),
        "policy_reward_sha256": policy_reward_sha,
        "decoding_config_sha256": decoding_sha,
        "sampling_backend_contract_sha256": backend_sha,
        "harness_contract_sha256": harness_sha,
        "exclusion_audit_sha256": _sha256_json(dict(exclusion_audit)),
        "duplicate_semantic_rollout_count": duplicate_semantic_rollout_count,
    }
    normalized_payload = {
        "schema_version": NESTED_SAMPLES_VERSION,
        "experiment_uid": _required_sha256(summary.get("experiment_uid"), "experiment_uid"),
        "reward_contract_sha256": policy_reward_sha,
        "collected_continuations_per_decision": collected_count,
        "train_fold_indices": list(TRAIN_FOLD_INDICES),
        "gate_fold_indices": list(GATE_FOLD_INDICES),
        "source_attestation": {
            "schema_version": NESTED_SOURCE_ATTESTATION_VERSION,
            "attested": True,
            "contract_sha256": _sha256_json(source_contract),
        },
        "states": normalized_state_rows,
    }
    result = _estimate_nested_decision_credit(normalized_payload)
    training_gate = result["training_gate"]
    if not isinstance(training_gate, dict):
        raise TypeError("nested estimator returned an invalid training gate")
    candidate_ready = bool(training_gate["statistical_candidate_ready"])
    training_gate["attested_collection"] = True
    training_gate["training_ready"] = candidate_ready
    training_gate["optimizer_unlock_allowed"] = candidate_ready
    training_gate["failed_checks"] = [
        reason
        for reason in training_gate["failed_checks"]
        if reason != "collection_artifacts_not_attested"
    ]
    result["source_contract"] = source_contract
    result["collector_safety_lock_preserved"] = True
    return result
