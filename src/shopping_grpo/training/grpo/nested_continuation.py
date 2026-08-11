"""Exact first-decision replay with independently sampled continuations.

This module is deliberately collection-only.  It turns observational active
suffix records into a frozen decision bank, replays each exact first Assistant
turn from the original branch, and forks multiple continuations only after the
post-action actor prompt and public harness state have been reproduced.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy

from shopping_grpo.training.grpo import nested_structure as _nested_structure
from shopping_grpo.training.grpo.active_suffix import (
    ACTIVE_SUFFIX_RESULT_VERSION,
    ActiveSuffixInfrastructureError,
    ActiveSuffixRunner,
    _canonical_json,
    _is_sha256,
    _sha256_json,
    sampling_backend_contract_sha256,
)
from shopping_grpo.training.grpo.nested_structure import (
    NESTED_DECISION_IDENTITY_VERSION,
    NESTED_STRUCTURE_EXCLUSION_REASONS,
    classify_nested_stage1_structure,
)
from shopping_grpo.training.grpo.pivotal_states import (
    canonical_replay_action,
    observation_sha256,
    replay_action_sha256,
    token_ids_sha256,
)

NESTED_DECISION_VERSION = "shopping-nested-decision-v1"
NESTED_CONTINUATION_VERSION = "shopping-nested-continuation-v2"
NESTED_COLLECTION_VERSION = "shopping-nested-continuation-collection-v2"
NESTED_HARNESS_CONTRACT_VERSION = "shopping-nested-harness-contract-v2"
NESTED_DECISION_CONTENT_VERSION = "shopping-nested-decision-content-v1"
NESTED_SEED_SCHEDULE_VERSION = "shopping-nested-state-crn-seed-v1"
NESTED_FOLD_CONTRACT_VERSION = "shopping-nested-train4-gate4-v1"
NESTED_ROLLOUT_CONTENT_VERSION = "shopping-nested-rollout-content-v2"
NESTED_EXCLUSION_AUDIT_VERSION = _nested_structure.NESTED_EXCLUSION_AUDIT_VERSION

MECHANICAL_CONTINUATIONS_PER_DECISION = 4
FORMAL_CONTINUATIONS_PER_DECISION = 8
TRAIN_CONTINUATION_INDICES = (0, 1, 2, 3)
GATE_CONTINUATION_INDICES = (4, 5, 6, 7)
MAX_SAMPLING_INVALID_RATE = 0.05


class _ProposalStructureExclusion(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code

_FORBIDDEN_KEYS = {
    "goal",
    "gold",
    "gold_asin",
    "hidden_goal",
    "reward_goal",
    "target",
    "target_asin",
}
_HARNESS_CONFIG_FIELDS = (
    "max_steps",
    "prompt_length",
    "response_length",
    "context_window",
    "context_generation_reserve",
    "context_safety_margin",
    "context_compaction_enable",
    "max_user_turns",
    "max_assistant_turns",
    "max_parallel_calls",
    "max_tool_response_length",
    "tool_response_truncate_side",
    "observation_token_budget",
    "observation_detail_token_budget",
    "observation_generic_token_budget",
    "observation_search_top_k",
    "observation_policy_sha256",
    "tool_parser",
)


def _find_forbidden_keys(value: object, prefix: str = "$") -> list[str]:
    found = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            path = f"{prefix}.{name}"
            if name.lower() in _FORBIDDEN_KEYS:
                found.append(path)
            found.extend(_find_forbidden_keys(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_find_forbidden_keys(item, f"{prefix}[{index}]"))
    return found


def _assert_no_hidden_goal(value: object) -> None:
    found = _find_forbidden_keys(value)
    if found:
        raise ActiveSuffixInfrastructureError(
            "nested artifact contains a forbidden hidden-goal field",
            code="hidden_goal_contract_invalid",
        )


def _integer_tokens(value: object, name: str, *, allow_empty: bool = False) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value):
        raise TypeError(f"{name} must contain non-negative integer token ids")
    if not value and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    return list(value)


def _logprobs(value: object, count: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{name} length must equal the token count")
    result = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise TypeError(f"{name} must contain finite numbers")
        item = float(item)
        if not math.isfinite(item):
            raise ValueError(f"{name} must contain finite numbers")
        result.append(item)
    return result


def _continuation_seed_uid(state_uid: str, continuation_index: int) -> str:
    if not _is_sha256(state_uid):
        raise ValueError("state_uid must be a SHA256 digest")
    if (
        not isinstance(continuation_index, int)
        or isinstance(continuation_index, bool)
        or continuation_index < 0
    ):
        raise ValueError("nested continuation index must be non-negative")
    return _sha256_json(
        {
            "version": NESTED_SEED_SCHEDULE_VERSION,
            "state_uid": state_uid,
            "continuation_index": continuation_index,
        }
    )


def _nested_seed(state_uid: str, continuation_index: int, turn_index: int) -> int:
    if not _is_sha256(state_uid):
        raise ValueError("state_uid must be a SHA256 digest")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (continuation_index, turn_index)
    ):
        raise ValueError("nested continuation seed indices must be non-negative")
    payload = _canonical_json(
        {
            "version": NESTED_SEED_SCHEDULE_VERSION,
            "state_uid": state_uid,
            "continuation_index": continuation_index,
            "turn_index": turn_index,
        }
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _continuation_fold(continuation_index: int) -> str:
    if continuation_index in TRAIN_CONTINUATION_INDICES:
        return "train"
    if continuation_index in GATE_CONTINUATION_INDICES:
        return "gate"
    raise ValueError("continuation index is outside the fixed train/gate fold contract")


def _continuation_uid(
    state_uid: str,
    decision_uid: str,
    continuation_index: int,
) -> str:
    seed_uid = _continuation_seed_uid(state_uid, continuation_index)
    return _sha256_json(
        {
            "state_uid": state_uid,
            "decision_uid": decision_uid,
            "continuation_index": continuation_index,
            "seed_schedule_version": NESTED_SEED_SCHEDULE_VERSION,
            "continuation_seed_uid": seed_uid,
        }
    )


def _with_rollout_content_sha256(record: dict[str, object]) -> dict[str, object]:
    if "rollout_content_sha256" in record:
        raise ValueError("rollout content hash must be added exactly once")
    record["rollout_content_sha256"] = _sha256_json(
        {
            "version": NESTED_ROLLOUT_CONTENT_VERSION,
            "record": record,
        }
    )
    return record


def build_nested_harness_contract(
    decoding_config: Mapping[str, object],
    *,
    policy_reward_sha256: str,
) -> dict[str, object]:
    """Bind every public counter/budget that affects a continuation boundary."""
    if not isinstance(decoding_config, Mapping):
        raise TypeError("decoding_config must be an object")
    missing = set(_HARNESS_CONFIG_FIELDS) - set(decoding_config)
    if missing:
        raise ValueError("decoding_config is missing nested harness fields")
    if not _is_sha256(policy_reward_sha256):
        raise ValueError("policy_reward_sha256 must be a SHA256 digest")
    return {
        "schema_version": NESTED_HARNESS_CONTRACT_VERSION,
        "decision_boundary": "after-first-decision-effect-v2",
        "prefix_runtime_restore": "shopping-active-prefix-runtime-v1",
        "recent_action_window": 3,
        "max_consecutive_guard_rejections": 3,
        "policy_reward_sha256": policy_reward_sha256,
        "config": {
            name: deepcopy(decoding_config[name]) for name in _HARNESS_CONFIG_FIELDS
        },
    }


def _harness_boundary_snapshot(boundary: Mapping[str, object]) -> dict[str, object]:
    state = boundary.get("state")
    prompt = _integer_tokens(boundary.get("prompt_token_ids"), "post-action prompt")
    if not isinstance(state, Mapping):
        raise ActiveSuffixInfrastructureError("post-action harness state is missing")
    first_action_sha256 = boundary.get("first_action_sha256")
    if not _is_sha256(first_action_sha256):
        raise ActiveSuffixInfrastructureError("post-action first action SHA256 is invalid")
    first_action_span = boundary.get("first_action_span")
    if (
        not isinstance(first_action_span, list)
        or len(first_action_span) != 2
        or first_action_span[0] != 0
        or not isinstance(first_action_span[1], int)
        or isinstance(first_action_span[1], bool)
        or first_action_span[1] < 1
    ):
        raise ActiveSuffixInfrastructureError("nested first action span is invalid")
    runtime_state = deepcopy(dict(state))
    runtime_state["latest_observation_raw"] = {
        "sha256": observation_sha256(state.get("latest_observation_raw", ""))
    }
    runtime_state["latest_observation"] = {
        "sha256": observation_sha256(state.get("latest_observation", ""))
    }
    _assert_no_hidden_goal(runtime_state)
    snapshot = {
        "first_action_sha256": first_action_sha256,
        "first_action_span": list(first_action_span),
        "post_action_prompt_sha256": token_ids_sha256(prompt),
        "post_action_prompt_token_count": len(prompt),
        "response_tokens_before": boundary.get("response_tokens_before"),
        "response_token_count": boundary.get("response_token_count"),
        "assistant_turns": boundary.get("assistant_turns"),
        "user_turns": boundary.get("user_turns"),
        "task_id": state.get("task_id"),
        "max_steps": state.get("max_steps"),
        "done": state.get("done"),
        "terminate": state.get("terminate"),
        "termination_reason": state.get("termination_reason"),
        "action_attempt_count": state.get("action_attempt_count"),
        "repeat_action_count": state.get("repeat_action_count"),
        "guard_rejection_count": state.get("guard_rejection_count"),
        "consecutive_guard_rejections": state.get("consecutive_guard_rejections"),
        "decision_count": state.get("decision_count"),
        "next_assistant_turn_id": state.get("next_assistant_turn_id"),
        "steps_sha256": _sha256_json(state.get("steps")),
        "action_events_sha256": _sha256_json(state.get("action_events")),
        "decision_events_sha256": _sha256_json(state.get("decision_events")),
        "replay_ledger_sha256": _sha256_json(state.get("replay_ledger")),
        "recent_action_signatures_sha256": _sha256_json(
            state.get("recent_action_signatures")
        ),
        "latest_raw_observation_sha256": observation_sha256(
            state.get("latest_observation_raw", "")
        ),
        "latest_visible_observation_sha256": observation_sha256(
            state.get("latest_observation", "")
        ),
        "runtime_state_sha256": _sha256_json(runtime_state),
    }
    scalar_integer_fields = (
        "response_tokens_before",
        "response_token_count",
        "assistant_turns",
        "user_turns",
        "task_id",
        "max_steps",
        "action_attempt_count",
        "repeat_action_count",
        "guard_rejection_count",
        "consecutive_guard_rejections",
        "decision_count",
        "next_assistant_turn_id",
    )
    if any(
        not isinstance(snapshot[name], int) or isinstance(snapshot[name], bool)
        for name in scalar_integer_fields
    ):
        raise ActiveSuffixInfrastructureError("post-action harness counters are invalid")
    if not isinstance(snapshot["done"], bool) or not isinstance(
        snapshot["terminate"], bool
    ):
        raise ActiveSuffixInfrastructureError("post-action terminal flags are invalid")
    terminal_boundary = snapshot["done"] or snapshot["terminate"]
    if snapshot["done"] and not snapshot["terminate"]:
        raise ActiveSuffixInfrastructureError("done boundary must also terminate")
    if not terminal_boundary and snapshot["termination_reason"] is not None:
        raise ActiveSuffixInfrastructureError(
            "non-terminal post-action boundary has a termination reason"
        )
    if terminal_boundary and (
        not isinstance(snapshot["termination_reason"], str)
        or not snapshot["termination_reason"]
    ):
        raise ActiveSuffixInfrastructureError(
            "terminal post-action boundary lacks a termination reason"
        )
    snapshot["boundary_kind"] = (
        "deterministic_terminal" if terminal_boundary else "continuation_required"
    )
    _assert_no_hidden_goal(snapshot)
    return snapshot


class _ForcedBoundClient:
    def __init__(self, owner: _ForcedDecisionCompletionClient, delegate):
        self.owner = owner
        self.delegate = delegate

    def complete(self, prompt_token_ids: Sequence[int], *, seed: int) -> dict[str, object]:
        return self.owner._complete(self.delegate, list(prompt_token_ids), seed=seed)


class _ForcedDecisionCompletionClient:
    """Inject one audited decision, then delegate only downstream requests."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.timeout = getattr(delegate, "timeout", None)
        self.context: dict[str, object] | None = None

    def attest_and_bind(self, **kwargs):
        return _ForcedBoundClient(self, self.delegate.attest_and_bind(**kwargs))

    def prepare(
        self,
        decision: Mapping[str, object],
        continuation_index: int,
        *,
        expected_boundary: Mapping[str, object] | None,
    ) -> None:
        if self.context is not None:
            raise ActiveSuffixInfrastructureError("nested completion context is already active")
        self.context = {
            "decision": decision,
            "continuation_index": continuation_index,
            "expected_boundary": expected_boundary,
            "forced_calls": 0,
            "downstream_request_seeds": [],
            "downstream_prompt_sha256": [],
            "boundary": None,
            "observed_boundary_summary": None,
        }

    def observe_boundary(self, boundary: Mapping[str, object]) -> None:
        context = self.context
        if context is None or context["forced_calls"] != 1:
            raise ActiveSuffixInfrastructureError("post-action boundary arrived out of order")
        if context["boundary"] is not None:
            raise ActiveSuffixInfrastructureError("post-action boundary was observed twice")
        decision = context["decision"]
        if boundary.get("group_index") != decision["group_index"] or boundary.get(
            "suffix_index"
        ) != decision["suffix_index"]:
            raise ActiveSuffixInfrastructureError("post-action boundary source changed")
        prompt = _integer_tokens(boundary.get("prompt_token_ids"), "post-action prompt")
        snapshot = _harness_boundary_snapshot(boundary)
        if boundary.get("first_action") != decision["first_action"] or boundary.get(
            "first_action_sha256"
        ) != decision["first_action_sha256"]:
            raise ActiveSuffixInfrastructureError("forced first action changed during replay")
        if boundary.get("first_action_span") != [
            0,
            len(decision["first_assistant_token_ids"]),
        ]:
            raise ActiveSuffixInfrastructureError("forced first decision span changed")
        observed = {
            "post_action_prompt_token_ids": prompt,
            "post_action_prompt_sha256": token_ids_sha256(prompt),
            "post_action_prompt_token_count": len(prompt),
            "harness_snapshot": snapshot,
            "harness_snapshot_sha256": _sha256_json(snapshot),
        }
        context["observed_boundary_summary"] = {
            "post_action_prompt_sha256": observed["post_action_prompt_sha256"],
            "post_action_prompt_token_count": observed[
                "post_action_prompt_token_count"
            ],
            "harness_snapshot_sha256": observed["harness_snapshot_sha256"],
            "boundary_kind": snapshot["boundary_kind"],
            "first_action_sha256": snapshot["first_action_sha256"],
        }
        expected = context["expected_boundary"]
        if expected is not None and observed != expected:
            raise ActiveSuffixInfrastructureError(
                "post-action prompt or harness differs across continuations",
                code="nested_boundary_parity_invalid",
            )
        context["boundary"] = observed

    def _complete(self, delegate, prompt: list[int], *, seed: int) -> dict[str, object]:
        context = self.context
        if context is None:
            raise ActiveSuffixInfrastructureError("nested completion context is not prepared")
        decision = context["decision"]
        if context["forced_calls"] == 0:
            if prompt != decision["pre_action_prompt_token_ids"]:
                raise ActiveSuffixInfrastructureError("forced decision pre-action prompt changed")
            if seed != decision["stage1_first_request_seed"]:
                raise ActiveSuffixInfrastructureError("forced decision source seed changed")
            context["forced_calls"] = 1
            return {
                "prompt_token_ids": list(prompt),
                "token_ids": list(decision["first_assistant_token_ids"]),
                "old_logprobs": list(decision["first_assistant_old_logprobs"]),
                "finish_reason": "stop",
                "stop_reason": None,
            }
        boundary = context.get("boundary")
        if not isinstance(boundary, Mapping):
            raise ActiveSuffixInfrastructureError(
                "downstream generation started before post-action boundary validation"
            )
        downstream_index = len(context["downstream_request_seeds"])
        if downstream_index == 0 and prompt != boundary["post_action_prompt_token_ids"]:
            raise ActiveSuffixInfrastructureError("first continuation prompt changed")
        nested_seed = _nested_seed(
            decision["state_uid"],
            context["continuation_index"],
            downstream_index,
        )
        if nested_seed in context["downstream_request_seeds"]:
            raise ActiveSuffixInfrastructureError("nested request seed collision")
        context["downstream_request_seeds"].append(nested_seed)
        context["downstream_prompt_sha256"].append(token_ids_sha256(prompt))
        return delegate.complete(prompt, seed=nested_seed)

    def finish(self) -> dict[str, object]:
        if self.context is None:
            raise ActiveSuffixInfrastructureError("nested completion context is not active")
        context = self.context
        self.context = None
        if context["forced_calls"] != 1 or not isinstance(context["boundary"], Mapping):
            raise ActiveSuffixInfrastructureError("nested decision boundary was not reproduced")
        snapshot = context["boundary"].get("harness_snapshot")
        if not isinstance(snapshot, Mapping):
            raise ActiveSuffixInfrastructureError("nested decision boundary snapshot is missing")
        if (
            context["downstream_request_seeds"]
            and snapshot.get("boundary_kind") == "deterministic_terminal"
        ):
            raise ActiveSuffixInfrastructureError(
                "terminal nested decision unexpectedly sampled a continuation"
            )
        return {
            "boundary": deepcopy(context["boundary"]),
            "downstream_request_seeds": list(context["downstream_request_seeds"]),
            "downstream_prompt_sha256": list(context["downstream_prompt_sha256"]),
            "observed_boundary_summary": deepcopy(
                context["observed_boundary_summary"]
            ),
        }

    def abort(self) -> dict[str, object]:
        context = self.context or {}
        self.context = None
        return {
            "boundary": deepcopy(context.get("boundary")),
            "downstream_request_seeds": list(
                context.get("downstream_request_seeds") or []
            ),
            "downstream_prompt_sha256": list(
                context.get("downstream_prompt_sha256") or []
            ),
            "observed_boundary_summary": deepcopy(
                context.get("observed_boundary_summary")
            ),
        }


class NestedContinuationCollector:
    """Replay a stage-one decision bank and fork exact downstream continuations."""

    def __init__(
        self,
        *,
        plan: Mapping[str, object],
        resolved_selections: Sequence[Mapping[str, object]],
        stage1_records: Sequence[Mapping[str, object]],
        stage1_source_sha256: str,
        actor_checkpoint,
        sampling_backend_contract: Mapping[str, object],
        completion_client,
        parser,
        encoder,
        env_factory,
        tool_schemas,
        required_environment_version: str,
        vllm_timeout_seconds: int = 180,
        environment_timeout_seconds: int = 60,
        expected_proposals: int | None = None,
        continuations_per_decision: int = FORMAL_CONTINUATIONS_PER_DECISION,
        policy_reward: object = None,
    ):
        if not _is_sha256(stage1_source_sha256):
            raise ValueError("stage1_source_sha256 must be a SHA256 digest")
        if expected_proposals is not None and (
            not isinstance(expected_proposals, int)
            or isinstance(expected_proposals, bool)
            or expected_proposals < 1
        ):
            raise ValueError("expected_proposals must be a positive integer or null")
        if (
            not isinstance(continuations_per_decision, int)
            or isinstance(continuations_per_decision, bool)
            or continuations_per_decision
            not in {
                MECHANICAL_CONTINUATIONS_PER_DECISION,
                FORMAL_CONTINUATIONS_PER_DECISION,
            }
        ):
            raise ValueError("continuations_per_decision must be exactly four or eight")
        # This is the stage-one proposal count. Exact duplicate decisions are
        # intentionally collapsed later while retaining their multiplicity.
        self.continuations_per_decision = continuations_per_decision
        self.stage1_source_sha256 = stage1_source_sha256
        self.forced_client = _ForcedDecisionCompletionClient(completion_client)
        self._lease_count = 0

        def counted_env_factory():
            env = env_factory()
            self._lease_count += 1
            return env

        self.runner = ActiveSuffixRunner(
            plan=plan,
            resolved_selections=resolved_selections,
            actor_checkpoint=actor_checkpoint,
            sampling_backend_contract=sampling_backend_contract,
            completion_client=self.forced_client,
            parser=parser,
            encoder=encoder,
            env_factory=counted_env_factory,
            tool_schemas=tool_schemas,
            required_environment_version=required_environment_version,
            policy_reward=policy_reward,
            vllm_timeout_seconds=vllm_timeout_seconds,
            environment_timeout_seconds=environment_timeout_seconds,
            expected_groups=len(plan.get("groups") or []),
            expected_suffixes_per_state=plan.get("suffixes_per_state"),
            post_first_decision_observer=self.forced_client.observe_boundary,
        )
        self.harness_contract = build_nested_harness_contract(
            self.runner.plan["decoding_config"],
            policy_reward_sha256=self.runner.policy_reward_sha256,
        )
        self.harness_contract_sha256 = _sha256_json(self.harness_contract)
        self.backend_sha256 = sampling_backend_contract_sha256(
            self.runner.backend_contract
        )
        planned_proposals = sum(
            len(group["suffixes"]) for group in self.runner.plan["groups"]
        )
        if expected_proposals is not None and expected_proposals != planned_proposals:
            raise ValueError("expected proposal count differs from the active plan")
        self.expected_proposals = planned_proposals
        self._source_audit_by_state: dict[str, list[dict[str, object]]] = {}
        self._state_exclusion_reasons: dict[str, list[dict[str, object]]] = {}
        self.decisions = self._build_decision_bank(stage1_records)
        self.stage1_structure = classify_nested_stage1_structure(
            self.runner.plan, stage1_records
        )
        if {
            decision["state_uid"] for decision in self.decisions
        } != set(self.stage1_structure["eligible_state_uids"]):
            raise ActiveSuffixInfrastructureError(
                "nested collector and pure stage-one classifier disagree",
                code="nested_structure_classifier_mismatch",
            )

    def _record_state_exclusion(
        self,
        state_uid: str,
        *,
        code: str,
        proposal_uids: Sequence[str],
        decision_uid: str | None = None,
    ) -> None:
        if code not in NESTED_STRUCTURE_EXCLUSION_REASONS:
            raise ActiveSuffixInfrastructureError(
                "nested collector produced a non-allowlisted exclusion reason",
                code="nested_structure_reason_invalid",
            )
        reason = {
            "code": code,
            "affected_proposal_uids": sorted(set(proposal_uids)),
            "decision_uid": decision_uid,
        }
        reasons = self._state_exclusion_reasons.setdefault(state_uid, [])
        if reason not in reasons:
            reasons.append(reason)

    def _validate_stage1_first_decision_structure(
        self,
        record: Mapping[str, object],
        *,
        group: Mapping[str, object],
        suffix: Mapping[str, object],
    ) -> None:
        identities = {
            "suffix_uid": suffix["suffix_uid"],
            "parent_branch_uid": group["parent_branch_uid"],
            "replay_state_id": group["replay_state_id"],
            "actor_checkpoint_sha256": self.runner.plan[
                "actor_checkpoint_sha256"
            ],
            "decoding_config_sha256": self.runner.plan[
                "decoding_config_sha256"
            ],
            "sampling_backend_contract_sha256": self.backend_sha256,
            "prompt_token_sha256": group["actor_prompt_tokens"]["sha256"],
        }
        if any(record.get(name) != value for name, value in identities.items()):
            raise _ProposalStructureExclusion("stage1_plan_identity_invalid")
        if record.get("task_id") != group["task_id"]:
            raise _ProposalStructureExclusion("stage1_task_identity_invalid")
        try:
            prompt = _integer_tokens(record.get("prompt_token_ids"), "pre-action prompt")
        except (TypeError, ValueError) as exc:
            raise _ProposalStructureExclusion("pre_action_prompt_invalid") from exc
        if (
            prompt != group["actor_prompt_tokens"]["tokens"]
            or token_ids_sha256(prompt) != record.get("prompt_token_sha256")
        ):
            raise _ProposalStructureExclusion("pre_action_prompt_invalid")
        try:
            response = _integer_tokens(record.get("response_ids"), "stage-one response")
            _logprobs(
                record.get("old_logprobs"), len(response), "stage-one old_logprobs"
            )
        except (TypeError, ValueError) as exc:
            raise _ProposalStructureExclusion("first_decision_tensors_invalid") from exc
        response_mask = record.get("response_mask")
        if (
            not isinstance(response_mask, list)
            or len(response_mask) != len(response)
            or any(value not in {0, 1} for value in response_mask)
        ):
            raise _ProposalStructureExclusion("first_decision_tensors_invalid")
        span = record.get("first_action_span")
        if (
            not isinstance(span, list)
            or len(span) != 2
            or span[0] != 0
            or not isinstance(span[1], int)
            or isinstance(span[1], bool)
            or not 0 < span[1] <= len(response)
            or any(value != 1 for value in response_mask[: span[1]])
        ):
            raise _ProposalStructureExclusion("first_decision_span_invalid")
        action = record.get("first_action")
        if not isinstance(action, Mapping):
            raise _ProposalStructureExclusion("first_decision_action_invalid")
        try:
            canonical_action = canonical_replay_action(
                action.get("tool"), action.get("parameters")
            )
            action_sha256 = replay_action_sha256(
                canonical_action["tool"], canonical_action["parameters"]
            )
        except (TypeError, ValueError) as exc:
            raise _ProposalStructureExclusion("first_decision_action_invalid") from exc
        if canonical_action != action or action_sha256 != record.get(
            "first_action_sha256"
        ):
            raise _ProposalStructureExclusion("first_decision_action_invalid")
        request_prompts = record.get("request_prompt_sha256")
        request_seeds = record.get("request_seeds")
        if (
            not isinstance(request_prompts, list)
            or not request_prompts
            or request_prompts[0] != group["actor_prompt_tokens"]["sha256"]
            or not isinstance(request_seeds, list)
            or not request_seeds
            or request_seeds[0] != suffix["seed"]
        ):
            raise _ProposalStructureExclusion("first_request_provenance_invalid")

    def _build_decision_bank(
        self, records: Sequence[Mapping[str, object]]
    ) -> list[dict[str, object]]:
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TypeError("stage1_records must be a sequence")
        if len(records) != self.expected_proposals:
            raise ValueError("stage-one proposal count does not match the contract")
        groups = {
            group["active_group_uid"]: (index, group)
            for index, group in enumerate(self.runner.plan["groups"])
        }
        decisions_by_uid: dict[tuple[str, str], dict[str, object]] = {}
        seen_sources = set()
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError("stage-one record must be an object")
            _assert_no_hidden_goal(record)
            if record.get("schema_version") != ACTIVE_SUFFIX_RESULT_VERSION:
                raise ValueError("stage-one result version mismatch")
            if (
                record.get("optimizer_enabled") is not False
                or record.get("uses_hidden_goal") is not False
            ):
                raise ValueError("stage-one safety contract is invalid")
            group_entry = groups.get(record.get("active_group_uid"))
            if group_entry is None:
                raise ValueError("stage-one record group is not in the active plan")
            group_index, group = group_entry
            suffix_index = record.get("suffix_index")
            if (
                not isinstance(suffix_index, int)
                or isinstance(suffix_index, bool)
                or not 0 <= suffix_index < len(group["suffixes"])
            ):
                raise ValueError("stage-one suffix index is invalid")
            source_key = (group_index, suffix_index)
            if source_key in seen_sources:
                raise ValueError("stage-one decision source is duplicated")
            seen_sources.add(source_key)
            suffix = group["suffixes"][suffix_index]
            state_uid = group["active_group_uid"]
            proposal_uid = _sha256_json(
                {
                    "state_uid": state_uid,
                    "proposal_index": suffix_index,
                }
            )
            stage1_record_sha256 = _sha256_json(dict(record))
            self._source_audit_by_state.setdefault(state_uid, []).append(
                {
                    "proposal_index": suffix_index,
                    "proposal_uid": proposal_uid,
                    "stage1_suffix_uid": suffix["suffix_uid"],
                    "stage1_record_sha256": stage1_record_sha256,
                }
            )
            try:
                self._validate_stage1_first_decision_structure(
                    record,
                    group=group,
                    suffix=suffix,
                )
            except _ProposalStructureExclusion as exc:
                self._record_state_exclusion(
                    state_uid,
                    code=exc.code,
                    proposal_uids=[proposal_uid],
                )
                continue
            identities = {
                "suffix_uid": suffix["suffix_uid"],
                "parent_branch_uid": group["parent_branch_uid"],
                "replay_state_id": group["replay_state_id"],
                "actor_checkpoint_sha256": self.runner.plan[
                    "actor_checkpoint_sha256"
                ],
                "decoding_config_sha256": self.runner.plan[
                    "decoding_config_sha256"
                ],
                "sampling_backend_contract_sha256": self.backend_sha256,
                "prompt_token_sha256": group["actor_prompt_tokens"]["sha256"],
            }
            if any(record.get(name) != value for name, value in identities.items()):
                raise ValueError("stage-one decision identity differs from the plan")
            if record.get("task_id") != group["task_id"]:
                raise ValueError("stage-one task differs from the plan")
            prompt = _integer_tokens(record.get("prompt_token_ids"), "pre-action prompt")
            if prompt != group["actor_prompt_tokens"]["tokens"]:
                raise ValueError("stage-one pre-action prompt differs from the plan")
            if token_ids_sha256(prompt) != record.get("prompt_token_sha256"):
                raise ValueError("stage-one pre-action prompt hash mismatch")
            response = _integer_tokens(record.get("response_ids"), "stage-one response")
            response_mask = record.get("response_mask")
            old_logprobs = _logprobs(
                record.get("old_logprobs"), len(response), "stage-one old_logprobs"
            )
            if (
                not isinstance(response_mask, list)
                or len(response_mask) != len(response)
                or any(value not in {0, 1} for value in response_mask)
            ):
                raise ValueError("stage-one response mask is invalid")
            span = record.get("first_action_span")
            if (
                not isinstance(span, list)
                or len(span) != 2
                or span[0] != 0
                or not isinstance(span[1], int)
                or isinstance(span[1], bool)
                or not 0 < span[1] <= len(response)
                or any(value != 1 for value in response_mask[: span[1]])
            ):
                raise ValueError("stage-one first decision span is invalid")
            first_tokens = response[: span[1]]
            first_logprobs = old_logprobs[: span[1]]
            action = record.get("first_action")
            if not isinstance(action, Mapping):
                raise TypeError("stage-one first action must be an object")
            canonical_action = canonical_replay_action(
                action.get("tool"), action.get("parameters")
            )
            action_sha256 = replay_action_sha256(
                canonical_action["tool"], canonical_action["parameters"]
            )
            if canonical_action != action or action_sha256 != record.get(
                "first_action_sha256"
            ):
                raise ValueError("stage-one first action hash mismatch")
            request_prompts = record.get("request_prompt_sha256")
            request_seeds = record.get("request_seeds")
            if (
                not isinstance(request_prompts, list)
                or not request_prompts
                or request_prompts[0] != group["actor_prompt_tokens"]["sha256"]
                or not isinstance(request_seeds, list)
                or not request_seeds
                or request_seeds[0] != suffix["seed"]
            ):
                raise ValueError("stage-one first request provenance is invalid")
            resolved = self.runner.resolved[group_index]
            trajectory = resolved.get("trajectory") or {}
            pre_action_prompt_sha256 = token_ids_sha256(prompt)
            first_assistant_token_sha256 = token_ids_sha256(first_tokens)
            decision_identity = {
                "version": NESTED_DECISION_IDENTITY_VERSION,
                "state_uid": state_uid,
                "pre_action_prompt_sha256": pre_action_prompt_sha256,
                "first_assistant_token_ids": first_tokens,
                "first_action_sha256": action_sha256,
            }
            decision_uid = _sha256_json(decision_identity)
            identity = {
                "decision_identity": decision_identity,
                "active_group_uid": state_uid,
                "parent_branch_uid": group["parent_branch_uid"],
                "replay_state_id": group["replay_state_id"],
                "task_id": group["task_id"],
                "environment_version": trajectory.get("environment_version"),
                "environment_manifest_sha256": group[
                    "environment_manifest_sha256"
                ],
                "policy_reward_sha256": self.runner.policy_reward_sha256,
                "harness_contract_sha256": self.harness_contract_sha256,
                "actor_checkpoint_sha256": self.runner.plan[
                    "actor_checkpoint_sha256"
                ],
                "decoding_config_sha256": self.runner.plan[
                    "decoding_config_sha256"
                ],
                "sampling_backend_contract_sha256": self.backend_sha256,
                "source_hashes": {
                    "active_branch_plan_sha256": self.runner.plan_sha256,
                    "resolved_selections_sha256": self.runner.resolved_sha256,
                    "stage1_source_sha256": self.stage1_source_sha256,
                },
            }
            if identity["environment_version"] != self.runner.required_environment_version:
                raise ValueError("stage-one environment version differs from the runner")
            source_proposal = {
                "proposal_index": suffix_index,
                "proposal_uid": proposal_uid,
                "source_uid": _sha256_json(
                    {
                        "proposal_uid": proposal_uid,
                        "stage1_suffix_uid": suffix["suffix_uid"],
                        "stage1_record_sha256": stage1_record_sha256,
                    }
                ),
                "stage1_suffix_uid": suffix["suffix_uid"],
                "stage1_suffix_index": suffix_index,
                "stage1_record_sha256": stage1_record_sha256,
                "stage1_first_request_seed": request_seeds[0],
                "stage1_first_request_prompt_sha256": request_prompts[0],
                "first_assistant_old_logprobs": first_logprobs,
                "first_assistant_old_logprobs_sha256": _sha256_json(first_logprobs),
            }
            key = (state_uid, decision_uid)
            existing = decisions_by_uid.get(key)
            if existing is not None:
                stable_fields = {
                    "identity": identity,
                    "group_index": group_index,
                    "task_id": group["task_id"],
                    "pre_action_prompt_token_ids": prompt,
                    "pre_action_prompt_sha256": pre_action_prompt_sha256,
                    "first_assistant_token_ids": first_tokens,
                    "first_assistant_token_sha256": first_assistant_token_sha256,
                    "first_assistant_loss_mask": [1] * len(first_tokens),
                    "first_action": canonical_action,
                    "first_action_sha256": action_sha256,
                }
                if any(existing.get(name) != value for name, value in stable_fields.items()):
                    raise ValueError("duplicate exact decision has inconsistent stable fields")
                if existing["first_assistant_old_logprobs"] != first_logprobs:
                    self._record_state_exclusion(
                        state_uid,
                        code="duplicate_decision_old_logprobs_mismatch",
                        proposal_uids=[
                            *existing.get("proposal_uids", []),
                            *[
                                item["proposal_uid"]
                                for item in existing["source_proposals"]
                            ],
                            proposal_uid,
                        ],
                        decision_uid=decision_uid,
                    )
                    continue
                existing["source_proposals"].append(source_proposal)
                continue
            decisions_by_uid[key] = {
                "schema_version": NESTED_DECISION_VERSION,
                "state_uid": state_uid,
                "decision_uid": decision_uid,
                "identity": identity,
                "group_index": group_index,
                "suffix_index": suffix_index,
                "active_group_uid": state_uid,
                "task_id": group["task_id"],
                "pre_action_prompt_token_ids": prompt,
                "pre_action_prompt_sha256": pre_action_prompt_sha256,
                "first_assistant_token_ids": first_tokens,
                "first_assistant_token_sha256": first_assistant_token_sha256,
                "first_assistant_old_logprobs": first_logprobs,
                "first_assistant_loss_mask": [1] * len(first_tokens),
                "first_action": canonical_action,
                "first_action_sha256": action_sha256,
                "stage1_first_request_seed": request_seeds[0],
                "source_proposals": [source_proposal],
                "proposal_multiplicity": 1,
                "continuations_expected": self.continuations_per_decision,
                "post_action_boundary": None,
                "continuation_uids": [],
                "structurally_valid": False,
                "optimizer_enabled": False,
                "training_ready": False,
                "uses_hidden_goal": False,
            }
        excluded_state_uids = set(self._state_exclusion_reasons)
        decisions = [
            decision
            for decision in decisions_by_uid.values()
            if decision["state_uid"] not in excluded_state_uids
        ]
        decisions.sort(
            key=lambda item: (
                item["group_index"],
                min(source["proposal_index"] for source in item["source_proposals"]),
                item["decision_uid"],
            )
        )
        state_decision_counts: Counter[str] = Counter()
        for decision in decisions:
            sources = sorted(
                decision["source_proposals"], key=lambda item: item["proposal_index"]
            )
            decision["source_proposals"] = sources
            decision["proposal_multiplicity"] = len(sources)
            decision["proposal_uids"] = [item["proposal_uid"] for item in sources]
            decision["decision_index"] = state_decision_counts[decision["state_uid"]]
            state_decision_counts[decision["state_uid"]] += 1
            replay_source = sources[0]
            decision["replay_source_proposal_uid"] = replay_source["proposal_uid"]
            decision["suffix_index"] = replay_source["stage1_suffix_index"]
            decision["stage1_first_request_seed"] = replay_source[
                "stage1_first_request_seed"
            ]
            decision["first_assistant_old_logprobs"] = replay_source[
                "first_assistant_old_logprobs"
            ]
            decision["decision_content_sha256"] = _sha256_json(
                {
                    "version": NESTED_DECISION_CONTENT_VERSION,
                    "state_uid": decision["state_uid"],
                    "decision_uid": decision["decision_uid"],
                    "pre_action_prompt_token_ids": decision[
                        "pre_action_prompt_token_ids"
                    ],
                    "first_assistant_token_ids": decision[
                        "first_assistant_token_ids"
                    ],
                    "first_assistant_old_logprobs": decision[
                        "first_assistant_old_logprobs"
                    ],
                    "first_assistant_loss_mask": decision[
                        "first_assistant_loss_mask"
                    ],
                    "first_action": decision["first_action"],
                    "first_action_sha256": decision["first_action_sha256"],
                }
            )
        return decisions

    async def _validate_decision_parsing(self) -> None:
        """Parse every frozen first turn before the first environment is leased."""
        for decision in self.decisions:
            parser_error_code = None
            try:
                calls = await self.runner._parse(decision["first_assistant_token_ids"])
                if not calls:
                    assistant_final = canonical_replay_action("assistant_final", {})
                    if decision["first_action"] != assistant_final:
                        parser_error_code = "first_decision_parse_empty"
                elif len(calls) > 1:
                    parallel = canonical_replay_action(
                        "parallel_tool_calls",
                        {"tools": [str(call["name"]) for call in calls]},
                    )
                    if decision["first_action"] != parallel:
                        parser_error_code = "first_decision_parallel_calls_mismatch"
                elif calls[0].get("arguments") is None:
                    malformed = canonical_replay_action(
                        "malformed_tool_arguments",
                        {"tool": str(calls[0]["name"])},
                    )
                    if decision["first_action"] != malformed:
                        parser_error_code = "first_decision_malformed_arguments_mismatch"
                else:
                    parsed = canonical_replay_action(
                        calls[0]["name"], calls[0]["arguments"]
                    )
                    if parsed != decision["first_action"]:
                        parser_error_code = "first_decision_parse_mismatch"
            except Exception:  # noqa: BLE001 - the parser is a runtime trust boundary.
                parser_error_code = "first_decision_parser_failure"
            if parser_error_code is not None:
                raise ActiveSuffixInfrastructureError(
                    "nested first-decision parser contract drift: "
                    f"{parser_error_code}",
                    code="nested_parser_contract_invalid",
                )

    def _build_exclusion_audit(self) -> dict[str, object]:
        eligible_proposal_uids = {
            source["proposal_uid"]
            for decision in self.decisions
            for source in decision["source_proposals"]
        }
        eligible_state_uids = {decision["state_uid"] for decision in self.decisions}
        if (
            eligible_state_uids != set(self.stage1_structure["eligible_state_uids"])
            or eligible_proposal_uids
            != set(self.stage1_structure["eligible_proposal_uids"])
            or set(self._state_exclusion_reasons)
            != set(self.stage1_structure["excluded_state_uids"])
        ):
            raise ActiveSuffixInfrastructureError(
                "nested decision bank differs from pure stage-one classification",
                code="nested_structure_classifier_mismatch",
            )
        audit = deepcopy(self.stage1_structure["exclusion_audit"])
        _assert_no_hidden_goal(audit)
        return audit

    def _invalid_continuation(
        self,
        decision: Mapping[str, object],
        continuation_index: int,
        exc: Exception,
        trace: Mapping[str, object],
        lease_sequence: int | None,
    ) -> dict[str, object]:
        continuation_uid = _continuation_uid(
            decision["state_uid"], decision["decision_uid"], continuation_index
        )
        record = {
            "schema_version": NESTED_CONTINUATION_VERSION,
            "continuation_uid": continuation_uid,
            "continuation_seed_uid": _continuation_seed_uid(
                decision["state_uid"], continuation_index
            ),
            "seed_schedule_version": NESTED_SEED_SCHEDULE_VERSION,
            "first_downstream_seed": (
                (trace.get("downstream_request_seeds") or [None])[0]
            ),
            "fold_contract_version": NESTED_FOLD_CONTRACT_VERSION,
            "fold": _continuation_fold(continuation_index),
            "generation_mode": "infrastructure_invalid",
            "state_uid": decision["state_uid"],
            "source_proposal_uids": list(decision["proposal_uids"]),
            "proposal_multiplicity": decision["proposal_multiplicity"],
            "decision_uid": decision["decision_uid"],
            "continuation_index": continuation_index,
            "task_id": decision["task_id"],
            "valid_for_learning": False,
            "infrastructure_invalid": True,
            "reward_invalid": False,
            "reward_unverifiable": False,
            "sampling_invalid": True,
            "infrastructure_error_code": (
                exc.code
                if isinstance(exc, ActiveSuffixInfrastructureError)
                else "external_failure"
            ),
            "infrastructure_error_class": exc.__class__.__name__,
            "invalid_reason": "nested_continuation_infrastructure_error",
            "post_action_prompt_sha256": (
                (trace.get("boundary") or {}).get("post_action_prompt_sha256")
            ),
            "harness_snapshot_sha256": (
                (trace.get("boundary") or {}).get("harness_snapshot_sha256")
            ),
            "observed_boundary_summary": deepcopy(
                trace.get("observed_boundary_summary")
            ),
            "downstream_request_seeds": list(
                trace.get("downstream_request_seeds") or []
            ),
            "strict": False,
            "policy_reward": 0.0,
            "terminal_utility": 0.0,
            "model_failure": False,
            "termination_reason": "nested_continuation_infrastructure_invalid",
            "lease_sequence": lease_sequence,
            "fresh_environment_lease": lease_sequence is not None,
            "release_verified": False,
            "optimizer_enabled": False,
            "uses_hidden_goal": False,
        }
        return _with_rollout_content_sha256(record)

    def _valid_continuation(
        self,
        decision: Mapping[str, object],
        continuation_index: int,
        result: Mapping[str, object],
        trace: Mapping[str, object],
        lease_sequence: int,
    ) -> dict[str, object]:
        first_count = len(decision["first_assistant_token_ids"])
        if result.get("first_action_span") != [0, first_count]:
            raise ActiveSuffixInfrastructureError("nested result first decision span changed")
        if result.get("response_ids", [])[:first_count] != decision[
            "first_assistant_token_ids"
        ]:
            raise ActiveSuffixInfrastructureError("nested result first decision tokens changed")
        if result.get("old_logprobs", [])[:first_count] != decision[
            "first_assistant_old_logprobs"
        ]:
            raise ActiveSuffixInfrastructureError("nested result first decision logprobs changed")
        if result.get("first_action") != decision["first_action"] or result.get(
            "first_action_sha256"
        ) != decision["first_action_sha256"]:
            raise ActiveSuffixInfrastructureError("nested result first action changed")
        replay = result.get("replay")
        if (
            not isinstance(replay, Mapping)
            or replay.get("verified") is not True
            or not isinstance(replay.get("verified_transitions"), int)
            or isinstance(replay.get("verified_transitions"), bool)
            or replay.get("verified_transitions") < 0
            or not _is_sha256(replay.get("raw_observation_sha256"))
            or not _is_sha256(replay.get("projected_observation_sha256"))
        ):
            raise ActiveSuffixInfrastructureError(
                "nested result lacks verified replay evidence",
                code="nested_replay_evidence_invalid",
            )
        boundary = trace.get("boundary")
        if not isinstance(boundary, Mapping):
            raise ActiveSuffixInfrastructureError("nested result lacks a validated boundary")
        downstream_request_seeds = list(trace.get("downstream_request_seeds") or [])
        expected_request_seeds = [
            _nested_seed(decision["state_uid"], continuation_index, turn_index)
            for turn_index in range(len(downstream_request_seeds))
        ]
        snapshot = boundary.get("harness_snapshot")
        if not isinstance(snapshot, Mapping):
            raise ActiveSuffixInfrastructureError("nested result boundary snapshot is missing")
        terminal_boundary = snapshot.get("boundary_kind") == "deterministic_terminal"
        if downstream_request_seeds != expected_request_seeds:
            raise ActiveSuffixInfrastructureError(
                "nested downstream request seed schedule changed",
                code="nested_seed_schedule_invalid",
            )
        if downstream_request_seeds and terminal_boundary:
            raise ActiveSuffixInfrastructureError(
                "nested generation mode differs from the terminal boundary",
                code="nested_generation_mode_invalid",
            )
        if downstream_request_seeds:
            generation_mode = "sampled_continuation"
        elif terminal_boundary:
            generation_mode = "deterministic_terminal"
        elif (
            result.get("model_failure") is True
            and isinstance(result.get("termination_reason"), str)
            and result.get("termination_reason")
        ):
            generation_mode = "no_generation_terminal"
        else:
            raise ActiveSuffixInfrastructureError(
                "non-terminal boundary produced no continuation or terminal result",
                code="nested_generation_mode_invalid",
            )
        continuation_uid = _continuation_uid(
            decision["state_uid"], decision["decision_uid"], continuation_index
        )
        valid_for_learning = bool(result["valid_for_learning"])
        infrastructure_invalid = bool(result["infrastructure_invalid"])
        model_failure = bool(result["model_failure"])
        invalid_reason = result["invalid_reason"]
        # Active-suffix collection may suppress an empty *new* completion because it
        # has no local tokens. Nested replay already owns the frozen first-decision
        # span, so the same model-attributable failure remains a trainable negative.
        if model_failure and not infrastructure_invalid:
            valid_for_learning = True
            invalid_reason = None
        reward_invalid = (
            not valid_for_learning and not infrastructure_invalid and not model_failure
        )
        reward_unverifiable = reward_invalid and invalid_reason == "reward_unverifiable"
        sampling_invalid = not valid_for_learning
        record = {
            "schema_version": NESTED_CONTINUATION_VERSION,
            "continuation_uid": continuation_uid,
            "continuation_seed_uid": _continuation_seed_uid(
                decision["state_uid"], continuation_index
            ),
            "seed_schedule_version": NESTED_SEED_SCHEDULE_VERSION,
            "first_downstream_seed": (
                downstream_request_seeds[0] if downstream_request_seeds else None
            ),
            "fold_contract_version": NESTED_FOLD_CONTRACT_VERSION,
            "fold": _continuation_fold(continuation_index),
            "generation_mode": generation_mode,
            "state_uid": decision["state_uid"],
            "source_proposal_uids": list(decision["proposal_uids"]),
            "proposal_multiplicity": decision["proposal_multiplicity"],
            "decision_uid": decision["decision_uid"],
            "continuation_index": continuation_index,
            "task_id": decision["task_id"],
            "post_action_prompt_sha256": boundary["post_action_prompt_sha256"],
            "post_action_prompt_token_count": boundary[
                "post_action_prompt_token_count"
            ],
            "harness_snapshot_sha256": boundary["harness_snapshot_sha256"],
            "observed_boundary_summary": deepcopy(
                trace.get("observed_boundary_summary")
            ),
            "replay": deepcopy(dict(replay)),
            "downstream_request_seeds": downstream_request_seeds,
            "downstream_prompt_sha256": list(trace["downstream_prompt_sha256"]),
            "response_ids": list(result["response_ids"]),
            "response_mask": list(result["response_mask"]),
            "old_logprobs": list(result["old_logprobs"]),
            "assistant_spans": deepcopy(result["assistant_spans"]),
            "valid_for_learning": valid_for_learning,
            "infrastructure_invalid": infrastructure_invalid,
            "reward_invalid": reward_invalid,
            "reward_unverifiable": reward_unverifiable,
            "sampling_invalid": sampling_invalid,
            "infrastructure_error_code": result["infrastructure_error_code"],
            "infrastructure_error_class": result["infrastructure_error_class"],
            "invalid_reason": invalid_reason,
            "strict": bool(result["strict"]),
            "policy_reward": float(result["policy_reward"]),
            "terminal_utility": float(result["terminal_utility"]),
            "reward_type": result["reward_type"],
            "model_failure": model_failure,
            "termination_reason": result["termination_reason"],
            "harness_limit_reason": result["harness_limit_reason"],
            "steps": int(result["steps"]),
            "guard_rejections": int(result["guard_rejections"]),
            "repeat_actions": int(result["repeat_actions"]),
            "lease_sequence": lease_sequence,
            "fresh_environment_lease": True,
            "release_verified": True,
            "optimizer_enabled": False,
            "uses_hidden_goal": False,
        }
        return _with_rollout_content_sha256(record)

    async def collect(self) -> dict[str, object]:
        await self._validate_decision_parsing()
        exclusion_audit = self._build_exclusion_audit()
        continuations = []
        decision_outputs = []
        for decision in self.decisions:
            expected_boundary = None
            decision_continuations = []
            for continuation_index in range(self.continuations_per_decision):
                lease_count_before = self._lease_count
                self.forced_client.prepare(
                    decision,
                    continuation_index,
                    expected_boundary=expected_boundary,
                )
                try:
                    result = await self.runner._collect_one(
                        decision["group_index"], decision["suffix_index"]
                    )
                    if self._lease_count != lease_count_before + 1:
                        raise ActiveSuffixInfrastructureError(
                            "nested continuation did not use exactly one fresh lease"
                        )
                    trace = self.forced_client.finish()
                    record = self._valid_continuation(
                        decision,
                        continuation_index,
                        result,
                        trace,
                        self._lease_count,
                    )
                except Exception as exc:  # noqa: BLE001 - persist bounded diagnostics.
                    trace = self.forced_client.abort()
                    lease_sequence = (
                        self._lease_count
                        if self._lease_count == lease_count_before + 1
                        else None
                    )
                    record = self._invalid_continuation(
                        decision,
                        continuation_index,
                        exc,
                        trace,
                        lease_sequence,
                    )
                if expected_boundary is None and not record["infrastructure_invalid"]:
                    expected_boundary = deepcopy(trace["boundary"])
                decision_continuations.append(record)
                continuations.append(record)
            output = deepcopy(decision)
            output["continuation_uids"] = [
                item["continuation_uid"] for item in decision_continuations
            ]
            if expected_boundary is not None:
                output["post_action_boundary"] = expected_boundary
            output["structurally_valid"] = all(
                not item["infrastructure_invalid"] for item in decision_continuations
            ) and len(decision_continuations) == self.continuations_per_decision
            output["credit_eligible"] = all(
                item["valid_for_learning"] for item in decision_continuations
            ) and len(decision_continuations) == self.continuations_per_decision
            decision_outputs.append(output)

        expected_continuations = len(self.decisions) * self.continuations_per_decision
        infrastructure_invalid = sum(
            int(item["infrastructure_invalid"]) for item in continuations
        )
        reward_invalid = sum(int(item["reward_invalid"]) for item in continuations)
        sampling_invalid = sum(int(item["sampling_invalid"]) for item in continuations)
        sampling_invalid_rate = (
            sampling_invalid / len(continuations) if continuations else 0.0
        )
        proposal_uids = [
            source["proposal_uid"]
            for decision in decision_outputs
            for source in decision["source_proposals"]
        ]
        proposal_count = len(proposal_uids)
        cardinality_complete = (
            len(decision_outputs) == len(self.decisions)
            and proposal_count == exclusion_audit["eligible_proposals"]
            and len(proposal_uids) == len(set(proposal_uids))
            and len(continuations) == expected_continuations
            and all(
                len(item["continuation_uids"])
                == self.continuations_per_decision
                for item in decision_outputs
            )
        )
        all_artifacts = {"decisions": decision_outputs, "continuations": continuations}
        _assert_no_hidden_goal(all_artifacts)
        summary = {
            "schema_version": NESTED_COLLECTION_VERSION,
            "experiment_uid": _sha256_json(
                {
                    "plan_sha256": self.runner.plan_sha256,
                    "stage1_source_sha256": self.stage1_source_sha256,
                    "harness_contract_sha256": self.harness_contract_sha256,
                    "continuations_per_decision": self.continuations_per_decision,
                    "fold_contract_version": NESTED_FOLD_CONTRACT_VERSION,
                }
            ),
            "plan_sha256": self.runner.plan_sha256,
            "resolved_selections_sha256": self.runner.resolved_sha256,
            "stage1_source_sha256": self.stage1_source_sha256,
            "harness_contract": self.harness_contract,
            "harness_contract_sha256": self.harness_contract_sha256,
            "exclusion_audit": exclusion_audit,
            "provenance": {
                "required_environment_version": self.runner.required_environment_version,
                "environment_manifest_sha256s": self.runner.environment_manifest_sha256s,
                "policy_reward_sha256": self.runner.policy_reward_sha256,
                "actor_checkpoint_sha256": self.runner.plan[
                    "actor_checkpoint_sha256"
                ],
                "decoding_config_sha256": self.runner.plan[
                    "decoding_config_sha256"
                ],
                "sampling_backend_contract_sha256": self.backend_sha256,
                "seed_schedule_version": NESTED_SEED_SCHEDULE_VERSION,
                "rollout_content_version": NESTED_ROLLOUT_CONTENT_VERSION,
            },
            "fold_contract": {
                "version": NESTED_FOLD_CONTRACT_VERSION,
                "formal_continuations_per_decision": FORMAL_CONTINUATIONS_PER_DECISION,
                "train_indices": list(TRAIN_CONTINUATION_INDICES),
                "gate_indices": list(GATE_CONTINUATION_INDICES),
                "gate_only_not_refit": True,
            },
            "aggregate": {
                "proposals": proposal_count,
                "pre_registered_proposals": exclusion_audit[
                    "pre_registered_proposals"
                ],
                "eligible_proposals": exclusion_audit["eligible_proposals"],
                "excluded_proposals": exclusion_audit["excluded_proposals"],
                "pre_registered_states": exclusion_audit["pre_registered_states"],
                "eligible_states": exclusion_audit["eligible_states"],
                "excluded_states": exclusion_audit["excluded_states"],
                "decisions": len(decision_outputs),
                "distinct_decisions": len(decision_outputs),
                "continuations_per_decision": self.continuations_per_decision,
                "continuations": len(continuations),
                "valid_decisions": sum(
                    int(item["structurally_valid"]) for item in decision_outputs
                ),
                "credit_eligible_decisions": sum(
                    int(item["credit_eligible"]) for item in decision_outputs
                ),
                "valid_for_learning_continuations": sum(
                    int(item["valid_for_learning"]) for item in continuations
                ),
                "infrastructure_invalid_continuations": infrastructure_invalid,
                "reward_invalid_continuations": reward_invalid,
                "reward_unverifiable_continuations": sum(
                    int(item["reward_unverifiable"]) for item in continuations
                ),
                "sampling_invalid_continuations": sampling_invalid,
                "sampling_invalid_rate": sampling_invalid_rate,
                "reward_invalid_reason_counts": dict(
                    sorted(
                        Counter(
                            str(item["invalid_reason"])
                            for item in continuations
                            if item["reward_invalid"]
                        ).items()
                    )
                ),
                "infrastructure_error_code_counts": dict(
                    sorted(
                        Counter(
                            str(item["infrastructure_error_code"])
                            for item in continuations
                            if item["infrastructure_invalid"]
                        ).items()
                    )
                ),
                "strict_success_continuations": sum(
                    int(item["strict"]) for item in continuations
                ),
                "model_failure_continuations": sum(
                    int(item["model_failure"]) for item in continuations
                ),
                "deterministic_terminal_continuations": sum(
                    item["generation_mode"] == "deterministic_terminal"
                    for item in continuations
                ),
                "cardinality_complete": cardinality_complete,
                "post_action_parity_complete": all(
                    item["structurally_valid"] for item in decision_outputs
                ),
                "fresh_lease_continuations": sum(
                    int(item["fresh_environment_lease"]) for item in continuations
                ),
                "release_verified_continuations": sum(
                    int(item["release_verified"]) for item in continuations
                ),
                "mechanical_collection_passed": (
                    bool(decision_outputs)
                    and cardinality_complete
                    and sampling_invalid_rate <= MAX_SAMPLING_INVALID_RATE
                ),
                "formal_fold_collection_passed": (
                    self.continuations_per_decision
                    == FORMAL_CONTINUATIONS_PER_DECISION
                    and bool(decision_outputs)
                    and cardinality_complete
                    and sampling_invalid_rate <= MAX_SAMPLING_INVALID_RATE
                ),
                "formal_experiment_gate_passed": False,
            },
            "safety": {
                "active_branch": True,
                "exact_first_decision_replay": True,
                "fresh_lease_per_continuation": True,
                "outcome_conditioned_resampling": False,
                "common_random_numbers_by_state_slot": True,
                "train_indices": list(TRAIN_CONTINUATION_INDICES),
                "gate_only_indices": list(GATE_CONTINUATION_INDICES),
                "all_eight_refit_allowed": False,
                "state_exclusion_no_backfill": True,
                "scale_collection_ready": False,
                "formal_scale_blockers": [
                    "streaming_journal_and_resume_not_implemented",
                    "actor_attestation_rehashed_per_continuation",
                ],
                "optimizer_enabled": False,
                "training_ready": False,
                "uses_hidden_goal": False,
            },
        }
        _assert_no_hidden_goal(summary)
        return {
            "decisions": decision_outputs,
            "continuations": continuations,
            "summary": summary,
        }
