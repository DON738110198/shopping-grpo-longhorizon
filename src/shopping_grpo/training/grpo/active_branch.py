"""Fail-closed contracts for controlled pivotal-state suffix branching.

The plan built here contains no rewards or terminal outcomes.  It binds one
captured actor-visible prompt to the exact public replay state, actor weights,
and decoding configuration before any environment lease or model request is
allowed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

from shopping_grpo.training.grpo.pivotal_states import (
    ACTOR_PROMPT_TOKENS_VERSION,
    branch_uid,
    token_ids_sha256,
)
from shopping_grpo.training.grpo.selection import (
    PIVOTAL_SELECTION_FIELDS,
    PIVOTAL_SELECTION_SOURCE_FIELDS,
)

ACTIVE_BRANCH_PLAN_VERSION = "shopping-active-branch-plan-v1"
ACTIVE_BRANCH_STRATEGY_VERSION = "pivotal-exact-prompt-k-suffix-v1"

_REQUIRED_DECODING_FIELDS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "max_tokens_mode",
    "seed_schedule",
    "min_tokens",
    "stop",
    "stop_token_ids",
    "ignore_eos",
    "max_steps",
    "prompt_length",
    "response_length",
    "context_window",
    "context_generation_reserve",
    "context_safety_margin",
    "context_input_budget",
    "context_preserve_recent_groups",
    "context_compaction_enable",
    "max_user_turns",
    "max_assistant_turns",
    "max_parallel_calls",
    "max_tool_response_length",
    "tool_response_truncate_side",
    "tokenization_sanity_check_mode",
    "apply_chat_template_kwargs",
    "mm_processor_kwargs",
    "tool_parser",
    "tool_schema_sha256",
    "generation_config_source",
    "sampling_backend_contract_sha256",
    "observation_token_budget",
    "observation_detail_token_budget",
    "observation_generic_token_budget",
    "observation_search_top_k",
    "observation_policy_sha256",
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


def observation_policy_sha256(config: Mapping[str, object]) -> str:
    """Hash the projection values that change actor-visible tool observations."""
    fields = (
        "observation_token_budget",
        "observation_detail_token_budget",
        "observation_generic_token_budget",
        "observation_search_top_k",
    )
    if not isinstance(config, Mapping):
        raise TypeError("observation policy config must be an object")
    if any(field not in config for field in fields):
        raise ValueError("observation policy config is incomplete")
    return _sha256_json({field: config[field] for field in fields})


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_number(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def validate_decoding_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate every behavior-changing suffix sampling input before hashing it."""
    if not isinstance(config, Mapping):
        raise TypeError("decoding_config must be an object")
    if set(config) != _REQUIRED_DECODING_FIELDS:
        raise ValueError("decoding_config fields do not match the active-branch contract")
    temperature = _finite_number(config.get("temperature"), "temperature")
    top_p = _finite_number(config.get("top_p"), "top_p")
    if temperature < 0.0:
        raise ValueError("temperature must be a finite number >= 0")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be a finite number in (0, 1]")
    top_k = config.get("top_k")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or (top_k != -1 and top_k < 1):
        raise ValueError("top_k must be -1 or a positive integer")
    min_p = _finite_number(config.get("min_p"), "min_p")
    if not 0.0 <= min_p <= 1.0:
        raise ValueError("min_p must be in [0, 1]")
    repetition_penalty = _finite_number(
        config.get("repetition_penalty"),
        "repetition_penalty",
    )
    if repetition_penalty <= 0.0:
        raise ValueError("repetition_penalty must be positive")
    presence_penalty = _finite_number(config.get("presence_penalty"), "presence_penalty")
    frequency_penalty = _finite_number(
        config.get("frequency_penalty"),
        "frequency_penalty",
    )
    if not -2.0 <= presence_penalty <= 2.0:
        raise ValueError("presence_penalty must be in [-2, 2]")
    if not -2.0 <= frequency_penalty <= 2.0:
        raise ValueError("frequency_penalty must be in [-2, 2]")
    max_tokens_mode = config.get("max_tokens_mode")
    if max_tokens_mode != "verl-v0.8-remaining-context":
        raise ValueError("max_tokens_mode must equal verl-v0.8-remaining-context")
    seed_schedule = config.get("seed_schedule")
    if seed_schedule != "first-suffix-then-sha256-turn-v1":
        raise ValueError("seed_schedule must equal first-suffix-then-sha256-turn-v1")
    min_tokens = _non_negative_int(config.get("min_tokens"), "min_tokens")
    stop = config.get("stop")
    if (
        not isinstance(stop, list)
        or any(not isinstance(value, str) or not value for value in stop)
        or len(stop) != len(set(stop))
    ):
        raise ValueError("stop must be a list of unique non-empty strings")
    stop_token_ids = config.get("stop_token_ids")
    if (
        not isinstance(stop_token_ids, list)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in stop_token_ids
        )
        or len(stop_token_ids) != len(set(stop_token_ids))
    ):
        raise ValueError("stop_token_ids must be unique non-negative integers")
    ignore_eos = config.get("ignore_eos")
    if not isinstance(ignore_eos, bool):
        raise TypeError("ignore_eos must be boolean")
    max_steps = _positive_int(config.get("max_steps"), "max_steps")
    prompt_length = _positive_int(config.get("prompt_length"), "prompt_length")
    response_length = _positive_int(config.get("response_length"), "response_length")
    context_window = _positive_int(config.get("context_window"), "context_window")
    context_generation_reserve = _positive_int(
        config.get("context_generation_reserve"),
        "context_generation_reserve",
    )
    context_safety_margin = _non_negative_int(
        config.get("context_safety_margin"),
        "context_safety_margin",
    )
    context_input_budget = _positive_int(
        config.get("context_input_budget"),
        "context_input_budget",
    )
    context_preserve_recent_groups = _positive_int(
        config.get("context_preserve_recent_groups"),
        "context_preserve_recent_groups",
    )
    context_compaction_enable = config.get("context_compaction_enable")
    if not isinstance(context_compaction_enable, bool):
        raise TypeError("context_compaction_enable must be boolean")
    if prompt_length + response_length > context_window:
        raise ValueError("prompt_length plus response_length must fit the context window")
    maximum_context_input = context_window - context_generation_reserve - context_safety_margin
    if maximum_context_input < 1:
        raise ValueError(
            "context_window must exceed context_generation_reserve plus context_safety_margin"
        )
    if context_input_budget > maximum_context_input:
        raise ValueError("context_input_budget does not fit the context window")
    max_user_turns = _positive_int(config.get("max_user_turns"), "max_user_turns")
    max_assistant_turns = _positive_int(
        config.get("max_assistant_turns"),
        "max_assistant_turns",
    )
    max_parallel_calls = _positive_int(
        config.get("max_parallel_calls"),
        "max_parallel_calls",
    )
    if max_parallel_calls != 1:
        raise ValueError("max_parallel_calls must equal 1")
    max_tool_response_length = _positive_int(
        config.get("max_tool_response_length"),
        "max_tool_response_length",
    )
    tool_response_truncate_side = config.get("tool_response_truncate_side")
    if tool_response_truncate_side not in {"left", "middle", "right"}:
        raise ValueError("tool_response_truncate_side must be left, middle, or right")
    tokenization_sanity_check_mode = config.get("tokenization_sanity_check_mode")
    if tokenization_sanity_check_mode not in {"disable", "strict", "ignore_strippable"}:
        raise ValueError("tokenization_sanity_check_mode is unsupported")
    apply_chat_template_kwargs = config.get("apply_chat_template_kwargs")
    mm_processor_kwargs = config.get("mm_processor_kwargs")
    if apply_chat_template_kwargs != {} or mm_processor_kwargs != {}:
        raise ValueError("active suffix processor kwargs must currently be empty objects")
    tool_parser = config.get("tool_parser")
    if not isinstance(tool_parser, str) or not tool_parser:
        raise ValueError("tool_parser must be a non-empty string")
    if config.get("generation_config_source") != "vllm":
        raise ValueError("generation_config_source must equal 'vllm'")
    for field in (
        "tool_schema_sha256",
        "observation_policy_sha256",
        "sampling_backend_contract_sha256",
    ):
        if not _is_sha256(config.get(field)):
            raise ValueError(f"{field} must be a lowercase SHA256 digest")
    observation_token_budget = _positive_int(
        config.get("observation_token_budget"),
        "observation_token_budget",
    )
    observation_detail_token_budget = _positive_int(
        config.get("observation_detail_token_budget"),
        "observation_detail_token_budget",
    )
    observation_generic_token_budget = _positive_int(
        config.get("observation_generic_token_budget"),
        "observation_generic_token_budget",
    )
    if (
        min(
            observation_token_budget,
            observation_detail_token_budget,
            observation_generic_token_budget,
        )
        < 64
    ):
        raise ValueError("observation token budgets must be at least 64")
    observation_search_top_k = _positive_int(
        config.get("observation_search_top_k"),
        "observation_search_top_k",
    )
    expected_observation_sha256 = observation_policy_sha256(config)
    if config.get("observation_policy_sha256") != expected_observation_sha256:
        raise ValueError("observation_policy_sha256 does not match observation settings")
    return {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "max_tokens_mode": str(max_tokens_mode),
        "seed_schedule": str(seed_schedule),
        "min_tokens": min_tokens,
        "stop": list(stop),
        "stop_token_ids": list(stop_token_ids),
        "ignore_eos": ignore_eos,
        "max_steps": max_steps,
        "prompt_length": prompt_length,
        "response_length": response_length,
        "context_window": context_window,
        "context_generation_reserve": context_generation_reserve,
        "context_safety_margin": context_safety_margin,
        "context_input_budget": context_input_budget,
        "context_preserve_recent_groups": context_preserve_recent_groups,
        "context_compaction_enable": context_compaction_enable,
        "max_user_turns": max_user_turns,
        "max_assistant_turns": max_assistant_turns,
        "max_parallel_calls": max_parallel_calls,
        "max_tool_response_length": max_tool_response_length,
        "tool_response_truncate_side": str(tool_response_truncate_side),
        "tokenization_sanity_check_mode": str(tokenization_sanity_check_mode),
        "apply_chat_template_kwargs": {},
        "mm_processor_kwargs": {},
        "tool_parser": tool_parser,
        "tool_schema_sha256": str(config["tool_schema_sha256"]),
        "generation_config_source": "vllm",
        "sampling_backend_contract_sha256": str(config["sampling_backend_contract_sha256"]),
        "observation_token_budget": observation_token_budget,
        "observation_detail_token_budget": observation_detail_token_budget,
        "observation_generic_token_budget": observation_generic_token_budget,
        "observation_search_top_k": observation_search_top_k,
        "observation_policy_sha256": expected_observation_sha256,
    }


def validate_actor_prompt_tokens(
    capture: Mapping[str, object],
    *,
    expected_sha256: str,
) -> dict[str, object]:
    """Validate exact prompt token materialization and its existing branch hash."""
    if not isinstance(capture, Mapping):
        raise TypeError("actor_prompt_tokens must be an object")
    if set(capture) != {"version", "sha256", "count", "tokens"}:
        raise ValueError("actor_prompt_tokens fields do not match the capture contract")
    if capture.get("version") != ACTOR_PROMPT_TOKENS_VERSION:
        raise ValueError("actor_prompt_tokens version mismatch")
    tokens = capture.get("tokens")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in tokens
        )
    ):
        raise ValueError("actor prompt tokens must be a non-empty list of token ids")
    count = capture.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count != len(tokens):
        raise ValueError("actor prompt token count mismatch")
    actual_sha256 = token_ids_sha256(tokens)
    if capture.get("sha256") != actual_sha256:
        raise ValueError("actor prompt token capture hash mismatch")
    if actual_sha256 != expected_sha256:
        raise ValueError("actor prompt tokens do not match the selected branch")
    return {
        "version": ACTOR_PROMPT_TOKENS_VERSION,
        "sha256": actual_sha256,
        "count": len(tokens),
        "tokens": list(tokens),
    }


def _suffix_seed(base_seed: int, active_group_uid: str, suffix_index: int) -> int:
    payload = f"{base_seed}:{active_group_uid}:{suffix_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _active_group_identity(
    *,
    parent_branch_uid: str,
    replay_state_id: str,
    task_id: int,
    prefix_action_count: int,
    actor_prompt_sha256: str,
    tokenizer_contract_sha256: str,
    environment_manifest_sha256: str,
    actor_checkpoint_sha256: str,
    decoding_config_sha256: str,
    base_seed: int,
    suffixes_per_state: int,
) -> dict[str, object]:
    return {
        "strategy_version": ACTIVE_BRANCH_STRATEGY_VERSION,
        "parent_branch_uid": parent_branch_uid,
        "replay_state_id": replay_state_id,
        "task_id": task_id,
        "prefix_action_count": prefix_action_count,
        "actor_prompt_sha256": actor_prompt_sha256,
        "tokenizer_contract_sha256": tokenizer_contract_sha256,
        "environment_manifest_sha256": environment_manifest_sha256,
        "actor_checkpoint_sha256": actor_checkpoint_sha256,
        "decoding_config_sha256": decoding_config_sha256,
        "base_seed": base_seed,
        "suffixes_per_state": suffixes_per_state,
    }


def build_active_branch_plan(
    resolved_selections: Sequence[Mapping[str, object]],
    *,
    actor_checkpoint_sha256: str,
    decoding_config: Mapping[str, object],
    seed: int,
    suffixes_per_state: int,
) -> dict[str, object]:
    """Build an outcome-blind K-suffix plan from resolved exact selections."""
    if not _is_sha256(actor_checkpoint_sha256):
        raise ValueError("actor_checkpoint_sha256 must be a lowercase SHA256 digest")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    suffix_count = _positive_int(suffixes_per_state, "suffixes_per_state")
    if suffix_count < 2:
        raise ValueError("suffixes_per_state must be at least 2")
    normalized_decoding = validate_decoding_config(decoding_config)
    decoding_sha256 = _sha256_json(normalized_decoding)
    if not resolved_selections:
        raise ValueError("resolved selection must contain at least one branch")

    groups = []
    seen_parent_branches = set()
    seen_group_ids = set()
    for group_index, resolved in enumerate(resolved_selections):
        if not isinstance(resolved, Mapping):
            raise TypeError(f"resolved selection {group_index} must be an object")
        selection = resolved.get("selection")
        trajectory = resolved.get("trajectory")
        event = resolved.get("event")
        if not all(isinstance(value, Mapping) for value in (selection, trajectory, event)):
            raise ValueError(f"resolved selection {group_index} is incomplete")
        if set(selection) != PIVOTAL_SELECTION_FIELDS:
            raise ValueError(f"resolved selection {group_index} fields do not match the contract")
        parent_branch_uid = str(selection.get("branch_uid") or "")
        replay_state_id = str(selection.get("replay_state_id") or "")
        if not _is_sha256(parent_branch_uid) or not _is_sha256(replay_state_id):
            raise ValueError(f"resolved selection {group_index} has invalid identities")
        if parent_branch_uid in seen_parent_branches:
            raise ValueError("resolved selection repeats parent branch_uid")
        seen_parent_branches.add(parent_branch_uid)
        if event.get("branch_uid") != parent_branch_uid:
            raise ValueError(f"resolved selection {group_index} event branch mismatch")
        if event.get("replay_state_id") != replay_state_id:
            raise ValueError(f"resolved selection {group_index} event state mismatch")
        actor_prompt_sha256 = str(event.get("actor_prompt_sha256") or "")
        tokenizer_sha256 = str(event.get("tokenizer_contract_sha256") or "")
        manifest_sha256 = str(trajectory.get("environment_manifest_sha256") or "")
        if not all(
            _is_sha256(value) for value in (actor_prompt_sha256, tokenizer_sha256, manifest_sha256)
        ):
            raise ValueError(f"resolved selection {group_index} has incomplete hash contracts")
        if branch_uid(replay_state_id, actor_prompt_sha256, tokenizer_sha256) != parent_branch_uid:
            raise ValueError(f"resolved selection {group_index} branch identity mismatch")
        prompt_capture = validate_actor_prompt_tokens(
            event.get("actor_prompt_tokens"),
            expected_sha256=actor_prompt_sha256,
        )
        task_id = _non_negative_int(selection.get("task_id"), "selection task_id")
        prefix_count = _non_negative_int(
            selection.get("prefix_action_count"),
            "selection prefix_action_count",
        )
        labels = selection.get("pivotal_labels")
        source = selection.get("source")
        if (
            not isinstance(labels, list)
            or not labels
            or labels != sorted(set(labels))
            or any(not isinstance(label, str) or not label for label in labels)
        ):
            raise ValueError(f"resolved selection {group_index} has invalid pivotal labels")
        if not isinstance(source, Mapping):
            raise TypeError(f"resolved selection {group_index} has invalid source")
        if set(source) != PIVOTAL_SELECTION_SOURCE_FIELDS:
            raise ValueError(
                f"resolved selection {group_index} source fields do not match the contract"
            )

        group_identity = _active_group_identity(
            parent_branch_uid=parent_branch_uid,
            replay_state_id=replay_state_id,
            task_id=task_id,
            prefix_action_count=prefix_count,
            actor_prompt_sha256=actor_prompt_sha256,
            tokenizer_contract_sha256=tokenizer_sha256,
            environment_manifest_sha256=manifest_sha256,
            actor_checkpoint_sha256=actor_checkpoint_sha256,
            decoding_config_sha256=decoding_sha256,
            base_seed=seed,
            suffixes_per_state=suffix_count,
        )
        active_group_uid = _sha256_json(group_identity)
        if active_group_uid in seen_group_ids:
            raise ValueError("active branch group identity collision")
        seen_group_ids.add(active_group_uid)
        suffixes = [
            {
                "suffix_index": suffix_index,
                "suffix_uid": _sha256_json(
                    {
                        "active_group_uid": active_group_uid,
                        "suffix_index": suffix_index,
                    }
                ),
                "seed": _suffix_seed(seed, active_group_uid, suffix_index),
            }
            for suffix_index in range(suffix_count)
        ]
        if len({suffix["seed"] for suffix in suffixes}) != suffix_count:
            raise ValueError("derived suffix seeds are not unique")
        groups.append(
            {
                "group_index": group_index,
                "active_group_uid": active_group_uid,
                "parent_branch_uid": parent_branch_uid,
                "replay_state_id": replay_state_id,
                "task_id": task_id,
                "prefix_action_count": prefix_count,
                "pivotal_labels": list(labels),
                "source": {name: source[name] for name in sorted(PIVOTAL_SELECTION_SOURCE_FIELDS)},
                "environment_manifest_sha256": manifest_sha256,
                "tokenizer_contract_sha256": tokenizer_sha256,
                "actor_prompt_tokens": prompt_capture,
                "suffixes": suffixes,
            }
        )

    plan = {
        "schema_version": ACTIVE_BRANCH_PLAN_VERSION,
        "strategy_version": ACTIVE_BRANCH_STRATEGY_VERSION,
        "seed": seed,
        "suffixes_per_state": suffix_count,
        "actor_checkpoint_sha256": actor_checkpoint_sha256,
        "decoding_config": normalized_decoding,
        "decoding_config_sha256": decoding_sha256,
        "aggregate": {
            "active_groups": len(groups),
            "unique_tasks": len({group["task_id"] for group in groups}),
            "planned_suffixes": len(groups) * suffix_count,
        },
        "groups": groups,
        "safety": {
            "active_branch": True,
            "outcome_blind": True,
            "outcome_fields_read": [],
            "uses_hidden_goal": False,
            "prompt_token_hash_verified": True,
            "optimizer_enabled": False,
            "training_ready": False,
        },
    }
    return validate_active_branch_plan(plan)


_PLAN_FIELDS = {
    "schema_version",
    "strategy_version",
    "seed",
    "suffixes_per_state",
    "actor_checkpoint_sha256",
    "decoding_config",
    "decoding_config_sha256",
    "aggregate",
    "groups",
    "safety",
}
_GROUP_FIELDS = {
    "group_index",
    "active_group_uid",
    "parent_branch_uid",
    "replay_state_id",
    "task_id",
    "prefix_action_count",
    "pivotal_labels",
    "source",
    "environment_manifest_sha256",
    "tokenizer_contract_sha256",
    "actor_prompt_tokens",
    "suffixes",
}
_SUFFIX_FIELDS = {"suffix_index", "suffix_uid", "seed"}
_AGGREGATE_FIELDS = {"active_groups", "unique_tasks", "planned_suffixes"}
_SAFETY_FIELDS = {
    "active_branch",
    "outcome_blind",
    "outcome_fields_read",
    "uses_hidden_goal",
    "prompt_token_hash_verified",
    "optimizer_enabled",
    "training_ready",
}
_EXPECTED_SAFETY = {
    "active_branch": True,
    "outcome_blind": True,
    "outcome_fields_read": [],
    "uses_hidden_goal": False,
    "prompt_token_hash_verified": True,
    "optimizer_enabled": False,
    "training_ready": False,
}


def _require_exact_fields(
    value: Mapping[str, object],
    expected: set[str] | frozenset[str],
    name: str,
) -> None:
    if set(value) != set(expected):
        raise ValueError(f"{name} fields do not match the active-branch contract")


def _validate_source(source: object, group_index: int) -> dict[str, object]:
    if not isinstance(source, Mapping):
        raise TypeError(f"active group {group_index} source must be an object")
    _require_exact_fields(
        source,
        PIVOTAL_SELECTION_SOURCE_FIELDS,
        f"active group {group_index} source",
    )
    input_index = _non_negative_int(source.get("input_index"), "source input_index")
    input_sha256 = str(source.get("input_sha256") or "")
    if not _is_sha256(input_sha256):
        raise ValueError(f"active group {group_index} source has invalid input_sha256")
    path = source.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"active group {group_index} source has invalid path")
    line = _positive_int(source.get("line"), "source line")
    trajectory_index = _non_negative_int(
        source.get("trajectory_index"),
        "source trajectory_index",
    )
    event_index = _non_negative_int(source.get("event_index"), "source event_index")
    nullable_indices = {}
    for field in ("global_step", "generation_batch"):
        value = source.get(field)
        nullable_indices[field] = (
            None if value is None else _non_negative_int(value, f"source {field}")
        )
    uid = source.get("uid")
    if not isinstance(uid, str):
        raise TypeError(f"active group {group_index} source uid must be a string")
    return {
        "input_index": input_index,
        "input_sha256": input_sha256,
        "path": path,
        "line": line,
        "global_step": nullable_indices["global_step"],
        "generation_batch": nullable_indices["generation_batch"],
        "uid": uid,
        "trajectory_index": trajectory_index,
        "event_index": event_index,
    }


def validate_active_branch_plan(plan: Mapping[str, object]) -> dict[str, object]:
    """Recompute every execution identity before a plan can reach model or environment I/O."""
    if not isinstance(plan, Mapping):
        raise TypeError("active branch plan must be an object")
    _require_exact_fields(plan, _PLAN_FIELDS, "active branch plan")
    if plan.get("schema_version") != ACTIVE_BRANCH_PLAN_VERSION:
        raise ValueError("active branch plan schema_version mismatch")
    if plan.get("strategy_version") != ACTIVE_BRANCH_STRATEGY_VERSION:
        raise ValueError("active branch plan strategy_version mismatch")
    seed = plan.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("active branch plan seed must be an integer")
    if seed < 0:
        raise ValueError("active branch plan seed must be non-negative")
    suffix_count = _positive_int(plan.get("suffixes_per_state"), "suffixes_per_state")
    if suffix_count < 2:
        raise ValueError("suffixes_per_state must be at least 2")
    actor_checkpoint_sha256 = str(plan.get("actor_checkpoint_sha256") or "")
    if not _is_sha256(actor_checkpoint_sha256):
        raise ValueError("actor_checkpoint_sha256 must be a lowercase SHA256 digest")
    decoding = plan.get("decoding_config")
    if not isinstance(decoding, Mapping):
        raise TypeError("decoding_config must be an object")
    normalized_decoding = validate_decoding_config(decoding)
    if _canonical_json(dict(decoding)) != _canonical_json(normalized_decoding):
        raise ValueError("decoding_config is not in canonical validated form")
    decoding_sha256 = _sha256_json(normalized_decoding)
    if plan.get("decoding_config_sha256") != decoding_sha256:
        raise ValueError("decoding_config_sha256 mismatch")

    raw_groups = plan.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("active branch plan groups must be a non-empty list")
    groups = []
    seen_parent_branches = set()
    seen_group_ids = set()
    seen_suffix_ids = set()
    for expected_group_index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, Mapping):
            raise TypeError(f"active group {expected_group_index} must be an object")
        _require_exact_fields(raw_group, _GROUP_FIELDS, f"active group {expected_group_index}")
        group_index = _non_negative_int(raw_group.get("group_index"), "active group index")
        if group_index != expected_group_index:
            raise ValueError(f"active group {expected_group_index} group_index mismatch")
        parent_branch_uid = str(raw_group.get("parent_branch_uid") or "")
        replay_state_fingerprint = str(raw_group.get("replay_state_id") or "")
        manifest_sha256 = str(raw_group.get("environment_manifest_sha256") or "")
        tokenizer_sha256 = str(raw_group.get("tokenizer_contract_sha256") or "")
        if not all(
            _is_sha256(value)
            for value in (
                parent_branch_uid,
                replay_state_fingerprint,
                manifest_sha256,
                tokenizer_sha256,
            )
        ):
            raise ValueError(f"active group {expected_group_index} has invalid hash identities")
        if parent_branch_uid in seen_parent_branches:
            raise ValueError("active branch plan repeats parent_branch_uid")
        seen_parent_branches.add(parent_branch_uid)
        task_id = _non_negative_int(raw_group.get("task_id"), "active group task_id")
        prefix_count = _non_negative_int(
            raw_group.get("prefix_action_count"),
            "active group prefix_action_count",
        )
        labels = raw_group.get("pivotal_labels")
        if (
            not isinstance(labels, list)
            or not labels
            or labels != sorted(set(labels))
            or any(not isinstance(label, str) or not label for label in labels)
        ):
            raise ValueError(f"active group {expected_group_index} has invalid pivotal labels")
        source = _validate_source(raw_group.get("source"), expected_group_index)
        prompt_capture_raw = raw_group.get("actor_prompt_tokens")
        if not isinstance(prompt_capture_raw, Mapping):
            raise TypeError(f"active group {expected_group_index} prompt capture must be an object")
        prompt_sha256 = str(prompt_capture_raw.get("sha256") or "")
        if not _is_sha256(prompt_sha256):
            raise ValueError(f"active group {expected_group_index} has invalid prompt hash")
        prompt_capture = validate_actor_prompt_tokens(
            prompt_capture_raw,
            expected_sha256=prompt_sha256,
        )
        if (
            branch_uid(replay_state_fingerprint, prompt_sha256, tokenizer_sha256)
            != parent_branch_uid
        ):
            raise ValueError(f"active group {expected_group_index} parent branch mismatch")
        expected_group_uid = _sha256_json(
            _active_group_identity(
                parent_branch_uid=parent_branch_uid,
                replay_state_id=replay_state_fingerprint,
                task_id=task_id,
                prefix_action_count=prefix_count,
                actor_prompt_sha256=prompt_sha256,
                tokenizer_contract_sha256=tokenizer_sha256,
                environment_manifest_sha256=manifest_sha256,
                actor_checkpoint_sha256=actor_checkpoint_sha256,
                decoding_config_sha256=decoding_sha256,
                base_seed=seed,
                suffixes_per_state=suffix_count,
            )
        )
        if raw_group.get("active_group_uid") != expected_group_uid:
            raise ValueError(f"active group {expected_group_index} active_group_uid mismatch")
        if expected_group_uid in seen_group_ids:
            raise ValueError("active branch plan repeats active_group_uid")
        seen_group_ids.add(expected_group_uid)

        raw_suffixes = raw_group.get("suffixes")
        if not isinstance(raw_suffixes, list) or len(raw_suffixes) != suffix_count:
            raise ValueError(f"active group {expected_group_index} suffix count mismatch")
        suffixes = []
        group_seeds = set()
        for suffix_index, raw_suffix in enumerate(raw_suffixes):
            if not isinstance(raw_suffix, Mapping):
                raise TypeError(
                    f"active group {expected_group_index} suffix {suffix_index} must be an object"
                )
            _require_exact_fields(
                raw_suffix,
                _SUFFIX_FIELDS,
                f"active group {expected_group_index} suffix {suffix_index}",
            )
            actual_suffix_index = _non_negative_int(
                raw_suffix.get("suffix_index"),
                "active suffix index",
            )
            if actual_suffix_index != suffix_index:
                raise ValueError(
                    f"active group {expected_group_index} suffix {suffix_index} index mismatch"
                )
            expected_suffix_uid = _sha256_json(
                {
                    "active_group_uid": expected_group_uid,
                    "suffix_index": suffix_index,
                }
            )
            if raw_suffix.get("suffix_uid") != expected_suffix_uid:
                raise ValueError(
                    f"active group {expected_group_index} suffix {suffix_index} uid mismatch"
                )
            if expected_suffix_uid in seen_suffix_ids:
                raise ValueError("active branch plan repeats suffix_uid")
            seen_suffix_ids.add(expected_suffix_uid)
            expected_seed = _suffix_seed(seed, expected_group_uid, suffix_index)
            actual_seed = _non_negative_int(raw_suffix.get("seed"), "active suffix seed")
            if actual_seed != expected_seed:
                raise ValueError(
                    f"active group {expected_group_index} suffix seed mismatch at {suffix_index}"
                )
            if expected_seed in group_seeds:
                raise ValueError(f"active group {expected_group_index} repeats suffix seed")
            group_seeds.add(expected_seed)
            suffixes.append(
                {
                    "suffix_index": suffix_index,
                    "suffix_uid": expected_suffix_uid,
                    "seed": expected_seed,
                }
            )
        groups.append(
            {
                "group_index": expected_group_index,
                "active_group_uid": expected_group_uid,
                "parent_branch_uid": parent_branch_uid,
                "replay_state_id": replay_state_fingerprint,
                "task_id": task_id,
                "prefix_action_count": prefix_count,
                "pivotal_labels": list(labels),
                "source": source,
                "environment_manifest_sha256": manifest_sha256,
                "tokenizer_contract_sha256": tokenizer_sha256,
                "actor_prompt_tokens": prompt_capture,
                "suffixes": suffixes,
            }
        )

    aggregate = plan.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise TypeError("active branch plan aggregate must be an object")
    _require_exact_fields(aggregate, _AGGREGATE_FIELDS, "active branch plan aggregate")
    expected_aggregate = {
        "active_groups": len(groups),
        "unique_tasks": len({group["task_id"] for group in groups}),
        "planned_suffixes": len(groups) * suffix_count,
    }
    for field in _AGGREGATE_FIELDS:
        _non_negative_int(aggregate.get(field), f"active branch aggregate {field}")
    if _canonical_json(dict(aggregate)) != _canonical_json(expected_aggregate):
        raise ValueError("active branch plan aggregate mismatch")
    safety = plan.get("safety")
    if not isinstance(safety, Mapping):
        raise TypeError("active branch plan safety must be an object")
    _require_exact_fields(safety, _SAFETY_FIELDS, "active branch plan safety")
    if _canonical_json(dict(safety)) != _canonical_json(_EXPECTED_SAFETY):
        raise ValueError("active branch plan safety contract mismatch")
    return {
        "schema_version": ACTIVE_BRANCH_PLAN_VERSION,
        "strategy_version": ACTIVE_BRANCH_STRATEGY_VERSION,
        "seed": seed,
        "suffixes_per_state": suffix_count,
        "actor_checkpoint_sha256": actor_checkpoint_sha256,
        "decoding_config": normalized_decoding,
        "decoding_config_sha256": decoding_sha256,
        "aggregate": expected_aggregate,
        "groups": groups,
        "safety": dict(_EXPECTED_SAFETY),
    }
