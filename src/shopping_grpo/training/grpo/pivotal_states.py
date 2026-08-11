"""Public-state and assistant-turn evidence for pivotal-state GRPO audits.

The helpers in this module are deliberately independent of veRL and
ShopSimulator internals.  They only consume Actor-visible observations,
accepted tool calls, response masks, and terminal diagnostics.  This keeps the
offline feasibility audit useful without introducing a hidden-goal side
channel into rollout metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy

from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS, tool_call_to_action

PIVOTAL_AUDIT_VERSION = "shopping-pivotal-state-audit-v1"
REPLAY_STATE_VERSION = "shopping-public-replay-state-v1"
TURN_SPAN_VERSION = "shopping-assistant-turn-spans-v1"
DEFAULT_REPLAY_MAX_STEPS = 35
_NAVIGATION_TO_SEARCH = {"search_products", "back_to_search", "next_page"}
_PRODUCT_SUBPAGES = {
    "view_description",
    "view_features",
    "view_reviews",
    "view_attributes",
    "prev_page",
}
_REPLAY_TOOL_PARAMETER_SCHEMAS = {
    str(schema["function"]["name"]): schema["function"]["parameters"]
    for schema in SHOP_TOOL_SCHEMAS
    if str(schema["function"]["name"]) != "think"
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _bounded_parameters(parameters: object) -> dict[str, object]:
    if not isinstance(parameters, Mapping):
        return {}
    bounded: dict[str, object] = {}
    for key, value in list(parameters.items())[:8]:
        if isinstance(value, str):
            bounded[str(key)] = value[:256]
        elif isinstance(value, (bool, int, float)) or value is None:
            bounded[str(key)] = value
        else:
            bounded[str(key)] = str(value)[:256]
    return bounded


def canonical_replay_parameters(parameters: object) -> dict[str, object]:
    """Return an exact JSON copy of tool parameters, or fail instead of truncating."""
    if not isinstance(parameters, Mapping):
        return {}
    if any(not isinstance(key, str) for key in parameters):
        raise ValueError("replay parameter keys must be strings")
    try:
        serialized = _canonical_json(dict(parameters))
    except (TypeError, ValueError) as exc:
        raise ValueError("replay parameters must be finite JSON values") from exc
    if len(serialized.encode("utf-8")) > 16 * 1024:
        raise ValueError("replay parameters exceed the 16 KiB exact-record limit")
    normalized = json.loads(serialized)
    if not isinstance(normalized, dict):  # pragma: no cover - Mapping serializes to an object.
        raise TypeError("replay parameters did not serialize to an object")
    return normalized


def canonical_replay_action(tool: object, parameters: object) -> dict[str, object]:
    """Return the exact public action representation used by replay hashes."""
    return {
        "tool": str(tool or ""),
        "parameters": canonical_replay_parameters(parameters),
    }


def replay_action_sha256(tool: object, parameters: object) -> str:
    action = canonical_replay_action(tool, parameters)
    return hashlib.sha256(_canonical_json(action).encode("utf-8")).hexdigest()


def observation_sha256(observation: object) -> str:
    """Hash a public observation without retaining its potentially large text."""
    return hashlib.sha256(str(observation or "").encode("utf-8")).hexdigest()


def replay_state_id(
    task_id: int,
    accepted_actions: Sequence[Mapping[str, object]],
    observation_fingerprint: str,
    *,
    observation_kind: str,
    environment_manifest_sha256: str = "",
    public_query_sha256: str = "",
) -> str:
    """Identify a deterministic replay prefix plus its public observation.

    The accepted action prefix is intentionally part of the identity.  Two
    visually similar pages reached through different state transitions are not
    assumed to be interchangeable.
    """
    payload = {
        "version": REPLAY_STATE_VERSION,
        "task_id": int(task_id),
        "environment_manifest_sha256": str(environment_manifest_sha256),
        "public_query_sha256": str(public_query_sha256),
        "accepted_actions": [
            canonical_replay_action(action.get("tool"), action.get("parameters"))
            for action in accepted_actions
            if str(action.get("tool") or "") != "think"
        ],
        "observation_kind": str(observation_kind),
        "observation_sha256": str(observation_fingerprint),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def token_ids_sha256(token_ids: Sequence[object]) -> str:
    """Hash the exact Actor prompt token ids used by one generation call."""
    normalized = [int(token_id) for token_id in token_ids]
    return hashlib.sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()


def tokenizer_contract_sha256(tokenizer: object) -> str:
    """Bind prompt hashes to the tokenizer/chat-template contract that produced them."""
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    if not isinstance(init_kwargs, Mapping):
        init_kwargs = {}
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    added_vocab = get_added_vocab() if callable(get_added_vocab) else {}
    if not isinstance(added_vocab, Mapping):
        added_vocab = {}
    payload = {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "chat_template": str(getattr(tokenizer, "chat_template", "") or ""),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "added_vocab": {str(key): int(value) for key, value in added_vocab.items()},
        "all_special_ids": [
            int(token_id) for token_id in (getattr(tokenizer, "all_special_ids", []) or [])
        ],
        "commit_hash": str(init_kwargs.get("_commit_hash") or ""),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def branch_uid(
    replay_state_fingerprint: str,
    actor_prompt_sha256: str,
    tokenizer_fingerprint: str,
) -> str:
    """Identify both the environment replay state and exact policy-visible prompt."""
    payload = {
        "version": REPLAY_STATE_VERSION,
        "replay_state_id": str(replay_state_fingerprint),
        "actor_prompt_sha256": str(actor_prompt_sha256),
        "tokenizer_contract_sha256": str(tokenizer_fingerprint),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _policy_scope_id(
    source_path: str,
    line_number: int,
    global_step: object,
    explicit_scope: object = None,
) -> str:
    """Keep feasibility groups inside one frozen-policy collection scope."""
    if _is_sha256(explicit_scope):
        return str(explicit_scope)
    if global_step is None:
        return f"{source_path}:line={line_number}"
    return f"{source_path}:global_step={global_step}"


def validate_replay_ledger(
    trajectory: Mapping[str, object],
    *,
    max_steps: int = DEFAULT_REPLAY_MAX_STEPS,
) -> tuple[list[Mapping[str, object]] | None, str | None]:
    ledger = trajectory.get("replay_ledger")
    initial_hash = trajectory.get("initial_public_observation_sha256")
    if not isinstance(ledger, list):
        return None, "missing_replay_ledger"
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        return None, "invalid_max_steps"
    if len(ledger) > max_steps:
        return None, "replay_ledger_exceeds_max_steps"
    if not _is_sha256(initial_hash):
        return None, "invalid_initial_observation_hash"
    expected_before = str(initial_hash)
    for index, raw_transition in enumerate(ledger):
        if not isinstance(raw_transition, Mapping):
            return None, f"transition_{index}_not_object"
        if raw_transition.get("sequence") != index:
            return None, f"transition_{index}_sequence_mismatch"
        if raw_transition.get("before_public_observation_sha256") != expected_before:
            return None, f"transition_{index}_before_hash_mismatch"
        tool = str(raw_transition.get("tool") or "")
        if not tool:
            return None, f"transition_{index}_missing_tool"
        parameters = raw_transition.get("parameters")
        if not isinstance(parameters, Mapping):
            return None, f"transition_{index}_invalid_parameters"
        try:
            exact_parameters = canonical_replay_parameters(parameters)
        except ValueError:
            return None, f"transition_{index}_invalid_exact_parameters"
        schema = _REPLAY_TOOL_PARAMETER_SCHEMAS.get(tool)
        if schema is None:
            return None, f"transition_{index}_unknown_or_non_environment_tool"
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            return None, f"transition_{index}_invalid_tool_schema"
        extra_names = sorted(set(exact_parameters) - set(properties))
        if extra_names:
            return None, f"transition_{index}_schema_extra_arguments"
        missing_names = sorted(set(required) - set(exact_parameters))
        if missing_names:
            return None, f"transition_{index}_schema_missing_arguments"
        for name, value in exact_parameters.items():
            property_schema = properties.get(name, {})
            if property_schema.get("type") == "string" and not isinstance(value, str):
                return None, f"transition_{index}_schema_type_mismatch"
            allowed_values = property_schema.get("enum")
            if isinstance(allowed_values, list) and value not in allowed_values:
                return None, f"transition_{index}_schema_enum_mismatch"
        try:
            action = tool_call_to_action(tool, exact_parameters)
        except (KeyError, TypeError, ValueError):
            return None, f"transition_{index}_action_conversion_failed"
        if not isinstance(action, str) or not action:
            return None, f"transition_{index}_action_conversion_failed"
        done = raw_transition.get("done")
        if not isinstance(done, bool):
            return None, f"transition_{index}_invalid_done"
        after_hash = raw_transition.get("after_public_observation_sha256")
        if done:
            if after_hash is not None:
                return None, f"transition_{index}_terminal_after_hash_present"
            if index + 1 != len(ledger):
                return None, f"transition_{index}_actions_after_terminal"
        elif not _is_sha256(after_hash):
            return None, f"transition_{index}_invalid_after_hash"
        expected_before = str(after_hash or "")
    return ledger, None


def validate_event_training_contract(
    trajectory: Mapping[str, object],
    event: Mapping[str, object],
) -> tuple[bool, str | None]:
    """Verify that an event can be replayed and mapped to one exact loss span."""
    if trajectory.get("valid_for_learning") is not True or trajectory.get("invalid_reason"):
        return False, "invalid_learning_outcome"
    if trajectory.get("replay_observation_v2_complete") is not True:
        return False, "observation_v2_incomplete"
    if trajectory.get("replay_contract_error"):
        return False, "producer_replay_contract_error"
    if trajectory.get("replay_state_version") != REPLAY_STATE_VERSION:
        return False, "replay_state_version_mismatch"
    if trajectory.get("turn_span_version") != TURN_SPAN_VERSION:
        return False, "turn_span_version_mismatch"
    if trajectory.get("turn_span_valid") is not True:
        return False, "turn_span_invalid"
    manifest_hash = trajectory.get("environment_manifest_sha256")
    query_hash = trajectory.get("public_query_sha256")
    if not _is_sha256(manifest_hash):
        return False, "invalid_environment_manifest_hash"
    if not _is_sha256(query_hash):
        return False, "invalid_public_query_hash"
    ledger, ledger_error = validate_replay_ledger(trajectory)
    if ledger_error is not None or ledger is None:
        return False, ledger_error
    raw_prefix_count = event.get("prefix_action_count")
    if not isinstance(raw_prefix_count, int) or isinstance(raw_prefix_count, bool):
        return False, "invalid_prefix_action_count"
    prefix_count = raw_prefix_count
    if not 0 <= prefix_count <= len(ledger):
        return False, "prefix_action_count_out_of_range"
    if prefix_count and bool(ledger[prefix_count - 1].get("done")):
        return False, "branch_after_terminal"
    current_observation_hash = (
        trajectory.get("initial_public_observation_sha256")
        if prefix_count == 0
        else ledger[prefix_count - 1].get("after_public_observation_sha256")
    )
    if event.get("raw_observation_sha256") != current_observation_hash:
        return False, "event_observation_hash_mismatch"
    accepted_prefix = [
        canonical_replay_action(item.get("tool"), item.get("parameters"))
        for item in ledger[:prefix_count]
    ]
    expected_state_id = replay_state_id(
        int(trajectory["task_id"]),
        accepted_prefix,
        str(current_observation_hash),
        observation_kind="raw_public_observation",
        environment_manifest_sha256=str(manifest_hash),
        public_query_sha256=str(query_hash),
    )
    if event.get("replay_state_id") != expected_state_id:
        return False, "replay_state_id_mismatch"
    actor_prompt_hash = event.get("actor_prompt_sha256")
    tokenizer_hash = event.get("tokenizer_contract_sha256")
    if not _is_sha256(actor_prompt_hash) or not _is_sha256(tokenizer_hash):
        return False, "invalid_prompt_contract_hash"
    expected_branch_uid = branch_uid(
        expected_state_id,
        str(actor_prompt_hash),
        str(tokenizer_hash),
    )
    if event.get("branch_uid") != expected_branch_uid:
        return False, "branch_uid_mismatch"
    action_hash = event.get("action_sha256")
    if not _is_sha256(action_hash):
        return False, "invalid_action_hash"
    replay_parameters = event.get("replay_parameters")
    if not isinstance(replay_parameters, Mapping):
        return False, "missing_exact_action_parameters"
    try:
        expected_action_hash = replay_action_sha256(
            event.get("tool"),
            replay_parameters,
        )
    except ValueError:
        return False, "invalid_exact_action_parameters"
    if action_hash != expected_action_hash:
        return False, "action_hash_mismatch"
    if event.get("accepted") is True and event.get("tool") not in {
        "think",
        "assistant_final",
    }:
        if prefix_count >= len(ledger):
            return False, "accepted_action_missing_from_ledger"
        expected_transition_action = canonical_replay_action(
            ledger[prefix_count].get("tool"),
            ledger[prefix_count].get("parameters"),
        )
        if canonical_replay_action(event.get("tool"), replay_parameters) != (
            expected_transition_action
        ):
            return False, "accepted_action_ledger_mismatch"
    turn_id = event.get("assistant_turn_id")
    spans = trajectory.get("turn_spans")
    if not isinstance(spans, list):
        return False, "missing_turn_spans"
    matching_spans = [
        span for span in spans if isinstance(span, Mapping) and span.get("turn_id") == turn_id
    ]
    if len(matching_spans) != 1:
        return False, "assistant_turn_span_mismatch"
    span = matching_spans[0]
    if span.get("credit_eligible") is not True:
        return False, "assistant_turn_not_credit_eligible"
    assistant_span = span.get("assistant_span")
    if (
        not isinstance(assistant_span, list)
        or len(assistant_span) != 2
        or not 0 <= int(assistant_span[0]) < int(assistant_span[1])
    ):
        return False, "invalid_assistant_span"
    tool_names = span.get("tool_names")
    if span.get("kind") == "assistant_termination":
        if (
            event.get("tool") != "assistant_final"
            or span.get("tool_call_count") != 0
            or tool_names != []
        ):
            return False, "assistant_final_turn_mismatch"
    elif (
        span.get("kind") != "tool_call"
        or span.get("tool_call_count") != 1
        or not isinstance(tool_names, list)
        or tool_names != [str(event.get("tool") or "")]
    ):
        return False, "assistant_turn_tool_mismatch"
    return True, None


def response_mask_spans(response_mask: Sequence[object]) -> list[tuple[int, int]]:
    """Return every contiguous Assistant-token run in a veRL response mask."""
    if response_mask and int(response_mask[0]) != 1:
        raise ValueError("response_mask must begin with an Assistant token")
    spans = []
    index = 0
    while index < len(response_mask):
        value = int(response_mask[index])
        if value not in {0, 1}:
            raise ValueError("response_mask must contain only 0/1 values")
        if value == 0:
            index += 1
            continue
        start = index
        while index < len(response_mask) and int(response_mask[index]) == 1:
            index += 1
        spans.append((start, index))
    return spans


def materialize_assistant_turn_spans(
    turn_records: Sequence[Mapping[str, object]],
    response_mask: Sequence[object],
) -> list[dict[str, object]]:
    """Resolve stable generation records into final, unpadded response coordinates.

    Context compaction removes complete leading Assistant/observation groups.
    Callers therefore drop the corresponding leading records when compaction is
    applied, and this function reconstructs coordinates only from the final
    response mask rather than carrying stale offsets across compaction.
    """
    assistant_runs = response_mask_spans(response_mask)
    if len(turn_records) not in {len(assistant_runs), len(assistant_runs) + 1}:
        raise ValueError(
            "assistant turn record count does not match response_mask runs: "
            f"records={len(turn_records)}, mask_runs={len(assistant_runs)}"
        )
    if len(turn_records) == len(assistant_runs) + 1:
        omitted = turn_records[-1]
        if omitted.get("kind") != "assistant_termination":
            raise ValueError("only a final truncated Assistant termination may be omitted")
    turn_ids = [int(record["turn_id"]) for record in turn_records]
    if len(turn_ids) != len(set(turn_ids)):
        raise ValueError("assistant turn ids must be unique")
    for index, record in enumerate(turn_records):
        kind = str(record.get("kind") or "")
        if kind not in {"tool_call", "assistant_termination"}:
            raise ValueError(f"assistant turn {record.get('turn_id')} has invalid kind")
        if kind == "assistant_termination" and index + 1 != len(turn_records):
            raise ValueError("assistant termination must be the final turn")
    spans = []
    previous_end = 0
    for index, (raw_record, (assistant_start, assistant_end)) in enumerate(
        zip(turn_records[: len(assistant_runs)], assistant_runs, strict=True)
    ):
        record = deepcopy(dict(raw_record))
        generated_tokens = int(record["generated_token_count"])
        retained_tokens = assistant_end - assistant_start
        truncated = generated_tokens > retained_tokens
        if generated_tokens != retained_tokens and not (
            index + 1 == len(turn_records)
            and record.get("kind") == "assistant_termination"
            and truncated
        ):
            raise ValueError(
                f"assistant turn {record.get('turn_id')} token count mismatch: "
                f"record={generated_tokens}, mask={retained_tokens}"
            )
        if assistant_start < previous_end:
            raise ValueError("assistant turn spans overlap")
        observation_end = (
            assistant_runs[index + 1][0] if index + 1 < len(assistant_runs) else len(response_mask)
        )
        if any(int(value) != 0 for value in response_mask[assistant_end:observation_end]):
            raise ValueError("observation span contains Assistant tokens")
        record.update(
            {
                "coordinate_space": "unpadded_response_v1",
                "assistant_span": [assistant_start, assistant_end],
                "observation_span": [assistant_end, observation_end],
            }
        )
        if truncated:
            record["truncated"] = True
            record["credit_eligible"] = False
        if record.get("kind") == "assistant_termination" and observation_end != assistant_end:
            raise ValueError("assistant termination cannot have an observation span")
        if (
            record.get("kind") == "tool_call"
            and observation_end == assistant_end
            and record.get("credit_eligible") is True
        ):
            raise ValueError("credit-eligible tool turn is missing its observation")
        spans.append(record)
        previous_end = observation_end
    return spans


def rebase_assistant_turn_spans(
    spans: Sequence[Mapping[str, object]],
    removed_tokens: int,
) -> list[dict[str, object]]:
    """Rebase stable turn spans after complete leading tool groups are compacted."""
    removed_tokens = int(removed_tokens)
    if removed_tokens < 0:
        raise ValueError("removed_tokens must be non-negative")
    rebased = []
    for raw_span in spans:
        span = deepcopy(dict(raw_span))
        start = int(span["start"])
        end = int(span["end"])
        if not 0 <= start < end:
            raise ValueError(f"invalid assistant turn span [{start}, {end})")
        if end <= removed_tokens:
            continue
        if start < removed_tokens < end:
            raise ValueError("context compaction split an assistant turn span")
        span["start"] = start - removed_tokens
        span["end"] = end - removed_tokens
        rebased.append(span)
    return rebased


def finalize_assistant_turn_spans(
    spans: Sequence[Mapping[str, object]],
    response_mask: Sequence[object],
) -> list[dict[str, object]]:
    """Clip spans to the retained response and prove exact mask alignment."""
    response_length = len(response_mask)
    finalized = []
    seen_turn_ids = set()
    for raw_span in spans:
        span = deepcopy(dict(raw_span))
        turn_id = int(span["turn_id"])
        if turn_id in seen_turn_ids:
            raise ValueError(f"duplicate assistant turn_id {turn_id}")
        seen_turn_ids.add(turn_id)
        start = int(span["start"])
        end = int(span["end"])
        if not 0 <= start < end:
            raise ValueError(f"invalid assistant turn span [{start}, {end})")
        if start >= response_length:
            continue
        if end > response_length:
            span["end"] = response_length
            span["truncated"] = True
        finalized.append(span)

    actual = [(int(span["start"]), int(span["end"])) for span in finalized]
    expected = response_mask_spans(response_mask)
    if actual != expected:
        raise ValueError(
            "assistant turn spans do not match response_mask runs: "
            f"spans={actual}, mask_runs={expected}"
        )
    return finalized


def _accepted_prefix(events: Sequence[Mapping[str, object]], end: int) -> list[dict[str, object]]:
    prefix = []
    for event in events[:end]:
        if event.get("accepted") is not True or event.get("error"):
            continue
        action_parameters = event.get("replay_parameters")
        if not isinstance(action_parameters, Mapping):
            action_parameters = event.get("parameters")
        action = canonical_replay_action(event.get("tool"), action_parameters)
        if action["tool"] != "think":
            prefix.append(action)
    return prefix


def _public_phase(actions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    candidate_open = False
    selected_options = 0
    for action in actions:
        tool = str(action.get("tool") or "")
        if tool == "open_product":
            candidate_open = True
            selected_options = 0
        elif tool == "select_option" and candidate_open:
            selected_options += 1
        elif tool in _NAVIGATION_TO_SEARCH:
            candidate_open = False
            selected_options = 0
        elif tool in _PRODUCT_SUBPAGES:
            continue
        elif tool in {"buy_now", "finish_without_purchase"}:
            candidate_open = False
            selected_options = 0
    return {
        "candidate_open": candidate_open,
        "selected_option_count": selected_options,
    }


def pivotal_labels(
    accepted_prefix: Sequence[Mapping[str, object]],
    current_event: Mapping[str, object],
    previous_event: Mapping[str, object] | None,
) -> tuple[str, ...]:
    """Classify only generic, Actor-visible decision states."""
    phase = _public_phase(accepted_prefix)
    tool = str(current_event.get("tool") or "")
    labels = []
    if phase["candidate_open"]:
        labels.append("candidate_open")
    if int(phase["selected_option_count"]) > 0:
        labels.append("option_selected")
    if previous_event is not None and previous_event.get("accepted") is False:
        labels.append("post_guard")
    if bool(current_event.get("repeated")):
        labels.append("repeat_attempt")
    if tool == "buy_now":
        labels.append("pre_commit")
    if tool == "think":
        labels.append("non_environment_think")
    if tool == "assistant_final":
        labels.append("premature_assistant_final")
    if phase["candidate_open"] and tool in _NAVIGATION_TO_SEARCH | {"open_product"}:
        labels.append("candidate_abandonment")
    if int(phase["selected_option_count"]) > 0 and tool == "select_option":
        labels.append("option_reselection")
    return tuple(labels)


def _required_finite_number(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _trajectory_visits(
    record: Mapping[str, object],
    trajectory: Mapping[str, object],
    *,
    source_path: str,
    line_number: int,
    trajectory_index: int,
) -> list[dict[str, object]]:
    events = trajectory.get("decision_trace", trajectory.get("action_trace"))
    if not isinstance(events, list):
        return []
    if trajectory.get("valid_for_learning") is not True or trajectory.get("invalid_reason"):
        return []
    task_id = int(trajectory["task_id"])
    strict_value = _required_finite_number(trajectory.get("strict"), "strict")
    if strict_value not in {0.0, 1.0}:
        raise ValueError("strict must equal 0 or 1")
    strict = bool(strict_value)
    terminal_utility = _required_finite_number(
        trajectory.get("terminal_utility"),
        "terminal_utility",
    )
    policy_reward = _required_finite_number(
        trajectory.get("policy_reward", terminal_utility),
        "policy_reward",
    )
    policy_scope_id = _policy_scope_id(
        source_path,
        line_number,
        record.get("global_step"),
        record.get("policy_scope_uid"),
    )
    if all(record.get(key) is not None for key in ("global_step", "generation_batch", "uid")):
        occurrence = (
            f"{policy_scope_id}:step{record['global_step']}:"
            f"batch{record['generation_batch']}:uid{record['uid']}:t{trajectory_index}"
        )
    else:
        occurrence = f"{source_path}:{line_number}:t{trajectory_index}"
    visits = []
    previous_event = None
    seen_visit_keys = set()
    for event_index, raw_event in enumerate(events):
        if not isinstance(raw_event, Mapping):
            continue
        event = dict(raw_event)
        prefix = _accepted_prefix(events, event_index)
        raw_observation_hash = str(event.get("raw_observation_sha256") or "")
        projected_observation_hash = str(event.get("observation_sha256") or "")
        if raw_observation_hash:
            observation_hash = raw_observation_hash
            observation_kind = "raw_public_observation"
        elif projected_observation_hash:
            observation_hash = projected_observation_hash
            observation_kind = "legacy_projected_observation"
        else:
            previous_event = event
            continue
        training_contract_complete, contract_error = validate_event_training_contract(
            trajectory,
            event,
        )
        if training_contract_complete:
            prefix_count = int(event["prefix_action_count"])
            prefix = [
                canonical_replay_action(item.get("tool"), item.get("parameters"))
                for item in trajectory["replay_ledger"][:prefix_count]
            ]
        state_id = str(
            event.get("replay_state_id")
            or replay_state_id(
                task_id,
                prefix,
                observation_hash,
                observation_kind=observation_kind,
            )
        )
        labels = pivotal_labels(prefix, event, previous_event)
        previous_event = event
        visit_key = (
            state_id,
            str(event.get("branch_uid") or "")
            if training_contract_complete
            else f"legacy_or_incomplete:{event_index}",
        )
        if not labels or visit_key in seen_visit_keys:
            continue
        seen_visit_keys.add(visit_key)
        action_parameters = (
            event.get("replay_parameters")
            if training_contract_complete
            else event.get("parameters")
        )
        action = canonical_replay_action(event.get("tool"), action_parameters)
        visits.append(
            {
                "replay_state_id": state_id,
                "replay_state_version": REPLAY_STATE_VERSION,
                "observation_kind": observation_kind,
                "task_id": task_id,
                "prefix_action_count": len(prefix),
                "pivotal_labels": labels,
                "next_action": action,
                "next_action_sha256": str(
                    event.get("action_sha256")
                    or replay_action_sha256(action["tool"], action["parameters"])
                ),
                "next_action_accepted": event.get("accepted") is True,
                "next_action_repeated": bool(event.get("repeated")),
                "assistant_turn_id": event.get("assistant_turn_id"),
                "branch_uid": str(event.get("branch_uid") or ""),
                "training_contract_complete": training_contract_complete,
                "training_contract_error": contract_error,
                "policy_scope_id": policy_scope_id,
                "strict": strict,
                "terminal_utility": terminal_utility,
                "policy_reward": policy_reward,
                "reward_type": str(trajectory.get("reward_type") or ""),
                "termination_reason": str(trajectory.get("termination_reason") or ""),
                "source": {
                    "path": source_path,
                    "line": line_number,
                    "global_step": record.get("global_step"),
                    "generation_batch": record.get("generation_batch"),
                    "uid": str(record.get("uid") or ""),
                    "trajectory_index": trajectory_index,
                    "occurrence": occurrence,
                },
            }
        )
    return visits


def _action_outcome_table(
    visits: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    buckets: dict[str, dict[str, object]] = {}
    for visit in visits:
        action_hash = str(visit["next_action_sha256"])
        bucket = buckets.setdefault(
            action_hash,
            {
                "action_sha256": action_hash,
                "action_summary": deepcopy(visit["next_action"]),
                "visits": 0,
                "strict_successes": 0,
                "strict_failures": 0,
                "utility_min": float(visit["terminal_utility"]),
                "utility_max": float(visit["terminal_utility"]),
            },
        )
        bucket["visits"] = int(bucket["visits"]) + 1
        bucket["strict_successes"] = int(bucket["strict_successes"]) + int(bool(visit["strict"]))
        bucket["strict_failures"] = int(bucket["strict_failures"]) + int(not bool(visit["strict"]))
        bucket["utility_min"] = min(
            float(bucket["utility_min"]),
            float(visit["terminal_utility"]),
        )
        bucket["utility_max"] = max(
            float(bucket["utility_max"]),
            float(visit["terminal_utility"]),
        )
    return sorted(
        buckets.values(),
        key=lambda bucket: (-int(bucket["visits"]), str(bucket["action_sha256"])),
    )


def audit_pivotal_states(
    records: Iterable[tuple[str, int, Mapping[str, object]]],
    *,
    min_visits: int = 2,
    training_min_visits: int = 4,
    training_min_groups: int = 50,
    training_min_tasks: int = 40,
    reward_tolerance: float = 1.0e-8,
    max_groups: int = 200,
) -> dict[str, object]:
    """Find replay-identical decision states with divergent actions and outcomes."""
    if int(min_visits) < 2:
        raise ValueError("min_visits must be at least 2")
    if int(training_min_visits) < int(min_visits):
        raise ValueError("training_min_visits must be at least min_visits")
    if int(training_min_groups) < 1:
        raise ValueError("training_min_groups must be positive")
    if int(training_min_tasks) < 1:
        raise ValueError("training_min_tasks must be positive")
    if reward_tolerance < 0 or not math.isfinite(reward_tolerance):
        raise ValueError("reward_tolerance must be finite and non-negative")
    if int(max_groups) < 1:
        raise ValueError("max_groups must be positive")

    visits_by_state: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    visits_by_branch: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    contract_errors: Counter[str] = Counter()
    source_records = 0
    trajectories = 0
    learning_valid_trajectories = 0
    learning_invalid_trajectories = 0
    learning_validity_unknown_trajectories = 0
    duplicate_semantic_trajectories = 0
    semantic_trajectory_digests: dict[tuple[object, ...], str] = {}
    pivotal_visits = 0
    for source_path, line_number, raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"{source_path}:{line_number} is not an object")
        source_records += 1
        raw_trajectories = raw_record.get("trajectories")
        if not isinstance(raw_trajectories, list):
            raise TypeError(f"{source_path}:{line_number} is missing trajectories")
        for trajectory_index, trajectory in enumerate(raw_trajectories):
            if not isinstance(trajectory, Mapping):
                raise TypeError(
                    f"{source_path}:{line_number} trajectory {trajectory_index} is not an object"
                )
            explicit_scope = raw_record.get("policy_scope_uid")
            if _is_sha256(explicit_scope) and all(
                raw_record.get(key) is not None
                for key in ("global_step", "generation_batch", "uid")
            ):
                semantic_key = (
                    str(explicit_scope),
                    raw_record.get("global_step"),
                    raw_record.get("generation_batch"),
                    str(raw_record.get("uid")),
                    trajectory_index,
                )
                trajectory_digest = hashlib.sha256(
                    _canonical_json(trajectory).encode("utf-8")
                ).hexdigest()
                prior_digest = semantic_trajectory_digests.get(semantic_key)
                if prior_digest is not None:
                    if prior_digest != trajectory_digest:
                        raise ValueError(
                            "semantic trajectory identity has conflicting content: "
                            f"scope={explicit_scope}, step={raw_record.get('global_step')}, "
                            f"batch={raw_record.get('generation_batch')}, "
                            f"uid={raw_record.get('uid')}, trajectory={trajectory_index}"
                        )
                    duplicate_semantic_trajectories += 1
                    continue
                semantic_trajectory_digests[semantic_key] = trajectory_digest
            trajectories += 1
            if trajectory.get("valid_for_learning") is True and not trajectory.get(
                "invalid_reason"
            ):
                learning_valid_trajectories += 1
            elif trajectory.get("valid_for_learning") is False or trajectory.get("invalid_reason"):
                learning_invalid_trajectories += 1
            else:
                learning_validity_unknown_trajectories += 1
            visits = _trajectory_visits(
                raw_record,
                trajectory,
                source_path=source_path,
                line_number=line_number,
                trajectory_index=trajectory_index,
            )
            pivotal_visits += len(visits)
            for visit in visits:
                scope = str(visit["policy_scope_id"])
                state_id = str(visit["replay_state_id"])
                visits_by_state[(scope, state_id)].append(visit)
                if visit["training_contract_complete"]:
                    visits_by_branch[(scope, str(visit["branch_uid"]))].append(visit)
                else:
                    contract_errors[str(visit["training_contract_error"] or "unknown")] += 1

    groups = []
    label_counts: Counter[str] = Counter()
    for (policy_scope_id, state_id), raw_visits in visits_by_state.items():
        # Repeated visits inside one trajectory are removed earlier; identical
        # serialized occurrences from duplicated input files are removed here.
        unique_visits = {str(visit["source"]["occurrence"]): visit for visit in raw_visits}
        visits = list(unique_visits.values())
        if len(visits) < int(min_visits):
            continue
        strict_values = {bool(visit["strict"]) for visit in visits}
        utilities = [float(visit["terminal_utility"]) for visit in visits]
        action_signatures = {str(visit["next_action_sha256"]) for visit in visits}
        labels = sorted({label for visit in visits for label in visit["pivotal_labels"]})
        label_counts.update(labels)
        group = {
            "replay_state_id": state_id,
            "policy_scope_id": policy_scope_id,
            "task_id": int(visits[0]["task_id"]),
            "observation_kind": str(visits[0]["observation_kind"]),
            "prefix_action_count": int(visits[0]["prefix_action_count"]),
            "pivotal_labels": labels,
            "visits": len(visits),
            "unique_next_actions": len(action_signatures),
            "action_divergent": len(action_signatures) > 1,
            "strict_successes": sum(bool(visit["strict"]) for visit in visits),
            "strict_failures": sum(not bool(visit["strict"]) for visit in visits),
            "mixed_strict": len(strict_values) > 1,
            "observationally_heterogeneous_strict": len(strict_values) > 1,
            "utility_min": min(utilities),
            "utility_max": max(utilities),
            "utility_varying": max(utilities) - min(utilities) > reward_tolerance,
            "eligible_mixed_strict": len(strict_values) > 1 and len(action_signatures) > 1,
            "eligible_mixed_utility": (
                max(utilities) - min(utilities) > reward_tolerance and len(action_signatures) > 1
            ),
            "examples": [
                {
                    key: deepcopy(visit[key])
                    for key in (
                        "next_action",
                        "next_action_accepted",
                        "next_action_repeated",
                        "strict",
                        "terminal_utility",
                        "policy_reward",
                        "reward_type",
                        "termination_reason",
                        "source",
                    )
                }
                for visit in visits[:8]
            ],
            "action_outcomes": _action_outcome_table(visits),
        }
        groups.append(group)

    groups.sort(
        key=lambda group: (
            not group["eligible_mixed_strict"],
            not group["eligible_mixed_utility"],
            -int(group["visits"]),
            int(group["task_id"]),
            str(group["replay_state_id"]),
        )
    )
    observational_candidates = [group for group in groups if group["eligible_mixed_strict"]]
    observational_candidates_per_task = Counter(
        int(group["task_id"]) for group in observational_candidates
    )
    observational_candidate_count = len(observational_candidates)
    observational_candidate_task_ess = (
        observational_candidate_count**2
        / sum(count**2 for count in observational_candidates_per_task.values())
        if observational_candidates_per_task
        else 0.0
    )
    exact_contract_states = []
    exact_branch_groups = []
    for (policy_scope_id, exact_uid), raw_visits in visits_by_branch.items():
        unique_visits = {str(visit["source"]["occurrence"]): visit for visit in raw_visits}
        visits = list(unique_visits.values())
        if not visits:
            continue
        exact_contract_states.append(
            {
                "branch_uid": exact_uid,
                "policy_scope_id": policy_scope_id,
                "replay_state_id": str(visits[0]["replay_state_id"]),
                "task_id": int(visits[0]["task_id"]),
                "prefix_action_count": int(visits[0]["prefix_action_count"]),
                "pivotal_labels": sorted(
                    {label for visit in visits for label in visit["pivotal_labels"]}
                ),
                "visits": len(visits),
                "source": deepcopy(visits[0]["source"]),
            }
        )
        if len(visits) < int(min_visits):
            continue
        strict_values = {bool(visit["strict"]) for visit in visits}
        utilities = [float(visit["terminal_utility"]) for visit in visits]
        actions = {str(visit["next_action_sha256"]) for visit in visits}
        exact_branch_groups.append(
            {
                "branch_uid": exact_uid,
                "policy_scope_id": policy_scope_id,
                "replay_state_id": str(visits[0]["replay_state_id"]),
                "task_id": int(visits[0]["task_id"]),
                "visits": len(visits),
                "unique_next_actions": len(actions),
                "mixed_strict": len(strict_values) > 1,
                "observationally_heterogeneous_strict": len(strict_values) > 1,
                "utility_varying": max(utilities) - min(utilities) > reward_tolerance,
                "eligible_mixed_strict": len(strict_values) > 1 and len(actions) > 1,
                "eligible_mixed_utility": (
                    max(utilities) - min(utilities) > reward_tolerance and len(actions) > 1
                ),
                "natural_suffix_group_candidate": (
                    len(visits) >= int(training_min_visits)
                    and len(strict_values) > 1
                    and len(actions) > 1
                ),
                "action_outcomes": _action_outcome_table(visits),
            }
        )
    exact_branch_groups.sort(
        key=lambda group: (
            not group["eligible_mixed_strict"],
            not group["eligible_mixed_utility"],
            -int(group["visits"]),
            int(group["task_id"]),
        )
    )
    exact_contract_states.sort(
        key=lambda state: (
            int(state["task_id"]),
            int(state["prefix_action_count"]),
            str(state["branch_uid"]),
        )
    )
    natural_suffix_group_candidates = [
        group for group in exact_branch_groups if group["natural_suffix_group_candidate"]
    ]
    exact_states_per_task = Counter(int(state["task_id"]) for state in exact_contract_states)
    exact_state_count = len(exact_contract_states)
    exact_state_unique_tasks = len(exact_states_per_task)
    exact_state_task_capped_count = sum(min(2, count) for count in exact_states_per_task.values())
    exact_state_task_cluster_ess = (
        exact_state_count**2 / sum(count**2 for count in exact_states_per_task.values())
        if exact_states_per_task
        else 0.0
    )
    hash_contract_feasible = (
        exact_state_count >= int(training_min_groups)
        and exact_state_unique_tasks >= int(training_min_tasks)
        and exact_state_task_capped_count >= int(training_min_groups)
    )
    aggregate = {
        "source_records": source_records,
        "trajectories": trajectories,
        "learning_valid_trajectories": learning_valid_trajectories,
        "learning_invalid_trajectories": learning_invalid_trajectories,
        "learning_validity_unknown_trajectories": learning_validity_unknown_trajectories,
        "duplicate_semantic_trajectories": duplicate_semantic_trajectories,
        "pivotal_visits": pivotal_visits,
        "unique_pivotal_states": len(visits_by_state),
        "shared_pivotal_states": len(groups),
        "action_divergent_states": sum(group["action_divergent"] for group in groups),
        "mixed_strict_states": sum(group["mixed_strict"] for group in groups),
        "mixed_utility_states": sum(group["utility_varying"] for group in groups),
        "eligible_mixed_strict_states": sum(group["eligible_mixed_strict"] for group in groups),
        "eligible_mixed_utility_states": sum(group["eligible_mixed_utility"] for group in groups),
        "observational_candidate_unique_tasks": len(observational_candidates_per_task),
        "observational_candidate_task_capped_groups": sum(
            min(2, count) for count in observational_candidates_per_task.values()
        ),
        "observational_candidate_task_cluster_ess": observational_candidate_task_ess,
        "exact_prompt_shared_branches": len(exact_branch_groups),
        "exact_prompt_mixed_strict_branches": sum(
            group["eligible_mixed_strict"] for group in exact_branch_groups
        ),
        "exact_prompt_mixed_utility_branches": sum(
            group["eligible_mixed_utility"] for group in exact_branch_groups
        ),
        "exact_contract_pivotal_states": exact_state_count,
        "exact_contract_unique_tasks": exact_state_unique_tasks,
        "exact_contract_task_capped_states": exact_state_task_capped_count,
        "exact_contract_task_cluster_ess": exact_state_task_cluster_ess,
        "natural_suffix_group_candidates": len(natural_suffix_group_candidates),
        "natural_group_min_visits": int(training_min_visits),
        "candidate_min_pivotal_states": int(training_min_groups),
        "candidate_min_unique_tasks": int(training_min_tasks),
        "training_contract_error_counts": dict(sorted(contract_errors.items())),
        "pivotal_label_counts": dict(sorted(label_counts.items())),
    }
    return {
        "schema_version": PIVOTAL_AUDIT_VERSION,
        "replay_state_version": REPLAY_STATE_VERSION,
        "turn_span_version": TURN_SPAN_VERSION,
        "aggregate": aggregate,
        "groups": groups[: int(max_groups)],
        "groups_truncated": max(0, len(groups) - int(max_groups)),
        "exact_prompt_branch_groups": exact_branch_groups[: int(max_groups)],
        "exact_prompt_branch_groups_truncated": max(0, len(exact_branch_groups) - int(max_groups)),
        "exact_contract_states": exact_contract_states[: int(max_groups)],
        "exact_contract_states_truncated": max(0, len(exact_contract_states) - int(max_groups)),
        "safety": {
            "uses_hidden_goal": False,
            "fingerprint_inputs": [
                "task_id",
                "environment_manifest_sha256",
                "public_query_sha256",
                "accepted_public_action_prefix",
                "public_observation_sha256",
            ],
            "branch_uid_additional_inputs": [
                "exact_actor_prompt_token_sha256",
                "tokenizer_contract_sha256",
            ],
            "hash_contract_feasible": hash_contract_feasible,
            "live_replay_verified": False,
            "prompt_tokens_materialized": False,
            "active_branching_ready": False,
            "training_ready": False,
            "training_blockers": [
                "historical outcomes are observational, not controlled suffix branches",
                "live reset-and-replay verification is not attached to these groups",
                "sampling audits do not contain prompt/response tensors for training",
            ],
            "training_started": False,
        },
    }


def render_pivotal_audit_markdown(audit: Mapping[str, object]) -> str:
    aggregate = audit["aggregate"]
    lines = [
        "# Pivotal-State Offline Audit",
        "",
        f"- Schema: `{audit['schema_version']}`",
        f"- Source records: `{aggregate['source_records']}`",
        f"- Trajectories: `{aggregate['trajectories']}`",
        f"- Pivotal visits: `{aggregate['pivotal_visits']}`",
        f"- Shared pivotal states: `{aggregate['shared_pivotal_states']}`",
        f"- Action-divergent states: `{aggregate['action_divergent_states']}`",
        f"- Observational mixed-strict states: `{aggregate['eligible_mixed_strict_states']}`",
        f"- Observational mixed-utility states: `{aggregate['eligible_mixed_utility_states']}`",
        f"- Exact-prompt mixed-strict branches: `{aggregate['exact_prompt_mixed_strict_branches']}`",
        f"- Exact-contract pivotal states: `{aggregate['exact_contract_pivotal_states']}`",
        f"- Exact-contract unique tasks: `{aggregate['exact_contract_unique_tasks']}`",
        f"- Natural mixed-suffix groups: `{aggregate['natural_suffix_group_candidates']}`",
        "- Hidden goal used: `false`",
        "- Training started: `false`",
        "",
        "## Decision Gate",
        "",
    ]
    replay_states = int(aggregate["eligible_mixed_strict_states"])
    exact_states = int(aggregate["exact_contract_pivotal_states"])
    required_states = int(aggregate["candidate_min_pivotal_states"])
    required_tasks = int(aggregate["candidate_min_unique_tasks"])
    unique_tasks = int(aggregate["exact_contract_unique_tasks"])
    lines.append(f"- Observational replay-state candidates: `{replay_states}`.")
    lines.append(
        f"- HASH PASS: at least {required_states} exact-contract pivotal states "
        f"across at least {required_tasks} tasks."
        if audit["safety"]["hash_contract_feasible"]
        else f"- HASH FAIL: need {required_states} exact-contract pivotal states "
        f"across {required_tasks} tasks; observed {exact_states} states across "
        f"{unique_tasks} tasks."
    )
    lines.append(
        "- TRAINING BLOCKED: historical suffixes are observational, live replay is not "
        "attached, and sampling-audit JSON does not contain trainable token tensors."
    )
    lines.extend(
        [
            "",
            "## Representative States",
            "",
            "| task | labels | visits | actions | strict S/F | utility range |",
            "| ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for group in audit.get("groups", [])[:30]:
        lines.append(
            "| {task_id} | {labels} | {visits} | {actions} | {successes}/{failures} | "
            "{minimum:.3f}..{maximum:.3f} |".format(
                task_id=group["task_id"],
                labels=", ".join(group["pivotal_labels"]),
                visits=group["visits"],
                actions=group["unique_next_actions"],
                successes=group["strict_successes"],
                failures=group["strict_failures"],
                minimum=group["utility_min"],
                maximum=group["utility_max"],
            )
        )
    lines.extend(
        [
            "",
            "These mixed outcomes are observational heterogeneity, not proof that the first action caused the outcome.",
            "The fingerprint contains only the task id, accepted public tool calls, and a public observation hash.",
            "It does not contain the hidden goal, target ASIN, or correct option values.",
            "",
        ]
    )
    return "\n".join(lines)
