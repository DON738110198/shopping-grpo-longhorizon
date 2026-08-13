"""Outcome-blind selection of exact public pivotal replay states."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence

from shopping_grpo.training.grpo.pivotal_states import (
    ACTOR_PROMPT_TOKENS_VERSION,
    canonical_replay_action,
    pivotal_labels,
    token_ids_sha256,
    validate_event_training_contract,
)

PIVOTAL_SELECTION_VERSION = "shopping-pivotal-state-selection-v1"
PIVOTAL_SELECTION_STRATEGY = "seeded-task-round-robin-v1"
SEARCH_DECISION_SELECTION_STRATEGY = "seeded-search-decision-stratified-v2"
SEARCH_DECISION_LABELS = (
    "search_query_decision",
    "search_result_open_decision",
)
MAX_STATES_PER_TASK = 2
SEARCH_MAX_STATES_PER_TASK = 1
PIVOTAL_SELECTION_FIELDS = frozenset(
    {
        "selection_index",
        "task_id",
        "replay_state_id",
        "branch_uid",
        "prefix_action_count",
        "pivotal_labels",
        "source",
    }
)
PIVOTAL_SELECTION_SOURCE_FIELDS = frozenset(
    {
        "input_index",
        "input_sha256",
        "path",
        "line",
        "global_step",
        "generation_batch",
        "uid",
        "trajectory_index",
        "event_index",
    }
)

_CONTRACT_TRAJECTORY_FIELDS = (
    "task_id",
    "replay_observation_v2_complete",
    "replay_contract_error",
    "replay_state_version",
    "turn_span_version",
    "turn_span_valid",
    "environment_manifest_sha256",
    "environment_version",
    "public_query_sha256",
    "initial_public_observation_sha256",
    "replay_ledger",
    "turn_spans",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _validate_actor_prompt_capture(event: Mapping[str, object]) -> str | None:
    capture = event.get("actor_prompt_tokens")
    if not isinstance(capture, Mapping):
        return "missing_actor_prompt_tokens"
    if set(capture) != {"version", "sha256", "count", "tokens"}:
        return "actor_prompt_token_fields_mismatch"
    if capture.get("version") != ACTOR_PROMPT_TOKENS_VERSION:
        return "actor_prompt_token_version_mismatch"
    tokens = capture.get("tokens")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in tokens
        )
    ):
        return "invalid_actor_prompt_tokens"
    count = capture.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count != len(tokens):
        return "actor_prompt_token_count_mismatch"
    captured_sha256 = capture.get("sha256")
    if not _is_sha256(captured_sha256):
        return "invalid_actor_prompt_token_hash"
    actual_sha256 = token_ids_sha256(tokens)
    if actual_sha256 != captured_sha256 or actual_sha256 != event.get("actor_prompt_sha256"):
        return "actor_prompt_token_hash_mismatch"
    return None


def _seeded_rank(seed: int, namespace: str, value: object) -> tuple[str, str]:
    payload = _canonical_json(
        {
            "seed": int(seed),
            "namespace": str(namespace),
            "value": value,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), str(value)


def _contract_view(trajectory: Mapping[str, object]) -> dict[str, object]:
    """Copy only structural fields, making outcome access impossible downstream."""
    view = {name: trajectory.get(name) for name in _CONTRACT_TRAJECTORY_FIELDS}
    # The shared validator also gates learning outcomes. Selection deliberately
    # substitutes constants so reward/outcome metadata cannot influence sampling.
    view["valid_for_learning"] = True
    view["invalid_reason"] = None
    return view


def _candidate_at(
    record: Mapping[str, object],
    trajectory: Mapping[str, object],
    *,
    input_index: int,
    input_sha256: str,
    source_path: str,
    line_number: int,
    trajectory_index: int,
    event_index: int,
    require_prompt_capture: bool,
    strategy: str = PIVOTAL_SELECTION_STRATEGY,
) -> tuple[dict[str, object] | None, str | None]:
    events = trajectory.get("decision_trace")
    if not isinstance(events, list):
        return None, "missing_decision_trace"
    if not 0 <= event_index < len(events):
        return None, "event_index_out_of_range"
    raw_event = events[event_index]
    if not isinstance(raw_event, Mapping):
        return None, "event_not_object"
    event = raw_event
    contract_trajectory = _contract_view(trajectory)
    valid, reason = validate_event_training_contract(contract_trajectory, event)
    if not valid:
        return None, str(reason or "invalid_event_contract")
    task_id = contract_trajectory.get("task_id")
    if not isinstance(task_id, int) or isinstance(task_id, bool):
        return None, "invalid_task_id"
    prefix_count = int(event["prefix_action_count"])
    ledger = contract_trajectory["replay_ledger"]
    if not isinstance(ledger, list):  # Covered by the validator; retained for typing.
        return None, "missing_replay_ledger"
    accepted_prefix = [
        canonical_replay_action(item.get("tool"), item.get("parameters"))
        for item in ledger[:prefix_count]
        if isinstance(item, Mapping)
    ]
    previous_event = events[event_index - 1] if event_index else None
    if not isinstance(previous_event, Mapping):
        previous_event = None
    if strategy == PIVOTAL_SELECTION_STRATEGY:
        labels = pivotal_labels(accepted_prefix, event, previous_event)
    elif strategy == SEARCH_DECISION_SELECTION_STRATEGY:
        tool = str(event.get("tool") or "")
        if event.get("accepted") is not True or event.get("error"):
            labels = ()
        elif tool == "search_products":
            labels = ("search_query_decision",)
        elif tool == "open_product":
            labels = ("search_result_open_decision",)
        else:
            labels = ()
    else:  # pragma: no cover - public entry points validate this first.
        raise ValueError("unknown pivotal selection strategy")
    if not labels:
        return None, "not_pivotal"
    if require_prompt_capture:
        capture_error = _validate_actor_prompt_capture(event)
        if capture_error is not None:
            return None, capture_error
    return {
        "task_id": task_id,
        "replay_state_id": str(event["replay_state_id"]),
        "branch_uid": str(event["branch_uid"]),
        "prefix_action_count": prefix_count,
        "pivotal_labels": sorted(set(labels)),
        "source": {
            "input_index": int(input_index),
            "input_sha256": str(input_sha256),
            "path": str(source_path),
            "line": int(line_number),
            "global_step": record.get("global_step"),
            "generation_batch": record.get("generation_batch"),
            "uid": str(record.get("uid") or ""),
            "trajectory_index": int(trajectory_index),
            "event_index": int(event_index),
        },
    }, None


def _source_rank(candidate: Mapping[str, object]) -> tuple[object, ...]:
    source = candidate["source"]
    if not isinstance(source, Mapping):  # pragma: no cover - internal construction.
        raise TypeError("candidate source must be an object")
    return (
        str(source["path"]),
        int(source["line"]),
        int(source["trajectory_index"]),
        int(source["event_index"]),
    )


def _validate_provenance(provenance: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    if not provenance:
        raise ValueError("selection provenance must contain at least one input")
    normalized = []
    seen_hashes = set()
    seen_paths = set()
    for index, item in enumerate(provenance):
        path = item.get("path")
        digest = item.get("sha256")
        if not isinstance(path, str) or not path:
            raise ValueError(f"selection provenance input {index} has an invalid path")
        if not _is_sha256(digest):
            raise ValueError(f"selection provenance input {index} has an invalid sha256")
        if digest in seen_hashes:
            raise ValueError("duplicate sampling-audit content is not allowed")
        if path in seen_paths:
            raise ValueError("duplicate sampling-audit paths are not allowed")
        seen_hashes.add(digest)
        seen_paths.add(path)
        normalized.append({"path": path, "sha256": str(digest)})
    return normalized


def select_pivotal_states(
    records: Iterable[tuple[int, str, int, Mapping[str, object]]],
    *,
    seed: int,
    max_states: int,
    provenance: Sequence[Mapping[str, object]],
    require_prompt_capture: bool = False,
    strategy: str = PIVOTAL_SELECTION_STRATEGY,
    label_quotas: Mapping[str, int] | None = None,
    excluded_task_ids: Sequence[int] = (),
) -> dict[str, object]:
    """Select exact pivotal branches without consulting reward or outcome fields.

    A bare ``branch_uid`` is deduplicated across every source record and global
    step. The seeded task round-robin first maximizes task coverage, then admits
    at most a second state from each task.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not isinstance(max_states, int) or isinstance(max_states, bool):
        raise TypeError("max_states must be an integer")
    if max_states < 1:
        raise ValueError("max_states must be a positive integer")
    if not isinstance(require_prompt_capture, bool):
        raise TypeError("require_prompt_capture must be boolean")
    if strategy not in {
        PIVOTAL_SELECTION_STRATEGY,
        SEARCH_DECISION_SELECTION_STRATEGY,
    }:
        raise ValueError("unknown pivotal selection strategy")
    if any(not isinstance(task_id, int) or isinstance(task_id, bool) for task_id in excluded_task_ids):
        raise TypeError("excluded task ids must be integers")
    normalized_excluded_task_ids = sorted(set(excluded_task_ids))
    if any(task_id < 0 for task_id in normalized_excluded_task_ids):
        raise ValueError("excluded task ids must be non-negative")
    normalized_label_quotas: dict[str, int] | None = None
    if strategy == PIVOTAL_SELECTION_STRATEGY:
        if label_quotas is not None or normalized_excluded_task_ids:
            raise ValueError("generic pivotal selection does not accept search constraints")
        max_states_per_task = MAX_STATES_PER_TASK
    else:
        if require_prompt_capture is not True:
            raise ValueError("search-decision selection requires exact prompt capture")
        if not isinstance(label_quotas, Mapping) or set(label_quotas) != set(
            SEARCH_DECISION_LABELS
        ):
            raise ValueError("search-decision selection requires exact label quotas")
        normalized_label_quotas = {}
        for label in SEARCH_DECISION_LABELS:
            quota = label_quotas.get(label)
            if not isinstance(quota, int) or isinstance(quota, bool) or quota < 0:
                raise ValueError("search-decision label quotas must be non-negative integers")
            normalized_label_quotas[label] = quota
        if sum(normalized_label_quotas.values()) != max_states:
            raise ValueError("search-decision label quotas must sum to max_states")
        max_states_per_task = SEARCH_MAX_STATES_PER_TASK
    excluded_task_set = set(normalized_excluded_task_ids)
    normalized_provenance = _validate_provenance(provenance)
    unique_candidates: dict[str, dict[str, object]] = {}
    source_records = 0
    source_trajectories = 0
    contract_valid_events = 0
    pivotal_occurrences = 0
    duplicate_occurrences = 0
    contract_errors: Counter[str] = Counter()

    for input_index, source_path, line_number, raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"{source_path}:{line_number} is not an object")
        if not 0 <= int(input_index) < len(normalized_provenance):
            raise ValueError(f"{source_path}:{line_number} has an invalid input index")
        if normalized_provenance[int(input_index)]["path"] != str(source_path):
            raise ValueError(f"{source_path}:{line_number} does not match provenance")
        source_records += 1
        trajectories = raw_record.get("trajectories")
        if not isinstance(trajectories, list):
            raise TypeError(f"{source_path}:{line_number} is missing trajectories")
        for trajectory_index, raw_trajectory in enumerate(trajectories):
            if not isinstance(raw_trajectory, Mapping):
                raise TypeError(
                    f"{source_path}:{line_number} trajectory {trajectory_index} is not an object"
                )
            source_trajectories += 1
            events = raw_trajectory.get("decision_trace")
            if not isinstance(events, list):
                contract_errors["missing_decision_trace"] += 1
                continue
            for event_index in range(len(events)):
                candidate, reason = _candidate_at(
                    raw_record,
                    raw_trajectory,
                    input_index=int(input_index),
                    input_sha256=normalized_provenance[int(input_index)]["sha256"],
                    source_path=str(source_path),
                    line_number=int(line_number),
                    trajectory_index=trajectory_index,
                    event_index=event_index,
                    require_prompt_capture=require_prompt_capture,
                    strategy=strategy,
                )
                if candidate is None:
                    if reason != "not_pivotal":
                        contract_errors[str(reason)] += 1
                    continue
                if candidate["task_id"] in excluded_task_set:
                    continue
                contract_valid_events += 1
                pivotal_occurrences += 1
                branch = str(candidate["branch_uid"])
                previous = unique_candidates.get(branch)
                if previous is None:
                    unique_candidates[branch] = candidate
                    continue
                duplicate_occurrences += 1
                identity = ("task_id", "replay_state_id", "prefix_action_count")
                if any(previous[name] != candidate[name] for name in identity):
                    raise ValueError(f"branch_uid collision for {branch}")
                if _source_rank(candidate) < _source_rank(previous):
                    unique_candidates[branch] = candidate

    selected: list[dict[str, object]] = []
    if strategy == PIVOTAL_SELECTION_STRATEGY:
        by_task: dict[int, list[dict[str, object]]] = defaultdict(list)
        for candidate in unique_candidates.values():
            candidate_task_id = candidate["task_id"]
            if not isinstance(candidate_task_id, int) or isinstance(candidate_task_id, bool):
                raise TypeError("candidate task_id is not an integer")
            by_task[candidate_task_id].append(candidate)
        for task_id, candidates in by_task.items():
            candidates.sort(
                key=lambda item: _seeded_rank(
                    seed, f"task:{task_id}:branch", item["branch_uid"]
                )
            )
        task_order = sorted(
            by_task,
            key=lambda task_id: _seeded_rank(seed, "task", task_id),
        )
        for task_rank in range(MAX_STATES_PER_TASK):
            for task_id in task_order:
                candidates = by_task[task_id]
                if task_rank >= len(candidates):
                    continue
                item = dict(candidates[task_rank])
                item["selection_index"] = len(selected)
                selected.append(item)
                if len(selected) >= max_states:
                    break
            if len(selected) >= max_states:
                break
    else:
        if normalized_label_quotas is None:  # pragma: no cover - validated above.
            raise AssertionError("search label quotas were not normalized")
        selected_tasks: set[int] = set()
        for label in SEARCH_DECISION_LABELS:
            quota = normalized_label_quotas[label]
            if quota == 0:
                continue
            by_task = defaultdict(list)
            for candidate in unique_candidates.values():
                if candidate["pivotal_labels"] != [label]:
                    continue
                by_task[int(candidate["task_id"])].append(candidate)
            task_order = sorted(
                by_task,
                key=lambda task_id: _seeded_rank(seed, f"search-label:{label}:task", task_id),
            )
            selected_for_label = 0
            for task_id in task_order:
                if task_id in selected_tasks:
                    continue
                candidates = sorted(
                    by_task[task_id],
                    key=lambda item: _seeded_rank(
                        seed,
                        f"search-label:{label}:task:{task_id}:branch",
                        item["branch_uid"],
                    ),
                )
                item = dict(candidates[0])
                item["selection_index"] = len(selected)
                selected.append(item)
                selected_tasks.add(task_id)
                selected_for_label += 1
                if selected_for_label >= quota:
                    break
            if selected_for_label != quota:
                raise ValueError(
                    "cannot fill search decision quota "
                    f"for {label}: requested {quota}, found {selected_for_label}"
                )

    return {
        "schema_version": PIVOTAL_SELECTION_VERSION,
        "strategy_version": strategy,
        "seed": seed,
        "constraints": {
            "max_states": max_states,
            "max_states_per_task": max_states_per_task,
            "require_prompt_capture": require_prompt_capture,
            **(
                {
                    "label_quotas": normalized_label_quotas,
                    "excluded_task_ids": normalized_excluded_task_ids,
                    "excluded_task_ids_sha256": hashlib.sha256(
                        _canonical_json(normalized_excluded_task_ids).encode("utf-8")
                    ).hexdigest(),
                }
                if strategy == SEARCH_DECISION_SELECTION_STRATEGY
                else {}
            ),
        },
        "provenance": {"inputs": normalized_provenance},
        "aggregate": {
            "source_records": source_records,
            "source_trajectories": source_trajectories,
            "contract_valid_pivotal_occurrences": contract_valid_events,
            "pivotal_candidate_occurrences": pivotal_occurrences,
            "unique_branch_candidates": len(unique_candidates),
            "duplicate_branch_occurrences": duplicate_occurrences,
            "selected_branches": len(selected),
            "selected_tasks": len(
                {_required_integer(item["task_id"], "selected task_id") for item in selected}
            ),
            "contract_error_counts": dict(sorted(contract_errors.items())),
        },
        "safety": {
            "outcome_blind": True,
            "outcome_fields_read": [],
            "uses_hidden_goal": False,
        },
        "selections": selected,
    }


_TOP_LEVEL_FIELDS = {
    "schema_version",
    "strategy_version",
    "seed",
    "constraints",
    "provenance",
    "aggregate",
    "safety",
    "selections",
}
_CONSTRAINT_FIELDS = {"max_states", "max_states_per_task", "require_prompt_capture"}
_SEARCH_CONSTRAINT_FIELDS = _CONSTRAINT_FIELDS | {
    "label_quotas",
    "excluded_task_ids",
    "excluded_task_ids_sha256",
}
_PROVENANCE_FIELDS = {"inputs"}
_INPUT_PROVENANCE_FIELDS = {"path", "sha256"}
_SAFETY_FIELDS = {"outcome_blind", "outcome_fields_read", "uses_hidden_goal"}
_AGGREGATE_FIELDS = {
    "source_records",
    "source_trajectories",
    "contract_valid_pivotal_occurrences",
    "pivotal_candidate_occurrences",
    "unique_branch_candidates",
    "duplicate_branch_occurrences",
    "selected_branches",
    "selected_tasks",
    "contract_error_counts",
}


def _required_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_exact_fields(
    value: Mapping[str, object],
    expected: set[str],
    name: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} has unexpected or missing fields")


def validate_pivotal_selection(
    selection: Mapping[str, object],
    *,
    expected_inputs: Sequence[Mapping[str, object]] | None = None,
    require_prompt_capture: bool | None = None,
) -> list[dict[str, object]]:
    """Validate the complete selector artifact before any environment lease."""
    if not isinstance(selection, Mapping):
        raise TypeError("selection must be an object")
    _require_exact_fields(selection, _TOP_LEVEL_FIELDS, "selection top-level fields")
    if selection.get("schema_version") != PIVOTAL_SELECTION_VERSION:
        raise ValueError("selection schema_version mismatch")
    strategy = selection.get("strategy_version")
    if strategy not in {
        PIVOTAL_SELECTION_STRATEGY,
        SEARCH_DECISION_SELECTION_STRATEGY,
    }:
        raise ValueError("selection strategy_version mismatch")
    _required_integer(selection.get("seed"), "selection seed")
    constraints = selection.get("constraints")
    if not isinstance(constraints, Mapping):
        raise TypeError("selection constraints must be an object")
    expected_constraint_fields = (
        _SEARCH_CONSTRAINT_FIELDS
        if strategy == SEARCH_DECISION_SELECTION_STRATEGY
        else _CONSTRAINT_FIELDS
    )
    _require_exact_fields(
        constraints,
        expected_constraint_fields,
        "selection constraints",
    )
    max_states = _required_integer(
        constraints.get("max_states"),
        "selection max_states",
        minimum=1,
    )
    expected_task_cap = (
        SEARCH_MAX_STATES_PER_TASK
        if strategy == SEARCH_DECISION_SELECTION_STRATEGY
        else MAX_STATES_PER_TASK
    )
    if constraints.get("max_states_per_task") != expected_task_cap:
        raise ValueError("selection max_states_per_task mismatch")
    capture_required = constraints.get("require_prompt_capture")
    if not isinstance(capture_required, bool):
        raise TypeError("selection require_prompt_capture must be boolean")
    if require_prompt_capture is not None:
        if not isinstance(require_prompt_capture, bool):
            raise TypeError("require_prompt_capture must be boolean")
        if capture_required is not require_prompt_capture:
            raise ValueError("selection require_prompt_capture mismatch")
    if strategy == SEARCH_DECISION_SELECTION_STRATEGY:
        if capture_required is not True:
            raise ValueError("search-decision selection requires exact prompt capture")
        quotas = constraints.get("label_quotas")
        if not isinstance(quotas, Mapping) or set(quotas) != set(SEARCH_DECISION_LABELS):
            raise ValueError("selection search label quotas mismatch")
        for label in SEARCH_DECISION_LABELS:
            _required_integer(quotas.get(label), f"selection quota {label}")
        if sum(int(quotas[label]) for label in SEARCH_DECISION_LABELS) != max_states:
            raise ValueError("selection search label quotas do not sum to max_states")
        excluded_task_ids = constraints.get("excluded_task_ids")
        if (
            not isinstance(excluded_task_ids, list)
            or any(
                not isinstance(task_id, int)
                or isinstance(task_id, bool)
                or task_id < 0
                for task_id in excluded_task_ids
            )
            or excluded_task_ids != sorted(set(excluded_task_ids))
        ):
            raise ValueError("selection excluded task ids are invalid")
        expected_excluded_sha256 = hashlib.sha256(
            _canonical_json(excluded_task_ids).encode("utf-8")
        ).hexdigest()
        if constraints.get("excluded_task_ids_sha256") != expected_excluded_sha256:
            raise ValueError("selection excluded task id hash mismatch")
    provenance = selection.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError("selection provenance must be an object")
    _require_exact_fields(provenance, _PROVENANCE_FIELDS, "selection provenance")
    raw_inputs = provenance.get("inputs")
    if not isinstance(raw_inputs, list):
        raise TypeError("selection provenance inputs must be a list")
    for input_index, raw_input in enumerate(raw_inputs):
        if not isinstance(raw_input, Mapping):
            raise TypeError(f"selection provenance input {input_index} must be an object")
        _require_exact_fields(
            raw_input,
            _INPUT_PROVENANCE_FIELDS,
            f"selection provenance input {input_index}",
        )
    inputs = _validate_provenance(raw_inputs)
    if expected_inputs is not None:
        normalized_expected = _validate_provenance(expected_inputs)
        if inputs != normalized_expected:
            raise ValueError("selection input provenance mismatch")
    safety = selection.get("safety")
    if not isinstance(safety, Mapping):
        raise TypeError("selection safety must be an object")
    _require_exact_fields(safety, _SAFETY_FIELDS, "selection safety")
    if safety.get("outcome_blind") is not True:
        raise ValueError("selection is not outcome-blind")
    if safety.get("outcome_fields_read") != [] or safety.get("uses_hidden_goal") is not False:
        raise ValueError("selection safety contract mismatch")
    raw_selections = selection.get("selections")
    if not isinstance(raw_selections, list):
        raise TypeError("selections must be a list")
    if len(raw_selections) > max_states:
        raise ValueError("selection exceeds max_states")

    validated = []
    seen_branches = set()
    task_counts: Counter[int] = Counter()
    for expected_index, raw_item in enumerate(raw_selections):
        if not isinstance(raw_item, Mapping):
            raise TypeError(f"selection {expected_index} must be an object")
        unexpected = set(raw_item) - PIVOTAL_SELECTION_FIELDS
        missing = PIVOTAL_SELECTION_FIELDS - set(raw_item)
        if unexpected or missing:
            raise ValueError(f"selection {expected_index} has unexpected fields or missing fields")
        if raw_item.get("selection_index") != expected_index:
            raise ValueError(f"selection_index mismatch at {expected_index}")
        task_id = _required_integer(raw_item.get("task_id"), "selection task_id")
        prefix_count = _required_integer(
            raw_item.get("prefix_action_count"),
            "selection prefix_action_count",
        )
        state_id = raw_item.get("replay_state_id")
        branch = raw_item.get("branch_uid")
        if not _is_sha256(state_id):
            raise ValueError(f"selection {expected_index} has an invalid replay_state_id")
        if not _is_sha256(branch):
            raise ValueError(f"selection {expected_index} has an invalid branch_uid")
        if branch in seen_branches:
            raise ValueError(f"selection {expected_index} repeats branch_uid")
        seen_branches.add(branch)
        labels = raw_item.get("pivotal_labels")
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(label, str) or not label for label in labels)
            or labels != sorted(set(labels))
        ):
            raise ValueError(f"selection {expected_index} has invalid pivotal_labels")
        if strategy == SEARCH_DECISION_SELECTION_STRATEGY and labels not in [
            [label] for label in SEARCH_DECISION_LABELS
        ]:
            raise ValueError(f"selection {expected_index} is not a search decision")
        source = raw_item.get("source")
        if not isinstance(source, Mapping):
            raise TypeError(f"selection {expected_index} source must be an object")
        unexpected_source = set(source) - PIVOTAL_SELECTION_SOURCE_FIELDS
        missing_source = PIVOTAL_SELECTION_SOURCE_FIELDS - set(source)
        if unexpected_source or missing_source:
            raise ValueError(f"selection {expected_index} source has invalid fields")
        input_index = _required_integer(
            source.get("input_index"),
            "selection source input_index",
        )
        if input_index >= len(inputs):
            raise ValueError(f"selection {expected_index} source input_index is out of range")
        if source.get("path") != inputs[input_index]["path"]:
            raise ValueError(f"selection {expected_index} source path mismatch")
        if source.get("input_sha256") != inputs[input_index]["sha256"]:
            raise ValueError(f"selection {expected_index} source sha256 mismatch")
        _required_integer(source.get("line"), "selection source line", minimum=1)
        _required_integer(
            source.get("trajectory_index"),
            "selection source trajectory_index",
        )
        _required_integer(source.get("event_index"), "selection source event_index")
        for name in ("global_step", "generation_batch"):
            value = source.get(name)
            if value is not None:
                _required_integer(value, f"selection source {name}")
        if not isinstance(source.get("uid"), str):
            raise TypeError(f"selection {expected_index} source uid must be a string")
        task_counts[task_id] += 1
        if task_counts[task_id] > expected_task_cap:
            raise ValueError(f"selection exceeds task cap for task {task_id}")
        validated.append(
            {
                "selection_index": expected_index,
                "task_id": task_id,
                "replay_state_id": str(state_id),
                "branch_uid": str(branch),
                "prefix_action_count": prefix_count,
                "pivotal_labels": list(labels),
                "source": dict(source),
            }
        )

    aggregate = selection.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise TypeError("selection aggregate must be an object")
    _require_exact_fields(aggregate, _AGGREGATE_FIELDS, "selection aggregate")
    for name in _AGGREGATE_FIELDS - {"contract_error_counts"}:
        _required_integer(aggregate.get(name), f"selection aggregate {name}")
    contract_error_counts = aggregate.get("contract_error_counts")
    if not isinstance(contract_error_counts, Mapping) or any(
        not isinstance(reason, str)
        or not reason
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        for reason, count in contract_error_counts.items()
    ):
        raise ValueError("selection aggregate contract_error_counts is invalid")
    if aggregate.get("selected_branches") != len(validated):
        raise ValueError("selection aggregate selected_branches mismatch")
    if aggregate.get("selected_tasks") != len(task_counts):
        raise ValueError("selection aggregate selected_tasks mismatch")
    occurrences = int(aggregate["pivotal_candidate_occurrences"])
    if aggregate["contract_valid_pivotal_occurrences"] != occurrences:
        raise ValueError("selection aggregate pivotal occurrence mismatch")
    if aggregate["unique_branch_candidates"] + aggregate["duplicate_branch_occurrences"] != (
        occurrences
    ):
        raise ValueError("selection aggregate branch deduplication mismatch")
    if len(validated) > aggregate["unique_branch_candidates"]:
        raise ValueError("selection aggregate has fewer candidates than selections")
    if strategy == SEARCH_DECISION_SELECTION_STRATEGY:
        quotas = constraints["label_quotas"]
        selected_label_counts = Counter(item["pivotal_labels"][0] for item in validated)
        if any(selected_label_counts[label] > quotas[label] for label in SEARCH_DECISION_LABELS):
            raise ValueError("selection exceeds a search decision label quota")
    return validated


def resolve_pivotal_selection(
    selection: Mapping[str, object],
    records: Iterable[tuple[int, str, int, Mapping[str, object]]],
    *,
    expected_inputs: Sequence[Mapping[str, object]],
    require_prompt_capture: bool | None = None,
) -> list[dict[str, object]]:
    """Resolve every selected locator exactly, rejecting the whole batch on drift."""
    selected = validate_pivotal_selection(
        selection,
        expected_inputs=expected_inputs,
        require_prompt_capture=require_prompt_capture,
    )
    if not selected:
        raise ValueError("selection contains no branches")
    record_list = list(records)
    constraints = selection["constraints"]
    if not isinstance(constraints, Mapping):  # pragma: no cover - validated above.
        raise TypeError("validated selection constraints are not an object")
    strategy = str(selection.get("strategy_version") or "")
    recomputed = select_pivotal_states(
        record_list,
        seed=_required_integer(selection.get("seed"), "selection seed"),
        max_states=_required_integer(
            constraints.get("max_states"),
            "selection max_states",
            minimum=1,
        ),
        provenance=expected_inputs,
        require_prompt_capture=bool(constraints["require_prompt_capture"]),
        strategy=strategy,
        label_quotas=(
            constraints.get("label_quotas")
            if strategy == SEARCH_DECISION_SELECTION_STRATEGY
            else None
        ),
        excluded_task_ids=(
            constraints.get("excluded_task_ids", [])
            if strategy == SEARCH_DECISION_SELECTION_STRATEGY
            else ()
        ),
    )
    if _canonical_json(dict(selection)) != _canonical_json(recomputed):
        raise ValueError("selection does not match the deterministic outcome-blind selector")
    records_by_locator: dict[tuple[int, int], tuple[str, Mapping[str, object]]] = {}
    for input_index, source_path, line_number, record in record_list:
        key = (int(input_index), int(line_number))
        if key in records_by_locator:
            raise ValueError(f"duplicate source record locator {key}")
        records_by_locator[key] = (str(source_path), record)

    resolved: list[dict[str, object]] = []
    for item in selected:
        source = item["source"]
        if not isinstance(source, Mapping):  # pragma: no cover - validated above.
            raise TypeError("validated selection source is not an object")
        locator = (int(source["input_index"]), int(source["line"]))
        located = records_by_locator.get(locator)
        if located is None:
            raise ValueError(f"selected source record is missing at {locator}")
        source_path, record = located
        if source_path != source["path"]:
            raise ValueError(f"selected source path drift at {locator}")
        if not isinstance(record, Mapping):
            raise TypeError(f"selected source record is not an object at {locator}")
        trajectories = record.get("trajectories")
        trajectory_index = int(source["trajectory_index"])
        if not isinstance(trajectories, list) or trajectory_index >= len(trajectories):
            raise ValueError(f"selected trajectory is missing at {locator}")
        trajectory = trajectories[trajectory_index]
        if not isinstance(trajectory, Mapping):
            raise TypeError(f"selected trajectory is not an object at {locator}")
        candidate, reason = _candidate_at(
            record,
            trajectory,
            input_index=int(source["input_index"]),
            input_sha256=str(source["input_sha256"]),
            source_path=source_path,
            line_number=int(source["line"]),
            trajectory_index=trajectory_index,
            event_index=int(source["event_index"]),
            require_prompt_capture=bool(constraints["require_prompt_capture"]),
            strategy=strategy,
        )
        if candidate is None:
            raise ValueError(f"selected event rejected at {locator}: {reason}")
        comparable_fields = (
            "task_id",
            "replay_state_id",
            "branch_uid",
            "prefix_action_count",
            "pivotal_labels",
            "source",
        )
        if any(candidate[name] != item[name] for name in comparable_fields):
            raise ValueError(f"selected event identity mismatch at {locator}")
        events = trajectory.get("decision_trace")
        if not isinstance(events, list):  # pragma: no cover - _candidate_at validated it.
            raise TypeError("resolved trajectory lost decision_trace")
        event = events[int(source["event_index"])]
        if not isinstance(event, Mapping):  # pragma: no cover - _candidate_at validated it.
            raise TypeError("resolved event is not an object")
        resolved.append(
            {
                "selection": item,
                "trajectory": trajectory,
                "event": event,
            }
        )
    if len(resolved) != len(selected):  # pragma: no cover - loop is exhaustive.
        raise AssertionError("selection resolution lost branches")
    return resolved
