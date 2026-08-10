"""Build targeted SFT suffixes from mixed-success GRPO rollout groups.

Only clean Reward v3 gold rollouts become demonstrations. Failed siblings from
the same prompt decide which late assistant turns receive labels; hidden goals
and raw reward payloads are never copied into the SFT rows.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any

from shopping_grpo.environment.client import ShopAgentEnv
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS, tool_call_to_action

RECOVERY_SFT_SCHEMA_VERSION = "shopping-recovery-sft-v1"
_TOOL_RESPONSE = re.compile(
    r"user\n<tool_response>\n(.*?)\n</tool_response>\nassistant\n",
    flags=re.DOTALL,
)
_BUCKET_PRIORITY = {
    "purchase_selection_regression": 0,
    "premature_finish_after_reference_candidate": 1,
    "reached_reference_candidate_and_option_but_failed_to_commit": 2,
    "reached_reference_candidate_but_option_or_commit_failed": 3,
    "premature_finish_before_reference_candidate": 4,
    "failed_to_reach_reference_candidate": 5,
}


def extract_tool_responses(output: object) -> list[str]:
    if not isinstance(output, str):
        return []
    return [match.group(1) for match in _TOOL_RESPONSE.finditer(output)]


def _shopping(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("shopping")
    return value if isinstance(value, Mapping) else {}


def _trace(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = _shopping(record).get("action_trace")
    if not isinstance(value, list):
        return []
    return [event for event in value if isinstance(event, Mapping)]


def _is_gold(record: Mapping[str, Any]) -> bool:
    shopping = _shopping(record)
    return (
        shopping.get("reward_type") == "gold_purchase"
        and shopping.get("termination_reason") == "gold_purchase"
        and shopping.get("valid_for_learning") is True
        and shopping.get("done") is True
        and shopping.get("infrastructure_invalid") is not True
    )


def _is_learning_failure(record: Mapping[str, Any]) -> bool:
    shopping = _shopping(record)
    return (
        shopping.get("reward_type") != "gold_purchase"
        and shopping.get("valid_for_learning") is True
        and shopping.get("infrastructure_invalid") is not True
    )


def _clean_gold_reject_reason(record: Mapping[str, Any]) -> str | None:
    if not _is_gold(record):
        return "not_gold"
    shopping = _shopping(record)
    trace = _trace(record)
    if not trace:
        return "empty_action_trace"
    if int(shopping.get("guard_rejections", 0) or 0) != 0:
        return "gold_has_guard_rejection"
    if int(shopping.get("repeat_actions", 0) or 0) != 0:
        return "gold_has_repeat_action"
    if any(event.get("accepted") is not True or event.get("error") for event in trace):
        return "gold_has_rejected_or_error_action"
    if trace[-1].get("tool") != "buy_now":
        return "gold_does_not_end_in_buy"
    if len(extract_tool_responses(record.get("output"))) != len(trace):
        return "tool_response_count_mismatch"
    return None


def _source_key(record: Mapping[str, Any]) -> tuple[str, int, int]:
    source = record.get("_source")
    source = source if isinstance(source, Mapping) else {}
    return (
        str(source.get("path") or ""),
        int(source.get("step", 0) or 0),
        int(source.get("line", 0) or 0),
    )


def _gold_sort_key(record: Mapping[str, Any]) -> tuple[int, int, tuple[str, int, int]]:
    return (
        len(_trace(record)),
        len(str(record.get("output") or "")),
        _source_key(record),
    )


def _canonical_option(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _reference_view(record: Mapping[str, Any]) -> dict[str, Any] | None:
    trace = _trace(record)
    buy_indices = [
        index
        for index, event in enumerate(trace)
        if event.get("accepted") is True and event.get("tool") == "buy_now"
    ]
    if not buy_indices:
        return None
    buy_index = buy_indices[-1]
    open_indices = [
        index
        for index, event in enumerate(trace[:buy_index])
        if event.get("accepted") is True
        and event.get("tool") == "open_product"
        and isinstance(event.get("parameters"), Mapping)
        and event["parameters"].get("asin") is not None
    ]
    if not open_indices:
        return None
    open_index = open_indices[-1]
    asin = str(trace[open_index]["parameters"]["asin"])
    option_indices = [
        index
        for index, event in enumerate(trace[open_index + 1 : buy_index], open_index + 1)
        if event.get("accepted") is True
        and event.get("tool") == "select_option"
        and isinstance(event.get("parameters"), Mapping)
        and event["parameters"].get("value") is not None
    ]
    return {
        "asin": asin,
        "open_event_index": open_index,
        "option_event_indices": option_indices,
        "options": [str(trace[index]["parameters"]["value"]) for index in option_indices],
        "buy_event_index": buy_index,
    }


def _failure_signature(record: Mapping[str, Any]) -> str:
    payload = {
        "termination_reason": _shopping(record).get("termination_reason"),
        "actions": [
            {
                "tool": event.get("tool"),
                "parameters": event.get("parameters"),
                "accepted": event.get("accepted"),
                "guard_reason": event.get("guard_reason"),
            }
            for event in _trace(record)
        ],
    }
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def diagnose_failure(
    failure: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    trace = _trace(failure)
    accepted = [event for event in trace if event.get("accepted") is True]
    opened_asins = {
        str((event.get("parameters") or {}).get("asin"))
        for event in accepted
        if event.get("tool") == "open_product"
        and isinstance(event.get("parameters"), Mapping)
        and event["parameters"].get("asin") is not None
    }
    selected_options = {
        _canonical_option(event["parameters"].get("value"))
        for event in accepted
        if event.get("tool") == "select_option"
        and isinstance(event.get("parameters"), Mapping)
        and event["parameters"].get("value") is not None
    }
    matching_options = [
        option
        for option in reference["options"]
        if _canonical_option(option) in selected_options
    ]
    reached = reference["asin"] in opened_asins
    attempted_purchase = any(event.get("tool") == "buy_now" for event in accepted)
    termination = str(_shopping(failure).get("termination_reason") or "unknown")
    if attempted_purchase:
        bucket = "purchase_selection_regression"
    elif termination in {
        "assistant_final",
        "assistant_finished_without_environment_done",
    }:
        position = "after" if reached else "before"
        bucket = f"premature_finish_{position}_reference_candidate"
    elif reached and matching_options:
        bucket = "reached_reference_candidate_and_option_but_failed_to_commit"
    elif reached:
        bucket = "reached_reference_candidate_but_option_or_commit_failed"
    else:
        bucket = "failed_to_reach_reference_candidate"
    return {
        "bucket": bucket,
        "termination_reason": termination,
        "reached_reference_asin": reached,
        "matching_reference_options": matching_options,
        "attempted_purchase": attempted_purchase,
    }


def _supervision_events(
    primary_bucket: str,
    reference: Mapping[str, Any],
) -> tuple[str, list[int]]:
    buy = int(reference["buy_event_index"])
    options = [int(index) for index in reference["option_event_indices"]]
    if primary_bucket == "failed_to_reach_reference_candidate":
        return "candidate_options_commit", [
            int(reference["open_event_index"]),
            *options,
            buy,
        ]
    if primary_bucket in {
        "purchase_selection_regression",
        "reached_reference_candidate_but_option_or_commit_failed",
        "premature_finish_before_reference_candidate",
    }:
        return "options_commit", [*options, buy]
    return "commit_only", [buy]


def reconstruct_action_messages(
    *,
    prompt: Iterable[Mapping[str, Any]],
    gold_record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    messages = [deepcopy(dict(message)) for message in prompt]
    trace = _trace(gold_record)
    responses = extract_tool_responses(gold_record.get("output"))
    if len(trace) != len(responses):
        raise ValueError("tool response count does not match action trace")
    task_id = int(_shopping(gold_record)["task_id"])
    source_digest = hashlib.sha256(
        str(gold_record.get("output") or "").encode("utf-8")
    ).hexdigest()[:12]
    for index, (event, response) in enumerate(zip(trace, responses)):
        name = str(event.get("tool") or "")
        parameters = event.get("parameters")
        if not name or not isinstance(parameters, Mapping):
            raise ValueError(f"action {index} is missing tool name or parameters")
        call_id = f"recovery-{task_id}-{source_digest}-{index:02d}"
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(
                                dict(parameters),
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": name,
                "tool_call_id": call_id,
                "content": "购买已完成。" if name == "buy_now" else response,
            }
        )
    return messages


def build_recovery_rows(
    *,
    records: Iterable[Mapping[str, Any]],
    prompts_by_task: Mapping[int, Iterable[Mapping[str, Any]]],
    held_out_task_ids: Iterable[int] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select one clean gold suffix per mixed-success task."""

    held_out = {int(task_id) for task_id in held_out_task_ids}
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    record_count = 0
    for record in records:
        record_count += 1
        source = record.get("_source")
        source = source if isinstance(source, Mapping) else {}
        group_id = str(source.get("group") or source.get("path") or "")
        task_id = int(_shopping(record)["task_id"])
        grouped[(group_id, task_id)].append(record)

    task_candidates: dict[int, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: {"gold": [], "failure": []}
    )
    mixed_groups = 0
    for (_, task_id), group_records in grouped.items():
        gold = [record for record in group_records if _is_gold(record)]
        failure = [record for record in group_records if _is_learning_failure(record)]
        if not gold or not failure:
            continue
        mixed_groups += 1
        task_candidates[task_id]["gold"].extend(gold)
        task_candidates[task_id]["failure"].extend(failure)

    rows = []
    rejected = []
    diagnostic_counts = Counter()
    supervision_counts = Counter()
    for task_id in sorted(task_candidates):
        if task_id in held_out:
            rejected.append({"task_id": task_id, "reason": "held_out_task"})
            continue
        prompt = prompts_by_task.get(task_id)
        if prompt is None:
            rejected.append({"task_id": task_id, "reason": "missing_prompt"})
            continue
        candidates = task_candidates[task_id]
        clean_gold = [
            record
            for record in candidates["gold"]
            if _clean_gold_reject_reason(record) is None
        ]
        if not clean_gold:
            reasons = Counter(
                _clean_gold_reject_reason(record) or "unknown"
                for record in candidates["gold"]
            )
            rejected.append(
                {
                    "task_id": task_id,
                    "reason": "no_clean_gold",
                    "gold_reject_reasons": dict(sorted(reasons.items())),
                }
            )
            continue
        gold = min(clean_gold, key=_gold_sort_key)
        reference = _reference_view(gold)
        if reference is None:
            rejected.append({"task_id": task_id, "reason": "gold_reference_missing"})
            continue
        unique_failures = {}
        for failure in candidates["failure"]:
            unique_failures.setdefault(_failure_signature(failure), failure)
        diagnoses = [
            diagnose_failure(failure, reference)
            for failure in unique_failures.values()
        ]
        primary = min(
            (diagnosis["bucket"] for diagnosis in diagnoses),
            key=lambda bucket: _BUCKET_PRIORITY[bucket],
        )
        supervision_mode, supervised_events = _supervision_events(primary, reference)
        messages = reconstruct_action_messages(prompt=prompt, gold_record=gold)
        supervised_message_indices = [2 + 2 * index for index in supervised_events]
        trace = _trace(gold)
        source = gold.get("_source")
        source = deepcopy(dict(source)) if isinstance(source, Mapping) else {}
        trajectory_hash = hashlib.sha256(
            str(gold.get("output") or "").encode("utf-8")
        ).hexdigest()
        row = {
            "schema_version": RECOVERY_SFT_SCHEMA_VERSION,
            "trajectory_id": f"recovery-{task_id}-{trajectory_hash[:16]}",
            "task_id": task_id,
            "messages": messages,
            "tools": deepcopy(SHOP_TOOL_SCHEMAS),
            "supervised_assistant_indices": supervised_message_indices,
            "recovery": {
                "primary_bucket": primary,
                "diagnostic_bucket_counts": dict(
                    sorted(Counter(item["bucket"] for item in diagnoses).items())
                ),
                "failure_termination_counts": dict(
                    sorted(
                        Counter(
                            item["termination_reason"] for item in diagnoses
                        ).items()
                    )
                ),
                "failure_trajectories": len(diagnoses),
                "reference_asin": reference["asin"],
                "reference_options": reference["options"],
                "supervision_mode": supervision_mode,
                "supervised_event_indices": supervised_events,
                "supervised_tools": [trace[index]["tool"] for index in supervised_events],
                "source": source,
                "source_output_sha256": trajectory_hash,
            },
        }
        rows.append(row)
        diagnostic_counts.update(item["bucket"] for item in diagnoses)
        supervision_counts[supervision_mode] += 1

    summary = {
        "schema_version": RECOVERY_SFT_SCHEMA_VERSION,
        "input_records": record_count,
        "prompt_groups": len(grouped),
        "mixed_success_groups": mixed_groups,
        "mixed_success_tasks": len(task_candidates),
        "recovery_rows": len(rows),
        "rejected_tasks": len(rejected),
        "diagnostic_bucket_counts": dict(sorted(diagnostic_counts.items())),
        "supervision_mode_counts": dict(sorted(supervision_counts.items())),
    }
    return rows, rejected, summary


def _stable_order(task_ids: Iterable[int], seed: int) -> list[int]:
    return sorted(
        {int(task_id) for task_id in task_ids},
        key=lambda task_id: hashlib.sha256(
            f"{seed}:{task_id}".encode()
        ).hexdigest(),
    )


def split_recovery_rows(
    *,
    recovery_rows: Iterable[Mapping[str, Any]],
    base_train_rows: Iterable[Mapping[str, Any]],
    base_validation_rows: Iterable[Mapping[str, Any]],
    validation_ratio: float = 0.1,
    seed: int = 20260810,
) -> dict[str, list[dict[str, Any]]]:
    """Preserve global task disjointness while mixing base and recovery rows."""

    recovery = [deepcopy(dict(row)) for row in recovery_rows]
    base_train = [deepcopy(dict(row)) for row in base_train_rows]
    base_validation = [deepcopy(dict(row)) for row in base_validation_rows]
    train_ids = {int(row["task_id"]) for row in base_train}
    validation_ids = {int(row["task_id"]) for row in base_validation}
    if train_ids & validation_ids:
        raise ValueError("base train and validation task IDs overlap")
    ratio = float(validation_ratio)
    if not 0 <= ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")

    recovery_by_task = {int(row["task_id"]): row for row in recovery}
    if len(recovery_by_task) != len(recovery):
        raise ValueError("recovery_rows must contain at most one row per task")
    unseen_ids = set(recovery_by_task) - train_ids - validation_ids
    ordered_unseen = _stable_order(unseen_ids, seed)
    validation_count = round(len(ordered_unseen) * ratio)
    if ordered_unseen and ratio > 0:
        validation_count = max(1, validation_count)
    new_validation_ids = set(ordered_unseen[:validation_count])
    recovery_validation_ids = validation_ids & set(recovery_by_task)
    recovery_validation_ids.update(new_validation_ids)
    recovery_train_ids = set(recovery_by_task) - recovery_validation_ids
    recovery_train = [
        recovery_by_task[task_id] for task_id in sorted(recovery_train_ids)
    ]
    recovery_validation = [
        recovery_by_task[task_id] for task_id in sorted(recovery_validation_ids)
    ]
    mixed_train = [*base_train, *recovery_train]
    mixed_validation = [*base_validation, *recovery_validation]
    mixed_train_ids = {int(row["task_id"]) for row in mixed_train}
    mixed_validation_ids = {int(row["task_id"]) for row in mixed_validation}
    if mixed_train_ids & mixed_validation_ids:
        raise AssertionError("mixed train and validation task IDs overlap")
    return {
        "recovery_train": recovery_train,
        "recovery_validation": recovery_validation,
        "mixed_train": mixed_train,
        "mixed_validation": mixed_validation,
    }


def replay_recovery_row(
    row: Mapping[str, Any],
    *,
    base_url: str,
    env_factory=ShopAgentEnv,
) -> dict[str, Any]:
    """Replay the clean source action sequence and report only public outcome facts."""

    task_id = int(row["task_id"])
    actions = []
    for message in row.get("messages") or []:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments", "{}")
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            actions.append(tool_call_to_action(function.get("name"), arguments))
    final = {}
    try:
        with env_factory(base_url=base_url) as env:
            env.reset(task_id)
            for action in actions:
                final = env.step(action)
                if final.get("done"):
                    break
        detail = final.get("reward_detail") or {}
        ok = (
            final.get("done") is True
            and final.get("over") is True
            and detail.get("reward_version") == "shopsimulator-reward-v3"
            and detail.get("reward_type") == "gold_purchase"
            and detail.get("reward_valid") is True
            and detail.get("purchase_success") is True
            and detail.get("termination_reason") == "gold_purchase"
        )
        return {
            "task_id": task_id,
            "ok": ok,
            "actions": len(actions),
            "reward_type": detail.get("reward_type"),
            "termination_reason": detail.get("termination_reason"),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - replay must audit per-task failures.
        return {
            "task_id": task_id,
            "ok": False,
            "actions": len(actions),
            "reward_type": None,
            "termination_reason": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def replay_recovery_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    base_url: str,
    workers: int = 8,
    env_factory=ShopAgentEnv,
) -> list[dict[str, Any]]:
    rows = list(rows)
    results = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(
                replay_recovery_row,
                row,
                base_url=base_url,
                env_factory=env_factory,
            ): int(row["task_id"])
            for row in rows
        }
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: item["task_id"])
