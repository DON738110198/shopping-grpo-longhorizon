"""Deterministic audit of strict-success flips between two complete runs.

The audit deliberately uses only actor-visible actions and the product selected
by the successful run. It never copies the environment's hidden goal into the
output. Diagnostic buckets describe observable failure locations, not inferred
model intent or a causal root cause.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from shopping_grpo.evaluation.metrics import compute_deterministic_metrics
from shopping_grpo.evaluation.summary import is_strict_success
from shopping_grpo.evaluation.trajectory import normalize_trajectory

TRANSITION_AUDIT_SCHEMA_VERSION = "shopping-strict-transition-audit-v1"
_METRIC_PATHS = {
    "executed_tool_steps": ("actions_and_efficiency", "executed_tool_steps"),
    "action_attempts": ("actions_and_efficiency", "action_attempts"),
    "guard_rejections": ("legality", "guard_rejection_count"),
    "duplicate_actions": ("repetition", "duplicate_canonical_action_count"),
    "consecutive_duplicate_actions": (
        "repetition",
        "consecutive_duplicate_action_count",
    ),
    "truncated_observations": ("context", "truncated_observation_count"),
}


def _index_complete_run(
    label: str,
    trajectories: Iterable[Mapping[str, Any]],
    expected_ids: set[int],
) -> dict[int, Mapping[str, Any]]:
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


def _bounded_text(value: object, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _action_summary(event: Mapping[str, Any]) -> str:
    name = str(event.get("tool_name") or "unknown")
    parameters = event.get("parameters")
    parameters = parameters if isinstance(parameters, Mapping) else {}
    preferred = next(
        (
            (key, parameters[key])
            for key in ("query", "asin", "value", "option")
            if key in parameters
        ),
        None,
    )
    argument = ""
    if preferred is not None:
        key, value = preferred
        argument = f"({key}={_bounded_text(value, 72)!r})"
    if event.get("event_type") == "guard_rejection":
        reason = _bounded_text(event.get("guard_reason") or "unknown", 64)
        return f"REJECTED {name}{argument} [{reason}]"
    return f"{name}{argument}"


def _unique(values: Iterable[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _canonical_option(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _metric_value(metrics: Mapping[str, Any], path: tuple[str, str]) -> int:
    section = metrics.get(path[0])
    section = section if isinstance(section, Mapping) else {}
    return int(section.get(path[1], 0) or 0)


def _trajectory_view(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_trajectory(trajectory)
    metrics = compute_deterministic_metrics(normalized)
    outcome = metrics["reward_and_outcome"]
    actions = metrics["actions_and_efficiency"]
    legality = metrics["legality"]
    repetition = metrics["repetition"]
    context = metrics["context"]
    validity = metrics["validity"]
    executed = [
        event
        for event in normalized["events"]
        if event.get("event_type") == "tool_step"
    ]
    opened_asins = _unique(
        str((event.get("parameters") or {}).get("asin"))
        for event in executed
        if event.get("tool_name") == "open_product"
        and (event.get("parameters") or {}).get("asin") is not None
    )
    selected_options = _unique(
        _bounded_text((event.get("parameters") or {}).get("value"), 160)
        for event in executed
        if event.get("tool_name") == "select_option"
        and (event.get("parameters") or {}).get("value") is not None
    )
    purchase = normalized["terminal"].get("purchase")
    purchase = purchase if isinstance(purchase, Mapping) else {}
    return {
        "strict_success": bool(outcome["strict_gold_success"]),
        "status": normalized["status"],
        "reward_type": outcome["reward_type"],
        "termination_reason": outcome["termination_reason"],
        "final_reward": outcome["final_reward"],
        "terminal_utility": outcome["terminal_utility"],
        "purchased_asin": (
            str(purchase["asin"]) if purchase.get("asin") is not None else None
        ),
        "executed_opened_asins": opened_asins,
        "executed_selected_options": selected_options,
        "executed_tool_steps": actions["executed_tool_steps"],
        "action_attempts": actions["action_attempts"],
        "buy_count": actions["buy_count"],
        "guard_rejection_count": legality["guard_rejection_count"],
        "guard_reason_counts": legality["guard_reason_counts"],
        "duplicate_canonical_action_count": repetition[
            "duplicate_canonical_action_count"
        ],
        "consecutive_duplicate_action_count": repetition[
            "consecutive_duplicate_action_count"
        ],
        "truncated_observation_count": context["truncated_observation_count"],
        "trajectory_error_type": validity["trajectory_error_type"],
        "infrastructure_invalid": validity["infrastructure_invalid"],
        "action_trace": [_action_summary(event) for event in normalized["events"]],
        "_metrics": metrics,
    }


def _failure_bucket(failure: Mapping[str, Any], bridge: Mapping[str, Any]) -> str:
    if int(failure["buy_count"]) > 0:
        return "purchase_selection_regression"
    reached = bool(bridge["failure_reached_reference_asin"])
    option_overlap = bool(bridge["matching_reference_options"])
    termination = str(failure["termination_reason"])
    if termination == "assistant_final":
        suffix = "after" if reached else "before"
        return f"premature_finish_{suffix}_reference_candidate"
    if reached and option_overlap:
        return "reached_reference_candidate_and_option_but_failed_to_commit"
    if reached:
        return "reached_reference_candidate_but_option_or_commit_failed"
    return "failed_to_reach_reference_candidate"


def _case_view(
    *,
    task_id: int,
    transition: str,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    source_view = _trajectory_view(source)
    target_view = _trajectory_view(target)
    success_view = target_view if transition == "gain" else source_view
    failure_view = source_view if transition == "gain" else target_view
    reference_asin = success_view["purchased_asin"]
    reference_options = success_view["executed_selected_options"]
    failure_options_by_canonical = {
        _canonical_option(value): value
        for value in failure_view["executed_selected_options"]
    }
    matching_options = _unique(
        failure_options_by_canonical[_canonical_option(value)]
        for value in reference_options
        if _canonical_option(value) in failure_options_by_canonical
    )
    bridge = {
        "reference_success_asin": reference_asin,
        "failure_reached_reference_asin": bool(
            reference_asin
            and reference_asin in failure_view["executed_opened_asins"]
        ),
        "reference_success_options": reference_options,
        "matching_reference_options": matching_options,
        "failure_attempted_purchase": int(failure_view["buy_count"]) > 0,
    }
    case = {
        "task_id": task_id,
        "transition": transition,
        "actor_query": _bounded_text(
            normalize_trajectory(source).get("actor_query"),
            360,
        ),
        "source": source_view,
        "target": target_view,
        "failure_bridge": bridge,
        "diagnostic_bucket": _failure_bucket(failure_view, bridge),
        "metric_delta_target_minus_source": {
            name: _metric_value(target_view["_metrics"], path)
            - _metric_value(source_view["_metrics"], path)
            for name, path in _METRIC_PATHS.items()
        },
    }
    del source_view["_metrics"]
    del target_view["_metrics"]
    return case


def _aggregate_cases(cases: list[Mapping[str, Any]], transition: str) -> dict[str, Any]:
    selected = [case for case in cases if case["transition"] == transition]
    failure_side = "source" if transition == "gain" else "target"
    deltas = Counter()
    for case in selected:
        deltas.update(case["metric_delta_target_minus_source"])
    count = len(selected)
    return {
        "cases": count,
        "failure_reward_type_counts": dict(
            sorted(Counter(case[failure_side]["reward_type"] for case in selected).items())
        ),
        "failure_termination_reason_counts": dict(
            sorted(
                Counter(
                    case[failure_side]["termination_reason"] for case in selected
                ).items()
            )
        ),
        "diagnostic_bucket_counts": dict(
            sorted(Counter(case["diagnostic_bucket"] for case in selected).items())
        ),
        "failure_reached_reference_asin": sum(
            case["failure_bridge"]["failure_reached_reference_asin"]
            for case in selected
        ),
        "failure_matched_reference_option": sum(
            bool(case["failure_bridge"]["matching_reference_options"])
            for case in selected
        ),
        "failure_attempted_purchase": sum(
            case["failure_bridge"]["failure_attempted_purchase"]
            for case in selected
        ),
        "metric_delta_target_minus_source": {
            name: {
                "total": deltas[name],
                "mean": deltas[name] / count if count else 0.0,
            }
            for name in _METRIC_PATHS
        },
    }


def audit_strict_transitions(
    *,
    expected_task_ids: Iterable[int],
    source_trajectories: Iterable[Mapping[str, Any]],
    target_trajectories: Iterable[Mapping[str, Any]],
    source_label: str,
    target_label: str,
) -> dict[str, Any]:
    """Audit every strict-success gain and loss on a complete paired task set."""

    expected = [int(task_id) for task_id in expected_task_ids]
    if not expected:
        raise ValueError("expected_task_ids must not be empty")
    if len(set(expected)) != len(expected):
        raise ValueError("expected_task_ids contains duplicates")
    expected_set = set(expected)
    source = _index_complete_run(source_label, source_trajectories, expected_set)
    target = _index_complete_run(target_label, target_trajectories, expected_set)

    cases = []
    transition_counts = Counter()
    source_successes = 0
    target_successes = 0
    for task_id in expected:
        source_success = is_strict_success(source[task_id])
        target_success = is_strict_success(target[task_id])
        source_successes += int(source_success)
        target_successes += int(target_success)
        transition_key = (
            f"{'success' if source_success else 'failure'}_to_"
            f"{'success' if target_success else 'failure'}"
        )
        transition_counts[transition_key] += 1
        if source_success == target_success:
            continue
        cases.append(
            _case_view(
                task_id=task_id,
                transition="gain" if target_success else "loss",
                source=source[task_id],
                target=target[task_id],
            )
        )

    return {
        "schema_version": TRANSITION_AUDIT_SCHEMA_VERSION,
        "source": {"label": source_label, "strict_successes": source_successes},
        "target": {"label": target_label, "strict_successes": target_successes},
        "paired_tasks": len(expected),
        "strict_success_transitions": dict(sorted(transition_counts.items())),
        "aggregate": {
            "gains": _aggregate_cases(cases, "gain"),
            "losses": _aggregate_cases(cases, "loss"),
        },
        "cases": cases,
    }


def render_transition_audit_markdown(audit: Mapping[str, Any]) -> str:
    """Render a compact Chinese evidence report without hidden task facts."""

    source_label = audit["source"]["label"]
    target_label = audit["target"]["label"]
    gains = audit["aggregate"]["gains"]
    losses = audit["aggregate"]["losses"]
    lines = [
        "# 严格成功翻转轨迹审计",
        "",
        "本报告只使用 Actor 可见动作，以及成功轨迹实际购买的商品和规格；",
        "不写入环境 hidden goal。诊断桶描述失败发生的位置，不等同于因果归因。",
        "",
        "## 总览",
        "",
        f"- 对比：`{source_label}` -> `{target_label}`",
        f"- 配对任务：{audit['paired_tasks']}",
        f"- gains：{gains['cases']}；losses：{losses['cases']}",
        f"- loss 中失败轨迹到过参考成功商品：{losses['failure_reached_reference_asin']}/{losses['cases']}",
        f"- loss 中失败轨迹选中过参考成功规格：{losses['failure_matched_reference_option']}/{losses['cases']}",
        f"- loss 中失败轨迹实际调用过购买：{losses['failure_attempted_purchase']}/{losses['cases']}",
        "",
        "### 失败终止类型",
        "",
        "| 翻转 | 失败侧 reward type | 数量 |",
        "| --- | --- | ---: |",
    ]
    for label, aggregate in (("gain", gains), ("loss", losses)):
        for reward_type, count in aggregate["failure_reward_type_counts"].items():
            lines.append(f"| {label} | `{reward_type}` | {count} |")
    lines.extend(
        [
            "",
            "### 可观察诊断桶",
            "",
            "| 翻转 | 诊断桶 | 数量 |",
            "| --- | --- | ---: |",
        ]
    )
    for label, aggregate in (("gain", gains), ("loss", losses)):
        for bucket, count in aggregate["diagnostic_bucket_counts"].items():
            lines.append(f"| {label} | `{bucket}` | {count} |")
    lines.extend(
        [
            "",
            "### 行为变化",
            "",
            f"以下均为 `{target_label} - {source_label}`。",
            "",
            "| 翻转 | steps | guard | duplicate | truncation |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, aggregate in (("gain", gains), ("loss", losses)):
        delta = aggregate["metric_delta_target_minus_source"]
        lines.append(
            "| {label} | {steps:+.2f} | {guard:+.2f} | {duplicate:+.2f} | {trunc:+.2f} |".format(
                label=label,
                steps=delta["executed_tool_steps"]["mean"],
                guard=delta["guard_rejections"]["mean"],
                duplicate=delta["duplicate_actions"]["mean"],
                trunc=delta["truncated_observations"]["mean"],
            )
        )

    for transition, title in (("gain", "Gains"), ("loss", "Losses")):
        lines.extend(["", f"## {title}", ""])
        for case in audit["cases"]:
            if case["transition"] != transition:
                continue
            source = case["source"]
            target = case["target"]
            bridge = case["failure_bridge"]
            lines.extend(
                [
                    f"### Task {case['task_id']}",
                    "",
                    f"- 用户需求：{case['actor_query']}",
                    f"- 结果：`{source['reward_type']}` -> `{target['reward_type']}`",
                    f"- 诊断桶：`{case['diagnostic_bucket']}`",
                    f"- 参考成功商品：`{bridge['reference_success_asin']}`；失败侧到达：{bridge['failure_reached_reference_asin']}",
                    f"- 参考规格重合：{bridge['matching_reference_options'] or '无'}；失败侧购买调用：{bridge['failure_attempted_purchase']}",
                    f"- {source_label}：" + " -> ".join(source["action_trace"]),
                    f"- {target_label}：" + " -> ".join(target["action_trace"]),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"
