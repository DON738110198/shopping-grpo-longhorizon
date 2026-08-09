"""veRL 有界动态采样补丁使用的纯 Python reward-group 选择逻辑。

默认每个 task/prompt 采样 K=4 条轨迹。若组内 policy reward 全相同，则相对 advantage
没有排序信息；若任一轨迹基础设施无效或 reward 不可验证，则整组也不能训练。
本文件只返回保留索引和统计量，真正按索引裁剪 tensor batch 的位置在 veRL 补丁中。
"""

from __future__ import annotations

import json
import math
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any


def aggregate_shopping_metrics(shopping_infos: Sequence[object]) -> dict[str, float]:
    """把 AgentLoop 轨迹诊断聚合为 veRL 每步指标。"""
    if not shopping_infos:
        return {}

    reward_keys = (
        "full",
        "strict",
        "native",
        "semantic",
        "policy_base",
        "total",
        "efficiency",
        "penalty_overlong",
        "penalty_unfinished",
        "penalty_guard",
        "penalty_repeat",
        "repeat_action_rate",
        "r_type",
        "r_att",
        "r_option",
        "r_price",
    )
    rewards = {key: [] for key in reward_keys}
    steps = []
    done = []
    max_steps = []
    infrastructure_invalid = []
    reward_unverifiable = []
    terminal_utilities = []
    purchase_success = []
    sampling_invalid = []
    match_scores = []
    evidence_coverage = []
    partial_purchase = []
    model_failure = []
    valid_for_learning = []
    guard_rejections = []
    repeat_actions = []
    for index, info in enumerate(shopping_infos):
        if not isinstance(info, Mapping) or not isinstance(info.get("reward"), Mapping):
            raise TypeError(f"shopping extra field at index {index} is missing reward diagnostics")
        reward = info["reward"]
        for key in reward_keys:
            try:
                value = float(reward[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"shopping reward at index {index} is missing numeric {key}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"shopping reward {key} at index {index} is not finite")
            rewards[key].append(value)
        steps.append(float(info.get("steps", 0)))
        done.append(float(info.get("done") is True))
        max_steps.append(float(info.get("termination_reason") == "max_steps"))
        infrastructure_invalid.append(float(bool(info.get("infrastructure_invalid"))))
        reward_unverifiable.append(float(bool(info.get("reward_unverifiable"))))
        terminal_utilities.append(
            float(reward.get("terminal_utility", reward["total"]))
        )
        purchase_success.append(
            float(bool(reward.get("purchase_success", reward["full"])))
        )
        sampling_invalid.append(
            float(
                bool(
                    reward.get(
                        "sampling_invalid",
                        info.get("infrastructure_invalid")
                        or info.get("reward_unverifiable"),
                    )
                )
            )
        )
        match_scores.append(float(reward.get("match_score", reward["r_att"])))
        evidence_coverage.append(
            float(reward.get("evidence_coverage", 0.0))
        )
        partial_purchase.append(
            float(info.get("reward_type") == "partial_alternative_purchase")
        )
        model_failure.append(float(bool(info.get("model_failure"))))
        valid_for_learning.append(float(bool(info.get("valid_for_learning"))))
        guard_rejections.append(float(info.get("guard_rejections", 0)))
        repeat_actions.append(float(info.get("repeat_actions", 0)))

    def mean(values):
        return sum(values) / len(values)

    return {
        "reward/full_mean": mean(rewards["full"]),
        "reward/strict_mean": mean(rewards["strict"]),
        "reward/native_mean": mean(rewards["native"]),
        "reward/semantic_mean": mean(rewards["semantic"]),
        "reward/policy_base_mean": mean(rewards["policy_base"]),
        "reward/shaped_min": min(rewards["total"]),
        "reward/shaped_mean": mean(rewards["total"]),
        "reward/shaped_max": max(rewards["total"]),
        "reward/terminal_utility_min": min(terminal_utilities),
        "reward/terminal_utility_mean": mean(terminal_utilities),
        "reward/terminal_utility_max": max(terminal_utilities),
        "reward/purchase_success_rate": mean(purchase_success),
        "reward/partial_purchase_rate": mean(partial_purchase),
        "reward/match_score_mean": mean(match_scores),
        "reward/evidence_coverage_mean": mean(evidence_coverage),
        "reward/efficiency_mean": mean(rewards["efficiency"]),
        "penalty/overlong_mean": mean(rewards["penalty_overlong"]),
        "penalty/unfinished_mean": mean(rewards["penalty_unfinished"]),
        "penalty/guard_mean": mean(rewards["penalty_guard"]),
        "penalty/repeat_mean": mean(rewards["penalty_repeat"]),
        "component/r_type_mean": mean(rewards["r_type"]),
        "component/r_att_mean": mean(rewards["r_att"]),
        "component/r_option_mean": mean(rewards["r_option"]),
        "component/r_price_mean": mean(rewards["r_price"]),
        "trajectory/average_steps": mean(steps),
        "trajectory/done_rate": mean(done),
        "trajectory/max_steps_rate": mean(max_steps),
        "trajectory/repeat_action_rate": mean(rewards["repeat_action_rate"]),
        "trajectory/guard_rejections_mean": mean(guard_rejections),
        "trajectory/repeat_actions_mean": mean(repeat_actions),
        "trajectory/model_failure_rate": mean(model_failure),
        "trajectory/valid_for_learning_rate": mean(valid_for_learning),
        "trajectory/infrastructure_invalid_rate": mean(infrastructure_invalid),
        "trajectory/reward_unverifiable_rate": mean(reward_unverifiable),
        "trajectory/sampling_invalid_rate": mean(sampling_invalid),
    }


def extract_shopping_group_signals(
    shopping_infos: Sequence[object],
) -> tuple[list[float], list[float], list[bool], list[bool], list[tuple[str, ...]]]:
    """Return policy/raw rewards, success, and explicit learning-validity reasons."""
    policy_rewards = []
    terminal_utilities = []
    purchase_success = []
    sampling_invalid = []
    invalid_reasons = []
    for index, info in enumerate(shopping_infos):
        if not isinstance(info, Mapping) or not isinstance(info.get("reward"), Mapping):
            raise TypeError(f"shopping extra field at index {index} is missing reward diagnostics")
        try:
            policy_reward = float(info["reward"]["total"])
            terminal_utility = float(info["reward"]["terminal_utility"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TypeError(
                f"shopping extra field at index {index} is missing policy or terminal reward"
            ) from exc
        if not math.isfinite(policy_reward) or not math.isfinite(terminal_utility):
            raise ValueError(
                f"shopping reward at index {index} is not finite"
            )
        if info["reward"].get("policy_reward_version") != "shopping-policy-reward-v1":
            raise ValueError(f"shopping extra field at index {index} has wrong policy reward version")
        raw_purchase_success = info["reward"].get("purchase_success")
        if not isinstance(raw_purchase_success, (bool, int, float)):
            raise TypeError(
                f"shopping extra field at index {index} is missing purchase_success"
            )
        if "infrastructure_invalid" not in info:
            raise ValueError(
                f"shopping extra field at index {index} is missing infrastructure_invalid"
            )
        reasons = []
        if bool(info["infrastructure_invalid"]):
            reasons.append("infrastructure_invalid")
        if bool(info.get("reward_unverifiable")):
            reasons.append("reward_unverifiable")
        reward_sampling_invalid = bool(
            info["reward"].get("sampling_invalid", False)
        )
        if reward_sampling_invalid and not reasons:
            reasons.append(str(info.get("invalid_reason") or "reward_sampling_invalid"))
        expected_valid = not reasons
        if bool(info.get("valid_for_learning")) != expected_valid:
            raise ValueError(
                f"shopping extra field at index {index} has inconsistent valid_for_learning"
            )
        policy_rewards.append(policy_reward)
        terminal_utilities.append(terminal_utility)
        purchase_success.append(bool(raw_purchase_success))
        sampling_invalid.append(bool(reasons))
        invalid_reasons.append(tuple(reasons))
    return (
        policy_rewards,
        terminal_utilities,
        purchase_success,
        sampling_invalid,
        invalid_reasons,
    )


def select_reward_varying_groups(
    uids: Sequence[Hashable],
    seq_rewards: Sequence[float],
    *,
    policy_rewards: Sequence[float] | None = None,
    terminal_utilities: Sequence[float] | None = None,
    purchase_success: Sequence[bool] | None = None,
    sampling_invalid: Sequence[bool] | None = None,
    sampling_invalid_reasons: Sequence[Sequence[str]] | None = None,
    tolerance: float = 1.0e-8,
) -> tuple[list[int], dict[str, Any]]:
    """返回 reward 有差异且全部有效的 group 所对应的 trajectory 索引。

    对 uid=u 的组，先计算 ``range_u = max(policy_reward_u) - min(policy_reward_u)``。仅当
    ``range_u > tolerance`` 且组内没有 sampling_invalid 时保留。返回索引保持原始
    顺序，使调用方能对 input_ids、attention_mask、old_log_probs、rewards 以及
    extra_fields 使用同一个 selection，避免张量与轨迹诊断错位。
    """

    if len(uids) != len(seq_rewards):
        raise ValueError(
            f"uids and seq_rewards must have equal length, got {len(uids)} and {len(seq_rewards)}"
        )
    optional_sequences = {
        "policy_rewards": policy_rewards,
        "terminal_utilities": terminal_utilities,
        "purchase_success": purchase_success,
        "sampling_invalid": sampling_invalid,
        "sampling_invalid_reasons": sampling_invalid_reasons,
    }
    for name, values in optional_sequences.items():
        if values is not None and len(values) != len(uids):
            raise ValueError(f"{name} must have the same length as uids")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be a finite non-negative number, got {tolerance!r}")

    policy_values = policy_rewards if policy_rewards is not None else seq_rewards
    terminal_values = terminal_utilities if terminal_utilities is not None else seq_rewards
    success_values = (
        purchase_success if purchase_success is not None else [False] * len(uids)
    )
    invalid_values = (
        sampling_invalid if sampling_invalid is not None else [False] * len(uids)
    )
    reason_values = (
        sampling_invalid_reasons
        if sampling_invalid_reasons is not None
        else [()] * len(uids)
    )
    # uid 是 prompt/task 的组标识；相同 uid 的 K 条 rollout 必须一起作决定。
    grouped: dict[Hashable, dict[str, Any]] = {}
    for index, (
        uid,
        raw_reward,
        raw_policy_reward,
        raw_terminal_utility,
        raw_success,
        raw_invalid,
        raw_reasons,
    ) in enumerate(
        zip(
            uids,
            seq_rewards,
            policy_values,
            terminal_values,
            success_values,
            invalid_values,
            reason_values,
            strict=True,
        )
    ):
        try:
            hash(uid)
        except TypeError as exc:
            raise ValueError(f"uid at index {index} is not hashable: {uid!r}") from exc

        reward = float(raw_reward)
        if not math.isfinite(reward):
            raise ValueError(f"seq_reward at index {index} is not finite: {raw_reward!r}")
        metadata_reward = float(raw_policy_reward)
        if not math.isfinite(metadata_reward):
            raise ValueError(
                f"policy_reward at index {index} is not finite: {raw_policy_reward!r}"
            )
        if not math.isclose(reward, metadata_reward, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(
                f"policy reward mismatch at index {index}: tensor={reward}, metadata={metadata_reward}"
            )
        terminal_utility = float(raw_terminal_utility)
        if not math.isfinite(terminal_utility):
            raise ValueError(
                f"terminal_utility at index {index} is not finite: {raw_terminal_utility!r}"
            )

        group = grouped.setdefault(
            uid,
            {
                "uid": uid,
                "indices": [],
                "rewards": [],
                "terminal_utilities": [],
                "purchase_success": [],
                "sampling_invalid": [],
                "sampling_invalid_reasons": [],
            },
        )
        group["indices"].append(index)
        group["rewards"].append(reward)
        group["terminal_utilities"].append(terminal_utility)
        group["purchase_success"].append(bool(raw_success))
        group["sampling_invalid"].append(bool(raw_invalid))
        group["sampling_invalid_reasons"].extend(str(reason) for reason in raw_reasons)

    kept_uids: list[Hashable] = []
    dropped_uids: list[Hashable] = []
    groups: list[dict[str, Any]] = []
    for uid, group in grouped.items():
        rewards = group["rewards"]
        utilities = group["terminal_utilities"]
        reward_min = min(rewards)
        reward_max = max(rewards)
        reward_varying = reward_max - reward_min > tolerance
        has_sampling_invalid = any(group["sampling_invalid"])
        reasons = tuple(sorted(set(group["sampling_invalid_reasons"])))
        if has_sampling_invalid:
            drop_reason = "sampling_invalid"
        elif not reward_varying:
            drop_reason = "constant_reward"
        else:
            drop_reason = None
        keep = drop_reason is None
        if keep:
            kept_uids.append(uid)
        else:
            dropped_uids.append(uid)
        groups.append(
            {
                "uid": uid,
                "indices": tuple(group["indices"]),
                "rewards": tuple(group["rewards"]),
                "terminal_utilities": tuple(utilities),
                "purchase_success": tuple(group["purchase_success"]),
                "reward_min": reward_min,
                "reward_max": reward_max,
                "utility_min": min(utilities),
                "utility_max": max(utilities),
                "reward_varying": reward_varying,
                "sampling_invalid": has_sampling_invalid,
                "sampling_invalid_reasons": reasons,
                "drop_reason": drop_reason,
                "kept": keep,
            }
        )

    kept_uid_set = set(kept_uids)
    trajectory_indices = [index for index, uid in enumerate(uids) if uid in kept_uid_set]
    stats = {
        "num_trajectories": len(uids),
        "num_groups": len(grouped),
        "kept_group_count": len(kept_uids),
        "dropped_group_count": len(dropped_uids),
        "kept_uids": tuple(kept_uids),
        "dropped_uids": tuple(dropped_uids),
        "all_equal_group_count": sum(
            not group["reward_varying"] for group in groups
        ),
        "all_zero_reward_group_count": sum(
            max(abs(value) for value in group["rewards"]) <= tolerance
            for group in groups
        ),
        # The veRL patch used this name before policy/raw rewards were separated.
        "all_zero_utility_group_count": sum(
            max(abs(value) for value in group["rewards"]) <= tolerance
            for group in groups
        ),
        "all_zero_terminal_utility_group_count": sum(
            max(abs(value) for value in group["terminal_utilities"]) <= tolerance
            for group in groups
        ),
        "all_purchase_success_group_count": sum(
            all(group["purchase_success"])
            for group in groups
        ),
        "no_purchase_success_group_count": sum(
            not any(group["purchase_success"]) for group in groups
        ),
        "sampling_invalid_group_count": sum(
            group["sampling_invalid"] for group in groups
        ),
        "sampling_invalid_reason_counts": {
            reason: sum(
                reason in group["sampling_invalid_reasons"] for group in groups
            )
            for reason in sorted(
                {
                    reason
                    for group in groups
                    for reason in group["sampling_invalid_reasons"]
                }
            )
        },
        # Compatibility aliases for existing monitoring code.
        "infrastructure_invalid_group_count": sum(
            group["sampling_invalid"] for group in groups
        ),
        "groups": tuple(groups),
    }
    return trajectory_indices, stats


def append_sampling_audit(
    output_dir: str | Path,
    *,
    global_step: int,
    generation_batch: int,
    group_stats: Mapping[str, Any],
    shopping_infos: Sequence[object],
) -> Path:
    """Append compact per-group rollout evidence from the central trainer process."""
    destination = Path(output_dir) / "sampling_audit.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for group in group_stats.get("groups", ()):
            trajectories = []
            for index in group["indices"]:
                info = shopping_infos[index]
                if not isinstance(info, Mapping):
                    raise TypeError(f"shopping audit entry at index {index} is not an object")
                reward = info.get("reward") or {}
                trajectories.append(
                    {
                        "task_id": info.get("task_id"),
                        "termination_reason": info.get("termination_reason"),
                        "reward_type": info.get("reward_type"),
                        "policy_reward": reward.get("total"),
                        "terminal_utility": reward.get("terminal_utility"),
                        "strict": reward.get("strict"),
                        "valid_for_learning": info.get("valid_for_learning"),
                        "invalid_reason": info.get("invalid_reason"),
                        "model_failure": info.get("model_failure"),
                        "steps": info.get("steps"),
                        "guard_rejections": info.get("guard_rejections"),
                        "repeat_actions": info.get("repeat_actions"),
                        "action_trace": info.get("action_trace", []),
                    }
                )
            record = {
                "global_step": int(global_step),
                "generation_batch": int(generation_batch),
                "uid": group["uid"],
                "kept": bool(group["kept"]),
                "drop_reason": group["drop_reason"],
                "sampling_invalid_reasons": group["sampling_invalid_reasons"],
                "policy_rewards": group["rewards"],
                "terminal_utilities": group["terminal_utilities"],
                "trajectories": trajectories,
            }
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return destination
