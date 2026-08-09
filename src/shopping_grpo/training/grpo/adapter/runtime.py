"""每条 veRL trajectory 的轻量运行状态；不保存 ShopSimulator 隐藏 goal。

这里是环境事实到训练信号之间的“防火墙”。只有公开 observation、动作、终局
Reward v3 和诊断计数能进入 veRL；环境用于判分的隐藏 gold goal 不会写入 state，
否则模型可能通过训练管线的数据侧信道看到答案。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from contextvars import ContextVar

current_environment: ContextVar = ContextVar("shopsimulator_environment", default=None)
current_runtime_state: ContextVar = ContextVar("shopsimulator_runtime_state", default=None)
POLICY_REWARD_VERSION = "shopping-policy-reward-v1"
DEFAULT_POLICY_REWARD_CONFIG = {
    "version": POLICY_REWARD_VERSION,
    "guard_rejection_penalty": 0.03,
    "guard_rejection_cap": 3,
    "repeat_action_penalty": 0.02,
    "repeat_action_cap": 3,
    "minimum": -1.0,
    "maximum": 1.0,
    "model_failure_rewards": {
        "assistant_finished_without_environment_done": -0.40,
        "max_steps": -0.50,
        "context_hard_limit_exceeded": -0.55,
        "parallel_tool_calls": -0.60,
        "too_many_guard_rejections": -0.70,
    },
}
ACTION_TRACE_LIMIT = 48
REWARD_V3_TYPES = {
    "gold_purchase",
    "valid_alternative_purchase",
    "partial_alternative_purchase",
    "graceful_stop",
    "early_abstain",
    "wrong_purchase",
    "repeat_loop",
    "max_steps",
    "reward_unverifiable",
}


def make_runtime_state(task_id: int, max_steps: int) -> dict:
    """创建只含公共运行诊断的状态，reward 仅在环境正常终局后写入。

    ``done`` 表示环境终局，``terminate`` 表示 AgentLoop 应停止；二者并不等价。
    例如上下文超限会 terminate，但不能伪造成一次合法的环境 done。
    """
    return {
        "task_id": int(task_id),
        "max_steps": int(max_steps),
        "steps": [],
        "done": False,
        "terminate": False,
        "termination_reason": None,
        "consecutive_guard_rejections": 0,
        "action_attempt_count": 0,
        "repeat_action_count": 0,
        "recent_action_signatures": [],
        "action_events": [],
        "terminal_result": {},
        "final_reward": 0.0,
        "reward_version": None,
        "reward_type": None,
        "reward_valid": True,
        "reward_unverifiable": False,
        "reward_detail": None,
        "infrastructure_invalid": False,
        "error": None,
        "context_compactions": 0,
        "context_tokens_removed": 0,
        "context_max_input_tokens": 0,
        "observation_projection_count": 0,
        "observation_truncated_count": 0,
        "observation_raw_tokens": 0,
        "observation_visible_tokens": 0,
        "observation_max_raw_tokens": 0,
        "observation_max_visible_tokens": 0,
        "observation_visible_asin_count": 0,
        "observation_visible_button_count": 0,
        "observation_any_truncated": False,
        "latest_observation_truncated": False,
        "observation_footer_failures": 0,
        "guard_rejection_count": 0,
        "guard_rejection_after_truncation_count": 0,
        "action_attempt_after_truncation_count": 0,
    }


def record_observation_projection(state: dict, meta: dict) -> None:
    """Aggregate public projection diagnostics without retaining hidden environment state."""
    raw_tokens = int(meta["raw_tokens"])
    visible_tokens = int(meta["visible_tokens"])
    state["observation_projection_count"] += 1
    state["observation_truncated_count"] += int(bool(meta["truncated"]))
    state["observation_raw_tokens"] += raw_tokens
    state["observation_visible_tokens"] += visible_tokens
    state["observation_max_raw_tokens"] = max(state["observation_max_raw_tokens"], raw_tokens)
    state["observation_max_visible_tokens"] = max(
        state["observation_max_visible_tokens"], visible_tokens
    )
    state["observation_visible_asin_count"] += int(meta["visible_asin_count"])
    state["observation_visible_button_count"] += int(meta["visible_button_count"])
    state["observation_any_truncated"] = (
        state["observation_any_truncated"] or bool(meta["truncated"])
    )
    state["latest_observation_truncated"] = bool(meta["truncated"])
    state["observation_footer_failures"] += int(
        not bool(meta["critical_footer_preserved"])
    )


def _bounded_parameters(parameters: dict) -> dict:
    """Keep only the small public tool arguments needed for rollout auditing."""
    bounded = {}
    for key, value in list(parameters.items())[:8]:
        if isinstance(value, str):
            bounded[str(key)] = value[:256]
        elif isinstance(value, (bool, int, float)) or value is None:
            bounded[str(key)] = value
        else:
            bounded[str(key)] = str(value)[:256]
    return bounded


def record_action_attempt(
    state: dict,
    tool_name: str,
    parameters: dict,
    observation: str,
) -> dict | None:
    """Record a public action summary and detect a repeat on the recent page state."""
    if tool_name == "think":
        return None
    canonical_parameters = json.dumps(
        parameters,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    observation_fingerprint = hashlib.sha256(str(observation).encode("utf-8")).hexdigest()
    signature = (str(tool_name), canonical_parameters, observation_fingerprint)
    recent = state["recent_action_signatures"]
    state["action_attempt_count"] += 1
    repeated = signature in recent
    if repeated:
        state["repeat_action_count"] += 1
    recent.append(signature)
    del recent[:-3]
    event = {
        "index": state["action_attempt_count"] - 1,
        "tool": str(tool_name),
        "parameters": _bounded_parameters(parameters),
        "observation_sha256": observation_fingerprint,
        "repeated": repeated,
        "accepted": None,
        "guard_reason": None,
        "error": None,
    }
    state["action_events"].append(event)
    del state["action_events"][:-ACTION_TRACE_LIMIT]
    return event


def record_action_outcome(
    event: dict | None,
    *,
    accepted: bool,
    guard_reason: str | None = None,
    error: str | None = None,
) -> None:
    """Complete an action event after the guard or environment has handled it."""
    if event is None:
        return
    event["accepted"] = bool(accepted)
    event["guard_reason"] = str(guard_reason) if guard_reason else None
    event["error"] = str(error)[:512] if error else None


def validate_reward(raw_detail: object) -> dict:
    """验证并最小化 Environment v2.1 / Reward v3 的公开诊断。

    这里采用 fail-closed：字段缺失、NaN、范围越界或枚举不一致都会让整条 rollout
    ``sampling_invalid``，而不是用默认值继续训练。返回对象只保留训练/监控需要字段。
    """
    if not isinstance(raw_detail, Mapping):
        raise TypeError("reward_detail must be an object")
    if raw_detail.get("reward_version") != "shopsimulator-reward-v3":
        raise ValueError("reward_detail has an unsupported reward_version")
    reward_type = str(raw_detail.get("reward_type", ""))
    if reward_type not in REWARD_V3_TYPES:
        raise ValueError(f"unknown Reward v3 reward_type: {reward_type!r}")
    if raw_detail.get("termination_reason") != reward_type:
        raise ValueError("termination_reason must equal reward_type")
    reward_valid = raw_detail.get("reward_valid")
    if not isinstance(reward_valid, bool):
        raise TypeError("reward_valid must be boolean")
    if (reward_type == "reward_unverifiable") != (not reward_valid):
        raise ValueError("only reward_unverifiable may set reward_valid=false")
    try:
        terminal_utility = float(raw_detail.get("terminal_utility"))
    except (TypeError, ValueError) as exc:
        raise ValueError("terminal_utility must be numeric") from exc
    if not math.isfinite(terminal_utility):
        raise ValueError("terminal_utility must be finite")
    purchase_success = raw_detail.get("purchase_success")
    sampling_invalid = raw_detail.get("sampling_invalid")
    if not isinstance(purchase_success, bool):
        raise TypeError("purchase_success must be boolean")
    if not isinstance(sampling_invalid, bool):
        raise TypeError("sampling_invalid must be boolean")
    if sampling_invalid != (not reward_valid):
        raise ValueError("sampling_invalid must equal not reward_valid")
    hard_gates = raw_detail.get("hard_gates")
    if not isinstance(hard_gates, Mapping):
        raise TypeError("hard_gates must be an object")
    public_gates = {}
    for name, raw_gate in hard_gates.items():
        if not isinstance(raw_gate, Mapping):
            raise TypeError(f"hard gate {name!r} must be an object")
        status = raw_gate.get("status")
        if status not in {"pass", "fail", "unverifiable"}:
            raise ValueError(f"hard gate {name!r} has invalid status")
        if raw_gate.get("passed") != (status == "pass"):
            raise ValueError(f"hard gate {name!r} has inconsistent passed")
        if raw_gate.get("verifiable") != (status != "unverifiable"):
            raise ValueError(f"hard gate {name!r} has inconsistent verifiable")
        public_gates[str(name)] = {
            "status": status,
            "passed": raw_gate["passed"],
            "verifiable": raw_gate["verifiable"],
            "comparator": str(raw_gate.get("comparator") or ""),
            "source_field": str(raw_gate.get("source_field") or ""),
        }
    try:
        weighted_score = float(raw_detail.get("weighted_score", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("weighted_score must be numeric") from exc
    if not math.isfinite(weighted_score) or not 0.0 <= weighted_score <= 1.0:
        raise ValueError("weighted_score must be finite and in [0, 1]")
    try:
        evidence_coverage = float(raw_detail.get("evidence_coverage", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("evidence_coverage must be numeric") from exc
    if (
        not math.isfinite(evidence_coverage)
        or not 0.0 <= evidence_coverage <= 1.0
    ):
        raise ValueError("evidence_coverage must be finite and in [0, 1]")
    raw_dimension_scores = raw_detail.get("dimension_scores") or {}
    if not isinstance(raw_dimension_scores, Mapping):
        raise TypeError("dimension_scores must be an object")
    dimension_scores = {}
    for name in ("brand", "model", "core_functions", "key_options"):
        try:
            score = float(raw_dimension_scores.get(name, 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"dimension score {name} must be numeric") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(
                f"dimension score {name} must be finite and in [0, 1]"
            )
        dimension_scores[name] = score
    return {
        "reward_version": "shopsimulator-reward-v3",
        "reward_type": reward_type,
        "reward_valid": reward_valid,
        "termination_reason": reward_type,
        "target_asin_match": bool(raw_detail.get("target_asin_match")),
        "hard_gates": public_gates,
        "weighted_score": weighted_score,
        "evidence_coverage": evidence_coverage,
        "dimension_scores": dimension_scores,
        "terminal_utility": terminal_utility,
        "purchase_success": purchase_success,
        "sampling_invalid": sampling_invalid,
    }


def _normal_terminal(state: dict) -> bool:
    terminal = state.get("terminal_result") or {}
    return (
        state.get("done") is True
        and terminal.get("done") is True
        and terminal.get("over") is True
    )


def validate_policy_reward_config(raw_config: object = None) -> dict:
    """Resolve and validate the training-only policy reward configuration."""
    raw_config = raw_config or {}
    if not isinstance(raw_config, Mapping):
        raise TypeError("policy_reward config must be an object")
    config = dict(DEFAULT_POLICY_REWARD_CONFIG)
    config.update({key: value for key, value in raw_config.items() if key != "model_failure_rewards"})
    failure_rewards = dict(DEFAULT_POLICY_REWARD_CONFIG["model_failure_rewards"])
    raw_failures = raw_config.get("model_failure_rewards", {})
    if not isinstance(raw_failures, Mapping):
        raise TypeError("policy_reward.model_failure_rewards must be an object")
    failure_rewards.update(raw_failures)
    config["model_failure_rewards"] = failure_rewards
    if config.get("version") != POLICY_REWARD_VERSION:
        raise ValueError(f"unsupported policy reward version: {config.get('version')!r}")
    for key in (
        "guard_rejection_penalty",
        "repeat_action_penalty",
        "minimum",
        "maximum",
    ):
        try:
            config[key] = float(config[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"policy_reward.{key} must be numeric") from exc
        if not math.isfinite(config[key]):
            raise ValueError(f"policy_reward.{key} must be finite")
    for key in ("guard_rejection_cap", "repeat_action_cap"):
        try:
            config[key] = int(config[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"policy_reward.{key} must be an integer") from exc
        if config[key] < 0:
            raise ValueError(f"policy_reward.{key} must be non-negative")
    if config["guard_rejection_penalty"] < 0 or config["repeat_action_penalty"] < 0:
        raise ValueError("policy reward behavior penalties must be non-negative")
    if config["minimum"] >= config["maximum"]:
        raise ValueError("policy_reward minimum must be smaller than maximum")
    for reason, raw_value in failure_rewards.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"model failure reward {reason!r} must be numeric") from exc
        if not math.isfinite(value) or not config["minimum"] <= value <= config["maximum"]:
            raise ValueError(f"model failure reward {reason!r} is outside policy reward bounds")
        failure_rewards[str(reason)] = value
    return config


def reward_breakdown(
    state: dict,
    policy_config: object = None,
) -> dict[str, float | bool | str | None]:
    """Convert Reward v3 and public trajectory behavior into the policy reward.

    Environment terminal utility remains an immutable diagnostic. Only the training-side
    ``total`` applies bounded behavior penalties or assigns explicit rewards to model-caused
    terminations. Unknown or infrastructure-caused failures remain invalid and yield no
    learning signal.
    """
    config = validate_policy_reward_config(policy_config)
    normal_terminal = _normal_terminal(state)
    infrastructure_invalid = bool(state.get("infrastructure_invalid"))
    native = float(state.get("final_reward", 0.0)) if normal_terminal else 0.0
    if not math.isfinite(native):
        infrastructure_invalid = True
        native = 0.0

    reward_v3 = state.get("reward_version") == "shopsimulator-reward-v3"
    reward_unverifiable = reward_v3 and not bool(state.get("reward_valid", True))
    valid_terminal = normal_terminal and reward_v3 and not reward_unverifiable
    termination_reason = str(state.get("termination_reason") or "")
    failure_rewards = config["model_failure_rewards"]
    model_failure = (
        not normal_terminal
        and not infrastructure_invalid
        and termination_reason in failure_rewards
    )

    invalid_reason = None
    if infrastructure_invalid:
        invalid_reason = str(state.get("error") or termination_reason or "infrastructure_invalid")
    elif reward_unverifiable:
        invalid_reason = "reward_unverifiable"
    elif not valid_terminal and not model_failure:
        invalid_reason = str(state.get("error") or termination_reason or "missing_terminal_reward")
    valid_for_learning = invalid_reason is None

    detail = (state.get("reward_detail") or {}) if reward_v3 else {}
    gates = detail.get("hard_gates") or {}
    dimension_scores = detail.get("dimension_scores") or {}
    component = lambda name: float(bool(gates.get(name, {}).get("passed")))
    full = float(valid_terminal and state.get("reward_type") == "gold_purchase")
    purchase_success = bool(
        valid_terminal
        and state.get("reward_type")
        in {"gold_purchase", "valid_alternative_purchase"}
    )
    policy_base = native if valid_terminal else float(failure_rewards.get(termination_reason, 0.0))

    guard_count = min(
        int(state.get("guard_rejection_count", 0)),
        config["guard_rejection_cap"],
    )
    repeat_count = min(
        int(state.get("repeat_action_count", 0)),
        config["repeat_action_cap"],
    )
    # Model-failure values are fixed anchors. Behavior penalties only refine otherwise
    # valid environment terminals, so each exceptional termination hits its documented score.
    penalty_guard = (
        guard_count * config["guard_rejection_penalty"] if valid_terminal else 0.0
    )
    penalty_repeat = (
        repeat_count * config["repeat_action_penalty"] if valid_terminal else 0.0
    )
    total = 0.0
    if valid_for_learning:
        total = min(
            config["maximum"],
            max(config["minimum"], policy_base - penalty_guard - penalty_repeat),
        )
    action_attempts = max(int(state.get("action_attempt_count", 0)), 1)
    return {
        "policy_reward_version": POLICY_REWARD_VERSION,
        "policy_base": policy_base,
        "r_type": component("category"),
        "r_att": float(detail.get("weighted_score", 0.0)),
        "r_option": float(dimension_scores.get("key_options", 0.0)),
        "r_price": component("budget"),
        "match_score": float(detail.get("weighted_score", 0.0)),
        "evidence_coverage": float(detail.get("evidence_coverage", 0.0)),
        "brand_score": float(dimension_scores.get("brand", 0.0)),
        "model_score": float(dimension_scores.get("model", 0.0)),
        "core_function_score": float(dimension_scores.get("core_functions", 0.0)),
        "option_score": float(dimension_scores.get("key_options", 0.0)),
        "full": full,
        "strict": full,
        "native": native,
        "semantic": float(purchase_success),
        "efficiency": 0.0,
        "penalty_overlong": abs(policy_base) if termination_reason in {"max_steps", "context_hard_limit_exceeded"} else 0.0,
        "penalty_unfinished": abs(policy_base) if termination_reason == "assistant_finished_without_environment_done" else 0.0,
        "penalty_guard": penalty_guard,
        "penalty_repeat": penalty_repeat,
        "repeat_action_rate": int(state.get("repeat_action_count", 0)) / action_attempts,
        "total": total,
        "terminal_utility": native,
        "purchase_success": float(purchase_success),
        "sampling_invalid": not valid_for_learning,
        "valid_for_learning": valid_for_learning,
        "invalid_reason": invalid_reason,
        "model_failure": model_failure,
        "infrastructure_invalid": infrastructure_invalid,
        "reward_unverifiable": reward_unverifiable,
    }


def terminal_reward(
    state: dict,
    mode: str = "native",
    policy_config: object = None,
) -> float:
    """Return either immutable environment reward or the training-only policy reward."""
    if mode == "policy_v1":
        return float(reward_breakdown(state, policy_config)["total"])
    if mode != "native":
        raise ValueError(f"unknown shopping reward mode: {mode!r}")
    if state.get("infrastructure_invalid") or state.get("error") or not _normal_terminal(state):
        return 0.0
    return float(state.get("final_reward", 0.0))


def task_id_from_kwargs(kwargs: dict) -> int:
    """从 veRL parquet 的 extra_info 读取当前任务，缺失时立即失败。"""
    extra_info = kwargs.get("extra_info")
    if hasattr(extra_info, "item"):
        extra_info = extra_info.item()
    if not isinstance(extra_info, dict) or "task_id" not in extra_info:
        raise ValueError("veRL sample extra_info is missing task_id")
    return int(extra_info["task_id"])
