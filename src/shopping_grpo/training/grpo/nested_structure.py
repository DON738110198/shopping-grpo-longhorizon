"""Outcome-blind structural classification for nested continuation sources.

The classifier in this module is intentionally pure: its answer is a function
only of a validated active-branch plan and the corresponding stage-one JSONL
records.  It never imports the rollout runner, parser, environment, or capture
code.  Both collection and offline credit estimation can therefore recompute
the same state-level no-backfill partition instead of trusting serialized
exclusion reasons.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from shopping_grpo.training.grpo.active_branch import validate_active_branch_plan
from shopping_grpo.training.grpo.pivotal_states import (
    canonical_replay_action,
    replay_action_sha256,
    token_ids_sha256,
)

NESTED_EXCLUSION_AUDIT_VERSION = "shopping-nested-state-exclusion-audit-v2"
NESTED_STAGE1_STRUCTURE_VERSION = "shopping-nested-stage1-structure-v1"
NESTED_DECISION_IDENTITY_VERSION = "shopping-nested-decision-identity-v1"
EXPECTED_STAGE1_RESULT_VERSION = "shopping-active-suffix-result-v1"

# Only these outcome-independent defects can remove a pre-registered state.
# Parser/runtime drift is deliberately absent: it is an infrastructure error
# that must abort collection rather than silently changing the state sample.
NESTED_STRUCTURE_EXCLUSION_REASONS = frozenset(
    {
        "stage1_plan_identity_invalid",
        "stage1_task_identity_invalid",
        "pre_action_prompt_invalid",
        "first_decision_tensors_invalid",
        "first_decision_span_invalid",
        "first_decision_action_invalid",
        "first_request_provenance_invalid",
        "duplicate_decision_old_logprobs_mismatch",
    }
)

_FORBIDDEN_KEYS = {
    "goal",
    "gold",
    "gold_asin",
    "hidden_goal",
    "reward_goal",
    "target",
    "target_asin",
}


class NestedStage1ContractError(ValueError):
    """The plan/JSONL source set is not a complete pre-registered partition."""


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


def _integer_tokens(
    value: object,
    *,
    allow_empty: bool = False,
) -> list[int] | None:
    if not isinstance(value, list):
        return None
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in value
    ):
        return None
    if not value and not allow_empty:
        return None
    return list(value)


def _finite_logprobs(value: object, count: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != count:
        return None
    result = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        result.append(number)
    return result


def _proposal_uid(state_uid: str, proposal_index: int) -> str:
    return _sha256_json(
        {
            "state_uid": state_uid,
            "proposal_index": proposal_index,
        }
    )


def _decision_uid(
    state_uid: str,
    prompt: Sequence[int],
    first_tokens: Sequence[int],
    action_sha256: str,
) -> str:
    return _sha256_json(
        {
            "version": NESTED_DECISION_IDENTITY_VERSION,
            "state_uid": state_uid,
            "pre_action_prompt_sha256": token_ids_sha256(prompt),
            "first_assistant_token_ids": list(first_tokens),
            "first_action_sha256": action_sha256,
        }
    )


def _reason(
    code: str,
    proposal_uids: Sequence[str],
    *,
    decision_uid: str | None = None,
) -> dict[str, object]:
    if code not in NESTED_STRUCTURE_EXCLUSION_REASONS:
        raise AssertionError(f"unregistered nested exclusion reason: {code}")
    return {
        "code": code,
        "affected_proposal_uids": sorted(set(proposal_uids)),
        "decision_uid": decision_uid,
    }


def _record_reason(
    reasons_by_state: dict[str, list[dict[str, object]]],
    state_uid: str,
    reason: dict[str, object],
) -> None:
    reasons = reasons_by_state.setdefault(state_uid, [])
    if reason not in reasons:
        reasons.append(reason)


def _classify_proposal(
    record: Mapping[str, object],
    *,
    plan: Mapping[str, object],
    group: Mapping[str, object],
    suffix: Mapping[str, object],
    proposal_uid: str,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Return decision evidence or one bounded structural exclusion reason."""
    identities = {
        "suffix_uid": suffix["suffix_uid"],
        "parent_branch_uid": group["parent_branch_uid"],
        "replay_state_id": group["replay_state_id"],
        "actor_checkpoint_sha256": plan["actor_checkpoint_sha256"],
        "decoding_config_sha256": plan["decoding_config_sha256"],
        "sampling_backend_contract_sha256": plan["decoding_config"][
            "sampling_backend_contract_sha256"
        ],
        "prompt_token_sha256": group["actor_prompt_tokens"]["sha256"],
    }
    if any(record.get(name) != value for name, value in identities.items()):
        return None, _reason("stage1_plan_identity_invalid", [proposal_uid])
    if record.get("task_id") != group["task_id"]:
        return None, _reason("stage1_task_identity_invalid", [proposal_uid])

    prompt = _integer_tokens(record.get("prompt_token_ids"))
    if (
        prompt is None
        or prompt != group["actor_prompt_tokens"]["tokens"]
        or token_ids_sha256(prompt) != record.get("prompt_token_sha256")
    ):
        return None, _reason("pre_action_prompt_invalid", [proposal_uid])

    response = _integer_tokens(record.get("response_ids"))
    if response is None:
        return None, _reason("first_decision_tensors_invalid", [proposal_uid])
    old_logprobs = _finite_logprobs(record.get("old_logprobs"), len(response))
    response_mask = record.get("response_mask")
    if (
        old_logprobs is None
        or not isinstance(response_mask, list)
        or len(response_mask) != len(response)
        or any(value not in {0, 1} for value in response_mask)
    ):
        return None, _reason("first_decision_tensors_invalid", [proposal_uid])

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
        return None, _reason("first_decision_span_invalid", [proposal_uid])

    action = record.get("first_action")
    if not isinstance(action, Mapping):
        return None, _reason("first_decision_action_invalid", [proposal_uid])
    try:
        canonical_action = canonical_replay_action(
            action.get("tool"), action.get("parameters")
        )
        action_sha256 = replay_action_sha256(
            canonical_action["tool"], canonical_action["parameters"]
        )
    except (TypeError, ValueError):
        return None, _reason("first_decision_action_invalid", [proposal_uid])
    if canonical_action != action or action_sha256 != record.get("first_action_sha256"):
        return None, _reason("first_decision_action_invalid", [proposal_uid])

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
        return None, _reason("first_request_provenance_invalid", [proposal_uid])

    first_tokens = response[: span[1]]
    decision_uid = _decision_uid(
        str(group["active_group_uid"]), prompt, first_tokens, action_sha256
    )
    return (
        {
            "decision_uid": decision_uid,
            "first_assistant_old_logprobs": old_logprobs[: span[1]],
            "proposal_uid": proposal_uid,
        },
        None,
    )


def classify_nested_stage1_structure(
    plan: Mapping[str, object],
    stage1_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Classify every pre-registered state without reading rollout outcomes.

    Any structural defect excludes the whole state and all of its proposals;
    there is no replacement or backfill.  Source-set/provenance corruption that
    prevents a complete plan partition raises ``NestedStage1ContractError``.
    """
    normalized_plan = validate_active_branch_plan(plan)
    if isinstance(stage1_records, (str, bytes)) or not isinstance(
        stage1_records, Sequence
    ):
        raise TypeError("stage1_records must be a sequence")

    groups = normalized_plan["groups"]
    groups_by_uid = {group["active_group_uid"]: group for group in groups}
    expected_proposals = sum(len(group["suffixes"]) for group in groups)
    if len(stage1_records) != expected_proposals:
        raise NestedStage1ContractError(
            "stage-one proposal count does not match the active plan"
        )

    source_by_state: dict[str, list[dict[str, object]]] = defaultdict(list)
    decisions_by_state: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    reasons_by_state: dict[str, list[dict[str, object]]] = {}
    seen_sources: set[tuple[str, int]] = set()

    for record in stage1_records:
        if not isinstance(record, Mapping):
            raise TypeError("stage-one record must be an object")
        forbidden = _find_forbidden_keys(record)
        if forbidden:
            raise NestedStage1ContractError(
                "stage-one record contains a forbidden hidden-goal field"
            )
        if record.get("schema_version") != EXPECTED_STAGE1_RESULT_VERSION:
            raise NestedStage1ContractError("stage-one result version mismatch")
        if (
            record.get("optimizer_enabled") is not False
            or record.get("uses_hidden_goal") is not False
        ):
            raise NestedStage1ContractError("stage-one safety contract is invalid")

        state_uid = record.get("active_group_uid")
        group = groups_by_uid.get(state_uid)
        if group is None:
            raise NestedStage1ContractError(
                "stage-one record group is not in the active plan"
            )
        suffix_index = record.get("suffix_index")
        if (
            not isinstance(suffix_index, int)
            or isinstance(suffix_index, bool)
            or not 0 <= suffix_index < len(group["suffixes"])
        ):
            raise NestedStage1ContractError("stage-one suffix index is invalid")
        source_key = (str(state_uid), suffix_index)
        if source_key in seen_sources:
            raise NestedStage1ContractError("stage-one decision source is duplicated")
        seen_sources.add(source_key)

        suffix = group["suffixes"][suffix_index]
        proposal_uid = _proposal_uid(str(state_uid), suffix_index)
        source_by_state[str(state_uid)].append(
            {
                "proposal_index": suffix_index,
                "proposal_uid": proposal_uid,
                "stage1_record_sha256": _sha256_json(dict(record)),
            }
        )
        decision, reason = _classify_proposal(
            record,
            plan=normalized_plan,
            group=group,
            suffix=suffix,
            proposal_uid=proposal_uid,
        )
        if reason is not None:
            _record_reason(reasons_by_state, str(state_uid), reason)
        elif decision is not None:
            decisions_by_state[str(state_uid)][str(decision["decision_uid"])].append(
                decision
            )

    expected_sources = {
        (str(group["active_group_uid"]), int(suffix["suffix_index"]))
        for group in groups
        for suffix in group["suffixes"]
    }
    if seen_sources != expected_sources:
        raise NestedStage1ContractError(
            "stage-one sources do not cover the active plan exactly"
        )

    for state_uid, by_decision in decisions_by_state.items():
        for decision_uid, sources in by_decision.items():
            old_logprobs = {
                _canonical_json(source["first_assistant_old_logprobs"])
                for source in sources
            }
            if len(old_logprobs) > 1:
                _record_reason(
                    reasons_by_state,
                    state_uid,
                    _reason(
                        "duplicate_decision_old_logprobs_mismatch",
                        [str(source["proposal_uid"]) for source in sources],
                        decision_uid=decision_uid,
                    ),
                )

    pre_registered_state_uids = set(groups_by_uid)
    excluded_state_uids = set(reasons_by_state)
    eligible_state_uids = pre_registered_state_uids - excluded_state_uids
    pre_registered_proposal_uids = {
        source["proposal_uid"]
        for sources in source_by_state.values()
        for source in sources
    }
    eligible_proposal_uids = {
        source["proposal_uid"]
        for state_uid in eligible_state_uids
        for source in source_by_state[state_uid]
    }
    excluded_proposal_uids = pre_registered_proposal_uids - eligible_proposal_uids

    reason_counts = Counter(
        str(reason["code"])
        for reasons in reasons_by_state.values()
        for reason in reasons
    )
    excluded_state_records = []
    for state_uid in sorted(excluded_state_uids):
        group = groups_by_uid[state_uid]
        sources = sorted(
            source_by_state[state_uid], key=lambda item: item["proposal_index"]
        )
        excluded_state_records.append(
            {
                "state_uid": state_uid,
                "task_id": group["task_id"],
                "proposal_uids": [source["proposal_uid"] for source in sources],
                "source_record_sha256s": [
                    source["stage1_record_sha256"] for source in sources
                ],
                "reasons": sorted(
                    reasons_by_state[state_uid],
                    key=lambda item: (
                        item["code"],
                        item["decision_uid"] or "",
                        item["affected_proposal_uids"],
                    ),
                ),
                "no_backfill": True,
            }
        )

    audit = {
        "schema_version": NESTED_EXCLUSION_AUDIT_VERSION,
        "pre_registered_states": len(pre_registered_state_uids),
        "pre_registered_proposals": len(pre_registered_proposal_uids),
        "eligible_states": len(eligible_state_uids),
        "eligible_proposals": len(eligible_proposal_uids),
        "excluded_states": len(excluded_state_uids),
        "excluded_proposals": len(excluded_proposal_uids),
        "reason_counts": dict(sorted(reason_counts.items())),
        "excluded_state_records": excluded_state_records,
        "proposal_partition_sha256": _sha256_json(
            {
                "pre_registered": sorted(pre_registered_proposal_uids),
                "eligible": sorted(eligible_proposal_uids),
                "excluded": sorted(excluded_proposal_uids),
            }
        ),
        "accounting_verified": True,
        "no_backfill": True,
    }
    return {
        "schema_version": NESTED_STAGE1_STRUCTURE_VERSION,
        "eligible_state_uids": sorted(eligible_state_uids),
        "excluded_state_uids": sorted(excluded_state_uids),
        "eligible_proposal_uids": sorted(eligible_proposal_uids),
        "excluded_proposal_uids": sorted(excluded_proposal_uids),
        "exclusion_audit": audit,
    }
