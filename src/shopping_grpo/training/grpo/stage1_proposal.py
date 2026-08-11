"""First-decision-only Stage-1 sampling for nested PSA credit.

Stage 1 samples exactly one Assistant turn from each pre-registered prompt.  It
never executes the sampled action, requests a downstream turn, or reads a
terminal reward.  The resulting compatibility records contain the tensor and
identity fields consumed by the existing nested structural classifier.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path

from shopping_grpo.training.grpo.active_branch import (
    build_active_branch_plan,
    validate_active_branch_plan,
    validate_actor_prompt_tokens,
)
from shopping_grpo.training.grpo.active_suffix import (
    ACTIVE_SUFFIX_RESULT_VERSION,
    ActiveSuffixInfrastructureError,
    _finite_logprobs,
    _is_sha256,
    _normalize_tool_calls,
    _token_ids,
    sampling_backend_contract_sha256,
    sha256_actor_checkpoint,
    tool_schema_sha256,
    validate_sampling_backend_contract,
)
from shopping_grpo.training.grpo.adapter.runtime import (
    validate_policy_reward_config,
)
from shopping_grpo.training.grpo.nested_structure import (
    classify_nested_stage1_structure,
)
from shopping_grpo.training.grpo.pivotal_states import (
    canonical_replay_action,
    replay_action_sha256,
)

FIRST_DECISION_PROPOSAL_VERSION = "shopping-first-decision-proposal-v1"
FIRST_DECISION_COLLECTION_VERSION = "shopping-first-decision-collection-v1"
FIRST_DECISION_JOURNAL_VERSION = "shopping-first-decision-journal-v4"
FIRST_DECISION_JOURNAL_ENTRY_VERSION = "shopping-first-decision-journal-entry-v2"
FIRST_DECISION_ATTEMPT_VERSION = "shopping-first-decision-attempt-v1"
FIRST_DECISION_ACTOR_RUN_START_VERSION = (
    "shopping-first-decision-actor-run-start-v1"
)
FIRST_DECISION_ACTOR_RUN_END_VERSION = "shopping-first-decision-actor-run-end-v1"
FIRST_DECISION_ACTOR_RUN_CHAIN_VERSION = (
    "shopping-first-decision-actor-run-chain-v1"
)
FIRST_DECISION_JOURNAL_COMPLETE_VERSION = (
    "shopping-first-decision-journal-complete-v2"
)
FIRST_DECISION_JOURNAL_ARTIFACT_VERSION = (
    "shopping-first-decision-journal-artifact-v2"
)
FIRST_DECISION_FINAL_MANIFEST_VERSION = "shopping-first-decision-final-manifest-v1"
FIRST_DECISION_RECORD_CONTENT_VERSION = "shopping-first-decision-record-content-v1"
FIRST_DECISION_SOURCE_PROVENANCE_VERSION = (
    "shopping-first-decision-source-provenance-v2"
)

_OUTCOME_FIELDS = {
    "strict",
    "terminal_utility",
    "policy_reward",
    "reward_type",
    "reward_detail",
    "reward",
    "reward_version",
    "reward_valid",
    "reward_components",
    "final_reward",
    "purchase_success",
    "done",
    "over",
    "termination_reason",
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_nonsymlink_artifact_path(
    value: str | Path,
    *,
    label: str,
) -> tuple[Path, Path]:
    """Lstat the caller-supplied path before resolving any filesystem alias."""
    literal = Path(value).expanduser()
    try:
        metadata = os.lstat(literal)
    except FileNotFoundError:
        metadata = None
    if metadata is not None and stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"Stage-1 {label} must not be a symbolic link")
    return literal, literal.resolve()


def _recheck_nonsymlink_artifact_path(
    literal: Path,
    resolved: Path,
    *,
    label: str,
) -> None:
    try:
        metadata = os.lstat(literal)
    except FileNotFoundError as exc:
        raise ValueError(f"Stage-1 {label} is missing") from exc
    if stat.S_ISLNK(metadata.st_mode) or literal.resolve() != resolved:
        raise ValueError(f"Stage-1 {label} path changed or became symbolic")


def _proposal_uid(state_uid: str, proposal_index: int) -> str:
    return _sha256_json(
        {"state_uid": state_uid, "proposal_index": proposal_index}
    )


def _slot_uid(
    plan_sha256: str,
    state_uid: str,
    proposal_index: int,
    proposal_uid: str,
) -> str:
    return _sha256_json(
        {
            "plan_sha256": plan_sha256,
            "active_group_uid": state_uid,
            "proposal_index": proposal_index,
            "proposal_uid": proposal_uid,
        }
    )


def _attempt_uid(
    journal_contract_sha256: str,
    slot_uid: str,
    actor_run_uid: str,
) -> str:
    return _sha256_json(
        {
            "journal_contract_sha256": journal_contract_sha256,
            "slot_uid": slot_uid,
            "actor_run_uid": actor_run_uid,
        }
    )


def _slot_specs_for_plan(
    plan: Mapping[str, object],
) -> list[dict[str, object]]:
    plan_sha256 = _sha256_json(plan)
    specs = []
    for group_index, group in enumerate(plan["groups"]):
        for proposal_index, suffix in enumerate(group["suffixes"]):
            proposal_uid = _proposal_uid(
                group["active_group_uid"], proposal_index
            )
            specs.append(
                {
                    "group_index": group_index,
                    "proposal_index": proposal_index,
                    "proposal_uid": proposal_uid,
                    "slot_uid": _slot_uid(
                        plan_sha256,
                        group["active_group_uid"],
                        proposal_index,
                        proposal_uid,
                    ),
                    "active_group_uid": group["active_group_uid"],
                    "suffix_uid": suffix["suffix_uid"],
                    "seed": suffix["seed"],
                }
            )
    return specs


def _build_journal_contract(
    *,
    plan: Mapping[str, object],
    slot_specs: Sequence[Mapping[str, object]],
    actor_tokenizer_contract_sha256: str,
    sampling_backend_contract_sha256: str,
    required_environment_version: str,
    environment_manifest_sha256s: Sequence[str],
    policy_reward_sha256: str,
    harness_contract_sha256: str,
    source_provenance: Mapping[str, object],
) -> dict[str, object]:
    normalized_specs = [
        json.loads(_canonical_json(dict(spec))) for spec in slot_specs
    ]
    slot_uid_schedule = [str(spec["slot_uid"]) for spec in normalized_specs]
    contract: dict[str, object] = {
        "schema_version": FIRST_DECISION_JOURNAL_VERSION,
        "proposal_version": FIRST_DECISION_PROPOSAL_VERSION,
        "plan_schema_version": plan["schema_version"],
        "plan_strategy_version": plan["strategy_version"],
        "plan_sha256": _sha256_json(plan),
        "actor_checkpoint_sha256": plan["actor_checkpoint_sha256"],
        "actor_tokenizer_contract_sha256": actor_tokenizer_contract_sha256,
        "decoding_config_sha256": plan["decoding_config_sha256"],
        "sampling_backend_contract_sha256": (
            sampling_backend_contract_sha256
        ),
        "required_environment_version": required_environment_version,
        "environment_manifest_sha256s": list(environment_manifest_sha256s),
        "policy_reward_sha256": policy_reward_sha256,
        "tool_schema_sha256": plan["decoding_config"]["tool_schema_sha256"],
        "harness_contract_sha256": harness_contract_sha256,
        "expected_states": len(plan["groups"]),
        "proposals_per_state": plan["suffixes_per_state"],
        "expected_proposals": len(normalized_specs),
        "slot_schedule": normalized_specs,
        "slot_uid_schedule": slot_uid_schedule,
        "slot_uid_schedule_sha256": _sha256_json(slot_uid_schedule),
        "source_provenance": json.loads(
            _canonical_json(dict(source_provenance))
        ),
        "scale_ready": _source_scale_ready(source_provenance),
        "optimizer_enabled": False,
        "outcome_fields_read": [],
        "uses_hidden_goal": False,
    }
    contract["contract_sha256"] = _sha256_json(contract)
    return contract


def _record_content_sha256(record: Mapping[str, object]) -> str:
    payload = dict(record)
    payload.pop("stage1_record_content_sha256", None)
    return _sha256_json(
        {"version": FIRST_DECISION_RECORD_CONTENT_VERSION, "record": payload}
    )


def _with_record_content_sha256(record: dict[str, object]) -> dict[str, object]:
    if "stage1_record_content_sha256" in record:
        raise ValueError("Stage-1 record content hash must be added exactly once")
    if _OUTCOME_FIELDS.intersection(record):
        raise ValueError("first-decision record must not contain rollout outcomes")
    record["stage1_record_content_sha256"] = _record_content_sha256(record)
    return record


def _normalize_source_provenance(value: Mapping[str, object] | None) -> dict[str, object]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError("source_provenance must be an object")
    normalized = json.loads(_canonical_json(dict(value)))
    git_sha = normalized.get("git_sha")
    if git_sha is None:
        raise ValueError("source_provenance must bind a full git_sha")
    if (
        not isinstance(git_sha, str)
        or len(git_sha) != 40
        or any(character not in "0123456789abcdef" for character in git_sha)
    ):
        raise ValueError("source_provenance git_sha must be a 40-character commit")
    _source_scale_ready(normalized)
    return normalized


def _source_scale_ready(source_provenance: Mapping[str, object]) -> bool:
    schema_version = source_provenance.get("schema_version")
    if schema_version != FIRST_DECISION_SOURCE_PROVENANCE_VERSION:
        return False
    execution_mode = source_provenance.get("execution_mode")
    clean = source_provenance.get("git_worktree_clean")
    dirty_override = source_provenance.get("dirty_source_override")
    declared_ready = source_provenance.get("scale_ready")
    if (
        execution_mode not in {"formal", "mechanical"}
        or not isinstance(clean, bool)
        or not isinstance(dirty_override, bool)
        or not isinstance(declared_ready, bool)
    ):
        raise ValueError("source_provenance readiness fields are invalid")
    expected_ready = execution_mode == "formal" and clean and not dirty_override
    if declared_ready != expected_ready:
        raise ValueError("source_provenance scale readiness is inconsistent")
    if execution_mode == "formal" and (not clean or dirty_override):
        raise ValueError("formal Stage-1 source must use a clean git worktree")
    if dirty_override and execution_mode != "mechanical":
        raise ValueError("dirty source override is restricted to mechanical runs")
    return declared_ready


def _build_harness_contract(
    decoding_config: Mapping[str, object],
    *,
    policy_reward_sha256: str,
) -> dict[str, object]:
    """Mirror the nested v2 harness identity without importing its collector."""
    missing = set(_HARNESS_CONFIG_FIELDS) - set(decoding_config)
    if missing:
        raise ValueError("decoding_config is missing Stage-1 harness fields")
    if not _is_sha256(policy_reward_sha256):
        raise ValueError("policy_reward_sha256 must be a SHA256 digest")
    return {
        "schema_version": "shopping-nested-harness-contract-v2",
        "decision_boundary": "after-first-decision-effect-v2",
        "prefix_runtime_restore": "shopping-active-prefix-runtime-v1",
        "recent_action_window": 3,
        "max_consecutive_guard_rejections": 3,
        "policy_reward_sha256": policy_reward_sha256,
        "config": {
            name: deepcopy(decoding_config[name])
            for name in _HARNESS_CONFIG_FIELDS
        },
    }


def _prefix_budget_context(
    resolved: Mapping[str, object],
    group: Mapping[str, object],
    config: Mapping[str, object],
) -> dict[str, int]:
    """Read only the captured turn locator and budgets before the selected event."""
    trajectory = resolved.get("trajectory")
    event = resolved.get("event")
    source = group.get("source")
    if not all(isinstance(item, Mapping) for item in (trajectory, event, source)):
        raise ValueError("resolved Stage-1 branch is incomplete")
    if (
        event.get("branch_uid") != group.get("parent_branch_uid")
        or event.get("replay_state_id") != group.get("replay_state_id")
        or event.get("actor_prompt_sha256")
        != group["actor_prompt_tokens"]["sha256"]
    ):
        raise ValueError("resolved Stage-1 event identity differs from the plan")
    event_index = source.get("event_index")
    events = trajectory.get("decision_trace")
    if (
        not isinstance(event_index, int)
        or isinstance(event_index, bool)
        or not isinstance(events, list)
        or not 0 <= event_index < len(events)
        or not isinstance(events[event_index], Mapping)
        or _canonical_json(dict(events[event_index]))
        != _canonical_json(dict(event))
    ):
        raise ValueError("resolved Stage-1 event locator is invalid")
    previous_turn_id = -1
    executed_steps = 0
    consecutive_guard_rejections = 0
    for decision_index, prefix_event in enumerate(events[:event_index]):
        if (
            not isinstance(prefix_event, Mapping)
            or prefix_event.get("decision_index") != decision_index
        ):
            raise ValueError("resolved Stage-1 prefix decisions are not contiguous")
        prefix_turn_id = prefix_event.get("assistant_turn_id")
        if (
            not isinstance(prefix_turn_id, int)
            or isinstance(prefix_turn_id, bool)
            or prefix_turn_id <= previous_turn_id
        ):
            raise ValueError("resolved Stage-1 prefix turns are not monotonic")
        previous_turn_id = prefix_turn_id
        decision_kind = prefix_event.get("decision_kind")
        if decision_kind == "think":
            executed_steps += 1
        elif decision_kind == "environment_tool":
            accepted = prefix_event.get("accepted")
            if accepted is True:
                executed_steps += 1
                consecutive_guard_rejections = 0
            elif accepted is False:
                consecutive_guard_rejections += 1
            else:
                raise ValueError("resolved Stage-1 prefix action outcome is missing")
        else:
            raise ValueError("resolved Stage-1 prefix contains a terminal decision")
    if executed_steps >= int(config["max_steps"]):
        raise ValueError("resolved Stage-1 branch exhausted max_steps")
    if consecutive_guard_rejections >= 3:
        raise ValueError("resolved Stage-1 branch follows terminal guard state")
    turn_id = event.get("assistant_turn_id")
    spans = trajectory.get("turn_spans")
    if (
        not isinstance(turn_id, int)
        or isinstance(turn_id, bool)
        or turn_id < 0
        or previous_turn_id >= turn_id
        or not isinstance(spans, list)
    ):
        raise ValueError("resolved Stage-1 turn locator is invalid")
    if event.get("decision_index") != event_index:
        raise ValueError("resolved Stage-1 decision index changed")
    matches = [
        span
        for span in spans
        if isinstance(span, Mapping) and span.get("turn_id") == turn_id
    ]
    if len(matches) != 1:
        raise ValueError("resolved Stage-1 turn span is missing or ambiguous")
    assistant_span = matches[0].get("assistant_span")
    if (
        matches[0].get("credit_eligible") is not True
        or not isinstance(assistant_span, list)
        or len(assistant_span) != 2
        or any(
            not isinstance(item, int) or isinstance(item, bool)
            for item in assistant_span
        )
        or not 0 <= assistant_span[0] < assistant_span[1]
    ):
        raise ValueError("resolved Stage-1 Assistant span is invalid")
    response_tokens_before = assistant_span[0]
    if response_tokens_before >= int(config["response_length"]):
        raise ValueError("resolved Stage-1 branch exhausted response_length")
    if turn_id >= int(config["max_assistant_turns"]):
        raise ValueError("resolved Stage-1 branch exhausted max_assistant_turns")
    if turn_id >= int(config["max_user_turns"]):
        raise ValueError("resolved Stage-1 branch exhausted max_user_turns")
    prompt_capture = group.get("actor_prompt_tokens")
    if not isinstance(prompt_capture, Mapping) or int(prompt_capture["count"]) > (
        int(config["context_window"])
        - int(config["context_generation_reserve"])
        - int(config["context_safety_margin"])
    ):
        raise ValueError("resolved Stage-1 branch exceeds the context hard limit")
    return {
        "assistant_turns": turn_id,
        "user_turns": turn_id,
        "response_tokens_before": response_tokens_before,
        "executed_steps": executed_steps,
        "consecutive_guard_rejections": consecutive_guard_rejections,
    }


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _checkpoint_stat_snapshot(root: Path) -> dict[str, object]:
    if root.is_symlink() or not root.is_dir():
        raise ActiveSuffixInfrastructureError(
            "Stage-1 actor checkpoint root is missing or symbolic",
            code="actor_attestation_invalid",
        )
    entries = []
    for path in [root, *sorted(root.rglob("*"))]:
        if path.is_symlink():
            raise ActiveSuffixInfrastructureError(
                "Stage-1 actor checkpoint gained a symbolic link",
                code="actor_attestation_invalid",
            )
        stat = path.stat()
        if path.is_dir():
            kind = "directory"
        elif path.is_file():
            kind = "file"
        else:
            raise ActiveSuffixInfrastructureError(
                "Stage-1 actor checkpoint has an unsupported entry",
                code="actor_attestation_invalid",
            )
        entries.append(
            {
                "path": "." if path == root else path.relative_to(root).as_posix(),
                "kind": kind,
                "mode": int(stat.st_mode),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "ctime_ns": int(stat.st_ctime_ns),
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
            }
        )
    return {
        "entry_count": len(entries),
        "entries_sha256": _sha256_json(entries),
        "entries": entries,
    }


class _Stage1ActorRunAttestation:
    """Full-hash one process run and stat-check every completion boundary."""

    def __init__(self, actor_checkpoint: Path, expected_sha256: str) -> None:
        if not _is_sha256(expected_sha256):
            raise ValueError("Stage-1 expected actor SHA256 is invalid")
        requested = actor_checkpoint.expanduser()
        if requested.is_symlink():
            raise ActiveSuffixInfrastructureError(
                "Stage-1 actor checkpoint root is symbolic",
                code="actor_attestation_invalid",
            )
        self.root = requested.resolve()
        self.expected_sha256 = expected_sha256
        self.run_uid = _sha256_json(
            {
                "actor_checkpoint_sha256": expected_sha256,
                "process_id": os.getpid(),
                "nonce": uuid.uuid4().hex,
            }
        )
        before = _checkpoint_stat_snapshot(self.root)
        start_full_sha256 = sha256_actor_checkpoint(self.root)
        after = _checkpoint_stat_snapshot(self.root)
        if before != after:
            raise ActiveSuffixInfrastructureError(
                "Stage-1 actor changed during run-start attestation",
                code="actor_attestation_start_drift",
            )
        if start_full_sha256 != expected_sha256:
            raise ActiveSuffixInfrastructureError(
                "Stage-1 actor run-start SHA256 mismatch",
                code="actor_attestation_start_mismatch",
            )
        self.baseline_snapshot = before
        self.stat_snapshot_sha256 = _sha256_json(before)
        self.start_full_sha256 = start_full_sha256
        self.runtime_stat_checks = 0
        self.completion_request_stat_checks = 0
        self.finished = False

    def start_report(self) -> dict[str, object]:
        return {
            "actor_run_uid": self.run_uid,
            "actor_checkpoint_sha256": self.expected_sha256,
            "start_full_sha256": self.start_full_sha256,
            "stat_snapshot_sha256": self.stat_snapshot_sha256,
            "stat_entry_count": self.baseline_snapshot["entry_count"],
            "read_only_run_binding": True,
        }

    def backend_binding(self) -> dict[str, object]:
        return {
            "actor_checkpoint_sha256": self.expected_sha256,
            "actor_run_uid": self.run_uid,
            "stat_snapshot_sha256": self.stat_snapshot_sha256,
            "stat_entry_count": self.baseline_snapshot["entry_count"],
            "read_only_run_binding": True,
        }

    def verify_runtime(self, *, completion_request: bool = False) -> None:
        if self.finished:
            raise ActiveSuffixInfrastructureError(
                "Stage-1 actor run attestation is closed",
                code="actor_attestation_closed",
            )
        if _checkpoint_stat_snapshot(self.root) != self.baseline_snapshot:
            raise ActiveSuffixInfrastructureError(
                "Stage-1 actor checkpoint changed during collection",
                code="actor_attestation_invalid",
            )
        self.runtime_stat_checks += 1
        if completion_request:
            self.completion_request_stat_checks += 1

    def finish(self) -> dict[str, object]:
        self.verify_runtime()
        before = _checkpoint_stat_snapshot(self.root)
        end_full_sha256 = sha256_actor_checkpoint(self.root)
        after = _checkpoint_stat_snapshot(self.root)
        if before != self.baseline_snapshot or after != self.baseline_snapshot:
            raise ActiveSuffixInfrastructureError(
                "Stage-1 actor changed during run-end attestation",
                code="actor_attestation_end_drift",
            )
        if end_full_sha256 != self.expected_sha256:
            raise ActiveSuffixInfrastructureError(
                "Stage-1 actor run-end SHA256 mismatch",
                code="actor_attestation_end_mismatch",
            )
        self.finished = True
        return {
            **self.start_report(),
            "end_full_sha256": end_full_sha256,
            "runtime_stat_checks": self.runtime_stat_checks,
            "completion_request_stat_checks": (
                self.completion_request_stat_checks
            ),
            "completed": True,
            "drift_detected": False,
        }


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link publishes the fully fsynced inode without overwriting an
        # existing slot on POSIX or Windows.  That makes concurrent/stale
        # resume attempts fail closed instead of replacing collected evidence.
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class _FirstDecisionJournal:
    def __init__(
        self,
        root: Path,
        *,
        contract: Mapping[str, object],
        slot_specs: Sequence[Mapping[str, object]],
        resume: bool,
    ) -> None:
        requested_root = root.expanduser()
        if requested_root.is_symlink():
            raise ValueError("Stage-1 journal root must not be a symbolic link")
        self.root = requested_root.resolve()
        self.contract_path = self.root / "journal_contract.json"
        self.attempts_root = self.root / "attempts"
        self.slots_root = self.root / "slots"
        self.runs_root = self.root / "actor_runs"
        self.complete_path = self.root / "journal_complete.json"
        self.contract = json.loads(_canonical_json(dict(contract)))
        contract_payload = dict(self.contract)
        contract_sha256 = contract_payload.pop("contract_sha256", None)
        if contract_sha256 != _sha256_json(contract_payload):
            raise ValueError("Stage-1 journal contract content hash mismatch")
        self.ordered_slot_specs = [
            json.loads(_canonical_json(dict(slot))) for slot in slot_specs
        ]
        self.slot_specs = {
            str(slot["slot_uid"]): slot for slot in self.ordered_slot_specs
        }
        if len(self.slot_specs) != len(self.ordered_slot_specs):
            raise ValueError("Stage-1 journal repeats a slot identity")
        slot_uid_schedule = [
            str(spec["slot_uid"]) for spec in self.ordered_slot_specs
        ]
        if (
            self.contract.get("schema_version")
            != FIRST_DECISION_JOURNAL_VERSION
            or self.contract.get("slot_schedule") != self.ordered_slot_specs
            or self.contract.get("slot_uid_schedule") != slot_uid_schedule
            or self.contract.get("slot_uid_schedule_sha256")
            != _sha256_json(slot_uid_schedule)
            or self.contract.get("expected_proposals")
            != len(self.ordered_slot_specs)
        ):
            raise ValueError("Stage-1 journal UID schedule differs from its contract")
        if self.root.exists() and not self.root.is_dir():
            raise ValueError("Stage-1 journal path must be a directory")
        if not resume:
            if self.root.exists() and any(self.root.iterdir()):
                raise ValueError("Stage-1 journal exists; pass resume=True to continue")
            self.attempts_root.mkdir(parents=True, exist_ok=True)
            self.slots_root.mkdir(parents=True, exist_ok=True)
            self.runs_root.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(self.contract_path, self.contract)
        else:
            if (
                self.contract_path.is_symlink()
                or self.attempts_root.is_symlink()
                or self.slots_root.is_symlink()
                or self.runs_root.is_symlink()
                or not self.contract_path.is_file()
                or not self.attempts_root.is_dir()
                or not self.slots_root.is_dir()
                or not self.runs_root.is_dir()
            ):
                raise ValueError("resumed Stage-1 journal is incomplete")
            try:
                existing_contract = json.loads(
                    self.contract_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("resumed Stage-1 journal header is invalid") from exc
            if existing_contract != self.contract:
                raise ValueError("resumed Stage-1 journal contract changed")
        self._validate_root_layout()
        self.actor_run_starts, self.actor_run_ends = self._load_actor_runs()
        self.attempts = self._load_attempts()
        self.entries = self._load_entries()
        self.active_run_uid: str | None = None
        if self.complete_path.exists():
            self._validate_complete_marker()

    @staticmethod
    def _entry_sha256(entry: Mapping[str, object]) -> str:
        payload = dict(entry)
        payload.pop("entry_sha256", None)
        return _sha256_json(payload)

    def _slot_path(self, spec: Mapping[str, object]) -> Path:
        return self.slots_root / (
            f"{int(spec['group_index']):06d}-"
            f"{int(spec['proposal_index']):03d}-"
            f"{spec['slot_uid']}.json"
        )

    def _attempt_path(self, spec: Mapping[str, object]) -> Path:
        return self.attempts_root / (
            f"{int(spec['group_index']):06d}-"
            f"{int(spec['proposal_index']):03d}-"
            f"{spec['slot_uid']}.json"
        )

    @staticmethod
    def _is_crash_temporary(path: Path) -> bool:
        return path.is_file() and path.name.startswith(".") and ".tmp." in path.name

    def _validate_root_layout(self) -> None:
        allowed = {
            self.contract_path.name,
            self.attempts_root.name,
            self.slots_root.name,
            self.runs_root.name,
            self.complete_path.name,
        }
        unexpected = [
            path
            for path in self.root.iterdir()
            if path.name not in allowed and not self._is_crash_temporary(path)
        ]
        if unexpected:
            raise ValueError("Stage-1 journal contains an unexpected root path")
        if (
            self.contract_path.is_symlink()
            or self.attempts_root.is_symlink()
            or self.slots_root.is_symlink()
            or self.runs_root.is_symlink()
            or not self.contract_path.is_file()
            or not self.attempts_root.is_dir()
            or not self.slots_root.is_dir()
            or not self.runs_root.is_dir()
        ):
            raise ValueError("Stage-1 journal root layout is invalid")
        if self.complete_path.is_symlink():
            raise ValueError("Stage-1 completion marker must not be symbolic")

    @staticmethod
    def _run_payload_sha256(payload: Mapping[str, object], field: str) -> str:
        content = dict(payload)
        content.pop(field, None)
        return _sha256_json(content)

    def _run_start_path(self, run_index: int) -> Path:
        return self.runs_root / f"{run_index:06d}-start.json"

    def _run_end_path(self, run_index: int) -> Path:
        return self.runs_root / f"{run_index:06d}-end.json"

    def _validate_run_start(
        self, value: Mapping[str, object], *, run_index: int
    ) -> dict[str, object]:
        expected_fields = {
            "schema_version",
            "run_index",
            "actor_run_uid",
            "predecessor_run_uid",
            "journal_contract_sha256",
            "actor_checkpoint_sha256",
            "start_full_sha256",
            "stat_snapshot_sha256",
            "stat_entry_count",
            "read_only_run_binding",
            "start_content_sha256",
        }
        if set(value) != expected_fields:
            raise ValueError("Stage-1 actor run-start fields do not match")
        if (
            value.get("schema_version")
            != FIRST_DECISION_ACTOR_RUN_START_VERSION
            or value.get("run_index") != run_index
            or value.get("journal_contract_sha256")
            != self.contract["contract_sha256"]
            or value.get("actor_checkpoint_sha256")
            != self.contract["actor_checkpoint_sha256"]
            or value.get("start_full_sha256")
            != self.contract["actor_checkpoint_sha256"]
            or not _is_sha256(value.get("actor_run_uid"))
            or not _is_sha256(value.get("stat_snapshot_sha256"))
            or not isinstance(value.get("stat_entry_count"), int)
            or isinstance(value.get("stat_entry_count"), bool)
            or value["stat_entry_count"] < 1
            or value.get("read_only_run_binding") is not True
            or value.get("start_content_sha256")
            != self._run_payload_sha256(value, "start_content_sha256")
        ):
            raise ValueError("Stage-1 actor run-start attestation is invalid")
        predecessor = value.get("predecessor_run_uid")
        if predecessor is not None and not _is_sha256(predecessor):
            raise ValueError("Stage-1 actor predecessor run UID is invalid")
        return json.loads(_canonical_json(dict(value)))

    def _validate_run_end(
        self,
        value: Mapping[str, object],
        *,
        run_index: int,
        start: Mapping[str, object],
    ) -> dict[str, object]:
        expected_fields = {
            "schema_version",
            "run_index",
            "actor_run_uid",
            "journal_contract_sha256",
            "actor_checkpoint_sha256",
            "start_full_sha256",
            "end_full_sha256",
            "stat_snapshot_sha256",
            "stat_entry_count",
            "runtime_stat_checks",
            "completion_request_stat_checks",
            "read_only_run_binding",
            "completed",
            "drift_detected",
            "end_content_sha256",
        }
        if set(value) != expected_fields:
            raise ValueError("Stage-1 actor run-end fields do not match")
        runtime_checks = value.get("runtime_stat_checks")
        completion_checks = value.get("completion_request_stat_checks")
        if (
            value.get("schema_version") != FIRST_DECISION_ACTOR_RUN_END_VERSION
            or value.get("run_index") != run_index
            or value.get("actor_run_uid") != start["actor_run_uid"]
            or value.get("journal_contract_sha256")
            != self.contract["contract_sha256"]
            or value.get("actor_checkpoint_sha256")
            != start["actor_checkpoint_sha256"]
            or value.get("start_full_sha256") != start["start_full_sha256"]
            or value.get("end_full_sha256") != start["start_full_sha256"]
            or value.get("stat_snapshot_sha256")
            != start["stat_snapshot_sha256"]
            or value.get("stat_entry_count") != start["stat_entry_count"]
            or not isinstance(runtime_checks, int)
            or isinstance(runtime_checks, bool)
            or runtime_checks < 1
            or not isinstance(completion_checks, int)
            or isinstance(completion_checks, bool)
            or not 0 <= completion_checks <= runtime_checks
            or value.get("read_only_run_binding") is not True
            or value.get("completed") is not True
            or value.get("drift_detected") is not False
            or value.get("end_content_sha256")
            != self._run_payload_sha256(value, "end_content_sha256")
        ):
            raise ValueError("Stage-1 actor run-end attestation is invalid")
        return json.loads(_canonical_json(dict(value)))

    def _load_actor_runs(
        self,
    ) -> tuple[dict[int, dict[str, object]], dict[int, dict[str, object]]]:
        unexpected = [
            path
            for path in self.runs_root.iterdir()
            if not (
                (path.is_file() and path.suffix == ".json" and not path.is_symlink())
                or self._is_crash_temporary(path)
            )
        ]
        if unexpected:
            raise ValueError("Stage-1 journal contains an unexpected actor-run path")
        raw_starts: dict[int, Mapping[str, object]] = {}
        raw_ends: dict[int, Mapping[str, object]] = {}
        for path in sorted(self.runs_root.glob("*.json")):
            if path.is_symlink():
                raise ValueError("Stage-1 actor-run evidence must not be symbolic")
            stem_parts = path.stem.rsplit("-", 1)
            if (
                len(stem_parts) != 2
                or not stem_parts[0].isdigit()
                or stem_parts[1] not in {"start", "end"}
            ):
                raise ValueError("Stage-1 actor-run filename is invalid")
            run_index = int(stem_parts[0])
            expected_path = (
                self._run_start_path(run_index)
                if stem_parts[1] == "start"
                else self._run_end_path(run_index)
            )
            if path != expected_path:
                raise ValueError("Stage-1 actor-run filename is not canonical")
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Stage-1 actor-run evidence is invalid JSON") from exc
            if not isinstance(value, Mapping):
                raise TypeError("Stage-1 actor-run evidence must be an object")
            target = raw_starts if stem_parts[1] == "start" else raw_ends
            if run_index in target:
                raise ValueError("Stage-1 actor-run index is duplicated")
            target[run_index] = value
        if set(raw_starts) != set(range(len(raw_starts))):
            raise ValueError("Stage-1 actor-run start schedule is not contiguous")
        if not set(raw_ends).issubset(raw_starts):
            raise ValueError("Stage-1 actor-run end lacks its start")
        starts: dict[int, dict[str, object]] = {}
        ends: dict[int, dict[str, object]] = {}
        for run_index in range(len(raw_starts)):
            starts[run_index] = self._validate_run_start(
                raw_starts[run_index], run_index=run_index
            )
            if run_index in raw_ends:
                ends[run_index] = self._validate_run_end(
                    raw_ends[run_index],
                    run_index=run_index,
                    start=starts[run_index],
                )
        self._validate_actor_run_sequence(starts, ends, require_closed=False)
        return starts, ends

    @staticmethod
    def _validate_actor_run_sequence(
        starts: Mapping[int, Mapping[str, object]],
        ends: Mapping[int, Mapping[str, object]],
        *,
        require_closed: bool,
    ) -> None:
        run_count = len(starts)
        run_uids = [starts[index]["actor_run_uid"] for index in range(run_count)]
        if len(run_uids) != len(set(run_uids)):
            raise ValueError("Stage-1 actor run UID is duplicated")
        for run_index in range(run_count):
            start = starts[run_index]
            predecessor = starts.get(run_index - 1)
            expected_predecessor_uid = (
                None if predecessor is None else predecessor["actor_run_uid"]
            )
            if start.get("predecessor_run_uid") != expected_predecessor_uid:
                raise ValueError("Stage-1 actor-run predecessor chain changed")
            if (
                predecessor is not None
                and (run_index - 1) not in ends
                and (
                    start["start_full_sha256"]
                    != predecessor["start_full_sha256"]
                    or start["stat_snapshot_sha256"]
                    != predecessor["stat_snapshot_sha256"]
                    or start["stat_entry_count"]
                    != predecessor["stat_entry_count"]
                )
            ):
                raise ValueError(
                    "Stage-1 successor start does not close its crashed predecessor"
                )
        if require_closed and (not starts or (run_count - 1) not in ends):
            raise ValueError("Stage-1 actor-run chain lacks a graceful final end")

    def begin_actor_run(
        self, start_report: Mapping[str, object]
    ) -> dict[str, object]:
        if self.complete_path.exists():
            raise ValueError("cannot start an actor run in a completed Stage-1 journal")
        self.actor_run_starts, self.actor_run_ends = self._load_actor_runs()
        run_index = len(self.actor_run_starts)
        predecessor = self.actor_run_starts.get(run_index - 1)
        payload: dict[str, object] = {
            "schema_version": FIRST_DECISION_ACTOR_RUN_START_VERSION,
            "run_index": run_index,
            "actor_run_uid": start_report.get("actor_run_uid"),
            "predecessor_run_uid": (
                None if predecessor is None else predecessor["actor_run_uid"]
            ),
            "journal_contract_sha256": self.contract["contract_sha256"],
            "actor_checkpoint_sha256": start_report.get(
                "actor_checkpoint_sha256"
            ),
            "start_full_sha256": start_report.get("start_full_sha256"),
            "stat_snapshot_sha256": start_report.get("stat_snapshot_sha256"),
            "stat_entry_count": start_report.get("stat_entry_count"),
            "read_only_run_binding": start_report.get("read_only_run_binding"),
        }
        payload["start_content_sha256"] = self._run_payload_sha256(
            payload, "start_content_sha256"
        )
        validated = self._validate_run_start(payload, run_index=run_index)
        if (
            predecessor is not None
            and (run_index - 1) not in self.actor_run_ends
            and (
                validated["start_full_sha256"]
                != predecessor["start_full_sha256"]
                or validated["stat_snapshot_sha256"]
                != predecessor["stat_snapshot_sha256"]
                or validated["stat_entry_count"]
                != predecessor["stat_entry_count"]
            )
        ):
            raise ValueError(
                "Stage-1 resume actor differs from the crashed predecessor"
            )
        _atomic_write_json(self._run_start_path(run_index), validated)
        self.actor_run_starts, self.actor_run_ends = self._load_actor_runs()
        self.active_run_uid = str(validated["actor_run_uid"])
        return deepcopy(validated)

    def finish_actor_run(self, end_report: Mapping[str, object]) -> dict[str, object]:
        if self.active_run_uid is None:
            raise ValueError("Stage-1 journal has no active actor run")
        run_index = len(self.actor_run_starts) - 1
        start = self.actor_run_starts[run_index]
        if end_report.get("actor_run_uid") != self.active_run_uid:
            raise ValueError("Stage-1 actor run-end UID differs from its start")
        payload: dict[str, object] = {
            "schema_version": FIRST_DECISION_ACTOR_RUN_END_VERSION,
            "run_index": run_index,
            "actor_run_uid": end_report.get("actor_run_uid"),
            "journal_contract_sha256": self.contract["contract_sha256"],
            "actor_checkpoint_sha256": end_report.get(
                "actor_checkpoint_sha256"
            ),
            "start_full_sha256": end_report.get("start_full_sha256"),
            "end_full_sha256": end_report.get("end_full_sha256"),
            "stat_snapshot_sha256": end_report.get("stat_snapshot_sha256"),
            "stat_entry_count": end_report.get("stat_entry_count"),
            "runtime_stat_checks": end_report.get("runtime_stat_checks"),
            "completion_request_stat_checks": end_report.get(
                "completion_request_stat_checks"
            ),
            "read_only_run_binding": end_report.get("read_only_run_binding"),
            "completed": end_report.get("completed"),
            "drift_detected": end_report.get("drift_detected"),
        }
        payload["end_content_sha256"] = self._run_payload_sha256(
            payload, "end_content_sha256"
        )
        validated = self._validate_run_end(
            payload, run_index=run_index, start=start
        )
        _atomic_write_json(self._run_end_path(run_index), validated)
        self.actor_run_starts, self.actor_run_ends = self._load_actor_runs()
        self.active_run_uid = None
        return deepcopy(validated)

    @staticmethod
    def _attempt_payload_sha256(payload: Mapping[str, object]) -> str:
        content = dict(payload)
        content.pop("attempt_content_sha256", None)
        return _sha256_json(content)

    def _validate_attempt(
        self, attempt: Mapping[str, object], path: Path
    ) -> dict[str, object]:
        if set(attempt) != {
            "schema_version",
            "slot_uid",
            "slot",
            "actor_run_uid",
            "request_prompt_sha256",
            "request_seed",
            "attempt_uid",
            "intent_persisted_before_request",
            "attempt_content_sha256",
        }:
            raise ValueError("Stage-1 sampling-attempt fields do not match")
        if attempt.get("schema_version") != FIRST_DECISION_ATTEMPT_VERSION:
            raise ValueError("Stage-1 sampling-attempt version mismatch")
        slot_uid = attempt.get("slot_uid")
        spec = self.slot_specs.get(str(slot_uid))
        if spec is None:
            raise ValueError("Stage-1 journal contains an unknown attempt slot")
        if path != self._attempt_path(spec):
            raise ValueError("Stage-1 sampling-attempt filename changed")
        actor_run_uid = attempt.get("actor_run_uid")
        known_run_uids = {
            start["actor_run_uid"] for start in self.actor_run_starts.values()
        }
        expected_attempt_uid = _attempt_uid(
            str(self.contract["contract_sha256"]),
            str(slot_uid),
            str(actor_run_uid),
        )
        if (
            attempt.get("slot") != spec
            or not _is_sha256(actor_run_uid)
            or actor_run_uid not in known_run_uids
            or not _is_sha256(attempt.get("request_prompt_sha256"))
            or attempt.get("request_seed") != spec["seed"]
            or attempt.get("attempt_uid") != expected_attempt_uid
            or attempt.get("intent_persisted_before_request") is not True
            or attempt.get("attempt_content_sha256")
            != self._attempt_payload_sha256(attempt)
        ):
            raise ValueError("Stage-1 sampling-attempt content is invalid")
        return json.loads(_canonical_json(dict(attempt)))

    def _load_attempts(self) -> dict[str, dict[str, object]]:
        attempts: dict[str, dict[str, object]] = {}
        unexpected = [
            path
            for path in self.attempts_root.iterdir()
            if not (
                (path.is_file() and path.suffix == ".json" and not path.is_symlink())
                or self._is_crash_temporary(path)
            )
        ]
        if unexpected:
            raise ValueError("Stage-1 journal contains an unexpected attempt path")
        for path in sorted(self.attempts_root.glob("*.json")):
            if path.is_symlink():
                raise ValueError("Stage-1 sampling-attempt must not be symbolic")
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "Stage-1 journal contains an invalid sampling-attempt"
                ) from exc
            if not isinstance(raw, Mapping):
                raise TypeError("Stage-1 sampling-attempt must be an object")
            attempt = self._validate_attempt(raw, path)
            slot_uid = str(attempt["slot_uid"])
            if slot_uid in attempts:
                raise ValueError("Stage-1 journal repeats a sampling-attempt")
            attempts[slot_uid] = attempt
        return attempts

    def begin_attempt(
        self,
        spec: Mapping[str, object],
        *,
        actor_run_uid: str,
        request_prompt_sha256: str,
        request_seed: int,
    ) -> dict[str, object]:
        if self.complete_path.exists():
            raise ValueError("cannot start an attempt in a completed Stage-1 journal")
        if self.active_run_uid is None or actor_run_uid != self.active_run_uid:
            raise ValueError("Stage-1 attempt is not bound to the active actor run")
        slot_uid = str(spec["slot_uid"])
        if slot_uid in self.entries:
            raise ValueError("refusing to attempt a completed Stage-1 slot")
        if slot_uid in self.attempts:
            raise ValueError("refusing to repeat a started Stage-1 attempt")
        attempt: dict[str, object] = {
            "schema_version": FIRST_DECISION_ATTEMPT_VERSION,
            "slot_uid": slot_uid,
            "slot": json.loads(_canonical_json(dict(spec))),
            "actor_run_uid": actor_run_uid,
            "request_prompt_sha256": request_prompt_sha256,
            "request_seed": request_seed,
            "attempt_uid": _attempt_uid(
                str(self.contract["contract_sha256"]), slot_uid, actor_run_uid
            ),
            "intent_persisted_before_request": True,
        }
        attempt["attempt_content_sha256"] = self._attempt_payload_sha256(attempt)
        path = self._attempt_path(spec)
        if path.exists():
            raise ValueError("Stage-1 sampling-attempt already exists")
        validated = self._validate_attempt(attempt, path)
        _atomic_write_json(path, validated)
        self.attempts[slot_uid] = self._validate_attempt(validated, path)
        return deepcopy(validated)

    def orphan_attempt(self, spec: Mapping[str, object]) -> dict[str, object] | None:
        slot_uid = str(spec["slot_uid"])
        if slot_uid in self.entries:
            return None
        attempt = self.attempts.get(slot_uid)
        return None if attempt is None else deepcopy(attempt)

    def _validate_entry(
        self, entry: Mapping[str, object], path: Path
    ) -> dict[str, object]:
        if set(entry) != {
            "schema_version",
            "slot_uid",
            "slot",
            "attempt_uid",
            "record",
            "entry_sha256",
        }:
            raise ValueError("Stage-1 journal entry fields do not match")
        if entry.get("schema_version") != FIRST_DECISION_JOURNAL_ENTRY_VERSION:
            raise ValueError("Stage-1 journal entry version mismatch")
        slot_uid = entry.get("slot_uid")
        spec = self.slot_specs.get(str(slot_uid))
        if spec is None:
            raise ValueError("Stage-1 journal contains an unknown slot")
        if path != self._slot_path(spec):
            raise ValueError("Stage-1 journal slot filename changed")
        if entry.get("slot") != spec:
            raise ValueError("Stage-1 journal slot contract changed")
        if entry.get("entry_sha256") != self._entry_sha256(entry):
            raise ValueError("Stage-1 journal entry content hash mismatch")
        record = entry.get("record")
        if not isinstance(record, Mapping):
            raise TypeError("Stage-1 journal entry record must be an object")
        attempt = self.attempts.get(str(slot_uid))
        if (
            attempt is None
            or entry.get("attempt_uid") != attempt["attempt_uid"]
            or record.get("sampling_attempt_uid") != attempt["attempt_uid"]
            or record.get("actor_run_uid") != attempt["actor_run_uid"]
            or record.get("prompt_token_sha256")
            != attempt["request_prompt_sha256"]
            or record.get("seed") != attempt["request_seed"]
        ):
            raise ValueError("Stage-1 journal entry does not close its attempt")
        actor_run_uid = record.get("actor_run_uid")
        known_run_uids = {
            start["actor_run_uid"] for start in self.actor_run_starts.values()
        }
        if not _is_sha256(actor_run_uid) or actor_run_uid not in known_run_uids:
            raise ValueError("Stage-1 journal record lacks a known actor run UID")
        if record.get("stage1_record_content_sha256") != _record_content_sha256(record):
            raise ValueError("Stage-1 proposal record content hash mismatch")
        if (
            record.get("active_group_uid") != spec["active_group_uid"]
            or record.get("suffix_index") != spec["proposal_index"]
            or record.get("suffix_uid") != spec["suffix_uid"]
            or record.get("proposal_uid") != spec["proposal_uid"]
            or record.get("journal_slot_uid") != spec["slot_uid"]
        ):
            raise ValueError("Stage-1 journal record slot identity changed")
        if _OUTCOME_FIELDS.intersection(record):
            raise ValueError("Stage-1 journal record contains rollout outcomes")
        return json.loads(_canonical_json(dict(entry)))

    def _load_entries(self) -> dict[str, dict[str, object]]:
        entries: dict[str, dict[str, object]] = {}
        unexpected = [
            path
            for path in self.slots_root.iterdir()
            if not (
                (path.is_file() and path.suffix == ".json" and not path.is_symlink())
                or self._is_crash_temporary(path)
            )
        ]
        if unexpected:
            raise ValueError("Stage-1 journal contains an unexpected slot path")
        for path in sorted(self.slots_root.glob("*.json")):
            if path.is_symlink():
                raise ValueError("Stage-1 journal entry must not be symbolic")
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Stage-1 journal contains invalid JSON") from exc
            if not isinstance(raw, Mapping):
                raise TypeError("Stage-1 journal entry must be an object")
            entry = self._validate_entry(raw, path)
            slot_uid = str(entry["slot_uid"])
            if slot_uid in entries:
                raise ValueError("Stage-1 journal repeats a completed slot")
            entries[slot_uid] = entry
        return entries

    def _entry_descriptors(self) -> list[dict[str, object]]:
        descriptors = []
        for spec in self.ordered_slot_specs:
            slot_uid = str(spec["slot_uid"])
            entry = self.entries.get(slot_uid)
            if entry is None:
                raise ValueError(
                    "Stage-1 journal does not cover the pre-registered partition"
                )
            path = self._slot_path(spec)
            raw = path.read_bytes()
            descriptors.append(
                {
                    "slot_uid": slot_uid,
                    "relative_path": path.relative_to(self.root).as_posix(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "byte_count": len(raw),
                    "entry_sha256": entry["entry_sha256"],
                    "record_content_sha256": entry["record"][
                        "stage1_record_content_sha256"
                    ],
                }
            )
        return descriptors

    def _attempt_descriptors(self) -> list[dict[str, object]]:
        descriptors = []
        for spec in self.ordered_slot_specs:
            slot_uid = str(spec["slot_uid"])
            attempt = self.attempts.get(slot_uid)
            if attempt is None:
                raise ValueError(
                    "Stage-1 journal lacks a pre-request attempt for a slot"
                )
            path = self._attempt_path(spec)
            raw = path.read_bytes()
            descriptors.append(
                {
                    "slot_uid": slot_uid,
                    "attempt_uid": attempt["attempt_uid"],
                    "actor_run_uid": attempt["actor_run_uid"],
                    "relative_path": path.relative_to(self.root).as_posix(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "byte_count": len(raw),
                    "attempt_content_sha256": attempt[
                        "attempt_content_sha256"
                    ],
                }
            )
        return descriptors

    def _actor_run_file_descriptors(self) -> list[dict[str, object]]:
        descriptors = []
        for run_index in range(len(self.actor_run_starts)):
            for boundary, path, payload in (
                (
                    "start",
                    self._run_start_path(run_index),
                    self.actor_run_starts[run_index],
                ),
                (
                    "end",
                    self._run_end_path(run_index),
                    self.actor_run_ends.get(run_index),
                ),
            ):
                if payload is None:
                    continue
                raw = path.read_bytes()
                descriptors.append(
                    {
                        "run_index": run_index,
                        "actor_run_uid": self.actor_run_starts[run_index][
                            "actor_run_uid"
                        ],
                        "boundary": boundary,
                        "relative_path": path.relative_to(self.root).as_posix(),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "byte_count": len(raw),
                        "content_sha256": payload[
                            f"{boundary}_content_sha256"
                        ],
                    }
                )
        return descriptors

    def actor_run_chain_report(self, *, require_closed: bool) -> dict[str, object]:
        self._validate_actor_run_sequence(
            self.actor_run_starts,
            self.actor_run_ends,
            require_closed=require_closed,
        )
        known_run_uids = {
            str(start["actor_run_uid"]) for start in self.actor_run_starts.values()
        }
        record_run_uid_schedule = [
            str(self.entries[str(spec["slot_uid"])]["record"].get("actor_run_uid"))
            for spec in self.ordered_slot_specs
            if str(spec["slot_uid"]) in self.entries
        ]
        attempt_run_uid_schedule = [
            str(self.attempts[str(spec["slot_uid"])].get("actor_run_uid"))
            for spec in self.ordered_slot_specs
            if str(spec["slot_uid"]) in self.attempts
        ]
        if any(uid not in known_run_uids for uid in record_run_uid_schedule):
            raise ValueError("Stage-1 record is not covered by an actor run-start")
        if any(uid not in known_run_uids for uid in attempt_run_uid_schedule):
            raise ValueError("Stage-1 attempt is not covered by an actor run-start")
        record_counts = {
            uid: record_run_uid_schedule.count(uid) for uid in known_run_uids
        }
        attempt_counts = {
            uid: attempt_run_uid_schedule.count(uid) for uid in known_run_uids
        }
        runs = []
        for run_index, start in self.actor_run_starts.items():
            end = self.actor_run_ends.get(run_index)
            if end is not None:
                closure = "graceful_end"
                closure_run_uid = start["actor_run_uid"]
                if end["completion_request_stat_checks"] != attempt_counts[
                    str(start["actor_run_uid"])
                ]:
                    raise ValueError(
                        "Stage-1 graceful run request count differs from its attempts"
                    )
            elif run_index + 1 in self.actor_run_starts:
                closure = "successor_start_continuity"
                closure_run_uid = self.actor_run_starts[run_index + 1][
                    "actor_run_uid"
                ]
            else:
                closure = "unclosed"
                closure_run_uid = None
            runs.append(
                {
                    "run_index": run_index,
                    "actor_run_uid": start["actor_run_uid"],
                    "predecessor_run_uid": start["predecessor_run_uid"],
                    "start_full_sha256": start["start_full_sha256"],
                    "end_full_sha256": (
                        None if end is None else end["end_full_sha256"]
                    ),
                    "stat_snapshot_sha256": start["stat_snapshot_sha256"],
                    "stat_entry_count": start["stat_entry_count"],
                    "attempt_count": attempt_counts[str(start["actor_run_uid"])],
                    "record_count": record_counts[str(start["actor_run_uid"])],
                    "closure": closure,
                    "closure_run_uid": closure_run_uid,
                }
            )
        all_runs_closed = bool(runs) and all(
            run["closure"] != "unclosed" for run in runs
        )
        report: dict[str, object] = {
            "schema_version": FIRST_DECISION_ACTOR_RUN_CHAIN_VERSION,
            "actor_checkpoint_sha256": self.contract[
                "actor_checkpoint_sha256"
            ],
            "run_count": len(runs),
            "record_count": len(record_run_uid_schedule),
            "attempt_count": len(attempt_run_uid_schedule),
            "attempt_run_uid_schedule": attempt_run_uid_schedule,
            "attempt_run_uid_schedule_sha256": _sha256_json(
                attempt_run_uid_schedule
            ),
            "record_run_uid_schedule": record_run_uid_schedule,
            "record_run_uid_schedule_sha256": _sha256_json(
                record_run_uid_schedule
            ),
            "covered_record_run_uids": sorted(set(record_run_uid_schedule)),
            "all_records_covered": (
                len(record_run_uid_schedule) == len(self.entries)
            ),
            "all_attempts_covered": (
                len(attempt_run_uid_schedule) == len(self.attempts)
            ),
            "all_runs_closed": all_runs_closed,
            "runs": runs,
        }
        report["chain_sha256"] = _sha256_json(report)
        if require_closed and (
            report["all_records_covered"] is not True
            or report["all_attempts_covered"] is not True
            or report["all_runs_closed"] is not True
        ):
            raise ValueError("Stage-1 actor-run chain is not formally closed")
        return report

    def _complete_payload(self) -> dict[str, object]:
        if (
            set(self.attempts) != set(self.slot_specs)
            or set(self.entries) != set(self.slot_specs)
        ):
            raise ValueError("Stage-1 journal cannot seal an incomplete partition")
        records = [
            self.entries[str(spec["slot_uid"])]["record"]
            for spec in self.ordered_slot_specs
        ]
        attempts = self._attempt_descriptors()
        attempt_uid_schedule = [
            str(attempt["attempt_uid"]) for attempt in attempts
        ]
        return {
            "schema_version": FIRST_DECISION_JOURNAL_COMPLETE_VERSION,
            "status": "complete",
            "journal_contract_sha256": self.contract["contract_sha256"],
            "journal_header_file_sha256": _sha256_file(self.contract_path),
            "expected_slot_count": len(self.ordered_slot_specs),
            "completed_slot_count": len(self.entries),
            "slot_uid_schedule": self.contract["slot_uid_schedule"],
            "slot_uid_schedule_sha256": self.contract[
                "slot_uid_schedule_sha256"
            ],
            "attempt_uid_schedule": attempt_uid_schedule,
            "attempt_uid_schedule_sha256": _sha256_json(attempt_uid_schedule),
            "attempts": attempts,
            "entries": self._entry_descriptors(),
            "records_canonical_sha256": _sha256_json(records),
            "actor_run_files": self._actor_run_file_descriptors(),
            "actor_run_chain": self.actor_run_chain_report(require_closed=True),
        }

    def _validate_complete_marker(self) -> dict[str, object]:
        if not self.complete_path.is_file() or self.complete_path.is_symlink():
            raise ValueError("Stage-1 journal completion marker is invalid")
        try:
            marker = json.loads(self.complete_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Stage-1 journal completion marker is invalid") from exc
        expected = self._complete_payload()
        if marker != expected:
            raise ValueError(
                "Stage-1 journal completion marker does not match its entries"
            )
        return json.loads(_canonical_json(expected))

    def _reload(self, *, require_complete: bool) -> None:
        self._validate_root_layout()
        self.actor_run_starts, self.actor_run_ends = self._load_actor_runs()
        self.attempts = self._load_attempts()
        self.entries = self._load_entries()
        if self.complete_path.exists():
            self._validate_complete_marker()
        elif require_complete:
            raise ValueError("completed Stage-1 journal marker is required")

    def commit(
        self,
        spec: Mapping[str, object],
        record: Mapping[str, object],
        *,
        interrupted_recovery: bool = False,
    ) -> None:
        if self.complete_path.exists():
            raise ValueError("cannot append to a completed Stage-1 journal")
        if self.active_run_uid is None:
            raise ValueError("Stage-1 journal has no active actor run")
        if not interrupted_recovery and record.get("actor_run_uid") != self.active_run_uid:
            raise ValueError("Stage-1 record is not bound to the active actor run")
        slot_uid = str(spec["slot_uid"])
        if slot_uid in self.entries:
            raise ValueError("refusing to resample a completed Stage-1 slot")
        attempt = self.attempts.get(slot_uid)
        if attempt is None:
            raise ValueError("Stage-1 record lacks a durable pre-request attempt")
        if (
            record.get("actor_run_uid") != attempt["actor_run_uid"]
            or record.get("sampling_attempt_uid") != attempt["attempt_uid"]
        ):
            raise ValueError("Stage-1 record differs from its sampling-attempt")
        if interrupted_recovery and (
            record.get("infrastructure_invalid") is not True
            or record.get("infrastructure_error_code")
            != "interrupted_after_attempt_intent"
        ):
            raise ValueError("Stage-1 interrupted attempt must fail closed")
        entry: dict[str, object] = {
            "schema_version": FIRST_DECISION_JOURNAL_ENTRY_VERSION,
            "slot_uid": slot_uid,
            "slot": json.loads(_canonical_json(dict(spec))),
            "attempt_uid": attempt["attempt_uid"],
            "record": json.loads(_canonical_json(dict(record))),
        }
        entry["entry_sha256"] = self._entry_sha256(entry)
        path = self._slot_path(spec)
        if path.exists():
            raise ValueError("Stage-1 journal slot already exists")
        _atomic_write_json(path, entry)
        self.entries[slot_uid] = self._validate_entry(entry, path)

    def seal(self) -> dict[str, object]:
        if self.active_run_uid is not None:
            raise ValueError("cannot seal Stage-1 journal before actor run-end")
        self._reload(require_complete=False)
        payload = self._complete_payload()
        if self.complete_path.exists():
            self._validate_complete_marker()
        else:
            _atomic_write_json(self.complete_path, payload)
        self._reload(require_complete=True)
        return self._validate_complete_marker()

    def ordered_records(self) -> list[dict[str, object]]:
        self._reload(require_complete=True)
        if set(self.entries) != set(self.slot_specs):
            raise ValueError("Stage-1 journal does not cover the pre-registered partition")
        return [
            deepcopy(self.entries[str(spec["slot_uid"])]["record"])
            for spec in self.ordered_slot_specs
        ]

    def content_attestation(self) -> dict[str, object]:
        self._reload(require_complete=True)
        attempts = self._attempt_descriptors()
        attempt_uid_schedule = [
            str(attempt["attempt_uid"]) for attempt in attempts
        ]
        rows = self._entry_descriptors()
        actor_run_files = self._actor_run_file_descriptors()
        actor_run_chain = self.actor_run_chain_report(require_closed=True)
        header_raw = self.contract_path.read_bytes()
        complete_raw = self.complete_path.read_bytes()
        byte_contract = {
            "header_sha256": hashlib.sha256(header_raw).hexdigest(),
            "header_byte_count": len(header_raw),
            "slot_uid_schedule": list(self.contract["slot_uid_schedule"]),
            "slot_uid_schedule_sha256": self.contract[
                "slot_uid_schedule_sha256"
            ],
            "attempt_uid_schedule": attempt_uid_schedule,
            "attempt_uid_schedule_sha256": _sha256_json(attempt_uid_schedule),
            "attempts": attempts,
            "entries": rows,
            "actor_run_files": actor_run_files,
            "actor_run_chain_sha256": actor_run_chain["chain_sha256"],
            "complete_sha256": hashlib.sha256(complete_raw).hexdigest(),
            "complete_byte_count": len(complete_raw),
        }
        return {
            "journal_contract_sha256": self.contract["contract_sha256"],
            "journal_contract_file_sha256": byte_contract["header_sha256"],
            "completed_slots": len(rows),
            "slot_uid_schedule": byte_contract["slot_uid_schedule"],
            "slot_uid_schedule_sha256": byte_contract[
                "slot_uid_schedule_sha256"
            ],
            "attempt_uid_schedule": byte_contract["attempt_uid_schedule"],
            "attempt_uid_schedule_sha256": byte_contract[
                "attempt_uid_schedule_sha256"
            ],
            "sampling_attempts": attempts,
            "slot_entries": rows,
            "actor_run_files": actor_run_files,
            "actor_run_chain": actor_run_chain,
            "journal_complete_marker_file_sha256": byte_contract[
                "complete_sha256"
            ],
            "journal_complete_marker_content_sha256": _sha256_json(
                self._validate_complete_marker()
            ),
            "journal_content_sha256": _sha256_json(byte_contract),
            "byte_contract": byte_contract,
        }


def _completed_journal_artifact(
    journal: _FirstDecisionJournal,
    *,
    commit_directory: Path,
    expected_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    commit_root = commit_directory.expanduser().resolve()
    try:
        relative_root = journal.root.relative_to(commit_root)
    except ValueError as exc:
        raise ValueError(
            "Stage-1 journal must be inside the manifest directory"
        ) from exc
    if not relative_root.parts or any(part in {"", ".", ".."} for part in relative_root.parts):
        raise ValueError("Stage-1 journal relative directory is invalid")

    journal_records = journal.ordered_records()
    normalized_expected = [
        json.loads(_canonical_json(dict(record))) for record in expected_records
    ]
    if len(journal_records) != len(normalized_expected):
        raise ValueError("Stage-1 records count differs from the sampling journal")
    for index, (journal_record, expected_record) in enumerate(
        zip(journal_records, normalized_expected, strict=True)
    ):
        if journal_record != expected_record:
            raise ValueError(
                f"Stage-1 record {index} differs from the sampling journal"
            )

    attestation = journal.content_attestation()
    byte_contract = attestation["byte_contract"]
    artifact: dict[str, object] = {
        "schema_version": FIRST_DECISION_JOURNAL_ARTIFACT_VERSION,
        "relative_directory": relative_root.as_posix(),
        "header": {
            "relative_path": journal.contract_path.relative_to(
                journal.root
            ).as_posix(),
            "sha256": byte_contract["header_sha256"],
            "byte_count": byte_contract["header_byte_count"],
        },
        "complete": {
            "relative_path": journal.complete_path.relative_to(
                journal.root
            ).as_posix(),
            "sha256": byte_contract["complete_sha256"],
            "byte_count": byte_contract["complete_byte_count"],
            "content_sha256": attestation[
                "journal_complete_marker_content_sha256"
            ],
        },
        "record_count": len(journal_records),
        "slot_uid_schedule": list(attestation["slot_uid_schedule"]),
        "slot_uid_schedule_sha256": attestation[
            "slot_uid_schedule_sha256"
        ],
        "attempt_uid_schedule": list(attestation["attempt_uid_schedule"]),
        "attempt_uid_schedule_sha256": attestation[
            "attempt_uid_schedule_sha256"
        ],
        "entries": deepcopy(attestation["slot_entries"]),
        "attempts": deepcopy(attestation["sampling_attempts"]),
        "actor_runs": deepcopy(attestation["actor_run_files"]),
        "actor_run_chain": deepcopy(attestation["actor_run_chain"]),
        "actor_run_chain_sha256": attestation["actor_run_chain"][
            "chain_sha256"
        ],
        "records_canonical_sha256": _sha256_json(journal_records),
        "journal_content_sha256": attestation["journal_content_sha256"],
    }
    artifact["artifact_sha256"] = _sha256_json(artifact)
    return artifact


def _actor_run_provenance_fields(
    journal: _FirstDecisionJournal,
    journal_attestation: Mapping[str, object],
) -> dict[str, object]:
    chain = journal_attestation.get("actor_run_chain")
    if (
        not isinstance(chain, Mapping)
        or not isinstance(chain.get("runs"), list)
        or not chain["runs"]
    ):
        raise ValueError("Stage-1 actor-run chain attestation is missing")
    first = chain["runs"][0]
    runtime_checks = sum(
        int(end["runtime_stat_checks"])
        for end in journal.actor_run_ends.values()
    )
    return {
        "actor_attested_at_start": True,
        "actor_attested_at_end": chain["all_runs_closed"],
        "actor_stat_snapshot_sha256": first["stat_snapshot_sha256"],
        "actor_stat_entry_count": first["stat_entry_count"],
        "actor_runtime_stat_checks": runtime_checks,
        "actor_run_chain_sha256": chain["chain_sha256"],
        "actor_run_count": chain["run_count"],
        "actor_record_run_uid_count": len(chain["covered_record_run_uids"]),
        "actor_all_records_covered": chain["all_records_covered"],
        "actor_all_runs_closed": chain["all_runs_closed"],
    }


class FirstDecisionStage1Collector:
    """Sample one frozen-policy Assistant decision for every planned slot."""

    def __init__(
        self,
        *,
        plan: Mapping[str, object],
        resolved_selections: Sequence[Mapping[str, object]],
        actor_checkpoint: str | Path,
        actor_tokenizer_contract_sha256: str,
        sampling_backend_contract: Mapping[str, object],
        completion_client,
        parser,
        tool_schemas: Sequence[Mapping[str, object]],
        required_environment_version: str,
        expected_states: int,
        proposals_per_state: int,
        source_provenance: Mapping[str, object] | None = None,
        policy_reward: object = None,
        vllm_timeout_seconds: int = 180,
    ) -> None:
        self.plan = validate_active_branch_plan(plan)
        self.plan_sha256 = _sha256_json(self.plan)
        self.resolved = list(resolved_selections)
        rebuilt = build_active_branch_plan(
            self.resolved,
            actor_checkpoint_sha256=self.plan["actor_checkpoint_sha256"],
            decoding_config=self.plan["decoding_config"],
            seed=self.plan["seed"],
            suffixes_per_state=self.plan["suffixes_per_state"],
        )
        if rebuilt != self.plan:
            raise ValueError("Stage-1 plan does not match resolved selections")
        if (
            not isinstance(expected_states, int)
            or isinstance(expected_states, bool)
            or expected_states < 1
            or len(self.plan["groups"]) != expected_states
        ):
            raise ValueError("Stage-1 expected state count mismatch")
        if (
            not isinstance(proposals_per_state, int)
            or isinstance(proposals_per_state, bool)
            or proposals_per_state < 2
            or self.plan["suffixes_per_state"] != proposals_per_state
        ):
            raise ValueError("Stage-1 proposal K mismatch")
        if len(self.resolved) != expected_states:
            raise ValueError("Stage-1 resolved selection count mismatch")
        self.expected_states = expected_states
        self.proposals_per_state = proposals_per_state
        self.actor_checkpoint = Path(actor_checkpoint).expanduser().absolute()
        if not _is_sha256(actor_tokenizer_contract_sha256) or any(
            group["tokenizer_contract_sha256"]
            != actor_tokenizer_contract_sha256
            for group in self.plan["groups"]
        ):
            raise ValueError("Stage-1 actor tokenizer contract differs from the plan")
        self.actor_tokenizer_contract_sha256 = actor_tokenizer_contract_sha256
        self.backend_contract = validate_sampling_backend_contract(
            sampling_backend_contract
        )
        self.backend_sha256 = sampling_backend_contract_sha256(
            self.backend_contract
        )
        decoding = self.plan["decoding_config"]
        if self.backend_sha256 != decoding["sampling_backend_contract_sha256"]:
            raise ValueError("Stage-1 backend contract differs from the plan")
        self.completion_client = completion_client
        self.parser = parser
        if getattr(completion_client, "timeout", None) != vllm_timeout_seconds:
            raise ValueError("Stage-1 completion timeout differs from the contract")
        self.vllm_timeout_seconds = vllm_timeout_seconds
        self.tool_schemas = [
            json.loads(_canonical_json(dict(schema))) for schema in tool_schemas
        ]
        if not self.tool_schemas or tool_schema_sha256(self.tool_schemas) != decoding[
            "tool_schema_sha256"
        ]:
            raise ValueError("Stage-1 tool schema contract mismatch")
        self.tool_names = {
            str((schema.get("function") or {}).get("name") or "")
            for schema in self.tool_schemas
        }
        if "" in self.tool_names or len(self.tool_names) != len(self.tool_schemas):
            raise ValueError("Stage-1 tool schemas must have unique names")
        parser_stops = getattr(parser, "stop_token_ids", None)
        if parser_stops is None or list(parser_stops) != decoding["stop_token_ids"]:
            raise ValueError("Stage-1 parser stop-token contract mismatch")
        self.required_environment_version = str(required_environment_version)
        self.policy_reward = validate_policy_reward_config(policy_reward)
        self.policy_reward_sha256 = _sha256_json(self.policy_reward)
        self.source_provenance = _normalize_source_provenance(source_provenance)
        self.scale_ready = _source_scale_ready(self.source_provenance)
        self.prefix_contexts = []
        environment_manifest_sha256s = set()
        for group, resolved in zip(self.plan["groups"], self.resolved, strict=True):
            trajectory = resolved.get("trajectory")
            if not isinstance(trajectory, Mapping):
                raise TypeError("resolved Stage-1 trajectory is missing")
            if trajectory.get("environment_version") != self.required_environment_version:
                raise ValueError("Stage-1 environment version mismatch")
            if (
                trajectory.get("environment_manifest_sha256")
                != group["environment_manifest_sha256"]
            ):
                raise ValueError("Stage-1 environment manifest mismatch")
            environment_manifest_sha256s.add(group["environment_manifest_sha256"])
            self.prefix_contexts.append(
                _prefix_budget_context(resolved, group, decoding)
            )
        self.environment_manifest_sha256s = sorted(environment_manifest_sha256s)
        self.harness_contract = _build_harness_contract(
            decoding,
            policy_reward_sha256=self.policy_reward_sha256,
        )
        self.harness_contract_sha256 = _sha256_json(self.harness_contract)
        self.slot_specs = self._build_slot_specs()
        self.journal_contract = _build_journal_contract(
            plan=self.plan,
            slot_specs=self.slot_specs,
            actor_tokenizer_contract_sha256=(
                self.actor_tokenizer_contract_sha256
            ),
            sampling_backend_contract_sha256=self.backend_sha256,
            required_environment_version=self.required_environment_version,
            environment_manifest_sha256s=self.environment_manifest_sha256s,
            policy_reward_sha256=self.policy_reward_sha256,
            harness_contract_sha256=self.harness_contract_sha256,
            source_provenance=self.source_provenance,
        )

    def _build_slot_specs(self) -> list[dict[str, object]]:
        return _slot_specs_for_plan(self.plan)

    def _attest_backend(self, actor_run: _Stage1ActorRunAttestation):
        kwargs = {
            "plan_sha256": self.plan_sha256,
            "expected_contract": self.backend_contract,
            "expected_contract_sha256": self.backend_sha256,
            "decoding_config": self.plan["decoding_config"],
            "actor_checkpoint": self.actor_checkpoint,
            "actor_checkpoint_sha256": self.plan["actor_checkpoint_sha256"],
        }
        prehashed = getattr(
            self.completion_client, "attest_and_bind_prehashed", None
        )
        if callable(prehashed):
            return prehashed(
                **kwargs,
                actor_runtime_binding=actor_run.backend_binding(),
            )
        return self.completion_client.attest_and_bind(
            **kwargs,
        )

    async def _parse(self, token_ids: Sequence[int]) -> list[dict[str, object]]:
        parsed = self.parser.parse(list(token_ids), self.tool_schemas)
        if hasattr(parsed, "__await__"):
            parsed = await parsed
        return _normalize_tool_calls(parsed)

    def _base_record(
        self,
        group_index: int,
        proposal_index: int,
        actor_run_uid: str,
        sampling_attempt_uid: str,
    ) -> dict[str, object]:
        group = self.plan["groups"][group_index]
        suffix = group["suffixes"][proposal_index]
        proposal_uid = _proposal_uid(group["active_group_uid"], proposal_index)
        capture = validate_actor_prompt_tokens(
            group["actor_prompt_tokens"],
            expected_sha256=group["actor_prompt_tokens"]["sha256"],
        )
        return {
            "schema_version": ACTIVE_SUFFIX_RESULT_VERSION,
            "stage1_proposal_version": FIRST_DECISION_PROPOSAL_VERSION,
            "collection_mode": "first_decision_only",
            "active_group_uid": group["active_group_uid"],
            "parent_branch_uid": group["parent_branch_uid"],
            "replay_state_id": group["replay_state_id"],
            "task_id": group["task_id"],
            "suffix_index": proposal_index,
            "suffix_uid": suffix["suffix_uid"],
            "proposal_uid": proposal_uid,
            "journal_slot_uid": _slot_uid(
                self.plan_sha256,
                group["active_group_uid"],
                proposal_index,
                proposal_uid,
            ),
            "seed": suffix["seed"],
            "plan_schema_version": self.plan["schema_version"],
            "plan_strategy_version": self.plan["strategy_version"],
            "plan_sha256": self.plan_sha256,
            "actor_checkpoint_sha256": self.plan["actor_checkpoint_sha256"],
            "actor_run_uid": actor_run_uid,
            "sampling_attempt_uid": sampling_attempt_uid,
            "actor_tokenizer_contract_sha256": (
                self.actor_tokenizer_contract_sha256
            ),
            "decoding_config_sha256": self.plan["decoding_config_sha256"],
            "sampling_backend_contract_sha256": self.backend_sha256,
            "environment_version": self.required_environment_version,
            "environment_manifest_sha256": group[
                "environment_manifest_sha256"
            ],
            "policy_reward_sha256": self.policy_reward_sha256,
            "tool_schema_sha256": self.plan["decoding_config"][
                "tool_schema_sha256"
            ],
            "harness_contract_sha256": self.harness_contract_sha256,
            "prompt_token_ids": list(capture["tokens"]),
            "prompt_token_count": capture["count"],
            "prompt_token_sha256": capture["sha256"],
            "optimizer_enabled": False,
            "uses_hidden_goal": False,
        }

    async def _sample_slot(
        self,
        bound_client,
        group_index: int,
        proposal_index: int,
        actor_run_uid: str,
        sampling_attempt_uid: str,
    ) -> dict[str, object]:
        record = self._base_record(
            group_index,
            proposal_index,
            actor_run_uid,
            sampling_attempt_uid,
        )
        prompt = list(record["prompt_token_ids"])
        seed = int(record["seed"])
        completion = bound_client.complete(prompt, seed=seed)
        if not isinstance(completion, Mapping):
            raise ActiveSuffixInfrastructureError("Stage-1 completion is not an object")
        if completion.get("prompt_token_ids") != prompt:
            raise ActiveSuffixInfrastructureError("Stage-1 prompt echo changed")
        completion_ids = _token_ids(
            completion.get("token_ids"),
            "Stage-1 completion token_ids",
            allow_empty=True,
        )
        old_logprobs = _finite_logprobs(
            completion.get("old_logprobs"), len(completion_ids)
        )
        finish_reason = completion.get("finish_reason")
        stop_reason = completion.get("stop_reason")
        if finish_reason not in {"stop", "length"}:
            raise ActiveSuffixInfrastructureError(
                "Stage-1 completion has an invalid finish_reason"
            )
        if not completion_ids and finish_reason != "stop":
            raise ActiveSuffixInfrastructureError(
                "Stage-1 empty completion must be immediate EOS/stop"
            )
        if stop_reason is not None and not isinstance(stop_reason, (int, str)):
            raise ActiveSuffixInfrastructureError(
                "Stage-1 completion has an invalid stop_reason"
            )
        context = self.prefix_contexts[group_index]
        config = self.plan["decoding_config"]
        response_budget = int(config["response_length"]) - int(
            context["response_tokens_before"]
        )
        harness_limit_reason = None
        first_action = None
        first_action_sha256 = None
        first_action_span = None
        first_action_credit_eligible = False
        output_ids = list(completion_ids)
        output_logprobs = list(old_logprobs)

        if completion_ids:
            if (
                context["response_tokens_before"] + len(completion_ids)
                >= int(config["response_length"])
            ):
                harness_limit_reason = "response_length"
            elif context["assistant_turns"] + 1 >= int(
                config["max_assistant_turns"]
            ):
                harness_limit_reason = "max_assistant_turns"
            elif context["user_turns"] >= int(config["max_user_turns"]):
                harness_limit_reason = "max_user_turns"

            if harness_limit_reason is not None:
                output_ids = output_ids[:response_budget]
                output_logprobs = output_logprobs[:response_budget]
                first_action = canonical_replay_action("harness_termination", {})
            else:
                calls = await self._parse(completion_ids)
                if not calls:
                    first_action = canonical_replay_action("assistant_final", {})
                    first_action_credit_eligible = True
                elif len(calls) > 1:
                    first_action = canonical_replay_action(
                        "parallel_tool_calls",
                        {"tools": [str(call["name"]) for call in calls]},
                    )
                else:
                    call = calls[0]
                    name = str(call["name"])
                    parameters = call["arguments"]
                    if name not in self.tool_names:
                        first_action = canonical_replay_action(
                            name,
                            parameters if isinstance(parameters, Mapping) else {},
                        )
                    elif parameters is None:
                        first_action = canonical_replay_action(
                            "malformed_tool_arguments", {"tool": name}
                        )
                    else:
                        first_action = canonical_replay_action(name, parameters)
                    first_action_credit_eligible = True
            if not output_ids:
                raise ActiveSuffixInfrastructureError(
                    "Stage-1 harness truncation removed the complete first decision"
                )
            first_action_span = [0, len(output_ids)]
            first_action_sha256 = replay_action_sha256(
                first_action["tool"], first_action["parameters"]
            )

        record.update(
            {
                "request_prompt_sha256": [record["prompt_token_sha256"]],
                "request_seeds": [seed],
                "sampling_attempted": True,
                "response_ids": output_ids,
                "response_mask": [1] * len(output_ids),
                "old_logprobs": output_logprobs,
                "assistant_spans": (
                    [list(first_action_span)] if first_action_span is not None else []
                ),
                "first_action": first_action,
                "first_action_sha256": first_action_sha256,
                "first_action_span": first_action_span,
                "first_action_credit_eligible": first_action_credit_eligible,
                "finish_reason": finish_reason,
                "stop_reason": stop_reason,
                "harness_limit_reason": harness_limit_reason,
                "zero_token_completion": not completion_ids,
                "infrastructure_invalid": False,
                "infrastructure_error_code": None,
                "infrastructure_error_class": None,
            }
        )
        return _with_record_content_sha256(record)

    def _invalid_record(
        self,
        group_index: int,
        proposal_index: int,
        actor_run_uid: str,
        sampling_attempt_uid: str,
        exc: Exception,
    ) -> dict[str, object]:
        record = self._base_record(
            group_index,
            proposal_index,
            actor_run_uid,
            sampling_attempt_uid,
        )
        record.update(
            {
                "request_prompt_sha256": [record["prompt_token_sha256"]],
                "request_seeds": [record["seed"]],
                "sampling_attempted": True,
                "response_ids": [],
                "response_mask": [],
                "old_logprobs": [],
                "assistant_spans": [],
                "first_action": None,
                "first_action_sha256": None,
                "first_action_span": None,
                "first_action_credit_eligible": False,
                "finish_reason": None,
                "stop_reason": None,
                "harness_limit_reason": None,
                "zero_token_completion": False,
                "infrastructure_invalid": True,
                "infrastructure_error_code": (
                    exc.code
                    if isinstance(exc, ActiveSuffixInfrastructureError)
                    else "external_failure"
                ),
                "infrastructure_error_class": exc.__class__.__name__[:128],
            }
        )
        return _with_record_content_sha256(record)

    async def collect(
        self,
        journal_dir: str | Path,
        *,
        resume: bool = False,
    ) -> dict[str, object]:
        """Fill each missing pre-registered slot exactly once, then attest the end."""
        journal = _FirstDecisionJournal(
            Path(journal_dir),
            contract=self.journal_contract,
            slot_specs=self.slot_specs,
            resume=resume,
        )
        newly_sampled = 0
        resumed_slots = len(journal.entries)
        if not journal.complete_path.exists():
            actor_run = _Stage1ActorRunAttestation(
                self.actor_checkpoint,
                self.plan["actor_checkpoint_sha256"],
            )
            journal.begin_actor_run(actor_run.start_report())
            bound_client = self._attest_backend(actor_run)
            for spec in self.slot_specs:
                if spec["slot_uid"] in journal.entries:
                    continue
                group_index = int(spec["group_index"])
                proposal_index = int(spec["proposal_index"])
                attempt = journal.orphan_attempt(spec)
                if attempt is not None:
                    interrupted = ActiveSuffixInfrastructureError(
                        "Stage-1 process ended after persisting request intent; "
                        "the slot is excluded instead of reissued",
                        code="interrupted_after_attempt_intent",
                    )
                    record = self._invalid_record(
                        group_index,
                        proposal_index,
                        str(attempt["actor_run_uid"]),
                        str(attempt["attempt_uid"]),
                        interrupted,
                    )
                    journal.commit(spec, record, interrupted_recovery=True)
                    resumed_slots += 1
                    continue
                prompt_sha256 = str(
                    self.plan["groups"][group_index]["actor_prompt_tokens"][
                        "sha256"
                    ]
                )
                attempt = journal.begin_attempt(
                    spec,
                    actor_run_uid=actor_run.run_uid,
                    request_prompt_sha256=prompt_sha256,
                    request_seed=int(spec["seed"]),
                )
                actor_run.verify_runtime(completion_request=True)
                try:
                    record = await self._sample_slot(
                        bound_client,
                        group_index,
                        proposal_index,
                        actor_run.run_uid,
                        str(attempt["attempt_uid"]),
                    )
                except Exception as exc:  # noqa: BLE001 - persist one bounded slot failure.
                    record = self._invalid_record(
                        group_index,
                        proposal_index,
                        actor_run.run_uid,
                        str(attempt["attempt_uid"]),
                        exc,
                    )
                journal.commit(spec, record)
                newly_sampled += 1
            actor_run.verify_runtime()
            self._attest_backend(actor_run)
            journal.finish_actor_run(actor_run.finish())
            journal.seal()
        records = journal.ordered_records()
        structure = classify_nested_stage1_structure(self.plan, records)
        journal_attestation = journal.content_attestation()
        actor_provenance = _actor_run_provenance_fields(
            journal, journal_attestation
        )
        invalid_records = sum(
            int(record["infrastructure_invalid"]) for record in records
        )
        excluded_states = len(structure["excluded_state_uids"])
        summary = {
            "schema_version": FIRST_DECISION_COLLECTION_VERSION,
            "proposal_version": FIRST_DECISION_PROPOSAL_VERSION,
            "plan_schema_version": self.plan["schema_version"],
            "plan_strategy_version": self.plan["strategy_version"],
            "plan_sha256": self.plan_sha256,
            "scale_ready": self.scale_ready,
            "journal_contract": deepcopy(self.journal_contract),
            "journal_attestation": journal_attestation,
            "source_provenance": deepcopy(self.source_provenance),
            "provenance": {
                "actor_checkpoint_sha256": self.plan["actor_checkpoint_sha256"],
                "actor_tokenizer_contract_sha256": (
                    self.actor_tokenizer_contract_sha256
                ),
                **actor_provenance,
                "decoding_config_sha256": self.plan["decoding_config_sha256"],
                "sampling_backend_contract_sha256": self.backend_sha256,
                "required_environment_version": self.required_environment_version,
                "environment_manifest_sha256s": self.environment_manifest_sha256s,
                "policy_reward_sha256": self.policy_reward_sha256,
                "tool_schema_sha256": self.plan["decoding_config"][
                    "tool_schema_sha256"
                ],
                "harness_contract_sha256": self.harness_contract_sha256,
            },
            "aggregate": {
                "states": self.expected_states,
                "proposals_per_state": self.proposals_per_state,
                "pre_registered_proposals": len(self.slot_specs),
                "completed_proposals": len(records),
                "newly_sampled_proposals": newly_sampled,
                "resumed_proposals": resumed_slots,
                "infrastructure_invalid_proposals": invalid_records,
                "infrastructure_invalid_proposal_rate": (
                    invalid_records / len(records)
                ),
                "eligible_states": len(structure["eligible_state_uids"]),
                "excluded_states": excluded_states,
                "state_exclusion_rate": excluded_states / self.expected_states,
                "partition_complete": len(records) == len(self.slot_specs),
            },
            "stage1_structure_sha256": _sha256_json(structure),
            "exclusion_audit": structure["exclusion_audit"],
            "safety": {
                "first_decision_only": True,
                "environment_leases": 0,
                "downstream_generation_requests": 0,
                "terminal_reward_read": False,
                "strict_outcome_read": False,
                "outcome_conditioned_resampling": False,
                "completed_slots_are_never_resampled": True,
                "started_attempts_are_never_reissued": True,
                "optimizer_enabled": False,
                "uses_hidden_goal": False,
                "scale_ready": self.scale_ready,
                "formal_source_clean": self.scale_ready,
            },
        }
        return {
            "records": records,
            "summary": summary,
            "journal_directory": str(journal.root),
        }


def write_first_decision_artifacts(
    collection: Mapping[str, object],
    *,
    records_output: str | Path,
    manifest_output: str | Path,
) -> dict[str, object]:
    """Atomically finalize plain compatibility JSONL and its bound manifest."""
    if not isinstance(collection, Mapping):
        raise TypeError("first-decision collection must be an object")
    records = collection.get("records")
    summary = collection.get("summary")
    journal_directory = collection.get("journal_directory")
    if not isinstance(records, list) or not isinstance(summary, Mapping):
        raise TypeError("first-decision collection is incomplete")
    if not isinstance(journal_directory, (str, Path)) or not str(
        journal_directory
    ):
        raise TypeError("first-decision collection is missing its journal")
    aggregate = summary.get("aggregate")
    journal_attestation = summary.get("journal_attestation")
    journal_contract = summary.get("journal_contract")
    summary_provenance = summary.get("provenance")
    summary_source_provenance = summary.get("source_provenance")
    if (
        summary.get("schema_version") != FIRST_DECISION_COLLECTION_VERSION
        or summary.get("proposal_version") != FIRST_DECISION_PROPOSAL_VERSION
        or not isinstance(aggregate, Mapping)
        or aggregate.get("partition_complete") is not True
        or aggregate.get("completed_proposals") != len(records)
        or not isinstance(journal_attestation, Mapping)
        or journal_attestation.get("completed_slots") != len(records)
        or not isinstance(journal_contract, Mapping)
        or not isinstance(journal_contract.get("slot_schedule"), list)
        or not isinstance(summary_provenance, Mapping)
        or not isinstance(summary_source_provenance, Mapping)
    ):
        raise ValueError("first-decision collection is not a complete partition")
    scale_ready = _source_scale_ready(summary_source_provenance)
    if (
        summary.get("scale_ready") is not scale_ready
        or journal_contract.get("scale_ready") is not scale_ready
    ):
        raise ValueError("first-decision scale readiness differs from its source")
    expected_contract_bindings = {
        "proposal_version": summary.get("proposal_version"),
        "plan_schema_version": summary.get("plan_schema_version"),
        "plan_strategy_version": summary.get("plan_strategy_version"),
        "plan_sha256": summary.get("plan_sha256"),
        "actor_checkpoint_sha256": summary_provenance.get(
            "actor_checkpoint_sha256"
        ),
        "actor_tokenizer_contract_sha256": summary_provenance.get(
            "actor_tokenizer_contract_sha256"
        ),
        "decoding_config_sha256": summary_provenance.get(
            "decoding_config_sha256"
        ),
        "sampling_backend_contract_sha256": summary_provenance.get(
            "sampling_backend_contract_sha256"
        ),
        "required_environment_version": summary_provenance.get(
            "required_environment_version"
        ),
        "environment_manifest_sha256s": summary_provenance.get(
            "environment_manifest_sha256s"
        ),
        "policy_reward_sha256": summary_provenance.get("policy_reward_sha256"),
        "tool_schema_sha256": summary_provenance.get("tool_schema_sha256"),
        "harness_contract_sha256": summary_provenance.get(
            "harness_contract_sha256"
        ),
        "expected_states": aggregate.get("states"),
        "proposals_per_state": aggregate.get("proposals_per_state"),
        "expected_proposals": aggregate.get("pre_registered_proposals"),
        "source_provenance": summary_source_provenance,
        "scale_ready": scale_ready,
        "optimizer_enabled": False,
        "outcome_fields_read": [],
        "uses_hidden_goal": False,
    }
    if any(
        journal_contract.get(name) != value
        for name, value in expected_contract_bindings.items()
    ):
        raise ValueError("first-decision summary differs from its journal header")
    for record in records:
        if (
            not isinstance(record, Mapping)
            or record.get("stage1_record_content_sha256")
            != _record_content_sha256(record)
            or _OUTCOME_FIELDS.intersection(record)
        ):
            raise ValueError("first-decision collection contains an invalid record")
    output_literal, output = _resolve_nonsymlink_artifact_path(
        records_output, label="records output"
    )
    manifest_literal, manifest = _resolve_nonsymlink_artifact_path(
        manifest_output, label="manifest output"
    )
    if output == manifest:
        raise ValueError("Stage-1 records and manifest outputs must differ")
    if manifest.exists():
        raise ValueError("refusing to overwrite finalized Stage-1 artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    journal = _FirstDecisionJournal(
        Path(journal_directory),
        contract=journal_contract,
        slot_specs=journal_contract["slot_schedule"],
        resume=True,
    )
    actual_attestation = journal.content_attestation()
    if actual_attestation != dict(journal_attestation):
        raise ValueError("Stage-1 sampling journal changed after collection")
    expected_actor_provenance = _actor_run_provenance_fields(
        journal, actual_attestation
    )
    if any(
        summary_provenance.get(name) != value
        for name, value in expected_actor_provenance.items()
    ):
        raise ValueError("Stage-1 actor-run provenance differs from its journal")
    journal_artifact = _completed_journal_artifact(
        journal,
        commit_directory=manifest.parent,
        expected_records=records,
    )
    records = journal.ordered_records()
    output_tmp = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    manifest_tmp = manifest.with_name(f".{manifest.name}.tmp.{os.getpid()}")
    output_tmp.unlink(missing_ok=True)
    manifest_tmp.unlink(missing_ok=True)
    try:
        with output_tmp.open("x", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        records_file_sha256 = _sha256_file(output_tmp)
        rechecked_journal_artifact = _completed_journal_artifact(
            journal,
            commit_directory=manifest.parent,
            expected_records=records,
        )
        if rechecked_journal_artifact != journal_artifact:
            raise ValueError("Stage-1 sampling journal changed during finalization")
        final_manifest = {
            "schema_version": FIRST_DECISION_FINAL_MANIFEST_VERSION,
            "status": "complete" if scale_ready else "mechanical_complete",
            "scale_ready": scale_ready,
            "collection_schema_version": summary["schema_version"],
            "proposal_version": summary["proposal_version"],
            "records_file_sha256": records_file_sha256,
            "records_canonical_sha256": _sha256_json(records),
            "record_count": len(records),
            "journal_contract_sha256": actual_attestation[
                "journal_contract_sha256"
            ],
            "journal_contract_file_sha256": actual_attestation[
                "journal_contract_file_sha256"
            ],
            "journal_content_sha256": actual_attestation[
                "journal_content_sha256"
            ],
            "journal_wal": deepcopy(journal_artifact),
            "plan_schema_version": summary["plan_schema_version"],
            "plan_strategy_version": summary["plan_strategy_version"],
            "plan_sha256": summary["plan_sha256"],
            "source_provenance": deepcopy(summary["source_provenance"]),
            "source_provenance_sha256": _sha256_json(
                summary["source_provenance"]
            ),
            "provenance": deepcopy(summary["provenance"]),
            "aggregate": deepcopy(summary["aggregate"]),
            "stage1_structure_sha256": summary["stage1_structure_sha256"],
            "exclusion_audit_sha256": _sha256_json(summary["exclusion_audit"]),
            "safety": deepcopy(summary["safety"]),
        }
        with manifest_tmp.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(final_manifest, ensure_ascii=False, indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(output_tmp, output)
        except FileExistsError:
            _recheck_nonsymlink_artifact_path(
                output_literal, output, label="records output"
            )
            if output.read_bytes() != output_tmp.read_bytes():
                raise ValueError(
                    "uncommitted Stage-1 records differ from resumed collection"
                )
        _recheck_nonsymlink_artifact_path(
            output_literal, output, label="records output"
        )
        os.link(manifest_tmp, manifest)
        _recheck_nonsymlink_artifact_path(
            manifest_literal, manifest, label="manifest output"
        )
        _fsync_directory(output.parent)
        if manifest.parent != output.parent:
            _fsync_directory(manifest.parent)
        return final_manifest
    finally:
        output_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)


def _load_jsonl_records(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Stage-1 records line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(value, Mapping):
                raise TypeError(
                    f"Stage-1 records line {line_number} must be an object"
                )
            records.append(json.loads(_canonical_json(dict(value))))
    if not records:
        raise ValueError("Stage-1 records artifact is empty")
    return records


def _resolve_manifest_journal_directory(
    manifest_file: Path, artifact: Mapping[str, object]
) -> Path:
    if not isinstance(artifact, Mapping):
        raise TypeError("Stage-1 manifest journal_wal must be an object")
    relative_value = artifact.get("relative_directory")
    if not isinstance(relative_value, str) or not relative_value:
        raise ValueError("Stage-1 journal relative directory is invalid")
    relative = Path(relative_value)
    if (
        relative.is_absolute()
        or relative.as_posix() != relative_value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("Stage-1 journal relative directory is invalid")
    literal = manifest_file.parent / relative
    if literal.is_symlink():
        raise ValueError("Stage-1 journal root must not be a symbolic link")
    root = literal.resolve()
    try:
        root.relative_to(manifest_file.parent.resolve())
    except ValueError as exc:
        raise ValueError("Stage-1 journal escapes the manifest directory") from exc
    if not root.is_dir():
        raise ValueError("Stage-1 journal directory is missing")
    return root


def verify_first_decision_artifacts(
    *,
    plan: Mapping[str, object],
    records_path: str | Path,
    manifest_path: str | Path,
    expected_source_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Verify a finalized Stage-1 JSONL before it enters nested collection."""
    normalized_plan = validate_active_branch_plan(plan)
    plan_sha256 = _sha256_json(normalized_plan)
    records_literal, records_file = _resolve_nonsymlink_artifact_path(
        records_path, label="records artifact"
    )
    manifest_literal, manifest_file = _resolve_nonsymlink_artifact_path(
        manifest_path, label="manifest artifact"
    )
    if not records_file.is_file() or not manifest_file.is_file():
        raise ValueError("finalized Stage-1 records and manifest must both exist")
    _recheck_nonsymlink_artifact_path(
        records_literal, records_file, label="records artifact"
    )
    _recheck_nonsymlink_artifact_path(
        manifest_literal, manifest_file, label="manifest artifact"
    )
    try:
        manifest_raw = manifest_file.read_bytes()
        manifest_value = json.loads(manifest_raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Stage-1 manifest is not valid JSON") from exc
    if not isinstance(manifest_value, Mapping):
        raise TypeError("Stage-1 manifest must be an object")
    manifest = json.loads(_canonical_json(dict(manifest_value)))
    expected_manifest_fields = {
        "schema_version",
        "status",
        "scale_ready",
        "collection_schema_version",
        "proposal_version",
        "records_file_sha256",
        "records_canonical_sha256",
        "record_count",
        "journal_contract_sha256",
        "journal_contract_file_sha256",
        "journal_content_sha256",
        "journal_wal",
        "plan_schema_version",
        "plan_strategy_version",
        "plan_sha256",
        "source_provenance",
        "source_provenance_sha256",
        "provenance",
        "aggregate",
        "stage1_structure_sha256",
        "exclusion_audit_sha256",
        "safety",
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError("Stage-1 manifest fields do not match the contract")
    if (
        manifest.get("schema_version") != FIRST_DECISION_FINAL_MANIFEST_VERSION
        or manifest.get("status") not in {"complete", "mechanical_complete"}
        or manifest.get("collection_schema_version")
        != FIRST_DECISION_COLLECTION_VERSION
        or manifest.get("proposal_version") != FIRST_DECISION_PROPOSAL_VERSION
    ):
        raise ValueError("Stage-1 manifest version or status mismatch")
    for name in (
        "records_file_sha256",
        "records_canonical_sha256",
        "journal_contract_sha256",
        "journal_contract_file_sha256",
        "journal_content_sha256",
        "plan_sha256",
        "stage1_structure_sha256",
        "exclusion_audit_sha256",
        "source_provenance_sha256",
    ):
        if not _is_sha256(manifest.get(name)):
            raise ValueError(f"Stage-1 manifest {name} must be a SHA256 digest")
    _recheck_nonsymlink_artifact_path(
        records_literal, records_file, label="records artifact"
    )
    _recheck_nonsymlink_artifact_path(
        manifest_literal, manifest_file, label="manifest artifact"
    )
    if _sha256_file(records_file) != manifest["records_file_sha256"]:
        raise ValueError("Stage-1 records file SHA256 mismatch")
    if (
        manifest["plan_schema_version"] != normalized_plan["schema_version"]
        or manifest["plan_strategy_version"]
        != normalized_plan["strategy_version"]
        or manifest["plan_sha256"] != plan_sha256
    ):
        raise ValueError("Stage-1 manifest plan identity mismatch")
    source_provenance = manifest.get("source_provenance")
    if not isinstance(source_provenance, Mapping):
        raise TypeError("Stage-1 source_provenance must be an object")
    if _sha256_json(dict(source_provenance)) != manifest[
        "source_provenance_sha256"
    ]:
        raise ValueError("Stage-1 source provenance SHA256 mismatch")
    source_provenance = _normalize_source_provenance(source_provenance)
    scale_ready = _source_scale_ready(source_provenance)
    expected_status = "complete" if scale_ready else "mechanical_complete"
    if (
        manifest.get("scale_ready") is not scale_ready
        or manifest.get("status") != expected_status
    ):
        raise ValueError("Stage-1 manifest scale readiness is invalid")
    if expected_source_provenance is not None and _canonical_json(
        dict(source_provenance)
    ) != _canonical_json(_normalize_source_provenance(expected_source_provenance)):
        raise ValueError("Stage-1 source provenance mismatch")

    provenance = manifest.get("provenance")
    expected_provenance_fields = {
        "actor_checkpoint_sha256",
        "actor_tokenizer_contract_sha256",
        "actor_attested_at_start",
        "actor_attested_at_end",
        "actor_stat_snapshot_sha256",
        "actor_stat_entry_count",
        "actor_runtime_stat_checks",
        "actor_run_chain_sha256",
        "actor_run_count",
        "actor_record_run_uid_count",
        "actor_all_records_covered",
        "actor_all_runs_closed",
        "decoding_config_sha256",
        "sampling_backend_contract_sha256",
        "required_environment_version",
        "environment_manifest_sha256s",
        "policy_reward_sha256",
        "tool_schema_sha256",
        "harness_contract_sha256",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != (
        expected_provenance_fields
    ):
        raise ValueError("Stage-1 provenance fields do not match the contract")
    environment_manifests = sorted(
        {group["environment_manifest_sha256"] for group in normalized_plan["groups"]}
    )
    expected_plan_provenance = {
        "actor_checkpoint_sha256": normalized_plan["actor_checkpoint_sha256"],
        "decoding_config_sha256": normalized_plan["decoding_config_sha256"],
        "sampling_backend_contract_sha256": normalized_plan["decoding_config"][
            "sampling_backend_contract_sha256"
        ],
        "environment_manifest_sha256s": environment_manifests,
        "tool_schema_sha256": normalized_plan["decoding_config"][
            "tool_schema_sha256"
        ],
    }
    tokenizer_contracts = {
        group["tokenizer_contract_sha256"] for group in normalized_plan["groups"]
    }
    if (
        len(tokenizer_contracts) != 1
        or provenance.get("actor_tokenizer_contract_sha256")
        != next(iter(tokenizer_contracts))
    ):
        raise ValueError("Stage-1 tokenizer provenance differs from the active plan")
    if any(provenance.get(name) != value for name, value in expected_plan_provenance.items()):
        raise ValueError("Stage-1 provenance differs from the active plan")
    if (
        provenance.get("actor_attested_at_start") is not True
        or provenance.get("actor_attested_at_end") is not True
        or not _is_sha256(provenance.get("actor_stat_snapshot_sha256"))
        or not isinstance(provenance.get("actor_stat_entry_count"), int)
        or isinstance(provenance.get("actor_stat_entry_count"), bool)
        or provenance["actor_stat_entry_count"] < 1
        or not isinstance(provenance.get("actor_runtime_stat_checks"), int)
        or isinstance(provenance.get("actor_runtime_stat_checks"), bool)
        or provenance["actor_runtime_stat_checks"] < 2
        or not _is_sha256(provenance.get("actor_run_chain_sha256"))
        or not isinstance(provenance.get("actor_run_count"), int)
        or isinstance(provenance.get("actor_run_count"), bool)
        or provenance["actor_run_count"] < 1
        or not isinstance(provenance.get("actor_record_run_uid_count"), int)
        or isinstance(provenance.get("actor_record_run_uid_count"), bool)
        or provenance["actor_record_run_uid_count"] < 1
        or provenance.get("actor_all_records_covered") is not True
        or provenance.get("actor_all_runs_closed") is not True
        or not isinstance(provenance.get("required_environment_version"), str)
        or not provenance["required_environment_version"]
        or not _is_sha256(provenance.get("policy_reward_sha256"))
        or not _is_sha256(provenance.get("harness_contract_sha256"))
    ):
        raise ValueError("Stage-1 runtime provenance is invalid")
    policy_reward_source = source_provenance.get("policy_reward")
    if (
        isinstance(policy_reward_source, Mapping)
        and policy_reward_source.get("resolved_sha256")
        != provenance["policy_reward_sha256"]
    ):
        raise ValueError("Stage-1 policy reward source binding mismatch")

    _recheck_nonsymlink_artifact_path(
        records_literal, records_file, label="records artifact"
    )
    _recheck_nonsymlink_artifact_path(
        manifest_literal, manifest_file, label="manifest artifact"
    )
    records = _load_jsonl_records(records_file)
    if _sha256_json(records) != manifest["records_canonical_sha256"]:
        raise ValueError("Stage-1 records canonical SHA256 mismatch")
    if manifest.get("record_count") != len(records):
        raise ValueError("Stage-1 manifest record count mismatch")
    slot_specs = _slot_specs_for_plan(normalized_plan)
    expected_journal_contract = _build_journal_contract(
        plan=normalized_plan,
        slot_specs=slot_specs,
        actor_tokenizer_contract_sha256=provenance[
            "actor_tokenizer_contract_sha256"
        ],
        sampling_backend_contract_sha256=provenance[
            "sampling_backend_contract_sha256"
        ],
        required_environment_version=provenance[
            "required_environment_version"
        ],
        environment_manifest_sha256s=provenance[
            "environment_manifest_sha256s"
        ],
        policy_reward_sha256=provenance["policy_reward_sha256"],
        harness_contract_sha256=provenance["harness_contract_sha256"],
        source_provenance=source_provenance,
    )
    if manifest["journal_contract_sha256"] != expected_journal_contract[
        "contract_sha256"
    ]:
        raise ValueError("Stage-1 journal contract differs from the manifest")
    journal_root = _resolve_manifest_journal_directory(
        manifest_file, manifest["journal_wal"]
    )
    journal = _FirstDecisionJournal(
        journal_root,
        contract=expected_journal_contract,
        slot_specs=slot_specs,
        resume=True,
    )
    actual_journal_artifact = _completed_journal_artifact(
        journal,
        commit_directory=manifest_file.parent,
        expected_records=records,
    )
    if actual_journal_artifact != manifest["journal_wal"]:
        raise ValueError("Stage-1 manifest differs from the sampling journal bytes")
    expected_actor_provenance = _actor_run_provenance_fields(
        journal, journal.content_attestation()
    )
    if any(
        provenance.get(name) != value
        for name, value in expected_actor_provenance.items()
    ):
        raise ValueError("Stage-1 actor-run chain differs from provenance")
    if (
        manifest["journal_contract_file_sha256"]
        != actual_journal_artifact["header"]["sha256"]
        or manifest["journal_content_sha256"]
        != actual_journal_artifact["journal_content_sha256"]
        or manifest["records_canonical_sha256"]
        != actual_journal_artifact["records_canonical_sha256"]
    ):
        raise ValueError("Stage-1 journal binding differs from the final manifest")
    groups_by_uid = {
        group["active_group_uid"]: group for group in normalized_plan["groups"]
    }
    for record in records:
        if (
            record.get("stage1_proposal_version")
            != FIRST_DECISION_PROPOSAL_VERSION
            or record.get("collection_mode") != "first_decision_only"
            or record.get("stage1_record_content_sha256")
            != _record_content_sha256(record)
            or not _is_sha256(record.get("actor_run_uid"))
            or not _is_sha256(record.get("sampling_attempt_uid"))
        ):
            raise ValueError("Stage-1 proposal content contract mismatch")
        group = groups_by_uid.get(record.get("active_group_uid"))
        suffix_index = record.get("suffix_index")
        if (
            group is None
            or not isinstance(suffix_index, int)
            or isinstance(suffix_index, bool)
            or not 0 <= suffix_index < len(group["suffixes"])
        ):
            raise ValueError("Stage-1 proposal plan slot is invalid")
        proposal_uid = _proposal_uid(group["active_group_uid"], suffix_index)
        expected_record_provenance = {
            "proposal_uid": proposal_uid,
            "journal_slot_uid": _slot_uid(
                plan_sha256,
                group["active_group_uid"],
                suffix_index,
                proposal_uid,
            ),
            "plan_schema_version": normalized_plan["schema_version"],
            "plan_strategy_version": normalized_plan["strategy_version"],
            "plan_sha256": plan_sha256,
            "environment_version": provenance["required_environment_version"],
            "actor_tokenizer_contract_sha256": provenance[
                "actor_tokenizer_contract_sha256"
            ],
            "environment_manifest_sha256": group[
                "environment_manifest_sha256"
            ],
            "policy_reward_sha256": provenance["policy_reward_sha256"],
            "tool_schema_sha256": provenance["tool_schema_sha256"],
            "harness_contract_sha256": provenance["harness_contract_sha256"],
        }
        if any(
            record.get(name) != value
            for name, value in expected_record_provenance.items()
        ):
            raise ValueError("Stage-1 proposal provenance mismatch")
        if _OUTCOME_FIELDS.intersection(record):
            raise ValueError("Stage-1 proposal contains rollout outcomes")

    structure = classify_nested_stage1_structure(normalized_plan, records)
    if (
        _sha256_json(structure) != manifest["stage1_structure_sha256"]
        or _sha256_json(structure["exclusion_audit"])
        != manifest["exclusion_audit_sha256"]
    ):
        raise ValueError("Stage-1 structural classification changed")
    aggregate = manifest.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise TypeError("Stage-1 manifest aggregate must be an object")
    expected_count = sum(
        len(group["suffixes"]) for group in normalized_plan["groups"]
    )
    expected_aggregate = {
        "states": len(normalized_plan["groups"]),
        "proposals_per_state": normalized_plan["suffixes_per_state"],
        "pre_registered_proposals": expected_count,
        "completed_proposals": expected_count,
        "eligible_states": len(structure["eligible_state_uids"]),
        "excluded_states": len(structure["excluded_state_uids"]),
        "state_exclusion_rate": (
            len(structure["excluded_state_uids"])
            / len(normalized_plan["groups"])
        ),
        "partition_complete": True,
    }
    infrastructure_invalid = sum(
        int(record.get("infrastructure_invalid") is True) for record in records
    )
    expected_aggregate.update(
        {
            "infrastructure_invalid_proposals": infrastructure_invalid,
            "infrastructure_invalid_proposal_rate": (
                infrastructure_invalid / expected_count
            ),
        }
    )
    if any(aggregate.get(name) != value for name, value in expected_aggregate.items()):
        raise ValueError("Stage-1 aggregate differs from the verified partition")
    if aggregate.get("newly_sampled_proposals", 0) + aggregate.get(
        "resumed_proposals", 0
    ) != expected_count:
        raise ValueError("Stage-1 aggregate sampling counts are inconsistent")
    safety = manifest.get("safety")
    required_safety = {
        "first_decision_only": True,
        "environment_leases": 0,
        "downstream_generation_requests": 0,
        "terminal_reward_read": False,
        "strict_outcome_read": False,
        "outcome_conditioned_resampling": False,
        "completed_slots_are_never_resampled": True,
        "started_attempts_are_never_reissued": True,
        "optimizer_enabled": False,
        "uses_hidden_goal": False,
        "scale_ready": scale_ready,
        "formal_source_clean": scale_ready,
    }
    if not isinstance(safety, Mapping) or any(
        safety.get(name) != value for name, value in required_safety.items()
    ):
        raise ValueError("Stage-1 safety contract mismatch")
    return {
        "manifest": manifest,
        "manifest_file_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "records": records,
        "structure": structure,
    }
