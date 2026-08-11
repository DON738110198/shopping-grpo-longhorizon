"""Crash-safe storage and actor attestation for nested continuation collection.

The completion manifest is the only commit point.  Journal entries and the
three collection artifacts may exist after an interrupted run, but consumers
must reject them until a complete manifest binds their exact bytes.
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

from shopping_grpo.training.grpo.active_suffix import (
    ActiveSuffixInfrastructureError,
    _canonical_json,
    _is_sha256,
    _sha256_json,
    sha256_actor_checkpoint,
)

ACTOR_RUN_ATTESTATION_VERSION = "shopping-actor-run-attestation-v2"
ACTOR_RUN_START_VERSION = "shopping-actor-run-start-v1"
ACTOR_RUN_CHAIN_VERSION = "shopping-actor-run-chain-v1"
NESTED_JOURNAL_VERSION = "shopping-nested-continuation-journal-v2"
NESTED_JOURNAL_ENTRY_VERSION = "shopping-nested-journal-entry-v1"
NESTED_REQUEST_INTENT_VERSION = "shopping-nested-request-intent-v1"
NESTED_JOURNAL_COMPLETE_VERSION = "shopping-nested-journal-complete-v2"
NESTED_JOURNAL_ARTIFACT_VERSION = "shopping-nested-journal-artifact-v2"
NESTED_ARTIFACT_MANIFEST_VERSION = "shopping-nested-artifact-manifest-v3"
NESTED_ARTIFACT_COMMIT_VERSION = "shopping-nested-manifest-commit-v1"
NESTED_SCALE_STORAGE_VERSION = "shopping-nested-scale-storage-v3"
NESTED_SOURCE_BINDING_VERSION = "shopping-nested-source-binding-v4"
NESTED_STAGE1_SOURCE_BINDING_VERSION = "shopping-nested-stage1-source-binding-v3"
FIRST_DECISION_FINAL_MANIFEST_VERSION = (
    "shopping-first-decision-final-manifest-v1"
)
_NESTED_COLLECTION_VERSION = "shopping-nested-continuation-collection-v3"
_NESTED_DECISION_VERSION = "shopping-nested-decision-v2"
_NESTED_CONTINUATION_VERSION = "shopping-nested-continuation-v2"
_NESTED_SEED_SCHEDULE_VERSION = "shopping-nested-state-crn-seed-v1"
_NESTED_FOLD_CONTRACT_VERSION = "shopping-nested-train4-gate4-v1"
_NESTED_ROLLOUT_CONTENT_VERSION = "shopping-nested-rollout-content-v2"


def _validate_record_request_intent_contract(
    record: Mapping[str, object],
    *,
    intents: Sequence[Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Bind every downstream request claim to its deterministic WAL intent."""
    if not isinstance(record, Mapping):
        raise TypeError("nested continuation record must be an object")
    completion_calls = record.get("completion_calls")
    seeds = record.get("downstream_request_seeds")
    prompt_hashes = record.get("downstream_prompt_sha256")
    intent_uids = record.get("completion_intent_uids")
    if (
        not isinstance(completion_calls, int)
        or isinstance(completion_calls, bool)
        or completion_calls < 0
        or not isinstance(seeds, list)
        or not isinstance(prompt_hashes, list)
        or not isinstance(intent_uids, list)
    ):
        raise ValueError("nested record request-intent cardinality is invalid")
    downstream_count = max(completion_calls - 1, 0)
    if not (
        len(seeds)
        == len(prompt_hashes)
        == len(intent_uids)
        == downstream_count
    ):
        raise ValueError("nested record request-intent cardinality is invalid")
    continuation_uid = record.get("continuation_uid")
    actor_run_uid = record.get("actor_run_uid")
    actor_stat_sha256 = record.get("actor_stat_snapshot_sha256")
    if (
        not _is_sha256(continuation_uid)
        or (downstream_count and not _is_sha256(actor_run_uid))
        or (downstream_count and not _is_sha256(actor_stat_sha256))
        or any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            for seed in seeds
        )
        or any(not _is_sha256(prompt_sha256) for prompt_sha256 in prompt_hashes)
        or any(not _is_sha256(intent_uid) for intent_uid in intent_uids)
        or len(intent_uids) != len(set(intent_uids))
    ):
        raise ValueError("nested record request-intent identity is invalid")

    expected: list[dict[str, object]] = []
    for request_index, (seed, prompt_sha256, intent_uid) in enumerate(
        zip(seeds, prompt_hashes, intent_uids, strict=True)
    ):
        identity = {
            "continuation_uid": continuation_uid,
            "actor_run_uid": actor_run_uid,
            "actor_stat_snapshot_sha256": actor_stat_sha256,
            "request_index": request_index,
            "seed": seed,
            "prompt_sha256": prompt_sha256,
        }
        expected_uid = _sha256_json(
            {"schema_version": NESTED_REQUEST_INTENT_VERSION, **identity}
        )
        if intent_uid != expected_uid:
            raise ValueError("nested record request-intent identity is invalid")
        expected_intent = {
            "schema_version": NESTED_REQUEST_INTENT_VERSION,
            "intent_uid": expected_uid,
            **identity,
        }
        expected_intent["intent_sha256"] = _sha256_json(expected_intent)
        expected.append(expected_intent)

    if intents is not None:
        actual = list(intents)
        if len(actual) != downstream_count or any(
            not isinstance(intent, Mapping) for intent in actual
        ):
            raise ValueError("nested record request-intent closure is invalid")
        if [dict(intent) for intent in actual] != expected:
            raise ValueError(
                "nested record request intents differ from request seed/prompt claims"
            )
    return expected


def _lstat_no_symlink(path: Path, name: str) -> os.stat_result | None:
    try:
        result = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(result.st_mode):
        raise ValueError(f"{name} must not be a symbolic link")
    return result


def _resolve_no_symlink_target(path: str | Path, name: str) -> Path:
    requested = Path(path).expanduser().absolute()
    target_stat = _lstat_no_symlink(requested, name)
    if target_stat is not None and not stat.S_ISREG(target_stat.st_mode):
        raise ValueError(f"{name} must be a regular file")
    return requested.resolve()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync; Windows does not permit this open mode."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, content: bytes, *, refuse_existing: bool) -> None:
    target_stat = _lstat_no_symlink(path, "atomic artifact target")
    if target_stat is not None and not stat.S_ISREG(target_stat.st_mode):
        raise ValueError("atomic artifact target must be a regular file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    if temporary.exists():
        raise ValueError(f"stale temporary artifact exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if refuse_existing:
                raise
            if path.read_bytes() != content:
                raise ValueError(f"existing artifact bytes differ: {path}")
            return
        else:
            _fsync_directory(path.parent)
            return
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    else:
        text = _canonical_json(value) + "\n"
    return text.encode("utf-8")


def _jsonl_bytes(values: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_json_bytes(dict(value)) for value in values)


def _checkpoint_stat_snapshot(root: Path) -> dict[str, object]:
    if root.is_symlink() or not root.is_dir():
        raise ActiveSuffixInfrastructureError(
            "actor checkpoint root is missing or symbolic",
            code="actor_checkpoint_runtime_drift",
        )
    entries: list[dict[str, object]] = []
    for path in [root, *sorted(root.rglob("*"))]:
        if path.is_symlink():
            raise ActiveSuffixInfrastructureError(
                "actor checkpoint gained a symbolic link",
                code="actor_checkpoint_runtime_drift",
            )
        stat = path.stat()
        if path.is_dir():
            kind = "directory"
        elif path.is_file():
            kind = "file"
        else:
            raise ActiveSuffixInfrastructureError(
                "actor checkpoint contains an unsupported filesystem entry",
                code="actor_checkpoint_runtime_drift",
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


class ActorCheckpointRunAttestation:
    """Full-hash an actor at run boundaries and stat-check it in between."""

    def __init__(self, path: str | Path, expected_sha256: str):
        if not _is_sha256(expected_sha256):
            raise ValueError("expected actor checkpoint SHA256 is invalid")
        requested = Path(path).expanduser()
        if requested.is_symlink():
            raise ValueError("actor checkpoint must not be a symbolic link")
        self.root = requested.resolve()
        self.expected_sha256 = str(expected_sha256)
        self._run_uid = _sha256_json(
            {
                "actor_checkpoint_sha256": self.expected_sha256,
                "process_id": os.getpid(),
                "nonce": uuid.uuid4().hex,
            }
        )
        before = _checkpoint_stat_snapshot(self.root)
        start_sha256 = sha256_actor_checkpoint(self.root)
        after = _checkpoint_stat_snapshot(self.root)
        if before != after:
            raise ActiveSuffixInfrastructureError(
                "actor checkpoint changed during start attestation",
                code="actor_checkpoint_start_drift",
            )
        if start_sha256 != self.expected_sha256:
            raise ActiveSuffixInfrastructureError(
                "actor checkpoint SHA256 mismatch",
                code="actor_checkpoint_start_mismatch",
            )
        self._baseline_snapshot = before
        self._snapshot_sha256 = _sha256_json(before)
        self._start_sha256 = start_sha256
        self._runtime_checks = 0
        self._completion_request_stat_checks = 0
        self._finished = False
        self._end_sha256: str | None = None

    def verify_runtime(
        self,
        *,
        actor_checkpoint: str | Path | None = None,
        actor_checkpoint_sha256: str | None = None,
        completion_request: bool = False,
    ) -> dict[str, object]:
        if self._finished:
            raise ActiveSuffixInfrastructureError(
                "actor run attestation is already closed",
                code="actor_checkpoint_attestation_closed",
            )
        if actor_checkpoint is not None:
            requested = Path(actor_checkpoint).expanduser()
            if requested.is_symlink() or requested.resolve() != self.root:
                raise ActiveSuffixInfrastructureError(
                    "actor checkpoint path changed during collection",
                    code="actor_checkpoint_runtime_drift",
                )
        if (
            actor_checkpoint_sha256 is not None
            and str(actor_checkpoint_sha256) != self.expected_sha256
        ):
            raise ActiveSuffixInfrastructureError(
                "actor checkpoint identity changed during collection",
                code="actor_checkpoint_runtime_drift",
            )
        current = _checkpoint_stat_snapshot(self.root)
        if current != self._baseline_snapshot:
            raise ActiveSuffixInfrastructureError(
                "actor checkpoint stat/content binding drifted during collection",
                code="actor_checkpoint_runtime_drift",
            )
        self._runtime_checks += 1
        if completion_request:
            self._completion_request_stat_checks += 1
        return {
            "schema_version": ACTOR_RUN_ATTESTATION_VERSION,
            "actor_checkpoint_sha256": self.expected_sha256,
            "actor_run_uid": self._run_uid,
            "stat_snapshot_sha256": self._snapshot_sha256,
            "stat_entry_count": self._baseline_snapshot["entry_count"],
            "read_only_run_binding": True,
        }

    def finish(self) -> dict[str, object]:
        if self._finished:
            raise ActiveSuffixInfrastructureError(
                "actor run attestation was finalized twice",
                code="actor_checkpoint_attestation_closed",
            )
        self.verify_runtime()
        before = _checkpoint_stat_snapshot(self.root)
        end_sha256 = sha256_actor_checkpoint(self.root)
        after = _checkpoint_stat_snapshot(self.root)
        if before != self._baseline_snapshot or after != self._baseline_snapshot:
            raise ActiveSuffixInfrastructureError(
                "actor checkpoint changed during end attestation",
                code="actor_checkpoint_end_drift",
            )
        if end_sha256 != self.expected_sha256:
            raise ActiveSuffixInfrastructureError(
                "actor checkpoint end SHA256 mismatch",
                code="actor_checkpoint_end_mismatch",
            )
        self._finished = True
        self._end_sha256 = end_sha256
        return self.report()

    def report(self) -> dict[str, object]:
        return {
            "schema_version": ACTOR_RUN_ATTESTATION_VERSION,
            "actor_checkpoint_sha256": self.expected_sha256,
            "actor_run_uid": self._run_uid,
            "start_full_sha256": self._start_sha256,
            "end_full_sha256": self._end_sha256,
            "stat_snapshot_sha256": self._snapshot_sha256,
            "stat_entry_count": self._baseline_snapshot["entry_count"],
            "runtime_stat_checks": self._runtime_checks,
            "completion_request_stat_checks": (
                self._completion_request_stat_checks
            ),
            "read_only_run_binding": True,
            "completed": self._finished,
            "drift_detected": False,
        }


def validate_actor_run_attestation(
    report: Mapping[str, object], *, expected_sha256: str
) -> dict[str, object]:
    if not isinstance(report, Mapping):
        raise TypeError("actor run attestation must be an object")
    expected_fields = {
        "schema_version",
        "actor_checkpoint_sha256",
        "actor_run_uid",
        "start_full_sha256",
        "end_full_sha256",
        "stat_snapshot_sha256",
        "stat_entry_count",
        "runtime_stat_checks",
        "completion_request_stat_checks",
        "read_only_run_binding",
        "completed",
        "drift_detected",
    }
    if set(report) != expected_fields:
        raise ValueError("actor run attestation fields do not match the contract")
    if report.get("schema_version") != ACTOR_RUN_ATTESTATION_VERSION:
        raise ValueError("actor run attestation version mismatch")
    if any(
        report.get(name) != expected_sha256
        for name in (
            "actor_checkpoint_sha256",
            "start_full_sha256",
            "end_full_sha256",
        )
    ):
        raise ValueError("actor run boundary hashes do not match the planned actor")
    if not _is_sha256(report.get("actor_run_uid")):
        raise ValueError("actor run UID is invalid")
    if not _is_sha256(report.get("stat_snapshot_sha256")):
        raise ValueError("actor stat snapshot SHA256 is invalid")
    for name in ("stat_entry_count", "runtime_stat_checks"):
        value = report.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"actor run {name} must be a positive integer")
    completion_checks = report.get("completion_request_stat_checks")
    if (
        not isinstance(completion_checks, int)
        or isinstance(completion_checks, bool)
        or completion_checks < 0
        or completion_checks > report["runtime_stat_checks"]
    ):
        raise ValueError("actor completion-request stat checks are invalid")
    if (
        report.get("read_only_run_binding") is not True
        or report.get("completed") is not True
        or report.get("drift_detected") is not False
    ):
        raise ValueError("actor run attestation did not complete without drift")
    return deepcopy(dict(report))


def _actor_run_start(report: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(report, Mapping):
        raise TypeError("actor run start attestation must be an object")
    if (
        report.get("schema_version") != ACTOR_RUN_ATTESTATION_VERSION
        or not _is_sha256(report.get("actor_checkpoint_sha256"))
        or report.get("start_full_sha256")
        != report.get("actor_checkpoint_sha256")
        or not _is_sha256(report.get("actor_run_uid"))
        or not _is_sha256(report.get("stat_snapshot_sha256"))
        or not isinstance(report.get("stat_entry_count"), int)
        or isinstance(report.get("stat_entry_count"), bool)
        or report["stat_entry_count"] < 1
        or report.get("read_only_run_binding") is not True
        or report.get("drift_detected") is not False
    ):
        raise ValueError("actor run start attestation is invalid")
    return {
        "schema_version": ACTOR_RUN_START_VERSION,
        "actor_checkpoint_sha256": report["actor_checkpoint_sha256"],
        "actor_run_uid": report["actor_run_uid"],
        "start_full_sha256": report["start_full_sha256"],
        "stat_snapshot_sha256": report["stat_snapshot_sha256"],
        "stat_entry_count": report["stat_entry_count"],
        "read_only_run_binding": True,
    }


def validate_nested_stage1_source_binding(
    binding: Mapping[str, object],
    *,
    expected_records_sha256: str | None = None,
) -> dict[str, object]:
    """Validate the transitive Stage-1 completion-manifest binding."""
    if not isinstance(binding, Mapping):
        raise TypeError("nested Stage-1 source binding must be an object")
    expected_fields = {
        "schema_version",
        "final_manifest_schema_version",
        "final_manifest_sha256",
        "records_file_sha256",
        "records_canonical_sha256",
        "record_count",
        "journal_contract_sha256",
        "journal_contract_file_sha256",
        "journal_content_sha256",
        "active_plan_sha256",
        "stage1_structure_sha256",
        "source_provenance_sha256",
        "source_git_sha",
        "source_git_worktree_clean",
        "source_git_status_sha256",
        "policy_reward_sha256",
        "harness_contract_sha256",
        "required_environment_version",
        "environment_manifest_sha256s",
    }
    if set(binding) != expected_fields:
        raise ValueError("nested Stage-1 source binding fields do not match")
    if (
        binding.get("schema_version") != NESTED_STAGE1_SOURCE_BINDING_VERSION
        or binding.get("final_manifest_schema_version")
        != FIRST_DECISION_FINAL_MANIFEST_VERSION
    ):
        raise ValueError("nested Stage-1 source binding version mismatch")
    digest_fields = expected_fields - {
        "schema_version",
        "final_manifest_schema_version",
        "record_count",
        "environment_manifest_sha256s",
        "required_environment_version",
        "source_git_sha",
        "source_git_worktree_clean",
    }
    if any(not _is_sha256(binding.get(name)) for name in digest_fields):
        raise ValueError("nested Stage-1 source binding contains an invalid digest")
    count = binding.get("record_count")
    manifests = binding.get("environment_manifest_sha256s")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 1
        or not isinstance(manifests, list)
        or not manifests
        or manifests != sorted(set(manifests))
        or any(not _is_sha256(value) for value in manifests)
    ):
        raise ValueError("nested Stage-1 source cardinality is invalid")
    if (
        not isinstance(binding.get("required_environment_version"), str)
        or not binding["required_environment_version"]
    ):
        raise ValueError("nested Stage-1 environment version is invalid")
    git_sha = binding.get("source_git_sha")
    if (
        not isinstance(git_sha, str)
        or len(git_sha) != 40
        or any(character not in "0123456789abcdef" for character in git_sha)
        or binding.get("source_git_worktree_clean") is not True
        or binding.get("source_git_status_sha256")
        != hashlib.sha256(b"").hexdigest()
    ):
        raise ValueError("nested Stage-1 source was not collected from clean Git")
    if (
        expected_records_sha256 is not None
        and binding.get("records_file_sha256") != expected_records_sha256
    ):
        raise ValueError("nested Stage-1 records differ from the finalized source")
    return deepcopy(dict(binding))


def build_nested_stage1_source_binding(
    manifest: Mapping[str, object],
    *,
    final_manifest_sha256: str,
) -> dict[str, object]:
    """Project a verified Stage-1 manifest into the Stage-2 source contract."""
    if not isinstance(manifest, Mapping):
        raise TypeError("finalized Stage-1 manifest must be an object")
    provenance = manifest.get("provenance")
    source_provenance = manifest.get("source_provenance")
    if (
        manifest.get("schema_version") != FIRST_DECISION_FINAL_MANIFEST_VERSION
        or not isinstance(provenance, Mapping)
        or not isinstance(source_provenance, Mapping)
    ):
        raise ValueError("Stage-1 manifest is not a completed formal source")
    if (
        source_provenance.get("git_worktree_clean") is not True
        or source_provenance.get("dirty_source_override") is not False
        or source_provenance.get("execution_mode") != "formal"
        or source_provenance.get("scale_ready") is not True
    ):
        raise ValueError("nested Stage-1 source was not collected from clean Git")
    if manifest.get("status") != "complete" or manifest.get("scale_ready") is not True:
        raise ValueError("Stage-1 manifest is not a completed formal source")
    binding = {
        "schema_version": NESTED_STAGE1_SOURCE_BINDING_VERSION,
        "final_manifest_schema_version": manifest["schema_version"],
        "final_manifest_sha256": final_manifest_sha256,
        "records_file_sha256": manifest["records_file_sha256"],
        "records_canonical_sha256": manifest["records_canonical_sha256"],
        "record_count": manifest["record_count"],
        "journal_contract_sha256": manifest["journal_contract_sha256"],
        "journal_contract_file_sha256": manifest[
            "journal_contract_file_sha256"
        ],
        "journal_content_sha256": manifest["journal_content_sha256"],
        "active_plan_sha256": manifest["plan_sha256"],
        "stage1_structure_sha256": manifest["stage1_structure_sha256"],
        "source_provenance_sha256": manifest["source_provenance_sha256"],
        "source_git_sha": source_provenance.get("git_sha"),
        "source_git_worktree_clean": source_provenance.get(
            "git_worktree_clean"
        ),
        "source_git_status_sha256": source_provenance.get(
            "git_status_sha256"
        ),
        "policy_reward_sha256": provenance["policy_reward_sha256"],
        "harness_contract_sha256": provenance["harness_contract_sha256"],
        "required_environment_version": provenance[
            "required_environment_version"
        ],
        "environment_manifest_sha256s": provenance[
            "environment_manifest_sha256s"
        ],
    }
    return validate_nested_stage1_source_binding(binding)


def build_nested_journal_contract(
    *,
    experiment_uid: str,
    active_plan_sha256: str,
    formal_plan_sha256: str,
    resolved_selections_sha256: str,
    stage1_source_sha256: str,
    stage1_manifest_sha256: str | None,
    stage1_source_finalized: bool,
    harness_contract_sha256: str,
    actor_runtime_binding: Mapping[str, object],
    continuations_per_decision: int,
    expected_continuation_uids: Sequence[str],
) -> dict[str, object]:
    digest_values = (
        experiment_uid,
        active_plan_sha256,
        formal_plan_sha256,
        resolved_selections_sha256,
        stage1_source_sha256,
        harness_contract_sha256,
    )
    if any(not _is_sha256(value) for value in digest_values):
        raise ValueError("nested journal source digest is invalid")
    if not isinstance(stage1_source_finalized, bool) or (
        stage1_source_finalized != _is_sha256(stage1_manifest_sha256)
    ):
        raise ValueError("nested journal Stage-1 finalization binding is invalid")
    if (
        not isinstance(actor_runtime_binding, Mapping)
        or actor_runtime_binding.get("read_only_run_binding") is not True
        or not _is_sha256(actor_runtime_binding.get("actor_checkpoint_sha256"))
        or not _is_sha256(actor_runtime_binding.get("stat_snapshot_sha256"))
    ):
        raise ValueError("nested journal actor runtime binding is invalid")
    stable_actor_binding = {
        "schema_version": actor_runtime_binding.get("schema_version"),
        "actor_checkpoint_sha256": actor_runtime_binding.get(
            "actor_checkpoint_sha256"
        ),
        "stat_snapshot_sha256": actor_runtime_binding.get(
            "stat_snapshot_sha256"
        ),
        "stat_entry_count": actor_runtime_binding.get("stat_entry_count"),
        "read_only_run_binding": True,
    }
    if (
        not isinstance(continuations_per_decision, int)
        or isinstance(continuations_per_decision, bool)
        or continuations_per_decision < 1
    ):
        raise ValueError("nested journal continuation cardinality is invalid")
    uids = list(expected_continuation_uids)
    if any(not _is_sha256(uid) for uid in uids) or len(uids) != len(set(uids)):
        raise ValueError("nested journal continuation UID schedule is invalid")
    return {
        "experiment_uid": experiment_uid,
        "plan_sha256": active_plan_sha256,
        "formal_plan_sha256": formal_plan_sha256,
        "resolved_selections_sha256": resolved_selections_sha256,
        "stage1_source_sha256": stage1_source_sha256,
        "stage1_manifest_sha256": stage1_manifest_sha256,
        "stage1_source_finalized": stage1_source_finalized,
        "harness_contract_sha256": harness_contract_sha256,
        "actor_runtime_binding": stable_actor_binding,
        "continuations_per_decision": continuations_per_decision,
        "expected_continuation_uids": uids,
    }


def nested_journal_header_sha256(contract: Mapping[str, object]) -> str:
    if not isinstance(contract, Mapping):
        raise TypeError("nested journal contract must be an object")
    return _sha256_json(
        {
            "schema_version": NESTED_JOURNAL_VERSION,
            "contract": json.loads(_canonical_json(dict(contract))),
        }
    )


class NestedContinuationJournal:
    """One atomically written envelope per deterministic continuation UID."""

    def __init__(self, root: str | Path, contract: Mapping[str, object]):
        if not isinstance(contract, Mapping):
            raise TypeError("nested journal contract must be an object")
        requested_root = Path(root).expanduser().absolute()
        _lstat_no_symlink(requested_root, "nested journal root")
        self.root = requested_root.resolve()
        self.entries_root = self.root / "entries"
        self.intents_root = self.root / "request_intents"
        self.actor_runs_root = self.root / "actor_runs"
        self.header_path = self.root / "header.json"
        self.complete_path = self.root / "complete.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.entries_root.mkdir(parents=True, exist_ok=True)
        self.intents_root.mkdir(parents=True, exist_ok=True)
        self.actor_runs_root.mkdir(parents=True, exist_ok=True)
        root_stat = _lstat_no_symlink(self.root, "nested journal root")
        entries_stat = _lstat_no_symlink(
            self.entries_root, "nested journal entries directory"
        )
        intents_stat = _lstat_no_symlink(
            self.intents_root, "nested journal request-intents directory"
        )
        actor_runs_stat = _lstat_no_symlink(
            self.actor_runs_root, "nested journal actor-runs directory"
        )
        if (
            root_stat is None
            or not stat.S_ISDIR(root_stat.st_mode)
            or entries_stat is None
            or not stat.S_ISDIR(entries_stat.st_mode)
            or intents_stat is None
            or not stat.S_ISDIR(intents_stat.st_mode)
            or actor_runs_stat is None
            or not stat.S_ISDIR(actor_runs_stat.st_mode)
        ):
            raise ValueError("nested journal directories are invalid")
        normalized_contract = json.loads(_canonical_json(dict(contract)))
        raw_uids = normalized_contract.get("expected_continuation_uids")
        if (
            not isinstance(raw_uids, list)
            or any(not _is_sha256(uid) for uid in raw_uids)
            or len(raw_uids) != len(set(raw_uids))
        ):
            raise ValueError("journal contract expected UIDs are invalid")
        self.expected_uids = tuple(str(uid) for uid in raw_uids)
        self.expected_uid_set = set(self.expected_uids)
        self.header = {
            "schema_version": NESTED_JOURNAL_VERSION,
            "contract": normalized_contract,
        }
        self.header_sha256 = _sha256_json(self.header)
        header_bytes = _json_bytes(self.header, pretty=True)
        header_stat = _lstat_no_symlink(
            self.header_path, "nested journal header"
        )
        if header_stat is not None:
            if not stat.S_ISREG(header_stat.st_mode):
                raise ValueError("nested journal header must be a regular file")
            if self.header_path.read_bytes() != header_bytes:
                raise ValueError("nested journal header differs from the requested run")
        else:
            _atomic_write_bytes(self.header_path, header_bytes, refuse_existing=False)
        self._actor_chain_required = isinstance(
            normalized_contract.get("actor_runtime_binding"), Mapping
        )
        self._actor_starts, self._actor_ends = self._load_actor_runs()
        self._intents = self._load_request_intents()
        self._entries = self._load_entries()
        self._validate_request_intent_closure()
        complete_stat = _lstat_no_symlink(
            self.complete_path, "nested journal completion marker"
        )
        if complete_stat is not None:
            if not stat.S_ISREG(complete_stat.st_mode):
                raise ValueError(
                    "nested journal completion marker must be a regular file"
                )
            self._verify_complete_marker()

    @staticmethod
    def _actor_file_parts(path: Path, suffix: str) -> tuple[int, str]:
        tail = f".{suffix}.json"
        if not path.name.endswith(tail):
            raise ValueError("nested journal actor-run filename is invalid")
        stem = path.name[: -len(tail)]
        sequence_text, separator, actor_run_uid = stem.partition("-")
        if (
            separator != "-"
            or len(sequence_text) != 8
            or not sequence_text.isdigit()
            or not _is_sha256(actor_run_uid)
        ):
            raise ValueError("nested journal actor-run filename is invalid")
        return int(sequence_text), actor_run_uid

    def _actor_path(self, sequence: int, actor_run_uid: str, suffix: str) -> Path:
        return self.actor_runs_root / (
            f"{sequence:08d}-{actor_run_uid}.{suffix}.json"
        )

    def _validate_actor_start(
        self, value: object, *, sequence: int, filename_uid: str
    ) -> dict[str, object]:
        if not isinstance(value, Mapping):
            raise TypeError("nested actor-run start must be an object")
        expected_fields = {
            "schema_version",
            "sequence_index",
            "predecessor_actor_run_uid",
            "actor_checkpoint_sha256",
            "actor_run_uid",
            "start_full_sha256",
            "stat_snapshot_sha256",
            "stat_entry_count",
            "read_only_run_binding",
            "start_attestation_sha256",
        }
        payload = dict(value)
        claimed = payload.pop("start_attestation_sha256", None)
        binding = self.header["contract"].get("actor_runtime_binding") or {}
        predecessor = value.get("predecessor_actor_run_uid")
        if (
            set(value) != expected_fields
            or value.get("schema_version") != ACTOR_RUN_START_VERSION
            or value.get("sequence_index") != sequence
            or value.get("actor_run_uid") != filename_uid
            or (predecessor is not None and not _is_sha256(predecessor))
            or value.get("actor_checkpoint_sha256")
            != binding.get("actor_checkpoint_sha256")
            or value.get("start_full_sha256")
            != binding.get("actor_checkpoint_sha256")
            or value.get("stat_snapshot_sha256")
            != binding.get("stat_snapshot_sha256")
            or value.get("stat_entry_count") != binding.get("stat_entry_count")
            or value.get("read_only_run_binding") is not True
            or not _is_sha256(claimed)
            or claimed != _sha256_json(payload)
        ):
            raise ValueError("nested actor-run start attestation is invalid")
        return json.loads(_canonical_json(dict(value)))

    def _load_actor_runs(
        self,
    ) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
        starts_by_sequence: dict[int, dict[str, object]] = {}
        ends: dict[str, dict[str, object]] = {}
        for path in self.actor_runs_root.iterdir():
            path_stat = _lstat_no_symlink(path, "nested journal actor-run file")
            if path_stat is None or not stat.S_ISREG(path_stat.st_mode):
                raise ValueError("nested journal contains an invalid actor-run path")
            if path.name.startswith(".") and path.suffix == ".tmp":
                continue
            if path.name.endswith(".start.json"):
                sequence, uid = self._actor_file_parts(path, "start")
                if sequence in starts_by_sequence:
                    raise ValueError("nested actor-run sequence is duplicated")
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError("nested actor-run start is invalid") from exc
                starts_by_sequence[sequence] = self._validate_actor_start(
                    value, sequence=sequence, filename_uid=uid
                )
            elif path.name.endswith(".end.json"):
                sequence, uid = self._actor_file_parts(path, "end")
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError("nested actor-run end is invalid") from exc
                end = validate_actor_run_attestation(
                    value,
                    expected_sha256=str(
                        (self.header["contract"].get("actor_runtime_binding") or {}).get(
                            "actor_checkpoint_sha256"
                        )
                        or ""
                    ),
                )
                if end["actor_run_uid"] != uid or uid in ends:
                    raise ValueError("nested actor-run end identity is invalid")
                end["_sequence_index"] = sequence
                ends[uid] = end
            else:
                raise ValueError("nested journal contains an unexpected actor-run file")
        if sorted(starts_by_sequence) != list(range(len(starts_by_sequence))):
            raise ValueError("nested actor-run sequence is not contiguous")
        starts = [starts_by_sequence[index] for index in range(len(starts_by_sequence))]
        for index, start in enumerate(starts):
            expected_predecessor = (
                None if index == 0 else starts[index - 1]["actor_run_uid"]
            )
            if start["predecessor_actor_run_uid"] != expected_predecessor:
                raise ValueError("nested actor-run predecessor chain is invalid")
        start_by_uid = {str(start["actor_run_uid"]): start for start in starts}
        if len(start_by_uid) != len(starts) or set(ends) - set(start_by_uid):
            raise ValueError("nested actor-run start/end set is invalid")
        for uid, end in ends.items():
            if end.pop("_sequence_index") != start_by_uid[uid]["sequence_index"]:
                raise ValueError("nested actor-run end sequence is invalid")
            start = start_by_uid[uid]
            if (
                end["start_full_sha256"] != start["start_full_sha256"]
                or end["stat_snapshot_sha256"] != start["stat_snapshot_sha256"]
                or end["stat_entry_count"] != start["stat_entry_count"]
            ):
                raise ValueError("nested actor-run end differs from its start")
        return starts, ends

    @property
    def is_complete(self) -> bool:
        return _lstat_no_symlink(
            self.complete_path, "nested journal completion marker"
        ) is not None

    def begin_actor_run(self, report: Mapping[str, object]) -> dict[str, object]:
        if not self._actor_chain_required:
            raise ValueError("nested journal contract has no actor-run binding")
        if self.is_complete:
            raise ValueError("cannot begin an actor run in a completed journal")
        core = _actor_run_start(report)
        existing = next(
            (
                start
                for start in self._actor_starts
                if start["actor_run_uid"] == core["actor_run_uid"]
            ),
            None,
        )
        if existing is not None:
            comparable = {
                key: existing[key]
                for key in core
            }
            if comparable != core:
                raise ValueError("idempotent actor-run start changed identity")
            return deepcopy(existing)
        sequence = len(self._actor_starts)
        start = {
            **core,
            "sequence_index": sequence,
            "predecessor_actor_run_uid": (
                self._actor_starts[-1]["actor_run_uid"]
                if self._actor_starts
                else None
            ),
        }
        start["start_attestation_sha256"] = _sha256_json(start)
        path = self._actor_path(sequence, str(core["actor_run_uid"]), "start")
        _lstat_no_symlink(path, "nested actor-run start")
        _atomic_write_bytes(path, _json_bytes(start, pretty=True), refuse_existing=True)
        self._actor_starts.append(start)
        return deepcopy(start)

    def finish_actor_run(self, report: Mapping[str, object]) -> dict[str, object]:
        if self.is_complete:
            raise ValueError("cannot finish an actor run in a completed journal")
        expected_sha = str(
            (self.header["contract"].get("actor_runtime_binding") or {}).get(
                "actor_checkpoint_sha256"
            )
            or ""
        )
        end = validate_actor_run_attestation(report, expected_sha256=expected_sha)
        uid = str(end["actor_run_uid"])
        start = next(
            (item for item in self._actor_starts if item["actor_run_uid"] == uid),
            None,
        )
        if start is None:
            raise ValueError("actor run ended without a persisted start")
        if (
            end["start_full_sha256"] != start["start_full_sha256"]
            or end["stat_snapshot_sha256"] != start["stat_snapshot_sha256"]
            or end["stat_entry_count"] != start["stat_entry_count"]
        ):
            raise ValueError("actor run end differs from its persisted start")
        path = self._actor_path(int(start["sequence_index"]), uid, "end")
        content = _json_bytes(end, pretty=True)
        _lstat_no_symlink(path, "nested actor-run end")
        if uid in self._actor_ends:
            if self._actor_ends[uid] != end or path.read_bytes() != content:
                raise ValueError("idempotent actor-run end changed attestation")
        else:
            _atomic_write_bytes(path, content, refuse_existing=True)
            self._actor_ends[uid] = end
        return deepcopy(end)

    def latest_actor_attestation(self) -> dict[str, object]:
        if not self._actor_starts:
            raise ValueError("nested journal has no actor-run attestation")
        uid = str(self._actor_starts[-1]["actor_run_uid"])
        if uid not in self._actor_ends:
            raise ValueError("latest nested actor run is not gracefully closed")
        return deepcopy(self._actor_ends[uid])

    def actor_run_chain_report(
        self, *, require_complete: bool
    ) -> dict[str, object]:
        if not self._actor_chain_required:
            report = {
                "schema_version": ACTOR_RUN_CHAIN_VERSION,
                "actor_checkpoint_sha256": None,
                "stat_snapshot_sha256": None,
                "run_count": 0,
                "record_count": len(self._entries),
                "request_intent_count": len(self._intents),
                "runs": [],
                "all_records_covered": not self._entries,
                "all_request_intents_covered": not self._intents,
            }
            report["actor_run_chain_sha256"] = _sha256_json(report)
            return report
        if not self._actor_starts:
            raise ValueError("nested journal lacks a persisted actor-run start")
        records_by_run: dict[str, list[Mapping[str, object]]] = {}
        for envelope in self._entries.values():
            record = envelope.get("record")
            if not isinstance(record, Mapping):
                raise TypeError("nested journal record is invalid")
            uid = record.get("actor_run_uid")
            records_by_run.setdefault(str(uid), []).append(record)
        intents_by_run: dict[str, list[Mapping[str, object]]] = {}
        for intent in self._intents.values():
            intents_by_run.setdefault(str(intent["actor_run_uid"]), []).append(
                intent
            )
        start_uids = {str(start["actor_run_uid"]) for start in self._actor_starts}
        if (set(records_by_run) | set(intents_by_run)) - start_uids:
            raise ValueError(
                "nested records or requests reference an unattested actor run"
            )
        runs = []
        for index, start in enumerate(self._actor_starts):
            uid = str(start["actor_run_uid"])
            end = self._actor_ends.get(uid)
            successor = (
                self._actor_starts[index + 1]
                if index + 1 < len(self._actor_starts)
                else None
            )
            if end is not None:
                closure_kind = "graceful_end"
                closure_uid = None
            elif successor is not None:
                if (
                    successor["predecessor_actor_run_uid"] != uid
                    or successor["actor_checkpoint_sha256"]
                    != start["actor_checkpoint_sha256"]
                    or successor["start_full_sha256"]
                    != start["start_full_sha256"]
                    or successor["stat_snapshot_sha256"]
                    != start["stat_snapshot_sha256"]
                    or successor["stat_entry_count"] != start["stat_entry_count"]
                ):
                    raise ValueError("actor successor-start continuity is invalid")
                closure_kind = "successor_start"
                closure_uid = successor["actor_run_uid"]
            else:
                closure_kind = "open"
                closure_uid = None
                if require_complete:
                    raise ValueError("latest nested actor run lacks a closing attestation")
            records = records_by_run.get(uid, [])
            intents = intents_by_run.get(uid, [])
            if any(
                intent["actor_stat_snapshot_sha256"]
                != start["stat_snapshot_sha256"]
                for intent in intents
            ):
                raise ValueError("actor-run request-intent coverage is invalid")
            completion_calls = 0
            checked_calls = 0
            for record in records:
                record_calls = record.get("completion_calls")
                record_checked = record.get("actor_stat_checked_completion_calls")
                if (
                    not isinstance(record_calls, int)
                    or isinstance(record_calls, bool)
                    or record_calls < 0
                    or record_checked != record_calls
                    or record.get("actor_stat_snapshot_sha256")
                    != start["stat_snapshot_sha256"]
                ):
                    raise ValueError("actor-run record coverage is invalid")
                completion_calls += record_calls
                checked_calls += int(record_checked)
            if (
                end is not None
                and end["completion_request_stat_checks"] < completion_calls
            ):
                raise ValueError("actor-run end does not cover persisted requests")
            runs.append(
                {
                    "sequence_index": index,
                    "actor_run_uid": uid,
                    "predecessor_actor_run_uid": start[
                        "predecessor_actor_run_uid"
                    ],
                    "start_attestation_sha256": start[
                        "start_attestation_sha256"
                    ],
                    "end_attestation_sha256": (
                        _sha256_json(end) if end is not None else None
                    ),
                    "closure_kind": closure_kind,
                    "closure_successor_actor_run_uid": closure_uid,
                    "record_count": len(records),
                    "request_intent_count": len(intents),
                    "completion_calls": completion_calls,
                    "actor_stat_checked_completion_calls": checked_calls,
                }
            )
        binding = self.header["contract"]["actor_runtime_binding"]
        report = {
            "schema_version": ACTOR_RUN_CHAIN_VERSION,
            "actor_checkpoint_sha256": binding["actor_checkpoint_sha256"],
            "stat_snapshot_sha256": binding["stat_snapshot_sha256"],
            "run_count": len(runs),
            "record_count": len(self._entries),
            "request_intent_count": len(self._intents),
            "runs": runs,
            "all_records_covered": sum(run["record_count"] for run in runs)
            == len(self._entries),
            "all_request_intents_covered": sum(
                run["request_intent_count"] for run in runs
            )
            == len(self._intents),
        }
        if (
            report["all_records_covered"] is not True
            or report["all_request_intents_covered"] is not True
        ):
            raise ValueError(
                "actor-run chain does not cover every journal record/request"
            )
        report["actor_run_chain_sha256"] = _sha256_json(report)
        return report

    def _validate_request_intent(self, value: object) -> dict[str, object]:
        if not isinstance(value, Mapping):
            raise TypeError("nested completion-request intent must be an object")
        expected_fields = {
            "schema_version",
            "intent_uid",
            "continuation_uid",
            "actor_run_uid",
            "actor_stat_snapshot_sha256",
            "request_index",
            "seed",
            "prompt_sha256",
            "intent_sha256",
        }
        payload = dict(value)
        claimed_content_sha = payload.pop("intent_sha256", None)
        identity = {
            name: value.get(name)
            for name in (
                "continuation_uid",
                "actor_run_uid",
                "actor_stat_snapshot_sha256",
                "request_index",
                "seed",
                "prompt_sha256",
            )
        }
        if (
            set(value) != expected_fields
            or value.get("schema_version") != NESTED_REQUEST_INTENT_VERSION
            or not _is_sha256(value.get("intent_uid"))
            or value.get("intent_uid")
            != _sha256_json(
                {"schema_version": NESTED_REQUEST_INTENT_VERSION, **identity}
            )
            or value.get("continuation_uid") not in self.expected_uid_set
            or not _is_sha256(value.get("actor_run_uid"))
            or not _is_sha256(value.get("actor_stat_snapshot_sha256"))
            or not isinstance(value.get("request_index"), int)
            or isinstance(value.get("request_index"), bool)
            or value["request_index"] < 0
            or not isinstance(value.get("seed"), int)
            or isinstance(value.get("seed"), bool)
            or value["seed"] < 0
            or not _is_sha256(value.get("prompt_sha256"))
            or not _is_sha256(claimed_content_sha)
            or claimed_content_sha != _sha256_json(payload)
        ):
            raise ValueError("nested completion-request intent is invalid")
        return json.loads(_canonical_json(dict(value)))

    def _load_request_intents(self) -> dict[str, dict[str, object]]:
        intents: dict[str, dict[str, object]] = {}
        for path in self.intents_root.iterdir():
            path_stat = _lstat_no_symlink(
                path, "nested journal request-intent path"
            )
            if path_stat is None or not stat.S_ISREG(path_stat.st_mode):
                raise ValueError("nested journal request-intent path is invalid")
            if path.name.startswith(".") and path.suffix == ".tmp":
                continue
            if path.suffix != ".json" or not _is_sha256(path.stem):
                raise ValueError(
                    "nested journal contains an unexpected request-intent file"
                )
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("nested completion-request intent is invalid") from exc
            intent = self._validate_request_intent(value)
            if intent["intent_uid"] != path.stem or path.stem in intents:
                raise ValueError("nested completion-request intent filename changed")
            intents[path.stem] = intent
        return intents

    def begin_completion_request(
        self, intent: Mapping[str, object]
    ) -> dict[str, object]:
        if not self._actor_chain_required:
            raise ValueError("nested journal contract has no request-intent binding")
        if self.is_complete:
            raise ValueError("cannot begin a request in a completed nested journal")
        core_fields = {
            "continuation_uid",
            "actor_run_uid",
            "actor_stat_snapshot_sha256",
            "request_index",
            "seed",
            "prompt_sha256",
        }
        if not isinstance(intent, Mapping) or set(intent) != core_fields:
            raise ValueError("nested completion-request intent fields do not match")
        identity = json.loads(_canonical_json(dict(intent)))
        value = {
            "schema_version": NESTED_REQUEST_INTENT_VERSION,
            "intent_uid": _sha256_json(
                {"schema_version": NESTED_REQUEST_INTENT_VERSION, **identity}
            ),
            **identity,
        }
        value["intent_sha256"] = _sha256_json(value)
        normalized = self._validate_request_intent(value)
        actor_run_uids = {
            str(start["actor_run_uid"]) for start in self._actor_starts
        }
        if normalized["actor_run_uid"] not in actor_run_uids:
            raise ValueError("request intent references an unattested actor run")
        uid = str(normalized["intent_uid"])
        if uid in self._intents:
            raise ValueError("completion-request intent was persisted twice")
        path = self.intents_root / f"{uid}.json"
        _lstat_no_symlink(path, "nested completion-request intent")
        _atomic_write_bytes(
            path, _json_bytes(normalized, pretty=True), refuse_existing=True
        )
        self._intents[uid] = normalized
        return deepcopy(normalized)

    def _validate_request_intent_closure(self) -> None:
        if not self._actor_chain_required:
            return
        referenced: list[str] = []
        for continuation_uid, envelope in self._entries.items():
            record = envelope.get("record")
            expected = sorted(
                (
                    intent
                    for intent in self._intents.values()
                    if intent["continuation_uid"] == continuation_uid
                ),
                key=lambda item: int(item["request_index"]),
            )
            _validate_record_request_intent_contract(record, intents=expected)
            referenced.extend(str(item["intent_uid"]) for item in expected)
        if len(referenced) != len(set(referenced)):
            raise ValueError("nested request intent was closed by multiple records")
        unclosed = set(self._intents) - set(referenced)
        if unclosed:
            raise ValueError(
                "nested journal has an unclosed completion-request intent; "
                "resume must fail closed"
            )

    @staticmethod
    def _envelope(
        record: Mapping[str, object], boundary: Mapping[str, object] | None
    ) -> dict[str, object]:
        continuation_uid = record.get("continuation_uid")
        if not _is_sha256(continuation_uid):
            raise ValueError("journal record continuation_uid is invalid")
        normalized_boundary = (
            None
            if boundary is None
            else json.loads(_canonical_json(dict(boundary)))
        )
        envelope: dict[str, object] = {
            "schema_version": NESTED_JOURNAL_ENTRY_VERSION,
            "continuation_uid": str(continuation_uid),
            "record": json.loads(_canonical_json(dict(record))),
            "post_action_boundary": normalized_boundary,
        }
        envelope["entry_sha256"] = _sha256_json(envelope)
        return envelope

    @staticmethod
    def _validate_envelope(value: object, *, filename_uid: str) -> dict[str, object]:
        if not isinstance(value, Mapping):
            raise TypeError("nested journal entry must be an object")
        if set(value) != {
            "schema_version",
            "continuation_uid",
            "record",
            "post_action_boundary",
            "entry_sha256",
        }:
            raise ValueError("nested journal entry fields do not match the contract")
        if value.get("schema_version") != NESTED_JOURNAL_ENTRY_VERSION:
            raise ValueError("nested journal entry version mismatch")
        continuation_uid = value.get("continuation_uid")
        if continuation_uid != filename_uid or not _is_sha256(continuation_uid):
            raise ValueError("nested journal filename/UID mismatch")
        record = value.get("record")
        boundary = value.get("post_action_boundary")
        if not isinstance(record, Mapping) or (
            boundary is not None and not isinstance(boundary, Mapping)
        ):
            raise TypeError("nested journal entry payload is invalid")
        if record.get("continuation_uid") != continuation_uid:
            raise ValueError("nested journal record UID mismatch")
        payload = dict(value)
        claimed = payload.pop("entry_sha256")
        if not _is_sha256(claimed) or claimed != _sha256_json(payload):
            raise ValueError("nested journal entry content SHA256 mismatch")
        return json.loads(_canonical_json(dict(value)))

    def _load_entries(self) -> dict[str, dict[str, object]]:
        entries: dict[str, dict[str, object]] = {}
        unexpected = [
            path
            for path in self.entries_root.iterdir()
            if not (
                (_lstat_no_symlink(path, "nested journal entry path") is not None)
                and stat.S_ISREG(path.lstat().st_mode)
                and (
                    path.suffix == ".json"
                    or (path.name.startswith(".") and path.suffix == ".tmp")
                )
            )
        ]
        if unexpected:
            raise ValueError("nested journal contains an unexpected entry path")
        for path in sorted(self.entries_root.glob("*.json")):
            path_stat = _lstat_no_symlink(path, "nested journal entry")
            if path_stat is None or not stat.S_ISREG(path_stat.st_mode):
                raise ValueError("nested journal entry must be a regular file")
            uid = path.stem
            if uid not in self.expected_uid_set:
                raise ValueError("nested journal contains an unregistered continuation UID")
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("nested journal entry is unreadable or truncated") from exc
            entries[uid] = self._validate_envelope(raw, filename_uid=uid)
        return entries

    def entries(self) -> dict[str, dict[str, object]]:
        return deepcopy(self._entries)

    def append(
        self,
        record: Mapping[str, object],
        boundary: Mapping[str, object] | None,
    ) -> None:
        if self.is_complete:
            raise ValueError("cannot append to a completed nested journal")
        envelope = self._envelope(record, boundary)
        uid = str(envelope["continuation_uid"])
        if uid not in self.expected_uid_set:
            raise ValueError("nested journal record UID was not pre-registered")
        if self._actor_chain_required:
            record = envelope["record"]
            expected_intents = sorted(
                (
                    intent
                    for intent in self._intents.values()
                    if intent["continuation_uid"] == uid
                ),
                key=lambda item: int(item["request_index"]),
            )
            _validate_record_request_intent_contract(
                record, intents=expected_intents
            )
        content = _json_bytes(envelope)
        path = self.entries_root / f"{uid}.json"
        _lstat_no_symlink(path, "nested journal entry")
        try:
            _atomic_write_bytes(path, content, refuse_existing=True)
        except FileExistsError:
            existing = self._validate_envelope(
                json.loads(path.read_text(encoding="utf-8")), filename_uid=uid
            )
            if existing != envelope:
                raise ValueError("idempotent nested journal write changed record content")
        self._entries[uid] = envelope
        self._validate_request_intent_closure()

    def _complete_payload(self) -> dict[str, object]:
        self._validate_request_intent_closure()
        ordered_hashes = [
            self._entries[uid]["entry_sha256"] for uid in self.expected_uids
        ]
        uid_order = {uid: index for index, uid in enumerate(self.expected_uids)}
        ordered_intents = sorted(
            self._intents.values(),
            key=lambda item: (
                uid_order[str(item["continuation_uid"])],
                int(item["request_index"]),
            ),
        )
        actor_chain = self.actor_run_chain_report(require_complete=True)
        return {
            "schema_version": NESTED_JOURNAL_COMPLETE_VERSION,
            "status": "complete",
            "header_sha256": self.header_sha256,
            "expected_record_count": len(self.expected_uids),
            "record_count": len(self._entries),
            "ordered_entry_sha256s_sha256": _sha256_json(ordered_hashes),
            "request_intent_count": len(ordered_intents),
            "ordered_request_intent_sha256s_sha256": _sha256_json(
                [item["intent_sha256"] for item in ordered_intents]
            ),
            "actor_run_count": actor_chain["run_count"],
            "actor_run_chain_sha256": actor_chain[
                "actor_run_chain_sha256"
            ],
        }

    def _verify_complete_marker(self) -> dict[str, object]:
        if set(self._entries) != self.expected_uid_set:
            raise ValueError("completed nested journal is missing registered records")
        marker_stat = _lstat_no_symlink(
            self.complete_path, "nested journal completion marker"
        )
        if marker_stat is None or not stat.S_ISREG(marker_stat.st_mode):
            raise ValueError("nested journal completion marker is invalid")
        try:
            marker = json.loads(self.complete_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("nested journal completion marker is invalid") from exc
        expected = self._complete_payload()
        if marker != expected:
            raise ValueError("nested journal completion marker does not match records")
        return expected

    def finalize(self) -> dict[str, object]:
        if set(self._entries) != self.expected_uid_set:
            missing = len(self.expected_uid_set - set(self._entries))
            raise ValueError(f"nested journal is incomplete ({missing} records missing)")
        payload = self._complete_payload()
        content = _json_bytes(payload, pretty=True)
        if self.is_complete:
            self._verify_complete_marker()
        else:
            _atomic_write_bytes(self.complete_path, content, refuse_existing=False)
        return deepcopy(payload)

    def completion_report(self) -> dict[str, object]:
        marker_stat = _lstat_no_symlink(
            self.complete_path, "nested journal completion marker"
        )
        if marker_stat is None or not stat.S_ISREG(marker_stat.st_mode):
            raise ValueError("completed nested journal marker is required")
        return deepcopy(self._verify_complete_marker())


def _completed_journal_artifact(
    journal: NestedContinuationJournal,
    *,
    commit_directory: Path,
    expected_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    for path, name in (
        (journal.root, "nested journal root"),
        (journal.entries_root, "nested journal entries directory"),
        (journal.intents_root, "nested journal request-intents directory"),
        (journal.actor_runs_root, "nested journal actor-runs directory"),
    ):
        path_stat = _lstat_no_symlink(path, name)
        if path_stat is None or not stat.S_ISDIR(path_stat.st_mode):
            raise ValueError(f"{name} is invalid")
    try:
        relative_root = journal.root.relative_to(commit_directory)
    except ValueError as exc:
        raise ValueError(
            "nested journal must be inside the manifest commit directory"
        ) from exc
    if not relative_root.parts or any(part == ".." for part in relative_root.parts):
        raise ValueError("nested journal relative directory is invalid")
    report = journal.completion_report()
    entries = journal.entries()
    ordered_records = list(expected_records)
    if len(ordered_records) != len(journal.expected_uids):
        raise ValueError("nested journal record count differs from collection")
    descriptors = []
    for uid, expected_record in zip(
        journal.expected_uids, ordered_records, strict=True
    ):
        envelope = entries.get(uid)
        if not isinstance(envelope, Mapping) or envelope.get("record") != dict(
            expected_record
        ):
            raise ValueError("nested journal record differs from collection artifact")
        path = journal.entries_root / f"{uid}.json"
        path_stat = _lstat_no_symlink(path, "nested journal entry")
        if path_stat is None or not stat.S_ISREG(path_stat.st_mode):
            raise ValueError("nested journal entry is not a regular file")
        raw = path.read_bytes()
        descriptors.append(
            {
                "continuation_uid": uid,
                "relative_path": f"entries/{uid}.json",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "byte_count": len(raw),
                "entry_sha256": envelope["entry_sha256"],
            }
        )
    for path, name in (
        (journal.header_path, "nested journal header"),
        (journal.complete_path, "nested journal completion marker"),
    ):
        path_stat = _lstat_no_symlink(path, name)
        if path_stat is None or not stat.S_ISREG(path_stat.st_mode):
            raise ValueError(f"{name} is invalid")
    header_raw = journal.header_path.read_bytes()
    complete_raw = journal.complete_path.read_bytes()
    uid_order = {uid: index for index, uid in enumerate(journal.expected_uids)}
    request_intent_descriptors = []
    for intent in sorted(
        journal._intents.values(),
        key=lambda item: (
            uid_order[str(item["continuation_uid"])],
            int(item["request_index"]),
        ),
    ):
        intent_uid = str(intent["intent_uid"])
        path = journal.intents_root / f"{intent_uid}.json"
        path_stat = _lstat_no_symlink(
            path, "nested completion-request intent"
        )
        if path_stat is None or not stat.S_ISREG(path_stat.st_mode):
            raise ValueError("nested completion-request intent is missing")
        raw = path.read_bytes()
        request_intent_descriptors.append(
            {
                "intent_uid": intent_uid,
                "continuation_uid": intent["continuation_uid"],
                "request_index": intent["request_index"],
                "relative_path": f"request_intents/{path.name}",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "byte_count": len(raw),
                "intent_sha256": intent["intent_sha256"],
            }
        )
    actor_chain_report = journal.actor_run_chain_report(require_complete=True)
    actor_files = []
    for start in journal._actor_starts:
        sequence = int(start["sequence_index"])
        uid = str(start["actor_run_uid"])
        for kind in ("start", "end"):
            path = journal._actor_path(sequence, uid, kind)
            path_stat = _lstat_no_symlink(path, f"nested actor-run {kind}")
            if path_stat is None:
                if kind == "end":
                    continue
                raise ValueError("nested actor-run start artifact is missing")
            if not stat.S_ISREG(path_stat.st_mode):
                raise ValueError("nested actor-run artifact is not a regular file")
            raw = path.read_bytes()
            actor_files.append(
                {
                    "sequence_index": sequence,
                    "actor_run_uid": uid,
                    "kind": kind,
                    "relative_path": f"actor_runs/{path.name}",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "byte_count": len(raw),
                }
            )
    actor_chain_artifact = {
        "report": actor_chain_report,
        "files": actor_files,
    }
    actor_chain_artifact["artifact_sha256"] = _sha256_json(
        actor_chain_artifact
    )
    artifact = {
        "schema_version": NESTED_JOURNAL_ARTIFACT_VERSION,
        "relative_directory": relative_root.as_posix(),
        "header": {
            "relative_path": "header.json",
            "sha256": hashlib.sha256(header_raw).hexdigest(),
            "byte_count": len(header_raw),
        },
        "complete": {
            "relative_path": "complete.json",
            "sha256": hashlib.sha256(complete_raw).hexdigest(),
            "byte_count": len(complete_raw),
        },
        "record_count": len(descriptors),
        "entries": descriptors,
        "request_intents": request_intent_descriptors,
        "actor_run_chain": actor_chain_artifact,
        "completion_report_sha256": _sha256_json(report),
    }
    artifact["artifact_sha256"] = _sha256_json(artifact)
    return artifact


def _open_completed_journal(
    root: str | Path,
    contract: Mapping[str, object],
) -> NestedContinuationJournal:
    requested_root = Path(root).expanduser().absolute()
    root_stat = _lstat_no_symlink(requested_root, "nested journal root")
    if root_stat is None or not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("completed nested journal root is invalid")
    journal_root = requested_root.resolve()
    header_stat = _lstat_no_symlink(
        journal_root / "header.json", "nested journal header"
    )
    complete_stat = _lstat_no_symlink(
        journal_root / "complete.json", "nested journal completion marker"
    )
    if (
        header_stat is None
        or not stat.S_ISREG(header_stat.st_mode)
        or complete_stat is None
        or not stat.S_ISREG(complete_stat.st_mode)
    ):
        raise ValueError("completed nested journal files are required")
    return NestedContinuationJournal(journal_root, contract)


def _validate_collector_artifact_contract(
    decisions: Sequence[object],
    continuations: Sequence[object],
    summary: Mapping[str, object],
) -> None:
    """Reject storage-shaped objects that were not emitted by the collector."""
    aggregate = summary.get("aggregate")
    safety = summary.get("safety")
    if (
        summary.get("schema_version") != _NESTED_COLLECTION_VERSION
        or not isinstance(aggregate, Mapping)
        or not isinstance(safety, Mapping)
        or not decisions
    ):
        raise ValueError("nested collection schema is not scale eligible")
    per_decision = aggregate.get("continuations_per_decision")
    if (
        not isinstance(per_decision, int)
        or isinstance(per_decision, bool)
        or per_decision not in {4, 8}
    ):
        raise ValueError("nested continuation cardinality is invalid")

    scheduled: list[str] = []
    decision_by_uid: dict[str, Mapping[str, object]] = {}
    proposal_count = 0
    for raw_decision in decisions:
        if not isinstance(raw_decision, Mapping):
            raise TypeError("nested decision must be an object")
        decision_uid = raw_decision.get("decision_uid")
        state_uid = raw_decision.get("state_uid")
        source_proposals = raw_decision.get("source_proposals")
        proposal_uids = raw_decision.get("proposal_uids")
        decision_uids = raw_decision.get("continuation_uids")
        if (
            raw_decision.get("schema_version") != _NESTED_DECISION_VERSION
            or not _is_sha256(decision_uid)
            or not _is_sha256(state_uid)
            or decision_uid in decision_by_uid
            or not isinstance(source_proposals, list)
            or not source_proposals
            or not isinstance(proposal_uids, list)
            or len(proposal_uids) != len(source_proposals)
            or raw_decision.get("proposal_multiplicity") != len(source_proposals)
            or raw_decision.get("continuations_expected") != per_decision
            or not isinstance(decision_uids, list)
            or len(decision_uids) != per_decision
            or any(not _is_sha256(uid) for uid in decision_uids)
            or not isinstance(raw_decision.get("structurally_valid"), bool)
            or not isinstance(raw_decision.get("credit_eligible"), bool)
            or raw_decision.get("optimizer_enabled") is not False
            or raw_decision.get("training_ready") is not False
            or raw_decision.get("uses_hidden_goal") is not False
        ):
            raise ValueError("nested decision schema is not scale eligible")
        decision_by_uid[str(decision_uid)] = raw_decision
        scheduled.extend(str(uid) for uid in decision_uids)
        proposal_count += len(source_proposals)

    actual_uids: list[str] = []
    sampling_invalid = 0
    for raw_continuation in continuations:
        if not isinstance(raw_continuation, Mapping):
            raise TypeError("nested continuation must be an object")
        decision_uid = raw_continuation.get("decision_uid")
        state_uid = raw_continuation.get("state_uid")
        index = raw_continuation.get("continuation_index")
        if (
            raw_continuation.get("schema_version")
            != _NESTED_CONTINUATION_VERSION
            or not _is_sha256(decision_uid)
            or not _is_sha256(state_uid)
            or decision_uid not in decision_by_uid
            or decision_by_uid[str(decision_uid)].get("state_uid") != state_uid
            or not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= per_decision
            or raw_continuation.get("seed_schedule_version")
            != _NESTED_SEED_SCHEDULE_VERSION
            or raw_continuation.get("fold_contract_version")
            != _NESTED_FOLD_CONTRACT_VERSION
            or not isinstance(
                raw_continuation.get("completion_intent_uids"), list
            )
            or any(
                not _is_sha256(uid)
                for uid in raw_continuation["completion_intent_uids"]
            )
            or len(raw_continuation["completion_intent_uids"])
            != len(set(raw_continuation["completion_intent_uids"]))
            or not all(
                isinstance(raw_continuation.get(field), bool)
                for field in (
                    "valid_for_learning",
                    "infrastructure_invalid",
                    "reward_invalid",
                    "reward_unverifiable",
                    "sampling_invalid",
                    "strict",
                    "model_failure",
                    "fresh_environment_lease",
                    "release_verified",
                )
            )
            or raw_continuation.get("optimizer_enabled") is not False
            or raw_continuation.get("uses_hidden_goal") is not False
        ):
            raise ValueError("nested continuation schema is not scale eligible")
        seed_uid = _sha256_json(
            {
                "version": _NESTED_SEED_SCHEDULE_VERSION,
                "state_uid": state_uid,
                "continuation_index": index,
            }
        )
        expected_uid = _sha256_json(
            {
                "state_uid": state_uid,
                "decision_uid": decision_uid,
                "continuation_index": index,
                "seed_schedule_version": _NESTED_SEED_SCHEDULE_VERSION,
                "continuation_seed_uid": seed_uid,
            }
        )
        content_record = dict(raw_continuation)
        claimed_content_sha = content_record.pop("rollout_content_sha256", None)
        expected_content_sha = _sha256_json(
            {"version": _NESTED_ROLLOUT_CONTENT_VERSION, "record": content_record}
        )
        if (
            raw_continuation.get("continuation_seed_uid") != seed_uid
            or raw_continuation.get("continuation_uid") != expected_uid
            or claimed_content_sha != expected_content_sha
        ):
            raise ValueError("nested continuation deterministic identity is invalid")
        _validate_record_request_intent_contract(raw_continuation)
        actual_uids.append(expected_uid)
        sampling_invalid += int(raw_continuation["sampling_invalid"])

    cardinality_complete = (
        len(decisions) == len(decision_by_uid)
        and len(continuations) == len(decisions) * per_decision
        and scheduled == actual_uids
        and len(actual_uids) == len(set(actual_uids))
    )
    sampling_invalid_rate = (
        sampling_invalid / len(continuations) if continuations else 0.0
    )
    expected_aggregate = {
        "proposals": proposal_count,
        "decisions": len(decisions),
        "distinct_decisions": len(decision_by_uid),
        "continuations": len(continuations),
        "sampling_invalid_continuations": sampling_invalid,
        "sampling_invalid_rate": sampling_invalid_rate,
        "cardinality_complete": cardinality_complete,
        "mechanical_collection_passed": (
            bool(decisions) and cardinality_complete and sampling_invalid_rate <= 0.05
        ),
    }
    if any(aggregate.get(key) != value for key, value in expected_aggregate.items()):
        raise ValueError("nested collection aggregate is not collector-consistent")
    if aggregate.get("mechanical_collection_passed") is not True:
        raise ValueError("nested collection did not pass its mechanical gate")
    required_safety = {
        "active_branch": True,
        "exact_first_decision_replay": True,
        "fresh_lease_per_continuation": True,
        "outcome_conditioned_resampling": False,
        "common_random_numbers_by_state_slot": True,
        "state_exclusion_no_backfill": True,
        "optimizer_enabled": False,
        "training_ready": False,
        "uses_hidden_goal": False,
    }
    if any(safety.get(key) != value for key, value in required_safety.items()):
        raise ValueError("nested collection safety contract is not scale eligible")


def finalize_scale_ready_collection(
    collection: Mapping[str, object],
    *,
    journal_path: str | Path,
    journal_report: Mapping[str, object],
    actor_attestation: Mapping[str, object],
) -> dict[str, object]:
    """Unlock scale collection only after journal and actor contracts complete."""
    if not isinstance(collection, Mapping):
        raise TypeError("nested collection must be an object")
    normalized = deepcopy(dict(collection))
    if set(normalized) != {"decisions", "continuations", "summary"}:
        raise ValueError("nested collection fields do not match the contract")
    decisions = normalized.get("decisions")
    summary = normalized.get("summary")
    continuations = normalized.get("continuations")
    if (
        not isinstance(decisions, list)
        or not isinstance(summary, dict)
        or not isinstance(continuations, list)
    ):
        raise TypeError("nested collection summary/continuations are invalid")
    _validate_collector_artifact_contract(decisions, continuations, summary)
    provenance = summary.get("provenance")
    safety = summary.get("safety")
    aggregate = summary.get("aggregate")
    if not all(isinstance(item, dict) for item in (provenance, safety, aggregate)):
        raise TypeError("nested collection summary contracts are invalid")
    stage1_binding = validate_nested_stage1_source_binding(
        provenance.get("stage1_source_binding"),
        expected_records_sha256=str(summary.get("stage1_source_sha256") or ""),
    )
    if (
        safety.get("finalized_stage1_source") is not True
        or safety.get("actor_prehashed_backend_binding") is not True
        or safety.get("actor_stat_checked_before_every_completion") is not True
        or stage1_binding["active_plan_sha256"] != summary.get("plan_sha256")
        or stage1_binding["policy_reward_sha256"]
        != provenance.get("policy_reward_sha256")
        or stage1_binding["harness_contract_sha256"]
        != summary.get("harness_contract_sha256")
        or stage1_binding["environment_manifest_sha256s"]
        != provenance.get("environment_manifest_sha256s")
        or stage1_binding["required_environment_version"]
        != provenance.get("required_environment_version")
    ):
        raise ValueError("nested collection lacks a finalized Stage-1 source contract")
    actor = validate_actor_run_attestation(
        actor_attestation,
        expected_sha256=str(provenance.get("actor_checkpoint_sha256") or ""),
    )
    completion_calls = 0
    checked_completion_calls = 0
    for continuation in continuations:
        if not isinstance(continuation, Mapping):
            raise TypeError("nested continuation must be an object")
        record_calls = continuation.get("completion_calls")
        record_checked = continuation.get(
            "actor_stat_checked_completion_calls"
        )
        if (
            not isinstance(record_calls, int)
            or isinstance(record_calls, bool)
            or record_calls < 0
            or record_checked != record_calls
            or continuation.get("actor_stat_snapshot_sha256")
            != actor["stat_snapshot_sha256"]
            or not _is_sha256(continuation.get("actor_run_uid"))
        ):
            raise ValueError(
                "nested continuation lacks per-request actor stat evidence"
            )
        completion_calls += record_calls
        checked_completion_calls += record_checked
    if (
        safety.get("completion_calls") != completion_calls
        or safety.get("actor_stat_checked_completion_calls")
        != checked_completion_calls
        or actor["runtime_stat_checks"]
        < actor["completion_request_stat_checks"] + 2
    ):
        raise ValueError(
            "actor runtime stat attestation did not cover completion requests"
        )
    expected_journal = {
        "schema_version",
        "status",
        "header_sha256",
        "expected_record_count",
        "record_count",
        "ordered_entry_sha256s_sha256",
        "request_intent_count",
        "ordered_request_intent_sha256s_sha256",
        "actor_run_count",
        "actor_run_chain_sha256",
    }
    if not isinstance(journal_report, Mapping) or set(journal_report) != expected_journal:
        raise ValueError("nested journal completion report fields do not match")
    if journal_report.get("schema_version") != NESTED_JOURNAL_COMPLETE_VERSION:
        raise ValueError("nested journal completion report version mismatch")
    continuation_count = len(continuations)
    continuations_per_decision = aggregate.get("continuations_per_decision")
    if (
        not isinstance(continuations_per_decision, int)
        or isinstance(continuations_per_decision, bool)
        or continuations_per_decision < 1
    ):
        raise ValueError("nested continuation cardinality is invalid")
    expected_uids: list[str] = []
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise TypeError("nested decision must be an object")
        decision_uids = decision.get("continuation_uids")
        if (
            not isinstance(decision_uids, list)
            or len(decision_uids) != continuations_per_decision
            or any(not _is_sha256(uid) for uid in decision_uids)
        ):
            raise ValueError("nested decision continuation schedule is invalid")
        expected_uids.extend(str(uid) for uid in decision_uids)
    actual_uids = []
    expected_request_intent_count = 0
    for continuation in continuations:
        if not isinstance(continuation, Mapping) or not _is_sha256(
            continuation.get("continuation_uid")
        ):
            raise ValueError("nested continuation UID is invalid")
        actual_uids.append(str(continuation["continuation_uid"]))
        expected_request_intent_count += len(
            _validate_record_request_intent_contract(continuation)
        )
    if expected_uids != actual_uids or len(actual_uids) != len(set(actual_uids)):
        raise ValueError("nested continuation artifact differs from its UID schedule")
    if (
        journal_report.get("status") != "complete"
        or journal_report.get("expected_record_count") != continuation_count
        or journal_report.get("record_count") != continuation_count
        or aggregate.get("continuations") != continuation_count
        or not _is_sha256(journal_report.get("header_sha256"))
        or not _is_sha256(journal_report.get("ordered_entry_sha256s_sha256"))
        or not _is_sha256(
            journal_report.get("ordered_request_intent_sha256s_sha256")
        )
        or not isinstance(journal_report.get("request_intent_count"), int)
        or isinstance(journal_report.get("request_intent_count"), bool)
        or journal_report["request_intent_count"]
        != expected_request_intent_count
        or not _is_sha256(journal_report.get("actor_run_chain_sha256"))
        or not isinstance(journal_report.get("actor_run_count"), int)
        or isinstance(journal_report.get("actor_run_count"), bool)
        or journal_report["actor_run_count"] < 1
    ):
        raise ValueError("nested journal did not complete the expected collection")
    formal_plan = summary.get("formal_plan")
    if (
        not isinstance(formal_plan, Mapping)
        or formal_plan.get("schema_version") != "shopping-nested-formal-plan-v2"
        or summary.get("formal_plan_sha256") != _sha256_json(dict(formal_plan))
    ):
        raise ValueError("nested formal plan v2 identity is invalid")
    actor_runtime_binding = {
        "schema_version": ACTOR_RUN_ATTESTATION_VERSION,
        "actor_checkpoint_sha256": actor["actor_checkpoint_sha256"],
        "stat_snapshot_sha256": actor["stat_snapshot_sha256"],
        "stat_entry_count": actor["stat_entry_count"],
        "read_only_run_binding": True,
    }
    expected_journal_contract = build_nested_journal_contract(
        experiment_uid=str(summary.get("experiment_uid") or ""),
        active_plan_sha256=str(summary.get("plan_sha256") or ""),
        formal_plan_sha256=str(summary.get("formal_plan_sha256") or ""),
        resolved_selections_sha256=str(
            summary.get("resolved_selections_sha256") or ""
        ),
        stage1_source_sha256=str(summary.get("stage1_source_sha256") or ""),
        stage1_manifest_sha256=str(stage1_binding["final_manifest_sha256"]),
        stage1_source_finalized=True,
        harness_contract_sha256=str(
            summary.get("harness_contract_sha256") or ""
        ),
        actor_runtime_binding=actor_runtime_binding,
        continuations_per_decision=continuations_per_decision,
        expected_continuation_uids=expected_uids,
    )
    if journal_report.get("header_sha256") != nested_journal_header_sha256(
        expected_journal_contract
    ):
        raise ValueError("nested journal header differs from the formal collection")
    completed_journal = _open_completed_journal(
        journal_path, expected_journal_contract
    )
    if completed_journal.completion_report() != dict(journal_report):
        raise ValueError("nested journal report differs from its completed files")
    actor_chain = completed_journal.actor_run_chain_report(require_complete=True)
    if (
        actor_chain["actor_run_chain_sha256"]
        != journal_report["actor_run_chain_sha256"]
        or actor_chain["run_count"] != journal_report["actor_run_count"]
        or actor_chain["request_intent_count"]
        != journal_report["request_intent_count"]
        or completed_journal.latest_actor_attestation() != actor
    ):
        raise ValueError("nested actor-run attestation chain is incomplete")
    _completed_journal_artifact(
        completed_journal,
        commit_directory=completed_journal.root.parent,
        expected_records=continuations,
    )
    safety.update(
        {
            "streaming_journal_and_resume": True,
            "manifest_commit_required": True,
            "actor_start_end_full_hash": True,
            "actor_run_chain_closed": True,
            "request_intent_write_ahead_log": True,
            "unclosed_request_intent_fail_closed": True,
            "actor_runtime_stat_binding": True,
            "actor_prehashed_backend_binding": True,
            "actor_stat_checked_before_every_completion": True,
            "finalized_stage1_source": True,
            "scale_collection_ready": True,
            "formal_scale_blockers": [],
        }
    )
    provenance["scale_storage_contract"] = {
        "schema_version": NESTED_SCALE_STORAGE_VERSION,
        "journal_complete_sha256": _sha256_json(dict(journal_report)),
        "actor_checkpoint_sha256": actor["actor_checkpoint_sha256"],
        "actor_stat_snapshot_sha256": actor["stat_snapshot_sha256"],
        "actor_run_chain_sha256": actor_chain["actor_run_chain_sha256"],
        "actor_run_count": actor_chain["run_count"],
        "request_intent_count": actor_chain["request_intent_count"],
    }
    return normalized


def _artifact_descriptor(path: Path, content: bytes, record_count: int) -> dict[str, object]:
    return {
        "filename": path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
        "record_count": record_count,
    }


def commit_nested_collection_artifacts(
    collection: Mapping[str, object],
    *,
    artifact_files: Mapping[str, str | Path],
    manifest_path: str | Path,
    journal_path: str | Path,
    journal_report: Mapping[str, object],
    actor_attestation: Mapping[str, object],
) -> dict[str, object]:
    """Persist the three artifacts and atomically publish their manifest last."""
    revalidated_collection = finalize_scale_ready_collection(
        collection,
        journal_path=journal_path,
        journal_report=journal_report,
        actor_attestation=actor_attestation,
    )
    if _canonical_json(revalidated_collection) != _canonical_json(dict(collection)):
        raise ValueError(
            "nested collection was not finalized through the scale-storage gate"
        )
    collection = revalidated_collection
    if set(artifact_files) != {"decisions", "continuations", "summary"}:
        raise ValueError("nested artifact file fields do not match the contract")
    paths = {
        name: _resolve_no_symlink_target(
            value, f"nested {name} artifact path"
        )
        for name, value in artifact_files.items()
    }
    manifest = _resolve_no_symlink_target(
        manifest_path, "nested manifest path"
    )
    if len(set(paths.values())) != 3 or manifest in paths.values():
        raise ValueError("nested artifact and manifest paths must be distinct")
    parents = {path.parent for path in [*paths.values(), manifest]}
    if len(parents) != 1:
        raise ValueError("nested artifacts and manifest must share one commit directory")
    if manifest.exists():
        raise FileExistsError("nested collection manifest already exists")
    decisions = collection.get("decisions")
    continuations = collection.get("continuations")
    summary = collection.get("summary")
    if not isinstance(decisions, list) or not isinstance(continuations, list):
        raise TypeError("nested decision and continuation artifacts must be lists")
    if not isinstance(summary, Mapping):
        raise TypeError("nested summary artifact must be an object")
    safety = summary.get("safety")
    if not isinstance(safety, Mapping) or safety.get("scale_collection_ready") is not True:
        raise ValueError("nested collection is not scale-storage ready")
    contents = {
        "decisions": _jsonl_bytes(decisions),
        "continuations": _jsonl_bytes(continuations),
        "summary": _json_bytes(dict(summary), pretty=True),
    }
    descriptors = {
        "decisions": _artifact_descriptor(paths["decisions"], contents["decisions"], len(decisions)),
        "continuations": _artifact_descriptor(
            paths["continuations"], contents["continuations"], len(continuations)
        ),
        "summary": _artifact_descriptor(paths["summary"], contents["summary"], 1),
    }
    actor = validate_actor_run_attestation(
        actor_attestation,
        expected_sha256=str((summary.get("provenance") or {}).get("actor_checkpoint_sha256") or ""),
    )
    provenance = summary.get("provenance")
    formal_plan = summary.get("formal_plan")
    if not isinstance(provenance, Mapping) or not isinstance(formal_plan, Mapping):
        raise TypeError("nested summary source provenance is invalid")
    source_contract = {
        "schema_version": NESTED_SOURCE_BINDING_VERSION,
        "active_branch_plan_schema_version": provenance.get(
            "active_branch_plan_schema_version"
        ),
        "active_branch_strategy_version": provenance.get(
            "active_branch_strategy_version"
        ),
        "active_branch_plan_sha256": summary.get("plan_sha256"),
        "nested_formal_plan_version": provenance.get(
            "nested_formal_plan_version"
        ),
        "nested_formal_plan_sha256": summary.get("formal_plan_sha256"),
        "required_environment_version": provenance.get(
            "required_environment_version"
        ),
        "environment_manifest_sha256s": provenance.get(
            "environment_manifest_sha256s"
        ),
        "policy_reward_sha256": provenance.get("policy_reward_sha256"),
        "harness_contract_sha256": summary.get("harness_contract_sha256"),
        "actor_checkpoint_sha256": provenance.get("actor_checkpoint_sha256"),
        "decoding_config_sha256": provenance.get("decoding_config_sha256"),
        "sampling_backend_contract_sha256": provenance.get(
            "sampling_backend_contract_sha256"
        ),
        "stage1_source_binding": deepcopy(
            provenance.get("stage1_source_binding")
        ),
    }
    digest_names = (
        "active_branch_plan_sha256",
        "nested_formal_plan_sha256",
        "policy_reward_sha256",
        "harness_contract_sha256",
        "actor_checkpoint_sha256",
        "decoding_config_sha256",
        "sampling_backend_contract_sha256",
    )
    if (
        source_contract["nested_formal_plan_version"]
        != "shopping-nested-formal-plan-v2"
        or formal_plan.get("schema_version")
        != source_contract["nested_formal_plan_version"]
        or any(not _is_sha256(source_contract[name]) for name in digest_names)
        or not isinstance(source_contract["required_environment_version"], str)
        or not source_contract["required_environment_version"]
        or not isinstance(source_contract["environment_manifest_sha256s"], list)
        or not source_contract["environment_manifest_sha256s"]
        or any(
            not _is_sha256(value)
            for value in source_contract["environment_manifest_sha256s"]
        )
    ):
        raise ValueError("nested source binding is incomplete or not formal plan v2")
    validate_nested_stage1_source_binding(
        source_contract["stage1_source_binding"],
        expected_records_sha256=str(summary.get("stage1_source_sha256") or ""),
    )
    completed_journal = _open_completed_journal(
        journal_path,
        build_nested_journal_contract(
            experiment_uid=str(summary["experiment_uid"]),
            active_plan_sha256=str(summary["plan_sha256"]),
            formal_plan_sha256=str(summary["formal_plan_sha256"]),
            resolved_selections_sha256=str(
                summary["resolved_selections_sha256"]
            ),
            stage1_source_sha256=str(summary["stage1_source_sha256"]),
            stage1_manifest_sha256=str(
                source_contract["stage1_source_binding"][
                    "final_manifest_sha256"
                ]
            ),
            stage1_source_finalized=True,
            harness_contract_sha256=str(summary["harness_contract_sha256"]),
            actor_runtime_binding={
                "schema_version": ACTOR_RUN_ATTESTATION_VERSION,
                "actor_checkpoint_sha256": actor["actor_checkpoint_sha256"],
                "stat_snapshot_sha256": actor["stat_snapshot_sha256"],
                "stat_entry_count": actor["stat_entry_count"],
                "read_only_run_binding": True,
            },
            continuations_per_decision=int(
                (summary.get("aggregate") or {})["continuations_per_decision"]
            ),
            expected_continuation_uids=[
                str(uid)
                for decision in decisions
                for uid in decision["continuation_uids"]
            ],
        ),
    )
    journal_artifact = _completed_journal_artifact(
        completed_journal,
        commit_directory=manifest.parent,
        expected_records=continuations,
    )
    actor_run_chain = deepcopy(
        journal_artifact["actor_run_chain"]["report"]
    )
    manifest_value = {
        "schema_version": NESTED_ARTIFACT_MANIFEST_VERSION,
        "status": "complete",
        "experiment_uid": summary.get("experiment_uid"),
        "artifacts": descriptors,
        "journal_completion": deepcopy(dict(journal_report)),
        "journal_artifact": journal_artifact,
        "actor_run_attestation": actor,
        "actor_run_chain": actor_run_chain,
        "source_contract": source_contract,
        "commit": {
            "schema_version": NESTED_ARTIFACT_COMMIT_VERSION,
            "manifest_is_only_commit_point": True,
            "artifacts_written_before_manifest": True,
        },
    }
    manifest_value["artifact_set_uid"] = _sha256_json(
        {
            "experiment_uid": manifest_value["experiment_uid"],
            "artifacts": descriptors,
            "journal_completion": manifest_value["journal_completion"],
            "journal_artifact": journal_artifact,
            "actor_run_attestation": actor,
            "actor_run_chain": actor_run_chain,
            "source_contract": source_contract,
        }
    )
    for name in ("decisions", "continuations", "summary"):
        _atomic_write_bytes(paths[name], contents[name], refuse_existing=False)
    _atomic_write_bytes(manifest, _json_bytes(manifest_value, pretty=True), refuse_existing=True)
    return manifest_value


def verify_completed_collection_manifest(
    manifest_path: str | Path,
    *,
    artifact_files: Mapping[str, str | Path],
    include_file_attestation: bool = False,
) -> dict[str, object]:
    """Reject incomplete, substituted, truncated, or post-commit artifacts."""
    if set(artifact_files) != {"decisions", "continuations", "summary"}:
        raise ValueError("nested artifact file fields do not match the contract")
    manifest_file = _resolve_no_symlink_target(
        manifest_path, "nested manifest path"
    )
    manifest_stat = _lstat_no_symlink(manifest_file, "nested manifest path")
    if manifest_stat is None or not stat.S_ISREG(manifest_stat.st_mode):
        raise ValueError("completed nested collection manifest is required")
    try:
        manifest_raw = manifest_file.read_bytes()
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("nested collection manifest is invalid") from exc
    if not isinstance(manifest, Mapping):
        raise TypeError("nested collection manifest must be an object")
    expected_fields = {
        "schema_version",
        "status",
        "experiment_uid",
        "artifacts",
        "journal_completion",
        "journal_artifact",
        "actor_run_attestation",
        "actor_run_chain",
        "source_contract",
        "commit",
        "artifact_set_uid",
    }
    if set(manifest) != expected_fields:
        raise ValueError("nested collection manifest fields do not match")
    if (
        manifest.get("schema_version") != NESTED_ARTIFACT_MANIFEST_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("commit")
        != {
            "schema_version": NESTED_ARTIFACT_COMMIT_VERSION,
            "manifest_is_only_commit_point": True,
            "artifacts_written_before_manifest": True,
        }
    ):
        raise ValueError("nested collection manifest is not a complete commit")
    raw_descriptors = manifest.get("artifacts")
    if not isinstance(raw_descriptors, Mapping) or set(raw_descriptors) != set(artifact_files):
        raise ValueError("nested collection manifest artifact set is invalid")
    paths = {
        name: _resolve_no_symlink_target(
            value, f"nested {name} artifact path"
        )
        for name, value in artifact_files.items()
    }
    if any(path.parent != manifest_file.parent for path in paths.values()):
        raise ValueError("nested artifacts must share the manifest commit directory")
    parsed_artifacts: dict[str, object] = {}
    for name, path in paths.items():
        descriptor = raw_descriptors.get(name)
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "filename",
            "sha256",
            "byte_count",
            "record_count",
        }:
            raise ValueError("nested artifact descriptor fields do not match")
        path_stat = _lstat_no_symlink(path, f"nested {name} artifact path")
        if (
            descriptor.get("filename") != path.name
            or path_stat is None
            or not stat.S_ISREG(path_stat.st_mode)
        ):
            raise ValueError("nested artifact path does not match its manifest")
        raw = path.read_bytes()
        if (
            descriptor.get("sha256") != hashlib.sha256(raw).hexdigest()
            or descriptor.get("byte_count") != len(raw)
        ):
            raise ValueError("nested artifact bytes differ from the completion manifest")
        if name == "summary":
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("nested summary artifact is invalid") from exc
            record_count = 1
            if not isinstance(parsed, Mapping):
                raise TypeError("nested summary artifact must be an object")
            parsed_artifacts[name] = dict(parsed)
            if parsed.get("experiment_uid") != manifest.get("experiment_uid"):
                raise ValueError("nested summary experiment differs from manifest")
            safety = parsed.get("safety")
            if (
                not isinstance(safety, Mapping)
                or safety.get("scale_collection_ready") is not True
                or safety.get("finalized_stage1_source") is not True
                or safety.get("actor_prehashed_backend_binding") is not True
                or safety.get("actor_stat_checked_before_every_completion")
                is not True
                or safety.get("actor_run_chain_closed") is not True
                or safety.get("request_intent_write_ahead_log") is not True
                or safety.get("unclosed_request_intent_fail_closed") is not True
            ):
                raise ValueError("nested summary is not scale-collection ready")
        else:
            try:
                rows = [
                    json.loads(line)
                    for line in raw.decode("utf-8").splitlines()
                    if line.strip()
                ]
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("nested JSONL artifact is invalid") from exc
            if any(not isinstance(row, Mapping) for row in rows):
                raise TypeError("nested JSONL artifact rows must be objects")
            parsed_artifacts[name] = [dict(row) for row in rows]
            record_count = len(rows)
        if descriptor.get("record_count") != record_count:
            raise ValueError("nested artifact record count differs from manifest")
    actor = manifest.get("actor_run_attestation")
    expected_actor = str((actor or {}).get("actor_checkpoint_sha256") or "")
    validate_actor_run_attestation(actor, expected_sha256=expected_actor)
    source_contract = manifest.get("source_contract")
    if not isinstance(source_contract, Mapping) or source_contract.get(
        "schema_version"
    ) != NESTED_SOURCE_BINDING_VERSION:
        raise ValueError("nested manifest source binding v4 is required")
    summary_value = json.loads(paths["summary"].read_text(encoding="utf-8"))
    provenance = summary_value.get("provenance") or {}
    expected_source_contract = {
        "schema_version": NESTED_SOURCE_BINDING_VERSION,
        "active_branch_plan_schema_version": provenance.get(
            "active_branch_plan_schema_version"
        ),
        "active_branch_strategy_version": provenance.get(
            "active_branch_strategy_version"
        ),
        "active_branch_plan_sha256": summary_value.get("plan_sha256"),
        "nested_formal_plan_version": provenance.get(
            "nested_formal_plan_version"
        ),
        "nested_formal_plan_sha256": summary_value.get("formal_plan_sha256"),
        "required_environment_version": provenance.get(
            "required_environment_version"
        ),
        "environment_manifest_sha256s": provenance.get(
            "environment_manifest_sha256s"
        ),
        "policy_reward_sha256": provenance.get("policy_reward_sha256"),
        "harness_contract_sha256": summary_value.get("harness_contract_sha256"),
        "actor_checkpoint_sha256": provenance.get("actor_checkpoint_sha256"),
        "decoding_config_sha256": provenance.get("decoding_config_sha256"),
        "sampling_backend_contract_sha256": provenance.get(
            "sampling_backend_contract_sha256"
        ),
        "stage1_source_binding": deepcopy(
            provenance.get("stage1_source_binding")
        ),
    }
    if dict(source_contract) != expected_source_contract or source_contract.get(
        "nested_formal_plan_version"
    ) != "shopping-nested-formal-plan-v2":
        raise ValueError("nested manifest source binding differs from summary")
    if actor.get("actor_checkpoint_sha256") != source_contract.get(
        "actor_checkpoint_sha256"
    ):
        raise ValueError("nested actor attestation differs from source binding")
    validate_nested_stage1_source_binding(
        source_contract.get("stage1_source_binding"),
        expected_records_sha256=str(summary_value.get("stage1_source_sha256") or ""),
    )
    journal = manifest.get("journal_completion")
    if (
        not isinstance(journal, Mapping)
        or set(journal)
        != {
            "schema_version",
            "status",
            "header_sha256",
            "expected_record_count",
            "record_count",
            "ordered_entry_sha256s_sha256",
            "request_intent_count",
            "ordered_request_intent_sha256s_sha256",
            "actor_run_count",
            "actor_run_chain_sha256",
        }
        or journal.get("schema_version") != NESTED_JOURNAL_COMPLETE_VERSION
        or journal.get("status") != "complete"
        or not _is_sha256(journal.get("header_sha256"))
        or not _is_sha256(journal.get("ordered_entry_sha256s_sha256"))
        or not _is_sha256(
            journal.get("ordered_request_intent_sha256s_sha256")
        )
        or not isinstance(journal.get("request_intent_count"), int)
        or isinstance(journal.get("request_intent_count"), bool)
        or journal["request_intent_count"] < 0
        or not _is_sha256(journal.get("actor_run_chain_sha256"))
        or not isinstance(journal.get("actor_run_count"), int)
        or isinstance(journal.get("actor_run_count"), bool)
        or journal["actor_run_count"] < 1
        or journal.get("record_count")
        != raw_descriptors["continuations"].get("record_count")
        or journal.get("expected_record_count") != journal.get("record_count")
    ):
        raise ValueError("nested manifest journal completion is invalid")
    journal_artifact = manifest.get("journal_artifact")
    if not isinstance(journal_artifact, Mapping) or set(journal_artifact) != {
        "schema_version",
        "relative_directory",
        "header",
        "complete",
        "record_count",
        "entries",
        "request_intents",
        "actor_run_chain",
        "completion_report_sha256",
        "artifact_sha256",
    }:
        raise ValueError("nested journal artifact descriptor is invalid")
    relative_directory = journal_artifact.get("relative_directory")
    if (
        journal_artifact.get("schema_version")
        != NESTED_JOURNAL_ARTIFACT_VERSION
        or not isinstance(relative_directory, str)
        or not relative_directory
    ):
        raise ValueError("nested journal artifact version or path is invalid")
    relative_path = Path(relative_directory)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("nested journal artifact path escapes the commit directory")
    journal_candidate = (manifest_file.parent / relative_path).absolute()
    journal_root_stat = _lstat_no_symlink(
        journal_candidate, "nested journal root"
    )
    if journal_root_stat is None or not stat.S_ISDIR(journal_root_stat.st_mode):
        raise ValueError("nested journal root artifact is invalid")
    journal_root = journal_candidate.resolve()
    try:
        journal_root.relative_to(manifest_file.parent)
    except ValueError as exc:
        raise ValueError(
            "nested journal artifact path escapes the commit directory"
        ) from exc
    try:
        header_path = journal_root / "header.json"
        header_stat = _lstat_no_symlink(header_path, "nested journal header")
        if header_stat is None or not stat.S_ISREG(header_stat.st_mode):
            raise ValueError("nested journal header artifact is invalid")
        header_value = json.loads(
            header_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("nested journal header artifact is invalid") from exc
    if not isinstance(header_value, Mapping) or not isinstance(
        header_value.get("contract"), Mapping
    ):
        raise TypeError("nested journal header contract is invalid")
    completed_journal = _open_completed_journal(
        journal_root, header_value["contract"]
    )
    actual_journal_artifact = _completed_journal_artifact(
        completed_journal,
        commit_directory=manifest_file.parent,
        expected_records=parsed_artifacts["continuations"],
    )
    if dict(journal_artifact) != actual_journal_artifact:
        raise ValueError("nested journal files differ from the completion manifest")
    actor_run_chain = manifest.get("actor_run_chain")
    actual_actor_run_chain = actual_journal_artifact["actor_run_chain"][
        "report"
    ]
    if (
        not isinstance(actor_run_chain, Mapping)
        or dict(actor_run_chain) != actual_actor_run_chain
        or actor_run_chain.get("actor_run_chain_sha256")
        != journal.get("actor_run_chain_sha256")
        or actor_run_chain.get("run_count") != journal.get("actor_run_count")
        or completed_journal.latest_actor_attestation() != dict(actor)
    ):
        raise ValueError("nested manifest actor-run chain is invalid")
    expected_uid = _sha256_json(
        {
            "experiment_uid": manifest.get("experiment_uid"),
            "artifacts": dict(raw_descriptors),
            "journal_completion": dict(journal),
            "journal_artifact": dict(manifest["journal_artifact"]),
            "actor_run_attestation": dict(actor),
            "actor_run_chain": dict(actor_run_chain),
            "source_contract": dict(source_contract),
        }
    )
    if manifest.get("artifact_set_uid") != expected_uid:
        raise ValueError("nested artifact-set UID mismatch")
    revalidated_collection = finalize_scale_ready_collection(
        {
            "decisions": parsed_artifacts["decisions"],
            "continuations": parsed_artifacts["continuations"],
            "summary": parsed_artifacts["summary"],
        },
        journal_path=(
            journal_root
        ),
        journal_report=journal,
        actor_attestation=actor,
    )
    if _canonical_json(revalidated_collection["summary"]) != _canonical_json(
        parsed_artifacts["summary"]
    ):
        raise ValueError("nested summary did not pass the scale-storage gate")
    verified_manifest = deepcopy(dict(manifest))
    if include_file_attestation:
        return {
            "manifest": verified_manifest,
            "manifest_file_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        }
    return verified_manifest
