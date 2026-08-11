from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shopping_grpo.training.grpo.active_suffix import (
    ActiveSuffixInfrastructureError,
    _sha256_json,
    sha256_actor_checkpoint,
)
from shopping_grpo.training.grpo.nested_artifacts import (
    ACTOR_RUN_ATTESTATION_VERSION,
    FIRST_DECISION_FINAL_MANIFEST_VERSION,
    NESTED_STAGE1_SOURCE_BINDING_VERSION,
    ActorCheckpointRunAttestation,
    NestedContinuationJournal,
    build_nested_journal_contract,
    commit_nested_collection_artifacts,
    finalize_scale_ready_collection,
    verify_completed_collection_manifest,
)


def _completed_journal(
    root: Path,
    collection: dict[str, object],
    actor_report: dict[str, object],
) -> tuple[NestedContinuationJournal, dict[str, object]]:
    summary = collection["summary"]
    assert isinstance(summary, dict)
    stage1_binding = summary["provenance"]["stage1_source_binding"]
    expected_uids = [
        uid
        for decision in collection["decisions"]
        for uid in decision["continuation_uids"]
    ]
    contract = build_nested_journal_contract(
        experiment_uid=summary["experiment_uid"],
        active_plan_sha256=summary["plan_sha256"],
        formal_plan_sha256=summary["formal_plan_sha256"],
        resolved_selections_sha256=summary["resolved_selections_sha256"],
        stage1_source_sha256=summary["stage1_source_sha256"],
        stage1_manifest_sha256=stage1_binding["final_manifest_sha256"],
        stage1_source_finalized=True,
        harness_contract_sha256=summary["harness_contract_sha256"],
        actor_runtime_binding={
            "schema_version": ACTOR_RUN_ATTESTATION_VERSION,
            "actor_checkpoint_sha256": actor_report[
                "actor_checkpoint_sha256"
            ],
            "stat_snapshot_sha256": actor_report["stat_snapshot_sha256"],
            "stat_entry_count": actor_report["stat_entry_count"],
            "read_only_run_binding": True,
        },
        continuations_per_decision=summary["aggregate"][
            "continuations_per_decision"
        ],
        expected_continuation_uids=expected_uids,
    )
    record_count = len(collection["continuations"])
    journal = NestedContinuationJournal(root, contract)
    journal.begin_actor_run(actor_report)
    boundaries = {
        decision["decision_uid"]: decision.get("post_action_boundary")
        for decision in collection["decisions"]
    }
    for record in collection["continuations"]:
        journal.append(record, boundaries.get(record.get("decision_uid")))
    journal.finish_actor_run(actor_report)
    report = journal.finalize()
    assert report["record_count"] == record_count
    return journal, report


def _raw_collection(
    actor_sha256: str,
    actor_stat_snapshot_sha256: str,
    actor_run_uid: str,
) -> dict[str, object]:
    state_uid = "0" * 64
    decision_uid = "2" * 64
    continuations = []
    for continuation_index in range(4):
        continuation_seed_uid = _sha256_json(
            {
                "version": "shopping-nested-state-crn-seed-v1",
                "state_uid": state_uid,
                "continuation_index": continuation_index,
            }
        )
        continuation_uid = _sha256_json(
            {
                "state_uid": state_uid,
                "decision_uid": decision_uid,
                "continuation_index": continuation_index,
                "seed_schedule_version": "shopping-nested-state-crn-seed-v1",
                "continuation_seed_uid": continuation_seed_uid,
            }
        )
        continuation = {
            "schema_version": "shopping-nested-continuation-v2",
            "continuation_uid": continuation_uid,
            "continuation_seed_uid": continuation_seed_uid,
            "seed_schedule_version": "shopping-nested-state-crn-seed-v1",
            "fold_contract_version": "shopping-nested-train4-gate4-v1",
            "fold": "train",
            "generation_mode": "deterministic_terminal",
            "state_uid": state_uid,
            "decision_uid": decision_uid,
            "continuation_index": continuation_index,
            "valid_for_learning": True,
            "infrastructure_invalid": False,
            "reward_invalid": False,
            "reward_unverifiable": False,
            "sampling_invalid": False,
            "strict": True,
            "model_failure": False,
            "fresh_environment_lease": True,
            "release_verified": True,
            "completion_calls": 0,
            "actor_stat_checked_completion_calls": 0,
            "completion_intent_uids": [],
            "downstream_request_seeds": [],
            "downstream_prompt_sha256": [],
            "actor_stat_snapshot_sha256": actor_stat_snapshot_sha256,
            "actor_run_uid": actor_run_uid,
            "optimizer_enabled": False,
            "uses_hidden_goal": False,
        }
        continuation["rollout_content_sha256"] = _sha256_json(
            {
                "version": "shopping-nested-rollout-content-v2",
                "record": continuation,
            }
        )
        continuations.append(continuation)
    stage1_source_sha256 = "b" * 64
    stage1_binding = {
        "schema_version": NESTED_STAGE1_SOURCE_BINDING_VERSION,
        "final_manifest_schema_version": FIRST_DECISION_FINAL_MANIFEST_VERSION,
        "final_manifest_sha256": "c" * 64,
        "records_file_sha256": stage1_source_sha256,
        "records_canonical_sha256": "d" * 64,
        "record_count": 1,
        "journal_contract_sha256": "e" * 64,
        "journal_contract_file_sha256": "f" * 64,
        "journal_content_sha256": "0" * 64,
        "active_plan_sha256": "4" * 64,
        "stage1_structure_sha256": "1" * 64,
        "source_provenance_sha256": "2" * 64,
        "source_git_sha": "c" * 40,
        "source_git_worktree_clean": True,
        "source_git_status_sha256": hashlib.sha256(b"").hexdigest(),
        "policy_reward_sha256": "8" * 64,
        "harness_contract_sha256": "6" * 64,
        "required_environment_version": "shopsimulator-environment-v2.1",
        "environment_manifest_sha256s": ["7" * 64],
    }
    formal_plan = {"schema_version": "shopping-nested-formal-plan-v2"}
    return {
        "decisions": [
            {
                "schema_version": "shopping-nested-decision-v2",
                "state_uid": state_uid,
                "decision_uid": decision_uid,
                "source_proposals": [{"proposal_uid": "1" * 64}],
                "proposal_uids": ["1" * 64],
                "proposal_multiplicity": 1,
                "continuations_expected": 4,
                "continuation_uids": [
                    continuation["continuation_uid"]
                    for continuation in continuations
                ],
                "structurally_valid": True,
                "credit_eligible": True,
                "optimizer_enabled": False,
                "training_ready": False,
                "uses_hidden_goal": False,
            }
        ],
        "continuations": continuations,
        "summary": {
            "schema_version": "shopping-nested-continuation-collection-v3",
            "experiment_uid": "3" * 64,
            "plan_sha256": "4" * 64,
            "stage1_source_sha256": stage1_source_sha256,
            "formal_plan": formal_plan,
            "formal_plan_sha256": _sha256_json(formal_plan),
            "resolved_selections_sha256": "5" * 64,
            "harness_contract_sha256": "6" * 64,
            "aggregate": {
                "proposals": 1,
                "decisions": 1,
                "distinct_decisions": 1,
                "continuations": 4,
                "continuations_per_decision": 4,
                "sampling_invalid_continuations": 0,
                "sampling_invalid_rate": 0.0,
                "cardinality_complete": True,
                "mechanical_collection_passed": True,
            },
            "provenance": {
                "active_branch_plan_schema_version": (
                    "shopping-active-branch-plan-v1"
                ),
                "active_branch_strategy_version": (
                    "pivotal-exact-prompt-k-suffix-v1"
                ),
                "nested_formal_plan_version": "shopping-nested-formal-plan-v2",
                "required_environment_version": (
                    "shopsimulator-environment-v2.1"
                ),
                "environment_manifest_sha256s": ["7" * 64],
                "policy_reward_sha256": "8" * 64,
                "actor_checkpoint_sha256": actor_sha256,
                "decoding_config_sha256": "9" * 64,
                "sampling_backend_contract_sha256": "a" * 64,
                "stage1_source_binding": stage1_binding,
            },
            "safety": {
                "active_branch": True,
                "exact_first_decision_replay": True,
                "fresh_lease_per_continuation": True,
                "outcome_conditioned_resampling": False,
                "common_random_numbers_by_state_slot": True,
                "state_exclusion_no_backfill": True,
                "finalized_stage1_source": True,
                "actor_prehashed_backend_binding": True,
                "actor_stat_checked_before_every_completion": True,
                "completion_calls": 0,
                "actor_stat_checked_completion_calls": 0,
                "scale_collection_ready": False,
                "formal_scale_blockers": ["not_finalized"],
                "optimizer_enabled": False,
                "training_ready": False,
                "uses_hidden_goal": False,
            },
        },
    }


class ActorCheckpointRunAttestationTests(unittest.TestCase):
    def test_runtime_stat_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "weights.bin"
            weights.write_bytes(b"weights-v1")
            actor_sha = sha256_actor_checkpoint(root)
            attestor = ActorCheckpointRunAttestation(root, actor_sha)
            attestor.verify_runtime(
                actor_checkpoint=root,
                actor_checkpoint_sha256=actor_sha,
            )

            weights.write_bytes(b"weights-v2")

            with self.assertRaisesRegex(
                ActiveSuffixInfrastructureError, "drifted"
            ):
                attestor.verify_runtime()

    @unittest.skipIf(os.name == "nt", "Windows st_ctime is file creation time")
    def test_runtime_ctime_detects_same_size_rewrite_with_restored_mtime(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "weights.bin"
            weights.write_bytes(b"weights-v1")
            actor_sha = sha256_actor_checkpoint(root)
            attestor = ActorCheckpointRunAttestation(root, actor_sha)
            baseline = weights.stat()

            weights.write_bytes(b"weights-v2")
            os.utime(
                weights,
                ns=(baseline.st_atime_ns, baseline.st_mtime_ns),
            )
            rewritten = weights.stat()

            self.assertEqual(rewritten.st_size, baseline.st_size)
            self.assertEqual(rewritten.st_ino, baseline.st_ino)
            self.assertEqual(rewritten.st_mtime_ns, baseline.st_mtime_ns)
            self.assertNotEqual(rewritten.st_ctime_ns, baseline.st_ctime_ns)
            with self.assertRaisesRegex(
                ActiveSuffixInfrastructureError, "drifted"
            ):
                attestor.verify_runtime()

    def test_start_and_end_full_hash_attestation_completes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "weights.bin").write_bytes(b"weights")
            actor_sha = sha256_actor_checkpoint(root)
            attestor = ActorCheckpointRunAttestation(root, actor_sha)
            attestor.verify_runtime()

            report = attestor.finish()

        self.assertTrue(report["completed"])
        self.assertEqual(report["start_full_sha256"], actor_sha)
        self.assertEqual(report["end_full_sha256"], actor_sha)
        self.assertGreaterEqual(report["runtime_stat_checks"], 2)

    def test_runtime_checks_do_not_rehash_the_actor_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "weights.bin").write_bytes(b"weights")
            actor_sha = sha256_actor_checkpoint(root)
            with patch(
                "shopping_grpo.training.grpo.nested_artifacts."
                "sha256_actor_checkpoint",
                wraps=sha256_actor_checkpoint,
            ) as full_hash:
                attestor = ActorCheckpointRunAttestation(root, actor_sha)
                for _ in range(5):
                    attestor.verify_runtime()
                attestor.finish()

        self.assertEqual(full_hash.call_count, 2)


class NestedContinuationJournalTests(unittest.TestCase):
    def test_journal_binds_request_intent_to_record_seed_and_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actor = root / "actor"
            actor.mkdir()
            (actor / "weights.bin").write_bytes(b"weights")
            actor_sha = sha256_actor_checkpoint(actor)
            attestor = ActorCheckpointRunAttestation(actor, actor_sha)
            binding = attestor.verify_runtime()
            continuation_uid = "d" * 64
            journal = NestedContinuationJournal(
                root / "journal",
                {
                    "expected_continuation_uids": [continuation_uid],
                    "actor_runtime_binding": binding,
                },
            )
            journal.begin_actor_run(attestor.report())
            intent = journal.begin_completion_request(
                {
                    "continuation_uid": continuation_uid,
                    "actor_run_uid": binding["actor_run_uid"],
                    "actor_stat_snapshot_sha256": binding[
                        "stat_snapshot_sha256"
                    ],
                    "request_index": 0,
                    "seed": 7,
                    "prompt_sha256": "a" * 64,
                }
            )
            record = {
                "continuation_uid": continuation_uid,
                "completion_calls": 2,
                "downstream_request_seeds": [8],
                "downstream_prompt_sha256": ["a" * 64],
                "completion_intent_uids": [intent["intent_uid"]],
                "actor_run_uid": binding["actor_run_uid"],
                "actor_stat_snapshot_sha256": binding[
                    "stat_snapshot_sha256"
                ],
            }
            with self.assertRaisesRegex(ValueError, "request-intent"):
                journal.append(record, None)

            record["downstream_request_seeds"] = [7]
            journal.append(record, None)

    def test_journal_rejects_symlinked_root_and_artifacts(self) -> None:
        def make_link(source: Path, target: Path, *, directory: bool = False):
            try:
                os.symlink(source, target, target_is_directory=directory)
            except OSError as exc:
                self.skipTest(f"symbolic links unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            contract = {"expected_continuation_uids": []}

            real_root = base / "root-real"
            NestedContinuationJournal(real_root, contract)
            linked_root = base / "root-link"
            make_link(real_root, linked_root, directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                NestedContinuationJournal(linked_root, contract)

            header_root = base / "header-journal"
            NestedContinuationJournal(header_root, contract)
            external_header = base / "external-header.json"
            (header_root / "header.json").replace(external_header)
            make_link(external_header, header_root / "header.json")
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                NestedContinuationJournal(header_root, contract)

            complete_root = base / "complete-journal"
            complete_journal = NestedContinuationJournal(complete_root, contract)
            complete_journal.finalize()
            external_complete = base / "external-complete.json"
            (complete_root / "complete.json").replace(external_complete)
            make_link(external_complete, complete_root / "complete.json")
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                NestedContinuationJournal(complete_root, contract)

            entries_root = base / "entries-journal"
            NestedContinuationJournal(entries_root, contract)
            external_entries = base / "external-entries"
            (entries_root / "entries").replace(external_entries)
            make_link(external_entries, entries_root / "entries", directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                NestedContinuationJournal(entries_root, contract)

            uid = "d" * 64
            entry_contract = {"expected_continuation_uids": [uid]}
            entry_root = base / "entry-journal"
            entry_journal = NestedContinuationJournal(entry_root, entry_contract)
            entry_journal.append({"continuation_uid": uid}, None)
            entry_path = entry_root / "entries" / f"{uid}.json"
            external_entry = base / "external-entry.json"
            entry_path.replace(external_entry)
            make_link(external_entry, entry_path)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                NestedContinuationJournal(entry_root, entry_contract)

    def test_partial_journal_resumes_and_duplicate_write_is_idempotent(self) -> None:
        uids = [str(index) * 64 for index in (1, 2, 3)]
        contract = {
            "experiment_uid": "f" * 64,
            "expected_continuation_uids": uids,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "journal"
            journal = NestedContinuationJournal(root, contract)
            first = {"continuation_uid": uids[0], "payload": "first"}
            journal.append(first, None)
            (root / "entries" / f".{uids[1]}.json.999.tmp").write_bytes(
                b'{"partial":'
            )

            resumed = NestedContinuationJournal(root, contract)
            self.assertEqual(set(resumed.entries()), {uids[0]})
            resumed.append(first, None)
            with self.assertRaisesRegex(ValueError, "changed record content"):
                resumed.append(
                    {"continuation_uid": uids[0], "payload": "conflict"}, None
                )
            resumed.append(
                {"continuation_uid": uids[1], "payload": "second"}, None
            )
            resumed.append(
                {"continuation_uid": uids[2], "payload": "third"}, None
            )
            report = resumed.finalize()

            verified = NestedContinuationJournal(root, contract)

        self.assertEqual(report["record_count"], 3)
        self.assertEqual(len(verified.entries()), 3)

    def test_tampered_journal_entry_is_rejected_on_resume(self) -> None:
        uid = "d" * 64
        contract = {
            "experiment_uid": "e" * 64,
            "expected_continuation_uids": [uid],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "journal"
            journal = NestedContinuationJournal(root, contract)
            journal.append({"continuation_uid": uid, "payload": 1}, None)
            entry_path = root / "entries" / f"{uid}.json"
            envelope = json.loads(entry_path.read_text(encoding="utf-8"))
            envelope["record"]["payload"] = 2
            entry_path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "content SHA256 mismatch"):
                NestedContinuationJournal(root, contract)


class NestedArtifactManifestTests(unittest.TestCase):
    def test_scale_ready_rejects_missing_request_intent_observer_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actor = root / "actor"
            actor.mkdir()
            (actor / "weights.bin").write_bytes(b"weights")
            actor_sha = sha256_actor_checkpoint(actor)
            actor_attestor = ActorCheckpointRunAttestation(actor, actor_sha)
            actor_attestor.verify_runtime()
            actor_report = actor_attestor.finish()
            raw = _raw_collection(
                actor_sha,
                actor_report["stat_snapshot_sha256"],
                actor_report["actor_run_uid"],
            )
            journal, report = _completed_journal(
                root / "journal", raw, actor_report
            )
            continuation = raw["continuations"][0]
            continuation["completion_calls"] = 2
            continuation["downstream_request_seeds"] = [7]
            continuation["downstream_prompt_sha256"] = ["a" * 64]
            continuation["rollout_content_sha256"] = _sha256_json(
                {
                    "version": "shopping-nested-rollout-content-v2",
                    "record": {
                        key: value
                        for key, value in continuation.items()
                        if key != "rollout_content_sha256"
                    },
                }
            )

            with self.assertRaisesRegex(ValueError, "request-intent cardinality"):
                finalize_scale_ready_collection(
                    raw,
                    journal_path=journal.root,
                    journal_report=report,
                    actor_attestation=actor_report,
                )

    def test_zero_call_resume_cannot_cover_an_unattested_old_actor_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actor = root / "actor"
            actor.mkdir()
            (actor / "weights.bin").write_bytes(b"weights")
            actor_sha = sha256_actor_checkpoint(actor)
            old_attestor = ActorCheckpointRunAttestation(actor, actor_sha)
            old_attestor.verify_runtime()
            old_report = old_attestor.finish()
            resumed_attestor = ActorCheckpointRunAttestation(actor, actor_sha)
            resumed_attestor.verify_runtime()
            resumed_report = resumed_attestor.finish()
            collection = _raw_collection(
                actor_sha,
                old_report["stat_snapshot_sha256"],
                old_report["actor_run_uid"],
            )
            summary = collection["summary"]
            stage1_binding = summary["provenance"]["stage1_source_binding"]
            contract = build_nested_journal_contract(
                experiment_uid=summary["experiment_uid"],
                active_plan_sha256=summary["plan_sha256"],
                formal_plan_sha256=summary["formal_plan_sha256"],
                resolved_selections_sha256=summary[
                    "resolved_selections_sha256"
                ],
                stage1_source_sha256=summary["stage1_source_sha256"],
                stage1_manifest_sha256=stage1_binding[
                    "final_manifest_sha256"
                ],
                stage1_source_finalized=True,
                harness_contract_sha256=summary["harness_contract_sha256"],
                actor_runtime_binding={
                    "schema_version": ACTOR_RUN_ATTESTATION_VERSION,
                    "actor_checkpoint_sha256": actor_sha,
                    "stat_snapshot_sha256": resumed_report[
                        "stat_snapshot_sha256"
                    ],
                    "stat_entry_count": resumed_report["stat_entry_count"],
                    "read_only_run_binding": True,
                },
                continuations_per_decision=4,
                expected_continuation_uids=[
                    item["continuation_uid"]
                    for item in collection["continuations"]
                ],
            )
            journal = NestedContinuationJournal(root / "journal", contract)
            journal.begin_actor_run(resumed_report)
            for record in collection["continuations"]:
                journal.append(record, None)
            journal.finish_actor_run(resumed_report)

            with self.assertRaisesRegex(ValueError, "unattested actor run"):
                journal.finalize()

    def test_manifest_is_commit_point_and_binds_bytes_and_counts(self) -> None:
        with (
            tempfile.TemporaryDirectory() as actor_tmp,
            tempfile.TemporaryDirectory() as output_tmp,
        ):
            actor = Path(actor_tmp)
            (actor / "weights.bin").write_bytes(b"weights")
            actor_sha = sha256_actor_checkpoint(actor)
            actor_attestor = ActorCheckpointRunAttestation(actor, actor_sha)
            actor_attestor.verify_runtime()
            actor_report = actor_attestor.finish()
            raw = _raw_collection(
                actor_sha,
                actor_report["stat_snapshot_sha256"],
                actor_report["actor_run_uid"],
            )
            root = Path(output_tmp)
            journal, report = _completed_journal(
                root / "journal", raw, actor_report
            )
            completed = finalize_scale_ready_collection(
                raw,
                journal_path=journal.root,
                journal_report=report,
                actor_attestation=actor_report,
            )
            artifacts = {
                "decisions": root / "decisions.jsonl",
                "continuations": root / "continuations.jsonl",
                "summary": root / "summary.json",
            }
            manifest = root / "manifest.json"

            with self.assertRaisesRegex(ValueError, "manifest is required"):
                verify_completed_collection_manifest(
                    manifest, artifact_files=artifacts
                )
            committed = commit_nested_collection_artifacts(
                completed,
                artifact_files=artifacts,
                manifest_path=manifest,
                journal_path=journal.root,
                journal_report=report,
                actor_attestation=actor_report,
            )
            verified_snapshot = verify_completed_collection_manifest(
                manifest,
                artifact_files=artifacts,
                include_file_attestation=True,
            )
            verified = verified_snapshot["manifest"]
            self.assertEqual(
                verified_snapshot["manifest_file_sha256"],
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )

            actor_start_path = next(
                (journal.actor_runs_root).glob("*.start.json")
            )
            actor_start_bytes = actor_start_path.read_bytes()
            actor_start_path.write_bytes(actor_start_bytes + b" ")
            with self.assertRaisesRegex(ValueError, "journal files differ"):
                verify_completed_collection_manifest(
                    manifest, artifact_files=artifacts
                )
            actor_start_path.write_bytes(actor_start_bytes)

            entry_path = next((journal.root / "entries").glob("*.json"))
            entry_bytes = entry_path.read_bytes()
            entry_path.write_bytes(entry_bytes + b" ")
            with self.assertRaisesRegex(ValueError, "journal files differ"):
                verify_completed_collection_manifest(
                    manifest, artifact_files=artifacts
                )
            entry_path.write_bytes(entry_bytes)

            external_continuations = root / "external-continuations.jsonl"
            artifacts["continuations"].replace(external_continuations)
            try:
                os.symlink(
                    external_continuations,
                    artifacts["continuations"],
                )
            except OSError:
                external_continuations.replace(artifacts["continuations"])
            else:
                with self.assertRaisesRegex(ValueError, "symbolic link"):
                    verify_completed_collection_manifest(
                        manifest, artifact_files=artifacts
                    )
                artifacts["continuations"].unlink()
                external_continuations.replace(artifacts["continuations"])

            artifacts["continuations"].write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                verify_completed_collection_manifest(
                    manifest, artifact_files=artifacts
                )

        self.assertEqual(verified["artifact_set_uid"], committed["artifact_set_uid"])
        self.assertEqual(
            committed["artifacts"]["continuations"]["record_count"], 4
        )
        self.assertTrue(completed["summary"]["safety"]["scale_collection_ready"])
        self.assertEqual(completed["summary"]["safety"]["formal_scale_blockers"], [])

    def test_scale_ready_rejects_incomplete_journal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            actor = Path(tmp) / "actor"
            actor.mkdir()
            (actor / "weights.bin").write_bytes(b"weights")
            actor_sha = sha256_actor_checkpoint(actor)
            actor_attestor = ActorCheckpointRunAttestation(actor, actor_sha)
            actor_attestor.verify_runtime()
            actor_report = actor_attestor.finish()
            raw = _raw_collection(
                actor_sha,
                actor_report["stat_snapshot_sha256"],
                actor_report["actor_run_uid"],
            )
            journal, report = _completed_journal(
                Path(tmp) / "journal", raw, actor_report
            )
            report["expected_record_count"] = 0
            report["record_count"] = 0

            with self.assertRaisesRegex(ValueError, "expected collection"):
                finalize_scale_ready_collection(
                    raw,
                    journal_path=journal.root,
                    journal_report=report,
                    actor_attestation=actor_report,
                )

    def test_scale_ready_rejects_storage_shaped_fake_collection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            actor = Path(tmp) / "actor"
            actor.mkdir()
            (actor / "weights.bin").write_bytes(b"weights")
            actor_sha = sha256_actor_checkpoint(actor)
            actor_attestor = ActorCheckpointRunAttestation(actor, actor_sha)
            actor_attestor.verify_runtime()
            actor_report = actor_attestor.finish()
            raw = _raw_collection(
                actor_sha,
                actor_report["stat_snapshot_sha256"],
                actor_report["actor_run_uid"],
            )
            journal, report = _completed_journal(
                Path(tmp) / "journal", raw, actor_report
            )
            raw["decisions"][0].pop("schema_version")

            with self.assertRaisesRegex(ValueError, "decision schema"):
                finalize_scale_ready_collection(
                    raw,
                    journal_path=journal.root,
                    journal_report=report,
                    actor_attestation=actor_report,
                )

    def test_scale_ready_rejects_journal_header_from_another_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            actor = Path(tmp) / "actor"
            actor.mkdir()
            (actor / "weights.bin").write_bytes(b"weights")
            actor_sha = sha256_actor_checkpoint(actor)
            actor_attestor = ActorCheckpointRunAttestation(actor, actor_sha)
            actor_attestor.verify_runtime()
            actor_report = actor_attestor.finish()
            raw = _raw_collection(
                actor_sha,
                actor_report["stat_snapshot_sha256"],
                actor_report["actor_run_uid"],
            )
            journal, report = _completed_journal(
                Path(tmp) / "journal", raw, actor_report
            )
            report["header_sha256"] = "0" * 64

            with self.assertRaisesRegex(ValueError, "formal collection"):
                finalize_scale_ready_collection(
                    raw,
                    journal_path=journal.root,
                    journal_report=report,
                    actor_attestation=actor_report,
                )

    def test_scale_ready_rejects_formal_plan_v1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            actor = Path(tmp) / "actor"
            actor.mkdir()
            (actor / "weights.bin").write_bytes(b"weights")
            actor_sha = sha256_actor_checkpoint(actor)
            actor_attestor = ActorCheckpointRunAttestation(actor, actor_sha)
            actor_attestor.verify_runtime()
            actor_report = actor_attestor.finish()
            raw = _raw_collection(
                actor_sha,
                actor_report["stat_snapshot_sha256"],
                actor_report["actor_run_uid"],
            )
            raw["summary"]["formal_plan"]["schema_version"] = (
                "shopping-nested-formal-plan-v1"
            )
            raw["summary"]["formal_plan_sha256"] = _sha256_json(
                raw["summary"]["formal_plan"]
            )
            journal, report = _completed_journal(
                Path(tmp) / "journal", raw, actor_report
            )

            with self.assertRaisesRegex(ValueError, "formal plan v2"):
                finalize_scale_ready_collection(
                    raw,
                    journal_path=journal.root,
                    journal_report=report,
                    actor_attestation=actor_report,
                )

    def test_manifest_commit_cannot_bypass_scale_finalizer(self) -> None:
        with (
            tempfile.TemporaryDirectory() as actor_tmp,
            tempfile.TemporaryDirectory() as output_tmp,
        ):
            actor = Path(actor_tmp)
            (actor / "weights.bin").write_bytes(b"weights")
            actor_sha = sha256_actor_checkpoint(actor)
            actor_attestor = ActorCheckpointRunAttestation(actor, actor_sha)
            actor_attestor.verify_runtime()
            actor_report = actor_attestor.finish()
            raw = _raw_collection(
                actor_sha,
                actor_report["stat_snapshot_sha256"],
                actor_report["actor_run_uid"],
            )
            root = Path(output_tmp)
            journal, report = _completed_journal(
                root / "journal", raw, actor_report
            )
            raw["summary"]["safety"]["scale_collection_ready"] = True
            raw["summary"]["safety"]["formal_scale_blockers"] = []
            with self.assertRaisesRegex(ValueError, "scale-storage gate"):
                commit_nested_collection_artifacts(
                    raw,
                    artifact_files={
                        "decisions": root / "decisions.jsonl",
                        "continuations": root / "continuations.jsonl",
                        "summary": root / "summary.json",
                    },
                    manifest_path=root / "manifest.json",
                    journal_path=journal.root,
                    journal_report=report,
                    actor_attestation=actor_report,
                )

    def test_manifest_retry_after_artifacts_written_is_byte_stable(self) -> None:
        with (
            tempfile.TemporaryDirectory() as actor_tmp,
            tempfile.TemporaryDirectory() as output_tmp,
        ):
            actor = Path(actor_tmp)
            (actor / "weights.bin").write_bytes(b"weights")
            actor_sha = sha256_actor_checkpoint(actor)
            first_attestor = ActorCheckpointRunAttestation(actor, actor_sha)
            first_attestor.verify_runtime()
            first_report = first_attestor.finish()
            raw = _raw_collection(
                actor_sha,
                first_report["stat_snapshot_sha256"],
                first_report["actor_run_uid"],
            )
            root = Path(output_tmp)
            journal, journal_report = _completed_journal(
                root / "journal", raw, first_report
            )
            first = finalize_scale_ready_collection(
                raw,
                journal_path=journal.root,
                journal_report=journal_report,
                actor_attestation=first_report,
            )
            artifacts = {
                "decisions": root / "decisions.jsonl",
                "continuations": root / "continuations.jsonl",
                "summary": root / "summary.json",
            }
            manifest = root / "manifest.json"
            commit_nested_collection_artifacts(
                first,
                artifact_files=artifacts,
                manifest_path=manifest,
                journal_path=journal.root,
                journal_report=journal_report,
                actor_attestation=first_report,
            )
            summary_bytes = artifacts["summary"].read_bytes()
            manifest.unlink()

            second_attestor = ActorCheckpointRunAttestation(actor, actor_sha)
            second_attestor.verify_runtime()
            second_attestor.finish()
            persisted_report = journal.latest_actor_attestation()
            second = finalize_scale_ready_collection(
                first,
                journal_path=journal.root,
                journal_report=journal_report,
                actor_attestation=persisted_report,
            )
            commit_nested_collection_artifacts(
                second,
                artifact_files=artifacts,
                manifest_path=manifest,
                journal_path=journal.root,
                journal_report=journal_report,
                actor_attestation=persisted_report,
            )
            retry_summary_bytes = artifacts["summary"].read_bytes()

        self.assertEqual(first["summary"], second["summary"])
        self.assertEqual(retry_summary_bytes, summary_bytes)

    def test_scale_ready_rejects_unfinalized_stage1_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            actor = Path(tmp) / "actor"
            actor.mkdir()
            (actor / "weights.bin").write_bytes(b"weights")
            actor_sha = sha256_actor_checkpoint(actor)
            actor_attestor = ActorCheckpointRunAttestation(actor, actor_sha)
            actor_attestor.verify_runtime()
            actor_report = actor_attestor.finish()
            collection = _raw_collection(
                actor_sha,
                actor_report["stat_snapshot_sha256"],
                actor_report["actor_run_uid"],
            )
            journal, report = _completed_journal(
                Path(tmp) / "journal", collection, actor_report
            )
            collection["summary"]["provenance"]["stage1_source_binding"] = None
            collection["summary"]["safety"]["finalized_stage1_source"] = False

            with self.assertRaisesRegex(TypeError, "Stage-1 source binding"):
                finalize_scale_ready_collection(
                    collection,
                    journal_path=journal.root,
                    journal_report=report,
                    actor_attestation=actor_report,
                )


if __name__ == "__main__":
    unittest.main()
