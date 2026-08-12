from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_active_suffix import (
    MANIFEST_SHA256,
    TOKENIZER_SHA256,
    AlwaysBuyClient,
    FakeEncoder,
    FakeEnvironment,
    FakeParser,
    FakePlanBoundClient,
    decoding_and_backend,
    resolved_branch,
    reward_detail,
)

from scripts.estimate_nested_credit import main as estimate_credit_main
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS
from shopping_grpo.training.grpo.active_branch import build_active_branch_plan
from shopping_grpo.training.grpo.active_suffix import (
    _sha256_json,
    sha256_actor_checkpoint,
)
from shopping_grpo.training.grpo.nested_artifacts import (
    ACTOR_RUN_ATTESTATION_VERSION,
    NESTED_REQUEST_INTENT_VERSION,
    ActorCheckpointRunAttestation,
    NestedContinuationJournal,
    build_nested_journal_contract,
    commit_nested_collection_artifacts,
    finalize_scale_ready_collection,
)
from shopping_grpo.training.grpo.nested_continuation import (
    NestedContinuationCollector,
    _with_rollout_content_sha256,
)
from shopping_grpo.training.grpo.nested_credit import (
    NESTED_CREDIT_VERSION,
    NESTED_HELDOUT_VERSION,
    NESTED_SAMPLES_VERSION,
    continuation_seed_uid,
    continuation_turn_seed,
    continuation_uid,
    estimate_nested_collection_credit,
    estimate_nested_decision_credit,
    proposal_uid,
    validate_nested_samples,
)
from shopping_grpo.training.grpo.pivotal_states import (
    canonical_replay_action,
    replay_action_sha256,
)
from shopping_grpo.training.grpo.stage1_proposal import (
    FirstDecisionStage1Collector,
    write_first_decision_artifacts,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _state(
    state_index: int,
    rewards_by_decision: list[list[float]],
    proposal_decisions: list[int],
    *,
    task_id: int | None = None,
) -> dict[str, object]:
    state_uid = _sha(f"state:{state_index}")
    decision_uids = [_sha(f"decision:{index}") for index in range(len(rewards_by_decision))]
    proposals = [
        {
            "proposal_index": proposal_index,
            "proposal_uid": proposal_uid(state_uid, proposal_index),
            "decision_uid": decision_uids[decision_index],
        }
        for proposal_index, decision_index in enumerate(proposal_decisions)
    ]
    decisions = []
    for decision_index, rewards in enumerate(rewards_by_decision):
        decision_uid = decision_uids[decision_index]
        decisions.append(
            {
                "decision_index": decision_index,
                "decision_uid": decision_uid,
                "continuations": [
                    {
                        "continuation_index": continuation_index,
                        "continuation_uid": continuation_uid(
                            state_uid, decision_uid, continuation_index
                        ),
                        "continuation_seed_uid": continuation_seed_uid(
                            state_uid, continuation_index
                        ),
                        "downstream_request_seeds": [
                            continuation_turn_seed(state_uid, continuation_index, 0)
                        ],
                        "rollout_content_sha256": _sha(
                            f"rollout:{state_index}:{decision_index}:{continuation_index}"
                        ),
                        "generation_mode": "sampled_continuation",
                        "reward": reward,
                        "valid_for_learning": True,
                        "infrastructure_invalid": False,
                        "reward_invalid": False,
                        "reward_unverifiable": False,
                        "sampling_invalid": False,
                        "model_failure": False,
                        "invalid_reason": None,
                    }
                    for continuation_index, reward in enumerate(rewards)
                ],
            }
        )
    return {
        "state_index": state_index,
        "state_uid": state_uid,
        "task_id": state_index if task_id is None else task_id,
        "proposals": proposals,
        "decisions": decisions,
    }


def _payload(
    states: list[dict[str, object]],
    continuation_count: int = 8,
    *,
    source_attested: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": NESTED_SAMPLES_VERSION,
        "experiment_uid": _sha("experiment"),
        "reward_contract_sha256": _sha("shopping-policy-reward-v1"),
        "collected_continuations_per_decision": continuation_count,
        "train_fold_indices": [0, 1, 2, 3],
        "gate_fold_indices": [4, 5, 6, 7],
        "source_attestation": {
            "schema_version": "shopping-psa-nested-source-attestation-v1",
            "attested": source_attested,
            "contract_sha256": _sha("source") if source_attested else None,
        },
        "states": states,
    }


class NestedCreditEstimatorTests(unittest.TestCase):
    def test_formula_uses_variance_of_mean_and_proposal_multiplicity(self) -> None:
        payload = _payload(
            [
                _state(
                    0,
                    [
                        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [0, 0, 0, 1],
                )
            ]
        )

        result = estimate_nested_decision_credit(payload)

        self.assertEqual(result["schema_version"], NESTED_CREDIT_VERSION)
        state = result["states"][0]
        self.assertTrue(state["structural_gate"]["eligible_for_credit"])
        self.assertAlmostEqual(state["mu_s"], 0.25)
        self.assertAlmostEqual(state["weighted_observed_variance"], 3.0 / 16.0)
        self.assertAlmostEqual(state["weighted_uncertainty"], 1.0 / 32.0)
        self.assertAlmostEqual(state["between_variance"], 5.0 / 32.0)
        decision_a, decision_b = state["decision_values"]
        self.assertEqual(decision_a["proposal_multiplicity"], 3)
        self.assertEqual(decision_b["proposal_multiplicity"], 1)
        self.assertAlmostEqual(decision_a["q_d"], 0.5)
        self.assertAlmostEqual(decision_b["q_d"], -0.5)
        self.assertAlmostEqual(decision_a["u_d"], 1.0 / 12.0)
        self.assertAlmostEqual(decision_b["u_d"], 1.0 / 12.0)
        self.assertAlmostEqual(decision_a["kappa_d"], 15.0 / 16.0, places=6)
        self.assertAlmostEqual(decision_b["kappa_d"], 5.0 / 8.0, places=6)
        self.assertAlmostEqual(decision_a["q_tilde_d"], 31.0 / 64.0, places=6)
        self.assertAlmostEqual(decision_b["q_tilde_d"], -7.0 / 32.0, places=6)
        self.assertAlmostEqual(state["v_s"], 79.0 / 256.0, places=6)
        self.assertAlmostEqual(decision_a["advantage"], 1.0 / 6.0, places=6)
        self.assertAlmostEqual(decision_b["advantage"], -0.5, places=6)
        self.assertEqual(decision_a["loss_weight"], 3)
        self.assertAlmostEqual(
            sum(row["loss_weight"] * row["advantage"] for row in state["decision_values"]),
            0.0,
        )
        self.assertFalse(result["estimator_contract"]["standard_deviation_normalization"])

    def test_crn_covariance_uses_all_paired_slots_and_stays_psd(self) -> None:
        payload = _payload(
            [
                _state(
                    0,
                    [
                        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    [0, 0, 1, 1],
                )
            ]
        )

        result = estimate_nested_decision_credit(payload)

        covariance = result["states"][0]["paired_crn_covariance"]
        self.assertTrue(covariance["slot_alignment_verified"])
        self.assertTrue(
            covariance["full_decision_slot_cartesian_product_verified"]
        )
        self.assertTrue(covariance["psd_by_convex_construction"])
        self.assertEqual(covariance["slot_indices"], [0, 1, 2, 3])
        self.assertAlmostEqual(
            covariance["raw_covariance_of_means"][0][1], 1.0 / 12.0
        )
        self.assertAlmostEqual(
            covariance["shrunk_covariance_of_means"][0][1], 1.0 / 24.0
        )
        self.assertAlmostEqual(
            result["states"][0]["weighted_uncertainty"], 1.0 / 48.0
        )

    def test_advantage_is_clipped_to_half(self) -> None:
        state = _state(0, [[1.0] * 8, [-1.0] * 8], [0, 1, 1, 1])

        result = estimate_nested_decision_credit(_payload([state]))

        advantages = [row["advantage"] for row in result["states"][0]["decision_values"]]
        self.assertEqual(advantages[0], 0.5)
        self.assertGreaterEqual(advantages[1], -0.5)
        self.assertLessEqual(advantages[1], 0.5)

    def test_constant_outcomes_remain_structurally_eligible_with_zero_advantage(self) -> None:
        state = _state(0, [[0.25] * 8, [0.25] * 8], [0, 0, 1, 1])

        result = estimate_nested_decision_credit(_payload([state]))

        estimated = result["states"][0]
        self.assertTrue(estimated["structural_gate"]["eligible_for_credit"])
        self.assertEqual(estimated["structural_gate"]["reward_dependent_checks"], [])
        self.assertEqual(
            [row["advantage"] for row in estimated["decision_values"]],
            [0.0, 0.0],
        )
        self.assertFalse(estimated["heldout_diagnostic"]["train_ranking_identifiable"])
        self.assertFalse(result["training_gate"]["training_ready"])

    def test_first_two_last_two_heldout_is_excluded_from_structural_gate(self) -> None:
        state = _state(
            0,
            [
                [1.0, 0.5, 1.0, 0.5, -1.0, -0.5, -1.0, -0.5],
                [0.0, -0.5, 0.0, -0.5, 1.0, 0.5, 1.0, 0.5],
            ],
            [0, 0, 1, 1],
        )

        result = estimate_nested_decision_credit(_payload([state]))

        diagnostic = result["states"][0]["heldout_diagnostic"]
        self.assertEqual(diagnostic["version"], NESTED_HELDOUT_VERSION)
        self.assertEqual(diagnostic["train_indices"], [0, 1, 2, 3])
        self.assertEqual(diagnostic["heldout_indices"], [4, 5, 6, 7])
        self.assertAlmostEqual(diagnostic["train_delta"], 1.0)
        self.assertAlmostEqual(diagnostic["heldout_delta"], -1.5)
        self.assertFalse(diagnostic["split_half_ranking_consistent"])
        self.assertFalse(diagnostic["used_by_structural_gate"])
        self.assertTrue(diagnostic["used_by_signal_gate"])
        self.assertTrue(result["states"][0]["structural_gate"]["eligible_for_credit"])

    def test_invalid_slot_closes_credit_without_becoming_a_reward_gate(self) -> None:
        state = _state(0, [[0.0] * 8, [0.0] * 8], [0, 0, 1, 1])
        continuation = state["decisions"][0]["continuations"][0]
        continuation.update(
            {
                "generation_mode": "infrastructure_invalid",
                "reward": None,
                "valid_for_learning": False,
                "infrastructure_invalid": True,
                "reward_invalid": False,
                "reward_unverifiable": False,
                "sampling_invalid": True,
                "invalid_reason": "token_alignment_invalid",
            }
        )

        result = estimate_nested_decision_credit(_payload([state]))

        estimated = result["states"][0]
        self.assertFalse(estimated["structural_gate"]["eligible_for_credit"])
        self.assertIn(
            "incomplete_valid_train_fold",
            estimated["structural_gate"]["failed_checks"],
        )
        self.assertEqual(estimated["decision_values"], [])

    def test_reward_unverifiable_slot_is_counted_but_never_enters_credit(self) -> None:
        state = _state(0, [[0.0] * 8, [0.0] * 8], [0, 0, 1, 1])
        continuation = state["decisions"][0]["continuations"][0]
        continuation.update(
            {
                "reward": None,
                "valid_for_learning": False,
                "reward_invalid": True,
                "reward_unverifiable": True,
                "sampling_invalid": True,
                "invalid_reason": "reward_unverifiable",
            }
        )

        result = estimate_nested_decision_credit(_payload([state]))

        estimated = result["states"][0]
        self.assertFalse(estimated["structural_gate"]["eligible_for_credit"])
        self.assertEqual(estimated["reward_invalid_continuations"], 1)
        self.assertEqual(estimated["sampling_invalid_continuations"], 1)
        self.assertEqual(estimated["decision_values"], [])
        self.assertIsNone(estimated["heldout_diagnostic"])

    def test_l_one_is_well_formed_but_not_credit_eligible(self) -> None:
        state = _state(0, [[0.0], [1.0]], [0, 0, 1, 1])

        result = estimate_nested_decision_credit(_payload([state], continuation_count=1))

        estimated = result["states"][0]
        self.assertFalse(estimated["structural_gate"]["eligible_for_credit"])
        self.assertIn(
            "formal_train_gate_cardinality_incomplete",
            estimated["structural_gate"]["failed_checks"],
        )
        self.assertEqual(estimated["decision_values"], [])

    def test_forged_source_attestation_cannot_unlock_optimizer(self) -> None:
        states = [
            _state(index, [[1.0] * 8, [-1.0] * 8], [0, 0, 1, 1])
            for index in range(32)
        ]

        result = estimate_nested_decision_credit(_payload(states, source_attested=True))

        gate = result["training_gate"]
        self.assertFalse(gate["training_ready"])
        self.assertFalse(gate["optimizer_unlock_allowed"])
        self.assertTrue(gate["structural_ready"])
        self.assertTrue(gate["signal_ready"])
        self.assertTrue(gate["statistical_ready"])
        self.assertTrue(gate["statistical_candidate_ready"])
        self.assertFalse(gate["attested_collection"])
        self.assertIn("collection_artifacts_not_attested", gate["failed_checks"])
        self.assertEqual(gate["observed"]["eligible_states"], 32)
        self.assertEqual(gate["observed"]["eligible_unique_tasks"], 32)
        self.assertAlmostEqual(
            gate["observed"]["state_proposal_weight_effective_sample_size"], 32.0
        )
        self.assertAlmostEqual(
            gate["observed"]["task_cluster_proposal_weight_effective_sample_size"],
            32.0,
        )
        signal = result["signal_gate"]
        self.assertTrue(signal["signal_ready"])
        self.assertEqual(signal["observed"]["high_kappa_state_rate"], 1.0)
        self.assertEqual(signal["observed"]["heldout_mean_delta"], 2.0)
        self.assertEqual(signal["observed"]["split_half_consistency"], 1.0)
        self.assertEqual(signal["bootstrap"]["mean_delta_ci95"], [2.0, 2.0])
        self.assertEqual(signal["bootstrap"]["consistency_ci95"], [1.0, 1.0])
        self.assertLess(signal["permutation"]["null_95th_percentile"], 2.0)

    def test_32_constant_states_pass_structure_but_not_signal(self) -> None:
        states = [
            _state(index, [[0.0] * 8, [0.0] * 8], [0, 0, 1, 1])
            for index in range(32)
        ]

        result = estimate_nested_decision_credit(_payload(states, source_attested=True))

        self.assertTrue(result["states"][0]["structural_gate"]["eligible_for_credit"])
        self.assertTrue(result["training_gate"]["structural_ready"])
        self.assertFalse(result["signal_gate"]["signal_ready"])
        self.assertFalse(result["training_gate"]["training_ready"])
        self.assertIn(
            "no_train_ranking_identifiable_tasks",
            result["signal_gate"]["failed_checks"],
        )

    def test_imbalanced_state_proposal_counts_fail_ess_even_with_32_states(self) -> None:
        states = [
            _state(index, [[0.0] * 8, [0.0] * 8], [0, 0, 1, 1])
            for index in range(32)
        ]
        states[0] = _state(0, [[0.0] * 8, [0.0] * 8], [0] * 99 + [1])

        gate = estimate_nested_decision_credit(
            _payload(states, source_attested=True)
        )["training_gate"]

        self.assertFalse(gate["training_ready"])
        self.assertNotIn("insufficient_eligible_states", gate["failed_checks"])
        self.assertIn("insufficient_state_effective_sample_size", gate["failed_checks"])

    def test_repeated_tasks_cannot_fake_the_global_training_gate(self) -> None:
        states = [
            _state(
                index,
                [[1.0] * 8, [-1.0] * 8],
                [0, 0, 1, 1],
                task_id=7,
            )
            for index in range(32)
        ]

        result = estimate_nested_decision_credit(_payload(states, source_attested=True))

        gate = result["training_gate"]
        self.assertFalse(gate["structural_ready"])
        self.assertFalse(gate["signal_ready"])
        self.assertFalse(gate["training_ready"])
        self.assertEqual(gate["observed"]["eligible_unique_tasks"], 1)
        self.assertAlmostEqual(
            gate["observed"]["task_cluster_proposal_weight_effective_sample_size"],
            1.0,
        )
        self.assertIn("insufficient_unique_tasks", gate["failed_checks"])
        self.assertIn(
            "insufficient_task_cluster_effective_sample_size", gate["failed_checks"]
        )
        self.assertIn(
            "signal:insufficient_identifiable_task_effective_sample_size",
            gate["failed_checks"],
        )


class NestedCollectionAdapterTests(unittest.TestCase):
    def _artifacts(
        self,
        root: Path,
        *,
        resolved_rows=None,
        suffixes_per_state: int = 2,
        continuations_per_decision: int = 4,
        stage1_mutator=None,
        stage1_client=None,
        env_factory=FakeEnvironment,
    ):
        actor = root / "actor"
        actor.mkdir()
        (actor / "weights.bin").write_bytes(b"weights")
        actor_sha = sha256_actor_checkpoint(actor)
        decoding, backend = decoding_and_backend(actor_sha)
        resolved = resolved_rows or [resolved_branch(17, [10, 20])]
        plan = build_active_branch_plan(
            resolved,
            actor_checkpoint_sha256=actor_sha,
            decoding_config=decoding,
            seed=20260811,
            suffixes_per_state=suffixes_per_state,
        )
        stage1_collector = FirstDecisionStage1Collector(
            plan=plan,
            resolved_selections=resolved,
            actor_checkpoint=actor,
            actor_tokenizer_contract_sha256=TOKENIZER_SHA256,
            sampling_backend_contract=backend,
            completion_client=stage1_client or FakePlanBoundClient(),
            parser=FakeParser(),
            tool_schemas=SHOP_TOOL_SCHEMAS,
            required_environment_version="shopsimulator-environment-v2.1",
            expected_states=len(resolved),
            proposals_per_state=suffixes_per_state,
            source_provenance={
                "schema_version": (
                    "shopping-first-decision-source-provenance-v2"
                ),
                "git_sha": "c" * 40,
                "git_worktree_clean": True,
                "git_status_sha256": hashlib.sha256(b"").hexdigest(),
                "execution_mode": "formal",
                "dirty_source_override": False,
                "scale_ready": True,
            },
        )
        stage1_collection = asyncio.run(
            stage1_collector.collect(root / "stage1-journal")
        )
        stage1 = stage1_collection["records"]
        if stage1_mutator is not None:
            stage1_mutator(stage1)
        stage1_records_path = root / "stage1.jsonl"
        stage1_manifest_path = root / "stage1-manifest.json"
        write_first_decision_artifacts(
            stage1_collection,
            records_output=stage1_records_path,
            manifest_output=stage1_manifest_path,
        )
        stage1_source_sha256 = hashlib.sha256(
            stage1_records_path.read_bytes()
        ).hexdigest()
        actor_attestor = ActorCheckpointRunAttestation(actor, actor_sha)
        collector = NestedContinuationCollector(
            plan=plan,
            resolved_selections=resolved,
            stage1_records=stage1,
            stage1_source_sha256=stage1_source_sha256,
            stage1_records_path=stage1_records_path,
            stage1_manifest_path=stage1_manifest_path,
            actor_checkpoint=actor,
            sampling_backend_contract=backend,
            completion_client=AlwaysBuyClient(),
            parser=FakeParser(),
            encoder=FakeEncoder(),
            env_factory=env_factory,
            tool_schemas=SHOP_TOOL_SCHEMAS,
            required_environment_version="shopsimulator-environment-v2.1",
            expected_proposals=len(resolved) * suffixes_per_state,
            continuations_per_decision=continuations_per_decision,
            actor_checkpoint_attestor=actor_attestor,
        )
        request_intents = []

        def record_request_intent(intent):
            normalized = {
                "schema_version": NESTED_REQUEST_INTENT_VERSION,
                "intent_uid": _sha256_json(
                    {
                        "schema_version": NESTED_REQUEST_INTENT_VERSION,
                        **intent,
                    }
                ),
                **copy.deepcopy(intent),
            }
            normalized["intent_sha256"] = _sha256_json(normalized)
            request_intents.append(normalized)
            return normalized

        collector.set_completion_intent_observer(record_request_intent)
        collection = asyncio.run(collector.collect())
        actor_attestation = actor_attestor.finish()
        if not hasattr(self, "_scale_contexts"):
            self._scale_contexts = {}
        self._scale_contexts[str(root.resolve())] = {
            "actor": actor,
            "actor_attestation": actor_attestation,
            "stage1_records_path": stage1_records_path,
            "stage1_manifest_path": stage1_manifest_path,
            "stage1_source_sha256": stage1_source_sha256,
            "request_intents": request_intents,
        }
        return (
            collection,
            plan,
            resolved,
            stage1,
            backend,
        )

    def _estimate(
        self,
        collection,
        root,
        plan,
        resolved,
        stage1,
        backend,
        *,
        persisted_collection=None,
        environment_manifest_sha256=MANIFEST_SHA256,
    ):
        manifest = json.loads(Path("data/environment.json").read_text(encoding="utf-8"))
        persisted = persisted_collection or collection
        scale_context = self._scale_contexts[str(root.resolve())]
        actor = scale_context["actor"]
        with tempfile.TemporaryDirectory() as artifact_tmp:
            artifact_root = Path(artifact_tmp)
            artifact_files = {
                "decisions": artifact_root / "decisions.jsonl",
                "continuations": artifact_root / "continuations.jsonl",
                "summary": artifact_root / "summary.json",
            }
            actor_attestation = scale_context["actor_attestation"]
            persisted_summary = persisted["summary"]
            actor_binding = {
                "schema_version": ACTOR_RUN_ATTESTATION_VERSION,
                "actor_checkpoint_sha256": actor_attestation[
                    "actor_checkpoint_sha256"
                ],
                "stat_snapshot_sha256": actor_attestation[
                    "stat_snapshot_sha256"
                ],
                "stat_entry_count": actor_attestation["stat_entry_count"],
                "read_only_run_binding": True,
            }
            journal_contract = build_nested_journal_contract(
                experiment_uid=persisted_summary["experiment_uid"],
                active_plan_sha256=persisted_summary["plan_sha256"],
                formal_plan_sha256=persisted_summary["formal_plan_sha256"],
                resolved_selections_sha256=persisted_summary[
                    "resolved_selections_sha256"
                ],
                stage1_source_sha256=persisted_summary["stage1_source_sha256"],
                stage1_manifest_sha256=persisted_summary["provenance"][
                    "stage1_source_binding"
                ]["final_manifest_sha256"],
                stage1_source_finalized=True,
                harness_contract_sha256=persisted_summary[
                    "harness_contract_sha256"
                ],
                actor_runtime_binding=actor_binding,
                continuations_per_decision=persisted_summary["aggregate"][
                    "continuations_per_decision"
                ],
                expected_continuation_uids=[
                    continuation_uid_value
                    for decision in persisted["decisions"]
                    for continuation_uid_value in decision["continuation_uids"]
                ],
            )
            journal = NestedContinuationJournal(
                artifact_root / "journal", journal_contract
            )
            journal.begin_actor_run(actor_attestation)
            boundaries = {
                decision["decision_uid"]: decision.get("post_action_boundary")
                for decision in persisted["decisions"]
            }
            for continuation in persisted["continuations"]:
                for intent in scale_context["request_intents"]:
                    if (
                        intent["continuation_uid"]
                        != continuation["continuation_uid"]
                    ):
                        continue
                    journal.begin_completion_request(
                        {
                            name: intent[name]
                            for name in (
                                "continuation_uid",
                                "actor_run_uid",
                                "actor_stat_snapshot_sha256",
                                "request_index",
                                "seed",
                                "prompt_sha256",
                            )
                        }
                    )
                journal.append(
                    continuation,
                    boundaries.get(continuation["decision_uid"]),
                )
            journal.finish_actor_run(actor_attestation)
            journal_report = journal.finalize()
            finalized_persisted = finalize_scale_ready_collection(
                persisted,
                journal_path=journal.root,
                journal_report=journal_report,
                actor_attestation=actor_attestation,
            )
            if collection is persisted_collection or collection is persisted:
                finalized_collection = finalized_persisted
            else:
                finalized_collection = copy.deepcopy(collection)
                finalized_collection["summary"]["safety"] = copy.deepcopy(
                    finalized_persisted["summary"]["safety"]
                )
                finalized_collection["summary"]["provenance"][
                    "scale_storage_contract"
                ] = copy.deepcopy(
                    finalized_persisted["summary"]["provenance"][
                        "scale_storage_contract"
                    ]
                )
            collection_manifest = artifact_root / "collection-manifest.json"
            commit_nested_collection_artifacts(
                finalized_persisted,
                artifact_files=artifact_files,
                manifest_path=collection_manifest,
                journal_path=journal.root,
                journal_report=journal_report,
                actor_attestation=actor_attestation,
            )
            return estimate_nested_collection_credit(
                finalized_collection,
                actor_checkpoint=actor,
                environment_manifest=manifest,
                environment_manifest_path=Path("data/environment.json"),
                environment_manifest_sha256=environment_manifest_sha256,
                active_branch_plan=plan,
                resolved_selections=resolved,
                stage1_records=stage1,
                stage1_source_sha256=scale_context["stage1_source_sha256"],
                stage1_records_path=scale_context["stage1_records_path"],
                stage1_manifest=scale_context["stage1_manifest_path"],
                sampling_backend_contract=backend,
                artifact_files=artifact_files,
                collection_manifest=collection_manifest,
            )

    def test_real_adapter_attests_source_but_mechanical_l4_stays_locked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection, plan, resolved, stage1, backend = self._artifacts(root)
            report = self._estimate(
                collection, root, plan, resolved, stage1, backend
            )

        self.assertTrue(report["source_attestation"]["attested"])
        self.assertTrue(report["training_gate"]["attested_collection"])
        self.assertNotIn(
            "source_collection_not_attested",
            report["training_gate"]["failed_checks"],
        )
        self.assertFalse(report["training_gate"]["training_ready"])
        self.assertFalse(report["training_gate"]["optimizer_unlock_allowed"])
        self.assertGreaterEqual(
            report["source_contract"]["duplicate_semantic_rollout_count"], 1
        )

    def test_adapter_recomputes_environment_manifest_file_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection, plan, resolved, stage1, backend = self._artifacts(root)

            with self.assertRaisesRegex(ValueError, "file SHA256 mismatch"):
                self._estimate(
                    collection,
                    root,
                    plan,
                    resolved,
                    stage1,
                    backend,
                    environment_manifest_sha256="a" * 64,
                )

    def test_adapter_rejects_tampered_stage1_completion_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection, plan, resolved, stage1, backend = self._artifacts(root)
            manifest_path = self._scale_contexts[str(root.resolve())][
                "stage1_manifest_path"
            ]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "incomplete"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "version or status"):
                self._estimate(collection, root, plan, resolved, stage1, backend)

    def test_policy_reward_is_recomputed_from_bound_public_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection, plan, resolved, stage1, backend = self._artifacts(root)
            tampered = copy.deepcopy(collection)
            continuation = tampered["continuations"][0]
            original_reward = float(continuation["policy_reward"])
            continuation["policy_reward"] = (
                original_reward - 0.125
                if original_reward >= 0.0
                else original_reward + 0.125
            )
            continuation.pop("rollout_content_sha256")
            continuation.update(_with_rollout_content_sha256(continuation))

            with self.assertRaisesRegex(ValueError, "reward recomputation mismatch"):
                self._estimate(tampered, root, plan, resolved, stage1, backend)

    def test_in_memory_collection_must_match_real_artifact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection, plan, resolved, stage1, backend = self._artifacts(root)
            tampered = copy.deepcopy(collection)
            tampered["continuations"][0]["strict"] = not tampered["continuations"][0][
                "strict"
            ]

            with self.assertRaisesRegex(ValueError, "differs from bound artifact bytes"):
                self._estimate(
                    tampered,
                    root,
                    plan,
                    resolved,
                    stage1,
                    backend,
                    persisted_collection=collection,
                )

    def test_decision_content_hash_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection, plan, resolved, stage1, backend = self._artifacts(root)
            tampered = copy.deepcopy(collection)
            tampered["decisions"][0]["decision_content_sha256"] = "0" * 64

            with self.assertRaisesRegex(ValueError, "decision content SHA256 mismatch"):
                self._estimate(tampered, root, plan, resolved, stage1, backend)

    def test_learning_invalid_flag_requires_real_infrastructure_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection, plan, resolved, stage1, backend = self._artifacts(root)
            tampered = copy.deepcopy(collection)
            continuation = tampered["continuations"][0]
            continuation["valid_for_learning"] = False
            continuation.pop("rollout_content_sha256")
            continuation.update(_with_rollout_content_sha256(continuation))

            with self.assertRaisesRegex(ValueError, "sampling-invalid flag mismatch"):
                self._estimate(tampered, root, plan, resolved, stage1, backend)

    def test_scale_gate_rejects_discarding_a_model_failure_as_sampling_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection, plan, resolved, stage1, backend = self._artifacts(root)
            tampered = copy.deepcopy(collection)
            continuation = tampered["continuations"][0]
            continuation.update(
                {
                    "valid_for_learning": False,
                    "sampling_invalid": True,
                    "model_failure": True,
                    "invalid_reason": "assistant_finished_without_environment_done",
                }
            )
            continuation.pop("rollout_content_sha256")
            continuation.update(_with_rollout_content_sha256(continuation))

            with self.assertRaisesRegex(
                ValueError, "aggregate is not collector-consistent"
            ):
                self._estimate(tampered, root, plan, resolved, stage1, backend)

    def test_attested_reward_unverifiable_is_excluded_from_q_not_rejected(self) -> None:
        class OneRewardInvalidEnvironment(FakeEnvironment):
            buy_count = 0

            def step(self, action):
                if action != "click[Buy Now]":
                    return super().step(action)
                self.actions.append(action)
                self.__class__.buy_count += 1
                detail = reward_detail()
                if self.__class__.buy_count == 1:
                    detail.update(
                        {
                            "reward_type": "reward_unverifiable",
                            "reward_valid": False,
                            "termination_reason": "reward_unverifiable",
                            "target_asin_match": False,
                            "terminal_utility": 0.0,
                            "purchase_success": False,
                            "sampling_invalid": True,
                        }
                    )
                    return {
                        "done": True,
                        "over": True,
                        "reward": 0.0,
                        "reward_detail": detail,
                    }
                return {
                    "done": True,
                    "over": True,
                    "reward": 1.0,
                    "reward_detail": detail,
                }

        resolved = [resolved_branch(task_id, [10, 20]) for task_id in (17, 18, 19)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            OneRewardInvalidEnvironment.buy_count = 0
            collection, plan, resolved, stage1, backend = self._artifacts(
                root,
                resolved_rows=resolved,
                continuations_per_decision=8,
                env_factory=OneRewardInvalidEnvironment,
            )
            report = self._estimate(collection, root, plan, resolved, stage1, backend)

        self.assertTrue(report["source_attestation"]["attested"])
        self.assertEqual(
            report["training_gate"]["observed"]["reward_invalid_continuations"], 1
        )
        affected = next(
            state for state in report["states"] if state["reward_invalid_continuations"]
        )
        self.assertFalse(affected["structural_gate"]["eligible_for_credit"])
        self.assertEqual(affected["decision_values"], [])

    def test_attested_boundary_failure_is_excluded_from_q_not_rejected(self) -> None:
        class BoundaryInvalidEnvironment(FakeEnvironment):
            def step(self, action):
                if (
                    self.task_id == 17
                    and action.startswith("click[")
                    and action != "click[Buy Now]"
                ):
                    raise AssertionError("forced action is unavailable")
                return super().step(action)

        resolved = [resolved_branch(task_id, [10, 20]) for task_id in range(17, 38)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection, plan, resolved, stage1, backend = self._artifacts(
                root,
                resolved_rows=resolved,
                continuations_per_decision=8,
                env_factory=BoundaryInvalidEnvironment,
            )
            self.assertTrue(
                any(
                    decision["post_action_boundary"] is None
                    for decision in collection["decisions"]
                )
            )
            report = self._estimate(
                collection, root, plan, resolved, stage1, backend
            )

        self.assertTrue(report["source_attestation"]["attested"])
        self.assertGreater(
            report["training_gate"]["observed"][
                "infrastructure_invalid_continuations"
            ],
            0,
        )
        affected = next(
            state
            for state in report["states"]
            if state["infrastructure_invalid_continuations"]
        )
        self.assertFalse(affected["structural_gate"]["eligible_for_credit"])
        self.assertEqual(affected["decision_values"], [])

    def test_exclusion_audit_is_recomputed_not_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection, plan, resolved, stage1, backend = self._artifacts(root)
            tampered = copy.deepcopy(collection)
            tampered["summary"]["exclusion_audit"]["reason_counts"] = {
                "wrong_purchase": 1
            }

            with self.assertRaisesRegex(
                ValueError, "outcome-blind stage-one recomputation"
            ):
                self._estimate(tampered, root, plan, resolved, stage1, backend)

    def test_fully_attested_formal_collection_can_unlock_optimizer(self) -> None:
        def make_two_decisions_per_state(records):
            buy_action = canonical_replay_action("buy_now", {})
            buy_action_sha = replay_action_sha256("buy_now", {})
            for record in records:
                if record["suffix_index"] % 2:
                    record["response_ids"][0] = 42
                    record["first_action"] = buy_action
                    record["first_action_sha256"] = buy_action_sha

        resolved = [
            resolved_branch(task_id, [10 + task_id, 20 + task_id])
            for task_id in range(17, 49)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection, plan, resolved, stage1, backend = self._artifacts(
                root,
                resolved_rows=resolved,
                suffixes_per_state=4,
                continuations_per_decision=8,
                stage1_mutator=make_two_decisions_per_state,
            )
            report = self._estimate(
                collection, root, plan, resolved, stage1, backend
            )

        gate = report["training_gate"]
        self.assertTrue(gate["attested_collection"])
        self.assertTrue(gate["statistical_candidate_ready"])
        self.assertTrue(gate["training_ready"])
        self.assertTrue(gate["optimizer_unlock_allowed"])
        self.assertEqual(gate["failed_checks"], [])

    def test_copied_slot_with_forged_uid_and_seed_fails_lease_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection, plan, resolved, stage1, backend = self._artifacts(root)
            tampered = copy.deepcopy(collection)
            decision = tampered["decisions"][0]
            copied = copy.deepcopy(tampered["continuations"][0])
            copied.pop("rollout_content_sha256")
            copied["continuation_index"] = 1
            copied["continuation_uid"] = continuation_uid(
                copied["state_uid"], copied["decision_uid"], 1
            )
            copied["continuation_seed_uid"] = continuation_seed_uid(
                copied["state_uid"], 1
            )
            copied["first_downstream_seed"] = continuation_turn_seed(
                copied["state_uid"], 1, 0
            )
            copied["downstream_request_seeds"] = [copied["first_downstream_seed"]]
            copied["completion_intent_uids"] = list(
                tampered["continuations"][1]["completion_intent_uids"]
            )
            copied = _with_rollout_content_sha256(copied)
            tampered["continuations"][1] = copied
            decision["continuation_uids"][1] = copied["continuation_uid"]

            with self.assertRaisesRegex(ValueError, "lease is not independent"):
                self._estimate(tampered, root, plan, resolved, stage1, backend)


class NestedCreditCliTests(unittest.TestCase):
    def _argv(self, root: Path) -> list[str]:
        json_paths = (
            "summary",
            "collection-manifest",
            "environment",
            "plan",
            "selection",
            "backend",
            "stage1-manifest",
        )
        for name in json_paths:
            (root / f"{name}.json").write_text("{}\n", encoding="utf-8")
        jsonl_paths = ("decisions", "continuations", "stage1", "input")
        for name in jsonl_paths:
            (root / f"{name}.jsonl").write_text("{}\n", encoding="utf-8")
        (root / "actor").mkdir()
        return [
            "--decisions",
            str(root / "decisions.jsonl"),
            "--continuations",
            str(root / "continuations.jsonl"),
            "--summary",
            str(root / "summary.json"),
            "--collection-manifest",
            str(root / "collection-manifest.json"),
            "--actor-checkpoint",
            str(root / "actor"),
            "--environment-manifest",
            str(root / "environment.json"),
            "--plan",
            str(root / "plan.json"),
            "--selection",
            str(root / "selection.json"),
            "--input",
            str(root / "input.jsonl"),
            "--stage1-source",
            str(root / "stage1.jsonl"),
            "--stage1-manifest",
            str(root / "stage1-manifest.json"),
            "--sampling-backend-contract",
            str(root / "backend.json"),
            "--output",
            str(root / "credit_report.json"),
        ]

    def test_not_ready_report_is_written_with_exit_zero(self) -> None:
        report = {
            "training_gate": {
                "training_ready": False,
                "structural_ready": False,
                "signal_ready": False,
                "failed_checks": ["formal_train_gate_cardinality_incomplete"],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self._argv(root)
            with (
                patch(
                    "scripts.estimate_nested_credit.resolve_pivotal_selection",
                    return_value=[],
                ),
                patch("scripts.estimate_nested_credit._records", return_value=[]),
                patch(
                    "scripts.estimate_nested_credit.estimate_nested_collection_credit",
                    return_value=report,
                ),
            ):
                self.assertEqual(estimate_credit_main(argv), 0)
            written = json.loads((root / "credit_report.json").read_text(encoding="utf-8"))
        self.assertFalse(written["training_gate"]["training_ready"])

    def test_invalid_contract_exits_nonzero_without_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self._argv(root)
            with (
                patch(
                    "scripts.estimate_nested_credit.resolve_pivotal_selection",
                    return_value=[],
                ),
                patch("scripts.estimate_nested_credit._records", return_value=[]),
                patch(
                    "scripts.estimate_nested_credit.estimate_nested_collection_credit",
                    side_effect=ValueError("actor mismatch"),
                ),
                self.assertRaisesRegex(SystemExit, "contract invalid"),
            ):
                estimate_credit_main(argv)
            self.assertFalse((root / "credit_report.json").exists())


class NestedCreditContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = _payload(
            [_state(0, [[0.0] * 8, [1.0] * 8], [0, 0, 1, 1])]
        )

    def test_rejects_unknown_outcome_field_instead_of_using_it_for_eligibility(self) -> None:
        tampered = copy.deepcopy(self.payload)
        tampered["states"][0]["mixed_strict"] = True

        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            validate_nested_samples(tampered)

    def test_rejects_boolean_task_identity(self) -> None:
        tampered = copy.deepcopy(self.payload)
        tampered["states"][0]["task_id"] = True

        with self.assertRaisesRegex(ValueError, "task_id"):
            validate_nested_samples(tampered)

    def test_rejects_proposal_identity_mismatch(self) -> None:
        tampered = copy.deepcopy(self.payload)
        tampered["states"][0]["proposals"][0]["proposal_uid"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "proposal_uid mismatch"):
            validate_nested_samples(tampered)

    def test_rejects_continuation_identity_mismatch(self) -> None:
        tampered = copy.deepcopy(self.payload)
        tampered["states"][0]["decisions"][0]["continuations"][0][
            "continuation_uid"
        ] = "0" * 64

        with self.assertRaisesRegex(ValueError, "continuation_uid mismatch"):
            validate_nested_samples(tampered)

    def test_rejects_continuation_cardinality_mismatch(self) -> None:
        tampered = copy.deepcopy(self.payload)
        tampered["states"][0]["decisions"][0]["continuations"].pop()

        with self.assertRaisesRegex(ValueError, "exactly 8 continuation slots"):
            validate_nested_samples(tampered)

    def test_rejects_orphan_decision_identity(self) -> None:
        tampered = copy.deepcopy(self.payload)
        tampered["states"][0]["proposals"][0]["decision_uid"] = _sha("orphan")

        with self.assertRaisesRegex(ValueError, "same identities"):
            validate_nested_samples(tampered)

    def test_model_failure_cannot_be_silently_marked_learning_invalid(self) -> None:
        tampered = copy.deepcopy(self.payload)
        continuation = tampered["states"][0]["decisions"][0]["continuations"][0]
        continuation.update(
            {
                "reward": None,
                "valid_for_learning": False,
                "infrastructure_invalid": False,
                "sampling_invalid": True,
                "model_failure": True,
                "invalid_reason": "assistant_finished",
            }
        )

        with self.assertRaisesRegex(ValueError, "must remain learning-valid"):
            validate_nested_samples(tampered)

    def test_rejects_non_finite_or_out_of_contract_reward(self) -> None:
        for reward in (float("nan"), 1.01):
            with self.subTest(reward=reward):
                tampered = copy.deepcopy(self.payload)
                tampered["states"][0]["decisions"][0]["continuations"][0][
                    "reward"
                ] = reward
                with self.assertRaisesRegex(ValueError, "reward"):
                    validate_nested_samples(tampered)


if __name__ == "__main__":
    unittest.main()
