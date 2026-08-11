import asyncio
import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.collect_first_decision_proposals import _git_binding, parse_args
from shopping_grpo.environment.observation import render_structured_observation
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS
from shopping_grpo.training.grpo import stage1_proposal as stage1_module
from shopping_grpo.training.grpo.active_branch import (
    ACTOR_PROMPT_TOKENS_VERSION,
    build_active_branch_plan,
    observation_policy_sha256,
)
from shopping_grpo.training.grpo.active_suffix import (
    build_sampling_backend_contract,
    sampling_backend_contract_sha256,
    sha256_actor_checkpoint,
    tool_schema_sha256,
)
from shopping_grpo.training.grpo.nested_continuation import (
    build_nested_harness_contract,
)
from shopping_grpo.training.grpo.nested_structure import (
    classify_nested_stage1_structure,
)
from shopping_grpo.training.grpo.pivotal_states import (
    branch_uid,
    observation_sha256,
    replay_action_sha256,
    replay_state_id,
    token_ids_sha256,
)
from shopping_grpo.training.grpo.stage1_proposal import (
    FIRST_DECISION_PROPOSAL_VERSION,
    FIRST_DECISION_SOURCE_PROVENANCE_VERSION,
    FirstDecisionStage1Collector,
    verify_first_decision_artifacts,
    write_first_decision_artifacts,
)

PRODUCT_ID = "123456789012"
TOKENIZER_SHA256 = "b" * 64
MANIFEST_SHA256 = "a" * 64


def _formal_source_provenance():
    return {
        "schema_version": FIRST_DECISION_SOURCE_PROVENANCE_VERSION,
        "git_sha": "c" * 40,
        "git_worktree_clean": True,
        "git_status_sha256": hashlib.sha256(b"").hexdigest(),
        "execution_mode": "formal",
        "dirty_source_override": False,
        "scale_ready": True,
    }


def _observation_state(page_type="search_home"):
    if page_type == "search_results":
        return {
            "observation_version": "shopping-observation-v2",
            "page_type": "search_results",
            "search_available": False,
            "actions": [PRODUCT_ID],
            "query": "mug",
            "normalized_query": "mug",
            "page": 1,
            "total_pages": 1,
            "total_results": 1,
            "rank_start": 1,
            "rank_end": 1,
            "products": [
                {
                    "rank": 1,
                    "asin": PRODUCT_ID,
                    "price": "10",
                    "brand": "B",
                    "category": "mug",
                    "key_attributes": [],
                    "title": "Mug",
                }
            ],
        }
    return {
        "observation_version": "shopping-observation-v2",
        "page_type": "search_home",
        "search_available": True,
        "actions": [],
    }


def _resolved_branch(task_id, prompt_tokens):
    initial = render_structured_observation(_observation_state())
    results = render_structured_observation(_observation_state("search_results"))
    initial_hash = observation_sha256(initial)
    results_hash = observation_sha256(results)
    query_hash = observation_sha256("public task")
    state_id = replay_state_id(
        task_id,
        [],
        initial_hash,
        observation_kind="raw_public_observation",
        environment_manifest_sha256=MANIFEST_SHA256,
        public_query_sha256=query_hash,
    )
    prompt_hash = token_ids_sha256(prompt_tokens)
    parent = branch_uid(state_id, prompt_hash, TOKENIZER_SHA256)
    source = {
        "input_index": 0,
        "input_sha256": "f" * 64,
        "path": "/audit.jsonl",
        "line": task_id,
        "global_step": 1,
        "generation_batch": 1,
        "uid": f"group-{task_id}",
        "trajectory_index": 0,
        "event_index": 0,
    }
    selected_event = {
        "decision_index": 0,
        "decision_kind": "environment_tool",
        "tool": "search_products",
        "parameters": {"query": "mug"},
        "replay_parameters": {"query": "mug"},
        "action_sha256": replay_action_sha256(
            "search_products", {"query": "mug"}
        ),
        "branch_uid": parent,
        "replay_state_id": state_id,
        "actor_prompt_sha256": prompt_hash,
        "tokenizer_contract_sha256": TOKENIZER_SHA256,
        "actor_prompt_tokens": {
            "version": ACTOR_PROMPT_TOKENS_VERSION,
            "sha256": prompt_hash,
            "count": len(prompt_tokens),
            "tokens": prompt_tokens,
        },
        "prefix_action_count": 0,
        "assistant_turn_id": 0,
        "raw_observation_sha256": initial_hash,
        "observation_sha256": initial_hash,
        "accepted": True,
        "repeated": False,
        "guard_reason": None,
        "error": None,
    }
    return {
        "selection": {
            "selection_index": task_id - 17,
            "task_id": task_id,
            "replay_state_id": state_id,
            "branch_uid": parent,
            "prefix_action_count": 0,
            "pivotal_labels": ["candidate_search"],
            "source": source,
        },
        "trajectory": {
            "task_id": task_id,
            "environment_manifest_sha256": MANIFEST_SHA256,
            "environment_version": "shopsimulator-environment-v2.1",
            "public_query_sha256": query_hash,
            "initial_public_observation_sha256": initial_hash,
            "replay_ledger": [
                {
                    "sequence": 0,
                    "tool": "search_products",
                    "parameters": {"query": "mug"},
                    "before_public_observation_sha256": initial_hash,
                    "after_public_observation_sha256": results_hash,
                    "done": False,
                }
            ],
            "decision_trace": [selected_event],
            "turn_spans": [
                {
                    "turn_id": 0,
                    "kind": "tool_call",
                    "tool_names": ["search_products"],
                    "tool_call_count": 1,
                    "credit_eligible": True,
                    "assistant_span": [0, 1],
                    "observation_span": [1, 2],
                }
            ],
            # These fields prove Stage 1 does not serialize or consume outcomes.
            "strict": True,
            "terminal_utility": 1.0,
        },
        "event": selected_event,
    }


def _decoding_and_backend(actor_sha256, **overrides):
    template = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "active_suffix_decoding_template.json"
    )
    decoding = json.loads(template.read_text(encoding="utf-8"))
    decoding.update(overrides)
    decoding["tool_schema_sha256"] = tool_schema_sha256(SHOP_TOOL_SCHEMAS)
    backend = build_sampling_backend_contract(
        server_version="0.25.1",
        served_model="shopping-agent",
        served_model_root_sha256=actor_sha256,
        actor_checkpoint_sha256=actor_sha256,
        max_model_len=decoding["context_window"],
        decoding_config=decoding,
    )
    decoding["sampling_backend_contract_sha256"] = (
        sampling_backend_contract_sha256(backend)
    )
    decoding["observation_policy_sha256"] = observation_policy_sha256(decoding)
    return decoding, backend


class _Parser:
    def __init__(self):
        self.stop_token_ids = []

    async def parse(self, token_ids, schemas):
        del schemas
        if token_ids == [41]:
            return [{"name": "open_product", "arguments": {"asin": PRODUCT_ID}}]
        if token_ids == [42]:
            return []
        if token_ids == [43]:
            return [
                {"name": "search_products", "arguments": {"query": "mug"}},
                {"name": "open_product", "arguments": {"asin": PRODUCT_ID}},
            ]
        if token_ids == [44]:
            return [{"name": "not_a_shop_tool", "arguments": {"value": 1}}]
        if token_ids == [45]:
            return [{"name": "search_products", "arguments": "{"}]
        if token_ids == [46]:
            return [{"name": "think", "arguments": {"thought": "compare"}}]
        if token_ids == [47]:
            return [{"name": "search_products", "arguments": {"query": "mug"}}]
        if token_ids == [48]:
            return [{"name": "not_a_shop_tool", "arguments": "{"}]
        if token_ids == [49]:
            raise RuntimeError("parser unavailable")
        if token_ids == [50]:
            return [{"name": "open_product", "arguments": {"asin": "bad"}}]
        return []


class _Client:
    timeout = 180

    def __init__(self, completions):
        self.completions = iter(completions)
        self.calls = []
        self.attestations = 0
        self.actor_runtime_bindings = []
        self.on_complete = None

    def attest_and_bind(self, **kwargs):
        self.attestations += 1
        self.attestation = kwargs
        return self

    def attest_and_bind_prehashed(self, *, actor_runtime_binding, **kwargs):
        self.actor_runtime_bindings.append(dict(actor_runtime_binding))
        return self.attest_and_bind(**kwargs)

    def complete(self, prompt_token_ids, *, seed):
        self.calls.append((list(prompt_token_ids), seed))
        completion = next(self.completions)
        if isinstance(completion, BaseException):
            raise completion
        tokens = list(completion)
        if self.on_complete is not None:
            self.on_complete(len(self.calls))
        return {
            "prompt_token_ids": list(prompt_token_ids),
            "token_ids": tokens,
            "old_logprobs": [-0.1] * len(tokens),
            "finish_reason": "stop",
            "stop_reason": None,
        }


class FirstDecisionStage1Tests(unittest.TestCase):
    def test_scale_cli_defaults_to_forty_states_and_four_proposals(self):
        args = parse_args(
            [
                "--plan",
                "plan.json",
                "--selection",
                "selection.json",
                "--input",
                "audit.jsonl",
                "--actor-checkpoint",
                "actor",
                "--sampling-backend-contract",
                "backend.json",
                "--served-model",
                "shopping-agent",
                "--output",
                "stage1.jsonl",
                "--manifest-output",
                "stage1-manifest.json",
            ]
        )
        self.assertEqual(args.expected_states, 40)
        self.assertEqual(args.proposals_per_state, 4)
        self.assertFalse(args.resume)
        self.assertFalse(args.mechanical)

    def test_formal_cli_rejects_dirty_override_and_mechanical_is_not_scale_ready(self):
        with self.assertRaisesRegex(ValueError, "restricted to --mechanical"):
            _git_binding(mechanical=False, allow_dirty=True)

        completed = [
            mock.Mock(stdout="c" * 40 + "\n"),
            mock.Mock(stdout=" M src/example.py\n"),
        ]
        with mock.patch(
            "scripts.collect_first_decision_proposals.subprocess.run",
            side_effect=completed,
        ):
            binding = _git_binding(mechanical=True, allow_dirty=True)

        self.assertEqual(binding["execution_mode"], "mechanical")
        self.assertTrue(binding["dirty_source_override"])
        self.assertFalse(binding["scale_ready"])

    def _fixture(
        self,
        root,
        completions,
        *,
        proposals_per_state=2,
        decoding_overrides=None,
        plan=None,
        resolved=None,
        actor=None,
        backend=None,
        source_provenance=None,
    ):
        if actor is None:
            actor = root / "actor"
            actor.mkdir()
            (actor / "weights.bin").write_bytes(b"weights")
        actor_sha = sha256_actor_checkpoint(actor)
        if resolved is None:
            resolved = [_resolved_branch(17, [10, 20])]
        if backend is None:
            decoding, backend = _decoding_and_backend(
                actor_sha, **(decoding_overrides or {})
            )
        else:
            decoding = plan["decoding_config"]
        if plan is None:
            plan = build_active_branch_plan(
                resolved,
                actor_checkpoint_sha256=actor_sha,
                decoding_config=decoding,
                seed=20260811,
                suffixes_per_state=proposals_per_state,
            )
        client = _Client(completions)
        collector = FirstDecisionStage1Collector(
            plan=plan,
            resolved_selections=resolved,
            actor_checkpoint=actor,
            actor_tokenizer_contract_sha256=TOKENIZER_SHA256,
            sampling_backend_contract=backend,
            completion_client=client,
            parser=_Parser(),
            tool_schemas=SHOP_TOOL_SCHEMAS,
            required_environment_version="shopsimulator-environment-v2.1",
            expected_states=len(resolved),
            proposals_per_state=proposals_per_state,
            source_provenance=(
                _formal_source_provenance()
                if source_provenance is None
                else source_provenance
            ),
        )
        return collector, client, plan, resolved, actor, backend

    def test_mechanical_collection_cannot_emit_a_scale_ready_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = {
                **_formal_source_provenance(),
                "execution_mode": "mechanical",
                "scale_ready": False,
            }
            collector, _, plan, *_ = self._fixture(
                root, [[41], [42]], source_provenance=source
            )
            result = asyncio.run(collector.collect(root / "journal"))
            records_path = root / "stage1.jsonl"
            manifest_path = root / "stage1-manifest.json"
            manifest = write_first_decision_artifacts(
                result,
                records_output=records_path,
                manifest_output=manifest_path,
            )
            verified = verify_first_decision_artifacts(
                plan=plan,
                records_path=records_path,
                manifest_path=manifest_path,
            )

        self.assertEqual(manifest["status"], "mechanical_complete")
        self.assertFalse(manifest["scale_ready"])
        self.assertFalse(verified["manifest"]["scale_ready"])

    def test_formal_scale_is_exactly_160_first_turn_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resolved = [
                _resolved_branch(task_id, [10, task_id])
                for task_id in range(17, 57)
            ]
            collector, client, plan, *_ = self._fixture(
                root,
                [[41]] * 160,
                proposals_per_state=4,
                resolved=resolved,
            )
            result = asyncio.run(collector.collect(root / "journal"))

        self.assertEqual(len(client.calls), 160)
        self.assertEqual(client.attestations, 2)
        self.assertEqual(result["summary"]["aggregate"]["states"], 40)
        self.assertEqual(
            result["summary"]["aggregate"]["pre_registered_proposals"], 160
        )
        self.assertEqual(
            len(
                classify_nested_stage1_structure(plan, result["records"])[
                    "eligible_state_uids"
                ]
            ),
            40,
        )

    def test_collects_only_first_decision_and_is_nested_structure_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector, client, plan, *_ = self._fixture(
                root, [[41], [41], [41], [41]], proposals_per_state=4
            )
            result = asyncio.run(collector.collect(root / "journal"))

        self.assertEqual(len(client.calls), 4)
        self.assertEqual(len({seed for _, seed in client.calls}), 4)
        self.assertEqual(client.attestations, 2)
        self.assertEqual(
            result["summary"]["provenance"]["actor_runtime_stat_checks"], 6
        )
        self.assertEqual(len(result["records"]), 4)
        self.assertEqual(
            collector.harness_contract,
            build_nested_harness_contract(
                plan["decoding_config"],
                policy_reward_sha256=collector.policy_reward_sha256,
            ),
        )
        self.assertEqual(
            classify_nested_stage1_structure(plan, result["records"])[
                "eligible_state_uids"
            ],
            [plan["groups"][0]["active_group_uid"]],
        )
        for record in result["records"]:
            self.assertEqual(
                record["stage1_proposal_version"], FIRST_DECISION_PROPOSAL_VERSION
            )
            self.assertEqual(record["response_ids"], [41])
            self.assertEqual(record["response_mask"], [1])
            self.assertEqual(record["first_action"]["tool"], "open_product")
            self.assertNotIn("strict", record)
            self.assertNotIn("terminal_utility", record)
            self.assertNotIn("policy_reward", record)

    def test_canonicalizes_every_first_decision_boundary_without_continuing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector, client, plan, *_ = self._fixture(
                root,
                [[42], [43], [44], [45], [46], [47], [48], [50]],
                proposals_per_state=8,
            )
            result = asyncio.run(collector.collect(root / "journal"))

        self.assertEqual(len(client.calls), 8)
        actions = [record["first_action"] for record in result["records"]]
        self.assertEqual(
            [action["tool"] for action in actions],
            [
                "assistant_final",
                "parallel_tool_calls",
                "not_a_shop_tool",
                "malformed_tool_arguments",
                "think",
                "search_products",
                "not_a_shop_tool",
                "open_product",
            ],
        )
        self.assertEqual(actions[1]["parameters"]["tools"], [
            "search_products",
            "open_product",
        ])
        self.assertEqual(actions[2]["parameters"], {"value": 1})
        self.assertEqual(actions[3]["parameters"], {"tool": "search_products"})
        self.assertEqual(actions[6]["parameters"], {})
        self.assertEqual(actions[7]["parameters"], {"asin": "bad"})
        self.assertEqual(
            classify_nested_stage1_structure(plan, result["records"])[
                "eligible_state_uids"
            ],
            [plan["groups"][0]["active_group_uid"]],
        )

    def test_harness_termination_is_a_first_decision_not_a_suffix_rollout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector, client, plan, *_ = self._fixture(
                root,
                [[41], [47]],
                decoding_overrides={"max_assistant_turns": 1},
            )
            result = asyncio.run(collector.collect(root / "journal"))

        self.assertEqual(len(client.calls), 2)
        for record in result["records"]:
            self.assertEqual(record["first_action"]["tool"], "harness_termination")
            self.assertEqual(record["harness_limit_reason"], "max_assistant_turns")
            self.assertFalse(record["first_action_credit_eligible"])
        self.assertEqual(
            classify_nested_stage1_structure(plan, result["records"])[
                "eligible_state_uids"
            ],
            [plan["groups"][0]["active_group_uid"]],
        )

    def test_zero_token_and_parser_failure_are_persisted_without_backfill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector, client, plan, *_ = self._fixture(root, [[], [49]])
            result = asyncio.run(collector.collect(root / "journal"))
            resumed, resumed_client, *_ = self._fixture(
                root,
                [],
                plan=plan,
                resolved=collector.resolved,
                actor=collector.actor_checkpoint,
                backend=collector.backend_contract,
            )
            resumed_result = asyncio.run(
                resumed.collect(root / "journal", resume=True)
            )

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(resumed_client.calls, [])
        self.assertTrue(result["records"][0]["zero_token_completion"])
        self.assertFalse(result["records"][0]["infrastructure_invalid"])
        self.assertTrue(result["records"][1]["infrastructure_invalid"])
        structure = classify_nested_stage1_structure(plan, result["records"])
        self.assertEqual(structure["eligible_state_uids"], [])
        self.assertEqual(len(structure["excluded_state_uids"]), 1)
        self.assertEqual(
            resumed_result["summary"]["aggregate"]["resumed_proposals"], 2
        )
        self.assertEqual(
            resumed_result["summary"]["aggregate"]["newly_sampled_proposals"], 0
        )

    def test_resume_never_reissues_an_attempt_interrupted_during_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector, client, plan, resolved, actor, backend = self._fixture(
                root, [[41], KeyboardInterrupt()]
            )
            with self.assertRaises(KeyboardInterrupt):
                asyncio.run(collector.collect(root / "journal"))
            first_slot = next((root / "journal" / "slots").glob("*.json"))
            first_slot_bytes = first_slot.read_bytes()

            resumed, resumed_client, *_ = self._fixture(
                root,
                [],
                plan=plan,
                resolved=resolved,
                actor=actor,
                backend=backend,
            )
            result = asyncio.run(resumed.collect(root / "journal", resume=True))
            unchanged_first_slot = first_slot.read_bytes()

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.attestations, 1)
        self.assertEqual(len(resumed_client.calls), 0)
        self.assertEqual(resumed_client.attestations, 2)
        self.assertEqual(unchanged_first_slot, first_slot_bytes)
        self.assertEqual(
            [record["response_ids"] for record in result["records"]],
            [[41], []],
        )
        self.assertEqual(
            result["records"][1]["infrastructure_error_code"],
            "interrupted_after_attempt_intent",
        )
        self.assertEqual(result["summary"]["aggregate"]["resumed_proposals"], 2)
        self.assertEqual(result["summary"]["aggregate"]["newly_sampled_proposals"], 0)

    def test_response_then_pre_commit_crash_is_closed_without_resampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector, client, plan, resolved, actor, backend = self._fixture(
                root, [[41], [42]]
            )
            original_commit = stage1_module._FirstDecisionJournal.commit

            def crash_before_second_entry(journal, spec, record, **kwargs):
                if spec["proposal_index"] == 1:
                    raise KeyboardInterrupt()
                return original_commit(journal, spec, record, **kwargs)

            with mock.patch.object(
                stage1_module._FirstDecisionJournal,
                "commit",
                new=crash_before_second_entry,
            ), self.assertRaises(KeyboardInterrupt):
                asyncio.run(collector.collect(root / "journal"))

            attempts_after_crash = list(
                (root / "journal" / "attempts").glob("*.json")
            )
            attempt_bytes_after_crash = {
                path.name: path.read_bytes() for path in attempts_after_crash
            }
            slots_after_crash = list((root / "journal" / "slots").glob("*.json"))
            resumed, resumed_client, *_ = self._fixture(
                root,
                [],
                plan=plan,
                resolved=resolved,
                actor=actor,
                backend=backend,
            )
            result = asyncio.run(resumed.collect(root / "journal", resume=True))
            attempt_bytes_after_resume = {
                path.name: path.read_bytes()
                for path in (root / "journal" / "attempts").glob("*.json")
            }

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(attempts_after_crash), 2)
        self.assertEqual(attempt_bytes_after_resume, attempt_bytes_after_crash)
        self.assertEqual(len(slots_after_crash), 1)
        self.assertEqual(resumed_client.calls, [])
        self.assertEqual(
            result["records"][1]["infrastructure_error_code"],
            "interrupted_after_attempt_intent",
        )
        self.assertEqual(
            result["records"][1]["actor_run_uid"],
            result["summary"]["journal_attestation"]["actor_run_chain"]["runs"][
                0
            ]["actor_run_uid"],
        )
        self.assertTrue(result["summary"]["aggregate"]["partition_complete"])
        self.assertEqual(
            classify_nested_stage1_structure(plan, result["records"])[
                "eligible_state_uids"
            ],
            [],
        )

    def test_actor_stat_drift_aborts_before_another_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector, client, *_ = self._fixture(root, [[41], [41]])
            weights = collector.actor_checkpoint / "weights.bin"
            client.on_complete = lambda call_count: (
                weights.write_bytes(b"mutated-weights") if call_count == 1 else None
            )
            with self.assertRaisesRegex(
                RuntimeError, "actor checkpoint changed during collection"
            ):
                asyncio.run(collector.collect(root / "journal"))
            completed_slots = list((root / "journal" / "slots").glob("*.json"))

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.attestations, 1)
        self.assertEqual(len(completed_slots), 1)

    def test_resume_rejects_a_tampered_journal_before_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector, _, plan, resolved, actor, backend = self._fixture(
                root, [[41], [41]]
            )
            asyncio.run(collector.collect(root / "journal"))
            slot = next((root / "journal" / "slots").glob("*.json"))
            entry = json.loads(slot.read_text(encoding="utf-8"))
            entry["record"]["response_ids"] = [99]
            slot.write_text(json.dumps(entry), encoding="utf-8")
            resumed, resumed_client, *_ = self._fixture(
                root,
                [],
                plan=plan,
                resolved=resolved,
                actor=actor,
                backend=backend,
            )
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                asyncio.run(resumed.collect(root / "journal", resume=True))

        self.assertEqual(resumed_client.calls, [])
        self.assertEqual(resumed_client.attestations, 0)

    def test_resume_closes_crashed_actor_run_without_resampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector, client, plan, resolved, actor, backend = self._fixture(
                root, [[41], [42]]
            )
            with mock.patch.object(
                stage1_module._FirstDecisionJournal,
                "finish_actor_run",
                side_effect=KeyboardInterrupt(),
            ), self.assertRaises(KeyboardInterrupt):
                asyncio.run(collector.collect(root / "journal"))
            slots_after_crash = list((root / "journal" / "slots").glob("*.json"))
            starts_after_crash = list(
                (root / "journal" / "actor_runs").glob("*-start.json")
            )
            ends_after_crash = list(
                (root / "journal" / "actor_runs").glob("*-end.json")
            )
            self.assertEqual(len(slots_after_crash), 2)
            self.assertEqual(len(starts_after_crash), 1)
            self.assertEqual(ends_after_crash, [])
            self.assertFalse((root / "journal" / "journal_complete.json").exists())

            resumed, resumed_client, *_ = self._fixture(
                root,
                [],
                plan=plan,
                resolved=resolved,
                actor=actor,
                backend=backend,
            )
            result = asyncio.run(resumed.collect(root / "journal", resume=True))
            records_path = root / "stage1.jsonl"
            manifest_path = root / "stage1-manifest.json"
            write_first_decision_artifacts(
                result,
                records_output=records_path,
                manifest_output=manifest_path,
            )
            verified = verify_first_decision_artifacts(
                plan=plan,
                records_path=records_path,
                manifest_path=manifest_path,
            )

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(resumed_client.calls, [])
        self.assertTrue(result["summary"]["aggregate"]["partition_complete"])
        chain = result["summary"]["journal_attestation"]["actor_run_chain"]
        self.assertEqual(chain["run_count"], 2)
        self.assertEqual(
            [run["closure"] for run in chain["runs"]],
            ["successor_start_continuity", "graceful_end"],
        )
        self.assertEqual(
            {record["actor_run_uid"] for record in result["records"]},
            {chain["runs"][0]["actor_run_uid"]},
        )
        self.assertEqual(verified["records"], result["records"])

    def test_resume_rejects_actor_stat_change_after_unclosed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector, _, plan, resolved, actor, backend = self._fixture(
                root, [[41], [42]]
            )
            with mock.patch.object(
                stage1_module._FirstDecisionJournal,
                "finish_actor_run",
                side_effect=KeyboardInterrupt(),
            ), self.assertRaises(KeyboardInterrupt):
                asyncio.run(collector.collect(root / "journal"))

            weights = actor / "weights.bin"
            before = weights.stat()
            os.utime(
                weights,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
            resumed, resumed_client, *_ = self._fixture(
                root,
                [],
                plan=plan,
                resolved=resolved,
                actor=actor,
                backend=backend,
            )
            with self.assertRaisesRegex(
                ValueError, "differs from the crashed predecessor"
            ):
                asyncio.run(resumed.collect(root / "journal", resume=True))

        self.assertEqual(resumed_client.calls, [])
        self.assertEqual(resumed_client.attestations, 0)

    def test_finalizer_rejects_rehashed_record_that_diverges_from_wal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector, _, *_ = self._fixture(root, [[41], [42]])
            result = asyncio.run(collector.collect(root / "journal"))
            tampered = copy.deepcopy(result)
            record = tampered["records"][0]
            record["response_ids"] = [99]
            record["stage1_record_content_sha256"] = (
                stage1_module._record_content_sha256(record)
            )

            with self.assertRaisesRegex(
                ValueError, "differs from the sampling journal"
            ):
                write_first_decision_artifacts(
                    tampered,
                    records_output=root / "stage1.jsonl",
                    manifest_output=root / "stage1-manifest.json",
                )

    def test_verifier_rejects_rehashed_records_despite_manifest_rehash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector, _, plan, *_ = self._fixture(root, [[41], [42]])
            result = asyncio.run(collector.collect(root / "journal"))
            records_path = root / "stage1.jsonl"
            manifest_path = root / "stage1-manifest.json"
            write_first_decision_artifacts(
                result,
                records_output=records_path,
                manifest_output=manifest_path,
            )
            records = [
                json.loads(line)
                for line in records_path.read_text(encoding="utf-8").splitlines()
            ]
            records[0]["response_ids"] = [99]
            records[0]["stage1_record_content_sha256"] = (
                stage1_module._record_content_sha256(records[0])
            )
            records_bytes = b"".join(
                (
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                ).encode("utf-8")
                for record in records
            )
            records_path.write_bytes(records_bytes)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["records_file_sha256"] = hashlib.sha256(
                records_bytes
            ).hexdigest()
            manifest["records_canonical_sha256"] = stage1_module._sha256_json(
                records
            )
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "differs from the sampling journal"
            ):
                verify_first_decision_artifacts(
                    plan=plan,
                    records_path=records_path,
                    manifest_path=manifest_path,
                )

    def test_verifier_binds_header_entries_and_complete_marker_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector, _, plan, *_ = self._fixture(root, [[41], [42]])
            result = asyncio.run(collector.collect(root / "journal"))
            records_path = root / "stage1.jsonl"
            manifest_path = root / "stage1-manifest.json"
            manifest = write_first_decision_artifacts(
                result,
                records_output=records_path,
                manifest_output=manifest_path,
            )
            wal = manifest["journal_wal"]
            targets = [
                root / "journal" / wal["header"]["relative_path"],
                root / "journal" / wal["attempts"][0]["relative_path"],
                root / "journal" / wal["entries"][0]["relative_path"],
                root / "journal" / wal["actor_runs"][0]["relative_path"],
                root / "journal" / wal["actor_runs"][1]["relative_path"],
                root / "journal" / wal["complete"]["relative_path"],
            ]
            for target in targets:
                with self.subTest(target=target.name):
                    original = target.read_bytes()
                    target.write_bytes(original + b" ")
                    with self.assertRaises(ValueError):
                        verify_first_decision_artifacts(
                            plan=plan,
                            records_path=records_path,
                            manifest_path=manifest_path,
                        )
                    target.write_bytes(original)

            verified = verify_first_decision_artifacts(
                plan=plan,
                records_path=records_path,
                manifest_path=manifest_path,
            )

        self.assertEqual(verified["records"], result["records"])

    def test_final_manifest_binds_journal_and_plain_nested_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector, _, plan, *_ = self._fixture(root, [[41], [42]])
            result = asyncio.run(collector.collect(root / "journal"))
            records_path = root / "stage1.jsonl"
            manifest_path = root / "stage1-manifest.json"
            manifest = write_first_decision_artifacts(
                result,
                records_output=records_path,
                manifest_output=manifest_path,
            )
            verified = verify_first_decision_artifacts(
                plan=plan,
                records_path=records_path,
                manifest_path=manifest_path,
                expected_source_provenance=_formal_source_provenance(),
            )
            records = [
                json.loads(line)
                for line in records_path.read_text(encoding="utf-8").splitlines()
            ]
            records_path.write_text(
                records_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "file SHA256 mismatch"):
                verify_first_decision_artifacts(
                    plan=plan,
                    records_path=records_path,
                    manifest_path=manifest_path,
                )

        self.assertEqual(
            classify_nested_stage1_structure(plan, records)["eligible_state_uids"],
            [plan["groups"][0]["active_group_uid"]],
        )
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["aggregate"]["completed_proposals"], 2)
        self.assertEqual(verified["records"], records)
        self.assertEqual(verified["manifest"], manifest)
        self.assertEqual(len(manifest["journal_contract_file_sha256"]), 64)
        self.assertEqual(len(manifest["journal_content_sha256"]), 64)
        self.assertEqual(
            manifest["journal_content_sha256"],
            manifest["journal_wal"]["journal_content_sha256"],
        )
        self.assertEqual(
            manifest["journal_wal"]["slot_uid_schedule"],
            [record["journal_slot_uid"] for record in records],
        )
        self.assertEqual(len(manifest["journal_wal"]["entries"]), 2)
        self.assertEqual(len(manifest["journal_wal"]["attempts"]), 2)
        self.assertEqual(len(manifest["journal_wal"]["actor_runs"]), 2)
        self.assertTrue(
            manifest["journal_wal"]["actor_run_chain"]["all_runs_closed"]
        )
        self.assertTrue(
            manifest["journal_wal"]["actor_run_chain"][
                "all_records_covered"
            ]
        )
        self.assertEqual(len(manifest["journal_wal"]["complete"]["sha256"]), 64)
        self.assertEqual(len(manifest["source_provenance_sha256"]), 64)
        self.assertEqual(
            manifest["provenance"]["actor_tokenizer_contract_sha256"],
            TOKENIZER_SHA256,
        )
        self.assertEqual(len(manifest["provenance"]["harness_contract_sha256"]), 64)
        self.assertTrue(manifest["safety"]["first_decision_only"])

    def test_final_manifest_resume_accepts_identical_uncommitted_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector, _, _, *_ = self._fixture(root, [[41], [42]])
            result = asyncio.run(collector.collect(root / "journal"))
            records_path = root / "stage1.jsonl"
            manifest_path = root / "stage1-manifest.json"
            first = write_first_decision_artifacts(
                result,
                records_output=records_path,
                manifest_output=manifest_path,
            )
            records_bytes = records_path.read_bytes()
            manifest_path.unlink()

            resumed = write_first_decision_artifacts(
                result,
                records_output=records_path,
                manifest_output=manifest_path,
            )
            resumed_records_bytes = records_path.read_bytes()

        self.assertEqual(first, resumed)
        self.assertEqual(resumed_records_bytes, records_bytes)

    def test_top_level_artifact_symlinks_are_rejected_before_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector, _, plan, *_ = self._fixture(root, [[41], [42]])
            result = asyncio.run(collector.collect(root / "journal"))
            records_path = root / "stage1.jsonl"
            manifest_path = root / "stage1-manifest.json"
            write_first_decision_artifacts(
                result,
                records_output=records_path,
                manifest_output=manifest_path,
            )
            real_records = root / "real-stage1.jsonl"
            records_path.rename(real_records)
            try:
                os.symlink(real_records, records_path)
            except OSError as exc:
                self.skipTest(f"symbolic links unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "records artifact.*symbolic"):
                verify_first_decision_artifacts(
                    plan=plan,
                    records_path=records_path,
                    manifest_path=manifest_path,
                )
            records_path.unlink()
            real_records.rename(records_path)

            real_manifest = root / "real-stage1-manifest.json"
            manifest_path.rename(real_manifest)
            os.symlink(real_manifest, manifest_path)
            with self.assertRaisesRegex(ValueError, "manifest artifact.*symbolic"):
                verify_first_decision_artifacts(
                    plan=plan,
                    records_path=records_path,
                    manifest_path=manifest_path,
                )
            manifest_path.unlink()

            linked_output = root / "linked-output.jsonl"
            target_output = root / "target-output.jsonl"
            target_output.write_text("target", encoding="utf-8")
            os.symlink(target_output, linked_output)
            with self.assertRaisesRegex(ValueError, "records output.*symbolic"):
                write_first_decision_artifacts(
                    result,
                    records_output=linked_output,
                    manifest_output=root / "unused-manifest.json",
                )

            linked_manifest = root / "linked-manifest.json"
            target_manifest = root / "target-manifest.json"
            target_manifest.write_text("target", encoding="utf-8")
            os.symlink(target_manifest, linked_manifest)
            with self.assertRaisesRegex(ValueError, "manifest output.*symbolic"):
                write_first_decision_artifacts(
                    result,
                    records_output=root / "unused-records.jsonl",
                    manifest_output=linked_manifest,
                )

    def test_verifier_lstats_top_level_paths_before_resolving(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collector, _, plan, *_ = self._fixture(root, [[41], [42]])
            result = asyncio.run(collector.collect(root / "journal"))
            records_path = root / "stage1.jsonl"
            manifest_path = root / "stage1-manifest.json"
            write_first_decision_artifacts(
                result,
                records_output=records_path,
                manifest_output=manifest_path,
            )
            real_lstat = os.lstat

            with mock.patch.object(
                stage1_module.os,
                "lstat",
                return_value=mock.Mock(st_mode=0o120777),
            ), self.assertRaisesRegex(ValueError, "records output.*symbolic"):
                write_first_decision_artifacts(
                    result,
                    records_output=root / "new-records.jsonl",
                    manifest_output=root / "new-manifest.json",
                )

            def output_manifest_is_link(path):
                if Path(path) == root / "new-manifest.json":
                    return mock.Mock(st_mode=0o120777)
                return real_lstat(path)

            with mock.patch.object(
                stage1_module.os,
                "lstat",
                side_effect=output_manifest_is_link,
            ), self.assertRaisesRegex(ValueError, "manifest output.*symbolic"):
                write_first_decision_artifacts(
                    result,
                    records_output=root / "new-records.jsonl",
                    manifest_output=root / "new-manifest.json",
                )

            def records_is_link(path):
                if Path(path) == records_path:
                    return mock.Mock(st_mode=0o120777)
                return real_lstat(path)

            with mock.patch.object(
                stage1_module.os, "lstat", side_effect=records_is_link
            ), self.assertRaisesRegex(ValueError, "records artifact.*symbolic"):
                verify_first_decision_artifacts(
                    plan=plan,
                    records_path=records_path,
                    manifest_path=manifest_path,
                )

            def manifest_is_link(path):
                if Path(path) == manifest_path:
                    return mock.Mock(st_mode=0o120777)
                return real_lstat(path)

            with mock.patch.object(
                stage1_module.os, "lstat", side_effect=manifest_is_link
            ), self.assertRaisesRegex(ValueError, "manifest artifact.*symbolic"):
                verify_first_decision_artifacts(
                    plan=plan,
                    records_path=records_path,
                    manifest_path=manifest_path,
                )


if __name__ == "__main__":
    unittest.main()
