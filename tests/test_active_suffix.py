import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

from scripts import collect_active_suffixes as collect_cli
from scripts.materialize_active_suffix_contract import (
    materialize_contracts,
)
from scripts.materialize_active_suffix_contract import (
    parse_args as parse_materialize_args,
)
from shopping_grpo.environment.observation import render_structured_observation
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS
from shopping_grpo.training.grpo.active_branch import (
    ACTOR_PROMPT_TOKENS_VERSION,
    build_active_branch_plan,
    observation_policy_sha256,
)
from shopping_grpo.training.grpo.active_suffix import (
    ActiveSuffixInfrastructureError,
    ActiveSuffixRunner,
    VllmTokenCompletionClient,
    _normalize_tool_calls,
    append_assistant_turn,
    append_tool_observation,
    attest_actor_checkpoint,
    build_sampling_backend_contract,
    generation_max_tokens,
    sampling_backend_contract_sha256,
    sha256_actor_checkpoint,
    summarize_active_suffix_group,
    tool_schema_sha256,
)
from shopping_grpo.training.grpo.pivotal_states import (
    branch_uid,
    canonical_replay_action,
    observation_sha256,
    replay_action_sha256,
    replay_state_id,
    token_ids_sha256,
)

PRODUCT_ID = "123456789012"
TOKENIZER_SHA256 = "b" * 64
MANIFEST_SHA256 = "a" * 64


def observation_state(page_type="search_home"):
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
    if page_type == "product_detail":
        return {
            "observation_version": "shopping-observation-v2",
            "page_type": "product_detail",
            "search_available": False,
            "actions": ["Buy Now"],
            "product": {
                "asin": PRODUCT_ID,
                "title": "Mug",
                "brand": "B",
                "category": "mug",
                "price": "10",
                "key_attributes": [],
            },
            "selected_options": {},
            "available_options": {},
        }
    return {
        "observation_version": "shopping-observation-v2",
        "page_type": "search_home",
        "search_available": True,
        "actions": [],
    }


def reward_detail():
    gate = {
        "status": "pass",
        "passed": True,
        "verifiable": True,
        "comparator": "eq",
        "source_field": "public",
    }
    return {
        "reward_version": "shopsimulator-reward-v3",
        "reward_type": "gold_purchase",
        "reward_valid": True,
        "termination_reason": "gold_purchase",
        "target_asin_match": True,
        "hard_gates": {"category": gate, "budget": gate},
        "weighted_score": 1.0,
        "evidence_coverage": 1.0,
        "dimension_scores": {
            "brand": 1.0,
            "model": 1.0,
            "core_functions": 1.0,
            "key_options": 1.0,
        },
        "terminal_utility": 1.0,
        "purchase_success": True,
        "sampling_invalid": False,
    }


def decoding_and_backend(actor_sha256):
    template = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "active_suffix_decoding_template.json"
    )
    decoding = json.loads(template.read_text(encoding="utf-8"))
    decoding["tool_schema_sha256"] = tool_schema_sha256(SHOP_TOOL_SCHEMAS)
    backend = build_sampling_backend_contract(
        server_version="0.25.1",
        served_model="shopping-agent",
        served_model_root_sha256=actor_sha256,
        actor_checkpoint_sha256=actor_sha256,
        max_model_len=decoding["context_window"],
        decoding_config=decoding,
    )
    decoding["sampling_backend_contract_sha256"] = sampling_backend_contract_sha256(
        backend
    )
    decoding["observation_policy_sha256"] = observation_policy_sha256(decoding)
    return decoding, backend


def resolved_branch(task_id, prompt_tokens):
    initial = render_structured_observation(observation_state())
    results = render_structured_observation(observation_state("search_results"))
    product = render_structured_observation(observation_state("product_detail"))
    initial_hash = observation_sha256(initial)
    results_hash = observation_sha256(results)
    product_hash = observation_sha256(product)
    query_hash = observation_sha256("public task")
    accepted = [{"tool": "search_products", "parameters": {"query": "mug"}}]
    state_id = replay_state_id(
        task_id,
        accepted,
        results_hash,
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
        "event_index": 1,
    }
    selected_event = {
        "decision_index": 1,
        "decision_kind": "environment_tool",
        "tool": "open_product",
        "parameters": {"asin": PRODUCT_ID},
        "replay_parameters": {"asin": PRODUCT_ID},
        "action_sha256": replay_action_sha256(
            "open_product", {"asin": PRODUCT_ID}
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
        "prefix_action_count": 1,
        "assistant_turn_id": 1,
        "raw_observation_sha256": results_hash,
        "observation_sha256": results_hash,
        "accepted": True,
        "repeated": False,
        "guard_reason": None,
        "error": None,
    }
    prior_event = {
        "decision_index": 0,
        "decision_kind": "environment_tool",
        "tool": "search_products",
        "parameters": {"query": "mug"},
        "replay_parameters": {"query": "mug"},
        "action_sha256": replay_action_sha256(
            "search_products", {"query": "mug"}
        ),
        "prefix_action_count": 0,
        "assistant_turn_id": 0,
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
            "prefix_action_count": 1,
            "pivotal_labels": ["candidate_open"],
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
                },
                {
                    "sequence": 1,
                    "tool": "open_product",
                    "parameters": {"asin": PRODUCT_ID},
                    "before_public_observation_sha256": results_hash,
                    "after_public_observation_sha256": product_hash,
                    "done": False,
                }
            ],
            "decision_trace": [prior_event, selected_event],
            "turn_spans": [
                {
                    "turn_id": 0,
                    "kind": "tool_call",
                    "tool_names": ["search_products"],
                    "tool_call_count": 1,
                    "credit_eligible": True,
                    "assistant_span": [0, 1],
                    "observation_span": [1, 2],
                },
                {
                    "turn_id": 1,
                    "kind": "tool_call",
                    "tool_names": ["open_product"],
                    "tool_call_count": 1,
                    "credit_eligible": True,
                    "assistant_span": [2, 3],
                    "observation_span": [3, 4],
                },
            ],
        },
        "event": selected_event,
    }


def resolved_initial_branch(task_id, prompt_tokens):
    initial = render_structured_observation(observation_state())
    results = render_structured_observation(observation_state("search_results"))
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
        },
        "event": selected_event,
    }


class FakeEnvironment:
    instances: ClassVar[list] = []

    def __init__(self):
        self.actions = []
        self.released = False
        self.timeout = 60
        self.__class__.instances.append(self)

    def reset(self, task_id):
        self.task_id = task_id
        return {
            "instruction": "public task",
            "observation_state": observation_state(),
            "environment_version": "shopsimulator-environment-v2.1",
            "environment_manifest_sha256": MANIFEST_SHA256,
            "goal": "must never be serialized",
        }

    def step(self, action):
        self.actions.append(action)
        if action == "search[mug]":
            return {
                "done": False,
                "reward": 0.0,
                "observation_state": observation_state("search_results"),
            }
        if action == f"click[{PRODUCT_ID}]":
            return {
                "done": False,
                "reward": 0.0,
                "observation_state": observation_state("product_detail"),
            }
        if action == "click[Buy Now]":
            return {
                "done": True,
                "over": True,
                "reward": 1.0,
                "reward_detail": reward_detail(),
                "goal": "must never be serialized",
            }
        raise AssertionError(action)

    def release(self):
        self.released = True


class FakeEncoder:
    tokenizer_contract_sha256 = TOKENIZER_SHA256

    @staticmethod
    def count_tokens(text):
        return max(1, len(str(text)) // 100)

    @staticmethod
    def encode_tool_observation(text):
        return [90, len(str(text)) % 89 + 1]


class FakeParser:
    stop_token_ids: ClassVar[list[int]] = []

    async def parse(self, token_ids, schemas):
        self.schemas = schemas
        if token_ids == [41]:
            return [{"name": "open_product", "arguments": {"asin": PRODUCT_ID}}]
        if token_ids == [42]:
            return [{"name": "buy_now", "arguments": {}}]
        return []


class FakePlanBoundClient:
    def __init__(self):
        self.calls = 0
        self.attested = False
        self.attestations = 0
        self.seeds = []
        self.timeout = 180

    def attest_and_bind(self, **kwargs):
        self.attested = True
        self.attestations += 1
        self.attestation = kwargs
        return self

    def complete(self, prompt_token_ids, *, seed):
        self.calls += 1
        self.seeds.append(seed)
        token = 41 if self.calls % 2 else 42
        return {
            "prompt_token_ids": list(prompt_token_ids),
            "token_ids": [token],
            "old_logprobs": [-0.1],
            "finish_reason": "stop",
            "stop_reason": None,
        }


class FakeImmediateEosClient(FakePlanBoundClient):
    def complete(self, prompt_token_ids, *, seed):
        self.calls += 1
        self.seeds.append(seed)
        return {
            "prompt_token_ids": list(prompt_token_ids),
            "token_ids": [],
            "old_logprobs": [],
            "finish_reason": "stop",
            "stop_reason": None,
        }


class FixedCompletionClient(FakePlanBoundClient):
    def __init__(self, token_ids):
        super().__init__()
        self.token_ids = list(token_ids)

    def complete(self, prompt_token_ids, *, seed):
        self.calls += 1
        self.seeds.append(seed)
        return {
            "prompt_token_ids": list(prompt_token_ids),
            "token_ids": list(self.token_ids),
            "old_logprobs": [-0.1] * len(self.token_ids),
            "finish_reason": "stop",
            "stop_reason": None,
        }


class FixedLengthEncoder(FakeEncoder):
    def __init__(self, observation_length):
        self.observation_length = int(observation_length)

    def encode_tool_observation(self, text):
        del text
        return [90] * self.observation_length


def with_response_prefix(resolved, response_tokens_before):
    value = copy.deepcopy(resolved)
    spans = value["trajectory"]["turn_spans"]
    spans[0]["observation_span"] = [1, response_tokens_before]
    spans[1]["assistant_span"] = [response_tokens_before, response_tokens_before + 1]
    spans[1]["observation_span"] = [response_tokens_before + 1, response_tokens_before + 2]
    return value


def with_guard_and_think_prefix(resolved):
    value = copy.deepcopy(resolved)
    selected = copy.deepcopy(value["event"])
    selected["decision_index"] = 4
    selected["assistant_turn_id"] = 4
    value["event"] = selected
    value["selection"]["source"]["event_index"] = 4
    initial_event = value["trajectory"]["decision_trace"][0]
    observation_hash = selected["observation_sha256"]
    think_parameters = {"reason": "inspect before buying"}
    guard_parameters = {}
    think_event = {
        "decision_index": 1,
        "decision_kind": "think",
        "tool": "think",
        "parameters": think_parameters,
        "replay_parameters": think_parameters,
        "action_sha256": replay_action_sha256("think", think_parameters),
        "prefix_action_count": 1,
        "assistant_turn_id": 1,
        "observation_sha256": observation_hash,
        "accepted": None,
        "repeated": False,
        "guard_reason": None,
        "error": None,
    }
    guard_event = {
        "decision_index": 2,
        "decision_kind": "environment_tool",
        "tool": "buy_now",
        "parameters": guard_parameters,
        "replay_parameters": guard_parameters,
        "action_sha256": replay_action_sha256("buy_now", guard_parameters),
        "prefix_action_count": 1,
        "assistant_turn_id": 2,
        "observation_sha256": observation_hash,
        "accepted": False,
        "repeated": False,
        "guard_reason": "click_not_in_previous_observation",
        "error": None,
    }
    repeated_guard = {**guard_event, "decision_index": 3, "assistant_turn_id": 3}
    repeated_guard["repeated"] = True
    value["trajectory"]["decision_trace"] = [
        initial_event,
        think_event,
        guard_event,
        repeated_guard,
        selected,
    ]
    value["trajectory"]["turn_spans"] = [
        {
            "turn_id": turn_id,
            "kind": "tool_call",
            "tool_names": ["search_products" if turn_id == 0 else "buy_now"],
            "tool_call_count": 1,
            "credit_eligible": True,
            "assistant_span": [turn_id * 2, turn_id * 2 + 1],
            "observation_span": [turn_id * 2 + 1, turn_id * 2 + 2],
        }
        for turn_id in range(5)
    ]
    value["trajectory"]["turn_spans"][1]["kind"] = "think"
    value["trajectory"]["turn_spans"][1]["tool_names"] = ["think"]
    value["trajectory"]["turn_spans"][4]["tool_names"] = ["open_product"]
    return value


def collect_single_state(resolved, client, *, encoder=None, parser=None):
    with tempfile.TemporaryDirectory() as tmp:
        actor = Path(tmp)
        (actor / "weights.bin").write_bytes(b"weights")
        actor_sha = sha256_actor_checkpoint(actor)
        decoding, backend = decoding_and_backend(actor_sha)
        plan = build_active_branch_plan(
            [resolved],
            actor_checkpoint_sha256=actor_sha,
            decoding_config=decoding,
            seed=20260811,
            suffixes_per_state=2,
        )
        runner = ActiveSuffixRunner(
            plan=plan,
            resolved_selections=[resolved],
            actor_checkpoint=actor,
            sampling_backend_contract=backend,
            completion_client=client,
            parser=parser or FakeParser(),
            encoder=encoder or FakeEncoder(),
            env_factory=FakeEnvironment,
            expected_groups=1,
            expected_suffixes_per_state=2,
        )
        return asyncio.run(runner.collect())


class ActiveSuffixPrimitiveTest(unittest.TestCase):
    def test_collection_cli_uses_separate_vllm_and_environment_timeouts(self):
        args = collect_cli.parse_args(
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
                "suffixes.jsonl",
                "--summary-output",
                "summary.json",
            ]
        )
        self.assertEqual(args.vllm_timeout, 180)
        self.assertEqual(args.environment_timeout, 60)

    def test_collection_cli_writes_invalid_artifacts_then_exits_nonzero(self):
        collection = {
            "schema_version": "shopping-active-suffix-collection-v1",
            "records": [{"infrastructure_invalid": True}],
            "groups": [],
            "aggregate": {"mechanical_smoke_passed": False},
            "safety": {},
        }

        async def fake_run(args):
            del args
            return collection

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "records.jsonl"
            summary = Path(tmp) / "summary.json"
            args = SimpleNamespace(output=output, summary_output=summary)
            with (
                patch.object(collect_cli, "parse_args", return_value=args),
                patch.object(collect_cli, "_run", new=fake_run),
                self.assertRaises(SystemExit) as raised,
            ):
                collect_cli.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertTrue(output.is_file())
            self.assertTrue(summary.is_file())

    def test_materializer_cli_defaults_to_repository_template(self):
        args = parse_materialize_args(
            [
                "--actor-checkpoint",
                "actor",
                "--served-model",
                "shopping-agent",
                "--backend-output",
                "backend.json",
                "--decoding-output",
                "decoding.json",
            ]
        )
        repo = Path(__file__).resolve().parents[1]
        self.assertEqual(
            args.decoding_template,
            repo / "configs" / "active_suffix_decoding_template.json",
        )

    def test_actor_checkpoint_digest_rejects_file_and_directory_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "weights.bin").write_bytes(b"weights")
            digest = sha256_actor_checkpoint(root)
            self.assertEqual(attest_actor_checkpoint(root, digest), digest)
            (root / "weights.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ActiveSuffixInfrastructureError, "SHA256 mismatch"):
                attest_actor_checkpoint(root, digest)
            actor_link = root.parent / f"{root.name}-actor-link"
            nested_target = root.parent / f"{root.name}-nested-target"
            nested_target.mkdir()
            try:
                actor_link.symlink_to(root, target_is_directory=True)
                (root / "nested-link").symlink_to(
                    nested_target,
                    target_is_directory=True,
                )
            except OSError:
                self.skipTest("directory symlinks are unavailable on this platform")
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                sha256_actor_checkpoint(actor_link)
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                sha256_actor_checkpoint(root)

    def test_vllm_client_requires_plan_binding_prompt_echo_and_logprobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            actor = Path(tmp)
            (actor / "weights.bin").write_bytes(b"weights")
            actor_sha = sha256_actor_checkpoint(actor)
            decoding, backend = decoding_and_backend(actor_sha)
            captured = {}

            def transport(url, payload, headers, timeout):
                captured.update({"url": url, "payload": payload})
                return {
                    "model": "shopping-agent",
                    "choices": [
                        {
                            "prompt_token_ids": list(payload["prompt"]),
                            "token_ids": [41, 42],
                            "logprobs": {"token_logprobs": [-0.1, -0.2]},
                            "finish_reason": "stop",
                            "stop_reason": 7,
                        }
                    ]
                }

            def metadata(url, headers, timeout):
                if url.endswith("/version"):
                    return {"version": "0.25.1"}
                return {
                    "data": [
                        {
                            "id": "shopping-agent",
                            "root": str(actor.resolve()),
                            "max_model_len": decoding["context_window"],
                        }
                    ]
                }

            client = VllmTokenCompletionClient(
                "shopping-agent",
                "http://127.0.0.1:8000/v1",
                "EMPTY",
                transport=transport,
                metadata_transport=metadata,
            )
            with self.assertRaisesRegex(ActiveSuffixInfrastructureError, "not plan-bound"):
                client.complete([11], seed=1)
            materialized = client.materialize_backend_contract(
                actor_checkpoint=actor,
                actor_checkpoint_sha256=actor_sha,
                decoding_config=decoding,
            )
            self.assertEqual(materialized, backend)
            bound = client.attest_and_bind(
                plan_sha256="1" * 64,
                expected_contract=backend,
                expected_contract_sha256=sampling_backend_contract_sha256(backend),
                decoding_config=decoding,
                actor_checkpoint=actor,
                actor_checkpoint_sha256=actor_sha,
            )
            completion = bound.complete([11, 12, 13], seed=3407)
            self.assertEqual(captured["payload"]["prompt"], [11, 12, 13])
            self.assertFalse(captured["payload"]["add_special_tokens"])
            self.assertTrue(captured["payload"]["return_token_ids"])
            self.assertEqual(
                captured["payload"]["max_tokens"],
                generation_max_tokens(decoding, 3),
            )
            self.assertEqual(
                backend["effective_request"]["seed_schedule"],
                "first-suffix-then-sha256-turn-v1",
            )
            self.assertNotIn("return_prompt_token_ids", captured["payload"])
            self.assertNotIn(
                "return_prompt_token_ids", backend["effective_request"]
            )
            self.assertEqual(completion["token_ids"], [41, 42])
            self.assertEqual(completion["old_logprobs"], [-0.1, -0.2])

    def test_repository_template_materializes_from_live_backend_metadata(self):
        repo = Path(__file__).resolve().parents[1]
        template = json.loads(
            (repo / "configs" / "active_suffix_decoding_template.json").read_text(
                encoding="utf-8"
            )
        )
        tools = json.loads(
            (repo / "configs" / "tools.json").read_text(encoding="utf-8")
        )
        schemas = [item["tool_schema"] for item in tools["tools"]]
        with tempfile.TemporaryDirectory() as tmp:
            actor = Path(tmp)
            (actor / "weights.bin").write_bytes(b"weights")

            def metadata(url, headers, timeout):
                if url.endswith("/version"):
                    return {"version": "0.25.1"}
                return {
                    "data": [
                        {
                            "id": "shopping-agent",
                            "root": str(actor.resolve()),
                            "max_model_len": template["context_window"],
                        }
                    ]
                }

            backend, decoding, actor_sha = materialize_contracts(
                decoding_template=template,
                tool_schemas=schemas,
                parser_stop_token_ids=[],
                actor_checkpoint=actor,
                completion_client=VllmTokenCompletionClient(
                    "shopping-agent",
                    "http://127.0.0.1:8000/v1",
                    "EMPTY",
                    metadata_transport=metadata,
                ),
            )
        self.assertEqual(backend["actor_checkpoint_sha256"], actor_sha)
        self.assertEqual(
            decoding["tool_schema_sha256"], tool_schema_sha256(schemas)
        )
        self.assertEqual(
            decoding["sampling_backend_contract_sha256"],
            sampling_backend_contract_sha256(backend),
        )
        self.assertNotEqual(decoding["observation_policy_sha256"], "0" * 64)
        self.assertNotIn("return_prompt_token_ids", backend["effective_request"])

    def test_non_object_json_arguments_follow_shopping_tool_normalization(self):
        for raw in ("[]", "null", '"scalar"', "7"):
            [call] = _normalize_tool_calls(
                [{"name": "buy_now", "arguments": raw}]
            )
            self.assertEqual(call["arguments"], {})
            self.assertIsNone(call["arguments_error"])

    def test_empty_completion_is_returned_as_immediate_eos_not_infrastructure(self):
        client = VllmTokenCompletionClient(
            "m",
            "http://localhost:1",
            "EMPTY",
            transport=lambda url, payload, headers, timeout: {
                "model": "m",
                "choices": [
                    {
                        "prompt_token_ids": list(payload["prompt"]),
                        "token_ids": [],
                        "logprobs": {"token_logprobs": []},
                        "finish_reason": "stop",
                        "stop_reason": None,
                    }
                ]
            },
        )
        client._bindings.add("bound")
        result = client._complete_bound(
            "bound",
            [1],
            seed=1,
            temperature=0.7,
            top_p=0.9,
            top_k=-1,
            min_p=0.0,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            max_tokens=8,
            min_tokens=0,
            stop=[],
            stop_token_ids=[],
            ignore_eos=False,
        )
        self.assertEqual(result["token_ids"], [])
        self.assertEqual(result["old_logprobs"], [])

    def test_tensor_spans_and_strict_group_gate(self):
        response_ids, response_mask, old_logprobs = [], [], []
        self.assertEqual(
            append_assistant_turn(
                response_ids,
                response_mask,
                old_logprobs,
                {"token_ids": [1, 2], "old_logprobs": [-0.1, -0.2]},
            ),
            (0, 2),
        )
        append_tool_observation(response_ids, response_mask, old_logprobs, [3, 4])
        self.assertEqual(response_mask, [1, 1, 0, 0])
        common = {
            "active_group_uid": "1" * 64,
            "parent_branch_uid": "2" * 64,
            "replay_state_id": "3" * 64,
            "task_id": 1,
            "actor_checkpoint_sha256": "4" * 64,
            "decoding_config_sha256": "5" * 64,
            "sampling_backend_contract_sha256": "6" * 64,
            "response_ids": [1],
            "response_mask": [1],
            "old_logprobs": [-0.1],
            "valid_for_learning": True,
            "first_action_credit_eligible": True,
            "first_action_span": [0, 1],
            "infrastructure_invalid": False,
            "infrastructure_error_code": None,
            "model_failure": False,
            "request_prompt_sha256": ["a" * 64],
        }
        first_action_a = canonical_replay_action("open_product", {"asin": PRODUCT_ID})
        first_action_b = canonical_replay_action("think", {"reason": "inspect"})
        summary = summarize_active_suffix_group(
            [
                {
                    **common,
                    "suffix_index": 0,
                    "suffix_uid": "7" * 64,
                    "seed": 1,
                    "request_seeds": [1],
                    "strict": True,
                    "first_action": first_action_a,
                    "first_action_sha256": replay_action_sha256(
                        first_action_a["tool"], first_action_a["parameters"]
                    ),
                },
                {
                    **common,
                    "suffix_index": 1,
                    "suffix_uid": "9" * 64,
                    "seed": 2,
                    "request_seeds": [2],
                    "strict": False,
                    "first_action": first_action_b,
                    "first_action_sha256": replay_action_sha256(
                        first_action_b["tool"], first_action_b["parameters"]
                    ),
                },
            ],
            expected_group_uid="1" * 64,
            expected_k=2,
        )
        self.assertTrue(summary["eligible_for_action_credit"])
        tampered = {
            **common,
            "suffix_index": 0,
            "suffix_uid": "7" * 64,
            "seed": 1,
            "request_seeds": [2],
            "strict": True,
            "first_action": first_action_a,
            "first_action_sha256": replay_action_sha256(
                first_action_a["tool"], first_action_a["parameters"]
            ),
        }
        with self.assertRaisesRegex(ValueError, "request seed schedule"):
            summarize_active_suffix_group([tampered])
        broken = {**common, "suffix_index": 0, "suffix_uid": "7" * 64, "seed": 1}
        broken["request_seeds"] = [1]
        broken["strict"] = 1
        broken["first_action"] = first_action_a
        broken["first_action_sha256"] = "8" * 64
        with self.assertRaisesRegex(TypeError, "strict must be boolean"):
            summarize_active_suffix_group([broken])


class ActiveSuffixRunnerTest(unittest.TestCase):
    def test_runner_rejects_vllm_timeout_contract_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            actor = Path(tmp)
            (actor / "weights.bin").write_bytes(b"weights")
            actor_sha = sha256_actor_checkpoint(actor)
            decoding, backend = decoding_and_backend(actor_sha)
            resolved = [resolved_branch(17, [10])]
            plan = build_active_branch_plan(
                resolved,
                actor_checkpoint_sha256=actor_sha,
                decoding_config=decoding,
                seed=1,
                suffixes_per_state=2,
            )
            client = FakePlanBoundClient()
            client.timeout = 181
            with self.assertRaisesRegex(ValueError, "client timeout"):
                ActiveSuffixRunner(
                    plan=plan,
                    resolved_selections=resolved,
                    actor_checkpoint=actor,
                    sampling_backend_contract=backend,
                    completion_client=client,
                    parser=FakeParser(),
                    encoder=FakeEncoder(),
                    env_factory=FakeEnvironment,
                    expected_groups=1,
                    expected_suffixes_per_state=2,
                )

    def test_fake_runner_replays_each_suffix_and_writes_no_hidden_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            actor = Path(tmp)
            (actor / "weights.bin").write_bytes(b"weights")
            actor_sha = sha256_actor_checkpoint(actor)
            decoding, backend = decoding_and_backend(actor_sha)
            resolved = [resolved_branch(17, [10, 20]), resolved_branch(18, [30, 40])]
            plan = build_active_branch_plan(
                resolved,
                actor_checkpoint_sha256=actor_sha,
                decoding_config=decoding,
                seed=20260811,
                suffixes_per_state=2,
            )
            FakeEnvironment.instances = []
            client = FakePlanBoundClient()
            runner = ActiveSuffixRunner(
                plan=plan,
                resolved_selections=resolved,
                actor_checkpoint=actor,
                sampling_backend_contract=backend,
                completion_client=client,
                parser=FakeParser(),
                encoder=FakeEncoder(),
                env_factory=FakeEnvironment,
                expected_groups=2,
                expected_suffixes_per_state=2,
            )
            result = asyncio.run(runner.collect())
            self.assertTrue(client.attested)
            self.assertEqual(client.attestations, 5)
            self.assertEqual(result["aggregate"]["suffixes"], 4)
            self.assertEqual(result["aggregate"]["valid_suffixes"], 4)
            self.assertTrue(result["aggregate"]["mechanical_smoke_passed"])
            self.assertEqual(
                result["provenance"]["required_environment_version"],
                "shopsimulator-environment-v2.1",
            )
            self.assertEqual(
                result["provenance"]["environment_manifest_sha256s"],
                [MANIFEST_SHA256],
            )
            self.assertEqual(len(result["provenance"]["policy_reward_sha256"]), 64)
            self.assertEqual(result["provenance"]["vllm_timeout_seconds"], 180)
            self.assertEqual(result["provenance"]["environment_timeout_seconds"], 60)
            self.assertFalse(result["safety"]["generation_config_api_attested"])
            self.assertEqual(len(FakeEnvironment.instances), 4)
            self.assertTrue(all(env.released for env in FakeEnvironment.instances))
            self.assertTrue(all(record["strict"] for record in result["records"]))
            self.assertTrue(
                all(
                    record["request_seeds"][0] == record["seed"]
                    and len(record["request_seeds"])
                    == len(record["request_prompt_sha256"])
                    and len(record["request_seeds"])
                    == len(set(record["request_seeds"]))
                    for record in result["records"]
                )
            )
            self.assertTrue(
                all(record["response_mask"] == [1, 0, 0, 1, 0, 0] for record in result["records"])
            )
            self.assertNotIn("must never be serialized", json.dumps(result))

    def test_immediate_eos_after_prefix_is_noninfra_untrainable_model_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            actor = Path(tmp)
            (actor / "weights.bin").write_bytes(b"weights")
            actor_sha = sha256_actor_checkpoint(actor)
            decoding, backend = decoding_and_backend(actor_sha)
            resolved = [resolved_branch(17, [10]), resolved_branch(18, [20])]
            plan = build_active_branch_plan(
                resolved,
                actor_checkpoint_sha256=actor_sha,
                decoding_config=decoding,
                seed=1,
                suffixes_per_state=2,
            )
            result = asyncio.run(
                ActiveSuffixRunner(
                    plan=plan,
                    resolved_selections=resolved,
                    actor_checkpoint=actor,
                    sampling_backend_contract=backend,
                    completion_client=FakeImmediateEosClient(),
                    parser=FakeParser(),
                    encoder=FakeEncoder(),
                    env_factory=FakeEnvironment,
                    expected_groups=2,
                    expected_suffixes_per_state=2,
                ).collect()
            )
            for record in result["records"]:
                self.assertTrue(record["model_failure"])
                self.assertFalse(record["infrastructure_invalid"])
                self.assertFalse(record["valid_for_learning"])
                self.assertEqual(record["invalid_reason"], "empty_suffix_no_trainable_tokens")
                self.assertEqual(
                    record["termination_reason"],
                    "assistant_finished_without_environment_done",
                )
                self.assertIsNone(record["infrastructure_error_code"])
                self.assertIsNone(record["first_action"])
                self.assertIsNone(record["first_action_span"])
                self.assertFalse(record["first_action_credit_eligible"])
                self.assertEqual(record["response_ids"], [])
                self.assertEqual(record["response_tokens_before"], 2)
                self.assertEqual(record["response_tokens_after"], 2)
            self.assertTrue(result["aggregate"]["mechanical_smoke_passed"])
            self.assertEqual(result["aggregate"]["valid_suffixes"], 0)

    def test_immediate_eos_with_empty_full_response_is_alignment_infrastructure(self):
        result = collect_single_state(
            resolved_initial_branch(17, [10]),
            FakeImmediateEosClient(),
        )
        for record in result["records"]:
            self.assertTrue(record["infrastructure_invalid"])
            self.assertFalse(record["valid_for_learning"])
            self.assertEqual(
                record["infrastructure_error_code"],
                "trajectory_alignment_invalid",
            )
        self.assertFalse(result["aggregate"]["mechanical_smoke_passed"])

    def test_external_exception_text_is_not_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            actor = Path(tmp)
            (actor / "weights.bin").write_bytes(b"weights")
            actor_sha = sha256_actor_checkpoint(actor)
            decoding, backend = decoding_and_backend(actor_sha)
            resolved = [resolved_branch(17, [10]), resolved_branch(18, [20])]
            plan = build_active_branch_plan(
                resolved,
                actor_checkpoint_sha256=actor_sha,
                decoding_config=decoding,
                seed=1,
                suffixes_per_state=2,
            )

            def poisoned_environment():
                raise RuntimeError("hidden_goal=must-never-reach-the-artifact")

            result = asyncio.run(
                ActiveSuffixRunner(
                    plan=plan,
                    resolved_selections=resolved,
                    actor_checkpoint=actor,
                    sampling_backend_contract=backend,
                    completion_client=FakePlanBoundClient(),
                    parser=FakeParser(),
                    encoder=FakeEncoder(),
                    env_factory=poisoned_environment,
                    expected_groups=2,
                    expected_suffixes_per_state=2,
                ).collect()
            )
        serialized = json.dumps(result)
        self.assertNotIn("must-never-reach-the-artifact", serialized)
        self.assertTrue(
            all(
                record["invalid_reason"] == "external_failure"
                and record["infrastructure_error_code"] == "external_failure"
                and record["infrastructure_error_class"] == "RuntimeError"
                for record in result["records"]
            )
        )

    def test_response_prefix_budget_uses_verl_greater_equal_boundary(self):
        resolved = with_response_prefix(resolved_branch(17, [10]), 20289)
        result = collect_single_state(
            resolved,
            FixedCompletionClient([41]),
            encoder=FixedLengthEncoder(190),
        )
        for record in result["records"]:
            self.assertEqual(record["response_tokens_before"], 20289)
            self.assertEqual(record["response_ids"], [41])
            self.assertEqual(record["response_mask"], [1])
            self.assertEqual(record["harness_limit_reason"], "response_length")
            self.assertFalse(record["response_truncated"])
            self.assertTrue(record["first_action_credit_eligible"])

    def test_generation_overflow_is_sliced_and_not_action_creditable(self):
        resolved = with_response_prefix(resolved_branch(17, [10]), 20289)
        result = collect_single_state(
            resolved,
            FixedCompletionClient([60] * 200),
        )
        for record in result["records"]:
            self.assertEqual(len(record["response_ids"]), 191)
            self.assertEqual(record["response_tokens_after"], 20480)
            self.assertTrue(record["response_truncated"])
            self.assertFalse(record["first_action_credit_eligible"])
            self.assertEqual(record["first_action_span"], [0, 191])

    def test_prefix_runtime_state_restores_guard_repeat_and_think_counters(self):
        resolved = with_guard_and_think_prefix(resolved_branch(17, [10]))
        result = collect_single_state(
            resolved,
            FixedCompletionClient([42]),
        )
        for record in result["records"]:
            self.assertEqual(record["termination_reason"], "too_many_guard_rejections")
            self.assertEqual(record["steps"], 2)
            self.assertEqual(record["guard_rejections"], 3)
            self.assertEqual(record["repeat_actions"], 2)
            self.assertEqual(record["assistant_turns"], 5)
            self.assertTrue(record["valid_for_learning"])
            self.assertFalse(record["infrastructure_invalid"])

    def test_response_cap_does_not_overwrite_existing_guard_termination(self):
        resolved = with_guard_and_think_prefix(resolved_branch(17, [10]))
        spans = resolved["trajectory"]["turn_spans"]
        spans[3]["observation_span"] = [7, 20477]
        spans[4]["assistant_span"] = [20477, 20478]
        spans[4]["observation_span"] = [20478, 20479]
        result = collect_single_state(
            resolved,
            FixedCompletionClient([42]),
            encoder=FixedLengthEncoder(2),
        )
        for record in result["records"]:
            self.assertEqual(record["termination_reason"], "too_many_guard_rejections")
            self.assertEqual(record["harness_limit_reason"], "response_length")
            self.assertEqual(record["guard_rejections"], 3)
            self.assertEqual(record["policy_reward"], -0.7)

    def test_prefix_runtime_marker_drift_fails_closed(self):
        resolved = with_guard_and_think_prefix(resolved_branch(17, [10]))
        resolved["trajectory"]["decision_trace"][3]["repeated"] = False
        result = collect_single_state(
            resolved,
            FixedCompletionClient([42]),
        )
        self.assertFalse(result["aggregate"]["mechanical_smoke_passed"])
        self.assertTrue(
            all(
                record["infrastructure_error_code"]
                == "prefix_runtime_state_invalid"
                for record in result["records"]
            )
        )

    def test_plan_or_prompt_drift_fails_before_environment_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            actor = Path(tmp)
            (actor / "weights.bin").write_bytes(b"weights")
            actor_sha = sha256_actor_checkpoint(actor)
            decoding, backend = decoding_and_backend(actor_sha)
            resolved = [resolved_branch(17, [10]), resolved_branch(18, [20])]
            plan = build_active_branch_plan(
                resolved,
                actor_checkpoint_sha256=actor_sha,
                decoding_config=decoding,
                seed=1,
                suffixes_per_state=2,
            )
            plan = copy.deepcopy(plan)
            plan["groups"][0]["actor_prompt_tokens"]["tokens"] = [11]
            with self.assertRaises((ValueError, ActiveSuffixInfrastructureError)):
                ActiveSuffixRunner(
                    plan=plan,
                    resolved_selections=resolved,
                    actor_checkpoint=actor,
                    sampling_backend_contract=backend,
                    completion_client=FakePlanBoundClient(),
                    parser=FakeParser(),
                    encoder=FakeEncoder(),
                    env_factory=lambda: self.fail("leased before plan validation"),
                    expected_groups=2,
                    expected_suffixes_per_state=2,
                )

    def test_runner_snapshots_inputs_and_rejects_internal_plan_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            actor = Path(tmp)
            (actor / "weights.bin").write_bytes(b"weights")
            actor_sha = sha256_actor_checkpoint(actor)
            decoding, backend = decoding_and_backend(actor_sha)
            resolved = [resolved_branch(17, [10])]
            plan = build_active_branch_plan(
                resolved,
                actor_checkpoint_sha256=actor_sha,
                decoding_config=decoding,
                seed=1,
                suffixes_per_state=2,
            )
            runner = ActiveSuffixRunner(
                plan=plan,
                resolved_selections=resolved,
                actor_checkpoint=actor,
                sampling_backend_contract=backend,
                completion_client=FakePlanBoundClient(),
                parser=FakeParser(),
                encoder=FakeEncoder(),
                env_factory=FakeEnvironment,
                expected_groups=1,
                expected_suffixes_per_state=2,
            )
            plan["safety"]["optimizer_enabled"] = True
            resolved[0]["trajectory"]["decision_trace"][0]["repeated"] = True
            result = asyncio.run(runner.collect())
            self.assertTrue(result["aggregate"]["mechanical_smoke_passed"])

            runner.plan["safety"]["optimizer_enabled"] = True
            result = asyncio.run(runner.collect())
            self.assertFalse(result["aggregate"]["mechanical_smoke_passed"])
            self.assertTrue(
                all(
                    record["infrastructure_error_code"] == "plan_contract_invalid"
                    for record in result["records"]
                )
            )


if __name__ == "__main__":
    unittest.main()
