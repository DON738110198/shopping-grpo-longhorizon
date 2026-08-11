"""不依赖 veRL 安装的最小适配层单测。"""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop

from shopping_grpo.training.grpo.adapter.agent_loop import (
    ShoppingToolAgentLoop,
    validate_actor_prompt_token_capture_config,
)
from shopping_grpo.training.grpo.adapter.runtime import (
    current_environment,
    current_runtime_state,
    make_runtime_state,
    record_action_attempt,
    record_action_outcome,
    record_replay_transition,
    reward_breakdown,
    task_id_from_kwargs,
    terminal_reward,
)
from shopping_grpo.training.grpo.adapter.session import ShopSimulatorSession
from shopping_grpo.training.grpo.adapter.tools import ShopSimulatorTool
from shopping_grpo.training.grpo.pivotal_states import (
    ACTOR_PROMPT_TOKENS_VERSION,
    observation_sha256,
    replay_state_id,
    token_ids_sha256,
)


def make_tool(name):
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Test-only {name} tool.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
    try:
        from verl.tools.schemas import OpenAIFunctionToolSchema
    except ImportError:
        tool_schema = schema
    else:
        tool_schema = OpenAIFunctionToolSchema.model_validate(schema)
    return ShopSimulatorTool({}, tool_schema)


class FakeGenerationTokenizer:
    def __init__(self):
        self.name_or_path = "test-tokenizer"
        self.chat_template = "test-template"
        self.vocab_size = 128
        self.all_special_ids = [0]
        self.init_kwargs = {"_commit_hash": "test-commit"}

    @staticmethod
    def get_added_vocab():
        return {}


class FakeGenerationServer:
    def __init__(self, token_ids):
        self.token_ids = list(token_ids)

    async def generate(self, **kwargs):
        del kwargs
        return SimpleNamespace(
            token_ids=self.token_ids,
            log_probs=None,
            num_preempted=0,
            extra_fields={},
            routed_experts=None,
        )


def make_generation_loop(*, response_length, max_assistant_turns):
    loop = object.__new__(ShoppingToolAgentLoop)
    loop.context_compaction_enable = False
    loop.context_window_tokens = 128
    loop.context_generation_reserve_tokens = 32
    loop.context_safety_margin_tokens = 8
    loop.context_input_budget = 64
    loop.context_preserve_recent_groups = 1
    loop.response_length = response_length
    loop.max_assistant_turns = max_assistant_turns
    loop.max_user_turns = 40
    loop.tool_parser = SimpleNamespace(stop_token_ids=[])
    loop.server_manager = FakeGenerationServer([11, 12, 13])
    loop.tokenizer = FakeGenerationTokenizer()
    return loop


def make_agent_data():
    agent_data = AgentData(
        messages=[],
        image_data=[],
        video_data=[],
        audio_data=None,
        mm_processor_kwargs={},
        metrics={},
        request_id="test-request",
        tools_kwargs={},
    )
    agent_data.prompt_ids = [101]
    return agent_data


class VerlAdapterRuntimeTest(unittest.TestCase):
    def test_actor_prompt_token_capture_config_is_bounded_and_fail_closed(self):
        config = validate_actor_prompt_token_capture_config(
            {
                "enabled": True,
                "task_ids": [7, 7, 3],
                "decision_indices": [2, 0],
                "replay_state_ids": ["a" * 64],
                "max_events_per_trajectory": 2,
            }
        )
        self.assertEqual(config["version"], ACTOR_PROMPT_TOKENS_VERSION)
        self.assertEqual(config["task_ids"], (3, 7))
        self.assertEqual(config["decision_indices"], (0, 2))
        self.assertEqual(config["replay_state_ids"], ("a" * 64,))

        with self.assertRaisesRegex(ValueError, "unknown actor_prompt_token_capture"):
            validate_actor_prompt_token_capture_config({"enabled": True, "typo": 1})
        with self.assertRaisesRegex(ValueError, "must be in"):
            validate_actor_prompt_token_capture_config(
                {"enabled": True, "max_events_per_trajectory": 0}
            )
        with self.assertRaisesRegex(ValueError, "lowercase sha256"):
            validate_actor_prompt_token_capture_config(
                {"enabled": True, "replay_state_ids": ["not-a-state"]}
            )
        for falsey_non_mapping in ([], "", 0, False):
            with (
                self.subTest(falsey_non_mapping=falsey_non_mapping),
                self.assertRaisesRegex(
                    TypeError,
                    "must be an object",
                ),
            ):
                validate_actor_prompt_token_capture_config(falsey_non_mapping)
        with self.assertRaisesRegex(TypeError, "task_ids must be a list"):
            validate_actor_prompt_token_capture_config({"task_ids": {7: True}})
        with self.assertRaisesRegex(TypeError, "replay_state_ids must be a list"):
            validate_actor_prompt_token_capture_config({"replay_state_ids": {"a" * 64: True}})

    def test_actor_prompt_tokens_capture_exact_post_compaction_prompt_once(self):
        async def fake_parent_generate(
            _loop,
            agent_data,
            sampling_params,
            ignore_termination=False,
        ):
            del sampling_params, ignore_termination
            self.assertEqual(agent_data.prompt_ids, [101, 12, 22])
            agent_data.prompt_ids.append(99)
            agent_data.response_mask.append(1)
            agent_data.tool_calls = [SimpleNamespace(name="open_product")]
            return AgentState.PROCESSING_TOOLS

        async def run():
            state_id = replay_state_id(
                7,
                [],
                observation_sha256("raw public page"),
                observation_kind="raw_public_observation",
                environment_manifest_sha256="a" * 64,
                public_query_sha256="b" * 64,
            )
            loop = make_generation_loop(response_length=64, max_assistant_turns=40)
            loop.context_compaction_enable = True
            loop.context_input_budget = 3
            loop.actor_prompt_token_capture = validate_actor_prompt_token_capture_config(
                {
                    "enabled": True,
                    "task_ids": [7],
                    "decision_indices": [0],
                    "replay_state_ids": [state_id],
                    "max_events_per_trajectory": 1,
                }
            )
            agent_data = make_agent_data()
            agent_data.prompt_ids = [101, 11, 21, 12, 22]
            agent_data.response_mask = [1, 0, 1, 0]
            agent_data.response_logprobs = []
            state = make_runtime_state(task_id=7, max_steps=35)
            state["environment_manifest_sha256"] = "a" * 64
            state["public_query_sha256"] = "b" * 64
            state["latest_observation"] = "projected public page"
            state["latest_observation_raw"] = "raw public page"
            state["replay_observation_v2_complete"] = True
            state["assistant_turn_records"] = [
                {"turn_id": 0, "kind": "tool_call"},
                {"turn_id": 1, "kind": "tool_call"},
            ]
            state["next_assistant_turn_id"] = 2
            token = current_runtime_state.set(state)
            try:
                with patch.object(
                    ToolAgentLoop,
                    "_handle_generating_state",
                    fake_parent_generate,
                ):
                    next_state = await loop._handle_generating_state(agent_data, {})
                event = record_action_attempt(
                    state,
                    "open_product",
                    {"asin": "123"},
                    state["latest_observation"],
                )
            finally:
                current_runtime_state.reset(token)
            return next_state, state, event

        next_state, state, event = asyncio.run(run())
        self.assertEqual(next_state, AgentState.PROCESSING_TOOLS)
        materialized = event["actor_prompt_tokens"]
        self.assertEqual(
            set(materialized),
            {"version", "sha256", "count", "tokens"},
        )
        self.assertEqual(materialized["version"], ACTOR_PROMPT_TOKENS_VERSION)
        self.assertEqual(materialized["tokens"], [101, 12, 22])
        self.assertEqual(materialized["count"], 3)
        self.assertEqual(materialized["sha256"], token_ids_sha256([101, 12, 22]))
        self.assertEqual(materialized["sha256"], event["actor_prompt_sha256"])
        self.assertEqual(state["actor_prompt_token_capture_count"], 1)
        self.assertEqual(state["context_tokens_removed"], 2)
        self.assertNotIn(
            "_actor_prompt_token_candidate",
            state["assistant_turn_records"][-1],
        )

    def test_actor_prompt_token_capture_respects_event_limit_and_is_default_off(self):
        async def fake_parent_generate(
            _loop,
            agent_data,
            sampling_params,
            ignore_termination=False,
        ):
            del sampling_params, ignore_termination
            agent_data.prompt_ids.append(99)
            agent_data.response_mask.append(1)
            agent_data.tool_calls = [SimpleNamespace(name="open_product")]
            return AgentState.PROCESSING_TOOLS

        async def run(config, captured_events):
            loop = make_generation_loop(response_length=64, max_assistant_turns=40)
            loop.actor_prompt_token_capture = config
            agent_data = make_agent_data()
            state = make_runtime_state(task_id=7, max_steps=35)
            state["actor_prompt_token_capture_count"] = captured_events
            token = current_runtime_state.set(state)
            try:
                with patch.object(
                    ToolAgentLoop,
                    "_handle_generating_state",
                    fake_parent_generate,
                ):
                    await loop._handle_generating_state(agent_data, {})
                return record_action_attempt(state, "open_product", {"asin": "123"}, "page")
            finally:
                current_runtime_state.reset(token)

        disabled = validate_actor_prompt_token_capture_config()
        at_limit = validate_actor_prompt_token_capture_config(
            {"enabled": True, "max_events_per_trajectory": 1}
        )
        wrong_task = validate_actor_prompt_token_capture_config(
            {"enabled": True, "task_ids": [8], "max_events_per_trajectory": 1}
        )
        wrong_decision = validate_actor_prompt_token_capture_config(
            {"enabled": True, "decision_indices": [1], "max_events_per_trajectory": 1}
        )
        wrong_state = validate_actor_prompt_token_capture_config(
            {"enabled": True, "replay_state_ids": ["f" * 64], "max_events_per_trajectory": 1}
        )
        self.assertNotIn("actor_prompt_tokens", asyncio.run(run(disabled, 0)))
        self.assertNotIn("actor_prompt_tokens", asyncio.run(run(at_limit, 1)))
        self.assertNotIn("actor_prompt_tokens", asyncio.run(run(wrong_task, 0)))
        self.assertNotIn("actor_prompt_tokens", asyncio.run(run(wrong_decision, 0)))
        self.assertNotIn("actor_prompt_tokens", asyncio.run(run(wrong_state, 0)))

    def test_actor_prompt_token_capture_failure_marks_trajectory_invalid_before_generation(self):
        async def run():
            loop = make_generation_loop(response_length=64, max_assistant_turns=40)
            loop.actor_prompt_token_capture = validate_actor_prompt_token_capture_config(
                {"enabled": True, "max_events_per_trajectory": 1}
            )
            agent_data = make_agent_data()
            # The ordinary hash normalizes this value, while exact capture rejects it.
            agent_data.prompt_ids = ["101"]
            state = make_runtime_state(task_id=7, max_steps=35)
            token = current_runtime_state.set(state)
            try:
                next_state = await loop._handle_generating_state(agent_data, {})
            finally:
                current_runtime_state.reset(token)
            return next_state, state

        next_state, state = asyncio.run(run())
        self.assertEqual(next_state, AgentState.TERMINATED)
        self.assertTrue(state["infrastructure_invalid"])
        self.assertTrue(state["terminate"])
        self.assertEqual(state["termination_reason"], "actor_prompt_token_capture_failed")
        self.assertIn("must be integers", state["actor_prompt_token_capture_error"])
        self.assertEqual(state["assistant_turn_records"], [])

    def test_response_length_limit_is_not_recorded_as_assistant_final(self):
        async def run():
            loop = make_generation_loop(response_length=3, max_assistant_turns=40)
            agent_data = make_agent_data()
            state = make_runtime_state(task_id=1, max_steps=35)
            token = current_runtime_state.set(state)
            try:
                next_state = await loop._handle_generating_state(agent_data, {})
            finally:
                current_runtime_state.reset(token)
            return next_state, state

        next_state, state = asyncio.run(run())
        self.assertEqual(next_state, AgentState.TERMINATED)
        self.assertEqual(state["decision_events"], [])
        self.assertEqual(len(state["assistant_turn_records"]), 1)
        turn = state["assistant_turn_records"][0]
        self.assertEqual(turn["generation_termination_reason"], "response_length")
        self.assertFalse(turn["credit_eligible"])

    def test_max_assistant_turns_limit_is_not_recorded_as_assistant_final(self):
        async def run():
            loop = make_generation_loop(response_length=32, max_assistant_turns=1)
            agent_data = make_agent_data()
            state = make_runtime_state(task_id=1, max_steps=35)
            token = current_runtime_state.set(state)
            try:
                next_state = await loop._handle_generating_state(agent_data, {})
            finally:
                current_runtime_state.reset(token)
            return next_state, state

        next_state, state = asyncio.run(run())
        self.assertEqual(next_state, AgentState.TERMINATED)
        self.assertEqual(state["decision_events"], [])
        self.assertEqual(len(state["assistant_turn_records"]), 1)
        turn = state["assistant_turn_records"][0]
        self.assertEqual(turn["generation_termination_reason"], "max_assistant_turns")
        self.assertFalse(turn["credit_eligible"])

    def test_agent_loop_preserves_real_verl_metrics_and_exports_shopping_diagnostics(self):
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.released = False
                created.append(self)

            def reset(self, task_id):
                return {
                    "instruction": f"task {task_id}",
                    "environment_version": "shopsimulator-environment-v2.1",
                }

            def release(self):
                self.released = True

        async def fake_parent_run(_loop, sampling_params, **kwargs):
            state = current_runtime_state.get()
            state.update(
                {
                    "done": True,
                    "terminal_result": {"done": True, "over": True},
                    "termination_reason": "gold_purchase",
                    "final_reward": 1.0,
                    "reward_version": "shopsimulator-reward-v3",
                    "reward_type": "gold_purchase",
                    "reward_valid": True,
                    "reward_detail": {
                        "weighted_score": 1.0,
                        "evidence_coverage": 1.0,
                        "dimension_scores": {"key_options": 1.0},
                        "hard_gates": {
                            "category": {"passed": True},
                            "budget": {"passed": True},
                        },
                    },
                }
            )
            return AgentLoopOutput(
                prompt_ids=[1],
                response_ids=[2],
                response_mask=[1],
                reward_score=None,
                metrics=AgentLoopMetrics(generate_sequences=0.25),
                extra_fields={},
            )

        async def run():
            loop = object.__new__(ShoppingToolAgentLoop)
            loop.base_url = "http://shop.test"
            loop.timeout = 60
            loop.max_steps = 35
            loop.required_environment_version = "shopsimulator-environment-v2.1"
            loop.reward_mode = "policy_v1"
            loop.policy_reward = {}
            loop.env_factory = FakeEnv
            with patch.object(ToolAgentLoop, "run", fake_parent_run):
                return await ShoppingToolAgentLoop.run(
                    loop,
                    {},
                    extra_info={"task_id": 42},
                )

        output = asyncio.run(run())
        self.assertIsInstance(output.metrics, AgentLoopMetrics)
        self.assertEqual(
            output.metrics.model_dump(),
            {
                "generate_sequences": 0.25,
                "tool_calls": 0.0,
                "compute_score": 0.0,
                "num_preempted": -1,
            },
        )
        self.assertEqual(output.reward_score, 1.0)
        self.assertEqual(output.extra_fields["shopping"]["task_id"], 42)
        self.assertEqual(
            output.extra_fields["shopping"]["reward"]["terminal_utility"],
            1.0,
        )
        self.assertTrue(created[0].released)

    def test_terminal_reward_only_uses_a_normal_environment_completion(self):
        done = make_runtime_state(task_id=1, max_steps=35)
        done.update(
            {"done": True, "terminal_result": {"done": True, "over": True}, "final_reward": 0.75}
        )
        self.assertEqual(terminal_reward(done), 0.75)

        unfinished = make_runtime_state(task_id=1, max_steps=35)
        unfinished.update({"final_reward": 1.0, "terminal_result": {"done": False}})
        self.assertEqual(terminal_reward(unfinished), 0.0)

        errored = make_runtime_state(task_id=1, max_steps=35)
        errored.update(
            {
                "done": True,
                "terminal_result": {"done": True, "over": True},
                "final_reward": 1.0,
                "error": "tool_error:timeout",
            }
        )
        self.assertEqual(terminal_reward(errored), 0.0)

    def test_context_state_is_task_local(self):
        state = make_runtime_state(task_id=2, max_steps=35)
        token = current_runtime_state.set(state)
        try:
            self.assertIs(current_runtime_state.get(), state)
        finally:
            current_runtime_state.reset(token)

    def test_runtime_state_has_no_hidden_goal_fields(self):
        state = make_runtime_state(task_id=2, max_steps=35)
        self.assertNotIn("goal", state)
        self.assertIsNone(state["reward_detail"])

    def test_action_attempt_records_replay_and_exact_prompt_contract(self):
        state = make_runtime_state(task_id=2, max_steps=35)
        state["latest_observation_raw"] = "public product page"
        state["environment_manifest_sha256"] = "a" * 64
        state["public_query_sha256"] = "b" * 64
        state["replay_observation_v2_complete"] = True
        state["current_assistant_turn_id"] = 3
        state["assistant_turn_records"].append(
            {
                "turn_id": 3,
                "actor_prompt_sha256": "c" * 64,
                "tokenizer_contract_sha256": "d" * 64,
            }
        )

        event = record_action_attempt(
            state,
            "open_product",
            {"asin": "123"},
            "projected product page",
        )
        self.assertTrue(event["branch_identity_complete"])
        self.assertEqual(event["assistant_turn_id"], 3)
        self.assertEqual(len(event["replay_state_id"]), 64)
        self.assertEqual(len(event["branch_uid"]), 64)
        record_action_outcome(state, event, accepted=True)
        record_replay_transition(
            state,
            event,
            parameters={"asin": "123"},
            after_observation="next public page",
            done=False,
        )
        self.assertEqual(state["replay_ledger"][0]["tool"], "open_product")
        self.assertNotIn("public product page", str(state["replay_ledger"]))

    def test_replay_ledger_keeps_exact_parameters_while_audit_summary_is_bounded(self):
        state = make_runtime_state(task_id=2, max_steps=35)
        state["latest_observation_raw"] = "public search page"
        query = "q" * 300
        event = record_action_attempt(
            state,
            "search_products",
            {"query": query},
            "projected search page",
        )
        record_action_outcome(state, event, accepted=True)
        record_replay_transition(
            state,
            event,
            parameters={"query": query},
            after_observation="results",
            done=False,
        )
        self.assertEqual(len(event["parameters"]["query"]), 256)
        self.assertEqual(event["replay_parameters"]["query"], query)
        self.assertEqual(len(state["action_events"][0]["parameters"]["query"]), 256)
        self.assertNotIn("replay_parameters", state["action_events"][0])
        self.assertEqual(state["replay_ledger"][0]["parameters"]["query"], query)

    def test_task_id_is_read_from_verl_extra_info(self):
        self.assertEqual(task_id_from_kwargs({"extra_info": {"task_id": 42}}), 42)

    def test_task_id_accepts_numpy_style_scalar_container(self):
        class Scalar:
            def item(self):
                return {"task_id": 43}

        self.assertEqual(task_id_from_kwargs({"extra_info": Scalar()}), 43)

    def test_missing_task_id_fails_before_acquiring_an_environment(self):
        with self.assertRaisesRegex(ValueError, "task_id"):
            task_id_from_kwargs({"extra_info": {"split": "train"}})

    def test_terminal_observation_is_not_returned_to_the_model(self):
        class FakeEnv:
            def step(self, action):
                self.action = action
                return {
                    "instruction": "Goal: hidden answer\nReward: hidden breakdown",
                    "done": True,
                    "over": True,
                    "reward": 1.0,
                    "goal": {"secret": True},
                    "reward_detail": {"secret": True},
                }

        async def run():
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "搜索功能是否可用: True"
            env_token = current_environment.set(FakeEnv())
            state_token = current_runtime_state.set(state)
            try:
                response, _, _ = await make_tool("search_products").execute(
                    "tool-1", {"query": "mug"}
                )
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertEqual(response.text, "Environment terminated.")
            self.assertTrue(state["terminate"])
            self.assertEqual(state["terminal_result"], {"done": True, "over": True})
            self.assertTrue(state["infrastructure_invalid"])
            self.assertIsNone(state["reward_detail"])
            self.assertNotIn("hidden", str(state))

        asyncio.run(run())

    def test_terminal_reward_components_are_validated_without_entering_tool_observation(self):
        class FakeEnv:
            def step(self, action):
                return {
                    "instruction": "Goal: hidden answer",
                    "done": True,
                    "over": True,
                    "reward": 0.6,
                    "termination_reason": "valid_alternative_purchase",
                    "goal": {"secret": True},
                    "reward_detail": {
                        "reward_version": "shopsimulator-reward-v3",
                        "reward_type": "valid_alternative_purchase",
                        "reward_valid": True,
                        "termination_reason": "valid_alternative_purchase",
                        "target_asin_match": False,
                        "terminal_utility": 0.6,
                        "purchase_success": True,
                        "sampling_invalid": False,
                        "hard_gates": {
                            "category": {
                                "status": "pass",
                                "passed": True,
                                "verifiable": True,
                            }
                        },
                        "weighted_score": 0.75,
                        "evidence_coverage": 0.8,
                        "dimension_scores": {
                            "brand": 1.0,
                            "model": 1.0,
                            "core_functions": 0.5,
                            "key_options": 0.5,
                        },
                        "hidden_answer": "do not retain",
                    },
                }

        async def run():
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "搜索功能是否可用: True"
            env_token = current_environment.set(FakeEnv())
            state_token = current_runtime_state.set(state)
            try:
                response, _, _ = await make_tool("search_products").execute(
                    "tool-1", {"query": "mug"}
                )
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)

            self.assertEqual(response.text, "Environment terminated.")
            self.assertFalse(state["infrastructure_invalid"])
            self.assertEqual(
                state["reward_detail"]["dimension_scores"],
                {
                    "brand": 1.0,
                    "model": 1.0,
                    "core_functions": 0.5,
                    "key_options": 0.5,
                },
            )
            self.assertNotIn("hidden", str(state))

        asyncio.run(run())

    def test_terminal_reward_keeps_unverifiable_separate_from_infrastructure(self):
        class FakeEnv:
            def step(self, action):
                return {
                    "instruction": "terminal",
                    "done": True,
                    "over": True,
                    "reward": 0.0,
                    "termination_reason": "reward_unverifiable",
                    "reward_valid": False,
                    "reward_detail": {
                        "reward_version": "shopsimulator-reward-v3",
                        "reward_type": "reward_unverifiable",
                        "reward_valid": False,
                        "termination_reason": "reward_unverifiable",
                        "target_asin_match": False,
                        "terminal_utility": 0.0,
                        "purchase_success": False,
                        "sampling_invalid": True,
                        "hard_gates": {
                            "category": {
                                "status": "unverifiable",
                                "passed": False,
                                "verifiable": False,
                            }
                        },
                        "weighted_score": 0.0,
                        "evidence_coverage": 0.0,
                        "dimension_scores": {},
                    },
                }

        async def run():
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "搜索功能是否可用: True"
            env_token = current_environment.set(FakeEnv())
            state_token = current_runtime_state.set(state)
            try:
                await make_tool("search_products").execute("tool-v2", {"query": "mug"})
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertFalse(state["infrastructure_invalid"])
            self.assertTrue(state["reward_unverifiable"])
            self.assertEqual(state["reward_type"], "reward_unverifiable")
            self.assertEqual(state["termination_reason"], "reward_unverifiable")

        asyncio.run(run())

    def test_reward_exposes_utility_success_and_sampling_validity_separately(self):
        class FakeEnv:
            def step(self, action):
                return {
                    "instruction": "terminal",
                    "done": True,
                    "over": True,
                    "reward": 0.55,
                    "termination_reason": "valid_alternative_purchase",
                    "reward_valid": True,
                    "reward_detail": {
                        "reward_version": "shopsimulator-reward-v3",
                        "reward_type": "valid_alternative_purchase",
                        "reward_valid": True,
                        "termination_reason": "valid_alternative_purchase",
                        "target_asin_match": False,
                        "terminal_utility": 0.55,
                        "purchase_success": True,
                        "sampling_invalid": False,
                        "weighted_score": 1.0,
                        "evidence_coverage": 1.0,
                        "dimension_scores": {
                            "brand": 0.0,
                            "model": 0.0,
                            "core_functions": 1.0,
                            "key_options": 1.0,
                        },
                        "hard_gates": {
                            "category": {
                                "status": "pass",
                                "passed": True,
                                "verifiable": True,
                                "comparator": "category_leaf_ancestor_chain",
                                "source_field": "category",
                            }
                        },
                    },
                }

        async def run():
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "搜索功能是否可用: True"
            env_token = current_environment.set(FakeEnv())
            state_token = current_runtime_state.set(state)
            try:
                await make_tool("search_products").execute(
                    "tool-v3",
                    {"query": "mug"},
                )
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertFalse(state["infrastructure_invalid"])
            self.assertFalse(state["reward_unverifiable"])
            self.assertEqual(
                state["reward_type"],
                "valid_alternative_purchase",
            )
            breakdown = reward_breakdown(state)
            self.assertEqual(breakdown["terminal_utility"], 0.55)
            self.assertEqual(breakdown["purchase_success"], 1.0)
            self.assertEqual(breakdown["r_att"], 1.0)
            self.assertEqual(breakdown["r_option"], 1.0)
            self.assertFalse(breakdown["sampling_invalid"])

        asyncio.run(run())

    def test_sync_environment_step_runs_off_the_event_loop_thread(self):
        main_thread = threading.get_ident()

        class FakeEnv:
            step_thread = None

            def step(self, action):
                self.step_thread = threading.get_ident()
                return {"instruction": "next", "done": False, "over": False, "reward": 0.0}

        async def run():
            env = FakeEnv()
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "搜索功能是否可用: True"
            env_token = current_environment.set(env)
            state_token = current_runtime_state.set(state)
            try:
                await make_tool("search_products").execute("tool-1", {"query": "mug"})
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertNotEqual(env.step_thread, main_thread)

        asyncio.run(run())

    def test_think_consumes_the_step_budget_and_terminates_at_the_exact_limit(self):
        async def run():
            state = make_runtime_state(task_id=2, max_steps=1)
            env_token = current_environment.set(object())
            state_token = current_runtime_state.set(state)
            try:
                response, _, _ = await make_tool("think").execute("tool-1", {"note": "plan"})
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertEqual(len(state["steps"]), 1)
            self.assertTrue(state["terminate"])
            self.assertEqual(state["error"], "max_steps")
            self.assertIn("maximum", response.text)

        asyncio.run(run())

    def test_repeated_guard_rejections_terminate_instead_of_looping_forever(self):
        async def run():
            state = make_runtime_state(task_id=2, max_steps=35)
            state["latest_observation"] = "可点击的按钮: []"
            state["latest_observation_truncated"] = True
            env_token = current_environment.set(object())
            state_token = current_runtime_state.set(state)
            try:
                tool = make_tool("open_product")
                for index in range(3):
                    response, _, _ = await tool.execute(f"tool-{index}", {"asin": "123456789012"})
            finally:
                current_runtime_state.reset(state_token)
                current_environment.reset(env_token)
            self.assertTrue(state["terminate"])
            self.assertEqual(state["error"], "too_many_guard_rejections")
            self.assertEqual(state["steps"], [])
            self.assertEqual(state["action_attempt_count"], 3)
            self.assertEqual(state["repeat_action_count"], 2)
            self.assertEqual(state["guard_rejection_count"], 3)
            self.assertEqual(state["guard_rejection_after_truncation_count"], 3)
            self.assertEqual(state["action_attempt_after_truncation_count"], 3)
            self.assertIn("maximum", response.text)

        asyncio.run(run())

    def test_session_releases_its_environment_on_close(self):
        """无论正常终局还是异常路径，veRL lifecycle 都必须归还 ShopSimulator 租约。"""
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.released = False
                created.append(self)

            def reset(self, task_id):
                return {"instruction": f"task {task_id}"}

            def release(self):
                self.released = True

        async def run():
            session = ShopSimulatorSession(max_steps=35, env_factory=FakeEnv)
            state = await session.start(task_id=8)
            state.update(
                {"done": True, "terminal_result": {"done": True, "over": True}, "final_reward": 1.0}
            )
            self.assertEqual(terminal_reward(state), 1.0)
            await session.close()

        asyncio.run(run())
        self.assertTrue(created[0].released)

    def test_session_reset_and_release_run_off_the_event_loop_thread(self):
        main_thread = threading.get_ident()
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.reset_thread = None
                self.release_thread = None
                created.append(self)

            def reset(self, task_id):
                self.reset_thread = threading.get_ident()
                return {"instruction": f"task {task_id}"}

            def release(self):
                self.release_thread = threading.get_ident()

        async def run():
            session = ShopSimulatorSession(env_factory=FakeEnv)
            await session.start(task_id=8)
            await session.close()

        asyncio.run(run())
        self.assertNotEqual(created[0].reset_thread, main_thread)
        self.assertNotEqual(created[0].release_thread, main_thread)

    def test_session_rejects_wrong_environment_version_and_releases(self):
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.released = False
                created.append(self)

            def reset(self, task_id):
                return {
                    "instruction": f"task {task_id}",
                    "environment_version": "unsupported-environment",
                }

            def release(self):
                self.released = True

        async def run():
            session = ShopSimulatorSession(
                required_environment_version="shopsimulator-environment-v2.1",
                env_factory=FakeEnv,
            )
            with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                await session.start(1)

        asyncio.run(run())
        self.assertTrue(created[0].released)

    def test_session_rejects_wrong_environment_manifest_and_releases(self):
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.released = False
                created.append(self)

            def reset(self, task_id):
                return {
                    "instruction": f"task {task_id}",
                    "environment_version": "shopsimulator-environment-v2.1",
                    "environment_manifest_sha256": "b" * 64,
                }

            def release(self):
                self.released = True

        async def run():
            session = ShopSimulatorSession(
                required_environment_version="shopsimulator-environment-v2.1",
                required_environment_manifest_sha256="a" * 64,
                env_factory=FakeEnv,
            )
            with self.assertRaisesRegex(RuntimeError, "manifest mismatch"):
                await session.start(1)

        asyncio.run(run())
        self.assertTrue(created[0].released)

    def test_reset_failure_still_releases_the_environment(self):
        created = []

        class FakeEnv:
            def __init__(self, **kwargs):
                self.released = False
                created.append(self)

            def reset(self, task_id):
                raise RuntimeError("reset failed")

            def release(self):
                self.released = True

        async def run():
            session = ShopSimulatorSession(env_factory=FakeEnv)
            with self.assertRaisesRegex(RuntimeError, "reset failed"):
                await session.start(task_id=8)

        asyncio.run(run())
        self.assertTrue(created[0].released)

    def test_release_failure_is_not_silently_hidden_or_forgotten(self):
        class FakeEnv:
            def __init__(self, **kwargs):
                pass

            def reset(self, task_id):
                return {"instruction": f"task {task_id}"}

            def release(self):
                raise RuntimeError("release failed")

        async def run():
            session = ShopSimulatorSession(env_factory=FakeEnv)
            await session.start(task_id=8)
            with self.assertRaisesRegex(RuntimeError, "release failed"):
                await session.close()
            self.assertEqual(session.state["error"], "release_error:RuntimeError:release failed")

        asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
