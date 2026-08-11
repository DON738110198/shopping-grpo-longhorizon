import unittest

from shopping_grpo.training.grpo.active_branch import (
    ACTOR_PROMPT_TOKENS_VERSION,
    build_active_branch_plan,
    observation_policy_sha256,
    validate_active_branch_plan,
    validate_actor_prompt_tokens,
)
from shopping_grpo.training.grpo.pivotal_states import (
    branch_uid,
    token_ids_sha256,
)


def resolved_branch(task_id=7, prompt_tokens=None):
    prompt_tokens = prompt_tokens or [10, 20, 30]
    prompt_hash = token_ids_sha256(prompt_tokens)
    state_id = f"{task_id:064x}"
    tokenizer_hash = "b" * 64
    parent_branch = branch_uid(state_id, prompt_hash, tokenizer_hash)
    selection = {
        "selection_index": 0,
        "task_id": task_id,
        "replay_state_id": state_id,
        "branch_uid": parent_branch,
        "prefix_action_count": 2,
        "pivotal_labels": ["candidate_open"],
        "source": {
            "input_index": 0,
            "input_sha256": "f" * 64,
            "path": "/audit.jsonl",
            "line": 1,
            "global_step": 1,
            "generation_batch": 1,
            "uid": "group-1",
            "trajectory_index": 0,
            "event_index": 2,
        },
    }
    event = {
        "branch_uid": parent_branch,
        "replay_state_id": state_id,
        "actor_prompt_sha256": prompt_hash,
        "tokenizer_contract_sha256": tokenizer_hash,
        "actor_prompt_tokens": {
            "version": ACTOR_PROMPT_TOKENS_VERSION,
            "sha256": prompt_hash,
            "count": len(prompt_tokens),
            "tokens": prompt_tokens,
        },
    }
    trajectory = {"environment_manifest_sha256": "a" * 64}
    return {"selection": selection, "trajectory": trajectory, "event": event}


DECODING = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": -1,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "max_tokens_mode": "verl-v0.8-remaining-context",
    "seed_schedule": "first-suffix-then-sha256-turn-v1",
    "min_tokens": 0,
    "stop": [],
    "stop_token_ids": [],
    "ignore_eos": False,
    "max_steps": 35,
    "prompt_length": 4096,
    "response_length": 20480,
    "context_window": 24576,
    "context_generation_reserve": 512,
    "context_safety_margin": 512,
    "context_input_budget": 16384,
    "context_preserve_recent_groups": 1,
    "context_compaction_enable": False,
    "max_user_turns": 40,
    "max_assistant_turns": 40,
    "max_parallel_calls": 1,
    "max_tool_response_length": 16384,
    "tool_response_truncate_side": "middle",
    "tokenization_sanity_check_mode": "ignore_strippable",
    "apply_chat_template_kwargs": {},
    "mm_processor_kwargs": {},
    "tool_parser": "qwen3_coder",
    "tool_schema_sha256": "c" * 64,
    "generation_config_source": "vllm",
    "sampling_backend_contract_sha256": "9" * 64,
    "observation_token_budget": 1536,
    "observation_detail_token_budget": 4096,
    "observation_generic_token_budget": 768,
    "observation_search_top_k": 20,
}
DECODING["observation_policy_sha256"] = observation_policy_sha256(DECODING)


class ActiveBranchPlanTest(unittest.TestCase):
    def test_plan_binds_prompt_actor_decoding_and_suffix_seeds(self):
        plan = build_active_branch_plan(
            [resolved_branch(7), resolved_branch(8, [40, 50])],
            actor_checkpoint_sha256="e" * 64,
            decoding_config=DECODING,
            seed=20260811,
            suffixes_per_state=4,
        )

        self.assertEqual(plan["aggregate"]["active_groups"], 2)
        self.assertEqual(plan["aggregate"]["unique_tasks"], 2)
        self.assertEqual(plan["aggregate"]["planned_suffixes"], 8)
        self.assertTrue(plan["safety"]["outcome_blind"])
        self.assertFalse(plan["safety"]["training_ready"])
        for group in plan["groups"]:
            self.assertEqual(len(group["suffixes"]), 4)
            self.assertEqual(len({item["seed"] for item in group["suffixes"]}), 4)
            self.assertTrue(all(len(item["suffix_uid"]) == 64 for item in group["suffixes"]))
        self.assertEqual(validate_active_branch_plan(plan), plan)

        other_seed = build_active_branch_plan(
            [resolved_branch(7), resolved_branch(8, [40, 50])],
            actor_checkpoint_sha256="e" * 64,
            decoding_config=DECODING,
            seed=20260812,
            suffixes_per_state=4,
        )
        self.assertNotEqual(
            plan["groups"][0]["active_group_uid"],
            other_seed["groups"][0]["active_group_uid"],
        )
        self.assertNotEqual(
            plan["groups"][0]["suffixes"][0]["suffix_uid"],
            other_seed["groups"][0]["suffixes"][0]["suffix_uid"],
        )

    def test_prompt_capture_fails_closed_on_token_or_hash_drift(self):
        resolved = resolved_branch()
        event = resolved["event"]
        capture = dict(event["actor_prompt_tokens"])
        capture["tokens"] = [10, 20, 31]

        with self.assertRaisesRegex(ValueError, "capture hash mismatch"):
            validate_actor_prompt_tokens(
                capture,
                expected_sha256=event["actor_prompt_sha256"],
            )

        event["actor_prompt_tokens"] = None
        with self.assertRaisesRegex(TypeError, "must be an object"):
            build_active_branch_plan(
                [resolved],
                actor_checkpoint_sha256="e" * 64,
                decoding_config=DECODING,
                seed=1,
                suffixes_per_state=4,
            )

    def test_plan_rejects_outcome_or_config_shortcuts(self):
        resolved = resolved_branch()
        resolved["selection"]["branch_uid"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "event branch mismatch"):
            build_active_branch_plan(
                [resolved],
                actor_checkpoint_sha256="e" * 64,
                decoding_config=DECODING,
                seed=1,
                suffixes_per_state=4,
            )

        injected = resolved_branch()
        injected["selection"]["source"]["reward"] = 1.0
        with self.assertRaisesRegex(ValueError, "source fields"):
            build_active_branch_plan(
                [injected],
                actor_checkpoint_sha256="e" * 64,
                decoding_config=DECODING,
                seed=1,
                suffixes_per_state=4,
            )

        bad_config = dict(DECODING)
        bad_config["temperature"] = float("nan")
        with self.assertRaisesRegex(ValueError, "temperature"):
            build_active_branch_plan(
                [resolved_branch()],
                actor_checkpoint_sha256="e" * 64,
                decoding_config=bad_config,
                seed=1,
                suffixes_per_state=4,
            )

    def test_prompt_capture_rejects_boolean_count(self):
        prompt_hash = token_ids_sha256([10])
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            validate_actor_prompt_tokens(
                {
                    "version": ACTOR_PROMPT_TOKENS_VERSION,
                    "sha256": prompt_hash,
                    "count": True,
                    "tokens": [10],
                },
                expected_sha256=prompt_hash,
            )

    def test_every_explicit_sampling_knob_changes_group_identity(self):
        baseline = build_active_branch_plan(
            [resolved_branch()],
            actor_checkpoint_sha256="e" * 64,
            decoding_config=DECODING,
            seed=1,
            suffixes_per_state=4,
        )["groups"][0]["active_group_uid"]
        alternatives = {
            "temperature": 0.6,
            "top_p": 0.8,
            "top_k": 50,
            "min_p": 0.1,
            "repetition_penalty": 1.1,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.1,
            "prompt_length": 4000,
            "min_tokens": 1,
            "stop": ["END"],
            "stop_token_ids": [1],
            "ignore_eos": True,
            "sampling_backend_contract_sha256": "8" * 64,
            "max_steps": 34,
            "response_length": 20000,
            "context_window": 25000,
            "context_generation_reserve": 768,
            "context_safety_margin": 256,
            "context_input_budget": 16000,
            "context_preserve_recent_groups": 2,
            "context_compaction_enable": True,
            "max_user_turns": 39,
            "max_assistant_turns": 39,
            "max_tool_response_length": 8192,
            "tool_response_truncate_side": "left",
            "tokenization_sanity_check_mode": "strict",
        }
        for field, value in alternatives.items():
            with self.subTest(field=field):
                config = {**DECODING, field: value}
                changed = build_active_branch_plan(
                    [resolved_branch()],
                    actor_checkpoint_sha256="e" * 64,
                    decoding_config=config,
                    seed=1,
                    suffixes_per_state=4,
                )["groups"][0]["active_group_uid"]
                self.assertNotEqual(baseline, changed)

        invalid_parallelism = {**DECODING, "max_parallel_calls": 2}
        with self.assertRaisesRegex(ValueError, "max_parallel_calls must equal 1"):
            build_active_branch_plan(
                [resolved_branch()],
                actor_checkpoint_sha256="e" * 64,
                decoding_config=invalid_parallelism,
                seed=1,
                suffixes_per_state=4,
            )

    def test_suffix_count_is_bound_to_group_and_suffix_identity(self):
        plan_four = build_active_branch_plan(
            [resolved_branch()],
            actor_checkpoint_sha256="e" * 64,
            decoding_config=DECODING,
            seed=1,
            suffixes_per_state=4,
        )
        plan_eight = build_active_branch_plan(
            [resolved_branch()],
            actor_checkpoint_sha256="e" * 64,
            decoding_config=DECODING,
            seed=1,
            suffixes_per_state=8,
        )

        self.assertNotEqual(
            plan_four["groups"][0]["active_group_uid"],
            plan_eight["groups"][0]["active_group_uid"],
        )
        self.assertNotEqual(
            plan_four["groups"][0]["suffixes"][0]["suffix_uid"],
            plan_eight["groups"][0]["suffixes"][0]["suffix_uid"],
        )

    def test_plan_validator_recomputes_all_execution_identities(self):
        plan = build_active_branch_plan(
            [resolved_branch()],
            actor_checkpoint_sha256="e" * 64,
            decoding_config=DECODING,
            seed=1,
            suffixes_per_state=4,
        )

        tampered_seed = {
            **plan,
            "groups": [
                {
                    **plan["groups"][0],
                    "suffixes": [
                        {**plan["groups"][0]["suffixes"][0], "seed": 7},
                        *plan["groups"][0]["suffixes"][1:],
                    ],
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "suffix seed mismatch"):
            validate_active_branch_plan(tampered_seed)

        tampered_group = {
            **plan,
            "groups": [{**plan["groups"][0], "prefix_action_count": 3}],
        }
        with self.assertRaisesRegex(ValueError, "active_group_uid mismatch"):
            validate_active_branch_plan(tampered_group)

        tampered_decoding = {
            **plan,
            "decoding_config": {**plan["decoding_config"], "temperature": 0.6},
        }
        with self.assertRaisesRegex(ValueError, "decoding_config_sha256 mismatch"):
            validate_active_branch_plan(tampered_decoding)


if __name__ == "__main__":
    unittest.main()
