import json
import unittest

from shopping_grpo.training.grpo.pivotal_states import (
    REPLAY_STATE_VERSION,
    TURN_SPAN_VERSION,
    audit_pivotal_states,
    branch_uid,
    canonical_replay_action,
    materialize_assistant_turn_spans,
    observation_sha256,
    replay_action_sha256,
    replay_state_id,
    token_ids_sha256,
    validate_event_training_contract,
)


def exact_trajectory(strict):
    task_id = 19
    manifest_hash = "a" * 64
    query_hash = "b" * 64
    tokenizer_hash = "c" * 64
    observation_hashes = [
        observation_sha256(name) for name in ("home", "results", "product", "selected", "back")
    ]
    actions = [
        canonical_replay_action("search_products", {"query": "red mug"}),
        canonical_replay_action("open_product", {"asin": "A"}),
        canonical_replay_action("select_option", {"value": "large"}),
    ]
    if strict:
        actions.append(canonical_replay_action("buy_now", {}))
    else:
        actions.extend(
            [
                canonical_replay_action("back_to_search", {}),
                canonical_replay_action(
                    "finish_without_purchase",
                    {"reason": "no_suitable_product"},
                ),
            ]
        )
    ledger = []
    events = []
    spans = []
    for index, action_value in enumerate(actions):
        before_hash = observation_hashes[min(index, len(observation_hashes) - 1)]
        done = index + 1 == len(actions)
        after_hash = None if done else observation_hashes[index + 1]
        ledger.append(
            {
                "sequence": index,
                "tool": action_value["tool"],
                "parameters": action_value["parameters"],
                "before_public_observation_sha256": before_hash,
                "after_public_observation_sha256": after_hash,
                "done": done,
            }
        )
        state_id = replay_state_id(
            task_id,
            actions[:index],
            before_hash,
            observation_kind="raw_public_observation",
            environment_manifest_sha256=manifest_hash,
            public_query_sha256=query_hash,
        )
        prompt_hash = "d" * 64 if index == 3 else f"{index + 1:064x}"
        events.append(
            {
                "index": index,
                "tool": action_value["tool"],
                "parameters": action_value["parameters"],
                "replay_parameters": action_value["parameters"],
                "action_sha256": replay_action_sha256(
                    action_value["tool"],
                    action_value["parameters"],
                ),
                "observation_sha256": before_hash,
                "raw_observation_sha256": before_hash,
                "replay_state_id": state_id,
                "branch_uid": branch_uid(state_id, prompt_hash, tokenizer_hash),
                "actor_prompt_sha256": prompt_hash,
                "tokenizer_contract_sha256": tokenizer_hash,
                "prefix_action_count": index,
                "assistant_turn_id": index,
                "accepted": True,
                "repeated": False,
                "guard_reason": None,
                "error": None,
            }
        )
        spans.append(
            {
                "turn_id": index,
                "kind": "tool_call",
                "tool_names": [action_value["tool"]],
                "tool_call_count": 1,
                "credit_eligible": True,
                "assistant_span": [index * 2, index * 2 + 1],
                "observation_span": [index * 2 + 1, index * 2 + 2],
            }
        )
    return {
        "task_id": task_id,
        "strict": float(strict),
        "terminal_utility": 1.0 if strict else -0.2,
        "policy_reward": 1.0 if strict else -0.2,
        "reward_type": "gold_purchase" if strict else "graceful_stop",
        "termination_reason": "gold_purchase" if strict else "graceful_stop",
        "valid_for_learning": True,
        "invalid_reason": None,
        "replay_state_version": REPLAY_STATE_VERSION,
        "environment_manifest_sha256": manifest_hash,
        "environment_version": "shopsimulator-environment-v2.1",
        "public_query_sha256": query_hash,
        "initial_public_observation_sha256": observation_hashes[0],
        "replay_observation_v2_complete": True,
        "replay_ledger": ledger,
        "turn_span_version": TURN_SPAN_VERSION,
        "turn_span_valid": True,
        "turn_spans": spans,
        "action_trace": events,
    }


def action(index, tool, parameters, observation, *, accepted=True, repeated=False):
    return {
        "index": index,
        "tool": tool,
        "parameters": parameters,
        "observation_sha256": observation_sha256(observation),
        "accepted": accepted,
        "repeated": repeated,
        "guard_reason": None if accepted else "not_visible",
        "error": None,
    }


def trajectory(task_id, strict, final_tool, *, secret=None):
    trace = [
        action(0, "search_products", {"query": "red mug"}, "home"),
        action(1, "open_product", {"asin": "A"}, "results"),
        action(2, "select_option", {"value": "large"}, "product"),
        action(3, final_tool, {}, "selected product"),
    ]
    row = {
        "task_id": task_id,
        "strict": 1.0 if strict else 0.0,
        "terminal_utility": 1.0 if strict else -0.65,
        "policy_reward": 1.0 if strict else -0.65,
        "reward_type": "gold_purchase" if strict else "repeat_loop",
        "termination_reason": "gold_purchase" if strict else "repeat_loop",
        "valid_for_learning": True,
        "invalid_reason": None,
        "action_trace": trace,
    }
    if secret is not None:
        row["goal"] = secret
    return row


class PivotalStateAuditTest(unittest.TestCase):
    def test_replay_state_hash_is_canonical_and_history_sensitive(self):
        prefix_a = [{"tool": "search_products", "parameters": {"query": "mug", "x": 1}}]
        prefix_b = [{"parameters": {"x": 1, "query": "mug"}, "tool": "search_products"}]
        kwargs = {
            "observation_kind": "raw_public_observation",
            "environment_manifest_sha256": "a" * 64,
            "public_query_sha256": "b" * 64,
        }
        state_a = replay_state_id(7, prefix_a, "c" * 64, **kwargs)
        state_b = replay_state_id(7, prefix_b, "c" * 64, **kwargs)
        self.assertEqual(state_a, state_b)
        self.assertNotEqual(
            state_a,
            replay_state_id(7, [], "c" * 64, **kwargs),
        )

    def test_branch_uid_binds_exact_actor_prompt(self):
        first = branch_uid("state", token_ids_sha256([1, 2]), "tokenizer")
        second = branch_uid("state", token_ids_sha256([1, 3]), "tokenizer")
        self.assertNotEqual(first, second)

    def test_materializes_turn_and_observation_spans_from_final_mask(self):
        records = [
            {
                "turn_id": 0,
                "kind": "tool_call",
                "generated_token_count": 2,
            },
            {
                "turn_id": 1,
                "kind": "tool_call",
                "generated_token_count": 1,
            },
            {
                "turn_id": 2,
                "kind": "assistant_termination",
                "generated_token_count": 2,
            },
        ]
        spans = materialize_assistant_turn_spans(
            records,
            [1, 1, 0, 0, 1, 0, 0, 1, 1],
        )
        self.assertEqual(spans[0]["assistant_span"], [0, 2])
        self.assertEqual(spans[0]["observation_span"], [2, 4])
        self.assertEqual(spans[1]["assistant_span"], [4, 5])
        self.assertEqual(spans[1]["observation_span"], [5, 7])
        self.assertEqual(spans[2]["assistant_span"], [7, 9])
        self.assertEqual(spans[2]["observation_span"], [9, 9])

    def test_turn_span_mismatch_fails_instead_of_clipping(self):
        with self.assertRaisesRegex(ValueError, "token count mismatch"):
            materialize_assistant_turn_spans(
                [
                    {
                        "turn_id": 0,
                        "kind": "tool_call",
                        "generated_token_count": 3,
                    }
                ],
                [1, 1],
            )

    def test_only_final_assistant_termination_may_be_truncated(self):
        spans = materialize_assistant_turn_spans(
            [
                {
                    "turn_id": 0,
                    "kind": "tool_call",
                    "generated_token_count": 2,
                    "credit_eligible": True,
                },
                {
                    "turn_id": 1,
                    "kind": "assistant_termination",
                    "generated_token_count": 5,
                    "credit_eligible": False,
                },
            ],
            [1, 1, 0, 1, 1],
        )
        self.assertEqual(spans[0]["assistant_span"], [0, 2])
        self.assertTrue(spans[1]["truncated"])
        self.assertFalse(spans[1]["credit_eligible"])

    def test_final_fully_truncated_termination_does_not_erase_prior_turns(self):
        spans = materialize_assistant_turn_spans(
            [
                {
                    "turn_id": 0,
                    "kind": "tool_call",
                    "generated_token_count": 2,
                    "credit_eligible": True,
                },
                {
                    "turn_id": 1,
                    "kind": "assistant_termination",
                    "generated_token_count": 3,
                    "credit_eligible": False,
                },
            ],
            [1, 1, 0],
        )
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["turn_id"], 0)

    def test_audit_finds_same_state_with_different_action_and_strict_outcome(self):
        record = {
            "global_step": 1,
            "generation_batch": 1,
            "uid": "prompt-one",
            "trajectories": [
                trajectory(7, True, "buy_now", secret="hidden-target-value"),
                trajectory(7, False, "view_features", secret="hidden-target-value"),
            ],
        }
        audit = audit_pivotal_states([("audit.jsonl", 1, record)])
        aggregate = audit["aggregate"]
        self.assertEqual(aggregate["eligible_mixed_strict_states"], 1)
        eligible = [group for group in audit["groups"] if group["eligible_mixed_strict"]]
        self.assertEqual(len(eligible), 1)
        self.assertIn("option_selected", eligible[0]["pivotal_labels"])
        self.assertEqual(eligible[0]["unique_next_actions"], 2)
        self.assertNotIn("hidden-target-value", json.dumps(audit))

    def test_guard_attempt_does_not_change_environment_replay_prefix(self):
        success = trajectory(8, True, "buy_now")
        failure = trajectory(8, False, "view_features")
        rejected = action(
            3,
            "open_product",
            {"asin": "stale"},
            "selected product",
            accepted=False,
        )
        failure["action_trace"].insert(3, rejected)
        failure["action_trace"][4]["index"] = 4
        audit = audit_pivotal_states(
            [
                (
                    "audit.jsonl",
                    1,
                    {
                        "global_step": 1,
                        "generation_batch": 1,
                        "uid": "prompt-two",
                        "trajectories": [success, failure],
                    },
                )
            ]
        )
        self.assertGreaterEqual(audit["aggregate"]["eligible_mixed_strict_states"], 1)
        eligible = [group for group in audit["groups"] if group["eligible_mixed_strict"]]
        self.assertTrue(any("post_guard" in group["pivotal_labels"] for group in eligible))

    def test_exact_contract_candidates_are_observational_and_never_training_ready(self):
        record = {
            "global_step": 5,
            "generation_batch": 1,
            "uid": "same-policy",
            "trajectories": [
                exact_trajectory(True),
                exact_trajectory(True),
                exact_trajectory(False),
                exact_trajectory(False),
            ],
        }
        audit = audit_pivotal_states([("audit.jsonl", 1, record)])
        self.assertEqual(audit["aggregate"]["natural_suffix_group_candidates"], 1)
        self.assertEqual(audit["aggregate"]["exact_contract_pivotal_states"], 2)
        self.assertEqual(
            audit["exact_prompt_branch_groups"][0]["action_outcomes"][0]["visits"],
            2,
        )
        self.assertFalse(audit["safety"]["training_ready"])
        self.assertFalse(audit["safety"]["hash_contract_feasible"])

    def test_invalid_learning_outcome_cannot_create_a_mixed_group(self):
        failure = exact_trajectory(False)
        failure["valid_for_learning"] = False
        failure["invalid_reason"] = "infrastructure_invalid"
        audit = audit_pivotal_states(
            [
                (
                    "audit.jsonl",
                    1,
                    {
                        "global_step": 5,
                        "generation_batch": 1,
                        "uid": "same-policy",
                        "trajectories": [exact_trajectory(True), failure],
                    },
                )
            ]
        )
        self.assertEqual(audit["aggregate"]["eligible_mixed_strict_states"], 0)

    def test_unknown_learning_validity_is_excluded_from_observational_groups(self):
        unknown = trajectory(9, False, "view_features")
        del unknown["valid_for_learning"]
        audit = audit_pivotal_states(
            [
                (
                    "audit.jsonl",
                    1,
                    {
                        "global_step": 1,
                        "generation_batch": 1,
                        "uid": "unknown-validity",
                        "trajectories": [trajectory(9, True, "buy_now"), unknown],
                    },
                )
            ]
        )
        self.assertEqual(audit["aggregate"]["eligible_mixed_strict_states"], 0)
        self.assertEqual(audit["aggregate"]["learning_validity_unknown_trajectories"], 1)

    def test_action_hash_is_recomputed_from_exact_event_action(self):
        candidate = exact_trajectory(True)
        event = candidate["action_trace"][2]
        event["action_sha256"] = "f" * 64
        valid, reason = validate_event_training_contract(candidate, event)
        self.assertFalse(valid)
        self.assertEqual(reason, "action_hash_mismatch")

    def test_long_action_uses_exact_preimage_not_bounded_display_parameters(self):
        candidate = exact_trajectory(True)
        event = candidate["action_trace"][0]
        long_query = "q" * 300
        event["parameters"] = {"query": long_query[:256]}
        event["replay_parameters"] = {"query": long_query}
        event["action_sha256"] = replay_action_sha256(
            "search_products",
            {"query": long_query},
        )
        candidate["replay_ledger"][0]["parameters"] = {"query": long_query}
        valid, reason = validate_event_training_contract(candidate, event)
        self.assertTrue(valid, reason)

    def test_semantic_duplicate_trajectory_is_counted_once(self):
        record = {
            "policy_scope_uid": "e" * 64,
            "global_step": 5,
            "generation_batch": 1,
            "uid": "same-slot",
            "trajectories": [exact_trajectory(True)],
        }
        audit = audit_pivotal_states(
            [
                ("first.jsonl", 1, record),
                ("partially-overlapping-copy.jsonl", 8, record),
            ]
        )
        self.assertEqual(audit["aggregate"]["trajectories"], 1)
        self.assertEqual(audit["aggregate"]["duplicate_semantic_trajectories"], 1)

    def test_same_branch_from_different_policy_steps_is_not_merged(self):
        records = [
            (
                "audit.jsonl",
                index,
                {
                    "global_step": index,
                    "generation_batch": 1,
                    "uid": "same-task",
                    "trajectories": [exact_trajectory(strict)],
                },
            )
            for index, strict in ((1, True), (2, False))
        ]
        audit = audit_pivotal_states(records)
        self.assertEqual(audit["aggregate"]["exact_prompt_shared_branches"], 0)


if __name__ == "__main__":
    unittest.main()
