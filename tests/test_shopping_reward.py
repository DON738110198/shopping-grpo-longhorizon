"""Tests for Reward v3 preservation and the training-only policy reward."""

import unittest

from shopping_grpo.training.grpo.adapter.runtime import (
    POLICY_REWARD_VERSION,
    make_runtime_state,
    record_action_attempt,
    reward_breakdown,
    terminal_reward,
    validate_policy_reward_config,
)


def terminal_state(
    *,
    reward_type="gold_purchase",
    terminal_utility=1.0,
    guard_rejections=0,
    repeat_actions=0,
):
    state = make_runtime_state(task_id=1, max_steps=35)
    state.update(
        {
            "done": True,
            "terminal_result": {"done": True, "over": True},
            "termination_reason": reward_type,
            "final_reward": terminal_utility,
            "reward_version": "shopsimulator-reward-v3",
            "reward_type": reward_type,
            "reward_valid": True,
            "reward_detail": {
                "hard_gates": {
                    "category": {"passed": reward_type != "wrong_purchase"},
                    "budget": {"passed": reward_type != "wrong_purchase"},
                },
                "weighted_score": 1.0 if reward_type == "gold_purchase" else 0.5,
                "evidence_coverage": 1.0,
                "dimension_scores": {
                    "brand": 1.0,
                    "model": 1.0,
                    "core_functions": 1.0,
                    "key_options": 1.0,
                },
            },
            "guard_rejection_count": guard_rejections,
            "repeat_action_count": repeat_actions,
            "action_attempt_count": max(guard_rejections + repeat_actions, 1),
        }
    )
    return state


class ShoppingPolicyRewardTest(unittest.TestCase):
    def test_clean_gold_preserves_reward_v3_utility(self):
        result = reward_breakdown(terminal_state())

        self.assertEqual(result["policy_reward_version"], POLICY_REWARD_VERSION)
        self.assertEqual(result["terminal_utility"], 1.0)
        self.assertEqual(result["total"], 1.0)
        self.assertEqual(result["strict"], 1.0)
        self.assertTrue(result["valid_for_learning"])
        self.assertFalse(result["sampling_invalid"])

    def test_behavior_penalties_are_bounded_and_do_not_change_raw_utility(self):
        result = reward_breakdown(
            terminal_state(guard_rejections=7, repeat_actions=9)
        )

        self.assertEqual(result["terminal_utility"], 1.0)
        self.assertAlmostEqual(result["penalty_guard"], 0.09)
        self.assertAlmostEqual(result["penalty_repeat"], 0.06)
        self.assertAlmostEqual(result["total"], 0.85)

    def test_negative_terminal_reward_is_clipped_after_behavior_penalties(self):
        result = reward_breakdown(
            terminal_state(
                reward_type="wrong_purchase",
                terminal_utility=-0.85,
                guard_rejections=3,
                repeat_actions=3,
            )
        )

        self.assertEqual(result["terminal_utility"], -0.85)
        self.assertEqual(result["total"], -1.0)
        self.assertTrue(result["valid_for_learning"])

    def test_model_caused_terminations_are_valid_negative_samples(self):
        expected = {
            "assistant_finished_without_environment_done": -0.40,
            "max_steps": -0.50,
            "context_hard_limit_exceeded": -0.55,
            "parallel_tool_calls": -0.60,
            "too_many_guard_rejections": -0.70,
        }
        for reason, value in expected.items():
            with self.subTest(reason=reason):
                state = make_runtime_state(task_id=1, max_steps=35)
                state["termination_reason"] = reason
                state["error"] = reason
                state["guard_rejection_count"] = 3
                state["repeat_action_count"] = 3
                result = reward_breakdown(state)
                self.assertEqual(result["total"], value)
                self.assertEqual(result["penalty_guard"], 0.0)
                self.assertEqual(result["penalty_repeat"], 0.0)
                self.assertTrue(result["model_failure"])
                self.assertTrue(result["valid_for_learning"])
                self.assertFalse(result["sampling_invalid"])

    def test_infrastructure_and_unverifiable_rewards_fail_closed(self):
        infrastructure = make_runtime_state(task_id=1, max_steps=35)
        infrastructure["infrastructure_invalid"] = True
        infrastructure["error"] = "tool_error:Timeout"
        invalid = reward_breakdown(infrastructure)
        self.assertEqual(invalid["total"], 0.0)
        self.assertFalse(invalid["valid_for_learning"])
        self.assertTrue(invalid["sampling_invalid"])

        unverifiable = terminal_state()
        unverifiable["reward_valid"] = False
        unverifiable["reward_unverifiable"] = True
        invalid = reward_breakdown(unverifiable)
        self.assertEqual(invalid["terminal_utility"], 1.0)
        self.assertEqual(invalid["total"], 0.0)
        self.assertEqual(invalid["invalid_reason"], "reward_unverifiable")

    def test_unknown_nonterminal_failure_is_not_silently_trained(self):
        state = make_runtime_state(task_id=1, max_steps=35)
        state["termination_reason"] = "unexpected_framework_exit"
        state["error"] = state["termination_reason"]
        result = reward_breakdown(state)
        self.assertTrue(result["sampling_invalid"])
        self.assertFalse(result["model_failure"])

    def test_terminal_reward_modes_are_explicit(self):
        state = terminal_state(guard_rejections=1)
        self.assertEqual(terminal_reward(state, mode="native"), 1.0)
        self.assertEqual(terminal_reward(state, mode="policy_v1"), 0.97)
        with self.assertRaisesRegex(ValueError, "unknown shopping reward mode"):
            terminal_reward(state, mode="constraint_aware")

    def test_policy_config_rejects_unsafe_values(self):
        with self.assertRaisesRegex(ValueError, "version"):
            validate_policy_reward_config({"version": "unknown"})
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_policy_reward_config({"guard_rejection_penalty": -0.1})
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_policy_reward_config(
                {"model_failure_rewards": {"max_steps": -2.0}}
            )

    def test_repeat_detection_and_action_trace_use_only_public_inputs(self):
        state = make_runtime_state(task_id=1, max_steps=35)
        record_action_attempt(state, "search_products", {"query": "mug"}, "page")
        record_action_attempt(state, "open_product", {"asin": "123"}, "page")
        event = record_action_attempt(
            state,
            "search_products",
            {"query": "mug"},
            "page",
        )

        self.assertEqual(state["repeat_action_count"], 1)
        self.assertTrue(event["repeated"])
        self.assertIn("observation_sha256", event)
        self.assertNotIn("goal", event)

    def test_think_is_not_an_environment_action_attempt(self):
        state = make_runtime_state(task_id=1, max_steps=35)
        self.assertIsNone(record_action_attempt(state, "think", {}, "page"))
        self.assertEqual(state["action_attempt_count"], 0)
        self.assertEqual(state["action_events"], [])


if __name__ == "__main__":
    unittest.main()
