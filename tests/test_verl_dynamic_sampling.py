"""Unit tests for policy-reward group filtering and rollout audit records."""

import json
import tempfile
import unittest
from pathlib import Path

from shopping_grpo.training.grpo.dynamic_sampling import (
    aggregate_shopping_metrics,
    append_sampling_audit,
    extract_shopping_group_signals,
    select_reward_varying_groups,
)


def shopping_info(
    policy_reward,
    *,
    terminal_utility=0.0,
    strict=False,
    invalid=False,
    invalid_reason=None,
    termination_reason="assistant_finished_without_environment_done",
    model_failure=False,
    guards=0,
    repeats=0,
):
    return {
        "task_id": 7,
        "steps": 10,
        "done": bool(strict),
        "termination_reason": termination_reason,
        "reward_type": "gold_purchase" if strict else None,
        "infrastructure_invalid": invalid_reason == "infrastructure_invalid",
        "reward_unverifiable": invalid_reason == "reward_unverifiable",
        "valid_for_learning": not invalid,
        "invalid_reason": invalid_reason,
        "model_failure": model_failure,
        "guard_rejections": guards,
        "repeat_actions": repeats,
        "action_trace": [{"tool": "search_products", "accepted": True}],
        "reward": {
            "policy_reward_version": "shopping-policy-reward-v1",
            "policy_base": policy_reward,
            "full": float(strict),
            "strict": float(strict),
            "native": terminal_utility,
            "semantic": float(strict),
            "total": policy_reward,
            "efficiency": 0.0,
            "penalty_overlong": 0.0,
            "penalty_unfinished": 0.0,
            "penalty_guard": guards * 0.03,
            "penalty_repeat": repeats * 0.02,
            "repeat_action_rate": 0.0,
            "terminal_utility": terminal_utility,
            "purchase_success": float(strict),
            "sampling_invalid": invalid,
            "r_type": float(strict),
            "r_att": float(strict),
            "r_option": float(strict),
            "r_price": float(strict),
            "match_score": float(strict),
            "evidence_coverage": float(strict),
        },
    }


class RewardGroupSelectionTest(unittest.TestCase):
    def test_constant_policy_reward_group_is_dropped(self):
        indices, stats = select_reward_varying_groups(["a"] * 4, [0.0] * 4)
        self.assertEqual(indices, [])
        self.assertEqual(stats["all_zero_reward_group_count"], 1)
        self.assertEqual(stats["groups"][0]["drop_reason"], "constant_reward")

    def test_same_terminal_utility_but_different_policy_reward_is_kept(self):
        rewards = [1.0, 0.97, 0.94, 1.0]
        indices, stats = select_reward_varying_groups(
            ["a"] * 4,
            rewards,
            policy_rewards=rewards,
            terminal_utilities=[1.0] * 4,
            purchase_success=[True] * 4,
            sampling_invalid=[False] * 4,
        )
        self.assertEqual(indices, [0, 1, 2, 3])
        self.assertEqual(stats["all_zero_terminal_utility_group_count"], 0)
        self.assertTrue(stats["groups"][0]["reward_varying"])

    def test_model_failure_is_a_valid_negative_sample(self):
        infos = [
            shopping_info(-0.4, model_failure=True),
            shopping_info(1.0, terminal_utility=1.0, strict=True),
            shopping_info(-0.5, model_failure=True, termination_reason="max_steps"),
            shopping_info(0.55, terminal_utility=0.55),
        ]
        policy, utility, success, invalid, reasons = extract_shopping_group_signals(infos)
        indices, _ = select_reward_varying_groups(
            ["a"] * 4,
            policy,
            policy_rewards=policy,
            terminal_utilities=utility,
            purchase_success=success,
            sampling_invalid=invalid,
            sampling_invalid_reasons=reasons,
        )
        self.assertEqual(indices, [0, 1, 2, 3])
        self.assertEqual(invalid, [False] * 4)

    def test_true_invalid_member_drops_the_whole_group(self):
        infos = [shopping_info(0.0) for _ in range(4)]
        infos[1] = shopping_info(
            0.2,
            invalid=True,
            invalid_reason="infrastructure_invalid",
        )
        policy, utility, success, invalid, reasons = extract_shopping_group_signals(infos)
        indices, stats = select_reward_varying_groups(
            ["a"] * 4,
            [0.0, 0.2, 0.0, 0.0],
            policy_rewards=policy,
            terminal_utilities=utility,
            purchase_success=success,
            sampling_invalid=invalid,
            sampling_invalid_reasons=reasons,
        )
        self.assertEqual(indices, [])
        self.assertEqual(stats["groups"][0]["drop_reason"], "sampling_invalid")
        self.assertEqual(
            stats["sampling_invalid_reason_counts"]["infrastructure_invalid"],
            1,
        )

    def test_tensor_and_metadata_policy_reward_must_match(self):
        with self.assertRaisesRegex(ValueError, "policy reward mismatch"):
            select_reward_varying_groups(
                ["a"] * 4,
                [0.0, 0.2, 0.0, 0.0],
                policy_rewards=[0.0, 0.1, 0.0, 0.0],
            )

    def test_invalid_validity_flag_fails_closed(self):
        info = shopping_info(0.0, invalid=True, invalid_reason="reward_unverifiable")
        info["valid_for_learning"] = True
        with self.assertRaisesRegex(ValueError, "valid_for_learning"):
            extract_shopping_group_signals([info])

    def test_metrics_include_strict_behavior_and_model_failures(self):
        infos = [
            shopping_info(1.0, terminal_utility=1.0, strict=True),
            shopping_info(-0.4, model_failure=True, guards=2, repeats=1),
        ]
        metrics = aggregate_shopping_metrics(infos)
        self.assertEqual(metrics["reward/strict_mean"], 0.5)
        self.assertEqual(metrics["reward/terminal_utility_mean"], 0.5)
        self.assertEqual(metrics["trajectory/model_failure_rate"], 0.5)
        self.assertEqual(metrics["trajectory/guard_rejections_mean"], 1.0)
        self.assertEqual(metrics["trajectory/repeat_actions_mean"], 0.5)

    def test_sampling_audit_contains_actions_without_hidden_goal(self):
        infos = [shopping_info(-0.4, model_failure=True) for _ in range(4)]
        _, stats = select_reward_varying_groups(
            ["a"] * 4,
            [-0.4, -0.5, -0.4, -0.4],
            policy_rewards=[-0.4, -0.5, -0.4, -0.4],
            terminal_utilities=[0.0] * 4,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = append_sampling_audit(
                tmp,
                global_step=1,
                generation_batch=1,
                group_stats=stats,
                shopping_infos=infos,
            )
            record = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(record["global_step"], 1)
        self.assertEqual(record["trajectories"][0]["action_trace"][0]["tool"], "search_products")
        self.assertNotIn("goal", json.dumps(record))


if __name__ == "__main__":
    unittest.main()
