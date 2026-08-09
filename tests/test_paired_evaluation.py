import unittest

from shopping_grpo.evaluation.paired import compare_strict_success


def trajectory(task_id, strict):
    reward_type = "gold_purchase" if strict else "wrong_purchase"
    return {
        "task_id": task_id,
        "status": "done",
        "done": True,
        "terminal_result": {
            "done": True,
            "over": True,
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v3",
                "reward_type": reward_type,
                "reward_valid": True,
                "purchase_success": strict,
                "termination_reason": reward_type,
            },
        },
    }


class PairedEvaluationTest(unittest.TestCase):
    def test_reports_transitions_exact_mcnemar_and_paired_interval(self):
        result = compare_strict_success(
            expected_task_ids=[1, 2, 3, 4],
            source_trajectories=[
                trajectory(1, True),
                trajectory(2, True),
                trajectory(3, False),
                trajectory(4, False),
            ],
            target_trajectories=[
                trajectory(1, True),
                trajectory(2, False),
                trajectory(3, True),
                trajectory(4, True),
            ],
            source_label="sft",
            target_label="grpo",
            bootstrap_resamples=200,
            bootstrap_seed=7,
        )

        self.assertEqual(result["source"]["strict_successes"], 2)
        self.assertEqual(result["target"]["strict_successes"], 3)
        self.assertEqual(result["gains"], 2)
        self.assertEqual(result["losses"], 1)
        self.assertEqual(result["gain_task_ids"], [3, 4])
        self.assertEqual(result["loss_task_ids"], [2])
        self.assertEqual(result["mcnemar"]["discordant_pairs"], 3)
        self.assertEqual(result["mcnemar"]["p_value"], 1.0)
        self.assertEqual(
            result["strict_success_rate_delta_target_minus_source"], 0.25
        )
        interval = result["paired_confidence_interval"]
        self.assertLessEqual(interval["lower"], 0.25)
        self.assertGreaterEqual(interval["upper"], 0.25)

    def test_rejects_missing_or_duplicate_tasks(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            compare_strict_success(
                expected_task_ids=[1, 2],
                source_trajectories=[trajectory(1, True)],
                target_trajectories=[trajectory(1, True), trajectory(2, False)],
                source_label="source",
                target_label="target",
                bootstrap_resamples=10,
            )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            compare_strict_success(
                expected_task_ids=[1, 2],
                source_trajectories=[
                    trajectory(1, True),
                    trajectory(1, False),
                    trajectory(2, False),
                ],
                target_trajectories=[trajectory(1, True), trajectory(2, False)],
                source_label="source",
                target_label="target",
                bootstrap_resamples=10,
            )


if __name__ == "__main__":
    unittest.main()
