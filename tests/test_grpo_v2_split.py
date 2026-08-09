"""Tests for the deterministic GRPO-v2 holdout split."""

import unittest

from scripts.build_grpo_v2_split import split_rows_by_length_bucket


class GrpoV2SplitTest(unittest.TestCase):
    def test_split_is_exact_deterministic_and_disjoint(self):
        rows = [
            {
                "task_id": index,
                "length_bucket": "short" if index < 60 else "long",
            }
            for index in range(100)
        ]
        train_a, tuning_a = split_rows_by_length_bucket(
            rows,
            tuning_size=15,
            seed=20260809,
        )
        train_b, tuning_b = split_rows_by_length_bucket(
            list(reversed(rows)),
            tuning_size=15,
            seed=20260809,
        )
        train_ids = {row["task_id"] for row in train_a}
        tuning_ids = {row["task_id"] for row in tuning_a}
        self.assertEqual(len(train_a), 85)
        self.assertEqual(len(tuning_a), 15)
        self.assertFalse(train_ids & tuning_ids)
        self.assertEqual(tuning_ids, {row["task_id"] for row in tuning_b})
        self.assertEqual(train_ids, {row["task_id"] for row in train_b})
        self.assertEqual(
            sum(row["length_bucket"] == "short" for row in tuning_a),
            9,
        )

    def test_invalid_holdout_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "tuning_size"):
            split_rows_by_length_bucket(
                [{"task_id": 1, "length_bucket": "short"}],
                tuning_size=1,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
