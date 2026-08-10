import json
import unittest

from shopping_grpo.training.sft.recovery import (
    build_recovery_rows,
    replay_recovery_row,
    split_recovery_rows,
)


def event(index, tool, parameters, *, accepted=True, guard_reason=None):
    return {
        "index": index,
        "tool": tool,
        "parameters": parameters,
        "accepted": accepted,
        "guard_reason": guard_reason,
        "error": None,
    }


def record(task_id, reward_type, trace, *, source="rollouts/1.jsonl", line=1):
    responses = [f"observation {index}" for index in range(len(trace) - 1)]
    responses.append("Environment terminated.")
    output = "assistant\n" + "".join(
        f"<tool_call>call</tool_call>user\n<tool_response>\n{response}\n"
        "</tool_response>\nassistant\n"
        for response in responses
    )
    termination = reward_type
    if reward_type == "unknown":
        termination = "assistant_finished_without_environment_done"
    return {
        "output": output,
        "shopping": {
            "task_id": task_id,
            "done": reward_type != "unknown",
            "reward_type": reward_type,
            "termination_reason": termination,
            "valid_for_learning": True,
            "infrastructure_invalid": False,
            "guard_rejections": sum(not item["accepted"] for item in trace),
            "repeat_actions": 0,
            "action_trace": trace,
        },
        "_source": {
            "path": source,
            "group": source,
            "step": 1,
            "line": line,
        },
    }


def gold_trace(asin="A"):
    return [
        event(0, "search_products", {"query": "product"}),
        event(1, "open_product", {"asin": asin}),
        event(2, "select_option", {"value": "small"}),
        event(3, "buy_now", {}),
    ]


class RecoverySftTest(unittest.TestCase):
    def test_mixed_group_builds_commit_only_supervision(self):
        gold = record(1, "gold_purchase", gold_trace(), line=1)
        failure = record(
            1,
            "repeat_loop",
            [
                event(0, "search_products", {"query": "product"}),
                event(1, "open_product", {"asin": "A"}),
                event(2, "select_option", {"value": "small"}),
                event(3, "view_features", {}),
            ],
            line=2,
        )

        rows, rejected, summary = build_recovery_rows(
            records=[gold, failure],
            prompts_by_task={
                1: [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "query"},
                ]
            },
        )

        self.assertFalse(rejected)
        self.assertEqual(summary["recovery_rows"], 1)
        row = rows[0]
        self.assertEqual(row["recovery"]["supervision_mode"], "commit_only")
        self.assertEqual(row["recovery"]["supervised_tools"], ["buy_now"])
        self.assertEqual(row["supervised_assistant_indices"], [8])
        self.assertNotIn("goal", json.dumps(row))

    def test_wrong_purchase_supervises_options_and_commit(self):
        gold = record(2, "gold_purchase", gold_trace("B"), source="rollouts/2.jsonl")
        failure = record(
            2,
            "partial_alternative_purchase",
            [
                event(0, "open_product", {"asin": "B"}),
                event(1, "select_option", {"value": "wrong"}),
                event(2, "buy_now", {}),
            ],
            source="rollouts/2.jsonl",
            line=2,
        )

        rows, _, _ = build_recovery_rows(
            records=[gold, failure],
            prompts_by_task={2: [{"role": "user", "content": "query"}]},
        )

        self.assertEqual(
            rows[0]["recovery"]["supervised_tools"],
            ["select_option", "buy_now"],
        )

    def test_split_preserves_task_disjointness(self):
        recovery = [{"task_id": task_id} for task_id in (1, 2, 3, 4)]
        split = split_recovery_rows(
            recovery_rows=recovery,
            base_train_rows=[{"task_id": 1}],
            base_validation_rows=[{"task_id": 2}],
            validation_ratio=0.5,
            seed=7,
        )

        train_ids = {row["task_id"] for row in split["mixed_train"]}
        validation_ids = {row["task_id"] for row in split["mixed_validation"]}
        self.assertIn(1, train_ids)
        self.assertIn(2, validation_ids)
        self.assertFalse(train_ids & validation_ids)

    def test_replay_requires_strict_gold_terminal(self):
        class FakeEnv:
            def __init__(self, base_url):
                self.base_url = base_url

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def reset(self, task_id):
                self.task_id = task_id

            def step(self, action):
                if action == "click[Buy Now]":
                    return {
                        "done": True,
                        "over": True,
                        "reward_detail": {
                            "reward_version": "shopsimulator-reward-v3",
                            "reward_type": "gold_purchase",
                            "reward_valid": True,
                            "purchase_success": True,
                            "termination_reason": "gold_purchase",
                        },
                    }
                return {"done": False}

        rows, _, _ = build_recovery_rows(
            records=[
                record(3, "gold_purchase", gold_trace("C"), source="rollouts/3.jsonl"),
                record(
                    3,
                    "repeat_loop",
                    [event(0, "open_product", {"asin": "C"})],
                    source="rollouts/3.jsonl",
                    line=2,
                ),
            ],
            prompts_by_task={3: [{"role": "user", "content": "query"}]},
        )

        result = replay_recovery_row(
            rows[0], base_url="http://shop", env_factory=FakeEnv
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["reward_type"], "gold_purchase")


if __name__ == "__main__":
    unittest.main()
