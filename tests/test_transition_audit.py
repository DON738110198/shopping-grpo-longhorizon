import unittest

from shopping_grpo.evaluation.transition_audit import (
    audit_strict_transitions,
    render_transition_audit_markdown,
)


def tool_call(name, parameters):
    return {
        "id": f"call-{name}",
        "type": "function",
        "function": {"name": name, "arguments": parameters},
    }


def step(index, name, parameters):
    return {
        "step_index": index,
        "tool_call": tool_call(name, parameters),
        "tool_name": name,
        "parameters": parameters,
        "reward": 0.0,
        "done": False,
    }


def trajectory(
    task_id,
    *,
    reward_type,
    steps=None,
    blocked=None,
    purchased_asin=None,
    termination_reason=None,
):
    strict = reward_type == "gold_purchase"
    return {
        "task_id": task_id,
        "trajectory_id": f"trajectory-{task_id}",
        "status": "done" if strict or reward_type != "assistant_final" else "assistant_final",
        "done": strict or reward_type not in {"assistant_final"},
        "initial_result": {"instruction": f"Instruction: public query {task_id}"},
        "steps": steps or [],
        "blocked_tool_calls": blocked or [],
        "terminal_result": {
            "done": strict or reward_type not in {"assistant_final"},
            "over": strict or reward_type not in {"assistant_final"},
            "purchase": {"asin": purchased_asin} if purchased_asin else {},
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v3",
                "reward_type": reward_type,
                "reward_valid": True,
                "purchase_success": strict,
                "termination_reason": termination_reason or reward_type,
            },
        },
    }


class StrictTransitionAuditTest(unittest.TestCase):
    def test_audits_recovery_gain_and_purchase_selection_loss(self):
        source = [
            trajectory(
                1,
                reward_type="repeat_loop",
                steps=[
                    step(0, "open_product", {"asin": "A"}),
                    step(1, "select_option", {"value": "small"}),
                ],
            ),
            trajectory(
                2,
                reward_type="gold_purchase",
                purchased_asin="B",
                steps=[
                    step(0, "open_product", {"asin": "B"}),
                    step(1, "select_option", {"value": "white"}),
                    step(2, "buy_now", {}),
                ],
            ),
        ]
        target = [
            trajectory(
                1,
                reward_type="gold_purchase",
                purchased_asin="A",
                steps=[
                    step(0, "open_product", {"asin": "A"}),
                    step(1, "select_option", {"value": "small"}),
                    step(2, "buy_now", {}),
                ],
            ),
            trajectory(
                2,
                reward_type="partial_alternative_purchase",
                purchased_asin="B",
                steps=[
                    step(0, "open_product", {"asin": "B"}),
                    step(1, "select_option", {"value": "white"}),
                    step(2, "select_option", {"value": "black"}),
                    step(3, "buy_now", {}),
                ],
            ),
        ]

        audit = audit_strict_transitions(
            expected_task_ids=[1, 2],
            source_trajectories=source,
            target_trajectories=target,
            source_label="sft",
            target_label="grpo",
        )

        self.assertEqual(audit["aggregate"]["gains"]["cases"], 1)
        self.assertEqual(audit["aggregate"]["losses"]["cases"], 1)
        gain, loss = audit["cases"]
        self.assertEqual(
            gain["diagnostic_bucket"],
            "reached_reference_candidate_and_option_but_failed_to_commit",
        )
        self.assertEqual(loss["diagnostic_bucket"], "purchase_selection_regression")
        self.assertEqual(loss["failure_bridge"]["matching_reference_options"], ["white"])
        self.assertIn("Task 2", render_transition_audit_markdown(audit))

    def test_guard_rejected_open_is_not_counted_as_reached(self):
        source = [
            trajectory(
                3,
                reward_type="gold_purchase",
                purchased_asin="C",
                steps=[step(0, "open_product", {"asin": "C"}), step(1, "buy_now", {})],
            )
        ]
        target = [
            trajectory(
                3,
                reward_type="repeat_loop",
                blocked=[
                    {
                        "step_index": 0,
                        "tool_call": tool_call("open_product", {"asin": "C"}),
                        "reason": "click_not_in_previous_observation",
                    }
                ],
            )
        ]

        audit = audit_strict_transitions(
            expected_task_ids=[3],
            source_trajectories=source,
            target_trajectories=target,
            source_label="sft",
            target_label="grpo",
        )

        case = audit["cases"][0]
        self.assertFalse(case["failure_bridge"]["failure_reached_reference_asin"])
        self.assertEqual(
            case["diagnostic_bucket"], "failed_to_reach_reference_candidate"
        )
        self.assertTrue(case["target"]["action_trace"][0].startswith("REJECTED"))


if __name__ == "__main__":
    unittest.main()
