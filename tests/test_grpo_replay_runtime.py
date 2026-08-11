import unittest

from shopping_grpo.training.grpo.adapter.runtime import (
    make_runtime_state,
    record_action_attempt,
    record_action_outcome,
    record_non_environment_decision,
    record_replay_transition,
)


def replay_ready_state():
    state = make_runtime_state(task_id=3, max_steps=35)
    state["latest_observation_raw"] = "public page"
    state["environment_manifest_sha256"] = "a" * 64
    state["public_query_sha256"] = "b" * 64
    state["replay_observation_v2_complete"] = True
    state["current_assistant_turn_id"] = 4
    state["assistant_turn_records"].append(
        {
            "turn_id": 4,
            "actor_prompt_sha256": "c" * 64,
            "tokenizer_contract_sha256": "d" * 64,
        }
    )
    return state


class ReplayRuntimeTest(unittest.TestCase):
    def test_think_is_a_policy_decision_but_not_an_environment_action(self):
        state = replay_ready_state()
        event = record_non_environment_decision(
            state,
            "think",
            {"note": "inspect"},
            "projected page",
        )
        self.assertEqual(event["decision_kind"], "think")
        self.assertTrue(event["branch_identity_complete"])
        self.assertEqual(state["action_attempt_count"], 0)
        self.assertEqual(state["action_events"], [])
        self.assertEqual(state["replay_ledger"], [])

    def test_assistant_final_is_retained_as_a_branch_alternative(self):
        state = replay_ready_state()
        event = record_non_environment_decision(
            state,
            "assistant_final",
            {},
            "projected page",
        )
        self.assertEqual(event["tool"], "assistant_final")
        self.assertEqual(event["prefix_action_count"], 0)
        self.assertEqual(event["assistant_turn_id"], 4)

    def test_audit_instrumentation_failure_does_not_abort_the_action(self):
        state = replay_ready_state()
        event = record_action_attempt(
            state,
            "search_products",
            {"query": "q" * (17 * 1024)},
            "projected page",
        )
        self.assertIsNotNone(event)
        self.assertIsNone(event["action_sha256"])
        self.assertFalse(event["branch_identity_complete"])
        self.assertEqual(state["action_attempt_count"], 1)
        self.assertEqual(state["replay_contract_error"], "invalid_exact_action_parameters")

    def test_exact_decision_parameters_are_separate_from_bounded_action_trace(self):
        state = replay_ready_state()
        query = "q" * 300
        event = record_action_attempt(
            state,
            "search_products",
            {"query": query},
            "projected page",
        )
        record_action_outcome(state, event, accepted=True)
        record_replay_transition(
            state,
            event,
            parameters={"query": query},
            after_observation="public results",
            done=False,
        )
        self.assertEqual(event["replay_parameters"]["query"], query)
        self.assertEqual(len(event["parameters"]["query"]), 256)
        self.assertEqual(len(state["action_events"][0]["parameters"]["query"]), 256)
        self.assertNotIn("replay_parameters", state["action_events"][0])
        self.assertTrue(state["action_events"][0]["accepted"])
        self.assertEqual(state["replay_ledger"][0]["parameters"]["query"], query)


if __name__ == "__main__":
    unittest.main()
