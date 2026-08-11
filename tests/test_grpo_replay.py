import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_pivotal_replay import _load_validated_manifest
from shopping_grpo.training.grpo.pivotal_states import observation_sha256
from shopping_grpo.training.grpo.replay import (
    verify_replay_trajectory,
    verify_replay_with_factory,
)

PRODUCT_ID = "123456789012"


def observation_state(*, page_type="search_home", divergent=False):
    if page_type == "search_results":
        return {
            "observation_version": "shopping-observation-v2",
            "page_type": "search_results",
            "search_available": False,
            "actions": [PRODUCT_ID],
            "query": "mug",
            "normalized_query": "mug",
            "page": 1,
            "total_pages": 1,
            "total_results": 1,
            "rank_start": 1,
            "rank_end": 1,
            "products": [
                {
                    "rank": 1,
                    "asin": PRODUCT_ID,
                    "price": "10",
                    "brand": "B",
                    "category": "mug",
                    "key_attributes": [],
                    "title": "Changed mug" if divergent else "Mug",
                }
            ],
        }
    return {
        "observation_version": "shopping-observation-v2",
        "page_type": "search_home",
        "search_available": True,
        "actions": [],
    }


def replay_trajectory(query="public task"):
    from shopping_grpo.environment.observation import render_structured_observation

    initial_hash = observation_sha256(render_structured_observation(observation_state()))
    results_hash = observation_sha256(
        render_structured_observation(observation_state(page_type="search_results"))
    )
    return {
        "task_id": 17,
        "environment_manifest_sha256": "a" * 64,
        "environment_version": "shopsimulator-environment-v2.1",
        "public_query_sha256": observation_sha256(query),
        "initial_public_observation_sha256": initial_hash,
        "replay_ledger": [
            {
                "sequence": 0,
                "tool": "search_products",
                "parameters": {"query": "mug"},
                "before_public_observation_sha256": initial_hash,
                "after_public_observation_sha256": results_hash,
                "done": False,
            },
            {
                "sequence": 1,
                "tool": "open_product",
                "parameters": {"asin": PRODUCT_ID},
                "before_public_observation_sha256": results_hash,
                "after_public_observation_sha256": None,
                "done": True,
            },
        ],
    }


class FakeReplayEnvironment:
    def __init__(self, *, divergent=False):
        self.divergent = divergent
        self.actions = []

    def reset(self, task_id):
        self.task_id = task_id
        return {
            "instruction": "public task",
            "observation_state": observation_state(),
            "environment_version": "shopsimulator-environment-v2.1",
            "environment_manifest_sha256": "a" * 64,
            "goal_options": {"secret": "never serialize me"},
        }

    def step(self, action):
        self.actions.append(action)
        if len(self.actions) == 1:
            return {
                "instruction": "public results",
                "observation_state": observation_state(
                    page_type="search_results",
                    divergent=self.divergent,
                ),
                "done": False,
                "goal": "never serialize me",
            }
        return {
            "instruction": "hidden terminal",
            "done": True,
            "reward_detail": {"target_asin": "never serialize me"},
        }

    def release(self):
        self.released = True


class ReplayVerificationTest(unittest.TestCase):
    def test_live_replay_cli_rejects_an_unvalidated_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "environment.json"
            path.write_text('{"environment_version":"not-enough"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid environment manifest"):
                _load_validated_manifest(path)

    def test_exact_replay_verifies_each_public_transition_without_hidden_fields(self):
        env = FakeReplayEnvironment()
        result = verify_replay_trajectory(
            env,
            replay_trajectory(),
            environment_manifest_sha256="a" * 64,
        )
        self.assertTrue(result["verified"])
        self.assertEqual(result["verified_transitions"], 2)
        self.assertEqual(env.actions, ["search[mug]", f"click[{PRODUCT_ID}]"])
        self.assertNotIn("never serialize me", json.dumps(result))

    def test_replay_reports_first_public_state_divergence(self):
        result = verify_replay_trajectory(
            FakeReplayEnvironment(divergent=True),
            replay_trajectory(),
            environment_manifest_sha256="a" * 64,
        )
        self.assertFalse(result["verified"])
        self.assertEqual(result["reason"], "after_public_observation_hash_mismatch")
        self.assertEqual(result["first_divergence_index"], 0)

    def test_manifest_mismatch_fails_before_reset(self):
        class MustNotReset:
            def reset(self, task_id):  # pragma: no cover - a failure would call this.
                raise AssertionError(task_id)

        result = verify_replay_trajectory(
            MustNotReset(),
            replay_trajectory(),
            environment_manifest_sha256="b" * 64,
        )
        self.assertEqual(result["stage"], "manifest")
        self.assertEqual(result["verified_transitions"], 0)

    def test_invalid_ledger_fails_before_reset(self):
        candidate = replay_trajectory()
        candidate["replay_ledger"][1]["before_public_observation_sha256"] = "c" * 64
        result = verify_replay_trajectory(
            FakeReplayEnvironment(),
            candidate,
            environment_manifest_sha256="a" * 64,
        )
        self.assertEqual(result["stage"], "contract")
        self.assertEqual(result["reason"], "transition_1_before_hash_mismatch")

    def test_guard_rejection_never_steps_the_environment(self):
        candidate = replay_trajectory()
        candidate["replay_ledger"][1]["parameters"] = {"asin": "999999999999"}
        env = FakeReplayEnvironment()
        result = verify_replay_trajectory(
            env,
            candidate,
            environment_manifest_sha256="a" * 64,
        )
        self.assertFalse(result["verified"])
        self.assertEqual(
            result["reason"],
            "guard_rejection:click_not_in_previous_observation",
        )
        self.assertEqual(len(env.actions), 1)

    def test_missing_trajectory_environment_version_fails_before_reset(self):
        candidate = replay_trajectory()
        del candidate["environment_version"]

        class MustNotReset:
            def reset(self, task_id):  # pragma: no cover - a failure would call this.
                raise AssertionError(task_id)

        result = verify_replay_trajectory(
            MustNotReset(),
            candidate,
            environment_manifest_sha256="a" * 64,
        )
        self.assertEqual(result["stage"], "contract")
        self.assertEqual(result["reason"], "trajectory_environment_version_mismatch")

    def test_reset_environment_version_is_checked_against_trusted_requirement(self):
        class WrongVersion(FakeReplayEnvironment):
            def reset(self, task_id):
                result = super().reset(task_id)
                result["environment_version"] = "unsupported-environment"
                return result

        result = verify_replay_trajectory(
            WrongVersion(),
            replay_trajectory(),
            environment_manifest_sha256="a" * 64,
            required_environment_version="shopsimulator-environment-v2.1",
        )
        self.assertEqual(result["stage"], "reset")
        self.assertEqual(result["reason"], "environment_version_mismatch")

    def test_server_must_attest_the_local_environment_manifest(self):
        class WrongManifest(FakeReplayEnvironment):
            def reset(self, task_id):
                result = super().reset(task_id)
                result["environment_manifest_sha256"] = "b" * 64
                return result

        result = verify_replay_trajectory(
            WrongManifest(),
            replay_trajectory(),
            environment_manifest_sha256="a" * 64,
        )
        self.assertEqual(result["stage"], "reset")
        self.assertEqual(result["reason"], "server_environment_manifest_mismatch")

    def test_invalid_tool_schema_fails_before_environment_factory(self):
        candidate = replay_trajectory()
        candidate["replay_ledger"][0]["parameters"] = {
            "query": "mug",
            "hidden_extra": "bad",
        }

        def must_not_create():  # pragma: no cover - a failure would call this.
            raise AssertionError("factory called")

        result = verify_replay_with_factory(
            candidate,
            environment_manifest_sha256="a" * 64,
            env_factory=must_not_create,
        )
        self.assertEqual(result["stage"], "contract")
        self.assertEqual(result["reason"], "transition_0_schema_extra_arguments")

    def test_replay_over_max_steps_fails_before_environment_factory(self):
        candidate = replay_trajectory()

        def must_not_create():  # pragma: no cover - a failure would call this.
            raise AssertionError("factory called")

        result = verify_replay_with_factory(
            candidate,
            environment_manifest_sha256="a" * 64,
            required_max_steps=1,
            env_factory=must_not_create,
        )
        self.assertEqual(result["stage"], "contract")
        self.assertEqual(result["reason"], "replay_ledger_exceeds_max_steps")

    def test_prefix_replay_stops_before_terminal_and_recomputes_state_id(self):
        result = verify_replay_trajectory(
            FakeReplayEnvironment(),
            replay_trajectory(),
            environment_manifest_sha256="a" * 64,
            prefix_action_count=1,
        )
        self.assertTrue(result["verified"])
        self.assertEqual(result["verified_transitions"], 1)
        self.assertEqual(len(result["recomputed_replay_state_id"]), 64)

    def test_factory_wrapper_releases_environment(self):
        environments = []

        def factory():
            environment = FakeReplayEnvironment()
            environments.append(environment)
            return environment

        result = verify_replay_with_factory(
            replay_trajectory(),
            environment_manifest_sha256="a" * 64,
            env_factory=factory,
            prefix_action_count=1,
        )
        self.assertTrue(result["verified"])
        self.assertTrue(result["release_ok"])
        self.assertTrue(environments[0].released)

    def test_release_failure_invalidates_a_successful_replay(self):
        class ReleaseFailure(FakeReplayEnvironment):
            def release(self):
                raise RuntimeError("hidden server detail")

        result = verify_replay_with_factory(
            replay_trajectory(),
            environment_manifest_sha256="a" * 64,
            env_factory=ReleaseFailure,
            prefix_action_count=1,
        )
        self.assertFalse(result["verified"])
        self.assertEqual(result["reason"], "release_error:RuntimeError")
        self.assertNotIn("hidden server detail", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
