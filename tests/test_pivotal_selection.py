import copy
import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.select_pivotal_states import main as select_main
from scripts.verify_pivotal_replay import main as verify_main
from scripts.verify_pivotal_replay import verify_selected_replays
from shopping_grpo.training.grpo.active_branch import (
    build_active_branch_plan,
    observation_policy_sha256,
    validate_active_branch_plan,
)
from shopping_grpo.training.grpo.pivotal_states import (
    ACTOR_PROMPT_TOKENS_VERSION,
    REPLAY_STATE_VERSION,
    TURN_SPAN_VERSION,
    branch_uid,
    canonical_replay_action,
    observation_sha256,
    replay_action_sha256,
    replay_state_id,
    token_ids_sha256,
)
from shopping_grpo.training.grpo.selection import (
    SEARCH_DECISION_SELECTION_STRATEGY,
    resolve_pivotal_selection,
    select_pivotal_states,
    validate_pivotal_selection,
)


def exact_trajectory(
    task_id: int,
    prompt_salt: int,
    *,
    capture_prompts: bool = False,
) -> dict[str, object]:
    manifest_hash = "a" * 64
    query_hash = observation_sha256(f"public task {task_id}")
    tokenizer_hash = "b" * 64
    observation_hashes = [
        observation_sha256(f"{task_id}:home"),
        observation_sha256(f"{task_id}:results"),
        observation_sha256(f"{task_id}:product"),
    ]
    actions = [
        canonical_replay_action("search_products", {"query": "red mug"}),
        canonical_replay_action("open_product", {"asin": "123456789012"}),
        canonical_replay_action("buy_now", {}),
    ]
    ledger = []
    events = []
    spans = []
    for index, action in enumerate(actions):
        done = index + 1 == len(actions)
        after_hash = None if done else observation_hashes[index + 1]
        ledger.append(
            {
                "sequence": index,
                "tool": action["tool"],
                "parameters": action["parameters"],
                "before_public_observation_sha256": observation_hashes[index],
                "after_public_observation_sha256": after_hash,
                "done": done,
            }
        )
        state_id = replay_state_id(
            task_id,
            actions[:index],
            observation_hashes[index],
            observation_kind="raw_public_observation",
            environment_manifest_sha256=manifest_hash,
            public_query_sha256=query_hash,
        )
        prompt_tokens = [task_id, prompt_salt, index + 1]
        prompt_hash = (
            token_ids_sha256(prompt_tokens)
            if capture_prompts
            else f"{prompt_salt * 10 + index + 1:064x}"
        )
        event = {
            "decision_index": index,
            "decision_kind": "environment_tool",
            "tool": action["tool"],
            "parameters": action["parameters"],
            "replay_parameters": action["parameters"],
            "action_sha256": replay_action_sha256(
                action["tool"],
                action["parameters"],
            ),
            "observation_sha256": observation_hashes[index],
            "raw_observation_sha256": observation_hashes[index],
            "replay_state_id": state_id,
            "replay_state_version": REPLAY_STATE_VERSION,
            "branch_uid": branch_uid(state_id, prompt_hash, tokenizer_hash),
            "branch_identity_complete": True,
            "actor_prompt_sha256": prompt_hash,
            "tokenizer_contract_sha256": tokenizer_hash,
            "prefix_action_count": index,
            "assistant_turn_id": index,
            "accepted": True,
            "repeated": False,
            "guard_reason": None,
            "error": None,
            "index": index,
        }
        if capture_prompts:
            event["actor_prompt_tokens"] = {
                "version": ACTOR_PROMPT_TOKENS_VERSION,
                "sha256": prompt_hash,
                "count": len(prompt_tokens),
                "tokens": prompt_tokens,
            }
        events.append(event)
        spans.append(
            {
                "turn_id": index,
                "kind": "tool_call",
                "tool_names": [action["tool"]],
                "tool_call_count": 1,
                "credit_eligible": True,
                "assistant_span": [index * 2, index * 2 + 1],
                "observation_span": [index * 2 + 1, index * 2 + 2],
            }
        )
    return {
        "task_id": task_id,
        "strict": 1.0,
        "terminal_utility": 1.0,
        "policy_reward": 1.0,
        "valid_for_learning": True,
        "invalid_reason": None,
        "replay_state_version": REPLAY_STATE_VERSION,
        "environment_manifest_sha256": manifest_hash,
        "environment_version": "shopsimulator-environment-v2.1",
        "public_query_sha256": query_hash,
        "initial_public_observation_sha256": observation_hashes[0],
        "replay_observation_v2_complete": True,
        "replay_contract_error": None,
        "replay_ledger": ledger,
        "turn_span_version": TURN_SPAN_VERSION,
        "turn_span_valid": True,
        "turn_spans": spans,
        "decision_trace": events,
    }


class OutcomePoison(Mapping[str, object]):
    forbidden = frozenset(
        {
            "strict",
            "utility",
            "terminal_utility",
            "policy_reward",
            "reward_type",
            "termination_reason",
            "model_failure",
            "valid_for_learning",
            "invalid_reason",
            "goal",
            "reward_detail",
        }
    )

    def __init__(self, values: Mapping[str, object]):
        self.values = values

    def __getitem__(self, key: str) -> object:
        if key in self.forbidden:
            raise AssertionError(f"selector read outcome field {key}")
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def get(self, key: str, default=None):
        if key in self.forbidden:
            raise AssertionError(f"selector read outcome field {key}")
        return self.values.get(key, default)


def record(step: int, trajectories: list[Mapping[str, object]]) -> dict[str, object]:
    return {
        "global_step": step,
        "generation_batch": 1,
        "uid": f"group-{step}",
        "trajectories": trajectories,
    }


PROVENANCE = [{"path": "/audit.jsonl", "sha256": "f" * 64}]


def active_decoding_config() -> dict[str, object]:
    config = {
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
    config["observation_policy_sha256"] = observation_policy_sha256(config)
    return config


class PivotalSelectionTest(unittest.TestCase):
    def test_search_decision_selector_is_stratified_fresh_and_outcome_blind(self):
        rows = [
            (
                0,
                "/audit.jsonl",
                task_id,
                record(
                    task_id,
                    [OutcomePoison(exact_trajectory(task_id, task_id, capture_prompts=True))],
                ),
            )
            for task_id in range(1, 7)
        ]

        result = select_pivotal_states(
            rows,
            seed=20260812,
            max_states=4,
            provenance=PROVENANCE,
            require_prompt_capture=True,
            strategy=SEARCH_DECISION_SELECTION_STRATEGY,
            label_quotas={
                "search_query_decision": 2,
                "search_result_open_decision": 2,
            },
            excluded_task_ids=[1, 2],
        )

        self.assertEqual(len(result["selections"]), 4)
        self.assertEqual(len({item["task_id"] for item in result["selections"]}), 4)
        self.assertFalse({1, 2} & {item["task_id"] for item in result["selections"]})
        self.assertEqual(
            Counter(item["pivotal_labels"][0] for item in result["selections"]),
            {"search_query_decision": 2, "search_result_open_decision": 2},
        )
        self.assertEqual(result["constraints"]["max_states_per_task"], 1)
        self.assertEqual(result["constraints"]["excluded_task_ids"], [1, 2])
        self.assertTrue(result["safety"]["outcome_blind"])
        resolved = resolve_pivotal_selection(
            result,
            rows,
            expected_inputs=PROVENANCE,
            require_prompt_capture=True,
        )
        self.assertEqual(len(resolved), 4)
        plan = build_active_branch_plan(
            resolved,
            actor_checkpoint_sha256="e" * 64,
            decoding_config=active_decoding_config(),
            seed=20260812,
            suffixes_per_state=4,
        )
        self.assertEqual(validate_active_branch_plan(plan), plan)

    def test_search_decision_selector_rejects_constraint_tampering(self):
        rows = [
            (
                0,
                "/audit.jsonl",
                task_id,
                record(task_id, [exact_trajectory(task_id, task_id, capture_prompts=True)]),
            )
            for task_id in range(1, 5)
        ]
        result = select_pivotal_states(
            rows,
            seed=20260812,
            max_states=2,
            provenance=PROVENANCE,
            require_prompt_capture=True,
            strategy=SEARCH_DECISION_SELECTION_STRATEGY,
            label_quotas={
                "search_query_decision": 1,
                "search_result_open_decision": 1,
            },
            excluded_task_ids=[9],
        )

        bad_hash = copy.deepcopy(result)
        bad_hash["constraints"]["excluded_task_ids_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "excluded task id hash"):
            validate_pivotal_selection(bad_hash, expected_inputs=PROVENANCE)

        bad_quota = copy.deepcopy(result)
        bad_quota["constraints"]["label_quotas"]["search_query_decision"] = 2
        with self.assertRaisesRegex(ValueError, "do not sum"):
            validate_pivotal_selection(bad_quota, expected_inputs=PROVENANCE)

    def test_selector_cli_writes_hashed_exact_source_locators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "sampling.jsonl"
            output_path = root / "selection.json"
            input_path.write_text(
                json.dumps(record(7, [exact_trajectory(4, 9, capture_prompts=True)])) + "\n",
                encoding="utf-8",
            )
            with patch.object(
                sys,
                "argv",
                [
                    "select_pivotal_states.py",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--seed",
                    "20260811",
                    "--max-states",
                    "50",
                    "--require-prompt-capture",
                ],
            ):
                select_main()

            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            source = artifact["selections"][0]["source"]
            self.assertEqual(source["path"], str(input_path.resolve()))
            self.assertEqual(source["input_sha256"], artifact["provenance"]["inputs"][0]["sha256"])
            self.assertEqual(
                (source["line"], source["trajectory_index"], source["event_index"]),
                (1, 0, 2),
            )
            self.assertTrue(artifact["constraints"]["require_prompt_capture"])

    def test_prompt_capture_requirement_filters_and_validates_candidates(self):
        rows = [
            (0, "/audit.jsonl", 1, record(1, [exact_trajectory(1, 1)])),
            (
                0,
                "/audit.jsonl",
                2,
                record(2, [exact_trajectory(2, 2, capture_prompts=True)]),
            ),
        ]

        result = select_pivotal_states(
            rows,
            seed=3407,
            max_states=10,
            provenance=PROVENANCE,
            require_prompt_capture=True,
        )

        self.assertEqual([item["task_id"] for item in result["selections"]], [2])
        self.assertTrue(result["constraints"]["require_prompt_capture"])
        self.assertEqual(
            result["aggregate"]["contract_error_counts"],
            {"missing_actor_prompt_tokens": 1},
        )
        resolved = resolve_pivotal_selection(
            result,
            rows,
            expected_inputs=PROVENANCE,
            require_prompt_capture=True,
        )
        self.assertEqual(len(resolved), 1)

        bad_rows = [
            (
                0,
                "/audit.jsonl",
                1,
                record(2, [exact_trajectory(2, 2, capture_prompts=True)]),
            )
        ]
        bad_event = bad_rows[0][3]["trajectories"][0]["decision_trace"][2]
        bad_event["actor_prompt_tokens"]["tokens"][-1] += 1
        rejected = select_pivotal_states(
            bad_rows,
            seed=3407,
            max_states=10,
            provenance=PROVENANCE,
            require_prompt_capture=True,
        )
        self.assertEqual(rejected["selections"], [])
        self.assertEqual(
            rejected["aggregate"]["contract_error_counts"],
            {"actor_prompt_token_hash_mismatch": 1},
        )

        capture_mutations = {
            "version": (
                lambda capture: capture.update({"version": "future-version"}),
                "actor_prompt_token_version_mismatch",
            ),
            "count": (
                lambda capture: capture.update({"count": capture["count"] + 1}),
                "actor_prompt_token_count_mismatch",
            ),
            "sha256": (
                lambda capture: capture.update({"sha256": "f" * 64}),
                "actor_prompt_token_hash_mismatch",
            ),
        }
        for name, (mutate, expected_reason) in capture_mutations.items():
            with self.subTest(capture_mutation=name):
                trajectory = exact_trajectory(3, 3, capture_prompts=True)
                mutate(trajectory["decision_trace"][2]["actor_prompt_tokens"])
                mutated = select_pivotal_states(
                    [(0, "/audit.jsonl", 1, record(3, [trajectory]))],
                    seed=3407,
                    max_states=10,
                    provenance=PROVENANCE,
                    require_prompt_capture=True,
                )
                self.assertEqual(mutated["selections"], [])
                self.assertEqual(
                    mutated["aggregate"]["contract_error_counts"],
                    {expected_reason: 1},
                )

    def test_capture_required_selection_resolves_into_valid_active_plan(self):
        rows = [
            (
                0,
                "/audit.jsonl",
                1,
                record(7, [exact_trajectory(7, 9, capture_prompts=True)]),
            )
        ]
        selection = select_pivotal_states(
            rows,
            seed=20260811,
            max_states=4,
            provenance=PROVENANCE,
            require_prompt_capture=True,
        )
        resolved = resolve_pivotal_selection(
            selection,
            rows,
            expected_inputs=PROVENANCE,
            require_prompt_capture=True,
        )

        plan = build_active_branch_plan(
            resolved,
            actor_checkpoint_sha256="e" * 64,
            decoding_config=active_decoding_config(),
            seed=20260811,
            suffixes_per_state=4,
        )

        self.assertEqual(plan["aggregate"]["active_groups"], 1)
        self.assertEqual(validate_active_branch_plan(plan), plan)

    def test_selection_is_outcome_blind_deduplicated_and_task_capped(self):
        repeated = exact_trajectory(1, 1)
        repeated["goal"] = "hidden-selection-secret"
        repeated["reward_detail"] = {"target_asin": "hidden-selection-secret"}
        for event in repeated["decision_trace"]:
            event["strict"] = 1.0
            event["policy_reward"] = 1.0
        repeated["decision_trace"] = [OutcomePoison(event) for event in repeated["decision_trace"]]
        rows = [
            (0, "/audit.jsonl", 1, record(1, [OutcomePoison(repeated)])),
            (0, "/audit.jsonl", 2, record(2, [OutcomePoison(repeated)])),
            (
                0,
                "/audit.jsonl",
                3,
                record(
                    3,
                    [
                        OutcomePoison(exact_trajectory(1, 2)),
                        OutcomePoison(exact_trajectory(1, 3)),
                        OutcomePoison(exact_trajectory(2, 4)),
                    ],
                ),
            ),
        ]

        result = select_pivotal_states(
            rows,
            seed=3407,
            max_states=10,
            provenance=PROVENANCE,
        )

        task_counts = Counter(item["task_id"] for item in result["selections"])
        self.assertEqual(task_counts, {1: 2, 2: 1})
        self.assertEqual(result["aggregate"]["duplicate_branch_occurrences"], 1)
        self.assertEqual(
            len({item["branch_uid"] for item in result["selections"]}),
            len(result["selections"]),
        )
        self.assertEqual(result["constraints"]["max_states_per_task"], 2)
        self.assertFalse(result["constraints"]["require_prompt_capture"])
        self.assertTrue(result["safety"]["outcome_blind"])
        self.assertNotIn("hidden-selection-secret", json.dumps(result))
        self.assertEqual(
            len(resolve_pivotal_selection(result, rows, expected_inputs=PROVENANCE)),
            3,
        )

    def test_selection_contract_rejects_duplicate_or_tampered_entries(self):
        rows = [
            (0, "/audit.jsonl", 1, record(1, [exact_trajectory(1, 1)])),
            (0, "/audit.jsonl", 2, record(2, [exact_trajectory(2, 2)])),
        ]
        result = select_pivotal_states(
            rows,
            seed=3407,
            max_states=10,
            provenance=PROVENANCE,
        )
        validated = validate_pivotal_selection(result, expected_inputs=PROVENANCE)
        self.assertEqual(len(validated), 2)

        bad_capture_mode = {
            **result,
            "constraints": {**result["constraints"], "require_prompt_capture": "yes"},
        }
        with self.assertRaisesRegex(TypeError, "require_prompt_capture"):
            validate_pivotal_selection(bad_capture_mode, expected_inputs=PROVENANCE)

        duplicate = {**result, "selections": [*result["selections"], result["selections"][0]]}
        duplicate["aggregate"] = {**result["aggregate"], "selected_branches": 3}
        with self.assertRaisesRegex(ValueError, "selection_index"):
            validate_pivotal_selection(duplicate, expected_inputs=PROVENANCE)

        tampered = {**result, "selections": [dict(item) for item in result["selections"]]}
        tampered["selections"][0]["strict"] = 1.0
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            validate_pivotal_selection(tampered, expected_inputs=PROVENANCE)

        top_level_secret = {**result, "strict": 1.0}
        with self.assertRaisesRegex(ValueError, "top-level fields"):
            validate_pivotal_selection(top_level_secret, expected_inputs=PROVENANCE)

        repeated_branch = {
            **result,
            "selections": [
                *result["selections"],
                {**result["selections"][0], "selection_index": 2},
            ],
            "aggregate": {**result["aggregate"], "selected_branches": 3},
        }
        with self.assertRaisesRegex(ValueError, "repeats branch_uid"):
            validate_pivotal_selection(repeated_branch, expected_inputs=PROVENANCE)

    def test_fixed_seed_is_invariant_to_record_iteration_order(self):
        rows = [
            (
                0,
                "/audit.jsonl",
                task_id,
                record(task_id, [exact_trajectory(task_id, task_id)]),
            )
            for task_id in range(1, 9)
        ]
        first = select_pivotal_states(
            rows,
            seed=20260811,
            max_states=5,
            provenance=PROVENANCE,
        )
        reversed_rows = select_pivotal_states(
            reversed(rows),
            seed=20260811,
            max_states=5,
            provenance=PROVENANCE,
        )
        self.assertEqual(first["selections"], reversed_rows["selections"])

    def test_resolver_fails_closed_on_locator_or_event_mismatch(self):
        rows = [(0, "/audit.jsonl", 1, record(1, [exact_trajectory(1, 1)]))]
        selection = select_pivotal_states(
            rows,
            seed=3407,
            max_states=10,
            provenance=PROVENANCE,
        )
        resolved = resolve_pivotal_selection(
            selection,
            rows,
            expected_inputs=PROVENANCE,
        )
        self.assertEqual(len(resolved), 1)
        self.assertEqual(
            resolved[0]["event"]["branch_uid"],
            selection["selections"][0]["branch_uid"],
        )

        tampered = {
            **selection,
            "selections": [
                {
                    **selection["selections"][0],
                    "source": {
                        **selection["selections"][0]["source"],
                        "event_index": 1,
                    },
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "deterministic outcome-blind selector"):
            resolve_pivotal_selection(tampered, rows, expected_inputs=PROVENANCE)

    def test_resolver_recomputes_the_outcome_blind_selection_algorithm(self):
        rows = [
            (
                0,
                "/audit.jsonl",
                1,
                record(
                    1,
                    [
                        exact_trajectory(1, 1),
                        exact_trajectory(1, 2),
                        exact_trajectory(1, 3),
                    ],
                ),
            )
        ]
        selected_one = select_pivotal_states(
            rows,
            seed=3407,
            max_states=1,
            provenance=PROVENANCE,
        )
        selected_two = select_pivotal_states(
            rows,
            seed=3407,
            max_states=2,
            provenance=PROVENANCE,
        )
        alternate = dict(selected_two["selections"][1])
        alternate["selection_index"] = 0
        tampered = {**selected_one, "selections": [alternate]}

        with self.assertRaisesRegex(ValueError, "deterministic outcome-blind selector"):
            resolve_pivotal_selection(tampered, rows, expected_inputs=PROVENANCE)

    def test_live_verifier_consumes_every_selected_branch_once_and_in_order(self):
        rows = [
            (0, "/audit.jsonl", 1, record(1, [exact_trajectory(1, 1)])),
            (0, "/audit.jsonl", 2, record(2, [exact_trajectory(2, 2)])),
        ]
        selection = select_pivotal_states(
            rows,
            seed=3407,
            max_states=10,
            provenance=PROVENANCE,
        )
        replay = Mock()
        replay.side_effect = lambda trajectory, **kwargs: {
            "verified": True,
            "stage": "complete",
            "reason": None,
            "recomputed_replay_state_id": next(
                item["replay_state_id"]
                for item in selection["selections"]
                if item["task_id"] == trajectory["task_id"]
            ),
            "release_ok": True,
        }
        with patch(
            "scripts.verify_pivotal_replay.verify_replay_with_factory",
            replay,
        ):
            results = verify_selected_replays(
                selection,
                rows,
                expected_inputs=PROVENANCE,
                environment_manifest_sha256="a" * 64,
                required_environment_version="shopsimulator-environment-v2.1",
                required_max_steps=35,
                env_factory=object,
            )

        self.assertEqual(replay.call_count, len(selection["selections"]))
        self.assertEqual(
            [item["branch_uid"] for item in results],
            [item["branch_uid"] for item in selection["selections"]],
        )
        self.assertEqual(
            [item["selection_index"] for item in results],
            list(range(len(results))),
        )

    def test_live_verifier_preflights_the_whole_selection_before_leasing(self):
        rows = [(0, "/audit.jsonl", 1, record(1, [exact_trajectory(1, 1)]))]
        selection = select_pivotal_states(
            rows,
            seed=3407,
            max_states=10,
            provenance=PROVENANCE,
        )
        selection["selections"][0]["source"]["line"] = 9
        factory = Mock()
        with self.assertRaisesRegex(ValueError, "deterministic outcome-blind selector"):
            verify_selected_replays(
                selection,
                rows,
                expected_inputs=PROVENANCE,
                environment_manifest_sha256="a" * 64,
                required_environment_version="shopsimulator-environment-v2.1",
                required_max_steps=35,
                env_factory=factory,
            )
        factory.assert_not_called()

    def test_live_verifier_writes_all_failures_then_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "sampling.jsonl"
            selection_path = root / "selection.json"
            output_path = root / "live.json"
            input_path.write_text(
                json.dumps(record(1, [exact_trajectory(1, 1)])) + "\n",
                encoding="utf-8",
            )
            digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
            provenance = [{"path": str(input_path.resolve()), "sha256": digest}]
            selection = select_pivotal_states(
                [(0, str(input_path.resolve()), 1, record(1, [exact_trajectory(1, 1)]))],
                seed=3407,
                max_states=10,
                provenance=provenance,
            )
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            failure = {
                "selection_index": 0,
                "task_id": 1,
                "branch_uid": selection["selections"][0]["branch_uid"],
                "expected_replay_state_id": selection["selections"][0]["replay_state_id"],
                "prefix_action_count": 2,
                "pivotal_labels": selection["selections"][0]["pivotal_labels"],
                "source": selection["selections"][0]["source"],
                "verification": {
                    "verified": False,
                    "stage": "transition",
                    "reason": "after_public_observation_hash_mismatch",
                },
            }
            manifest_path = Path(__file__).parents[1] / "data" / "environment.json"
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "verify_pivotal_replay.py",
                        "--selection",
                        str(selection_path),
                        "--input",
                        str(input_path),
                        "--environment-manifest",
                        str(manifest_path),
                        "--output",
                        str(output_path),
                    ],
                ),
                patch(
                    "scripts.verify_pivotal_replay.verify_selected_replays",
                    return_value=[failure],
                ),
                self.assertRaisesRegex(SystemExit, "1"),
            ):
                verify_main()

            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact["aggregate"]["selected_branches"], 1)
            self.assertEqual(artifact["aggregate"]["attempted_branches"], 1)
            self.assertEqual(artifact["aggregate"]["failed_branches"], 1)
            self.assertFalse(artifact["aggregate"]["all_selected_verified"])


if __name__ == "__main__":
    unittest.main()
