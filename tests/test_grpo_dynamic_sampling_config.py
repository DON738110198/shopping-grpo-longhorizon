"""CPU-only checks for the project dynamic-sampling configuration gate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_grpo_runtime import (
    PATCH_MARKER,
    compose_runtime_config,
    validate_capture_only,
    validate_dynamic_sampling,
    validate_training_memory_budget,
)


class DynamicSamplingConfigTest(unittest.TestCase):
    def test_capture_only_is_explicitly_disabled_by_default(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("grpo.yaml", "dapo.yaml"):
            source = (root / "configs" / name).read_text(encoding="utf-8")
            self.assertIn("shopping_capture_only:\n  enable: false", source)
        validate_capture_only(
            {
                "shopping_capture_only": {"enable": False},
                "shopping_dynamic_sampling": {"enable": True},
            }
        )

    def test_capture_only_requires_dynamic_audit_and_strict_schema(self):
        validate_capture_only(
            {
                "shopping_capture_only": {"enable": True},
                "shopping_dynamic_sampling": {"enable": True},
                "trainer": {
                    "val_before_train": False,
                    "test_freq": -1,
                    "val_only": False,
                },
                "reward_model": {"enable": False},
            }
        )

        with self.assertRaisesRegex(SystemExit, "requires shopping_dynamic_sampling"):
            validate_capture_only(
                {
                    "shopping_capture_only": {"enable": True},
                    "shopping_dynamic_sampling": {"enable": False},
                    "trainer": {
                        "val_before_train": False,
                        "test_freq": -1,
                        "val_only": False,
                    },
                    "reward_model": {"enable": False},
                }
            )

        with self.assertRaisesRegex(SystemExit, "unsupported keys: typo"):
            validate_capture_only(
                {
                    "shopping_capture_only": {"enable": False, "typo": True},
                    "shopping_dynamic_sampling": {"enable": True},
                }
            )

        with self.assertRaisesRegex(SystemExit, "must be a boolean"):
            validate_capture_only(
                {
                    "shopping_capture_only": {"enable": 1},
                    "shopping_dynamic_sampling": {"enable": True},
                }
            )
        with self.assertRaisesRegex(SystemExit, "config is required"):
            validate_capture_only({"shopping_dynamic_sampling": {"enable": True}})
        with self.assertRaisesRegex(SystemExit, "enable must be explicit"):
            validate_capture_only(
                {
                    "shopping_capture_only": {},
                    "shopping_dynamic_sampling": {"enable": True},
                }
            )
        with self.assertRaisesRegex(SystemExit, "must be an object"):
            validate_capture_only(
                {
                    "shopping_capture_only": False,
                    "shopping_dynamic_sampling": {"enable": True},
                }
            )

    def test_capture_only_rejects_validation_and_reward_model_work(self):
        base = {
            "shopping_capture_only": {"enable": True},
            "shopping_dynamic_sampling": {"enable": True},
            "trainer": {
                "val_before_train": False,
                "test_freq": -1,
                "val_only": False,
            },
            "reward_model": {"enable": False},
        }
        cases = (
            ("trainer.val_before_train=false", "trainer", "val_before_train", True),
            ("trainer.test_freq<=0", "trainer", "test_freq", 1),
            ("trainer.test_freq<=0", "trainer", "test_freq", "disabled"),
            ("trainer.val_only=false", "trainer", "val_only", True),
            ("reward_model.enable=false", "reward_model", "enable", True),
        )
        for expected, section, key, value in cases:
            with self.subTest(expected=expected, value=value):
                config = {
                    name: dict(payload) if isinstance(payload, dict) else payload
                    for name, payload in base.items()
                }
                config[section][key] = value
                with self.assertRaisesRegex(SystemExit, expected):
                    validate_capture_only(config)
        for expected, section, key in (
            ("trainer.val_before_train=false", "trainer", "val_before_train"),
            ("trainer.test_freq<=0", "trainer", "test_freq"),
            ("trainer.val_only=false", "trainer", "val_only"),
            ("reward_model.enable=false", "reward_model", "enable"),
        ):
            with self.subTest(expected=expected, missing=key):
                config = {
                    name: dict(payload) if isinstance(payload, dict) else payload
                    for name, payload in base.items()
                }
                del config[section][key]
                with self.assertRaisesRegex(SystemExit, expected):
                    validate_capture_only(config)

    def test_training_memory_budget_enforces_real_micro_batch_one(self):
        config = compose_runtime_config([])
        validate_training_memory_budget(config)
        self.assertEqual(config.data.max_response_length, 20480)
        self.assertEqual(config.actor_rollout_ref.rollout.max_model_len, 24576)
        self.assertFalse(config.actor_rollout_ref.actor.use_dynamic_bsz)
        self.assertEqual(
            config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu, 1
        )
        self.assertFalse(
            config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
        )
        self.assertEqual(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu, 1
        )
        self.assertFalse(config.actor_rollout_ref.ref.log_prob_use_dynamic_bsz)
        self.assertEqual(
            config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu, 1
        )

    def test_training_memory_budget_rejects_unsafe_overrides(self):
        unsafe_response = compose_runtime_config(["data.max_response_length=24576"])
        with self.assertRaisesRegex(SystemExit, "unsafe GRPO response budget"):
            validate_training_memory_budget(unsafe_response)

        dynamic_actor = compose_runtime_config(
            ["actor_rollout_ref.actor.use_dynamic_bsz=true"]
        )
        with self.assertRaisesRegex(SystemExit, "actor.use_dynamic_bsz must be false"):
            validate_training_memory_budget(dynamic_actor)

        dynamic_rollout_log_prob = compose_runtime_config(
            ["actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true"]
        )
        with self.assertRaisesRegex(
            SystemExit, "rollout.log_prob_use_dynamic_bsz must be false"
        ):
            validate_training_memory_budget(dynamic_rollout_log_prob)

    def test_hydra_overrides_resolve_project_top_level_config(self):
        config = compose_runtime_config(
            [
                "shopping_dynamic_sampling.enable=true",
                "shopping_dynamic_sampling.metric=seq_reward",
                "shopping_dynamic_sampling.max_num_gen_batches=3",
                "shopping_dynamic_sampling.max_consecutive_skipped_updates=10",
                "shopping_dynamic_sampling.reward_tolerance=1e-8",
            ]
        )
        self.assertTrue(config.shopping_dynamic_sampling.enable)
        self.assertEqual(config.shopping_dynamic_sampling.metric, "seq_reward")
        self.assertEqual(config.shopping_dynamic_sampling.max_num_gen_batches, 3)
        self.assertEqual(
            config.shopping_dynamic_sampling.max_consecutive_skipped_updates, 10
        )
        self.assertEqual(config.shopping_dynamic_sampling.reward_tolerance, 1.0e-8)
        self.assertTrue(config.algorithm.rollout_correction.bypass_mode)
        self.assertTrue(config.actor_rollout_ref.rollout.calculate_log_probs)

    def test_enabled_config_requires_installed_patch_marker(self):
        config = compose_runtime_config(["shopping_dynamic_sampling.enable=true"])
        with tempfile.TemporaryDirectory() as temp_dir:
            verl_source = Path(temp_dir) / "verl" / "__init__.py"
            trainer_source = verl_source.parent / "trainer" / "ppo" / "ray_trainer.py"
            trainer_source.parent.mkdir(parents=True)
            verl_source.write_text("", encoding="utf-8")
            trainer_source.write_text("# unpatched\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "patch marker is missing"):
                validate_dynamic_sampling(config, verl_source, {"verl": "0.8.0"})

            trainer_source.write_text(f"# {PATCH_MARKER}\n", encoding="utf-8")
            validate_dynamic_sampling(config, verl_source, {"verl": "0.8.0"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
