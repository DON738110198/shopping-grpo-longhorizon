"""Public CPU and parameterized GRPO entry-point tests."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_grpo_runtime import validate_agent_loop_config
from scripts.train_grpo import build_command, parse_args
from scripts.train_grpo import main as train_grpo_main
from shopping_grpo.cli import main as cli_main
from shopping_grpo.smoke import run_cpu_smoke


class PublicEntrypointTest(unittest.TestCase):
    def test_public_grpo_launcher_defaults_to_non_capture_agent_loop(self):
        root = Path(__file__).resolve().parents[1]
        with patch.object(sys, "argv", ["train_grpo.py"]):
            args = parse_args()
        self.assertEqual(
            args.agent_loop_config.resolve(),
            (root / "configs/agent_loop.yaml").resolve(),
        )

    def test_cpu_smoke_covers_public_contracts(self):
        result = run_cpu_smoke()

        self.assertEqual(
            result["checks"],
            [
                "action_schema",
                "trajectory_normalization",
                "reward_sample",
                "sft_label_mask",
                "dynamic_sampling_grouping",
            ],
        )

    def test_offline_example_cli_runs_without_models_or_environment(self):
        root = Path(__file__).resolve().parents[1]
        with patch.object(
            sys,
            "argv",
            [
                "shopping-grpo",
                "evaluate",
                str(root / "examples/trajectories.jsonl"),
            ],
        ), patch("builtins.print") as output:
            cli_main()

        summary = json.loads(output.call_args.args[0])
        self.assertEqual(summary["trajectory_count"], 3)
        self.assertEqual(summary["strict_gold_success_count"], 1)

    def test_public_grpo_launcher_accepts_sharded_weights_and_console(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors.index.json").write_text(
                "{}",
                encoding="utf-8",
            )
            train = temporary / "train.parquet"
            train.write_bytes(b"example")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"example")
            custom_config = temporary / "custom_grpo.yaml"
            custom_config.write_text(
                (root / "configs/grpo.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            output = temporary / "output"
            with patch.object(
                sys,
                "argv",
                [
                    "train_grpo.py",
                    "--model",
                    str(model),
                    "--train-data",
                    str(train),
                    "--val-data",
                    str(validation),
                    "--output",
                    str(output),
                    "--config",
                    str(custom_config),
                    "--agent-loop-config",
                    str(root / "configs/agent_loop_active_capture.yaml"),
                    "--logger",
                    "console",
                    "--dry-run",
                ],
            ):
                args = parse_args()
            command, environment = build_command(args)

        self.assertIn("verl.trainer.main_ppo", command)
        self.assertEqual(environment["GRPO_MODEL_PATH"], str(model))
        self.assertEqual(environment["GRPO_TRAIN_FILE"], str(train))
        self.assertEqual(environment["GRPO_VAL_FILE"], str(validation))
        self.assertEqual(environment["GRPO_CONFIG_DIR"], str(temporary.resolve()))
        self.assertEqual(
            environment["SHOPPING_AGENT_LOOP_CONFIG"],
            str((root / "configs/agent_loop_active_capture.yaml").resolve()),
        )
        self.assertEqual(environment["VLLM_USE_FLASHINFER_SAMPLER"], "0")
        self.assertIn("trainer.logger=[console]", command)
        self.assertIn("data.seed=42", command)
        self.assertIn("actor_rollout_ref.rollout.engine_kwargs.vllm.seed=42", command)

    def test_public_grpo_launcher_runs_preflight_before_training(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            train = temporary / "train.parquet"
            train.write_bytes(b"example")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"example")
            output = temporary / "output"
            argv = [
                "train_grpo.py",
                "--model",
                str(model),
                "--train-data",
                str(train),
                "--val-data",
                str(validation),
                "--output",
                str(output),
                "--config",
                str(root / "configs/grpo.yaml"),
                "--agent-loop-config",
                str(root / "configs/agent_loop_active_capture.yaml"),
                "--logger",
                "console",
                "--experiment-name",
                "preflight-regression",
                "--",
                "trainer.total_training_steps=1",
            ]
            with patch.object(sys, "argv", argv), patch(
                "scripts.train_grpo.subprocess.call",
                side_effect=[0, 0],
            ) as subprocess_call, patch(
                "scripts.train_grpo._write_run_evidence"
            ) as write_evidence, self.assertRaises(SystemExit) as completed:
                train_grpo_main()

            self.assertEqual(completed.exception.code, 0)
            self.assertEqual(subprocess_call.call_count, 2)
            preflight = subprocess_call.call_args_list[0].args[0]
            training = subprocess_call.call_args_list[1].args[0]
            for command in (preflight, training):
                self.assertIn("trainer.logger=[console]", command)
                self.assertIn(
                    "trainer.experiment_name=preflight-regression",
                    command,
                )
                self.assertIn("trainer.total_training_steps=1", command)
            self.assertIn("check_grpo_runtime.py", preflight[1])
            self.assertIn("verl.trainer.main_ppo", training)
            write_evidence.assert_called_once()
            self.assertEqual(
                write_evidence.call_args.kwargs["agent_config"],
                (root / "configs/agent_loop_active_capture.yaml").resolve(),
            )

    @unittest.skipUnless(importlib.util.find_spec("omegaconf"), "requires OmegaConf")
    def test_active_capture_profile_changes_only_the_bounded_capture_contract(self):
        from omegaconf import OmegaConf

        root = Path(__file__).resolve().parents[1]
        base = OmegaConf.to_container(
            OmegaConf.load(root / "configs/agent_loop.yaml"),
            resolve=False,
        )[0]
        active = OmegaConf.to_container(
            OmegaConf.load(root / "configs/agent_loop_active_capture.yaml"),
            resolve=False,
        )[0]
        base_capture = base.pop("actor_prompt_token_capture")
        active_capture = active.pop("actor_prompt_token_capture")

        self.assertEqual(base, active)
        self.assertFalse(base_capture["enabled"])
        self.assertEqual(base_capture["max_events_per_trajectory"], 1)
        self.assertTrue(active_capture["enabled"])
        self.assertEqual(active_capture["max_events_per_trajectory"], 48)

    @unittest.skipUnless(importlib.util.find_spec("omegaconf"), "requires OmegaConf")
    def test_preflight_resolves_the_launcher_selected_agent_loop(self):
        from omegaconf import OmegaConf

        root = Path(__file__).resolve().parents[1]
        selected = (root / "configs/agent_loop_active_capture.yaml").resolve()
        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {
                        "agent": {"agent_loop_config_path": str(selected)}
                    }
                }
            }
        )
        with patch.dict(
            os.environ,
            {"SHOPPING_AGENT_LOOP_CONFIG": str(selected)},
            clear=False,
        ), patch("builtins.print"):
            validate_agent_loop_config(config, lambda raw: raw)

        with patch.dict(
            os.environ,
            {"SHOPPING_AGENT_LOOP_CONFIG": str(root / "configs/agent_loop.yaml")},
            clear=False,
        ), self.assertRaisesRegex(SystemExit, "does not match launcher selection"):
            validate_agent_loop_config(config, lambda raw: raw)


if __name__ == "__main__":
    unittest.main()
