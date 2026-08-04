"""Public CPU and parameterized GRPO entry-point tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.train_grpo import build_command, main as train_grpo_main, parse_args
from shopping_grpo.cli import main as cli_main
from shopping_grpo.smoke import run_cpu_smoke


class PublicEntrypointTest(unittest.TestCase):
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
                    str(root / "configs/grpo.yaml"),
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
        self.assertIn("trainer.logger=[console]", command)

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
            ) as subprocess_call, self.assertRaises(SystemExit) as completed:
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


if __name__ == "__main__":
    unittest.main()
