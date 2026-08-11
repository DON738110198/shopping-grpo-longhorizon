"""Reproducibility tests for the pinned veRL source patch."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from verl import DataProto

from scripts import apply_verl_dynamic_sampling_patch as patcher
from shopping_grpo.training.grpo.dynamic_sampling import (
    extract_shopping_group_signals,
    select_reward_varying_groups,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/apply_verl_dynamic_sampling_patch.py"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_source() -> Path:
    installed = patcher.resolve_installed_ray_trainer()
    if file_sha256(installed) == patcher.EXPECTED_ORIGINAL_SHA256:
        return installed
    backup = Path(str(installed) + patcher.BACKUP_SUFFIX)
    if file_sha256(backup) != patcher.EXPECTED_ORIGINAL_SHA256:
        raise AssertionError("installed veRL source and its backup are not the pinned original")
    return backup


class VerlPatchScriptTest(unittest.TestCase):
    def run_script(self, target: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--target", str(target), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_apply_is_idempotent_and_restore_recovers_original(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "ray_trainer.py"
            shutil.copy2(original_source(), target)

            first = self.run_script(target)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(file_sha256(target), patcher.EXPECTED_PATCHED_SHA256)
            self.assertIn(patcher.PATCH_MARKER, target.read_text(encoding="utf-8"))

            second = self.run_script(target)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("already applied", second.stdout)
            self.assertEqual(file_sha256(target), patcher.EXPECTED_PATCHED_SHA256)

            restored = self.run_script(target, "--restore")
            self.assertEqual(restored.returncode, 0, restored.stderr)
            self.assertEqual(file_sha256(target), patcher.EXPECTED_ORIGINAL_SHA256)

    def test_unknown_sha256_is_rejected_without_modification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "ray_trainer.py"
            shutil.copy2(original_source(), target)
            target.write_text(target.read_text(encoding="utf-8") + "\n# unknown change\n", encoding="utf-8")
            before = file_sha256(target)

            result = self.run_script(target)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to patch unknown ray_trainer.py", result.stderr)
            self.assertEqual(file_sha256(target), before)
            self.assertFalse(Path(str(target) + patcher.BACKUP_SUFFIX).exists())

    def test_patched_fit_preserves_bypass_and_defers_reference_and_update(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "ray_trainer.py"
            shutil.copy2(original_source(), target)
            result = self.run_script(target)
            self.assertEqual(result.returncode, 0, result.stderr)

            patched_source = target.read_text(encoding="utf-8")
            self.assertIn('f"val-shopping/{key}"', patched_source)
            self.assertIn('validation_dump_infos["shopping"]', patched_source)
            self.assertIn('reward_extra_infos_to_dump["shopping"]', patched_source)
            fit_source = patched_source.split("    def fit(self):", 1)[1]
            generation = fit_source.index("generate_sequences(combined_gen_batch)")
            reward_filter = fit_source.index("SHOPPING_GRPO_DYNAMIC_SAMPLING_BATCH")
            skipped = fit_source.index("SHOPPING_GRPO_DYNAMIC_SAMPLING_SKIPPED")
            ready = fit_source.index("SHOPPING_GRPO_DYNAMIC_SAMPLING_READY")
            sleep_before_training = fit_source.index(
                "self.checkpoint_manager.sleep_replicas()", ready
            )
            bypass = fit_source.index("apply_bypass_mode", sleep_before_training)
            recomputed_old = fit_source.index(
                "self._compute_old_log_prob(batch)", bypass
            )
            reference = fit_source.index("self._compute_ref_log_prob(batch)", bypass)
            advantage = fit_source.index("batch = compute_advantage(", reference)
            update = fit_source.index("actor_output = self._update_actor(batch)", advantage)

            self.assertLess(generation, reward_filter)
            self.assertLess(reward_filter, skipped)
            self.assertLess(reward_filter, ready)
            self.assertLess(ready, sleep_before_training)
            self.assertLess(sleep_before_training, bypass)
            self.assertLess(bypass, recomputed_old)
            self.assertLess(recomputed_old, reference)
            self.assertLess(reference, advantage)
            self.assertLess(advantage, update)
            self.assertIn(
                "if not dynamic_sampling_enabled:\n"
                "                            self.checkpoint_manager.sleep_replicas()",
                fit_source,
            )
            self.assertIn(
                "if bypass_recomputing_logprobs:  # Use `rollout_log_probs`",
                fit_source,
            )
            self.assertIn("extract_shopping_group_signals", fit_source)
            self.assertIn("aggregate_shopping_metrics", fit_source)
            self.assertIn("append_sampling_audit", fit_source)
            self.assertIn("policy_rewards=policy_rewards", fit_source)
            self.assertIn("terminal_utilities=terminal_utilities", fit_source)
            self.assertIn("sampling_invalid=sampling_invalid", fit_source)
            self.assertIn('"drop_reason": group["drop_reason"]', fit_source)
            self.assertIn('"group/all_zero_utility_ratio"', fit_source)
            self.assertIn('"group/no_purchase_success_ratio"', fit_source)
            self.assertIn('"group/all_purchase_success_ratio"', fit_source)
            self.assertIn('"group/sampling_invalid"', fit_source)
            self.assertIn('"training/optimizer_updated": 0', fit_source)
            self.assertIn(
                "logger.log(data=skipped_metrics, step=self.global_steps)",
                fit_source,
            )
            self.assertIn(
                "dynamic_accepted_batches = []",
                fit_source[skipped:ready],
            )
            self.assertIn("dynamic_consecutive_skips", fit_source)
            self.assertIn(">= dynamic_max_consecutive_skips", fit_source)
            self.assertNotIn(
                "exhausted max_num_gen_batches=",
                fit_source,
            )
            self.assertLess(
                fit_source.index("SHOPPING_GRPO_DYNAMIC_SAMPLING_SKIPPED"),
                fit_source.index("self.checkpoint_manager.sleep_replicas()", ready),
            )

    def test_capture_only_audits_then_skips_every_policy_operation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "ray_trainer.py"
            shutil.copy2(original_source(), target)
            result = self.run_script(target)
            self.assertEqual(result.returncode, 0, result.stderr)

            fit_source = target.read_text(encoding="utf-8").split(
                "    def fit(self):", 1
            )[1]
            config = fit_source.index(
                'capture_only_config = self.config.get("shopping_capture_only")'
            )
            config_log = fit_source.index(
                "SHOPPING_GRPO_CAPTURE_ONLY_CONFIG", config
            )
            initial_validation_guard = fit_source.index(
                "if not capture_only_enabled and self.config.trainer.get(\n"
                '            "val_before_train", True\n'
                "        ):"
            )
            initial_validation = fit_source.index(
                "val_metrics = self._validate()", initial_validation_guard
            )
            bootstrap_sync = fit_source.index(
                "self.checkpoint_manager.update_weights(self.global_steps)"
            )
            generation = fit_source.index("generate_sequences(combined_gen_batch)")
            audit = fit_source.index("append_sampling_audit(", generation)
            capture_step = fit_source.index("SHOPPING_GRPO_CAPTURE_ONLY_STEP", audit)
            policy_guard = fit_source.index(
                "if not capture_only_enabled:", capture_step
            )
            rollout_logging = fit_source.index(
                "# Log rollout generations if enabled", policy_guard
            )
            periodic_validation_guard = fit_source.index(
                "if (\n"
                "                    not capture_only_enabled\n"
                "                    and self.config.trainer.test_freq > 0",
                rollout_logging,
            )
            periodic_validation = fit_source.index(
                "val_metrics: dict = self._validate()", periodic_validation_guard
            )
            guarded_policy_source = fit_source[policy_guard:rollout_logging]

            self.assertIn("trainer.val_before_train=false", fit_source[config:generation])
            self.assertIn("trainer.test_freq<=0", fit_source[config:generation])
            self.assertIn("trainer.val_only=false", fit_source[config:generation])
            self.assertIn(
                "reward.reward_model.enable=false", fit_source[config:generation]
            )
            self.assertIn(
                'reward_config = self.config.get("reward")',
                fit_source[config:generation],
            )
            self.assertNotIn(
                "self.config.reward_model",
                fit_source[config:generation],
            )
            self.assertIn(
                'if capture_only_config is None:\n'
                '            raise ValueError("shopping_capture_only config is required")',
                fit_source[config:generation],
            )
            self.assertIn(
                'if "enable" not in capture_only_config:\n'
                '            raise ValueError("shopping_capture_only.enable must be explicit")',
                fit_source[config:generation],
            )
            self.assertLess(config, config_log)
            self.assertLess(config, generation)
            self.assertLess(initial_validation_guard, initial_validation)
            self.assertLess(bootstrap_sync, generation)
            self.assertEqual(
                fit_source.count(
                    "self.checkpoint_manager.update_weights(self.global_steps)"
                ),
                1,
                "capture-only permits exactly one pre-generation checkpoint bootstrap sync",
            )
            self.assertNotIn(
                "self.checkpoint_manager.update_weights(",
                fit_source[generation:policy_guard],
                "capture-only must not synchronize changed weights after rollout",
            )
            self.assertNotIn(
                "optim.lr",
                fit_source[config:generation],
                "capture-only must never be inferred from lr=0",
            )
            self.assertLess(generation, audit)
            self.assertLess(audit, capture_step)
            self.assertLess(capture_step, policy_guard)
            self.assertLess(periodic_validation_guard, periodic_validation)
            self.assertEqual(fit_source.count("self._validate()"), 2)
            for operation in (
                "self._compute_old_log_prob(batch)",
                "self._compute_ref_log_prob(batch)",
                "batch = compute_advantage(",
                "self._update_critic(batch)",
                "self._update_actor(batch)",
                "self._save_checkpoint()",
                "self.checkpoint_manager.update_weights(current_step)",
            ):
                self.assertIn(operation, guarded_policy_source)
                self.assertNotIn(
                    operation,
                    fit_source[audit:policy_guard],
                    f"{operation} escaped the capture-only policy guard",
                )
            self.assertIn("self.global_steps = current_step", fit_source[audit:policy_guard])
            self.assertIn(
                'batch.batch["token_level_scores"] = reward_tensor',
                fit_source[audit:policy_guard],
            )
            self.assertIn('"training/optimizer_updated": int(optimizer_updated)', fit_source)
            self.assertIn('"training/capture_only": int(capture_only_enabled)', fit_source)
            self.assertIn(
                "dynamic_sampling_enabled and not optimizer_updated and not capture_only_enabled",
                fit_source,
            )
            self.assertIn(
                "if capture_only_enabled:\n"
                "                            dynamic_accepted_batches.append(batch)",
                fit_source,
            )
            self.assertIn(
                "else:\n"
                "                            remaining_prompts = dynamic_target_prompts",
                fit_source,
            )
            self.assertIn(
                "else:\n"
                "                            print(\n"
                '                                "SHOPPING_GRPO_DYNAMIC_SAMPLING_READY "',
                fit_source,
            )

    def test_select_and_concat_keep_all_trajectory_fields_aligned(self):
        def make_batch(offset: int, uid_prefix: str) -> DataProto:
            row_ids = torch.arange(offset, offset + 8, dtype=torch.int64)
            uids = np.array([f"{uid_prefix}-drop"] * 4 + [f"{uid_prefix}-keep"] * 4)
            rewards = torch.tensor([0.0] * 4 + [2 / 7, 4 / 7, 2 / 7, 2 / 7])
            return DataProto.from_dict(
                tensors={
                    "responses": row_ids[:, None],
                    "response_mask": row_ids[:, None],
                    "rollout_log_probs": row_ids[:, None].float(),
                    "token_level_scores": rewards[:, None],
                    "rm_scores": rewards[:, None],
                    "attention_mask": row_ids[:, None],
                    "position_ids": row_ids[:, None],
                },
                non_tensors={
                    "uid": uids,
                    "extra_info": np.array(
                        [{"task_id": int(row_id)} for row_id in row_ids], dtype=object
                    ),
                    "shopping": np.array(
                        [
                            {
                                "infrastructure_invalid": False,
                                "reward_unverifiable": False,
                                "valid_for_learning": True,
                                "reward": {
                                    "policy_reward_version": "shopping-policy-reward-v1",
                                    "total": float(reward),
                                    "terminal_utility": float(reward),
                                    "purchase_success": bool(reward > 0),
                                    "sampling_invalid": False,
                                },
                            }
                            for reward in rewards
                        ],
                        dtype=object,
                    ),
                },
                meta_info={"reward_extra_keys": []},
            )

        selected_batches = []
        for batch in (make_batch(0, "a"), make_batch(8, "b")):
            rewards = batch.batch["rm_scores"].sum(dim=-1).tolist()
            policy, utility, success, invalid, reasons = extract_shopping_group_signals(
                batch.non_tensor_batch["shopping"].tolist()
            )
            indices, _ = select_reward_varying_groups(
                batch.non_tensor_batch["uid"].tolist(),
                rewards,
                policy_rewards=policy,
                terminal_utilities=utility,
                purchase_success=success,
                sampling_invalid=invalid,
                sampling_invalid_reasons=reasons,
            )
            selected_batches.append(batch.select_idxs(indices))

        combined = DataProto.concat(selected_batches)
        expected_ids = [4, 5, 6, 7, 12, 13, 14, 15]
        self.assertEqual(combined.batch["responses"].flatten().tolist(), expected_ids)
        for key in (
            "response_mask",
            "rollout_log_probs",
            "attention_mask",
            "position_ids",
        ):
            self.assertEqual(combined.batch[key].flatten().tolist(), expected_ids)
        self.assertEqual(
            [item["task_id"] for item in combined.non_tensor_batch["extra_info"]],
            expected_ids,
        )
        self.assertEqual(
            combined.non_tensor_batch["uid"].tolist(),
            ["a-keep"] * 4 + ["b-keep"] * 4,
        )
        self.assertTrue(
            torch.allclose(
                combined.batch["token_level_scores"].flatten(),
                torch.tensor([2 / 7, 4 / 7, 2 / 7, 2 / 7] * 2),
            )
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
