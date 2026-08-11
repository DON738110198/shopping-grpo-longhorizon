"""Validate the DAPO pilot as a narrow GRPO-stage configuration variant."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from scripts.check_grpo_runtime import compose_runtime_config


class DapoConfigTest(unittest.TestCase):
    def test_dapo_changes_only_the_intended_optimization_recipe(self):
        with patch.dict(os.environ, {"GRPO_CONFIG_NAME": "dapo"}):
            config = compose_runtime_config([])

        self.assertEqual(config.shopping_dapo.variant, "dapo-clip-higher-pilot-v1")
        self.assertTrue(config.shopping_dapo.clip_higher)
        self.assertTrue(config.shopping_dapo.dynamic_sampling)
        self.assertTrue(config.shopping_dapo.token_level_policy_gradient)
        self.assertFalse(config.shopping_dapo.reward_std_normalization)
        self.assertFalse(config.shopping_dapo.soft_overlong_reward)
        self.assertEqual(config.actor_rollout_ref.actor.clip_ratio_low, 0.20)
        self.assertEqual(config.actor_rollout_ref.actor.clip_ratio_high, 0.28)
        self.assertEqual(config.actor_rollout_ref.actor.clip_ratio_c, 10.0)
        self.assertEqual(
            config.actor_rollout_ref.actor.loss_agg_mode,
            "token-mean",
        )
        self.assertFalse(config.algorithm.norm_adv_by_std_in_grpo)
        self.assertFalse(config.algorithm.use_kl_in_reward)
        self.assertFalse(config.actor_rollout_ref.actor.use_kl_loss)
        self.assertTrue(config.shopping_dynamic_sampling.enable)

        with patch.dict(os.environ, {"GRPO_CONFIG_NAME": "grpo"}):
            grpo_config = compose_runtime_config([])

        dapo_payload = OmegaConf.to_container(config, resolve=False)
        grpo_payload = OmegaConf.to_container(grpo_config, resolve=False)
        dapo_payload.pop("shopping_dapo")
        dapo_actor = dapo_payload["actor_rollout_ref"]["actor"]
        grpo_actor = grpo_payload["actor_rollout_ref"]["actor"]
        dapo_actor["clip_ratio_high"] = grpo_actor["clip_ratio_high"]
        dapo_actor["clip_ratio_c"] = grpo_actor["clip_ratio_c"]
        self.assertEqual(dapo_payload, grpo_payload)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
