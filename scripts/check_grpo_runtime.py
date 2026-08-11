#!/usr/bin/env python3
"""在加载模型前拒绝污染或版本不匹配的 GRPO 环境。"""

from __future__ import annotations

import json
import math
import os
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

EXPECTED_VERSIONS = {
    "verl": "0.8.0",
    "vllm": "0.25.1",
    "torch": "2.11.0",
    "transformers": "5.15.0.dev0",
    "ray": "2.56.1",
    "tensordict": "0.10.0",
    "numpy": "2.2.6",
    "swanlab": "0.9.1",
}
EXPECTED_TRANSFORMERS_REVISION = "7ea2320c76117e6742364808a666ef6f2fb40a67"
PATCH_MARKER = "SHOPPING_GRPO_DYNAMIC_SAMPLING_PATCH_V5"
MAX_SAFE_RESPONSE_LENGTH = 20480
MAX_SAFE_SEQUENCE_LENGTH = 24576
CURRENT_RUNTIME_FILES = {
    "observation.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/observation.py",
    "pack_api.py": "environments/ShopSimulator/shop_env/shop_env/pack_api.py",
    "reward.py": "environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py",
    "slot_lease_pool.py": "environments/ShopSimulator/shop_env/shop_env/slot_lease_pool.py",
    "web_agent_text_env.py": "environments/ShopSimulator/shop_env/web_agent_site/envs/web_agent_text_env.py",
}


def validate_reward_runtime_files(manifest, root):
    if manifest.get("lease_contract") != "explicit-client-release-v1":
        raise SystemExit(
            "Environment v2.1 manifest must select explicit-client-release-v1"
        )
    expected = manifest.get("runtime_files_sha256")
    if not isinstance(expected, dict) or set(expected) != set(CURRENT_RUNTIME_FILES):
        raise SystemExit(
            "Environment v2.1 manifest runtime_files_sha256 is missing or incomplete"
        )
    from shopping_grpo.environment.manifest import sha256_file

    mismatches = {}
    for name, relative_path in CURRENT_RUNTIME_FILES.items():
        actual = sha256_file(Path(root) / relative_path)
        if actual != expected[name]:
            mismatches[name] = {"expected": expected[name], "actual": actual}
    if mismatches:
        raise SystemExit(
            "Environment v2.1 runtime file hash mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )


def validate_environment_contract():
    required_version = os.environ.get(
        "SHOPPING_ENVIRONMENT_VERSION",
        "shopsimulator-environment-v2.1",
    )
    if required_version != "shopsimulator-environment-v2.1":
        raise SystemExit(
            "this repository supports only shopsimulator-environment-v2.1"
        )
    manifest_path = os.environ.get("SHOPPING_ENV_MANIFEST")
    if not manifest_path or not Path(manifest_path).is_file():
        raise SystemExit(
            f"{required_version} requires SHOPPING_ENV_MANIFEST pointing to a frozen manifest"
        )
    try:
        from shopping_grpo.environment.manifest import validate_manifest

        manifest = validate_manifest(
            json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        )
    except (ImportError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {required_version} manifest: {exc}") from exc
    actual_environment_version = manifest.get(
        "environment_version",
        "shopsimulator-environment-v2.1",
    )
    if actual_environment_version != required_version:
        raise SystemExit(
            "environment manifest version mismatch: "
            f"expected {required_version}, got {actual_environment_version}"
        )
    tools_path = Path(
        os.environ.get(
            "SHOPPING_TOOL_CONFIG",
            Path(__file__).resolve().parents[1]
            / "configs/tools.json",
        )
    )
    tools = json.loads(tools_path.read_text(encoding="utf-8")).get("tools", [])
    tool_names = {
        item.get("tool_schema", {}).get("function", {}).get("name")
        for item in tools
    }
    if "finish_without_purchase" not in tool_names:
        raise SystemExit("Environment v2 tool config is missing finish_without_purchase")
    if int(manifest["max_steps"]) != 35:
        raise SystemExit("Environment v2 GRPO contract requires max_steps=35")
    validate_reward_runtime_files(
        manifest,
        Path(__file__).resolve().parents[1],
    )
    print(
        f"{required_version} manifest preflight passed: "
        + json.dumps(
            {
                "manifest": str(Path(manifest_path).resolve()),
                "shopsimulator_commit": manifest["shopsimulator_commit"],
                "observation_version": manifest["observation_version"],
                "reward_version": manifest["reward"]["version"],
                "search_version": manifest["search"]["version"],
                "lease_contract": manifest.get("lease_contract"),
                "runtime_file_count": len(manifest.get("runtime_files_sha256") or {}),
            },
            sort_keys=True,
        )
    )


def compose_runtime_config(overrides):
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
    except ImportError as exc:
        raise SystemExit(f"cannot parse GRPO config before preflight: {exc}") from exc

    GlobalHydra.instance().clear()
    raw_config_dir = os.environ.get("GRPO_CONFIG_DIR")
    if not raw_config_dir:
        raise SystemExit("GRPO_CONFIG_DIR is required")
    config_dir = Path(raw_config_dir).expanduser().resolve()
    if not config_dir.is_dir():
        raise SystemExit(f"GRPO_CONFIG_DIR does not exist: {config_dir}")
    config_name = os.environ.get("GRPO_CONFIG_NAME", "grpo")
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=list(overrides))


def validate_agent_loop_config(config, capture_validator):
    """Resolve the exact AgentLoop YAML selected by the public launcher."""
    config_path = os.environ.get("SHOPPING_AGENT_LOOP_CONFIG")
    if not config_path:
        raise SystemExit("SHOPPING_AGENT_LOOP_CONFIG is required")
    selected_path = Path(config_path).expanduser().resolve()
    if not selected_path.is_file():
        raise SystemExit(f"AgentLoop config does not exist: {selected_path}")

    configured_path = Path(
        str(config.actor_rollout_ref.rollout.agent.agent_loop_config_path)
    ).expanduser().resolve()
    if configured_path != selected_path:
        raise SystemExit(
            "resolved AgentLoop config path does not match launcher selection: "
            f"resolved={configured_path}, selected={selected_path}"
        )

    try:
        from omegaconf import OmegaConf

        payload = OmegaConf.to_container(
            OmegaConf.load(selected_path),
            resolve=True,
        )
    except Exception as exc:
        raise SystemExit(f"cannot resolve AgentLoop config {selected_path}: {exc}") from exc
    if not isinstance(payload, list) or len(payload) != 1:
        raise SystemExit("AgentLoop config must contain exactly one loop definition")
    loop = payload[0]
    if not isinstance(loop, dict):
        raise SystemExit("AgentLoop definition must be an object")
    if loop.get("name") != "shopping_tool_agent":
        raise SystemExit("AgentLoop config must define shopping_tool_agent")
    if loop.get("_target_") != (
        "shopping_grpo.training.grpo.adapter.agent_loop.ShoppingToolAgentLoop"
    ):
        raise SystemExit("AgentLoop config selects an unsupported implementation")
    try:
        capture = capture_validator(loop.get("actor_prompt_token_capture"))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"invalid actor prompt capture config: {exc}") from exc
    print(
        "AgentLoop config preflight passed: "
        + json.dumps(
            {
                "path": str(selected_path),
                "prompt_capture_enabled": bool(capture["enabled"]),
                "prompt_capture_limit": int(capture["max_events_per_trajectory"]),
            },
            sort_keys=True,
        )
    )


def validate_transformers_revision():
    """The Qwen3.5 runtime uses one pinned upstream Transformers revision."""
    dist = distribution("transformers")
    direct_url = Path(dist.locate_file("transformers-5.15.0.dev0.dist-info/direct_url.json"))
    if not direct_url.is_file():
        raise SystemExit(
            "cannot verify pinned Transformers revision: direct_url.json is missing"
        )
    try:
        metadata = json.loads(direct_url.read_text(encoding="utf-8"))
        revision = metadata["vcs_info"]["commit_id"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid Transformers direct_url.json: {exc}") from exc
    if revision != EXPECTED_TRANSFORMERS_REVISION:
        raise SystemExit(
            "incompatible Transformers revision: expected "
            f"{EXPECTED_TRANSFORMERS_REVISION}, got {revision}"
        )
    print(f"pinned Transformers revision preflight passed: {revision}")


def validate_dynamic_sampling(config, verl_source: Path, installed):
    dynamic_config = config.get("shopping_dynamic_sampling", {})
    if not bool(dynamic_config.get("enable", False)):
        return

    if installed.get("verl") != "0.8.0":
        raise SystemExit(
            f"shopping dynamic sampling requires verl==0.8.0, got {installed.get('verl')}"
        )
    ray_trainer = verl_source.parent / "trainer" / "ppo" / "ray_trainer.py"
    if not ray_trainer.is_file():
        raise SystemExit(f"cannot locate installed RayPPOTrainer source: {ray_trainer}")
    if PATCH_MARKER not in ray_trainer.read_text(encoding="utf-8"):
        raise SystemExit(
            "shopping dynamic sampling is enabled but the pinned veRL patch marker is missing; "
            "run scripts/apply_verl_dynamic_sampling_patch.py first"
        )

    try:
        from shopping_grpo.training.grpo.dynamic_sampling import (
            extract_shopping_group_signals,
            select_reward_varying_groups,
        )
    except ImportError as exc:
        raise SystemExit(f"shopping dynamic sampling helper is unavailable: {exc}") from exc
    policy, utility, success, invalid, reasons = extract_shopping_group_signals(
        [
            {
                "infrastructure_invalid": False,
                "reward_unverifiable": False,
                "valid_for_learning": True,
                "reward": {
                    "policy_reward_version": "shopping-policy-reward-v1",
                    "total": reward,
                    "terminal_utility": reward,
                    "purchase_success": reward > 0,
                    "sampling_invalid": False,
                },
            }
            for reward in (0.0, 1.0, 0.0, 0.0)
        ]
    )
    indices, _ = select_reward_varying_groups(
        ["preflight"] * 4,
        [0.0, 1.0, 0.0, 0.0],
        policy_rewards=policy,
        terminal_utilities=utility,
        purchase_success=success,
        sampling_invalid=invalid,
        sampling_invalid_reasons=reasons,
    )
    if indices != [0, 1, 2, 3]:
        raise SystemExit("shopping dynamic sampling helper failed its import-time sanity check")

    if dynamic_config.get("metric") != "seq_reward":
        raise SystemExit("shopping_dynamic_sampling.metric must be seq_reward")
    if int(dynamic_config.get("max_num_gen_batches", 0)) <= 0:
        raise SystemExit("shopping_dynamic_sampling.max_num_gen_batches must be positive")
    if int(dynamic_config.get("max_consecutive_skipped_updates", 0)) <= 0:
        raise SystemExit(
            "shopping_dynamic_sampling.max_consecutive_skipped_updates must be positive"
        )
    reward_tolerance = float(dynamic_config.get("reward_tolerance", -1))
    if reward_tolerance < 0 or not math.isfinite(reward_tolerance):
        raise SystemExit("shopping_dynamic_sampling.reward_tolerance must be finite and non-negative")
    if not bool(config.algorithm.rollout_correction.get("bypass_mode", False)):
        raise SystemExit("shopping dynamic sampling requires rollout_correction.bypass_mode=true")
    if not bool(config.actor_rollout_ref.rollout.get("calculate_log_probs", False)):
        raise SystemExit("shopping dynamic sampling requires rollout.calculate_log_probs=true")

    print(
        "shopping dynamic sampling preflight passed: "
        + json.dumps(
            {
                "enable": True,
                "metric": str(dynamic_config.metric),
                "max_num_gen_batches": int(dynamic_config.max_num_gen_batches),
                "max_consecutive_skipped_updates": int(
                    dynamic_config.max_consecutive_skipped_updates
                ),
                "reward_tolerance": reward_tolerance,
                "ray_trainer": str(ray_trainer),
                "marker": PATCH_MARKER,
            },
            sort_keys=True,
        )
    )


def validate_capture_only(config):
    """Fail closed around the explicit fixed-policy rollout collection mode."""
    capture_config = config.get("shopping_capture_only")
    if capture_config is None:
        raise SystemExit("shopping_capture_only config is required")
    if not hasattr(capture_config, "keys"):
        raise SystemExit("shopping_capture_only must be an object")
    unknown_keys = sorted(set(capture_config.keys()) - {"enable"})
    if unknown_keys:
        raise SystemExit(
            "shopping_capture_only contains unsupported keys: "
            + ", ".join(unknown_keys)
        )
    if "enable" not in capture_config:
        raise SystemExit("shopping_capture_only.enable must be explicit")
    enabled = capture_config.get("enable")
    if type(enabled) is not bool:
        raise SystemExit("shopping_capture_only.enable must be a boolean")
    if enabled and not bool(
        config.get("shopping_dynamic_sampling", {}).get("enable", False)
    ):
        raise SystemExit(
            "shopping_capture_only requires shopping_dynamic_sampling.enable=true "
            "so every rollout is written to sampling_audit.jsonl"
        )
    if enabled:
        trainer = config.get("trainer", {})
        reward_model = config.get("reward_model", {})
        test_freq = trainer.get("test_freq")
        violations = []
        if trainer.get("val_before_train") is not False:
            violations.append("trainer.val_before_train=false")
        if (
            isinstance(test_freq, bool)
            or not isinstance(test_freq, (int, float))
            or test_freq > 0
        ):
            violations.append("trainer.test_freq<=0")
        if trainer.get("val_only") is not False:
            violations.append("trainer.val_only=false")
        if reward_model.get("enable") is not False:
            violations.append("reward_model.enable=false")
        if violations:
            raise SystemExit(
                "shopping_capture_only requires " + ", ".join(violations)
            )
    print(
        "shopping capture-only preflight passed: "
        + json.dumps(
            {
                "enable": enabled,
                "policy_update_path_enabled": not enabled,
                "selection_source": "sampling_audit.jsonl" if enabled else None,
            },
            sort_keys=True,
        )
    )


def validate_swanlab_tracking(config):
    """Validate SwanLab only when the user explicitly enables it."""
    logger_backends = list(config.trainer.get("logger", []))
    if "swanlab" not in logger_backends:
        return
    forbidden = {"wandb", "tracking", "vemlp_wandb"} & set(logger_backends)
    if forbidden:
        raise SystemExit(
            "Reward v3 GRPO forbids W&B logger backends: "
            + ", ".join(sorted(forbidden))
        )
    if os.environ.get("SWANLAB_MODE") != "online":
        raise SystemExit("Reward v3 GRPO requires SWANLAB_MODE=online")
    if not os.environ.get("SWANLAB_API_KEY"):
        raise SystemExit(
            "Reward v3 GRPO requires SWANLAB_API_KEY in the launching environment"
        )
    log_dir = os.environ.get("SWANLAB_LOG_DIR")
    if not log_dir:
        raise SystemExit("Reward v3 GRPO requires SWANLAB_LOG_DIR")
    resolved_log_dir = Path(log_dir).resolve()
    if str(config.trainer.get("project_name")) != "shopping-grpo":
        raise SystemExit("Reward v3 GRPO SwanLab project must be shopping-grpo")
    print(
        "SwanLab online preflight passed: "
        + json.dumps(
            {
                "api_key": "present",
                "logger": logger_backends,
                "log_dir": str(resolved_log_dir),
                "mode": "online",
                "project": str(config.trainer.project_name),
                "run_name": str(config.trainer.experiment_name),
            },
            sort_keys=True,
        )
    )


def validate_training_memory_budget(config):
    prompt_length = int(config.data.max_prompt_length)
    response_length = int(config.data.max_response_length)
    total_length = prompt_length + response_length
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    reference = config.actor_rollout_ref.ref

    if response_length > MAX_SAFE_RESPONSE_LENGTH:
        raise SystemExit(
            "unsafe GRPO response budget: "
            f"max_response_length={response_length} exceeds {MAX_SAFE_RESPONSE_LENGTH}"
        )
    if total_length > MAX_SAFE_SEQUENCE_LENGTH:
        raise SystemExit(
            "unsafe GRPO sequence budget: "
            f"max_prompt_length + max_response_length = {total_length}, "
            f"limit is {MAX_SAFE_SEQUENCE_LENGTH}"
        )
    for name, value in (
        ("rollout.max_model_len", int(rollout.max_model_len)),
        ("rollout.max_num_batched_tokens", int(rollout.max_num_batched_tokens)),
        (
            "rollout.log_prob_max_token_len_per_gpu",
            int(rollout.log_prob_max_token_len_per_gpu),
        ),
        ("actor.ppo_max_token_len_per_gpu", int(actor.ppo_max_token_len_per_gpu)),
        ("ref.log_prob_max_token_len_per_gpu", int(reference.log_prob_max_token_len_per_gpu)),
    ):
        if value != MAX_SAFE_SEQUENCE_LENGTH:
            raise SystemExit(
                f"unsafe or inconsistent GRPO memory budget: {name} must equal "
                f"{MAX_SAFE_SEQUENCE_LENGTH}, got {value}"
            )
    if bool(actor.use_dynamic_bsz):
        raise SystemExit(
            "actor.use_dynamic_bsz must be false so ppo_micro_batch_size_per_gpu=1 is enforced"
        )
    if int(actor.ppo_micro_batch_size_per_gpu) != 1:
        raise SystemExit("actor.ppo_micro_batch_size_per_gpu must equal 1")
    if bool(rollout.log_prob_use_dynamic_bsz):
        raise SystemExit(
            "rollout.log_prob_use_dynamic_bsz must be false so "
            "log_prob_micro_batch_size_per_gpu=1 is enforced"
        )
    if int(rollout.log_prob_micro_batch_size_per_gpu) != 1:
        raise SystemExit("rollout.log_prob_micro_batch_size_per_gpu must equal 1")
    if bool(reference.log_prob_use_dynamic_bsz):
        raise SystemExit(
            "ref.log_prob_use_dynamic_bsz must be false so "
            "log_prob_micro_batch_size_per_gpu=1 is enforced"
        )
    if int(reference.log_prob_micro_batch_size_per_gpu) != 1:
        raise SystemExit("ref.log_prob_micro_batch_size_per_gpu must equal 1")

    print(
        "GRPO training memory budget preflight passed: "
        + json.dumps(
            {
                "max_prompt_length": prompt_length,
                "max_response_length": response_length,
                "max_sequence_length": total_length,
                "actor_micro_batch_size_per_gpu": 1,
                "actor_dynamic_batch": False,
                "rollout_log_prob_micro_batch_size_per_gpu": 1,
                "rollout_log_prob_dynamic_batch": False,
                "reference_micro_batch_size_per_gpu": 1,
                "reference_dynamic_batch": False,
            },
            sort_keys=True,
        )
    )


def main():
    config = compose_runtime_config(sys.argv[1:])
    validate_environment_contract()
    required_paths = {
        "GRPO_TRAIN_FILE": os.environ.get("GRPO_TRAIN_FILE"),
        "GRPO_VAL_FILE": os.environ.get("GRPO_VAL_FILE"),
    }
    missing = [name for name, value in required_paths.items() if not value or not Path(value).is_file()]
    if missing:
        raise SystemExit("missing GRPO parquet file(s): " + ", ".join(missing))
    validate_training_memory_budget(config)

    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"incompatible Python: expected 3.12, got {sys.version.split()[0]}")

    installed = {}
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            installed[package] = version(package)
        except PackageNotFoundError as exc:
            raise SystemExit(f"missing GRPO dependency: {package}=={expected}") from exc
        if installed[package].split("+", 1)[0] != expected:
            raise SystemExit(
                f"incompatible GRPO dependency: expected {package}=={expected}, got {installed[package]}"
            )
    validate_transformers_revision()

    try:
        import torch
        import verl
        from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
        from verl.experimental.agent_loop.tool_parser import ToolParser
        from verl.tools.base_tool import BaseTool
        from verl.utils.tracking import Tracking

        from shopping_grpo.training.grpo.adapter.agent_loop import (
            ShoppingToolAgentLoop,
            validate_actor_prompt_token_capture_config,
        )
        from shopping_grpo.training.grpo.adapter.tools import ShopSimulatorTool
        from shopping_grpo.training.grpo.compat import install_torch_padding_fallback
    except ImportError as exc:
        raise SystemExit(
            "incompatible veRL 0.8 install: required AgentLoop/Tool APIs are unavailable; "
            f"original error: {exc}"
        ) from exc

    verl_source = Path(verl.__file__).resolve()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in the GRPO environment")
    if (
        not issubclass(ShoppingToolAgentLoop, ToolAgentLoop)
        or not issubclass(ShopSimulatorTool, BaseTool)
        or AgentState.TERMINATED.value != "terminated"
        or not hasattr(ToolAgentLoop, "_handle_processing_tools_state")
    ):
        raise SystemExit("incompatible veRL ToolAgentLoop lifecycle API")
    if "qwen3_coder" not in ToolParser._registry:
        raise SystemExit("veRL 0.8 built-in qwen3_coder parser is unavailable")
    if "swanlab" not in Tracking.supported_backend:
        raise SystemExit("veRL 0.8 SwanLab tracking backend is unavailable")
    validate_agent_loop_config(
        config,
        validate_actor_prompt_token_capture_config,
    )
    validate_capture_only(config)
    validate_dynamic_sampling(config, verl_source, installed)
    validate_swanlab_tracking(config)
    install_torch_padding_fallback()
    print(
        "GRPO runtime preflight passed: "
        + ", ".join(f"{name}={value}" for name, value in installed.items())
        + f", source={verl_source}"
    )


if __name__ == "__main__":
    main()
