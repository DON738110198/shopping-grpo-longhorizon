#!/usr/bin/env python3
"""Run the repository's single supported Shopping Agent GRPO recipe.

这个文件是“启动器”，不是 GRPO 算法实现。它做三件事：

1. 验证 SFT merged checkpoint、Parquet 数据和空输出目录；
2. 把项目路径写入环境变量，让 Hydra YAML 在 Ray worker 中也能解析同一份资源；
3. 先运行严格 preflight，再 ``python -m verl.trainer.main_ppo``。

真正的批采样/PPO 更新在 veRL 中；本仓库的领域逻辑从 ``configs/agent_loop.yaml``
进入 ``ShoppingToolAgentLoop``，再调用 ``ShopSimulatorTool``。
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/grpo.yaml"
DEFAULT_AGENT_CONFIG = ROOT / "configs/agent_loop.yaml"
DEFAULT_TOOL_CONFIG = ROOT / "configs/tools.json"
DEFAULT_MANIFEST = ROOT / "data/environment.json"
DEFAULT_MODEL = ROOT / "outputs/models/sft-merged"
DEFAULT_TRAIN_DATA = ROOT / "data/grpo/train.parquet"
DEFAULT_VAL_DATA = ROOT / "data/grpo/validation.parquet"


def _model_has_weights(path: Path) -> bool:
    """接受单文件或 Hugging Face 分片索引，但拒绝只有 config 的空壳目录。"""
    candidates = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any((path / name).is_file() for name in candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--val-data", type=Path, default=DEFAULT_VAL_DATA)
    parser.add_argument("--env-url", default="http://127.0.0.1:5700")
    parser.add_argument("--output", type=Path, default=Path("outputs/models/grpo"))
    parser.add_argument(
        "--logger",
        choices=("console", "swanlab"),
        default="console",
    )
    parser.add_argument("--experiment-name", default="shopping-agent-grpo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--agent-loop-config",
        type=Path,
        default=DEFAULT_AGENT_CONFIG,
        help=(
            "veRL AgentLoop YAML to resolve, preflight, and record in the run "
            "manifest"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "hydra_overrides",
        nargs=argparse.REMAINDER,
        help="additional veRL Hydra overrides after --",
    )
    return parser.parse_args()


def _validated_path(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"{description} does not exist: {resolved}")
    return resolved


def _hydra_overrides(args: argparse.Namespace) -> list[str]:
    """把稳定的项目覆盖项与用户在 ``--`` 后传入的 Hydra 覆盖项合并。

    同一列表同时交给 preflight 和 main_ppo，保证“检查的配置”就是“实际训练的配置”。
    """
    logger_override = (
        "trainer.logger=[console,swanlab]"
        if args.logger == "swanlab"
        else "trainer.logger=[console]"
    )
    extra = list(args.hydra_overrides)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    return [
        logger_override,
        f"trainer.experiment_name={args.experiment_name}",
        *extra,
        f"data.seed={args.seed}",
        f"actor_rollout_ref.actor.data_loader_seed={args.seed}",
        f"actor_rollout_ref.actor.fsdp_config.seed={args.seed}",
        f"actor_rollout_ref.ref.fsdp_config.seed={args.seed}",
        f"actor_rollout_ref.rollout.engine_kwargs.vllm.seed={args.seed}",
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_evidence(path: Path) -> dict:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _model_evidence(model: Path) -> list[dict]:
    candidates = []
    for pattern in ("config.json", "*.safetensors", "*.bin", "*.index.json"):
        candidates.extend(model.glob(pattern))
    return [_file_evidence(path) for path in sorted(set(candidates)) if path.is_file()]


def _write_run_evidence(
    *,
    args: argparse.Namespace,
    command: list[str],
    environment: dict[str, str],
    output: Path,
    model: Path,
    train_data: Path,
    val_data: Path,
    config: Path,
    agent_config: Path,
) -> None:
    resolved = subprocess.run(
        [*command, "--cfg", "job", "--resolve"],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    (output / "resolved_config.yaml").write_text(resolved.stdout, encoding="utf-8")
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "schema_version": "shopping-grpo-run-v2",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_sha": git_sha,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
        "runtime_environment": {
            key: environment.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "VLLM_USE_FLASHINFER_SAMPLER",
                "CPATH",
            )
        },
        "seed": args.seed,
        "experiment_name": args.experiment_name,
        "command": command,
        "environment_contract": {
            "environment": environment["SHOPPING_ENVIRONMENT_VERSION"],
            "manifest": _file_evidence(Path(environment["SHOPPING_ENV_MANIFEST"])),
        },
        "model": {
            "path": str(model),
            "files": _model_evidence(model),
        },
        "data": {
            "train": _file_evidence(train_data),
            "validation": _file_evidence(val_data),
        },
        "config": _file_evidence(config),
        "agent_loop_config": _file_evidence(agent_config),
        "tool_config": _file_evidence(DEFAULT_TOOL_CONFIG),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "verl", "vllm")
        },
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    """返回 ``(argv, env)``，只构造进程规格，不执行训练。

    这种纯构造函数让 ``--dry-run`` 和单元测试可以验证最终命令，而不导入 CUDA、
    Ray 或模型权重。``env`` 也显式复制当前环境，避免覆盖代理/CUDA 等调用方设置。
    """
    model = _validated_path(args.model, "model directory")
    if not model.is_dir() or not (model / "config.json").is_file():
        raise SystemExit(f"model directory is missing config.json: {model}")
    if not _model_has_weights(model):
        raise SystemExit(
            "model directory has no supported weight file or sharded index: "
            f"{model}"
        )
    train_data = _validated_path(args.train_data, "train parquet")
    val_data = _validated_path(args.val_data, "validation parquet")
    config = _validated_path(args.config, "GRPO example config")
    agent_config = _validated_path(args.agent_loop_config, "AgentLoop config")
    if not agent_config.is_file():
        raise SystemExit(f"AgentLoop config must be a file: {agent_config}")
    output = args.output.expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise SystemExit(f"output must be a directory: {output}")
        if any(output.iterdir()):
            raise SystemExit(f"output directory must be new or empty: {output}")
    if args.logger == "swanlab" and not os.environ.get("SWANLAB_API_KEY"):
        raise SystemExit("--logger swanlab requires SWANLAB_API_KEY")

    # Hydra 配置中的 ${oc.env:...} 以及 Ray 子进程都从这份 environment 取值。
    # 使用绝对路径是为了让 worker 即使改变 cwd 也读取同一个模型、数据和 manifest。
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "SHOPPING_GRPO_ROOT": str(ROOT),
            "SHOPPING_ENVIRONMENT_VERSION": "shopsimulator-environment-v2.1",
            "SHOPPING_ENV_MANIFEST": str(DEFAULT_MANIFEST),
            "GRPO_MODEL_PATH": str(model),
            "GRPO_TRAIN_FILE": str(train_data),
            "GRPO_VAL_FILE": str(val_data),
            "GRPO_OUTPUT_DIR": str(output),
            "SHOPSIM_BASE_URL": str(args.env_url),
            "SHOPPING_AGENT_LOOP_CONFIG": str(agent_config),
            "SHOPPING_TOOL_CONFIG": str(DEFAULT_TOOL_CONFIG),
            "GRPO_CONFIG_DIR": str(config.parent),
            "GRPO_CONFIG_NAME": config.stem,
            # This FlashInfer build rejects Blackwell SM 12.x during sampler warmup.
            # PyTorch sampling remains deterministic under the configured vLLM seed.
            "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get(
                "VLLM_USE_FLASHINFER_SAMPLER", "0"
            ),
        }
    )
    if args.logger == "swanlab":
        environment.update(
            {
                "SWANLAB_MODE": "online",
                "SWANLAB_LOG_DIR": str(output / "swanlab"),
            }
        )
    overrides = _hydra_overrides(args)
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={config.parent}",
        f"--config-name={config.stem}",
        *overrides,
    ]
    return command, environment


def main() -> None:
    args = parse_args()
    command, environment = build_command(args)
    # 在创建输出目录前打印完整审计信息；dry-run 到这里就结束，绝不会碰 GPU。
    audit = {
        "command": command,
        "model": environment["GRPO_MODEL_PATH"],
        "train_data": environment["GRPO_TRAIN_FILE"],
        "val_data": environment["GRPO_VAL_FILE"],
        "env_url": environment["SHOPSIM_BASE_URL"],
        "output": environment["GRPO_OUTPUT_DIR"],
        "logger": args.logger,
        "config": str(args.config.resolve()),
        "agent_loop_config": environment["SHOPPING_AGENT_LOOP_CONFIG"],
        "seed": args.seed,
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    Path(environment["GRPO_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    # preflight 会校验 Python/CUDA/依赖精确版本、运行文件 SHA、token 内存护栏和
    # veRL 动态采样补丁。任何一项失败都不会进入 main_ppo。
    preflight = [
        sys.executable,
        str(ROOT / "scripts/check_grpo_runtime.py"),
        *_hydra_overrides(args),
    ]
    preflight_status = subprocess.call(preflight, cwd=ROOT, env=environment)
    if preflight_status:
        raise SystemExit(preflight_status)
    _write_run_evidence(
        args=args,
        command=command,
        environment=environment,
        output=Path(environment["GRPO_OUTPUT_DIR"]),
        model=Path(environment["GRPO_MODEL_PATH"]),
        train_data=Path(environment["GRPO_TRAIN_FILE"]),
        val_data=Path(environment["GRPO_VAL_FILE"]),
        config=args.config.expanduser().resolve(),
        agent_config=Path(environment["SHOPPING_AGENT_LOOP_CONFIG"]),
    )
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=environment))


if __name__ == "__main__":
    main()
