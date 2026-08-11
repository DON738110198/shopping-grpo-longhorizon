#!/usr/bin/env python3
"""Collect a resumable 40-state x 4-proposal first-decision Stage 1."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
from pathlib import Path

from shopping_grpo.training.grpo.active_suffix import VllmTokenCompletionClient
from shopping_grpo.training.grpo.adapter.runtime import (
    validate_policy_reward_config,
)
from shopping_grpo.training.grpo.pivotal_states import (
    tokenizer_contract_sha256,
)
from shopping_grpo.training.grpo.selection import resolve_pivotal_selection
from shopping_grpo.training.grpo.stage1_proposal import (
    FIRST_DECISION_SOURCE_PROVENANCE_VERSION,
    FirstDecisionStage1Collector,
    verify_first_decision_artifacts,
    write_first_decision_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--sampling-backend-contract", type=Path, required=True)
    parser.add_argument(
        "--tools-config", type=Path, default=ROOT / "configs" / "tools.json"
    )
    parser.add_argument("--policy-reward-config", type=Path)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-api-key-env", default="VLLM_API_KEY")
    parser.add_argument(
        "--environment-version", default="shopsimulator-environment-v2.1"
    )
    parser.add_argument("--vllm-timeout", type=int, default=180)
    parser.add_argument("--expected-states", type=int, default=40)
    parser.add_argument("--proposals-per-state", type=int, default=4)
    parser.add_argument("--journal-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mechanical", action="store_true")
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser.parse_args(argv)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_object(path: Path, name: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _load_tool_schemas(path: Path) -> list[dict]:
    value = _load_object(path, "tools config")
    tools = value.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("tools config must contain a non-empty tools list")
    schemas = []
    for index, item in enumerate(tools):
        if not isinstance(item, dict) or not isinstance(item.get("tool_schema"), dict):
            raise TypeError(f"tools config entry {index} lacks tool_schema")
        schemas.append(item["tool_schema"])
    return schemas


def _records(paths: list[Path]):
    for input_index, path in enumerate(paths):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    yield input_index, str(path), line_number, json.loads(line)


class Qwen3CoderParser:
    """Thin adapter around veRL 0.8's exact qwen3_coder parser."""

    def __init__(self, tokenizer, schemas):
        from verl.experimental.agent_loop.tool_parser import ToolParser
        from verl.tools.schemas import OpenAIFunctionToolSchema

        self._parser = ToolParser.get_tool_parser("qwen3_coder", tokenizer)
        self.stop_token_ids = list(self._parser.stop_token_ids)
        self._schemas = [
            OpenAIFunctionToolSchema.model_validate(schema) for schema in schemas
        ]

    async def parse(self, token_ids, schemas):
        del schemas
        _, calls = await self._parser.extract_tool_calls(token_ids, self._schemas)
        return calls


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_binding(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256_file(path)}


def _git_binding(
    *, mechanical: bool, allow_dirty: bool
) -> dict[str, object]:
    if allow_dirty and not mechanical:
        raise ValueError(
            "--allow-dirty-source is restricted to --mechanical runs"
        )

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout

    git_sha = run("rev-parse", "HEAD").strip().lower()
    if len(git_sha) != 40 or any(
        character not in "0123456789abcdef" for character in git_sha
    ):
        raise ValueError("git HEAD is not a full commit SHA")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    clean = not bool(status)
    if not clean and not (mechanical and allow_dirty):
        raise ValueError(
            "formal Stage-1 requires a clean worktree; dirty source is allowed "
            "only with --mechanical --allow-dirty-source"
        )
    scale_ready = clean and not mechanical
    return {
        "git_sha": git_sha,
        "git_worktree_clean": clean,
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "execution_mode": "mechanical" if mechanical else "formal",
        "dirty_source_override": bool(not clean and allow_dirty),
        "scale_ready": scale_ready,
    }


async def _run(args):
    paths = {
        "plan": args.plan.expanduser().resolve(),
        "selection": args.selection.expanduser().resolve(),
        "backend": args.sampling_backend_contract.expanduser().resolve(),
        "tools": args.tools_config.expanduser().resolve(),
    }
    if args.policy_reward_config is not None:
        paths["policy_reward"] = args.policy_reward_config.expanduser().resolve()
    inputs = [path.expanduser().resolve() for path in args.input]
    actor = args.actor_checkpoint.expanduser().absolute()
    for path in [*paths.values(), *inputs]:
        if not path.is_file():
            raise ValueError(f"required input does not exist: {path}")
    if not actor.is_dir():
        raise ValueError(f"actor checkpoint directory does not exist: {actor}")
    if args.vllm_timeout < 1:
        raise ValueError("vLLM timeout must be positive")
    if args.expected_states < 1 or args.proposals_per_state < 2:
        raise ValueError("Stage-1 scale must contain states and at least two proposals")

    input_bindings = [_file_binding(path) for path in inputs]
    plan = _load_object(paths["plan"], "active branch plan")
    selection = _load_object(paths["selection"], "pivotal selection")
    backend_contract = _load_object(paths["backend"], "sampling backend contract")
    tool_schemas = _load_tool_schemas(paths["tools"])
    policy_reward = (
        _load_object(paths["policy_reward"], "policy reward config")
        if "policy_reward" in paths
        else None
    )
    resolved_policy_reward = validate_policy_reward_config(policy_reward)
    resolved = resolve_pivotal_selection(
        selection,
        _records(inputs),
        expected_inputs=input_bindings,
        require_prompt_capture=True,
    )

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(actor, trust_remote_code=True)
    tokenizer = processor.tokenizer
    parser = Qwen3CoderParser(tokenizer, tool_schemas)
    client = VllmTokenCompletionClient(
        args.served_model,
        args.vllm_base_url,
        os.environ.get(args.vllm_api_key_env, "EMPTY"),
        timeout=args.vllm_timeout,
    )
    git_binding = _git_binding(
        mechanical=args.mechanical,
        allow_dirty=args.allow_dirty_source,
    )
    source_provenance = {
        "schema_version": FIRST_DECISION_SOURCE_PROVENANCE_VERSION,
        **git_binding,
        "plan": _file_binding(paths["plan"]),
        "selection": _file_binding(paths["selection"]),
        "sampling_backend_contract": _file_binding(paths["backend"]),
        "tools_config": _file_binding(paths["tools"]),
        "policy_reward": (
            {
                **_file_binding(paths["policy_reward"]),
                "source": "file",
                "resolved_sha256": _sha256_json(resolved_policy_reward),
            }
            if "policy_reward" in paths
            else {
                "source": "embedded_default",
                "resolved_sha256": _sha256_json(resolved_policy_reward),
            }
        ),
        "inputs": input_bindings,
    }
    collector = FirstDecisionStage1Collector(
        plan=plan,
        resolved_selections=resolved,
        actor_checkpoint=actor,
        actor_tokenizer_contract_sha256=tokenizer_contract_sha256(tokenizer),
        sampling_backend_contract=backend_contract,
        completion_client=client,
        parser=parser,
        tool_schemas=tool_schemas,
        required_environment_version=args.environment_version,
        expected_states=args.expected_states,
        proposals_per_state=args.proposals_per_state,
        source_provenance=source_provenance,
        policy_reward=resolved_policy_reward,
        vllm_timeout_seconds=args.vllm_timeout,
    )
    return await collector.collect(args.journal_dir, resume=args.resume)


def main():
    args = parse_args()
    if args.allow_dirty_source and not args.mechanical:
        raise SystemExit(
            "--allow-dirty-source is restricted to --mechanical runs"
        )
    output = args.output.expanduser().resolve()
    manifest_output = args.manifest_output.expanduser().resolve()
    if output == manifest_output:
        raise SystemExit("Stage-1 output and manifest paths must differ")
    if manifest_output.exists():
        raise SystemExit("refusing to overwrite finalized Stage-1 artifacts")
    if output.exists() and not args.resume:
        raise SystemExit(
            "uncommitted Stage-1 records require --resume before manifest recovery"
        )
    args.journal_dir = (
        args.journal_dir.expanduser().resolve()
        if args.journal_dir is not None
        else manifest_output.parent / ".first-decision-stage1-journal"
    )
    try:
        collection = asyncio.run(_run(args))
        written_manifest = write_first_decision_artifacts(
            collection,
            records_output=output,
            manifest_output=manifest_output,
        )
        plan = _load_object(args.plan.expanduser().resolve(), "active branch plan")
        verified = verify_first_decision_artifacts(
            plan=plan,
            records_path=output,
            manifest_path=manifest_output,
            expected_source_provenance=collection["summary"]["source_provenance"],
        )
        if verified["manifest"] != written_manifest:
            raise ValueError("finalized Stage-1 manifest changed after commit")
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"first-decision Stage-1 collection failed: {exc}") from exc
    print(json.dumps(collection["summary"]["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
