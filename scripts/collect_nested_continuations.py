#!/usr/bin/env python3
"""Fork L exact continuations for every audited stage-one first decision."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

if __package__:
    from scripts.collect_active_suffixes import (
        AgentLoopToolObservationEncoder,
        Qwen3CoderParser,
        _load_object,
        _load_tool_schemas,
        _records,
        verify_verl_tool_observation_parity,
    )
else:
    from collect_active_suffixes import (  # type: ignore[no-redef]
        AgentLoopToolObservationEncoder,
        Qwen3CoderParser,
        _load_object,
        _load_tool_schemas,
        _records,
        verify_verl_tool_observation_parity,
    )
from shopping_grpo.environment.client import ShopAgentEnv
from shopping_grpo.training.grpo.active_suffix import VllmTokenCompletionClient
from shopping_grpo.training.grpo.nested_continuation import (
    NestedContinuationCollector,
)
from shopping_grpo.training.grpo.selection import resolve_pivotal_selection

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--active-suffixes", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--sampling-backend-contract", type=Path, required=True)
    parser.add_argument(
        "--tools-config", type=Path, default=ROOT / "configs" / "tools.json"
    )
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--environment-base-url", default="http://127.0.0.1:5700")
    parser.add_argument(
        "--environment-version", default="shopsimulator-environment-v2.1"
    )
    parser.add_argument("--vllm-timeout", type=int, default=180)
    parser.add_argument("--environment-timeout", type=int, default=60)
    parser.add_argument("--expected-proposals", type=int)
    parser.add_argument("--continuations-per-decision", type=int, default=8)
    parser.add_argument("--decisions-output", type=Path, required=True)
    parser.add_argument("--continuations-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path, name: str) -> list[dict]:
    values = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{name} line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise TypeError(f"{name} line {line_number} must be an object")
            values.append(value)
    if not values:
        raise ValueError(f"{name} must contain at least one record")
    return values


async def _run(args):
    paths = {
        "plan": args.plan.expanduser().resolve(),
        "selection": args.selection.expanduser().resolve(),
        "backend": args.sampling_backend_contract.expanduser().resolve(),
        "tools": args.tools_config.expanduser().resolve(),
        "stage1": args.active_suffixes.expanduser().resolve(),
    }
    inputs = [path.expanduser().resolve() for path in args.input]
    actor = args.actor_checkpoint.expanduser().absolute()
    for path in [*paths.values(), *inputs]:
        if not path.is_file():
            raise ValueError(f"required input does not exist: {path}")
    if not actor.is_dir():
        raise ValueError(f"actor checkpoint directory does not exist: {actor}")
    if args.vllm_timeout < 1 or args.environment_timeout < 1:
        raise ValueError("vLLM and environment timeouts must be positive")
    if args.continuations_per_decision not in {4, 8}:
        raise ValueError("continuations-per-decision must be exactly four or eight")

    provenance = [{"path": str(path), "sha256": _sha256(path)} for path in inputs]
    plan = _load_object(paths["plan"], "active branch plan")
    selection = _load_object(paths["selection"], "pivotal selection")
    backend_contract = _load_object(paths["backend"], "sampling backend contract")
    stage1_records = _jsonl(paths["stage1"], "active suffixes")
    tool_schemas = _load_tool_schemas(paths["tools"])
    resolved = resolve_pivotal_selection(
        selection,
        _records(inputs),
        expected_inputs=provenance,
        require_prompt_capture=True,
    )

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(actor, trust_remote_code=True)
    tokenizer = processor.tokenizer
    parser = Qwen3CoderParser(tokenizer, tool_schemas)
    decoding = plan.get("decoding_config") or {}
    encoder = AgentLoopToolObservationEncoder(
        processor,
        prompt_length=decoding.get("prompt_length"),
        apply_chat_template_kwargs=decoding.get("apply_chat_template_kwargs"),
        mm_processor_kwargs=decoding.get("mm_processor_kwargs"),
    )
    await verify_verl_tool_observation_parity(encoder)
    client = VllmTokenCompletionClient(
        args.served_model,
        args.vllm_base_url,
        os.environ.get(args.vllm_api_key_env, "EMPTY"),
        timeout=args.vllm_timeout,
    )
    collector = NestedContinuationCollector(
        plan=plan,
        resolved_selections=resolved,
        stage1_records=stage1_records,
        stage1_source_sha256=_sha256(paths["stage1"]),
        actor_checkpoint=actor,
        sampling_backend_contract=backend_contract,
        completion_client=client,
        parser=parser,
        encoder=encoder,
        env_factory=lambda: ShopAgentEnv(
            base_url=args.environment_base_url,
            timeout=args.environment_timeout,
        ),
        tool_schemas=tool_schemas,
        required_environment_version=args.environment_version,
        vllm_timeout_seconds=args.vllm_timeout,
        environment_timeout_seconds=args.environment_timeout,
        expected_proposals=args.expected_proposals,
        continuations_per_decision=args.continuations_per_decision,
    )
    return await collector.collect()


def _write_jsonl(path: Path, values: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    args = parse_args()
    outputs = [
        args.decisions_output.expanduser().resolve(),
        args.continuations_output.expanduser().resolve(),
        args.summary_output.expanduser().resolve(),
    ]
    if len(set(outputs)) != len(outputs):
        raise SystemExit("nested output paths must be distinct")
    if any(path.exists() for path in outputs):
        raise SystemExit("refusing to overwrite an existing nested artifact")
    try:
        collection = asyncio.run(_run(args))
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"nested continuation collection failed: {exc}") from exc

    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    temporaries = [path.with_suffix(path.suffix + ".tmp") for path in outputs]
    try:
        _write_jsonl(temporaries[0], collection["decisions"])
        _write_jsonl(temporaries[1], collection["continuations"])
        temporaries[2].write_text(
            json.dumps(collection["summary"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for temporary, output in zip(temporaries, outputs, strict=True):
            temporary.replace(output)
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)
    aggregate = collection["summary"]["aggregate"]
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    if aggregate["mechanical_collection_passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
