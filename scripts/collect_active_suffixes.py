#!/usr/bin/env python3
"""Collect a fixed 2-state x 4-suffix observational active-branch probe."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from shopping_grpo.environment.client import ShopAgentEnv
from shopping_grpo.training.grpo.active_suffix import (
    ActiveSuffixRunner,
    VllmTokenCompletionClient,
)
from shopping_grpo.training.grpo.pivotal_states import tokenizer_contract_sha256
from shopping_grpo.training.grpo.selection import resolve_pivotal_selection

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--sampling-backend-contract", type=Path, required=True)
    parser.add_argument(
        "--tools-config",
        type=Path,
        default=ROOT / "configs" / "tools.json",
    )
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--environment-base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--environment-version", default="shopsimulator-environment-v2.1")
    parser.add_argument("--vllm-timeout", type=int, default=180)
    parser.add_argument("--environment-timeout", type=int, default=60)
    parser.add_argument("--expected-states", type=int, default=2)
    parser.add_argument("--suffixes-per-state", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(paths: list[Path]):
    for input_index, path in enumerate(paths):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    yield input_index, str(path), line_number, json.loads(line)


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


class AgentLoopToolObservationEncoder:
    """Reproduce veRL ToolAgentLoop's text-only tool turn tokenization."""

    def __init__(
        self,
        processor,
        *,
        prompt_length: int,
        apply_chat_template_kwargs: dict,
        mm_processor_kwargs: dict,
    ):
        from verl.utils.chat_template import initialize_system_prompt

        if apply_chat_template_kwargs != {} or mm_processor_kwargs != {}:
            raise ValueError("active suffix processor kwargs must currently be empty")
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.prompt_length = int(prompt_length)
        self.apply_chat_template_kwargs = dict(apply_chat_template_kwargs)
        self.mm_processor_kwargs = dict(mm_processor_kwargs)
        self._system_prompt = list(
            initialize_system_prompt(
                processor,
                **self.apply_chat_template_kwargs,
            )
        )
        self.tokenizer_contract_sha256 = tokenizer_contract_sha256(self.tokenizer)

    def count_tokens(self, text):
        return len(self.tokenizer.encode(str(text), add_special_tokens=False))

    def encode_tool_observation(self, text):
        from verl.utils.chat_template import apply_chat_template
        from verl.utils.tokenizer import (
            build_multimodal_processor_inputs,
            normalize_token_ids,
        )

        rendered = apply_chat_template(
            self.processor,
            [{"role": "tool", "content": str(text)}],
            tools=None,
            add_generation_prompt=True,
            tokenize=False,
            **self.apply_chat_template_kwargs,
        )
        model_inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[rendered],
            images=None,
            videos=None,
            audio=None,
            mm_processor_kwargs=self.mm_processor_kwargs,
        )
        token_ids = normalize_token_ids(model_inputs.pop("input_ids"))
        prefix = len(self._system_prompt)
        if token_ids[:prefix] != self._system_prompt:
            raise ValueError("tool observation system-prompt prefix mismatch")
        return token_ids[prefix:][-self.prompt_length :]


async def verify_verl_tool_observation_parity(encoder):
    """Compare short and long text against veRL's actual AgentLoopBase path."""
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase

    shim = SimpleNamespace(
        processor=encoder.processor,
        tokenizer=encoder.tokenizer,
        loop=asyncio.get_running_loop(),
        apply_chat_template_kwargs=encoder.apply_chat_template_kwargs,
        system_prompt=encoder._system_prompt,
        rollout_config=SimpleNamespace(prompt_length=encoder.prompt_length),
    )
    shim._get_mm_processor_kwargs = lambda audios=None: dict(
        encoder.mm_processor_kwargs
    )
    shim._cap_text_prompt_length = lambda token_ids: list(token_ids)[
        -encoder.prompt_length :
    ]
    samples = [
        "Environment terminated.",
        "long tool observation " * (encoder.prompt_length + 256),
    ]
    for text in samples:
        reference = await AgentLoopBase.apply_chat_template(
            shim,
            [{"role": "tool", "content": text}],
            tools=None,
            images=None,
            videos=None,
            audios=None,
            mm_processor_kwargs=encoder.mm_processor_kwargs,
            remove_system_prompt=True,
        )
        if list(reference) != encoder.encode_tool_observation(text):
            raise ValueError("active suffix tool observation token parity failed")


async def _run(args):
    paths = {
        "plan": args.plan.expanduser().resolve(),
        "selection": args.selection.expanduser().resolve(),
        "backend": args.sampling_backend_contract.expanduser().resolve(),
        "tools": args.tools_config.expanduser().resolve(),
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
    provenance = [{"path": str(path), "sha256": _sha256(path)} for path in inputs]
    plan = _load_object(paths["plan"], "active branch plan")
    selection = _load_object(paths["selection"], "pivotal selection")
    backend_contract = _load_object(paths["backend"], "sampling backend contract")
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
    runner = ActiveSuffixRunner(
        plan=plan,
        resolved_selections=resolved,
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
        expected_groups=args.expected_states,
        expected_suffixes_per_state=args.suffixes_per_state,
    )
    return await runner.collect()


def main():
    args = parse_args()
    output = args.output.expanduser().resolve()
    summary_output = args.summary_output.expanduser().resolve()
    if output == summary_output:
        raise SystemExit("output and summary-output must be different files")
    if output.exists() or summary_output.exists():
        raise SystemExit("refusing to overwrite an existing active suffix artifact")
    try:
        collection = asyncio.run(_run(args))
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"active suffix collection failed: {exc}") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output.with_suffix(output.suffix + ".tmp")
    summary_tmp = summary_output.with_suffix(summary_output.suffix + ".tmp")
    try:
        with output_tmp.open("x", encoding="utf-8", newline="\n") as handle:
            for record in collection["records"]:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        summary = {key: value for key, value in collection.items() if key != "records"}
        summary_tmp.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_tmp.replace(output)
        summary_tmp.replace(summary_output)
    finally:
        output_tmp.unlink(missing_ok=True)
        summary_tmp.unlink(missing_ok=True)
    print(json.dumps(collection["aggregate"], ensure_ascii=False, indent=2))
    if collection["aggregate"]["mechanical_smoke_passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
