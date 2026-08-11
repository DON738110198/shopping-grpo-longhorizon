#!/usr/bin/env python3
"""Attest live vLLM and materialize active-suffix backend/decoding contracts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from shopping_grpo.training.grpo.active_branch import (
    observation_policy_sha256,
    validate_decoding_config,
)
from shopping_grpo.training.grpo.active_suffix import (
    VllmTokenCompletionClient,
    sampling_backend_contract_sha256,
    sha256_actor_checkpoint,
    tool_schema_sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decoding-template",
        type=Path,
        default=ROOT / "configs" / "active_suffix_decoding_template.json",
    )
    parser.add_argument(
        "--tools-config",
        type=Path,
        default=ROOT / "configs" / "tools.json",
    )
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--backend-output", type=Path, required=True)
    parser.add_argument("--decoding-output", type=Path, required=True)
    return parser.parse_args(argv)


def _load_object(path: Path, name: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _tool_schemas(path: Path) -> list[dict]:
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


def _write_new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def materialize_contracts(
    *,
    decoding_template: dict,
    tool_schemas: list[dict],
    parser_stop_token_ids: list[int],
    actor_checkpoint: Path,
    completion_client: VllmTokenCompletionClient,
) -> tuple[dict, dict, str]:
    """Replace template attestations using the actor and live vLLM metadata."""
    decoding = json.loads(json.dumps(decoding_template))
    decoding["tool_schema_sha256"] = tool_schema_sha256(tool_schemas)
    decoding["stop_token_ids"] = list(parser_stop_token_ids)
    decoding["sampling_backend_contract_sha256"] = "0" * 64
    decoding["observation_policy_sha256"] = observation_policy_sha256(decoding)
    actor_sha256 = sha256_actor_checkpoint(actor_checkpoint)
    backend = completion_client.materialize_backend_contract(
        actor_checkpoint=actor_checkpoint,
        actor_checkpoint_sha256=actor_sha256,
        decoding_config=decoding,
    )
    if backend["max_model_len"] != int(decoding["context_window"]):
        raise ValueError("live vLLM max_model_len differs from decoding context_window")
    decoding["sampling_backend_contract_sha256"] = sampling_backend_contract_sha256(
        backend
    )
    decoding["observation_policy_sha256"] = observation_policy_sha256(decoding)
    return backend, validate_decoding_config(decoding), actor_sha256


def main():
    args = parse_args()
    template_path = args.decoding_template.expanduser().resolve()
    tools_path = args.tools_config.expanduser().resolve()
    actor = args.actor_checkpoint.expanduser().absolute()
    backend_output = args.backend_output.expanduser().resolve()
    decoding_output = args.decoding_output.expanduser().resolve()
    if backend_output == decoding_output:
        raise SystemExit("backend-output and decoding-output must be different files")
    if backend_output.exists() or decoding_output.exists():
        raise SystemExit("refusing to overwrite an existing contract artifact")
    for path in (template_path, tools_path):
        if not path.is_file():
            raise SystemExit(f"required input does not exist: {path}")
    if not actor.is_dir():
        raise SystemExit(f"actor checkpoint directory does not exist: {actor}")
    try:
        client = VllmTokenCompletionClient(
            args.served_model,
            args.vllm_base_url,
            os.environ.get(args.vllm_api_key_env, "EMPTY"),
            timeout=args.timeout,
        )
        decoding_template = _load_object(template_path, "decoding template")
        from transformers import AutoProcessor
        from verl.experimental.agent_loop.tool_parser import ToolParser

        processor = AutoProcessor.from_pretrained(actor, trust_remote_code=True)
        tokenizer = processor.tokenizer
        parser = ToolParser.get_tool_parser(
            str(decoding_template.get("tool_parser") or ""), tokenizer
        )
        parser_stop_token_ids = list(parser.stop_token_ids)
        backend, decoding, actor_sha256 = materialize_contracts(
            decoding_template=decoding_template,
            tool_schemas=_tool_schemas(tools_path),
            parser_stop_token_ids=parser_stop_token_ids,
            actor_checkpoint=actor,
            completion_client=client,
        )
        _write_new_json(backend_output, backend)
        try:
            _write_new_json(decoding_output, decoding)
        except Exception:
            backend_output.unlink(missing_ok=True)
            raise
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"active suffix contract materialization failed: {exc}") from exc
    print(
        json.dumps(
            {
                "actor_checkpoint_sha256": actor_sha256,
                "tool_schema_sha256": decoding["tool_schema_sha256"],
                "sampling_backend_contract_sha256": decoding[
                    "sampling_backend_contract_sha256"
                ],
                "server_version": backend["server_version"],
                "served_model": backend["served_model"],
                "max_model_len": backend["max_model_len"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
