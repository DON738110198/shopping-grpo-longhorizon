#!/usr/bin/env python3
"""Fork L exact continuations for every audited stage-one first decision."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
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
from shopping_grpo.training.grpo.active_suffix import (
    VllmTokenCompletionClient,
    _sha256_json,
    sampling_backend_contract_sha256,
    sha256_actor_checkpoint,
    tool_schema_sha256,
    validate_sampling_backend_contract,
)
from shopping_grpo.training.grpo.nested_artifacts import (
    ActorCheckpointRunAttestation,
    NestedContinuationJournal,
    commit_nested_collection_artifacts,
    finalize_scale_ready_collection,
    verify_completed_collection_manifest,
)
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
    parser.add_argument("--stage1-manifest", type=Path, required=True)
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
    parser.add_argument("--journal-dir", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_clean_git_head(expected_git_sha: object) -> str:
    if (
        not isinstance(expected_git_sha, str)
        or len(expected_git_sha) != 40
        or any(character not in "0123456789abcdef" for character in expected_git_sha)
    ):
        raise ValueError("formal nested source lacks a full Git commit SHA")

    def run(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError("formal nested Git attestation failed") from exc
        return completed.stdout

    current = run("rev-parse", "HEAD").strip().lower()
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    if current != expected_git_sha or status:
        raise ValueError(
            "formal nested collection requires the clean Stage-1 Git commit"
        )
    return current


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
        "stage1_manifest": args.stage1_manifest.expanduser().resolve(),
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
    stage1_manifest = _load_object(
        paths["stage1_manifest"], "Stage-1 completion manifest"
    )
    _require_clean_git_head(
        (stage1_manifest.get("source_provenance") or {}).get("git_sha")
    )
    stage1_records = _jsonl(paths["stage1"], "active suffixes")
    tool_schemas = _load_tool_schemas(paths["tools"])
    resolved = resolve_pivotal_selection(
        selection,
        _records(inputs),
        expected_inputs=provenance,
        require_prompt_capture=True,
    )
    actor_attestor = ActorCheckpointRunAttestation(
        actor,
        str(plan.get("actor_checkpoint_sha256") or ""),
    )

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(actor, trust_remote_code=True)
    actor_attestor.verify_runtime(
        actor_checkpoint=actor,
        actor_checkpoint_sha256=plan.get("actor_checkpoint_sha256"),
    )
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
        stage1_records_path=paths["stage1"],
        stage1_manifest_path=paths["stage1_manifest"],
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
        actor_checkpoint_attestor=actor_attestor,
    )
    actor_binding = actor_attestor.verify_runtime(
        actor_checkpoint=actor,
        actor_checkpoint_sha256=plan.get("actor_checkpoint_sha256"),
    )
    journal = NestedContinuationJournal(
        args.journal_dir,
        collector.journal_contract(actor_binding),
    )
    completed_before_resume = journal.is_complete
    if not completed_before_resume:
        journal.begin_actor_run(actor_attestor.report())
    collector.set_completion_intent_observer(journal.begin_completion_request)
    collection = await collector.collect(
        resumed_entries=journal.entries(),
        persist_continuation=journal.append,
    )
    current_actor_report = actor_attestor.finish()
    if completed_before_resume:
        journal_report = journal.completion_report()
        actor_report = journal.latest_actor_attestation()
    else:
        journal.finish_actor_run(current_actor_report)
        actor_report = current_actor_report
        journal_report = journal.finalize()
    return {
        "collection": finalize_scale_ready_collection(
            collection,
            journal_path=journal.root,
            journal_report=journal_report,
            actor_attestation=actor_report,
        ),
        "journal_report": journal_report,
        "actor_attestation": actor_report,
        "journal_path": journal.root,
    }


def _verify_completed_request(args, manifest, summary) -> None:
    """Refuse an idempotent exit when any requested source identity changed."""
    source_contract = manifest.get("source_contract")
    if not isinstance(source_contract, dict):
        raise TypeError("completed nested source contract is missing")
    requested_files = {
        "plan": args.plan.expanduser().resolve(),
        "selection": args.selection.expanduser().resolve(),
        "backend": args.sampling_backend_contract.expanduser().resolve(),
        "tools": args.tools_config.expanduser().resolve(),
        "stage1": args.active_suffixes.expanduser().resolve(),
        "stage1_manifest": args.stage1_manifest.expanduser().resolve(),
    }
    inputs = [path.expanduser().resolve() for path in args.input]
    if any(not path.is_file() for path in [*requested_files.values(), *inputs]):
        raise ValueError("a requested completed-run source file is missing")
    actor = args.actor_checkpoint.expanduser().absolute()
    if not actor.is_dir():
        raise ValueError("requested completed-run actor checkpoint is missing")

    plan = _load_object(requested_files["plan"], "active branch plan")
    backend = validate_sampling_backend_contract(
        _load_object(requested_files["backend"], "sampling backend contract")
    )
    selection = _load_object(requested_files["selection"], "pivotal selection")
    provenance = [{"path": str(path), "sha256": _sha256(path)} for path in inputs]
    resolved = resolve_pivotal_selection(
        selection,
        _records(inputs),
        expected_inputs=provenance,
        require_prompt_capture=True,
    )
    stage1_binding = source_contract.get("stage1_source_binding")
    if not isinstance(stage1_binding, dict):
        raise TypeError("completed nested Stage-1 binding is missing")
    _require_clean_git_head(stage1_binding.get("source_git_sha"))
    requested_tool_sha256 = tool_schema_sha256(
        _load_tool_schemas(requested_files["tools"])
    )
    requested_actor_sha256 = sha256_actor_checkpoint(actor)
    expected_proposals = stage1_binding.get("record_count")
    if (
        _sha256_json(plan) != source_contract.get("active_branch_plan_sha256")
        or _sha256_json(resolved) != summary.get("resolved_selections_sha256")
        or sampling_backend_contract_sha256(backend)
        != source_contract.get("sampling_backend_contract_sha256")
        or requested_actor_sha256
        != source_contract.get("actor_checkpoint_sha256")
        or requested_tool_sha256
        != (plan.get("decoding_config") or {}).get("tool_schema_sha256")
        or args.environment_version
        != source_contract.get("required_environment_version")
        or args.served_model != backend.get("served_model")
        or args.continuations_per_decision
        != (summary.get("aggregate") or {}).get("continuations_per_decision")
        or (
            args.expected_proposals is not None
            and args.expected_proposals != expected_proposals
        )
        or _sha256(requested_files["stage1"])
        != stage1_binding.get("records_file_sha256")
        or _sha256(requested_files["stage1_manifest"])
        != stage1_binding.get("final_manifest_sha256")
    ):
        raise ValueError("completed nested collection differs from the requested run")


def main():
    args = parse_args()
    outputs = {
        "decisions": args.decisions_output.expanduser().absolute(),
        "continuations": args.continuations_output.expanduser().absolute(),
        "summary": args.summary_output.expanduser().absolute(),
    }
    output_parent = outputs["summary"].parent
    args.journal_dir = (
        args.journal_dir.expanduser().absolute()
        if args.journal_dir is not None
        else output_parent / ".nested-continuation-journal"
    )
    args.manifest_output = (
        args.manifest_output.expanduser().absolute()
        if args.manifest_output is not None
        else output_parent / "nested-collection-manifest.json"
    )
    if len(set(outputs.values())) != len(outputs):
        raise SystemExit("nested output paths must be distinct")
    if args.manifest_output in outputs.values():
        raise SystemExit("nested manifest path must be distinct from artifacts")
    if any(path.parent != output_parent for path in outputs.values()) or (
        args.manifest_output.parent != output_parent
    ):
        raise SystemExit("nested artifacts and manifest must share one directory")
    try:
        args.journal_dir.relative_to(output_parent)
    except ValueError as exc:
        raise SystemExit(
            "nested journal must be inside the artifact commit directory"
        ) from exc
    if args.manifest_output.exists():
        try:
            manifest = verify_completed_collection_manifest(
                args.manifest_output,
                artifact_files=outputs,
            )
            summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
            _verify_completed_request(args, manifest, summary)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"completed nested manifest is invalid: {exc}") from exc
        aggregate = summary["aggregate"]
        print(
            json.dumps(
                {
                    **aggregate,
                    "artifact_set_uid": manifest["artifact_set_uid"],
                    "already_committed": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if aggregate["mechanical_collection_passed"] is not True:
            raise SystemExit(2)
        return
    try:
        result = asyncio.run(_run(args))
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"nested continuation collection failed: {exc}") from exc
    try:
        manifest = commit_nested_collection_artifacts(
            result["collection"],
            artifact_files=outputs,
            manifest_path=args.manifest_output,
            journal_path=result["journal_path"],
            journal_report=result["journal_report"],
            actor_attestation=result["actor_attestation"],
        )
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"nested artifact commit failed: {exc}") from exc
    aggregate = result["collection"]["summary"]["aggregate"]
    print(
        json.dumps(
            {**aggregate, "artifact_set_uid": manifest["artifact_set_uid"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    if aggregate["mechanical_collection_passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
