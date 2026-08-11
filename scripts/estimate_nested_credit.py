#!/usr/bin/env python3
"""Attest nested collector artifacts and write a fail-closed credit report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from scripts.collect_active_suffixes import _records
else:
    from collect_active_suffixes import _records  # type: ignore[no-redef]

from shopping_grpo.training.grpo.nested_credit import (
    estimate_nested_collection_credit,
)
from shopping_grpo.training.grpo.selection import resolve_pivotal_selection


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(path: Path, name: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _jsonl(
    path: Path, name: str, *, allow_empty: bool = False
) -> list[dict[str, object]]:
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{name} line {line_number} must be an object")
        values.append(value)
    if not values and not allow_empty:
        raise ValueError(f"{name} must contain at least one record")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--continuations", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--stage1-source", type=Path, required=True)
    parser.add_argument("--sampling-backend-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = {
        "decisions": args.decisions.expanduser().resolve(),
        "continuations": args.continuations.expanduser().resolve(),
        "summary": args.summary.expanduser().resolve(),
        "environment_manifest": args.environment_manifest.expanduser().resolve(),
        "plan": args.plan.expanduser().resolve(),
        "selection": args.selection.expanduser().resolve(),
        "stage1": args.stage1_source.expanduser().resolve(),
        "backend": args.sampling_backend_contract.expanduser().resolve(),
    }
    input_paths = [path.expanduser().resolve() for path in args.input]
    actor = args.actor_checkpoint.expanduser().absolute()
    output = args.output.expanduser().resolve()
    for path in [*paths.values(), *input_paths]:
        if not path.is_file():
            raise SystemExit(f"nested credit input does not exist: {path}")
    if not actor.is_dir():
        raise SystemExit(f"actor checkpoint does not exist: {actor}")
    if output.exists():
        raise SystemExit(f"refusing to overwrite nested credit report: {output}")

    try:
        provenance = [
            {"path": str(path), "sha256": _sha256(path)} for path in input_paths
        ]
        selection = _object(paths["selection"], "pivotal selection")
        resolved = resolve_pivotal_selection(
            selection,
            _records(input_paths),
            expected_inputs=provenance,
            require_prompt_capture=True,
        )
        decisions = _jsonl(
            paths["decisions"], "nested decisions", allow_empty=True
        )
        continuations = _jsonl(
            paths["continuations"], "nested continuations", allow_empty=True
        )
        summary = _object(paths["summary"], "nested summary")
        report = estimate_nested_collection_credit(
            {
                "decisions": decisions,
                "continuations": continuations,
                "summary": summary,
            },
            actor_checkpoint=actor,
            environment_manifest=_object(
                paths["environment_manifest"], "environment manifest"
            ),
            environment_manifest_sha256=_sha256(paths["environment_manifest"]),
            active_branch_plan=_object(paths["plan"], "active branch plan"),
            resolved_selections=resolved,
            stage1_records=_jsonl(paths["stage1"], "stage-one suffixes"),
            stage1_source_sha256=_sha256(paths["stage1"]),
            sampling_backend_contract=_object(paths["backend"], "sampling backend"),
            artifact_files={
                name: paths[name]
                for name in ("decisions", "continuations", "summary")
            },
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"nested credit contract invalid: {exc}") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gate = report["training_gate"]
    print(
        json.dumps(
            {
                "training_ready": gate["training_ready"],
                "structural_ready": gate["structural_ready"],
                "signal_ready": gate["signal_ready"],
                "failed_checks": gate["failed_checks"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
