#!/usr/bin/env python3
"""Audit strict-success gains and losses between two complete trajectory runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from shopping_grpo.evaluation.transition_audit import (
    audit_strict_transitions,
    render_transition_audit_markdown,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    benchmark = _read_jsonl(args.benchmark)
    result = audit_strict_transitions(
        expected_task_ids=[task["task_id"] for task in benchmark],
        source_trajectories=_read_jsonl(args.source),
        target_trajectories=_read_jsonl(args.target),
        source_label=args.source_label,
        target_label=args.target_label,
    )
    result["provenance"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in (
            ("benchmark", args.benchmark),
            ("source", args.source),
            ("target", args.target),
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(
        render_transition_audit_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
