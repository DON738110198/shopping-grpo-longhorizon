#!/usr/bin/env python3
"""Audit saved GRPO groups for replay-identical pivotal decision states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from shopping_grpo.training.grpo.pivotal_states import (
    audit_pivotal_states,
    render_pivotal_audit_markdown,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--min-visits", type=int, default=2)
    parser.add_argument("--training-min-visits", type=int, default=4)
    parser.add_argument("--training-min-groups", type=int, default=50)
    parser.add_argument("--training-min-tasks", type=int, default=40)
    parser.add_argument("--max-groups", type=int, default=200)
    parser.add_argument("--reward-tolerance", type=float, default=1.0e-8)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(paths: list[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    yield str(path), line_number, json.loads(line)


def main():
    args = parse_args()
    paths = [path.expanduser().resolve() for path in args.input]
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"sampling audit does not exist: {path}")
    provenance = [{"path": str(path), "sha256": _sha256(path)} for path in paths]
    duplicate_hashes = sorted(
        digest
        for digest in {item["sha256"] for item in provenance}
        if sum(item["sha256"] == digest for item in provenance) > 1
    )
    if duplicate_hashes:
        raise SystemExit(
            "duplicate sampling-audit content is not allowed: " + ", ".join(duplicate_hashes)
        )
    result = audit_pivotal_states(
        _records(paths),
        min_visits=args.min_visits,
        training_min_visits=args.training_min_visits,
        training_min_groups=args.training_min_groups,
        training_min_tasks=args.training_min_tasks,
        reward_tolerance=args.reward_tolerance,
        max_groups=args.max_groups,
    )
    result["provenance"] = provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(
        render_pivotal_audit_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
