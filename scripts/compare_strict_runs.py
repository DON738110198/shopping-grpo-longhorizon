#!/usr/bin/env python3
"""Compare two complete evaluation JSONL files on paired strict Gold success."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.evaluation.paired import compare_strict_success


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    args = parse_args()
    benchmark = _read_jsonl(args.benchmark)
    result = compare_strict_success(
        expected_task_ids=[task["task_id"] for task in benchmark],
        source_trajectories=_read_jsonl(args.source),
        target_trajectories=_read_jsonl(args.target),
        source_label=args.source_label,
        target_label=args.target_label,
        confidence=args.confidence,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    result["provenance"] = {
        "benchmark": str(args.benchmark),
        "source": str(args.source),
        "target": str(args.target),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
