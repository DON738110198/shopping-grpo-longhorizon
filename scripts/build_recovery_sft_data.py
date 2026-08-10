#!/usr/bin/env python3
"""Build leak-free recovery SFT suffixes from saved GRPO rollout groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from shopping_grpo.collection.sft import read_jsonl, write_jsonl
from shopping_grpo.training.sft.recovery import (
    build_recovery_rows,
    replay_recovery_rows,
    split_recovery_rows,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, action="append", required=True)
    parser.add_argument("--prompt-parquet", type=Path, required=True)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--base-validation", type=Path, required=True)
    parser.add_argument("--held-out", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--replay-base-url", default=None)
    parser.add_argument("--replay-workers", type=int, default=8)
    parser.add_argument(
        "--allow-replay-failures",
        action="store_true",
        help="Write only replay-passing rows instead of failing the build.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_prompts(path: Path) -> dict[int, list[dict]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("--prompt-parquet requires pyarrow") from exc
    prompts = {}
    for row in pq.read_table(path, columns=["prompt", "extra_info"]).to_pylist():
        task_id = int(row["extra_info"]["task_id"])
        if task_id in prompts:
            raise SystemExit(f"duplicate prompt task_id {task_id}")
        prompts[task_id] = row["prompt"]
    return prompts


def _read_rollouts(directories: list[Path]) -> tuple[list[dict], dict]:
    records = []
    file_hashes = []
    for directory in directories:
        paths = sorted(
            directory.glob("*.jsonl"),
            key=lambda path: int(path.stem) if path.stem.isdigit() else path.name,
        )
        for path in paths:
            file_hashes.append((str(path), _sha256(path)))
            for line_number, row in enumerate(read_jsonl(path), 1):
                row["_source"] = {
                    "path": str(path),
                    "step": int(path.stem) if path.stem.isdigit() else None,
                    "line": line_number,
                    "group": str(path),
                }
                records.append(row)
    combined = hashlib.sha256()
    for path, digest in file_hashes:
        combined.update(f"{path}\0{digest}\n".encode())
    return records, {
        "directories": [str(path) for path in directories],
        "files": len(file_hashes),
        "rows": len(records),
        "combined_sha256": combined.hexdigest(),
    }


def _held_out_ids(paths: list[Path]) -> set[int]:
    result = set()
    for path in paths:
        result.update(int(row["task_id"]) for row in read_jsonl(path))
    return result


def _file_view(path: Path) -> dict:
    return {
        "path": str(path),
        "rows": sum(1 for _ in read_jsonl(path)),
        "sha256": _sha256(path),
    }


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    args = parse_args()
    records, rollout_source = _read_rollouts(args.rollout_dir)
    prompts = _read_prompts(args.prompt_parquet)
    held_out = _held_out_ids(args.held_out)
    recovery_rows, rejected, summary = build_recovery_rows(
        records=records,
        prompts_by_task=prompts,
        held_out_task_ids=held_out,
    )
    replay_results = []
    if args.replay_base_url:
        replay_results = replay_recovery_rows(
            recovery_rows,
            base_url=args.replay_base_url,
            workers=args.replay_workers,
        )
        failed_ids = {item["task_id"] for item in replay_results if not item["ok"]}
        if failed_ids and not args.allow_replay_failures:
            raise SystemExit(
                f"replay rejected {len(failed_ids)} task(s): {sorted(failed_ids)[:20]}"
            )
        if failed_ids:
            rejected.extend(
                {"task_id": task_id, "reason": "replay_failed"}
                for task_id in sorted(failed_ids)
            )
            recovery_rows = [
                row for row in recovery_rows if int(row["task_id"]) not in failed_ids
            ]

    base_train = list(read_jsonl(args.base_train))
    base_validation = list(read_jsonl(args.base_validation))
    split = split_recovery_rows(
        recovery_rows=recovery_rows,
        base_train_rows=base_train,
        base_validation_rows=base_validation,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "recovery_all": output / "recovery_all.jsonl",
        "recovery_train": output / "recovery_train.jsonl",
        "recovery_validation": output / "recovery_validation.jsonl",
        "mixed_train": output / "mixed_train.jsonl",
        "mixed_validation": output / "mixed_validation.jsonl",
        "rejected": output / "rejected.jsonl",
        "replay": output / "replay_results.jsonl",
    }
    write_jsonl(paths["recovery_all"], recovery_rows)
    for name in (
        "recovery_train",
        "recovery_validation",
        "mixed_train",
        "mixed_validation",
    ):
        write_jsonl(paths[name], split[name])
    write_jsonl(paths["rejected"], rejected)
    write_jsonl(paths["replay"], replay_results)
    train_ids = {int(row["task_id"]) for row in split["mixed_train"]}
    validation_ids = {int(row["task_id"]) for row in split["mixed_validation"]}
    recovery_ids = {int(row["task_id"]) for row in recovery_rows}
    base_train_ids = {int(row["task_id"]) for row in base_train}
    base_validation_ids = {int(row["task_id"]) for row in base_validation}
    recovery_modes = Counter(
        row["recovery"]["supervision_mode"] for row in recovery_rows
    )
    metadata = {
        **summary,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "command": [sys.executable, *sys.argv],
        "recovery_rows_after_replay": len(recovery_rows),
        "replay": {
            "enabled": bool(args.replay_base_url),
            "base_url": args.replay_base_url,
            "workers": args.replay_workers if args.replay_base_url else 0,
            "passed": sum(item["ok"] for item in replay_results),
            "failed": sum(not item["ok"] for item in replay_results),
        },
        "split": {
            "seed": args.seed,
            "validation_ratio_for_unseen_tasks": args.validation_ratio,
            "recovery_train": len(split["recovery_train"]),
            "recovery_validation": len(split["recovery_validation"]),
            "mixed_train": len(split["mixed_train"]),
            "mixed_validation": len(split["mixed_validation"]),
            "train_validation_task_overlap": len(train_ids & validation_ids),
            "recovery_base_train_task_overlap": len(
                recovery_ids & base_train_ids
            ),
            "recovery_base_validation_task_overlap": len(
                recovery_ids & base_validation_ids
            ),
        },
        "supervision_mode_counts_after_replay": dict(sorted(recovery_modes.items())),
        "leakage": {
            "held_out_paths": [str(path) for path in args.held_out],
            "held_out_task_count": len(held_out),
            "recovery_held_out_overlap": len(
                recovery_ids & held_out
            ),
        },
        "source": {
            "rollouts": rollout_source,
            "prompt_parquet": {
                "path": str(args.prompt_parquet),
                "sha256": _sha256(args.prompt_parquet),
            },
            "base_train": _file_view(args.base_train),
            "base_validation": _file_view(args.base_validation),
        },
        "files": {name: _file_view(path) for name, path in paths.items()},
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
