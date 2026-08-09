#!/usr/bin/env python3
"""Build the deterministic GRPO-v2 train/tuning split from committed task assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "data/grpo"
DEFAULT_FINAL_TASKS = ROOT / "data/evaluation/tasks.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/data/grpo_v2_seed20260809"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "task_id" not in row:
                raise ValueError(f"{path}:{line_number} is missing task_id")
            rows.append(row)
    task_ids = [int(row["task_id"]) for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{path} contains duplicate task IDs")
    return rows


def _stable_key(seed: int, task_id: int) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def split_rows_by_length_bucket(
    rows: list[dict],
    *,
    tuning_size: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Hold out an exactly sized, deterministic sample while preserving bucket ratios."""
    if not 0 < tuning_size < len(rows):
        raise ValueError("tuning_size must be between zero and the source row count")
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        bucket = str(row.get("length_bucket") or "unknown")
        buckets[bucket].append(row)

    exact = {
        bucket: tuning_size * len(bucket_rows) / len(rows)
        for bucket, bucket_rows in buckets.items()
    }
    quotas = {bucket: int(value) for bucket, value in exact.items()}
    remainder = tuning_size - sum(quotas.values())
    for bucket in sorted(buckets, key=lambda name: (-(exact[name] - quotas[name]), name))[:remainder]:
        quotas[bucket] += 1

    tuning_ids = set()
    for bucket, bucket_rows in buckets.items():
        ranked = sorted(
            bucket_rows,
            key=lambda row: (_stable_key(seed, int(row["task_id"])), int(row["task_id"])),
        )
        tuning_ids.update(int(row["task_id"]) for row in ranked[: quotas[bucket]])

    train_rows = [row for row in rows if int(row["task_id"]) not in tuning_ids]
    tuning_rows = [row for row in rows if int(row["task_id"]) in tuning_ids]
    if len(tuning_rows) != tuning_size:
        raise AssertionError("deterministic split produced the wrong tuning size")
    return train_rows, tuning_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--final-tasks", type=Path, default=DEFAULT_FINAL_TASKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tuning-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_train = _read_jsonl(source_dir / "train.jsonl")
    source_validation = _read_jsonl(source_dir / "validation.jsonl")
    final_rows = _read_jsonl(args.final_tasks.expanduser().resolve())
    train_rows, tuning_rows = split_rows_by_length_bucket(
        source_train,
        tuning_size=args.tuning_size,
        seed=args.seed,
    )

    train_ids = {int(row["task_id"]) for row in train_rows}
    tuning_ids = {int(row["task_id"]) for row in tuning_rows}
    validation_ids = {int(row["task_id"]) for row in source_validation}
    final_ids = {int(row["task_id"]) for row in final_rows}
    named_sets = {
        "train": train_ids,
        "tuning150": tuning_ids,
        "validation50": validation_ids,
        "final200": final_ids,
    }
    names = list(named_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = named_sets[left] & named_sets[right]
            if overlap:
                raise ValueError(f"{left}/{right} overlap: {sorted(overlap)[:10]}")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required to build the GRPO-v2 parquet split") from exc
    source_table = pq.read_table(source_dir / "train.parquet")
    parquet_ids = [
        int(value["task_id"])
        for value in source_table.column("extra_info").to_pylist()
    ]
    if set(parquet_ids) != {int(row["task_id"]) for row in source_train}:
        raise ValueError("train.jsonl and train.parquet task IDs differ")
    selected_indices = [index for index, task_id in enumerate(parquet_ids) if task_id in train_ids]
    train_table = source_table.take(pa.array(selected_indices, type=pa.int64()))

    train_jsonl = output_dir / "train850.jsonl"
    tuning150_jsonl = output_dir / "tuning150.jsonl"
    tuning200_jsonl = output_dir / "tuning200.jsonl"
    train_parquet = output_dir / "train850.parquet"
    _write_jsonl(train_jsonl, train_rows)
    _write_jsonl(tuning150_jsonl, tuning_rows)
    _write_jsonl(tuning200_jsonl, [*source_validation, *tuning_rows])
    pq.write_table(train_table, train_parquet)

    metadata = {
        "schema_version": "shopping-grpo-v2-split-v1",
        "seed": args.seed,
        "source_tasks": len(source_train),
        "train_tasks": len(train_rows),
        "tuning150_tasks": len(tuning_rows),
        "validation50_tasks": len(source_validation),
        "tuning200_tasks": len(source_validation) + len(tuning_rows),
        "final200_tasks": len(final_rows),
        "length_buckets": {
            "source": Counter(str(row.get("length_bucket") or "unknown") for row in source_train),
            "train": Counter(str(row.get("length_bucket") or "unknown") for row in train_rows),
            "tuning150": Counter(str(row.get("length_bucket") or "unknown") for row in tuning_rows),
        },
        "overlaps": {
            f"{left}_{right}": len(named_sets[left] & named_sets[right])
            for index, left in enumerate(names)
            for right in names[index + 1 :]
        },
        "files": {
            path.name: {"tasks": count, "sha256": _sha256(path)}
            for path, count in (
                (train_jsonl, len(train_rows)),
                (train_parquet, len(train_rows)),
                (tuning150_jsonl, len(tuning_rows)),
                (tuning200_jsonl, len(source_validation) + len(tuning_rows)),
            )
        },
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=dict) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
