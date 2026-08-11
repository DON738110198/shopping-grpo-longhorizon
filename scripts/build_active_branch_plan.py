#!/usr/bin/env python3
"""Build a no-optimizer K-suffix plan from captured pivotal-state prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from shopping_grpo.training.grpo.active_branch import build_active_branch_plan
from shopping_grpo.training.grpo.active_suffix import sha256_actor_checkpoint
from shopping_grpo.training.grpo.selection import resolve_pivotal_selection


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--decoding-config", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--suffixes-per-state", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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


def main():
    args = parse_args()
    input_paths = [path.expanduser().resolve() for path in args.input]
    selection_path = args.selection.expanduser().resolve()
    decoding_path = args.decoding_config.expanduser().resolve()
    actor_checkpoint = args.actor_checkpoint.expanduser().absolute()
    for path in [*input_paths, selection_path, decoding_path]:
        if not path.is_file():
            raise SystemExit(f"required input does not exist: {path}")
    if not actor_checkpoint.is_dir():
        raise SystemExit(f"actor checkpoint directory does not exist: {actor_checkpoint}")
    provenance = [{"path": str(path), "sha256": _sha256(path)} for path in input_paths]
    try:
        selection = _load_object(selection_path, "selection")
        decoding_config = _load_object(decoding_path, "decoding config")
        actor_checkpoint_sha256 = sha256_actor_checkpoint(actor_checkpoint)
        resolved = resolve_pivotal_selection(
            selection,
            _records(input_paths),
            expected_inputs=provenance,
            require_prompt_capture=True,
        )
        plan = build_active_branch_plan(
            resolved,
            actor_checkpoint_sha256=actor_checkpoint_sha256,
            decoding_config=decoding_config,
            seed=args.seed,
            suffixes_per_state=args.suffixes_per_state,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SystemExit(f"active branch plan failed: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plan["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
