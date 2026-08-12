#!/usr/bin/env python3
"""Select outcome-blind, task-capped exact pivotal states from sampling audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from shopping_grpo.training.grpo.selection import (
    PIVOTAL_SELECTION_STRATEGY,
    SEARCH_DECISION_SELECTION_STRATEGY,
    select_pivotal_states,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-states", type=int, default=100)
    parser.add_argument(
        "--strategy",
        choices=(PIVOTAL_SELECTION_STRATEGY, SEARCH_DECISION_SELECTION_STRATEGY),
        default=PIVOTAL_SELECTION_STRATEGY,
    )
    parser.add_argument("--search-query-states", type=int)
    parser.add_argument("--search-open-states", type=int)
    parser.add_argument(
        "--exclude-task-ids",
        type=Path,
        help="JSON list of task ids excluded from a fresh confirmatory selection",
    )
    parser.add_argument(
        "--require-prompt-capture",
        action="store_true",
        help="select only pivotal events with exact actor prompt token captures",
    )
    return parser.parse_args()


def _load_records(paths: list[Path]):
    provenance = []
    records = []
    for input_index, path in enumerate(paths):
        payload = path.read_bytes()
        provenance.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
            if line.strip():
                records.append((input_index, str(path), line_number, json.loads(line)))
    return provenance, records


def main():
    args = parse_args()
    paths = [path.expanduser().resolve() for path in args.input]
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"sampling audit does not exist: {path}")
    excluded_task_ids = []
    if args.exclude_task_ids is not None:
        exclusion_path = args.exclude_task_ids.expanduser().resolve()
        try:
            excluded_task_ids = json.loads(exclusion_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read excluded task ids: {exc}") from exc
        if not isinstance(excluded_task_ids, list):
            raise SystemExit("excluded task ids must be a JSON list")
    label_quotas = None
    if args.strategy == SEARCH_DECISION_SELECTION_STRATEGY:
        if args.search_query_states is None or args.search_open_states is None:
            raise SystemExit(
                "search-decision strategy requires --search-query-states and "
                "--search-open-states"
            )
        label_quotas = {
            "search_query_decision": args.search_query_states,
            "search_result_open_decision": args.search_open_states,
        }
    elif (
        args.search_query_states is not None
        or args.search_open_states is not None
        or excluded_task_ids
    ):
        raise SystemExit("search-specific constraints require the search-decision strategy")
    try:
        provenance, records = _load_records(paths)
        result = select_pivotal_states(
            records,
            seed=args.seed,
            max_states=args.max_states,
            provenance=provenance,
            require_prompt_capture=args.require_prompt_capture,
            strategy=args.strategy,
            label_quotas=label_quotas,
            excluded_task_ids=excluded_task_ids,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SystemExit(f"pivotal selection failed: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
