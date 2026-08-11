#!/usr/bin/env python3
"""Live-verify an exact, outcome-blind pivotal-state selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from shopping_grpo.environment.client import ShopAgentEnv
from shopping_grpo.environment.manifest import validate_manifest
from shopping_grpo.training.grpo.replay import verify_replay_with_factory
from shopping_grpo.training.grpo.selection import resolve_pivotal_selection


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def _load_validated_manifest_snapshot(path: Path) -> tuple[dict[str, object], str]:
    try:
        payload = path.read_bytes()
        manifest = json.loads(payload.decode("utf-8"))
        validated = validate_manifest(manifest)
        return validated, hashlib.sha256(payload).hexdigest()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid environment manifest: {exc}") from exc


def _load_validated_manifest(path: Path) -> dict[str, object]:
    return _load_validated_manifest_snapshot(path)[0]


def _load_selection(path: Path) -> tuple[Mapping[str, object], str]:
    try:
        payload = path.read_bytes()
        selection = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid pivotal selection: {exc}") from exc
    if not isinstance(selection, Mapping):
        raise TypeError("invalid pivotal selection: root must be an object")
    return selection, hashlib.sha256(payload).hexdigest()


def _load_records(paths: Sequence[Path]):
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


def verify_selected_replays(
    selection: Mapping[str, object],
    records: Iterable[tuple[int, str, int, Mapping[str, object]]],
    *,
    expected_inputs: Sequence[Mapping[str, object]],
    environment_manifest_sha256: str,
    required_environment_version: str,
    required_max_steps: int,
    env_factory,
) -> list[dict[str, object]]:
    """Preflight all locators, then replay every selected branch exactly once."""
    candidates = resolve_pivotal_selection(
        selection,
        records,
        expected_inputs=expected_inputs,
    )
    results = []
    for candidate in candidates:
        selected = candidate["selection"]
        trajectory = candidate["trajectory"]
        if not isinstance(selected, Mapping) or not isinstance(trajectory, Mapping):
            raise TypeError("resolved selection contract is invalid")
        verification = verify_replay_with_factory(
            trajectory,
            environment_manifest_sha256=environment_manifest_sha256,
            required_environment_version=required_environment_version,
            required_max_steps=required_max_steps,
            env_factory=env_factory,
            prefix_action_count=int(selected["prefix_action_count"]),
        )
        expected_state_id = str(selected["replay_state_id"])
        if (
            verification.get("verified")
            and verification.get("recomputed_replay_state_id") != expected_state_id
        ):
            verification.update(
                {
                    "verified": False,
                    "stage": "state_identity",
                    "reason": "recomputed_replay_state_id_mismatch",
                }
            )
        results.append(
            {
                "selection_index": int(selected["selection_index"]),
                "task_id": int(selected["task_id"]),
                "branch_uid": str(selected["branch_uid"]),
                "expected_replay_state_id": expected_state_id,
                "prefix_action_count": int(selected["prefix_action_count"]),
                "pivotal_labels": list(selected["pivotal_labels"]),
                "source": dict(selected["source"]),
                "verification": verification,
            }
        )
    if len(results) != len(candidates):  # pragma: no cover - loop is exhaustive.
        raise AssertionError("live verifier lost selected branches")
    return results


def main():
    args = parse_args()
    selection_path = args.selection.expanduser().resolve()
    input_paths = [path.expanduser().resolve() for path in args.input]
    manifest_path = args.environment_manifest.expanduser().resolve()
    if not selection_path.is_file():
        raise SystemExit(f"pivotal selection does not exist: {selection_path}")
    for path in input_paths:
        if not path.is_file():
            raise SystemExit(f"sampling audit does not exist: {path}")
    if not manifest_path.is_file():
        raise SystemExit(f"environment manifest does not exist: {manifest_path}")
    try:
        selection, selection_hash = _load_selection(selection_path)
        manifest, manifest_hash = _load_validated_manifest_snapshot(manifest_path)
        input_provenance, records = _load_records(input_paths)
        required_environment_version = str(
            manifest.get("environment_version", "shopsimulator-environment-v2.1")
        )
        required_max_steps = int(manifest["max_steps"])
        results = verify_selected_replays(
            selection,
            records,
            expected_inputs=input_provenance,
            environment_manifest_sha256=manifest_hash,
            required_environment_version=required_environment_version,
            required_max_steps=required_max_steps,
            env_factory=lambda: ShopAgentEnv(
                base_url=args.base_url,
                timeout=args.timeout,
            ),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"pivotal replay preflight failed: {exc}") from exc

    selected_count = len(selection["selections"])
    attempted = len(results)
    if attempted != selected_count:
        raise SystemExit("pivotal replay internal error: selected/result cardinality mismatch")
    verified = sum(bool(result["verification"].get("verified")) for result in results)
    all_selected_verified = bool(results) and verified == attempted == selected_count
    output = {
        "schema_version": "shopping-pivotal-live-replay-audit-v2",
        "provenance": {
            "selection": str(selection_path),
            "selection_sha256": selection_hash,
            "inputs": input_provenance,
            "environment_manifest": str(manifest_path),
            "environment_manifest_sha256": manifest_hash,
            "required_environment_version": required_environment_version,
            "required_max_steps": required_max_steps,
            "base_url": str(args.base_url),
        },
        "aggregate": {
            "selected_branches": selected_count,
            "resolved_branches": attempted,
            "attempted_branches": attempted,
            "verified_branches": verified,
            "failed_branches": attempted - verified,
            "verification_rate": verified / attempted if attempted else 0.0,
            "all_selected_verified": all_selected_verified,
        },
        "safety": {
            "selection_outcome_blind": True,
            "uses_hidden_goal": False,
            "raw_environment_payloads_saved": False,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["aggregate"], ensure_ascii=False, indent=2))
    if not all_selected_verified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
